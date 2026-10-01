//! Present llama.cpp's text GGUF and mmproj GGUF as one streaming model.
//! The Gemma 4 wgpu engine expects Ollama's combined GGUF layout. This
//! adapter builds only a virtual header; tensor bytes stay in the two files.

use std::{collections::HashMap, path::Path, sync::Arc};

use async_trait::async_trait;
use rullama_engine::{
    error::{Result as EngineResult, RullamaError},
    gguf::{
        FileFetcher, GgmlDtype, GgufReader, GgufValue, InMemoryFetcher, TensorDesc, TensorFetcher,
    },
};

use crate::{Error, Result};

struct Segment {
    start: u64,
    end: u64,
    source_offset: u64,
    source: Arc<dyn TensorFetcher>,
    conversion: Option<Conversion>,
}

enum Conversion {
    Q8ToF16(u64),
    Bf16ToF16(u64),
}

struct PairedFetcher {
    header: Vec<u8>,
    segments: Vec<Segment>,
    total: u64,
}

#[async_trait(?Send)]
impl TensorFetcher for PairedFetcher {
    fn total_len(&self) -> u64 {
        self.total
    }

    async fn fetch(&self, offset: u64, len: u64) -> EngineResult<Vec<u8>> {
        let end = offset
            .checked_add(len)
            .filter(|&end| end <= self.total)
            .ok_or_else(|| RullamaError::Gguf("virtual GGUF read is out of bounds".into()))?;
        let mut out = Vec::with_capacity(len as usize);
        let mut pos = offset;
        while pos < end {
            if pos < self.header.len() as u64 {
                let stop = end.min(self.header.len() as u64);
                out.extend_from_slice(&self.header[pos as usize..stop as usize]);
                pos = stop;
                continue;
            }
            if let Some(segment) = self
                .segments
                .iter()
                .find(|segment| segment.start <= pos && pos < segment.end)
            {
                let stop = end.min(segment.end);
                if let Some(conversion) = &segment.conversion {
                    let (raw_len, convert): (u64, fn(&[u8]) -> EngineResult<Vec<u8>>) =
                        match conversion {
                            Conversion::Q8ToF16(len) => (*len, q8_to_f16),
                            Conversion::Bf16ToF16(len) => (*len, bf16_to_f16),
                        };
                    let raw = segment.source.fetch(segment.source_offset, raw_len).await?;
                    let converted = convert(&raw)?;
                    let begin = (pos - segment.start) as usize;
                    out.extend_from_slice(&converted[begin..begin + (stop - pos) as usize]);
                } else {
                    let bytes = segment
                        .source
                        .fetch(segment.source_offset + pos - segment.start, stop - pos)
                        .await?;
                    out.extend_from_slice(&bytes);
                }
                pos = stop;
            } else {
                let stop = self
                    .segments
                    .iter()
                    .filter(|segment| segment.start > pos)
                    .map(|segment| segment.start)
                    .min()
                    .unwrap_or(end)
                    .min(end);
                out.resize(out.len() + (stop - pos) as usize, 0);
                pos = stop;
            }
        }
        Ok(out)
    }
}

// TensorFetcher itself is local-only (?Send async methods); the engine API requires Arc.
#[allow(clippy::arc_with_non_send_sync)]
pub async fn open(model_path: &Path, projector_path: &Path) -> Result<Arc<dyn TensorFetcher>> {
    let model_file: Arc<dyn TensorFetcher> = Arc::new(FileFetcher::open(model_path).map_err(err)?);
    let projector_file: Arc<dyn TensorFetcher> =
        Arc::new(FileFetcher::open(projector_path).map_err(err)?);
    let model = GgufReader::new_streaming(model_file.clone())
        .await
        .map_err(err)?;
    let projector = GgufReader::new_streaming(projector_file.clone())
        .await
        .map_err(err)?;
    if model
        .get("general.architecture")
        .map_err(err)?
        .as_str()
        .map_err(err)?
        != "gemma4"
    {
        return Err(Error::Invalid(
            "model GGUF must have gemma4 architecture".into(),
        ));
    }
    if projector
        .get("general.architecture")
        .map_err(err)?
        .as_str()
        .map_err(err)?
        != "clip"
    {
        return Err(Error::Invalid(
            "mmproj GGUF must have clip architecture".into(),
        ));
    }
    if projector.tensor("v.patch_embd.weight").is_err() {
        return Err(Error::Invalid(
            "mmproj GGUF has no Gemma 4 vision tower".into(),
        ));
    }

    let mut metadata = model.metadata().clone();
    for (from, to) in [
        ("clip.vision.block_count", "gemma4.vision.block_count"),
        (
            "clip.vision.embedding_length",
            "gemma4.vision.embedding_length",
        ),
        (
            "clip.vision.feed_forward_length",
            "gemma4.vision.feed_forward_length",
        ),
        (
            "clip.vision.attention.head_count",
            "gemma4.vision.attention.head_count",
        ),
        ("clip.vision.patch_size", "gemma4.vision.patch_size"),
        (
            "clip.vision.attention.layer_norm_epsilon",
            "gemma4.vision.attention.layer_norm_epsilon",
        ),
    ] {
        let value = projector.get(from).map_err(err)?.clone();
        metadata.insert(to.to_owned(), value);
    }
    // llama.cpp's Gemma 4 mmproj uses 3x3 pooling and RGB input.
    metadata.insert(
        "gemma4.vision.projector.scale_factor".into(),
        GgufValue::U32(3),
    );
    metadata.insert("gemma4.vision.num_channels".into(), GgufValue::U32(3));

    let (mut tensors, mut segments, mut cursor) = model_parts(&model, model_file);
    for tensor in projector.tensors().iter().filter(|tensor| {
        tensor.name.starts_with("v.") || tensor.name.starts_with("mm.input_projection")
    }) {
        if tensors.iter().any(|existing| existing.name == tensor.name) {
            return Err(Error::Invalid(format!(
                "duplicate GGUF tensor {}",
                tensor.name
            )));
        }
        let mut virtual_tensor = tensor.clone();
        virtual_tensor.offset = cursor;
        let conversion = match tensor.dtype {
            GgmlDtype::Q8_0 => {
                virtual_tensor.dtype = GgmlDtype::F16;
                Some(Conversion::Q8ToF16(tensor.byte_len()))
            }
            GgmlDtype::BF16 => {
                virtual_tensor.dtype = GgmlDtype::F16;
                Some(Conversion::Bf16ToF16(tensor.byte_len()))
            }
            _ => None,
        };
        let end = cursor + virtual_tensor.byte_len();
        segments.push(Segment {
            start: cursor,
            end,
            source_offset: projector.data_section_offset() + tensor.offset,
            source: projector_file.clone(),
            conversion,
        });
        cursor = align(end, model.alignment());
        tensors.push(virtual_tensor);
    }
    let clamps = vision_clamps(&projector).await?;
    let clamp_len = clamps.len() as u64;
    tensors.push(TensorDesc {
        name: "v.clamp_data".into(),
        dims: vec![clamp_len / 4],
        dtype: GgmlDtype::F32,
        offset: cursor,
    });
    segments.push(Segment {
        start: cursor,
        end: cursor + clamp_len,
        source_offset: 0,
        source: Arc::new(InMemoryFetcher::new(clamps)),
        conversion: None,
    });
    cursor = align(cursor + clamp_len, model.alignment());

    finish_virtual_gguf(metadata, tensors, segments, cursor, model.alignment())
}

/// Present a text GGUF with BF16 tensors as F16 to the wgpu matmul kernels.
pub async fn open_text(model_path: &Path) -> Result<Arc<dyn TensorFetcher>> {
    let file: Arc<dyn TensorFetcher> = Arc::new(FileFetcher::open(model_path).map_err(err)?);
    let model = GgufReader::new_streaming(file.clone()).await.map_err(err)?;
    let (tensors, segments, cursor) = model_parts(&model, file);
    finish_virtual_gguf(
        model.metadata().clone(),
        tensors,
        segments,
        cursor,
        model.alignment(),
    )
}

fn model_parts(
    model: &GgufReader,
    file: Arc<dyn TensorFetcher>,
) -> (Vec<TensorDesc>, Vec<Segment>, u64) {
    let model_data_len = file.total_len() - model.data_section_offset();
    let mut cursor = align(model_data_len, model.alignment());
    let mut tensors: Vec<_> = model
        .tensors()
        .iter()
        .filter(|tensor| tensor.dtype != GgmlDtype::BF16)
        .cloned()
        .collect();
    let mut segments = vec![Segment {
        start: 0,
        end: model_data_len,
        source_offset: model.data_section_offset(),
        source: file.clone(),
        conversion: None,
    }];
    for original in model
        .tensors()
        .iter()
        .filter(|tensor| tensor.dtype == GgmlDtype::BF16)
    {
        let mut converted = original.clone();
        converted.dtype = GgmlDtype::F16;
        converted.offset = cursor;
        let end = cursor + converted.byte_len();
        segments.push(Segment {
            start: cursor,
            end,
            source_offset: model.data_section_offset() + original.offset,
            source: file.clone(),
            conversion: Some(Conversion::Bf16ToF16(original.byte_len())),
        });
        cursor = align(end, model.alignment());
        tensors.push(converted);
    }
    (tensors, segments, cursor)
}

// TensorFetcher has local-only async methods; the model API still requires Arc.
#[allow(clippy::arc_with_non_send_sync)]
fn finish_virtual_gguf(
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorDesc>,
    mut segments: Vec<Segment>,
    cursor: u64,
    alignment: u64,
) -> Result<Arc<dyn TensorFetcher>> {
    let mut header = Vec::new();
    header.extend_from_slice(b"GGUF");
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    write_metadata(&mut header, &metadata)?;
    for tensor in &tensors {
        write_string(&mut header, &tensor.name);
        header.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for dim in &tensor.dims {
            header.extend_from_slice(&dim.to_le_bytes());
        }
        header.extend_from_slice(&(tensor.dtype as u32).to_le_bytes());
        header.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    header.resize(align(header.len() as u64, alignment) as usize, 0);
    let base = header.len() as u64;
    for segment in &mut segments {
        segment.start += base;
        segment.end += base;
    }
    let total = base + cursor;
    Ok(Arc::new(PairedFetcher {
        header,
        segments,
        total,
    }))
}

fn q8_to_f16(raw: &[u8]) -> EngineResult<Vec<u8>> {
    if !raw.len().is_multiple_of(34) {
        return Err(RullamaError::Gguf("invalid Q8_0 tensor byte length".into()));
    }
    let mut out = Vec::with_capacity(raw.len() / 34 * 64);
    for block in raw.as_chunks::<34>().0 {
        let scale = half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
        for &value in &block[2..] {
            let result = half::f16::from_f32(scale * (value as i8 as f32));
            if !result.is_finite() {
                return Err(RullamaError::Gguf(
                    "Q8_0 vision weight is nonfinite after F16 conversion".into(),
                ));
            }
            out.extend_from_slice(&result.to_bits().to_le_bytes());
        }
    }
    Ok(out)
}

fn bf16_to_f16(raw: &[u8]) -> EngineResult<Vec<u8>> {
    if !raw.len().is_multiple_of(2) {
        return Err(RullamaError::Gguf("invalid BF16 tensor byte length".into()));
    }
    let mut out = Vec::with_capacity(raw.len());
    for pair in raw.as_chunks::<2>().0 {
        let value = half::bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32();
        let result = half::f16::from_f32(value);
        if !result.is_finite() {
            return Err(RullamaError::Gguf(
                "BF16 model weight is nonfinite after F16 conversion".into(),
            ));
        }
        out.extend_from_slice(&result.to_bits().to_le_bytes());
    }
    Ok(out)
}

async fn vision_clamps(projector: &GgufReader) -> Result<Vec<u8>> {
    let layers = projector
        .get("clip.vision.block_count")
        .map_err(err)?
        .as_u32()
        .map_err(err)?;
    let mut out = Vec::with_capacity(layers as usize * 7 * 4 * 4);
    for layer in 0..layers {
        for linear in [
            "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down",
        ] {
            let names = ["input_min", "input_max", "output_min", "output_max"];
            for (i, suffix) in names.into_iter().enumerate() {
                let name = format!("v.blk.{layer}.{linear}.{suffix}");
                let value = if projector.tensor(&name).is_ok() {
                    let bytes = projector.fetch_tensor_bytes(&name).await.map_err(err)?;
                    if bytes.len() != 4 {
                        return Err(Error::Backend(format!("invalid vision clamp {name}")));
                    }
                    f32::from_le_bytes(bytes.try_into().expect("four bytes"))
                } else if i.is_multiple_of(2) {
                    f32::MIN
                } else {
                    f32::MAX
                };
                if !value.is_finite() {
                    return Err(Error::Backend(format!("nonfinite vision clamp {name}")));
                }
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    Ok(out)
}

fn align(n: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        n
    } else {
        n.div_ceil(alignment) * alignment
    }
}

fn write_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u64).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

fn write_metadata(out: &mut Vec<u8>, metadata: &HashMap<String, GgufValue>) -> Result<()> {
    let mut entries: Vec<_> = metadata.iter().collect();
    entries.sort_by_key(|(key, _)| *key);
    for (key, value) in entries {
        write_string(out, key);
        write_value(out, value)?;
    }
    Ok(())
}

fn write_value(out: &mut Vec<u8>, value: &GgufValue) -> Result<()> {
    macro_rules! scalar {
        ($type:expr, $v:expr) => {{
            out.extend_from_slice(&$type.to_le_bytes());
            out.extend_from_slice(&$v.to_le_bytes());
        }};
    }
    macro_rules! array {
        ($type:expr, $values:expr, $body:expr) => {{
            out.extend_from_slice(&9u32.to_le_bytes());
            out.extend_from_slice(&$type.to_le_bytes());
            out.extend_from_slice(&($values.len() as u64).to_le_bytes());
            for value in $values {
                $body(out, value);
            }
        }};
    }
    match value {
        GgufValue::U8(v) => scalar!(0u32, v),
        GgufValue::I8(v) => scalar!(1u32, v),
        GgufValue::U16(v) => scalar!(2u32, v),
        GgufValue::I16(v) => scalar!(3u32, v),
        GgufValue::U32(v) => scalar!(4u32, v),
        GgufValue::I32(v) => scalar!(5u32, v),
        GgufValue::F32(v) => scalar!(6u32, v),
        GgufValue::Bool(v) => {
            out.extend_from_slice(&7u32.to_le_bytes());
            out.push(u8::from(*v));
        }
        GgufValue::String(v) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_string(out, v);
        }
        GgufValue::U64(v) => scalar!(10u32, v),
        GgufValue::I64(v) => scalar!(11u32, v),
        GgufValue::F64(v) => scalar!(12u32, v),
        GgufValue::ArrayU8(v) => array!(0u32, v, |o: &mut Vec<u8>, x: &u8| o.push(*x)),
        GgufValue::ArrayI8(v) => array!(1u32, v, |o: &mut Vec<u8>, x: &i8| o.push(*x as u8)),
        GgufValue::ArrayU16(v) => array!(2u32, v, |o: &mut Vec<u8>, x: &u16| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayI16(v) => array!(3u32, v, |o: &mut Vec<u8>, x: &i16| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayU32(v) => array!(4u32, v, |o: &mut Vec<u8>, x: &u32| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayI32(v) => array!(5u32, v, |o: &mut Vec<u8>, x: &i32| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayF32(v) => array!(6u32, v, |o: &mut Vec<u8>, x: &f32| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayBool(v) => {
            array!(7u32, v, |o: &mut Vec<u8>, x: &bool| o.push(u8::from(*x)))
        }
        GgufValue::ArrayString(v) => {
            array!(8u32, v, |o: &mut Vec<u8>, x: &String| write_string(o, x))
        }
        GgufValue::ArrayU64(v) => array!(10u32, v, |o: &mut Vec<u8>, x: &u64| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayI64(v) => array!(11u32, v, |o: &mut Vec<u8>, x: &i64| o
            .extend_from_slice(&x.to_le_bytes())),
        GgufValue::ArrayF64(v) => array!(12u32, v, |o: &mut Vec<u8>, x: &f64| o
            .extend_from_slice(&x.to_le_bytes())),
    }
    Ok(())
}

fn err(error: impl std::fmt::Display) -> Error {
    Error::Backend(error.to_string())
}

#[cfg(test)]
#[test]
fn converted_vision_and_model_weights_keep_numeric_values() {
    let mut q8 = Vec::from(half::f16::from_f32(0.5).to_bits().to_le_bytes());
    q8.extend([0u8, 1, 254, 127]);
    q8.resize(34, 0);
    let converted = q8_to_f16(&q8).unwrap();
    let values: Vec<f32> = converted
        .as_chunks::<2>()
        .0
        .iter()
        .take(4)
        .map(|pair| half::f16::from_bits(u16::from_le_bytes(*pair)).to_f32())
        .collect();
    assert_eq!(values, [0.0, 0.5, -1.0, 63.5]);

    let raw = half::bf16::from_f32(1.5).to_bits().to_le_bytes();
    let converted = bf16_to_f16(&raw).unwrap();
    assert_eq!(
        half::f16::from_bits(u16::from_le_bytes(converted.try_into().unwrap())).to_f32(),
        1.5
    );
}

#[cfg(test)]
#[test]
fn text_gguf_exposes_bf16_weights_as_f16() {
    let original = half::bf16::from_f32(1.5).to_bits().to_le_bytes();
    let plain = half::f16::from_f32(2.0).to_bits().to_le_bytes();
    let tensors = [
        TensorDesc {
            name: "bf16.weight".into(),
            dims: vec![1],
            dtype: GgmlDtype::BF16,
            offset: 0,
        },
        TensorDesc {
            name: "f16.weight".into(),
            dims: vec![1],
            dtype: GgmlDtype::F16,
            offset: 64,
        },
    ];
    let mut source = Vec::new();
    source.extend_from_slice(b"GGUF");
    source.extend_from_slice(&3u32.to_le_bytes());
    source.extend_from_slice(&2u64.to_le_bytes());
    source.extend_from_slice(&1u64.to_le_bytes());
    write_string(&mut source, "general.alignment");
    write_value(&mut source, &GgufValue::U32(64)).unwrap();
    for tensor in &tensors {
        write_string(&mut source, &tensor.name);
        source.extend_from_slice(&1u32.to_le_bytes());
        source.extend_from_slice(&1u64.to_le_bytes());
        source.extend_from_slice(&(tensor.dtype as u32).to_le_bytes());
        source.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    source.resize(align(source.len() as u64, 64) as usize, 0);
    let data_offset = source.len();
    source.extend_from_slice(&original);
    source.resize(data_offset + 64, 0);
    source.extend_from_slice(&plain);

    let path = test_file("text-bf16", &source);
    let virtual_file = pollster::block_on(open_text(&path)).unwrap();
    let virtual_reader = pollster::block_on(GgufReader::new_streaming(virtual_file)).unwrap();
    assert_eq!(
        virtual_reader.tensor("bf16.weight").unwrap().dtype,
        GgmlDtype::F16
    );
    assert_eq!(
        virtual_reader.tensor("f16.weight").unwrap().dtype,
        GgmlDtype::F16
    );
    assert_eq!(
        pollster::block_on(virtual_reader.fetch_tensor_bytes("bf16.weight")).unwrap(),
        bf16_to_f16(&original).unwrap()
    );
    assert_eq!(
        pollster::block_on(virtual_reader.fetch_tensor_bytes("f16.weight")).unwrap(),
        plain
    );
    std::fs::remove_file(path).unwrap();
}

#[cfg(test)]
fn test_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path =
        std::env::temp_dir().join(format!("l2s1-{name}-{}-{nonce}.gguf", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

#[cfg(test)]
#[test]
#[ignore = "requires SKID_VISION_MODEL and SKID_VISION_MMPROJ"]
fn paired_files_expose_both_tensor_sections() {
    let model_path = std::env::var("SKID_VISION_MODEL").expect("SKID_VISION_MODEL");
    let projector_path = std::env::var("SKID_VISION_MMPROJ").expect("SKID_VISION_MMPROJ");
    let fetcher = pollster::block_on(open(Path::new(&model_path), Path::new(&projector_path)))
        .expect("pair GGUFs");
    let reader = pollster::block_on(GgufReader::new_streaming(fetcher)).expect("virtual GGUF");
    assert_eq!(
        reader
            .get("general.architecture")
            .unwrap()
            .as_str()
            .unwrap(),
        "gemma4"
    );
    assert_eq!(
        reader
            .get("gemma4.vision.block_count")
            .unwrap()
            .as_u32()
            .unwrap(),
        16
    );
    for name in [
        "token_embd.weight",
        "per_layer_model_proj.weight",
        "v.patch_embd.weight",
        "mm.input_projection.weight",
    ] {
        let tensor = reader.tensor(name).expect(name);
        let bytes = pollster::block_on(reader.fetch_tensor_bytes(name)).expect(name);
        assert_eq!(bytes.len() as u64, tensor.byte_len());
        let source_path = if name == "token_embd.weight" || name == "per_layer_model_proj.weight" {
            &model_path
        } else {
            &projector_path
        };
        let source = Arc::new(FileFetcher::open(Path::new(source_path)).unwrap());
        let original = pollster::block_on(GgufReader::new_streaming(source)).unwrap();
        let raw = pollster::block_on(original.fetch_tensor_bytes(name)).unwrap();
        if name == "per_layer_model_proj.weight" {
            assert_eq!(tensor.dtype, GgmlDtype::F16);
            assert_eq!(bytes, bf16_to_f16(&raw).unwrap());
        } else if name != "token_embd.weight"
            && original.tensor(name).unwrap().dtype == GgmlDtype::Q8_0
        {
            assert_eq!(tensor.dtype, GgmlDtype::F16);
            assert_eq!(bytes, q8_to_f16(&raw).unwrap());
        } else {
            assert_eq!(bytes, raw);
        }
    }
}
