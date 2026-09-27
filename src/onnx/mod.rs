//! Laya marker-head inference through ONNX Runtime. No language-model mass is invented.
#[allow(dead_code)]
mod pyjson;
use crate::http::HttpDecisionBackend;
use crate::{Decision, DecisionKind, DecisionRequest, Error, Result};
use ndarray::Array2;
use ort::{
    session::{Session, builder::GraphOptimizationLevel},
    value::Tensor,
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
use tokenizers::Tokenizer;

fn err(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}
#[derive(Deserialize)]
struct Special {
    cls: u32,
    sep: u32,
    mask: u32,
    pad: u32,
    mask_text: String,
}
#[derive(Deserialize)]
struct Config {
    engine: String,
    layout: String,
    max_len: usize,
    head_max_len: usize,
    special_tokens: Special,
    #[serde(default = "two")]
    min_markers: usize,
}
fn two() -> usize {
    2
}
struct Row {
    ids: Vec<u32>,
    markers: Vec<usize>,
    qtype: usize,
}

/// One resident ONNX session. CPU and CUDA are explicit, without CPU fallback on CUDA errors.
pub struct OnnxBackend {
    session: Session,
    tokenizer: Tokenizer,
    config: Config,
    calibration: Value,
    identity: String,
    device: String,
    min_probability: f64,
}
impl OnnxBackend {
    pub fn load(
        directory: &Path,
        cuda: bool,
        threads: usize,
        min_probability: f64,
    ) -> Result<Self> {
        if threads == 0 || !min_probability.is_finite() || !(0.0..=1.0).contains(&min_probability) {
            return Err(Error::Invalid(
                "positive threads and probability in [0,1] required".into(),
            ));
        }
        let config: Config =
            serde_json::from_slice(&std::fs::read(directory.join("decision.json")).map_err(err)?)
                .map_err(err)?;
        if config.engine != "onnx"
            || config.layout != "laya-markers-v1"
            || !(16..=8192).contains(&config.max_len)
            || config.head_max_len >= config.max_len
            || config.min_markers < 2
            || config.min_markers > 255
        {
            return Err(Error::Invalid(
                "unsupported Laya ONNX layout or context".into(),
            ));
        }
        let mut tokenizer = Tokenizer::from_file(directory.join("tokenizer.json")).map_err(err)?;
        tokenizer.with_truncation(None).map_err(err)?;
        tokenizer.with_padding(None);
        let calibration: Value = serde_json::from_slice(
            &std::fs::read(directory.join("calibration.json")).map_err(err)?,
        )
        .map_err(err)?;
        if calibration
            .get("temperature_map")
            .is_some_and(|v| !v.is_null())
        {
            return Err(Error::Invalid(
                "input-conditioned temperature map is unsupported by this Laya backend".into(),
            ));
        }
        let mut hasher = Sha256::new();
        let mut weights_digest = Sha256::new();
        // The graph references the author's external safetensors file.
        for name in [
            "model.onnx",
            "model.safetensors",
            "tokenizer.json",
            "decision.json",
            "calibration.json",
        ] {
            hasher.update(name.as_bytes());
            let mut f = File::open(directory.join(name)).map_err(err)?;
            let mut buffer = vec![0; 65536];
            loop {
                let n = f.read(&mut buffer).map_err(err)?;
                if n == 0 {
                    break;
                }
                hasher.update(&buffer[..n]);
                if name == "model.safetensors" {
                    weights_digest.update(&buffer[..n]);
                }
            }
        }
        let weights_digest = format!("{:x}", weights_digest.finalize());
        let mut external =
            File::open(directory.join(format!("sha256-{weights_digest}"))).map_err(err)?;
        let mut external_digest = Sha256::new();
        let mut buffer = vec![0; 65536];
        loop {
            let n = external.read(&mut buffer).map_err(err)?;
            if n == 0 {
                break;
            }
            external_digest.update(&buffer[..n]);
        }
        if format!("{:x}", external_digest.finalize()) != weights_digest {
            return Err(Error::Invalid(
                "ONNX external weights do not match model.safetensors".into(),
            ));
        }
        let device = if cuda { "cuda" } else { "cpu" }.to_string();
        hasher.update(format!(
            "l2s1-laya-strict-v1:{device}:{threads}:{min_probability}"
        ));
        hasher.update(include_bytes!("mod.rs"));
        hasher.update(include_bytes!("pyjson.rs"));
        hasher.update(include_bytes!("../discriminative.rs"));
        let runtime = std::env::var_os("ORT_DYLIB_PATH").ok_or_else(|| {
            Error::Invalid("set ORT_DYLIB_PATH to the ONNX Runtime shared library".into())
        })?;
        if cuda {
            let filename = if cfg!(target_os = "windows") {
                "onnxruntime_providers_cuda.dll"
            } else {
                "libonnxruntime_providers_cuda.so"
            };
            let path = Path::new(&runtime)
                .parent()
                .unwrap_or(Path::new("."))
                .join(filename);
            let mut provider = File::open(path).map_err(err)?;
            let mut bytes = vec![0; 65536];
            loop {
                let n = provider.read(&mut bytes).map_err(err)?;
                if n == 0 {
                    break;
                }
                hasher.update(&bytes[..n]);
            }
        }
        let mut runtime = File::open(runtime).map_err(err)?;
        let mut buffer = vec![0; 65536];
        loop {
            let n = runtime.read(&mut buffer).map_err(err)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let identity = format!("{:x}", hasher.finalize());
        let mut builder = Session::builder()
            .map_err(err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(err)?
            .with_intra_threads(threads)
            .map_err(err)?;
        if cuda {
            #[cfg(feature = "onnx-cuda")]
            {
                builder = builder
                    .with_execution_providers([ort::ep::CUDA::default()
                        .with_tf32(false)
                        .build()
                        .error_on_failure()])
                    .map_err(err)?;
            }
            #[cfg(not(feature = "onnx-cuda"))]
            {
                return Err(Error::Invalid("rebuild with onnx-cuda to use CUDA".into()));
            }
        }
        let session = builder
            .commit_from_file(directory.join("model.onnx"))
            .map_err(err)?;
        Ok(Self {
            session,
            tokenizer,
            config,
            calibration,
            identity,
            device,
            min_probability,
        })
    }
    pub fn identity(&self) -> &str {
        &self.identity
    }
    fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(err)
    }
    fn encode(&self, state: &[u32], decision: &Decision) -> Result<Row> {
        let (qtype, name) = match decision.kind {
            DecisionKind::Choice { .. } => (0, "choice"),
            DecisionKind::Ordinal { .. } => (1, "score"),
            DecisionKind::Binary { .. } => (2, "noul"),
        };
        let options = decision.options();
        if options.len() > 255 || (qtype == 1 && options.len() > 10) {
            return Err(Error::Invalid(
                "Laya supports at most 255 choices or 10 ordinal levels".into(),
            ));
        }
        let sp = &self.config.special_tokens;
        let clean = |s: &str| s.replace(&sp.mask_text, " ");
        let head = self.tokenize(&format!(
            "{name} question: {}",
            clean(&decision.instruction)
        ))?;
        let mut ids = vec![sp.cls];
        ids.extend(&head);
        ids.push(sp.sep);
        let mut markers = Vec::new();
        let mut option_count = 0;
        for (i, opt) in options.iter().enumerate() {
            let text = if qtype == 1 {
                format!("level {i}: {}", opt.criterion)
            } else if opt.criterion.is_empty() {
                opt.id.clone()
            } else {
                format!("{}: {}", opt.id, opt.criterion)
            };
            let tokens = self.tokenize(&format!(" {}", clean(&text)))?;
            // Upstream truncates options/instructions. Fail closed instead of silently changing rules.
            if tokens.len() > 48 {
                return Err(Error::Invalid(
                    "Laya option exceeds 48 tokens; truncation disabled".into(),
                ));
            }
            option_count += 1 + tokens.len();
            markers.push(ids.len());
            ids.push(sp.mask);
            ids.extend(tokens);
        }
        let instruction_budget = self.config.head_max_len.saturating_sub(option_count);
        if instruction_budget < 16 || head.len() > instruction_budget.max(8) {
            return Err(Error::Invalid(
                "Laya instruction/options exceed head budget; truncation disabled".into(),
            ));
        }
        ids.push(sp.sep);
        ids.extend_from_slice(state);
        ids.push(sp.sep);
        if ids.len() > self.config.max_len {
            return Err(Error::Invalid(
                "Laya input exceeds context; truncation disabled".into(),
            ));
        }
        Ok(Row {
            ids,
            markers,
            qtype,
        })
    }
    fn temperature(&self, row: &Row) -> f64 {
        let name = ["choice", "score", "noul"][row.qtype];
        let k = row.markers.len();
        let bucket = if k <= 2 {
            "2"
        } else if k <= 5 {
            "3-5"
        } else if k <= 10 {
            "6-10"
        } else {
            "11+"
        };
        self.calibration["temperature_by_options"][format!("{name}:{bucket}")]
            .as_f64()
            .or_else(|| self.calibration["temperature"][row.qtype].as_f64())
            .unwrap_or(1.0)
            .clamp(0.5, 5.0)
    }
    fn forward(&mut self, rows: &[Row]) -> Result<Vec<Vec<f32>>> {
        let n = rows.len();
        let seq = rows.iter().map(|r| r.ids.len()).max().unwrap_or(0);
        let k = rows
            .iter()
            .map(|r| r.markers.len())
            .max()
            .unwrap_or(2)
            .max(self.config.min_markers);
        let mut ids = Array2::<i64>::from_elem((n, seq), self.config.special_tokens.pad.into());
        let mut attention = Array2::<i64>::zeros((n, seq));
        let mut positions = Array2::<i64>::zeros((n, k));
        let mut mask = Array2::<bool>::from_elem((n, k), false);
        for (i, row) in rows.iter().enumerate() {
            for (j, &id) in row.ids.iter().enumerate() {
                ids[(i, j)] = id.into();
                attention[(i, j)] = 1;
            }
            for (j, &p) in row.markers.iter().enumerate() {
                positions[(i, j)] = p as i64;
                mask[(i, j)] = true;
            }
        }
        let qtypes = ndarray::Array1::from_iter(rows.iter().map(|r| r.qtype as i64));
        let outputs=self.session.run(ort::inputs!{
            "input_ids"=>Tensor::from_array(ids).map_err(err)?,"attention_mask"=>Tensor::from_array(attention).map_err(err)?,
            "marker_pos"=>Tensor::from_array(positions).map_err(err)?,"marker_mask"=>Tensor::from_array(mask).map_err(err)?,
            "qtype"=>Tensor::from_array(qtypes).map_err(err)?}).map_err(err)?;
        let logits = outputs["logits"].try_extract_array::<f32>().map_err(err)?;
        if logits.shape() != [n, k] {
            return Err(err("unexpected ONNX logits shape"));
        }
        Ok(rows
            .iter()
            .enumerate()
            .map(|(i, r)| (0..r.markers.len()).map(|j| logits[[i, j]]).collect())
            .collect())
    }
    pub fn inspect_input(&self, request: &DecisionRequest) -> Result<Value> {
        request.validate()?;
        let text = match &request.state {
            Value::String(s) => s.clone(),
            v => pyjson::dumps(v, false),
        };
        let state = self.tokenize(&text.replace(&self.config.special_tokens.mask_text, " "))?;
        request
            .decisions
            .iter()
            .map(|d| {
                self.encode(&state, d).map(
                    |row| json!({"id":d.id,"ids":row.ids,"markers":row.markers,"qtype":row.qtype}),
                )
            })
            .collect::<Result<Vec<_>>>()
            .map(|v| json!(v))
    }
    pub fn decide_many(&mut self, requests: &[DecisionRequest]) -> Result<Vec<Value>> {
        if requests.len() > 128 || requests.iter().map(|r| r.decisions.len()).sum::<usize>() > 128 {
            return Err(Error::Invalid(
                "ONNX batch permits at most 128 requests and decisions".into(),
            ));
        }
        let mut rows = Vec::new();
        for request in requests {
            request.validate()?;
            let text = match &request.state {
                Value::String(s) => s.clone(),
                v => pyjson::dumps(v, false),
            };
            let state = self.tokenize(&text.replace(&self.config.special_tokens.mask_text, " "))?;
            for d in &request.decisions {
                rows.push(self.encode(&state, d)?);
            }
        }
        let mut logits = Vec::with_capacity(rows.len());
        let mut start = 0;
        while start < rows.len() {
            let mut end = start;
            let mut longest = 0;
            while end < rows.len() {
                let next = longest.max(rows[end].ids.len());
                if next * (end - start + 1) > 32768 {
                    break;
                }
                longest = next;
                end += 1;
            }
            if end == start {
                return Err(err("row exceeds batch budget"));
            }
            logits.extend(self.forward(&rows[start..end])?);
            start = end;
        }
        let mut offset = 0;
        let mut outputs = Vec::new();
        for request in requests {
            let mut results = Vec::new();
            for d in &request.decisions {
                let row = &rows[offset];
                results.push(crate::discriminative::score(
                    d,
                    &logits[offset],
                    self.temperature(row),
                    self.min_probability,
                    &self.identity,
                    row.ids.len(),
                )?);
                offset += 1;
            }
            outputs.push(json!({"backend":{"runtime":"onnx-laya","model":self.identity,"details":{"device":self.device,"layout":"laya-markers-v1","strict_input":true}},
                "policy":null,"results":results}));
        }
        Ok(outputs)
    }
}
impl HttpDecisionBackend for OnnxBackend {
    fn capabilities(&self) -> Value {
        json!({"api_version":1,"backend":{"runtime":"onnx-laya","model":self.identity},
        "artifact_id":self.identity,"decision_types":["binary","choice","ordinal"],"evidence":"discriminative",
        "media":{"image":{"supported":false,"max_per_decision":0,"max_bytes_each":0}},
        "limits":{"max_decisions":128,"context_tokens":self.config.max_len},
        "request_policy":{"supported":false,"target_error_rate":"unsupported; no language-model candidate mass"},
        "batch":{"supported":true,"enabled":true,"execution":"native_parallel","max_requests":128,"max_decisions":128,
            "max_decisions_per_wave":128,"text":true,"image":false,"mixed_media":false,"reasoning_modes":["direct"]}})
    }
    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> Result<Value> {
        if !images.is_empty() {
            return Err(Error::Invalid("Laya has no image input".into()));
        }
        Ok(self.decide_many(std::slice::from_ref(request))?.remove(0))
    }
    fn decide_native_batch_json(
        &mut self,
        requests: &[DecisionRequest],
        images: &[Vec<&[u8]>],
    ) -> Result<Vec<Value>> {
        if requests.len() != images.len() || images.iter().any(|i| !i.is_empty()) {
            return Err(Error::Invalid(
                "Laya batch requires matching text-only requests".into(),
            ));
        }
        self.decide_many(requests)
    }
}
