//! Still-image tokenization, projector reuse and independent decoder scheduling.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]
use crate::{NativeVisionBatchMetrics, NativeVisionInput, bridge::*, raw::*};
use std::{
    ffi::{CStr, c_char},
    ptr, slice,
    sync::Arc,
};
struct Bitmap(*mut mtmd_bitmap);
impl Drop for Bitmap {
    fn drop(&mut self) {
        unsafe {
            mtmd_bitmap_free(self.0);
        }
    }
}
struct Chunks(*mut mtmd_input_chunks);
impl Drop for Chunks {
    fn drop(&mut self) {
        unsafe {
            mtmd_input_chunks_free(self.0);
        }
    }
}
struct ProjectorBatch(*mut mtmd_batch);
impl Drop for ProjectorBatch {
    fn drop(&mut self) {
        unsafe {
            mtmd_batch_free(self.0);
        }
    }
}
struct Attention {
    ctx: *mut llama_context,
    non_causal: bool,
}
impl Drop for Attention {
    fn drop(&mut self) {
        if self.non_causal {
            unsafe {
                llama_set_causal_attn(self.ctx, true);
            }
        }
    }
}

unsafe fn tokenize(e: &Engine, input: &NativeVisionInput) -> Result<Chunks> {
    if e.vision.is_null()
        || input.prefix.is_null()
        || input.data_before.is_null()
        || input.image.is_null()
        || input.image_len == 0
        || input.data_after.is_null()
        || input.suffix.is_null()
    {
        return Err("invalid vision inference arguments".into());
    }
    let wrapped = mtmd_helper_bitmap_init_from_buf(
        e.vision,
        input.image,
        input.image_len,
        false,
        mtmd_helper_init_opt_default(),
    );
    let bitmap = Bitmap(wrapped.bitmap);
    if !wrapped.video_ctx.is_null() {
        mtmd_helper_video_free(wrapped.video_ctx);
        return Err("video input is unsupported; provide one still image".into());
    }
    if bitmap.0.is_null() || mtmd_bitmap_is_audio(bitmap.0) {
        return Err("image must be a supported still-image format".into());
    }
    let texts = [
        mtmd_input_text {
            text: input.prefix,
            text_len: input.prefix_len,
            add_special: false,
            parse_special: true,
        },
        mtmd_input_text {
            text: input.data_before,
            text_len: input.before_len,
            add_special: false,
            parse_special: false,
        },
        mtmd_input_text {
            text: input.data_after,
            text_len: input.after_len,
            add_special: false,
            parse_special: false,
        },
        mtmd_input_text {
            text: input.suffix,
            text_len: input.suffix_len,
            add_special: false,
            parse_special: true,
        },
    ];
    let parts = [
        mtmd_input_part {
            text: &texts[0],
            bitmap: ptr::null(),
        },
        mtmd_input_part {
            text: &texts[1],
            bitmap: ptr::null(),
        },
        mtmd_input_part {
            text: ptr::null(),
            bitmap: bitmap.0,
        },
        mtmd_input_part {
            text: &texts[2],
            bitmap: ptr::null(),
        },
        mtmd_input_part {
            text: &texts[3],
            bitmap: ptr::null(),
        },
    ];
    let ptrs = parts.each_ref().map(|p| p as *const mtmd_input_part);
    let chunks = Chunks(mtmd_input_chunks_init());
    if chunks.0.is_null()
        || mtmd_tokenize_from_parts(e.vision, chunks.0, ptrs.as_ptr(), ptrs.len(), true) != 0
    {
        return Err("vision prompt tokenization failed".into());
    }
    Ok(chunks)
}
unsafe fn copy_output(
    e: &Engine,
    output: *const f32,
    ids: Option<&[i32]>,
    logits: &mut [f32],
    normalizer: *mut f64,
) -> Result<()> {
    if output.is_null() {
        return Err("missing vision final logits".into());
    }
    let output = slice::from_raw_parts(output, e.vocab() as usize);
    if let Some(ids) = ids {
        *normalizer = compact(output, ids, logits)?;
    } else {
        logits.copy_from_slice(output);
    }
    Ok(())
}
unsafe fn forward_vision(
    e: *mut Engine,
    input: NativeVisionInput,
    continuation: *const i32,
    continuation_count: usize,
    logits: *mut f32,
    logits_count: usize,
    input_tokens: *mut usize,
    candidate_ids: *const i32,
    candidate_count: usize,
    log_normalizer: *mut f64,
    error: *mut c_char,
    cap: usize,
) -> bool {
    let ok = run(e, error, cap, |e| {
        if logits.is_null()
            || input_tokens.is_null()
            || (continuation_count > 0 && continuation.is_null())
            || logits_count
                != if candidate_ids.is_null() {
                    e.vocab() as usize
                } else {
                    candidate_count
                }
        {
            return Err("invalid vision inference arguments".into());
        }
        e.vision_metrics = NativeVisionBatchMetrics::default();
        let ids = if candidate_ids.is_null() {
            None
        } else {
            if log_normalizer.is_null() || !(2..=26).contains(&candidate_count) {
                return Err("missing vision compact normalizer or invalid candidates".into());
            }
            let ids = slice::from_raw_parts(candidate_ids, candidate_count);
            validate_candidates(ids, e.vocab())?;
            Some(ids)
        };
        e.ensure_sequences(1, 0, false, true)?;
        e.clear();
        let chunks = tokenize(e, &input)?;
        let tokens = mtmd_helper_get_n_tokens(chunks.0);
        let positions = mtmd_helper_get_n_pos(chunks.0);
        if tokens == 0
            || tokens > e.context_size as usize
            || positions <= 0
            || positions as u32 > e.context_size
            || continuation_count > e.context_size as usize - positions as usize
        {
            return Err("vision input exceeds context; truncation is disabled".into());
        }
        let mut end = 0;
        e.memory_dirty = true;
        if mtmd_helper_eval_chunks(
            e.vision,
            e.ctx,
            chunks.0,
            0,
            0,
            e.batch_size as i32,
            true,
            &mut end,
        ) != 0
            || end != positions
        {
            return Err("vision decode failed".into());
        }
        if continuation_count > 0 {
            let continuation = slice::from_raw_parts(continuation, continuation_count);
            let mut b = Batch::new(e.batch_size)?;
            for (part, ts) in continuation.chunks(e.batch_size as usize).enumerate() {
                b.0.n_tokens = 0;
                for (j, &token) in ts.iter().enumerate() {
                    let index = part * e.batch_size as usize + j;
                    if token < 0 || token >= e.vocab() {
                        return Err("vision continuation token outside vocabulary".into());
                    }
                    b.add(
                        token,
                        end + index as i32,
                        0,
                        index + 1 == continuation_count,
                    );
                }
                e.memory_dirty = true;
                if llama_decode(e.ctx, b.0) != 0 {
                    return Err("vision continuation decode failed".into());
                }
            }
        }
        copy_output(
            e,
            llama_get_logits_ith(e.ctx, -1),
            ids,
            slice::from_raw_parts_mut(logits, logits_count),
            log_normalizer,
        )?;
        *input_tokens = tokens;
        Ok(())
    });
    sd_clear(e);
    ok
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_vision(
    e: *mut Engine,
    prefix: *const c_char,
    prefix_len: usize,
    data_before: *const c_char,
    before_len: usize,
    image: *const u8,
    image_len: usize,
    data_after: *const c_char,
    after_len: usize,
    suffix: *const c_char,
    suffix_len: usize,
    continuation: *const i32,
    continuation_count: usize,
    logits: *mut f32,
    logits_count: usize,
    input_tokens: *mut usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    forward_vision(
        e,
        NativeVisionInput {
            prefix,
            prefix_len,
            data_before,
            before_len,
            image,
            image_len,
            data_after,
            after_len,
            suffix,
            suffix_len,
        },
        continuation,
        continuation_count,
        logits,
        logits_count,
        input_tokens,
        ptr::null(),
        0,
        ptr::null_mut(),
        error,
        cap,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_vision_compact(
    e: *mut Engine,
    prefix: *const c_char,
    prefix_len: usize,
    data_before: *const c_char,
    before_len: usize,
    image: *const u8,
    image_len: usize,
    data_after: *const c_char,
    after_len: usize,
    suffix: *const c_char,
    suffix_len: usize,
    continuation: *const i32,
    continuation_count: usize,
    candidate_ids: *const i32,
    candidate_count: usize,
    logits: *mut f32,
    logits_count: usize,
    log_normalizer: *mut f64,
    input_tokens: *mut usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if candidate_ids.is_null() || log_normalizer.is_null() {
        sd_clear(e);
        report(error, cap, "null vision compact argument");
        return false;
    }
    forward_vision(
        e,
        NativeVisionInput {
            prefix,
            prefix_len,
            data_before,
            before_len,
            image,
            image_len,
            data_after,
            after_len,
            suffix,
            suffix_len,
        },
        continuation,
        continuation_count,
        logits,
        logits_count,
        input_tokens,
        candidate_ids,
        candidate_count,
        log_normalizer,
        error,
        cap,
    )
}
struct Segment {
    chunk: *const mtmd_input_chunk,
    embeddings: Option<Arc<[f32]>>,
    offset: usize,
    image_ordinal: usize,
}
struct Sequence {
    _chunks: Chunks,
    segments: Vec<Segment>,
    current: usize,
    position: llama_pos,
}
impl Sequence {
    unsafe fn advance(&mut self) {
        while let Some(seg) = self.segments.get(self.current) {
            if seg.offset < mtmd_input_chunk_get_n_tokens(seg.chunk) {
                break;
            }
            self.position += mtmd_input_chunk_get_n_pos(seg.chunk);
            self.current += 1;
        }
    }
}
#[derive(Clone, Copy)]
struct Row {
    seq: usize,
    segment: usize,
    offset: usize,
    base: llama_pos,
    output: bool,
}
unsafe fn equal_image(
    e: &Engine,
    a: &Segment,
    b: &Segment,
    ai: &NativeVisionInput,
    bi: &NativeVisionInput,
    mrope: bool,
) -> bool {
    if a.image_ordinal != b.image_ordinal
        || ai.image_len != bi.image_len
        || slice::from_raw_parts(ai.image, ai.image_len)
            != slice::from_raw_parts(bi.image, bi.image_len)
        || mtmd_input_chunk_get_n_tokens(a.chunk) != mtmd_input_chunk_get_n_tokens(b.chunk)
        || mtmd_input_chunk_get_n_pos(a.chunk) != mtmd_input_chunk_get_n_pos(b.chunk)
        || mtmd_decode_use_non_causal(e.vision, a.chunk)
            != mtmd_decode_use_non_causal(e.vision, b.chunk)
    {
        return false;
    }
    let aid = mtmd_input_chunk_get_id(a.chunk);
    let bid = mtmd_input_chunk_get_id(b.chunk);
    if aid.is_null() != bid.is_null()
        || (!aid.is_null() && CStr::from_ptr(aid) != CStr::from_ptr(bid))
    {
        return false;
    }
    if mrope {
        let at = mtmd_input_chunk_get_tokens_image(a.chunk);
        let bt = mtmd_input_chunk_get_tokens_image(b.chunk);
        if at.is_null() || bt.is_null() {
            return false;
        }
        for t in 0..mtmd_input_chunk_get_n_tokens(a.chunk) {
            let ap = mtmd_image_tokens_get_decoder_pos(at, 0, t);
            let bp = mtmd_image_tokens_get_decoder_pos(bt, 0, t);
            if (ap.t, ap.y, ap.x, ap.z) != (bp.t, bp.y, bp.x, bp.z) {
                return false;
            }
        }
    }
    true
}
unsafe fn forward_parallel(
    e: *mut Engine,
    inputs: *const NativeVisionInput,
    sequences: i32,
    capacity: u32,
    dynamic: bool,
    logits: *mut f32,
    logits_count: usize,
    input_tokens: *mut usize,
    candidate_ids: *const *const i32,
    candidate_counts: *const usize,
    normalizers: *mut f64,
    error: *mut c_char,
    cap: usize,
) -> bool {
    let ok = run(e, error, cap, |e| {
        if e.vision.is_null()
            || inputs.is_null()
            || logits.is_null()
            || input_tokens.is_null()
            || sequences < 1
            || sequences as u32 > capacity
            || capacity > 32
        {
            return Err("invalid parallel vision arguments".into());
        }
        e.vision_metrics = NativeVisionBatchMetrics::default();
        if e.recurrent() {
            return Err("parallel vision is unsupported for recurrent/hybrid models".into());
        }
        let inputs = slice::from_raw_parts(inputs, sequences as usize);
        let vocab = e.vocab() as usize;
        let mut candidate_offsets = vec![0; inputs.len() + 1];
        let mut candidates = Vec::new();
        if !candidate_ids.is_null() {
            if candidate_counts.is_null() || normalizers.is_null() {
                return Err("missing parallel compact arguments".into());
            }
            for i in 0..inputs.len() {
                let count = *candidate_counts.add(i);
                let ids = *candidate_ids.add(i);
                if ids.is_null() || !(2..=26).contains(&count) {
                    return Err("invalid compact candidate bank".into());
                }
                let ids = slice::from_raw_parts(ids, count);
                validate_candidates(ids, vocab as i32)?;
                candidates.push(ids);
                candidate_offsets[i + 1] = candidate_offsets[i] + count;
            }
        }
        if logits_count
            != if candidates.is_empty() {
                vocab * inputs.len()
            } else {
                *candidate_offsets.last().unwrap()
            }
        {
            return Err("wrong parallel vision logits buffer size".into());
        }
        let mut seqs = Vec::new();
        let mut media = Vec::new();
        let mut maximum_tokens = 0;
        let n_embd = llama_model_n_embd_inp(e.model) as usize;
        let mrope = mtmd_decode_use_mrope(e.vision);
        for (s, input) in inputs.iter().enumerate() {
            let chunks = tokenize(e, input)?;
            let tokens = mtmd_helper_get_n_tokens(chunks.0);
            let positions = mtmd_helper_get_n_pos(chunks.0);
            if tokens == 0
                || tokens > e.context_size as usize
                || positions <= 0
                || positions as u32 > e.context_size
            {
                return Err(
                    "parallel vision input exceeds per-question context; truncation is disabled"
                        .into(),
                );
            }
            *input_tokens.add(s) = tokens;
            maximum_tokens = maximum_tokens.max(tokens);
            let mut segments = Vec::new();
            let mut image_ordinal = 0;
            for c in 0..mtmd_input_chunks_size(chunks.0) {
                let chunk = mtmd_input_chunks_get(chunks.0, c);
                if mtmd_input_chunk_get_n_tokens(chunk) == 0 {
                    continue;
                }
                let kind = mtmd_input_chunk_get_type(chunk);
                if kind != mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT
                    && kind != mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_IMAGE
                {
                    return Err("parallel vision only supports text and image chunks".into());
                }
                let ordinal = if kind == mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_IMAGE {
                    let ordinal = image_ordinal;
                    image_ordinal += 1;
                    media.push((s, segments.len()));
                    ordinal
                } else {
                    0
                };
                segments.push(Segment {
                    chunk,
                    embeddings: None,
                    offset: 0,
                    image_ordinal: ordinal,
                });
            }
            if segments.last().is_none_or(|seg| {
                mtmd_input_chunk_get_type(seg.chunk)
                    != mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT
            }) {
                return Err(
                    "parallel vision requires a final text suffix for decision logits".into(),
                );
            }
            seqs.push(Sequence {
                _chunks: chunks,
                segments,
                current: 0,
                position: 0,
            });
        }
        // Indices replace C++ segment pointers, so Vec growth cannot invalidate a reference.
        let mut duplicates = Vec::new();
        if e.vision_projector_reuse {
            let mut unique: Vec<(usize, usize)> = Vec::new();
            for (s, c) in media {
                let same = unique.iter().copied().find(|&(bs, bc)| {
                    equal_image(
                        e,
                        &seqs[s].segments[c],
                        &seqs[bs].segments[bc],
                        &inputs[s],
                        &inputs[bs],
                        mrope,
                    )
                });
                if let Some(other) = same {
                    duplicates.push(((s, c), other));
                } else {
                    unique.push((s, c));
                }
            }
            media = unique;
        }
        let maximum = u64::from(e.context_size) * u64::from(capacity);
        if maximum > i32::MAX as u64 {
            return Err("parallel vision context exceeds INT_MAX".into());
        }
        let per_sequence =
            u64::from(e.context_size).min(maximum_tokens as u64 + u64::from(e.batch_size));
        let requested = if dynamic && capacity > 1 {
            per_sequence * u64::from(capacity)
        } else {
            maximum
        };
        e.ensure_sequences(capacity, requested as u32, dynamic, false)?;
        e.clear();
        let mut start = 0;
        while start < media.len() {
            let batch = ProjectorBatch(mtmd_batch_init(e.vision));
            if batch.0.is_null() {
                return Err("vision projector batch allocation failed".into());
            }
            let mut end = start;
            while end < media.len() {
                let (s, c) = media[end];
                let rc = mtmd_batch_add_chunk(batch.0, seqs[s].segments[c].chunk);
                if rc == 0 {
                    end += 1;
                    if e.vision_projector_reuse {
                        break;
                    }
                    continue;
                }
                if end > start && (rc == 2 || rc == 3) {
                    break;
                }
                return Err("vision projector batch preparation failed".into());
            }
            if mtmd_batch_encode(batch.0) != 0 {
                return Err("vision projector batch encoding failed".into());
            }
            e.vision_metrics.projector_encode_calls += 1;
            e.vision_metrics.projector_batch_max =
                e.vision_metrics.projector_batch_max.max(end - start);
            for &(s, c) in &media[start..end] {
                let seg = &mut seqs[s].segments[c];
                let embd = mtmd_batch_get_output_embd(batch.0, seg.chunk);
                if embd.is_null() {
                    return Err("missing batched vision embeddings".into());
                }
                let count = mtmd_input_chunk_get_n_tokens(seg.chunk)
                    .checked_mul(n_embd)
                    .ok_or("vision embedding size overflow")?;
                seg.embeddings = Some(Arc::from(slice::from_raw_parts(embd, count)));
            }
            start = end;
        }
        for &((s, c), (bs, bc)) in &duplicates {
            seqs[s].segments[c].embeddings = seqs[bs].segments[bc].embeddings.clone();
        }
        e.vision_metrics.projector_reused_chunks = duplicates.len();
        let mut completed = 0;
        while completed < inputs.len() {
            for seq in &mut seqs {
                seq.advance();
            }
            let text_mode = seqs.iter().any(|seq| {
                seq.segments.get(seq.current).is_some_and(|seg| {
                    mtmd_input_chunk_get_type(seg.chunk)
                        == mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT
                })
            });
            let non_causal = !text_mode
                && seqs
                    .iter()
                    .find_map(|seq| {
                        seq.segments
                            .get(seq.current)
                            .map(|seg| mtmd_decode_use_non_causal(e.vision, seg.chunk))
                    })
                    .unwrap_or(false);
            let limit = if non_causal {
                e.batch_size.min(llama_n_ubatch(e.ctx))
            } else {
                e.batch_size
            } as usize;
            let mut rows = Vec::new();
            if non_causal {
                for (s, seq) in seqs.iter_mut().enumerate() {
                    let Some(seg) = seq.segments.get_mut(seq.current) else {
                        continue;
                    };
                    if !mtmd_decode_use_non_causal(e.vision, seg.chunk) {
                        continue;
                    }
                    let count = mtmd_input_chunk_get_n_tokens(seg.chunk);
                    if count > limit {
                        return Err(
                            "non-causal vision image requires batch and ubatch >= image tokens"
                                .into(),
                        );
                    }
                    if rows.len() + count > limit {
                        continue;
                    }
                    while seg.offset < count {
                        rows.push(Row {
                            seq: s,
                            segment: seq.current,
                            offset: seg.offset,
                            base: seq.position,
                            output: false,
                        });
                        seg.offset += 1;
                    }
                }
            } else {
                let mut progress = true;
                while progress && rows.len() < limit {
                    progress = false;
                    for (s, seq) in seqs.iter_mut().enumerate() {
                        if rows.len() >= limit {
                            break;
                        }
                        seq.advance();
                        let last_segment = seq.current + 1 == seq.segments.len();
                        let Some(seg) = seq.segments.get_mut(seq.current) else {
                            continue;
                        };
                        let text = mtmd_input_chunk_get_type(seg.chunk)
                            == mtmd_input_chunk_type_MTMD_INPUT_CHUNK_TYPE_TEXT;
                        if text != text_mode
                            || (!text && mtmd_decode_use_non_causal(e.vision, seg.chunk))
                        {
                            continue;
                        }
                        let output = last_segment
                            && seg.offset + 1 == mtmd_input_chunk_get_n_tokens(seg.chunk);
                        rows.push(Row {
                            seq: s,
                            segment: seq.current,
                            offset: seg.offset,
                            base: seq.position,
                            output,
                        });
                        seg.offset += 1;
                        progress = true;
                    }
                }
            }
            if rows.is_empty() {
                return Err("parallel vision scheduler made no progress".into());
            }
            let count = rows.len();
            let axes = if !text_mode && mrope { 4 } else { 1 };
            let mut tokens = vec![0; if text_mode { count } else { 0 }];
            let mut embeddings = vec![0.0; if text_mode { 0 } else { count * n_embd }];
            let mut positions = vec![0; count * axes];
            let mut n_seq = vec![1; count];
            let mut ids = vec![0; count];

            let mut outputs = vec![0; count];
            let mut participating = vec![false; inputs.len()];
            for (i, row) in rows.iter().enumerate() {
                let seg = &seqs[row.seq].segments[row.segment];
                ids[i] = row.seq as i32;
                outputs[i] = i8::from(row.output);
                participating[row.seq] = true;
                if text_mode {
                    let mut size = 0;
                    let ts = mtmd_input_chunk_get_tokens_text(seg.chunk, &mut size);
                    if ts.is_null() || row.offset >= size {
                        return Err("missing image text tokens".into());
                    }
                    tokens[i] = *ts.add(row.offset);
                    positions[i] = row.base + row.offset as i32;
                } else {
                    let embd = seg.embeddings.as_ref().ok_or("missing image embeddings")?;
                    embeddings[i * n_embd..(i + 1) * n_embd]
                        .copy_from_slice(&embd[row.offset * n_embd..(row.offset + 1) * n_embd]);
                    if mrope {
                        let image = mtmd_input_chunk_get_tokens_image(seg.chunk);
                        if image.is_null() {
                            return Err("missing image position metadata".into());
                        }
                        let pos = mtmd_image_tokens_get_decoder_pos(image, row.base, row.offset);
                        positions[i] = pos.t as i32;
                        positions[i + count] = pos.y as i32;
                        positions[i + count * 2] = pos.x as i32;
                        positions[i + count * 3] = pos.z as i32;
                    } else {
                        positions[i] = row.base + row.offset as i32;
                    }
                }
            }
            let mut id_ptrs: Vec<_> = ids.iter_mut().map(|id| id as *mut i32).collect();
            let batch = llama_batch {
                n_tokens: count as i32,
                token: if text_mode {
                    tokens.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                embd: if text_mode {
                    ptr::null_mut()
                } else {
                    embeddings.as_mut_ptr()
                },
                pos: positions.as_mut_ptr(),
                n_seq_id: n_seq.as_mut_ptr(),
                seq_id: id_ptrs.as_mut_ptr(),
                logits: outputs.as_mut_ptr(),
            };
            let _guard = Attention {
                ctx: e.ctx,
                non_causal,
            };
            if non_causal {
                llama_set_causal_attn(e.ctx, false);
            }
            e.memory_dirty = true;
            let rc = llama_decode(e.ctx, batch);
            if rc != 0 {
                return Err(format!("parallel vision decode failed (llama_decode={rc})"));
            }
            e.vision_metrics.decoder_calls += 1;
            e.vision_metrics.decoder_batch_max_sequences = e
                .vision_metrics
                .decoder_batch_max_sequences
                .max(participating.iter().filter(|&&p| p).count());
            for (i, row) in rows.iter().enumerate().filter(|(_, r)| r.output) {
                let output = llama_get_logits_ith(e.ctx, i as i32);
                let s = row.seq;
                if candidates.is_empty() {
                    copy_output(
                        e,
                        output,
                        None,
                        slice::from_raw_parts_mut(logits.add(s * vocab), vocab),
                        ptr::null_mut(),
                    )?;
                } else {
                    copy_output(
                        e,
                        output,
                        Some(candidates[s]),
                        slice::from_raw_parts_mut(
                            logits.add(candidate_offsets[s]),
                            candidates[s].len(),
                        ),
                        normalizers.add(s),
                    )?;
                }
                completed += 1;
            }
        }
        Ok(())
    });
    sd_clear(e);
    ok
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_vision_parallel(
    e: *mut Engine,
    inputs: *const NativeVisionInput,
    sequences: i32,
    capacity: u32,
    dynamic: bool,
    logits: *mut f32,
    logits_count: usize,
    input_tokens: *mut usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    forward_parallel(
        e,
        inputs,
        sequences,
        capacity,
        dynamic,
        logits,
        logits_count,
        input_tokens,
        ptr::null(),
        ptr::null(),
        ptr::null_mut(),
        error,
        cap,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_vision_parallel_compact(
    e: *mut Engine,
    inputs: *const NativeVisionInput,
    sequences: i32,
    capacity: u32,
    dynamic: bool,
    candidate_ids: *const *const i32,
    candidate_counts: *const usize,
    logits: *mut f32,
    logits_count: usize,
    normalizers: *mut f64,
    input_tokens: *mut usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if candidate_ids.is_null() || candidate_counts.is_null() || normalizers.is_null() {
        sd_clear(e);
        report(error, cap, "null parallel vision compact argument");
        return false;
    }
    forward_parallel(
        e,
        inputs,
        sequences,
        capacity,
        dynamic,
        logits,
        logits_count,
        input_tokens,
        candidate_ids,
        candidate_counts,
        normalizers,
        error,
        cap,
    )
}
