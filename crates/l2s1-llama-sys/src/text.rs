//! Text decoding, shared prefixes, bounded thinking and state snapshots.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]
use crate::{
    NativeRestoreMetrics, SD_PARALLEL_RETAIN_PREFIX, SD_PARALLEL_TOKEN_PREFIX, bridge::*, raw::*,
};
use std::{ffi::c_char, ptr, slice, time::Instant};

unsafe fn forward_logits(
    e: &mut Engine,
    tokens: &[i32],
    reuse: bool,
    reused: &mut i32,
    boundary: Option<usize>,
) -> Result<*const f32> {
    e.ensure_sequences(1, 0, false, true)?;
    if tokens.is_empty() || tokens.len() > llama_n_ctx(e.ctx) as usize {
        return Err("input exceeds context or is empty; truncation is disabled".into());
    }
    let memory = llama_get_memory(e.ctx);
    if memory.is_null() {
        return Err("decoder memory unavailable".into());
    }
    if boundary.is_some_and(|p| p == 0 || p >= tokens.len())
        || (boundary.is_some() && e.recurrent())
    {
        return Err("invalid fixed prefix boundary or unsupported recurrent model".into());
    }
    let mut common = 0;
    if reuse && !e.recurrent() && e.cached_boundary == boundary {
        common = e
            .cached_tokens
            .iter()
            .zip(tokens)
            .take(tokens.len() - 1)
            .take_while(|(a, b)| a == b)
            .count();
        common = match boundary {
            Some(p) if common >= p => p,
            Some(_) => 0,
            None => common - common % e.batch_size as usize,
        };
    }
    if common == 0 || !llama_memory_seq_rm(memory, 0, common as i32, -1) {
        let cache = if boundary.is_some() && reuse {
            std::mem::take(&mut e.split_cache)
        } else {
            Default::default()
        };
        e.clear();
        e.split_cache = cache;
        common = 0;
        if let Some(p) = boundary
            && reuse
            && let Some(index) = e
                .split_cache
                .iter()
                .position(|(key, _)| key == &tokens[..p])
        {
            let entry = e.split_cache.remove(index).unwrap();
            if llama_state_seq_set_data(e.ctx, entry.1.as_ptr(), entry.1.len(), 0) != entry.1.len()
            {
                return Err("fixed prefix snapshot restore failed".into());
            }
            e.memory_dirty = true;
            e.split_cache.push_back(entry);
            common = p;
        }
    }
    e.cached_tokens.clear();
    let mut batch = Batch::new(e.batch_size)?;
    if let Some(p) = boundary {
        if common == 0 {
            decode_range(e, &mut batch, tokens, 0, p, false)?;
            if reuse {
                let bytes = llama_state_seq_get_size(e.ctx, 0);
                const LIMIT: usize = 256 * 1024 * 1024;
                if bytes > 0 && bytes <= LIMIT {
                    while !e.split_cache.is_empty()
                        && (e.split_cache.len() >= 8
                            || e.split_cache
                                .iter()
                                .map(|(_, data)| data.len())
                                .sum::<usize>()
                                + bytes
                                > LIMIT)
                    {
                        e.split_cache.pop_front();
                    }
                    let mut snapshot = vec![0; bytes];
                    if llama_state_seq_get_data(e.ctx, snapshot.as_mut_ptr(), bytes, 0) != bytes {
                        return Err("fixed prefix snapshot save failed".into());
                    }
                    e.split_cache.push_back((tokens[..p].to_vec(), snapshot));
                }
            }
        }
        decode_range(e, &mut batch, tokens, p, tokens.len(), true)?;
    } else {
        decode_range(e, &mut batch, tokens, common, tokens.len(), true)?;
    }
    let output = llama_get_logits_ith(e.ctx, -1);
    if output.is_null() {
        return Err("missing final logits".into());
    }
    if reuse {
        e.cached_tokens.extend_from_slice(tokens);
    }
    e.cached_boundary = boundary;
    *reused = common as i32;
    Ok(output)
}
unsafe fn decode_range(
    e: &mut Engine,
    b: &mut Batch,
    tokens: &[i32],
    mut start: usize,
    end: usize,
    final_output: bool,
) -> Result<()> {
    while start < end {
        b.0.n_tokens = 0;
        let stop = end.min(start + e.batch_size as usize);
        for (position, &token) in tokens.iter().enumerate().take(stop).skip(start) {
            b.add(
                token,
                position as i32,
                0,
                final_output && position == end - 1,
            );
        }
        e.memory_dirty = true;
        if llama_decode(e.ctx, b.0) != 0 {
            return Err("llama_decode failed".into());
        }
        e.last_feature_row = b.0.n_tokens - 1;
        start = stop;
    }
    Ok(())
}
unsafe fn input<'a>(tokens: *const i32, count: i32) -> Result<&'a [i32]> {
    if tokens.is_null() || count <= 0 {
        return Err("input exceeds context or is empty; truncation is disabled".into());
    }
    Ok(slice::from_raw_parts(tokens, count as usize))
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward(
    e: *mut Engine,
    tokens: *const i32,
    count: i32,
    reuse: bool,
    reused: *mut i32,
    logits: *mut f32,
    logits_count: usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if !reused.is_null() {
        *reused = 0;
    }
    run(e, error, cap, |e| {
        if reused.is_null() || logits.is_null() || logits_count != e.vocab() as usize {
            return Err("wrong logits buffer size".into());
        }
        let output = forward_logits(e, input(tokens, count)?, reuse, &mut *reused, None)?;
        ptr::copy_nonoverlapping(output, logits, logits_count);
        Ok(())
    })
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_compact(
    e: *mut Engine,
    tokens: *const i32,
    count: i32,
    reuse: bool,
    reused: *mut i32,
    candidate_ids: *const i32,
    candidate_count: usize,
    candidate_logits: *mut f32,
    candidate_logits_count: usize,
    log_normalizer: *mut f64,
    vocabulary_size: *mut i32,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if !reused.is_null() {
        *reused = 0;
    }
    let ok = run(e, error, cap, |e| {
        if tokens.is_null()
            || reused.is_null()
            || candidate_ids.is_null()
            || candidate_logits.is_null()
            || log_normalizer.is_null()
            || vocabulary_size.is_null()
        {
            return Err("null compact evidence argument".into());
        }
        if !(2..=26).contains(&candidate_count) || candidate_logits_count != candidate_count {
            return Err("invalid compact evidence buffer size".into());
        }
        let ids = slice::from_raw_parts(candidate_ids, candidate_count);
        let vocab = e.vocab();
        validate_candidates(ids, vocab)?;
        let output = forward_logits(e, input(tokens, count)?, reuse, &mut *reused, None)?;
        *log_normalizer = compact(
            slice::from_raw_parts(output, vocab as usize),
            ids,
            slice::from_raw_parts_mut(candidate_logits, candidate_count),
        )?;
        *vocabulary_size = vocab;
        Ok(())
    });
    if !ok && !reused.is_null() {
        *reused = 0;
    }
    ok
}
fn greedy(logits: &[f32]) -> Result<i32> {
    let mut next = None;
    let mut maximum = f32::NEG_INFINITY;
    for (token, &value) in logits.iter().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            return Err("invalid thinking vocabulary logit".into());
        }
        // Strict greater preserves vocabulary-order ties.
        if value > maximum {
            maximum = value;
            next = Some(token as i32);
        }
    }
    next.ok_or_else(|| "all thinking logits are non-finite".into())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_thinking(
    e: *mut Engine,
    tokens: *const i32,
    count: i32,
    close_token: i32,
    suffix: *const i32,
    suffix_count: usize,
    max_tokens: usize,
    generated_tokens: *mut usize,
    completed: *mut bool,
    logits: *mut f32,
    logits_count: usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if !generated_tokens.is_null() {
        *generated_tokens = 0;
    }
    if !completed.is_null() {
        *completed = false;
    }
    let ok = run(e, error, cap, |e| {
        if tokens.is_null()
            || suffix.is_null()
            || generated_tokens.is_null()
            || completed.is_null()
            || logits.is_null()
            || e.architecture.as_c_str() != c"qwen3"
            || !e.vision.is_null()
            || e.features_enabled
            || !(1..=1024).contains(&max_tokens)
            || suffix_count == 0
            || count <= 0
            || logits_count != e.vocab() as usize
        {
            return Err("unsupported or invalid thinking invocation".into());
        }
        let vocab = llama_model_get_vocab(e.model);
        if close_token < 0 || close_token >= e.vocab() {
            return Err("invalid thinking closure token".into());
        }
        let mut closing = [0; 64];
        let n = llama_token_to_piece(
            vocab,
            close_token,
            closing.as_mut_ptr(),
            closing.len() as i32,
            0,
            true,
        );
        if n != 8 || slice::from_raw_parts(closing.as_ptr().cast::<u8>(), 8) != b"</think>" {
            return Err("thinking requires the native </think> token".into());
        }
        if (count as usize)
            .checked_add(max_tokens)
            .and_then(|v| v.checked_add(suffix_count))
            .is_none_or(|v| v > e.context_size as usize)
        {
            return Err(
                "thinking input plus generation budget exceeds context; truncation is disabled"
                    .into(),
            );
        }
        let mut output = forward_logits(e, input(tokens, count)?, false, &mut 0, None)?;
        let mut b = Batch::new(1)?;
        let mut position = count;
        let mut decode = |e: &mut Engine, token: i32| -> Result<*const f32> {
            if token < 0 || token >= e.vocab() || position as u32 >= llama_n_ctx(e.ctx) {
                return Err("thinking decode exceeds context or vocabulary".into());
            }
            b.0.n_tokens = 0;
            b.add(token, position, 0, true);
            position += 1;
            e.memory_dirty = true;
            if llama_decode(e.ctx, b.0) != 0 {
                return Err("thinking llama_decode failed".into());
            }
            let out = llama_get_logits_ith(e.ctx, -1);
            if out.is_null() {
                return Err("thinking logits unavailable".into());
            }
            Ok(out)
        };
        for _ in 0..max_tokens {
            let next = greedy(slice::from_raw_parts(output, logits_count))?;
            if llama_vocab_is_eog(vocab, next) {
                return Err(format!(
                    "reasoning_incomplete: model ended before </think>; generated={} max_tokens={max_tokens}",
                    *generated_tokens
                ));
            }
            output = decode(e, next)?;
            *generated_tokens += 1;
            if next == close_token {
                for &token in slice::from_raw_parts(suffix, suffix_count) {
                    output = decode(e, token)?;
                }
                ptr::copy_nonoverlapping(output, logits, logits_count);
                *completed = true;
                return Ok(());
            }
        }
        Err(format!(
            "reasoning_limit: </think> was not generated; generated={} max_tokens={max_tokens}",
            *generated_tokens
        ))
    });
    sd_clear(e);
    ok
}
unsafe fn inputs<'a>(
    tokens: *const *const i32,
    counts: *const i32,
    sequences: i32,
    context: u32,
) -> Result<Vec<&'a [i32]>> {
    if tokens.is_null() || counts.is_null() || sequences <= 0 {
        return Err("invalid sequence inputs".into());
    }
    (0..sequences as usize)
        .map(|s| {
            let count = *counts.add(s);
            if count <= 0 || count as u32 > context {
                return Err("input exceeds per-question context; truncation is disabled".into());
            }
            input(*tokens.add(s), count)
        })
        .collect()
}
fn common_prefix(inputs: &[&[i32]], batch: u32) -> usize {
    if inputs.len() == 1 {
        return 0;
    }
    let mut common = inputs[0].len() - 1;
    for tokens in &inputs[1..] {
        common = inputs[0]
            .iter()
            .zip(*tokens)
            .take(common.min(tokens.len() - 1))
            .take_while(|(a, b)| a == b)
            .count();
    }
    common - common % batch as usize
}
/// Token-granular sharing skips segments too short to repay an extra decode.
const MIN_TOKEN_SEGMENT: usize = 16;

#[derive(Debug, PartialEq, Eq)]
enum PrefixOp {
    /// Evaluate `tokens[start..end]` of `seq` in its own sequence.
    Decode {
        seq: usize,
        start: usize,
        end: usize,
    },
    /// Share `src` KV for positions `[start, end)` with `dst`.
    Copy {
        src: usize,
        dst: usize,
        start: usize,
        end: usize,
    },
}

#[derive(Debug, PartialEq, Eq)]
struct PrefixPlan {
    ops: Vec<PrefixOp>,
    /// First position each sequence evaluates in the suffix phase.
    suffix_start: Vec<usize>,
    /// Tokens each sequence did not evaluate itself in this call.
    reused: Vec<usize>,
    /// Sequence 0's shared extent, retained when requested.
    root: usize,
}

/// Plan shared prefills as a prefix tree. Sequences that agree on a longer
/// exact prefix keep sharing it after another sequence in the wave diverges.
/// Every sequence keeps at least its final token for the suffix phase. With
/// `aligned`, shared segments end on complete prefill batches, so each KV row
/// is computed with the same decode boundaries as a serial prefill. `kept`
/// tokens of sequence 0 are already resident from a retained prefix.
fn plan_prefixes(inputs: &[&[i32]], batch: usize, aligned: bool, kept: usize) -> PrefixPlan {
    let n = inputs.len();
    let mut plan = PrefixPlan {
        ops: Vec::new(),
        suffix_start: vec![0; n],
        reused: vec![0; n],
        root: kept,
    };
    let mut resident = vec![0; n];
    resident[0] = kept;
    plan.reused[0] = kept;
    let align = |end: usize| {
        if aligned {
            end - end % batch.max(1)
        } else {
            end
        }
    };
    // (members, position from which members may still differ)
    let mut stack = vec![((0..n).collect::<Vec<_>>(), 0)];
    let mut first = true;
    while let Some((members, scan)) = stack.pop() {
        if members.len() < 2 {
            continue;
        }
        let lead = inputs[members[0]];
        let limit = members.iter().map(|&m| inputs[m].len() - 1).min().unwrap();
        let mut common = limit;
        for &m in &members[1..] {
            common = scan
                + lead[scan..common]
                    .iter()
                    .zip(&inputs[m][scan..common])
                    .take_while(|(a, b)| a == b)
                    .count();
        }
        let end = align(common);
        // The member with the most resident KV evaluates the shared segment;
        // ties keep request order.
        let owner = *members
            .iter()
            .max_by_key(|&&m| (resident[m], usize::MAX - m))
            .unwrap();
        let start = resident[owner];
        let gain: usize = members
            .iter()
            .filter(|&&m| m != owner)
            .map(|&m| end.saturating_sub(resident[m]))
            .sum();
        if gain > 0 && (aligned || first || end <= start || gain >= MIN_TOKEN_SEGMENT) {
            if end > start {
                plan.ops.push(PrefixOp::Decode {
                    seq: owner,
                    start,
                    end,
                });
                resident[owner] = end;
            }
            for &m in &members {
                if m != owner && resident[m] < end {
                    plan.ops.push(PrefixOp::Copy {
                        src: owner,
                        dst: m,
                        start: resident[m],
                        end,
                    });
                    plan.reused[m] += end - resident[m];
                    resident[m] = end;
                }
            }
        }
        if first {
            plan.root = resident[0];
            first = false;
        }
        // Split at the first differing position; exhausted members are leaves.
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &m in &members {
            if inputs[m].len() - 1 <= common {
                continue;
            }
            match groups
                .iter_mut()
                .find(|g| inputs[g[0]][common] == inputs[m][common])
            {
                Some(group) => group.push(m),
                None => groups.push(vec![m]),
            }
        }
        // Depth-first in request order keeps the plan deterministic.
        for group in groups.into_iter().rev() {
            stack.push((group, common + 1));
        }
    }
    plan.suffix_start = resident;
    plan
}

/// Longest exact prefix of the retained tokens that sequence 0 can keep.
fn retained_prefix(retained: &[i32], first: &[i32], batch: usize, aligned: bool) -> usize {
    let common = retained
        .iter()
        .zip(first)
        .take(first.len() - 1)
        .take_while(|(a, b)| a == b)
        .count();
    if aligned {
        common - common % batch.max(1)
    } else {
        common
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_parallel(
    e: *mut Engine,
    tokens: *const *const i32,
    counts: *const i32,
    sequences: i32,
    capacity: u32,
    dynamic: bool,
    flags: u32,
    reused: *mut i32,
    logits: *mut f32,
    logits_count: usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    let aligned = flags & SD_PARALLEL_TOKEN_PREFIX == 0;
    let retain = flags & SD_PARALLEL_RETAIN_PREFIX != 0;
    let ok = run(e, error, cap, |e| {
        if sequences < 1 || sequences as u32 > capacity || capacity > 32 {
            return Err("invalid parallel sequence count".into());
        }
        if flags & !(SD_PARALLEL_TOKEN_PREFIX | SD_PARALLEL_RETAIN_PREFIX) != 0 {
            return Err("unknown parallel flags".into());
        }
        if e.recurrent() {
            return Err(
                "parallel prefix sharing is unsupported for recurrent/hybrid models".into(),
            );
        }
        let vocab = e.vocab() as usize;
        if reused.is_null() || logits.is_null() || logits_count != vocab * sequences as usize {
            return Err("wrong parallel logits buffer size".into());
        }
        let reused = slice::from_raw_parts_mut(reused, sequences as usize);
        reused.fill(0);
        let inputs = inputs(tokens, counts, sequences, e.context_size)?;
        let total: u64 = inputs.iter().map(|s| s.len() as u64).sum();
        let maximum = u64::from(e.context_size) * u64::from(capacity);
        if maximum > i32::MAX as u64 {
            return Err("parallel width requires 1..32 and context * width <= INT_MAX".into());
        }
        let requested = if dynamic && capacity > 1 {
            maximum.min(total + u64::from(e.batch_size))
        } else {
            maximum
        };
        // A context rebuild clears the engine, including any retained prefix.
        e.ensure_sequences(capacity, requested as u32, dynamic, true)?;
        let batch = e.batch_size as usize;
        let kept = if retain && !e.parallel_retained.is_empty() {
            retained_prefix(&e.parallel_retained, inputs[0], batch, aligned)
        } else {
            0
        };
        if kept == 0 {
            e.clear();
        } else {
            // Only sequence 0's matching prefix survives from the previous call.
            let memory = llama_get_memory(e.ctx);
            for s in 1..capacity as i32 {
                llama_memory_seq_rm(memory, s, -1, -1);
            }
            llama_memory_seq_rm(memory, 0, kept as i32, -1);
            e.parallel_retained.truncate(kept);
        }
        let plan = plan_prefixes(&inputs, batch, aligned, kept);
        let mut b = Batch::new(e.batch_size)?;
        let memory = llama_get_memory(e.ctx);
        for op in &plan.ops {
            match *op {
                PrefixOp::Decode {
                    seq,
                    mut start,
                    end,
                } => {
                    while start < end {
                        b.0.n_tokens = 0;
                        let stop = end.min(start + batch);
                        for (position, &token) in
                            inputs[seq].iter().enumerate().take(stop).skip(start)
                        {
                            b.add(token, position as i32, seq as i32, false);
                        }
                        e.memory_dirty = true;
                        let rc = llama_decode(e.ctx, b.0);
                        if rc != 0 {
                            return Err(format!(
                                "parallel shared prefill failed (llama_decode={rc}, batch_tokens={}, context={}; code 1 means no KV slot for this batch; see llama.cpp stderr for other codes)",
                                b.0.n_tokens,
                                llama_n_ctx(e.ctx)
                            ));
                        }
                        start = stop;
                    }
                }
                PrefixOp::Copy {
                    src,
                    dst,
                    start,
                    end,
                } => {
                    llama_memory_seq_cp(memory, src as i32, dst as i32, start as i32, end as i32);
                }
            }
        }
        for (reuse, &count) in reused.iter_mut().zip(&plan.reused) {
            *reuse = count as i32;
        }
        let mut positions = plan.suffix_start.clone();
        let mut completed = 0;
        while completed < inputs.len() {
            b.0.n_tokens = 0;
            let mut outputs = Vec::new();
            let mut added = true;
            while added && b.0.n_tokens < e.batch_size as i32 {
                added = false;
                for s in 0..inputs.len() {
                    if b.0.n_tokens >= e.batch_size as i32 {
                        break;
                    }
                    if positions[s] == inputs[s].len() {
                        continue;
                    }
                    let last = positions[s] == inputs[s].len() - 1;
                    let j = b.add(inputs[s][positions[s]], positions[s] as i32, s as i32, last);
                    positions[s] += 1;
                    if last {
                        outputs.push((s, j));
                    }
                    added = true;
                }
            }
            if b.0.n_tokens == 0 {
                return Err("parallel scheduler made no progress".into());
            }
            e.memory_dirty = true;
            let rc = llama_decode(e.ctx, b.0);
            if rc != 0 {
                return Err(format!(
                    "parallel suffix decode failed (llama_decode={rc}, batch_tokens={}, completed={completed}/{sequences}, context={}; code 1 means no KV slot for this batch; see llama.cpp stderr for other codes)",
                    b.0.n_tokens,
                    llama_n_ctx(e.ctx)
                ));
            }
            for (s, j) in outputs {
                let output = llama_get_logits_ith(e.ctx, j);
                if output.is_null() {
                    return Err("missing parallel final logits".into());
                }
                ptr::copy_nonoverlapping(output, logits.add(s * vocab), vocab);
                completed += 1;
            }
        }
        if retain && plan.root > 0 {
            for s in 1..capacity as i32 {
                llama_memory_seq_rm(memory, s, -1, -1);
            }
            llama_memory_seq_rm(memory, 0, plan.root as i32, -1);
            e.parallel_retained = inputs[0][..plan.root].to_vec();
        } else {
            e.clear();
        }
        Ok(())
    });
    if !retain {
        sd_clear(e);
    }
    ok
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_restore(
    e: *mut Engine,
    tokens: *const *const i32,
    counts: *const i32,
    sequences: i32,
    limit: usize,
    reused: *mut i32,
    logits: *mut f32,
    logits_count: usize,
    metrics: *mut NativeRestoreMetrics,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if !metrics.is_null() {
        *metrics = NativeRestoreMetrics::default();
    }
    let ok = run(e, error, cap, |e| {
        e.ensure_sequences(1, 0, false, true)?;
        e.clear();
        let vocab = e.vocab() as usize;
        if sequences < 1
            || logits_count != vocab * sequences as usize
            || logits.is_null()
            || metrics.is_null()
            || reused.is_null()
        {
            return Err("invalid snapshot sequence or logits count".into());
        }
        let metrics = &mut *metrics;
        let reused = slice::from_raw_parts_mut(reused, sequences as usize);
        reused.fill(0);
        let inputs = inputs(tokens, counts, sequences, e.context_size)?;
        let common = common_prefix(&inputs, e.batch_size);
        let fresh = |e: &mut Engine,
                     reused: &mut [i32],
                     metrics: &mut NativeRestoreMetrics|
         -> Result<()> {
            for (s, tokens) in inputs.iter().enumerate() {
                let start = Instant::now();
                let output = forward_logits(e, tokens, false, &mut reused[s], None)?;
                ptr::copy_nonoverlapping(output, logits.add(s * vocab), vocab);
                metrics.suffix_ms += start.elapsed().as_secs_f64() * 1000.0;
            }
            Ok(())
        };
        if common == 0 {
            metrics.fallback = 1;
            return fresh(e, reused, metrics);
        }
        // A disabled snapshot budget cannot retain even one byte. Avoid doing
        // a common-prefix prefill that the fresh fallback would repeat.
        if limit == 0 {
            metrics.fallback = 2;
            return fresh(e, reused, metrics);
        }
        let mut b = Batch::new(e.batch_size)?;
        let start = Instant::now();
        decode_range(e, &mut b, inputs[0], 0, common, false)?;
        metrics.prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let bytes = llama_state_seq_get_size(e.ctx, 0);
        if bytes == 0 {
            metrics.fallback = 3;
            return fresh(e, reused, metrics);
        }
        if bytes > limit {
            metrics.fallback = 2;
            return fresh(e, reused, metrics);
        }
        let mut snapshot = vec![0; bytes];
        metrics.snapshot_bytes = bytes;
        if llama_state_seq_get_data(e.ctx, snapshot.as_mut_ptr(), bytes, 0) != bytes {
            metrics.fallback = 3;
            return fresh(e, reused, metrics);
        }
        metrics.save_ms = start.elapsed().as_secs_f64() * 1000.0;
        for (s, tokens) in inputs.iter().enumerate() {
            if s > 0 {
                let start = Instant::now();
                // For the single dense sequence, upstream state_read_meta
                // removes the old sequence before restoring its cells. A full
                // KV-buffer memset here duplicates that work. Keep the existing
                // clear for recurrent/hybrid memory and the diagnostic override;
                // request boundaries and every error still clear the engine.
                if e.recurrent() || e.force_kv_clear {
                    e.clear();
                }
                e.memory_dirty = true;
                if llama_state_seq_set_data(e.ctx, snapshot.as_ptr(), bytes, 0) != bytes {
                    metrics.fallback = 4;
                    metrics.restores = 0;
                    return fresh(e, reused, metrics);
                }
                metrics.restore_ms += start.elapsed().as_secs_f64() * 1000.0;
                metrics.restores += 1;
            }
            let start = Instant::now();
            decode_range(e, &mut b, tokens, common, tokens.len(), true)?;
            let output = llama_get_logits_ith(e.ctx, -1);
            if output.is_null() {
                return Err("missing snapshot final logits".into());
            }
            ptr::copy_nonoverlapping(output, logits.add(s * vocab), vocab);
            metrics.suffix_ms += start.elapsed().as_secs_f64() * 1000.0;
            reused[s] = if s == 0 { 0 } else { common as i32 };
        }
        Ok(())
    });
    sd_clear(e);
    ok
}
/// Explicit split plan: cold and warm runs use identical decode boundaries.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_forward_split(
    e: *mut Engine,
    tokens: *const i32,
    count: i32,
    boundary: usize,
    reuse: bool,
    reused: *mut i32,
    logits: *mut f32,
    logits_count: usize,
    error: *mut c_char,
    cap: usize,
) -> bool {
    if !reused.is_null() {
        *reused = 0;
    }
    run(e, error, cap, |e| {
        if reused.is_null() || logits.is_null() || logits_count != e.vocab() as usize {
            return Err("wrong split logits buffer size".into());
        }
        let output = forward_logits(
            e,
            input(tokens, count)?,
            reuse,
            &mut *reused,
            Some(boundary),
        )?;
        ptr::copy_nonoverlapping(output, logits, logits_count);
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn greedy_ties_and_nonfinite_values() {
        assert_eq!(greedy(&[f32::NEG_INFINITY, 4.0, 4.0]).unwrap(), 1);
        assert!(greedy(&[f32::NEG_INFINITY]).is_err());
        assert!(greedy(&[0.0, f32::NAN]).is_err());
        assert!(greedy(&[0.0, f32::INFINITY]).is_err());
    }
    #[test]
    fn compact_preserves_full_vocabulary_mass() {
        let mut out = [0.0; 2];
        let normalizer = compact(&[0.0, 1.0, 2.0, f32::NEG_INFINITY], &[2, 0], &mut out).unwrap();
        assert_eq!(out, [2.0, 0.0]);
        assert!((normalizer - (1.0_f64 + 1.0_f64.exp() + 2.0_f64.exp()).ln()).abs() < 1e-14);
        for ids in [[0, 0], [0, 4], [-1, 0], [0, 3]] {
            assert!(compact(&[0.0, 1.0, 2.0, f32::NEG_INFINITY], &ids, &mut out).is_err());
        }
        for bad in [f32::NAN, f32::INFINITY] {
            assert!(compact(&[0.0, 1.0, bad], &[0, 1], &mut out).is_err());
        }
    }
    #[test]
    fn shared_prefix_leaves_final_token_and_aligns_batches() {
        let a = [1, 2, 3, 4, 5, 6];
        let b = [1, 2, 3, 4, 8];
        assert_eq!(common_prefix(&[&a, &b], 2), 4);
        assert_eq!(common_prefix(&[&a, &a], 2), 4);
        assert_eq!(common_prefix(&[&a], 2), 0);
        assert_eq!(common_prefix(&[&a, &[1]], 2), 0);
    }
    fn decode(seq: usize, start: usize, end: usize) -> PrefixOp {
        PrefixOp::Decode { seq, start, end }
    }
    fn copy(src: usize, dst: usize, start: usize, end: usize) -> PrefixOp {
        PrefixOp::Copy {
            src,
            dst,
            start,
            end,
        }
    }
    #[test]
    fn prefix_plan_matches_whole_wave_sharing() {
        let a = [1, 2, 3, 4, 5, 6];
        let b = [1, 2, 3, 4, 8];
        let plan = plan_prefixes(&[&a, &b], 2, true, 0);
        assert_eq!(plan.ops, [decode(0, 0, 4), copy(0, 1, 0, 4)]);
        assert_eq!(plan.suffix_start, [4, 4]);
        assert_eq!(plan.reused, [0, 4]);
        assert_eq!(plan.root, 4);
        // Batch alignment keeps the old rounding; one input shares nothing.
        let plan = plan_prefixes(&[&a, &[1, 2, 3, 9]], 2, true, 0);
        assert_eq!(plan.ops, [decode(0, 0, 2), copy(0, 1, 0, 2)]);
        let plan = plan_prefixes(&[&a], 2, true, 0);
        assert!(plan.ops.is_empty());
        assert_eq!(plan.suffix_start, [0]);
        // Identical prompts keep their final token for the suffix phase.
        let plan = plan_prefixes(&[&a, &a, &a], 1, true, 0);
        assert_eq!(plan.suffix_start, [5, 5, 5]);
        assert_eq!(plan.reused, [0, 5, 5]);
    }
    #[test]
    fn prefix_plan_keeps_sharing_after_an_early_divergence() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        let b = [1, 2, 7, 7, 7, 7, 7, 7, 7];
        let c = [1, 2, 3, 4, 5, 6, 7, 8, 10];
        let plan = plan_prefixes(&[&a, &b, &c], 2, true, 0);
        assert_eq!(
            plan.ops,
            [
                decode(0, 0, 2),
                copy(0, 1, 0, 2),
                copy(0, 2, 0, 2),
                decode(0, 2, 8),
                copy(0, 2, 2, 8),
            ]
        );
        assert_eq!(plan.suffix_start, [8, 2, 8]);
        assert_eq!(plan.reused, [0, 2, 8]);
        assert_eq!(plan.root, 2);
        // Batch alignment rounds each nested segment down.
        let plan = plan_prefixes(&[&a, &b, &c], 4, true, 0);
        assert_eq!(
            plan.ops,
            [decode(0, 0, 8), copy(0, 2, 0, 8)],
            "the root rounds to zero, the nested group still shares"
        );
        assert_eq!(plan.suffix_start, [8, 0, 8]);
        assert_eq!(plan.root, 0);
    }
    #[test]
    fn token_prefix_plan_skips_only_short_nested_segments() {
        let long: Vec<i32> = (0..40).collect();
        let mut a = long.clone();
        a.extend([100, 1]);
        let mut b = long.clone();
        b.extend([100, 2]);
        let mut c = long.clone();
        c.extend([200, 3]);
        let plan = plan_prefixes(&[&a, &b, &c], 16, false, 0);
        // The root shares all 40 tokens; one more common token is not worth
        // a separate decode.
        assert_eq!(
            plan.ops,
            [decode(0, 0, 40), copy(0, 1, 0, 40), copy(0, 2, 0, 40)]
        );
        assert_eq!(plan.suffix_start, [40, 40, 40]);
        let plan = plan_prefixes(&[&a, &b, &c], 16, true, 0);
        assert_eq!(
            plan.ops,
            [decode(0, 0, 32), copy(0, 1, 0, 32), copy(0, 2, 0, 32)]
        );
    }
    #[test]
    fn retained_prefix_is_not_evaluated_again() {
        let a = [1, 2, 3, 4, 5, 6, 7];
        let b = [1, 2, 3, 4, 5, 6, 8];
        assert_eq!(retained_prefix(&[1, 2, 3, 4, 9], &a, 2, true), 4);
        assert_eq!(retained_prefix(&[1, 2, 3, 9], &a, 2, true), 2);
        assert_eq!(retained_prefix(&[1, 2, 3, 9], &a, 2, false), 3);
        assert_eq!(retained_prefix(&a, &a, 1, true), 6);
        let plan = plan_prefixes(&[&a, &b], 2, true, 4);
        assert_eq!(plan.ops, [decode(0, 4, 6), copy(0, 1, 0, 6)]);
        assert_eq!(plan.reused, [4, 6]);
        assert_eq!(plan.root, 6);
        // A single question reuses the retained prefix and keeps it.
        let plan = plan_prefixes(&[&a], 2, true, 4);
        assert!(plan.ops.is_empty());
        assert_eq!((plan.suffix_start[0], plan.reused[0], plan.root), (4, 4, 4));
        // A retained prefix longer than this wave's common prefix stays whole.
        let plan = plan_prefixes(&[&a, &[1, 9]], 2, true, 4);
        assert_eq!(plan.ops, []);
        assert_eq!(plan.root, 4);
    }
}
