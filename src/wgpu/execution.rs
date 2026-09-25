//! Request-local execution for the Gemma 4 wgpu model.
//!
//! The model owns one mutable KV stream. All modes keep decisions serial and
//! clear that stream at request boundaries. A snapshot never crosses requests.

use super::backend_error;
use crate::{Error, ExecutionMode, Result, StateRestoreMetrics};
use rullama_engine::api::Model;
use serde::Serialize;
use std::time::Instant;

const DEFAULT_SNAPSHOT_LIMIT: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
pub struct WgpuExecutionReport {
    pub requested_mode: ExecutionMode,
    pub effective_mode: ExecutionMode,
    pub fallback_reason: Option<String>,
    pub reused_prefix_tokens: Vec<usize>,
    pub state_restore: StateRestoreMetrics,
}

pub struct WgpuExecutor {
    mode: ExecutionMode,
    snapshot_limit: usize,
}

impl Default for WgpuExecutor {
    fn default() -> Self {
        Self {
            mode: ExecutionMode::Fresh,
            snapshot_limit: DEFAULT_SNAPSHOT_LIMIT,
        }
    }
}

impl WgpuExecutor {
    pub fn set_mode(&mut self, mode: ExecutionMode) -> Result<()> {
        if mode == ExecutionMode::Parallel {
            return Err(Error::Invalid(
                "wgpu does not support independent parallel KV sequences".into(),
            ));
        }
        self.mode = mode;
        Ok(())
    }

    pub fn set_snapshot_limit(&mut self, bytes: usize) {
        self.snapshot_limit = bytes;
    }

    pub fn run(
        &self,
        model: &mut Model,
        prompts: &[Vec<u32>],
        image_embeddings: Option<&[f32]>,
        single_token_codes: bool,
    ) -> Result<(Vec<Vec<f32>>, WgpuExecutionReport)> {
        let mut report = WgpuExecutionReport {
            requested_mode: self.mode,
            effective_mode: self.mode,
            fallback_reason: None,
            reused_prefix_tokens: vec![0; prompts.len()],
            state_restore: StateRestoreMetrics::default(),
        };
        if prompts.is_empty() {
            return Ok((Vec::new(), report));
        }
        if !single_token_codes {
            report.effective_mode = ExecutionMode::Fresh;
            if self.mode != ExecutionMode::Fresh {
                report.fallback_reason = Some("code_sequence_requires_fresh".into());
                report.state_restore.fallback_reason = report.fallback_reason.clone();
            }
            return Ok((Vec::new(), report));
        }
        let common = common_prefix(prompts);
        if self.mode != ExecutionMode::Fresh && common == 0 {
            report.fallback_reason = Some("no_shared_prefix".into());
        }
        if report.fallback_reason.is_some() || self.mode == ExecutionMode::Fresh {
            return fresh(model, prompts, image_embeddings, report);
        }

        let image_begin = model.image_sentinel_ids_native().map(|pair| pair.0);
        let image_rows = image_embeddings
            .map(|soft| soft.len() / model.forward().cfg().d_model as usize)
            .unwrap_or(0);
        let prefix_tokens = common
            + usize::from(image_begin.is_some_and(|begin| prompts[0][..common].contains(&begin)))
                * image_rows;
        if self.mode == ExecutionMode::StateRestore {
            // The engine's save API allocates and reads back the entire KV state.
            // Bound it before calling save, using a conservative f32 upper bound.
            let estimate = snapshot_upper_bound(model, prefix_tokens)?;
            if estimate > self.snapshot_limit {
                report.fallback_reason = Some("snapshot_memory_budget".into());
                return fresh(model, prompts, image_embeddings, report);
            }
        }

        model.reset_native();
        let start = Instant::now();
        let mut last = Vec::new();
        for &token in &prompts[0][..common] {
            last = step(model, token, image_begin, image_embeddings)?;
        }
        report.state_restore.prefill_ms = start.elapsed().as_secs_f64() * 1000.0;
        let prefill_position = model.position_native();

        let snapshot = if self.mode == ExecutionMode::StateRestore {
            let start = Instant::now();
            match pollster::block_on(model.save_kv_state_native()) {
                Ok(bytes) if bytes.len() <= self.snapshot_limit => {
                    report.state_restore.snapshot_bytes = bytes.len();
                    report.state_restore.save_ms = start.elapsed().as_secs_f64() * 1000.0;
                    Some(bytes)
                }
                Ok(_) => {
                    report.fallback_reason = Some("snapshot_memory_budget".into());
                    return fresh(model, prompts, image_embeddings, report);
                }
                Err(_) => {
                    report.fallback_reason = Some("snapshot_save_unavailable".into());
                    return fresh(model, prompts, image_embeddings, report);
                }
            }
        } else {
            None
        };

        let mut outputs = Vec::with_capacity(prompts.len());
        for (index, prompt) in prompts.iter().enumerate() {
            if index > 0 {
                let start = Instant::now();
                if let Some(snapshot) = &snapshot {
                    if model.restore_kv_state_native(snapshot).is_err() {
                        report.fallback_reason = Some("snapshot_restore_unavailable".into());
                        report.state_restore.restores = 0;
                        report.reused_prefix_tokens.fill(0);
                        return fresh(model, prompts, image_embeddings, report);
                    }
                    report.state_restore.restores += 1;
                    report.state_restore.restore_ms += start.elapsed().as_secs_f64() * 1000.0;
                } else {
                    model.truncate_kv_native(prefill_position);
                }
                report.reused_prefix_tokens[index] = prefill_position as usize;
            }
            let start = Instant::now();
            for &token in &prompt[common..] {
                last = step(model, token, image_begin, image_embeddings)?;
            }
            report.state_restore.suffix_ms += start.elapsed().as_secs_f64() * 1000.0;
            outputs.push(last.clone());
        }
        model.reset_native();
        Ok((outputs, report))
    }
}

fn fresh(
    model: &mut Model,
    prompts: &[Vec<u32>],
    image_embeddings: Option<&[f32]>,
    mut report: WgpuExecutionReport,
) -> Result<(Vec<Vec<f32>>, WgpuExecutionReport)> {
    report.effective_mode = ExecutionMode::Fresh;
    report.reused_prefix_tokens.fill(0);
    let image_begin = model.image_sentinel_ids_native().map(|pair| pair.0);
    let mut outputs = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        model.reset_native();
        let start = Instant::now();
        let mut last = Vec::new();
        for &token in prompt {
            last = step(model, token, image_begin, image_embeddings)?;
        }
        report.state_restore.suffix_ms += start.elapsed().as_secs_f64() * 1000.0;
        outputs.push(last);
    }
    model.reset_native();
    report.state_restore.fallback_reason = report.fallback_reason.clone();
    Ok((outputs, report))
}

fn step(
    model: &mut Model,
    token: u32,
    image_begin: Option<u32>,
    image_embeddings: Option<&[f32]>,
) -> Result<Vec<f32>> {
    let mut logits = pollster::block_on(model.forward_mut().step(token)).map_err(backend_error)?;
    if Some(token) == image_begin
        && let Some(soft) = image_embeddings
    {
        let width = model.forward().cfg().d_model as usize;
        for row in soft.chunks_exact(width) {
            logits = pollster::block_on(model.forward_mut().step_with_embedding(row))
                .map_err(backend_error)?;
        }
    }
    Ok(logits)
}

fn common_prefix(prompts: &[Vec<u32>]) -> usize {
    let first = &prompts[0];
    let mut common = first.len().saturating_sub(1);
    for prompt in prompts.iter().skip(1) {
        common = common.min(prompt.len().saturating_sub(1));
        common = first[..common]
            .iter()
            .zip(&prompt[..common])
            .take_while(|(a, b)| a == b)
            .count();
    }
    if prompts.len() == 1 { 0 } else { common }
}

fn snapshot_upper_bound(model: &Model, tokens: usize) -> Result<usize> {
    let cfg = model.forward().cfg();
    let mut bytes = 4096usize + 16 + cfg.n_layers as usize * 12;
    for layer in 0..cfg.n_layers {
        let elements = tokens
            .checked_mul(cfg.n_kv_heads(layer) as usize)
            .and_then(|value| value.checked_mul(cfg.head_dim(layer) as usize))
            .and_then(|value| value.checked_mul(8))
            .ok_or_else(|| Error::Invalid("snapshot size exceeds address space".into()))?;
        bytes = bytes
            .checked_add(elements)
            .ok_or_else(|| Error::Invalid("snapshot size exceeds address space".into()))?;
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::common_prefix;

    #[test]
    fn common_prefix_keeps_one_token_for_each_answer_boundary() {
        assert_eq!(common_prefix(&[vec![1, 2, 3], vec![1, 2, 4]]), 2);
        assert_eq!(common_prefix(&[vec![1, 2], vec![1, 2]]), 1);
        assert_eq!(common_prefix(&[vec![1, 2]]), 0);
    }
}
