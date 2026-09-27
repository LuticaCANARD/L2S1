use super::{LlamaBackend, sd_clear, sd_recurrent_or_hybrid};
use crate::{Decision, DecisionRequest, DecisionResponse, Error, ExecutionMode, Result};

/// Reuse a fixed instruction and candidate schema while evaluating new states.
///
/// Exclusively borrows the backend: model, schema, policy, layout and artifacts
/// cannot change during the session. Only exact, complete native token batches
/// are reused. Each new state replaces the previous suffix; no answers are cached.
/// With the default sorted JSON maps, Legacy places instruction/options before
/// state and usually exposes more reusable tokens than StateFirst. Always check
/// reused_prefix_tokens; downstream serde_json features can affect Legacy order.
///
/// KV is cleared at creation, on failure and on drop. Ordinary requests remain
/// isolated. This session retains one sequence, not a multi-schema cache.
#[must_use = "keep the session alive while evaluating states against the fixed decision"]
pub struct SharedDecisionSession<'a> {
    backend: &'a mut LlamaBackend,
    request: DecisionRequest,
}

impl LlamaBackend {
    /// Create a scoped schema-reuse session. Configure PrefixReuse first;
    /// recurrent/hybrid models and output heads are currently unsupported.
    pub fn shared_decision(&mut self, decision: Decision) -> Result<SharedDecisionSession<'_>> {
        unsafe { sd_clear(self.engine.as_ptr()) };
        if self.execution_mode != ExecutionMode::PrefixReuse {
            return Err(Error::Invalid(
                "shared-decision sessions require explicit PrefixReuse execution mode".into(),
            ));
        }
        if unsafe { sd_recurrent_or_hybrid(self.engine.as_ptr()) } {
            return Err(Error::Backend(
                "shared-decision sessions do not support recurrent/hybrid models".into(),
            ));
        }
        if self.output_head.is_some() {
            return Err(Error::Invalid(
                "shared-decision sessions do not support output heads".into(),
            ));
        }
        let request = DecisionRequest {
            state: serde_json::Value::Null,
            decisions: vec![decision],
        };
        request.validate()?;
        self.check_artifacts(&request)?;
        Ok(SharedDecisionSession {
            backend: self,
            request,
        })
    }
}

impl SharedDecisionSession<'_> {
    pub fn preparation_cache_stats(&self) -> super::PreparationCacheStats {
        self.backend.preparation_cache_stats()
    }

    pub fn take_timings(&mut self) -> super::InferenceTimings {
        self.backend.take_timings()
    }

    /// Evaluate a new state with the immutable schema. No state or question is
    /// concatenated to a prior request. An error clears KV before the next call.
    pub fn decide(&mut self, state: serde_json::Value) -> Result<DecisionResponse> {
        self.request.state = state;
        self.backend.restore_metrics = Default::default();
        let result = self
            .backend
            .evaluate(&self.request.state, &self.request.decisions[0]);
        let backend = self.backend.info_for_request(&self.request);
        self.request.state = serde_json::Value::Null;
        if result.is_err() {
            unsafe { sd_clear(self.backend.engine.as_ptr()) };
        }
        Ok(DecisionResponse {
            backend,
            policy: self.backend.policy.clone(),
            results: vec![result?],
        })
    }
}

impl Drop for SharedDecisionSession<'_> {
    fn drop(&mut self) {
        unsafe { sd_clear(self.backend.engine.as_ptr()) };
    }
}
