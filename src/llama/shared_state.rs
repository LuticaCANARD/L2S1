use super::{LlamaBackend, sd_clear, sd_recurrent_or_hybrid};
use crate::{Decision, DecisionRequest, DecisionResponse, Error, ExecutionMode, Result};

/// An explicit lifetime for decoder prefix reuse over one immutable state.
///
/// The backend is exclusively borrowed, so model, policy, layout and artifacts
/// cannot change while the session is active. Each call may use new decision
/// IDs, instructions, option counts and decision kinds. Only exact, complete
/// prefill batches are reused; questions never attend to earlier answers.
///
/// This is a decoder prefix session, not a learned bidirectional state encoder.
/// State-first layout usually exposes a longer reusable prefix for different
/// questions. Ordinary `DecisionBackend::decide` remains request-isolated.
/// Native KV state is cleared on creation, any error, and drop.
#[must_use = "keep the session alive while issuing questions against shared state"]
pub struct SharedStateSession<'a> {
    backend: &'a mut LlamaBackend,
    request: DecisionRequest,
}

impl LlamaBackend {
    /// Begin explicit prefix sharing without silently changing execution mode.
    /// Configure `ExecutionMode::PrefixReuse` before registering calibration and
    /// starting the session. Recurrent/hybrid memory and output heads are not
    /// supported. Prompt layout and evidence transfer mode remain unchanged.
    pub fn shared_state(&mut self, state: serde_json::Value) -> Result<SharedStateSession<'_>> {
        unsafe { sd_clear(self.engine.as_ptr()) };
        if self.execution_mode != ExecutionMode::PrefixReuse {
            return Err(Error::Invalid(
                "shared-state sessions require explicit PrefixReuse execution mode".into(),
            ));
        }
        if unsafe { sd_recurrent_or_hybrid(self.engine.as_ptr()) } {
            return Err(Error::Backend(
                "shared-state sessions do not support recurrent/hybrid models".into(),
            ));
        }
        if self.output_head.is_some() {
            return Err(Error::Invalid(
                "shared-state sessions do not support output heads".into(),
            ));
        }
        Ok(SharedStateSession {
            backend: self,
            request: DecisionRequest {
                state,
                decisions: Vec::new(),
            },
        })
    }
}

impl SharedStateSession<'_> {
    /// Model-local preparation-cache counters, including this session's calls.
    pub fn preparation_cache_stats(&self) -> super::PreparationCacheStats {
        self.backend.preparation_cache_stats()
    }

    /// Drain inference timings without ending the session or clearing its KV.
    pub fn take_timings(&mut self) -> super::InferenceTimings {
        self.backend.take_timings()
    }

    /// Evaluate independent decisions against the session's immutable state.
    /// IDs must be unique within this call; the same IDs may be reused in later
    /// calls. A failed call clears native state and can be followed by a new call.
    pub fn decide(&mut self, decisions: Vec<Decision>) -> Result<DecisionResponse> {
        // Reuse the owned state without cloning or serializing it for validation.
        self.request.decisions = decisions;
        let results = (|| {
            self.request.validate()?;
            self.backend.check_artifacts(&self.request)?;
            self.backend.restore_metrics = Default::default();
            self.request
                .decisions
                .iter()
                .map(|decision| self.backend.evaluate(&self.request.state, decision))
                .collect::<Result<Vec<_>>>()
        })();
        let backend_info = self.backend.info_for_request(&self.request);
        self.request.decisions.clear();
        if results.is_err() {
            unsafe { sd_clear(self.backend.engine.as_ptr()) };
        }
        Ok(DecisionResponse {
            backend: backend_info,
            policy: self.backend.policy.clone(),
            results: results?,
        })
    }
}

impl Drop for SharedStateSession<'_> {
    fn drop(&mut self) {
        unsafe { sd_clear(self.backend.engine.as_ptr()) };
    }
}
