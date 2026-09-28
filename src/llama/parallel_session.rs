use super::{LlamaBackend, sd_clear, sd_recurrent_or_hybrid};
use crate::{DecisionRequest, DecisionResponse, Error, ExecutionMode, Result};

/// Keep exact shared-prefix KV between `parallel` calls.
///
/// Ordinary parallel calls clear native KV at every request boundary, so a
/// common prefix (for example knowledge and examples in a request `shared`
/// field) is evaluated again by each call. Within this session, sequence 0
/// keeps the latest wave's root shared prefix. A later wave whose first prompt
/// starts with the same exact tokens reuses it instead of evaluating it again;
/// with the default batch alignment, only complete prefill batches are kept.
/// Questions never attend to another question's suffix or to earlier answers.
///
/// The backend is exclusively borrowed, so model, layout, policy and artifacts
/// cannot change. Sequence capacity stays at the configured parallel width.
/// Native KV is cleared on creation, any error, and drop.
#[must_use = "keep the session alive while issuing parallel requests"]
pub struct ParallelPrefixSession<'a> {
    backend: &'a mut LlamaBackend,
}

impl LlamaBackend {
    /// Begin cross-call prefix retention. Configure `ExecutionMode::Parallel`
    /// first; recurrent/hybrid models are not supported.
    pub fn parallel_prefix_session(&mut self) -> Result<ParallelPrefixSession<'_>> {
        unsafe { sd_clear(self.engine.as_ptr()) };
        if self.execution_mode != ExecutionMode::Parallel {
            return Err(Error::Invalid(
                "parallel prefix sessions require explicit Parallel execution mode".into(),
            ));
        }
        if unsafe { sd_recurrent_or_hybrid(self.engine.as_ptr()) } {
            return Err(Error::Backend(
                "parallel prefix sessions do not support recurrent/hybrid models".into(),
            ));
        }
        self.parallel_retain_calls = true;
        Ok(ParallelPrefixSession { backend: self })
    }
}

impl ParallelPrefixSession<'_> {
    pub fn preparation_cache_stats(&self) -> super::PreparationCacheStats {
        self.backend.preparation_cache_stats()
    }

    /// Drain inference timings without ending the session or clearing its KV.
    pub fn take_timings(&mut self) -> super::InferenceTimings {
        self.backend.take_timings()
    }

    /// Evaluate one request. See [`Self::decide_batch`].
    pub fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let mut responses = self.decide_batch(std::slice::from_ref(request))?;
        Ok(responses.remove(0))
    }

    /// Evaluate independent requests like `LlamaBackend::decide_batch`, keeping
    /// the retained prefix afterwards. A failed call clears native state and
    /// can be followed by a new call.
    pub fn decide_batch(&mut self, requests: &[DecisionRequest]) -> Result<Vec<DecisionResponse>> {
        let responses = self.backend.decide_batch_inner(requests);
        if responses.is_err() {
            unsafe { sd_clear(self.backend.engine.as_ptr()) };
        }
        responses
    }
}

impl Drop for ParallelPrefixSession<'_> {
    fn drop(&mut self) {
        self.backend.parallel_retain_calls = false;
        unsafe { sd_clear(self.backend.engine.as_ptr()) };
    }
}
