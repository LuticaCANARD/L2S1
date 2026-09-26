//! Direct image input with fresh evaluation or isolated parallel sequences.
use super::prepared_cache::TokenCacheValue;
use super::*;

/// The exact rendered prompt and answer mapping; no image bytes or KV state are
/// retained. The model-local cache is invalidated by every preparation setter.
pub(super) struct PreparedVision {
    parts: Vec<crate::prompt::PromptPart>,
    after: String,
    candidates: Vec<i32>,
}

impl Clone for PreparedVision {
    fn clone(&self) -> Self {
        Self {
            parts: self
                .parts
                .iter()
                .map(|part| crate::prompt::PromptPart {
                    text: part.text.clone(),
                    parse_special: part.parse_special,
                })
                .collect(),
            after: self.after.clone(),
            candidates: self.candidates.clone(),
        }
    }
}
impl TokenCacheValue for PreparedVision {
    fn retained_bytes(&self) -> usize {
        self.parts.capacity() * std::mem::size_of::<crate::prompt::PromptPart>()
            + self
                .parts
                .iter()
                .map(|part| part.text.capacity())
                .sum::<usize>()
            + self.after.capacity()
            + self.candidates.retained_bytes()
    }
}

/// Native vision calls clear their own request-local memory on every exit. This
/// guard owns Rust-only failures and unwinding before/after those calls.
struct VisionCleanup {
    engine: *mut c_void,
    armed: bool,
}
impl VisionCleanup {
    fn new(engine: *mut c_void) -> Self {
        Self {
            engine,
            armed: true,
        }
    }
    fn complete(&mut self) {
        self.armed = false;
    }
}
impl Drop for VisionCleanup {
    fn drop(&mut self) {
        if self.armed {
            unsafe { sd_clear(self.engine) };
        }
    }
}

impl LlamaBackend {
    /// Attach a vision projector compatible with this GGUF. Text decisions
    /// remain available on the same backend.
    pub fn load_vision_projector(&mut self, path: &Path) -> Result<()> {
        let path_text = path.to_string_lossy().into_owned();
        let path_c = CString::new(path_text.as_bytes())
            .map_err(|_| Error::Invalid("vision projector path contains NUL".into()))?;
        let hash = crate::interoperability::file_digest(path)?;
        let mut error = [0 as c_char; 1024];
        if !unsafe {
            sd_load_vision_projector(
                self.engine.as_ptr(),
                path_c.as_ptr(),
                error.as_mut_ptr(),
                error.len(),
            )
        } {
            return Err(native_error(&error));
        }
        self.vision_projector_path = Some(path_text);
        self.vision_projector_sha256 = Some(hash);
        self.clear_preparation_cache();
        Ok(())
    }

    /// Score typed decisions from the original state and one still image.
    /// The image is encoded by libmtmd; no generated caption is substituted.
    pub fn decide_vision(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse> {
        if self.execution_mode == ExecutionMode::Parallel {
            return self
                .decide_vision_batch(std::slice::from_ref(request), &[image])
                .map(|mut responses| responses.remove(0));
        }
        let mut cleanup = VisionCleanup::new(self.engine.as_ptr());
        let result = self.decide_vision_inner(request, image);
        if result.is_ok() {
            cleanup.complete();
        }
        result
    }

    /// Batch independent image requests into isolated native sequences. Each
    /// request keeps its own state and image; results preserve request and
    /// decision order. Fresh execution stays serial. Parallel execution uses
    /// at most `parallel_width` decisions per wave and currently requires
    /// single-token answer codes (at most 26 options).
    /// No native KV state survives this call, including validation failures.
    pub fn decide_vision_batch(
        &mut self,
        requests: &[DecisionRequest],
        images: &[&[u8]],
    ) -> Result<Vec<DecisionResponse>> {
        let mut cleanup = VisionCleanup::new(self.engine.as_ptr());
        let responses = (|| {
            if requests.len() != images.len() {
                return Err(Error::Invalid(
                    "one image is required per vision request".into(),
                ));
            }
            for (request, image) in requests.iter().zip(images) {
                self.validate_vision_request(request, image)?;
            }
            if requests.is_empty() {
                // No native call will establish a clean request boundary.
                unsafe { sd_clear(self.engine.as_ptr()) };
                return Ok(Vec::new());
            }
            if self.execution_mode == ExecutionMode::Fresh {
                return requests
                    .iter()
                    .zip(images)
                    .map(|(request, image)| self.decide_vision_inner(request, image))
                    .collect();
            }
            let questions = requests
                .iter()
                .zip(images)
                .flat_map(|(request, image)| {
                    request
                        .decisions
                        .iter()
                        .map(move |decision| (&request.state, decision, *image))
                })
                .collect::<Vec<_>>();
            if questions
                .iter()
                .any(|(_, decision, _)| decision.options().len() > 26)
            {
                return Err(Error::Invalid(
                    "parallel vision supports at most 26 options; use fresh execution for multi-token answer codes".into(),
                ));
            }
            let vocab = unsafe { sd_vocab_size(self.engine.as_ptr()) };
            if vocab <= 0 {
                return Err(Error::Backend("invalid vocabulary size".into()));
            }
            let width = self.parallel_width.min(questions.len());
            let mut results = Vec::with_capacity(questions.len());
            for wave in questions.chunks(width) {
                let started = Instant::now();
                let prepared = wave
                    .iter()
                    .map(|(state, decision, _)| self.prepare_vision(state, decision))
                    .collect::<Result<Vec<_>>>()?;
                self.timings.prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
                let before = "Image:\n";
                // All backing prompt strings and image slices outlive the call.
                let inputs = prepared
                    .iter()
                    .zip(wave)
                    .map(|(prepared, (_, _, image))| NativeVisionInput {
                        prefix: prepared.parts[0].text.as_ptr().cast(),
                        prefix_len: prepared.parts[0].text.len(),
                        data_before: before.as_ptr().cast(),
                        before_len: before.len(),
                        image: image.as_ptr(),
                        image_len: image.len(),
                        data_after: prepared.after.as_ptr().cast(),
                        after_len: prepared.after.len(),
                        suffix: prepared.parts[2].text.as_ptr().cast(),
                        suffix_len: prepared.parts[2].text.len(),
                    })
                    .collect::<Vec<_>>();
                let compact = self.evidence_transfer == EvidenceTransfer::Compact;
                let candidate_ids = prepared
                    .iter()
                    .map(|p| p.candidates.as_ptr())
                    .collect::<Vec<_>>();
                let candidate_counts = prepared
                    .iter()
                    .map(|p| p.candidates.len())
                    .collect::<Vec<_>>();
                let mut offsets = vec![0usize];
                for count in &candidate_counts {
                    offsets.push(offsets.last().unwrap() + count);
                }
                let mut logits = vec![
                    0.0;
                    if compact {
                        *offsets.last().unwrap()
                    } else {
                        wave.len() * vocab as usize
                    }
                ];
                let mut normalizers = vec![0.0; wave.len()];
                let mut input_tokens = vec![0; wave.len()];
                let mut error = [0 as c_char; 1024];
                self.failure_stage = (
                    "parallel_vision_inference",
                    FailureKind::BackendFailure,
                    None,
                );
                let started = Instant::now();
                let ok = unsafe {
                    if compact {
                        sd_forward_vision_parallel_compact(
                            self.engine.as_ptr(),
                            inputs.as_ptr(),
                            wave.len() as i32,
                            width as u32,
                            self.parallel_context_dynamic,
                            candidate_ids.as_ptr(),
                            candidate_counts.as_ptr(),
                            logits.as_mut_ptr(),
                            logits.len(),
                            normalizers.as_mut_ptr(),
                            input_tokens.as_mut_ptr(),
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    } else {
                        sd_forward_vision_parallel(
                            self.engine.as_ptr(),
                            inputs.as_ptr(),
                            wave.len() as i32,
                            width as u32,
                            self.parallel_context_dynamic,
                            logits.as_mut_ptr(),
                            logits.len(),
                            input_tokens.as_mut_ptr(),
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    }
                };
                self.timings.native_ms += started.elapsed().as_secs_f64() * 1000.0;
                if !ok {
                    return Err(native_error(&error));
                }
                let started = Instant::now();
                for (index, ((_, decision, _), prepared)) in wave.iter().zip(&prepared).enumerate()
                {
                    self.failure_stage = (
                        "score",
                        FailureKind::InvalidEvidence,
                        Some(decision.id.clone()),
                    );
                    let mut result = if compact {
                        ExactEvidence::from_native_summary(
                            decision,
                            &logits[offsets[index]..offsets[index + 1]],
                            &prepared.candidates,
                            vocab as usize,
                            normalizers[index],
                        )?
                        .score(
                            decision,
                            input_tokens[index],
                            &self.policy,
                        )?
                    } else {
                        let offset = index * vocab as usize;
                        score_logits(
                            decision,
                            &logits[offset..offset + vocab as usize],
                            &prepared.candidates,
                            input_tokens[index],
                            &self.policy,
                        )?
                    };
                    self.restore_code_metadata(&mut result);
                    results.push(result);
                }
                self.timings.score_ms += started.elapsed().as_secs_f64() * 1000.0;
                self.timings.decisions += wave.len();
            }
            let mut results = results.into_iter();
            Ok(requests
                .iter()
                .map(|request| {
                    let mut info = self.vision_info(request);
                    info.prompt_version.push_str("/vision-parallel-v1");
                    DecisionResponse {
                        backend: info,
                        policy: self.policy.clone(),
                        results: results.by_ref().take(request.decisions.len()).collect(),
                    }
                })
                .collect())
        })();
        if responses.is_ok() {
            cleanup.complete();
        }
        responses
    }

    /// Native execution counters for the latest parallel image wave. These
    /// remain available after request-local KV cleanup and measure actual
    /// projector batches and decoder sequences, rather than HTTP grouping.
    pub fn vision_batch_metrics(&self) -> Result<NativeVisionBatchMetrics> {
        let mut metrics = NativeVisionBatchMetrics::default();
        if !unsafe { sd_vision_batch_metrics(self.engine.as_ptr(), &mut metrics) } {
            return Err(Error::Backend(
                "unable to read native vision batch metrics".into(),
            ));
        }
        Ok(metrics)
    }

    /// Enable compatible opt-in vision optimizations on a resident backend.
    /// Compute batch size and flash attention are fixed when constructing the
    /// backend; use `ComputeOptions::vision_optimized()` for those settings.
    /// This profile can change scores through parallel decoder/projector math.
    pub fn enable_vision_optimizations(&mut self) -> Result<()> {
        if self.vision_projector_path.is_none()
            || self.output_head.is_some()
            || !self.calibrations.is_empty()
            || self.collect_features
        {
            return Err(Error::Invalid("vision optimizations require a loaded projector without output heads, calibration or feature export".into()));
        }
        if unsafe { sd_recurrent_or_hybrid(self.engine.as_ptr()) } {
            return Err(Error::Invalid(
                "optimized vision requires a non-recurrent, non-hybrid decoder".into(),
            ));
        }
        self.set_parallel_width(4)?;
        self.set_execution_mode(ExecutionMode::Parallel);
        self.set_parallel_context_dynamic(true);
        self.set_evidence_transfer(EvidenceTransfer::Compact)?;
        self.set_preparation_cache(PreparationCacheConfig {
            max_entries: 128,
            max_bytes: 8 * 1024 * 1024,
        });
        self.set_vision_projector_reuse(true);
        Ok(())
    }

    /// Enable preparation and evidence-copy optimizations while retaining the
    /// original serial native execution, batch/ubatch 256 and flash attention off.
    /// Exact prompt preparation is cached; inference results, image embeddings
    /// and KV state are never cached. Compact evidence keeps the full-vocabulary
    /// normalizer and supports at most 26 single-token answer codes.
    /// Every compatibility check completes before changing backend settings.
    pub fn enable_vision_preserving_optimizations(&mut self) -> Result<()> {
        if self.vision_projector_path.is_none()
            || self.output_head.is_some()
            || !self.calibrations.is_empty()
            || self.collect_features
        {
            return Err(Error::Invalid("preserving vision optimizations require a loaded projector without output heads, calibration or feature export".into()));
        }
        if self.compute.batch != 256
            || self.compute.ubatch != 256
            || self.compute.flash_attention != FlashAttention::Off
        {
            return Err(Error::Invalid("preserving vision optimizations require batch 256, ubatch 256 and flash attention off".into()));
        }
        self.set_parallel_width(1)?;
        self.set_execution_mode(ExecutionMode::Fresh);
        self.set_parallel_context_dynamic(false);
        self.set_evidence_transfer(EvidenceTransfer::Compact)?;
        self.set_preparation_cache(PreparationCacheConfig {
            max_entries: 128,
            max_bytes: 8 * 1024 * 1024,
        });
        self.set_vision_projector_reuse(false);
        Ok(())
    }

    /// Opt in to reusing identical image embeddings inside one native wave.
    /// Image bytes and ordered projector chunks must match exactly. Different
    /// projector batch shapes can change scores, so this is disabled by default.
    pub fn set_vision_projector_reuse(&mut self, enabled: bool) {
        unsafe { sd_set_vision_projector_reuse(self.engine.as_ptr(), enabled) };
        self.vision_projector_reuse = enabled;
    }

    fn prepare_vision(
        &self,
        state: &serde_json::Value,
        decision: &Decision,
    ) -> Result<PreparedVision> {
        let key = self
            .preparation_cache_enabled
            .then(|| serde_json::to_string(&(state, decision)))
            .transpose()
            .map_err(|e| Error::Invalid(e.to_string()))?;
        if let Some(key) = &key
            && let Some(prepared) = self
                .vision_prepared_cache
                .borrow_mut()
                .get("vision-model-local-v1", key)
        {
            return Ok(prepared);
        }
        let parts = match &self.chat_skeleton {
            Some(skeleton) => crate::prompt::compile_model_prompt_with_detail(
                skeleton,
                state,
                decision,
                self.prompt_layout,
                self.prompt_detail,
                self.code_rotation,
            )?,
            None => compile_prompt_with_detail(
                state,
                decision,
                self.prompt_layout,
                self.prompt_detail,
                self.code_rotation,
            ),
        };
        let candidates = if decision.options().len() <= 26 {
            self.prepare_candidate_tokens(decision, &parts[2].text)
                .map_err(|failure| Error::Backend(failure.message))?
        } else {
            Vec::new()
        };
        let after = format!("\nDecision data:\n{}", parts[1].text);
        let prepared = PreparedVision {
            parts,
            after,
            candidates,
        };
        if let Some(key) = key {
            self.vision_prepared_cache.borrow_mut().insert(
                "vision-model-local-v1".into(),
                key,
                prepared.clone(),
            );
        }
        Ok(prepared)
    }

    fn vision_info(&self, request: &DecisionRequest) -> BackendInfo {
        let mut info = self.info_for_request(request);
        info.runtime = "local-libllama-mtmd".into();
        info.prompt_version.push_str("/vision-image-v1");
        info
    }

    fn validate_vision_request(&self, request: &DecisionRequest, image: &[u8]) -> Result<()> {
        request.validate()?;
        if self.vision_projector_path.is_none() {
            return Err(Error::Invalid("vision projector is not loaded".into()));
        }
        crate::validate_image(image)?;
        if !matches!(
            self.execution_mode,
            ExecutionMode::Fresh | ExecutionMode::Parallel
        ) || self.output_head.is_some()
            || !self.calibrations.is_empty()
            || self.collect_features
        {
            return Err(Error::Invalid(
                "vision decisions require fresh or parallel execution without output heads, calibration or feature export".into(),
            ));
        }
        let marker = unsafe { CStr::from_ptr(sd_vision_marker()) }
            .to_str()
            .map_err(|_| Error::Backend("invalid native vision marker".into()))?;
        if serde_json::to_string(request)
            .map_err(|e| Error::Invalid(e.to_string()))?
            .contains(marker)
        {
            return Err(Error::Invalid(
                "request text contains reserved vision marker".into(),
            ));
        }
        Ok(())
    }

    fn decide_vision_inner(
        &mut self,
        request: &DecisionRequest,
        image: &[u8],
    ) -> Result<DecisionResponse> {
        self.validate_vision_request(request, image)?;
        let mut results = Vec::with_capacity(request.decisions.len());
        for decision in &request.decisions {
            let started = Instant::now();
            let prepared = self.prepare_vision(&request.state, decision)?;
            self.timings.prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
            let parts = &prepared.parts;
            // Media stays in the user-data segment; control tokens remain in
            // the trusted prefix and suffix.
            let before = "Image:\n";
            let after = &prepared.after;
            let size = unsafe { sd_vocab_size(self.engine.as_ptr()) };
            if size <= 0 {
                return Err(Error::Backend("invalid vocabulary size".into()));
            }
            if decision.options().len() > 26 {
                if self.evidence_transfer == EvidenceTransfer::Compact {
                    return Err(Error::Invalid("compact vision supports at most 26 options; use full evidence for multi-token codes".into()));
                }
                let paths = self.prepare_code_paths(&parts[2].text, decision.options().len())?;
                let mut input_tokens = 0;
                let mut prefix_evaluations = 0;
                let mut evaluated_tokens = 0;
                let scores = crate::codes::sequence_log_probabilities(&paths, |prefix| {
                    self.logits_buffer.resize(size as usize, 0.0);
                    let mut observed_tokens = 0;
                    let mut error = [0 as c_char; 1024];
                    let start = Instant::now();
                    let ok = unsafe {
                        sd_forward_vision(
                            self.engine.as_ptr(),
                            parts[0].text.as_ptr().cast(),
                            parts[0].text.len(),
                            before.as_ptr().cast(),
                            before.len(),
                            image.as_ptr(),
                            image.len(),
                            after.as_ptr().cast(),
                            after.len(),
                            parts[2].text.as_ptr().cast(),
                            parts[2].text.len(),
                            prefix.as_ptr(),
                            prefix.len(),
                            self.logits_buffer.as_mut_ptr(),
                            self.logits_buffer.len(),
                            &mut observed_tokens,
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    };
                    self.timings.native_ms += start.elapsed().as_secs_f64() * 1000.0;
                    if !ok {
                        return Err(native_error(&error));
                    }
                    if prefix.is_empty() {
                        input_tokens = observed_tokens;
                    }
                    prefix_evaluations += 1;
                    evaluated_tokens += observed_tokens + prefix.len();
                    Ok(self.logits_buffer.clone())
                })?;
                let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let log_mass = max + scores.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
                if log_mass > 1e-8 {
                    return Err(Error::Backend(
                        "overlapping answer-code probability mass".into(),
                    ));
                }
                let representatives: Vec<_> = paths
                    .iter()
                    .map(|path| if path.len() == 1 { path[0] } else { -1 })
                    .collect();
                let mut scored = crate::decision::score_candidate_logits(
                    decision,
                    &scores,
                    &representatives,
                    input_tokens,
                    log_mass.exp().min(1.0),
                    &self.policy,
                )?;
                for (score, path) in scored.scores.iter_mut().zip(paths) {
                    score.token_ids = path;
                }
                scored.scoring_method = "code_sequence_conditional_softmax_v1".into();
                scored.code_prefix_evaluations = prefix_evaluations;
                scored.code_evaluated_tokens = evaluated_tokens;
                self.restore_code_metadata(&mut scored);
                results.push(scored);
                self.timings.decisions += 1;
                continue;
            }
            let candidates = &prepared.candidates;
            let compact = self.evidence_transfer == EvidenceTransfer::Compact;
            self.logits_buffer.resize(
                if compact {
                    candidates.len()
                } else {
                    size as usize
                },
                0.0,
            );
            let mut input_tokens = 0;
            let mut normalizer = 0.0;
            let mut error = [0 as c_char; 1024];
            let start = Instant::now();
            let ok = unsafe {
                if compact {
                    sd_forward_vision_compact(
                        self.engine.as_ptr(),
                        parts[0].text.as_ptr().cast(),
                        parts[0].text.len(),
                        before.as_ptr().cast(),
                        before.len(),
                        image.as_ptr(),
                        image.len(),
                        after.as_ptr().cast(),
                        after.len(),
                        parts[2].text.as_ptr().cast(),
                        parts[2].text.len(),
                        std::ptr::null(),
                        0,
                        candidates.as_ptr(),
                        candidates.len(),
                        self.logits_buffer.as_mut_ptr(),
                        self.logits_buffer.len(),
                        &mut normalizer,
                        &mut input_tokens,
                        error.as_mut_ptr(),
                        error.len(),
                    )
                } else {
                    sd_forward_vision(
                        self.engine.as_ptr(),
                        parts[0].text.as_ptr().cast(),
                        parts[0].text.len(),
                        before.as_ptr().cast(),
                        before.len(),
                        image.as_ptr(),
                        image.len(),
                        after.as_ptr().cast(),
                        after.len(),
                        parts[2].text.as_ptr().cast(),
                        parts[2].text.len(),
                        std::ptr::null(),
                        0,
                        self.logits_buffer.as_mut_ptr(),
                        self.logits_buffer.len(),
                        &mut input_tokens,
                        error.as_mut_ptr(),
                        error.len(),
                    )
                }
            };
            self.timings.native_ms += start.elapsed().as_secs_f64() * 1000.0;
            if !ok {
                return Err(native_error(&error));
            }
            let started = Instant::now();
            let mut scored = if compact {
                ExactEvidence::from_native_summary(
                    decision,
                    &self.logits_buffer,
                    candidates,
                    size as usize,
                    normalizer,
                )?
                .score(decision, input_tokens, &self.policy)?
            } else {
                score_logits(
                    decision,
                    &self.logits_buffer,
                    candidates,
                    input_tokens,
                    &self.policy,
                )?
            };
            self.timings.score_ms += started.elapsed().as_secs_f64() * 1000.0;
            self.restore_code_metadata(&mut scored);
            results.push(scored);
            self.timings.decisions += 1;
        }
        Ok(DecisionResponse {
            backend: self.vision_info(request),
            policy: self.policy.clone(),
            results,
        })
    }
}
