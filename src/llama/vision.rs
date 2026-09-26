//! Direct image input with fresh evaluation or isolated parallel sequences.
use super::*;

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
        unsafe { sd_clear(self.engine.as_ptr()) };
        let result = if self.execution_mode == ExecutionMode::Parallel {
            self.decide_vision_batch(std::slice::from_ref(request), &[image])
                .map(|mut responses| responses.remove(0))
        } else {
            self.decide_vision_inner(request, image)
        };
        unsafe { sd_clear(self.engine.as_ptr()) };
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
        unsafe { sd_clear(self.engine.as_ptr()) };
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
                    .map(|(state, decision, _)| {
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
                        let (_, candidates) = self.prepare(state, decision)?;
                        let after = format!("\nDecision data:\n{}", parts[1].text);
                        Ok((parts, after, candidates))
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.timings.prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
                let before = "Image:\n";
                // All backing prompt strings and image slices outlive the call.
                let inputs = prepared
                    .iter()
                    .zip(wave)
                    .map(|((parts, after, _), (_, _, image))| NativeVisionInput {
                        prefix: parts[0].text.as_ptr().cast(),
                        prefix_len: parts[0].text.len(),
                        data_before: before.as_ptr().cast(),
                        before_len: before.len(),
                        image: image.as_ptr(),
                        image_len: image.len(),
                        data_after: after.as_ptr().cast(),
                        after_len: after.len(),
                        suffix: parts[2].text.as_ptr().cast(),
                        suffix_len: parts[2].text.len(),
                    })
                    .collect::<Vec<_>>();
                let mut logits = vec![0.0; wave.len() * vocab as usize];
                let mut input_tokens = vec![0; wave.len()];
                let mut error = [0 as c_char; 1024];
                self.failure_stage = (
                    "parallel_vision_inference",
                    FailureKind::BackendFailure,
                    None,
                );
                let started = Instant::now();
                let ok = unsafe {
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
                };
                self.timings.native_ms += started.elapsed().as_secs_f64() * 1000.0;
                if !ok {
                    return Err(native_error(&error));
                }
                let started = Instant::now();
                for (index, ((_, decision, _), (_, _, candidates))) in
                    wave.iter().zip(&prepared).enumerate()
                {
                    self.failure_stage = (
                        "score",
                        FailureKind::InvalidEvidence,
                        Some(decision.id.clone()),
                    );
                    let offset = index * vocab as usize;
                    let mut result = score_logits(
                        decision,
                        &logits[offset..offset + vocab as usize],
                        candidates,
                        input_tokens[index],
                        &self.policy,
                    )?;
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
        unsafe { sd_clear(self.engine.as_ptr()) };
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
        ) || self.evidence_transfer != EvidenceTransfer::Full
            || self.output_head.is_some()
            || !self.calibrations.is_empty()
            || self.collect_features
        {
            return Err(Error::Invalid(
                "vision decisions require fresh or parallel full-evidence execution without output heads, calibration or feature export".into(),
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
            let parts = match &self.chat_skeleton {
                Some(skeleton) => crate::prompt::compile_model_prompt_with_detail(
                    skeleton,
                    &request.state,
                    decision,
                    self.prompt_layout,
                    self.prompt_detail,
                    self.code_rotation,
                )?,
                None => compile_prompt_with_detail(
                    &request.state,
                    decision,
                    self.prompt_layout,
                    self.prompt_detail,
                    self.code_rotation,
                ),
            };
            // Keep media in the user-data segment; model control tokens remain
            // confined to the trusted prefix and suffix.
            let before = "Image:\n";
            let after = format!("\nDecision data:\n{}", parts[1].text);
            let size = unsafe { sd_vocab_size(self.engine.as_ptr()) };
            if size <= 0 {
                return Err(Error::Backend("invalid vocabulary size".into()));
            }
            if decision.options().len() > 26 {
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
            let (_, candidates) = self.prepare(&request.state, decision)?;
            self.logits_buffer.resize(size as usize, 0.0);
            let mut input_tokens = 0;
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
                    std::ptr::null(),
                    0,
                    self.logits_buffer.as_mut_ptr(),
                    self.logits_buffer.len(),
                    &mut input_tokens,
                    error.as_mut_ptr(),
                    error.len(),
                )
            };
            self.timings.native_ms += start.elapsed().as_secs_f64() * 1000.0;
            if !ok {
                return Err(native_error(&error));
            }
            let mut scored = score_logits(
                decision,
                &self.logits_buffer,
                &candidates,
                input_tokens,
                &self.policy,
            )?;
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
