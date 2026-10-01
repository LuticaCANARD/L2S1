use super::*;

impl LlamaBackend {
    pub fn reasoning(&self) -> ReasoningOptions {
        self.reasoning
    }

    /// Configuration admission only; request token length and closure-token
    /// validation still happen before generation.
    pub fn supports_thinking(&self) -> bool {
        self.check_thinking_config().is_ok()
    }

    pub fn set_reasoning(&mut self, options: ReasoningOptions) -> Result<()> {
        options.validate()?;
        let previous = self.reasoning;
        self.reasoning = options;
        let admitted = self.check_reasoning_config().and_then(|()| {
            if options.is_thinking() && self.tokenize("</think>", true)?.len() != 1 {
                return Err(Error::Invalid(
                    "thinking requires a single native </think> token".into(),
                ));
            }
            Ok(())
        });
        if let Err(error) = admitted {
            self.reasoning = previous;
            return Err(error);
        }
        unsafe { sd_clear(self.engine.as_ptr()) };
        self.clear_preparation_cache();
        Ok(())
    }

    pub(super) fn check_reasoning_config(&self) -> Result<()> {
        self.reasoning.validate()?;
        if !self.reasoning.is_thinking() {
            return Ok(());
        }
        self.check_thinking_config()
    }

    fn check_thinking_config(&self) -> Result<()> {
        if self.profile != PromptProfile::Qwen3 || self.architecture != "qwen3" {
            return Err(Error::Invalid("thinking currently requires a dense Qwen3 text model and qwen3 prompt profile; vision and other architectures are unsupported".into()));
        }
        if self.execution_mode != ExecutionMode::Fresh
            || self.preparation_cache_enabled
            || self.evidence_transfer != EvidenceTransfer::Full
            || self.vision_projector_path.is_some()
            || self.lora_path.is_some()
            || self.output_head.is_some()
            || self.collect_features
            || self.has_calibrations()
        {
            return Err(Error::Invalid("thinking requires fresh text execution, full logits, disabled preparation cache, and no projector, LoRA, head, feature export or calibration".into()));
        }
        Ok(())
    }

    pub(super) fn evaluate_thinking(
        &mut self,
        state: crate::PromptInput<'_>,
        decision: &Decision,
    ) -> Result<DecisionResult> {
        self.check_reasoning_config()?;
        if decision.options().len() > 26 {
            return Err(Error::Invalid(
                "thinking currently supports at most 26 single-token answer codes".into(),
            ));
        }
        if !unsafe { sd_set_features(self.engine.as_ptr(), false) } {
            return Err(Error::Backend(
                "cannot disable thinking feature extraction".into(),
            ));
        }
        let started = Instant::now();
        let (input, candidates) = self.prepare(state, decision)?;
        let closure = self.tokenize("</think>", true)?;
        let suffix = self.tokenize("\n\n", true)?;
        if closure.len() != 1 {
            return Err(Error::Invalid(
                "thinking requires a single native </think> token".into(),
            ));
        }
        self.timings.prepare_ms += started.elapsed().as_secs_f64() * 1000.;
        let size = unsafe { sd_vocab_size(self.engine.as_ptr()) };
        if size <= 0 {
            return Err(Error::Backend("invalid vocabulary size".into()));
        }
        self.logits_buffer.resize(size as usize, 0.);
        let mut generated_tokens = 0;
        let mut completed = false;
        let mut error = [0 as c_char; 1024];
        self.failure_stage = (
            "reasoning",
            FailureKind::BackendFailure,
            Some(decision.id.clone()),
        );
        let started = Instant::now();
        let ok = unsafe {
            sd_forward_thinking(
                self.engine.as_ptr(),
                input.as_ptr(),
                input.len() as i32,
                closure[0],
                suffix.as_ptr(),
                suffix.len(),
                self.reasoning.max_tokens,
                &mut generated_tokens,
                &mut completed,
                self.logits_buffer.as_mut_ptr(),
                self.logits_buffer.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        self.timings.native_ms += started.elapsed().as_secs_f64() * 1000.;
        if !ok {
            return Err(native_error(&error));
        }
        if !completed {
            return Err(Error::Backend(
                "reasoning_incomplete: native completion flag absent".into(),
            ));
        }
        let started = Instant::now();
        let mut result = score_logits(
            decision,
            &self.logits_buffer,
            &candidates,
            input.len(),
            &self.policy,
        )?;
        self.restore_code_metadata(&mut result);
        result.reasoning = Some(ReasoningUsage {
            mode: ReasoningMode::Thinking,
            generated_tokens,
            completed,
        });
        result.scoring_method.push_str("/thinking-greedy");
        self.timings.score_ms += started.elapsed().as_secs_f64() * 1000.;
        self.timings.decisions += 1;
        Ok(result)
    }
}
