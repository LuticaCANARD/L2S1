use super::*;
use crate::interoperability::digest;

impl LlamaBackend {
    pub fn identity(&self) -> ModelIdentity {
        let info = self.info();
        let template = unsafe { CStr::from_ptr(sd_chat_template(self.engine.as_ptr())) }.to_bytes();
        ModelIdentity {
            weights_sha256: self.weights_sha256.clone(),
            template_sha256: digest(template),
            prompt_profile: info.prompt_profile,
            prompt_version: info.prompt_version,
            loaded_runtime_sha256: self.loaded_runtime_sha256.clone(),
            runtime_build_sha256: env!("L2S1_RUNTIME_BUILD_SHA256").into(),
            adapter_sha256: self.adapter_sha256.clone(),
            head_sha256: self.head_sha256.clone(),
            device: unsafe { CStr::from_ptr(sd_device(self.engine.as_ptr())) }
                .to_string_lossy()
                .into_owned(),
            compute: self.compute,
            execution_mode: self.execution_mode,
            parallel_width: info.parallel_width,
        }
    }
    pub fn inspect(&self) -> ModelInspection {
        let hybrid = unsafe { sd_recurrent_or_hybrid(self.engine.as_ptr()) };
        let mut modes = vec![
            ExecutionMode::Fresh,
            ExecutionMode::PrefixReuse,
            ExecutionMode::StateRestore,
        ];
        if !hybrid {
            modes.push(ExecutionMode::Parallel);
        }
        ModelInspection {
            identity: self.identity(),
            capabilities: ModelCapabilities {
                architecture: self.architecture.clone(),
                vocabulary_size: unsafe { sd_vocab_size(self.engine.as_ptr()) } as usize,
                configured_context: self.context,
                training_context: unsafe { sd_training_context(self.engine.as_ptr()) },
                recurrent_or_hybrid: hybrid,
                exact_full_vocabulary_logits: true,
                execution_modes: modes,
                prefix_reuse_fallback: hybrid.then(|| "recurrent_or_hybrid_memory".into()),
                hidden_features: self.architecture == "gemma4",
                snapshot_limit_bytes: self.snapshot_limit_bytes,
                evidence_status: "loaded_metadata; request preflight and measured quality are separate; state_restore is experimental",
            },
        }
    }
    pub fn set_snapshot_limit_bytes(&mut self, limit: usize) {
        unsafe { sd_clear(self.engine.as_ptr()) };
        self.snapshot_limit_bytes = limit;
    }
    pub fn load_calibration(&mut self, path: &Path) -> Result<()> {
        let bytes = std::fs::read(path).map_err(|e| Error::Backend(e.to_string()))?;
        let artifact: ScalarCalibration =
            serde_json::from_slice(&bytes).map_err(|e| Error::Invalid(e.to_string()))?;
        self.register_calibration(artifact)
    }
    pub fn register_calibration(&mut self, artifact: ScalarCalibration) -> Result<()> {
        if self.output_head.is_some() {
            return Err(Error::Invalid(
                "scalar calibration and output head cannot be combined".into(),
            ));
        }
        artifact.validate(&self.identity())?;
        if self
            .calibrations
            .iter()
            .any(|a| a.decision_id == artifact.decision_id || a.id == artifact.id)
        {
            return Err(Error::Invalid("duplicate calibration task or ID".into()));
        }
        self.calibrations.push(artifact);
        Ok(())
    }
    pub fn clear_calibrations(&mut self) {
        self.calibrations.clear();
        unsafe { sd_clear(self.engine.as_ptr()) };
    }
    pub(super) fn check_artifacts(&self, request: &DecisionRequest) -> Result<()> {
        if let Some(head) = &self.output_head {
            self.check_head_config(head)?;
            for d in &request.decisions {
                head.applies_to(d)?;
            }
        }
        if !self.calibrations.is_empty() {
            let identity = self.identity();
            for artifact in &self.calibrations {
                artifact.validate(&identity)?;
                for d in &request.decisions {
                    artifact.applies_to(d)?;
                }
            }
        }
        Ok(())
    }
    pub(super) fn apply_calibration(
        &self,
        decision: &Decision,
        mut result: DecisionResult,
    ) -> Result<DecisionResult> {
        for artifact in &self.calibrations {
            result = artifact.apply(decision, result, &self.policy)?;
        }
        Ok(result)
    }
    /// Tokenizes and checks the actual answer boundary; performs no forward pass.
    pub fn preflight(
        &self,
        request: &DecisionRequest,
    ) -> std::result::Result<PreflightReport, DecisionFailure> {
        request
            .validate()
            .map_err(|e| DecisionFailure::new(FailureKind::InvalidRequest, "validate", None, e))?;
        self.check_artifacts(request).map_err(|e| {
            DecisionFailure::new(FailureKind::IncompatibleArtifact, "artifact", None, e)
        })?;
        let model = self.inspect();
        if !model
            .capabilities
            .execution_modes
            .contains(&self.execution_mode)
        {
            return Err(DecisionFailure::new(
                FailureKind::UnsupportedCapability,
                "execution_mode",
                None,
                "parallel prefix sharing is unsupported for recurrent/hybrid models",
            ));
        }
        let mut decisions = Vec::new();
        for d in &request.decisions {
            let (input, candidates) = self.prepare_checked(&request.state, d)?;
            decisions.push(PreparedDecisionReport {
                decision_id: d.id.clone(),
                input_tokens: input.len(),
                prompt_tokens_sha256: token_hash(&input),
                candidate_token_ids: candidates,
                option_ids: d.options().into_iter().map(|o| o.id).collect(),
            });
        }
        Ok(PreflightReport { model, decisions })
    }
    /// Timings belong only to this call. Preflight hashing/validation are outside inference timings.
    pub fn decide_detailed(
        &mut self,
        request: &DecisionRequest,
    ) -> std::result::Result<DiagnosticResponse, DecisionFailure> {
        unsafe { sd_clear(self.engine.as_ptr()) };
        let preflight = self.preflight(request)?;
        let previous = std::mem::take(&mut self.timings);
        self.failure_stage = ("inference", FailureKind::BackendFailure, None);
        let outcome = self.decide(request);
        let timings = std::mem::replace(&mut self.timings, previous);
        let response = outcome.map_err(|e| {
            DecisionFailure::new(
                self.failure_stage.1,
                self.failure_stage.0,
                self.failure_stage.2.as_deref(),
                e,
            )
        })?;
        let decisions = response
            .results
            .iter()
            .zip(preflight.decisions)
            .enumerate()
            .map(|(i, (r, p))| {
                let fallback = if self.execution_mode == ExecutionMode::PrefixReuse
                    && preflight.model.capabilities.recurrent_or_hybrid
                {
                    Some("recurrent_or_hybrid_memory".into())
                } else if self.execution_mode == ExecutionMode::StateRestore {
                    self.restore_metrics.fallback_reason.clone()
                } else if self.execution_mode == ExecutionMode::PrefixReuse
                    && r.reused_prefix_tokens == 0
                {
                    Some(
                        if i == 0 {
                            "request_start"
                        } else {
                            "no_aligned_prefix_or_memory_removal_failed"
                        }
                        .into(),
                    )
                } else {
                    None
                };
                let effective = if matches!(
                    self.execution_mode,
                    ExecutionMode::PrefixReuse | ExecutionMode::StateRestore
                ) && fallback.is_some()
                {
                    ExecutionMode::Fresh
                } else {
                    self.execution_mode
                };
                ExecutionDiagnostic {
                    decision_id: r.id.clone(),
                    prompt_tokens_sha256: p.prompt_tokens_sha256,
                    requested_mode: self.execution_mode,
                    effective_mode: effective,
                    reused_prefix_tokens: r.reused_prefix_tokens,
                    fallback_reason: fallback,
                    evidence_kind: evidence_kind(&r.scoring_method),
                    calibration_id: r.calibration_id.clone(),
                }
            })
            .collect();
        Ok(DiagnosticResponse {
            response,
            model: preflight.model.identity,
            decisions,
            timings,
            state_restore: self.restore_metrics.clone(),
        })
    }
    pub(super) fn evaluate_restore(
        &mut self,
        request: &DecisionRequest,
    ) -> Result<Vec<DecisionResult>> {
        if !unsafe { sd_set_features(self.engine.as_ptr(), false) } {
            return Err(Error::Backend("feature mode reset failed".into()));
        }
        let started = Instant::now();
        let prepared = request
            .decisions
            .iter()
            .map(|d| self.prepare(&request.state, d))
            .collect::<Result<Vec<_>>>()?;
        self.timings.prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
        let pointers: Vec<_> = prepared.iter().map(|p| p.0.as_ptr()).collect();
        let counts: Vec<_> = prepared.iter().map(|p| p.0.len() as i32).collect();
        let vocab = unsafe { sd_vocab_size(self.engine.as_ptr()) } as usize;
        let length = vocab
            .checked_mul(prepared.len())
            .ok_or_else(|| Error::Invalid("snapshot output buffer overflow".into()))?;
        let sequences = i32::try_from(prepared.len())
            .map_err(|_| Error::Invalid("too many snapshot decisions".into()))?;
        let mut logits = vec![0.0; length];
        let mut reused = vec![0; prepared.len()];
        let mut metrics = NativeRestoreMetrics::default();
        let mut error = [0 as c_char; 1024];
        self.failure_stage = ("state_restore", FailureKind::BackendFailure, None);
        let started = Instant::now();
        let ok = unsafe {
            sd_forward_restore(
                self.engine.as_ptr(),
                pointers.as_ptr(),
                counts.as_ptr(),
                sequences,
                self.snapshot_limit_bytes,
                reused.as_mut_ptr(),
                logits.as_mut_ptr(),
                logits.len(),
                &mut metrics,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        self.timings.native_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.restore_metrics = StateRestoreMetrics {
            snapshot_bytes: metrics.snapshot_bytes,
            save_ms: metrics.save_ms,
            restore_ms: metrics.restore_ms,
            prefill_ms: metrics.prefill_ms,
            suffix_ms: metrics.suffix_ms,
            restores: metrics.restores,
            fallback_reason: match metrics.fallback {
                0 => None,
                1 => Some("no_aligned_common_prefix".into()),
                2 => Some("snapshot_memory_budget".into()),
                3 => Some("snapshot_save_unavailable".into()),
                _ => Some("snapshot_restore_unavailable".into()),
            },
        };
        if !ok {
            return Err(native_error(&error));
        }
        let started = Instant::now();
        let mut results = Vec::new();
        for (i, (d, (input, candidates))) in request.decisions.iter().zip(&prepared).enumerate() {
            self.failure_stage = ("score", FailureKind::InvalidEvidence, Some(d.id.clone()));
            let mut result = score_logits(
                d,
                &logits[i * vocab..(i + 1) * vocab],
                candidates,
                input.len(),
                &self.policy,
            )?;
            result.reused_prefix_tokens = reused[i] as usize;
            results.push(self.apply_calibration(d, result)?);
        }
        self.timings.score_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.timings.decisions += results.len();
        Ok(results)
    }
}
fn token_hash(tokens: &[i32]) -> String {
    digest(
        &tokens
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

fn evidence_kind(scoring_method: &str) -> &'static str {
    if scoring_method.starts_with("learned_") {
        "learned_head_scores_with_native_base_mass_v1"
    } else {
        "native_full_vocabulary_logits_v1"
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn diagnostic_evidence_distinguishes_learned_head_scores() {
        assert_eq!(
            super::evidence_kind("learned_hidden_softmax_with_base_mass_v1"),
            "learned_head_scores_with_native_base_mass_v1"
        );
        assert_eq!(
            super::evidence_kind("learned_logit_affine_softmax_with_base_mass_v1"),
            "learned_head_scores_with_native_base_mass_v1"
        );
        assert_eq!(
            super::evidence_kind("temperature_softmax_with_base_mass_v1"),
            "native_full_vocabulary_logits_v1"
        );
    }
}
