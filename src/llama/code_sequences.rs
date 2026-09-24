//! Full-code likelihoods for automatically sized uppercase answer codes.
use super::*;

impl LlamaBackend {
    pub(super) fn check_sequence_config(&self) -> Result<()> {
        if !matches!(
            self.execution_mode,
            ExecutionMode::Fresh | ExecutionMode::PrefixReuse
        ) || self.evidence_transfer != EvidenceTransfer::Full
            || self.output_head.is_some()
            || !self.calibrations.is_empty()
            || self.collect_features
        {
            return Err(Error::Invalid("multi-letter codes currently require fresh or prefix-reuse execution and full evidence without output heads, scalar calibration or feature export".into()));
        }
        Ok(())
    }

    /// Exact assistant-continuation token paths in canonical semantic order.
    /// Unlike encode_decision, supports codes that span multiple tokens.
    pub fn encode_decision_sequences(
        &self,
        state: &serde_json::Value,
        decision: &Decision,
    ) -> Result<(Vec<i32>, Vec<Vec<i32>>)> {
        DecisionRequest {
            state: serde_json::Value::Null,
            decisions: vec![decision.clone()],
        }
        .validate()?;
        if decision.options().len() <= 26 {
            let (input, tokens) = self.prepare(state, decision)?;
            return Ok((input, tokens.into_iter().map(|t| vec![t]).collect()));
        }
        self.check_sequence_config()?;
        let key = self
            .preparation_cache_enabled
            .then(|| serde_json::to_string(&(state, decision)))
            .transpose()
            .map_err(|e| Error::Invalid(e.to_string()))?;
        if let Some(key) = &key
            && let Some((input, CandidateTokens::Sequences(paths))) = self
                .prepared_cache
                .borrow_mut()
                .get("model-local-code-sequences-v1", key)
        {
            return Ok((input, paths));
        }
        let prepared = self.prepare_code_sequences_uncached(state, decision)?;
        if let Some(key) = key {
            self.prepared_cache.borrow_mut().insert(
                "model-local-code-sequences-v1".into(),
                key,
                (
                    prepared.0.clone(),
                    CandidateTokens::Sequences(prepared.1.clone()),
                ),
            );
        }
        Ok(prepared)
    }

    fn prepare_code_sequences_uncached(
        &self,
        state: &serde_json::Value,
        decision: &Decision,
    ) -> Result<(Vec<i32>, Vec<Vec<i32>>)> {
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
        let mut input = Vec::new();
        for part in &parts {
            input.extend(self.tokenize(&part.text, part.parse_special)?);
        }
        let bos = unsafe { sd_required_bos(self.engine.as_ptr()) };
        if bos >= 0 && input.first() != Some(&bos) {
            input.insert(0, bos);
        }
        let paths =
            self.prepare_code_paths(&parts.last().unwrap().text, decision.options().len())?;
        let longest_prefix = paths.iter().map(|p| p.len() - 1).max().unwrap();
        if input
            .len()
            .checked_add(longest_prefix)
            .is_none_or(|n| n > self.context)
        {
            return Err(Error::Invalid(
                "prompt plus answer-code prefix exceeds context; truncation is disabled".into(),
            ));
        }
        Ok((input, paths))
    }

    /// Build exact assistant code continuations for both text and image prompts.
    /// Each execution path checks its own full input against the context limit.
    pub(super) fn prepare_code_paths(&self, tail: &str, count: usize) -> Result<Vec<Vec<i32>>> {
        let candidate_key = self
            .preparation_cache_enabled
            .then(|| format!("{count}:{tail}"));
        let cached = candidate_key.as_ref().and_then(|key| {
            self.candidate_cache
                .borrow_mut()
                .get("model-local-code-sequences-v1", key)
        });
        let cache_hit = matches!(cached, Some(CandidateTokens::Sequences(_)));
        let paths = if let Some(CandidateTokens::Sequences(paths)) = cached {
            paths
        } else {
            let tail_tokens = self.tokenize(tail, true)?;
            let mut paths = Vec::with_capacity(count);
            for i in 0..count {
                let code = option_code(i, count)?;
                let combined = self.tokenize(&format!("{tail}{code}"), true)?;
                if !combined.starts_with(&tail_tokens) || combined.len() <= tail_tokens.len() {
                    return Err(Error::Backend(format!(
                        "candidate {code} changes the assistant token boundary"
                    )));
                }
                paths.push(combined[tail_tokens.len()..].to_vec());
            }
            crate::codes::validate_code_paths(&paths)?;
            paths.rotate_right(self.code_rotation % count);
            paths
        };
        if !cache_hit && let Some(key) = candidate_key {
            self.candidate_cache.borrow_mut().insert(
                "model-local-code-sequences-v1".into(),
                key,
                CandidateTokens::Sequences(paths.clone()),
            );
        }
        Ok(paths)
    }

    pub(super) fn evaluate_code_sequences(
        &mut self,
        state: &serde_json::Value,
        decision: &Decision,
    ) -> Result<DecisionResult> {
        self.check_sequence_config()?;
        unsafe {
            if self.execution_mode == ExecutionMode::Fresh {
                sd_clear(self.engine.as_ptr());
            }
            sd_set_features(self.engine.as_ptr(), false);
        }
        let outcome = (|| {
            let started = Instant::now();
            let (input, paths) = self.encode_decision_sequences(state, decision)?;
            self.timings.prepare_ms += started.elapsed().as_secs_f64() * 1000.0;
            let vocab = unsafe { sd_vocab_size(self.engine.as_ptr()) };
            if vocab <= 0 {
                return Err(Error::Backend("invalid vocabulary size".into()));
            }
            let mut prefix_evaluations = 0usize;
            let mut evaluated_tokens = 0usize;
            let mut root_reused_tokens = 0usize;
            let scores = crate::codes::sequence_log_probabilities(&paths, |prefix| {
                let mut tokens = input.clone();
                tokens.extend_from_slice(prefix);
                let mut error = [0 as c_char; 1024];
                let mut reused = 0;
                self.logits_buffer.resize(vocab as usize, 0.0);
                let started = Instant::now();
                // Reuse only exact, complete prefill batches.
                // Each branch re-evaluates its suffix from the original prompt;
                // one candidate's continuation never leaks into another branch.
                let ok = unsafe {
                    sd_forward(
                        self.engine.as_ptr(),
                        tokens.as_ptr(),
                        tokens.len() as i32,
                        true,
                        &mut reused,
                        self.logits_buffer.as_mut_ptr(),
                        self.logits_buffer.len(),
                        error.as_mut_ptr(),
                        error.len(),
                    )
                };
                self.timings.native_ms += started.elapsed().as_secs_f64() * 1000.0;
                if !ok {
                    return Err(native_error(&error));
                }
                if prefix.is_empty() {
                    root_reused_tokens = reused as usize;
                }
                prefix_evaluations += 1;
                evaluated_tokens += tokens.len() - reused as usize;
                Ok(self.logits_buffer.clone())
            })?;
            let started = Instant::now();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_mass = max + scores.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            if log_mass > 1e-8 {
                return Err(Error::Backend(
                    "overlapping answer-code probability mass".into(),
                ));
            }
            let representatives: Vec<_> = paths
                .iter()
                .map(|p| if p.len() == 1 { p[0] } else { -1 })
                .collect();
            let mut result = crate::decision::score_candidate_logits(
                decision,
                &scores,
                &representatives,
                input.len(),
                log_mass.exp().min(1.0),
                &self.policy,
            )?;
            for (score, path) in result.scores.iter_mut().zip(paths) {
                score.token_ids = path;
            }
            result.scoring_method = "code_sequence_conditional_softmax_v1".into();
            result.code_prefix_evaluations = prefix_evaluations;
            result.code_evaluated_tokens = evaluated_tokens;
            result.reused_prefix_tokens = root_reused_tokens;
            self.restore_code_metadata(&mut result);
            self.timings.score_ms += started.elapsed().as_secs_f64() * 1000.0;
            self.timings.decisions += 1;
            Ok(result)
        })();
        if self.execution_mode == ExecutionMode::Fresh || outcome.is_err() {
            unsafe {
                sd_clear(self.engine.as_ptr());
            }
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(count: usize) -> DecisionRequest {
        DecisionRequest {
            state: serde_json::json!({"wanted": "intent_39"}),
            decisions: vec![Decision {
                id: "wide".into(),
                instruction: "Select the intent matching state.wanted.".into(),
                kind: DecisionKind::Choice {
                    options: (0..count)
                        .map(|i| OptionSpec {
                            id: format!("intent_{i}"),
                            criterion: format!("intent_{i}"),
                        })
                        .collect(),
                },
            }],
        }
    }

    #[test]
    #[ignore = "requires SKID_MODEL and optionally SKID_CUDA=1"]
    fn wide_codes_match_independent_fresh_teacher_forcing_and_preserve_legacy() {
        let model = std::env::var("SKID_MODEL").expect("SKID_MODEL");
        let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
        let mut backend = LlamaBackend::load(
            model.as_ref(),
            16384,
            256,
            4,
            cuda,
            DecisionPolicy::default(),
        )
        .unwrap();
        let legacy = request(3);
        let before = serde_json::to_value(backend.decide(&legacy).unwrap()).unwrap();
        for count in [27, 77] {
            let req = request(count);
            let (input, paths) = backend
                .encode_decision_sequences(&req.state, &req.decisions[0])
                .unwrap();
            let preflight = backend.preflight(&req).unwrap();
            assert_eq!(preflight.decisions[0].candidate_token_sequences, paths);
            let response = backend.decide(&req).unwrap();
            assert!(
                response
                    .backend
                    .prompt_version
                    .ends_with("/fixed-width-code-sequences-v1")
            );
            let result = &response.results[0];
            assert_eq!(result.scores.len(), count);
            assert_eq!(
                result.scoring_method,
                "code_sequence_conditional_softmax_v1"
            );
            let vocab = unsafe { sd_vocab_size(backend.engine.as_ptr()) } as usize;
            let mut reference = std::collections::BTreeMap::new();
            for path in &paths {
                for depth in 0..path.len() {
                    let prefix = path[..depth].to_vec();
                    if reference.contains_key(&prefix) {
                        continue;
                    }
                    let mut tokens = input.clone();
                    tokens.extend_from_slice(&prefix);
                    let mut logits = vec![0.0; vocab];
                    let mut reused = 0;
                    let mut error = [0 as c_char; 1024];
                    // No prefix reuse at all in this reference evaluation.
                    assert!(unsafe {
                        sd_forward(
                            backend.engine.as_ptr(),
                            tokens.as_ptr(),
                            tokens.len() as i32,
                            false,
                            &mut reused,
                            logits.as_mut_ptr(),
                            logits.len(),
                            error.as_mut_ptr(),
                            error.len(),
                        )
                    });
                    assert_eq!(reused, 0);
                    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                    let z = max
                        + logits
                            .iter()
                            .map(|&v| (v as f64 - max).exp())
                            .sum::<f64>()
                            .ln();
                    reference.insert(
                        prefix,
                        logits.into_iter().map(|v| v as f64 - z).collect::<Vec<_>>(),
                    );
                }
            }
            for (score, path) in result.scores.iter().zip(&paths) {
                let expected: f64 = (0..path.len())
                    .map(|depth| reference[&path[..depth]][path[depth] as usize])
                    .sum();
                assert!(
                    (score.raw_logit - expected).abs() < 0.002,
                    "joint likelihood mismatch: {}",
                    score.code
                );
                assert_eq!(&score.token_ids, path);
            }
            assert!(
                (result.candidate_mass
                    - result.scores.iter().map(|s| s.raw_logit.exp()).sum::<f64>())
                .abs()
                    < 1e-10
            );
            let repeated = backend.decide(&req).unwrap();
            for (a, b) in result.scores.iter().zip(&repeated.results[0].scores) {
                assert!((a.option_probability - b.option_probability).abs() < 1e-10);
            }
        }
        // AAA expansion is checked against actual tokenizer paths, not just strings.
        let three = request(677);
        let (_, paths) = backend
            .encode_decision_sequences(&three.state, &three.decisions[0])
            .unwrap();
        assert_eq!(paths.len(), 677);
        assert_eq!(option_code(676, 677).unwrap(), "BAA");
        assert_eq!(
            serde_json::to_value(backend.decide(&legacy).unwrap()).unwrap(),
            before
        );
        backend.set_execution_mode(ExecutionMode::Parallel);
        assert!(backend.decide(&request(77)).is_err());
        backend.set_execution_mode(ExecutionMode::Fresh);
        assert_eq!(
            serde_json::to_value(backend.decide(&legacy).unwrap()).unwrap(),
            before
        );
    }
}
