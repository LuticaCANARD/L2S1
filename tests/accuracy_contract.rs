use l2s1::*;

#[test]
fn historical_backend_json_defaults_to_minimal_unrotated_prompts() {
    let legacy = serde_json::json!({
        "model_path": "local.gguf", "model_description": "fixture",
        "model_architecture": "llama", "prompt_profile": "model",
        "prompt_layout": "legacy", "prompt_version": "gguf-jinja-decision-v1",
        "runtime": "local-libllama", "execution_mode": "fresh", "parallel_width": 1,
        "offload_requested": false, "offload_device": null
    });
    let mut info: BackendInfo = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(info.prompt_detail, PromptDetail::Minimal);
    assert_eq!(info.code_rotation, 0);
    assert_eq!(serde_json::to_value(&info).unwrap(), legacy);
    info.prompt_detail = PromptDetail::TypedExamples;
    info.code_rotation = 2;
    let changed = serde_json::to_value(&info).unwrap();
    assert_eq!(changed["prompt_detail"], "typed_examples");
    assert_eq!(changed["code_rotation"], 2);
}

#[cfg(feature = "llama")]
mod native {
    use super::*;
    use l2s1::llama::LlamaBackend;

    fn encoded(backend: &LlamaBackend, request: &DecisionRequest) -> Vec<(Vec<i32>, Vec<i32>)> {
        request
            .decisions
            .iter()
            .map(|decision| backend.encode_decision(&request.state, decision).unwrap())
            .collect()
    }

    fn assert_semantics(
        response: &DecisionResponse,
        request: &DecisionRequest,
        prepared: &[(Vec<i32>, Vec<i32>)],
        rotation: usize,
    ) {
        assert_eq!(response.results.len(), request.decisions.len());
        for ((result, decision), (input, candidates)) in response
            .results
            .iter()
            .zip(&request.decisions)
            .zip(prepared)
        {
            assert_eq!(result.id, decision.id);
            assert_eq!(result.input_tokens, input.len());
            assert!(!result.truncated);
            assert!(result.candidate_mass.is_finite());
            assert!((0.0..=1.0).contains(&result.candidate_mass));
            let options = decision.options();
            assert_eq!(result.scores.len(), options.len());
            for (index, (score, option)) in result.scores.iter().zip(&options).enumerate() {
                assert_eq!(score.id, option.id);
                assert_eq!(score.token_id, candidates[index]);
                let position = (index + options.len() - rotation % options.len()) % options.len();
                assert_eq!(score.code, ((b'A' + position as u8) as char).to_string());
                assert!(score.raw_logit.is_finite());
                assert!(score.option_probability.is_finite());
            }
            assert!(
                (result
                    .scores
                    .iter()
                    .map(|s| s.option_probability)
                    .sum::<f64>()
                    - 1.0)
                    .abs()
                    < 1e-10
            );
            match (&result.value, &decision.kind) {
                (DecisionValue::Binary { p_true, .. }, DecisionKind::Binary { .. }) => {
                    assert_eq!(*p_true, result.scores[1].option_probability);
                }
                (DecisionValue::Choice { selected }, DecisionKind::Choice { .. }) => {
                    assert!(
                        selected
                            .as_ref()
                            .is_none_or(|id| options.iter().any(|o| &o.id == id))
                    );
                }
                (
                    DecisionValue::Ordinal {
                        expected_value,
                        selected,
                    },
                    DecisionKind::Ordinal { levels },
                ) => {
                    let expected: f64 = result
                        .scores
                        .iter()
                        .zip(levels)
                        .map(|(score, level)| score.option_probability * level.value)
                        .sum();
                    assert!((expected - expected_value).abs() < 1e-12);
                    assert!(
                        selected
                            .as_ref()
                            .is_none_or(|id| levels.iter().any(|level| &level.id == id))
                    );
                }
                _ => panic!("result kind changed"),
            }
        }
    }

    /// Real checkpoint contract checks; no accuracy labels, downloads or reports.
    #[test]
    #[ignore = "requires SKID_MODEL and optionally SKID_CUDA=1"]
    fn typed_prompts_rotations_calibration_and_mixtures_keep_semantic_contracts() {
        let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
        let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
        let request: DecisionRequest =
            serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
        let original_request = serde_json::to_value(&request).unwrap();
        let mut backend =
            LlamaBackend::load(model.as_ref(), 2048, 32, 4, cuda, DecisionPolicy::default())
                .unwrap();
        let default_identity = backend.identity();
        let default_tokens = encoded(&backend, &request);
        let default_response = backend.decide(&request).unwrap();
        let default_json = serde_json::to_value(&default_response).unwrap();
        assert!(default_json["backend"].get("prompt_detail").is_none());
        assert!(default_json["backend"].get("code_rotation").is_none());
        backend.set_prompt_detail(PromptDetail::Minimal);
        backend.set_code_rotation(0).unwrap();
        assert_eq!(encoded(&backend, &request), default_tokens);
        assert_eq!(backend.identity(), default_identity);
        assert_eq!(
            serde_json::to_value(backend.decide(&request).unwrap()).unwrap(),
            default_json
        );

        backend.set_preparation_cache(PreparationCacheConfig {
            max_entries: 32,
            max_bytes: 1024 * 1024,
        });
        let mut identities = std::collections::HashSet::new();
        for detail in [
            PromptDetail::Minimal,
            PromptDetail::Typed,
            PromptDetail::TypedExamples,
        ] {
            backend.set_prompt_detail(detail);
            assert_eq!(backend.preparation_cache_stats().prompts.entries, 0);
            assert_eq!(backend.preparation_cache_stats().candidates.entries, 0);
            for rotation in [0, 1, 2] {
                backend.set_code_rotation(rotation).unwrap();
                assert_eq!(backend.preparation_cache_stats().prompts.entries, 0);
                assert_eq!(backend.preparation_cache_stats().candidates.entries, 0);
                let prepared = encoded(&backend, &request);
                assert_eq!(encoded(&backend, &request), prepared);
                assert!(backend.preparation_cache_stats().prompts.hits > 0);
                assert!(backend.preparation_cache_stats().candidates.entries > 0);
                if detail != PromptDetail::Minimal {
                    for (typed, original) in prepared.iter().zip(&default_tokens) {
                        assert_ne!(typed.0, original.0);
                    }
                }
                for ((_, candidates), (_, original)) in prepared.iter().zip(&default_tokens) {
                    for (index, token) in candidates.iter().enumerate() {
                        let code_position = (index + candidates.len()
                            - rotation % candidates.len())
                            % candidates.len();
                        assert_eq!(*token, original[code_position]);
                    }
                }
                let identity = backend.inspect().identity;
                assert!(identities.insert(identity.fingerprint()));
                if detail != PromptDetail::Minimal || rotation != 0 {
                    assert_ne!(identity.prompt_version, default_identity.prompt_version);
                }
                let response = backend.decide(&request).unwrap();
                assert_eq!(response.backend.prompt_detail, detail);
                assert_eq!(response.backend.code_rotation, rotation);
                assert_semantics(&response, &request, &prepared, rotation);
            }
        }
        let before_bad_rotation = backend.identity();
        assert!(backend.set_code_rotation(26).is_err());
        assert_eq!(backend.identity(), before_bad_rotation);

        backend.set_prompt_detail(PromptDetail::Minimal);
        backend.set_code_rotation(0).unwrap();
        assert_eq!(encoded(&backend, &request), default_tokens);
        assert_eq!(
            serde_json::to_value(backend.decide(&request).unwrap()).unwrap(),
            default_json
        );
        let calibration = ScalarCalibration::fit(
            "accuracy-contract-fixture".into(),
            &backend.identity(),
            &request.decisions[0],
            &[CalibrationRecord {
                group: "fixture-only-not-quality-evaluation".into(),
                raw_logits: default_response.results[0]
                    .scores
                    .iter()
                    .map(|score| score.raw_logit)
                    .collect(),
                correct_option: 1,
            }],
        )
        .unwrap();
        backend.register_calibration(calibration).unwrap();
        for detail in [PromptDetail::Typed, PromptDetail::TypedExamples] {
            backend.set_prompt_detail(detail);
            assert_eq!(
                backend.preflight(&request).unwrap_err().kind,
                FailureKind::IncompatibleArtifact
            );
            assert!(backend.decide(&request).is_err());
            backend.set_prompt_detail(PromptDetail::Minimal);
            assert_eq!(
                backend.decide(&request).unwrap().results[0]
                    .calibration_id
                    .as_deref(),
                Some("accuracy-contract-fixture")
            );
        }
        backend.set_code_rotation(1).unwrap();
        assert!(backend.decide(&request).is_err());
        backend.set_code_rotation(0).unwrap();
        assert!(
            backend.decide(&request).unwrap().results[0]
                .calibration_id
                .is_some()
        );
        backend.clear_calibrations();

        // Score actual native rotated observations, retaining full-vocabulary mass.
        backend.set_prompt_detail(PromptDetail::Typed);
        let mut passes = Vec::new();
        for rotation in [0, 1, 2] {
            backend.set_code_rotation(rotation).unwrap();
            passes.push(backend.decide(&request).unwrap().results.remove(0));
        }
        let mixture =
            score_semantic_mixture(&request.decisions[0], &passes, &DecisionPolicy::default())
                .unwrap();
        assert_eq!(mixture.scoring_method, "semantic_probability_mixture_v1");
        assert!(mixture.calibration_id.is_none());
        let expected_mass =
            passes.iter().map(|p| p.candidate_mass).sum::<f64>() / passes.len() as f64;
        assert_eq!(mixture.candidate_mass, expected_mass);

        backend.set_code_rotation(1).unwrap();
        backend.set_prompt_layout(PromptLayout::StateFirst);
        // Three independent question kinds force two parallel waves at width
        // two, and provide an actual common prefix for snapshot save/restore.
        backend.set_parallel_width(2).unwrap();
        for mode in [
            ExecutionMode::Fresh,
            ExecutionMode::PrefixReuse,
            ExecutionMode::StateRestore,
            ExecutionMode::Parallel,
        ] {
            if !backend
                .inspect()
                .capabilities
                .execution_modes
                .contains(&mode)
            {
                continue;
            }
            backend.set_execution_mode(mode);
            let prepared = encoded(&backend, &request);
            let detailed = backend.decide_detailed(&request).unwrap();
            assert_eq!(detailed.response.backend.execution_mode, mode);
            assert_semantics(&detailed.response, &request, &prepared, 1);
            if mode == ExecutionMode::Parallel {
                assert_eq!(detailed.response.backend.parallel_width, 2);
            }
            if mode == ExecutionMode::StateRestore {
                // Unsupported snapshot APIs may fall back, but this workload
                // must reach snapshot handling rather than the single-question
                // or no-common-prefix fast fallback.
                assert_ne!(
                    detailed.state_restore.fallback_reason.as_deref(),
                    Some("no_aligned_common_prefix")
                );
                if detailed.state_restore.fallback_reason.is_none() {
                    assert_eq!(detailed.state_restore.restores, request.decisions.len());
                    assert!(detailed.state_restore.snapshot_bytes > 0);
                    assert!(detailed.response.results[1].reused_prefix_tokens > 0);
                }
            }
        }
        backend.set_execution_mode(ExecutionMode::Fresh);
        let mut bad = request.clone();
        bad.decisions[1].id = bad.decisions[0].id.clone();
        assert!(backend.decide(&bad).is_err());
        let prepared = encoded(&backend, &request);
        assert_semantics(&backend.decide(&request).unwrap(), &request, &prepared, 1);
        assert_eq!(serde_json::to_value(&request).unwrap(), original_request);
    }
}
