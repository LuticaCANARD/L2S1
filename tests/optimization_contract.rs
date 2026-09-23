use l2s1::*;

fn request() -> DecisionRequest {
    serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap()
}

#[test]
fn compact_identity_is_explicit_and_calibration_cannot_cross_transfer_modes() {
    let legacy = serde_json::json!({
        "weights_sha256": "a".repeat(64),
        "template_sha256": "b".repeat(64),
        "prompt_profile": "model",
        "prompt_version": "v1",
        "loaded_runtime_sha256": "d".repeat(64),
        "runtime_build_sha256": "c".repeat(64),
        "adapter_sha256": null,
        "head_sha256": null,
        "device": "CPU",
        "compute": {
            "context": 512,
            "batch": 32,
            "ubatch": 32,
            "threads": 1,
            "flash_attention": "off"
        },
        "execution_mode": "fresh",
        "parallel_width": 1
    });
    let full: ModelIdentity = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(full.evidence_transfer, EvidenceTransfer::Full);
    assert_eq!(serde_json::to_value(&full).unwrap(), legacy);
    let mut compact = full.clone();
    compact.evidence_transfer = EvidenceTransfer::Compact;
    assert_ne!(full.fingerprint(), compact.fingerprint());
    assert_eq!(
        serde_json::to_value(&compact).unwrap()["evidence_transfer"],
        "compact"
    );
    let decision = &request().decisions[0];
    let records: Vec<_> = (0..6)
        .map(|i| CalibrationRecord {
            group: format!("source-{i}"),
            raw_logits: vec![1.0, 0.0, -1.0],
            correct_option: i % 3,
        })
        .collect();
    for (identity, other) in [(&full, &compact), (&compact, &full)] {
        let calibration =
            ScalarCalibration::fit("transfer-bound".into(), identity, decision, &records).unwrap();
        calibration.validate(identity).unwrap();
        assert!(calibration.validate(other).is_err());
    }
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

    fn compare_evidence(reference: &DecisionResult, actual: &DecisionResult) {
        assert_eq!(reference.id, actual.id);
        assert_eq!(reference.input_tokens, actual.input_tokens);
        assert_eq!(reference.reused_prefix_tokens, actual.reused_prefix_tokens);
        assert_eq!(reference.scores.len(), actual.scores.len());
        assert!((reference.candidate_mass - actual.candidate_mass).abs() < 1e-10);
        assert!((reference.top_option_probability - actual.top_option_probability).abs() < 1e-10);
        assert!((reference.entropy_confidence - actual.entropy_confidence).abs() < 1e-10);
        for (a, b) in reference.scores.iter().zip(&actual.scores) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.token_id, b.token_id);
            assert_eq!(a.raw_logit, b.raw_logit);
            assert!((a.option_probability - b.option_probability).abs() < 1e-10);
        }
        let raw_top = |result: &DecisionResult| {
            result
                .scores
                .iter()
                .max_by(|a, b| a.raw_logit.total_cmp(&b.raw_logit))
                .unwrap()
                .id
                .clone()
        };
        assert_eq!(raw_top(reference), raw_top(actual));
        let accepted = |result: &DecisionResult| {
            let mut value = serde_json::to_value(&result.value).unwrap();
            value.as_object_mut().unwrap().remove("p_true");
            value.as_object_mut().unwrap().remove("expected_value");
            value
        };
        assert_eq!(accepted(reference), accepted(actual));
        assert_eq!(
            serde_json::to_value(&reference.abstention_reasons).unwrap(),
            serde_json::to_value(&actual.abstention_reasons).unwrap()
        );
    }

    /// Exercises public preparation and inference boundaries with a real model.
    /// Run separately for each checkpoint/device; this asserts equivalence and
    /// retained-cache bounds, not a speedup or a process-RSS limit.
    #[test]
    #[ignore = "requires SKID_MODEL and optionally SKID_CUDA=1"]
    fn preparation_cache_and_compact_preserve_dynamic_decision_contracts() {
        let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
        let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
        let base = request();
        let mut backend =
            LlamaBackend::load(model.as_ref(), 2048, 32, 4, cuda, DecisionPolicy::default())
                .unwrap();
        let reference_tokens = encoded(&backend, &base);
        let reference = backend.decide(&base).unwrap();
        let default_json = serde_json::to_value(&reference).unwrap();
        assert!(default_json["backend"].get("evidence_transfer").is_none());
        assert!(
            serde_json::to_value(backend.identity())
                .unwrap()
                .get("evidence_transfer")
                .is_none()
        );

        let mut variants = Vec::new();
        let mut state = base.clone();
        state.state["storage_requirement"] = serde_json::json!("ambient");
        variants.push(state);
        let mut instruction = base.clone();
        instruction.decisions[0].instruction =
            "Read the storage requirement and select the corresponding storage zone.".into();
        variants.push(instruction);
        let mut options = base.clone();
        if let DecisionKind::Choice { options } = &mut options.decisions[0].kind {
            options.swap(0, 1);
        }
        variants.push(options);
        let variant_references: Vec<_> = variants
            .iter()
            .map(|request| {
                let tokens = encoded(&backend, request);
                assert_ne!(tokens[0].0, reference_tokens[0].0);
                (
                    tokens,
                    serde_json::to_value(backend.decide(request).unwrap()).unwrap(),
                )
            })
            .collect();
        const BUDGET: usize = 128 * 1024;
        backend.set_preparation_cache(PreparationCacheConfig {
            max_entries: 8,
            max_bytes: BUDGET,
        });
        assert_eq!(encoded(&backend, &base), reference_tokens);
        assert_eq!(encoded(&backend, &base), reference_tokens);
        for _ in 0..2 {
            assert_eq!(
                serde_json::to_value(backend.decide(&base).unwrap()).unwrap(),
                default_json
            );
        }
        let hits = backend.preparation_cache_stats().prompts.hits;
        assert!(hits >= base.decisions.len() as u64 * 3);
        for (request, (tokens, response)) in variants.iter().zip(&variant_references) {
            assert_eq!(&encoded(&backend, request), tokens);
            assert_eq!(
                &serde_json::to_value(backend.decide(request).unwrap()).unwrap(),
                response
            );
        }
        for i in 0..24 {
            let mut altered = base.clone();
            altered.state["cache_pressure"] = serde_json::json!(i);
            encoded(&backend, &altered);
            let stats = backend.preparation_cache_stats();
            assert!(stats.prompts.entries <= 8);
            assert!(stats.candidates.entries <= 8);
            assert!(stats.prompts.retained_bytes + stats.candidates.retained_bytes <= BUDGET);
        }
        assert!(backend.preparation_cache_stats().prompts.evictions > 0);
        assert!(backend.preparation_cache_stats().candidates.hits > 0);
        backend.set_prompt_layout(PromptLayout::StateFirst);
        let invalidated = backend.preparation_cache_stats();
        assert_eq!(invalidated.prompts.entries, 0);
        assert_eq!(invalidated.candidates.entries, 0);
        assert_eq!(
            invalidated.prompts.retained_bytes + invalidated.candidates.retained_bytes,
            0
        );
        let state_first = encoded(&backend, &base);
        assert_ne!(state_first[0].0, reference_tokens[0].0);
        backend.set_preparation_cache(PreparationCacheConfig::default());
        assert_eq!(encoded(&backend, &base), state_first);
        backend.set_prompt_layout(PromptLayout::Legacy);
        assert_eq!(encoded(&backend, &base), reference_tokens);

        let full_identity = backend.identity();
        backend
            .set_evidence_transfer(EvidenceTransfer::Compact)
            .unwrap();
        assert_eq!(
            backend.inspect().capabilities.execution_modes,
            vec![ExecutionMode::Fresh, ExecutionMode::PrefixReuse]
        );
        assert_ne!(
            backend.identity().fingerprint(),
            full_identity.fingerprint()
        );
        let compact = backend.decide(&base).unwrap();
        assert_eq!(
            serde_json::to_value(&compact).unwrap()["backend"]["evidence_transfer"],
            "compact"
        );
        for (reference, actual) in reference.results.iter().zip(&compact.results) {
            compare_evidence(reference, actual);
        }
        // The unsupported combination is rejected before attempting to read a
        // nonexistent artifact, so this needs neither training nor a fake head.
        let head_error = backend
            .load_output_head(std::path::Path::new("/nonexistent/l2s1-head.json"))
            .unwrap_err();
        assert!(
            matches!(head_error, Error::Invalid(ref message) if message.contains("full evidence"))
        );
        for mode in [ExecutionMode::Parallel, ExecutionMode::StateRestore] {
            backend.set_execution_mode(mode);
            assert!(backend.preflight(&base).is_err());
            assert!(backend.decide(&base).is_err());
        }
        backend.set_execution_mode(ExecutionMode::Fresh);
        let mut oversized = base.clone();
        oversized.decisions[0].instruction = "word ".repeat(4096);
        assert!(backend.decide(&oversized).is_err());
        let recovered = backend.decide(&base).unwrap();
        for (reference, actual) in reference.results.iter().zip(&recovered.results) {
            compare_evidence(reference, actual);
        }
        backend
            .set_evidence_transfer(EvidenceTransfer::Full)
            .unwrap();
        assert_eq!(backend.identity(), full_identity);
        assert_eq!(
            serde_json::to_value(backend.decide(&base).unwrap()).unwrap(),
            default_json
        );
    }
}
