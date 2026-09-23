#![cfg(feature = "llama")]
use l2s1::{llama::LlamaBackend, *};

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

fn equivalent(a: &DecisionResult, b: &DecisionResult) {
    assert_eq!(
        serde_json::to_value(&a.value).unwrap(),
        serde_json::to_value(&b.value).unwrap()
    );
    assert!((a.candidate_mass - b.candidate_mass).abs() < 0.002);
    assert_eq!(a.scores.len(), b.scores.len());
    for (x, y) in a.scores.iter().zip(&b.scores) {
        assert_eq!(x.code, y.code);
        assert_eq!(x.token_ids, y.token_ids);
        assert!((x.raw_logit - y.raw_logit).abs() < 0.002);
        assert!((x.option_probability - y.option_probability).abs() < 0.002);
    }
}

#[test]
#[ignore = "requires SKID_MODEL and optionally SKID_CUDA=1"]
fn wide_preparation_and_session_caches_preserve_scores_and_isolation() {
    let model = std::env::var("SKID_MODEL").expect("SKID_MODEL");
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        16384,
        256,
        4,
        std::env::var("SKID_CUDA").as_deref() == Ok("1"),
        DecisionPolicy::default(),
    )
    .unwrap();
    for count in [3, 60, 77, 677] {
        backend.set_execution_mode(ExecutionMode::Fresh);
        backend.set_preparation_cache(PreparationCacheConfig {
            max_entries: 0,
            max_bytes: 0,
        });
        let req = request(count);
        let fresh = backend.decide(&req).unwrap();
        backend.set_preparation_cache(PreparationCacheConfig {
            max_entries: 8,
            max_bytes: 4 * 1024 * 1024,
        });
        let first = backend.decide(&req).unwrap();
        equivalent(&fresh.results[0], &first.results[0]);
        let before = backend.preparation_cache_stats();
        let cached = backend.decide(&req).unwrap();
        assert_eq!(
            backend.preparation_cache_stats().prompts.hits,
            before.prompts.hits + 1
        );
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&cached).unwrap()
        );
        let mut changed = req.clone();
        changed.state["wanted"] = "intent_1".into();
        backend.decide(&changed).unwrap();
        assert_eq!(
            backend.preparation_cache_stats().candidates.hits,
            before.candidates.hits + 1
        );
        // A cached candidate mapping must still enforce this prompt's context limit.
        changed.state["padding"] = " word".repeat(20000).into();
        assert!(backend.decide(&changed).is_err());
        assert_eq!(backend.preparation_cache_stats().prompts.entries, 2);
        backend.set_execution_mode(ExecutionMode::PrefixReuse);
        {
            let mut session = backend.shared_state(req.state.clone()).unwrap();
            let first = session.decide(req.decisions.clone()).unwrap();
            assert_eq!(first.results[0].reused_prefix_tokens, 0);
            equivalent(&fresh.results[0], &first.results[0]);
            let hot = session.decide(req.decisions.clone()).unwrap();
            equivalent(&fresh.results[0], &hot.results[0]);
            if first.results[0].input_tokens > 256 {
                assert!(hot.results[0].reused_prefix_tokens > 0);
                assert_eq!(hot.results[0].reused_prefix_tokens % 256, 0);
            }
            if count > 26 {
                assert!(
                    hot.backend
                        .prompt_version
                        .ends_with("/fixed-width-code-sequences-v1")
                );
            }
            assert!(session.preparation_cache_stats().prompts.hits > 0);
            assert!(session.take_timings().decisions > 0);
            let mut invalid = req.decisions.clone();
            invalid.push(invalid[0].clone());
            assert!(session.decide(invalid).is_err());
            let recovered = session.decide(req.decisions.clone()).unwrap();
            assert_eq!(recovered.results[0].reused_prefix_tokens, 0);
            equivalent(&fresh.results[0], &recovered.results[0]);
        }
        for _ in 0..2 {
            let isolated = backend.decide(&req).unwrap();
            assert_eq!(isolated.results[0].reused_prefix_tokens, 0);
            equivalent(&fresh.results[0], &isolated.results[0]);
        }
        if count > 26 {
            backend.set_execution_mode(ExecutionMode::Parallel);
            assert!(
                backend
                    .encode_decision_sequences(&req.state, &req.decisions[0])
                    .is_err()
            );
            backend.set_execution_mode(ExecutionMode::Fresh);
            backend
                .set_evidence_transfer(EvidenceTransfer::Compact)
                .unwrap();
            assert!(backend.decide(&req).is_err());
            backend
                .set_evidence_transfer(EvidenceTransfer::Full)
                .unwrap();
            backend
                .encode_decision_sequences(&req.state, &req.decisions[0])
                .unwrap();
            backend.set_code_rotation(1).unwrap();
            assert_eq!(backend.preparation_cache_stats().prompts.entries, 0);
            let (_, rotated) = backend
                .encode_decision_sequences(&req.state, &req.decisions[0])
                .unwrap();
            let rotated_cached = backend
                .encode_decision_sequences(&req.state, &req.decisions[0])
                .unwrap()
                .1;
            assert_eq!(rotated, rotated_cached);
            backend.set_code_rotation(0).unwrap();
            let (_, mut original) = backend
                .encode_decision_sequences(&req.state, &req.decisions[0])
                .unwrap();
            original.rotate_right(1);
            assert_eq!(rotated, original);
        }
        eprintln!(
            "Verified {count} candidates: preparation hits, session reuse, recovery and request isolation"
        );
    }
}

#[test]
#[ignore = "requires a decoder SKID_MODEL and optionally SKID_CUDA=1"]
fn changing_wide_schemas_in_one_session_preserves_independent_results() {
    let model = std::env::var("SKID_MODEL").expect("SKID_MODEL");
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        8192,
        256,
        4,
        std::env::var("SKID_CUDA").as_deref() == Ok("1"),
        DecisionPolicy::default(),
    )
    .unwrap();
    backend.set_preparation_cache(PreparationCacheConfig {
        max_entries: 4,
        max_bytes: 4 * 1024 * 1024,
    });
    let requests: Vec<_> = [27, 60, 77].into_iter().map(request).collect();
    let fresh: Vec<_> = requests
        .iter()
        .map(|r| backend.decide(r).unwrap())
        .collect();
    backend.set_execution_mode(ExecutionMode::PrefixReuse);
    let mut session = backend.shared_state(requests[0].state.clone()).unwrap();
    for index in [0, 1, 2, 0, 2, 1] {
        let response = session.decide(requests[index].decisions.clone()).unwrap();
        equivalent(&fresh[index].results[0], &response.results[0]);
        let repeated = session.decide(requests[index].decisions.clone()).unwrap();
        equivalent(&fresh[index].results[0], &repeated.results[0]);
        assert!(repeated.results[0].reused_prefix_tokens > 0);
        eprintln!(
            "{} candidates: {} prefix evaluations, {} reused prompt tokens",
            requests[index].decisions[0].options().len(),
            repeated.results[0].code_prefix_evaluations,
            repeated.results[0].reused_prefix_tokens
        );
    }
}
