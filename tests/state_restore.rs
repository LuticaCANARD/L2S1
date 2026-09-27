#![cfg(feature = "llama")]
use l2s1::{DecisionPolicy, DecisionRequest, ExecutionMode, PromptLayout, llama::LlamaBackend};

fn assert_same(a: &l2s1::DecisionResponse, b: &l2s1::DecisionResponse) {
    assert_eq!(a.results.len(), b.results.len());
    for (x, y) in a.results.iter().zip(&b.results) {
        assert_eq!(serde_json::json!(x.value), serde_json::json!(y.value));
        assert_eq!(
            serde_json::json!(x.abstention_reasons),
            serde_json::json!(y.abstention_reasons)
        );
        assert!((x.candidate_mass - y.candidate_mass).abs() < 0.02);
        let top = |r: &l2s1::DecisionResult| {
            r.scores
                .iter()
                .max_by(|a, b| a.option_probability.total_cmp(&b.option_probability))
                .unwrap()
                .id
                .clone()
        };
        assert_eq!(top(x), top(y));
        for (left, right) in x.scores.iter().zip(&y.scores) {
            assert_eq!(left.id, right.id);
            assert!((left.option_probability - right.option_probability).abs() < 0.02);
        }
    }
}

#[test]
#[ignore = "requires SKID_MODEL; real snapshot replacement, budget fallback and error isolation"]
fn state_restore_replaces_suffix_and_skips_disabled_prefill() {
    let model = std::env::var("SKID_MODEL").unwrap();
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        2048,
        32,
        4,
        false,
        DecisionPolicy::default(),
    )
    .unwrap();
    backend.set_prompt_layout(PromptLayout::StateFirst);
    let mut request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    request.state["history"] = serde_json::json!("packing ".repeat(80));
    let mut duplicate = request.decisions[0].clone();
    duplicate.id = "duplicate".into();
    request.decisions.push(duplicate);
    request.decisions[1]
        .instruction
        .push_str(&" Check all shipment fields.".repeat(8));

    backend.set_execution_mode(ExecutionMode::Fresh);
    let fresh = backend.decide_detailed(&request).unwrap();
    backend.set_execution_mode(ExecutionMode::StateRestore);
    let restored = backend.decide_detailed(&request).unwrap();
    assert_eq!(restored.state_restore.fallback_reason, None);
    assert_eq!(restored.state_restore.restores, request.decisions.len() - 1);
    assert!(restored.state_restore.snapshot_bytes > 0);
    assert!(restored.response.results[1].reused_prefix_tokens > 0);
    assert_same(&fresh.response, &restored.response);

    for budget in [0, 1] {
        backend.set_snapshot_limit_bytes(budget);
        let limited = backend.decide_detailed(&request).unwrap();
        assert_eq!(
            limited.state_restore.fallback_reason.as_deref(),
            Some("snapshot_memory_budget")
        );
        assert_eq!(limited.state_restore.snapshot_bytes, 0);
        assert_eq!(limited.state_restore.restores, 0);
        if budget == 0 {
            // Structural work avoidance, not a flaky elapsed-time threshold.
            assert_eq!(limited.state_restore.prefill_ms, 0.0);
            assert_eq!(limited.state_restore.save_ms, 0.0);
        }
        assert!(
            limited
                .response
                .results
                .iter()
                .all(|r| r.reused_prefix_tokens == 0)
        );
        assert_same(&fresh.response, &limited.response);
    }
    backend.set_snapshot_limit_bytes(256 * 1024 * 1024);
    let mut invalid = request.clone();
    invalid.decisions[1].instruction = "overlong ".repeat(4000);
    assert!(backend.decide_detailed(&invalid).is_err());
    request.state["storage_requirement"] = serde_json::json!("frozen");
    backend.set_execution_mode(ExecutionMode::Fresh);
    let changed = backend.decide_detailed(&request).unwrap();
    backend.set_execution_mode(ExecutionMode::StateRestore);
    let recovered = backend.decide_detailed(&request).unwrap();
    assert_same(&changed.response, &recovered.response);
    assert_eq!(recovered.response.results[0].reused_prefix_tokens, 0);
    request.decisions.truncate(1);
    let single = backend.decide_detailed(&request).unwrap();
    assert_eq!(
        single.state_restore.fallback_reason.as_deref(),
        Some("no_aligned_common_prefix")
    );
    assert_eq!(single.state_restore.snapshot_bytes, 0);
}
