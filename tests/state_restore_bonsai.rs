#![cfg(feature = "llama")]

use l2s1::{
    ComputeOptions, DecisionPolicy, DecisionRequest, DecisionResponse, ExecutionMode, PromptLayout,
    PromptProfile, llama::LlamaBackend,
};

fn assert_same_decisions(reference: &DecisionResponse, actual: &DecisionResponse) {
    assert_eq!(reference.results.len(), actual.results.len());
    for (a, b) in reference.results.iter().zip(&actual.results) {
        assert_eq!(a.id, b.id);
        assert_eq!(
            serde_json::to_value(&a.value).unwrap(),
            serde_json::to_value(&b.value).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&a.abstention_reasons).unwrap(),
            serde_json::to_value(&b.abstention_reasons).unwrap()
        );
        assert_eq!(a.scores.len(), b.scores.len());
        assert!((a.candidate_mass - b.candidate_mass).abs() < 1e-6);
        for (x, y) in a.scores.iter().zip(&b.scores) {
            assert_eq!(x.id, y.id);
            assert_eq!(x.token_id, y.token_id);
            assert!((x.raw_logit - y.raw_logit).abs() < 1e-3);
            assert!((x.option_probability - y.option_probability).abs() < 1e-6);
        }
    }
}

#[test]
#[ignore = "requires L2S1_BONSAI_MODEL and the matching native CUDA build on RTX 3060"]
fn hybrid_state_restore_reuses_prefix_and_recovers_after_budget_fallback() {
    let model = std::env::var("L2S1_BONSAI_MODEL").expect("set L2S1_BONSAI_MODEL");
    let options: ComputeOptions = serde_json::from_value(serde_json::json!({
        "context":4096,"batch":256,"ubatch":256,"threads":8,"flash_attention":"off"
    }))
    .unwrap();
    let mut backend = LlamaBackend::load_with_options(
        model.as_ref(),
        options,
        true,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )
    .unwrap();
    assert!(backend.inspect().capabilities.recurrent_or_hybrid);
    backend.set_prompt_layout(PromptLayout::StateFirst);
    let request: DecisionRequest = serde_json::from_str(include_str!(
        "../benchmarks/jv-gist-rtx3060-20260925/request-16.json"
    ))
    .unwrap();

    let baseline = backend.decide_detailed(&request).unwrap();
    backend.set_execution_mode(ExecutionMode::StateRestore);
    let restored = backend.decide_detailed(&request).unwrap();
    assert_eq!(restored.state_restore.fallback_reason, None);
    assert!(restored.state_restore.snapshot_bytes > 0);
    assert_eq!(restored.state_restore.restores, 15);
    assert_eq!(restored.response.results[0].reused_prefix_tokens, 0);
    assert!(
        restored.response.results[1..]
            .iter()
            .all(|result| result.reused_prefix_tokens > 0)
    );
    assert_same_decisions(&baseline.response, &restored.response);

    backend.set_snapshot_limit_bytes(0);
    let fallback = backend.decide_detailed(&request).unwrap();
    assert_eq!(
        fallback.state_restore.fallback_reason.as_deref(),
        Some("snapshot_memory_budget")
    );
    assert_eq!(fallback.state_restore.restores, 0);
    assert!(
        fallback
            .response
            .results
            .iter()
            .all(|result| result.reused_prefix_tokens == 0)
    );
    assert_same_decisions(&baseline.response, &fallback.response);

    backend.set_snapshot_limit_bytes(256 * 1024 * 1024);
    let recovered = backend.decide_detailed(&request).unwrap();
    assert_eq!(recovered.state_restore.fallback_reason, None);
    assert_eq!(recovered.state_restore.restores, 15);
    assert_same_decisions(&baseline.response, &recovered.response);
}
