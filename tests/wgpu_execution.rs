#![cfg(feature = "wgpu")]

use l2s1::{DecisionPolicy, DecisionRequest, ExecutionMode, PromptLayout, wgpu::WgpuBackend};
use std::path::Path;

fn assert_same(reference: &l2s1::DecisionResponse, actual: &l2s1::DecisionResponse) {
    assert_eq!(reference.results.len(), actual.results.len());
    for (a, b) in reference.results.iter().zip(&actual.results) {
        assert_eq!(
            serde_json::to_value(&a.value).unwrap(),
            serde_json::to_value(&b.value).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&a.abstention_reasons).unwrap(),
            serde_json::to_value(&b.abstention_reasons).unwrap()
        );
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
#[ignore = "requires L2S1_WGPU_GEMMA_MODEL, L2S1_WGPU_GEMMA_MMPROJ, and a wgpu adapter"]
fn gemma4_request_local_execution_matches_fresh() {
    let model = std::env::var("L2S1_WGPU_GEMMA_MODEL").expect("set L2S1_WGPU_GEMMA_MODEL");
    let mmproj = std::env::var("L2S1_WGPU_GEMMA_MMPROJ").expect("set L2S1_WGPU_GEMMA_MMPROJ");
    let mut backend = WgpuBackend::load_with_software_adapter(
        Path::new(&model),
        Path::new(&mmproj),
        DecisionPolicy::default(),
        true,
    )
    .unwrap();
    backend.set_prompt_layout(PromptLayout::StateFirst);
    let mut request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    request.decisions.truncate(2);
    let (fresh, _) = backend.decide_detailed(&request).unwrap();

    backend
        .set_execution_mode(ExecutionMode::PrefixReuse)
        .unwrap();
    let (prefix, prefix_report) = backend.decide_detailed(&request).unwrap();
    assert_eq!(prefix_report.effective_mode, ExecutionMode::PrefixReuse);
    assert!(prefix_report.reused_prefix_tokens[1] > 0);
    assert_same(&fresh, &prefix);

    backend
        .set_execution_mode(ExecutionMode::StateRestore)
        .unwrap();
    let (restored, report) = backend.decide_detailed(&request).unwrap();
    assert_eq!(report.effective_mode, ExecutionMode::StateRestore);
    assert_eq!(report.state_restore.restores, 1);
    assert!(report.state_restore.snapshot_bytes > 0);
    assert!(report.reused_prefix_tokens[1] > 0);
    assert_same(&fresh, &restored);

    backend.set_snapshot_limit_bytes(0);
    let (fallback, report) = backend.decide_detailed(&request).unwrap();
    assert_eq!(report.effective_mode, ExecutionMode::Fresh);
    assert_eq!(
        report.fallback_reason.as_deref(),
        Some("snapshot_memory_budget")
    );
    assert_same(&fresh, &fallback);
}
