#![cfg(feature = "llama")]
use l2s1::{llama::LlamaBackend, *};
fn values(result: &DecisionResult) -> serde_json::Value {
    match &result.value {
        DecisionValue::Binary { value, .. } => serde_json::json!(value),
        DecisionValue::Choice { selected } | DecisionValue::Ordinal { selected, .. } => {
            serde_json::json!(selected)
        }
    }
}
fn top(result: &DecisionResult) -> &str {
    &result
        .scores
        .iter()
        .max_by(|a, b| a.option_probability.total_cmp(&b.option_probability))
        .unwrap()
        .id
}
#[test]
#[ignore = "requires L2S1_CONFORMANCE_MODELS (colon-separated paths); optional SKID_CUDA=1"]
fn model_replacement_contract() {
    let paths = std::env::var("L2S1_CONFORMANCE_MODELS").expect("set L2S1_CONFORMANCE_MODELS");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let mut reports = Vec::new();
    for path in paths.split(':') {
        let mut backend =
            LlamaBackend::load(path.as_ref(), 2048, 32, 4, cuda, DecisionPolicy::default())
                .unwrap();
        backend.set_prompt_layout(PromptLayout::StateFirst);
        let mut request: DecisionRequest =
            serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
        request.state["history"] =
            serde_json::json!("The package was scanned at the depot. ".repeat(12));
        let inspection = backend.inspect();
        assert_eq!(inspection.identity.weights_sha256.len(), 64);
        let prepared = backend.preflight(&request).unwrap();
        assert_eq!(prepared.decisions.len(), 3);
        let fresh = backend.decide_detailed(&request).unwrap();
        assert_eq!(fresh.timings.decisions, 3);
        for ((result, decision), prep) in fresh
            .response
            .results
            .iter()
            .zip(&request.decisions)
            .zip(&prepared.decisions)
        {
            assert_eq!(result.id, decision.id);
            assert_eq!(prep.input_tokens, result.input_tokens);
            assert_eq!(
                result.scores.iter().map(|s| &s.id).collect::<Vec<_>>(),
                prep.option_ids.iter().collect::<Vec<_>>()
            );
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
            assert!((0.0..=1.0).contains(&result.candidate_mass));
            assert_eq!(result.calibration_id, None);
            match (&result.value, &decision.kind) {
                (DecisionValue::Binary { .. }, DecisionKind::Binary { .. })
                | (DecisionValue::Choice { .. }, DecisionKind::Choice { .. }) => {}
                (
                    DecisionValue::Ordinal { expected_value, .. },
                    DecisionKind::Ordinal { levels },
                ) => assert!(
                    (expected_value
                        - result
                            .scores
                            .iter()
                            .zip(levels)
                            .map(|(s, l)| s.option_probability * l.value)
                            .sum::<f64>())
                    .abs()
                        < 1e-12
                ),
                _ => panic!("changed result kind"),
            }
        }
        let mut modes = Vec::new();
        for mode in [
            ExecutionMode::PrefixReuse,
            ExecutionMode::StateRestore,
            ExecutionMode::Parallel,
        ] {
            backend.set_execution_mode(mode);
            if !inspection.capabilities.execution_modes.contains(&mode) {
                assert_eq!(
                    backend.preflight(&request).unwrap_err().kind,
                    FailureKind::UnsupportedCapability
                );
                assert!(backend.decide(&request).is_err());
                continue;
            }
            let actual = backend.decide_detailed(&request).unwrap();
            let (mut pd, mut md, mut top_changes, mut accepted_changes) = (0.0_f64, 0.0_f64, 0, 0);
            for (a, b) in fresh.response.results.iter().zip(&actual.response.results) {
                for (x, y) in a.scores.iter().zip(&b.scores) {
                    assert_eq!(x.token_id, y.token_id);
                    pd = pd.max((x.option_probability - y.option_probability).abs());
                }
                md = md.max((a.candidate_mass - b.candidate_mass).abs());
                top_changes += usize::from(top(a) != top(b));
                accepted_changes += usize::from(values(a) != values(b));
            }
            let equivalent = pd < 0.02 && md < 0.02 && top_changes == 0 && accepted_changes == 0;
            // Parallel already has documented batch-shape drift. Report failure at
            // the unchanged criterion; it is not part of the serial reuse guarantee.
            if mode != ExecutionMode::Parallel {
                assert!(
                    equivalent,
                    "{} {:?}: probability={pd}, mass={md}, top={top_changes}, accepted={accepted_changes}",
                    path, mode
                );
            }
            if mode == ExecutionMode::StateRestore && actual.state_restore.fallback_reason.is_none()
            {
                assert!(actual.state_restore.snapshot_bytes > 0);
                assert_eq!(actual.state_restore.restores, 3);
                assert!(actual.response.results[1].reused_prefix_tokens > 0);
            }
            modes.push(serde_json::json!({"mode":mode,"within_existing_equivalence_tolerance":equivalent,"max_probability_delta":pd,"max_mass_delta":md,"top_changes":top_changes,"accepted_changes":accepted_changes,"diagnostics":actual.decisions,"state_restore":actual.state_restore}));
        }
        // Bound snapshot memory, explicit fallback, and fresh recovery after oversized/invalid requests.
        backend.set_execution_mode(ExecutionMode::StateRestore);
        backend.set_snapshot_limit_bytes(0);
        let limited = backend.decide_detailed(&request).unwrap();
        assert_eq!(
            limited.state_restore.fallback_reason.as_deref(),
            Some("snapshot_memory_budget")
        );
        assert_eq!(limited.state_restore.snapshot_bytes, 0);
        assert!(
            limited
                .decisions
                .iter()
                .all(|d| d.effective_mode == ExecutionMode::Fresh)
        );
        backend.set_snapshot_limit_bytes(256 * 1024 * 1024);
        let mut bad = request.clone();
        bad.decisions[1].instruction = "overflow ".repeat(4096);
        assert_eq!(
            backend.decide_detailed(&bad).unwrap_err().kind,
            FailureKind::ContextExceeded
        );
        assert!(backend.decide(&bad).is_err());
        backend.set_execution_mode(ExecutionMode::Fresh);
        let after = backend.decide_detailed(&request).unwrap();
        assert_eq!(after.timings.decisions, 3);
        for (a, b) in fresh.response.results.iter().zip(&after.response.results) {
            assert_eq!(values(a), values(b));
            for (x, y) in a.scores.iter().zip(&b.scores) {
                assert!((x.raw_logit - y.raw_logit).abs() < 1e-3);
            }
        }
        // Calibration binding includes prompt, model and execution mode. No implicit invalidation.
        let task = &request.decisions[0];
        let record = CalibrationRecord {
            group: "calibration-source".into(),
            raw_logits: fresh.response.results[0]
                .scores
                .iter()
                .map(|s| s.raw_logit)
                .collect(),
            correct_option: 1,
        };
        let artifact = ScalarCalibration::fit(
            "fixture-calibration".into(),
            &backend.identity(),
            task,
            &[record],
        )
        .unwrap();
        backend.register_calibration(artifact).unwrap();
        backend.preflight(&request).unwrap();
        let calibrated = backend.decide(&request).unwrap();
        assert_eq!(
            calibrated.results[0].calibration_id.as_deref(),
            Some("fixture-calibration")
        );
        assert!(calibrated.results[1].calibration_id.is_none());
        assert_eq!(
            calibrated.results[0].candidate_mass,
            after.response.results[0].candidate_mass
        );
        backend.set_prompt_layout(PromptLayout::Legacy);
        assert_eq!(
            backend.preflight(&request).unwrap_err().kind,
            FailureKind::IncompatibleArtifact
        );
        assert!(backend.decide(&request).is_err());
        backend.clear_calibrations();
        backend.preflight(&request).unwrap();
        reports.push(serde_json::json!({"model":inspection,"modes":modes,"snapshot_budget_fallback":limited.state_restore,"scope":"synthetic real-model contract; not labeled workload quality or production evidence"}));
    }
    if let Ok(path) = std::env::var("L2S1_CONFORMANCE_REPORT") {
        std::fs::write(path, serde_json::to_vec_pretty(&reports).unwrap()).unwrap();
    }
    println!("{}", serde_json::to_string(&reports).unwrap());
}
