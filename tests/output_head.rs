use l2s1::*;
fn fixture() -> (Decision, OutputHead, DecisionPolicy) {
    let options = vec![
        OptionSpec {
            id: "no".into(),
            criterion: "Negative".into(),
        },
        OptionSpec {
            id: "yes".into(),
            criterion: "Positive".into(),
        },
    ];
    let d = Decision {
        id: "sentiment".into(),
        instruction: "Classify".into(),
        kind: DecisionKind::Choice {
            options: options.clone(),
        },
    };
    let h = OutputHead {
        version: 1,
        id: "test-head".into(),
        feature_kind: "hidden".into(),
        model_sha256: "a".repeat(64),
        device: "CPU".into(),
        compute: ComputeOptions {
            context: 2048,
            batch: 256,
            ubatch: 256,
            threads: 4,
            flash_attention: FlashAttention::Off,
        },
        prompt_version: "test".into(),
        decision_id: d.id.clone(),
        instruction: d.instruction.clone(),
        options,
        weights: vec![vec![1., 0.], vec![0., 1.]],
        bias: vec![0., 0.],
        temperature: 2.,
    };
    (d, h, DecisionPolicy::default())
}
#[test]
fn semantic_permutations_and_calibration_preserve_base_mass() {
    let (mut d, h, p) = fixture();
    h.validate(2).unwrap();
    if let DecisionKind::Choice { options } = &mut d.kind {
        options.reverse();
    }
    let base = score_logits(&d, &[0., 0., 0.], &[0, 1], 10, &p).unwrap();
    let mass = base.candidate_mass;
    let r = h.apply(&d, &[0., 4.], base, &p).unwrap();
    assert_eq!(r.scores[0].id, "yes");
    assert_eq!(r.scores[0].raw_logit, 4.);
    assert!((r.scores[0].option_probability - 1. / (1. + (-2.0_f64).exp())).abs() < 1e-12);
    assert_eq!(r.candidate_mass, mass);
    assert_eq!(r.calibration_id.as_deref(), Some("test-head"));
    let base = score_logits(&d, &[-10., -10., 10.], &[0, 1], 10, &p).unwrap();
    let r = h.apply(&d, &[0., 40.], base, &p).unwrap();
    assert!(
        r.abstention_reasons
            .iter()
            .any(|r| matches!(r, AbstentionReason::LowCandidateMass))
    );
}
#[test]
fn incompatible_signatures_fail_and_unrelated_tasks_fall_back() {
    let (mut d, h, p) = fixture();
    d.instruction.push('!');
    assert!(h.applies_to(&d).is_err());
    d.id = "other".into();
    let base = score_logits(&d, &[0., 1.], &[0, 1], 10, &p).unwrap();
    let old = serde_json::to_value(&base).unwrap();
    assert_eq!(
        old,
        serde_json::to_value(h.apply(&d, &[], base, &p).unwrap()).unwrap()
    );
}
#[test]
fn malformed_heads_and_features_are_rejected() {
    let (d, mut h, p) = fixture();
    h.temperature = 0.;
    assert!(h.validate(2).is_err());
    h.temperature = 1.;
    h.weights[0].pop();
    assert!(h.validate(2).is_err());
    let base = score_logits(&d, &[0., 1.], &[0, 1], 10, &p).unwrap();
    assert!(h.apply(&d, &[0., f32::NAN], base, &p).is_err());
}
#[test]
fn affine_features_follow_semantic_ids() {
    let (mut d, mut h, p) = fixture();
    h.feature_kind = "logit_affine".into();
    h.temperature = 1.;
    if let DecisionKind::Choice { options } = &mut d.kind {
        options.reverse();
    }
    let base = score_logits(&d, &[3., 1.], &[0, 1], 10, &p).unwrap();
    let r = h.apply(&d, &[], base, &p).unwrap();
    assert_eq!(r.scores[0].raw_logit, 3.);
    assert_eq!(r.scores[1].raw_logit, 1.);
}
