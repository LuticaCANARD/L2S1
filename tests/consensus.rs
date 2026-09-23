use l2s1::{
    Decision, DecisionKind, DecisionPolicy, DecisionResult, DecisionValue, Level, OptionSpec,
    score_logits, score_semantic_mixture,
};

fn choice() -> Decision {
    Decision {
        id: "task".into(),
        instruction: "Choose the matching alternative.".into(),
        kind: DecisionKind::Choice {
            options: ["left", "right"]
                .into_iter()
                .map(|id| OptionSpec {
                    id: id.into(),
                    criterion: format!("{id} criterion"),
                })
                .collect(),
        },
    }
}

fn policy() -> DecisionPolicy {
    DecisionPolicy {
        min_top_probability: 0.0,
        min_candidate_mass: 0.0,
    }
}

fn pass(decision: &Decision, probabilities: &[f64], mass: f64, rotation: usize) -> DecisionResult {
    let tokens: Vec<_> = (0..probabilities.len()).map(|i| i as i32).collect();
    let mut result = score_logits(
        decision,
        &vec![0.0; probabilities.len()],
        &tokens,
        10,
        &policy(),
    )
    .unwrap();
    result.candidate_mass = mass;
    result.reused_prefix_tokens = 2;
    for (index, score) in result.scores.iter_mut().enumerate() {
        let code_position =
            (index + probabilities.len() - rotation % probabilities.len()) % probabilities.len();
        score.code = ((b'A' + code_position as u8) as char).to_string();
        score.token_id = code_position as i32;
        score.option_probability = probabilities[index];
        score.raw_logit = probabilities[index].ln();
    }
    result
}

fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-12,
        "actual={actual}, expected={expected}"
    );
}

#[test]
fn semantic_alignment_cancels_code_bias_and_preserves_first_pass_metadata() {
    let decision = choice();
    let first = pass(&decision, &[0.1, 0.9], 0.4, 1);
    let mut second = pass(&decision, &[0.9, 0.1], 0.4, 0);
    second.scores.reverse(); // Storage order is not semantic or display order.
    let result = score_semantic_mixture(&decision, &[first, second], &policy()).unwrap();
    near(result.scores[0].option_probability, 0.5);
    near(result.scores[1].option_probability, 0.5);
    near(result.candidate_mass, 0.4);
    assert_eq!(result.scores[0].id, "left");
    assert_eq!(result.scores[0].code, "B");
    assert_eq!(result.scores[0].token_id, 1);
    assert_eq!(result.scoring_method, "semantic_probability_mixture_v1");
    assert!(result.calibration_id.is_none());
    assert_eq!(result.input_tokens, 20);
    assert_eq!(result.reused_prefix_tokens, 4);
    assert!(matches!(
        result.value,
        DecisionValue::Choice { selected: None }
    ));
}

#[test]
fn full_probability_mixture_weights_each_pass_by_its_candidate_mass() {
    let decision = choice();
    let result = score_semantic_mixture(
        &decision,
        &[
            pass(&decision, &[0.9, 0.1], 0.2, 0),
            pass(&decision, &[0.1, 0.9], 0.8, 1),
        ],
        &policy(),
    )
    .unwrap();
    near(result.candidate_mass, 0.5);
    near(result.scores[0].option_probability, 0.26);
    near(result.scores[1].option_probability, 0.74);
    near(result.scores[0].raw_logit, 0.13_f64.ln());
    near(result.scores[1].raw_logit, 0.37_f64.ln());
    assert!(
        matches!(result.value, DecisionValue::Choice { selected: Some(ref id) } if id == "right")
    );
}

#[test]
fn original_ordinal_scale_and_binary_semantics_are_retained() {
    let decision = Decision {
        id: "ordinal".into(),
        instruction: "Select a level.".into(),
        kind: DecisionKind::Ordinal {
            levels: [-2.0, 1.0, 10.0]
                .into_iter()
                .enumerate()
                .map(|(i, value)| Level {
                    id: format!("level_{i}"),
                    criterion: format!("criterion_{i}"),
                    value,
                })
                .collect(),
        },
    };
    let result = score_semantic_mixture(
        &decision,
        &[
            pass(&decision, &[0.2, 0.3, 0.5], 0.6, 1),
            pass(&decision, &[0.2, 0.3, 0.5], 0.6, 2),
        ],
        &policy(),
    )
    .unwrap();
    assert_eq!(
        result
            .scores
            .iter()
            .map(|s| s.id.as_str())
            .collect::<Vec<_>>(),
        ["level_0", "level_1", "level_2"]
    );
    let DecisionValue::Ordinal {
        expected_value,
        selected,
    } = result.value
    else {
        panic!("wrong result type")
    };
    near(expected_value, 4.9);
    assert_eq!(selected.as_deref(), Some("level_2"));
    let decision = Decision {
        id: "binary".into(),
        instruction: "Evaluate.".into(),
        kind: DecisionKind::Binary {
            false_label: "False".into(),
            true_label: "True".into(),
        },
    };
    let result = score_semantic_mixture(
        &decision,
        &[
            pass(&decision, &[0.2, 0.8], 0.5, 1),
            pass(&decision, &[0.2, 0.8], 0.5, 0),
        ],
        &policy(),
    )
    .unwrap();
    let DecisionValue::Binary { p_true, value } = result.value else {
        panic!("wrong result type")
    };
    near(p_true, 0.8);
    assert_eq!(value, Some(true));
}

#[test]
fn rejects_non_native_invalid_or_mismatched_evidence() {
    let decision = choice();
    type Alteration = Box<dyn Fn(&mut DecisionResult)>;
    let alterations: Vec<Alteration> = vec![
        Box::new(|p| p.scoring_method = "semantic_probability_mixture_v1".into()),
        Box::new(|p| p.scoring_method = "learned_head".into()),
        Box::new(|p| p.calibration_id = Some("temperature".into())),
        Box::new(|p| p.truncated = true),
        Box::new(|p| p.id = "another-task".into()),
        Box::new(|p| p.candidate_mass = 0.0),
        Box::new(|p| p.candidate_mass = 1.01),
        Box::new(|p| p.candidate_mass = f64::NAN),
        Box::new(|p| p.scores[0].id = "right".into()),
        Box::new(|p| p.scores[0].id = "unknown".into()),
        Box::new(|p| {
            p.scores.pop();
        }),
        Box::new(|p| p.scores[0].token_id = -1),
        Box::new(|p| p.scores[0].token_id = p.scores[1].token_id),
        Box::new(|p| p.scores[0].token_id = 999),
        Box::new(|p| p.scores[0].code = "AA".into()),
        Box::new(|p| p.scores[0].code = "Z".into()),
        Box::new(|p| p.scores[0].code = p.scores[1].code.clone()),
        Box::new(|p| p.scores[0].raw_logit = f64::INFINITY),
        Box::new(|p| p.scores[0].option_probability = 0.0),
        Box::new(|p| p.scores[0].option_probability = f64::NAN),
        Box::new(|p| p.scores[0].option_probability = 0.7),
        Box::new(|p| p.reused_prefix_tokens = p.input_tokens + 1),
    ];
    for (index, alter) in alterations.into_iter().enumerate() {
        let first = pass(&decision, &[0.2, 0.8], 0.5, 0);
        let mut second = pass(&decision, &[0.2, 0.8], 0.5, 1);
        alter(&mut second);
        assert!(
            score_semantic_mixture(&decision, &[first, second], &policy()).is_err(),
            "alteration {index} accepted"
        );
    }
    assert!(score_semantic_mixture(&decision, &[], &policy()).is_err());
    assert!(
        score_semantic_mixture(
            &decision,
            &[pass(&decision, &[0.2, 0.8], 0.5, 0)],
            &policy()
        )
        .is_err()
    );
}

#[test]
fn token_accounting_is_checked_and_tiny_full_probabilities_do_not_underflow() {
    let decision = choice();
    let mut first = pass(&decision, &[0.2, 0.8], 0.5, 0);
    first.input_tokens = usize::MAX;
    assert!(
        score_semantic_mixture(
            &decision,
            &[first, pass(&decision, &[0.2, 0.8], 0.5, 1)],
            &policy()
        )
        .is_err()
    );
    let tiny = f64::MIN_POSITIVE;
    let result = score_semantic_mixture(
        &decision,
        &[
            pass(&decision, &[tiny, 1.0], tiny, 0),
            pass(&decision, &[tiny, 1.0], tiny, 1),
        ],
        &policy(),
    )
    .unwrap();
    assert!(result.scores[0].raw_logit.is_finite());
    assert!(result.scores[0].option_probability > 0.0);
    assert_eq!(result.candidate_mass, tiny);
}
