use l2s1::*;

fn choice(count: usize) -> Decision {
    Decision {
        id: "intent".into(),
        instruction: "Choose the matching intent.".into(),
        kind: DecisionKind::Choice {
            options: (0..count)
                .map(|i| OptionSpec {
                    id: format!("intent_{i}"),
                    criterion: format!("meaning {i}"),
                })
                .collect(),
        },
    }
}

#[test]
fn codes_expand_at_every_base26_boundary_without_overflow() {
    for (count, first, last) in [
        (26, "A", "Z"),
        (27, "AA", "BA"),
        (60, "AA", "CH"),
        (77, "AA", "CY"),
        (676, "AA", "ZZ"),
        (677, "AAA", "BAA"),
        (17576, "AAA", "ZZZ"),
        (17577, "AAAA", "BAAA"),
    ] {
        assert_eq!(option_code(0, count).unwrap(), first);
        assert_eq!(option_code(count - 1, count).unwrap(), last);
        let codes: std::collections::HashSet<_> =
            (0..count).map(|i| option_code(i, count).unwrap()).collect();
        assert_eq!(codes.len(), count);
        assert!(
            codes
                .iter()
                .all(|c| c.len() == first.len() && c.bytes().all(|b| b.is_ascii_uppercase()))
        );
    }
    assert!(option_code(0, 1).is_err());
    assert!(option_code(27, 27).is_err());
    assert_eq!(
        option_code(usize::MAX - 1, usize::MAX).unwrap().len(),
        option_code_width(usize::MAX)
    );
}

#[test]
fn wide_prompt_and_probabilities_keep_every_candidate_in_semantic_order() {
    for count in [27, 60, 77, 677] {
        let d = choice(count);
        DecisionRequest {
            state: serde_json::json!({}),
            decisions: vec![d.clone()],
        }
        .validate()
        .unwrap();
        for rotation in [0, 26, count - 1, usize::MAX] {
            let parts = compile_prompt_with_detail(
                &serde_json::json!({}),
                &d,
                PromptLayout::Legacy,
                PromptDetail::Minimal,
                rotation,
            );
            assert!(parts[0].text.contains(&format!(
                "exactly {} uppercase letters",
                option_code_width(count)
            )));
            assert!(!parts[1].parse_special);
            let payload: serde_json::Value = serde_json::from_str(&parts[1].text).unwrap();
            assert_eq!(payload["options"].as_array().unwrap().len(), count);
            for i in 0..count {
                assert_eq!(
                    payload["options"][i]["code"],
                    option_code(i, count).unwrap()
                );
                assert_eq!(
                    payload["options"][i]["criterion"],
                    format!("meaning {}", (i + rotation % count) % count)
                );
            }
        }
        let mut logits = vec![-10.0; count + 1];
        logits[count - 1] = 10.0;
        let result = score_logits(
            &d,
            &logits,
            &(0..count as i32).collect::<Vec<_>>(),
            10,
            &DecisionPolicy::default(),
        )
        .unwrap();
        assert!(
            matches!(result.value, DecisionValue::Choice { selected: Some(ref id) } if id==&format!("intent_{}",count-1))
        );
        assert_eq!(
            result.scores[count - 1].code,
            option_code(count - 1, count).unwrap()
        );
        assert!(
            (result
                .scores
                .iter()
                .map(|s| s.option_probability)
                .sum::<f64>()
                - 1.0)
                .abs()
                < 1e-12
        );
    }
}
