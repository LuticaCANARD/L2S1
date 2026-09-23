#![cfg(feature = "llama")]

use l2s1::{
    Decision, DecisionBackend, DecisionKind, DecisionPolicy, DecisionRequest, DecisionResult,
    DecisionValue, ExecutionMode, PromptLayout, llama::LlamaBackend,
};

fn compare(reference: &DecisionResult, actual: &DecisionResult) {
    assert_eq!(actual.id, reference.id);
    assert_eq!(actual.input_tokens, reference.input_tokens);
    assert_eq!(actual.scores.len(), reference.scores.len());
    assert_eq!(actual.truncated, reference.truncated);
    assert!((actual.candidate_mass - reference.candidate_mass).abs() <= 0.02);
    for (a, b) in actual.scores.iter().zip(&reference.scores) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.code, b.code);
        assert_eq!(a.token_id, b.token_id);
        assert!((a.option_probability - b.option_probability).abs() <= 0.02);
    }
    match (&actual.value, &reference.value) {
        (
            DecisionValue::Binary {
                p_true: a,
                value: av,
            },
            DecisionValue::Binary {
                p_true: b,
                value: bv,
            },
        ) => {
            assert_eq!(av, bv);
            assert!((a - b).abs() <= 0.02);
        }
        (DecisionValue::Choice { selected: a }, DecisionValue::Choice { selected: b }) => {
            assert_eq!(a, b);
        }
        (
            DecisionValue::Ordinal { selected: a, .. },
            DecisionValue::Ordinal { selected: b, .. },
        ) => assert_eq!(a, b),
        _ => panic!("decision kind changed"),
    }
    assert_eq!(
        serde_json::to_value(&actual.abstention_reasons).unwrap(),
        serde_json::to_value(&reference.abstention_reasons).unwrap()
    );
}

/// Model-dependent numerical differences are bounded separately from selection
/// equality. This fixture checks token identity, changing schemas, invalid-call
/// recovery and request isolation; it is not a quality or throughput benchmark.
#[test]
#[ignore = "requires SKID_MODEL and optionally SKID_CUDA=1"]
fn shared_state_supports_dynamic_questions_and_clears_boundaries() {
    let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    let mut backend =
        LlamaBackend::load(model.as_ref(), 2048, 32, 4, cuda, DecisionPolicy::default()).unwrap();
    backend.set_prompt_layout(PromptLayout::StateFirst);
    // Session creation must never silently switch away from the default mode.
    assert!(backend.shared_state(request.state.clone()).is_err());
    assert_eq!(
        backend.inspect().identity.execution_mode,
        ExecutionMode::Fresh
    );
    if backend.inspect().capabilities.recurrent_or_hybrid {
        backend.set_execution_mode(ExecutionMode::PrefixReuse);
        assert!(backend.shared_state(request.state.clone()).is_err());
        return;
    }

    let mut questions = request.decisions.clone();
    let mut changed = questions[0].clone();
    changed.instruction = "Read storage_requirement and choose its exact storage zone.".into();
    if let DecisionKind::Choice { options } = &mut changed.kind {
        options.swap(0, 1);
        options.pop();
    }
    // Reuse the original ID with changed instruction, candidate order and arity.
    questions.push(changed);
    questions.push(request.decisions[0].clone());
    let reference_tokens: Vec<_> = questions
        .iter()
        .map(|question| backend.encode_decision(&request.state, question).unwrap())
        .collect();
    let references: Vec<_> = questions
        .iter()
        .map(|question| {
            backend
                .decide(&DecisionRequest {
                    state: request.state.clone(),
                    decisions: vec![question.clone()],
                })
                .unwrap()
                .results
                .remove(0)
        })
        .collect();
    backend.set_execution_mode(ExecutionMode::PrefixReuse);
    for (question, tokens) in questions.iter().zip(&reference_tokens) {
        assert_eq!(
            backend.encode_decision(&request.state, question).unwrap(),
            *tokens
        );
    }
    {
        let mut session = backend.shared_state(request.state.clone()).unwrap();
        let mut observed_reuse = false;
        for (index, (question, reference)) in questions.iter().zip(&references).enumerate() {
            let actual = session.decide(vec![question.clone()]).unwrap();
            compare(reference, &actual.results[0]);
            if index == 0 {
                assert_eq!(actual.results[0].reused_prefix_tokens, 0);
            } else {
                observed_reuse |= actual.results[0].reused_prefix_tokens > 0;
            }
        }
        assert!(
            observed_reuse,
            "fixture should expose reusable full batches"
        );
        let oversized = Decision {
            id: "oversized".into(),
            instruction: "word ".repeat(4096),
            ..questions[0].clone()
        };
        let invalid_calls: Vec<Vec<Decision>> = vec![
            Vec::new(),
            vec![questions[0].clone(), questions[0].clone()],
            // Fail after the first decision has already populated native KV.
            vec![questions[0].clone(), oversized],
            vec![Decision {
                kind: DecisionKind::Choice {
                    options: Vec::new(),
                },
                ..questions[0].clone()
            }],
        ];
        for invalid in invalid_calls {
            assert!(session.decide(invalid).is_err());
            let recovered = session.decide(vec![questions[0].clone()]).unwrap();
            assert_eq!(recovered.results[0].reused_prefix_tokens, 0);
            compare(&references[0], &recovered.results[0]);
        }
    }
    // Dropping a session clears KV even before another ordinary request.
    {
        let mut next = backend.shared_state(request.state.clone()).unwrap();
        let first = next.decide(vec![questions[0].clone()]).unwrap();
        assert_eq!(first.results[0].reused_prefix_tokens, 0);
        compare(&references[0], &first.results[0]);
    }
    let mut independent = request.clone();
    independent.state["storage_requirement"] = serde_json::json!("ambient");
    independent.decisions.truncate(1);
    let actual = backend.decide(&independent).unwrap();
    assert_eq!(actual.results[0].reused_prefix_tokens, 0);
    backend.set_execution_mode(ExecutionMode::Fresh);
    let reference = backend.decide(&independent).unwrap();
    compare(&reference.results[0], &actual.results[0]);
}
