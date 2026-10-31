use l2s1::*;
use serde_json::{Value, json};

fn sample_identity() -> ModelIdentity {
    ModelIdentity {
        evidence_transfer: EvidenceTransfer::Full,
        weights_sha256: "a".repeat(64),
        template_sha256: "b".repeat(64),
        prompt_profile: "model".into(),
        prompt_version: "v1".into(),
        loaded_runtime_sha256: "d".repeat(64),
        runtime_build_sha256: "c".repeat(64),
        adapter_sha256: None,
        head_sha256: None,
        device: "CPU".into(),
        compute: ComputeOptions::default(),
        execution_mode: ExecutionMode::Fresh,
        parallel_width: 1,
        parallel_context_dynamic: false,
        parallel_prefix_alignment: ParallelPrefixAlignment::Batch,
        parallel_wave_order: ParallelWaveOrder::Request,
    }
}

fn choice_decision() -> Decision {
    Decision {
        id: "storage_zone".into(),
        instruction: "Select the storage zone matching storage_requirement.".into(),
        kind: DecisionKind::Choice {
            options: vec![
                OptionSpec {
                    id: "ambient".into(),
                    criterion: "Ambient storage is required.".into(),
                },
                OptionSpec {
                    id: "chilled".into(),
                    criterion: "Chilled storage is required.".into(),
                },
                OptionSpec {
                    id: "frozen".into(),
                    criterion: "Frozen storage is required.".into(),
                },
            ],
        },
    }
}

#[test]
fn temperature_and_policy_tuning_improves_ece_and_meets_target_accuracy() {
    let decision = choice_decision();
    // Overconfident logits where high-logit items are right and low-logit items have errors.
    let examples = vec![
        TuneExample {
            group: "g1".into(),
            shared: None,
            state: json!({"x": 1}),
            expected: json!("ambient"),
            raw_logits: Some(vec![9.0, 0.0, 0.0]),
            candidate_mass: Some(0.95),
        },
        TuneExample {
            group: "g2".into(),
            shared: None,
            state: json!({"x": 2}),
            expected: json!("chilled"),
            raw_logits: Some(vec![0.0, 9.0, 0.0]),
            candidate_mass: Some(0.92),
        },
        TuneExample {
            group: "g3".into(),
            shared: None,
            state: json!({"x": 3}),
            expected: json!("frozen"),
            raw_logits: Some(vec![0.0, 0.0, 9.0]),
            candidate_mass: Some(0.91),
        },
        TuneExample {
            group: "g4".into(),
            shared: None,
            state: json!({"x": 4}),
            expected: json!("chilled"),
            // Wrong prediction with lower margin and lower candidate mass.
            raw_logits: Some(vec![2.5, 1.0, 0.0]),
            candidate_mass: Some(0.20),
        },
        TuneExample {
            group: "g5".into(),
            shared: None,
            state: json!({"x": 5}),
            expected: json!("frozen"),
            // Wrong prediction with lower margin.
            raw_logits: Some(vec![2.0, 0.5, 0.0]),
            candidate_mass: Some(0.15),
        },
    ];

    let request = TuneRequest {
        id: Some("zone-cal".into()),
        shared: None,
        decision: decision.clone(),
        examples,
        held_out: Vec::new(),
        mode: TuningMode::PolicyAndTemperature,
        target_accepted_accuracy: Some(1.0),
        min_coverage: Some(0.5),
        l2_regularization: None,
        rotation_passes: None,
        apply: false,
    };

    let response = tune_records(&request, &sample_identity()).unwrap();
    assert_eq!(response.id, "zone-cal");
    assert_eq!(response.recommended_method, "temperature_calibration");
    assert!(response.target_met);
    assert!(response.tuned_fit.nll <= response.uncalibrated_fit.nll);
    assert_eq!(response.uncalibrated_fit.raw_top_accuracy, 0.6);
    assert_eq!(response.tuned_fit.accepted_accuracy, Some(1.0));
    assert!(response.tuned_fit.coverage >= 0.6);
    let cal = response.scalar_calibration.as_ref().unwrap();
    cal.validate(&sample_identity()).unwrap();
    assert!(cal.applies_to(&decision).unwrap());
    assert!(!response.policy_frontier.is_empty());
}

#[test]
fn logit_affine_tuning_corrects_systematic_prior_bias_and_improves_raw_accuracy() {
    let decision = Decision {
        id: "binary_flag".into(),
        instruction: "Is the item defective?".into(),
        kind: DecisionKind::Binary {
            false_label: "Normal".into(),
            true_label: "Defective".into(),
        },
    };
    // The base model has a +4.0 additive bias toward option 0 ("false"),
    // causing it to predict "false" even when "true" has higher relative signal!
    // True items have raw_logits [3.0, 1.5], False items have raw_logits [5.0, 0.0].
    let mut examples = Vec::new();
    for i in 0..8 {
        examples.push(TuneExample {
            group: format!("fit-false-{i}"),
            shared: None,
            state: json!({"i": i}),
            expected: json!(false),
            raw_logits: Some(vec![5.0, 0.0]),
            candidate_mass: Some(0.9),
        });
        examples.push(TuneExample {
            group: format!("fit-true-{i}"),
            shared: None,
            state: json!({"i": i + 100}),
            expected: json!(true),
            raw_logits: Some(vec![3.0, 1.5]),
            candidate_mass: Some(0.9),
        });
    }
    let mut held_out = Vec::new();
    for i in 0..4 {
        held_out.push(TuneExample {
            group: format!("held-false-{i}"),
            shared: None,
            state: json!({"h": i}),
            expected: json!("false"),
            raw_logits: Some(vec![4.8, 0.1]),
            candidate_mass: Some(0.9),
        });
        held_out.push(TuneExample {
            group: format!("held-true-{i}"),
            shared: None,
            state: json!({"h": i + 100}),
            expected: json!("true"),
            raw_logits: Some(vec![2.9, 1.6]),
            candidate_mass: Some(0.9),
        });
    }

    let request = TuneRequest {
        id: Some("bias-correct".into()),
        shared: None,
        decision: decision.clone(),
        examples,
        held_out,
        mode: TuningMode::Auto,
        target_accepted_accuracy: Some(0.95),
        min_coverage: Some(0.8),
        l2_regularization: Some(1e-2),
        rotation_passes: None,
        apply: false,
    };

    let response = tune_records(&request, &sample_identity()).unwrap();
    // Before tuning, raw_top_accuracy is 0.50 (all true examples are misclassified as false).
    assert_eq!(response.uncalibrated_fit.raw_top_accuracy, 0.5);
    assert_eq!(
        response
            .uncalibrated_held_out
            .as_ref()
            .unwrap()
            .raw_top_accuracy,
        0.5
    );
    // Auto mode selects logit_affine_head because it raises raw_top_accuracy from 0.50 to 1.00!
    assert_eq!(response.recommended_method, "logit_affine_head");
    assert_eq!(response.tuned_fit.raw_top_accuracy, 1.0);
    assert_eq!(
        response.tuned_held_out.as_ref().unwrap().raw_top_accuracy,
        1.0
    );
    assert!(response.target_met);

    // Verify that the returned OutputHead can be applied to a base DecisionResult!
    let head = response.output_head.as_ref().unwrap();
    head.validate(2).unwrap();
    let base = score_logits(
        &decision,
        &[2.9, 1.6, -5.0],
        &[0, 1],
        16,
        &DecisionPolicy::default(),
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&base.value).unwrap()["value"],
        json!(false)
    );
    let corrected = head
        .apply(&decision, &[], base, &response.recommended_policy)
        .unwrap();
    assert_eq!(
        serde_json::to_value(&corrected.value).unwrap()["value"],
        json!(true)
    );
}

#[test]
fn held_out_group_leakage_and_invalid_inputs_are_rejected() {
    let decision = choice_decision();
    let example = TuneExample {
        group: "shared-group".into(),
        shared: None,
        state: json!({}),
        expected: json!("ambient"),
        raw_logits: Some(vec![2.0, 1.0, 0.0]),
        candidate_mass: Some(0.9),
    };
    let leaked = TuneRequest {
        id: None,
        shared: None,
        decision: decision.clone(),
        examples: vec![example.clone()],
        held_out: vec![example.clone()],
        mode: TuningMode::Auto,
        target_accepted_accuracy: None,
        min_coverage: None,
        l2_regularization: None,
        rotation_passes: None,
        apply: false,
    };
    assert!(
        tune_records(&leaked, &sample_identity())
            .unwrap_err()
            .to_string()
            .contains("overlap")
    );

    let mut bad_option = example;
    bad_option.expected = json!("nonexistent_option");
    let bad_req = TuneRequest {
        id: None,
        shared: None,
        decision,
        examples: vec![bad_option],
        held_out: Vec::new(),
        mode: TuningMode::Auto,
        target_accepted_accuracy: None,
        min_coverage: None,
        l2_regularization: None,
        rotation_passes: None,
        apply: false,
    };
    assert!(tune_records(&bad_req, &sample_identity()).is_err());
}

struct MockTuneBackend;

impl DecisionBackend for MockTuneBackend {
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        let zone = request.state["temp"].as_str().unwrap_or("ambient");
        let logits = match zone {
            "ambient" => [4.0f32, 1.0, 0.0, -5.0],
            "chilled" => [0.5f32, 4.5, 0.5, -5.0],
            _ => [0.0f32, 0.5, 5.0, -5.0],
        };
        let result = score_logits(
            &request.decisions[0],
            &logits,
            &[0, 1, 2],
            12,
            &DecisionPolicy::default(),
        )?;
        Ok(DecisionResponse {
            backend: BackendInfo {
                prompt_detail: PromptDetail::Minimal,
                code_rotation: 0,
                evidence_transfer: EvidenceTransfer::Full,
                model_path: "mock.gguf".into(),
                vision_projector_path: None,
                vision_projector_sha256: None,
                vision_projector_reuse: false,
                lora_path: None,
                output_head_path: None,
                model_description: "mock".into(),
                model_architecture: "llama".into(),
                prompt_profile: "model".into(),
                prompt_layout: PromptLayout::Legacy,
                prompt_version: "v1".into(),
                runtime: "mock".into(),
                compute: Some(ComputeOptions::default()),
                execution_mode: ExecutionMode::Fresh,
                parallel_width: 1,
                parallel_context_dynamic: false,
                parallel_context_tokens: None,
                parallel_prefix_alignment: ParallelPrefixAlignment::Batch,
                parallel_wave_order: ParallelWaveOrder::Request,
                parallel_prefix_retained: false,
                offload_requested: false,
                offload_device: None,
            },
            policy: DecisionPolicy::default(),
            results: vec![result],
        })
    }
}

impl http::HttpDecisionBackend for MockTuneBackend {
    fn capabilities(&self) -> Value {
        json!({
            "api_version": 1,
            "backend": {"runtime": "mock", "model": "mock.gguf"},
            "evidence": "model_scored"
        })
    }
    fn decide_json(&mut self, request: &DecisionRequest, _: &[&[u8]]) -> Result<Value> {
        let response = self.decide(request)?;
        Ok(json!({
            "backend": {"runtime": "mock", "model": "mock.gguf", "details": response.backend},
            "policy": response.policy,
            "results": response.results.into_iter().map(|r| json!({
                "id": r.id,
                "value": {"type": "choice", "selected": "ambient"},
                "status": "selected",
                "abstention_reasons": r.abstention_reasons,
                "evidence": {
                    "type": "model_scored",
                    "scores": r.scores,
                    "candidate_mass": r.candidate_mass,
                    "top_option_probability": r.top_option_probability
                },
                "usage": {"input_tokens": r.input_tokens, "reused_prefix_tokens": 0}
            })).collect::<Vec<_>>()
        }))
    }
}

#[test]
fn tune_backend_and_stdio_evaluate_state_examples_end_to_end() {
    let decision = choice_decision();
    let request = TuneRequest {
        id: Some("live-tune".into()),
        shared: Some(json!({"facility": "seoul-1"})),
        decision,
        examples: vec![
            TuneExample {
                group: "s1".into(),
                shared: None,
                state: json!({"temp": "ambient"}),
                expected: json!("ambient"),
                raw_logits: None,
                candidate_mass: None,
            },
            TuneExample {
                group: "s2".into(),
                shared: None,
                state: json!({"temp": "chilled"}),
                expected: json!("chilled"),
                raw_logits: None,
                candidate_mass: None,
            },
            TuneExample {
                group: "s3".into(),
                shared: None,
                state: json!({"temp": "frozen"}),
                expected: json!("frozen"),
                raw_logits: None,
                candidate_mass: None,
            },
        ],
        held_out: Vec::new(),
        mode: TuningMode::Auto,
        target_accepted_accuracy: Some(0.95),
        min_coverage: Some(0.9),
        l2_regularization: None,
        rotation_passes: None,
        apply: false,
    };

    let mut backend = MockTuneBackend;
    let direct = tune_backend(&mut backend, &request, None).unwrap();
    assert_eq!(direct.tuned_fit.raw_top_accuracy, 1.0);
    assert!(direct.target_met);

    // Verify stdio "op": "tune" uses the exact same wire contract.
    let call = json!({
        "id": "tune-req-1",
        "op": "tune",
        "body": request
    });
    let mut input = serde_json::to_vec(&call).unwrap();
    input.push(b'\n');
    let mut output = Vec::new();
    stdio::serve_stream(&mut backend, std::io::Cursor::new(input), &mut output).unwrap();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["id"], "tune-req-1");
    assert_eq!(envelope["result"]["api_version"], 1);
    assert_eq!(envelope["result"]["request_id"], "tune-req-1");
    assert_eq!(envelope["result"]["id"], "live-tune");
    assert_eq!(envelope["result"]["tuned_fit"]["raw_top_accuracy"], 1.0);
    assert!(envelope["result"]["scalar_calibration"].is_object());
    assert!(envelope["result"]["output_head"].is_object());
}
