use l2s1::*;
fn request() -> DecisionRequest {
    serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap()
}
fn identity() -> ModelIdentity {
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
        compute: ComputeOptions {
            context: 512,
            batch: 32,
            ubatch: 32,
            threads: 1,
            flash_attention: FlashAttention::Off,
            gpu_layers: None,
            cpu_moe_layers: 0,
            model_load_mode: l2s1::ModelLoadMode::Auto,
        },
        execution_mode: ExecutionMode::Fresh,
        parallel_width: 1,
        parallel_context_dynamic: false,
        parallel_prefix_alignment: ParallelPrefixAlignment::Batch,
        parallel_wave_order: ParallelWaveOrder::Request,
    }
}

#[test]
fn dynamic_parallel_context_has_separate_artifact_identity() {
    let legacy = identity();
    let serialized = serde_json::to_value(&legacy).unwrap();
    assert!(serialized.get("parallel_context_dynamic").is_none());
    let historical: ModelIdentity = serde_json::from_value(serialized).unwrap();
    assert_eq!(legacy.fingerprint(), historical.fingerprint());
    let mut dynamic = legacy.clone();
    dynamic.parallel_context_dynamic = true;
    assert_ne!(dynamic.fingerprint(), legacy.fingerprint());
}
#[test]
fn exact_evidence_matches_independent_full_softmax_for_all_kinds() {
    let policy = DecisionPolicy {
        min_top_probability: 0.0,
        min_candidate_mass: 0.0,
    };
    for decision in request().decisions {
        let logits = [0.7f32, -1.5, 1.8, 0.2, -3.0];
        let tokens = if decision.options().len() == 2 {
            vec![2, 0]
        } else {
            vec![2, 0, 3]
        };
        let evidence = ExactEvidence::from_logits(&decision, &logits, &tokens).unwrap();
        let actual = evidence.score(&decision, 42, &policy).unwrap();
        let full: f64 = logits.iter().map(|&z| (z as f64).exp()).sum();
        let candidates: f64 = tokens
            .iter()
            .map(|&t| (logits[t as usize] as f64).exp())
            .sum();
        assert!((actual.candidate_mass - candidates / full).abs() < 1e-14);
        for (s, &t) in actual.scores.iter().zip(&tokens) {
            assert!(
                (s.option_probability - (logits[t as usize] as f64).exp() / candidates).abs()
                    < 1e-14
            );
        }
        let mut changed = decision.clone();
        if let DecisionKind::Choice { options } = &mut changed.kind {
            options.swap(0, 1);
            assert!(evidence.score(&changed, 42, &policy).is_err());
        }
        assert_eq!(
            serde_json::to_value(&actual).unwrap(),
            serde_json::to_value(score_logits(&decision, &logits, &tokens, 42, &policy).unwrap())
                .unwrap()
        );
        assert!(ExactEvidence::from_logits(&decision, &logits, &vec![0; tokens.len()]).is_err());
        assert!(ExactEvidence::from_logits(&decision, &[f32::NAN; 5], &tokens).is_err());
        assert!(ExactEvidence::from_logits(&decision, &logits, &tokens[..1]).is_err());
        assert!(ExactEvidence::from_logits(&decision, &logits, &vec![99; tokens.len()]).is_err());
    }
}
#[test]
fn calibration_preserves_base_evidence_and_rejects_cross_model_task_and_leakage() {
    for task in request().decisions {
        let n = task.options().len();
        let records: Vec<_> = (0..12)
            .map(|i| CalibrationRecord {
                group: format!("source-{i}"),
                raw_logits: (0..n).map(|j| if j == 0 { 8.0 } else { 0.0 }).collect(),
                correct_option: i % n,
            })
            .collect();
        let artifact =
            ScalarCalibration::fit("held-out-temperature".into(), &identity(), &task, &records)
                .unwrap();
        assert!(artifact.temperature > 1.0);
        assert!(artifact.calibrated_fit_metrics.nll < artifact.uncalibrated_fit_metrics.nll);
        artifact.validate(&identity()).unwrap();
        let mut wrong = identity();
        wrong.weights_sha256 = "d".repeat(64);
        assert!(artifact.validate(&wrong).is_err());
        wrong = identity();
        wrong.template_sha256 = "e".repeat(64);
        assert!(artifact.validate(&wrong).is_err());
        wrong = identity();
        wrong.adapter_sha256 = Some("f".repeat(64));
        assert!(artifact.validate(&wrong).is_err());
        wrong = identity();
        wrong.execution_mode = ExecutionMode::PrefixReuse;
        assert!(artifact.validate(&wrong).is_err());
        let mut changed = task.clone();
        changed.instruction.push('!');
        assert!(artifact.applies_to(&changed).is_err());
        assert!(artifact.evaluate_held_out(&records).is_err());
        let mut held = records.clone();
        for r in &mut held {
            r.group.push_str("-independent");
        }
        artifact.evaluate_held_out(&held).unwrap();
        let mut wrong_width = held.clone();
        for r in &mut wrong_width {
            r.raw_logits.push(0.0);
        }
        assert!(artifact.evaluate_held_out(&wrong_width).is_err());
        let logits: Vec<_> = (0..n + 1)
            .map(|i| if i == 0 { 8.0f32 } else { 0.0 })
            .collect();
        let tokens: Vec<_> = (0..n as i32).collect();
        let policy = DecisionPolicy {
            min_top_probability: 0.8,
            min_candidate_mass: 0.0,
        };
        let base = score_logits(&task, &logits, &tokens, 20, &policy).unwrap();
        let expected = serde_json::to_value(&base).unwrap();
        let result = artifact.apply(&task, base, &policy).unwrap();
        assert_eq!(
            result.candidate_mass,
            expected["candidate_mass"].as_f64().unwrap()
        );
        assert!(!result.abstention_reasons.is_empty());
        assert_eq!(result.scores[0].raw_logit, 8.0);
        assert_eq!(
            result.calibration_id.as_deref(),
            Some("held-out-temperature")
        );
        assert_eq!(
            result
                .scores
                .iter()
                .max_by(|a, b| a.option_probability.total_cmp(&b.option_probability))
                .unwrap()
                .token_id,
            0
        );
    }
}
#[test]
fn worker_owns_non_send_backend_bounds_admission_and_releases_memory() {
    use std::{rc::Rc, sync::mpsc};
    struct Backend {
        _not_send: Rc<()>,
        owner: std::thread::ThreadId,
        entered: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
        dropped: mpsc::Sender<bool>,
    }
    impl DecisionBackend for Backend {
        fn decide(&mut self, _: &DecisionRequest) -> Result<DecisionResponse> {
            assert_eq!(self.owner, std::thread::current().id());
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            Err(Error::Backend("fixture failure".into()))
        }
    }
    impl Drop for Backend {
        fn drop(&mut self) {
            let _ = self.dropped.send(self.owner == std::thread::current().id());
        }
    }
    let budget = MemoryBudget::new(100);
    let (entered, entry) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let (dropped, drop_rx) = mpsc::channel();
    let mut worker = BackendWorker::spawn(1, 10000, &budget, 80, move || {
        Ok(Backend {
            _not_send: Rc::new(()),
            owner: std::thread::current().id(),
            entered,
            release: gate,
            dropped,
        })
    })
    .unwrap();
    assert_eq!(budget.reserved_bytes(), 80);
    assert!(BackendWorker::spawn::<Backend, _>(1, 10000, &budget, 30, || unreachable!()).is_err());
    let first = worker.submit(request()).unwrap();
    entry.recv().unwrap();
    let second = worker.submit(request()).unwrap();
    assert!(worker.submit(request()).is_err());
    release.send(()).unwrap();
    entry.recv().unwrap();
    release.send(()).unwrap();
    worker.close().unwrap();
    assert!(drop_rx.recv().unwrap());
    assert_eq!(budget.reserved_bytes(), 0);
    assert!(first.wait().is_err());
    assert!(second.wait().is_err());
    assert!(worker.submit(request()).is_err());
    worker.close().unwrap();
    assert!(
        BackendWorker::spawn::<Backend, _>(1, 10000, &budget, 80, || Err(Error::Backend(
            "factory failure".into()
        )))
        .is_err()
    );
    assert_eq!(budget.reserved_bytes(), 0);
    assert!(
        BackendWorker::spawn::<Backend, _>(1, 10000, &budget, 80, || panic!("fixture panic"))
            .is_err()
    );
    assert_eq!(budget.reserved_bytes(), 0);
}

#[test]
fn held_out_policy_metrics_keep_mass_and_tie_gates() {
    let task = request().decisions.remove(1);
    let fit = vec![CalibrationRecord {
        group: "fit".into(),
        raw_logits: vec![0.0, 3.0],
        correct_option: 1,
    }];
    let mut artifact = ScalarCalibration::fit("policy".into(), &identity(), &task, &fit).unwrap();
    artifact.temperature = 1.0;
    let observations = vec![
        CalibrationPolicyRecord {
            observation: CalibrationRecord {
                group: "held-a".into(),
                raw_logits: vec![0.0, 3.0],
                correct_option: 1,
            },
            base_candidate_mass: 0.9,
        },
        CalibrationPolicyRecord {
            observation: CalibrationRecord {
                group: "held-b".into(),
                raw_logits: vec![0.0, 3.0],
                correct_option: 0,
            },
            base_candidate_mass: 0.001,
        },
        CalibrationPolicyRecord {
            observation: CalibrationRecord {
                group: "held-c".into(),
                raw_logits: vec![1.0, 1.0],
                correct_option: 1,
            },
            base_candidate_mass: 0.9,
        },
    ];
    let metrics = artifact
        .evaluate_policy(
            &observations,
            &DecisionPolicy {
                min_top_probability: 0.0,
                min_candidate_mass: 0.05,
            },
        )
        .unwrap();
    assert_eq!(metrics.accepted, 1);
    assert_eq!(metrics.accepted_accuracy, Some(1.0));
    assert_eq!(metrics.coverage, 1.0 / 3.0);
    let none = artifact
        .evaluate_policy(
            &observations,
            &DecisionPolicy {
                min_top_probability: 1.0,
                min_candidate_mass: 1.0,
            },
        )
        .unwrap();
    assert_eq!(none.accepted_accuracy, None);
    let forced = artifact
        .evaluate_policy(&observations, &DecisionPolicy::default())
        .unwrap();
    assert_eq!(forced.accepted, 3);
    assert_eq!(forced.coverage, 1.0);
    assert_eq!(forced.accepted_accuracy, Some(1.0 / 3.0));
    let mut invalid = observations;
    invalid[0].base_candidate_mass = f64::NAN;
    assert!(
        artifact
            .evaluate_policy(&invalid, &DecisionPolicy::default())
            .is_err()
    );
}

fn overconfident(group: &str, n: usize, count: usize) -> Vec<CalibrationRecord> {
    // Always 8 logits in favour of option 0, correct only 1/n of the time.
    (0..count)
        .map(|i| CalibrationRecord {
            group: format!("{group}-{i}"),
            raw_logits: (0..n).map(|j| if j == 0 { 8.0 } else { 0.0 }).collect(),
            correct_option: i % n,
        })
        .collect()
}

#[test]
fn calibration_metrics_report_top_label_ece() {
    // Confidence ~1.0 with accuracy 1/3: ECE close to 2/3.
    let metrics = calibration_metrics(&overconfident("s", 3, 30), 1.0).unwrap();
    assert!((metrics.ece.unwrap() - (1.0 - 1.0 / 3.0)).abs() < 1e-3);
    // Uniform scores that are right exactly 1/3 of the time are calibrated.
    let uniform: Vec<_> = (0..30)
        .map(|i| CalibrationRecord {
            group: format!("u-{i}"),
            raw_logits: vec![0.0; 3],
            correct_option: i % 3,
        })
        .collect();
    assert!(calibration_metrics(&uniform, 1.0).unwrap().ece.unwrap() < 1e-9);
    // Artifacts written before ECE was reported still deserialize.
    let old: CalibrationMetrics =
        serde_json::from_value(serde_json::json!({"examples": 1, "nll": 0.5, "brier": 0.1}))
            .unwrap();
    assert!(old.ece.is_none());
}

#[test]
fn family_calibration_generalizes_by_kind_and_reports_leave_one_task_out() {
    let decisions = request().decisions;
    let binary: Vec<_> = decisions
        .iter()
        .filter(|d| decision_kind_name(d) == "binary")
        .cloned()
        .collect();
    let base = binary[0].clone();
    let tasks: Vec<_> = (0..3)
        .map(|t| {
            let mut decision = base.clone();
            decision.id = format!("task-{t}");
            decision.instruction = format!("Question {t}?");
            FamilyCalibrationTask {
                decision,
                records: overconfident(&format!("task-{t}"), 2, 12),
            }
        })
        .collect();
    let family = FamilyCalibration::fit("binary-family".into(), &identity(), &tasks).unwrap();
    assert_eq!(
        (family.decision_kind.as_str(), family.option_count),
        ("binary", 2)
    );
    assert!(family.temperature > 1.0);
    assert!(family.calibrated_fit_metrics.nll < family.uncalibrated_fit_metrics.nll);
    assert_eq!(family.leave_one_task_out.len(), 3);
    for held in &family.leave_one_task_out {
        assert!(held.calibrated.nll < held.uncalibrated.nll);
        assert!(held.calibrated.ece.unwrap() < held.uncalibrated.ece.unwrap());
    }
    family.validate(&identity()).unwrap();
    let mut wrong = identity();
    wrong.parallel_prefix_alignment = ParallelPrefixAlignment::Token;
    assert!(
        family.validate(&wrong).is_err(),
        "alignment is part of the identity"
    );
    wrong = identity();
    wrong.parallel_wave_order = ParallelWaveOrder::Prefix;
    assert!(
        family.validate(&wrong).is_err(),
        "wave order is part of the identity"
    );

    // Applies to an unseen task of the same kind and width, not to others.
    let mut unseen = base.clone();
    unseen.id = "unseen".into();
    unseen.instruction = "A new question?".into();
    assert!(family.applies_to(&unseen));
    let three = decisions.iter().find(|d| d.options().len() == 3);
    if let Some(three) = three {
        assert!(!family.applies_to(three));
    }
    let policy = DecisionPolicy {
        min_top_probability: 0.8,
        min_candidate_mass: 0.0,
    };
    let logits = [8.0f32, 0.0, 0.0];
    let fresh = || score_logits(&unseen, &logits, &[0, 1], 20, &policy).unwrap();
    let base_result = fresh();
    let calibrated = family.apply(&unseen, fresh(), &policy).unwrap();
    assert_eq!(calibrated.calibration_id.as_deref(), Some("binary-family"));
    assert_eq!(
        calibrated.scoring_method,
        "family_temperature_softmax_with_base_mass_v1"
    );
    assert_eq!(calibrated.scores[0].raw_logit, 8.0);
    assert_eq!(calibrated.candidate_mass, base_result.candidate_mass);
    let top = |r: &DecisionResult| r.scores[0].option_probability;
    assert!(top(&calibrated) < top(&base_result));
    // A result a task-specific calibration already handled stays unchanged.
    let task_artifact = ScalarCalibration::fit(
        "task".into(),
        &identity(),
        &unseen,
        &overconfident("x", 2, 8),
    )
    .unwrap();
    let specific = || task_artifact.apply(&unseen, fresh(), &policy).unwrap();
    let after = family.apply(&unseen, specific(), &policy).unwrap();
    assert_eq!(after.calibration_id.as_deref(), Some("task"));
    assert_eq!(top(&after), top(&specific()));

    // Held-out evaluation rejects source-group leakage.
    assert!(family.evaluate_held_out(&tasks[0].records).is_err());
    family
        .evaluate_held_out(&overconfident("independent", 2, 6))
        .unwrap();

    // Fitting needs two distinct tasks of one kind and width.
    assert!(FamilyCalibration::fit("one".into(), &identity(), &tasks[..1]).is_err());
    let duplicate = vec![tasks[0].clone(), tasks[0].clone()];
    assert!(FamilyCalibration::fit("dup".into(), &identity(), &duplicate).is_err());
    if let Some(three) = three {
        let mixed = vec![
            tasks[0].clone(),
            FamilyCalibrationTask {
                decision: three.clone(),
                records: overconfident("three", 3, 6),
            },
        ];
        assert!(FamilyCalibration::fit("mixed".into(), &identity(), &mixed).is_err());
    }
}
