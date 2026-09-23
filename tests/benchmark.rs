//! Labeled, opt-in model benchmark. Ordinary tests validate the fixture and metrics.
use l2s1::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Suite {
    id: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Case {
    id: String,
    domain: String,
    request: DecisionRequest,
    /// Semantic option IDs; binary answers use "false" and "true".
    expected: BTreeMap<String, String>,
}

fn suite() -> Suite {
    let suite: Suite =
        serde_json::from_str(include_str!("fixtures/decision_benchmark.json")).unwrap();
    assert!(!suite.id.is_empty());
    assert!(!suite.cases.is_empty());
    let mut ids = HashSet::new();
    for case in &suite.cases {
        assert!(!case.domain.is_empty() && ids.insert(&case.id));
        case.request.validate().unwrap();
        assert_eq!(case.expected.len(), case.request.decisions.len());
        for decision in &case.request.decisions {
            let expected = case.expected.get(&decision.id).expect("missing label");
            assert!(decision.options().iter().any(|o| &o.id == expected));
        }
    }
    suite
}

#[derive(Default, Debug, Serialize)]
struct Counts {
    decisions: usize,
    accepted: usize,
    accepted_correct: usize,
    accepted_wrong: usize,
    abstained: usize,
    top1_correct: usize,
}

fn selected(result: &DecisionResult) -> Option<String> {
    match &result.value {
        DecisionValue::Binary { value, .. } => value.map(|v| v.to_string()),
        DecisionValue::Choice { selected } | DecisionValue::Ordinal { selected, .. } => {
            selected.clone()
        }
    }
}

fn top1(result: &DecisionResult) -> Option<&str> {
    let best = result
        .scores
        .iter()
        .max_by(|a, b| a.option_probability.total_cmp(&b.option_probability))?;
    (result
        .scores
        .iter()
        .filter(|s| (s.option_probability - best.option_probability).abs() < 1e-12)
        .count()
        == 1)
        .then_some(best.id.as_str())
}

impl Counts {
    fn add(&mut self, result: &DecisionResult, expected: &str) {
        self.decisions += 1;
        self.top1_correct += usize::from(top1(result) == Some(expected));
        match selected(result) {
            Some(answer) => {
                assert!(result.abstention_reasons.is_empty());
                self.accepted += 1;
                if answer == expected {
                    self.accepted_correct += 1;
                } else {
                    self.accepted_wrong += 1;
                }
            }
            None => {
                assert!(!result.abstention_reasons.is_empty());
                self.abstained += 1;
            }
        }
    }

    fn report(&self) -> serde_json::Value {
        let ratio = |n, d| (d > 0).then(|| n as f64 / d as f64);
        serde_json::json!({
            "counts": self,
            "coverage": ratio(self.accepted, self.decisions),
            "abstention_rate": ratio(self.abstained, self.decisions),
            "accepted_accuracy": ratio(self.accepted_correct, self.accepted),
            "correct_fraction": ratio(self.accepted_correct, self.decisions),
            "top1_accuracy_before_abstention": ratio(self.top1_correct, self.decisions),
        })
    }
}

fn percentile(sorted: &[f64], percent: usize) -> f64 {
    assert!(!sorted.is_empty() && (1..=100).contains(&percent));
    sorted[(sorted.len() * percent).div_ceil(100) - 1]
}

#[test]
fn benchmark_labels_match_independent_rules() {
    let suite = suite();
    assert_eq!(suite.cases.len(), 12);
    for case in suite.cases {
        let s = &case.request.state;
        let expected: BTreeMap<String, String> = match case.domain.as_str() {
            "warehouse" => {
                let hours = s["hours_until_dispatch"].as_u64().unwrap();
                let storage = s["storage_requirement"].as_str().unwrap();
                [
                    ("storage_zone", storage),
                    (
                        "cold_chain_required",
                        if storage == "ambient" {
                            "false"
                        } else {
                            "true"
                        },
                    ),
                    (
                        "dispatch_priority",
                        match hours {
                            0..=6 => "high",
                            7..=24 => "medium",
                            _ => "low",
                        },
                    ),
                ]
                .map(|(k, v)| (k.into(), v.into()))
                .into()
            }
            "access" => {
                let role = s["role"].as_str().unwrap();
                let can_edit =
                    s["enabled"].as_bool().unwrap() && matches!(role, "editor" | "admin");
                [
                    (
                        "role_scope",
                        match role {
                            "reader" => "read",
                            "editor" => "edit",
                            "admin" => "administer",
                            _ => panic!("unknown role"),
                        },
                    ),
                    ("can_edit", if can_edit { "true" } else { "false" }),
                    (
                        "review_priority",
                        match s["failed_attempts"].as_u64().unwrap() {
                            0 => "low",
                            1..=2 => "medium",
                            _ => "high",
                        },
                    ),
                ]
                .map(|(k, v)| (k.into(), v.into()))
                .into()
            }
            _ => panic!("unknown benchmark domain"),
        };
        assert_eq!(case.expected, expected, "{}", case.id);
    }
}

#[test]
fn benchmark_metrics_distinguish_wrong_and_abstained_answers() {
    let decision = &suite().cases[0].request.decisions[0];
    let expected = &decision.options()[0].id;
    let mut counts = Counts::default();
    for (logits, policy) in [
        ([8.0, 0.0, 0.0, 0.0], DecisionPolicy::default()),
        ([0.0, 8.0, 0.0, 0.0], DecisionPolicy::default()),
        ([8.0, 0.0, 0.0, 20.0], DecisionPolicy::default()),
    ] {
        counts.add(
            &score_logits(decision, &logits, &[0, 1, 2], 10, &policy).unwrap(),
            expected,
        );
    }
    assert_eq!(
        (
            counts.accepted_correct,
            counts.accepted_wrong,
            counts.abstained
        ),
        (1, 1, 1)
    );
    let report = counts.report();
    assert_eq!(report["accepted_accuracy"], 0.5);
    assert_eq!(report["correct_fraction"], 1.0 / 3.0);
    assert_eq!(report["top1_accuracy_before_abstention"], 2.0 / 3.0);
    let tied = score_logits(
        decision,
        &[0.0, 0.0, 0.0],
        &[0, 1, 2],
        10,
        &DecisionPolicy::default(),
    )
    .unwrap();
    let mut all_abstained = Counts::default();
    all_abstained.add(&tied, expected);
    assert!(all_abstained.report()["accepted_accuracy"].is_null());
    assert_eq!(all_abstained.top1_correct, 0);
    assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50), 2.0);
    assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 95), 4.0);
}

#[cfg(feature = "llama")]
mod native {
    use super::*;
    use l2s1::llama::LlamaBackend;
    use std::{env, fs, path::Path, time::Instant};

    fn setting(name: &str, default: u32, minimum: u32) -> u32 {
        let value = match env::var(name) {
            Ok(value) => value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be an integer")),
            Err(env::VarError::NotPresent) => default,
            Err(error) => panic!("{name}: {error}"),
        };
        assert!(value >= minimum, "{name} must be at least {minimum}");
        value
    }

    fn threshold(name: &str, default: f64) -> f64 {
        match env::var(name) {
            Ok(value) => value
                .parse()
                .unwrap_or_else(|_| panic!("{name} must be numeric")),
            Err(env::VarError::NotPresent) => default,
            Err(error) => panic!("{name}: {error}"),
        }
    }

    #[test]
    #[ignore = "requires SKID_MODEL; synthetic decision accuracy and inference performance benchmark"]
    fn model_decision_benchmark() {
        let suite = suite();
        let model = env::var("SKID_MODEL").expect("set SKID_MODEL to an existing local GGUF");
        let output =
            env::var("SKID_BENCH_OUTPUT").expect("set SKID_BENCH_OUTPUT to the JSON report path");
        let cuda = match env::var("SKID_CUDA").as_deref() {
            Ok("1") => true,
            Ok("0") | Err(env::VarError::NotPresent) => false,
            _ => panic!("SKID_CUDA must be 0 or 1"),
        };
        let iterations = setting("SKID_BENCH_ITERATIONS", 3, 1);
        let warmup = setting("SKID_BENCH_WARMUP", 1, 0);
        let context = setting("SKID_CONTEXT", 2048, 1);
        let batch = setting("SKID_BATCH", 256, 1);
        let threads = i32::try_from(setting("SKID_THREADS", 4, 1)).unwrap();
        let policy = DecisionPolicy {
            min_top_probability: threshold("SKID_MIN_TOP_PROBABILITY", 0.8),
            min_candidate_mass: threshold("SKID_MIN_CANDIDATE_MASS", 0.05),
        };
        policy.validate().unwrap();
        let started = Instant::now();
        let mut backend = LlamaBackend::load(
            Path::new(&model),
            context,
            batch,
            threads,
            cuda,
            policy.clone(),
        )
        .unwrap();
        let load_ms = started.elapsed().as_secs_f64() * 1000.0;
        backend.set_prompt_layout(match env::var("SKID_PROMPT_LAYOUT").as_deref() {
            Ok("legacy") | Err(env::VarError::NotPresent) => PromptLayout::Legacy,
            Ok("state-first") => PromptLayout::StateFirst,
            _ => panic!("SKID_PROMPT_LAYOUT must be legacy or state-first"),
        });
        backend.set_execution_mode(match env::var("SKID_EXECUTION_MODE").as_deref() {
            Ok("fresh") | Err(env::VarError::NotPresent) => ExecutionMode::Fresh,
            Ok("prefix-reuse") => ExecutionMode::PrefixReuse,
            _ => panic!("SKID_EXECUTION_MODE must be fresh or prefix-reuse"),
        });
        let started = Instant::now();
        let first = backend
            .decide(&suite.cases[0].request)
            .expect("first inference failed");
        let first_request_ms = started.elapsed().as_secs_f64() * 1000.0;
        // Warm every case; exclude both the first request and all warmup passes.
        for _ in 0..warmup {
            for case in &suite.cases {
                backend.decide(&case.request).expect("warmup failed");
            }
        }
        let mut total = Counts::default();
        let mut groups: BTreeMap<String, Counts> = BTreeMap::new();
        let mut latencies = Vec::new();
        let mut samples = Vec::new();
        let mut input_tokens = 0;
        let mut reused_prefix_tokens = 0;
        let mut baseline = BTreeMap::new();
        let mut repeated_comparisons = 0;
        let mut changed_outputs = 0;
        for pass in 0..iterations {
            // Deterministic cyclic ordering reduces repeated adjacent-case effects.
            for offset in 0..suite.cases.len() {
                let case = &suite.cases[(offset + pass as usize) % suite.cases.len()];
                let started = Instant::now();
                let response = backend
                    .decide(&case.request)
                    .unwrap_or_else(|e| panic!("{}: {e}", case.id));
                let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                assert!(elapsed_ms > 0.0 && elapsed_ms.is_finite());
                assert_eq!(response.results.len(), case.request.decisions.len());
                for (result, decision) in response.results.iter().zip(&case.request.decisions) {
                    assert_eq!(result.id, decision.id);
                    assert!(!result.truncated && result.input_tokens > 0);
                    assert_eq!(result.scores.len(), decision.options().len());
                    assert!(
                        result
                            .scores
                            .iter()
                            .all(|s| s.raw_logit.is_finite() && s.option_probability.is_finite())
                    );
                    let expected = &case.expected[&result.id];
                    total.add(result, expected);
                    groups
                        .entry(format!("domain/{}", case.domain))
                        .or_default()
                        .add(result, expected);
                    let kind = match decision.kind {
                        DecisionKind::Choice { .. } => "choice",
                        DecisionKind::Binary { .. } => "binary",
                        DecisionKind::Ordinal { .. } => "ordinal",
                    };
                    groups
                        .entry(format!("kind/{kind}"))
                        .or_default()
                        .add(result, expected);
                    let key = format!("{}/{}", case.id, result.id);
                    let signature = (selected(result), top1(result).map(str::to_owned));
                    if pass == 0 {
                        baseline.insert(key, signature);
                    } else {
                        repeated_comparisons += 1;
                        changed_outputs += usize::from(baseline[&key] != signature);
                    }
                    input_tokens += result.input_tokens;
                    reused_prefix_tokens += result.reused_prefix_tokens;
                }
                latencies.push(elapsed_ms);
                samples.push(serde_json::json!({"pass":pass,"case":case.id,"domain":case.domain,"latency_ms":elapsed_ms,"expected":case.expected,"results":response.results}));
            }
            println!("Completed benchmark pass {}/{}", pass + 1, iterations);
        }
        let seconds = latencies.iter().sum::<f64>() / 1000.0;
        let mut sorted = latencies.clone();
        sorted.sort_by(f64::total_cmp);
        let grouped: BTreeMap<_, _> = groups.iter().map(|(k, v)| (k, v.report())).collect();
        let report = serde_json::json!({
            "schema_version":1,"suite":suite.id,"unique_cases":suite.cases.len(),
            "unique_decisions":suite.cases.iter().map(|c| c.request.decisions.len()).sum::<usize>(),
            "iterations":iterations,"warmup_passes":warmup,"context":context,"batch":batch,"threads":threads,
            "policy":policy,"backend":first.backend,"model_bytes":fs::metadata(&model).unwrap().len(),
            "crate_version":env!("CARGO_PKG_VERSION"),"build_profile":if cfg!(debug_assertions) {"debug"} else {"release"},
            "host":{"os":env::consts::OS,"arch":env::consts::ARCH},
            "load_ms":load_ms,"first_request_ms":first_request_ms,"measured_inference_seconds":seconds,
            "latency_ms":{"samples":latencies,"mean":seconds*1000.0/sorted.len() as f64,"p50":percentile(&sorted,50),"p95":percentile(&sorted,95),"min":sorted[0],"max":sorted[sorted.len()-1]},
            "requests_per_second":samples.len() as f64/seconds,"decisions_per_second":total.decisions as f64/seconds,
            "accepted_correct_decisions_per_second":total.accepted_correct as f64/seconds,
            "input_tokens":input_tokens,"input_tokens_per_second":input_tokens as f64/seconds,
            "reused_prefix_tokens":reused_prefix_tokens,
            "evaluated_tokens":input_tokens-reused_prefix_tokens,
            "quality":total.report(),"by_group":grouped,
            "repeat_consistency":{"comparisons":repeated_comparisons,"changed_outputs":changed_outputs},
            "samples":samples,
            "measurement":"Sequential end-to-end decide calls, including prefill and scoring; no generated tokens. Warmups/loading excluded. Repeated passes are not additional independent accuracy examples."
        });
        let output = Path::new(&output);
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
        println!("Benchmark report: {}", output.display());
    }
}
