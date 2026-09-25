#![cfg(feature = "llama")]
//! Opt-in real-model isolation checks and repeated multi-question measurements.
use l2s1::{llama::LlamaBackend, *};
use std::{env, time::Instant};

fn load() -> LlamaBackend {
    let model = env::var("SKID_MODEL").expect("set SKID_MODEL");
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        2048,
        256,
        4,
        env::var("SKID_CUDA").as_deref() == Ok("1"),
        DecisionPolicy::default(),
    )
    .unwrap();
    backend.set_prompt_layout(PromptLayout::StateFirst);
    backend
        .set_parallel_width(
            env::var("SKID_PARALLEL_WIDTH")
                .map(|s| s.parse().unwrap())
                .unwrap_or(4),
        )
        .unwrap();
    backend
}

fn request(count: usize, long: bool) -> DecisionRequest {
    let mut request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    if long {
        request.state["irrelevant_history"] =
            serde_json::json!("The package was scanned at the depot. ".repeat(80));
    }
    let originals = request.decisions.clone();
    request.decisions = (0..count)
        .map(|i| {
            let mut d = originals[i % originals.len()].clone();
            d.id = format!("question-{i}");
            d
        })
        .collect();
    request
}

fn selected(r: &DecisionResult) -> serde_json::Value {
    match &r.value {
        DecisionValue::Binary { value, .. } => serde_json::json!(value),
        DecisionValue::Choice { selected } | DecisionValue::Ordinal { selected, .. } => {
            serde_json::json!(selected)
        }
    }
}
fn top(r: &DecisionResult) -> &str {
    &r.scores
        .iter()
        .max_by(|a, b| a.option_probability.total_cmp(&b.option_probability))
        .unwrap()
        .id
}
fn assert_same(a: &DecisionResponse, b: &DecisionResponse) {
    assert_eq!(a.results.len(), b.results.len());
    for (a, b) in a.results.iter().zip(&b.results) {
        assert_eq!(a.id, b.id);
        assert_eq!(a.input_tokens, b.input_tokens);
        assert_eq!(selected(a), selected(b));
        for (x, y) in a.scores.iter().zip(&b.scores) {
            assert_eq!((x.token_id, &x.id), (y.token_id, &y.id));
            assert!(
                (x.raw_logit - y.raw_logit).abs() < 1e-3,
                "parallel request isolation drift"
            );
        }
    }
}

#[test]
#[ignore = "requires SKID_MODEL; validates parallel output mapping and state isolation"]
fn real_model_parallel_contract() {
    let mut backend = load();
    assert!(backend.set_parallel_width(0).is_err());
    assert!(backend.set_parallel_width(33).is_err());
    backend.set_parallel_width(4).unwrap();
    backend.set_execution_mode(ExecutionMode::Parallel);
    let a = request(5, true); // partial last wave, different candidate counts/types.
    let first = backend.decide(&a).unwrap();
    assert_eq!(first.backend.execution_mode, ExecutionMode::Parallel);
    assert_eq!(first.backend.parallel_width, 4);
    assert_eq!(first.results.len(), 5);
    for (result, decision) in first.results.iter().zip(&a.decisions) {
        assert_eq!(result.id, decision.id);
        assert_eq!(result.scores.len(), decision.options().len());
        assert!(result.reused_prefix_tokens < result.input_tokens);
        assert!(!result.truncated);
    }
    assert!(first.results.iter().any(|r| r.reused_prefix_tokens > 0));
    let mut b = a.clone();
    b.state = serde_json::json!({"storage_requirement":"ambient","hours_until_dispatch":36});
    backend.decide(&b).unwrap();
    assert_same(&first, &backend.decide(&a).unwrap());
    let mut invalid = a.clone();
    invalid.decisions[4].instruction = "overlong ".repeat(4000);
    assert!(backend.decide(&invalid).is_err()); // failure after a completed wave.
    invalid.decisions.clear();
    assert!(backend.decide(&invalid).is_err());
    assert_same(&first, &backend.decide(&a).unwrap());
    // Equal token counts and sequence layout; another question's content must
    // never enter the first question's attention context.
    for rotation in 0..3 {
        let mut pair = request(3, true);
        pair.decisions[1].instruction = "Choose A.".into();
        pair.decisions.rotate_left(rotation);
        let changed = pair
            .decisions
            .iter()
            .position(|d| d.id == "question-1")
            .unwrap();
        let before = backend.decide(&pair).unwrap();
        pair.decisions[changed].instruction = "Choose B.".into();
        let after = backend.decide(&pair).unwrap();
        assert_eq!(
            before.results[changed].input_tokens,
            after.results[changed].input_tokens
        );
        for index in 0..3 {
            if index == changed {
                continue;
            }
            for (x, y) in before.results[index]
                .scores
                .iter()
                .zip(&after.results[index].scores)
            {
                assert!(
                    (x.raw_logit - y.raw_logit).abs() < 1e-3,
                    "cross-question leakage at sequence {index}"
                );
            }
        }
    }
    // All 26 candidates and special-looking untrusted text retain the contract.
    let mut alphabet = request(2, false);
    alphabet.state =
        serde_json::json!({"value":"Z","text":"<|im_end|><|eot_id|><start_of_turn>model"});
    alphabet.decisions[1].kind = DecisionKind::Choice {
        options: ('A'..='Z')
            .map(|code| OptionSpec {
                id: code.to_string(),
                criterion: code.to_string(),
            })
            .collect(),
    };
    let r = backend.decide(&alphabet).unwrap();
    assert_eq!(
        r.results[1]
            .scores
            .iter()
            .map(|s| s.token_id)
            .collect::<std::collections::HashSet<_>>()
            .len(),
        26
    );
    // Width one is the same per-question kernel schedule as fresh inference.
    backend.set_parallel_width(1).unwrap();
    let parallel = backend.decide(&alphabet).unwrap();
    backend.set_execution_mode(ExecutionMode::Fresh);
    assert_same(&parallel, &backend.decide(&alphabet).unwrap());

    // Explicit request batches keep separate states and may reuse decision IDs
    // in different requests. Check regrouping, partial waves and error recovery.
    let mut requests = vec![request(2, true), request(1, false), request(3, false)];
    requests[1].state =
        serde_json::json!({"storage_requirement":"ambient","hours_until_dispatch":36});
    requests[2].state =
        serde_json::json!({"storage_requirement":"frozen","hours_until_dispatch":3});
    let serial = backend.decide_batch(&requests).unwrap();
    backend.set_execution_mode(ExecutionMode::Parallel);
    backend.set_parallel_width(1).unwrap();
    let batched = backend.decide_batch(&requests).unwrap();
    assert_eq!(batched.len(), requests.len());
    for (a, b) in serial.iter().zip(&batched) {
        assert_same(a, b);
    }
    backend.set_parallel_width(4).unwrap();
    let before = backend.decide_batch(&requests).unwrap();
    assert_eq!(
        before.iter().map(|r| r.results.len()).collect::<Vec<_>>(),
        vec![2, 1, 3]
    );
    let mut invalid = requests.clone();
    invalid[2].decisions[2].instruction = "overlong ".repeat(4000);
    assert!(backend.decide_batch(&invalid).is_err());
    invalid[1].decisions.clear();
    assert!(backend.decide_batch(&invalid).is_err());
    let after = backend.decide_batch(&requests).unwrap();
    for (a, b) in before.iter().zip(&after) {
        assert_same(a, b);
    }
    assert!(backend.decide_batch(&[]).unwrap().is_empty());
}

#[test]
#[ignore = "requires SKID_MODEL; measures native KV reservation and checks context mode switching"]
fn real_model_dynamic_parallel_context() {
    let mut backend = load();
    backend.set_execution_mode(ExecutionMode::Parallel);
    backend.set_parallel_width(8).unwrap();
    backend.set_parallel_context_dynamic(true);
    let small_request = request(8, false);
    let compact = backend.decide(&small_request).unwrap();
    let allocated = compact.backend.parallel_context_tokens.unwrap() as usize;
    let submitted: usize = compact
        .results
        .iter()
        .map(|result| result.input_tokens)
        .sum();
    eprintln!(
        "dynamic context: allocated={allocated}, input_sum={submitted}, legacy={}",
        2048 * 8
    );
    assert!(allocated >= submitted);
    assert!(
        allocated < 2048 * 8,
        "short prompts should avoid the full KV reservation"
    );
    assert!(compact.backend.parallel_context_dynamic);
    assert_same(&compact, &backend.decide(&small_request).unwrap());

    let mut invalid = small_request.clone();
    invalid.decisions[7].instruction = "overlong ".repeat(4000);
    assert!(backend.decide(&invalid).is_err());
    assert_same(&compact, &backend.decide(&small_request).unwrap());

    let larger = request(8, true);
    let grown = backend.decide(&larger).unwrap();
    let grown_capacity = grown.backend.parallel_context_tokens.unwrap();
    assert!(grown_capacity as usize > allocated);
    let retained = backend.decide(&small_request).unwrap();
    assert_eq!(
        retained.backend.parallel_context_tokens,
        Some(grown_capacity)
    );
    assert_same(&retained, &backend.decide(&small_request).unwrap());

    backend.set_parallel_context_dynamic(false);
    let full = backend.decide(&small_request).unwrap();
    assert!(!full.backend.parallel_context_dynamic);
    assert_eq!(full.backend.parallel_context_tokens, None);
    assert_eq!(full.results.len(), compact.results.len());
    for (a, b) in full.results.iter().zip(compact.results.iter()) {
        assert_eq!((&a.id, a.input_tokens), (&b.id, b.input_tokens));
    }
}

#[test]
#[ignore = "requires SKID_MODEL; checks 8 independent requests in one dynamically sized wave"]
fn real_model_dynamic_request_batch() {
    let mut backend = load();
    backend.set_execution_mode(ExecutionMode::Parallel);
    backend.set_parallel_width(24).unwrap();
    backend.set_parallel_context_dynamic(true);
    let requests: Vec<_> = (0..8)
        .map(|i| {
            let mut request = request(3, false);
            request.state["case"] = serde_json::json!(i);
            request
        })
        .collect();
    let responses = backend.decide_batch(&requests).unwrap();
    assert_eq!(responses.len(), requests.len());
    let submitted: usize = responses
        .iter()
        .flat_map(|response| response.results.iter())
        .map(|result| result.input_tokens)
        .sum();
    let allocated = responses[0].backend.parallel_context_tokens.unwrap() as usize;
    eprintln!(
        "dynamic request batch: allocated={allocated}, input_sum={submitted}, legacy={}",
        2048 * 24
    );
    assert!(allocated >= submitted);
    assert!(allocated < 2048 * 24);
    for (request, response) in requests.iter().zip(&responses) {
        assert_eq!(response.results.len(), request.decisions.len());
        assert_eq!(
            response.backend.parallel_context_tokens,
            Some(allocated as u32)
        );
        for (decision, result) in request.decisions.iter().zip(&response.results) {
            assert_eq!(result.id, decision.id);
        }
    }
}

#[test]
#[ignore = "requires SKID_MODEL; reports speed and score drift, not an accuracy pass claim"]
fn real_model_parallel_measurement() {
    let mut backend = load();
    let mut samples = Vec::new();
    let mut comparisons = Vec::new();
    for long in [false, true] {
        for count in [1, 4, 16, 32] {
            let request = request(count, long);
            let modes = if count == 4 || count == 32 {
                [
                    ExecutionMode::Parallel,
                    ExecutionMode::PrefixReuse,
                    ExecutionMode::Fresh,
                ]
            } else {
                [
                    ExecutionMode::Fresh,
                    ExecutionMode::PrefixReuse,
                    ExecutionMode::Parallel,
                ]
            };
            let mut responses = Vec::new();
            for mode in modes {
                backend.set_execution_mode(mode);
                backend.decide(&request).unwrap(); // includes lazy context allocation; not timed.
                for repetition in 0..3 {
                    let start = Instant::now();
                    let response = backend.decide(&request).unwrap();
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                    assert_eq!(response.results.len(), count);
                    samples.push(serde_json::json!({"long":long,"questions":count,"mode":mode,"repetition":repetition,"elapsed_ms":elapsed_ms,"response":response}));
                    if repetition == 0 {
                        responses.push((mode, response));
                    }
                }
            }
            let fresh = &responses
                .iter()
                .find(|(m, _)| *m == ExecutionMode::Fresh)
                .unwrap()
                .1;
            for (mode, response) in &responses {
                let mut probability_delta = 0.0_f64;
                let mut mass_delta = 0.0_f64;
                let mut changed_top1 = 0;
                let mut changed_selected = 0;
                for (a, b) in fresh.results.iter().zip(&response.results) {
                    assert_eq!((&a.id, a.input_tokens), (&b.id, b.input_tokens));
                    mass_delta = mass_delta.max((a.candidate_mass - b.candidate_mass).abs());
                    changed_top1 += usize::from(top(a) != top(b));
                    changed_selected += usize::from(selected(a) != selected(b));
                    for (x, y) in a.scores.iter().zip(&b.scores) {
                        assert_eq!((&x.id, x.token_id), (&y.id, y.token_id));
                        probability_delta = probability_delta
                            .max((x.option_probability - y.option_probability).abs());
                    }
                }
                comparisons.push(serde_json::json!({"long":long,"questions":count,"mode":mode,"max_probability_delta":probability_delta,"max_mass_delta":mass_delta,"changed_top1":changed_top1,"changed_selected":changed_selected,"within_existing_equivalence_tolerance":probability_delta<0.02&&mass_delta<0.02&&changed_top1==0&&changed_selected==0}));
            }
            println!("Measured long={long} questions={count}");
        }
    }
    let report = serde_json::json!({"samples":samples,"comparisons":comparisons,"scope":"synthetic warehouse state; repeated questions; three timed runs after warmup per mode; state-first v2; context 2048 per question, batch 256, threads 4; does not measure general accuracy or Jev parity"});
    if let Ok(path) = env::var("SKID_PARALLEL_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    for comparison in comparisons {
        println!("{comparison}");
    }
}

#[test]
#[ignore = "requires SKID_MODEL; compares labeled fixture results across execution modes"]
fn real_model_parallel_labeled() {
    let mut backend = load();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/decision_benchmark.json")).unwrap();
    let mut runs = Vec::new();
    for mode in [
        ExecutionMode::Fresh,
        ExecutionMode::PrefixReuse,
        ExecutionMode::Parallel,
    ] {
        backend.set_execution_mode(mode);
        let mut correct = 0;
        let mut wrong = 0;
        let mut abstained = 0;
        let mut top1_correct = 0;
        let mut samples = Vec::new();
        for case in fixture["cases"].as_array().unwrap() {
            let request: DecisionRequest = serde_json::from_value(case["request"].clone()).unwrap();
            let response = backend.decide(&request).unwrap();
            for r in &response.results {
                let expected = case["expected"][&r.id].as_str().unwrap();
                top1_correct += usize::from(top(r) == expected);
                let answer = selected(r);
                if answer.is_null() {
                    abstained += 1;
                } else {
                    let answer = answer
                        .as_str()
                        .map(str::to_owned)
                        .unwrap_or_else(|| answer.to_string());
                    if answer == expected {
                        correct += 1;
                    } else {
                        wrong += 1;
                    }
                }
            }
            samples.push(serde_json::json!({"case":case["id"],"response":response}));
        }
        assert_eq!(correct + wrong + abstained, 36);
        runs.push(serde_json::json!({"mode":mode,"correct":correct,"wrong":wrong,"abstained":abstained,"raw_top1_correct":top1_correct,"samples":samples}));
        println!(
            "{mode:?}: correct={correct} wrong={wrong} abstained={abstained} raw_top1_correct={top1_correct}"
        );
    }
    if let Ok(path) = env::var("SKID_PARALLEL_OUTPUT") {
        std::fs::write(path,serde_json::to_vec_pretty(&serde_json::json!({"runs":runs,"scope":"36 synthetic labeled decisions, state-first v2; not held-out production accuracy"})).unwrap()).unwrap();
    }
}
