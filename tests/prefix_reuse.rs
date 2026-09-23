#![cfg(feature = "llama")]
//! Model-dependent equivalence and timing: run explicitly against local GGUFs.
use l2s1::{llama::LlamaBackend, *};
use std::{env, time::Instant};

fn signature(result: &DecisionResult) -> serde_json::Value {
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
#[ignore = "requires SKID_MODEL; compares fresh and reused prefixes on real inference"]
fn real_model_prefix_reuse() {
    let model = env::var("SKID_MODEL").expect("set SKID_MODEL");
    let cuda = env::var("SKID_CUDA").as_deref() == Ok("1");
    let batch = env::var("SKID_BATCH")
        .map(|s| {
            s.parse::<u32>()
                .expect("SKID_BATCH must be a positive integer")
        })
        .unwrap_or(256);
    assert!(batch > 0);
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        2048,
        batch,
        4,
        cuda,
        DecisionPolicy::default(),
    )
    .unwrap();
    backend.set_prompt_layout(PromptLayout::StateFirst);
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/decision_benchmark.json")).unwrap();
    let mut requests: Vec<DecisionRequest> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| serde_json::from_value(case["request"].clone()).unwrap())
        .collect();
    let mut long = requests[0].clone();
    long.state["irrelevant_history"] =
        serde_json::json!("The package was scanned at the depot. ".repeat(80));
    let originals = long.decisions.clone();
    // Includes exact duplicate prompts (only IDs differ), shorter suffixes, and
    // longer suffixes. All must regenerate final logits after trimming old KV.
    for index in 0..3 {
        let mut d = originals[index % originals.len()].clone();
        d.id = format!("repeat-{index}");
        long.decisions.push(d);
    }
    requests.push(long);
    let mut identical = requests[0].clone();
    identical.state = requests.last().unwrap().state.clone();
    identical.decisions = (0..3)
        .map(|i| {
            let mut d = originals[0].clone();
            d.id = format!("identical-{i}");
            d
        })
        .collect();
    requests.push(identical);
    // Warm up both paths without including their timing.
    for mode in [ExecutionMode::Fresh, ExecutionMode::PrefixReuse] {
        backend.set_execution_mode(mode);
        backend.decide(&requests[0]).unwrap();
    }
    let mut comparisons = 0;
    let mut changed_top1 = 0;
    let mut changed_selected = 0;
    let mut max_probability_delta = 0.0_f64;
    let mut max_mass_delta = 0.0_f64;
    let mut samples = Vec::new();
    let mut reused_total = 0;
    for (index, request) in requests.iter().enumerate() {
        // Alternate order to reduce timing bias from always measuring reuse second.
        let modes = if index % 2 == 0 {
            [ExecutionMode::Fresh, ExecutionMode::PrefixReuse]
        } else {
            [ExecutionMode::PrefixReuse, ExecutionMode::Fresh]
        };
        let mut pair = Vec::new();
        for mode in modes {
            backend.set_execution_mode(mode);
            let start = Instant::now();
            let response = backend.decide(request).unwrap();
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(response.backend.execution_mode, mode);
            assert_eq!(response.backend.prompt_layout, PromptLayout::StateFirst);
            assert!(response.backend.prompt_version.ends_with("state-first-v2"));
            assert_eq!(response.results[0].reused_prefix_tokens, 0);
            for r in &response.results {
                assert!(r.reused_prefix_tokens < r.input_tokens);
                if mode == ExecutionMode::Fresh {
                    assert_eq!(r.reused_prefix_tokens, 0);
                }
            }
            samples.push(serde_json::json!({"case":index,"mode":mode,"elapsed_ms":elapsed_ms,"results":response.results}));
            pair.push(response);
        }
        if modes[0] != ExecutionMode::Fresh {
            pair.swap(0, 1);
        }
        for (fresh, reused) in pair[0].results.iter().zip(&pair[1].results) {
            comparisons += 1;
            changed_top1 += usize::from(top(fresh) != top(reused));
            changed_selected += usize::from(signature(fresh) != signature(reused));
            max_mass_delta =
                max_mass_delta.max((fresh.candidate_mass - reused.candidate_mass).abs());
            reused_total += reused.reused_prefix_tokens;
            assert_eq!(fresh.input_tokens, reused.input_tokens);
            for (a, b) in fresh.scores.iter().zip(&reused.scores) {
                assert_eq!((&a.id, a.token_id), (&b.id, b.token_id));
                max_probability_delta =
                    max_probability_delta.max((a.option_probability - b.option_probability).abs());
            }
        }
    }
    // Failure after one successful branch must not leave stale state for the next request.
    backend.set_execution_mode(ExecutionMode::PrefixReuse);
    let before = backend.decide(&requests[0]).unwrap();
    backend.decide(&requests[1]).unwrap();
    let mut bad = requests[0].clone();
    bad.decisions[1].instruction = "overlong ".repeat(4000);
    assert!(backend.decide(&bad).is_err());
    let mut invalid = requests[0].clone();
    invalid.decisions.clear();
    assert!(backend.decide(&invalid).is_err());
    let after = backend.decide(&requests[0]).unwrap();
    assert_eq!(after.results[0].reused_prefix_tokens, 0);
    for (a, b) in before.results.iter().zip(&after.results) {
        for (x, y) in a.scores.iter().zip(&b.scores) {
            assert!(
                (x.raw_logit - y.raw_logit).abs() < 1e-3,
                "request/error isolation failed"
            );
        }
    }
    let report = serde_json::json!({
        "backend":after.backend,"comparisons":comparisons,"changed_top1":changed_top1,
        "changed_selected":changed_selected,"max_probability_delta":max_probability_delta,
        "max_candidate_mass_delta":max_mass_delta,"reused_prefix_tokens":reused_total,
        "context":2048,"batch":batch,"threads":4,"samples":samples,
        "scope":"12 synthetic labeled requests plus long-state and identical-prompt cases; sequential request-local reuse; no production accuracy claim"
    });
    if let Ok(path) = env::var("SKID_REUSE_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
    println!(
        "comparisons={comparisons} changed_top1={changed_top1} changed_selected={changed_selected} max_probability_delta={max_probability_delta} reused_tokens={reused_total}"
    );
    // Preserve the existing native-suite probability tolerance. Do not hide a
    // failed model/device comparison by increasing this threshold.
    assert!(
        max_probability_delta < 0.02,
        "prefix reuse exceeds probability tolerance"
    );
    assert!(
        max_mass_delta < 0.02,
        "prefix reuse exceeds candidate mass tolerance"
    );
    assert_eq!(changed_top1, 0, "prefix reuse changed raw top-1 decisions");
    assert_eq!(
        changed_selected, 0,
        "prefix reuse changed accepted decisions"
    );
    // The dense models in the fixture should actually exercise the cache path.
    if matches!(
        before.backend.model_architecture.as_str(),
        "qwen3" | "llama"
    ) {
        assert!(reused_total > 0);
    }
}
