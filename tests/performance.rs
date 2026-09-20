#![cfg(feature = "llama")]
//! Opt-in measurements of a local checkpoint; no downloads or model distribution.
use skid_desion::{llama::LlamaBackend, *};
use std::{env, fs, path::Path, time::Instant};

fn setting(name: &str, default: u32, minimum: u32) -> u32 {
    let value = match env::var(name) {
        Ok(value) => value
            .parse::<u32>()
            .unwrap_or_else(|_| panic!("{name} must be an integer")),
        Err(env::VarError::NotPresent) => default,
        Err(error) => panic!("cannot read {name}: {error}"),
    };
    assert!(value >= minimum, "{name} must be at least {minimum}");
    value
}

/// Nearest-rank percentile. Input must be sorted and nonempty.
fn percentile(sorted: &[f64], percent: usize) -> f64 {
    sorted[(sorted.len() * percent).div_ceil(100).saturating_sub(1)]
}

#[test]
fn latency_percentiles_use_nearest_rank() {
    assert_eq!(percentile(&[7.0], 95), 7.0);
    assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 50), 2.0);
    assert_eq!(percentile(&[1.0, 2.0, 3.0, 4.0], 95), 4.0);
}

#[test]
#[ignore = "requires SKID_MODEL; measures real local inference without a timing pass threshold"]
fn model_inference_performance() {
    let model = env::var("SKID_MODEL").expect("set SKID_MODEL to an existing local chat GGUF");
    let cuda = match env::var("SKID_CUDA").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(env::VarError::NotPresent) => false,
        _ => panic!("SKID_CUDA must be 0 or 1"),
    };
    let iterations = setting("SKID_PERF_ITERATIONS", 5, 1);
    let warmup = setting("SKID_PERF_WARMUP", 1, 0);
    let context = setting("SKID_CONTEXT", 2048, 1);
    let batch = setting("SKID_BATCH", 256, 1);
    let threads = i32::try_from(setting("SKID_THREADS", 4, 1)).expect("SKID_THREADS exceeds i32");
    let request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    request.validate().unwrap();
    let policy = DecisionPolicy::default();
    let started = Instant::now();
    let mut backend = LlamaBackend::load(
        Path::new(&model),
        context,
        batch,
        threads,
        cuda,
        policy.clone(),
    )
    .expect("failed to load the local model on the requested device");
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;

    // First request and additional warmups are excluded from measured samples.
    // Files may already be in the OS page cache, so this is not a cold-disk test.
    let started = Instant::now();
    let first = backend.decide(&request).expect("first inference failed");
    let first_request_ms = started.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(first.results.len(), request.decisions.len());
    for _ in 0..warmup {
        backend.decide(&request).expect("warmup inference failed");
    }

    let mut latency_ms = Vec::new();
    let mut input_tokens = 0usize;
    let mut abstentions = 0usize;
    let mut last = first;
    for _ in 0..iterations {
        let started = Instant::now();
        let response = backend.decide(&request).expect("measured inference failed");
        let elapsed = started.elapsed().as_secs_f64();
        assert!(elapsed > 0.0 && elapsed.is_finite());
        assert_eq!(response.results.len(), request.decisions.len());
        for (result, decision) in response.results.iter().zip(&request.decisions) {
            assert_eq!(result.id, decision.id);
            assert_eq!(result.scores.len(), decision.options().len());
            assert!(!result.truncated);
            assert!(result.input_tokens > 0);
            assert!(result.scores.iter().all(|s| s.raw_logit.is_finite()));
            input_tokens += result.input_tokens;
            abstentions += usize::from(!result.abstention_reasons.is_empty());
        }
        latency_ms.push(elapsed * 1000.0);
        last = response;
    }
    let total_ms = latency_ms.iter().sum::<f64>();
    let total_seconds = total_ms / 1000.0;
    let decisions = u64::from(iterations) * request.decisions.len() as u64;
    let mut sorted = latency_ms.clone();
    sorted.sort_by(f64::total_cmp);
    let cpu_model = fs::read_to_string("/proc/cpuinfo").ok().and_then(|text| {
        text.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "model name").then(|| value.trim().to_owned())
        })
    });
    let report = serde_json::json!({
        "schema_version": 1,
        "workload": "warehouse-v1",
        "measurement": "sequential end-to-end decide calls; prefill and scoring; no text generation",
        "crate_version": env!("CARGO_PKG_VERSION"),
        "build_profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        "host": {"os": env::consts::OS, "arch": env::consts::ARCH, "cpu_model": cpu_model},
        "backend": last.backend,
        "model_bytes": fs::metadata(&model).unwrap().len(),
        "context": context, "batch": batch, "threads": threads,
        "policy": policy,
        "iterations": iterations, "additional_warmup_requests": warmup,
        "decisions_per_request": request.decisions.len(),
        "load_ms": load_ms, "first_request_ms": first_request_ms,
        "latency_ms": {
            "samples": latency_ms, "mean": total_ms / f64::from(iterations),
            "p50": percentile(&sorted, 50), "p95": percentile(&sorted, 95),
            "min": sorted[0], "max": sorted[sorted.len()-1]
        },
        "measured_inference_seconds": total_seconds,
        "requests_per_second": f64::from(iterations) / total_seconds,
        "decisions_per_second": decisions as f64 / total_seconds,
        "input_tokens": input_tokens,
        "input_tokens_per_second": input_tokens as f64 / total_seconds,
        "abstained_decisions": abstentions,
        "last_results": last.results,
    });
    if let Ok(path) = env::var("SKID_PERF_OUTPUT") {
        let path = Path::new(&path);
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent).expect("cannot create performance report directory");
        }
        fs::write(path, serde_json::to_vec_pretty(&report).unwrap())
            .expect("cannot write performance report");
    }
    println!("SKID_PERF_REPORT={report}");
}
