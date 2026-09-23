use l2s1::*;

#[test]
fn compute_options_reject_invalid_bounds() {
    let good = ComputeOptions {
        context: 2048,
        batch: 1024,
        ubatch: 256,
        threads: 4,
        flash_attention: FlashAttention::On,
    };
    assert!(good.validate().is_ok());
    for invalid in [
        ComputeOptions { ubatch: 0, ..good },
        ComputeOptions {
            ubatch: 1025,
            ..good
        },
        ComputeOptions { batch: 0, ..good },
        ComputeOptions {
            batch: 4096,
            ..good
        },
        ComputeOptions {
            context: u32::MAX,
            ..good
        },
        ComputeOptions { threads: 0, ..good },
    ] {
        assert!(invalid.validate().is_err());
    }
}

#[test]
fn streaming_normalizer_matches_allocating_f64_reference() {
    let request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    let decision = &request.decisions[0];
    let n = decision.options().len();
    let candidates: Vec<i32> = (0..n as i32).collect();
    for size in [32, 262_144] {
        for offset in [-1000.0f32, 0.0, 1000.0] {
            let mut logits: Vec<f32> = (0..size)
                .map(|i| offset + ((i * 17 % 97) as f32 - 48.0) * 0.25)
                .collect();
            logits[size - 1] = f32::NEG_INFINITY;
            let lse = |values: Vec<f64>| {
                let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
            };
            let full = lse(logits.iter().map(|&v| v as f64).collect());
            let selected = lse(logits[..n].iter().map(|&v| v as f64).collect());
            let result = score_logits(
                decision,
                &logits,
                &candidates,
                20,
                &DecisionPolicy::default(),
            )
            .unwrap();
            assert_eq!(
                result.candidate_mass,
                (selected - full).exp().clamp(0.0, 1.0)
            );
            for (score, &z) in result.scores.iter().zip(&logits[..n]) {
                assert_eq!(score.option_probability, (z as f64 - selected).exp());
            }
        }
    }
}

#[cfg(feature = "llama")]
#[test]
#[ignore = "requires SKID_MODEL and optionally SKID_CUDA, SKID_BATCH, SKID_UBATCH, SKID_FA"]
fn real_model_tuned_context_survives_width_changes() {
    use l2s1::llama::LlamaBackend;
    let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
    let batch = std::env::var("SKID_BATCH")
        .map(|v| v.parse().unwrap())
        .unwrap_or(1024);
    let ubatch = std::env::var("SKID_UBATCH")
        .map(|v| v.parse().unwrap())
        .unwrap_or(batch);
    let flash_attention = if std::env::var("SKID_FA").as_deref() == Ok("1") {
        FlashAttention::On
    } else {
        FlashAttention::Off
    };
    let mut backend = LlamaBackend::load_with_options(
        model.as_ref(),
        ComputeOptions {
            context: 2048,
            batch,
            ubatch,
            threads: 4,
            flash_attention,
        },
        std::env::var("SKID_CUDA").as_deref() == Ok("1"),
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )
    .unwrap();
    let request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    let fresh = backend.decide(&request).unwrap();
    let compute = fresh.backend.compute.unwrap();
    assert_eq!(compute.ubatch, ubatch);
    assert_eq!(compute.flash_attention, flash_attention);
    let values = |r: &DecisionResponse| {
        r.results
            .iter()
            .flat_map(|d| d.scores.iter().map(|s| s.raw_logit))
            .collect::<Vec<_>>()
    };
    backend.set_parallel_width(1).unwrap();
    backend.set_execution_mode(ExecutionMode::Parallel);
    assert_eq!(values(&fresh), values(&backend.decide(&request).unwrap()));
    backend.set_parallel_width(4).unwrap();
    let first = backend.decide(&request).unwrap();
    let mut bad = request.clone();
    bad.state = serde_json::json!({"text": "too long ".repeat(10000)});
    assert!(backend.decide(&bad).is_err());
    assert_eq!(values(&first), values(&backend.decide(&request).unwrap()));
    backend.set_execution_mode(ExecutionMode::Fresh);
    assert_eq!(values(&fresh), values(&backend.decide(&request).unwrap()));
}
