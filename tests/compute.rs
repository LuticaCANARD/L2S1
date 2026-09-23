use l2s1::*;

#[cfg(feature = "llama")]
#[test]
#[ignore = "requires SKID_MODEL; optionally SKID_CUDA=1 and SKID_CPU_MOE_LAYERS"]
fn read_loading_preserves_decisions_and_cached_evidence() {
    use l2s1::llama::LlamaBackend;
    let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let mut options: ComputeOptions = serde_json::from_value(serde_json::json!({
        "context":8192,"batch":256,"ubatch":256,"threads":8,"flash_attention":"off"
    }))
    .unwrap();
    options.cpu_moe_layers = std::env::var("SKID_CPU_MOE_LAYERS")
        .map(|s| s.parse().unwrap())
        .unwrap_or(0);
    let mut request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    request.decisions.truncate(1); // Repeat one exact prompt within the session.
    request.state["history"] =
        serde_json::json!("The package was scanned at the depot. ".repeat(100));
    let mut auto = LlamaBackend::load_with_options(
        model.as_ref(),
        options,
        cuda,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )
    .unwrap();
    let baseline = auto.decide(&request).unwrap();
    let identity = auto.identity();
    drop(auto); // Never hold both copies of a large model at once.
    options.model_load_mode = ModelLoadMode::Read;
    let mut read = LlamaBackend::load_with_options(
        model.as_ref(),
        options,
        cuda,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )
    .unwrap();
    assert_ne!(identity, read.identity());
    assert_eq!(read.identity().compute.model_load_mode, ModelLoadMode::Read);
    read.set_preparation_cache(PreparationCacheConfig {
        max_entries: 32,
        max_bytes: 1024 * 1024,
    });
    let fresh = read.decide(&request).unwrap();
    assert_eq!(
        serde_json::to_value(&baseline.results).unwrap(),
        serde_json::to_value(&fresh.results).unwrap()
    );
    let repeat = read.decide(&request).unwrap();
    assert_eq!(
        serde_json::to_value(&fresh).unwrap(),
        serde_json::to_value(&repeat).unwrap()
    );
    assert!(read.preparation_cache_stats().prompts.hits > 0);
    read.set_execution_mode(ExecutionMode::PrefixReuse);
    let mut session = read.shared_state(request.state.clone()).unwrap();
    let first = session.decide(request.decisions.clone()).unwrap();
    let cached = session.decide(request.decisions.clone()).unwrap();
    assert!(cached.results.iter().any(|r| r.reused_prefix_tokens > 0));
    let evidence = |response: &DecisionResponse| {
        let mut value = serde_json::to_value(&response.results).unwrap();
        for result in value.as_array_mut().unwrap() {
            result
                .as_object_mut()
                .unwrap()
                .remove("reused_prefix_tokens");
            result
                .as_object_mut()
                .unwrap()
                .remove("code_evaluated_tokens");
        }
        value
    };
    assert_eq!(evidence(&first), evidence(&cached));
    assert_eq!(evidence(&fresh), evidence(&cached));
}

#[test]
fn compute_options_reject_invalid_bounds() {
    let good = ComputeOptions {
        context: 2048,
        batch: 1024,
        ubatch: 256,
        threads: 4,
        flash_attention: FlashAttention::On,
        gpu_layers: None,
        cpu_moe_layers: 0,
        model_load_mode: l2s1::ModelLoadMode::Auto,
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
            gpu_layers: None,
            cpu_moe_layers: 0,
            model_load_mode: l2s1::ModelLoadMode::Auto,
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

#[test]
fn placement_defaults_preserve_legacy_json_and_validate_device() {
    let legacy = serde_json::json!({"context":8192,"batch":256,"ubatch":256,"threads":4,"flash_attention":"off"});
    let defaults: ComputeOptions = serde_json::from_value(legacy.clone()).unwrap();
    assert_eq!(defaults.gpu_layers, None);
    assert_eq!(defaults.cpu_moe_layers, 0);
    assert_eq!(defaults.model_load_mode, ModelLoadMode::Auto);
    assert_eq!(serde_json::to_value(defaults).unwrap(), legacy);
    let read = ComputeOptions {
        model_load_mode: ModelLoadMode::Read,
        ..defaults
    };
    assert_ne!(read, defaults);
    assert_eq!(
        serde_json::to_value(read).unwrap()["model_load_mode"],
        "read"
    );
    assert_eq!(
        serde_json::from_value::<ComputeOptions>(serde_json::to_value(read).unwrap()).unwrap(),
        read
    );
    let mut invalid_mode = legacy.clone();
    invalid_mode["model_load_mode"] = serde_json::json!("unknown");
    assert!(serde_json::from_value::<ComputeOptions>(invalid_mode).is_err());
    assert!(defaults.validate_device(false).is_ok());
    let split = ComputeOptions {
        gpu_layers: Some(18),
        cpu_moe_layers: 14,
        ..defaults
    };
    assert!(split.validate_device(true).is_ok());
    assert!(split.validate_device(false).is_err());
    assert_ne!(split, defaults);
    assert_eq!(
        serde_json::from_value::<ComputeOptions>(serde_json::to_value(split).unwrap()).unwrap(),
        split
    );
    for invalid in [
        ComputeOptions {
            gpu_layers: Some(u32::MAX),
            ..defaults
        },
        ComputeOptions {
            cpu_moe_layers: u32::MAX,
            ..defaults
        },
    ] {
        assert!(invalid.validate().is_err());
    }
    assert!(
        ComputeOptions {
            gpu_layers: Some(0),
            ..defaults
        }
        .validate_device(false)
        .is_ok()
    );
}

#[cfg(feature = "llama")]
#[test]
#[ignore = "requires SKID_MODEL and CUDA; records placement drift separately"]
fn partial_gpu_placement_preserves_cache_and_artifact_identity() {
    use l2s1::llama::LlamaBackend;
    let selection = |result: &DecisionResult| match &result.value {
        DecisionValue::Binary { value, .. } => serde_json::to_value(value).unwrap(),
        DecisionValue::Choice { selected } | DecisionValue::Ordinal { selected, .. } => {
            serde_json::to_value(selected).unwrap()
        }
    };
    let model = std::env::var("SKID_MODEL").unwrap();
    let request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    let defaults: ComputeOptions = serde_json::from_value(serde_json::json!({"context":2048,"batch":256,"ubatch":256,"threads":4,"flash_attention":"off"})).unwrap();
    let mut backend = LlamaBackend::load_with_options(
        model.as_ref(),
        defaults,
        true,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )
    .unwrap();
    let original_identity = backend.identity();
    let base = backend.decide(&request).unwrap();
    drop(backend);
    for layers in [0, 4] {
        let options = ComputeOptions {
            gpu_layers: Some(layers),
            ..defaults
        };
        let mut backend = LlamaBackend::load_with_options(
            model.as_ref(),
            options,
            true,
            DecisionPolicy::default(),
            PromptProfile::Auto,
        )
        .unwrap();
        assert_ne!(backend.identity(), original_identity);
        assert_eq!(backend.identity().compute, options);
        backend.set_preparation_cache(PreparationCacheConfig {
            max_entries: 32,
            max_bytes: 1024 * 1024,
        });
        let first = backend.decide(&request).unwrap();
        let repeat = backend.decide(&request).unwrap();
        assert_eq!(
            serde_json::to_value(&first).unwrap(),
            serde_json::to_value(&repeat).unwrap()
        );
        assert!(backend.preparation_cache_stats().prompts.hits > 0);
        let max_delta = first
            .results
            .iter()
            .zip(&base.results)
            .flat_map(|(a, b)| {
                a.scores
                    .iter()
                    .zip(&b.scores)
                    .map(|(x, y)| (x.option_probability - y.option_probability).abs())
            })
            .fold(0.0, f64::max);
        // CPU and CUDA quantized kernels are distinct compute profiles. Keep
        // the existing 0.02 criterion visible rather than claiming equivalence
        // or increasing its tolerance. Cache equivalence is tested above and
        // below within each placement profile.
        let selection_changes = first
            .results
            .iter()
            .zip(&base.results)
            .filter(|(a, b)| selection(a) != selection(b))
            .count();
        let max_mass_delta = first
            .results
            .iter()
            .zip(&base.results)
            .map(|(a, b)| (a.candidate_mass - b.candidate_mass).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "gpu_layers={layers}, full_gpu_probability_delta={max_delta}, full_gpu_mass_delta={max_mass_delta}, selection_changes={selection_changes}, full_gpu_equivalent={}",
            max_delta < 0.02 && max_mass_delta < 0.02 && selection_changes == 0
        );
        backend.set_execution_mode(ExecutionMode::PrefixReuse);
        let mut session = backend.shared_state(request.state.clone()).unwrap();
        session.decide(request.decisions.clone()).unwrap();
        let hot = session.decide(request.decisions.clone()).unwrap();
        for (a, b) in hot.results.iter().zip(&first.results) {
            assert_eq!(selection(a), selection(b));
            for (x, y) in a.scores.iter().zip(&b.scores) {
                assert!((x.option_probability - y.option_probability).abs() < 0.02);
            }
        }
    }
    // Reject incompatible placement before trying to open a model file.
    assert!(
        LlamaBackend::load_with_options(
            std::path::Path::new("missing.gguf"),
            ComputeOptions {
                gpu_layers: Some(1),
                ..defaults
            },
            false,
            DecisionPolicy::default(),
            PromptProfile::Auto
        )
        .err()
        .unwrap()
        .to_string()
        .contains("CUDA")
    );
}
