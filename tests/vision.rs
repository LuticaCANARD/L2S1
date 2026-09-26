#![cfg(feature = "llama")]
use l2s1::{DecisionPolicy, DecisionRequest, DecisionResponse, ExecutionMode, llama::LlamaBackend};

fn assert_vision_equivalent(serial: &DecisionResponse, batched: &DecisionResponse) {
    assert_eq!(serial.results.len(), batched.results.len());
    for (a, b) in serial.results.iter().zip(&batched.results) {
        assert_eq!((&a.id, a.input_tokens), (&b.id, b.input_tokens));
        assert_eq!(
            serde_json::to_value(&a.value).unwrap(),
            serde_json::to_value(&b.value).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&a.abstention_reasons).unwrap(),
            serde_json::to_value(&b.abstention_reasons).unwrap()
        );
        assert!((a.candidate_mass - b.candidate_mass).abs() < 0.02);
        let top = |result: &l2s1::DecisionResult| {
            result
                .scores
                .iter()
                .max_by(|x, y| x.option_probability.total_cmp(&y.option_probability))
                .unwrap()
                .id
                .clone()
        };
        assert_eq!(top(a), top(b));
        for (x, y) in a.scores.iter().zip(&b.scores) {
            assert_eq!((&x.id, x.token_id), (&y.id, y.token_id));
            assert!((x.option_probability - y.option_probability).abs() < 0.02);
        }
    }
}

fn batch_fixture() -> (LlamaBackend, Vec<DecisionRequest>, [&'static [u8]; 5]) {
    let model = std::env::var("SKID_VISION_MODEL").expect("set SKID_VISION_MODEL");
    let projector = std::env::var("SKID_VISION_MMPROJ").expect("set SKID_VISION_MMPROJ");
    let optimized_compute = std::env::var("SKID_VISION_OPTIMIZED").as_deref() == Ok("1");
    let mut backend = LlamaBackend::load_with_options(
        model.as_ref(),
        if optimized_compute {
            l2s1::ComputeOptions::vision_optimized()
        } else {
            l2s1::ComputeOptions::default()
        },
        std::env::var("SKID_CUDA").as_deref() == Ok("1"),
        DecisionPolicy::default(),
        l2s1::PromptProfile::Auto,
    )
    .unwrap();
    backend.load_vision_projector(projector.as_ref()).unwrap();
    if optimized_compute {
        backend.set_parallel_context_dynamic(true);
        backend.set_vision_projector_reuse(true);
    }
    let red = include_bytes!("fixtures/vision_red_64.png").as_slice();
    let blue = include_bytes!("fixtures/vision_blue_64.png").as_slice();
    // Repeated decision IDs are valid across independent requests. Vary state,
    // image order, prompt length and decision counts to expose regrouping leaks.
    let requests: Vec<DecisionRequest> = (0..5).map(|i| {
        serde_json::from_value(serde_json::json!({
            "state": {"case": i, "reference": if i % 2 == 0 {"red"} else {"blue"}},
            "decisions": (0..if i == 1 {2} else {1}).map(|j| serde_json::json!({
                "id": format!("color{j}"),
                "instruction": if j == 0 {"Which color fills the image?"} else {"Compare the image color to state.reference. Choose whether they match."},
                "kind": {"type": "choice", "options": [
                    {"id":"red", "criterion":"The image is red."},
                    {"id":"blue", "criterion":"The image is blue."}
                ]}
            })).collect::<Vec<_>>()
        })).unwrap()
    }).collect();
    let images = [red, blue, blue, red, blue];
    (backend, requests, images)
}

#[test]
#[ignore = "requires SKID_VISION_MODEL and SKID_VISION_MMPROJ; native image sequence batching"]
fn real_vision_batch_preserves_image_state_order_and_recovers_after_errors() {
    let (mut backend, requests, images) = batch_fixture();
    let serial: Vec<_> = requests
        .iter()
        .zip(&images)
        .map(|(r, i)| backend.decide_vision(r, i).unwrap())
        .collect();
    backend.set_execution_mode(ExecutionMode::Parallel);
    backend.set_parallel_width(4).unwrap();
    let batched = backend.decide_vision_batch(&requests, &images).unwrap();
    let metrics = backend.vision_batch_metrics().unwrap();
    assert!(metrics.projector_encode_calls > 0);
    assert!(metrics.decoder_batch_max_sequences > 1);
    assert_eq!(batched.len(), requests.len());
    for (a, b) in serial.iter().zip(&batched) {
        assert_eq!(a.results.len(), b.results.len());
        for (a, b) in a.results.iter().zip(&b.results) {
            assert_eq!((&a.id, a.input_tokens), (&b.id, b.input_tokens));
        }
    }
    // Changing one image must affect its own logits while leaving the other
    // independent requests intact. The two fixtures have the same dimensions.
    let mut changed_images = images;
    changed_images[0] = images[1];
    let changed = backend
        .decide_vision_batch(&requests, &changed_images)
        .unwrap();
    assert!(
        batched[0].results[0]
            .scores
            .iter()
            .zip(&changed[0].results[0].scores)
            .any(|(a, b)| (a.raw_logit - b.raw_logit).abs() > 1e-4)
    );
    for (a, b) in batched.iter().zip(&changed).skip(1) {
        assert_vision_equivalent(a, b);
        for (a, b) in a.results.iter().zip(&b.results) {
            for (a, b) in a.scores.iter().zip(&b.scores) {
                assert!((a.raw_logit - b.raw_logit).abs() < 1e-3);
            }
        }
    }
    let mut broken = images;
    broken[3] = b"not an image";
    assert!(backend.decide_vision_batch(&requests, &broken).is_err());
    assert!(
        backend
            .decide_vision_batch(&requests, &images[..4])
            .is_err()
    );
    let mut invalid = requests.clone();
    invalid[4].decisions[0].instruction = "overlong input ".repeat(4096);
    assert!(backend.decide_vision_batch(&invalid, &images).is_err());
    for (a, b) in batched
        .iter()
        .zip(backend.decide_vision_batch(&requests, &images).unwrap())
    {
        assert_vision_equivalent(a, &b);
    }
    assert!(backend.decide_vision_batch(&[], &[]).unwrap().is_empty());
    backend.set_parallel_width(1).unwrap();
    for (a, b) in serial
        .iter()
        .zip(backend.decide_vision_batch(&requests, &images).unwrap())
    {
        assert_vision_equivalent(a, &b);
    }
    backend.set_execution_mode(ExecutionMode::Fresh);
    assert_vision_equivalent(
        &serial[0],
        &backend.decide_vision(&requests[0], images[0]).unwrap(),
    );
}

#[test]
#[ignore = "requires vision model/projector; numerical parity is separate from isolation and can fail on CUDA"]
fn real_vision_batch_equivalence_on_color_fixture() {
    let (mut backend, requests, images) = batch_fixture();
    let serial: Vec<_> = requests
        .iter()
        .zip(&images)
        .map(|(request, image)| backend.decide_vision(request, image).unwrap())
        .collect();
    backend.set_execution_mode(ExecutionMode::Parallel);
    backend.set_parallel_width(4).unwrap();
    let batched = backend.decide_vision_batch(&requests, &images).unwrap();
    for (a, b) in serial.iter().zip(&batched) {
        assert_vision_equivalent(a, b);
    }
}

#[test]
#[ignore = "requires real vision GGUF/projector; compact, cache, duplicate-image reuse and cleanup"]
fn real_vision_optimizations_preserve_full_evidence_and_recover() {
    let (mut backend, requests, images) = batch_fixture();
    let mut requests: Vec<_> = (0..4)
        .map(|i| {
            let mut request = requests[0].clone();
            request.state = serde_json::json!({"case": i});
            request
        })
        .collect();
    // Different candidate counts exercise compact output offsets independently
    // of per-sequence vocabulary rows.
    if let l2s1::DecisionKind::Choice { options } = &mut requests[3].decisions[0].kind {
        options.push(l2s1::OptionSpec {
            id: "other".into(),
            criterion: "Neither red nor blue.".into(),
        });
    }
    let repeated_images = [images[0]; 4];
    backend.enable_vision_optimizations().unwrap();
    let compact = backend
        .decide_vision_batch(&requests, &repeated_images)
        .unwrap();
    let metrics = backend.vision_batch_metrics().unwrap();
    // Some projectors create a global image plus tiles. Reuse each matching
    // chunk, without assuming that one input picture means one encoder chunk.
    assert!(metrics.projector_encode_calls > 0);
    assert!(metrics.projector_reused_chunks >= 3);
    assert_eq!(
        metrics.kv_clear_calls, 1,
        "one physical clear per successful wave"
    );
    assert!(metrics.kv_clear_skipped > 0);
    assert!(metrics.decoder_batch_max_sequences > 1);
    let first_cache = backend.preparation_cache_stats();
    let repeated = backend
        .decide_vision_batch(&requests, &repeated_images)
        .unwrap();
    assert!(backend.preparation_cache_stats().vision.hits >= first_cache.vision.hits + 4);
    for (a, b) in compact.iter().zip(&repeated) {
        assert_vision_equivalent(a, b);
    }

    backend
        .set_evidence_transfer(l2s1::EvidenceTransfer::Full)
        .unwrap();
    let full = backend
        .decide_vision_batch(&requests, &repeated_images)
        .unwrap();
    for (a, b) in compact.iter().zip(&full) {
        assert_vision_equivalent(a, b);
        for (a, b) in a.results.iter().zip(&b.results) {
            assert!((a.candidate_mass - b.candidate_mass).abs() < 1e-12);
            for (a, b) in a.scores.iter().zip(&b.scores) {
                assert_eq!(a.raw_logit, b.raw_logit);
                assert!((a.option_probability - b.option_probability).abs() < 1e-12);
            }
        }
    }
    let mut invalid = repeated_images;
    invalid[2] = b"invalid image";
    assert!(backend.decide_vision_batch(&requests, &invalid).is_err());
    let mut overlong = requests.clone();
    overlong[0].decisions[0].instruction = "context overflow ".repeat(4096);
    assert!(
        backend
            .decide_vision_batch(&overlong, &repeated_images)
            .is_err()
    );
    for (a, b) in full.iter().zip(
        backend
            .decide_vision_batch(&requests, &repeated_images)
            .unwrap(),
    ) {
        assert_vision_equivalent(a, &b);
    }
    backend.set_code_rotation(1).unwrap();
    assert_eq!(backend.preparation_cache_stats().vision.entries, 0);
    backend.set_code_rotation(0).unwrap();
    backend.set_execution_mode(ExecutionMode::Fresh);
    backend
        .set_evidence_transfer(l2s1::EvidenceTransfer::Full)
        .unwrap();
    let fresh_full = backend.decide_vision(&requests[0], images[0]).unwrap();
    backend
        .set_evidence_transfer(l2s1::EvidenceTransfer::Compact)
        .unwrap();
    let fresh_compact = backend.decide_vision(&requests[0], images[0]).unwrap();
    assert_vision_equivalent(&fresh_full, &fresh_compact);
    assert!(
        (fresh_full.results[0].candidate_mass - fresh_compact.results[0].candidate_mass).abs()
            < 1e-12
    );
}

#[test]
#[ignore = "requires SKID_VISION_MODEL and SKID_VISION_MMPROJ"]
fn real_image_changes_logits_and_failed_image_does_not_leak_state() {
    let model = std::env::var("SKID_VISION_MODEL").expect("set SKID_VISION_MODEL");
    let projector = std::env::var("SKID_VISION_MMPROJ").expect("set SKID_VISION_MMPROJ");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let request: DecisionRequest = serde_json::from_value(serde_json::json!({
        "state": {},
        "decisions": [{
            "id": "color",
            "instruction": "Which color fills the image?",
            "kind": {"type": "choice", "options": [
                {"id": "red", "criterion": "The image is red."},
                {"id": "blue", "criterion": "The image is blue."}
            ]}
        }]
    }))
    .unwrap();
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        2048,
        256,
        4,
        cuda,
        DecisionPolicy::default(),
    )
    .unwrap();
    backend.load_vision_projector(projector.as_ref()).unwrap();
    let red = include_bytes!("fixtures/vision_red_64.png");
    let blue = include_bytes!("fixtures/vision_blue_64.png");
    let first = backend.decide_vision(&request, red).unwrap();
    let second = backend.decide_vision(&request, blue).unwrap();
    assert_eq!(first.backend.runtime, "local-libllama-mtmd");
    assert_eq!(
        first
            .backend
            .vision_projector_sha256
            .as_ref()
            .map(String::len),
        Some(64)
    );
    assert!(first.results[0].input_tokens > 0);
    assert!(
        first.results[0]
            .scores
            .iter()
            .zip(&second.results[0].scores)
            .any(|(a, b)| (a.raw_logit - b.raw_logit).abs() > 1e-4)
    );
    assert!(backend.decide_vision(&request, b"not an image").is_err());
    let again = backend.decide_vision(&request, red).unwrap();
    for (a, b) in first.results[0].scores.iter().zip(&again.results[0].scores) {
        assert!((a.raw_logit - b.raw_logit).abs() < 1e-4);
    }
}

#[test]
#[ignore = "requires SKID_VISION_MODEL and SKID_VISION_MMPROJ"]
fn wide_image_codes_share_text_code_paths_and_leave_narrow_scoring_intact() {
    let model = std::env::var("SKID_VISION_MODEL").expect("set SKID_VISION_MODEL");
    let projector = std::env::var("SKID_VISION_MMPROJ").expect("set SKID_VISION_MMPROJ");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        8192,
        256,
        4,
        cuda,
        DecisionPolicy::default(),
    )
    .unwrap();
    backend.load_vision_projector(projector.as_ref()).unwrap();
    let image = include_bytes!("fixtures/vision_red_64.png");
    let narrow: DecisionRequest = serde_json::from_value(serde_json::json!({
        "state": {},
        "decisions": [{"id": "color", "instruction": "Which color fills the image?",
            "kind": {"type": "choice", "options": [
                {"id": "red", "criterion": "The image is red."},
                {"id": "blue", "criterion": "The image is blue."}
            ]}}]
    }))
    .unwrap();
    let before = backend.decide_vision(&narrow, image).unwrap();
    for count in [27, 77] {
        let request: DecisionRequest = serde_json::from_value(serde_json::json!({
            "state": {},
            "decisions": [{"id": "wide", "instruction": "Choose the matching image category.",
                "kind": {"type": "choice", "options": (0..count).map(|i|
                    serde_json::json!({"id": format!("category_{i}"),
                        "criterion": format!("The image belongs to category {i}.")})
                ).collect::<Vec<_>>()}}]
        }))
        .unwrap();
        let (_, text_paths) = backend
            .encode_decision_sequences(&request.state, &request.decisions[0])
            .unwrap();
        let result = backend.decide_vision(&request, image).unwrap();
        let scored = &result.results[0];
        assert_eq!(scored.scores.len(), count);
        assert_eq!(scored.scores[0].code, "AA");
        assert_eq!(
            scored.scoring_method,
            "code_sequence_conditional_softmax_v1"
        );
        assert!(scored.input_tokens > 0);
        assert!(scored.code_prefix_evaluations > 0);
        assert_eq!(scored.reused_prefix_tokens, 0);
        for (score, path) in scored.scores.iter().zip(&text_paths) {
            assert_eq!(&score.token_ids, path);
        }
        assert!(
            result
                .backend
                .prompt_version
                .contains("fixed-width-code-sequences-v1")
        );
        let probability_sum: f64 = scored.scores.iter().map(|s| s.option_probability).sum();
        assert!((probability_sum - 1.0).abs() < 1e-10);
        if scored.scores.iter().any(|s| s.token_ids.len() > 1) {
            assert!(scored.code_prefix_evaluations > 1);
        }
    }
    let after = backend.decide_vision(&narrow, image).unwrap();
    for (a, b) in before.results[0]
        .scores
        .iter()
        .zip(&after.results[0].scores)
    {
        assert!((a.raw_logit - b.raw_logit).abs() < 1e-4);
    }
}

#[test]
#[ignore = "requires real vision GGUF/projector; exact fresh-preserving profile parity and recovery"]
fn real_vision_preserving_profile_matches_fresh_and_recovers() {
    assert!(
        std::env::var_os("SKID_VISION_OPTIMIZED").is_none(),
        "unset SKID_VISION_OPTIMIZED to use the original batch256/flash-off fixture"
    );
    let (mut backend, requests, images) = batch_fixture();
    let baseline_info = backend.info();
    assert_eq!(baseline_info.execution_mode, ExecutionMode::Fresh);
    assert_eq!(
        baseline_info.evidence_transfer,
        l2s1::EvidenceTransfer::Full
    );
    assert_eq!(baseline_info.compute, Some(l2s1::ComputeOptions::default()));
    let baseline: Vec<_> = requests
        .iter()
        .zip(&images)
        .map(|(request, image)| backend.decide_vision(request, image).unwrap())
        .collect();

    fn assert_exact_results(expected: &[DecisionResponse], actual: &[DecisionResponse]) {
        assert_eq!(expected.len(), actual.len());
        for (expected, actual) in expected.iter().zip(actual) {
            assert_eq!(expected.results.len(), actual.results.len());
            for (a, b) in expected.results.iter().zip(&actual.results) {
                assert_eq!((&a.id, a.input_tokens), (&b.id, b.input_tokens));
                assert_eq!(a.candidate_mass, b.candidate_mass);
                assert_eq!(a.top_option_probability, b.top_option_probability);
                assert_eq!(a.entropy_confidence, b.entropy_confidence);
                assert_eq!(a.scores.len(), b.scores.len());
                for (a, b) in a.scores.iter().zip(&b.scores) {
                    assert_eq!((&a.id, &a.code, a.token_id), (&b.id, &b.code, b.token_id));
                    assert_eq!(a.raw_logit, b.raw_logit);
                    assert_eq!(a.option_probability, b.option_probability);
                }
                let ranking = |result: &l2s1::DecisionResult| {
                    let mut scores = result.scores.iter().collect::<Vec<_>>();
                    scores.sort_by(|a, b| b.option_probability.total_cmp(&a.option_probability));
                    scores
                        .into_iter()
                        .map(|score| score.id.clone())
                        .collect::<Vec<_>>()
                };
                assert_eq!(ranking(a), ranking(b));
                // Includes selected value, abstention reasons, token metadata,
                // scoring method and every remaining result field exactly.
                assert_eq!(
                    serde_json::to_value(a).unwrap(),
                    serde_json::to_value(b).unwrap()
                );
            }
        }
    }

    backend.enable_vision_preserving_optimizations().unwrap();
    let inspection = backend.inspect();
    assert_eq!(inspection.identity.compute, l2s1::ComputeOptions::default());
    assert_eq!(inspection.identity.execution_mode, ExecutionMode::Fresh);
    assert_eq!(
        inspection.identity.evidence_transfer,
        l2s1::EvidenceTransfer::Compact
    );
    assert!(!inspection.identity.parallel_context_dynamic);
    let info = backend.info();
    assert_eq!(info.compute, baseline_info.compute);
    assert_eq!(info.execution_mode, ExecutionMode::Fresh);
    assert_eq!(info.evidence_transfer, l2s1::EvidenceTransfer::Compact);
    assert!(!info.parallel_context_dynamic);
    assert!(!info.vision_projector_reuse);
    let preserving = backend.decide_vision_batch(&requests, &images).unwrap();
    assert_exact_results(&baseline, &preserving);
    let cache = backend.preparation_cache_stats();
    assert!(cache.vision.entries > 0);
    let repeated = backend.decide_vision_batch(&requests, &images).unwrap();
    assert_exact_results(&baseline, &repeated);
    let decisions = requests.iter().map(|r| r.decisions.len()).sum::<usize>();
    assert!(backend.preparation_cache_stats().vision.hits >= cache.vision.hits + decisions as u64);

    let mut invalid_images = images;
    invalid_images[3] = b"invalid image";
    assert!(
        backend
            .decide_vision_batch(&requests, &invalid_images)
            .is_err()
    );
    assert_exact_results(
        &baseline,
        &backend.decide_vision_batch(&requests, &images).unwrap(),
    );
    let mut overlong = requests.clone();
    // Earlier requests complete before this native context-overflow failure.
    overlong[4].decisions[0].instruction = "context overflow ".repeat(4096);
    assert!(backend.decide_vision_batch(&overlong, &images).is_err());
    assert_exact_results(
        &baseline,
        &backend.decide_vision_batch(&requests, &images).unwrap(),
    );

    backend.set_code_rotation(1).unwrap();
    assert_eq!(backend.preparation_cache_stats().vision.entries, 0);
    backend.set_code_rotation(0).unwrap();
    assert_exact_results(
        &baseline,
        &backend.decide_vision_batch(&requests, &images).unwrap(),
    );
}
