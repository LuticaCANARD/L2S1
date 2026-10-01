#![cfg(feature = "llama")]

use l2s1::{
    ComputeOptions, DecisionPolicy, DecisionRequest, DecisionResponse, EvidenceTransfer,
    ExecutionMode, PromptProfile, llama::LlamaBackend,
};

fn probabilities(response: &DecisionResponse) -> Vec<f64> {
    response.results[0]
        .scores
        .iter()
        .map(|score| score.option_probability)
        .collect()
}

fn assert_same(a: &DecisionResponse, b: &DecisionResponse) {
    assert_eq!(a.results[0].input_tokens, b.results[0].input_tokens);
    for (x, y) in probabilities(a).iter().zip(probabilities(b)) {
        assert!((x - y).abs() < 0.02, "{x} != {y}");
    }
}

#[test]
#[ignore = "requires SKID_VISION_MODEL and SKID_VISION_MMPROJ; seven-image native encoding"]
fn real_vision_images_share_one_context_preserve_single_and_recover() {
    let model = std::env::var("SKID_VISION_MODEL").expect("set SKID_VISION_MODEL");
    let projector = std::env::var("SKID_VISION_MMPROJ").expect("set SKID_VISION_MMPROJ");
    let mut backend = LlamaBackend::load_with_options(
        model.as_ref(),
        ComputeOptions {
            context: 8192,
            ..ComputeOptions::default()
        },
        std::env::var("SKID_CUDA").as_deref() == Ok("1"),
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )
    .unwrap();
    backend.load_vision_projector(projector.as_ref()).unwrap();
    let request: DecisionRequest = serde_json::from_value(serde_json::json!({
        "state": {"image_order": "Images 1 through 6 are references. Image 7 is the target."},
        "decisions": [{
            "id": "color",
            "instruction": "Classify the color of the last target image, image 7. Use images 1 through 6 only as references.",
            "kind": {"type": "choice", "options": [
                {"id": "red", "criterion": "The target image is red."},
                {"id": "blue", "criterion": "The target image is blue."}
            ]}
        }]
    })).unwrap();
    let red = include_bytes!("fixtures/vision_red_64.png").as_slice();
    let blue = include_bytes!("fixtures/vision_blue_64.png").as_slice();
    let images = [red, red, red, red, red, red, blue];
    let seven = backend.decide_vision_images(&request, &images).unwrap();
    let metrics = backend.vision_images_metrics().unwrap();
    assert_eq!(metrics.images, 7);
    assert!(metrics.image_chunks >= 7);
    eprintln!(
        "seven ordered images: input_tokens={}, images={}, image_chunks={}",
        seven.results[0].input_tokens, metrics.images, metrics.image_chunks
    );

    // This checks actual visual dependence, without treating model accuracy
    // on the color task as a requirement of the request transport contract.
    let reordered = [blue, red, red, red, red, red, red];
    let swapped = backend.decide_vision_images(&request, &reordered).unwrap();
    assert!(
        probabilities(&seven)
            .iter()
            .zip(probabilities(&swapped))
            .any(|(x, y)| (x - y).abs() > 1e-6)
    );

    backend
        .set_evidence_transfer(EvidenceTransfer::Compact)
        .unwrap();
    let compact = backend.decide_vision_images(&request, &images).unwrap();
    assert_same(&seven, &compact);
    backend
        .set_evidence_transfer(EvidenceTransfer::Full)
        .unwrap();

    assert!(backend.decide_vision_images(&request, &[]).is_err());
    assert!(backend.decide_vision_images(&request, &[red; 9]).is_err());
    let bad = [red, b"unsupported image".as_slice(), blue];
    assert!(backend.decide_vision_images(&request, &bad).is_err());
    let mut oversized = serde_json::to_value(&request).unwrap();
    oversized["state"]["padding"] = " many tokens".repeat(9000).into();
    let oversized: DecisionRequest = serde_json::from_value(oversized).unwrap();
    let error = backend
        .decide_vision_images(&oversized, &images)
        .unwrap_err();
    assert!(error.to_string().contains("truncation is disabled"));
    let recovered = backend.decide_vision_images(&request, &images).unwrap();
    assert_same(&seven, &recovered);

    let single = backend.decide_vision(&request, blue).unwrap();
    let single_images = backend.decide_vision_images(&request, &[blue]).unwrap();
    assert_eq!(
        seven.backend.prompt_version,
        single_images.backend.prompt_version
    );
    assert_same(&single, &single_images);
    assert_eq!(backend.vision_images_metrics().unwrap().images, 1);
    assert!(seven.results[0].input_tokens > single.results[0].input_tokens);
    backend.set_execution_mode(ExecutionMode::Parallel);
    let error = backend.decide_vision_images(&request, &images).unwrap_err();
    assert!(error.to_string().contains("require fresh execution"));
    // Single-image calls still reach the unchanged independent sequence API.
    assert!(backend.decide_vision_images(&request, &[blue]).is_ok());
}
