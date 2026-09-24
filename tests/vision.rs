#![cfg(feature = "llama")]
use l2s1::{DecisionPolicy, DecisionRequest, llama::LlamaBackend};

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
