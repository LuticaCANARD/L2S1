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
