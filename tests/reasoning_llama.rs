#![cfg(feature = "llama")]

use l2s1::{
    Decision, DecisionBackend, DecisionKind, DecisionPolicy, DecisionRequest, ExecutionMode,
    PreparationCacheConfig, ReasoningMode, ReasoningOptions, llama::LlamaBackend,
};
use std::path::Path;

/// Opt-in real-model regression: no synthetic logits substitute for generated
/// thinking or the KV state recovered after a failed generation budget.
#[test]
fn real_qwen3_thinking_budget_completion_and_request_isolation() {
    let Ok(model) = std::env::var("L2S1_TEST_REASONING_MODEL") else {
        eprintln!(
            "set L2S1_TEST_REASONING_MODEL to a Qwen3 dense GGUF to run native reasoning regression"
        );
        return;
    };
    let mut backend = LlamaBackend::load(
        Path::new(&model),
        2048,
        256,
        4,
        false,
        DecisionPolicy::default(),
    )
    .unwrap();
    let request = DecisionRequest {
        shared: None,
        state: serde_json::json!({"text":"The animal is a cat."}),
        decisions: vec![Decision {
            id: "animal".into(),
            instruction: "Does the text explicitly name a cat?".into(),
            kind: DecisionKind::Binary {
                false_label: "No".into(),
                true_label: "Yes".into(),
            },
        }],
    };
    let baseline = backend.decide(&request).unwrap();
    let direct_identity = backend.identity();
    assert!(backend.supports_thinking());
    assert!(baseline.results[0].reasoning.is_none());
    backend
        .set_reasoning(ReasoningOptions {
            mode: ReasoningMode::Thinking,
            max_tokens: 1,
        })
        .unwrap();
    let failure = backend.decide(&request).unwrap_err().to_string();
    assert!(
        failure.contains("reasoning_limit")
            && failure.contains("generated=1")
            && failure.contains("max_tokens=1"),
        "{failure}"
    );
    backend.set_reasoning(ReasoningOptions::default()).unwrap();
    assert_eq!(
        backend.identity().prompt_version,
        direct_identity.prompt_version
    );
    let recovered = backend.decide(&request).unwrap();
    for (a, b) in baseline.results[0]
        .scores
        .iter()
        .zip(&recovered.results[0].scores)
    {
        assert_eq!(a.option_probability, b.option_probability);
    }
    backend
        .set_reasoning(ReasoningOptions {
            mode: ReasoningMode::Thinking,
            max_tokens: 512,
        })
        .unwrap();
    assert_ne!(
        backend.identity().prompt_version,
        direct_identity.prompt_version
    );
    let thinking = backend.decide(&request).unwrap();
    let usage = thinking.results[0].reasoning.unwrap();
    assert!(usage.completed && usage.generated_tokens > 1 && usage.generated_tokens <= 512);
    assert_eq!(usage.mode, ReasoningMode::Thinking);
    let repeat = backend.decide(&request).unwrap();
    assert_eq!(repeat.results[0].reasoning, Some(usage));
    for (a, b) in thinking.results[0]
        .scores
        .iter()
        .zip(&repeat.results[0].scores)
    {
        assert_eq!(a.option_probability, b.option_probability);
    }
    backend.set_execution_mode(ExecutionMode::PrefixReuse);
    assert!(!backend.supports_thinking());
    assert!(
        backend
            .decide(&request)
            .unwrap_err()
            .to_string()
            .contains("fresh text execution")
    );
    backend.set_execution_mode(ExecutionMode::Fresh);
    backend.set_preparation_cache(PreparationCacheConfig {
        max_bytes: 1024,
        max_entries: 4,
    });
    assert!(!backend.supports_thinking());
    assert!(
        backend
            .decide(&request)
            .unwrap_err()
            .to_string()
            .contains("disabled preparation cache")
    );
    backend.set_preparation_cache(PreparationCacheConfig {
        max_bytes: 0,
        max_entries: 0,
    });
    backend.set_reasoning(ReasoningOptions::default()).unwrap();
    assert!(backend.supports_thinking());
    let after_thinking = backend.decide(&request).unwrap();
    for (a, b) in baseline.results[0]
        .scores
        .iter()
        .zip(&after_thinking.results[0].scores)
    {
        assert_eq!(a.option_probability, b.option_probability);
    }
    eprintln!(
        "thinking completed in {} generated tokens; limit errors and mode changes preserve direct probabilities",
        usage.generated_tokens
    );
    // Exercise the actual HTTP adapter's save/restore wrapper, including the
    // failure path, without replacing native thinking with a fixture backend.
    use l2s1::http::HttpDecisionBackend;
    let requests = std::slice::from_ref(&request);
    let images = vec![vec![]];
    let http_direct = backend
        .decide_json_batch_with_reasoning(requests, &images, None)
        .unwrap();
    let limit = ReasoningOptions {
        mode: ReasoningMode::Thinking,
        max_tokens: 1,
    };
    let error = backend
        .decide_json_batch_with_reasoning(requests, &images, Some(&limit))
        .unwrap_err();
    assert!(error.to_string().contains("reasoning_limit"));
    assert_eq!(backend.reasoning(), ReasoningOptions::default());
    let options = ReasoningOptions {
        mode: ReasoningMode::Thinking,
        max_tokens: 128,
    };
    let http_thinking = backend
        .decide_json_batch_with_reasoning(requests, &images, Some(&options))
        .unwrap();
    assert_eq!(
        http_thinking[0]["results"][0]["usage"]["reasoning"]["completed"],
        true
    );
    assert_eq!(backend.reasoning(), ReasoningOptions::default());
    let restored = backend
        .decide_json_batch_with_reasoning(requests, &images, None)
        .unwrap();
    assert_eq!(http_direct, restored);
}
