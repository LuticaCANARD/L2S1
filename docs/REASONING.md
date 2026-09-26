# Direct scoring and bounded thinking

[English](en/REASONING.md) · [한국어](ko/REASONING.md) · [日本語](ja/REASONING.md)

[English index](en/README.md) · [한국어 색인](ko/README.md) · [日本語索引](ja/README.md)

Direct mode remains the default and preserves the existing prompt bytes. It
scores the candidate codes after a closed, empty thinking block without
generating a reasoning sequence. Explicit `thinking` mode first generates
tokens greedily, waits for the model to emit its native `</think>` token,
evaluates the final separator, and then scores the same typed candidate set
from full-vocabulary logits. The final answer code is scored, not generated.

Initial thinking support is limited to **dense Qwen3 text GGUFs** whose chat
template supports `enable_thinking`, with the `qwen3` prompt profile (selected
automatically for supported models). Image thinking, other architectures,
more than 26 candidates, prefix reuse, parallel execution, state restoration,
preparation caches, compact evidence, LoRA, output heads, learned calibration,
and hidden-feature export are rejected. Image decisions continue to use direct
mode with a compatible vision model and projector.

## CLI

```sh
cargo build --release --locked --features llama --bin l2s1
target/release/l2s1 --model models/Qwen3-0.6B-Q8_0.gguf \
  --input request.json --reasoning-mode thinking --max-reasoning-tokens 128
```

Use `--reasoning-mode direct` for the historical path. The thinking budget
defaults to 128 and accepts 1–1024. The input, full token budget and final
separator must fit the configured context; input truncation is disabled.
If the model never closes thinking within the budget, inference fails with
`reasoning_limit`, the generated token count and budget. An end-of-generation
token before closure fails with `reasoning_incomplete`. Neither path inserts
a synthetic closing token or converts an unfinished sequence into an answer.

## Library and HTTP

```rust,ignore
backend.set_reasoning(l2s1::ReasoningOptions {
    mode: l2s1::ReasoningMode::Thinking,
    max_tokens: 128,
})?;
let response = backend.decide(&request)?;
backend.set_reasoning(l2s1::ReasoningOptions::default())?;
```

The JSON HTTP request accepts `"reasoning":{"mode":"thinking","max_tokens":128}`.
The adapter restores the previous reasoning mode after each request, including
failures. Supported model information is available through the capability
endpoint; selecting thinking on a vision request is an explicit error.

Successful CLI/library results contain `reasoning`; HTTP results place it under
`usage.reasoning`. For example:

```json
{"mode":"thinking","generated_tokens":88,"completed":true}
```

The count includes the model-generated closing token. It excludes the trusted
final separator and the ungenerated answer code. `input_tokens` still counts
the prepared input prompt. The reasoning trace is not returned. Prompt/artifact
identity includes `thinking-greedy-v1` and the budget, preventing direct-mode
calibration from being silently reused.

## Verified boundary

On September 26, 2026, the real Qwen3-0.6B Q8_0 CPU runtime completed a cat-text
binary question in 88 generated tokens. A one-token budget failed explicitly.
After that failure and after completed thinking, returning to direct mode
reproduced every candidate probability exactly on the same backend instance.
Repeated thinking also reproduced token usage and probabilities exactly.
Invalid cache/execution combinations were rejected and recovered correctly.

The same question also completed through the actual CUDA HTTP browser demo on
an RTX 3080 in 96 generated tokens, and a one-token budget returned HTTP 422
with the user's custom `reasoning_limit` message. CPU and CUDA generated-token
counts are not assumed identical. These are completion checks, not controlled
latency or cross-device numerical-equivalence measurements.

This is a native generation and isolation regression, not a quality benchmark
or a claim that thinking improves accuracy. The published typed-decisions
measurements use direct mode. Candidate probabilities and `candidate_mass`
retain their existing conditional-scoring meanings after thinking.

Run the real-model regression explicitly:

```sh
L2S1_TEST_REASONING_MODEL=/path/to/Qwen3-0.6B-Q8_0.gguf \
  cargo test --locked --features llama --test reasoning_llama -- --nocapture
```

Local raw direct/thinking/limit evidence is in the gitignored
`results/reasoning-native-20260926/` directory.
