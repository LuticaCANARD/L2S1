# L2S1 — LLM to System 1

**Swap the model while keeping your application's decision contract.**

L2S1 is a Rust library and CLI that turns a compatible local GGUF chat model into a typed decision backend. Applications supply a state and a list of binary, choice, or ordinal decisions. L2S1 prepares each question for the selected model, reads next-token scores, and returns a decision or an explicit abstention.

Application option IDs, result types, and acceptance rules stay consistent across models. Templates, token IDs, predictions, and calibration are model-specific. Changing a model does not guarantee the same answer or accuracy.

The crate is named `l2s1`. The default inference executable uses **llama.cpp and local GGUF files**. An optional `wgpu` feature provides a separate `l2s1-wgpu` executable for native GPU inference. An optional `openrouter` feature provides a selection-only remote executable. Model switching currently means loading a new backend or replacing an owned worker; there is no automatic model router or live hot-swap service.

## Optional wgpu backend

The `wgpu` feature uses the pinned [rullama-engine](https://github.com/Brainwires/rullama-framework/tree/main/engine/rullama-engine) Gemma 4 text and vision implementation. It accepts a Gemma 4 text GGUF alone or with its matching `mmproj` GGUF. A streaming adapter exposes paired files as one virtual GGUF to the engine; model weights are read from the original files. GPU inference and image encoding use Rust wgpu. The feature requires a wgpu GPU adapter and rejects software Vulkan adapters unless `--allow-software-adapter` is explicitly set for development. Set `WGPU_BACKEND=metal` and `--require-metal` on macOS to enforce the native Metal adapter. The Gemma 4 E2B Q8_0 pair has been checked locally through software wgpu; native GPU quality and other checkpoint sizes still need verification.

```sh
cargo run --release --locked --features wgpu --bin l2s1-wgpu -- \
  --model /models/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /models/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --image photo.jpg \
  --input examples/warehouse.json
```

Omit `--mmproj` and `--image` for text decisions. Add `--listen 127.0.0.1:8080` to serve the same `POST /v1/decisions` and `GET /healthz` API described below; send `media` for vision requests. Images are decoded up to 25 megapixels and resized to at most 432 pixels on the longer side, aligned to the vision encoder's 48-pixel grid. The wgpu path scores full-vocabulary mass and complete multi-token answer codes for binary, choice and ordinal decisions. It supports request-local `fresh`, `prefix-reuse`, and `state-restore` execution, with `--snapshot-limit-bytes` and `--diagnostics` on the CLI. Multi-token answer codes use fresh evaluation. `parallel`, LoRA, output heads, and scalar calibration use the generic llama.cpp path below. GPU scores may differ from llama.cpp; validate each model and task before treating them as calibrated probabilities.

This paired-GGUF adapter is specific to Gemma 4 and rejects Qwen model/projector files. For Bonsai, Qwen, SmolLM and other compatible GGUF chat models, use the existing llama.cpp backend. It reads the selected GGUF's tokenizer and chat template and exposes the same typed decision and HTTP APIs. Its CUDA build supports NVIDIA GPUs; the `llama-metal` build supports Apple Metal on macOS:

```sh
cargo run --release --locked --features llama-cuda -- \
  --model /models/Qwen3-0.6B-Q8_0.gguf \
  --device cuda --input examples/warehouse.json

# macOS
cargo run --release --locked --features llama-metal -- \
  --model /models/Qwen3-0.6B-Q8_0.gguf \
  --device metal --input examples/warehouse.json
```

Change `--model` to load another compatible GGUF. Text models need only the model file; for a supported vision model, supply its matching `--mmproj` file and send the image through the CLI or HTTP API. The llama.cpp Metal path retains the existing output heads, LoRA, calibration, execution modes, and HTTP contract. Each model needs its own backend instance and task-quality evaluation. This route does not require FlareLLM or a Qwen-specific Rust wgpu adapter. A [two-model CUDA smoke measurement](benchmarks/gguf-cuda-20260925/README.md) records the model identities, decisions, abstentions, and timings; it does not establish Metal performance.

| Metal route | GGUF models | Vision | Execution | Additional features |
| --- | --- | --- | --- | --- |
| Rust wgpu | Gemma 4 | Matching Gemma 4 `mmproj` | fresh, prefix-reuse, state-restore; multi-token codes use fresh | Same typed CLI and HTTP decisions |
| llama.cpp | Models supported by the pinned llama.cpp revision, including compatible Bonsai and Qwen GGUF | Supported model with matching `mmproj` | Existing fresh, prefix-reuse, state-restore, parallel contracts; vision remains fresh | LoRA, calibration, output heads, diagnostics, HTTP |

The `wgpu` executable rejects `parallel`; use the llama.cpp Metal route when it is needed. A Mac with compatible model files is needed to verify actual Metal inference and result equivalence.

## Architecture

```mermaid
flowchart TD
    A[Application or CLI] --> B[DecisionRequest: state and decisions]
    B --> C[Backend: model-specific prompt and token preparation]
    C --> D[llama.cpp or Gemma 4 Rust wgpu: GGUF inference]
    D --> E[Candidate logits or complete answer-code likelihoods]
    E --> F[Shared scoring, optional calibration or head, and DecisionPolicy]
    F --> G[DecisionResponse: typed values, scores, and abstention reasons]
```

| Component | Responsibility |
| --- | --- |
| [`decision.rs`](src/decision.rs) | Request/response types, `DecisionBackend`, shared scoring and acceptance policy |
| [`prompt.rs`](src/prompt.rs) | Compile state, instructions and options into the selected prompt layout |
| [`llama.rs`](src/llama.rs) | Own the model/context, select the prompt profile, tokenize inputs, dispatch inference and assemble results |
| [`wgpu.rs`](src/wgpu.rs), [`paired_gguf.rs`](src/wgpu/paired_gguf.rs) | Rust wgpu Gemma 4 text/vision inference and paired GGUF streaming |
| [`vision.rs`](src/vision.rs), [`http/contract.rs`](src/http/contract.rs), [`http.rs`](src/http.rs) | Image validation, wire contract and backend mapping, HTTP connection handling |
| [`openrouter.rs`](src/openrouter.rs) | Remote chat completions and selection-only response mapping |
| [`l2s1-llama-sys`](crates/l2s1-llama-sys), [`bridge.cpp`](crates/l2s1-llama-sys/native/bridge.cpp), [`chat.cpp`](crates/l2s1-llama-sys/native/chat.cpp) | Call llama.cpp, render GGUF Jinja templates, manage sequence memory and copy inference evidence |
| [`evidence.rs`](src/evidence.rs) | Validate complete vocabulary logits and preserve semantic option/token mappings |
| [`codes.rs`](src/codes.rs), [`llama/code_sequences.rs`](src/llama/code_sequences.rs) | Size A-Z/AA-ZZ/AAA-ZZZ codes and score complete token paths for larger candidate sets |
| [`calibration.rs`](src/calibration.rs), [`output_head.rs`](src/output_head.rs) | Optional task-scoped temperature calibration or learned output scoring |
| [`interoperability.rs`](src/interoperability.rs), [`llama/interchange.rs`](src/llama/interchange.rs) | Model fingerprints, capabilities, request preflight, structured failures and execution diagnostics |
| [`worker.rs`](src/worker.rs) | Bounded admission and dedicated-thread ownership of a backend |

The standard CLI calls `LlamaBackend` directly; the optional GPU CLI calls `WgpuBackend`. Long-running applications can place either backend behind `BackendWorker`.

## Direct image input and HTTP API

Load a vision-capable chat GGUF with its matching multimodal projector GGUF (`mmproj`). The projector encodes a still image through llama.cpp `libmtmd`; L2S1 then scores the same typed options from the resulting next-token logits. `LlamaBackend::load_vision_projector(path)` and `LlamaBackend::decide_vision(&request, image_bytes)` expose the Rust API. Existing text requests still use `decide`.

The opt-in HTTP listener accepts JSON at `POST /v1/decisions` and reports readiness at `GET /healthz`:

```sh
cargo run --release --locked --features llama-cuda -- \
  --model /models/vision-model.gguf --mmproj /models/mmproj.gguf \
  --device cuda --context 4096 --listen 127.0.0.1:8080
```

The versioned API accepts shared `state`, named `media`, and decisions. Omit `media` for text. Each decision can set `media_ids` to select images; omission selects all images and `[]` selects none. The response always has `api_version`, `request_id`, `backend`, `policy`, and `results`. Every result has `id`, typed `value`, `status`, `abstention_reasons`, `evidence`, and `usage`. Local evidence has `type: "model_scored"` with scores and candidate mass; OpenRouter evidence has `type: "selection_only"` and no invented probabilities. Local `policy` is populated; remote `policy` is `null`. Use `GET /v1/capabilities` to inspect the loaded model, supported image input, evidence type, and limits.

```json
{
  "state": {"task": "identify the object"},
  "media": [
    {"id": "front", "type": "image", "data_base64": "..."},
    {"id": "side", "type": "image", "data_base64": "..."}
  ],
  "decisions": [{
    "id": "object", "instruction": "Choose the main object",
    "kind": {"type": "choice", "options": [
      {"id": "box", "criterion": "a box"},
      {"id": "bag", "criterion": "a bag"}
    ]},
    "media_ids": ["front"]
  }]
}
```

A selection-only response has the same result envelope as a local response:

```json
{
  "api_version": 1,
  "request_id": "req-1",
  "backend": {"runtime": "openrouter-chat-completions", "model": "example/model", "details": null},
  "policy": null,
  "results": [{
    "id": "object", "value": {"type": "choice", "selected": "box"},
    "status": "selected", "abstention_reasons": [],
    "evidence": {"type": "selection_only", "selected_code": "A", "provider_model": "example/model"},
    "usage": {"input_tokens": 42, "output_tokens": 1}
  }]
}
```

For an existing text request, this command attaches one image:

```sh
jq --arg image "$(base64 -w0 photo.jpg)" '. + {media: [{id: "photo", type: "image", data_base64: $image}]}' \
  examples/warehouse.json | \
  curl -sS -H 'Content-Type: application/json' --data-binary @- \
  http://127.0.0.1:8080/v1/decisions
```

The library still accepts original image bytes without base64. The CLI equivalent is `--mmproj /models/mmproj.gguf --image photo.jpg --input request.json`. The local llama.cpp and wgpu backends accept one image per decision; the OpenRouter adapter accepts up to four images per decision, subject to the selected provider model's own limits. Image bytes are limited to 8 MiB each, each request to 128 decisions, and the HTTP body to 44 MiB. Invalid media references or backend limits fail before inference. More than 26 options use fixed-width answer codes. Local vision uses full-vocabulary scoring; llama.cpp vision uses fresh execution and rejects output heads, scalar calibration, and parallel/prefix-reuse modes for image requests. Gemma 4 wgpu vision accepts request-local prefix reuse and state restoration. The listener accepts up to 32 connections with a 16-request inference queue and a 192 MiB in-flight body budget. Local GPU inference remains serial; health and capability requests stay responsive while inference is busy when a connection slot remains. Bind to loopback or place an authenticated reverse proxy in front of it for remote clients. Errors have `error.code`, `error.message`, and `error.request_id`.

Additional Rust backends implement `HttpDecisionBackend::capabilities` and `decide_json`. The contract layer validates result IDs and the common result fields before sending any response. A backend can add evidence fields under its `evidence.type` without changing the shared `value` and `status` fields.

Gemma is a possible vision backend: Gemma 3 4B/12B/27B and Gemma 4 E2B/E4B have image-capable variants in llama.cpp. Gemma 3 1B is text-only. Pair a vision checkpoint with its matching `mmproj`; a text-only GGUF file alone cannot accept pixels. See the [llama.cpp multimodal model list](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md) and [Gemma 3 vision guide](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal/gemma3.md).

For CPU/CUDA latency measurements with Gemma 4 and two labeled image fixtures, see the [direct vision benchmark](VISION_BENCHMARK.md).
For a 30-class, 150-image CUDA run through the HTTP vision API, see the [Caltech-101 benchmark](benchmarks/caltech101-vision-20260924/README.md).
For a 70-image cat/dog validation with both answer orders, see the [Cats and Dogs vision report](benchmarks/cats-dogs-vision-20260924/REPORT.md); it records a historical HTTP request shape and does not claim general image accuracy.
For six-class waste-material classification and a paired prompt/acceptance-threshold study, see the [TrashNet vision benchmark](benchmarks/trashnet-vision-20260925/REPORT.md).

## OpenRouter adapter

Set `OPENROUTER_API_KEY` in the process environment and select an [OpenRouter model](https://openrouter.ai/models) that accepts the requested modality. The optional executable does not build the native llama.cpp backend. For example, `prism-ml/ternary-bonsai-2-27b` accepts text and images; it is a different checkpoint from a local Bonsai 27B Q1_0 GGUF.

```sh
cargo run --release --locked --no-default-features --features openrouter \
  --bin l2s1-openrouter -- \
  --model prism-ml/ternary-bonsai-2-27b --reasoning-effort none \
  --input examples/warehouse.json
```

For one image on the CLI, add `--image photo.jpg`. The adapter accepts PNG, JPEG, GIF and WebP up to 8 MiB. To serve HTTP, replace `--input ...` with `--listen 127.0.0.1:8081`. The listener exposes `POST /v1/decisions`, `GET /v1/capabilities`, and `GET /healthz`. It runs up to four remote HTTP requests in parallel; decisions with the same media selection are also sent in batches of up to four parallel completions. `--max-tokens` sets a per-decision completion limit (default 1024, maximum 4096). `--reasoning-effort` is optional and passed through only when requested; use a value supported by the selected model.

Codes must match exactly after surrounding whitespace is removed; malformed or incomplete model output abstains. Fixed-width codes support more than 26 options. The remote response has no `scores`, `candidate_mass`, `top_option_probability`, `p_true` or ordinal expected value, and the local probability policy is not applied. An ordinal result has the selected level's `level_value`. Provider or transport failures return HTTP 502. A mock-server test checks the adapter; a live OpenRouter request requires an API key. The provider's image and request limits depend on the selected model, so `GET /v1/capabilities` reports the adapter limits and marks the provider limit as model dependent.

## The decision contract

A request has shared JSON `state` and one or more decisions. Each decision supplies an ID, an instruction and its output kind:

| Kind | Definition | Result |
| --- | --- | --- |
| `binary` | False and true criteria | `p_true` and an optional Boolean |
| `choice` | Semantic option IDs and criteria | An optional selected option ID |
| `ordinal` | Ordered levels with strictly increasing numeric values | Expected value and an optional selected level ID |

For example:

```json
{
  "state": { "storage_requirement": "chilled" },
  "decisions": [
    {
      "id": "storage_zone",
      "instruction": "Select the storage zone matching storage_requirement.",
      "kind": {
        "type": "choice",
        "options": [
          { "id": "ambient", "criterion": "Ambient storage is required." },
          { "id": "chilled", "criterion": "Chilled storage is required." },
          { "id": "frozen", "criterion": "Frozen storage is required." }
        ]
      }
    }
  ]
}
```

L2S1 maps these semantic IDs to answer codes and checks their tokenization at the model's actual assistant answer boundary. Up to 26 options retain the original `A`–`Z` single-token path. Larger candidate sets automatically use fixed-width codes: `AA`–`ZZ`, then `AAA`–`ZZZ`, and so on. Complete code-sequence likelihoods are scored when codes span multiple tokens. The application receives `chilled`, not a model-specific code, as its selected value. Token paths and raw scores remain available as evidence. See [answer-code expansion and intent evaluation](INTENT_BENCHMARK.md).

Each decision is evaluated independently. A request can contain different decision kinds over the same state; it is not encoded as a conversation in which later questions see earlier answers. See [`examples/warehouse.json`](examples/warehouse.json) for all three kinds.

Rust callers can construct ordinal decisions with `Decision::ordinal(id, instruction, levels)` using typed `Level` values. `ComputeOptions::default()` uses the CLI defaults (2048 context, 256 batch and ubatch, four threads, flash attention off, automatic model loading, and no explicit GPU layer override).

### Scores and abstention

For native candidate logits `z`, L2S1 computes two separate quantities:

```text
option_probability[i] = exp(z[i] - logsumexp(candidate logits))
candidate_mass        = exp(logsumexp(candidate logits) - logsumexp(all vocabulary logits))
```

`option_probability` compares the supplied options. `candidate_mass` measures how much of the model's next-token probability belongs to those options at all. A high candidate-relative probability alone does not establish a reliable answer.

The default `DecisionPolicy` requires top-option probability at least **0.8**, candidate mass at least **0.05**, and no tied top candidates. Otherwise, the selected value is `null` and `abstention_reasons` explains why. Scores are still returned. Ordinal expected values are probability-weighted level values; they remain available when selection is withheld.

These are model scores, not universal probabilities of correctness. Partial top-k responses or model-generated numeric estimates do not satisfy the exact native evidence contract.

## Build

The pure Rust library supports validation, scoring, scalar calibration and worker ownership without native inference:

```sh
cargo test --locked
```

The inference backend and CLI support Linux (CPU or CUDA) and macOS (CPU or Metal). They require Rust with edition 2024 support, CMake, and a C++17 compiler. Metal builds require an Xcode toolchain with the Metal compiler. The `l2s1-llama-sys` workspace dependency builds llama.cpp and the matching native bridge together.

```sh
cargo build --release --locked --features llama
# CUDA toolkit required for GPU support:
cargo build --release --locked --features llama-cuda
# macOS with Metal:
cargo build --release --locked --features llama-metal
```

To produce independent CUDA builds for particular compute capabilities, run `scripts/build_cuda_arch.sh 86 89`. Each architecture gets a separate Cargo target directory and a `release/run-l2s1` launcher with its matching native libraries. The script requires `readelf` and a CUDA toolkit. The [sm_86 build and shared-state cache measurement](benchmarks/shared-state-cache-20260925/REPORT.md) were checked on an RTX 3080, including a fresh build and smoke on this PR branch; other architecture builds still need their own verification.

The architecture build script automatically uses `ccache` for C/C++/CUDA when installed. It respects an existing `L2S1_NATIVE_COMPILER_LAUNCHER`; set `L2S1_BUILD_CACHE=off` to disable automatic detection. To try Rust caching, set `RUSTC_WRAPPER=sccache` explicitly. In the measured independent `sm_86` builds, `sccache` had no Rust hits across separate Cargo target directories, so it is not enabled automatically. Keep `L2S1_CUDA_TARGET_ROOT` stable across repeated builds to reuse Cargo's local artifacts. An empty `ccache` can make the first build slower; see the [measured build-cache report](benchmarks/build-cache-20260925/REPORT.md) before adopting it for one-off builds.

The default CPU build uses CMake FetchContent to download and verify llama.cpp revision `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`; the first build needs network access. For an offline build or another revision, set `L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp`; the legacy `LLAMA_CPP_DIR` source override also works. `LLAMA_LIB_DIR` is no longer used. Validate custom revisions with the native contract tests. See [verification commands](VERIFICATION.md) and [native dependency details](crates/l2s1-llama-sys/README.md).

For an independent source release, publish `l2s1-llama-sys` before `l2s1`. The build embeds the native-library rpath for local Linux and macOS executables. A downstream crate can use `DEP_L2S1_LIBDIR` in its build script to set the rpath for its own executable. Prebuilt executables must ship matching native shared libraries with a portable loader path; swapping only `libllama.so` or `libllama.dylib` is unsupported.

### Dataset and benchmark tools

Data preparation, local benchmark orchestration, saved-prediction audits, and report generation use the repository-only Rust `l2s1-tools` binary. It does not link llama.cpp; inference commands launch the separately built `evaluate_jsonl` example. Model training and direct PyTorch probes remain Python workflows.

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools --help
```

See the [tool command map](crates/l2s1-tools/README.md) and each benchmark guide for arguments and evidence boundaries. Historical Python adapters remain available for artifact comparison and training imports.

Model files are supplied by the caller. Place an appropriate text chat/instruct GGUF under `models/` or another directory. The CLI does not download weights. Loading records checkpoint and runtime identities; a successful model load computes its file checksum once and caches it by file identity for subsequent loads. Set the absolute `L2S1_MODEL_HASH_CACHE_DIR` to relocate the cache, or remove its entries to force a fresh checksum.

## Inspect, validate and run

After building, inspect a model without reading a request:

```sh
./target/release/l2s1 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf --inspect
```

Check the actual request's template, candidate tokens, context usage, execution support and active artifact bindings without a forward pass:

```sh
./target/release/l2s1 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --preflight
```

Run the same request with either compatible model:

```sh
./target/release/l2s1 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json

./target/release/l2s1 \
  --model models/Qwen3-0.6B-Q8_0.gguf \
  --input examples/warehouse.json
```

The request and result schema stay the same. Each model uses its own tokenizer and prompt profile. Auto selection uses Qwen3's non-thinking profile for compatible dense Qwen3 checkpoints, Harmony final prefill for GPT-OSS, and the embedded GGUF Jinja template for other supported models.

CPU is the default. Select `--device cuda` or `--device metal` explicitly for GPU inference. An unavailable selected device or unsupported model produces an error; oversized inputs are rejected without truncation. `--context`, `--batch`, `--ubatch`, `--threads` and `--flash-attention off|auto|on` control requested compute settings. `--input -` reads stdin, ordinary results go to stdout, and native logs go to stderr. On macOS, build with `--features llama-metal` before selecting Metal. Native logs use `L2S1_LOG=warn` by default; set `error`, `info`, `debug`, or `off` to change verbosity.

Add `--diagnostics` to receive a separate envelope containing the normal response, model identity, prompt-token fingerprints, requested/effective execution mode, fallback reasons, calibration IDs and request-local timings. Ordinary `decide()` responses retain their existing shape. Request-stage failures under `--preflight` or `--diagnostics` have structured JSON and a nonzero exit status; model loading or malformed JSON can fail before that envelope.

## Rust integration

Enable the crate's `llama` feature to use `LlamaBackend`. A caller can keep its request unchanged while passing a different model path:

```rust
use std::path::Path;
use l2s1::{
    DecisionBackend, DecisionPolicy, DecisionRequest, DecisionResponse,
    llama::LlamaBackend,
};

fn decide_with_model(
    model: &Path,
    request: &DecisionRequest,
) -> l2s1::Result<DecisionResponse> {
    let mut backend = LlamaBackend::load(
        model, 2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    backend.decide(request)
}
```

For repeated requests, retain the backend instead of loading it for every call. `inspect()`, `preflight()` and `decide_detailed()` expose the corresponding inspection, validation and diagnostics APIs. `decide_batch()` accepts independent requests and preserves their result grouping.

`BackendWorker::spawn()` constructs a backend on its owner thread. Its factory can return the non-Send/non-Sync `LlamaBackend`; the native context never moves between threads. The worker bounds queued request count and serialized request size, reserves an operator-estimated memory budget, rejects a full queue immediately, and returns a `DecisionTicket` for admitted work. `close()` drains work and drops the backend on that same thread. Reservations control admission, not operating-system RSS. See [worker usage and lifecycle](MODEL_INTERCHANGEABILITY.md#bounded-ownership-and-scheduling).

On Metal, call `BackendWorker::close()` and wait for its owner thread to release `LlamaBackend` before process exit. Direct users should drop `LlamaBackend` before exit. This avoids llama.cpp Metal teardown assertions when a model remains loaded.

`BackendWorker::spawn_batched()` additionally collects requests under explicit request-count, model-input-token and collection-wait limits. With `LlamaBackend`, native batching requires selecting `ExecutionMode::Parallel`; other modes keep requests serial. Collection alone does not make inference parallel, and the existing parallel score-drift limits still apply.

### Optional prompt detail and answer-code mixtures

`--prompt-detail minimal` and `--code-rotation 0` preserve the existing prompt and remain the defaults. `typed` adds the decision kind, semantic option IDs, ordinal values and exact-comparison guidance. `typed-examples` also adds generic numerical interval examples before the input. These variants remain model-neutral, accept changing schemas and can change predictions and context usage; they are not measured accuracy guarantees.

```sh
./target/release/l2s1 --model models/Qwen3-0.6B-Q8_0.gguf \
  --input examples/warehouse.json --prompt-detail typed-examples --code-rotation 1
```

Rotation changes code assignment: displayed position `i` represents canonical option `(i + rotation) % option_count`. The backend accepts nonnegative rotations, reduces them modulo the option count, and returns scores in the original semantic order, including the original ordinal scale. Returned codes and token IDs describe the actual rotated assignment. The corresponding Rust setters are `set_prompt_detail(PromptDetail::TypedExamples)` and `set_code_rotation(1)?`. Changing either clears preparation caches and changes the prompt identity, so calibrations and heads must match that configuration. Default response JSON omits the added detail/rotation fields.

For multiple rotated passes, `score_semantic_mixture(&decision, &passes, &policy)` aligns results by semantic option ID and pools **full candidate probabilities**:

```text
q(y)                    = mean(candidate_mass[pass] * option_probability[pass][y])
mixture candidate_mass  = mean(candidate_mass[pass])
mixture option_probability[y] = q(y) / sum(q)
```

This retains the mass gate instead of setting it to one. It accepts at least two uncalibrated native passes; learned-head, calibrated and previously mixed results are rejected. The caller must use the same model, state, task and inference configuration, changing only code rotation. Mixture scores are not calibrated probabilities of correctness. The result is marked `semantic_probability_mixture_v1`; its `raw_logit` is `ln(q)`, code/token metadata represents the first pass, and token counts sum all passes. Additional passes consume additional inference time.

The [paired evaluation example](examples/evaluate_accuracy.rs) runs these variants on JSONL records containing `id`, optional `group`, and a `request` with one decision:

```sh
cargo run --release --locked --features llama --example evaluate_accuracy -- \
  --model models/Qwen3-0.6B-Q8_0.gguf --input cases.jsonl \
  --output /tmp/l2s1-accuracy-passes.jsonl \
  --prompt-details minimal,typed,typed-examples --all-rotations
```

The output must be new. The evaluator records native passes, mixture results, configuration identities and timing without reading answer labels. Evaluate task accuracy and acceptance coverage separately on held-out labels before selecting a variant.

## Model-specific identity and calibration

A `ModelIdentity` fingerprints the checkpoint, embedded template, effective prompt profile/version, runtime build and loaded libraries, active adapter/head, device label and compute/execution configuration. `preflight()` combines that identity with checks of the actual request. Capability inspection establishes available operations; labeled evaluation establishes task quality.

Optional scalar temperature calibration is bound to the model/configuration fingerprint and exact task signature, including option order and ordinal values. Changing a binding rejects the artifact instead of applying it to another model. Calibration preserves raw logits and base candidate mass, while acceptance probabilities and coverage may change.

```sh
cargo run --release --locked --example fit_calibration -- \
  fit-input.json task-temperature.json

./target/release/l2s1 \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json \
  --calibration task-temperature.json --diagnostics
```

The [calibration input schema and evaluation contract](MODEL_INTERCHANGEABILITY.md#scoped-scalar-calibration) describe measured raw-score records, independent source groups, held-out NLL/Brier and policy coverage checks. No trained calibration is bundled. Multiple task-scoped artifacts can be registered, but scalar calibration cannot be stacked with an output head.

A compatible GGUF LoRA can be loaded with `--lora`. A learned task-specific scorer can be loaded with `--output-head`; hidden-feature heads currently require Gemma4, and heads require their recorded fresh-execution configuration. These are optional specializations. See [LoRA training](DECISION_FINETUNE.md) and [output-head contracts](OUTPUT_HEAD.md).

## Execution and memory

| `--execution-mode` | Behavior | Status |
| --- | --- | --- |
| `fresh` | Evaluate every decision from an empty sequence state | Default |
| `prefix-reuse` | Reuse complete prefill batches from an exact common token prefix within one request | Opt-in; recurrent/hybrid memory falls back to fresh |
| `state-restore` | Save a common prefix's whole sequence state and restore it for later independent suffixes | Supported, opt-in; works with tested hybrid Bonsai GGUF |
| `parallel` | Batch independent questions into isolated sequences with shared-prefix prefill | Experimental; rejects recurrent/hybrid models and can change scores |

The default prompt layout is `legacy`. `--prompt-layout state-first` places shared state earlier and can expose longer reusable prefixes, but it also changes the prompt and can change predictions.

```sh
./target/release/l2s1 --model models/Qwen3-0.6B-Q8_0.gguf \
  --input examples/warehouse.json \
  --prompt-layout state-first --execution-mode state-restore \
  --snapshot-limit-bytes 268435456 --diagnostics
```

State restoration limits its snapshot buffer to 256 MiB by default and reports fresh fallback when a snapshot cannot be used. It requires full evidence transfer; use `state-restore` explicitly for recurrent/hybrid models because `prefix-reuse` still falls back to fresh on those models. `--parallel-width` bounds questions per parallel wave and increases context memory. Ordinary requests clear native KV state at request boundaries and after errors; snapshots never survive their native call. Neither snapshot limits nor worker reservations are whole-process memory limits.

`--parallel-context-dynamic` opts into sizing each parallel KV context from that wave's actual input tokens plus one batch of headroom. `--context` remains the per-question input limit. The context grows for larger later waves and retains its largest allocation at the same effective question count. Responses report `backend.parallel_context_tokens`; check decision equivalence on the target model because changing context size can change scores.

State copying has a cost and does not guarantee a speedup. The [Bonsai RTX 3060 validation](benchmarks/bonsai-state-restore-20260925/REPORT.md) records a real hybrid-model reuse and exact fresh-result parity on a fixed 16-decision fixture. Parallel execution has measured probability and top-choice differences on some checkpoints. Both remain explicit options; see [execution details](MODEL_INTERCHANGEABILITY.md#request-local-state-restoration) and [parallel execution](PARALLEL_EXECUTION.md).

### Optional preparation and evidence optimizations

For models larger than VRAM, CUDA loading accepts `--gpu-layers N` or
`--cpu-moe-layers N` to place part of the weights in CPU RAM. Placement is recorded
in compute identity and can change numerical scores. See
[CPU/GPU placement](MODEL_INTERCHANGEABILITY.md#cpugpu-placement).

For lower model-loading peak process RSS, use `--model-load-mode read`; see [loading behavior and measurement limits](MODEL_INTERCHANGEABILITY.md#model-loading-and-peak-host-rss).

Legacy prompts, fresh execution, full evidence transfer and disabled preparation caching remain the defaults. Existing repeated requests and fixed decision schemas continue to work; none of these options requires a fixed schema.

```sh
./target/release/l2s1 --model models/Qwen3-0.6B-Q8_0.gguf \
  --input examples/warehouse.json --evidence-transfer compact \
  --preparation-cache-bytes 8388608 --preparation-cache-entries 128
```

- **Preparation cache:** retain exact prepared prompts and answer-boundary candidate mappings within one loaded backend. `PreparationCacheConfig` limits entries per cache and divides one byte budget between them. It stores token preparation, not scores or KV state; new states and schemas use the normal preparation path. Repeated calls benefit most when the backend stays resident.
- **Compact evidence:** retain candidate logits and the full-vocabulary normalizer while avoiding the complete native-host-to-Rust logits copy. llama.cpp still computes the full vocabulary and makes it available on the host. This is not GPU-side reduction or output-head elimination. It supports fresh/prefix-reuse execution without a learned output head and has a distinct calibration identity.
- **Explicit shared state:** `backend.shared_state(state)?` borrows a backend configured for `PrefixReuse`. Repeated `session.decide(decisions)` calls may change IDs, instructions, option counts and decision kinds while sharing exact decoder prefixes over that immutable state. Session creation, errors and drop clear native state. Recurrent/hybrid models and output heads are rejected. This is a decoder session, not a separately trained state encoder.

See [optimization APIs and limits](MODEL_INTERCHANGEABILITY.md#optional-execution-optimizations). The [local benchmark harness](examples/benchmark_optimizations.rs) compares fresh, cached, compact and shared-state paths using mixed-schema warehouse questions, records raw scores and selection differences, and refuses to overwrite its output:

```sh
cargo run --release --locked --features llama --example benchmark_optimizations -- \
  --model models/Qwen3-0.6B-Q8_0.gguf --output /tmp/l2s1-optimizations.json
```

Repeat `--model` for additional checkpoints; use `--cuda` explicitly for GPU runs. The default benchmark uses three rounds, 1/4/16 questions and short/long synthetic states. It warms each path before timing, rotates path order and compares shared-state results to the same state-first prompt layout. Timings establish local workload behavior, not a general speedup or task accuracy.

## Compatibility and validation

A compatible checkpoint must be a decoder-only model supported by the linked runtime, have a renderable GGUF chat template that preserves the payload, and fit the chosen device/context. Decisions require at least two candidates; answer-code width grows automatically without an alphabet-derived count ceiling. The original path requires unique single-token continuations; multi-letter codes require stable, unique, prefix-free token sequences and support fresh or prefix-reuse execution with full evidence, without output heads, scalar calibration or feature export. Prompt length, answer-prefix length and available memory still bound real workloads. Compatibility is checked against actual model behavior rather than a general family-name promise.

Local conformance checks have covered SmolLM2, Qwen3, Gemma3, TinyLlama, Gemma4, a Qwen3.8 file with `qwen35` hybrid architecture, and GPT-OSS across CPU/CUDA configurations. Support remains checkpoint- and configuration-specific; use the [verification guide](VERIFICATION.md) for your model. Base models without suitable templates, encoder-only models and unverified multimodal configurations are outside the validated contract. The direct image path requires its own checkpoint and task validation.

```sh
cargo test --locked
cargo test --release --locked --features llama

L2S1_CONFORMANCE_MODELS=/path/to/model-a.gguf:/path/to/model-b.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/conformance.json \
  cargo test --release --locked --features llama \
  --test conformance -- --ignored --nocapture
```

The general test run skips model-dependent tests; invoke them explicitly with local checkpoints. Add `SKID_CUDA=1` to run conformance on CUDA. Tests do not download weights. Contract checks establish synthetic compatibility, not production accuracy. Keep generated conformance reports and benchmark artifacts in local output directories.

## Recorded model comparison

The September 23, 2026 JevBench matrix measured **22 GGUF checkpoints on all 231 public items** using the same frozen project build on an RTX 3060 12 GiB. All 5,082 predictions in the completed comparison runs were valid, with no inference errors or truncation. The 23 runtime configurations include one failed default GPT-OSS attempt and its successful CUDA Graphs-disabled recovery.

The original 22 matrix rows use identical request and evaluator hashes, fresh/legacy execution, context 8192, batch/ubatch 256, four threads and FlashAttention off, without reasoning-token generation, LoRA, an output head or learned calibration. The explicit GPT-OSS exception is marked below. Rows marked † are separate September 24 runs; ‡ is a separate September 23 run.

| Checkpoint | Argmax accuracy | Hard accuracy | Accepted wrong | Abstained / 231 | p50 / p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Gemma 4 31B Q4_K_M](https://huggingface.co/google/gemma-4-31B-it) † | 89.61% | 78.38% | 22 | 3 | 2287.40 / 24524.56 |
| Gemma 4 26B A4B UD-Q4_K_M ‡ | 84.85% | 70.27% | 29 | 11 | 787.01 / 8814.01 |
| Qwen3.5-4B-Q8_0 | 79.65% | 61.26% | 9 | 93 | 86.88 / 1137.30 |
| Qwen3.5-9B-Q4_K_M | 77.92% | 58.56% | 13 | 64 | 130.55 / 1697.14 |
| Qwen3.5-9B-Q8_0 | 77.92% | 56.76% | 14 | 66 | 124.68 / 1613.79 |
| gemma-4-E4B-it-Q4_K_M | 77.49% | 55.86% | 30 | 37 | 86.62 / 1186.47 |
| gemma-4-E4B-it-Q8_0 | 76.62% | 54.05% | 32 | 37 | 85.04 / 1146.21 |
| Qwen3.5-4B-Q4_K_M | 76.19% | 55.86% | 10 | 91 | 88.47 / 1175.18 |
| [Ternary Bonsai 27B Q2_g64](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf) † | 75.32% | 53.15% | 15 | 80 | 187.15 / 2462.18 |
| Qwen3.8-27B-UD-IQ2_XXS | 73.16% | 47.75% | 18 | 84 | 414.61 / 5460.22 |
| Qwen3-8B-Q8_0 | 71.43% | 48.65% | 56 | 15 | 121.45 / 1817.94 |
| [Bonsai 27B Q1_0](https://huggingface.co/prism-ml/Bonsai-27B-gguf) † | 71.00% | 46.85% | 14 | 91 | 184.51 / 2489.64 |
| gemma-4-E2B-it-Q8_0 | 67.97% | 43.24% | 58 | 22 | 46.79 / 701.11 |
| Qwen3-4B-Q8_0 | 65.80% | 44.14% | 63 | 29 | 85.26 / 1349.28 |
| Ministral-3-8B-Instruct-2512-Q4_K_M | 65.37% | 47.75% | 23 | 93 | 441.43 / 2399.70 |
| gpt-oss-20b-Q4_K_M (CUDA Graphs off) | 64.94% | 48.65% | 31 | 82 | 181.93 / 2262.28 |
| Qwen3.5-2B-Q8_0 | 62.34% | 48.65% | 15 | 133 | 40.99 / 513.03 |
| gemma-3-4b-it-Q8_0 | 59.74% | 36.04% | 88 | 9 | 74.71 / 879.92 |
| Phi-4-mini-instruct.Q8_0 | 56.71% | 42.34% | 30 | 115 | 70.50 / 971.95 |
| Qwen3.5-0.8B-Q8_0 | 51.95% | 42.34% | 16 | 183 | 27.93 / 350.53 |
| SmolLM3-3B-Q8_0 | 46.32% | 31.53% | 41 | 127 | 71.47 / 884.76 |
| Llama-3.2-3B-Instruct-Q8_0 | 43.72% | 30.63% | 46 | 147 | 68.18 / 884.75 |
| gemma-3-1b-it-Q8_0 | 38.96% | 28.83% | 121 | 30 | 25.06 / 309.24 |
| tinyllama-1.1b-chat-v1.0.Q4_K_M | 33.77% | 36.04% | 1 | 230 | 30.08 / 702.65 |
| Qwen3-0.6B-Q8_0 | 31.60% | 31.53% | 129 | 41 | 34.01 / 445.12 |
| SmolLM2-135M-Instruct-Q8_0 | 30.30% | 29.73% | 8 | 211 | 14.88 / 253.65 |

The † rows used the same public dataset (SHA-256 `dc3995d8ae1e2fc8e81ce38431add509eb8bb39b85aadfd0c7c32079382dde51`) and byte-identical 231 request JSONL (SHA-256 `6f96c4fc2b924ec0bef4aaa94c25b2456522909fdbebffe0c0df8fa44e5c2faa`). Candidate-argmax scores were 164/231 for Bonsai Q1_0, 174/231 for Ternary Bonsai Q2_g64 and 207/231 for Gemma 4 31B Q4_K_M. All 693 predictions were valid, with no inference errors or truncation; independent recounts reproduced each tier total. The Gemma 4 31B run also reproduced every candidate probability and policy value from its earlier validated run. The default policy accepted 140/151/228 decisions respectively, of which 126/136/206 were correct.

These runs used fresh/legacy execution, context 8192, batch/ubatch 256, and no LoRA, output head or learned calibration. The Bonsai runs used an RTX 3080 with full GPU offload and four threads; Gemma 4 31B used an RTX 3060 with 24 GPU layers, read-mode loading and eight threads. Evaluator binaries also differed, so the † latency figures are not a controlled speed comparison with each other or the original matrix. Their local, gitignored evidence is in `results/bonsai-27b-20260924/` and `results/jevbench-gemma31-rust-20260924/`; these artifacts are not included in the repository.

The ‡ 26B run used the same 231 public task IDs on an RTX 3060 with 18 CPU expert layers and eight threads. It used a different request serialization and evaluator build from the original matrix and † reruns, so the table combines task scores from distinct runs, not a controlled latency comparison. All 231 decisions completed without errors or truncation; the default policy accepted 220, including 191 correct. See the [26B measurement details](JEVBENCH.md#gemma-4-26b-a4b-separate-run-2026-09-23). Its raw evidence is local and gitignored.

Qwen3.5-4B Q8_0 had the highest argmax accuracy in the original 22-checkpoint matrix: 184/231 (79.65%), including 68/111 Hard items (61.26%). Its default policy accepted 138 decisions: 129 correct and 9 wrong, for 93.48% accepted accuracy at 59.74% coverage. Qwen3.5-9B Q4_K_M covered 72.29% with 13 accepted errors; Gemma4 E2B covered 90.48% with 58 accepted errors. Accuracy before abstention, accepted accuracy and coverage answer different questions.

GPT-OSS 20B Q4_K_M exhausted GPU memory in `cudaGraphInstantiate` after 129 predictions. With `GGML_CUDA_DISABLE_GRAPHS=1`, it completed all 231 at 64.94% accuracy and a sampled peak of 11,901 MiB. Its first 129 probability distributions were identical to the failed run. Keep this runtime exception when reproducing its result.

After model downloads finished, complete reruns of Qwen3.5-4B Q8_0 and Gemma4 E2B produced identical probabilities for every item. Their confirmation p50/p95 latencies were 86.99/1139.49 ms and 46.08/699.14 ms respectively; the table retains the original matrix timings.

These are public-subset, local inference measurements, not an official full-suite score, rank or production validation. Latency excludes loading and warmup; small models may exceed their training context. For the original 22-checkpoint matrix, raw predictions, model/source hashes, memory samples, failed-attempt evidence and independent accuracy/Brier/ECE recount are in the local, gitignored `results/jevbench-matrix-20260923/` directory; they are not included in this repository. The measured source snapshot is kept with those artifacts, and later working-tree optimizations are outside this frozen comparison. See the [evaluation method](JEVBENCH.md).

## Additional recorded model measurements

The [complete-label intent evaluation](INTENT_BENCHMARK.md#measured-results) used 200 BANKING77 English and 200 MASSIVE Korean examples per checkpoint on the RTX 3060. Each request included all 77 or 60 official labels. Accuracy below is raw top-choice accuracy before the abstention policy; the p50 values are local inference milliseconds. The four original configurations and separate 26B run completed both samples without inference errors or truncation.

| Checkpoint | BANKING77 correct / 200 | MASSIVE Korean correct / 200 | English / Korean p50 ms |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 123 (61.5%) | 103 (51.5%) | 214.15 / 157.52 |
| Gemma 4 E4B Q8_0 | 130 (65.0%) | 143 (71.5%) | 371.72 / 277.05 |
| Gemma 4 E4B Q4_K_M | 130 (65.0%) | 138 (69.0%) | 383.61 / 287.00 |
| Gemma 4 26B A4B UD-Q4_K_M ‡ | 152 (76.0%) | 156 (78.0%) | 2880.52 / 2134.52 |
| Qwen3-8B Q8_0 | 111 (55.5%) | 98 (49.0%) | 877.64 / 593.45 |

The ‡ 26B row used 18 CPU expert layers, eight threads, and a separate evaluator run. Its p50 values are first-call medians; immediate repeated calls with request-local cache measured 669.15 / 656.70 ms and preserved all 400 result evidence objects. These timings are not controlled comparisons with the four original rows.

The [CPU/GPU cache check](LAYA_BENCHMARK.md) additionally measured one larger checkpoint with 24 GPU layers and eight CPU threads. Times below cover two typed cases and their immediate repeats.

| Checkpoint | Batch 256 fresh | Batch 64 fresh / prefix reuse | Batch 128 fresh / prefix reuse |
| --- | ---: | ---: | ---: |
| Gemma 4 31B Q4_K_M | 41.677 s | 108.552 / 69.904 s | 62.438 / 42.770 s |

Both same-batch reuse comparisons preserved all 20 probability vectors and selections. This small repeated-input check measures cache behavior, not labeled task accuracy. Neither cached setting beat the batch-256 fresh baseline.

These intent and cache measurements use different tasks and settings from the JevBench matrix above. Their detailed protocols and local-artifact limits are in the linked guides.

## Further documentation

| Topic | Document |
| --- | --- |
| Identity, preflight, calibration, diagnostics and worker API | [Model interchangeability](MODEL_INTERCHANGEABILITY.md) |
| Build and model-specific validation | [Verification guide](VERIFICATION.md) |
| Prefix reuse and parallel execution | [Prefix algorithm](SEMIF_ALGORITHM.md), [parallel execution](PARALLEL_EXECUTION.md) |
| Evaluation methods | [Synthetic benchmark](BENCHMARK.md), [AG News](KAGGLE_BENCHMARK.md), [JevBench](JEVBENCH.md), [Laya/Jev tasks and CPU caching](LAYA_BENCHMARK.md) |
| Optional model/task adaptation | [Decision fine-tuning](DECISION_FINETUNE.md), [output heads](OUTPUT_HEAD.md) |

## License

Project source is [MIT-licensed](LICENSE). Preserve the dependency notices in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) when distributing the native components. Model weights have their own licenses; see [LICENSING.md](LICENSING.md) for the source and model distinction.

Model weights, local build/results directories and the separate `web/` directory are excluded from the Cargo package. No weights are bundled with the source or tests.
