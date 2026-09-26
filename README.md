# L2S1 — LLM to System 1

**Turn local GGUF model scores into typed decisions.**

English · [한국어](README.ko.md) · [Documentation](docs/README.md) · [Model results](docs/MODEL_RESULTS.md)

L2S1 is a Rust library and CLI for binary, choice, and ordinal decisions with local chat models. Give it JSON state, a question, and candidate criteria; receive a typed value, model scores, and an explicit abstention when the acceptance policy is not met.

Use it to classify messages, route requests, check conditions, or assign ordered levels. Questions and candidate IDs are supplied by your application at request time. You can change the compatible GGUF model while keeping the same request and result types.

## Quick start

You need a current stable Rust toolchain with edition 2024 support, CMake 3.24+, a C++17 compiler, and a compatible chat/instruct GGUF. The first native build downloads the pinned llama.cpp source. Model weights are supplied separately.

```sh
git clone https://github.com/LuticaCANARD/L2S1.git
cd L2S1
cargo build --release --locked --features llama --bin l2s1

./target/release/l2s1 \
  --model /path/to/chat-model.gguf \
  --input examples/warehouse.json
```

The [warehouse request](examples/warehouse.json) asks three independent questions about the same shipment: storage zone, cold-chain requirement, and dispatch priority. Its first decision looks like this:

```json
{
  "state": { "storage_requirement": "chilled" },
  "decisions": [{
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
  }]
}
```

A [recorded Gemma 4 CUDA run](examples/warehouse.gemma4.cuda.output.json) returned this result for `storage_zone` (excerpt):

```json
{
  "id": "storage_zone",
  "value": { "type": "choice", "selected": "chilled" },
  "abstention_reasons": []
}
```

The complete CLI response includes backend information, policy, candidate scores, probability mass, and token usage. Predictions and scores depend on the model and configuration; validate them on labeled examples for your task.

## Features

- **Typed outputs.** Binary checks, categorical choices, and ordered levels share one request format. Semantic option IDs stay consistent across compatible models.
- **Direct local scoring.** Read scores at the model's assistant answer boundary. Larger candidate sets use complete answer-code likelihoods when codes span multiple tokens.
- **Explicit abstention.** Keep the scores and explain why a selection was withheld. The default policy checks both relative option probability and full-vocabulary candidate mass.
- **Text and images.** Use a supported vision model with its matching `mmproj` GGUF for still-image decisions.
- **Rust, CLI, and HTTP.** Keep a backend resident in your application, run JSON from a file or stdin, or serve the versioned decision API.
- **Inspectable execution.** Inspect model identity, preflight requests, and record diagnostics. Optional caching, state restoration, parallel execution, LoRA, and task-scoped calibration have explicit contracts.

## Decision types and scores

| Kind | You provide | Local result |
| --- | --- | --- |
| `binary` | False and true criteria | Boolean or `null`, plus `p_true` |
| `choice` | Candidate IDs and criteria | Selected ID or `null` |
| `ordinal` | Ordered levels with increasing numeric values | Selected level ID or `null`, plus expected value |

Each decision is evaluated independently. All kinds return candidate scores and abstention reasons. Up to 26 candidates use `A`–`Z`; larger sets use fixed-width codes such as `AA`–`ZZ`. Tokenization and context limits are checked against the selected model.

Two scores determine default acceptance:

- `option_probability`: the candidate's probability relative to the supplied candidates.
- `candidate_mass`: the probability assigned to the candidates within the complete vocabulary or answer-code paths.

The default policy requires top-option probability **≥ 0.8**, candidate mass **≥ 0.05**, and no tied top candidates. Otherwise, the selected value is `null`. These are model scores; they are not calibrated probabilities of correctness. Optional calibration is bound to a specific model, configuration, and task.

See the [decision contract](docs/GUIDE.md#the-decision-contract) for formulas, answer codes, and validation rules.

## Backends and hardware

| Backend | Models and input | Hardware | Evidence |
| --- | --- | --- | --- |
| `llama` / `llama-cuda` / `llama-metal` | Compatible GGUF chat models; vision needs a matching projector | CPU, NVIDIA CUDA, Apple Metal | Local model scores |
| `wgpu` | Gemma 4 text GGUF, optionally with a matching vision projector | Native wgpu GPU adapter | Local model scores |
| `openrouter` | Provider models supporting the requested modality | Remote API | Selection only; no probabilities |

The default features are empty. Enable `llama` for the local CLI. Models must be supported by the pinned runtime, have a usable chat template, and fit the selected device and context. The wgpu adapter is specific to Gemma 4. Model changes can change predictions, latency, tokenization, and calibration.

For local GPU inference, build the matching feature and select the device:

```sh
# NVIDIA CUDA: requires the CUDA toolkit.
cargo build --release --locked --features llama-cuda --bin l2s1
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --device cuda --input examples/warehouse.json

# macOS Metal: requires an Xcode toolchain with the Metal compiler.
cargo build --release --locked --features llama-metal --bin l2s1
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --device metal --input examples/warehouse.json
```

CPU is the default. An explicitly requested GPU must be available. The first build fetches the native source; offline builds can set `L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp`. See the [build guide](docs/GUIDE.md#build) for native libraries, packaging, and separate CUDA architecture builds.

For [wgpu](docs/GUIDE.md#optional-wgpu-backend), use the separate `l2s1-wgpu` executable. For [OpenRouter](docs/GUIDE.md#openrouter-adapter), use `l2s1-openrouter` and supply `OPENROUTER_API_KEY`. OpenRouter responses use the shared typed HTTP envelope with `selection_only` evidence and no local probability policy.

## Inspect and run

```sh
# Inspect the loaded model and capabilities.
./target/release/l2s1 --model /path/to/chat-model.gguf --inspect

# Validate the actual request without a forward pass.
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --input examples/warehouse.json --preflight

# Run with execution diagnostics.
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --input examples/warehouse.json --diagnostics
```

`--input -` reads stdin. Results go to stdout; native logs go to stderr. Inputs that exceed the configured context are rejected without truncation. See [inspection and diagnostics](docs/GUIDE.md#inspect-validate-and-run) for compute settings and structured failures.

## HTTP and image input

Start a local server:

```sh
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --listen 127.0.0.1:8080
```

From another terminal:

```sh
curl -sS -H 'Content-Type: application/json' \
  --data-binary @examples/warehouse.json \
  http://127.0.0.1:8080/v1/decisions
```

The server exposes `POST /v1/decisions`, `GET /v1/capabilities`, and `GET /healthz`. HTTP responses carry an API version, request ID, backend, policy, and typed results with evidence. Bind to loopback or use an authenticated reverse proxy for remote access.

For a still image, load the vision model and its matching projector:

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/projector.gguf --image photo.jpg \
  --input examples/warehouse.json
```

HTTP image requests use named `media` and per-decision `media_ids`. Local backends accept one image per decision. See the [image and HTTP contract](docs/GUIDE.md#direct-image-input-and-http-api) for payloads, limits, and backend-specific behavior.

## Use from Rust

Enable the `llama` feature on the `l2s1` dependency. Load once and keep the backend for repeated requests:

```rust
use std::path::Path;
use l2s1::{
    Decision, DecisionBackend, DecisionKind, DecisionPolicy, DecisionRequest,
    Level, OptionSpec, llama::LlamaBackend,
};
use serde_json::{Map, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = DecisionRequest {
        state: Value::Object(Map::from_iter([
            ("shipment_id".into(), Value::from("BOX-103")),
            ("storage_requirement".into(), Value::from("chilled")),
            ("hours_until_dispatch".into(), Value::from(4)),
        ])),
        decisions: vec![
            Decision {
                id: "storage_zone".into(),
                instruction: "Select the storage zone that matches the shipment's storage_requirement.".into(),
                kind: DecisionKind::Choice {
                    options: vec![
                        OptionSpec {
                            id: "ambient".into(),
                            criterion: "The shipment requires ambient storage.".into(),
                        },
                        OptionSpec {
                            id: "chilled".into(),
                            criterion: "The shipment requires chilled storage.".into(),
                        },
                        OptionSpec {
                            id: "frozen".into(),
                            criterion: "The shipment requires frozen storage.".into(),
                        },
                    ],
                },
            },
            Decision {
                id: "cold_chain_required".into(),
                instruction: "Does this shipment need temperature-controlled storage? Chilled and frozen shipments do; ambient shipments do not.".into(),
                kind: DecisionKind::Binary {
                    false_label: "No temperature control is required.".into(),
                    true_label: "Temperature control is required.".into(),
                },
            },
            Decision {
                id: "dispatch_priority".into(),
                instruction: "Choose the priority using hours_until_dispatch and the exact thresholds in the levels.".into(),
                kind: DecisionKind::Ordinal {
                    levels: vec![
                        Level {
                            id: "low".into(),
                            criterion: "More than 24 hours remain until dispatch.".into(),
                            value: 0.0,
                        },
                        Level {
                            id: "medium".into(),
                            criterion: "More than 6 hours and at most 24 hours remain until dispatch.".into(),
                            value: 1.0,
                        },
                        Level {
                            id: "high".into(),
                            criterion: "At most 6 hours remain until dispatch.".into(),
                            value: 2.0,
                        },
                    ],
                },
            },
        ],
    };
    request.validate()?;
    let mut backend = LlamaBackend::load(
        Path::new("/path/to/chat-model.gguf"),
        2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    let response = backend.decide(&request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
```

Add `serde_json` to your dependencies. The request is constructed directly in Rust; no input file is needed. A model-free [runnable example](examples/warehouse.rs) constructs and validates the same request with `cargo run --locked --example warehouse`. Use `BackendWorker` for dedicated-thread ownership and bounded admission. On Metal, release the backend before process exit; worker users should call `close()` and wait for its owner thread. See [Rust integration](docs/GUIDE.md#rust-integration) for lifecycle and native linking details.

## Execution modes and vision throughput

The llama.cpp backend supports four execution modes:

| Mode | Use |
| --- | --- |
| `fresh` | Independent evaluation from empty sequence state; default |
| `prefix-reuse` | Reuse an exact common token prefix within one request |
| `state-restore` | Snapshot and restore a common prefix before independent suffixes |
| `parallel` | Batch independent questions into isolated sequences |

For GPU vision workloads, `--vision-optimized` enables four decoder streams, dynamic context reservation, Flash Attention, compact evidence, bounded preparation caching, and identical-image projector reuse:

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/projector.gguf --device cuda --vision-optimized \
  --image photo.jpg --input examples/warehouse.json
```

Parallel execution and the vision profile are supported features with a different numerical execution path from `fresh`: scores and selections can change. Parallel mode rejects recurrent/hybrid models, and parallel vision supports at most 26 options. The vision profile requires CUDA or Metal and compatible GPU kernels; its Metal performance remains unverified. Validate task quality and acceptance coverage on your checkpoint. Model-dependent tests are opt-in and do not download weights. See [execution and memory](docs/GUIDE.md#execution-and-memory), [optimized vision](docs/GUIDE.md#optimized-vision), and [verification](docs/VERIFICATION.md).

## Recorded measurements

| Study | Recorded scope | Report |
| --- | --- | --- |
| JevBench public subset | Original matrix: 22 checkpoints × 231 items; 5,082 valid predictions | [Model results](docs/MODEL_RESULTS.md), [method](docs/JEVBENCH.md) |
| Intent classification | 77 English labels and 60 Korean labels; 200 examples per language per checkpoint | [Intent benchmark](docs/INTENT_BENCHMARK.md) |
| Vision decisions | Still-image classification and execution-mode studies | [Vision benchmark](docs/VISION_BENCHMARK.md), [TrashNet study](benchmarks/trashnet-vision-20260925/REPORT.md) |

These are recorded local experiments at their stated revisions, hardware, and settings. Report accuracy, accepted accuracy, and coverage separately. Some raw benchmark artifacts remain local and gitignored; the reports identify their locations and reproduction procedures. The JevBench public subset is not an official full-suite score or rank.

## Documentation

For AI agents, use the portable [L2S1 skill](skills/l2s1/SKILL.md) and optional
stdio MCP adapter. MCP exposes documentation, typed request validation and the
resident HTTP backend's decisions. See [agent setup](docs/AGENT_INTEGRATION.md).

| Topic | Read |
| --- | --- |
| Build, requests, Rust API, HTTP, runtime options | [Guide](docs/GUIDE.md) |
| Model identity, preflight, calibration, worker ownership | [Model interchangeability](docs/MODEL_INTERCHANGEABILITY.md) |
| Model-specific test commands | [Verification](docs/VERIFICATION.md) |
| Prefix reuse and parallel execution | [Prefix algorithm](docs/SEMIF_ALGORITHM.md), [parallel execution](docs/PARALLEL_EXECUTION.md) |
| Learned specialization | [LoRA training](docs/DECISION_FINETUNE.md), [output heads](docs/OUTPUT_HEAD.md) |
| Dataset preparation and report tools | [l2s1-tools](crates/l2s1-tools/README.md) |
| Recorded model comparisons and their limits | [Model results](docs/MODEL_RESULTS.md) |

## Repository and development

| Path | Purpose |
| --- | --- |
| [`src/decision.rs`](src/decision.rs) | Typed requests, results, shared scoring, and policy |
| [`src/llama.rs`](src/llama.rs), [`src/wgpu.rs`](src/wgpu.rs) | Local inference backends |
| [`src/http.rs`](src/http.rs), [`src/http/contract.rs`](src/http/contract.rs) | HTTP server and versioned wire contract |
| [`crates/l2s1-llama-sys/`](crates/l2s1-llama-sys) | Pinned llama.cpp build and native bridge |
| [`crates/l2s1-tools/`](crates/l2s1-tools) | Dataset and benchmark tools |
| [`examples/`](examples) | Requests, saved outputs, and integration examples |
| [`docs/`](docs/README.md) | Detailed guides, design notes, and evaluation reports |
| [`web/`](web) | Svelte documentation site |

Run the pure Rust checks with `cargo test --locked`. Native and model-specific checks are documented in [VERIFICATION.md](docs/VERIFICATION.md). For a bug report, open a [GitHub issue](https://github.com/LuticaCANARD/L2S1/issues) with the command, checkpoint/quantization, runtime/device, and error. Keep English and Korean README changes aligned when submitting documentation updates.

## License

L2S1 source is [MIT-licensed](LICENSE). Model weights have their own licenses and are not bundled. Preserve the [third-party notices](THIRD_PARTY_LICENSES.txt) when distributing native components; see [LICENSING.md](docs/LICENSING.md).
