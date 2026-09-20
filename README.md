# skid-desion

A Rust library and CLI for typed decisions from local GGUF chat models. It reads next-token logits for option codes A–Z and returns structured choice, binary, or ordinal results. Option IDs remain separate from model output codes. No additional head training or generated explanation is required.

The backend supports multiple model families through the chat template embedded in each GGUF. Qwen3 retains its existing non-thinking profile; other models use llama.cpp's Jinja engine with the model's own BOS and EOS tokens.

## Build

You need Rust, a C++17 compiler, and a matching llama.cpp source checkout and shared-library build. Enable the `llama` feature for inference and the CLI; the default library feature set only includes request validation and scoring.

```sh
export LLAMA_CPP_DIR=/path/to/llama.cpp
export LLAMA_LIB_DIR="$LLAMA_CPP_DIR/build-cuda/bin"

cargo test --locked
cargo build --locked --features llama
```

The source checkout must include `common/jinja`, its JSON and Unicode helpers, and `vendor/nlohmann`. The Rust build compiles the C ABI bridge and these template helpers; it does not build libllama or its CPU/CUDA backends. Headers, helper sources, and shared libraries must come from the same llama.cpp version. The version used for verification is recorded in [VERIFICATION.md](VERIFICATION.md).

Without environment overrides, the build looks for `../llama.cpp` and its `build-cuda/bin` directory. Set both variables if your checkout or library build lives elsewhere. A CPU-only llama.cpp build is also usable; point `LLAMA_LIB_DIR` to its shared-library directory.

## Models

Download a text chat/instruct GGUF from Hugging Face, put it in `models/` or another local directory, and pass its path with `--model`. The CLI does not download models automatically and does not load Transformers `.safetensors` checkpoints directly.

| Model | GGUF file | Prompt selection |
| --- | --- | --- |
| [Qwen3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF) | `Qwen3-0.6B-Q8_0.gguf` | Qwen3 non-thinking |
| [SmolLM2 135M Instruct](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF) | `SmolLM2-135M-Instruct-Q8_0.gguf` | Embedded Jinja / ChatML |
| [Gemma 3 1B IT](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) | `gemma-3-1b-it-Q8_0.gguf` | Embedded Jinja / Gemma |
| [Gemma 4 E2B Instruct](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF) | `gemma-4-E2B-it-Q8_0.gguf` | Embedded Jinja / Gemma 4 (text only) |
| [TinyLlama 1.1B Chat](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | Embedded Jinja / Zephyr-style |

Qwen3, SmolLM2, Gemma 3 1B IT Q8_0, and Gemma 4 E2B Instruct Q8_0 pass the CPU and CUDA native suites. This does not imply identical decisions across CPU and CUDA. TinyLlama passes the CPU suite and runs on CUDA, but its CUDA batch-size consistency check fails; use CPU for its fully passing configuration. See the measured difference in [VERIFICATION.md](VERIFICATION.md#tinyllama-cuda-limitation).

The SmolLM2 and TinyLlama files above are community GGUF conversions. Their pinned revisions, hashes, and exact validation scope are listed in [VERIFICATION.md](VERIFICATION.md). These checkpoints are compatibility examples, not task-accuracy recommendations. Gemma 4 E2B Q8_0 occupies about 4.97 GB on disk; the text-only path does not load its optional multimodal projector or MTP companion.

For example, download the pinned SmolLM2 checkpoint:

```sh
mkdir -p models
curl --fail --location \
  https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q8_0.gguf \
  --output models/SmolLM2-135M-Instruct-Q8_0.gguf
```

Verify its SHA256 against the verification record before using it for reproducible runs. Model weights and build artifacts are excluded from version control.

### Compatibility requirements

A model must meet all of the following:

- Its decoder-only architecture and quantization are supported by the linked libllama.
- Its GGUF contains a chat template that the bundled Jinja engine can render for a text user message and an assistant generation prefix.
- Its template preserves the decision payload as one contiguous segment.
- Every requested option code is a unique, stable, single-token continuation at the actual assistant boundary.
- Its weights, context, and working buffers fit the selected device.

This allows additional Llama, Gemma, Phi, Mistral, and other chat-model variants without adding an architecture allowlist. It does not establish compatibility for every checkpoint in those families. Synthetic template tests cover their common role-delimiter patterns; only the checkpoints explicitly listed in the verification record have real-model evidence.

Missing templates, rendering failures, unsupported architectures, and unstable candidate tokenization return errors. Base models without chat templates, encoder/embedding models, multimodal inputs, and multi-token candidate scoring are not supported. Models that require reasoning before answering may have low candidate mass: `enable_thinking=false` only affects templates that honor that variable.

## Run

```sh
cargo run --locked --features llama -- \
  --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json \
  --device cpu
```

Switch models by changing `--model`:

```sh
cargo run --locked --features llama -- \
  --model models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
  --input examples/warehouse.json \
  --device cuda
```

CPU is the default. Explicit CUDA requests fail if no CUDA device is available or loading fails; there is no automatic CPU fallback. Use `--context`, `--batch`, and `--threads` to control resources. Oversized input is rejected without truncation.

`--input -` reads JSON from stdin. Results are written to stdout; model logs and errors go to stderr.

### Prompt profiles

| `--prompt-profile` | Behavior |
| --- | --- |
| `auto` (default) | Uses the legacy non-thinking profile for dense Qwen3 with `enable_thinking` metadata; otherwise renders the embedded Jinja template. |
| `model` | Renders the embedded Jinja template with the model's BOS/EOS strings, `add_generation_prompt=true`, and `enable_thinking=false`. |
| `qwen3` | Requires the dense Qwen3 profile; rejects incompatible models. |

The generic profile places the decision instructions and JSON payload in one user message, including for templates that reject a separate system role. Template control tokens are parsed as special tokens. Input data containing special-token strings is tokenized as ordinary text. This separation does not solve natural-language prompt injection.

Required BOS tokens are inserted once, including when the template already supplies one. Candidate token IDs are derived from the assistant continuation, rather than isolated letters, to account for tokenizer differences such as SentencePiece.

Library callers can use `LlamaBackend::load(...)` for automatic selection or `LlamaBackend::load_with_profile(..., PromptProfile::Model)` for an explicit profile. The public `compile_prompt(...)` function remains the legacy Qwen3 prompt helper.

## Results

- `choice`: the selected option ID, or `null` on abstention.
- `binary`: candidate-relative `p_true` and a boolean decision, or `null` on abstention.
- `ordinal`: the expected value over the supplied levels and the selected level ID, or `null` on abstention.
- `scores[].option_probability`: probabilities normalized over candidate codes only, not probabilities of correctness.
- `candidate_mass`: the total probability assigned to candidate tokens by the full-vocabulary softmax; a format diagnostic, not an accuracy guarantee.
- `entropy_confidence`: concentration of the candidate distribution, not an estimated success rate.
- `calibration_id`: currently always `null`; learned calibration is not implemented.
- `abstention_reasons`: low candidate mass, low top-candidate probability, or tied candidates. Scores remain available on abstention.

Default abstention thresholds of 0.8 for top-candidate probability and 0.05 for candidate mass are initial policy settings, not validated accuracy standards. Adjust them with `--min-top-probability` and `--min-candidate-mass`. Evaluate accuracy, abstention rate, and sensitivity to option order on a labeled dataset before automating decisions.

Backend metadata includes the model path, model description, architecture, resolved prompt profile, prompt version, and offload information. Pin the model SHA256 and llama.cpp build for reproducible deployment. Scores from different models are not calibrated for direct comparison.

## Architecture and limits

`DecisionBackend::decide` is the public inference boundary. `decision.rs` handles request validation, typed results, stable scoring, and abstention. `prompt.rs` constructs decision payloads and separates trusted template segments from input data. `llama.rs` owns model handles and inference calls. `native/bridge.cpp` adapts libllama's C API, while `native/chat.cpp` renders embedded Jinja templates.

Questions run sequentially. Each question clears KV memory, prefills long prompts in chunks, and copies only the final logits into Rust-owned memory. Multi-sequence batching and prefix caching are not implemented. Each decision supports 2–26 options.

FFI handles have one Rust owner and are neither `Send` nor `Sync`. Multi-request servers need an explicit worker/model/context strategy. Calibration, custom heads, and explanation generation are not implemented.

## Verification

```sh
cargo fmt --check
cargo test --locked
cargo test --locked --features llama
cargo clippy --locked --all-targets --features llama -- -D warnings

# Run explicitly for each checkpoint; omit SKID_CUDA for CPU.
SKID_MODEL=/absolute/path/to/model.gguf SKID_CUDA=1 \
  cargo test --locked --features llama --test native -- --ignored
```

Default tests cover scoring and typed request handling without model weights. Enabling `llama` also tests Jinja rendering, role boundaries, EOS substitution, non-thinking template variables, and rejection of malformed templates.

The ignored native test uses a real checkpoint to check request isolation, chunk-size consistency, all 26 candidate codes, profile validation, and rejection of oversized input. These checks establish local integration behavior, not business accuracy or throughput. See [VERIFICATION.md](VERIFICATION.md) for recorded results and model provenance.

## License

The project source is available under the [MIT license](LICENSE). The reviewed llama.cpp/Jinja and nlohmann/json code is MIT-licensed; preserve the applicable copyright and permission notices when distributing it. [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) records the reviewed dependency notices, including the additional Unicode license.

Model weights have their own licenses. The tested Gemma 3 checkpoint is subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms), not this project's MIT license. Gemma 4 instead uses [Apache 2.0](https://ai.google.dev/gemma/apache_2); its weights retain that license even when used with this MIT application. Cargo packages explicitly exclude `models/`, `target/`, and `results/`. See [LICENSING.md](LICENSING.md) for the reviewed scope and source, binary, and model-distribution distinctions.

## Inference performance test

The opt-in test in `tests/performance.rs` measures an existing local GGUF using the English warehouse example. It never downloads or bundles a model. Build libllama first and set the build paths described above.

```sh
SKID_MODEL=/absolute/path/to/chat-model.gguf \
SKID_CUDA=1 \
SKID_PERF_ITERATIONS=20 \
SKID_PERF_WARMUP=2 \
SKID_PERF_OUTPUT=results/performance/model.cuda.json \
  cargo test --release --locked --features llama --test performance \
  model_inference_performance -- --ignored --nocapture
```

Use `SKID_CUDA=0` for CPU. Defaults are five measured requests, one additional warmup, context 2048, batch 256, and four threads. Override resources with `SKID_CONTEXT`, `SKID_BATCH`, and `SKID_THREADS`. The first request is measured separately and is always excluded from the steady-state samples.

The JSON report includes model load time, first-request latency, individual request samples, mean/p50/p95 latency, requests/second, decisions/second, input tokens/second, abstention count, and final decision outputs. One request contains three sequential decisions. Input-token throughput includes prompt processing and scoring; it is not generated-token throughput. Percentiles use nearest rank, and five samples only provide a small smoke measurement.

The test does not impose a hardware-dependent speed threshold or treat an abstained answer as a timing failure. Record model hashes, the libllama build, and device settings when comparing runs. OS page-cache state affects load time; this is not a cold-disk benchmark. Reports remain in the excluded `results/` directory when using the command above.

## Svelte website

The English introduction site is in `web/`. It is a prerendered SvelteKit site with a labeled illustrative scoring demo; it does not load model weights or call an inference service.

```sh
cd web
npm ci
npm run check
npm run build
npm run dev
```

The static output is `web/build/`. Browser checks run with `npm test` after installing Playwright Chromium. The website is separate from the Rust crate and is excluded from its Cargo package.
