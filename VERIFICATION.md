# Local verification record

State-first v2 update: the earlier measurements below retain their original v1 prompt scope. Optional prefix-reuse correctness, prompt comparison, and timing results are recorded separately in [SEMIF_ALGORITHM.md](SEMIF_ALGORITHM.md).

GPT-OSS update: `gpt-oss-final-prefill-decision-v1` passed the CUDA native suite, including the unchanged batch-size tolerance, request isolation, all 26 candidate tokens, and oversized-input rejection. The earlier open-header GPT-OSS failure is superseded for this measured implementation only. See [GPT-OSS final-prefill results](GPT_OSS_FINAL_RESULTS.md) for the frozen Kaggle before/after comparison and later working-tree changes not covered by this run. CPU verification for this new profile remains unperformed.

Date: 2026-09-21. The current fixture is the English warehouse scenario in `examples/warehouse.json`. It replaces the earlier ticket scenario; results from those scenarios are not interchangeable. This is local integration and smoke-performance evidence, not production or task-accuracy certification.

## Runtime and scope

- llama.cpp: `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`, using its matching Linux/WSL shared libraries and Jinja helper sources.
- CPU: Intel Core i9-9900K, four inference threads.
- CUDA: NVIDIA GeForce RTX 3080, with native GPU access outside the execution sandbox.
- Real-model tests and CLI: release build. Context 2048; performance/CLI batch 256. Ordinary unit tests also passed in the debug profile.
- Inputs: three sequential questions about storage zone, cold-chain requirements, and dispatch priority.
- Local weights are not distributed. No model is downloaded by the tests or the website.

## Checkpoint provenance

All model files live in the ignored `models/` directory. Downloaded file SHA256 values matched the Hugging Face LFS metadata for the pinned revisions.

| Model | Repository and pinned revision | File | Bytes | SHA256 |
| --- | --- | --- | --- | --- |
| Qwen3 0.6B | [Qwen/Qwen3-0.6B-GGUF](https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/tree/23749fefcc72300e3a2ad315e1317431b06b590a) | `Qwen3-0.6B-Q8_0.gguf` | 639446688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` |
| SmolLM2 135M Instruct | [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/tree/09816acd5d99df7be770d85ea30822623dab342c) | `SmolLM2-135M-Instruct-Q8_0.gguf` | 144811360 | `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83` |
| Gemma 3 1B IT | [ggml-org/gemma-3-1b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF/tree/f9c28bcd85737ffc5aef028638d3341d49869c27) | `gemma-3-1b-it-Q8_0.gguf` | 1069306368 | `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5` |
| Gemma 4 E2B Instruct | [ggml-org/gemma-4-E2B-it-GGUF](https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF/tree/b4243c156154b6dca9324415f8c7ccc098b4aed1) | `gemma-4-E2B-it-Q8_0.gguf` | 4967497152 | `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a` |
| TinyLlama 1.1B Chat | [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tree/52e7645ba7c309695bec7ac98f4f005b139cf465) | `tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf` | 668788096 | `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0` |

SmolLM2 and TinyLlama use community GGUF conversions. Both report `general.architecture=llama` but use different vocabularies and embedded templates. Qwen3 reports `qwen3`; Gemma 3 reports `gemma3`. The Gemma file hash was verified against its pinned Hugging Face LFS metadata.

## Current correctness checks

The native suite reuses a context for A–B–A request isolation, checks unique single-token continuations for all 26 codes, compares batches 32 and 512 with an unchanged 0.02 probability tolerance, rejects incompatible explicit profiles, and rejects oversized input. Model handles are released between independent checks to limit test memory use.

| Checkpoint | CPU native suite | CUDA native suite | CPU/CUDA performance test and CLI |
| --- | --- | --- | --- |
| Gemma 4 E2B Q8_0 | Passed | Passed | Passed / Passed |
| Gemma 3 1B Q8_0 | Passed | Failed: batch consistency | Passed / Passed |
| Qwen3 0.6B Q8_0 | Passed | Passed | Passed / Passed |
| SmolLM2 135M Q8_0 | Passed | Passed | Passed / Passed |
| TinyLlama 1.1B Q4_K_M | Passed | Failed: batch consistency | Passed / Passed |

The CUDA consistency failures are reproducible on this warehouse input:

- TinyLlama option A: 0.5203199292 at batch 32 versus 0.4535476556 at batch 512; difference 0.0667722736.
- Gemma 3 option B: 0.9527877347 at batch 32 versus 0.9904579977 at batch 512; difference 0.0376702630.

Their A–B–A and alphabet checks execute before this failure. The later explicit-profile and oversized-input checks are not claimed for these CUDA runs; they pass on CPU. Both checkpoints run at the fixed performance batch of 256, but these timings do not establish batch-independent decisions. CPU/CUDA numerical equality is not asserted for any model. Gemma 4's native suite passes on both devices. No tolerance was relaxed.

## English example outputs

| Checkpoint | CPU | CUDA |
| --- | --- | --- |
| Gemma 4 E2B Q8_0 | [JSON](examples/warehouse.gemma4.cpu.output.json) | [JSON](examples/warehouse.gemma4.cuda.output.json) |
| Gemma 3 1B Q8_0 | [JSON](examples/warehouse.gemma3.cpu.output.json) | [JSON](examples/warehouse.gemma3.cuda.output.json) |
| Qwen3 0.6B Q8_0 | [JSON](examples/warehouse.qwen3.cpu.output.json) | [JSON](examples/warehouse.qwen3.cuda.output.json) |
| SmolLM2 135M Q8_0 | [JSON](examples/warehouse.smollm2.cpu.output.json) | [JSON](examples/warehouse.smollm2.cuda.output.json) |
| TinyLlama 1.1B Q4_K_M | [JSON](examples/warehouse.tinyllama.cpu.output.json) | [JSON](examples/warehouse.tinyllama.cuda.output.json) |

The fixture's rule-derived answers are `chilled`, `true`, and `high`. An inference result or high candidate probability is not an accuracy guarantee. TinyLlama and SmolLM2 abstain on all three questions with the default policy in the measured runs. The other measured models accept their three answers. This single scenario is not a representative evaluation dataset. Published output files normalize only the model path to a portable `models/<filename>`; scores and decisions retain their measured values.

## Inference performance test

`tests/performance.rs` is opt-in and uses `SKID_MODEL` to locate an existing checkpoint. It measures loading, the first request, additional warmups, and individual steady-state request durations. See the README for the command and environment variables.

These measurements used five samples after one separately measured first request and one additional warmup. One request includes three sequential decisions. Model files had already been accessed by the correctness run, so load times are not cold-disk measurements. Normal host activity was not isolated; the small sample count is intended to verify the measurement path rather than establish stable performance rankings.

| Checkpoint | Device | Load ms | First request ms | Request p50 ms | Request p95 ms | Decisions/s | Abstained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 E2B Q8_0 | CPU | 983.0 | 7482.4 | 7537.0 | 7699.1 | 0.40 | 0/15 |
| Gemma 4 E2B Q8_0 | CUDA | 1554.9 | 263.6 | 96.8 | 102.6 | 30.69 | 0/15 |
| Gemma 3 1B Q8_0 | CPU | 529.7 | 2767.6 | 2796.4 | 3028.6 | 1.05 | 0/15 |
| Gemma 3 1B Q8_0 | CUDA | 812.7 | 184.9 | 61.8 | 72.7 | 46.44 | 0/15 |
| Qwen3 0.6B Q8_0 | CPU | 447.5 | 2272.2 | 2334.4 | 2843.8 | 1.26 | 0/15 |
| Qwen3 0.6B Q8_0 | CUDA | 632.9 | 194.7 | 63.5 | 67.0 | 47.83 | 0/15 |
| SmolLM2 135M Q8_0 | CPU | 176.4 | 2250.5 | 720.4 | 1695.9 | 3.33 | 15/15 |
| SmolLM2 135M Q8_0 | CUDA | 316.3 | 181.4 | 52.2 | 65.0 | 56.82 | 15/15 |
| TinyLlama 1.1B Q4_K_M | CPU | 369.4 | 3677.8 | 3797.0 | 4144.4 | 0.79 | 15/15 |
| TinyLlama 1.1B Q4_K_M | CUDA | 374.7 | 171.1 | 49.4 | 50.1 | 60.95 | 15/15 |

Latency covers the complete `decide` call, including prompt preparation, prefill, logits transfer, and scoring. Input tokens/second in the JSON reports is not generated tokens/second. Throughput counts completed decisions, including abstentions. Percentiles use nearest rank; five samples make p95 equal to the maximum observed latency. There is no hardware-dependent pass threshold.

Full local reports are in the excluded `results/performance/` directory, with execution logs and statuses in `results/verification-warehouse/`. The website uses a selected, path-free summary in `web/src/lib/benchmarks.json`; its playground uses explicitly labeled synthetic probabilities instead of running a model in the browser.

## Source and website checks

- Rust: 14 ordinary tests passed with the `llama` feature; the three real-model tests are opt-in. Native and single-fixture performance results are listed above; the separate labeled comparison is recorded in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md). Formatting and Clippy with `-D warnings` passed; upstream C++ headers still emit unused-function warnings.
- SvelteKit: type and accessibility checks completed with zero errors and warnings; the static production build passed.
- Chromium: all five browser tests passed, covering prerendered content without JavaScript, keyboard-controlled abstention, decision types, clipboard interaction, documentation downloads, and viewport overflow at 1440 and 375 pixels. Desktop and mobile screenshots were visually inspected.
- Packaging: `cargo package --locked --offline --allow-dirty` built and verified the source archive. Its contents include the performance test, English fixture, and license notices, and exclude weights, local results, and the separate website. Static website assets were also inspected for model files and local filesystem paths in JSON examples.
- No website deployment was performed.

## Distribution and licensing

The project source is MIT-licensed. Upstream notices are in `THIRD_PARTY_LICENSES.txt`; model licenses are separate and reviewed in `LICENSING.md`. Gemma 3 uses its own terms; Gemma 4 uses Apache-2.0.

Cargo excludes local model/build/result directories, GGUF and safetensors files, and the separate website. The actual source archive is inspected for model weights and required notices after packaging. Neither the website nor tests redistribute weights. The previous ticket assets are retained only in excluded local history, not in the distributed examples.


## Model interchangeability implementation (2026-09-23)

The identity/preflight, evidence/policy boundary, scalar calibration, diagnostics,
whole-sequence restoration and bounded-worker checks are recorded separately in
[MODEL_INTERCHANGEABILITY_RESULTS.md](MODEL_INTERCHANGEABILITY_RESULTS.md).
The report distinguishes original-response regression checks, real-model contracts,
synthetic calibration/lifecycle tests and experimental parallel equivalence.
