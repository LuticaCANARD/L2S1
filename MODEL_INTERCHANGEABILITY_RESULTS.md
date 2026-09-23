# Model interchangeability implementation checks

Run date: 2026-09-23. See [the implementation guide](MODEL_INTERCHANGEABILITY.md) and [machine-readable evidence](MODEL_INTERCHANGEABILITY_RESULTS.json). These are local contract and regression checks on synthetic requests, not workload-quality or production evidence.

## Checks

- Default-feature tests: **21 passed**.
- Release tests with `llama`: **30 passed**, 13 environment-dependent tests ignored in the general run. The conformance tests below were invoked explicitly.
- `cargo clippy --release --locked --offline --all-targets --features llama -- -D warnings`, formatting and whitespace checks passed. Upstream C++ unused-function warnings remain separate from Rust lint checks.
- CLI inspection, actual request preflight, diagnostics, structured context-overflow failure, snapshot-budget fallback, single-decision fresh fallback, offline scalar fitting, artifact loading and execution-binding rejection passed. Calibration examples use synthetic records; they establish no useful fitted task temperature.
- Worker tests verify construction/use/destruction on one owner thread with a non-Send backend, full-queue rejection, memory admission, draining, idempotent close and reservation release after construction failure/panic. These are synthetic lifecycle tests.
- Exact evidence is checked against an independent full-softmax calculation for binary, choice and ordinal tasks. Invalid/duplicate/missing token mappings are rejected. Calibration tests cover identity/task mismatch, held-out group leakage, held-out option-count mismatch, retained logits/mass, ties, coverage and accepted accuracy.

## Original-response regression

A separately built copy of the pre-change working tree was compared with the implementation on CPU using default legacy/fresh settings and `examples/warehouse.json`. Parsed response JSON was exactly equal for all **12 decisions** across SmolLM2, Qwen3, Gemma3 and TinyLlama, including probabilities, candidate mass, abstention and backend fields. JSON key order is not part of this comparison.

## Real-model conformance matrix

Every row ran the same three decision kinds with a shared warehouse state plus repeated history, context 2048, batch/ubatch 32, four CPU threads, FlashAttention off and state-first prompts. Tests verify IDs, semantic options, token mappings, ordinal interpretation, explicit unsupported modes, artifact binding, snapshot budgets, failure recovery and request-local diagnostics. Replacing models is not expected to preserve their predictions.

The seven-checkpoint inference matrix was completed before final metadata refinements: held-out calibration dimension validation, learned-head diagnostic provenance, and a fingerprint of the compiled bridge/compiler configuration. The native snapshot and base scoring algorithms were unchanged. Final unit/native/CLI checks and a further SmolLM2 CUDA conformance run cover those refinements; each JSON row preserves its measured build identity.

| Checkpoint | Device | Snapshot MiB | Restore max probability / mass delta | Parallel max probability delta | Parallel equivalence |
| --- | --- | ---: | --- | ---: | --- |
| SmolLM2-135M-Instruct-Q8_0.gguf | CPU | 4.222 | 0 / 0 | 0.0833158 | Fail |
| Qwen3-0.6B-Q8_0.gguf | CPU | 17.502 | 0 / 0 | 0.00667669 | Pass |
| gemma-3-1b-it-Q8_0.gguf | CPU | 4.067 | 0 / 0 | 5.56596e-06 | Pass |
| tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | CPU | 4.128 | 0 / 0 | 0.0538332 | Fail |
| gemma-4-E2B-it-Q8_0.gguf | RTX 3080 CUDA | 2.817 | 0 / 0 | 0.00110224 | Pass |
| Qwen3.8-27B-UD-IQ2_XXS.gguf | RTX 3080 CUDA | 159.630 | 0 / 0 | — | Unsupported for hybrid memory |
| gpt-oss-20b-MXFP4.gguf | CPU | 8.255 | 0 / 0 | 0.13541 | Fail |
| SmolLM2-135M-Instruct-Q8_0.gguf (final build) | RTX 3080 CUDA | 4.222 | 0 / 0 | 0.0534828 | Fail |

State restoration had **zero probability/mass differences and no changed top-choice or accepted decisions** in these comparisons. It allocated a real snapshot and restored all three branches in each successful restore run. A zero-byte budget produced explicit fresh fallback with no snapshot buffer. Fresh recovery after invalid/oversize requests passed. Qwen3.8’s GGUF declares architecture `qwen35`; its prefix-reuse mode explicitly fell back to fresh, while full state restoration worked.

The equivalence criterion remains probability/mass delta below 0.02, no changed top-choice and no changed accepted result. **Parallel execution fails this criterion on SmolLM2 CPU/CUDA, TinyLlama CPU and GPT-OSS CPU.** GPT-OSS changed one raw top-choice; these rows had no changed accepted decisions. This preserves the earlier documented [parallel-mode limitation](PARALLEL_EXECUTION.md); it does not relax thresholds or enable parallelism by default.

## Memory and timing scope

The `time -v` peak process RSS was 1444.4 MiB for the four-model CPU invocation, 7561.2 MiB for the Gemma4/Qwen hybrid CUDA invocation, and 13943.3 MiB for GPT-OSS CPU. These include model/runtime allocations and are neither snapshot-only memory nor GPU VRAM measurements.

On the CUDA fixture, Gemma4 saved a 2.817 MiB snapshot and the hybrid Qwen saved 159.630 MiB. Save plus three restores cost about 0.884 seconds and 3.307 seconds respectively, whereas shared-prefix computation cost about 0.043 and 0.299 seconds. These observations do not establish a speedup; the mode remains experimental. Snapshot-copy timing includes the native clear/restore work. Commands shared host resources, and GPT-OSS was briefly paused to reduce contention, so whole-command elapsed times are not comparable benchmarks.

## Reproduction and boundaries

The matching llama.cpp source revision was `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. All full checkpoint, template, runtime and final-source fingerprints are in the JSON report. Set `LLAMA_CPP_DIR` and `LLAMA_LIB_DIR` to that matching local build before running the commands in the implementation guide. The CUDA device was an RTX 3080 with 10 GiB VRAM.

No new real LoRA/learned-head quality evaluation, remote-provider execution, service deployment, physical-device test, universal model-compatibility claim or production accuracy claim is made here. Existing head behavior is covered by its scoring tests; the new head-origin diagnostic has its own focused test. Model-specific workload calibration and quality evaluation remain necessary before choosing acceptance thresholds.
