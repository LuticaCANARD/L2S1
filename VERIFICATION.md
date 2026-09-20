# Local verification record

Date: 2026-09-20. These results cover local integration behavior, not task accuracy, production readiness, or throughput.

## Runtime

- llama.cpp commit: `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.
- Shared libraries: the existing Linux/WSL `build-cuda/bin` build.
- Jinja: `common/jinja`, JSON, and Unicode helper sources from the same checkout, compiled into the adapter.
- GPU: NVIDIA GeForce RTX 3080.
- The external llama.cpp checkout and its existing shared-library build were not modified.

The current project location requires explicit `LLAMA_CPP_DIR` and `LLAMA_LIB_DIR`; there is no sibling `../llama.cpp` checkout. CUDA initialization failed inside the execution sandbox. Re-running with native GPU access succeeded; the final CUDA results below use that environment.

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

## Checks

- `cargo fmt --check`: passed.
- `cargo test --locked --offline`: 7 tests passed; no real model required.
- `cargo test --locked --offline --features llama`: 11 tests passed; the real-model test is ignored by default.
- `cargo clippy --locked --offline --all-targets --features llama -- -D warnings`: passed. Upstream C++ Jinja headers emit unused-function warnings; Rust Clippy reported no errors.
- Explicit Qwen3 `--prompt-profile model` CLI run on CPU: passed, in addition to the default Qwen3 profile.

The Jinja tests use synthetic fixtures for ChatML, Llama 3, Gemma, Phi, Mistral, and Zephyr-style delimiters. They verify payload separation, role suffixes, BOS/EOS substitution, generation/non-thinking flags, and rejection of malformed or unusable templates. These fixture tests are not real-weight validation for Llama 3, Phi, Mistral, or unlisted Gemma checkpoints. The Gemma 3 1B checkpoint now has the separate real-weight evidence below.

## Real-model matrix

| Checkpoint | CPU native suite | CUDA native suite | CPU/CUDA CLI JSON |
| --- | --- | --- | --- |
| Qwen3 0.6B Q8_0 | Passed | Passed | Passed / Passed |
| SmolLM2 135M Q8_0 | Passed | Passed | Passed / Passed |
| Gemma 3 1B IT Q8_0 | Passed | Passed | Passed / Passed |
| Gemma 4 E2B Instruct Q8_0 | Passed | Passed | Passed / Passed |
| TinyLlama 1.1B Q4_K_M | Passed | **Failed: batch-size probability difference** | Passed / Passed |

Each successful native suite checks A–B–A request/KV isolation, candidate-probability differences below 0.02 between batch sizes 32 and 512, unique single-token continuations for all 26 codes, and rejection of an oversized request. Non-Qwen checkpoints additionally reject an explicit `qwen3` profile. CPU/CUDA numerical equality is not asserted.

The CLI runs use all three decision types in `examples/ticket.json` at the default batch size of 256. The output artifacts are:

| Model | CPU | CUDA |
| --- | --- | --- |
| Qwen3 | [JSON](examples/ticket.qwen3.cpu.output.json) | [JSON](examples/ticket.qwen3.cuda.output.json) |
| SmolLM2 | [JSON](examples/ticket.smollm2.cpu.output.json) | [JSON](examples/ticket.smollm2.cuda.output.json) |
| Gemma 3 | [JSON](examples/ticket.gemma3.cpu.output.json) | [JSON](examples/ticket.gemma3.cuda.output.json) |
| Gemma 4 | [JSON](examples/ticket.gemma4.cpu.output.json) | [JSON](examples/ticket.gemma4.cuda.output.json) |
| TinyLlama | [JSON](examples/ticket.tinyllama.cpu.output.json) | [JSON](examples/ticket.tinyllama.cuda.output.json) |

GPU logs confirm full layer offload for Qwen3 (29/29) and SmolLM2 (31/31), and Gemma 3 (27/27). Detailed logs are available locally under the ignored `target/verification-multimodel/` directory.

### TinyLlama CUDA limitation

TinyLlama Q4_K_M loads and returns valid JSON on CUDA, and its A–B–A isolation check passed. However, the department question produced these candidate probabilities:

| Batch | A | B | C |
| --- | --- | --- | --- |
| 32 | 0.7580996680 | 0.1571134331 | 0.0847868989 |
| 256 | 0.6735461659 | 0.2375104192 | 0.0889434149 |
| 512 | 0.6735461659 | 0.2375104192 | 0.0889434149 |

The A probability difference of **0.0845535021** exceeds the existing 0.02 test tolerance. The tolerance was not relaxed. The recorded TinyLlama CUDA native run stopped at that assertion; alphabet/profile/oversized-input checks from that run are not claimed for TinyLlama CUDA. (The subsequent Gemma 4 follow-up moves alphabet checks earlier and releases model handles between comparisons.) Those checks passed on CPU. The underlying CUDA numerical cause has not been established; this checkpoint is not validated for batch-independent CUDA results. Use CPU for the fully passing TinyLlama configuration, or treat a fixed CUDA batch/build as a separately evaluated configuration.

## Interpretation

Successful execution is not evidence of correct decisions. For example, the small SmolLM2 checkpoint returned `false` for the refund question in this Korean ticket; the Qwen3 checkpoint returned `true`. These examples do not establish either model's general accuracy. Evaluate model choice, language coverage, option ordering, and abstention policies on task-specific labeled data.

The earlier Qwen3-only CUDA output remains at [ticket.cuda.output.json](examples/ticket.cuda.output.json) for historical reference. The model-specific artifacts above were generated after the multi-model changes.

## Gemma 3 follow-up

Gemma 3 1B IT Q8_0 passed the existing native suite on CPU and RTX 3080 CUDA without inference-code changes or relaxed tolerances. The generic embedded-Jinja profile was selected. CUDA logs report a 1013.61 MiB model buffer; this is not total GPU/process memory.

The Korean ticket example yielded:

| Decision | CPU | CUDA |
| --- | --- | --- |
| Department | `billing` | Abstained |
| Refund requested | `true` (`p_true=0.998981`) | `true` (`p_true=0.999907`) |
| Urgency | Abstained (expected value 0.531813) | `low` (expected value 0.198202) |

The batch-size checks compare executions within one backend. Passing them does not establish CPU/CUDA numerical equality or decision equivalence. This single example is not an accuracy benchmark.

## Distribution review

The initial `cargo package --list` included local GGUF files despite `.gitignore`. Explicit Cargo exclusions now prevent model weights and local build/results directories from entering the source package. The project MIT license, third-party license texts, and [licensing assessment](LICENSING.md) are included. An actual source `.crate` was built and verified locally with `cargo package --locked --offline --allow-dirty`; its archive contains the license documents and no GGUF files. This review does not publish a crate or redistribute any model weights.

## Gemma 4 follow-up

Gemma 4 E2B Instruct Q8_0 passed the native suite and all three CLI decision types on CPU and RTX 3080 CUDA with the same llama.cpp revision. It reports architecture `gemma4` and uses `gguf-jinja-decision-v1`. No inference-code changes or tolerance relaxations were needed. The model's pinned SHA256 matched Hugging Face LFS metadata. Only text weights were loaded; image/audio projection and MTP were not tested.

The native test now releases each model/context after its checks, before constructing the next comparison context. This avoids loading several copies of the roughly 4.97 GB model simultaneously. A–B–A request isolation still reuses one live context; batch-size comparisons, all 26 candidate tokens, invalid-profile rejection, and oversized-input rejection retain their assertions. Gemma 3 was re-run successfully on CPU and CUDA after this test resource change.

For the Korean ticket, both Gemma 4 backends selected `billing`, returned `true` for the refund request, and selected urgency `medium`. The refund probabilities were 0.999999772 (CPU) and 0.999999725 (CUDA). Matching labels on one example are not a general accuracy or backend-equivalence guarantee.

Gemma 4 weights use Apache-2.0, whereas the tested Gemma 3 weights use the Gemma Terms of Use. See [LICENSING.md](LICENSING.md) for the implications when distributing this MIT application with or without weights.
