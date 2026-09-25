# Verification guide

Run checks for the exact checkpoint, runtime and execution configuration you intend to use. Unit tests, native contract tests and labeled task evaluations answer different questions.

## Build and general checks

The pinned llama.cpp revision is `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. CMake FetchContent downloads it on the first default build and verifies the archive hash. The sys dependency builds matching source, headers and shared libraries. The native backend currently requires Linux. Set `L2S1_LLAMA_CPP_SOURCE` to use a local checkout for offline checks or to test another upstream revision.

```sh
cargo fmt --all -- --check
cargo test --locked --offline
cargo test --release --locked --offline --features llama
cargo clippy --release --locked --offline --all-targets --features llama -- -D warnings
python3 -m unittest discover -s scripts -p 'test_*.py'
```

For GPU checks, build with `--features llama-cuda` and use `SKID_CUDA=1` for native tests. A CPU-only build rejects a CUDA request.

For another GGUF family on CUDA, use the same binary with the new model path. Run `--inspect`, `--preflight --input examples/warehouse.json`, and a real decision request before comparing task quality. The multi-model conformance test below accepts colon-separated GGUF paths and `SKID_CUDA=1`. A successful load or a finite candidate mass establishes compatibility with this decision path, not correctness on labeled tasks.

For direct still-image input, provide a compatible vision GGUF and its matching `mmproj` GGUF. The ignored test uses two different PNGs, checks that image content affects raw logits, and checks recovery after an invalid image. It does not establish task accuracy:

```sh
SKID_VISION_MODEL=/path/to/vision-model.gguf \
SKID_VISION_MMPROJ=/path/to/mmproj.gguf \
cargo test --locked --offline --features llama --test vision -- --ignored
```

Use `--features llama-cuda` and `SKID_CUDA=1` for the same contract on a permitted CUDA host. The HTTP path additionally needs a live loopback check of `/healthz`, a valid `POST /v1/decisions` with `image_base64`, and a malformed request returning HTTP 400.

Offline Cargo commands also need a previously populated CMake source cache or `L2S1_LLAMA_CPP_SOURCE` set to a local checkout. General test runs skip tests that need model files; they do not download weights or establish real-model compatibility. Upstream C++ helper warnings are separate from Rust lint results.

## Model contract checks

```sh
L2S1_CONFORMANCE_MODELS=/path/to/model-a.gguf:/path/to/model-b.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/l2s1-conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

Set `SKID_CUDA=1` for CUDA. The conformance suite checks semantic IDs, all decision kinds, token mappings, artifact bindings, context failures, recovery, snapshot limits and execution diagnostics. It compares optimized modes with fresh execution using the existing probability/mass tolerance of 0.02, unchanged top choices and unchanged accepted results. Parallel differences are reported separately; a completed report is not automatically an equivalence pass.

Use [prefix-reuse checks](SEMIF_ALGORITHM.md#reproduce) and [parallel contract checks](PARALLEL_EXECUTION.md#validation-and-measurement) for their specific execution paths. Parallel mode has known numerical differences and remains opt-in. A model that loads or passes preflight still needs inference and labeled workload evaluation.

## Optional optimization checks

```sh
SKID_MODEL=/path/to/model.gguf \
  cargo test --release --locked --offline --features llama \
  --test native_compact --test optimization_contract --test shared_state \
  --test worker_native -- --include-ignored --test-threads=1

cargo run --release --locked --offline --features llama \
  --example benchmark_optimizations -- \
  --model /path/to/model-a.gguf --model /path/to/model-b.gguf \
  --output /tmp/l2s1-optimizations.json --repeats 3
```

Repeat the contract checks for each checkpoint; use `SKID_CUDA=1` for tests and `--cuda` for the benchmark when measuring CUDA. The contracts cover exact cached token preparation, dynamic instructions and option order, compact/full evidence equivalence, session error cleanup, and native worker ticket correlation. Compact transfer retains the full-vocabulary normalizer. It removes a host-side buffer copy into Rust; it does not remove vocabulary projection or GPU-to-host transfer.

The benchmark measures 1, 4 and 16 questions with short and extended synthetic states, rotates execution order, excludes loading, and records raw responses, identities, stage timings and cache statistics. Cache measurements intentionally use repeated inputs after warmup. Shared-state measurements issue separate calls with different questions inside one session. Compare each optimization to its same-layout fresh baseline; state-first prompting can change predictions independently of reuse. Amortized time per question is not standalone request latency, and these workloads do not establish task accuracy or production throughput. Reports refuse overwrites and belong in ignored local output directories.

## Task quality and evidence

- [Synthetic benchmark](BENCHMARK.md): correctness, abstention, consistency and latency on the rule fixture.
- [AG News evaluation](KAGGLE_BENCHMARK.md): a frozen classification protocol.
- [JevBench evaluation](JEVBENCH.md): public task mapping, upstream scoring and separate acceptance metrics.

Retain checkpoint, runtime, prompt and configuration identities alongside raw predictions and labels in ignored local output directories. Keep latency, coverage, accepted accuracy and candidate-relative confidence separate. Generated reports and host-specific execution plans are not source documentation. The README contains a compact model-comparison summary; it is not an official leaderboard or production-accuracy claim.
