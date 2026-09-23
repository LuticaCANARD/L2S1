# Verification guide

Run checks for the exact checkpoint, runtime and execution configuration you intend to use. Unit tests, native contract tests and labeled task evaluations answer different questions.

## Build and general checks

The locally verified llama.cpp revision is `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. Its source, headers and shared libraries must match. Set paths to your own build; the native backend currently requires Linux.

```sh
export LLAMA_CPP_DIR=/path/to/llama.cpp
export LLAMA_LIB_DIR="$LLAMA_CPP_DIR/build-cuda/bin"

cargo fmt --all -- --check
cargo test --locked --offline
cargo test --release --locked --offline --features llama
cargo clippy --release --locked --offline --all-targets --features llama -- -D warnings
python3 -m unittest discover -s scripts -p 'test_*.py'
```

Offline Cargo commands require cached dependencies. General test runs skip tests that need model files; they do not download weights or establish real-model compatibility. Upstream C++ helper warnings are separate from Rust lint results.

## Model contract checks

```sh
L2S1_CONFORMANCE_MODELS=/path/to/model-a.gguf:/path/to/model-b.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/l2s1-conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

Set `SKID_CUDA=1` for CUDA. The conformance suite checks semantic IDs, all decision kinds, token mappings, artifact bindings, context failures, recovery, snapshot limits and execution diagnostics. It compares optimized modes with fresh execution using the existing probability/mass tolerance of 0.02, unchanged top choices and unchanged accepted results. Parallel differences are reported separately; a completed report is not automatically an equivalence pass.

Use [prefix-reuse checks](SEMIF_ALGORITHM.md#reproduce) and [parallel contract checks](PARALLEL_EXECUTION.md#validation-and-measurement) for their specific execution paths. Parallel mode has known numerical differences and remains opt-in. A model that loads or passes preflight still needs inference and labeled workload evaluation.

## Task quality and evidence

- [Synthetic benchmark](BENCHMARK.md): correctness, abstention, consistency and latency on the rule fixture.
- [AG News evaluation](KAGGLE_BENCHMARK.md): a frozen classification protocol.
- [JevBench evaluation](JEVBENCH.md): public task mapping, upstream scoring and separate acceptance metrics.

Retain checkpoint, runtime, prompt and configuration identities alongside raw predictions and labels in ignored local output directories. Keep latency, coverage, accepted accuracy and candidate-relative confidence separate. Generated reports and host-specific execution plans are not source documentation. The README contains a compact model-comparison summary; it is not an official leaderboard or production-accuracy claim.
