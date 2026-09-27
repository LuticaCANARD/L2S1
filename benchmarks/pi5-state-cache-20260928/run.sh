#!/bin/bash
set -euo pipefail
ROOT=/home/lutica/l2s1-arm64-fresh-20260927
for part in performance quality session; do
    python3 "$ROOT/state-cache/measure.py" --root "$ROOT/state-cache" \
        --model /home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf \
        --suite "$ROOT/source/tests/fixtures/decision_benchmark.json" --part "$part"
done
# Compile/run additional regression only after the performance measurements.
export CARGO_HOME=/home/lutica/l2s1-pi-20260927/cargo
export RUSTUP_HOME=/home/lutica/l2s1-pi-20260927/rustup
export PATH="/home/lutica/l2s1-pi-20260927/venv/bin:$CARGO_HOME/bin:$ROOT/node/bin:$PATH"
export LIBCLANG_PATH="$ROOT/deps/usr/lib/aarch64-linux-gnu"
export BINDGEN_EXTRA_CLANG_ARGS="-resource-dir=$ROOT/deps/usr/lib/llvm-19/lib/clang/19"
export L2S1_PORTABLE_BUILD=1 L2S1_ARM64_DISPATCH=1 L2S1_OPENMP=0 CCACHE_DISABLE=1
export CARGO_BUILD_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3
export SKID_MODEL=/home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf
cd "$ROOT/source"
cargo test --release --locked --features llama --test state_restore --test fixed_schema -- --ignored --nocapture > "$ROOT/state-cache/native-tests.log" 2>&1
L2S1_FORCE_KV_CLEAR=1 cargo test --release --locked --features llama --test state_restore -- --ignored --nocapture > "$ROOT/state-cache/force-clear-tests.log" 2>&1
