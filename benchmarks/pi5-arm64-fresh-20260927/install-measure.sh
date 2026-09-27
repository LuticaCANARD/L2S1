#!/bin/bash
set -euo pipefail
ROOT=/home/lutica/l2s1-arm64-fresh-20260927
OLD=/home/lutica/l2s1-pi-20260927
export CARGO_HOME="$OLD/cargo" RUSTUP_HOME="$OLD/rustup"
export PATH="$OLD/venv/bin:$CARGO_HOME/bin:$ROOT/node/bin:$PATH"
export LIBCLANG_PATH="$ROOT/deps/usr/lib/aarch64-linux-gnu"
export BINDGEN_EXTRA_CLANG_ARGS="-resource-dir=$ROOT/deps/usr/lib/llvm-19/lib/clang/19"
export L2S1_PORTABLE_BUILD=1 L2S1_ARM64_DISPATCH=1 L2S1_OPENMP=0 CCACHE_DISABLE=1
export CARGO_BUILD_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3
cd "$ROOT/source"
cargo test --release --locked --features llama --lib --bin l2s1 --test fixed_schema --test prefix_reuse --no-run > "$ROOT/test-build.log" 2>&1
# Only begin measurement after all compilation is finished.
for variant in baseline kernels openmp combined; do
    mkdir -p "$ROOT/installed/$variant"
    cd "$ROOT/installed/$variant"
    printf '{"private":true}\n' > package.json
    npm install --offline --ignore-scripts --no-audit --no-fund "$ROOT/builds/$variant/l2s1-runtime-linux-arm64-0.1.3.tgz"
    node "$ROOT/source/sdks/typescript/scripts/verify-runtime.mjs" "$PWD/node_modules/@l2s1/runtime-linux-arm64"
done
cd "$ROOT"
python3 inspect-builds.py "$ROOT" > inspect.log
cc -shared -fPIC -march=armv8-a mask-features.c -ldl -o mask-features.so
python3 measure.py --root "$ROOT" --suite "$ROOT/source/tests/fixtures/decision_benchmark.json" --model /home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf --part factorial
python3 measure.py --root "$ROOT" --suite "$ROOT/source/tests/fixtures/decision_benchmark.json" --model /home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf --part quality
python3 measure.py --root "$ROOT" --suite "$ROOT/source/tests/fixtures/decision_benchmark.json" --model /home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf --part fallback
python3 report.py "$ROOT" > report.log

cd "$ROOT/source"
cargo test --release --locked --features llama --lib --bin l2s1 > "$ROOT/rust-tests.log" 2>&1
export SKID_MODEL=/home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf
export SKID_BATCH=32 SKID_REUSE_OUTPUT="$ROOT/prefix-equivalence.json"
cargo test --release --locked --features llama --test fixed_schema --test prefix_reuse -- --ignored --nocapture > "$ROOT/native-equivalence.log" 2>&1
