#!/bin/bash
set -euo pipefail
ROOT=/home/lutica/l2s1-arm64-fresh-20260927
export CARGO_HOME=/home/lutica/l2s1-pi-20260927/cargo
export RUSTUP_HOME=/home/lutica/l2s1-pi-20260927/rustup
export PATH="/home/lutica/l2s1-pi-20260927/venv/bin:$CARGO_HOME/bin:$ROOT/node/bin:$PATH"
export LIBCLANG_PATH="$ROOT/deps/usr/lib/aarch64-linux-gnu"
export BINDGEN_EXTRA_CLANG_ARGS="-resource-dir=$ROOT/deps/usr/lib/llvm-19/lib/clang/19"
export L2S1_PORTABLE_BUILD=1 L2S1_ARM64_DISPATCH=1 L2S1_OPENMP=0 CCACHE_DISABLE=1
export CARGO_BUILD_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3
mkdir -p "$ROOT/state-cache/after"
cd "$ROOT/source"
cargo build --release --locked --features llama --example state_cache_probe --bin l2s1 --message-format=json-render-diagnostics > "$ROOT/state-cache/after/cargo.jsonl" 2> "$ROOT/state-cache/after/build.log"
cp target/release/examples/state_cache_probe "$ROOT/state-cache/after/probe"
node sdks/typescript/scripts/bundle-runtime.mjs --cargo-log "$ROOT/state-cache/after/cargo.jsonl" --output "$ROOT/state-cache/after/package"
node sdks/typescript/scripts/verify-runtime.mjs "$ROOT/state-cache/after/package"
npm pack "$ROOT/state-cache/after/package" --pack-destination "$ROOT/state-cache/after" --ignore-scripts
mkdir -p "$ROOT/state-cache/installed"
cd "$ROOT/state-cache/installed"
printf '{"private":true}\n' > package.json
npm install --offline --ignore-scripts --no-audit --no-fund "$ROOT/state-cache/after/l2s1-runtime-linux-arm64-0.1.3.tgz"
node "$ROOT/source/sdks/typescript/scripts/verify-runtime.mjs" "$PWD/node_modules/@l2s1/runtime-linux-arm64"
