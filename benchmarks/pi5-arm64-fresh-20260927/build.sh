#!/bin/bash
set -euo pipefail
ROOT=/home/lutica/l2s1-arm64-fresh-20260927
OLD=/home/lutica/l2s1-pi-20260927
export CARGO_HOME="$OLD/cargo" RUSTUP_HOME="$OLD/rustup"
export PATH="$OLD/venv/bin:$CARGO_HOME/bin:$ROOT/node/bin:$PATH"
export CARGO_BUILD_JOBS=3 CMAKE_BUILD_PARALLEL_LEVEL=3 CCACHE_DISABLE=1
cd "$ROOT"
if [ ! -x node/bin/node ]; then
  curl -fL --retry 3 https://nodejs.org/dist/v24.13.0/node-v24.13.0-linux-arm64.tar.xz -o node.tar.xz
  mkdir -p node
  tar -xJf node.tar.xz -C node --strip-components=1
fi
cd source
for variant in baseline kernels openmp combined; do
  case "$variant" in
    baseline) dispatch=0; omp=0;; kernels) dispatch=1; omp=0;;
    openmp) dispatch=0; omp=1;; combined) dispatch=1; omp=1;;
  esac
  mkdir -p "$ROOT/builds/$variant"
  export L2S1_PORTABLE_BUILD=1 L2S1_ARM64_DISPATCH="$dispatch" L2S1_OPENMP="$omp"
  echo "START $variant $(date -Is)"
  cargo build --release --locked --features llama --bin l2s1 --message-format=json-render-diagnostics > "$ROOT/builds/$variant/cargo.jsonl" 2> "$ROOT/builds/$variant/build.log"
  node sdks/typescript/scripts/bundle-runtime.mjs --cargo-log "$ROOT/builds/$variant/cargo.jsonl" --output "$ROOT/builds/$variant/package"
  node sdks/typescript/scripts/verify-runtime.mjs "$ROOT/builds/$variant/package"
  npm pack "$ROOT/builds/$variant/package" --pack-destination "$ROOT/builds/$variant" --ignore-scripts
  echo "DONE $variant $(date -Is)"
done
