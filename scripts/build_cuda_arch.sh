#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/build_cuda_arch.sh CUDA_ARCH [CUDA_ARCH ...]

Build one L2S1 CUDA executable per compute capability, for example 86 or 89.
Each architecture has its own Cargo target directory and matching native libraries.
Set L2S1_CUDA_TARGET_ROOT to change the output root (default: target/cuda-architectures).
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
if (( $# == 0 )); then
  usage >&2
  exit 2
fi

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
target_root=${L2S1_CUDA_TARGET_ROOT:-target/cuda-architectures}

for arch in "$@"; do
  if [[ ! $arch =~ ^[0-9]{2,3}$ ]]; then
    printf 'Invalid CUDA architecture: %s\n' "$arch" >&2
    exit 2
  fi
done
command -v readelf >/dev/null || {
  printf 'readelf is required to locate matching native libraries\n' >&2
  exit 1
}

for arch in "$@"; do
  target_dir="$target_root/sm_$arch"
  printf 'Building CUDA sm_%s in %s\n' "$arch" "$target_dir"
  L2S1_CUDA_ARCHITECTURES="$arch" CARGO_TARGET_DIR="$target_dir" \
    cargo build --release --locked --features llama-cuda --bin l2s1
  binary="$target_dir/release/l2s1"
  test -x "$binary"
  native_runpath=$(readelf -d "$binary" | sed -n 's/.*Library runpath: \[\([^]]*\)\].*/\1/p')
  native_lib_dir=${native_runpath%%:*}
  if [[ ! -f $native_lib_dir/libllama.so ]]; then
    printf 'Could not find matching native libraries for %s\n' "$binary" >&2
    exit 1
  fi
  native_out="$target_dir/release/native-lib"
  mkdir -p "$native_out"
  cp -a "$native_lib_dir"/lib*.so* "$native_out/"
  cat >"$target_dir/release/run-l2s1" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
release_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export LD_LIBRARY_PATH="$release_dir/native-lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$release_dir/l2s1" "$@"
EOF
  chmod +x "$target_dir/release/run-l2s1"
  printf 'Built %s\n' "$target_dir/release/run-l2s1"
done
