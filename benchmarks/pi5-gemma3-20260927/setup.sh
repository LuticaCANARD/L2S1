set -eu
PI_WORK="$HOME/l2s1-pi-20260927"
export CARGO_HOME="$PI_WORK/cargo"
export RUSTUP_HOME="$PI_WORK/rustup"
export PATH="$PI_WORK/venv/bin:$CARGO_HOME/bin:$PATH"
export CARGO_BUILD_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CCACHE_DISABLE=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$PI_WORK"
tar -xzf source.tar.gz -C source
python3 -m venv venv
venv/bin/python -m pip install 'cmake==4.2.3'
curl --proto '=https' --tlsv1.2 --retry 3 -sSf https://sh.rustup.rs -o rustup-init.sh
sh rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain stable
rustc --version
cargo --version
cmake --version
cd source
cargo build --release --locked --features llama --bin l2s1 --example evaluate_jsonl
cargo test --release --locked --features llama --test benchmark --no-run --message-format=json > "$PI_WORK/benchmark-build.jsonl"
