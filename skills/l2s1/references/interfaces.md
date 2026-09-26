# Build and interfaces

Run repository commands from a verified L2S1 checkout. Model paths are supplied
by the user/application; weights are not bundled. Rust needs edition 2024 support;
native builds need CMake 3.24+ and C++17. Check CLI `--help` when adapting flags.

```sh
cargo build --release --locked --features llama --bin l2s1
./target/release/l2s1 --model /path/to/model.gguf --inspect
./target/release/l2s1 --model /path/to/model.gguf --input request.json --preflight
./target/release/l2s1 --model /path/to/model.gguf --input request.json --diagnostics
# Or pipe a text request to --input -; JSON stdout, native logs stderr.
```

For GPU builds substitute `llama-cuda` / `llama-metal`, and run with `--device cuda`
/ `--device metal`. For offline builds set
`L2S1_LLAMA_CPP_SOURCE=/path/to/matching/llama.cpp`. Source, headers, native bridge
and libraries must be rebuilt together; do not swap an arbitrary runtime library.
If native logs mention an unwritable ccache, use a writable `CCACHE_DIR` or
`CCACHE_DISABLE=1`. Linux loaders may need the matching generated native library
directory on `LD_LIBRARY_PATH`; see `docs/GUIDE.md`.

## Resident HTTP

```sh
./target/release/l2s1 --model /path/to/model.gguf --listen 127.0.0.1:8080
curl -sS http://127.0.0.1:8080/v1/capabilities
curl -sS -H 'Content-Type: application/json' --data-binary @request.json \
  http://127.0.0.1:8080/v1/decisions
```

Health is `GET /healthz`. Keep unauthenticated native HTTP on loopback. For remote
use, provide the application's authenticated reverse proxy. For images, start the
backend with `--mmproj /path/to/matching-projector.gguf`; HTTP uses base64 media,
while CLI uses `--image /path/to/photo.jpg`. Text CLI preflight does not validate
HTTP images.

## MCP

Install `mcp/requirements.txt` in a Python 3.10+ virtual environment and launch
that environment's Python with `/path/to/L2S1/mcp/server.py`. Use
`--backend-url http://127.0.0.1:8080`; `--root /path/to/L2S1` is optional when the
server stays in the repository. Stdio MCP is a separate adapter process, not the
native HTTP endpoint. Do not register `/v1/decisions` as a Streamable HTTP MCP URL.

Tools: `l2s1_document`, `l2s1_example`, `l2s1_validate`, `l2s1_capabilities`,
`l2s1_decide`. Resources include `l2s1://schema/request`,
`l2s1://docs/guide`, and `l2s1://examples/warehouse`. The `design_decision` prompt
helps turn a task into a valid request. See `docs/AGENT_INTEGRATION.md` for client
configuration. The adapter connects to an operator-selected origin and performs
no shell commands, file writes, downloads, model loading or automatic retries.

## Rust

Enable the matching runtime feature on the `l2s1` dependency and include
`serde_json`. Load once and retain the backend for multiple decisions:

```rust
use std::path::Path;
use l2s1::{DecisionBackend, DecisionPolicy, DecisionRequest, llama::LlamaBackend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: DecisionRequest =
        serde_json::from_str(&std::fs::read_to_string("request.json")?)?;
    request.validate()?;
    let mut backend = LlamaBackend::load(
        Path::new("/path/to/model.gguf"),
        2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    let response = backend.decide(&request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    drop(backend);
    Ok(())
}
```

For GPU/vision/options/worker ownership use the current APIs in `docs/GUIDE.md`
and `docs/MODEL_INTERCHANGEABILITY.md`. Calibration and output heads are bound to
their model/task/configuration; changing GGUF does not preserve their validity.
