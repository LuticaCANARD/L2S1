# Build and interfaces

[English](../../../docs/en/skills/l2s1/references/interfaces.md) · [한국어](../../../docs/ko/skills/l2s1/references/interfaces.md) · [日本語](../../../docs/ja/skills/l2s1/references/interfaces.md)

[English index](../../../docs/en/README.md) · [한국어 색인](../../../docs/ko/README.md) · [日本語索引](../../../docs/ja/README.md)

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
`serde_json`. Construct the request directly with Rust types; no input file or JSON
parsing is needed. Load once and retain the backend for multiple decisions:

```rust
use std::path::Path;
use l2s1::{
    Decision, DecisionBackend, DecisionKind, DecisionPolicy, DecisionRequest,
    Level, OptionSpec, llama::LlamaBackend,
};
use serde_json::{Map, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = DecisionRequest {
        state: Value::Object(Map::from_iter([
            ("shipment_id".into(), Value::from("BOX-103")),
            ("storage_requirement".into(), Value::from("chilled")),
            ("hours_until_dispatch".into(), Value::from(4)),
        ])),
        decisions: vec![
            Decision {
                id: "storage_zone".into(),
                instruction: "Select the storage zone that matches the shipment's storage_requirement.".into(),
                kind: DecisionKind::Choice {
                    options: vec![
                        OptionSpec {
                            id: "ambient".into(),
                            criterion: "The shipment requires ambient storage.".into(),
                        },
                        OptionSpec {
                            id: "chilled".into(),
                            criterion: "The shipment requires chilled storage.".into(),
                        },
                        OptionSpec {
                            id: "frozen".into(),
                            criterion: "The shipment requires frozen storage.".into(),
                        },
                    ],
                },
            },
            Decision {
                id: "cold_chain_required".into(),
                instruction: "Does this shipment need temperature-controlled storage? Chilled and frozen shipments do; ambient shipments do not.".into(),
                kind: DecisionKind::Binary {
                    false_label: "No temperature control is required.".into(),
                    true_label: "Temperature control is required.".into(),
                },
            },
            Decision {
                id: "dispatch_priority".into(),
                instruction: "Choose the priority using hours_until_dispatch and the exact thresholds in the levels.".into(),
                kind: DecisionKind::Ordinal {
                    levels: vec![
                        Level {
                            id: "low".into(),
                            criterion: "More than 24 hours remain until dispatch.".into(),
                            value: 0.0,
                        },
                        Level {
                            id: "medium".into(),
                            criterion: "More than 6 hours and at most 24 hours remain until dispatch.".into(),
                            value: 1.0,
                        },
                        Level {
                            id: "high".into(),
                            criterion: "At most 6 hours remain until dispatch.".into(),
                            value: 2.0,
                        },
                    ],
                },
            },
        ],
    };
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
