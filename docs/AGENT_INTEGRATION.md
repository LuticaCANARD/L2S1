# AI agent integration

L2S1 includes a portable [skill](../skills/l2s1/SKILL.md) and a stdio MCP adapter.
The skill teaches agents to design requests, choose Rust/CLI/HTTP integration,
validate the actual model, and preserve abstention and evidence. MCP exposes
maintained repository documents and typed tools so an agent does not need shell
access to browse the library or call an already running backend.

## Install the skill

Copy the complete `skills/l2s1` directory into your agent's skill directory. For
agents that discover project skills under `.agents/skills`, from an L2S1 checkout:

```sh
mkdir -p /path/to/consumer-project/.agents/skills
cp -R skills/l2s1 /path/to/consumer-project/.agents/skills/l2s1
```

Keep `references/` and `agents/` with `SKILL.md`. Choose a destination that does
not already contain a different skill; review an existing copy before replacing
it. The skill has automatic discovery enabled and can also be invoked as
`$l2s1` in clients that support named skills. It works without MCP through the
Rust, CLI or HTTP interfaces. The repository keeps the distributable source at
`skills/l2s1`; installing it into an agent is a separate step.

## Install and connect MCP

The adapter uses the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
v1 API and [stdio transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports).
Python 3.10+ is required. Dependencies are isolated from the native Rust build:

```sh
python3 -m venv /path/to/l2s1-mcp-venv
/path/to/l2s1-mcp-venv/bin/python -m pip install -r /path/to/L2S1/mcp/requirements.txt
```

Register the following command in a stdio-capable MCP client. Replace the absolute
paths; no working-directory assumption is needed. Clients that use a
`mcpServers` JSON configuration can use:

```json
{
  "mcpServers": {
    "l2s1": {
      "command": "/path/to/l2s1-mcp-venv/bin/python",
      "args": [
        "/path/to/L2S1/mcp/server.py",
        "--backend-url", "http://127.0.0.1:8080"
      ]
    }
  }
}
```

The client launches the adapter. Manual command inspection is available with
`python /path/to/L2S1/mcp/server.py --help`; running without `--help` waits for MCP
messages on stdin. Stdout is reserved for protocol messages.

Documents, examples, schema and offline validation are immediately available.
For inference, separately start the native HTTP backend (CPU example):

```sh
cargo build --release --locked --features llama --bin l2s1
./target/release/l2s1 --model /path/to/model.gguf --listen 127.0.0.1:8080
```

The backend owns the model and stays resident across calls. Use `llama-cuda` with
`--device cuda`, or `llama-metal` with `--device metal`, when required. For images
also supply the matching `--mmproj`. The adapter defaults to
`http://127.0.0.1:8080` and a 180-second request timeout, configurable with
`--backend-url` and `--timeout`. A copied server can use `--root /path/to/L2S1`.
It requires a source checkout for the maintained documents and examples.

The native API is JSON HTTP, **not** a Streamable HTTP MCP endpoint. The adapter
does not start the backend, download weights, execute selected actions, or retry
failed inference. It connects only to its configured origin and does not inherit
HTTP proxies or credentials. Keep native HTTP on loopback; this adapter does not
provide remote authentication. A backend configured with OpenRouter sends state
and media to the remote provider and can incur charges.

## Agent interface

| Tool | Behavior |
| --- | --- |
| `l2s1_document(name)` | Read `overview`, `guide`, `models`, `verification`, `parallel`, `tools`, `agents`, or `skill` from an allowlist |
| `l2s1_example(name)` | Get the `warehouse` text request or `image` HTTP template |
| `l2s1_validate(request)` | Check v1 request structure, IDs, ordinal order, media references, sizes and optional reasoning/policy fields without a model |
| `l2s1_capabilities()` | Query the resident backend's evidence type, modality support and limits |
| `l2s1_decide(request)` | Return the native HTTP result unchanged, including abstention, scores and usage |

Both request tools expose detailed JSON Schema for `binary`, `choice`, `ordinal`,
and image fields during tool discovery. Static resources expose the same
documents at `l2s1://docs/<name>`, examples at `l2s1://examples/<name>`, and schema
at `l2s1://schema/request`. Tool equivalents support clients that do not expose
resources to their agent. The `design_decision(task)` prompt assists request design.

The adapter supports HTTP v1 fields: `state`, `decisions`, optional `media`,
and per-decision `media_ids`, plus optional `reasoning`, `policy`,
`target_error_rate` and `failure_reasons`. Inspect capabilities before using optional
features; an older backend can reject fields it does not support. A request error
rate is a score threshold, not a correctness guarantee. Unknown fields are rejected.
The adapter checks the HTTP ceilings of 128 decisions, 4 media items,
8 MiB per decoded image and 44 MiB
per body. Validation is structural: image decoding, model/token/context
preflight, and backend-specific limits still require the runtime. Images in the
template use a placeholder that must be replaced with actual standard base64 bytes.

A normal workflow reads the guide/example, designs the request, validates it,
inspects backend capabilities and then calls `l2s1_decide`. An abstained result
is successful tool execution with `status: "abstained"` and a null selection.
Transport, validation and backend failures become MCP tool errors. Backend HTTP
status and its structured error body are retained in the error message. A timeout
can leave inference running; the adapter does not automatically resubmit it.

## Verification

```sh
python /path/to/L2S1/mcp/server.py --help
python -m unittest discover -s mcp -p 'test_*.py' -v
python /path/to/L2S1/mcp/smoke_client.py
```

Run these with the virtual environment's Python. Tests use an HTTP fixture and an
official SDK client over subprocess stdio; they cover discovery, schema and
resources, validation failures, unchanged evidence/abstention, backend errors and
the prompt, optional policy/reasoning fields, and timeouts without retries. The
CI is configured to run the same tests on Python 3.10 and 3.14 without model weights.
The smoke client requires the real HTTP backend and verifies one warehouse request
across the actual MCP → HTTP → model path. It reports
selections and abstentions without assuming a particular model answer.
