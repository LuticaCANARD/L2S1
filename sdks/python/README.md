# L2S1 for Python

Python 3.11+ async SDK for the resident Rust engine. Requests and responses use the
same HTTP v1 field names, decision kinds, policies and evidence as `@l2s1/node`.
The package includes Pydantic runtime validation and PEP 561 type information.
It is prepared for repository/wheel distribution; it has not been published to PyPI.

## Install

From a checkout:

```sh
python -m pip install ./sdks/python
```

From a built package:

```sh
python -m pip install ./l2s1-0.1.0-py3-none-any.whl
```

The wheel contains the Python SDK. Supply the Rust engine and GGUF weights
separately. `load()` accepts `binary_path`, finds `l2s1` on PATH, or accepts
`runtime_dir` pointing to an extracted `@l2s1/runtime-<platform>` package. This
reuses the TypeScript bundle's executable and matching native libraries without
Node.js. Bundle version, OS/architecture, device and file checksums are checked.
There are no model downloads, native builds or install scripts in the wheel.

## Repeat a decision with new data

```python
import asyncio
from pydantic import BaseModel, ConfigDict
from l2s1 import BinaryKind, BinaryValue, Decision, L2S1, LoadOptions

class Temperature(BaseModel):
    model_config = ConfigDict(strict=True)
    temperature_c: float

async def main() -> None:
    async with await L2S1.load(LoadOptions(
        model="/path/to/model.gguf",
        binary_path="/path/to/l2s1",
        execution_mode="parallel",  # Native batching; transport defaults to stdio.
    )) as engine:
        plan = engine.prepare([Decision(
            id="cold",
            instruction="Is temperature_c below 10?",
            kind=BinaryKind(false_label="At least 10.", true_label="Below 10."),
        )], state_type=Temperature)

        first = await plan.decide(Temperature(temperature_c=6))
        second = await plan.decide(Temperature(temperature_c=15))
        results = await plan.decide_batch([
            Temperature(temperature_c=2), Temperature(temperature_c=20),
        ])
        result = first.results[0]
        if isinstance(result.value, BinaryValue):
            print(result.value.value)  # bool | None; preserve abstention

asyncio.run(main())
```

`prepare()` snapshots fixed definitions. Passing `state_type` returns
`PreparedDecision[Temperature]`: mypy checks calls and the SDK checks the model
type at runtime. Without `state_type`, the plan accepts JSON values. Input schema
checks do not establish model correctness. Preparation here does not precompile
model tokens or retain a KV prefix across requests.

`load()` defaults to `transport="stdio"`: it launches the compiled Rust executable
once and exchanges JSON lines through stdin/stdout. No port, HTTP server or
Node.js is needed. This is a resident native process, not an in-process PyO3
extension. Use `transport="http"` explicitly for a local HTTP server, or
`connect()` for a shared remote server. Both use the same typed JSON contract.

`decide_batch()` sends the entire array once to Rust's native parallel path.
Load with `execution_mode="parallel"`; `parallel_width` controls wave width.
For HTTP connections the endpoint is `POST /v1/decision-batches` with
`{"requests": [...]}`. Responses preserve request order and independent state,
IDs, media and policy; IDs may repeat across requests. The batch envelope
advertises `execution: "native_parallel"`. `timeout_ms` applies to the whole call.
There is no serial fallback or automatic retry. Unsupported backends report
`batch_unsupported`, and engines outside parallel mode report `batch_not_enabled`.
Validate `capabilities().batch` before selecting optional batch features.

The native wire batch accepts at most 128 requests and 128 total decisions.
Current llama.cpp parallel execution supports direct reasoning and at most 26
options per decision. A batch can be all text or use one image for every decision
with a matching projector; mixing text and image groups is rejected. All wire
requests are validated before inference. An execution error fails the whole
batch; already executed waves are not rolled back or replayed. Separate
`asyncio.gather(engine.decide(...))` calls are queued individually and are not
implicitly combined. See [the batching review](../../docs/BATCHING_API_REVIEW.md).

## Connect, send existing JSON, or adapt another transport

```python
from l2s1 import DecisionRequest, L2S1

async with L2S1.connect("http://127.0.0.1:8080") as engine:
    capabilities = await engine.capabilities()
    # payload can be the exact JSON object used by the TypeScript SDK.
    request = DecisionRequest.model_validate(payload)
    response = await engine.decide(request, timeout_ms=180_000)
    payload_for_typescript = response.model_dump(exclude_unset=True)
```

`load()`, `connect()` and `from_backend()` expose the same application API.
`DecisionBackend` is a typed protocol with async `decide()` and `capabilities()`.
`BatchDecisionBackend` adds async `decide_batch()` for custom native adapters.
An optional synchronous or asynchronous `close()` hook transfers resource
ownership to the facade. `L2S1Client` is also available as a standalone transport.

| TypeScript | Python |
| --- | --- |
| `L2S1.load({ model, binaryPath })` | `await L2S1.load(LoadOptions(model=..., binary_path=...))` |
| `L2S1.connect({ baseUrl })` | `L2S1.connect(base_url)` |
| `L2S1.fromBackend(backend)` | `L2S1.from_backend(backend)` |
| `engine.decide(request)` | `await engine.decide(request)` |
| `engine.prepare<State>(decisions)` | `engine.prepare(decisions, state_type=State)` |
| `plan.decide(state)` | `await plan.decide(state)` |
| `plan.decideBatch(states)` | `await plan.decide_batch(states)` |
| `engine.decideBatch(requests)` | `await engine.decide_batch(requests)` |
| `close()` / async disposal | `await close()` / `async with` |
| `AbortSignal` | `asyncio` task cancellation |

Both SDKs preserve `false`, `null` selections, ordered result IDs and
`model_scored` versus `selection_only` evidence. Python uses snake case for
application methods and options; JSON field names are unchanged. Use
`model_dump(exclude_unset=True)` for round trips so absent extension fields stay
absent. Decision-kind discriminator defaults are always serialized.

Inspect `capabilities()` before selecting image, reasoning or request-policy
extensions. Unsupported controls are never silently dropped. Model scores and
`target_error_rate` are not correctness guarantees. `L2S1Error` retains server
`code`, `status`, `request_id` and `user_reason`; network errors and task
cancellation retain their HTTPX/asyncio exceptions. There are no automatic retries.

`close()` is idempotent. Closing an HTTP client cancels its local HTTP tasks and
leaves the remote server running. Closing an owned engine terminates and waits
for the Rust child. Cancelling or timing out a call does not guarantee
that Rust inference stopped. Local stdio startup opens no socket. Explicit HTTP
startup uses an OS-selected loopback port. Native diagnostic output is
continuously drained. The stdio client admits at most 16 outstanding calls;
timed-out calls occupy their slot until the native reply arrives.

## Build and verify

```sh
python -m pip install './sdks/python[dev]'
python -m mypy --config-file sdks/python/pyproject.toml sdks/python/src sdks/python/examples sdks/python/tests/typecheck.py
python -m unittest discover -s sdks/python/tests -v
python -m build sdks/python

# Actual Rust stdio/HTTP/scoring boundary and Python ↔ TypeScript JSON equality:
cargo build --locked --example typescript_fixture
npm --prefix sdks/typescript ci
npm --prefix sdks/typescript run build
L2S1_TEST_BINARY="$PWD/target/debug/examples/typescript_fixture" \
  python -m unittest discover -s sdks/python/tests -v
```

On Windows use `typescript_fixture.exe`. The Rust fixture exercises resident
process lifetime, stdio/HTTP batch validation and scoring without a model; it is not native
GGUF, CUDA or Metal quality/performance evidence. The CI matrix targets Python
3.11 and 3.14 on Linux, macOS and Windows; local verification covers only the
actual host. Wheel/sdist artifacts are uploaded by CI; registry publishing is
not enabled.

The [release pipeline](../../docs/RELEASE_PIPELINE.md) builds and publishes GitHub Release, npm, PyPI and native Cargo packages after validation. External registry account/trusted-publisher setup is required; this local work has not published packages.
