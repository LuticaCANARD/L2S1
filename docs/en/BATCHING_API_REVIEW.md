# Repeated decisions and native batching

Rust, TypeScript and Python can keep one model resident and change only request
data. TS `prepare<State>(decisions)` and Python `prepare(decisions, state_type=State)`
snapshot fixed definitions. Python supplies Pydantic validation and PEP 561 types.

| Boundary | API | Execution |
| --- | --- | --- |
| Rust library | `LlamaBackend::decide_batch(&requests)` | Independent native sequences in parallel mode; serial in other modes |
| Local TS | `load({ executionMode: 'parallel' })` → `decideBatch(requests)` | Default stdio, one array sent to the compiled Rust process |
| Local Python | `await load(LoadOptions(execution_mode="parallel", ...))` → `decide_batch(requests)` | Default stdio, same Rust executable and runtime bundle |
| HTTP | `POST /v1/decision-batches`, `{"requests": [...]}` | One native parallel array call |
| Rust collection worker | `BackendWorker::spawn_batched()` | Microbatches bounded by request count, input tokens and wait time |

## Compiled mode without a port

Run `l2s1 --model model.gguf --execution-mode parallel --stdio`. SDK `load()` uses
this by default and exchanges JSON lines through stdin/stdout. Python needs no
HTTP server or Node.js. This is a resident compiled process, not in-process N-API
or PyO3 bindings. Rust callers can link the library directly. Explicit local HTTP
uses `transport: 'http'` / `transport="http"`; remote servers use `connect()`.

The RPC envelope is `{id:"local-1", op:"decide_batch", body:{requests:[...]}}`.
Operations are `health`, `capabilities`, `decide` and `decide_batch`. Replies contain
the same id and `result` or `error`. Stdout is reserved for RPC and native logs use
stderr. SDKs admit 16 outstanding calls; a timed-out/cancelled call retains its slot
until its reply arrives. Closing reaps the child. Timeout does not stop native inference.

## Native batch contract

All wire inputs are validated before inference. Media groups are flattened into
one native batch dispatch, then Rust parallel sequences execute in bounded waves.
State, media, IDs, policy and failure messages remain independent. Results follow
input order; IDs may repeat across requests. Request policies apply only to their
own response and preserve raw scores.

Success is `{api_version:1, request_id, execution:"native_parallel", responses:[...]}`.
Each item is a v1 response with `request_id = batch-id/input-index`. Stdio and HTTP
share the wire contract, allowing TS/Python request and response interoperability.

- At most 128 requests, 128 total decisions and 44 MiB per batch body.
- `parallel_width` / `parallelWidth` bounds each wave; larger batches use multiple waves.
- Current llama.cpp parallel execution supports direct reasoning and at most 26 options per decision.
- Use all text, or one image for every decision with a matching projector. Mixed text/image
  groups are rejected; existing recurrent/hybrid parallel limitations still apply.
- `capabilities().batch` describes support, activation, media, reasoning and limits.
- Unsupported backends return `batch_unsupported`; inactive parallel mode returns
  `batch_not_enabled`. There is no serial fallback or automatic retry. Custom backends
  must implement the native batch method.
- Timeout covers the whole call. An execution error fails the whole batch; completed
  waves are not rolled back or replayed. This is separate from an item-level Result API.

## Repetition and performance evidence

`Promise.all(engine.decide(...))` and `asyncio.gather(...)` enqueue separate calls.
The current stdio/HTTP paths do not automatically combine them. Use explicit
`decideBatch()` / `decide_batch()`. Automatic coalescing would require adapting wire
media, reasoning, policy and deadlines to the existing bounded Rust worker.

Preparation reuses definitions and types, not compiled tokens or persistent KV.
Changing state misses the existing `(state, decision)` preparation cache.
`SharedStateSession` instead fixes state and varies questions. Native waves share
only exact token prefixes and maintain isolated question suffix sequences.

Batch shape can change scores, selection and abstention. Measure throughput, total
completion time, p50/p95, memory, score differences, top-1 and policy acceptance
separately. Fixture stdio/HTTP and TS↔Python JSON tests do not establish model
performance. Real GGUF CPU smoke does not verify CUDA/Metal speed or numerical equivalence.
