# L2S1 for TypeScript

[English](README.md) · [한국어](../../ko/typescript/README.md) · [日本語](../../ja/typescript/README.md)

[English index](../README.md) · [한국어 색인](../../ko/README.md) · [日本語索引](../../ja/README.md)

`@l2s1/node` uses the existing Rust inference engine from Node.js. One model process stays resident across calls. Binary, choice, ordinal, image, policy, reasoning and evidence fields use the Rust HTTP v1 schema. Validation, tokenization, scoring, abstention and GPU execution remain in Rust.

Node.js 22+ is required. The wrapper selects an optional prebuilt runtime package for the current OS and CPU architecture. Each runtime package contains the Rust executable and matching llama.cpp/GGML shared libraries; installation runs no compilation or download scripts. Model weights are supplied separately. [Version 0.1.1](https://www.npmjs.com/package/@l2s1/node/v/0.1.1) is published on npm.

See the [npm publishing review](PUBLISHING.md) for scope access, package ordering, authentication and platform release conditions.

## Install

From npm:

```sh
npm install @l2s1/node@0.1.1
```

The wrapper selects the matching runtime through optional dependencies. Keep optional dependencies enabled. Runtime and wrapper versions must match.

### Install from this repository

You can also install the wrapper and platform runtime tarballs from [v0.1.1](https://github.com/LuticaCANARD/L2S1/releases/tag/v0.1.1). For example, on Linux x64:

```sh
npm install ./l2s1-node-0.1.1.tgz ./l2s1-runtime-linux-x64-0.1.1.tgz
```

For a custom native build, build the Rust executable at the repository root:

```sh
cargo build --release --locked --features llama --bin l2s1
```

For NVIDIA CUDA use `--features llama-cuda`; for macOS Metal use `--features llama-metal`. See the [native build guide](../GUIDE.md#build) for toolchain requirements and shared libraries. Keep the executable and its required native libraries together when distributing it.

Build and pack the TypeScript package:

```sh
cd sdks/typescript
npm ci
npm run build
npm pack
# In your application:
npm install /path/to/L2S1/sdks/typescript/l2s1-node-0.1.1.tgz
```

## Load a local model

```ts
import { L2S1 } from '@l2s1/node';

const engine = await L2S1.load({
  model: '/path/to/chat-model.gguf',
  device: 'cpu',
});
try {
  const response = await engine.decide({
    state: { temperature_c: 6 },
    decisions: [{
      id: 'cold',
      instruction: 'Is temperature_c below 10?',
      kind: {
        type: 'binary',
        false_label: 'Temperature is at least 10.',
        true_label: 'Temperature is below 10.',
      },
    }],
  });
  for (const result of response.results) {
    console.log(result.id, result.value, result.status);
    if (result.evidence.type === 'model_scored') {
      console.log(result.evidence.estimate.p_true, result.evidence.candidate_mass);
    }
  }
} finally {
  await engine.close();
}
```

`load()` selects the installed runtime for the current OS/architecture. Override it
with `binaryPath: '/path/to/l2s1'` or a name on PATH. The default
`transport: 'stdio'` starts the compiled Rust executable once and exchanges JSON
lines through stdin/stdout; no HTTP server or port is opened. This is a resident
native process, not an in-process N-API binding. Paths with spaces work.
`transport: 'http'` explicitly starts a loopback server on an OS-selected port.
Both transports wait for health and reuse the same model process. `close()` is
idempotent, cancels local calls, terminates the owned process and waits for exit.
A single call's timeout/cancellation does not guarantee native inference stopped.
Use `connect()` below for a shared server. Stdio admits at most 16 outstanding
calls; timed-out calls occupy their slot until the native reply arrives.

The build workflow covers Linux x64/arm64 (glibc), macOS x64/arm64 and Windows x64. Linux and Windows packages expose CPU; macOS arm64 exposes CPU and Metal. CUDA and other custom builds use `binaryPath`. Linux packages require the system glibc/C++ runtime; Windows packages require the Microsoft Visual C++ x64 runtime. Unsupported platforms produce a clear error. These are workflow targets; local verification on one platform does not establish that the other platform artifacts have passed CI.

TypeScript applications supporting explicit resource management can write `await using engine = await L2S1.load(...)`; this calls `close()` on scope exit. The package is ESM.

`LoadOptions` exposes CPU/CUDA/Metal, context/batch/thread counts, a vision projector (`mmproj`), LoRA, execution mode, parallel width, prompt layout/detail and startup policy. Less common Rust flags can be passed in `extraArgs`; `--stdio` and `--listen` are reserved. `startupTimeoutMs` defaults to 120,000 and call `timeoutMs` to 180,000. A startup `signal` cancels loading. `onStderr` receives native log chunks. Calls accept `{ signal, timeoutMs }` as their second argument. Failed inference requests are never automatically retried.

Run the [warehouse example](../../../sdks/typescript/examples/warehouse.ts) with Node.js 24's TypeScript support after building:

```sh
L2S1_BINARY=/absolute/path/to/l2s1 node examples/warehouse.ts /path/to/chat-model.gguf
```

## Connect to a server

Application code can keep one API while switching backends:

```ts
import { L2S1, type DecisionBackend } from '@l2s1/node';

const local = await L2S1.load({ model: '/path/to/model.gguf' });
const remote = L2S1.connect({ baseUrl: 'https://inference.example.com/l2s1' });
// Both expose decide(request), capabilities(), close() and async disposal.
// const response = await remote.decide(request);

function useCustomBackend(backend: DecisionBackend) {
  return L2S1.fromBackend(backend);
}
```

`DecisionBackend` requires async `decide(request, options)` and `capabilities(options)` methods returning the exported response types. An optional `close()` hook releases owned resources. This permits custom IPC, RPC or other runtime adapters without changing application decision code. `fromBackend()` transfers lifecycle ownership to the facade. Closing an HTTP connection cancels this client's requests and leaves the shared remote server running.

## Repeat fixed decisions with new state

```ts
const batchEngine = await L2S1.load({
  model: '/path/to/model.gguf', executionMode: 'parallel', parallelWidth: 4,
});
type Temperature = { temperature_c: number };
const plan = batchEngine.prepare<Temperature>([{
  id: 'cold', instruction: 'Is temperature_c below 10?',
  kind: { type: 'binary', false_label: 'At least 10.', true_label: 'Below 10.' },
}]);
const first = await plan.decide({ temperature_c: 6 });
const second = await plan.decide({ temperature_c: 15 });
const responses = await plan.decideBatch([{ temperature_c: 2 }, { temperature_c: 20 }]);
// Different definitions per item: await batchEngine.decideBatch(requests).
await batchEngine.close();
```

`prepare()` snapshots fixed definitions and supplies a typed state input. It does
not compile model tokens or retain KV state across requests. `decideBatch()`
sends one array to Rust's native parallel execution path through stdio, or one
`POST /v1/decision-batches` for HTTP. Load with `executionMode: 'parallel'`;
`parallelWidth` controls the wave width. Request state, media, ID and policy remain
independent; response order follows input order. The envelope reports
`execution: 'native_parallel'`. `timeoutMs` applies to the whole batch.
There is no serial fallback or automatic retry, including after an inference
failure. `batch_unsupported` and `batch_not_enabled` are explicit errors.
A custom `DecisionBackend` can implement optional `decideBatch()` for native batching.

Wire batches allow 128 requests and 128 total decisions. Current native parallel
execution supports direct reasoning and at most 26 options per decision. Use
all text, or one image for every decision with a matching projector; mixed text
and image groups are rejected. All wire requests are validated before inference;
an execution failure fails the whole batch without rolling back executed waves.
Check `capabilities().batch`. `Promise.all(engine.decide(...))` queues separate
requests; it does not combine them into a native batch. See the
[batching review](../BATCHING_API_REVIEW.md).

The [Python SDK](../python/README.md) exposes the same lifecycle and HTTP v1 JSON
contract, with `prepare(..., state_type=State)` and `decide_batch()` equivalents.
Both SDKs can use the same extracted runtime bundle, Rust server and GGUF files.

Use `@l2s1/node/http` for an existing Rust server. This subpath has no Node built-in imports and can also be bundled for browsers. Browser calls need a same-origin proxy or a reverse proxy that provides CORS; the Rust server does not add CORS headers. This does not run native inference inside the browser.

```ts
import { L2S1Client, L2S1Error } from '@l2s1/node/http';

const client = new L2S1Client({
  baseUrl: 'http://127.0.0.1:8080',
  // headers: { Authorization: 'Bearer proxy-token' },
});
await client.health();
const capabilities = await client.capabilities();
// await client.decide(request, { signal: abortController.signal });
```

`L2S1Error` retains `code`, `status`, `requestId` and `userReason` from Rust failures. Transport cancellation and network errors retain the original fetch error. The client checks the versioned response envelope, decision order/IDs/kinds, result status and evidence kind. Model scores and provider selections have distinct TypeScript evidence types; `selection_only` has no probabilities. The client also supports the Rust wgpu and OpenRouter HTTP servers.

## Policies, reasoning and images

Request `policy` uses `{ min_top_probability, min_candidate_mass }`. Both must be supplied. `target_error_rate` maps to `min_top_probability = 1 - rate`; it is not a correctness guarantee. `failure_reasons` supplies messages for known abstention/failure codes. `reasoning: { mode: 'thinking', max_tokens: 128 }` requires a backend and model advertising support; unsupported requests fail in Rust. Read `capabilities()` before selecting optional features.

The bundled v1 engine accepts request-local policy, error budgets and failure messages. It advertises direct reasoning only; thinking requests fail explicitly. Older servers reject unsupported fields with HTTP 400. `L2S1.load({ policy })` also sets the default startup policy. The client never silently ignores a requested control.

Images use standard base64 without a data URL prefix:

```ts
const request = {
  state: { task: 'classify the image' },
  media: [{ type: 'image' as const, id: 'photo', data_base64: imageBytes.toString('base64') }],
  decisions: [{
    id: 'cat', instruction: 'Is there a cat in the image?', media_ids: ['photo'],
    kind: { type: 'binary' as const, false_label: 'No cat.', true_label: 'A cat is present.' },
  }],
};
// Load with { model: visionModel, mmproj: matchingProjector } before deciding.
```

Omitted `media_ids` uses all request media; `[]` selects text only. Rust enforces media, decision, body and model limits. See the [HTTP contract](../GUIDE.md#direct-image-input-and-http-api).

## Verification and portability

```sh
npm run check
npm test
npm run test:rust
npm pack --dry-run
```

`test:rust` builds a small deterministic Rust backend and exercises the actual Rust scoring, validation and stdio/HTTP batch envelopes through managed Node calls. It needs a Rust toolchain but no model or C++ toolchain. It is fixture evidence, not model quality evidence.

Optional real-model smoke (at the repository root, build the native binary first):

```sh
cd sdks/typescript
L2S1_BINARY=/absolute/path/to/l2s1 L2S1_MODEL=/path/to/chat-model.gguf node --test test/model.integration.mjs
```

WASM is a separate runtime port. Compiling the Rust wrapper for WASM does not package the C++ inference engine or preserve CUDA/Metal execution. A WASM distribution needs a separately built inference engine, its JS/WASM boundary and a CPU/WebGPU path. This package uses prebuilt native runtime packages instead.

## Build distribution artifacts

The [runtime workflow](../../../.github/workflows/typescript-runtimes.yml) runs on `v*` version-tag pushes or manual dispatch, and builds all five platforms and uploads wrapper/runtime `.tgz` files as workflow artifacts. It does not publish packages. Publish all runtime packages at the matching version before publishing the wrapper. The workflow verifies checksums, executable startup, installation into a fresh npm project and automatic runtime resolution. Invalid-model startup in CI is not model inference validation.

To build the current platform locally, run at the repository root:

```sh
# On macOS arm64, use llama-metal and --devices cpu,metal to include Metal.
L2S1_PORTABLE_BUILD=1 cargo build --release --locked --features llama \
  --bin l2s1 --message-format=json-render-diagnostics > native-build.jsonl
node sdks/typescript/scripts/bundle-runtime.mjs --cargo-log native-build.jsonl
node sdks/typescript/scripts/verify-runtime.mjs sdks/typescript/runtime-packages/linux-x64
cd sdks/typescript/runtime-packages/linux-x64
npm pack --pack-destination ../../ --ignore-scripts
cd ../..
npm pack
npm run test:package -- l2s1-node-0.1.1.tgz l2s1-runtime-linux-x64-0.1.1.tgz
```

Use a fresh output directory when rebuilding a runtime. `L2S1_PORTABLE_BUILD=1` disables build-host CPU instructions and OpenMP dependencies; its general CPU kernels may be slower than a host-optimized custom build. Each artifact includes license notices and a SHA-256 manifest. The verifier checks that Linux llama.cpp/GGML dependencies resolve from the bundle directory.

The [release pipeline](../RELEASE_PIPELINE.md) publishes verified SDK and native runtime packages.
