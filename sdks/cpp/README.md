# L2S1 for C++

C++17 SDK for the resident Rust engine. `Engine::load()` starts one executable
with `--stdio`; subsequent calls reuse the loaded model. Binary, choice, ordinal,
image, reasoning, policy and evidence fields follow the same v1 contract as the
Python and TypeScript SDKs. Model execution and validation stay in Rust.

## Build and install

Install CMake 3.18+, a C++17 compiler, and nlohmann/json 3.11+ (MIT licensed).
The SDK does not download dependencies or model weights during configuration.

```sh
cmake -S sdks/cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp --config Release
ctest --test-dir build/cpp -C Release --output-on-failure
cmake --install build/cpp --config Release --prefix /your/prefix
```

For an offline single-header checkout, configure with
`-DL2S1_JSON_INCLUDE_DIR=/directory/containing/nlohmann`. The installed package
expects the consumer to provide `nlohmann_json` through its CMake package config.

In a consuming CMake project:

```cmake
find_package(l2s1_cpp 0.2 CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE l2s1::cpp)
```

Or use `add_subdirectory(path/to/L2S1/sdks/cpp)` and link `l2s1::cpp` directly.
Supply the Rust executable, its matching native libraries, and a GGUF model
separately. Build the executable with `cargo build --release --locked --features
llama --bin l2s1`; CUDA uses `llama-cuda`, macOS Metal uses `llama-metal`.
See the [native build requirements](../../crates/l2s1-llama-sys/README.md).

## Reuse a decision

Omit `execution_mode` to inherit automatic fixed-schema prefix reuse from a
compatible resident runtime. Set `execution_mode = "fresh"` or
`fixed_schema = false` to opt out; `fixed_schema = true` requires support.
An explicit execution mode takes precedence over `fixed_schema = false`.
The runtime clears retained snapshots whenever the ordered decision schema changes,
including when an older prepared plan is used again. State changes reuse the
prefix and compute new results. Check `capabilities()["prefix_reuse"]` and
`Result::usage["reused_prefix_tokens"]`; `prepare()` itself performs no inference.
These C++ defaults and schema invalidation require runtime/SDK 0.2.1 or later.
`execution_mode` is now `std::optional<std::string>`; `std::nullopt` inherits the runtime default.

The example below explicitly selects parallel execution for `decide_batch`:

```cpp
#include <l2s1/l2s1.hpp>

l2s1::LoadOptions options;
options.binary_path = "/path/to/l2s1";
options.model = "/path/to/model.gguf";
options.execution_mode = "parallel";
options.parallel_width = 2;
auto engine = l2s1::Engine::load(options);
auto plan = engine.prepare({{
    "cold", "Is temperature_c below 10?",
    l2s1::Binary{"At least 10.", "Below 10."}, std::nullopt
}});

auto response = plan.decide({{"temperature_c", 6}});
auto value = std::get<l2s1::BinaryValue>(response.results.at(0).value).value;
// value is std::optional<bool>; nullopt means abstention, never false.
auto batch = plan.decide_batch({
    {{"temperature_c", 2}}, {{"temperature_c", 20}}
});
```

`prepare()` copies definitions and options; it does not create a KV cache.
The engine must outlive the plan and must not move while a plan uses it.
`Response::raw` retains the complete server envelope. `Result::evidence` retains
all scoring fields, including candidate mass, probabilities and calibration.
`ChoiceValue::selected` and `OrdinalValue::selected` also preserve abstention.

`Request::options` accepts advanced v1 fields: `media`, `reasoning`, `policy`,
`target_error_rate`, and `failure_reasons`. `Decision::media_ids` distinguishes
omission (all request media) from an empty vector (text only). `decide_json()` and
`decide_batch_json()` accept the complete wire request directly, and still check
response ordering, kinds, selections and abstention. Rust validates requests
before inference. User-supplied failure text is returned as `Error::user_reason`.

Image order follows explicit `media_ids`; omission uses upload order and `[]` selects text only.
llama.cpp `fresh` supports up to 8 images per decision, `parallel` and wgpu one;
llama.cpp `prefix-reuse`/`state-restore` do not support images. Check the running
backend's `media.image.max_per_decision` capability. Requests allow 8 uploads,
up to 8 MiB each, within the 44 MiB body limit.

`decide_batch()` sends one native batch RPC; it does not loop over serial
inference. Select `execution_mode = "parallel"`, and check `capabilities()["batch"]`
for the active model's limits. Responses retain input order. Unsupported batching
returns the Rust error code; there is no automatic fallback or retry.

## Lifetime and errors

The API is synchronous. Serialize calls, moves and `close()` on an instance.
Different instances own independent processes. Destruction and idempotent
`close()` terminate and reap the owned process. Native stderr is inherited so
model diagnostics remain visible without pipe backpressure.

Startup defaults to 120 seconds; calls default to 180 seconds. A timeout closes
and terminates the owned engine to prevent a stale response from reaching the
next call. Create another engine to continue. `Error` exposes `code`, `request_id`
and `user_reason`; a returned inference error leaves the connection usable.
Arguments are passed directly to `posix_spawnp` or `CreateProcessW`, without a
shell. UTF-8 paths with spaces are supported.

There is no HTTP client or bundled runtime in this first C++ package. The process
transport has Linux/macOS and Windows implementations; local verification covers
Linux only. The CI matrix exercises SDK contract fixtures on all three systems;
workflow configuration alone is not evidence that those jobs passed.

Run the real-model example with:

```sh
build/cpp/l2s1_cpp_example /path/to/l2s1 /path/to/model.gguf
```

Contract tests use a fixture child and cover typed values, abstention, batching,
errors, wrong IDs, malformed JSON, paths, timeouts, startup failure and cleanup.
They do not establish real-model quality or GPU behavior.
