# Rust native bridge and C++ SDK validation

L2S1's model/context ownership, text decoding, prefix reuse, snapshot restoration,
bounded thinking helper, compact evidence reduction, image projector reuse and
vision batch scheduler now live in Rust. `sd_*` function signatures and C-layout
input/metrics structs remain compatible. Bindgen uses the headers from the same
source checkout that CMake builds; bindings and Rust sources enter the runtime
fingerprint.

The upstream llama.cpp/GGML/mtmd kernels remain C/C++. The small Jinja adapter
retains existing model-template semantics. `native/exception.cpp` protects
fallible upstream initialization, allocation, metadata, memory and decode calls
from C++ exception unwinding through Rust. It contains no decision policy.
Rust owners release contexts, models, batches, bitmaps and chunks on failure.
Failed initialization remains an explicit error on subsequent loads.

## Validation scope

| Check | Observed result |
| --- | --- |
| Linux root unit/integration suite with `llama` | 90 passed; 33 opt-in tests ignored |
| Rust bridge unit tests | 3 passed: greedy ties/nonfinite logits, full-vocabulary normalization, aligned common prefixes |
| Strict sys-crate Clippy and rustfmt | Passed |
| C++/Rust text reference comparison, SmolLM2-135M Q8_0 on CPU | Exact full-vocabulary logits and reused-token counts across repeated prefixes, static/dynamic parallel contexts, snapshot restoration and snapshot-budget fallback |
| C++/Rust vision reference comparison, SmolVLM-256M Q8_0 + matching projector on CPU | Exact full-vocabulary logits and token counts for two-image batches, with projector reuse both disabled and enabled |
| C++17 SDK contract fixtures | Build, tests and install passed on Linux, macOS and Windows CI |
| C++ SDK installed-package consumer on Linux | `find_package(l2s1_cpp)` and `l2s1::cpp` compile/link/run passed with a consumer-provided nlohmann/json package |
| C++ SDK real SmolLM2 inference on Linux | Resident repeated calls and one native batch completed; abstentions remained null optional values |

The reference bridge was saved before migration (SHA-256
`b1baa7f134fbff4bc2c07dba87d27d9e51ee07ce3e6e0edcca90ff00b5eee30d`)
and compiled as a separate shared object with `-Wl,-Bsymbolic`. This prevents
reference `sd_*` calls from resolving to Rust implementations. Both sides use
the same upstream headers and runtime build. The committed opt-in comparison
is `crates/l2s1-llama-sys/tests/cpp_parity.rs`; it compares all vocabulary floats,
not only selected labels.

Additional working-checkout regressions passed for real-model compact evidence,
dynamic context reservation, independent text request batches, and Qwen3-0.6B
thinking completion, budget failure and recovery. The broad CPU benchmark/vision
suites were stopped in favor of the focused reference comparisons above; they
are not counted as completed validation.

This records correctness and integration checks. It does not claim an accuracy
improvement, latency improvement, complete model-family coverage, or new CUDA/
Metal hardware validation. SDK platform fixtures are distinct from running real
models on those platforms.

## Reproduce the opt-in comparisons

Build a saved pre-migration `bridge.cpp` as a shared library against the same
llama.cpp/mtmd checkout and installed libraries used by Cargo, with
`-Wl,-Bsymbolic` on Linux. Then run:

```sh
L2S1_CPP_REFERENCE=/path/to/libreference.so \
SKID_MODEL=/path/to/SmolLM2-135M-Instruct-Q8_0.gguf \
SKID_VISION_MODEL=/path/to/SmolVLM-256M-Instruct-Q8_0.gguf \
SKID_VISION_MMPROJ=/path/to/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf \
L2S1_VISION_IMAGE="$PWD/tests/fixtures/vision_red_64.png" \
cargo test --locked -p l2s1-llama-sys --test cpp_parity -- --ignored --test-threads=1
```

The reference loader test is Linux-only. Normal tests do not require a model or
reference shared library. Native source builds now require libclang as well as
CMake and a C++17 compiler. See the [native crate guide](../crates/l2s1-llama-sys/README.md)
and [C++ SDK guide](../sdks/cpp/README.md).
