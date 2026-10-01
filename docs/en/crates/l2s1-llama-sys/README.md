<a id="l2s1-llamacpp-native-dependency"></a>
# L2S1 llama.cpp native dependency

[English](README.md) · [한국어](../../../ko/crates/l2s1-llama-sys/README.md) · [日本語](../../../ja/crates/l2s1-llama-sys/README.md)

[English index](../../README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../../ja/README.md)

This crate implements L2S1's model ownership, text/vision scheduling, prefix reuse,
state snapshots, bounded thinking and compact evidence in Rust. It builds the
upstream llama.cpp/GGML/mtmd runtime and generates Rust ABI bindings from the
exact same source headers. The small `native/chat.cpp` adapter retains upstream
Jinja rendering so model templates keep their existing behavior. A narrow
`native/exception.cpp` adapter catches upstream C++ exceptions before they can
unwind across Rust frames; scheduling and scoring remain in Rust.
CMake FetchContent downloads ggml-org/llama.cpp commit
`3d82ef62d47fd74e18f36c5eccbdcf965b617b17` and verifies the source archive's
SHA-256 before building. See `cmake/CMakeLists.txt` and `UPSTREAM_COMMIT`.
The native source is not checked into this repository. The upstream source archive
contains its license; this crate includes `THIRD_PARTY_LICENSES.txt`.

The default build is CPU-only. Enable `l2s1/llama-cuda` for CUDA or
`l2s1/llama-metal` for Metal on macOS. CMake 3.24+ and a C++17 compiler are
required, along with Clang/libclang for bindgen; CUDA additionally requires the CUDA toolkit.
On Debian/Ubuntu install `libclang-dev`; on macOS install Homebrew `llvm` and
set `LIBCLANG_PATH="$(brew --prefix llvm)/lib"` if discovery fails. Windows
builds use LLVM (`LIBCLANG_PATH` points to its `bin` directory) alongside MSVC. CUDA and Metal cannot
be enabled together in one build.
On macOS and Linux, the `l2s1` build script embeds the llama.cpp library
directory as an executable rpath. Installed native libraries also search their
own directory for dependent llama.cpp and GGML libraries. Downstream executables
can read `DEP_L2S1_LIBDIR` from their build
script and add their own rpath. Native logs default to warnings; set
`L2S1_LOG=error|warn|info|debug|off` before process startup to change the
level. Model-load failures include the last llama.cpp error message.
The CUDA build disables host-specific architecture selection for distributable
artifacts. Set `L2S1_CUDA_ARCHITECTURES` (for example `86`) to limit the CUDA
architectures when building for a known target.
Set `L2S1_NATIVE_COMPILER_LAUNCHER` to a compiler launcher such as the full
path to `ccache` to wrap llama.cpp's C, C++, and CUDA compilation. The
`scripts/build_cuda_arch.sh` entry point detects `ccache` automatically when
installed. Set `RUSTC_WRAPPER=sccache` explicitly to try Rust caching. The `cc`
crate also uses `sccache` for the retained C++ Jinja adapter when that Rust wrapper is
selected.

The first default build needs network access to download the pinned source.
For an offline build or another llama.cpp source checkout, set
`L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp`. The legacy `LLAMA_CPP_DIR` is accepted
when the new variable is unset. The selected checkout must contain `include/llama.h`,
`src/llama-ext.h`, `tools/mtmd/mtmd.h`, `common/jinja`, and their dependencies. This crate builds its own
native libraries from that source; a separately built library is never mixed
with its headers. Alternate upstream revisions are not guaranteed compatible
and must pass the native contract tests before use.

The build includes `libmtmd` for direct still-image input and links it with the
same `libllama` revision. Video support is disabled. Source and binary packages
must include the matching `libmtmd` and GGML shared libraries.

Publish this native package before the `l2s1` package, which depends on its
versioned release. Prebuilt executables also need the native shared libraries
packaged with an appropriate loader path; a source-package build does not
produce a relocatable binary archive by itself.

The stable `sd_*` ABI remains available to Rust callers. The former
`native/bridge.cpp` is replaced by `src/bridge.rs`, `src/text.rs` and
`src/vision.rs`. Native allocations have scoped Rust owners; error paths clear
possibly dirty KV state. Bindings and Rust implementation sources participate
in the runtime fingerprint. This is not a Rust reimplementation of llama.cpp.
See the [C++ application SDK](../../../../sdks/cpp/README.md) for typed resident-process use.

See [migration validation](../../../RUST_BRIDGE_MIGRATION.md) for the exact
C++/Rust comparison scope and opt-in reproduction commands.


## Portable Linux ARM64

`L2S1_PORTABLE_BUILD=1` keeps `GGML_NATIVE=OFF`. On Linux aarch64 it
builds upstream's loadable ARM CPU variants, including the ARMv8 baseline.
GGML checks Linux HWCAP/HWCAP2 before selecting dotprod, FP16, SVE, i8mm or SME
variants. Feature checks are compiled without the variant ISA flags and with
LTO disabled. The bridge loads modules beside the linked GGML library; keep the
whole matching library set when moving a binary. Cargo tests use that same set.
GCC 14 is the validated compiler for this upstream variant list (including ARMv9.2/SME). The packaged
Linux ARM64 CI build uses Ubuntu 24.04 and GCC 14; it requires a compatible glibc
and C++ runtime (older Linux distributions need a source build).

`L2S1_ARM64_DISPATCH=0` selects the former portable CPU implementation for
controlled comparisons. `L2S1_OPENMP=0` or `1` independently disables or enables
OpenMP. These switches are part of the native build cache key. Portable builds
keep OpenMP off by default; custom OpenMP bundles require their compiler's
OpenMP runtime, such as `libgomp.so.1` with GCC. The package verifier checks
runtime dependencies of every shared library, including dynamically loaded CPU
modules. Do not substitute a library from another build or upstream revision.

The Rust evidence SIMD scan does not accelerate GGML matrix multiplication.
Measure inference with `--execution-mode fresh` when comparing CPU kernels;
resident prefix reuse changes the amount of work performed.
