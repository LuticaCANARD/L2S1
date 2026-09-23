# L2S1 llama.cpp native dependency

This crate builds llama.cpp and L2S1's C++ bridge from the same source revision.
The bundled default source is ggml-org/llama.cpp commit
`3d82ef62d47fd74e18f36c5eccbdcf965b617b17` (see
`vendor/llama.cpp/UPSTREAM_COMMIT`). The upstream license and third-party notices
are included under `vendor/llama.cpp/` and `THIRD_PARTY_LICENSES.txt`.

The default build is CPU-only. Enable `l2s1/llama-cuda` for CUDA; CMake, a C++17
compiler, and the CUDA toolkit are required for that feature.
The CUDA build disables host-specific architecture selection for distributable
artifacts. Set `L2S1_CUDA_ARCHITECTURES` (for example `86`) to limit the CUDA
architectures when building for a known target.

To build against another llama.cpp source checkout, set
`L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp`. The legacy `LLAMA_CPP_DIR` is accepted
when the new variable is unset. The selected checkout must contain `include/llama.h`,
`src/llama-ext.h`, `common/jinja`, and its dependencies. This crate builds its own
native libraries from that source; a separately built library is never mixed
with its headers. Alternate upstream revisions are not guaranteed compatible
and must pass the native contract tests before use.

Publish this native package before the `l2s1` package, which depends on its
versioned release. Prebuilt executables also need the native shared libraries
packaged with an appropriate loader path; a source-package build does not
produce a relocatable binary archive by itself.
