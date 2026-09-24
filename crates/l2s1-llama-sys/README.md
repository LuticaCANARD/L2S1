# L2S1 llama.cpp native dependency

This crate builds llama.cpp and L2S1's C++ bridge from the same source revision.
CMake FetchContent downloads ggml-org/llama.cpp commit
`3d82ef62d47fd74e18f36c5eccbdcf965b617b17` and verifies the source archive's
SHA-256 before building. See `cmake/CMakeLists.txt` and `UPSTREAM_COMMIT`.
The native source is not checked into this repository. The upstream source archive
contains its license; this crate includes `THIRD_PARTY_LICENSES.txt`.

The default build is CPU-only. Enable `l2s1/llama-cuda` for CUDA; CMake 3.24+,
a C++17 compiler, and the CUDA toolkit are required for that feature.
The CUDA build disables host-specific architecture selection for distributable
artifacts. Set `L2S1_CUDA_ARCHITECTURES` (for example `86`) to limit the CUDA
architectures when building for a known target.

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
