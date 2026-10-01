# CUDA architecture build cache check (2026-09-25)

[English](../../docs/en/benchmarks/build-cache-20260925/REPORT.md) · [한국어](../../docs/ko/benchmarks/build-cache-20260925/REPORT.md) · [日本語](../../docs/ja/benchmarks/build-cache-20260925/REPORT.md)

[English index](../../docs/en/README.md) · [한국어 색인](../../docs/ko/README.md) · [日本語索引](../../docs/ja/README.md)

## Setup

On one 4-core Linux host with an RTX 3080, 15 GiB RAM, and 4 GiB swap, build the same `sm_86` release source from one checkout into separate, previously empty Cargo target directories. Use the complete local llama.cpp source cache, `CARGO_BUILD_JOBS=3`, and `CMAKE_BUILD_PARALLEL_LEVEL=3`. The source is merged PR #28 (`f61b6e6`) plus the compiler-cache changes in this PR. The native build uses `ccache` 4.12.3; runs A and B also use `sccache` 0.13.0 for Rust and the C++ bridge. Both tools were unpacked into ignored local build artifacts for this check; they are not repository dependencies.

| Fresh target directory | Cache state | Rust wrapper | Wall time | Native `ccache` hits |
| --- | --- | --- | ---: | ---: |
| A | Empty cache | `sccache` | 1,749.353 s (29m 09s) | 0/266 |
| B | After A | `sccache` | 171.595 s (2m 52s) | 254/266 (95.49%) |
| C | After B | None | 204.940 s (3m 25s) | 254/266 (95.49%) |

On B, the opt-in Rust `sccache` wrapper reported zero Rust hits and 49 Rust misses across the independent target directory; its nine hits were C/C++ bridge compilations. A smaller Rust-library probe also missed across two different target directories but hit on a forced rebuild within the same directory. The script therefore auto-detects `ccache` and leaves `sccache` opt-in through `RUSTC_WRAPPER`.

The B and C launchers each completed a three-decision Qwen3 0.6B smoke request on the RTX 3080, with 29/29 model layers offloaded. The local build logs and smoke stderr are in ignored `results/compiler-cache-bench-{a,b,c}*` artifacts. Runs B and C were 10.2x and 8.5x faster than A respectively, with 254 native cache hits. These are one run per setup, not repeated statistical trials. The cache-hit count and speedup describe this host, source snapshot, compiler versions, and warm cache; they do not predict another machine's build time. The empty-cache A run was slower than a previous 17-minute uncached build, but these were separate runs and the difference was not isolated to the cache wrapper.
