# Shared-state prefix reuse, changing questions

Measured 2026-09-25 on an RTX 3080 (compute capability 8.6), using the current working tree at commit `72fdef3eabaae579d640cb7b51d18a29dc83791e` with uncommitted changes. The release CUDA build targeted `sm_86`. Both models used context 2048, batch/ubatch 32, four CPU threads, full evidence and the same state-first prompt in both paths. Model loading and checksum verification were excluded.

This report records that local snapshot. The pull request containing it is based on a later `main`; the timing numbers below were not taken on that later tree.

Each path received one question per call. The 16 questions had distinct IDs and instruction text, so later calls could reuse the state prefix without repeating an identical full prompt. A short warehouse state and a version with 200 irrelevant `packing` words were tested. Five measured rounds per condition each had one untimed warmup; path order alternated between rounds. Times below are medians of complete runs, including session creation/drop and Rust preparation, but excluding loading and report serialization. Ratios are fresh median divided by shared-session median.

The preparation cache remained disabled. `Reused / input tokens` therefore refers to actual native prefix reuse, not tokenization-cache hits.

| Model | State | Questions | Fresh ms | Shared ms | Ratio | Reused / input tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8 | short | 1 | 27.10 | 27.34 | 0.99x | 0 / 151 |
| Qwen3 0.6B Q8 | short | 4 | 143.33 | 113.84 | 1.26x | 192 / 615 |
| Qwen3 0.6B Q8 | short | 16 | 597.02 | 430.50 | 1.39x | 960 / 2477 |
| Qwen3 0.6B Q8 | 200 words | 1 | 89.81 | 94.05 | 0.95x | 0 / 358 |
| Qwen3 0.6B Q8 | 200 words | 4 | 371.47 | 172.70 | 2.15x | 768 / 1443 |
| Qwen3 0.6B Q8 | 200 words | 16 | 1461.77 | 531.49 | 2.75x | 3840 / 5789 |
| Gemma 4 E2B Q8 | short | 1 | 56.34 | 55.60 | 1.01x | 0 / 147 |
| Gemma 4 E2B Q8 | short | 4 | 266.81 | 204.54 | 1.30x | 192 / 601 |
| Gemma 4 E2B Q8 | short | 16 | 1099.71 | 781.90 | 1.41x | 960 / 2423 |
| Gemma 4 E2B Q8 | 200 words | 1 | 156.98 | 157.73 | 1.00x | 0 / 355 |
| Gemma 4 E2B Q8 | 200 words | 4 | 678.41 | 317.08 | 2.14x | 768 / 1433 |
| Gemma 4 E2B Q8 | 200 words | 16 | 2735.54 | 985.42 | 2.78x | 3840 / 5751 |

Across 210 paired decision comparisons per model, raw candidate logits, candidate-relative probabilities, candidate mass, selected values and abstention reasons matched exactly. The report JSON retains per-round timings, reused token counts, model/runtime hashes and numerical differences: [Qwen3](qwen3-0.6b-sm86.json), [Gemma 4](gemma4-e2b-sm86.json). The benchmark source is [`examples/benchmark_shared_state_cache.rs`](../../examples/benchmark_shared_state_cache.rs).

The observed gain supports retaining a prefix for several questions about an unchanged state. A single question has no reuse and no measured benefit. This is an explicit in-process session, not an automatic cross-request LRU: HTTP parsing, cache lookup, eviction, multiple retained states and their memory cost were not measured. The synthetic workload has no quality labels and does not establish production latency or accuracy. The five rounds characterize this host and workload only.

## Architecture-specific build check

`scripts/build_cuda_arch.sh 86` completed in a separate `target/cuda-architectures/sm_86/` Cargo target directory, using an existing local llama.cpp source checkout to avoid a network fetch. Its CUDA compile flags contained only `compute_86` and `sm_86`. The script copied the matching native libraries beside the executable and created `release/run-l2s1`. That launcher completed a three-decision smoke request on the RTX 3080; the local native log confirms 29/29 model layers offloaded to CUDA. See the [build log](build-sm86.log) and [smoke response](sm86-smoke.json). The repetitive native log remains in ignored local results. Other CUDA architectures were not compiled or run in this check.

## Rebuild on the PR branch

Commit `0260180a5c950981cb642a3235b750529b7e4c3a` was also built with `scripts/build_cuda_arch.sh 86` from its clean PR worktree. The release build completed in 17 minutes using a complete local llama.cpp source cache. Its CUDA flags targeted only `compute_86` and `sm_86`; the launcher was created with its matching native libraries. A fresh three-decision Qwen3 0.6B smoke request completed on the RTX 3080, with 29/29 model layers offloaded. The [rebuild evidence](pr28-sm86-rebuild.json) includes the binary, native library and local-log hashes, flags, and scope; the [smoke response](pr28-sm86-smoke.json) contains the backend device and decisions. The shared-state timing table above was not remeasured on this commit.
