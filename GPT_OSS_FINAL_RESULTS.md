# GPT-OSS final-prefill verification

Measurement version: `gpt-oss-final-prefill-decision-v1`, verified in all 400 output records. Concurrent working-tree changes introduced `state-first-v2` prompt ordering and prefix reuse after this build was started; those changes are not covered by these results or the native-suite claim below. The recorded source/executable hashes identify the tested implementation.

Date: 2026-09-21. The same GPT-OSS-20B MXFP4 file and the same frozen Kaggle AG News 400 articles were evaluated before and after completing the assistant `final` header. No model weights, instructions, labels, class order, scoring formula, or abstention thresholds were changed.

## Change

`auto` now selects `gpt-oss-final` only for GGUF architecture `gpt-oss`. The embedded template is rendered, then a recognized open assistant header is completed to:

```text
<|start|>assistant<|channel|>final<|message|>
```

Candidate tokenization is checked at this new boundary. Decisions use unmasked vocabulary logits, with minimum top probability 0.8 and minimum candidate mass 0.05. No reasoning tokens are generated. Explicit `--prompt-profile model` preserves raw-template rendering for diagnostics. Incompatible architectures and unexpected/nonempty answer boundaries are rejected.

## Kaggle comparison

| Mode | Correct | Wrong | Abstained | Correct / all | Accepted accuracy | Coverage | Raw top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before: open header | 0 | 0 | 400 | 0.00% | n/a | 0.00% | 24.25% |
| After: final prefill | 134 | 41 | 225 | 33.50% | 76.57% | 43.75% | 55.25% |

Accepted accuracy excludes abstentions. Correct/all includes every sampled article. The balanced one-class baseline is 25%. These are application results on historical public articles; pretraining overlap is unknown. Direct-final prefill does not establish the model's accuracy when allowed to generate reasoning.

| Mode | Candidate mass min / median / max | p50 / p95 ms per article |
| --- | --- | ---: |
| Before | 1.58019e-19 / 9.82205e-19 / 3.09785e-18 | 1511.01 / 1640.13 |
| After | 0.996241 / 0.999682 / 0.999981 | 1420.81 / 1631.40 |

Final-prefill correct/all 95% Wilson interval: 29.05%–38.26%. Timing includes the first article but excludes loading; runs were not isolated hardware measurements.

## Validation

- 17 ordinary release tests passed, including three new regression tests for profile routing, Harmony completion/data isolation, and invalid answer boundaries. Existing tests remain enabled.
- Release all-target Clippy with `-D warnings` and formatting passed. Upstream C++ unused-function warnings remain.
- Real GPT-OSS CUDA native suite passed: A–B–A request isolation, 26 distinct candidate tokens, batch 32 vs 512 probability tolerance unchanged at 0.02, incompatible profile rejection, and oversized input rejection. This replaces the previous CUDA chunking failure for the final-prefill profile only.
- Warehouse CLI: chilled / true / high, all correct and accepted. Candidate mass 0.999926–0.999992. See [portable example output](examples/warehouse.gpt-oss.cuda.output.json).
- Gemma 4 regression: four existing articles retained their answers, profile, and prompt version. Other model routing is also covered by unit tests.
- Both 400-article runs were independently recounted against separate gold labels: no missing outcomes, inference errors, or truncation.
- CPU behavior for the new GPT-OSS final-prefill profile was not tested. No production deployment or GitHub push was performed.

## Reproduce and provenance

The original baseline remains in [Kaggle results](KAGGLE_BENCHMARK_RESULTS.md). The data selection and scoring definitions are in [Kaggle protocol](KAGGLE_BENCHMARK.md). Baseline and updated source/executable hashes are recorded separately; use the recorded baseline code for the original auto-profile behavior.

- GGUF SHA256: `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`.
- Frozen request SHA256: `1489716ed040e95087a6e973819ac1f346f39651db50cd2908da2b723da50a54`.
- llama.cpp: `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`.
- Runtime: CUDA on RTX 3080 10 GiB; context 2048, batch 256, four threads. CUDA offload does not prove exclusive residency in physical VRAM.
- New detailed predictions, metrics, hashes and runtime metadata: `results/kaggle-ag-news-gpt-oss-final/`. Native/CLI logs: `results/gpt-oss-final/`. Weights and detailed local reports remain ignored.

```sh
export LLAMA_CPP_DIR=/path/to/llama.cpp
export LLAMA_LIB_DIR="$LLAMA_CPP_DIR/build-cuda/bin"
cargo build --release --locked --features llama --bin l2s1 --example evaluate_jsonl
# Copy the existing frozen requests.jsonl and selection.json into a NEW output folder.
python3 scripts/kaggle_ag_news.py run --folder results/NEW_FOLDER --model gpt-oss-20b
```
