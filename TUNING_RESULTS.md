# Gemma 4 compute tuning and calibration — 2026-09-22

This is local RTX 3080 (10 GiB) evidence using Gemma 4 E2B Instruct Q8_0. No remote node was used. Legacy prompts and the original 0.8 probability / 0.05 candidate-mass policy are unchanged. Compute settings remain opt-in.

## Decision

For throughput experiments, parallel width 4 / batch 1024 / microbatch 1024 / FlashAttention on is the latency-throughput compromise; width 16 / batch 2048 maximizes measured throughput. Both change decisions and fail the existing serial-equivalence gate. Keep fresh / batch256 / FlashAttention off as the default. Accuracy differences of one or two articles on this reused 400-item sample do not establish a quality improvement.

## Changes

- Added explicit token microbatch and FlashAttention controls to the CLI, JSONL evaluator, and `ComputeOptions` library API. Existing loaders retain caller-supplied resource settings, microbatch equal to batch, and FlashAttention off; CLI defaults remain batch 256.
- Added batch preparation/native/scoring wall-time counters. Native time includes GPU work, synchronization, and logits copies; these counters do not isolate PCIe transfer cost.
- Removed the full-vocabulary temporary f64 allocation from scoring, preserving f64 arithmetic and summation order. Reference-normalizer tests require exact results.
- Added reproducible compute-matrix and task-specific temperature-calibration scripts. Calibration is an offline experiment; it is not silently enabled in the general decision API.

## Protocol

The original frozen 400-article Kaggle AG News set is now a development benchmark because it has been reused during tuning. Its request SHA-256 remains `1489716ed040e95087a6e973819ac1f346f39651db50cd2908da2b723da50a54`. Screening used 64 articles: eight original groups with the largest prior probability drift plus eight control groups. Screening accuracy is not an unbiased quality estimate. The initial FlashAttention-off screening ran while the separate runtime was compiling; FlashAttention screening ran after that build. Screening times were used only to shortlist configurations. Final runs execute sequentially after compilation, in forward, reverse, and mixed configuration order, three repeats each. This is a local workstation measurement, not an isolated-server latency SLA.

The original llama.cpp runtime was built with `GGML_CUDA_FA=OFF`. A separate build from the same source commit (`3d82ef62d47fd74e18f36c5eccbdcf965b617b17`) enables CUDA FlashAttention, f16 KV kernels, and SM86. The upstream checkout and original runtime were not modified. Runtime and binary hashes are stored in each run summary. One untimed warmup batch precedes every measured run. Model loading, file output, queue accumulation, and network latency are excluded from inference timing. Full batch completion time is latency; batch time divided by article count is amortized cost.

## Final 400-article runs

| Setting | Runs | Median total (s) | Articles/s | Batch p50 (ms) | Correct / wrong / abstain | Raw top-1 | Max probability delta vs original fresh | Changed selections |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| original-fresh | 3 | 16.970 | 23.57 | 40.43 | 305 / 83 / 12 | 77.50% | 0.000000 | 0 |
| optimized-fresh | 3 | 17.142 | 23.33 | 41.27 | 305 / 83 / 12 | 77.50% | 0.000000 | 0 |
| parallel4-b256-off | 3 | 16.204 | 24.69 | 161.09 | 304 / 81 / 15 | 77.50% | 0.513354 | 9 |
| parallel4-b1024-off | 3 | 13.925 | 28.72 | 137.76 | 306 / 80 / 14 | 77.50% | 0.409259 | 16 |
| fresh-b256-fa | 3 | 16.347 | 24.47 | 38.94 | 307 / 81 / 12 | 77.75% | 0.680735 | 12 |
| parallel4-b1024-fa | 3 | 11.838 | 33.79 | 117.30 | 305 / 79 / 16 | 77.25% | 0.377614 | 10 |
| parallel16-b2048-fa | 3 | 11.116 | 35.98 | 441.52 | 306 / 82 / 12 | 77.75% | 0.368241 | 8 |

Per-setting total-time ranges (seconds): original-fresh: 16.508–17.443; optimized-fresh: 16.581–19.497; parallel4-b256-off: 15.878–18.310; parallel4-b1024-off: 13.418–14.349; fresh-b256-fa: 16.237–16.365; parallel4-b1024-fa: 11.309–11.848; parallel16-b2048-fa: 11.033–11.160.

All final runs retain 400 distinct IDs with no errors, missing records, or truncation. The equivalence gate remains no changed top-1/accepted selections and probability/candidate-mass deltas <= 0.02; it is not relaxed for faster configurations.

### Profile breakdown

| Setting | Prepare (ms/article) | Native (ms/article) | Score (ms/article) |
|---|---:|---:|---:|
| original-fresh | 2.417 | 37.238 | 1.856 |
| optimized-fresh | 2.467 | 37.826 | 1.626 |
| parallel4-b256-off | 2.152 | 36.362 | 1.670 |
| parallel4-b1024-off | 1.961 | 30.895 | 1.655 |
| fresh-b256-fa | 2.398 | 36.009 | 1.619 |
| parallel4-b1024-fa | 2.020 | 25.592 | 1.654 |
| parallel16-b2048-fa | 1.934 | 23.882 | 1.648 |

The allocation change lowered median score time from 1.856 to 1.626 ms/article and avoids a 2 MiB temporary f64 buffer for this 262144-token vocabulary. End-to-end fresh time was 17.142 vs 16.970 seconds, so no end-to-end speedup is established for that change alone. The main measured gains come from token batching and attention kernels.

All 8400 final outcomes were independently recounted. Within each compute configuration, all three repeats had exactly identical option probabilities, candidate masses, and selected answers. Across configurations, substantial differences remain.

## Numerical investigation

Forcing `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` changes cuBLAS computation where that path is used; it does not turn quantized weights or every CUDA kernel into FP32. Across the full 400 articles, its serial-vs-parallel comparison still changed 8 accepted selections and 4 top-1 choices, with maximum probability delta 0.449860. It is not a stability fix.

The four-article group containing the largest previous drift was also run on CPU. CPU fresh-vs-parallel changed 2 accepted selections and 1 top-1 choice, with maximum probability delta 0.563185. CPU quantized inference is a diagnostic reference, not ground truth. Execution-shape sensitivity therefore occurs outside the CUDA parallel path too; these observations do not establish a single underlying kernel defect.

- `GGML_CUDA_DISABLE_GRAPHS=1`: serial-vs-parallel on 64 diagnostic articles changed 5 selections and 3 top-1 choices; maximum probability delta 0.513354.
- `GGML_CUDA_DISABLE_FUSION=1`: serial-vs-parallel on 64 diagnostic articles changed 5 selections and 2 top-1 choices; maximum probability delta 0.496118.

## Temperature calibration

Using the pinned Kaggle archive, sampled 400 fit and 400 validation articles from `train.csv`, balanced by class and seeded 20260922. Removed duplicates against all of `test.csv` and between both new splits. Labels remain separate from inference inputs. Fit only one scalar temperature by minimizing fit-set NLL over T in [0.05, 100]; validation labels did not select the temperature or acceptance thresholds. The model was not trained. Public-data pretraining contamination cannot be excluded.

Fitted **T = 4.781453**, for the original fresh / batch256 / FlashAttention-off configuration only. The transform is `softmax(candidate_logits / T)`. Full-vocabulary candidate mass remains the original raw-model mass; temperature scaling does not calibrate that quantity. Refit and validate after model, prompt, label, or compute changes.

| Held-out validation metric (400 articles) | Before | After |
|---|---:|---:|
| Raw top-1 accuracy | 80.25% | 80.25% |
| Mean top confidence | 98.30% | 79.82% |
| ECE (10 equal-width bins) | 18.05% | 6.36% |
| NLL | 1.8014 | 0.6250 |
| Brier (sum over four classes) | 0.3743 | 0.3029 |
| Coverage at fixed 0.8 threshold | 95.75% | 60.25% |
| Accuracy among accepted answers | 82.51% | 91.70% |
| Correct accepted / all articles | 79.00% | 55.25% |

Accepted answers change from 316/383 correct to 221/241 correct. Better reliability comes with more abstentions; raw classification accuracy is unchanged. This is not evidence that overall accuracy rose to 91.70%.

On the original 400-article development set, the independently fitted temperature reduced ECE from 20.57% to 2.81%. Accepted accuracy rose from 78.61% to 91.23%, while coverage fell from 97% to 57% (208 correct accepted / 400 total). Raw top-1 stayed 77.50%. These numbers were not used to fit the temperature.

## Reproduction and artifacts

Raw logs, JSONL outcomes, matrix files, binaries, runtime build, source snapshots and summaries are under `results/tuning-20260922/` (ignored by Git). Keep them for exact reproduction; no model weights or Kaggle text are committed.

```sh
export LLAMA_CPP_DIR=/home/lutica/personal/Openweight-Test/llama.cpp
export LLAMA_LIB_DIR="$LLAMA_CPP_DIR/build-cuda/bin"
cargo build --release --locked --features llama --example evaluate_jsonl

# Example opt-in compute settings (require the separate FA-capable runtime):
LD_LIBRARY_PATH=results/tuning-20260922/build-fa/bin \
  target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
  --input results/kaggle-ag-news/requests.jsonl --output /tmp/tuned-new.jsonl \
  --batch 1024 --ubatch 1024 --flash-attention on \
  --execution-mode parallel --parallel-width 4 --request-batch-size 4 --warmup

# The preparation command refuses to overwrite existing selections:
python3 scripts/calibrate_ag_news.py prepare \
  --archive results/kaggle-ag-news/dataset.zip \
  --template results/kaggle-ag-news/requests.jsonl --output /tmp/ag-calibration-new
```

Use `scripts/tune_compute.py --help` to run saved matrix JSON files against an explicit runtime directory. Evaluate the prepared fit and validation JSONL separately with default compute options, then use `scripts/calibrate_ag_news.py fit --selection ... --fit-file ... --validation-file ... --output ...`. This emits a scoped calibration artifact and before/after metrics; it does not change application defaults.

## Calibration for the faster configuration

The width4 / batch1024 / FlashAttention-on configuration was chosen from timing results, then independently fitted on the same 400 fit IDs. It was evaluated on the same disjoint 400 validation IDs, not a new untouched validation set. No validation labels or acceptance thresholds entered fitting. This produces a separate artifact, `results/tuning-20260922/calibration/temperature-fast.json`, rather than reusing the serial temperature.

Fitted T = 4.701892. Validation ECE fell from 18.22% to 6.21%, NLL from 1.7825 to 0.6233, and mean top confidence from 98.22% to 80.10%. Raw top-1 stayed 80.00%. At the unchanged 0.8 acceptance threshold, accepted accuracy rose from 81.96% (318/388) to 91.74% (222/242), while coverage fell from 97.00% to 60.50%. Correct accepted / all fell from 79.50% to 55.50%.

## Validation

- Default-feature Rust suite: 13 passed. Llama-feature Rust suite: 21 passed; model-dependent tests are opt-in.
- New real-Gemma CUDA regression checks passed with batch1024/microbatch256/FA-off and batch1024/microbatch1024/FA-on. They cover width changes, width1 equivalence, metadata, oversized-input failure, and state recovery.
- Python benchmark/calibration suite: 10 passed. Clippy with `-D warnings`, formatting, CLI build, and `git diff --check` passed.
- Final run audit: 8400 complete outcomes, independently counted, with exact repeated scores per configuration. No serial-equivalence failure was hidden or converted into a pass.

## Building the separate RTX 3080 runtime

```sh
cmake -S "$LLAMA_CPP_DIR" -B results/tuning-20260922/build-fa \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_CUDA_FA=ON \
  -DGGML_CUDA_FA_QUANTS=f16-f16 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF
cmake --build results/tuning-20260922/build-fa --target llama -j 4
```
