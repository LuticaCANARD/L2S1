# JevBench public evaluation

The Rust `l2s1-tools jevbench-public` command connects the existing `evaluate_jsonl` example to the public [JevBench](https://github.com/fstandhartinger/jevbench) tasks. Its public scorer follows the reviewed upstream revision. Earlier recorded Gemma 4 E2B Q8_0 and multi-model matrix results came from the historical Python adapter; new runs use Rust.

The reviewed upstream revision is `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`. Preparation rejects another revision or a dirty upstream checkout. The three published JSONL files contain 231 decisions: easy 48, original 72, hard 111. This is a public-subset evaluation, not the complete 534-item evaluation or an official leaderboard submission.

## Reproduce

Requirements: a clean upstream checkout at the pinned revision, a compatible GGUF, and a CUDA-enabled project build. The evaluator verifies CUDA offload; use `--expected-gpu 'RTX 3060'` (or the intended device name) to additionally require a specific GPU. The harness and public task files retain their upstream MIT notices; model terms remain separate.

```bash
# On a CUDA host with the Rust and native toolchains configured:
cd /path/to/L2S1
cargo build --release --locked --features llama-cuda --bin l2s1 --example evaluate_jsonl
cargo build --release --locked -p l2s1-tools

git clone https://github.com/fstandhartinger/jevbench.git ../upstream-new
git -C ../upstream-new checkout --detach f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84

target/release/l2s1-tools jevbench-public run \
  --jevbench ../upstream-new \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --context 8192 --expected-gpu 'RTX 3060' --output ../run-gemma4-new
```

The output directory must not already exist. No external inference API, API key, or billable service is used.

For CPU/GPU weight placement, the harness also accepts `--gpu-layers N`,
`--cpu-moe-layers N`, and `--threads N`. For the Gemma 4 26B A4B Q4 checkpoint
on the RTX 3060, use `--cpu-moe-layers 18 --threads 8 --context 8192`.
The manifest records these settings and the harness checks the reported placement.
Omitting these options preserves the baseline placement and four-thread setup.
The optional `--model-load-mode read` changes model loading only; the default
`auto` preserves automatic loading. Its requested mode is recorded and verified.

## Mapping and measurement contract

- Pass only `state`, instructions, and criteria to inference. Keep expected answers, rationales, gold distributions, and provenance in a separate scoring file.
- Preserve the canonical `labels` order. Map `choice` to ordered options, `noul` to false/true with probabilities mapped back to no/yes, and `score` to numeric ordinal levels in ascending order.
- Use the model's candidate-relative probabilities directly. The official argmax score is computed even when the project's default decision policy abstains. Report that policy's accepted accuracy, wrong accepted answers, and coverage separately.
- The Rust scorer follows the pinned upstream `score_task` and `summarize` definitions. Brier uses the full multiclass sum, including both binary labels. ECE uses ten equal-width confidence bins. Saved public predictions and aggregate results were compared with the upstream Python scorer.
- Baseline configuration: legacy prompt, fresh execution, context 8,192, batch/microbatch 256, four threads, one request at a time, FlashAttention off. No LoRA, output head, calibration artifact, or training on these items.
- Input overflow remains an error, never truncation. Retain inference failures; reject missing, duplicated, or unknown result IDs and candidate mapping mismatches.
- Report local Rust inference-call latency with loading and one warmup excluded. It includes prompt preparation and inference, but excludes an HTTP/network path. It is not the official board's remote-endpoint latency.
- Local cost is unknown (`null`), not zero. No overall JevBench Score or rank is assigned because the full dataset, official deployment latency, and a supported price basis are unavailable.

Outputs include the exact inference requests, original tasks with gold, raw predictions, normalized JevBench records, official metric summaries by public tier/family, selective-policy results, logs, commands, and data/model/evaluator hashes. `native_candidate_softmax` identifies the probability source; these values are not asserted to be calibrated correctness probabilities.

Mapping tests:

```bash
python3 -m unittest discover -s scripts -p 'test_jevbench_public.py'
```

## Local RTX 3080 rerun (2026-09-23)

A fresh five-model run used context 16,384 and `--expected-gpu 'RTX 3080'`.
All 1,155 predictions completed without inference errors or truncation and passed
an independent saved-prediction recount of accuracy, Brier, ECE and abstention
totals. Gemma 4 E2B Q8_0 scored 159/231 (68.83%); Gemma 3 1B Q8_0 93/231,
TinyLlama 1.1B Q4_K_M 77/231, Qwen3 0.6B Q8_0 73/231, and SmolLM2 135M Q8_0
71/231. These scores use argmax before the default abstention policy.

The public datasets and scoring modules are byte-identical to the revision
referenced by the [Open-Jev report](https://zefan-cai.github.io/open-jev/benchmarks/).
This evaluates L2S1 with local GGUFs; Open-Jev's trained checkpoints were not run.
The local full report (local artifact: `results/jevbench-local-20260923T083530Z/REPORT.md`, not committed) records
tier scores, selective accuracy, timing limits and reproduction commands. Raw
predictions, source snapshots and replay evidence remain beside that report in
the ignored results directory. This is separate from the RTX 3060 matrix below.

### Prompt-order diagnostics on that RTX 3080 run

Four post-hoc diagnostic streams kept each model, public task set and inference
settings fixed while changing either the JSON field order (`state-first`) or
reversing the 139 Choice option lists with their IDs and criteria paired. The
original model scores are the baseline from the five-model run above. Correct
answers below are semantic IDs, so a displayed option changing position does
not itself count as a changed answer.

| Model | Change | Baseline / 231 | New / 231 | Semantic answers changed | Wrong to correct / correct to wrong |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q8_0 | State first | 73 | 78 | 73/231 | 29 / 24 |
| Qwen3 0.6B Q8_0 | Choice order reversed | 73 | 76 | 131/139 | 29 / 26 |
| Gemma 4 E2B Q8_0 | State first | 159 | 151 | 50/231 | 15 / 23 |
| Gemma 4 E2B Q8_0 | Choice order reversed | 159 | 157 | 33/139 | 9 / 11 |

All four streams completed without inference errors or truncation. This is
evidence that prompt and option presentation can change these checkpoints'
outputs; it is not an independent held-out improvement test or a claim that
state-first is better for other models. The local, gitignored
`results/jevbench-diagnosis-20260923/REPORT.md` retains the full diagnosis,
commands and saved predictions. Legacy ordering remains the default.

## Multi-model comparison

The [recorded model comparison](MODEL_RESULTS.md#recorded-model-comparison) summarizes the completed September 23 RTX 3060 matrix: 22 GGUF checkpoints each scored all 231 items, with no errors in those completed runs. There were 23 runtime configurations: GPT-OSS failed with CUDA Graphs enabled, then completed with `GGML_CUDA_DISABLE_GRAPHS=1`. The original failed attempt remains in the evidence. Latency values belong to their respective settings and should not be mixed with separate runs.

The complete report, CSV, raw predictions, model plan, runtime hashes and measured source snapshot are in the local, gitignored `results/jevbench-matrix-20260923/` directory; they are not included in this repository. Qwen3.5-4B Q8_0 led this selected matrix at 184/231 (79.65%). Full reruns after model downloads finished reproduced all probabilities for that model and Gemma4 E2B exactly. The snapshot records the frozen build used by this comparison, independently of subsequent working-tree changes.

`l2s1-tools jevbench-matrix download/run` downloads a pinned plan or runs available checkpoints serially through the public evaluator. `l2s1-tools report-jevbench-matrix` recounts saved predictions against gold labels and checks shared request/evaluator hashes before producing the report. Regenerating it requires a local model plan, the full per-model run directories and `matrix-status.json`. Generated reports, run artifacts and host-specific plans remain local rather than being versioned with the source.

## Gemma 4 26B A4B separate run (2026-09-23)

The UD-Q4_K_M checkpoint (SHA-256 `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`) ran on an RTX 3060 with 18 CPU expert layers, eight threads, context 8192 and batch/ubatch 256. The pinned public scorer counted 196/231 correct (84.85%): Easy 48/48, Original 70/72, Hard 78/111. The default policy accepted 220 decisions, with 191 correct and 29 wrong; 11 abstained. Brier was 0.278098 and ECE 0.132128. There were no inference errors or truncated inputs.

The measured fresh/legacy inference p50/p95 was 787.01/8814.01 ms, excluding model load and one warmup. This run used the same public task IDs as the matrix but a different evaluator build and request serialization; its latency is not a controlled comparison with the original matrix or the later 31B run. Saved predictions were independently recounted. The local, gitignored `results/jevbench-gemma26-20260923T132214Z/` folder retains the source, model, request, evaluator and prediction hashes plus raw evidence.

## Gemma 4 rebuild confirmation (2026-09-23)

The current Rust source and native C++ bridge were freshly compiled in a separate directory on an RTX 3060 host, reusing the pinned llama.cpp CUDA libraries. Gemma 4 E2B Q8_0, E4B Q8_0 and E4B Q4_K_M each completed the same public 231 items with the baseline settings above. They scored 157/231 (67.97%), 177/231 (76.62%) and 179/231 (77.49%), respectively. All 693 candidate probability vectors and decision values exactly matched the previous RTX 3060 matrix; no inference errors or truncation occurred.

The rebuild report (local artifact: `results/gemma4-rebuild-20260923T103912Z/REPORT.md`, not committed) records the before/after comparison, latency, source and binary hashes, and build logs. Default tests passed 49/49; the llama feature suite passed 59 tests with 18 ignored, followed by a separately executed Gemma 4 CUDA integration test and six mapping tests. These results cover the frozen source and baseline path; optional optimization modes were not enabled for this rerun.
