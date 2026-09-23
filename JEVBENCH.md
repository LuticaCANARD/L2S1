# JevBench public evaluation

`scripts/jevbench_public.py` connects the existing `evaluate_jsonl` example to the public [JevBench](https://github.com/fstandhartinger/jevbench) tasks and the benchmark author's scoring functions. It records the Gemma 4 E2B Q8_0 evaluation and a multi-model matrix on the remote RTX 3060.

The reviewed upstream revision is `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`. The script rejects another revision or a dirty upstream checkout. The three published JSONL files contain 231 decisions: easy 48, original 72, hard 111. This is a public-subset evaluation, not the complete 534-item evaluation or an official leaderboard submission.

## Reproduce

Requirements: Python 3.11+, a clean upstream checkout at the pinned revision, the Gemma 4 E2B IT Q8_0 GGUF, and a CUDA-enabled project build. The current evaluator script checks for RTX 3060 offload to prevent this recorded run from silently falling back to another device. The harness and public task files retain their upstream MIT notices; model terms remain separate.

```bash
# On the prepared server:
source ~/.local/opt/skid-desion/skid-test-env.sh
cd ~/personal/skid/jevbench-20260923/source
cargo build --release --locked --features llama --bin l2s1 --example evaluate_jsonl

git clone https://github.com/fstandhartinger/jevbench.git ../upstream-new
git -C ../upstream-new checkout --detach f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84

python3 scripts/jevbench_public.py \
  --jevbench ../upstream-new \
  --evaluator target/release/examples/evaluate_jsonl \
  --model ~/personal/skid/skid-desion/models/gemma-4-E2B-it-Q8_0.gguf \
  --context 8192 --output ../run-gemma4-new
```

The output directory must not already exist. No external inference API, API key, or billable service is used.

## Mapping and measurement contract

- Pass only `state`, instructions, and criteria to inference. Keep expected answers, rationales, gold distributions, and provenance in a separate scoring file.
- Preserve the canonical `labels` order. Map `choice` to ordered options, `noul` to false/true with probabilities mapped back to no/yes, and `score` to numeric ordinal levels in ascending order.
- Use the model's candidate-relative probabilities directly. The official argmax score is computed even when the project's default decision policy abstains. Report that policy's accepted accuracy, wrong accepted answers, and coverage separately.
- Invoke upstream `score_task` and `summarize`; do not implement an alternative JevBench scoring formula. Brier uses the full multiclass sum, including both binary labels. ECE uses ten equal-width confidence bins.
- Baseline configuration: legacy prompt, fresh execution, context 8,192, batch/microbatch 256, four threads, one request at a time, FlashAttention off. No LoRA, output head, calibration artifact, or training on these items.
- Input overflow remains an error, never truncation. Retain inference failures; reject missing, duplicated, or unknown result IDs and candidate mapping mismatches.
- Report local Rust inference-call latency with loading and one warmup excluded. It includes prompt preparation and inference, but excludes an HTTP/network path. It is not the official board's remote-endpoint latency.
- Local cost is unknown (`null`), not zero. No overall JevBench Score or rank is assigned because the full dataset, official deployment latency, and a supported price basis are unavailable.

Outputs include the exact inference requests, original tasks with gold, raw predictions, normalized JevBench records, official metric summaries by public tier/family, selective-policy results, logs, commands, and data/model/evaluator hashes. `native_candidate_softmax` identifies the probability source; these values are not asserted to be calibrated correctness probabilities.

Mapping tests:

```bash
python3 -m unittest discover -s scripts -p 'test_jevbench_public.py'
```

The September 23 result is in [JEVBENCH_RESULTS.md](JEVBENCH_RESULTS.md).

## Multi-model snapshot

The [September 23 matrix report](docs/benchmarks/jevbench-20260923/REPORT.md), [JSON metrics](docs/benchmarks/jevbench-20260923/REPORT.json) and [model plan](docs/benchmarks/jevbench-20260923/plan.json) preserve the supplied snapshot: nine configurations scored, none failed before complete scoring, and 13 pending out of 22 planned. The model paths in the plan are specific to the benchmark host. This matrix is a separate invocation from the original Gemma4 run above, so its latency measurements differ.

`scripts/jevbench_matrix.py` downloads a pinned plan or runs available checkpoints serially through the public evaluator. `scripts/report_jevbench_matrix.py` recounts saved predictions against gold labels and checks shared request/evaluator hashes before producing the report. Regenerating the report requires the full per-model run directories and `matrix-status.json`; the three committed snapshot files alone contain aggregate evidence, not every prediction.
