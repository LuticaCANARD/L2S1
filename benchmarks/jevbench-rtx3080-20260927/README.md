# JevBench public rerun — RTX 3080, 2026-09-27

Five checkpoints completed all 231 public tasks (1,155 predictions). See the existing [model comparison](../../docs/MODEL_RESULTS.md) and [protocol / schema reuse analysis](../../docs/JEVBENCH.md#rtx3080-rerun-20260927). This is a public-subset local measurement, not the full 534-item suite or an official leaderboard. It does not rerun the supplied M5 Max / jv.py results.

- `summary.json`: independently recounted quality, policy, latency, tier, calibration and source identities, plus four paired schema diagnostics.
- `predictions.jsonl`: compact per-item probabilities, labels, policy selections, token counts and timings for the five fresh runs. No input states or rationales. Expected labels derive from JevBench commit `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`; its MIT notice is retained in `JEVBENCH-LICENSE`.
- `audit.py`: reconstructs this export and checks the Rust reports against complete native responses. Requires the ignored full run directory; the compact predictions do not pretend to bundle those responses, models or logs.

Measured source is the original dirty v0.1.1 checkout at `03d5bc0` with the scoped source and binary hashes in the summary. The PR is based on newer v0.1.2 main, retaining its Metal support; these figures are not a timing claim about a newly rebuilt final PR binary. CPU decision-rules evidence is recorded separately in `../schema-reuse-20260927/`.

## Reproduction

Build `l2s1-tools` and both examples using the recorded llama.cpp pin and CUDA architecture 86. Use the checkpoint hashes from `summary.json` and the pinned upstream JevBench checkout. Commands below illustrate the Qwen3 run; repeat the fresh run for all five recorded checkpoints and the paired diagnostic for Qwen3 and Gemma 4 E2B at batch 64 and 256. Never run models concurrently when comparing latency.

```sh
cargo build --release --locked -p l2s1-tools
L2S1_CUDA_ARCHITECTURES=86 cargo build --release --locked --features llama-cuda \
  --example evaluate_jsonl --example benchmark_schema_reuse

target/release/l2s1-tools jevbench-public run \
  --jevbench /path/to/pinned/jevbench \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/Qwen3-0.6B-Q8_0.gguf --device cuda --expected-gpu 'RTX 3080' \
  --context 16384 --threads 4 --timeout 1800 --output results/jev-rerun/qwen3

target/release/examples/benchmark_schema_reuse \
  --input results/jev-rerun/qwen3/requests.jsonl \
  --model models/Qwen3-0.6B-Q8_0.gguf --device cuda \
  --context 16384 --batch 64 --threads 4 --rounds 1 \
  --output results/jev-rerun/qwen3-schema-b64.json
```

The original full run directory is `results/jevbench-prefix-review-20260927/`, with a shared `prepared/` directory, five model directories and `{qwen3,gemma4}-schema-b{64,256}.json`. Recount the recorded artifacts:

```sh
python3 benchmarks/jevbench-rtx3080-20260927/audit.py \
  --runs results/jevbench-prefix-review-20260927 \
  --output /tmp/jevbench-audit
```

Within each paired batch, probabilities, mass, top-1, accepted selections and abstentions are identical. Batch-64 reuse saves few tokens and loses to batch-256 fresh; reducing batch size also changes some answers. One paired pass per setting is a diagnostic, not a stable speed distribution. Raw-confidence risk/coverage thresholds use test labels and exclude native mass/tie gates; they are not deployment policies or error guarantees.
