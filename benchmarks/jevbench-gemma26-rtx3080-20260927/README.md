# Gemma 4 26B-A4B — RTX 3080, 2026-09-27

This extends the [five-checkpoint run](../jevbench-rtx3080-20260927/README.md) with `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf`. Its SHA-256 is `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f`, identical to the separately recorded September 23 RTX 3060 checkpoint. The [pinned public download](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/blob/c099eb48e663fd284577b04978a94ffccb261841/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf) is 16,947,541,728 bytes.

Fresh accuracy uses **all 231 public JevBench tasks**. Paired prefix diagnostics use **135 tasks belonging to the 21 repeating schemas**, selected using only exact serialized decision definitions, never gold labels. The 96 singleton schemas are omitted only from paired diagnostics. Their whole-pass timings are not comparable with the earlier five-model, 231-item diagnostic passes or with request p50.

Settings: RTX 3080 10 GiB / WSL2, four CPU threads, context 8192, 22 CPU expert layers, automatic GPU layer placement, FlashAttention off, full evidence, legacy prompts, default 0.8 top-probability / 0.05 candidate-mass policy. CUDA and CPU expert placement apply identically to fresh and reuse. Accuracy uses batch/ubatch 256. Paired diagnostics use batches 64 and 256, one full untimed warmup per path and one measured full pass. Preparation cache is off. Timings exclude model load and warmup.

The frozen fresh evaluator and scorer are identical to the five-model rerun. The schema harness adds `--cpu-moe-layers` / `--gpu-layers` and phase messages; native inference sources are unchanged. Source and binary identities are recorded in `summary.json`. The original measured checkout is dirty v0.1.1; the PR uses a newer v0.1.2 base. This is not a final-PR-binary timing claim, an M5/MLX rerun, the complete 534-item JevBench suite or an official score.

- `summary.json`: independently recounted quality, policy, calibration, tier and latency metrics, two paired diagnostics, runtime identities and provenance.
- `predictions.jsonl`: all 231 compact per-item fresh predictions, expected labels, selections, probabilities, mass, tokens and latency. No state text or rationales.
- `audit.py`: independently checks the native reports and rebuilds this export; requires ignored complete run artifacts. It verifies that the diagnostic subset is exactly the repeated schemas.
- `JEVBENCH-LICENSE`: upstream MIT notice for derived task IDs and labels at commit `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`.

```sh
target/release/l2s1-tools jevbench-public run \
  --jevbench /path/to/pinned/jevbench \
  --evaluator target/release/examples/evaluate_jsonl \
  --model models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --device cuda --expected-gpu 'RTX 3080' --context 8192 \
  --threads 4 --cpu-moe-layers 22 --timeout 7200 \
  --output results/jevbench-gemma26-prefix-20260927/gemma26

target/release/examples/benchmark_schema_reuse \
  --model models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf \
  --device cuda --context 8192 --threads 4 --cpu-moe-layers 22 \
  --input results/jevbench-gemma26-prefix-20260927/repeated-schema-requests.jsonl \
  --batch 64 --rounds 1 \
  --output results/jevbench-gemma26-prefix-20260927/gemma26-schema-b64.json
```

Repeat the schema command at batch 256 with a different output path. The repeated-schema input is derived from the generated requests:

```python
import json
from collections import Counter
from pathlib import Path
root = Path('results/jevbench-gemma26-prefix-20260927')
rows = [json.loads(line) for line in (root/'gemma26/requests.jsonl').read_text().splitlines()]
keys = [json.dumps(row['request']['decisions'][0], sort_keys=True) for row in rows]
counts = Counter(keys)
selected = [row for row, key in zip(rows, keys) if counts[key] > 1]
assert len(selected) == 135
(root/'repeated-schema-requests.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
```

The recorded full directory also contains shared `prepared/` data, provenance, smoke outputs and logs. Run the independent audit on those artifacts:

```sh
python3 benchmarks/jevbench-gemma26-rtx3080-20260927/audit.py \
  --runs results/jevbench-gemma26-prefix-20260927 --output /tmp/jevbench-gemma26-audit
```

Raw-confidence risk/coverage thresholds use test labels and ignore native mass/tie gates; they are descriptive, not deployment policies or error guarantees. Repeated diagnostics do not add independent accuracy samples. Sampled GPU totals include other desktop usage and are not model-exclusive allocations.

## Outcome

Raw correct: 193/231 (83.55%). Accepted: 223, including 192 correct and 31 wrong (86.10% accepted accuracy at 96.54% coverage). Fresh p50/p95: 1,213.764/12,498.787 ms. No inference errors or truncated inputs.

Both matched fresh/session pairs have exactly zero probability and candidate-mass deltas, with unchanged top-1, accepted selections and abstentions. Batch-64 reuse is 1.58x faster than matched fresh, but **15.2% slower than batch-256 fresh** (196.769 versus 170.766 s). Batch 256 reuses zero tokens, so its 2% timing difference is not evidence of KV acceleration. Changing 256→64 itself changes two raw answers and three accepted selections; maximum candidate-probability drift is 0.993726 even though both subsets score 129/135. Thus equal aggregate accuracy does not establish decision equivalence across batches. The grouped batch-256 fresh pass versus the original fresh run has no answer/selection changes but a maximum probability drift of 0.000156. These observations are specific to this CPU/GPU mixed configuration. CPU expert offload is an explicit test setting; the runtime and harness defaults remain `cpu_moe_layers=0`. The raw-confidence frontier at empirical risk ≤5% retains 175/231 items with eight errors, selected after seeing test labels; it is not a deployment guarantee.
