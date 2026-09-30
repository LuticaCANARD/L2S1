# Training PR review: shared-evidence preservation

The custom Jev preparation path discarded `shared` before both training token export and evaluation. The exporter change in PR #91 alone therefore did not preserve an application's inputs. Preparation now preserves shared evidence in every request and flattened training question. Training also checks native prompt layout against the seal, rejects invalid context/empty token sequences, and reports named Score levels by their values.

The before arm reproduces the original #91 head (`bca9a3b796a412fd2bb0afe95f924ede9fc303c6`) exactly at the parsed-request boundary; see `before-reproduction.json`. Both arms use the same evaluator, model weights, prompt settings and labels. Only shared evidence differs. No adapter was trained or selected in this study.

## Frozen evaluation

Synthetic directory lookup, with one Choice, Noul and named-level Score question per case. Preparation-only train: 16 cases; development: 16 cases / 48 decisions; separate test: 64 cases / 192 decisions. IDs/states are disjoint. All fixture hashes, settings and model choices were fixed before the first run (`protocol.json`). Development results did not change the implementation or configuration. These cases measure the specific missing-evidence regression, not general reasoning, public JevBench, unseen task families, or LoRA quality.

RTX 3080 10 GiB, i9-9900K, WSL2, driver 596.21, pinned llama.cpp `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. Native in-process CUDA; fresh execution, state-first/typed, context 4096, batch/ubatch 256, FA off, four threads, one excluded warmup and one measured pass. Policy remains top probability >=0.8 and candidate mass >=0.05. Model, evaluator, source and request hashes are in `summary.json`. The 12B model is QAT Q4_0; E2B is Q8_0.

| Model / test arm | Raw correct / all | Accepted correct / accepted | Coverage | Correct accepted / all | Wrong accepted | Abstained |
| --- | --- | --- | --- | --- | --- | --- |
| E2B / before | 73/192 (38.02%) | 64/179 (35.75%) | 93.23% | 33.33% | 115 | 13 |
| E2B / after | 177/192 (92.19%) | 177/189 (93.65%) | 98.44% | 92.19% | 12 | 3 |
| 12B / before | 72/192 (37.50%) | 72/192 (37.50%) | 100% | 37.50% | 120 | 0 |
| 12B / after | 192/192 (100%) | 192/192 (100%) | 100% | 100% | 0 | 0 |

Raw accuracy change: E2B +54.17 percentage points (paired case-bootstrap 95% interval +44.79 to +63.02); 12B +62.50 points (+56.25 to +68.75). Bootstrap resamples 64 cases, keeping their three decisions together, 10,000 draws with seed 20260930091. It describes sampling uncertainty within this synthetic fixture, not generalization beyond it. Development raw counts: E2B 16→44/48, 12B 11→48/48.

## Latency and memory observations

Latency is per complete request containing three decisions. One pass is insufficient to claim a stable speed difference. Restoring the evidence adds input tokens; it is an accuracy fix, with higher observed inference time.

| Model / test arm | p50 / p95 ms | Input tokens, all 192 decisions | Process RSS/HWM max GiB | Whole-board GPU max MiB |
| --- | --- | --- | --- | --- |
| E2B / before | 170.10 / 189.45 | 55,327 | 4.95 | 3,812 |
| E2B / after | 188.62 / 207.52 | 75,439 | 4.96 | 3,815 |
| 12B / before | 418.59 / 427.65 | 56,095 | 6.81 | 9,449 |
| 12B / after | 529.84 / 537.94 | 76,207 | 6.83 | 9,446 |

RSS/HWM and GPU samples include loading; GPU readings include the desktop. They are separate measures, not allocator-exact peaks. Reused prefix tokens are zero in every fresh run.

## Validation and reproduction

- Python package contracts: 22 tests, 20 passed; the native-export and MLX-runtime tests are optional. Native export was separately enabled and passed all six layout/detail combinations on both E2B and 12B. MLX training itself was not rerun on this Linux/CUDA host.
- Native model tests: two passed on each model. They compare vocab-only and full-load token IDs, shared evidence, every layout/detail/rotation, and >2048-token inputs with a 4096 context; default context rejects oversized input.
- Jev contracts: 14 passed; accuracy contracts: 12 passed; evaluator contracts: four passed against the built executable.
- Named Score reporting completed on all 64 test cases / 192 decisions for the 12B after arm.
- Fixtures regenerate byte-for-byte with `python3 benchmarks/training-pr-review-20260930/prepare.py --output /tmp/new-fixture-directory`.
- `python3 benchmarks/training-pr-review-20260930/verify.py` checks input/record hashes and recomputes eight complete runs from `records.jsonl` (320 case predictions / 960 decision predictions). Repeated before/after/model arms do not multiply the 192-decision test denominator.

To rerun with matching local binaries, libraries and models:

```sh
python3 benchmarks/training-pr-review-20260930/run.py \
  --evaluator "$EVALUATOR" --models "$MODELS" \
  --llama-source "$LLAMA_SOURCE" --output /tmp/new-evidence-run
```

Set the matching shared-library search path for the evaluator. Full local predictions, stderr and the measured binary are retained under `results/training-pr-review-20260930/`; compact checked-in records support the reported metrics without model files.
