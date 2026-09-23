# Gemma decision LoRA pilot — 2026-09-23

The end-to-end training and deployment path works, but this adapter should **not
replace the default model**. The NF4 training runtime improved uncertainty and
slightly improved raw accuracy; those accuracy gains did not transfer to the
existing Q8 llama.cpp deployment. The calibrated base remains a competitive,
faster baseline. No default model or decision policy was changed.

This is a single-domain supervised experiment inspired by the goal of typed,
calibrated decisions. It does not reproduce Jev's RLCD, architecture, or claimed
parallel-question scaling. See [the frozen protocol](DECISION_FINETUNE.md).

## Training and evaluation boundary

- Gemma 4 E2B IT, official revision
  `3e22461f65e89153144f8adb70e3b8c2cc9845a7`.
- Training ran on the authorized remote RTX 3060 12 GB host. The local GGUF
  deployment checks used RTX 3080 10 GB. Cross-device timing is not compared.
- Excluded all 800 previously used airline examples. New disjoint splits contain
  900 training, 400 calibration and 400 test tweets, with balanced classes.
- Three cyclic option orders per training tweet: 2,700 inputs, one epoch,
  225 optimizer updates. Input IDs came directly from the production Rust prompt
  and GGUF tokenizer; maximum length 235 tokens, no truncation.
- NF4 base, bf16 compute, rank-8 Q/V LoRA, 1,339,392 trainable parameters.
  Candidate cross-entropy plus 0.1 times negative log candidate mass.
- Training took 716.66 seconds (11.94 minutes); peak allocated CUDA memory was
  6.43 GiB. This excludes download, model loading and evaluations.
- The final epoch was fixed in advance. No test-based checkpoint selection or
  subsequent retraining occurred. Each runtime/checkpoint received its own
  temperature fitted only on its 400 calibration examples.
- A 60-case test subset, fixed before inference, was evaluated in three cyclic
  option orders. Its 180 calls are not 180 independent test examples.

## Primary deployment result: local Q8 GGUF

Both columns use the same frozen 400 test cases, base GGUF, llama.cpp runtime,
legacy prompt, fresh execution, batch/microbatch 256, four threads and FA off.
The second column adds the converted F32 LoRA at scale 1.

| Metric | Base | Base + LoRA |
|---|---:|---:|
| Raw correct / 400 | 308 | 303 |
| Raw accuracy | 77.00% | 75.75% |
| Fitted temperature | 6.8565 | 2.1522 |
| Raw NLL | 2.0723 | 0.7398 |
| Calibrated NLL | 0.5688 | 0.5731 |
| Calibrated Brier score | 0.3158 | 0.3212 |
| Raw ECE, 10 equal-width bins | 21.61% | 16.50% |
| Calibrated ECE | 8.01% | 5.53% |
| Order-consistent cases / 60 | 51 | 53 |
| Correct order-probe calls / 180 | 143 | 144 |
| Median time for 400 requests, three runs | 15.153 s | 16.594 s |
| Median of per-run request p50 | 37.63 ms | 41.24 ms |

The paired disagreement counts were 17 base-only correct and 12 LoRA-only
correct (two-sided exact McNemar p = 0.4583). This sample does not establish a
general accuracy difference, and it does not demonstrate an improvement.

Repeated timing order alternated between base and LoRA. Every candidate logit
matched its quality-run reference exactly across the three repeats. LoRA added
9.51% to total inference time. Measurements exclude model/context loading,
warmup, network and file output; requests were processed sequentially.

### Calibrated acceptance tradeoff

Candidate mass must still be at least 0.05; ties abstain.

| Threshold | Base correct / wrong / abstain | Base accepted accuracy | LoRA correct / wrong / abstain | LoRA accepted accuracy |
|---|---|---:|---|---:|
| 0.6 | 283 / 50 / 67 | 84.98% | 281 / 56 / 63 | 83.38% |
| 0.7 | 251 / 29 / 120 | 89.64% | 252 / 37 / 111 | 87.20% |
| 0.8 | 199 / 21 / 180 | 90.45% | 194 / 14 / 192 | 93.27% |
| 0.9 | 119 / 1 / 280 | 99.17% | 111 / 1 / 288 | 99.11% |
| 1.0 | 0 / 0 / 400 | undefined | 0 / 0 / 400 | undefined |

At threshold 0.8, the higher accepted accuracy comes with lower coverage:
55.00% to 52.00%. At threshold 0.6, four fewer abstentions come with six more
wrong answers. Neither is an unqualified improvement.

For an equal-count confidence-ranking diagnostic, the most confident 320 of 400
cases (80% coverage) have **85.94% accuracy for both models**. At 60% coverage,
accuracy is 90.00% versus 90.83% (two additional correct answers out of 240).
These are retrospective ranking diagnostics, not deployable fitted thresholds;
they do not apply the mass gate. Temperature scaling does not change the raw
selected class or guarantee calibration on future data.

## Same-runtime training result: Transformers NF4

| Metric, same 400 test cases | Base | LoRA |
|---|---:|---:|
| Raw correct / 400 | 299 | 307 |
| Raw accuracy, ties not counted correct | 74.75% | 76.75% |
| Raw NLL | 2.0806 | 0.5168 |
| Raw ECE | 23.85% | 5.22% |
| Fitted temperature | 5.4113 | 1.1079 |
| Calibrated NLL | 0.6323 | 0.5153 |
| Calibrated ECE | 6.69% | 3.85% |
| Coverage at calibrated 0.8 | 57.50% | 61.25% |
| Accepted accuracy at calibrated 0.8 | 88.70% | 92.24% |
| Order-consistent cases / 60 | 49 | 49 |

There were 18 base-only and 26 LoRA-only correct cases (exact McNemar p = 0.2912).
Six trained NF4 test cases tied for maximum logit and were not counted correct;
the original had no test ties. Order consistency excludes tied answers. The
uncertainty improvement is stronger than the evidence for better class selection.

## Precision and conversion diagnostic

Because the NF4 benefit did not transfer to Q8, an additional **inference-only**
probe evaluated the first 100 calibration inputs in Transformers BF16. It did
not retrain, select a checkpoint or change the final test results.

- All 100 F32 GGUF LoRA matrices equal their PEFT source matrices exactly.
  Adapter alpha/rank are 16/8, with runtime scale 1.
- On these 100 inputs, Q8 and BF16 agree on the top choice in **100/100 cases
  before and 100/100 after applying LoRA**.
- NF4 agrees with BF16 in 87/100 base cases and 90/100 adapted cases.
- After LoRA, mean top probability is 81.14% in NF4, 92.34% in BF16 and 92.54%
  in Q8. Centered candidate-logit MAE against BF16 is 0.9574 for NF4 versus
  0.0812 for Q8.

This supports a precision-dependent adaptation/transfer explanation, rather than
corruption of adapter matrices during conversion. It does not prove equivalence
on all inputs or fully isolate every runtime numerical effect. NF4 calibration
parameters must not be reused for the Q8 deployment.

## Other-task regression

The previous AG News 400-case development set gives raw accuracy **77.50% to
74.00%** (310 to 296 correct) after applying the airline adapter. This is a
regression diagnostic on an already-used benchmark, not a new untouched test.
No AG News data trained this adapter, and airline temperatures were not applied.
The result does not support treating this single-domain adapter as a general
replacement for the base decision model.

## Delivered artifacts and checks

- Final PEFT adapter: `results/finetune-20260923/pilot/adapter/`.
- Final F32 GGUF adapter: `results/finetune-20260923/pilot/adapter.gguf` (~5.4 MB).
- Data manifest and production token hashes: `results/finetune-20260923/data/`.
- Training metadata, dependency lock, exact executed script, all 225 update logs,
  raw scores, conversion audit, paired audits and three timing repeats are under
  `results/finetune-20260923/`.
- Python tests: 14 passed. Rust feature-enabled suite: 21 passed, real-model
  tests excluded from that count. Clippy passed. Separate real-GPU tests passed
  for exact token export and adapter retention across failed reloads and context
  resizing. Adapter retention was checked with both the discarded smoke adapter
  and the final trained adapter.
- The CLI and JSONL evaluator support explicit `--lora`; metadata includes the
  adapter path. Base-model defaults remain unchanged. Calibration remains an
  offline artifact.

The next experiment should align training and deployment precision, include
multiple decision tasks to measure retention, and reserve another untouched test
set. This run does not justify simply training longer or promoting the adapter.
