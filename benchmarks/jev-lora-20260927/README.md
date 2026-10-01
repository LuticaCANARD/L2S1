# Jev-type LoRA pilot: Gemma4 and the shared model pipeline

Gemma4 E2B was trained on 120 cases / 600 typed decisions and evaluated on the separate public test split of 400 cases / 2,000 decisions. This is a **specialist on four seen workflows**, scored against synthetic teacher labels. It is not general-model quality, an official JevBench result, or a comparison against an unseen-workflow generalist.

Run: Linux, NVIDIA GeForce RTX 3080 10GB. This is separate from the user-supplied Windows RTX 5090 decision-rules run. The GGUF hashes also differ. Both checkpoints below use the same local Q8_0 base, native executable, CUDA libraries, fresh/minimal/legacy prompts, context 8192, batch 256 and four CPU threads. One warmup and loading are excluded. Latency is per five-decision request.

## Paired native evaluation

| Metric | Base | LoRA |
| --- | ---: | ---: |
| Raw label accuracy | 54.30% | 57.85% |
| Coverage | 92.75% | 48.35% |
| Accepted accuracy | 55.69% | 66.80% |
| Correct accepted / all | 51.65% | 32.30% |
| Soft KL, lower is better | 3.7739 | 0.7544 |
| Soft Brier, lower is better | 0.5668 | 0.2942 |
| Hard NLL | 3.4975 | 1.1528 |
| Hard Brier | 0.8516 | 0.6075 |
| Hard ECE, 15 bins | 0.4202 | 0.1889 |
| Expected Score MAE | 0.5240 | 0.4638 |
| p50 ms/request | 348.0 | 363.0 |
| p95 ms/request | 465.4 | 516.1 |

Probability metrics use unrounded probabilities over all three types. Native acceptance stays at top probability ≥0.8 and candidate mass ≥0.05. Soft-target training can change coverage substantially; compare it with accepted accuracy and correct/all. No threshold or temperature was fitted to this test split.

## By output type

| Type | Test decisions | Base raw | LoRA raw | Base soft KL | LoRA soft KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| choice | 600 | 56.17% | 55.67% | 3.5114 | 1.0685 |
| noul | 600 | 50.67% | 63.83% | 5.6963 | 0.3573 |
| score | 800 | 55.62% | 55.00% | 2.5289 | 0.8165 |

## Existing rules regression

One pass over the previously seen 36-decision `decision-rules-v1` fixture, with probability ties within 1e-12 counted incorrect. These are regression cases, not additional independent holdout examples. Do not substitute these local results for the Windows RTX 5090 measurements.

| Metric | Base | LoRA |
| --- | ---: | ---: |
| Raw accuracy | 97.22% | 91.67% |
| Coverage | 94.44% | 94.44% |
| Accepted accuracy | 97.06% | 97.06% |
| Correct accepted / all | 91.67% | 91.67% |
| Accepted wrong | 1 | 1 |

## Training and reproducibility

- One fixed final epoch: 600 inputs, 50 optimizer steps, rank 8 / alpha 16 Q/V LoRA. NF4 training, bf16 compute, fp32 adapter parameters. The smoke adapter was discarded. No test-based checkpoint, hyperparameter or threshold selection.
- Training took about 505 seconds after model loading; peak allocated GPU memory was 6.55 GiB. The converted f16 GGUF adapter is about 2.7 MB. No model or adapter weights are committed.
- Train/development/unused/test splits are disjoint by source ID and exact canonical state. Near duplicates and pretraining exposure are not excluded. One training seed does not establish run-to-run robustness.
- Raw paired predictions, logs, training tokens and adapter remain under `results/jev-gemma4-20260927/`. Published reports retain hashes, counts, per-type/per-workflow metrics and all 36 regression outcomes.
- The initial baseline was run during a concurrent native build and is excluded from this paired latency table. The final pair ran sequentially after training/build activity stopped. Timing is an observation on this machine, not a controlled speedup claim.

## Model coverage and output contract

The shared pipeline has profiles for **SmolLM2, Qwen3, Gemma3, TinyLlama and Gemma4**, plus a custom profile registry. All five passed tiny CPU architecture/gradient tests; only Gemma4 has completed real-weight training, GGUF conversion and native evaluation in this report. `--models all` enables serial runs, preserving failures and before/after metrics.

[Ollaya at f9e2d11](https://github.com/ollaya-dev/ollaya/tree/f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea) supplies the reference output contract. Choice/Score use normalized maximum-probability confidence and four-decimal wire rounding; Noul is P(true) without a confidence field. Score is an expected ordered index. All 300 synthetic answer JSONs matched compiled unmodified Ollaya decision source. Native abstention remains separately visible. This is answer-rendering parity, not HTTP API or model parity.

See [the actual Jev-shaped output](example-jev-output.json), [validation scope](validation.json), and [training instructions](../../docs/JEV_LORA.md). Source dataset: [LocalLLaMA/typed-decisions, pinned revision](https://huggingface.co/datasets/LocalLLaMA/typed-decisions/tree/c76749ec58bd8c3d2ea706b31c333a9059c38f90). Dataset license: Apache-2.0, as declared in its pinned card.
