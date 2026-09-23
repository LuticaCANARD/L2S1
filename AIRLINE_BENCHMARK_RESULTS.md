# Airline sentiment: seven-model comparison

Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment), Kaggle version 4, attributed to CrowdFlower / Figure Eight. The data card specifies CC BY-NC-SA 4.0. This local benchmark keeps all tweet text and label files in ignored `results/`; it does not redistribute the dataset.

Task: classify sentiment toward an airline or its service as **negative, neutral, or positive**. The prompt explicitly asks the model to account for negation and sarcasm; it was fixed before inference and is identical across models. Inputs contain tweet text only. Labels, annotator confidence, negative-reason labels, user metadata, and other source columns are excluded from prompts.

## Data and protocol

The source contains 14,640 rows. We removed 188 duplicate rows and all 73 rows belonging to text groups with conflicting sentiment labels. Identity normalizes HTML entities, whitespace, and case. Near-duplicates and model pretraining exposure are not excluded. Crowd labels may be ambiguous; no annotator-confidence filter was applied.

Seed 20260923 selects **400 calibration-fit and 400 validation examples**, disjoint by normalized text. Each split has 134 negative, 133 neutral, and 133 positive examples. This is our balanced split, not an official test split or a sample preserving deployment class prevalence. The majority-class baseline is 33.5%.

All seven models use local RTX 3080 CUDA, fresh execution, context 2048, token batch/microbatch 256, four CPU threads, FlashAttention off, one warmup request, and the same frozen inputs. This differs from the earlier parallel Gemma 4 throughput experiment. Model-specific automatic templates remain in effect: Qwen3 non-thinking, GPT-OSS final-channel prefill, embedded model template otherwise. Recurrent/hybrid Qwen35 cannot use the current parallel bridge, motivating uniform fresh execution. Quantizations differ and are recorded; this is a comparison of these local GGUFs and this decision adapter, not a general model capability ranking.

Each model gets its own temperature fitted only on its 400 fit examples by NLL minimization over [0.05, 100]. Validation labels do not fit the temperature or thresholds. Full-vocabulary candidate mass remains uncalibrated, with the same minimum 0.05; ties abstain. Temperatures do not change argmax. Values at the search boundary are explicitly reported. Thresholds 0.6, 0.7, 0.8, 0.9, 1.0 are evaluated as requested.

## Model comparison at calibrated threshold 0.6

| Model file | Raw top-1 | Correct / wrong / abstain | Coverage | Accepted accuracy | 400-item inference (s) | Temperature |
|---|---:|---|---:|---:|---:|---:|
| gemma-4-E2B-it-Q8_0.gguf | 77.75% | 288 / 58 / 54 | 86.50% | 83.24% | 15.252 | 7.2025 |
| gemma-3-1b-it-Q8_0.gguf | 33.50% | 0 / 0 / 400 | 0.00% | n/a | 9.834 | 100.0000 (upper bound) |
| Qwen3-0.6B-Q8_0.gguf | 33.50% | 0 / 0 / 400 | 0.00% | n/a | 9.538 | 100.0000 (upper bound) |
| SmolLM2-135M-Instruct-Q8_0.gguf | 26.00% | 0 / 0 / 400 | 0.00% | n/a | 8.391 | 100.0000 (upper bound) |
| tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf | 33.50% | 0 / 0 / 400 | 0.00% | n/a | 9.991 | 100.0000 (upper bound) |
| Qwen3.8-27B-UD-IQ2_XXS.gguf | 78.50% | 263 / 45 / 92 | 77.00% | 85.39% | 97.964 | 1.8342 |
| gpt-oss-20b-MXFP4.gguf | 62.00% | 166 / 42 / 192 | 52.00% | 79.81% | 553.105 | 4.3191 |

Inference timing is a single measured pass, excludes model loading, warmup and file output, and is not a repeated latency benchmark. GPT-OSS weights exceed this GPU’s physical VRAM; its timing reflects this hardware constraint. Do not compare these numbers to the earlier parallel article-batch timings as if the execution conditions were identical.

## Threshold sweep

| Model | Threshold | Correct | Wrong | Abstain | Coverage | Accepted accuracy |
|---|---:|---:|---:|---:|---:|---:|
| gemma4 | 0.6 | 288 | 58 | 54 | 86.50% | 83.24% |
| gemma4 | 0.7 | 258 | 42 | 100 | 75.00% | 86.00% |
| gemma4 | 0.8 | 212 | 25 | 163 | 59.25% | 89.45% |
| gemma4 | 0.9 | 115 | 8 | 277 | 30.75% | 93.50% |
| gemma4 | 1.0 | 0 | 0 | 400 | 0.00% | n/a |
| gemma3 | 0.6 | 0 | 0 | 400 | 0.00% | n/a |
| gemma3 | 0.7 | 0 | 0 | 400 | 0.00% | n/a |
| gemma3 | 0.8 | 0 | 0 | 400 | 0.00% | n/a |
| gemma3 | 0.9 | 0 | 0 | 400 | 0.00% | n/a |
| gemma3 | 1.0 | 0 | 0 | 400 | 0.00% | n/a |
| qwen3-0.6b | 0.6 | 0 | 0 | 400 | 0.00% | n/a |
| qwen3-0.6b | 0.7 | 0 | 0 | 400 | 0.00% | n/a |
| qwen3-0.6b | 0.8 | 0 | 0 | 400 | 0.00% | n/a |
| qwen3-0.6b | 0.9 | 0 | 0 | 400 | 0.00% | n/a |
| qwen3-0.6b | 1.0 | 0 | 0 | 400 | 0.00% | n/a |
| smollm2 | 0.6 | 0 | 0 | 400 | 0.00% | n/a |
| smollm2 | 0.7 | 0 | 0 | 400 | 0.00% | n/a |
| smollm2 | 0.8 | 0 | 0 | 400 | 0.00% | n/a |
| smollm2 | 0.9 | 0 | 0 | 400 | 0.00% | n/a |
| smollm2 | 1.0 | 0 | 0 | 400 | 0.00% | n/a |
| tinyllama | 0.6 | 0 | 0 | 400 | 0.00% | n/a |
| tinyllama | 0.7 | 0 | 0 | 400 | 0.00% | n/a |
| tinyllama | 0.8 | 0 | 0 | 400 | 0.00% | n/a |
| tinyllama | 0.9 | 0 | 0 | 400 | 0.00% | n/a |
| tinyllama | 1.0 | 0 | 0 | 400 | 0.00% | n/a |
| qwen38 | 0.6 | 263 | 45 | 92 | 77.00% | 85.39% |
| qwen38 | 0.7 | 234 | 30 | 136 | 66.00% | 88.64% |
| qwen38 | 0.8 | 197 | 16 | 187 | 53.25% | 92.49% |
| qwen38 | 0.9 | 99 | 5 | 296 | 26.00% | 95.19% |
| qwen38 | 1.0 | 0 | 0 | 400 | 0.00% | n/a |
| gpt-oss-20b | 0.6 | 166 | 42 | 192 | 52.00% | 79.81% |
| gpt-oss-20b | 0.7 | 109 | 11 | 280 | 30.00% | 90.83% |
| gpt-oss-20b | 0.8 | 49 | 2 | 349 | 12.75% | 96.08% |
| gpt-oss-20b | 0.9 | 4 | 0 | 396 | 1.00% | 100.00% |
| gpt-oss-20b | 1.0 | 0 | 0 | 400 | 0.00% | n/a |

## Uncalibrated default policy and low candidate mass

The following uses the existing raw-probability threshold 0.8 and mass threshold 0.05. It separates the adapter’s existing behavior from the offline temperature experiment. A model with no accepted predictions has undefined accepted accuracy, not zero. Lowering the probability threshold cannot override low candidate mass.

| Model | Correct / wrong / abstain | Raw top-1 prediction totals | Low-mass cases |
|---|---|---|---:|
| gemma4 | 309 / 83 / 8 | {'negative': 164, 'neutral': 97, 'positive': 139} | 0 |
| gemma3 | 134 / 265 / 1 | {'negative': 398, 'positive': 2} | 0 |
| qwen3-0.6b | 134 / 266 / 0 | {'negative': 400} | 0 |
| smollm2 | 0 / 0 / 400 | {'negative': 244, 'positive': 156} | 0 |
| tinyllama | 0 / 0 / 400 | {'negative': 399, 'neutral': 1} | 400 |
| qwen38 | 251 / 39 / 110 | {'negative': 157, 'neutral': 150, 'positive': 93} | 0 |
| gpt-oss-20b | 216 / 117 / 67 | {'negative': 141, 'neutral': 231, 'positive': 28} | 0 |

## Reproduction

Archive SHA-256: `c0dbee48cac32110a607430dc5c1941a1728383adef20a893ded91adbbba2de2`. Request hashes, selected row IDs, labels, and annotations are frozen in `results/kaggle-airline-20260922/selection.json`. Each model directory contains raw outputs, actual prompt/backend metadata, model/runtime/binary hashes, commands, and logs. `comparison-audit.json` independently recounts completeness and original selected outcomes. No application defaults or model weights were changed.

```sh
# Download the pinned archive into a new output folder, then:
python3 scripts/kaggle_airline.py prepare --folder results/airline-new
python3 scripts/kaggle_airline.py run --folder results/airline-new \
  --models gemma4 gemma3 qwen3-0.6b smollm2 tinyllama qwen38 gpt-oss-20b
```

The runner uses the archived evaluator and original CUDA runtime from this workspace. It refuses to overwrite existing model runs. The dataset-selection tests cover normalization, contradictory labels, disjoint splits, reproducibility, and mass/tie abstention gates. Python test suite: `python3 -m unittest discover -s scripts -p "test_*.py"`.

## Interpretation and diagnostic checks

On this balanced sample, Qwen3.8 has 314/400 raw-correct answers versus Gemma 4 with 311/400. The paired discordances are 38 Qwen-only correct and 35 Gemma-only correct (two-sided exact McNemar p = 0.8151). This does not establish a quality advantage; Gemma 4 is substantially faster on this hardware. At calibrated threshold 0.6 it also accepts more cases, at lower accepted accuracy. These are different coverage/accuracy tradeoffs.

GPT-OSS correctly classifies only 27 of the 133 positive examples; 95 are predicted neutral and 11 negative. No validation case is rejected for candidate mass below 0.05, so candidate-mass gating does not explain this error pattern. The final-channel prefill is active. This identifies an observed failure pattern, not its cause.

Gemma 3, Qwen3 0.6B, SmolLM2, and TinyLlama reach the temperature-search upper bound. Temperature scaling can reduce unjustified confidence but cannot fix their low raw accuracy or missing task understanding in this adapter. TinyLlama additionally fails the candidate-mass gate on all 400 validation cases.

### Post-hoc option-order probe

Selected the first four frozen validation cases per class (12 unique cases) and reran all three cyclic candidate orders for Gemma 3 and Qwen3 0.6B. This is a 72-call diagnostic, not another independent accuracy benchmark. No recalibration was applied.

| Model | First candidate | A predictions / 12 | Correct / 12 |
|---|---|---:|---:|
| gemma3 | negative | 12 | 4 |
| gemma3 | neutral | 0 | 4 |
| gemma3 | positive | 12 | 4 |
| qwen3-0.6b | negative | 12 | 4 |
| qwen3-0.6b | neutral | 12 | 4 |
| qwen3-0.6b | positive | 12 | 4 |

Qwen3 0.6B chose A in all 36 calls even though A rotated from negative to neutral to positive. Gemma 3 chose A in all cases for the negative-first and positive-first orders, while the neutral-first order chose B in 11 cases and C in one. Both scored 4/12 for each order. These results demonstrate decision-format/order sensitivity on this diagnostic, not that the models have no general sentiment capability.

### Verification

- All seven models completed both 400-example splits: 5600 inference outcomes, no errors, missing IDs, or truncation. These reuse 800 distinct examples, not 5600 independent test samples.
- Independent audits recount raw and original-policy outcomes; the split/normalization/gating/calibration Python suite passes all 12 tests. The additional 72 option-order calls also completed.
- A reporting-only p50 index was corrected from a fixed 100-batch assumption to nearest rank over the actual 400 batches. Raw outcomes, total inference time, calibration, and accuracy were unchanged; details are recorded in `timing-report-correction.json`.
- No inference-engine defaults, prompts after the frozen selection, or model weights were changed for the seven-model comparison.
