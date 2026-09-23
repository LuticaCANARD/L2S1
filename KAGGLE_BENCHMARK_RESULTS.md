# Kaggle AG News results

This is the original baseline, recorded before GPT-OSS final prefill was implemented. Its GPT-OSS row uses the open-header `model` profile. See [GPT-OSS final-prefill results](GPT_OSS_FINAL_RESULTS.md) for the same frozen articles after the fix; retain this table as historical evidence.

Date: 2026-09-21. Same frozen 400-article sample, 100 per category; one CUDA pass per model. See [protocol and reproduction](KAGGLE_BENCHMARK.md).

| Model | Correct | Wrong | Abstained | Errors / missing | Correct / all | Accepted accuracy | Coverage | Raw top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma4 | 305 | 83 | 12 | 0 / 0 | 76.25% | 78.61% | 97.00% | 77.50% |
| gemma3 | 100 | 300 | 0 | 0 / 0 | 25.00% | 25.00% | 100.00% | 25.00% |
| qwen38 | 252 | 35 | 113 | 0 / 0 | 63.00% | 87.80% | 71.75% | 78.75% |
| gpt-oss-20b | 0 | 0 | 400 | 0 / 0 | 0.00% | n/a | 0.00% | 24.25% |

Accepted accuracy excludes abstentions; correct/all retains every sampled article. Raw top-1 ignores the application abstention policy. An all-abstained model has undefined accepted accuracy. The balanced always-one-class baseline is 25%.

| Model | Correct/all 95% Wilson interval | Accepted accuracy 95% Wilson interval | p50 / p95 ms per article | Exit code |
| --- | --- | --- | ---: | ---: |
| gemma4 | 71.84%–80.16% | 74.26%–82.40% | 38.88 / 53.40 | 0 |
| gemma3 | 21.01%–29.47% | 21.01%–29.47% | 25.11 / 34.58 | 0 |
| qwen38 | 58.17%–67.59% | 83.51%–91.10% | 255.72 / 312.86 | 0 |
| gpt-oss-20b | 0.00%–0.95% | n/a | 1511.01 / 1640.13 | 0 |

## Per-class correct / 100 (abstentions remain in denominator)

| Model | World | Sports | Business | Science/Technology |
| --- | ---: | ---: | ---: | ---: |
| gemma4 | 91 | 87 | 81 | 46 |
| gemma3 | 100 | 0 | 0 | 0 |
| qwen38 | 81 | 97 | 61 | 13 |
| gpt-oss-20b | 0 | 0 | 0 | 0 |

## Provenance and scope

- Kaggle dataset: [AG News, version 2](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset). Archive SHA256: `6dce0e4e9d48d02fc63649d853dd906ba031ce9a38113bc01ad12e20ad04e23a`.
- Frozen unlabeled request SHA256: `1489716ed040e95087a6e973819ac1f346f39651db50cd2908da2b723da50a54`. Seed: `20260921`.
- Models: Gemma 4 E2B Q8_0, Gemma 3 1B Q8_0, Qwen3.8-27B UD-IQ2_XXS, GPT-OSS-20B MXFP4. Different sizes and quantizations; these are results for the existing application configuration, not model-family rankings.
- No fine-tuning, few-shot examples, prompt selection, threshold adjustment, or retries to improve scores. Historical public data may overlap model pretraining.
- Hardware: RTX 3080 10 GiB, i9-9900K, 16 GiB system RAM. Context 2048, batch 256, four threads. CUDA buffer allocation does not prove physical-VRAM residency.
- GPU model runs were sequential. A four-article Gemma 3 CPU diagnostic overlapped part of the Qwen GPU run; timing is exploratory, not a controlled hardware comparison.
- Gemma 3 selected World for every GPU article. A separate one-article-per-class CPU diagnostic also selected World for all four; this diagnostic is not included in the 400-article scores.
- Existing Gemma 3 and GPT-OSS CUDA batch-consistency failures are not resolved by this fixed-batch evaluation. GPT-OSS template/channel behavior also remains unchanged.
- Detailed predictions, errors, confusion matrices, model/executable hashes, runtime revision and source hashes are in the ignored `results/kaggle-ag-news/` directory. No dataset text or model weights were published.
