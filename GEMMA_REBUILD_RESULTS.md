# Gemma evaluation after rebuild

Date: 2026-09-22 (Asia/Seoul). One CUDA pass per checkpoint on the same frozen Kaggle AG News sample: 400 articles, 100 per class. The archived executable is exactly the one used for the preceding GPT-OSS rebuild test.

Settings: `legacy` prompt layout, `fresh` execution, automatic model chat profile, context 2,048, batch 256, four threads, minimum option probability 0.8 and candidate mass 0.05.

| Model | Correct | Wrong | Abstained | Correct/all | Accepted accuracy | Coverage | Raw top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemma4 | 305 | 83 | 12 | 76.25% | 78.61% | 97.00% | 77.50% |
| gemma3 | 100 | 300 | 0 | 25.00% | 25.00% | 100.00% | 25.00% |
| gpt-oss-20b (preceding run) | 129 | 43 | 228 | 32.25% | 75.00% | 43.00% | 55.25% |

Accepted accuracy excludes abstentions. Correct/all includes all 400 articles. These are measurements on this public dataset and particular quantized checkpoints; they are not calibrated confidence or a universal model ranking.

| Model | p50 / p95 ms per article | Errors / missing |
| --- | ---: | ---: |
| gemma4 | 41.25 / 57.29 | 0 / 0 |
| gemma3 | 26.53 / 34.76 | 0 / 0 |

## Comparison and validation

- gemma4: 0 selections/abstentions changed from the initial Kaggle benchmark; previous correct/wrong/abstained 305/83/12, current 305/83/12.
- gemma3: 0 selections/abstentions changed from the initial Kaggle benchmark; previous correct/wrong/abstained 100/300/0, current 100/300/0.
- Gemma 3 selected World for every article. Its 25% result equals the always-one-class baseline on this balanced sample.
- Both processes exited successfully. All 800 outputs had unique expected IDs, no errors or missing articles, and independently recounted correct/wrong/abstained totals.
- Recorded response metadata confirms legacy/fresh CUDA execution. Every result reused zero prefix tokens. This run does not evaluate state-first or prefix-cache speedups.

## Reproducibility

- Evaluator SHA256: `7c60b41200667dd1a162c5c41baeff302ea1571c69f454a7a2c71d8dbdda7aa0`.
- Frozen requests SHA256: `1489716ed040e95087a6e973819ac1f346f39651db50cd2908da2b723da50a54`.
- gemma-4-E2B-it-Q8_0.gguf SHA256: `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`.
- gemma-3-1b-it-Q8_0.gguf SHA256: `b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5`.
- Commands, checkpoint hashes, settings, per-article predictions, confusion matrices, Wilson intervals, logs and independent recounts: `results/gemma-rebuild-20260922/` (local, Git-ignored).
- Source and executable snapshot: `results/rebuild-20260921/`. No model weights or dataset text were published.
