# TrashNet material classification with local L2S1 vision

Evaluated on 2026-09-25. Source: [garythung/trashnet](https://github.com/garythung/trashnet), commit `6fa2b878c6c1b4304b91109070ce0edf9279bb31`, `data/dataset-resized.zip`. The archive SHA-256 is `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`. The source has 2,527 photographs in six material classes. Its creator states that objects were photographed on a white posterboard and resized to 512 × 384.

## Protocol

- Fixed seed `20260925`; 20 unique original JPEGs sampled from each of cardboard, glass, metal, paper, plastic, and trash, giving 120 images. `selection.json` contains filenames and per-image SHA-256 digests. The same shuffled order was used for all models and prompt variants.
- One image per serial local `/v1/decisions` request. The prompt asked for the main waste item's primary material, using six fixed English descriptions. The request contained only the image, prompt, descriptions, and empty state; filename and ground truth were added to observations after the response.
- Local CUDA build `target/cuda-architectures/sm_86/release/run-l2s1` on NVIDIA GeForce RTX 3080. Each model and its matching projector were loaded once. No fine tuning, demonstrations, threshold changes, warmup requests, or image augmentation. Default L2S1 abstention thresholds were used. Qwen3-VL 2B Q8_0 and its projector came from [ggml-org/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/tree/ea6a11058182570be6436b9a2e4ee7f7b49f908d); their SHA-256 values (`b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39`, `69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b`) match the earlier Caltech evaluation artifacts.
- The source archive has no official held-out split for this run. These are zero-shot results for this balanced frozen sample, not full-dataset or deployment accuracy. Published TrashNet CNN results use a different train/test protocol and are not directly comparable.

## Results

| Model (Q8_0) | Correct / all | Coverage | Correct / accepted | Raw top-1 / all | HTTP latency p50 / p95¹ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B it | 64/120 = 53.3% | 113/120 = 94.2% | 64/113 = 56.6% | 64/120 = 53.3% | 79.1 / 88.9 ms |
| Qwen3-VL 2B Instruct | 91/120 = 75.8% | 115/120 = 95.8% | 91/115 = 79.1% | 95/120 = 79.2% | 101.1 / 116.2 ms |
| SmolVLM 256M Instruct | 0/120 = 0% | 0/120 = 0% | undefined | 13/120 = 10.8% | 108.7 / 126.0 ms |

¹ Serial request time on this machine, with models already loaded; the first request is included. Startup to HTTP health: Gemma 28.3 s, Qwen3-VL 11.3 s, and SmolVLM 2.5 s. These figures do not represent concurrent throughput or other hardware.

The balanced six-class sample has a 16.7% constant-class baseline. Qwen3-VL got 27 more accepted answers right than Gemma: on matched images, Qwen alone was correct on 30, Gemma alone on 3, both on 61, and neither on 26. Qwen's five abstentions were all due to low top-option probability; four had the correct raw top-1. It missed every `trash` image despite performing well on the other five classes. Gemma also had uneven class performance. SmolVLM's 0% correct/all is caused by **120 abstentions**, not by 120 accepted wrong answers. Its six-option candidate mass ranged from 0.00000157 to 0.000980, below the default minimum; 113 responses also had low top-option probability. Candidate mass is coverage of the named options in the model's next-token distribution, not a probability that the answer is correct. Its forced top-1 result is also below the constant-class baseline.

### Per-class correct / 20

| Ground truth | Gemma accepted correct | Qwen accepted correct | SmolVLM raw top-1 correct |
| --- | ---: | ---: | ---: |
| cardboard | 19 | 19 | 5 |
| glass | 14 | 19 | 7 |
| metal | 14 | 19 | 0 |
| paper | 7 | 18 | 1 |
| plastic | 9 | 16 | 0 |
| trash | 1 | 0 | 0 |

Gemma most often confused paper with cardboard (11/20) and miscellaneous trash with paper (9/20) or cardboard (7/20). Its prediction distribution leaned toward cardboard (49/120 raw top-1). The full decision scores and abstention reasons are preserved per image.

Qwen classified all 20 `trash` images as either plastic (11) or paper (9). This class is miscellaneous by construction; the observed scores should not be interpreted as material probabilities or a measure of visual ambiguity.

## Material-traits prompt comparison

Before the second run, `prompts.json` froze the exact `traits_v1` instruction and its SHA-256 (`2f295757f010e5e6a18bdba7d26a79f802e2315a0a17b280cd75df08f341dbc1`). It describes observable cues for all six categories: board thickness and corrugation; glass rigidity and reflections; metal sheen, rims, and foil; flexible paper sheets; molded or film plastic; and residual, soiled, or mixed waste for `trash`. It directs the model to use the dominant visible material and not infer from background, printed words, or recyclability. The six option descriptions were unchanged. The image identities and order, model/projector files, CUDA binary, and default abstention policy match the baseline runs. The instruction was frozen before any `traits_v1` inference request.

| Model | Baseline correct / all | Traits correct / all | Baseline coverage | Traits coverage | Raw top-1 baseline → traits | HTTP p50 baseline → traits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-VL 2B | 91/120 | **89/120** | 115/120 | 107/120 | 95 → 93 | 101.1 → 129.4 ms |
| Gemma 4 E2B | 64/120 | **65/120** | 113/120 | 115/120 | 64 → 65 | 79.1 → 126.4 ms |
| SmolVLM 256M | 0/120 | **0/120** | 0/120 | 0/120 | 13 → 19 | 108.7 → 124.7 ms |

Qwen changed two previously wrong or abstained images to accepted correct answers, while four previously correct answers became wrong or abstained. Gemma made 13 such gains and 12 losses. SmolVLM still abstained on every image because candidate mass stayed below the default minimum. Longer requests had higher measured median latency in these single serial runs.

| Ground truth | Qwen baseline → traits correct / 20 | Gemma baseline → traits correct / 20 |
| --- | ---: | ---: |
| cardboard | 19 → 19 | 19 → 12 |
| glass | 19 → 20 | 14 → 17 |
| metal | 19 → 19 | 14 → 14 |
| paper | 18 → 15 | 7 → 4 |
| plastic | 16 → 15 | 9 → 10 |
| trash | 0 → 1 | 1 → 8 |

The detailed traits improved `trash` recall, especially for Gemma, but did not materially improve correct/all: Qwen lost two correct answers and Gemma gained one on this set. These paired results are exploratory. The prompt was written after seeing baseline aggregate errors, so repeating on this same sample cannot establish an unbiased generalization gain. A fresh held-out set would be needed for that claim.

## More detailed traits and a lower acceptance threshold

The next experiment froze `traits_v2` in `experiment-traits-v2.json` before inference. It expands the material cues and distinguishing cases to 288 words, without examples from the selected test images or any labels/filenames in requests. Its SHA-256 is `beb2a5980ae5ec92cdaec1e073f17905871597cac61f5349fd70d32d81880662`. The six option IDs and descriptions are unchanged. The same executable, checkpoint/projector pairs, images, order, CUDA device, and request protocol were used. The CLI acceptance setting changed from `min_top_probability=0.8` to `0.5`; `min_candidate_mass=0.05` stayed fixed.

`top_option_probability` is a conditional softmax over the six named answer tokens. The 0.5 setting allows a top option that carries at least half of that six-way mass to be selected, provided candidate mass and tie checks also pass. It is **not** calibrated probability that the answer is correct. Lowering this threshold changes selection/abstention after scoring; it does not change the model's answer ranking. Consequently, 0.5 results for earlier prompts and 0.8 results for `traits_v2` can be computed exactly from the stored scores. The counterfactual calculation was verified against the actual selections in every run.

### Correct / all, with accepted count in parentheses

| Model | Prompt | Threshold 0.8 | Threshold 0.5 | Raw top-1 correct |
| --- | --- | ---: | ---: | ---: |
| Qwen3-VL 2B | Baseline | 91 (115) | 95 (120) | 95 |
| Qwen3-VL 2B | Traits v1 | 89 (107) | 93 (118) | 93 |
| Qwen3-VL 2B | **Traits v2** | 85 (97) | **95 (117)** | 96 |
| Gemma 4 E2B | Baseline | 64 (113) | 64 (120) | 64 |
| Gemma 4 E2B | Traits v1 | 65 (115) | 65 (120) | 65 |
| Gemma 4 E2B | **Traits v2** | 69 (117) | **69 (119)** | 69 |
| SmolVLM 256M | Baseline | 0 (0) | 0 (0) | 13 |
| SmolVLM 256M | Traits v1 | 0 (0) | 0 (0) | 19 |
| SmolVLM 256M | **Traits v2** | 0 (0) | **0 (0)** | 21 |

For Qwen, lowering the old prompt's threshold alone yields 95 correct, the same total as the detailed prompt at 0.5. The detailed prompt moves Qwen's accepted `trash` results from 0/20 to 13/20, but loses correct answers in cardboard (19→16), metal (19→15), and paper (18→15); plastic rises from 16→17. At 0.8, the detailed prompt accepts fewer answers and scores 85/120. For Gemma, lowering the old threshold adds no correct answers, whereas the detailed prompt increases raw top-1 and accepted correct from 64 to 69. These are different class tradeoffs, not a consistent model-wide gain from detail alone.

The actual `traits_v2`/0.5 HTTP median times were 149.4 ms for Qwen, 145.3 ms for Gemma, and 138.1 ms for SmolVLM; these are single serial runs with a longer prompt. SmolVLM still abstained on all 120: its candidate mass was at most 0.00293, below the unchanged 0.05 floor. A lower top-option threshold cannot resolve that failure mode. Because prompts were revised after inspecting earlier results on this same sample, the observed changes are exploratory and need a fresh held-out sample before treating them as generalizable.

## Reproduce and inspect

- `evaluate.py`: pinned source validation, deterministic selection, live HTTP evaluation, summary calculation.
- `selection.json`: frozen image identities and hashes.
- `gemma4/observations.jsonl`, `qwen3vl/observations.jsonl`, `smolvlm/observations.jsonl`: one response-derived record per image.
- `gemma4/summary.json`, `qwen3vl/summary.json`, `smolvlm/summary.json`: counts, per-class results, latency, and binary/model/projector hashes.
- `runtime-checks.json`: original local server-log hashes and image-encoding counts for all nine runs. The repetitive raw logs remain in ignored local results and are excluded from this PR.
- `prompts.json`: the exact baseline and traits instructions, unchanged option descriptions, and the traits prompt hash.
- `gemma4-traits-v1/`, `qwen3vl-traits-v1/`, `smolvlm-traits-v1/`: observation, summary, and server log artifacts for the second run.
- `prompt-comparison.json`: an audited paired comparison verifying identical ordered image identities and hashes, selection, executable, checkpoint, and projector for each model.
- `experiment-traits-v2.json`: the exact more detailed instruction, option schema, threshold settings, and sample hash frozen before the third inference run.
- `qwen3vl-traits-v2-p50/`, `gemma4-traits-v2-p50/`, `smolvlm-traits-v2-p50/`: the third run's raw observations and summaries.
- `threshold-analysis.json`: the paired threshold recomputation from all nine runs, with checks against actual accepted decisions.

The 41 MB source image archive and model weights are not committed. Download the [pinned source archive](https://raw.githubusercontent.com/garythung/trashnet/6fa2b878c6c1b4304b91109070ce0edf9279bb31/data/dataset-resized.zip) as `dataset-resized.zip` and verify the SHA-256 above. Run `python3 evaluate.py prepare --source dataset-resized.zip --output selection.json --count 20 --seed 20260925` in a fresh output directory, followed by `python3 evaluate.py run --help` for model and binary inputs. The existing `selection.json` is intentionally opened with exclusive creation to prevent accidental replacement.
