<a id="trashnet-material-classification-with-local-l2s1-vision"></a>
# TrashNet material classification with local L2S1 vision

[English](REPORT.md) · [한국어](../../../ko/benchmarks/trashnet-vision-20260925/REPORT.md) · [日本語](../../../ja/benchmarks/trashnet-vision-20260925/REPORT.md)

[English index](../../README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../../ja/README.md)


Evaluated on 2026-09-25. Source: [garythung/trashnet](https://github.com/garythung/trashnet), commit `6fa2b878c6c1b4304b91109070ce0edf9279bb31`, `data/dataset-resized.zip`. The archive SHA-256 is `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`. The source has 2,527 photographs in six material classes. Its creator states that objects were photographed on a white posterboard and resized to 512 × 384.

<a id="protocol"></a>
## Protocol

- Fixed seed `20260925`; 20 unique original JPEGs sampled from each of cardboard, glass, metal, paper, plastic, and trash, giving 120 images. `selection.json` contains filenames and per-image SHA-256 digests. The same shuffled order was used for all models and prompt variants.
- One image per serial local `/v1/decisions` request. The prompt asked for the main waste item's primary material, using six fixed English descriptions. The request contained only the image, prompt, descriptions, and empty state; filename and ground truth were added to observations after the response.
- Local CUDA build `target/cuda-architectures/sm_86/release/run-l2s1` on NVIDIA GeForce RTX 3080. Each model and its matching projector were loaded once. No fine tuning, demonstrations, threshold changes, warmup requests, or image augmentation. Default L2S1 abstention thresholds were used. Qwen3-VL 2B Q8_0 and its projector came from [ggml-org/Qwen3-VL-2B-Instruct-GGUF](https://huggingface.co/ggml-org/Qwen3-VL-2B-Instruct-GGUF/tree/ea6a11058182570be6436b9a2e4ee7f7b49f908d); their SHA-256 values (`b7802e29f71a9e5b5e3f83f613df898a2204342dcea71a231ea501d481813c39`, `69066c8f279ec85ff48ab4059f6ebba0d2932ca57667f2bbdac7d9805bca9e7b`) match the earlier Caltech evaluation artifacts.
- The source archive has no official held-out split for this run. These are zero-shot results for this balanced frozen sample, not full-dataset or deployment accuracy. Published TrashNet CNN results use a different train/test protocol and are not directly comparable.

<a id="results"></a>
## Results

| Model (Q8_0) | Correct / all | Coverage | Correct / accepted | Raw top-1 / all | HTTP latency p50 / p95¹ | Fresh → native batch4 mean/ image² | Native correct / all (accepted)² | Optimized mean/image³ | Optimized correct/all (accepted)³ | Live fresh → serial cache/compact mean/image⁴ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gemma 4 E2B it | 64/120 = 53.3% | 113/120 = 94.2% | 64/113 = 56.6% | 64/120 = 53.3% | 79.1 / 88.9 ms | 77.4 → 69.7 ms (-10.0%) | 63/120 (112) | 51.8 ms (-33.5%) | 62/120 (115) | 79.9 → 81.1 ms (1.6%) |
| Qwen3-VL 2B Instruct | 91/120 = 75.8% | 115/120 = 95.8% | 91/115 = 79.1% | 95/120 = 79.2% | 101.1 / 116.2 ms | 101.7 → 86.7 ms (-14.7%) | 93/120 (117) | 57.3 ms (-43.6%) | 92/120 (116) | 101.9 → 102.4 ms (0.5%) |
| SmolVLM 256M Instruct | 0/120 = 0% | 0/120 = 0% | undefined | 13/120 = 10.8% | 108.7 / 126.0 ms | 109.1 → 67.1 ms (-38.5%) | 0/120 (0) | 42.7 ms (-60.4%) | 0/120 (0) | 117.3 → 124.6 ms (6.2%) |

¹ Serial request time on this machine, with models already loaded; the first request is included. Startup to HTTP health: Gemma 28.3 s, Qwen3-VL 11.3 s, and SmolVLM 2.5 s. These figures do not represent concurrent throughput or other hardware.

² Measured on 2026-09-26 with the same frozen baseline prompt and images. Both modes use four-image HTTP requests; `fresh` runs them serially and `parallel` uses four independent native sequences/KV streams. One untimed group warms context/graph allocation, then three full 120-image passes are timed in each mode. Mean/image is amortized group completion time, not individual response latency; model startup is excluded. These new columns use a different warmup/repetition protocol from the original single-image latency column. **All three CUDA comparisons fail the existing numerical equivalence criterion; batching remains experimental.** Native accepted counts are in parentheses. SmolVLM still abstains on all images. See the native batch section below.

³ Combined profile measured on 2026-09-26, with the same frozen 120 images and three timed passes. Percentages compare the final candidate against validated, earlier fresh baselines of qwen3vl: 101.6 ms/image, gemma4: 77.9 ms/image, smolvlm: 107.8 ms/image. These are amortized four-image group costs on RTX 3080, not per-image response times. Recorded baseline timings are reused explicitly rather than rerun alongside the final candidate. See the combined optimization section.

⁴ Historical live same-binary fresh/full/cache-off versus serial cache/compact components, one warmup group then three 120-image passes per mode. Batch/microbatch256, Flash Attention off, context4096, threads4 and read loading match. Both execute each image serially; four-image HTTP grouping is not native decoder batching. Scores, complete option rankings, selections, abstentions and token counts match exactly on all images/passes. These fresh timings are newly measured rather than copied from an earlier baseline; compare per-pass ranges below before interpreting small latency changes.

The balanced six-class sample has a 16.7% constant-class baseline. Qwen3-VL got 27 more accepted answers right than Gemma: on matched images, Qwen alone was correct on 30, Gemma alone on 3, both on 61, and neither on 26. Qwen's five abstentions were all due to low top-option probability; four had the correct raw top-1. It missed every `trash` image despite performing well on the other five classes. Gemma also had uneven class performance. SmolVLM's 0% correct/all is caused by **120 abstentions**, not by 120 accepted wrong answers. Its six-option candidate mass ranged from 0.00000157 to 0.000980, below the default minimum; 113 responses also had low top-option probability. Candidate mass is coverage of the named options in the model's next-token distribution, not a probability that the answer is correct. Its forced top-1 result is also below the constant-class baseline.

<a id="per-class-correct--20"></a>
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

<a id="material-traits-prompt-comparison"></a>
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

<a id="more-detailed-traits-and-a-lower-acceptance-threshold"></a>
## More detailed traits and a lower acceptance threshold

The next experiment froze `traits_v2` in `experiment-traits-v2.json` before inference. It expands the material cues and distinguishing cases to 288 words, without examples from the selected test images or any labels/filenames in requests. Its SHA-256 is `beb2a5980ae5ec92cdaec1e073f17905871597cac61f5349fd70d32d81880662`. The six option IDs and descriptions are unchanged. The same executable, checkpoint/projector pairs, images, order, CUDA device, and request protocol were used. The CLI acceptance setting changed from `min_top_probability=0.8` to `0.5`; `min_candidate_mass=0.05` stayed fixed.

`top_option_probability` is a conditional softmax over the six named answer tokens. The 0.5 setting allows a top option that carries at least half of that six-way mass to be selected, provided candidate mass and tie checks also pass. It is **not** calibrated probability that the answer is correct. Lowering this threshold changes selection/abstention after scoring; it does not change the model's answer ranking. Consequently, 0.5 results for earlier prompts and 0.8 results for `traits_v2` can be computed exactly from the stored scores. The counterfactual calculation was verified against the actual selections in every run.

<a id="correct--all-with-accepted-count-in-parentheses"></a>
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

<a id="native-four-image-batching-2026-09-26"></a>
## Native four-image batching (2026-09-26)

The native bridge batches compatible mtmd projector chunks and schedules text/image decoder rows from independent prompts. The decoder uses separate KV streams and preserves each image’s M-RoPE positions. All 90 timed parallel HTTP requests per model reported `decoder_batch_max_sequences=4`; encoder batching is a separate measurement. Qwen’s projector stayed at one image per encode, Gemma reached four, and the exact shape distributions are in `native-batch4-summary.json`.

The same executable SHA-256 was used for all final runs: `78b54a2347490c4a245d3bcf02a8e5ade28f65bda1fa44b73f81b9b5059691ee`. CUDA architecture 86, context 4096 per question, token batch/microbatch 256, threads 4, flash attention off, width 4, model loading by read, full evidence, and the unchanged 0.8/0.05 policy were used. Modes ran fresh-first for Qwen/SmolVLM and parallel-first for Gemma. All three repetitions gave the same selected values and rankings within each mode.

| Model | Changed selected images /120 | Changed raw top-1 images /120 | Max option probability delta | Max candidate mass delta | Existing equivalence criterion |
|---|---:|---:|---:|---:|---|
| Qwen3-VL 2B Instruct | 2 | 0 | 0.086350 | 1.68464e-05 | failed |
| Gemma 4 E2B it | 6 | 5 | 0.955545 | 0.000122204 | failed |
| SmolVLM 256M Instruct | 0 | 8 | 0.193323 | 0.0006789 | failed |

Qwen accepted correct answers increased from 91 to 93, while raw top-1 stayed 95/120. Gemma accepted correct answers changed from 64 to 63; raw top-1 stayed 64/120 in aggregate but five individual predictions changed. SmolVLM accepted none in either mode and raw top-1 correct changed 13 → 12. These are task-specific changes from a different numerical execution path, not an accuracy improvement claim. No tolerance was enlarged: equivalence requires probability/mass differences below 0.02 and unchanged selections, rankings, abstention reasons, and token counts. Input token counts matched throughout.

Functional isolation is tested separately: the native color-fixture test passed on all three CUDA models, including changing one image without changing other image logits, request order, partial waves, invalid inputs, and recovery. Fresh-versus-batch numerical parity remains a separate ignored real-model test. A four-image Qwen CPU diagnostic matched serial scores exactly; the corresponding CUDA diagnostic did not. Gemma’s batched projector embeddings also differed from serial embeddings on CUDA, and disabling CUDA fusion reduced but did not eliminate a four-image score difference. Those probes do not establish the cause or full-dataset CPU equivalence.

The measured improvements therefore apply to this machine and this experimental execution mode. The default remains fresh. Metal runtime was not verified for this patch.

Reproduce with `python3 benchmarks/trashnet-vision-20260925/evaluate_batch.py --help`. The script freezes each four-image payload, checks original image hashes and RTX 3080 offload, records full real responses and native counters, and terminates/waits for each server. `native-batch4-summary.json` retains model/projector/binary/sample identities, every pass duration, quality counts, changed image indices, and counter distributions. Full response JSONL and logs remain in the ignored `results/vision-trashnet-20260926-native-batch4-final/` directory.

<a id="combined-vision-optimizations-2026-09-26"></a>
## Combined vision optimizations (2026-09-26)

The final executable SHA-256 is `067bcc5ff58b6e9b0739c697035d4f9a62f2f0542afc4687c0ee3841d54e02ae`. The opt-in `--vision-optimized` profile combines four independent decoder streams, dynamic KV reservation, batch/microbatch 1024, Flash Attention, compact evidence, an 8 MiB bounded preparation cache, and exact duplicate-image projector reuse. Already-clean KV is not physically zeroed again; successful native vision calls perform their own cleanup, with a Rust error/unwind guard. The mtmd/helper/CLIP logger uses the same controlled callback as llama.cpp.

Each unique image chunk is encoded independently in reuse mode. An isolation regression exposed that changing duplicate-image membership otherwise changed encoder batch shape/order and Gemma scores for other images. The final path fixes this while retaining four-way decoder inference. All 90 timed requests per model report decoder width 4, projector batch maximum 1, one physical KV clear, and one skipped clear. These 120 distinct images reuse **zero projector chunks** and zero prefix KV tokens. Repeated state/decision prompts hit the preparation cache; the last wave reports four vision entries, four misses and 360 hits including warmup. Embeddings are only reused inside a wave, never across images with different bytes or across HTTP requests.

| Model | Changed selected images /120 | Changed raw top-1 images /120 | Max option probability delta | Raw correct fresh → optimized | Equivalence |
|---|---:|---:|---:|---:|---|
| Qwen3-VL 2B | 1 | 0 | 0.191486 | 95 → 95 | failed |
| Gemma 4 E2B | 6 | 3 | 0.691368 | 64 → 64 | failed |
| SmolVLM 256M | 0 | 13 | 0.131205 | 13 → 13 | failed |

The existing equivalence criterion is unchanged and **fails for all three optimized runs**. Correct/all and acceptance counts are in the existing results table; the three repetitions have identical within-mode quality counts. Execution changes are not an accuracy improvement claim. SmolVLM still abstains on all images.

Separate one-pass controls compared the prepatch binary with the final binary in ordinary fresh/full/batch256/FlashOff mode. All 120 images per model preserved probabilities and candidate mass exactly (maximum delta 0), with no selected value, top-1, abstention-reason or token-count changes. These control runs establish numerical preservation of the default on this sample; they do not measure an isolated speed benefit from skipped clears.

Six real-model optimized vision tests passed across the three checkpoints, covering image substitution/isolation, duplicate-image reuse, unequal candidate banks, compact/full equality at the same compute configuration, partial waves, cache invalidation, malformed images, context overflow and recovery. Real CUDA text prefix reuse, shared-state boundaries, compact failure recovery, and Bonsai hybrid snapshot restoration also passed. Metal remains unverified.

`optimized-batch4-summary.json` retains final binary/model/input identities, every pass duration, quality/equivalence results, counter distributions, default-preservation controls and real-model test outcomes. Full real responses and logs remain in ignored `results/vision-optimized-20260926/*-optimized-final/` and `*-fresh-control-final/`. The evaluator can use `--candidate-vision-optimized`; `--baseline-recorded-run` records and validates an earlier fresh baseline against its frozen requests, runtime configuration and raw timing samples.

<a id="optimization-component-ablation-2026-09-26"></a>
## Optimization component ablation (2026-09-26)

Eight configurations were screened independently on each model, using the same frozen 120 images, prompts and policy. All cache/compact variants preserved candidate logits, probabilities, candidate mass, complete option ranking, selected values, abstention reasons and input token counts exactly. The stricter `exact_evidence_match` does not enlarge the existing0.02 equivalence threshold.

| Change from fresh at batch 256/FlashOff/full/cache-off | Qwen exact | Gemma exact | Smol exact |
|---|---|---|---|
| Preparation cache | yes | yes | yes |
| Compact evidence | yes | yes | yes |
| Cache + compact | yes | yes | yes |
| Batch/microbatch 1024 only | yes | yes | yes |
| Flash Attention only | no | no | no |
| Parallel 4, unchanged batch/attention | no | no | no |
| Parallel 4, independent encoder chunks | no | no | no |
| Parallel 4, independent chunks + compact/cache/dynamicKV | no | no | no |

With Flash Attention changed alone, selected outcomes changed on Qwen 1 / Gemma 4 / Smol 0 images and raw top-1 on 0/2/8. With parallel inference changed alone, selected outcomes changed on 2/6/0 images and raw top-1 on 0/5/8. Full option rankings changed as well, including Smol responses that all abstain. Independently encoded unique chunks remove the encoder grouping dependence but do not make the parallel decoder match fresh. Batch 1024 alone matched on this short prompt sample; that result does not establish parity for longer states or different image layouts. The serial component comparison retained 256.

The serial component experiment used the original vision helper with batch/microbatch 256 and Flash Attention off, exact 8 MiB preparation caching and compact evidence. It did not replay saved predictions or cache image embeddings or KV. The measured binary hash was `aa71c1acd2497174cbda6bbe6e6d49e52d662d1766c3ccc493307d90770930ff`. These are constituent optimizations of Final optimized batch4. Its parallel decoder and Flash Attention still alter scores. The separate serial preset was removed after this experiment found no speedup; historical commands and test names in the immutable summary describe the measured revision, not current CLI/API availability.

Three live 120-image passes per mode on that measured binary preserved all measured logits/probabilities/mass/rankings/choices/reasons/token counts exactly. Quality and coverage remain fresh: Qwen 91 accepted correct / 115 accepted (raw 95), Gemma 64/113 (raw 64), and Smol 0/0 (raw 13). Each final image call reports one physical KV clear and one skipped clear; the final prepared vision cache reports 4 entries, 4 misses and 360 hits including warmup. These are preparation hits, not decoder KV reuse.

<a id="live-per-pass-processing-costs"></a>
### Live per-pass processing costs

| Model | Fresh ms/image by pass | Serial cache/compact ms/image by pass | Mean time change |
|---|---|---|---:|
| Qwen3-VL | 101.41, 101.44, 102.70 | 103.10, 102.24, 101.88 | 0.55% |
| Gemma 4 | 80.65, 79.16, 79.85 | 78.20, 81.36, 83.83 | 1.55% |
| SmolVLM | 119.86, 124.47, 107.71 | 113.45, 147.19, 113.27 | 6.21% |

The measured means were higher with the serial cache/compact components; no speedup was established. SmolVLM varied substantially by pass in both modes. These are local amortized costs with resident models and one warmup group, not individual image completion latency or a general speed guarantee. Both final modes were executed live using the same binary.

<a id="isolated-physical-kv-clear-check"></a>
### Isolated physical KV clear check

`L2S1_FORCE_KV_CLEAR=1` is a per-engine benchmark diagnostic. It reproduces physical zeroing even when KV is already clean while retaining the same bridge and wrapper calls. On all 120 images for each model, the forced mode reports 2 physical clears / 0 skips per last native call, and the default mode 1 clear / 1 skip. Logits, probabilities, mass, ranking and decisions matched exactly in all three pairs. One-pass timings are retained as diagnostics; they do not establish a robust isolated speedup or measure the earlier wrapper-call removal.

Three real-model component tests passed, including different image, state and request order, exact equality of every decision result field, warm cache reuse, cache invalidation, malformed images, late context overflow and recovery. The same request groups also passed the live HTTP shared-metadata contract. Metal remains unverified.

`preserving-ablation-summary.json` retains the 24 screening runs, final 3-pass timings, quality/equivalence, diagnostic counter distributions and real-model tests. `ablation-config.json` freezes the eight variant flags; reproduce a model matrix with `python3 benchmarks/trashnet-vision-20260925/evaluate_ablation.py --help`. `evaluate_batch.py --candidate-execution-mode fresh --candidate-evidence-transfer compact --candidate-cache-bytes 8388608` reproduces the serial component comparison; add `--baseline-force-kv-clear` to a fresh/full candidate comparison for the clear probe. Raw responses/logs remain ignored under `results/vision-ablation-20260926/`.

<a id="component-integration-verification"></a>
### Component integration verification

After removing the separate serial preset, the sm_86 CUDA build passed 20 library tests and one CLI test. On each of Qwen3-VL, Gemma 4 E2B and SmolVLM, `real_fresh_vision_cache_and_compact_match_full_and_recover` passed exact component equality at original compute settings, and `real_vision_optimizations_preserve_full_evidence_and_recover` passed at optimized compute settings. These six live tests cover cache hits/invalidation, compact/full scoring, duplicate-image reuse, native batch counters and error recovery. CLI help exposes only the combined `--vision-optimized` preset. This functional verification used executable SHA-256 `ff49cff478fc1c9c2e6d5531a31fca4f5d44dd6747595099627ffef15c4d6b0d`; throughput was not remeasured because the inference implementation and optimized settings were unchanged. Logs remain ignored under `results/vision-components-20260926/`.

<a id="reproduce-and-inspect"></a>
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
