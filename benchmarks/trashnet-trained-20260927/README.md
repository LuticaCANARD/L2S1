# TrashNet classifier improvement, 2026-09-27

The visual example now compares the historical zero-shot L2S1 results with a
**separate supervised image classifier**: frozen SigLIP2 features and a trained
multinomial logistic classifier. This is not a modification to L2S1's GGUF
scoring, a new L2S1 output head, or a LoRA improvement to Qwen.

| Evaluation | Original Qwen3-VL 2B raw top-1 | Trained classifier raw top-1 |
| --- | ---: | ---: |
| Same historical 120 photos | 95/120 (79.2%) | **113/120 (94.2%)** |
| Trash class within those 120 | 0/20 | **17/20** |
| Separate final test | Not rerun | **466/481 (96.9%)** |

The historical Qwen acceptance policy accepted 115 answers, of which 91 were
correct: correct/all 75.8%, coverage 95.8%, correct/accepted 79.1%. The trained
classifier always selects the highest scoring class (100% coverage). Its
scores are not calibrated probabilities of correctness, and it has no
full-vocabulary `candidate_mass`. Compare raw top-1 with raw top-1.

On the same 120 images, 20 Qwen raw errors became correct and two previously
correct answers became wrong: net +18 correct, +15.0 percentage points. All
seven remaining errors can be inspected in the gallery. The independent test
has 15 errors, also visible with the **Independent test → Wrong** filters.

## Split and model selection

- Original pinned TrashNet archive SHA-256:
  `0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2`.
- 2,527 source images, six classes. Historical gallery: 120 images excluded
  from both training and validation. Four perceptually similar neighbors were
  excluded as well.
- DCT perceptual hashes (63 AC bits, Hamming distance ≤ 6) and exact SHA-256
  duplicates were grouped before splitting; transitive groups stay together.
  This is a conservative automated similarity check, not proof that all
  photographs of the same physical item have been identified.
- Seed 20260927. Stratified group splits: **1,441 train / 481 validation /
  481 final test**. Every path, hash, group and membership is in `split.json`.
- [google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224),
  revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, frozen vision encoder.
  Original and horizontally flipped image features are averaged, then L2
  normalized. CUDA float32 on RTX 3080. No public TrashNet-finetuned checkpoint
  was used. The generic backbone's pretraining-image overlap is not audited.
- Eight fixed candidates: class-balanced logistic regression and RBF SVM,
  each with C ∈ {0.1, 1, 10, 100}. Select by validation **balanced accuracy**,
  using probability argmax consistently. No test/gallery tuning or refit.
- Selected `logistic_C100`: validation 459/481 (95.4%), balanced accuracy
  95.56%. The SVM C10 had slightly higher ordinary validation accuracy but
  lower balanced accuracy; it was not selected. Final test balanced accuracy:
  **96.35%**.

These are within-dataset results on controlled-background photographs, not
deployment accuracy or a same-training-budget comparison with the zero-shot
baselines. Historical examples have been inspected before; the separately
reserved 481-image test was not used to choose the classifier.

## Per-class final test

| Class | Correct / total |
| --- | ---: |
| cardboard | 75/77 |
| glass | 95/95 |
| metal | 78/78 |
| paper | 109/115 |
| plastic | 88/93 |
| trash | 21/23 |

## Serving and detection

The local HTTP server was run on all historical 120 JPEGs, submitting image
bytes without labels or filenames. Predictions matched the recorded evaluation
on **120/120**. Observed HTTP p50/p95: **13.8/25.4 ms**, including decoding,
preprocessing, two image views and the classifier; model load excluded and first
request included. This is a separate protocol from the older Qwen timings,
not a controlled speedup claim. `serving-verification.json` records that check.
`feature_ms_per_image` is batched feature-extraction throughput, never displayed
as individual request latency.

The detection page also offers an **experimental OWL-ViT comparison**, using a
fixed set of 15 waste-related phrases and class-agnostic overlap suppression.
It runs real Q8 inference on the same 12 photos as DETR. It still misidentifies
materials and objects. Its WASM browser test failed after downloading the model; browser execution is disabled and the UI exposes CPU recordings only. **No detector accuracy gain or localization mAP is
claimed**. TrashNet does not provide bounding-box ground truth here. Whole-image
material predictions are shown separately and are not assigned to each box.

## Reproduce

Package versions are recorded in `environment.json`. Use a Python 3.12 virtual
environment with those versions; the feature extraction run requires CUDA.
From the repository root:

```sh
python scripts/train_trashnet_classifier.py \
  --archive /path/to/dataset-resized.zip \
  --output results/trashnet-trained-new
python scripts/serve_trashnet_classifier.py \
  --model-dir results/trashnet-trained-new
```

In another terminal, export the audited results and run the UI:

```sh
python scripts/export_trashnet_improvement.py \
  --archive /path/to/dataset-resized.zip \
  --run results/trashnet-trained-new \
  --live-url http://127.0.0.1:8766
cd web
npm run dev -- --port 5173
```

Open `http://127.0.0.1:5173/trashnet?lang=ko`. The static recordings work without
the classifier server; new image classification uses the loopback server via
Vite's `/vision-inference` proxy. Production static hosting does not include
that server. Detector inference runs in browser WASM.

The classifier checkpoint and downloaded backbone remain in ignored `results/`;
retraining produces the checkpoint. The original baseline data is unchanged.
Training, validation and test observations plus the fixed selection grid are
included here for audit. No model weights are committed.

## Final verification scope

Type checking, lint and the production build pass. In the isolated PR checkout,
the full web suite passes 74 tests, with one opt-in external-model test skipped.
All 10 focused non-network visual-example checks pass. Real classifier uploads, a deliberately misleading upload filename,
holdout error filters and mobile dark mode pass against the running local server.

The DETR browser test successfully detected bottles twice in an earlier run,
but its final repeat failed during browser model startup; the root cause was not
established. Thus browser detector execution is not consistently verified in this
environment. Recorded detection inspection and the local trained classifier are
available independently. OWL-ViT browser inference is disabled after its own
failed check. These detector limitations do not affect the classifier evaluation.
