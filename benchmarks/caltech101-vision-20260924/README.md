# Caltech-101: 30-way vision decision benchmark

This is an exploratory zero-shot image classification run of the L2S1 HTTP
vision API. It exercises the 30-option code sequence path introduced in the
preceding vision PR, beyond the one-token alphabet limit of 26 options.

## Data and protocol

- Source: [Caltech-101 Kaggle mirror](https://www.kaggle.com/datasets/imbikramsaha/caltech-101), source ZIP SHA-256 `c29bfcb9f72b1b03bdfb6b871907ba08ec436bd4049870c9e0424a6f489d4855`.
- Before inference, `scripts/prepare_caltech101_vision.py` used seed `20260924` to sample 30 of the 101 object classes, then five images per class. Only `BACKGROUND_Google` was excluded. The sample ZIP SHA-256 is `421e1a1ad05a288a57ad5e837a4f22d1941854ca561fe67a2b1be0131de4b1de`.
- The original JPEG bytes were sent one by one to `POST /v1/decisions` as `image_base64`. Each request used the same alphabetically ordered 30 options. Neither the image filename nor ground truth was sent to the model. No warmup requests were used.
- This mirror provides no held-out split for this run. These 150 images were sampled from the archive as supplied; pretraining overlap is unknown. The result measures this fixed sample and prompt, not general classification performance. A balanced random-choice baseline is 1/30 (3.33%).

`selection.json` fixes every sampled filename and image hash. `observations.jsonl`
contains each HTTP result's 30 scores, selected option, raw top option, latency,
and abstention reason. Images and model weights are not included in Git.

## Runtime

| Item | Value |
| --- | --- |
| Code | `7e33eca` (vision wide codes) |
| Model | Gemma 4 E2B IT Q8_0 GGUF, SHA-256 `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a` |
| Projector | Matching multimodal GGUF, SHA-256 `9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292` |
| GPU | NVIDIA GeForce RTX 3060, CUDA offload confirmed in every HTTP response |
| Inference | fresh request context, context 4096, batch 256, 4 CPU threads, model load mode `read`, default policy (`min_candidate_mass=0.05`, `min_top_probability=0.8`) |
| Execution | one local HTTP request at a time; startup excluded from request latency |

The GPU used 3855 MiB and reported 74 °C near the end of the run. The server
reached `/healthz` after 5014 ms. These are single-run observations.

## Results

| Metric | Result |
| --- | ---: |
| Images / classes | 150 / 30 (5 each) |
| Correct over all images | 122/150 (81.33%) |
| Coverage | 148/150 (98.67%) |
| Accuracy among accepted decisions | 122/148 (82.43%) |
| Abstentions | 2/150, both `low_top_probability` on pagoda images |
| Raw top-1, ignoring abstention policy | 122/150 (81.33%) |
| Loaded-model HTTP latency | mean 196.609 ms; p50 195.282 ms; p95 nearest rank 200.662 ms |
| Code prefix evaluations | 1 per image for all 150 requests |

The per-class counts are in `summary.json`. Three classes scored 0/5 in this
sample: `flamingo_head`, `kangaroo`, and `okapi`. The first request is included
in latency (maximum 308.351 ms). No comparison with the earlier two-class
cats/dogs score is implied: the choices and sample differ substantially.

## Reproduce

Download the Kaggle mirror as a ZIP and confirm the source hash above. From
the repository root, with a CUDA-enabled `l2s1` binary and matching GGUF files:

```sh
python3 scripts/prepare_caltech101_vision.py --source /path/to/dataset.zip --output /path/to/sample-dir
python3 scripts/benchmark_caltech101_vision.py \
  --selection /path/to/sample-dir/selection.json \
  --sample /path/to/sample-dir/sample-30x5.zip \
  --binary /path/to/l2s1 --model /path/to/gemma-4-E2B-it-Q8_0.gguf \
  --mmproj /path/to/mmproj-gemma-4-E2B-it-Q8_0.gguf \
  --output /path/to/new-result-dir
```

The benchmark refuses to reuse an existing output directory and verifies the
sample ZIP and per-image SHA-256 hashes before scoring. It writes the raw
observations, summary, and server log to the output directory. To compare a
rerun, check model/projector/source hashes and the exact class order first.
