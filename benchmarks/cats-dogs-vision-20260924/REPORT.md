# Kaggle Cats and Dogs validation: L2S1 vision HTTP

Source: https://www.kaggle.com/datasets/marquis03/cats-and-dogs

- Source archive SHA-256: `47ac0f845f65b5f2b9cb8a184d9d17d4345f8bd77bf1ed63e79b426d90c3b99b`
- Validation: 70 original JPEGs, 24 cats and 46 dogs; exact image-hash overlap with the supplied training split: 0.
- Model: gemma4 E2B Q8_0; offload: NVIDIA GeForce RTX 3060; matching multimodal projector.
- Binary SHA-256: `586574207661e23078899973320c60a139d8f740fb75cd62709f22661fc7f53e`
- Model SHA-256: `996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a`
- Projector SHA-256: `9406f99c16d68cda4f1f0552192dcc99021ea1fc6d2fd50b1dc3ccf30d04b292`
- Protocol: one image per serial loopback HTTP request; fixed prompt asking for the main animal, without file name, path, label, examples or training rows; context 2048, batch 256, 4 CPU threads, default decision thresholds. Loaded-model latency includes the first request and excludes model startup. Original order is cat then dog; second run reverses the options.

| Option order | HTTP 200 | Accepted | Accepted correct / all | Accepted accuracy | Coverage | Raw top-1 correct / all | Mean | p50 | p95 nearest rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cat, dog | 70/70 | 70/70 | 69/70 (98.57%) | 98.57% | 100% | 69/70 | 101.795 ms | 99.176 ms | 114.351 ms |
| dog, cat | 70/70 | 70/70 | 69/70 (98.57%) | 98.57% | 100% | 69/70 | 101.392 ms | 98.745 ms | 113.744 ms |

Confusion in both runs: cats 24/24 correct; dogs 45/46 correct, 1 classified cat. No abstentions or HTTP errors. The same labeled Japanese Chin photo is the only error in both runs. Its conditional candidate probability is not a calibrated probability of being correct.

An always-dog classifier would score 46/70 (65.71%) because of class imbalance. The descriptive 95% Wilson interval for 69/70 is approximately 92.34% to 99.75%, conditional on this sample and protocol. Exact image-hash overlap was absent, but visual near-duplicates and model pretraining contamination were not assessed. This is a small, single-dataset evaluation, not a general image-classification accuracy claim.

This directory includes both summary JSON files, per-image observation records,
and the historical `evaluate-order-check.py` evaluator. The source image archive
and server logs remain local; no image bytes or model weights are published.
The evaluator sends the `image_base64` field accepted by the recorded binary.
Current L2S1 HTTP releases use the versioned `media` request contract, so the
script needs a request-shape update before replay against current `main`.
The binary SHA-256 above identifies the version that produced these results.
The GPU process was stopped after each run.
