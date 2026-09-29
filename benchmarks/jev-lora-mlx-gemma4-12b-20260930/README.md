# Jev LoRA on Apple Silicon (MLX): Gemma 4 12B IT

`l2s1-train run --backend mlx` on one Apple Silicon Mac (36 GiB unified memory),
same pinned data, split, seed, loss and schedule as the CUDA pilot
([jev-lora-20260927](../jev-lora-20260927/README.md)): 120 cases / 600 typed
decisions for training, the separate 400-case / 2,000-decision test split for
evaluation. Specialist on seen workflows, scored against synthetic teacher labels.

- Checkpoint: `google/gemma-4-12B-it` @ `707f0a3b` (`gemma4_unified`), mlx-lm
  main (`0.31.4.dev132`), MLX 0.32.3. 4-bit affine base (group 64), LoRA rank 8 /
  alpha 16 on Q/V (48 q_proj + 40 v_proj; global layers share K/V), gradient
  checkpointing. Training: 610 s, peak Metal 12.8 GiB.
- Native: HF → Q8_0 GGUF with the pinned llama.cpp converter; PEFT-format adapter
  → `convert_lora_to_gguf.py` (176 tensors, 10 MB) → evaluated with `--metal --lora`.
  Token export used the vocab-only loader.

| metric | base | adapter |
| --- | ---: | ---: |
| raw accuracy | 0.6865 | 0.7285 |
| coverage | 0.9395 | 0.3100 |
| accepted accuracy | 0.6998 | 0.9629 |
| correct / all | 0.6575 | 0.2985 |
| soft KL | 2.6121 | 0.2083 |
| soft Brier | 0.3907 | 0.0988 |
| score MAE | 0.4845 | 0.2872 |
| p50 latency (ms) | 1784 | 1828 |

Raw accuracy by type (base → adapter): choice 0.590 → 0.703, noul 0.797 → 0.807,
score 0.676 → 0.689. As in the CUDA pilot, the adapter trades coverage for
accepted accuracy under the fixed 0.8 / 0.05 policy; these probabilities are not
claimed calibrated. One run, one seed. Checkpoints, GGUFs and the adapter are not
published.
