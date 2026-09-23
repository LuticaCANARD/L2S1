# Tested model specifications

Extracted on 2026-09-21 from the five local GGUF files. Parameter counts are the sum of stored tensor dimensions, including embeddings. File sizes include metadata. No weights are included in this report; exact checkpoint revisions and hashes are in [VERIFICATION.md](VERIFICATION.md). Machine-readable details are in [MODEL_SPECS.json](MODEL_SPECS.json).

## Local checkpoint inventory

| Model | Stored parameters | Quantization | File size (MiB) | GGUF context tokens | License |
| --- | ---: | --- | ---: | ---: | --- |
| Qwen3 0.6B | 596,049,920 | Q8_0 | 609.82 | 40,960 | Apache-2.0 |
| SmolLM2 135M Instruct | 134,515,008 | Q8_0 | 138.10 | 8,192 | Apache-2.0 |
| Gemma 3 1B IT | 999,885,952 | Q8_0 | 1,019.77 | 32,768 | Gemma terms |
| Gemma 4 E2B IT | 4,647,450,147 | Q8_0 | 4,737.37 | 131,072 | Apache-2.0 |
| TinyLlama 1.1B Chat v1.0 | 1,100,048,384 | Q4_K_M | 637.81 | 2,048 | Apache-2.0 |

MiB means 1,048,576 bytes. Quantization is the checkpoint format label, not a claim that every tensor has that precision. All five files are GGUF v3 and contain a chat template. Context metadata is not evidence of successful inference at that length; all recorded integration/performance runs used 2,048 tokens.

## Architecture extracted from GGUF

| Model | Architecture | Layers | Hidden width | Attention heads / KV heads | Vocabulary | FFN width |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Qwen3 0.6B | `qwen3` | 28 | 1,024 | 16 / 8 | 151,936 | 3,072 |
| SmolLM2 135M Instruct | `llama` | 30 | 576 | 9 / 3 | 49,152 | 1,536 |
| Gemma 3 1B IT | `gemma3` | 26 | 1,152 | 4 / 1 | 262,144 | 6,912 |
| Gemma 4 E2B IT | `gemma4` | 35 | 1,536 | 8 / 1 | 262,144 | 6,144 (first 15); 12,288 (last 20) |
| TinyLlama 1.1B Chat v1.0 | `llama` | 22 | 2,048 | 32 / 4 | 32,000 | 5,632 |

Gemma 3 and Gemma 4 use 512-token sliding windows alongside global attention. The Gemma 4 file also declares 20 shared-KV layers and per-layer input embeddings of width 256. SmolLM2 and TinyLlama share the GGUF architecture label `llama`, but have different dimensions, vocabularies, and templates.

## Model-card distinctions

- **Gemma 4 E2B:** Google reports 2.3B effective parameters and 5.1B with embeddings for the full model, including separate vision (~150M) and audio (~300M) encoders. This text GGUF contains 4,647,450,147 stored parameters. The full model supports text, images, and audio; this repository tests only text and does not load multimodal projectors or an MTP drafter. The difference in scope is consistent with the omitted encoders; no exact encoder parameter audit was performed. [Official model card](https://ai.google.dev/gemma/docs/core/model_card_4).
- **Qwen3:** the model card lists a 32,768-token context, while the current upstream configuration and the tested GGUF declare 40,960. Both figures are retained rather than treating the larger number as a tested guarantee. The repository uses its non-thinking prompt profile. [Model card](https://huggingface.co/Qwen/Qwen3-0.6B), [configuration](https://huggingface.co/Qwen/Qwen3-0.6B/raw/main/config.json).
- **SmolLM2:** the upstream configuration confirms 8,192 positions, 30 layers, and hidden width 576. Its model card identifies an English text model and Apache-2.0. [Model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct), [configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/config.json).
- **TinyLlama:** the official chat model is tagged English and Apache-2.0. The local file is a community Q4_K_M conversion. [Model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0).
- **Gemma 3:** the 1B checkpoint has a 32K context and uses the separate Gemma terms. Do not apply Gemma 4's Apache license to it. [Model card](https://huggingface.co/google/gemma-3-1b-it), [Gemma terms](https://ai.google.dev/gemma/terms), [Gemma 4 Apache license](https://ai.google.dev/gemma/apache_2).

## Observed allocation sizes, not minimum hardware requirements

The prior RTX 3080 CLI runs used context 2,048 and batch 256. Values below are llama.cpp-reported buffer allocations in MiB. CPU-mapped sizes are virtual mappings, not resident RAM; these values do not measure total RSS or total VRAM and must not be added to claim whole-process usage. Driver/runtime overhead, tokenizer storage, and other allocations are excluded. Larger contexts or batches can increase memory use.

| Model | CUDA model | CUDA KV total | CUDA compute | CPU-mapped model |
| --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B | 604.15 | 224.00 | 149.38 | 157.65 |
| SmolLM2 135M Instruct | 136.49 | 45.00 | 48.56 | 28.69 |
| Gemma 3 1B IT | 1,013.61 | 52.00 | 257.12 | 306.00 |
| Gemma 4 E2B IT | 2,342.30 | 48.00 | 275.26 | 2,788.00 |
| TinyLlama 1.1B Chat v1.0 | 601.02 | 44.00 | 74.50 | 35.16 |

Gemma 3 and TinyLlama failed the CUDA batch-consistency check; their fixed-batch inference runs completed. SmolLM2 and TinyLlama abstained on all warehouse decisions at the default policy. These specifications and a single fixture do not establish model quality or an accuracy ranking. See [the full verification record](VERIFICATION.md) for timings and limitations.
