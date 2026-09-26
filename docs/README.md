# Documentation

[English introduction](../README.md) · [한국어 소개](../README.ko.md)

Start with the root README to build L2S1 and run a first decision. This directory contains the detailed English guides, execution contracts, and evaluation methods.

## Usage and integration

| Document | Contents |
| --- | --- |
| [Guide](GUIDE.md) | Build requirements, decision schema, scores, CLI, Rust, HTTP, images, and backend setup |
| [AI agent integration](AGENT_INTEGRATION.md) | Portable skill, stdio MCP, request validation, documentation resources, and resident inference |
| [Model interchangeability](MODEL_INTERCHANGEABILITY.md) | Model identity, request preflight, calibration, diagnostics, worker ownership, and memory controls |
| [Verification](VERIFICATION.md) | Build and model-dependent validation commands |
| [Licensing](LICENSING.md) | Source, dependency, and model license boundaries |

## Execution and specialization

| Document | Contents |
| --- | --- |
| [Parallel execution](PARALLEL_EXECUTION.md) | Supported text/image batching, dynamic context, projector reuse, and vision throughput profile |
| [Prefix algorithm](SEMIF_ALGORITHM.md) | Prefix preparation and reuse |
| [Decision fine-tuning](DECISION_FINETUNE.md) | LoRA workflow and its recorded evaluation |
| [Output heads](OUTPUT_HEAD.md) | Task-specific heads and artifact bindings |

Parallel execution and optimized vision are supported features within their documented model, device, and layout limits. The guides describe measured numerical differences and validation procedures for each execution mode.

## Evaluation methods and results

| Document | Contents |
| --- | --- |
| [Recorded model results](MODEL_RESULTS.md) | Checkpoint comparisons, scope, and measurement limits |
| [Synthetic benchmark](BENCHMARK.md) | Correctness, abstention, consistency, and latency on rule fixtures |
| [JevBench](JEVBENCH.md) | Public task mapping, upstream scoring, and acceptance metrics |
| [Intent classification](INTENT_BENCHMARK.md) | Wide answer codes, BANKING77, and MASSIVE Korean |
| [AG News](KAGGLE_BENCHMARK.md) | Frozen classification protocol |
| [Laya/Jev tasks and caching](LAYA_BENCHMARK.md) | Task conversion and CPU/GPU cache checks |
| [Vision benchmark](VISION_BENCHMARK.md) | Direct image inference and timing scope |

Task and hardware reports remain with their artifacts under [`benchmarks/`](../benchmarks):

- [Caltech-101](../benchmarks/caltech101-vision-20260924/README.md)
- [Cats and dogs](../benchmarks/cats-dogs-vision-20260924/REPORT.md)
- [TrashNet and vision throughput](../benchmarks/trashnet-vision-20260925/REPORT.md)
- [Generic GGUF CUDA smoke](../benchmarks/gguf-cuda-20260925/README.md)
- [Bonsai state restoration](../benchmarks/bonsai-state-restore-20260925/REPORT.md)
- [Shared-state caching](../benchmarks/shared-state-cache-20260925/REPORT.md)
- [Build caching](../benchmarks/build-cache-20260925/REPORT.md)

## Source and tooling

- [Architecture and component map](GUIDE.md#architecture)
- [Native llama.cpp dependency](../crates/l2s1-llama-sys/README.md)
- [Dataset and report tools](../crates/l2s1-tools/README.md)
- [Request and integration examples](../examples)
- [Documentation website](../web/README.md)

Code and dependency notices stay at the repository root in [LICENSE](../LICENSE) and [THIRD_PARTY_LICENSES.txt](../THIRD_PARTY_LICENSES.txt). Model weights are supplied separately and keep their own terms.
