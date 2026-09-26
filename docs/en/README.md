# Documentation index

[English](../en/README.md) · [한국어](../ko/README.md) · [日本語](../ja/README.md)

Start with the [English README](../../README.md) to build L2S1 and run a first decision. This index covers all public supplementary guides, package documentation, and benchmark reports. Every listed guide, package document, and benchmark report has a complete English, Korean, and Japanese body. Use the language links in a document to switch to the same document in another language. Code, commands, identifiers, and measurements retain their source values.

## Usage and integration

| Document | Contents |
| --- | --- |
| [Guide](GUIDE.md) | Build, schema, scores, CLI, Rust, HTTP, images, backends |
| [TypeScript library](typescript/README.md) | Resident model process, HTTP client, packaging, verification |
| [Python SDK](python/README.md) | Typed async calls, reusable decisions, wheels, shared Rust runtimes |
| [Batching API review](BATCHING_API_REVIEW.md) | Repeated state inputs, native batch boundaries and HTTP integration |
| [Release pipeline](RELEASE_PIPELINE.md) | GitHub Release, npm, PyPI and native Cargo publication |
| [AI agent integration](AGENT_INTEGRATION.md) | Portable skill, stdio MCP, request validation, resident inference |
| [Browser WebGPU demo](WEBGPU_DEMO.md) | Local Qwen3 ONNX inference, WebGPU, bounded thinking, policy |
| [Image and text demo](IMAGE_DEMO.md) | Recorded responses, policy, failure explanations, local server |
| [Pages deployment](PAGES_DEPLOYMENT.md) | Cloudflare Pages and n2s1.luticalab.net DNS |
| [Model interchangeability](MODEL_INTERCHANGEABILITY.md) | Model identity, preflight, calibration, diagnostics, workers, memory |
| [Verification](VERIFICATION.md) | Build and model-dependent validation commands |
| [Licensing](LICENSING.md) | Source, dependency, and model license boundaries |

## Execution and specialization

| Document | Contents |
| --- | --- |
| [Reasoning modes](REASONING.md) | Direct/native thinking, token bounds, runtime scope |
| [Parallel execution](PARALLEL_EXECUTION.md) | Text/image batching, dynamic context, projector reuse, vision profile |
| [Prefix algorithm](SEMIF_ALGORITHM.md) | Prefix preparation and reuse |
| [Decision fine-tuning](DECISION_FINETUNE.md) | LoRA workflow and recorded evaluation |
| [Output heads](OUTPUT_HEAD.md) | Task-specific heads and artifact bindings |

## Evaluation methods and results

| Document | Contents |
| --- | --- |
| [Recorded model results](MODEL_RESULTS.md) | Checkpoint comparisons, scope, measurement limits |
| [Synthetic benchmark](BENCHMARK.md) | Correctness, abstention, consistency, latency on rule fixtures |
| [JevBench](JEVBENCH.md) | Public task mapping, upstream scoring, acceptance metrics |
| [Ollaya comparison](OLLAYA_COMPARISON.md) | Evidence-based differences and remaining quality/product gaps |
| [typed-decisions](TYPED_DECISIONS_BENCHMARK.md) | Complete 2,000-judgment test, calibration, measured protocol |
| [Intent classification](INTENT_BENCHMARK.md) | Wide answer codes, BANKING77, MASSIVE Korean |
| [AG News](KAGGLE_BENCHMARK.md) | Frozen classification protocol |
| [Laya/Jev tasks and caching](LAYA_BENCHMARK.md) | Task conversion and CPU/GPU cache checks |
| [Vision benchmark](VISION_BENCHMARK.md) | Direct image inference and timing scope |

## Task and hardware reports

| Document | Contents |
| --- | --- |
| [Caltech-101](benchmarks/caltech101-vision-20260924/README.md) | Still-image classification |
| [Cats and dogs](benchmarks/cats-dogs-vision-20260924/REPORT.md) | Binary image classification |
| [TrashNet and vision throughput](benchmarks/trashnet-vision-20260925/REPORT.md) | Image classification and execution modes |
| [Generic GGUF CUDA smoke](benchmarks/gguf-cuda-20260925/README.md) | CUDA model smoke measurements |
| [Bonsai state restoration](benchmarks/bonsai-state-restore-20260925/REPORT.md) | State-restore measurements |
| [Shared-state caching](benchmarks/shared-state-cache-20260925/REPORT.md) | Cache measurements |
| [Build caching](benchmarks/build-cache-20260925/REPORT.md) | Native build cache measurements |
| [typed-decisions artifacts](benchmarks/typed-decisions-20260926/README.md) | Checked-in summaries, records, reproducibility metadata |

## Source and tooling

| Document | Contents |
| --- | --- |
| [Architecture and component map](GUIDE.md#architecture) | Source layout |
| [Native llama.cpp dependency](crates/l2s1-llama-sys/README.md) | Pinned native runtime and bridge |
| [Dataset and report tools](crates/l2s1-tools/README.md) | Dataset preparation, evaluation, recounting |
| [Warehouse request example](../../examples/warehouse.json) | Binary, choice, ordinal requests |
| [Documentation website](web/README.md) | Svelte site, demos, public document export |
| [npm publishing](typescript/PUBLISHING.md) | Platform runtime packaging, publication conditions and validation scope |
| [Experiment tool notices](crates/l2s1-tools/NOTICE.md) | JevBench and Laya attribution and dataset boundaries |
| [L2S1 skill](skills/l2s1/SKILL.md) | Integration instructions and evidence handling |
| [Skill request reference](skills/l2s1/references/decisions.md) | Request examples, typed results, HTTP and MCP fields |
| [Skill interface reference](skills/l2s1/references/interfaces.md) | Build, CLI, HTTP, MCP and Rust integration |

Source and dependency notices: [LICENSE](../../LICENSE), [THIRD_PARTY_LICENSES.txt](../../THIRD_PARTY_LICENSES.txt). Model weights are supplied separately and keep their own terms. These reports describe recorded experiments within their stated scope; report accuracy, accepted accuracy, and coverage separately.
