<a id="ollaya-comparison-and-l2s1-opportunities"></a>
# Ollaya comparison and L2S1 opportunities

[English](OLLAYA_COMPARISON.md) · [한국어](../ko/OLLAYA_COMPARISON.md) · [日本語](../ja/OLLAYA_COMPARISON.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)







This comparison references [Ollaya revision 8989f88](https://github.com/ollaya-dev/ollaya/tree/8989f88d92bd2191c548fa915b6a897db0a85f32), inspected September 26, 2026. Statements about an absent field concern that documented API snapshot, not every possible future backend.

Ollaya already offers a stronger distribution experience: a model registry, resumable model pulls, daemon lifecycle, language routing, desktop applications, MCP and TypeSafe-compatible clients. Reimplementing those conveniences is a substantial product task. Its source-reported model timings and parity measurements are not measurements performed by L2S1.

| Opportunity | Ollaya reference | L2S1 deliverable and evidence |
| --- | --- | --- |
| Image-grounded typed decisions | The documented request is state plus typed questions; this API snapshot has no image-media request contract | Native projector-backed choice/binary/ordinal image requests; actual photo demo and vision studies. The published photo example visibly contains an error. |
| Explicit refusal when input cannot fit | API §5.3 describes state truncation to model context; native /api/decide reports state_truncated, /v1 cannot report it | Context overflow is refused; model-scored responses retain truncation metadata and explicit abstention. Verify the relevant backend instead of inferring a universal quality benefit. |
| User control over computation | README describes decision inference without generated text | Direct mode or bounded native Qwen3 thinking; actual 88-token completion and token-limit failure demonstrated. No measured full-suite thinking accuracy improvement is claimed. |
| User acceptance and failure explanations | The API documents stable error codes and model temperatures | Request-local top-score/mass thresholds, requested-error threshold mapping, null abstention, and user messages alongside standard reason codes. Raw evidence remains unchanged. Actual error rates require labelled validation. |
| Reproducible quality and acceptance evidence | Published typed-decisions model scores | Complete 400-case / 2,000-judgment direct runs, hashes and downloadable per-judgment records; raw accuracy, coverage, accepted accuracy and correct/all are separate. |
| Browser-side text execution | The referenced README/API document native daemon and client/backend execution | Pages-hosted Qwen3 ONNX WebGPU demo downloads only on request and processes text locally in a worker. Actual direct inference is verified on a SwiftShader software adapter; hardware GPU speed and GGUF numerical equivalence are unverified. |

**There is no demonstrated L2S1 accuracy lead.** The same-sized Gemma 4 E2B Q8_0 checkpoint scored 54.3% in this L2S1 protocol; Ollaya reports 56.6% in its own protocol. Prompts, model execution, calibration and evaluators differ, so this is a reference comparison, not a controlled runtime experiment. Qwen3 0.6B scored 31.25% here. See the [full benchmark](TYPED_DECISIONS_BENCHMARK.md).

The strongest current positioning is controllable, inspectable image and text decisions: explicit failure boundaries, optional computation, unchanged raw score evidence, and reusable native execution. Prioritize task-specific calibration and a controlled same-model/same-request comparison before making correctness, speed or numerical-equivalence claims.

Sources: [Ollaya README](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/README.md), [API §5.3](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/api.md#53-model-specific-limits), [GGUF measurements](https://github.com/ollaya-dev/ollaya/blob/8989f88d92bd2191c548fa915b6a897db0a85f32/docs/families/llm-logits.md).
