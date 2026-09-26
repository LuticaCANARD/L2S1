<a id="브라우저-webgpu-실제-추론-데모"></a>
# Actual browser WebGPU inference demo

[English](WEBGPU_DEMO.md) · [한국어](../ko/WEBGPU_DEMO.md) · [日本語](../ja/WEBGPU_DEMO.md)

[English index](README.md) · [한국어 색인](../ko/README.md) · [日本語索引](../ja/README.md)

`/webgpu` runs Qwen3 0.6B ONNX in the browser. It does not present stored examples or synthetic scores as inference results. Inputs and generated thought tokens are processed inside a Web Worker and are not sent to a server inference API. Opening the page does not automatically download the model.

<a id="사용"></a>
## Usage

1. Open `/webgpu` on HTTPS or localhost. The page checks for a WebGPU adapter without downloading anything.
2. Click **Download and load model**. This downloads a pinned-revision Hugging Face model/tokenizer and a pinned-version jsDelivr inference runtime. Model weights are **569,789,750 bytes (543.4 MiB)** for q4f16 or **919,096,585 bytes (876.5 MiB)** for q4; runtime and metadata are additional.
3. When ready, edit the text state and click **Analyze in this browser**. The default example makes a binary decision on whether the text explicitly mentions a cat. The logistics example runs choice, binary, and ordinal questions together.
4. Inspect candidate probabilities, candidate-code mass, selection or abstention reasons, and full JSON. Changing input or policy clears previous results.

q4f16 requires adapter `shader-f16` support. An unsupported adapter selects q4 while retaining WebGPU as the execution provider. If WebGPU or an adapter is unavailable, the model is not downloaded. Unsupported execution is not replaced by WASM or a server backend. ONNX Runtime may use WASM/CPU for host and control work, so this does not mean every operation runs on the GPU.

Detected software adapters are explicitly labeled. Software WebGPU such as SwiftShader does not represent actual GPU hardware performance. Model loading, shader compilation, and inference times depend on browser, device, and memory.

**Unload model** releases memory but keeps the download cache. **Clear cache** removes only this demo's `l2s1-webgpu-qwen3-da145310` cache. **Stop and release memory** terminates the worker; load the model again before another run. Other sites' or applications' caches are not removed.

<a id="점수와-생각-후-판단"></a>
## Scores and decisions after thinking

Direct disables thinking in the official Qwen3 chat template, then evaluates actual final logits at the fixed `Answer:\n` boundary. It checks that A–Z codes extend this boundary by exactly one token without changing prior tokens; duplicate tokens or multi-token codes do not receive scores.

Thinking greedily generates from the actual `<think>` token through the actual `</think>` token on the same input. The limit is 1–256 generated tokens. Browser token counts include the opening and closing think tokens. Missing opening tokens yield `unsupported_thinking`; exhaustion yields `reasoning_limit`; generation ending before the limit yields `reasoning_incomplete`. Unfinished thinking is not artificially closed or replaced with direct scores. After completion, the fixed answer boundary is appended to the generated token sequence and scored with a new prefill. No KV-reuse speedup is claimed. Thought content is neither displayed nor included in response JSON.

`option_probability` is conditional probability among candidate codes. `candidate_mass` is those codes' probability mass in the full-vocabulary softmax. Separate log-sum-exp normalizations allow relative scores even for small candidate mass. Relative probabilities and mass are not probabilities of correctness.

The allowed error-rate input becomes the score acceptance criterion `min_top_probability = 1 - input_ratio`. **It does not guarantee an actual correctness error rate.** Candidate mass is configured independently. Ties abstain with a `null` selection. Custom failure text is displayed while retaining standard codes and original errors.

Input plus reserved tokens is limited to 1,024, with 1–8 questions per request and 2–26 candidates per question. Questions are evaluated independently. `usage.input_tokens` counts input before generation; `usage.scoring_input_tokens` counts the full scoring prefill including thought tokens and the answer boundary.

<a id="런타임과-측정-범위"></a>
## Runtime and measurement scope

- npm: `@huggingface/transformers` **4.3.0**, pinned by lockfile.
- ONNX Runtime Web: **1.31.0-dev.20260914-8d85527a0**, pinned as a Transformers.js dependency. Runtime WASM files come from jsDelivr and share the dedicated cache. Model weights and large WASM files are not bundled into Pages.
- Model: [onnx-community/Qwen3-0.6B-ONNX](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/tree/da1453100cf3ff33ef56d17983fc7a8648706db6), revision `da1453100cf3ff33ef56d17983fc7a8648706db6`.
- Candidate scoring: `web/src/lib/webgpu/scoring.ts`. Execution: `web/src/lib/webgpu/worker.ts`.

This browser ONNX path differs from Rust llama.cpp GGUF in model files, quantization, runtime, and prompts. No score, selection, or performance equivalence is claimed. The existing typed-decisions table is measured on RTX 3080 GGUF, not this page's WebGPU execution. The model license is Apache 2.0; code/dependency notices are in [web dependency licenses](../../web/THIRD_PARTY_LICENSES.txt).

<a id="실제-실행-검증"></a>
## Actual execution verification

The September 26, 2026 build was tested with the real model in Chromium using **Google SwiftShader software WebGPU**, q4. This is not hardware GPU performance or q4f16 execution validation.

- Direct scoring of `The animal is a cat.` selected `true`, with candidate probability **0.9732201940** and full-vocabulary candidate-code mass **0.9212860617**. Actual logits and token IDs were returned, with 0 generated thought tokens.
- A 1-token thinking limit returned `reasoning_limit` with the custom **WebGPU 지정 오류 문구** and no scores.
- Repeating direct on the same input changed probability by **0**. Model unloading also succeeded.
- Browser errors and inference-server requests were both **0**; model/runtime downloads are separate.
- A separate real-worker experiment observed generation through 48 tokens before stopping slow software execution. **Successful scoring after completed WebGPU thinking is still unverified.** Native CPU/CUDA completion is recorded [separately](REASONING.md).

This single input is not a quality benchmark or proof of superior performance. The first software-adapter direct decision took 63.54 seconds excluding model load. Local response and screenshots are gitignored under `results/typed-decisions-20260926/webgpu-ui-real.json`, `webgpu-ui-direct.png`, and `webgpu-ui-limit.png`.

<a id="로컬-개발"></a>
## Local development

```sh
cd web
npm ci
npm run dev
```

Open `http://127.0.0.1:5173/webgpu`. No separate L2S1 inference server is needed. Executing the WebGPU model requires a supported browser/adapter and network access for downloads.

Implementation references were the [official Hugging Face Qwen3 WebGPU worker](https://github.com/huggingface/transformers.js-examples/blob/main/qwen3-webgpu/src/worker.js), [WebGPU guide](https://huggingface.co/docs/transformers.js/guides/webgpu), [quantization guide](https://huggingface.co/docs/transformers.js/guides/dtypes), and the pinned npm package's model forward, generation, Tensor, and ONNX backend sources.
