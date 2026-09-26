<a id="브라우저-webgpu-실제-추론-데모"></a>
# 브라우저 WebGPU 실제 추론 데모

[English](../en/WEBGPU_DEMO.md) · [한국어](WEBGPU_DEMO.md) · [日本語](../ja/WEBGPU_DEMO.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)







`/webgpu`는 Qwen3 0.6B ONNX를 브라우저에서 실제 실행합니다. 저장된 예시나 합성 점수를 추론 결과로 표시하지 않습니다. 입력과 생성된 생각 토큰은 Web Worker 안에서 처리하며 서버 추론 API로 보내지 않습니다. 페이지를 열 때 모델을 자동 다운로드하지 않습니다.

<a id="사용"></a>
## 사용

1. HTTPS 사이트 또는 localhost에서 `/webgpu`를 엽니다. 페이지는 다운로드 없이 WebGPU 어댑터를 확인합니다.
2. **모델 다운로드·로드**를 누릅니다. 고정 revision의 Hugging Face 모델·토크나이저와 고정 버전의 jsDelivr 추론 런타임을 내려받습니다. 모델 본체는 q4f16 **569,789,750바이트 (543.4 MiB)**, q4 **919,096,585바이트 (876.5 MiB)**이며 런타임·메타데이터는 별도로 필요합니다.
3. 준비 완료 후 텍스트 상태를 편집하고 **이 브라우저에서 분석**을 누릅니다. 기본 예시는 텍스트에 고양이가 명시됐는지 binary 판단을 합니다. 물류 예시는 choice·binary·ordinal 세 질문을 함께 실행합니다.
4. 후보 확률·후보 코드 질량·선택 또는 보류 이유와 전체 JSON을 확인합니다. 입력과 정책을 수정하면 이전 결과는 지워집니다.

q4f16은 어댑터의 `shader-f16` 지원이 필요합니다. 미지원 어댑터에서는 q4를 선택하며 실행 provider는 그대로 WebGPU입니다. WebGPU가 없거나 어댑터를 얻지 못하면 모델을 다운로드하지 않습니다. 지원 실패를 WASM 백엔드나 서버 실행으로 대체하지 않습니다. ONNX Runtime의 호스트·제어 처리에는 WASM/CPU가 사용될 수 있으므로 모든 연산이 GPU에서 수행된다는 의미는 아닙니다.

소프트웨어 어댑터가 감지되면 화면에 명시합니다. SwiftShader 같은 소프트웨어 WebGPU 실행은 실제 GPU 하드웨어의 성능을 나타내지 않습니다. 모델 로드나 셰이더 컴파일, 추론 시간은 브라우저·장치·메모리에 따라 달라집니다.

**모델 해제**는 메모리를 해제하고 다운로드 캐시는 남깁니다. **캐시 삭제**는 이 데모 전용 `l2s1-webgpu-qwen3-da145310` 캐시만 삭제합니다. **중단·메모리 해제**는 worker를 종료하므로 다시 실행하려면 모델을 다시 로드해야 합니다. 다른 사이트나 다른 앱의 캐시는 지우지 않습니다.

<a id="점수와-생각-후-판단"></a>
## 점수와 생각 후 판단

Direct는 Qwen3 공식 chat template에서 thinking을 비활성화한 뒤 고정 답변 경계 `Answer:\n`의 실제 마지막 logits를 평가합니다. 코드 A~Z가 이 경계에서 기존 토큰을 변경하지 않고 정확히 하나의 토큰으로 이어지는지 검사하며, 중복된 토큰이나 여러 토큰 코드에는 점수를 반환하지 않습니다.

Thinking은 같은 입력에서 실제 `<think>` 토큰부터 실제 `</think>` 토큰까지 greedy 생성합니다. 최대 1~256개 생성 토큰을 설정합니다. 브라우저의 생성 토큰 수는 시작·종료 think 토큰을 포함합니다. 시작 토큰이 없으면 `unsupported_thinking`, 한도를 소진하면 `reasoning_limit`, 한도 전에 생성이 끝나면 `reasoning_incomplete` 오류를 반환합니다. 완료되지 않은 생각을 임의로 닫거나 direct 점수로 대체하지 않습니다. 완료 후 생성된 토큰 시퀀스에 고정 답변 경계를 붙여 새 prefill로 점수를 계산합니다. KV 재사용에 의한 속도 향상을 주장하지 않습니다. 생각 내용은 표시하거나 응답 JSON에 포함하지 않습니다.

`option_probability`는 후보 코드 사이의 조건부 확률입니다. `candidate_mass`는 전체 어휘 softmax에서 이 코드들이 차지하는 확률 질량입니다. 서로 다른 log-sum-exp 정규화를 사용해 작은 후보 질량의 상대 점수도 계산합니다. 후보 상대 확률이나 질량은 정답일 확률이 아닙니다.

허용 오류율 입력은 `min_top_probability = 1 - 입력 비율`로 변환하는 점수 수락 기준입니다. **실제 정답 오류율을 보장하지 않습니다.** 후보 질량 기준도 독립적으로 설정합니다. 동률은 보류하며 선택값은 `null`입니다. 사용자 지정 실패 문구는 표준 코드와 원래 오류를 유지한 채 함께 표시합니다.

입력 및 예약 토큰은 최대 1,024개이고, 요청당 질문은 1~8개, 질문당 후보는 2~26개입니다. 모든 질문은 독립적으로 평가합니다. `usage.input_tokens`는 생성 전 입력 토큰, `usage.scoring_input_tokens`는 생각 토큰과 답변 경계를 포함해 점수를 계산한 전체 prefill 길이입니다.

<a id="런타임과-측정-범위"></a>
## 런타임과 측정 범위

- npm: `@huggingface/transformers` **4.3.0**, lockfile로 고정.
- ONNX Runtime Web: **1.31.0-dev.20260914-8d85527a0**, Transformers.js 의존성으로 고정. 런타임 WASM 파일은 jsDelivr에서 받아 동일 전용 캐시에 저장합니다. 모델 가중치와 대형 WASM 파일을 Pages에 번들하지 않습니다.
- 모델: [onnx-community/Qwen3-0.6B-ONNX](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX/tree/da1453100cf3ff33ef56d17983fc7a8648706db6), revision `da1453100cf3ff33ef56d17983fc7a8648706db6`.
- 후보 점수 구현: `web/src/lib/webgpu/scoring.ts`. 실행 구현: `web/src/lib/webgpu/worker.ts`.

이 브라우저 ONNX 경로는 Rust llama.cpp GGUF 경로와 모델 파일·양자화·런타임·프롬프트가 다릅니다. 점수·선택·성능의 동등성을 주장하지 않습니다. 기존 typed-decisions 표는 RTX 3080 GGUF 측정이며 이 페이지의 WebGPU 측정이 아닙니다. 모델 라이선스는 Apache 2.0이며 코드·의존성 고지는 [웹 의존성 라이선스](../../web/THIRD_PARTY_LICENSES.txt)에 있습니다.

<a id="실제-실행-검증"></a>
## 실제 실행 검증

2026-09-26에 빌드된 페이지를 Chromium에서 실제 모델로 확인했습니다. 검증 어댑터는 **Google SwiftShader 소프트웨어 WebGPU**, 모델 형식은 q4입니다. 하드웨어 GPU 성능이나 q4f16 실행 검증은 아닙니다.

- `The animal is a cat.`의 직접 판단은 `true`를 선택했습니다. 후보 상대 확률은 **0.9732201940**, 전체 어휘의 후보 코드 질량은 **0.9212860617**이며, 실제 logits와 토큰 id도 반환했습니다. 생성 생각 토큰은 0개였습니다.
- 생각 한도를 1개로 설정하면 `reasoning_limit`이 발생하고 **WebGPU 지정 오류 문구**가 함께 표시됐습니다. 이 요청에는 점수가 반환되지 않았습니다.
- 같은 입력을 다시 direct로 실행한 후보 확률은 이전 결과와 차이가 **0**이었습니다. 모델 해제도 성공했습니다.
- 브라우저 오류와 추론 서버 요청은 모두 **0건**이었습니다. 모델 파일·런타임 다운로드는 별도입니다.
- 별도 실제 worker 실험에서 48개까지 생성 진행을 관찰했으나, 느린 소프트웨어 실행으로 중단했습니다. **WebGPU에서 생각 완료 후 판단 성공은 아직 검증하지 않았습니다.** 네이티브 CPU/CUDA의 완료 검증은 [별도 추론 기록](REASONING.md)에 있습니다.

이 한 입력의 검증은 품질 벤치마크나 성능 우위 증명이 아닙니다. 소프트웨어 어댑터의 첫 직접 판단은 모델 로드를 제외하고 63.54초였습니다. 로컬 실행 응답과 화면 증거는 gitignore된 `results/typed-decisions-20260926/webgpu-ui-real.json`, `webgpu-ui-direct.png`, `webgpu-ui-limit.png`에 보관했습니다.

<a id="로컬-개발"></a>
## 로컬 개발

```sh
cd web
npm ci
npm run dev
```

`http://127.0.0.1:5173/webgpu`를 엽니다. 별도 L2S1 추론 서버는 필요하지 않습니다. WebGPU 모델을 실행하려면 지원 브라우저·어댑터와 모델을 내려받을 네트워크가 필요합니다.

구현 확인에는 [Hugging Face 공식 Qwen3 WebGPU worker](https://github.com/huggingface/transformers.js-examples/blob/main/qwen3-webgpu/src/worker.js), [WebGPU 사용 문서](https://huggingface.co/docs/transformers.js/guides/webgpu), [양자화 문서](https://huggingface.co/docs/transformers.js/guides/dtypes), 고정 npm 패키지의 모델 forward·generation·Tensor·ONNX backend 소스를 사용했습니다.
