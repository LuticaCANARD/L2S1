<a id="generic-gguf-cuda-path-two-model-smoke-measurement"></a>
# 일반 GGUF CUDA 경로: 2가지 모델 스모크 측정

[English](../../../en/benchmarks/gguf-cuda-20260925/README.md) · [한국어](README.md) · [日本語](../../../ja/benchmarks/gguf-cuda-20260925/README.md)

[English index](../../../en/README.md) · [한국어 색인](../../README.md) · [日本語索引](../../../ja/README.md)

이 실행에서는 기존 llama.cpp CUDA 실행 파일이 두 개의 서로 다른 GGUF 모델 계열에서 형식화된 판단을 생성할 수 있는지 확인합니다. [`examples/warehouse.json`](../../../../examples/warehouse.json)(SHA-256 `68e3bffae421112de90658c62c5d52293e53b719f7610abb5377b9ad4bbd4944`)의 세 가지 판단에 대해 워밍업 없이 모델당 하나의 호출입니다. 픽스처의 예상 값은 `chilled`, `true` 및 `high`입니다.

<a id="runtime-and-method"></a>
## 런타임 및 방법

- 엔비디아 지포스 RTX 3060(12 GB); 각 응답은 이 장치를 CUDA 오프로드 대상으로 보고합니다.
- L2S1 CUDA 실행 파일이 2026-09-24, SHA-256 `b99f8f1e40d6ae542ed780eb3868e72a1c93f735ff488663b073c2335578af62`의 서버에 복사되었습니다. 서버 복사본에는 Git 메타데이터가 없으므로 이는 이 PR 커밋의 빌드가 아닌 기존 CUDA 경로를 확인하는 것입니다. 두 응답 모두 컴파일된 런타임 SHA-256 `771abedbe730b88af57cc1bc35543c4353b30382de7bf9c475a78e9814e66043`와 로드된 런타임 SHA-256 `ae19e1bb27af03e2057408582ca17f35a748b7fec5aab75db5bef03763a026d8`를 기록합니다.
- 새로 실행, 컨텍스트 2048, 배치/ubatch 256, 4 CPU 스레드, FlashAttention 꺼짐, 기본 판단 정책. 모델 간에 프롬프트 또는 런타임 옵션이 변경되지 않았습니다.
- 명령: `l2s1 --model /path/to/model.gguf --device cuda --diagnostics --input examples/warehouse.json`. 바이너리는 복사된 `lib/` 디렉터리의 일치하는 llama.cpp 라이브러리와 GPU 서버의 CUDA 런타임 라이브러리를 사용했습니다.
- `native_ms`는 응답의 요청-로컬 추론 타이머입니다. 실제 시간에는 로드, 모델 해싱, JSON 출력 및 프로세스 시작이 포함됩니다.

<a id="results"></a>
## 결과

| 모델 | GGUF SHA-256 | 네이티브 추론 | 전체 경과 시간 | 수락된 정답 / 전체 | 선택 / 전체 | 정답 / 선택 | 정책 적용 전 top-1 / 전체 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q8_0 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | 130.73 ms | 1.29 s | 0/3 | 3/3 | 0/3 | 0/3 |
| SmolLM2-135M Instruct Q8_0 | `5a1395716f7913741cc51d98581b9b1228d80987a9f7d3664106742eb06bba83` | 111.14 ms | 0.76 s | 0/3 | 0/3 | 정의되지 않음 | 0/3 |

Qwen는 `ambient`, `false` 및 `low`를 선택했습니다. SmolLM2는 세 가지 판단 모두에서 판단 보류했습니다. 수락된 판단의 정답률이 정의되지 않았습니다. 결과는 두 GGUF가 모두 CUDA 판단 경로를 통해 실행되었음을 증명합니다. 세 가지 합성 판단과 모델당 하나의 타이밍 샘플은 작업 정답률 또는 안정적인 대기 시간을 설정하지 않습니다.

전체 모델 ID, 옵션 점수, 후보 질량, 선택, 판단 보류 이유 및 단계 타이밍은 [`qwen3-q8-rtx3060.json`](../../../../benchmarks/gguf-cuda-20260925/qwen3-q8-rtx3060.json) 및 [`smollm2-q8-rtx3060.json`](../../../../benchmarks/gguf-cuda-20260925/smollm2-q8-rtx3060.json). 모델 중량은 포함되지 않습니다.
