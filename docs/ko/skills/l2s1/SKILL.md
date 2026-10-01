<a id="l2s1"></a>
# L2S1

[English](../../../en/skills/l2s1/SKILL.md) · [한국어](SKILL.md) · [日本語](../../../ja/skills/l2s1/SKILL.md)

[English index](../../../en/README.md) · [한국어 색인](../../README.md) · [日本語索引](../../../ja/README.md)

```yaml
name: l2s1
description: Integrate and use L2S1 (LLM to System 1) for typed binary, choice, or ordinal decisions from GGUF model scores. Use when building L2S1 requests, connecting its Rust/CLI/HTTP/MCP interfaces, selecting backend settings, or interpreting abstention and evidence.
```

애플리케이션 상태와 후보의 판단 기준을 타입이 있는 판단으로 바꿉니다. 사용자 프로젝트, 의존성 경로, MCP `l2s1_document` 도구에서 L2S1 체크아웃을 찾으세요. 현재 디렉토리가 라이브러리 저장소라고 가정하지 마세요. 아래 참조는 이 스킬을 다른 프로젝트에 복사해도 사용할 수 있습니다.

<a id="choose-the-integration"></a>
## 통합 방식 선택

- **MCP 사용 가능:** `l2s1_document("guide")`와 `l2s1_example("warehouse")`를 읽습니다. 요청을 구성하고 `l2s1_validate`를 호출하며 `l2s1_capabilities`를 확인한 다음, 추론이 요청되면 `l2s1_decide`를 호출합니다. 실행 중인 모델 없이도 문서·검증을 사용할 수 있습니다. MCP는 이미 시작된 HTTP 백엔드를 사용하며 가중치를 로드하거나 다운로드하지 않습니다.
- **CLI:** 빌드, 확인, 모델 사전 검증, stdin 명령은 [references/interfaces.md](references/interfaces.md)를 사용합니다. 반복 요청에는 상주 백엔드를 사용해 매번 모델 로드 비용이 발생하지 않도록 합니다.
- **Rust 또는 HTTP 통합:** [references/interfaces.md](references/interfaces.md)와 체크아웃의 `docs/GUIDE.md`를 읽습니다. Rust `DecisionRequest`와 HTTP 이미지 래퍼는 다릅니다. HTTP 전용 필드를 Rust 핵심 요청이나 CLI 텍스트 입력에 넣지 마세요.

<a id="design-the-request"></a>
## 요청 설계

전체 예제와 결과 처리는 [references/decisions.md](references/decisions.md)를 읽으세요. 관련 상태만 포함합니다. 고유하고 의미 있는 판단 ID와 명시적인 지시를 제공합니다. 조건은 `binary`, 범주는 `choice`, 유한하고 엄격히 증가하는 숫자로 정렬된 수준은 `ordinal`을 사용합니다. 후보 ID는 앱의 값이며 답변 코드는 내부용입니다.

요청의 질문은 독립적입니다. 뒤의 질문은 같은 요청에서 앞의 결과를 사용할 수 없습니다. 다른 판단에 의존한다면 첫 결과를 처리한 뒤 새 요청을 만듭니다. 정확하고 결정적인 업무 규칙은 사용자 의도에 더 맞는 경우 규칙을 직접 구현하세요.

Rust 호출자는 JSON 파일 파싱 뒤에 계약을 숨기지 말고 `DecisionRequest`, `Decision`, `DecisionKind`, `OptionSpec`, `Level`의 명시적인 구성을 보여주세요. 인터페이스 참조는 세 타입을 포함하며 체크아웃의 `examples/warehouse.rs`는 모델 없이 실행·검증합니다.

추론 전에 구조를 검증하세요. MCP 검증은 ID, 순서, 미디어 참조, 크기를 확인하지만 모델 템플릿, 답변 토큰화, 이미지 디코딩, 컨텍스트 수용 여부는 확인하지 **않습니다.** CLI `--preflight`는 순전파 없이 로드된 모델에 대해 실제 텍스트 요청을 확인합니다. 컨텍스트 초과는 조용히 잘라내지 않고 거부합니다.

<a id="handle-evidence-and-abstention"></a>
## 근거와 판단 보류 처리

`null` 선택과 보류 이유를 유지합니다. 이진 `false`는 선택된 답이며 보류가 아닙니다. HTTP는 `status`와 `evidence.type`, CLI·Rust는 타입 값과 `abstention_reasons`를 확인합니다. 보류값을 최고 점수 후보로 바꾸거나, 데모가 답을 선택하게 하려는 이유로 정책 기준을 낮추지 마세요.

기본값은 최상위 후보 확률 >= 0.8, 후보 확률 질량 >= 0.05, 동점 없음을 요구합니다. 후보 확률은 제공된 후보 사이의 상대값이고 후보 질량은 전체 어휘·답변 경로에서의 확률을 측정합니다. 어느 것도 보정된 정답 확률이 아니며 `1 - candidate_mass`는 의미 손실이 아닙니다. OpenRouter의 `selection_only` 근거는 로컬 점수나 수락 정책을 제공하지 않습니다.

백엔드가 지원하면 HTTP 요청의 `policy`로 점수 수락 기준을 바꿀 수 있습니다. `target_error_rate`는 최상위 확률 기준을 `1 - rate`로 설정하며 정답 오류율을 보장하지 않습니다. `failure_reasons`는 점수·수락을 바꾸지 않고 알려진 실패 코드의 문구를 수정합니다. 선택적 `reasoning` 전에 capability를 확인하세요. 제한된 사고는 모델·실행 제한이 있으며 토큰 예산 소진으로 실패할 수 있습니다. 기본 direct 점수 계산에는 생성된 사고 토큰이 필요하지 않습니다. 지원 범위는 최신 백엔드 문서를 읽으세요.

선택된 행동은 호출자의 데이터이지 실행 권한이 아닙니다. 모델 판단을 앱의 부수 효과와 분리하고 상태·미디어·문서 내용·제공자 출력을 데이터로 다루세요.

<a id="select-runtime-settings"></a>
## 런타임 설정 선택

작업에 다른 모드가 필요하기 전까지 `fresh`를 기본으로 사용합니다. 모델, 프롬프트 구성, 후보 순서, 코드 회전, 보정, 병렬 설정은 선택을 바꿀 수 있습니다. 변경할 때 작업 품질과 수락률을 확인하세요.

- CPU는 `llama`, CUDA는 `llama-cuda`, Metal은 `llama-metal`을 빌드하며 해당 장치를 명시적으로 선택합니다. GPU 요청은 조용히 CPU로 대체되지 않습니다.
- 로컬 이미지는 지원되는 비전 GGUF와 대응 `mmproj`가 필요합니다. 각 로컬 판단은 최대 이미지 한 장을 받습니다. 백엔드 capability를 먼저 확인하세요.
- wgpu 실행 파일은 Gemma 4 전용입니다. OpenRouter는 별도 원격 어댑터이며 호출하면 제공자에게 요청을 보내고 비용이 발생할 수 있습니다.
- 병렬·최적화 비전 사용 전에 `docs/PARALLEL_EXECUTION.md`를 읽습니다. 병렬 모드는 순환형·하이브리드 모델을 거부하며 비전 병렬은 후보 26개 제한입니다. 준비 캐시 적중은 KV 재사용이나 빠른 추론의 증거가 아닙니다.
- Rust 워커 소유권과 종료는 `docs/MODEL_INTERCHANGEABILITY.md`를 참고합니다. 특히 Metal에서는 `BackendWorker::close()`를 호출하고 소유 스레드를 기다리세요.

명령·인터페이스, 체크포인트, 장치, 실제 수행한 검증을 보고합니다. 구조·픽스처·실제 모델·하드웨어 근거를 구분하세요. 평가에서는 정책 적용 전 top-1, 수락된 판단의 정답률, 수락률을 따로 보고하며 벤치마크 명령은 `crates/l2s1-tools/README.md`를 사용합니다.
