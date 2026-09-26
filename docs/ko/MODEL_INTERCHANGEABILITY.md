<a id="model-interchangeability-in-l2s1"></a>
# L2S1의 모델 호환성

[English](../en/MODEL_INTERCHANGEABILITY.md) · [한국어](MODEL_INTERCHANGEABILITY.md) · [日本語](../ja/MODEL_INTERCHANGEABILITY.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

L2S1는 호환되는 로컬 GGUF 모델을 교체하면서 애플리케이션 판단 계약을 유지합니다. ID 및 사전 검증는 호환성을 설명합니다. 품질 및 임계값 적합성을 위해서는 여전히 레이블이 지정된 작업 부하가 필요합니다.

<a id="inspect-validate-run"></a>
## 검사, 검증, 실행

CPU의 경우 `llama` 기능을 사용하고 CUDA의 경우 `llama-cuda` 기능을 사용하여 빌드하세요. sys 종속성은 일치하는 llama.cpp 소스 및 라이브러리를 빌드합니다. 예제에서는 컴파일된 바이너리를 사용합니다.

```sh
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf --inspect
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --preflight
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --diagnostics
```

`LlamaBackend::inspect()`는 체크포인트 SHA-256(GGUF 토크나이저 포함), 내장형 템플릿 해시, 효과적인 프롬프트 프로필/버전, 브리지/런타임 빌드 해시, 로드된 라마/ggml 라이브러리 콘텐츠 해시를 보고합니다. 어댑터/헤드 해시, 장치 및 컴퓨팅 구성. 체크포인트 및 로드된 라이브러리 해시는 백엔드 등록 시 한 번 계산됩니다. 따라서 시작 시 전체 체크포인트를 읽습니다. 최적화된 빌드를 권장합니다. 등록된 중량/어댑터/런타임 파일을 사용하는 동안 불변으로 유지하십시오. 이 ID는 동시 파일 수정에 대한 서명이나 보호가 아닌 콘텐츠 출처입니다. GPU 드라이버, 펌웨어 및 운영 체제 ID는 지문이 채취되지 않습니다. CPU 장치 레이블은 고유한 프로세서 모델이 아닌 CPU 백엔드를 식별합니다. 구성 일치는 모든 물리적 시스템에서 동일한 동작을 설정하지 않습니다.

로드된 라이브러리 검색은 현재 Linux의 동적 로더를 사용합니다. 확인된 런타임 검색을 사용할 수 없는 경우 네이티브 백엔드가 명시적으로 실패합니다. 순수 Rust 스코어링, 보정 및 작업자 모듈에는 Linux 또는 llama.cpp가 필요하지 않습니다. 런타임 빌드 해시는 브리지/점수/프롬프트/보정 소스, 종속성 잠금 파일, Jinja 소스, 연결된 코어 라이브러리, 컴파일된 브리지 아카이브(전이적 헤더 및 C++ 컴파일 포함), 대상/프로필/Rust 플래그 및 컴파일러 버전을 바인딩합니다. 로드 라이브러리 해시는 동적으로 로드된 ggml 장치 백엔드도 포함합니다.

`preflight()`는 추론과 동일한 모델 바인딩 준비를 사용합니다. 실제 응답 경계에서 요청 구조, 실행 지원, 활성 아티팩트, 컨텍스트 길이 및 안정적인 후보 연속성을 확인합니다. 최대 26 후보는 기존 단일 토큰 매핑을 사용합니다. 더 큰 세트는 고정 너비 대문자 코드를 사용하고 스칼라 `candidate_token_ids` 대신 `candidate_token_sequences`를 반환합니다. 토큰 시퀀스는 고유해야 하며 접두사가 없어야 합니다. 이러한 판단은 출력 헤드, 스칼라 보정 또는 기능 내보내기 없이 전체 증거를 사용하여 새로운 실행 또는 접두사 재사용 실행을 지원합니다. `encode_decision_sequences()`는 전체 경로를 노출합니다. 이전 스칼라 내보내기 API는 와이드 코드를 명시적으로 거부합니다. [시퀀스 점수 계약](INTENT_BENCHMARK.md)를 참조하세요. 사전 검증는 추론 없이 토큰 수, 후보 매핑 및 프롬프트 토큰 지문을 반환합니다. 기능 플래그는 메타데이터 기반이며 모델 품질이나 수치적 동등성을 보장하지 않습니다. 하이브리드/반복 모델은 병렬 실행을 거부합니다. 접두사 재사용은 명시적으로 새로운 것으로 대체됩니다. 상태 복원은 전체 증거 전송이 포함된 명시적 직렬 모드로 제공됩니다. 로드된 헤드는 여전히 기존의 신규 전용/장치/컴퓨팅 제한을 적용합니다.

기존 `DecisionBackend::decide()` 및 응답 JSON는 계속 사용할 수 있습니다. `decide_detailed()`는 모델 ID, 판단별 요청/유효 실행, 대체 이유, 증거 원본, 보정 ID, 요청 로컬 타이밍 및 스냅샷 계정이 포함된 별도의 봉투를 추가합니다. 원시 프롬프트 텍스트를 반환하지 않습니다. 사전 검증의 준비/토큰 해시는 추론 타이밍 외부에 있습니다. 병렬 배치의 네이티브 시간은 한 번 계산됩니다. 보통 판단 보류는 성공적인 결과를 유지합니다. 자세한 실패는 유효하지 않은 요청, 지원되지 않는 기능, 호환되지 않는 아티팩트, 컨텍스트 오버플로, 유효하지 않은 점수 증거 및 네이티브 실패를 구별합니다. CLI `--preflight`/`--diagnostics` 반환 요청 단계 실패 JSON stdout 및 0이 아닌 종료 상태; 모델 로딩 및 잘못된 형식의 JSON는 이 엔벨로프 이전에 여전히 실패할 수 있습니다.

<a id="exact-evidence-and-policy"></a>
## 정확한 증거와 정책

`ExactEvidence::from_logits()`는 완전한 어휘 벡터에서 검증된 증거를 생성합니다. 공유 채점자는 정규화, 바이너리/선택/서열 해석, 동점 처리 및 두 가지 승인 게이트를 모두 소유합니다. 원래의 f64 합산 순서를 유지합니다. 증거에는 의미 체계 ID, 토큰 ID, 네이티브 logits, 어휘 크기 및 전체 어휘 로그 정규화가 유지됩니다. 후보자 전용 점수, 부분적인 상위 k 결과 및 생성된 숫자 추정치는 역직렬화 또는 공개 필드 돌연변이를 통해 이 정확한 증거 유형을 인스턴스화할 수 없습니다.

전체 어휘 후보 확률 질량은 보정된 확률과 독립적으로 유지됩니다. 기존 출력 헤드는 기본 모델 매스 게이트를 유지하고 동일한 정책 점수 측정기를 사용합니다. 증거는 네이티브 백엔드의 소유권 및 진단 봉투에 의해 준비된 판단/모델에 바인딩됩니다. 휴대용 인증 추론 영수증이 아닙니다.

<a id="optional-execution-optimizations"></a>
## 선택적 실행 최적화

모든 최적화는 명시적으로 선택합니다. 기본값은 legacy 프롬프트, fresh 실행, 전체 근거 전송, 비활성화된 준비 캐시입니다. 앱은 반복·고정 스키마 작업을 유지하거나 요청 사이에 스키마를 바꿀 수 있습니다. 학습된 상태 인코더나 새 모델 체크포인트는 필요하지 않습니다.

<a id="bounded-preparation-reuse"></a>
### 제한된 준비 재사용

```rust,ignore
backend.set_preparation_cache(l2s1::PreparationCacheConfig {
    max_entries: 128,
    max_bytes: 8 * 1024 * 1024,
});
let response = backend.decide(&request)?;
let cache_stats = backend.preparation_cache_stats();
backend.clear_preparation_cache();
```

백엔드는 텍스트 준비, 비전 프롬프트 준비, 후보 매핑의 FIFO 캐시 세 개를 소유합니다. 전체 준비의 키는 정확하게 직렬화된 상태와 판단입니다. 후보 매핑의 키는 실제 어시스턴트 응답 경계 텍스트와 후보 수입니다. 모델·토크나이저·템플릿의 소유권은 백엔드 내부에 있으며 관련 설정 변경은 캐시된 준비를 무효화합니다. 질문·후보 수·지시·상태가 달라 캐시가 빗나가면 일반 준비를 수행합니다. 성공한 준비만 캐시하며 예측이나 네이티브 KV 상태는 저장하지 않습니다. 반환 토큰 벡터는 계속 복사합니다. 단일 문자 매핑과 여러 문자 토큰 경로는 같은 제한을 공유합니다. 넓은 매핑은 외부 벡터와 각 토큰 경로의 할당을 모두 셉니다. 캐시 적중은 실행 모드·아티팩트 검사를 우회하지 않습니다.

`max_entries`는 각 캐시에 따로 적용되어 전체 항목 수는 한도의 세 배에 도달할 수 있습니다. 하나의 전체 `max_bytes` 예산은 텍스트 프롬프트에 절반, 비전 프롬프트와 후보 매핑에 각각 1/4을 배정합니다. 보유한 키·토큰 할당과 인라인 항목을 세며 할당자 메타데이터, 남는 큐 용량, 임시 입력 키, 반환 사본은 이 범위 밖입니다. 큰 항목은 유용한 항목을 제거하지 않고 저장을 건너뜁니다. 항목 수 제한이나 바이트 제한 중 하나가 0이면 저장을 비활성화합니다. `clear_preparation_cache()`는 항목을 지우되 적중·실패·퇴거 카운터를 유지합니다. `set_preparation_cache()`는 캐시 세 개를 교체하고 카운터를 초기화합니다. 일반 요청 경계는 준비된 토큰을 유지하지만 네이티브 KV 상태는 지웁니다.

CLI에 해당하는 항목은 `--preparation-cache-bytes 8388608 --preparation-cache-entries 128`입니다. 바이트 기본값은 0이므로 요청될 때까지 캐싱이 비활성화됩니다. CLI 호출은 각각 새로운 백엔드를 로드합니다. 호출 전반에 걸쳐 재사용하려면 상주 라이브러리 백엔드 또는 작업자가 필요합니다.

<a id="compact-native-evidence-transfer"></a>
### 컴팩트 네이티브 증거 전송

```rust,ignore
backend.set_evidence_transfer(l2s1::EvidenceTransfer::Compact)?;
```

`--evidence-transfer compact`는 CLI에서 동일한 경로를 선택합니다. `Full`는 기본값으로 유지됩니다. 컴팩트 모드는 C++ 브리지에서 전체 어휘 로그 노멀라이저를 계산하고 해당 노멀라이저, 어휘 크기 및 후보 logits만 Rust에 반환합니다. `candidate_mass`의 의미를 유지합니다. 분모를 후보 전용 정규화로 대체하지 않습니다. 네이티브 logits는 여전히 호스트의 전체 어휘를 ​​포함합니다. 이는 어휘 출력 프로젝션이나 llama.cpp의 장치-호스트 작업이 아닌 Rust로의 호스트 버퍼 복사본을 제거합니다. 동일한 부동 소수점 구현을 가정하는 대신 실제 런타임에서 숫자 델타를 비교하십시오.

텍스트 축약 모드는 명시적 공유 상태 세션을 포함한 fresh·prefix-reuse 실행을 지원합니다. 텍스트 state-restore·parallel 실행과 학습된 출력 헤드는 거부합니다. 비전 축약 모드는 단일 토큰 답변 코드(최대 후보 26개)의 fresh·parallel 실행을 지원하며 넓은 이미지 코드는 fresh·전체 근거가 필요합니다. 각 이미지 시퀀스는 자체 전체 어휘 정규화값을 유지합니다. 선택한 전송 모드는 모델·보정 식별 정보에 포함되며 축약 응답에 보고합니다. 전체 모드 JSON은 호환성을 위해 추가 필드를 생략합니다. 보정 등록 전에 모드를 선택하세요. 호환되지 않는 모드로 바꾸면 다른 모드에 학습한 아티팩트를 조용히 재사용하지 않고 실패합니다.

<a id="explicit-immutable-state-sessions"></a>
### 명시적 불변 상태 세션

```rust,ignore
backend.set_execution_mode(l2s1::ExecutionMode::PrefixReuse);
backend.set_prompt_layout(l2s1::PromptLayout::StateFirst);
{
    let mut session = backend.shared_state(state)?;
    let first = session.decide(first_questions)?;
    let next = session.decide(different_questions)?;
} // Native KV state is cleared here.
```

`SharedStateSession`는 하나의 백엔드를 독점적으로 빌리고 하나의 불변 JSON 상태를 소유합니다. 각 `decide(Vec<Decision>)` 호출은 판단 ID, 종류, 지침, 옵션 개수 및 판단 기준을 변경할 수 있습니다. ID는 통화 내에서 고유해야 하며 이후 통화에서 반복될 수 있습니다. 모델, 아티팩트, 정책 및 프롬프트 구성은 대여 중에 변경할 수 없습니다. 각 질문은 여전히 ​​독립적으로 컴파일된 프롬프트를 받습니다. 정확한 공통 토큰 접두어의 완전한 사전 채우기 배치만 재사용되며 질문은 이전 답변에 관여하지 않습니다.

세션에는 명시적으로 선택된 `PrefixReuse` 모드가 필요하며 순환/하이브리드 메모리 및 출력 헤드를 거부합니다. 선택한 프롬프트 레이아웃과 증거 전송 모드를 유지합니다. 상태 우선 레이아웃은 질문이 변경될 때 더 긴 공통 접두사를 노출할 수 있지만 레이아웃을 변경하면 예측이 바뀔 수 있습니다. 이는 양방향 상태 인코더나 작은 학습 읽기 헤드가 아닌 명시적 세션 호출 전반에 걸쳐 기존 인과 디코더의 접두사 재사용입니다. 재사용은 실제 일치하는 토큰에 따라 달라집니다. 하나의 짧은 질문이 더 빨라질 필요는 없습니다.

생성, 호출 실패, 해제 시 네이티브 KV 상태를 지웁니다. 실패 후에도 같은 세션은 다른 유효한 호출을 받을 수 있습니다. 일반 `DecisionBackend::decide()`는 요청 경계에서 격리되며 세션은 그 수명 계약을 바꾸지 않습니다. 각 결과의 `reused_prefix_tokens`는 실제 재사용을 보고합니다. 여러 문자 코드는 루트 프롬프트 평가의 재사용을 보고하며 `code_evaluated_tokens`는 모든 코드 프리픽스 분기에서 디코딩한 토큰을 셉니다. 넓은 코드는 전체 근거가 필요합니다. 세션을 빌리는 동안 `session.preparation_cache_stats()`와 `session.take_timings()`로 카운터를 확인합니다.

<a id="scoped-scalar-calibration"></a>
## 범위가 지정된 스칼라 보정

`ScalarCalibration::fit()`는 0.05에서 20까지의 경계 검색과 피팅이 악화될 경우 온도 1 폴백을 사용하여 보정 세트 NLL에 의해 양의 온도에 적합합니다. NLL. 판단 3종 모두를 지원합니다. 각 아티팩트는 전체 모델/구성 지문, 정확한 작업 서명(옵션 순서 및 서열 값 포함), 네이티브 점수 의미 체계, 보정 레코드 해시 및 소스 그룹 해시를 바인딩합니다. 다른 모델인 양자화, 어댑터, 템플릿/프로필, 컴퓨팅 설정 또는 실행 모드는 요청 시 아티팩트를 거부합니다. 구성을 변경해도 로드된 보정은 자동으로 삭제되지 않습니다.

```sh
cargo run --release --offline --example fit_calibration -- fit-input.json task-temperature.json
l2s1 --model models/SmolLM2-135M-Instruct-Q8_0.gguf \
  --input examples/warehouse.json --calibration task-temperature.json --diagnostics
```

`fit-input.json`은 `id`, `model`(검사의 `identity`), `decision`(정확한 요청 판단 하나), `calibration`, 독립된 `held_out` 배열을 포함합니다. 각 레코드는 `group`, 작업 후보 순서의 `raw_logits`, 0부터 시작하는 `correct_option`을 포함합니다. 정확한 모델·설정으로 측정한 logits를 제공하세요. 보정 학습 도구는 숫자 기록만으로 출처를 입증할 수 없습니다. 아티팩트를 쓰기 전에 겹치는 원본 그룹 ID와 검증 후보 수 불일치를 거부하고 보정 전후 검증 NLL·Brier를 출력합니다. 원본별로 바꿔 쓴 문장과 후보 변형을 같은 그룹 ID에 묶어야 합니다. 정확한 ID 검사는 잘못 붙인 그룹이나 근접 중복을 탐지하지 않습니다.

`evaluate_policy()`는 측정된 베이스 후보 확률 질량을 추가로 허용하고 지정된 `DecisionPolicy`에서 수락률, 수락된 판단의 정답률(아무 것도 허용되지 않으면 null) 및 최고 선택 정답률을 보고합니다. 유지된 수용 곡선에 대해 여러 임계값을 사용합니다. 피팅 측정항목은 정답률 테스트가 아닌 적합 측정항목으로 명시적으로 라벨이 지정됩니다. 훈련되었거나 보편적으로 적합한 온도는 번들로 제공되지 않습니다.

`register_calibration()`/`load_calibration()`에 아티팩트를 등록합니다. CLI는 반복되는 `--calibration`를 허용합니다. 작업/ID당 최대 하나의 아티팩트가 허용됩니다. 다른 작업 ID는 네이티브 점수를 유지합니다. 작업 의미가 변경된 일치 ID가 실패합니다. 원시 logits 및 기본 질량은 유지됩니다. 보정된 채점 및 아티팩트 ID는 명시적입니다. 스칼라 보정은 출력 헤드와 누적될 수 없습니다. `clear_calibrations()`는 명시적 제거 작업입니다.

<a id="request-local-state-restoration"></a>
## 요청-로컬 상태 복원

```sh
l2s1 --model MODEL.gguf --input examples/warehouse.json \
  --prompt-layout state-first --execution-mode state-restore \
  --snapshot-limit-bytes 268435456 --diagnostics
```

상태 복원은 하나의 요청에서 모든 판단에 공통된 정확한 접두사를 찾아 반올림하여 사전 채우기 배치를 완료합니다. 해당 접두사를 계산하고 llama.cpp의 시퀀스 상태 API를 사용하여 전체 시퀀스 상태를 저장합니다. 첫 번째 판단은 원래 사전 채우기에서 계속됩니다. 나중에 판단은 접미사를 평가하기 전에 저장된 상태를 지우고 복원합니다. 각 답변에는 독립적인 logits가 있으며 최소한 최종 토큰은 항상 평가됩니다. 첫 번째 판단은 접두사 사전 채우기 비용을 지불하고 재사용된 토큰이 0이라고 보고합니다. 후속 판단은 공유 접두사를 보고합니다. `restores` 진단은 실제 상태 로드를 계산하므로 성공적인 N-판단 요청은 N-1 복원을 보고합니다. 단일 판단 또는 짧은 공통 접두사 요청이 새로 실행됩니다.

스냅샷은 네이티브 호출을 벗어나지 않으며 요청, 오류, 컨텍스트 크기 조정, 프롬프트 변경 또는 어댑터 변경 후에도 유지되지 않습니다. 스냅샷 버퍼는 하나만 존재합니다. 크기는 할당 전에 구성된 바이트 제한에 대해 확인됩니다(기본값 256 MiB, 공통 접두사가 있는 경우 0은 예산 폴백을 강제 적용). 저장/복원 지원이 없거나 예산 초과로 인해 명시적인 새로운 대체가 트리거됩니다. 네이티브 디코드 실패는 오류로 남아 있습니다. 진단은 스냅샷 바이트, 저장/복원/미리 채우기/접미사 벽 시간 및 복원 횟수를 노출합니다. 이는 총 RSS, GPU 할당, logits 버퍼 또는 llama.cpp의 내부 스크래치 메모리가 아닌 스냅샷 버퍼를 제한합니다. 전체 프로세스 피크 RSS를 별도로 측정합니다.

이는 하이브리드 메모리를 포함한 직렬 상태 복원입니다. 완전한 증거 전송이 필요하며 하이브리드 병렬 시퀀스 또는 지속적인 공유 상태 세션을 활성화하지 않습니다. 이는 명시적 실행 모드입니다. `prefix-reuse`는 계속해서 KV 롤백을 사용하고 순환/하이브리드 모델에서는 새로운 상태로 돌아갑니다. 속도 향상은 약속되지 않습니다. 상태 복사는 재계산보다 비용이 더 많이 들 수 있습니다. 배포 전에 새로운 실행과 정확한 모델/구성을 비교하세요. 상태 우선 프롬프트는 변경된 순서가 실행 모드와 관계없이 예측을 변경할 수 있기 때문에 옵트인 상태로 유지됩니다. [Bonsai RTX 3060 검증](benchmarks/bonsai-state-restore-20260925/REPORT.md)는 실제 상태 저장 및 복원을 통해 하이브리드 경로를 연습합니다.

<a id="bounded-ownership-and-scheduling"></a>
## 제한된 소유권 및 일정 관리

`BackendWorker::spawn()`은 백엔드 팩토리, 큐 용량, 최대 직렬화 요청 바이트, 공유 `MemoryBudget`, 양의 모델 메모리 예약 추정값을 받습니다. 팩토리는 전용 스레드 안에 백엔드를 만들며 `LlamaBackend`는 non-Send/non-Sync를 유지합니다. 요청은 소유 스레드에서 직렬 처리합니다. `submit()`은 가득 찬 큐를 바로 거부하며 받아들인 작업에 `DecisionTicket`을 반환합니다. `close()`는 받은 작업을 모두 처리하고 스레드 종료를 기다린 뒤 소유 백엔드·예약을 해제합니다. 티켓 해제는 작업을 취소하지 않습니다. 생성 실패·패닉은 예약을 해제하며 중지된 워커는 연결 해제를 보고합니다.

```rust,ignore
let budget = l2s1::MemoryBudget::new(2 * 1024 * 1024 * 1024);
let mut worker = l2s1::BackendWorker::spawn(
    8, 1024 * 1024, &budget, 1024 * 1024 * 1024,
    move || l2s1::llama::LlamaBackend::load(
        &model_path, 2048, 256, 4, false, l2s1::DecisionPolicy::default()),
)?;
let response = worker.submit(request)?.wait()?;
worker.close()?;
```

`BackendWorker::spawn_batched()`는 팩토리 인수 앞에 `BatchPolicy`를 추가하고 `BatchDecisionBackend`가 필요합니다.

```rust,ignore
let policy = l2s1::BatchPolicy {
    max_requests: 4,
    max_input_tokens: 8192,
    max_wait: std::time::Duration::from_millis(2),
};
let mut worker = l2s1::BackendWorker::spawn_batched(
    8, 1024 * 1024, &budget, 1024 * 1024 * 1024, policy,
    move || {
        let mut backend = l2s1::llama::LlamaBackend::load(
            &model_path, 2048, 256, 4, false, l2s1::DecisionPolicy::default())?;
        backend.set_execution_mode(l2s1::ExecutionMode::Parallel);
        backend.set_parallel_width(4)?;
        Ok(backend)
    },
)?;
```

토큰 허용은 실제 모델 준비를 사용하고 문자나 요청 수를 추정하는 대신 모든 판단 프롬프트를 합산합니다. 배치당 토큰 제한을 초과하는 요청은 자체 티켓을 통해 실패합니다. 나머지 배치에 맞지 않는 유효한 요청은 다음 배치를 기다립니다. 결과는 입력 요청 순서와 그룹화를 유지합니다. `max_wait`는 첫 번째 요청 승인 시간부터 예산을 수집합니다. 이는 추론 기한, 엄격한 사전 검증 시간 제한 또는 종단 간 대기 시간 보장이 아닙니다. 제로 대기는 추가 요청 수집을 건너뜁니다. 대기열 및 직렬화된 바이트 제한은 계속 적용됩니다.

`LlamaBackend`를 사용하면 명시적으로 구성된 `Parallel` 모드만 승인된 요청을 네이티브 웨이브로 병합합니다. 신규, 접두사 재사용 및 상태 복원 모드는 직렬로 유지되며 요청 격리를 유지합니다. 네이티브 병렬 실패는 추론을 재생하지 않고 영향을 받은 배치에 실패합니다. 병렬 모드는 모델 제한, 추가 KV 메모리 및 측정된 점수 드리프트 위험을 유지합니다. 토큰 승인 및 수집에도 시간이 걸립니다. 일괄 처리만으로는 대기 시간이 줄어들거나 처리량이 늘어나지 않습니다.

예약에는 가중치, KV/반복 상태, 출력 및 여백이 있는 네이티브 스크래치 공간이 포함되어야 합니다. 운영 체제 메모리 제한이 아닌 운영자가 제공한 추정치를 기준으로 승인을 적용합니다. 대기열 제한은 호출자의 기존 할당이 아니라 승인된 요청 수 및 직렬화된 입력 크기를 제한합니다. 닫기는 실행 중인 네이티브 호출을 기다릴 수 있습니다. 강제 취소는 제공되지 않습니다. 모델을 변경하려면 작업자를 비우거나 닫고 다른 작업자를 구성하거나 두 가지 모두에 대해 명시적으로 충분한 예산을 확보하십시오. HTTP 서비스, 원격 공급자, 지속적인 모델 간 캐시 또는 라이브 핫스왑 정책이 도입되지 않았습니다.

<a id="validation-and-compatibility"></a>
## 검증 및 호환성

[검증 가이드](VERIFICATION.md)를 참조하세요. 동일한 적합성 테스트는 콜론으로 구분된 모델 경로를 허용하고 모든 결과 종류/매핑, 아티팩트, 실패, 복구, 스냅샷 제한 및 최신 대 최적화 점수를 확인합니다. 접두사 재사용/상태 복원은 기존 0.02 확률/질량 판단 기준와 변경되지 않은 최상위 선택 및 허용된 결과를 유지합니다. 병렬 실행에는 [알려진 배치 형태 드리프트](PARALLEL_EXECUTION.md)가 있습니다. 변경되지 않은 동등성 판단 기준을 별도로 보고하고 옵트인 상태로 유지됩니다.

```sh
cargo test --locked --offline
cargo test --release --locked --offline --features llama
L2S1_CONFORMANCE_MODELS=MODEL_A.gguf:MODEL_B.gguf \
  L2S1_CONFORMANCE_REPORT=/tmp/conformance.json \
  cargo test --release --locked --offline --features llama \
  --test conformance -- --ignored --nocapture
```

[`examples/benchmark_optimizations.rs`](../../examples/benchmark_optimizations.rs)를 사용하여 선택적 경로를 로컬 모델과 비교합니다.

```sh
cargo run --release --locked --offline --features llama \
  --example benchmark_optimizations -- \
  --model MODEL_A.gguf --model MODEL_B.gguf \
  --output /tmp/l2s1-optimizations.json --repeats 3
```

출력은 새로운 것이어야 합니다. 모델당 하나의 백엔드가 상주합니다. 하네스는 짧은/긴 합성 웨어하우스 상태를 사용하고 1/4/16는 사이클링 선택, 바이너리 및 서열 스키마에 대해 질문하고 경로 순서를 회전하며 각 측정된 경로/시나리오/라운드 직전에 시간이 정해지지 않은 워밍업을 한 번 실행합니다. 전체 응답, ID, 캐시 카운터, 입력/재사용 토큰, 타이밍 및 최대 확률/질량/logit 차이를 기록합니다. 캐시된/압축 경로는 기존의 새로운 경로와 비교됩니다. 공유 상태 호출은 상태 우선 신규 호출과 비교됩니다. 질문당 분할 평균 시간은 개별 요청 대기 시간이 아닙니다. 동일한 준비 입력으로 반복적인 준비가 수행됩니다. 새로운 질문에 대한 캐시 이득을 설정하지 않습니다. 이 하네스는 작업자 일괄 처리 또는 레이블이 지정된 작업 정답률을 측정하지 않습니다.

`ExecutionMode::StateRestore` 및 추가된 `EvidenceTransfer` 구성에는 다운스트림 전체 일치가 필요하며 해당되는 경우 업데이트하려면 수동으로 생성된 Rust 메타데이터 구조가 필요합니다. 기본 전체 전송 JSON는 추가된 전송 필드를 생략합니다. 레거시 프롬프트/새 실행 기본값과 기본 점수 의미 체계가 유지됩니다. 테스트 설비 및 계약 동등성은 운영 환경 또는 레이블이 지정된 워크로드 품질을 설정하지 않습니다.

<a id="cpugpu-placement"></a>
## CPU/GPU 배치

CUDA 로드는 `--gpu-layers N` 및 `--cpu-moe-layers N`를 허용합니다. 첫 번째 제한은 GPU 레이어 배치입니다. 두 번째는 CPU RAM의 첫 번째 N MoE 레이어의 전문가 가중치를 유지하면서 주의 및 공유 가중치를 위한 일반적인 배치를 유지합니다. 이러한 설정은 결합될 수 있습니다. 이는 체중 상주를 제어합니다. llama.cpp는 사전 채우기 중에 호스트 가중치를 사용하여 작업을 CUDA로 계속 오프로드할 수 있습니다. 원래 CUDA 로딩 동작을 유지하려면 둘 다 생략하세요. CPU 전용 로드는 CUDA 배치 요청을 거부합니다.

```sh
l2s1 --model models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --device cuda \
  --cpu-moe-layers 18 --context 8192 --batch 256 --threads 8 \
  --preparation-cache-bytes 67108864 --input request.json
```

Rust 필드는 `ComputeOptions.gpu_layers: Option<u32>` 및 `cpu_moe_layers: u32`입니다. 수동으로 생성된 구조체는 레거시 기본값에 대해 `None` 및 `0`를 제공해야 합니다. 이전 JSON는 읽기 가능한 상태로 유지되며 기본 직렬화에서는 두 필드가 모두 생략됩니다. 기본값이 아닌 값은 응답 컴퓨팅 메타데이터 및 모델 ID에 나타나므로 보정/헤드/worker ID는 배치를 구별합니다.

CPU 전문가 분할에는 MoE 체크포인트 및 N이 레이어 수보다 크지 않아야 합니다. 패턴은 llama.cpp의 전문적인 텐서 이름 지정을 사용하며 로드된 엔진의 소유로 유지됩니다. 이는 CUDA 할당 실패 후 CPU 폴백이 아닌 명시적인 가중치 배치입니다. 네이티브 로그는 실제 CPU/GPU 버퍼를 보고합니다. 요청된 레이어 수만으로는 VRAM 예산을 증명할 수 없습니다. CPU RAM, KV 및 컴퓨팅 버퍼도 맞아야 합니다. RTX 3060 12 GiB에서 컨텍스트 8192 및 배치 256가 있는 26B Q4 체크포인트가 14를 사용하여 컨텍스트 할당에 실패했습니다. CPU 전문가 레이어,  18 레이어는 모든 400 인텐트 사례(200 BANKING77 영어 및 200 MASSIVE 한국어)를 완료했습니다. 캐시된 반복 하나. 모든 400 캐시된 호출은 전체 판단 증거를 정확하게 보존했습니다. 이는 임의의 컨텍스트 길이나 동시 모델 인스턴스가 아닌 테스트된 배치 및 컨텍스트를 검증합니다.

양자화된 CPU 및 GPU 커널은 서로 다른 점수를 생성할 수 있습니다. CPU 상주 레이어가 있는 Qwen3 0.6B Q8 프로브는 전체 CUDA에 대한 기존 0.02 확률 차이 판단 기준을 초과했습니다(최대 0.028); 이는 동등 주장이 아닙니다. 의도한 배치에서 정답률을 평가합니다. 준비 및 세션 캐시 검사는 동일한 배치의 새로운 추론과 비교됩니다.

<a id="model-loading-and-peak-host-rss"></a>
## 모델 로딩 및 피크 호스트 RSS

`--model-load-mode read`는 자동 메모리 매핑 대신 llama.cpp의 일반 읽기/업로드 경로를 선택합니다. 가중치가 기존 CPU 및 CUDA 버퍼에 도달하는 방식을 변경합니다. 체크포인트, 양자화, 레이어 분할, 컨텍스트 또는 추론 알고리즘은 변경되지 않습니다. `auto`는 기본값으로 유지됩니다. 이 옵션은 CLI, JSONL 평가자, 인텐트 캐시 벤치마크 및 JevBench 하네스에서 사용할 수 있습니다.

26B CPU/GPU 분할의 경우 배치 명령에 추가합니다.

```sh
l2s1 --model models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --device cuda \
  --cpu-moe-layers 18 --model-load-mode read \
  --context 8192 --batch 256 --threads 8 --input request.json
```

Rust 필드는 `ComputeOptions.model_load_mode: ModelLoadMode`입니다. 기존 Rust 구조체 리터럴은 `ModelLoadMode::Auto`를 추가해야 합니다. 이전 JSON는 읽기 가능한 상태로 유지되며 기본 직렬화에서는 필드가 생략됩니다. 명시적 `read`는 컴퓨팅 메타데이터 및 모델/아티팩트 ID에 기록됩니다. 원래 네이티브 진입점은 외부 참조 호출자에 대한 자동 로딩을 유지합니다.

감소된 프로세스 RSS는 총 물리적 메모리 수요 감소와 동일하지 않습니다. 읽기 로드는 할당된 백엔드 버퍼(잠재적으로 CUDA 고정 호스트 메모리)를 사용하는 반면 매핑된 가중치는 깨끗한 파일 지원 페이지입니다. OS 파일 캐시는 RSS 프로세스 외부에 남아 있습니다. 의도된 호스트에서 높은 물 RSS 로드, 안정적인 추론 RSS 및 메모리 압력을 별도로 측정합니다. 모든 체크포인트 또는 CPU 전용 배포에서 읽기 로딩이 더 좋다고 가정하지 마십시오.

하나의 RTX 3060 호스트에서 동일한 Gemma 4 26B A4B Q4 체크포인트의 모드당 3개의 교대 웜 캐시 로드와 18 CPU 전문가 레이어가 측정되었습니다. `auto`의 16.174 GiB 및 `read`(42.2% 하위)의 9.349 GiB의 중간 피크 엔진 RSS. 짧은 포스트 로드 RSS 중앙값은 10.332 대 9.334 GiB였습니다. 모델 로딩에는 10.361와 13.931초가 소요되었습니다. `read` 실행의 모든 ​​231 공개 JevBench 증거 개체는 이전 `auto` 실행과 정확히 일치했습니다. 엔진 RSS 피크에서 샘플링된 시스템 `MemAvailable`는 `auto`의 28.903 GiB이고 `read`의 19.941 GiB였으므로 하위 프로세스 RSS는 증거가 아닙니다. 총 물리적 메모리 수요가 더 낮습니다. 이는 Metal 결과나 일반적인 로드 시간 보장이 아닌 로컬 Linux/CUDA 측정이었습니다. 전체 로컬 보고서 및 소스 해시는 `results/gemma26-lowrss-20260923T135227Z/REPORT.md`에서 무시됩니다.

<a id="vision-optimization-components"></a>
## 비전 최적화 구성 요소

`--vision-optimized`(또는 `ComputeOptions::vision_optimized()`가 포함된 `enable_vision_optimizations()`)는 4개의 독립적인 네이티브 이미지 시퀀스를 준비 캐싱, 압축 증거 및 더티 전용 KV 지우기와 결합합니다. 일치하는 프로젝터와 최대 26 답변 옵션이 필요합니다. 캐시된 준비 및 압축 증거는 변경되지 않은 컴퓨팅 구성에서 점수를 보존합니다. 결합된 병렬/Flash Attention 경로는 예측을 변경할 수 있습니다. 모델, 컴퓨팅 설정 및 하드웨어는 수치 검증에 사용된 참조와 일치해야 합니다. 사전 추론 결과는 캐시되지 않습니다.
