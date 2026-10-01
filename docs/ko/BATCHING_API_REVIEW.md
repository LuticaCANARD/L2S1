# 반복 decide 호출과 native batching API

Rust·TypeScript·Python에서 같은 모델을 유지하고 데이터만 바꿔 호출할 수 있다.
TS `prepare<State>(decisions)`와 Python `prepare(decisions, state_type=State)`는
고정 질문을 보관하고 state만 받는다. Python은 Pydantic 검증과 PEP 561 타입을 제공한다.

| 경계 | API | 실행 |
| --- | --- | --- |
| Rust 라이브러리 | `LlamaBackend::decide_batch(&requests)` | parallel 모드에서 독립 native sequences; 다른 모드에서는 직렬 |
| TS 로컬 SDK | `load({ executionMode: 'parallel' })` → `decideBatch(requests)` | 기본 stdio; 빌드된 Rust 프로세스에 배열을 한 번 전달 |
| Python 로컬 SDK | `await load(LoadOptions(execution_mode="parallel", ...))` → `decide_batch(requests)` | 기본 stdio; 같은 Rust 실행 파일·런타임 번들 사용 |
| HTTP | `POST /v1/decision-batches`, `{"requests": [...]}` | native parallel 배열 호출 |
| Rust 자동 수집 워커 | `BackendWorker::spawn_batched()` | 요청 수·입력 토큰·대기 시간으로 제한한 microbatch |

## 포트 없는 빌드 모드

Rust 실행 파일은 `l2s1 --model model.gguf --execution-mode parallel --stdio`로
상주한다. SDK `load()`는 이것을 기본 사용하고 stdin/stdout JSON lines로 통신한다.
실행 파일과 GGUF를 제공하면 HTTP 서버·Node.js 없이 Python에서도 동작한다.
TS의 N-API나 Python의 PyO3로 같은 프로세스에 로드하는 확장은 아니다.
Rust 사용자는 라이브러리를 직접 링크하여 호출할 수 있다. 로컬 HTTP 모드는
`transport: 'http'` / `transport="http"`로 명시하고 원격은 `connect()`를 사용한다.

stdio envelope는 `{ "id": "local-1", "op": "decide_batch", "body": {"requests": [...]} }`다.
`health`, `capabilities`, `decide`, `decide_batch`를 제공한다. 응답은 동일 id와
`result` 또는 `error`를 반환한다. stdout은 RPC 전용이며 native 로그는 stderr를 사용한다.
SDK는 미완료 호출을 최대 16개 허용한다. timeout·취소 후에도 응답 도착까지 슬롯을
유지하며 종료는 자식 프로세스를 정리한다. timeout은 native 실행 중단을 보장하지 않는다.

## native batch 계약

전체 wire 입력을 검증한 후 media 그룹을 평탄화하여 native batch 경계를 한 번
호출한다. Rust parallel 모드가 독립 sequence를 wave 단위로 처리한다.
요청별 state·media·ID·policy·failure messages를 보존하고 결과는 입력 순서다.
서로 다른 요청의 ID는 같아도 된다. 요청 정책은 해당 응답에만 적용하고 원시 점수는 유지한다.

성공 envelope는 `{api_version:1, request_id, execution:"native_parallel", responses:[...]}`다.
각 항목은 기존 v1 응답이고 `request_id`는 `batch-id/입력인덱스`다. stdio와 HTTP에서
같은 request/response 스키마를 쓰므로 TS·Python이 상호 호환된다.

- 최대 128개 요청, 총 128개 decisions, 전체 body 44 MiB.
- `parallel_width` / `parallelWidth`는 wave 폭을 정한다. 한 wave를 넘으면 여러 wave로 처리한다.
- 현재 llama.cpp native parallel은 direct reasoning, 질문별 최대 26개 선택지 대상이다.
- 모두 text이거나 모든 질문에 이미지가 하나씩 있는 배치와 일치하는 projector를 사용한다.
  text/image 그룹 혼합은 거부하며 recurrent/hybrid 모델의 기존 parallel 제약도 유지한다.
- `capabilities().batch`에 지원·활성화·media·reasoning·개수 한도를 표시한다.
- 지원하지 않는 backend는 `batch_unsupported`, parallel 미활성은 `batch_not_enabled`다.
  조용한 직렬 fallback이나 자동 재시도는 없다. 사용자 backend가 native batch 메서드를 구현해야 한다.
- timeout은 배치 전체에 적용한다. 실행 오류는 전체 배치를 실패 처리한다. 앞서 실행한
  wave의 추론은 되돌리거나 재실행하지 않는다. 항목별 성공/실패 API와는 별개다.

## 반복 호출과 성능 검증

`Promise.all(engine.decide(...))`와 `asyncio.gather(...)`는 별도 요청을 큐에 넣는다.
현재 stdio/HTTP 경로는 이들을 자동으로 합치지 않는다. 명시적 `decideBatch()` /
`decide_batch()`를 사용한다. 자동 합치기는 기존 bounded Rust 워커에 wire media·
reasoning·정책과 deadline을 연결하는 후속 작업이다.

`prepare()`는 정의·타입 재사용이며 token 사전 컴파일이나 영속 KV cache가 아니다.
state가 바뀌면 기존 `(state, decision)` 준비 캐시도 miss다. `SharedStateSession`은
state를 고정하고 질문을 바꾸는 별도 API다. native wave는 정확히 일치하는 token
prefix만 공유하고 질문별 suffix sequence를 분리한다.

batch shape가 바뀌면 확률·선택·abstention이 달라질 수 있다. 처리량·총 완료 시간·
p50/p95·메모리·점수 차이·top-1·정책 승인을 각각 검증해야 한다. Rust fixture의
stdio/HTTP 배치 및 TS↔Python JSON 검사는 모델 성능 증거가 아니다. 실제 GGUF CPU
smoke도 CUDA/Metal 성능이나 정확도 동등성을 검증하지 않는다.
