<a id="l2s1-for-python"></a>
# Python용 L2S1

[English](../../en/python/README.md) · [한국어](README.md) · [日本語](../../ja/python/README.md)

Python 3.11+ 비동기 SDK입니다. `@l2s1/node`와 같은 Rust 상주 엔진·HTTP v1 JSON을
사용하며, Pydantic 런타임 검증과 PEP 561 정적 타입 정보를 제공합니다.
저장소·wheel 배포를 준비했으며 PyPI에는 게시하지 않았습니다.

<a id="install"></a>
## 설치

저장소 루트에서 설치하거나 빌드한 wheel을 사용합니다.

```sh
python -m pip install ./python
python -m pip install ./python/dist/l2s1-0.1.0-py3-none-any.whl
```

wheel은 Python SDK를 포함합니다. Rust 실행 파일과 GGUF는 별도로 제공합니다.
`load()`에는 `binary_path`를 지정하거나 PATH의 `l2s1`을 사용합니다.
`runtime_dir`에는 압축 해제한 `@l2s1/runtime-<platform>` 디렉토리를 지정할 수
있습니다. 버전·OS·CPU·장치·파일 체크섬을 검사하고 동일한 실행 파일과 네이티브
라이브러리를 사용합니다. Node.js는 필요하지 않으며 자동 다운로드·컴파일은 없습니다.

<a id="repeat-a-decision-with-new-data"></a>
## 데이터만 바꿔 반복 호출

```python
import asyncio
from pydantic import BaseModel, ConfigDict
from l2s1 import BinaryKind, BinaryValue, Decision, L2S1, LoadOptions

class Temperature(BaseModel):
    model_config = ConfigDict(strict=True)
    temperature_c: float

async def main() -> None:
    async with await L2S1.load(LoadOptions(
        model="/path/to/model.gguf", binary_path="/path/to/l2s1",
        execution_mode="parallel",
    )) as engine:
        plan = engine.prepare([Decision(
            id="cold", instruction="Is temperature_c below 10?",
            kind=BinaryKind(false_label="At least 10.", true_label="Below 10."),
        )], state_type=Temperature)
        first = await plan.decide(Temperature(temperature_c=6))
        second = await plan.decide(Temperature(temperature_c=15))
        results = await plan.decide_batch([
            Temperature(temperature_c=2), Temperature(temperature_c=20),
        ])
        result = first.results[0]
        if isinstance(result.value, BinaryValue):
            print(result.value.value)  # bool | None

asyncio.run(main())
```

`prepare()`는 고정 질문을 복사해 보관합니다. `state_type`을 지정하면
`PreparedDecision[Temperature]`를 반환하며 mypy가 호출 타입을 확인하고 SDK가
런타임 모델 타입을 확인합니다. 생략하면 JSON 값을 받습니다. 입력 검증은 모델의
정답을 보장하지 않습니다. 여기서 준비한다는 것은 토큰 컴파일이나 KV 재사용이 아닙니다.

`load()`는 기본적으로 `transport="stdio"`를 사용합니다. 빌드된 Rust 실행 파일을
한 번 로드하고 stdin/stdout JSON으로 통신하므로 포트·HTTP 서버·Node.js가 필요
없습니다. PyO3로 같은 프로세스에 로드하는 방식은 아닙니다. 로컬 HTTP 연결은
`transport="http"`, 공유 서버는 `connect()`를 사용합니다.

`decide_batch()`는 요청 배열 전체를 한 번에 Rust native parallel 실행기에 전달합니다.
`execution_mode="parallel"`로 로드하고 `parallel_width`로 wave 폭을 지정합니다.
HTTP에서는 `POST /v1/decision-batches`와 `{"requests": [...]}`를 사용합니다.
각 요청의 state·ID·media·정책을 분리하고 입력 순서로 반환합니다. 서로 다른 요청에
같은 ID를 사용할 수 있습니다. 응답 envelope는 `execution: "native_parallel"`을
표시하고 `timeout_ms`는 배치 전체에 적용됩니다. 직렬 fallback과 자동 재시도는
없습니다. 지원하지 않으면 `batch_unsupported`, parallel 모드가 꺼져 있으면
`batch_not_enabled` 오류가 납니다. 사용자 native adapter는 `BatchDecisionBackend`를 구현합니다.

최대 128개 요청·총 128개 decisions이며 현재 native parallel은 direct reasoning과
질문별 최대 26개 선택지를 지원합니다. 모두 text이거나 모든 질문에 이미지가 하나씩
있는 배치와 일치하는 projector를 사용합니다. text/image 혼합은 거부합니다.
전체 wire 입력을 검증한 뒤 실행하며 실패 시 전체 배치를 실패 처리합니다. 이미
실행한 wave는 되돌리거나 재실행하지 않습니다. `asyncio.gather(decide(...))`는
자동 배치가 아닙니다. [배치 API 검토](../BATCHING_API_REVIEW.md)를 참고하세요.

<a id="connect-send-existing-json-or-adapt-another-transport"></a>
## HTTP 연결과 사용자 정의 백엔드

```python
from l2s1 import DecisionRequest, L2S1

async with L2S1.connect("http://127.0.0.1:8080") as engine:
    capabilities = await engine.capabilities()
    request = DecisionRequest.model_validate(payload)
    response = await engine.decide(request, timeout_ms=180_000)
    payload_for_typescript = response.model_dump(exclude_unset=True)
```

`payload`는 TypeScript가 사용하는 JSON과 같습니다. `load/connect/from_backend`는
같은 애플리케이션 API를 제공합니다. `DecisionBackend` Protocol은 비동기
`decide/capabilities`를 요구합니다. 선택적 동기·비동기 `close()`가 있으면 facade가
리소스 소유권을 맡습니다. 독립 HTTP 전송용 `L2S1Client`도 제공합니다.

| TypeScript | Python |
| --- | --- |
| `load({ model, binaryPath })` | `await load(LoadOptions(model=..., binary_path=...))` |
| `connect({ baseUrl })` | `connect(base_url)` |
| `fromBackend(backend)` | `from_backend(backend)` |
| `decide(request)` | `await decide(request)` |
| `prepare<State>(decisions)` | `prepare(decisions, state_type=State)` |
| `plan.decide(state)` | `await plan.decide(state)` |
| `decideBatch(items)` | `await decide_batch(items)` |
| async disposal | `async with` / `await close()` |
| `AbortSignal` | asyncio task 취소 |

Python 메서드·옵션은 snake case이고 JSON 키는 같습니다. `false`, `null`, 결과 순서·ID,
`model_scored/selection_only` 구분을 보존합니다. `model_dump(exclude_unset=True)`로
생략된 확장 필드를 유지하며 판단 종류의 기본 `type`은 항상 직렬화합니다.

이미지·reasoning·요청 정책은 `capabilities()`에서 지원을 확인하세요. 요청한
제어를 조용히 버리지 않습니다. 점수·`target_error_rate`는 정답률 보장이 아닙니다.
`L2S1Error`는 서버 `code/status/request_id/user_reason`을 보존하고 네트워크·취소는
HTTPX/asyncio 예외를 보존합니다. 자동 재시도는 없습니다.

`close()`는 멱등적입니다. HTTP 클라이언트 종료는 로컬 요청을 취소하고 원격 서버를
계속 유지합니다. 소유 엔진 종료는 Rust 자식 프로세스를 종료하고 기다립니다.
호출 취소·timeout은 네이티브 추론 중단을 보장하지 않습니다. 기본 stdio는 포트를
열지 않고 stderr를 계속 읽습니다. 명시적 HTTP만 loopback 포트를 사용합니다.
stdio는 최대 16개의 미완료 호출을 허용하며 timeout 후에도 native 응답까지 슬롯을 유지합니다.

<a id="build-and-verify"></a>
## 빌드와 검증

```sh
python -m pip install './python[dev]'
python -m mypy --config-file python/pyproject.toml python/src python/examples python/tests/typecheck.py
python -m unittest discover -s python/tests -v
python -m build python
cargo build --locked --example typescript_fixture
npm --prefix typescript ci
npm --prefix typescript run build
L2S1_TEST_BINARY="$PWD/target/debug/examples/typescript_fixture" \
  python -m unittest discover -s python/tests -v
```

Windows에서는 실행 파일에 `.exe`를 붙입니다. Rust fixture는 실제 프로세스·stdio/HTTP 배치·
점수 계산과 Python↔TypeScript JSON 동등성을 확인하며 모델을 사용하지 않습니다.
GGUF/CUDA/Metal 품질·성능 검증은 별도입니다. CI는 Linux·macOS·Windows에서
Python 3.11/3.14를 대상으로 wheel/sdist를 업로드합니다. 로컬 성공은 다른 플랫폼
성공을 뜻하지 않으며 레지스트리 업로드는 켜지 않았습니다.

[배포 파이프라인](../RELEASE_PIPELINE.md)은 GitHub Release·npm·PyPI·Cargo에 검증한 설치 파일을 게시합니다. 외부 계정·trusted publisher 설정은 별도로 필요하며 현재 작업에서 실제 게시하지 않았습니다.
