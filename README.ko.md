# L2S1 — LLM to System 1

**로컬 GGUF 모델의 점수를 타입이 있는 판단으로 바꿉니다.**

[English](README.md) · 한국어 · [日本語](README.ja.md) · [문서 색인](docs/ko/README.md) · [모델 측정 결과](docs/ko/MODEL_RESULTS.md)

L2S1은 로컬 채팅 모델로 이진·선택·서열 판단을 수행하는 Rust 라이브러리와 CLI입니다. JSON 상태, 질문, 후보별 판단 기준을 전달하면 타입이 있는 결과와 모델 점수를 반환합니다. 수락 정책을 충족하지 못하면 판단을 보류하고 그 이유를 알려줍니다.

메시지 분류, 요청 라우팅, 조건 확인, 단계별 수준 평가에 사용할 수 있습니다. 질문과 후보 ID는 애플리케이션이 요청마다 지정합니다. 호환되는 GGUF 모델을 바꿔도 요청과 결과 타입을 유지할 수 있습니다.

[Python SDK](python/README.md), [native 배치](docs/ko/BATCHING_API_REVIEW.md), [GitHub Release·npm·PyPI·Cargo 배포 파이프라인](docs/ko/RELEASE_PIPELINE.md)을 제공합니다.

## 빠른 시작

Rust 2024 에디션을 지원하는 최신 stable 툴체인, CMake 3.24 이상, C++17 컴파일러, 호환되는 chat/instruct GGUF가 필요합니다. 첫 네이티브 빌드에서는 고정된 llama.cpp 소스를 내려받습니다. 모델 가중치는 별도로 준비합니다.

```sh
git clone https://github.com/LuticaCANARD/L2S1.git
cd L2S1
cargo build --release --locked --features llama --bin l2s1

./target/release/l2s1 \
  --model /path/to/chat-model.gguf \
  --input examples/warehouse.json
```

[창고 예제 요청](examples/warehouse.json)은 같은 배송 상태를 바탕으로 보관 구역, 온도 관리 필요 여부, 출고 우선순위를 각각 판단합니다. 첫 번째 질문은 다음과 같은 구조입니다.

```json
{
  "state": { "storage_requirement": "chilled" },
  "decisions": [{
    "id": "storage_zone",
    "instruction": "Select the storage zone matching storage_requirement.",
    "kind": {
      "type": "choice",
      "options": [
        { "id": "ambient", "criterion": "Ambient storage is required." },
        { "id": "chilled", "criterion": "Chilled storage is required." },
        { "id": "frozen", "criterion": "Frozen storage is required." }
      ]
    }
  }]
}
```

[기록된 Gemma 4 CUDA 실행](examples/warehouse.gemma4.cuda.output.json)의 `storage_zone` 결과를 발췌하면 다음과 같습니다.

```json
{
  "id": "storage_zone",
  "value": { "type": "choice", "selected": "chilled" },
  "abstention_reasons": []
}
```

전체 CLI 응답에는 백엔드 정보, 수락 정책, 후보별 점수, 확률 질량, 토큰 사용량도 포함됩니다. 예측과 점수는 모델과 설정에 따라 달라집니다. 실제 작업의 정답이 있는 예제로 검증하세요.

## 주요 기능

- **타입이 있는 출력.** 이진 판단, 범주 선택, 서열 평가가 같은 요청 형식을 사용합니다. 호환되는 모델 사이에서도 의미를 나타내는 후보 ID가 유지됩니다.
- **로컬 모델 점수 직접 사용.** 모델의 어시스턴트 응답 경계에서 점수를 읽습니다. 후보가 많아 답변 코드가 여러 토큰으로 나뉘면 전체 코드의 우도를 계산합니다.
- **명시적인 판단 보류.** 선택을 보류해도 점수를 반환하고 이유를 설명합니다. 기본 정책은 후보 간 상대 확률과 전체 어휘에 대한 후보 확률 질량을 함께 확인합니다.
- **텍스트와 이미지.** 지원되는 비전 모델과 그에 맞는 `mmproj` GGUF를 사용해 정지 이미지도 판단할 수 있습니다.
- **Rust, TypeScript, CLI, HTTP.** 모델을 애플리케이션에 상주시켜 사용하거나, [TypeScript 패키지](docs/ko/typescript/README.md)로 Node.js에서 호출하거나, 파일·표준 입력 또는 HTTP API로 요청할 수 있습니다.
- **실행 정보 확인.** 모델 식별 정보, 요청 사전 검증, 실행 진단을 제공합니다. 캐시, 상태 복원, 병렬 실행, LoRA, 작업별 보정에는 각각 명시된 사용 조건이 있습니다.

## 판단 타입과 점수

| 타입 | 입력 | 로컬 결과 |
| --- | --- | --- |
| `binary` | 거짓·참에 대한 기준 | Boolean 또는 `null`, `p_true` |
| `choice` | 후보 ID와 기준 | 선택된 ID 또는 `null` |
| `ordinal` | 숫자 값이 증가하는 순서로 정렬된 수준 | 선택된 수준 ID 또는 `null`, 기댓값 |

각 질문은 독립적으로 평가됩니다. 모든 타입은 후보별 점수와 판단 보류 이유를 반환합니다. 후보가 26개 이하면 `A`–`Z`, 더 많으면 `AA`–`ZZ` 같은 고정 길이 코드를 사용합니다. 토큰화와 컨텍스트 제한은 선택한 모델을 기준으로 확인합니다.

기본 수락 여부는 두 점수를 사용합니다.

- `option_probability`: 주어진 후보 사이에서 계산한 상대 확률입니다.
- `candidate_mass`: 전체 어휘 또는 답변 코드 경로에서 후보에 할당된 확률 질량입니다.

기본 정책은 최상위 후보 확률 **0.8 이상**, 후보 확률 질량 **0.05 이상**, 최상위 후보 간 동점 없음을 요구합니다. 조건을 충족하지 못하면 선택값은 `null`입니다. 이 수치는 모델 점수이며, 정답일 확률로 보정된 값은 아닙니다. 선택적으로 적용하는 보정은 특정 모델·설정·작업에 연결됩니다.

수식, 답변 코드, 검증 규칙은 [판단 계약](docs/ko/GUIDE.md#the-decision-contract)을 참고하세요.

## 백엔드와 하드웨어

| 백엔드 | 모델과 입력 | 하드웨어 | 근거 데이터 |
| --- | --- | --- | --- |
| `llama` / `llama-cuda` / `llama-metal` | 호환되는 GGUF 채팅 모델; 비전은 대응하는 프로젝터 필요 | CPU, NVIDIA CUDA, Apple Metal | 로컬 모델 점수 |
| `wgpu` | Gemma 4 텍스트 GGUF, 선택적으로 대응하는 비전 프로젝터 | 네이티브 wgpu GPU 어댑터 | 로컬 모델 점수 |
| `openrouter` | 요청한 입력 형식을 지원하는 제공자 모델 | 원격 API | 선택 결과만 반환; 확률 없음 |

기본 Cargo feature는 비어 있습니다. 로컬 CLI를 사용하려면 `llama`를 활성화합니다. 모델은 고정된 런타임에서 지원되어야 하며, 사용할 수 있는 채팅 템플릿이 있고 선택한 장치와 컨텍스트에 들어가야 합니다. wgpu 어댑터는 Gemma 4 전용입니다. 모델을 바꾸면 예측, 지연 시간, 토큰화, 보정도 달라질 수 있습니다.

로컬 GPU 추론은 해당 feature로 빌드한 다음 장치를 지정합니다.

```sh
# NVIDIA CUDA: CUDA toolkit이 필요합니다.
cargo build --release --locked --features llama-cuda --bin l2s1
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --device cuda --input examples/warehouse.json

# macOS Metal: Metal compiler를 포함한 Xcode 툴체인이 필요합니다.
cargo build --release --locked --features llama-metal --bin l2s1
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --device metal --input examples/warehouse.json
```

기본 장치는 CPU입니다. GPU를 명시적으로 지정했다면 해당 장치가 사용 가능해야 합니다. 첫 빌드에서는 네이티브 소스를 내려받으며, 오프라인 빌드는 `L2S1_LLAMA_CPP_SOURCE=/path/to/llama.cpp`로 소스 위치를 지정할 수 있습니다. 네이티브 라이브러리, 배포 패키징, CUDA 아키텍처별 빌드는 [빌드 가이드](docs/ko/GUIDE.md#build)에 설명되어 있습니다.

[wgpu](docs/ko/GUIDE.md#optional-wgpu-backend)는 별도 실행 파일인 `l2s1-wgpu`를 사용합니다. [OpenRouter](docs/ko/GUIDE.md#openrouter-adapter)는 `l2s1-openrouter`와 `OPENROUTER_API_KEY`를 사용합니다. OpenRouter도 공통 타입을 가진 HTTP 응답 구조를 사용하지만, 근거는 `selection_only`이며 로컬 확률 수락 정책을 적용하지 않습니다.

## 확인과 실행

```sh
# 로드한 모델과 기능을 확인합니다.
./target/release/l2s1 --model /path/to/chat-model.gguf --inspect

# 추론 없이 실제 요청을 사전 검증합니다.
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --input examples/warehouse.json --preflight

# 실행 진단 정보를 함께 받습니다.
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --input examples/warehouse.json --diagnostics
```

`--input -`는 표준 입력을 읽습니다. 결과는 stdout, 네이티브 로그는 stderr로 출력됩니다. 설정한 컨텍스트를 초과하는 입력은 잘라내지 않고 오류로 처리합니다. 컴퓨팅 설정과 구조화된 실패 정보는 [모델 확인과 진단](docs/ko/GUIDE.md#inspect-validate-and-run)을 참고하세요.

## HTTP와 이미지 입력

로컬 서버를 시작합니다.

```sh
./target/release/l2s1 --model /path/to/chat-model.gguf \
  --listen 127.0.0.1:8080
```

다른 터미널에서 요청을 보냅니다.

```sh
curl -sS -H 'Content-Type: application/json' \
  --data-binary @examples/warehouse.json \
  http://127.0.0.1:8080/v1/decisions
```

서버는 `POST /v1/decisions`, `GET /v1/capabilities`, `GET /healthz`를 제공합니다. HTTP 응답에는 API 버전, 요청 ID, 백엔드, 수락 정책, 근거 데이터를 포함한 타입별 결과가 담깁니다. 루프백에 바인딩하거나 원격 접근을 위해 인증된 리버스 프록시를 사용하세요.

정지 이미지를 전달하려면 비전 모델과 그에 맞는 프로젝터를 함께 로드합니다.

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/projector.gguf --image photo.jpg \
  --input examples/warehouse.json
```

HTTP 이미지 요청은 이름이 있는 `media`와 질문별 `media_ids`를 사용합니다. 로컬 백엔드는 질문 하나당 이미지 한 장을 받습니다. 요청 구조, 제한, 백엔드별 동작은 [이미지와 HTTP 계약](docs/ko/GUIDE.md#direct-image-input-and-http-api)에 설명되어 있습니다.

## TypeScript에서 사용하기

[`@l2s1/node`](docs/ko/typescript/README.md)는 현재 OS·CPU에 맞는 사전 빌드 Rust 런타임을 선택하며, 타입이 있는 `load()`, `decide()`, `capabilities()`, `close()`를 제공합니다. 빌드 워크플로가 만드는 래퍼·런타임 tarball을 설치합니다. npm에는 아직 배포하지 않았으며 GGUF 가중치는 별도로 준비합니다. 같은 애플리케이션 API에서 `connect()`는 HTTP 서버를, `fromBackend()`는 사용자 정의 백엔드를 사용합니다.

Node.js 24 이상과 Bash 또는 Zsh에서 저장소 루트를 기준으로 실행합니다. 로컬 CPU 런타임과 TypeScript 패키지를 빌드하고, 예제의 타입을 검사한 뒤 [창고 분류 예제](typescript/examples/warehouse.ts)를 실행합니다. 모델 경로는 준비한 GGUF 파일의 절대 경로로 바꾸세요. `npm pack`은 `typescript/l2s1-node-0.1.0.tgz`를 생성합니다.

```sh
cargo build --release --locked --features llama --bin l2s1
cd typescript
npm ci
npm run build
npm run check
L2S1_BINARY=../target/release/l2s1 \
  node examples/warehouse.ts /absolute/path/to/chat-model.gguf
npm pack
cd ..
```

```ts
import { L2S1 } from '@l2s1/node';

const engine = await L2S1.load({
  model: '/path/to/chat-model.gguf',
});
try {
  const response = await engine.decide({
    state: { x: 1 },
    decisions: [{ id: 'positive', instruction: 'Is x positive?',
      kind: { type: 'binary', false_label: 'x <= 0', true_label: 'x > 0' } }],
  });
  console.log(response.results);
} finally { await engine.close(); }
```

실행 중인 서버나 브라우저 앱에서는 `@l2s1/node/http`의 `L2S1Client`를 사용합니다. 설치, 이미지, 추론 모드, 오류, 배포 조건은 [패키지 안내](docs/ko/typescript/README.md)를 참고하세요.

## Rust에서 사용하기

`l2s1` 의존성의 `llama` feature를 활성화합니다. 모델은 한 번 로드하고 반복 요청에 재사용합니다.

```rust
use std::path::Path;
use l2s1::{
    Decision, DecisionBackend, DecisionKind, DecisionPolicy, DecisionRequest,
    Level, OptionSpec, llama::LlamaBackend,
};
use serde_json::{Map, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = DecisionRequest {
        state: Value::Object(Map::from_iter([
            ("shipment_id".into(), Value::from("BOX-103")),
            ("storage_requirement".into(), Value::from("chilled")),
            ("hours_until_dispatch".into(), Value::from(4)),
        ])),
        decisions: vec![
            Decision {
                id: "storage_zone".into(),
                instruction: "Select the storage zone that matches the shipment's storage_requirement.".into(),
                kind: DecisionKind::Choice {
                    options: vec![
                        OptionSpec {
                            id: "ambient".into(),
                            criterion: "The shipment requires ambient storage.".into(),
                        },
                        OptionSpec {
                            id: "chilled".into(),
                            criterion: "The shipment requires chilled storage.".into(),
                        },
                        OptionSpec {
                            id: "frozen".into(),
                            criterion: "The shipment requires frozen storage.".into(),
                        },
                    ],
                },
            },
            Decision {
                id: "cold_chain_required".into(),
                instruction: "Does this shipment need temperature-controlled storage? Chilled and frozen shipments do; ambient shipments do not.".into(),
                kind: DecisionKind::Binary {
                    false_label: "No temperature control is required.".into(),
                    true_label: "Temperature control is required.".into(),
                },
            },
            Decision {
                id: "dispatch_priority".into(),
                instruction: "Choose the priority using hours_until_dispatch and the exact thresholds in the levels.".into(),
                kind: DecisionKind::Ordinal {
                    levels: vec![
                        Level {
                            id: "low".into(),
                            criterion: "More than 24 hours remain until dispatch.".into(),
                            value: 0.0,
                        },
                        Level {
                            id: "medium".into(),
                            criterion: "More than 6 hours and at most 24 hours remain until dispatch.".into(),
                            value: 1.0,
                        },
                        Level {
                            id: "high".into(),
                            criterion: "At most 6 hours remain until dispatch.".into(),
                            value: 2.0,
                        },
                    ],
                },
            },
        ],
    };
    request.validate()?;
    let mut backend = LlamaBackend::load(
        Path::new("/path/to/chat-model.gguf"),
        2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    let response = backend.decide(&request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    Ok(())
}
```

`serde_json`을 의존성에 추가합니다. 요청은 Rust 구조체로 직접 작성하므로 입력 파일이 필요하지 않습니다. 같은 요청을 만들고 검증하는 [실행 가능한 예제](examples/warehouse.rs)는 모델 없이 `cargo run --locked --example warehouse`로 실행할 수 있습니다. 전용 스레드에서 백엔드를 소유하고 요청 수를 제한하려면 `BackendWorker`를 사용합니다. Metal에서는 프로세스 종료 전에 백엔드를 해제해야 합니다. worker를 사용한다면 `close()`를 호출하고 소유 스레드의 종료를 기다리세요. 수명 관리와 네이티브 링크 설정은 [Rust 통합 가이드](docs/ko/GUIDE.md#rust-integration)를 참고하세요.

## 실행 모드와 비전 처리량 최적화

llama.cpp 백엔드는 네 가지 실행 모드를 지원합니다.

| 모드 | 용도 |
| --- | --- |
| `fresh` | 빈 시퀀스 상태에서 독립적으로 평가; 기본값 |
| `prefix-reuse` | 한 요청 안에서 정확히 일치하는 공통 토큰 접두사 재사용 |
| `state-restore` | 공통 접두사 상태를 저장하고 독립적인 후속 입력 전에 복원 |
| `parallel` | 독립적인 질문을 분리된 시퀀스로 배치 처리 |

GPU 비전 작업에서 `--vision-optimized`를 사용하면 디코더 스트림 4개, 동적 컨텍스트 할당, Flash Attention, compact 근거 전달, 크기가 제한된 준비 캐시, 동일 이미지의 프로젝터 결과 재사용을 함께 활성화합니다.

```sh
./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/projector.gguf --device cuda --vision-optimized \
  --image photo.jpg --input examples/warehouse.json
```

병렬 실행과 비전 프로파일은 정식 지원 기능이며, `fresh`와 다른 수치 계산 경로를 사용하므로 점수와 선택 결과가 달라질 수 있습니다. 병렬 모드는 recurrent/hybrid 모델을 지원하지 않으며, 병렬 비전은 후보 26개 이하를 지원합니다. 비전 프로파일에는 CUDA 또는 Metal과 호환되는 GPU 커널이 필요합니다. Metal 처리 성능은 아직 측정되지 않았습니다. 사용하는 체크포인트에서 작업 품질과 수락 비율을 확인하세요. 실제 모델이 필요한 테스트는 명시적으로 실행해야 하며 가중치를 자동으로 내려받지 않습니다. [실행과 메모리](docs/ko/GUIDE.md#execution-and-memory), [비전 최적화](docs/ko/GUIDE.md#optimized-vision), [검증 안내](docs/ko/VERIFICATION.md)를 참고하세요.

[이미지·텍스트 데모](https://n2s1.luticalab.net/demo)에서 실제 모델 실행 기록을 확인하고, 로컬 텍스트 추론의 즉시 판단·생각 후 판단 모드와 수락 기준·실패 설명을 지정할 수 있습니다. [데모 실행](docs/ko/IMAGE_DEMO.md) · [추론 계약](docs/ko/REASONING.md). Pages에서는 기록을 제공하며 새 입력 추론에는 안내된 로컬 네이티브 서버가 필요합니다.

[브라우저 WebGPU 데모](https://n2s1.luticalab.net/webgpu)는 Qwen3 0.6B ONNX를 선택적으로 내려받아 내 텍스트를 브라우저에서 직접 판단합니다. 즉시 판단·생각 후 판단, 수락 기준·실패 설명을 지정할 수 있습니다. WebGPU 어댑터가 필요하며 모델 크기는 q4f16 543.4 MiB, q4 876.5 MiB입니다. [실행과 측정 범위](docs/ko/WEBGPU_DEMO.md).

## 기록된 측정 결과

| 측정 | 기록된 범위 | 보고서 |
| --- | --- | --- |
| JevBench 공개 부분집합 | 최초 비교: 체크포인트 22개 × 항목 231개; 유효한 예측 5,082개 | [모델 결과](docs/ko/MODEL_RESULTS.md), [측정 방법](docs/ko/JEVBENCH.md) |
| 의도 분류 | 영어 레이블 77개와 한국어 레이블 60개; 체크포인트마다 언어별 예제 200개 | [의도 분류 벤치마크](docs/ko/INTENT_BENCHMARK.md) |
| typed-decisions | 전체 테스트: 모델별 400개 사례·2,000개 판단; Gemma 4 E2B 54.30%, Qwen3 0.6B 31.25% 원시 정확도 | [방법과 결과](docs/ko/TYPED_DECISIONS_BENCHMARK.md) |
| 비전 판단 | 정지 이미지 분류와 실행 모드 비교 | [비전 벤치마크](docs/ko/VISION_BENCHMARK.md), [TrashNet 연구](docs/ko/benchmarks/trashnet-vision-20260925/REPORT.md) |

명시된 리비전, 하드웨어, 설정에서 수행한 로컬 실험 기록입니다. 전체 정확도, 수락한 판단의 정확도, 수락 비율은 따로 확인해야 합니다. 일부 원시 결과는 로컬에만 보관되어 Git에서 제외되어 있으며, 각 보고서에 위치와 재현 절차가 명시되어 있습니다. JevBench 공개 부분집합의 결과는 공식 전체 평가 점수나 순위가 아닙니다.

## 문서

AI Agent용 [L2S1 skill](docs/ko/skills/l2s1/SKILL.md)과 stdio MCP 어댑터를 제공합니다.
MCP에서 문서 조회, 요청 형식 검증, 상주 HTTP 백엔드의 추론을 사용할 수 있습니다.
설치와 연결 방법은 [에이전트 연동 가이드](docs/ko/AGENT_INTEGRATION.md)를 참고하세요.

아래 상세 가이드는 영어로 제공됩니다.

| 주제 | 문서 |
| --- | --- |
| 빌드, 요청, Rust API, HTTP, 실행 옵션 | [사용 가이드](docs/ko/GUIDE.md) |
| 모델 식별 정보, 사전 검증, 보정, worker 소유권 | [모델 교체와 계약](docs/ko/MODEL_INTERCHANGEABILITY.md) |
| 모델별 테스트 명령 | [검증 안내](docs/ko/VERIFICATION.md) |
| 접두사 재사용과 병렬 실행 | [접두사 알고리즘](docs/ko/SEMIF_ALGORITHM.md), [병렬 실행](docs/ko/PARALLEL_EXECUTION.md) |
| 학습을 통한 특화 | [LoRA 학습](docs/ko/DECISION_FINETUNE.md), [출력 헤드](docs/ko/OUTPUT_HEAD.md) |
| 데이터 준비와 보고서 도구 | [l2s1-tools](docs/ko/crates/l2s1-tools/README.md) |
| 모델 비교 기록과 측정 한계 | [모델 결과](docs/ko/MODEL_RESULTS.md) |

## 저장소와 개발

| 경로 | 역할 |
| --- | --- |
| [`src/decision.rs`](src/decision.rs) | 타입별 요청·결과, 공통 점수 계산, 수락 정책 |
| [`src/llama.rs`](src/llama.rs), [`src/wgpu.rs`](src/wgpu.rs) | 로컬 추론 백엔드 |
| [`src/http.rs`](src/http.rs), [`src/http/contract.rs`](src/http/contract.rs) | HTTP 서버와 버전이 있는 통신 계약 |
| [`crates/l2s1-llama-sys/`](crates/l2s1-llama-sys) | 고정된 llama.cpp 빌드와 네이티브 브리지 |
| [`crates/l2s1-tools/`](crates/l2s1-tools) | 데이터와 벤치마크 도구 |
| [`examples/`](examples) | 요청, 저장된 출력, 통합 예제 |
| [`docs/`](docs/README.md) | 상세 가이드, 설계 설명, 평가 보고서 |
| [`web/`](web) | Svelte 문서 사이트 |

순수 Rust 검증은 `cargo test --locked`로 실행합니다. 네이티브와 모델별 검증은 [VERIFICATION.md](docs/ko/VERIFICATION.md)에 정리되어 있습니다. 버그를 보고할 때는 실행 명령, 체크포인트·양자화, 런타임·장치, 오류 내용을 포함해 [GitHub 이슈](https://github.com/LuticaCANARD/L2S1/issues)를 작성하세요. 문서를 수정할 때는 영어·한국어·일본어 README의 내용을 함께 맞춰주세요.

## 라이선스

L2S1 소스는 [MIT 라이선스](LICENSE)입니다. 모델 가중치는 각자의 라이선스를 따르며 저장소에 포함되지 않습니다. 네이티브 구성 요소를 배포할 때는 [서드파티 고지](THIRD_PARTY_LICENSES.txt)를 유지해야 합니다. 자세한 내용은 [LICENSING.md](docs/ko/LICENSING.md)를 참고하세요.
