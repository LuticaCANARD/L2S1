<a id="build-and-interfaces"></a>
# 빌드와 인터페이스

[English](../../../../en/skills/l2s1/references/interfaces.md) · [한국어](interfaces.md) · [日本語](../../../../ja/skills/l2s1/references/interfaces.md)

[English index](../../../../en/README.md) · [한국어 색인](../../../README.md) · [日本語索引](../../../../ja/README.md)

확인된 L2S1 체크아웃에서 저장소 명령을 실행하세요. 모델 경로는 사용자·앱이 준비하며 가중치는 포함하지 않습니다. Rust는 2024 에디션 지원, 네이티브 빌드는 CMake 3.24 이상과 C++17이 필요합니다. 플래그를 바꿀 때 CLI `--help`를 확인하세요.

```sh
cargo build --release --locked --features llama --bin l2s1
./target/release/l2s1 --model /path/to/model.gguf --inspect
./target/release/l2s1 --model /path/to/model.gguf --input request.json --preflight
./target/release/l2s1 --model /path/to/model.gguf --input request.json --diagnostics
# Or pipe a text request to --input -; JSON stdout, native logs stderr.
```

GPU 빌드는 `llama-cuda`·`llama-metal`로 바꾸고 `--device cuda`·`--device metal`로 실행합니다. 오프라인 빌드는 `L2S1_LLAMA_CPP_SOURCE=/path/to/matching/llama.cpp`를 설정합니다. 소스·헤더·네이티브 브리지·라이브러리는 함께 다시 빌드해야 하며 임의 런타임 라이브러리로 바꾸면 안 됩니다. 네이티브 로그에 쓰기 불가능한 ccache가 나오면 쓰기 가능한 `CCACHE_DIR` 또는 `CCACHE_DISABLE=1`을 사용합니다. Linux 로더는 대응하는 생성 라이브러리 디렉토리를 `LD_LIBRARY_PATH`에 요구할 수 있습니다. `docs/GUIDE.md`를 참고하세요.

<a id="resident-http"></a>
## 상주 HTTP

```sh
./target/release/l2s1 --model /path/to/model.gguf --listen 127.0.0.1:8080
curl -sS http://127.0.0.1:8080/v1/capabilities
curl -sS -H 'Content-Type: application/json' --data-binary @request.json \
  http://127.0.0.1:8080/v1/decisions
```

상태 확인은 `GET /healthz`입니다. 인증 없는 네이티브 HTTP는 루프백에 유지합니다. 원격 사용은 앱의 인증된 리버스 프록시를 제공합니다. 이미지는 `--mmproj /path/to/matching-projector.gguf`로 백엔드를 시작합니다. HTTP는 base64 미디어, CLI는 `--image /path/to/photo.jpg`를 사용합니다. 텍스트 CLI 사전 검증은 HTTP 이미지를 검증하지 않습니다.

<a id="mcp"></a>
## MCP

Python 3.10 이상 가상환경에 `mcp/requirements.txt`를 설치하고 그 환경의 Python으로 `/path/to/L2S1/mcp/server.py`를 실행합니다. `--backend-url http://127.0.0.1:8080`을 사용합니다. 서버가 저장소에 있으면 `--root /path/to/L2S1`은 선택 사항입니다. Stdio MCP는 별도 어댑터 프로세스이며 네이티브 HTTP 엔드포인트가 아닙니다. `/v1/decisions`를 Streamable HTTP MCP URL로 등록하지 마세요.

도구는 `l2s1_document`, `l2s1_example`, `l2s1_validate`, `l2s1_capabilities`, `l2s1_decide`입니다. 리소스에는 `l2s1://schema/request`, `l2s1://docs/guide`, `l2s1://examples/warehouse`가 있습니다. `design_decision`은 작업을 유효한 요청으로 만드는 데 도움을 줍니다. 클라이언트 설정은 `docs/AGENT_INTEGRATION.md`를 참고하세요. 어댑터는 운영자가 선택한 origin에 연결하며 셸 명령, 파일 쓰기, 다운로드, 모델 로드, 자동 재시도를 하지 않습니다.

<a id="rust"></a>
## Rust

`l2s1` 의존성에 대응 런타임 feature를 활성화하고 `serde_json`을 추가합니다. Rust 타입으로 직접 요청을 구성하며 입력 파일이나 JSON 파싱은 필요하지 않습니다. 한 번 로드해 여러 판단에 백엔드를 유지합니다.

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
        Path::new("/path/to/model.gguf"),
        2048, 256, 4, false, DecisionPolicy::default(),
    )?;
    let response = backend.decide(&request)?;
    println!("{}", serde_json::to_string_pretty(&response)?);
    drop(backend);
    Ok(())
}
```

GPU·비전·옵션·워커 소유권은 `docs/GUIDE.md`와 `docs/MODEL_INTERCHANGEABILITY.md`의 최신 API를 사용합니다. 보정과 출력 헤드는 모델·작업·설정에 연결되며 GGUF 변경 후 유효성이 유지되지 않습니다.
