<a id="ai-agent-integration"></a>
# AI 에이전트 통합

[English](../en/AGENT_INTEGRATION.md) · [한국어](AGENT_INTEGRATION.md) · [日本語](../ja/AGENT_INTEGRATION.md)

[English index](../en/README.md) · [한국어 색인](README.md) · [日本語索引](../ja/README.md)

L2S1은 범용 [스킬](skills/l2s1/SKILL.md)과 stdio MCP 어댑터를 제공합니다. 스킬은 요청 설계, Rust·CLI·HTTP 통합 선택, 실제 모델 검증, 판단 보류와 근거 보존 방법을 안내합니다. MCP는 관리되는 저장소 문서와 타입이 있는 도구를 공개하므로, 에이전트가 라이브러리를 살펴보거나 실행 중인 백엔드를 호출할 때 셸 접근이 필요하지 않습니다.

<a id="install-the-skill"></a>
## 스킬 설치

`skills/l2s1` 전체 디렉토리를 에이전트의 스킬 디렉토리에 복사하세요. `.agents/skills`에서 프로젝트 스킬을 찾는 에이전트라면 L2S1 체크아웃에서 다음을 실행합니다.

```sh
mkdir -p /path/to/consumer-project/.agents/skills
cp -R skills/l2s1 /path/to/consumer-project/.agents/skills/l2s1
```

`references/`와 `agents/`를 `SKILL.md`와 함께 유지하세요. 다른 스킬이 없는 대상을 선택하고 기존 사본을 교체하기 전 확인하세요. 이 스킬은 자동 검색이 활성화되어 있으며, 명명된 스킬을 지원하는 클라이언트에서 `$l2s1`로 호출할 수도 있습니다. MCP 없이도 Rust, CLI, HTTP 인터페이스로 동작합니다. 배포용 원본은 저장소의 `skills/l2s1`에 있고 에이전트 설치는 별도 단계입니다.

<a id="install-and-connect-mcp"></a>
## MCP 설치와 연결

어댑터는 [공식 MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)의 v1 API와 [stdio 전송](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)을 사용합니다. Python 3.10 이상이 필요합니다. 의존성은 네이티브 Rust 빌드와 분리됩니다.

```sh
python3 -m venv /path/to/l2s1-mcp-venv
/path/to/l2s1-mcp-venv/bin/python -m pip install -r /path/to/L2S1/mcp/requirements.txt
```

stdio 지원 MCP 클라이언트에 다음 명령을 등록하세요. 절대 경로를 수정하면 작업 디렉토리에 의존하지 않습니다. `mcpServers` JSON 설정을 사용하는 클라이언트는 다음을 사용할 수 있습니다.

```json
{
  "mcpServers": {
    "l2s1": {
      "command": "/path/to/l2s1-mcp-venv/bin/python",
      "args": [
        "/path/to/L2S1/mcp/server.py",
        "--backend-url", "http://127.0.0.1:8080"
      ]
    }
  }
}
```

클라이언트가 어댑터를 시작합니다. `python /path/to/L2S1/mcp/server.py --help`로 명령을 확인할 수 있습니다. `--help` 없이 실행하면 표준 입력의 MCP 메시지를 기다립니다. 표준 출력은 프로토콜 메시지 전용입니다.

문서, 예제, 스키마, 오프라인 검증은 바로 사용할 수 있습니다. 추론은 네이티브 HTTP 백엔드를 별도로 실행합니다. CPU 예시는 다음과 같습니다.

```sh
cargo build --release --locked --features llama --bin l2s1
./target/release/l2s1 --model /path/to/model.gguf --listen 127.0.0.1:8080
```

백엔드가 모델을 소유하며 호출 사이에도 상주합니다. 필요하면 `llama-cuda`와 `--device cuda`, 또는 `llama-metal`과 `--device metal`을 사용하세요. 이미지에는 대응하는 `--mmproj`도 지정합니다. 어댑터의 기본 주소는 `http://127.0.0.1:8080`, 요청 시간 제한은 180초이며 `--backend-url`과 `--timeout`으로 설정합니다. 복사한 서버는 `--root /path/to/L2S1`을 사용할 수 있습니다. 관리되는 문서와 예제를 제공하려면 소스 체크아웃이 필요합니다.

네이티브 API는 JSON HTTP이며 **Streamable HTTP MCP 엔드포인트가 아닙니다.** 어댑터는 백엔드 시작, 가중치 다운로드, 선택한 행동의 실행, 실패한 추론 재시도를 하지 않습니다. 설정된 origin에만 연결하며 HTTP 프록시나 자격 증명을 상속하지 않습니다. 네이티브 HTTP는 루프백에 두세요. 이 어댑터는 원격 인증을 제공하지 않습니다. OpenRouter를 사용하는 백엔드는 상태와 미디어를 원격 제공자에게 보내며 요금이 발생할 수 있습니다.

<a id="agent-interface"></a>
## 에이전트 인터페이스

| 도구 | 동작 |
| --- | --- |
| `l2s1_document(name)` | 허용 목록의 `overview`, `guide`, `models`, `verification`, `parallel`, `tools`, `agents`, `skill` 읽기 |
| `l2s1_example(name)` | `warehouse` 텍스트 요청 또는 `image` HTTP 템플릿 가져오기 |
| `l2s1_validate(request)` | 모델 없이 v1 요청 구조, ID, 서열 순서, 미디어 참조, 크기, 선택적 추론·정책 필드 검사 |
| `l2s1_capabilities()` | 상주 백엔드의 근거 타입, 모달리티 지원, 한도 조회 |
| `l2s1_decide(request)` | 판단 보류, 점수, 사용량을 포함한 네이티브 HTTP 결과를 그대로 반환 |

두 요청 도구는 검색 시 `binary`, `choice`, `ordinal`, 이미지 필드의 상세 JSON Schema를 제공합니다. 정적 리소스는 같은 문서를 `l2s1://docs/<name>`, 예제를 `l2s1://examples/<name>`, 스키마를 `l2s1://schema/request`에 공개합니다. 리소스를 에이전트에 제공하지 않는 클라이언트도 같은 기능의 도구를 사용할 수 있습니다. `design_decision(task)` 프롬프트는 요청 설계를 돕습니다.

어댑터는 HTTP v1의 `state`, `decisions`, 선택적인 `media`, 질문별 `media_ids`, 선택적인 `reasoning`, `policy`, `target_error_rate`, `failure_reasons`를 지원합니다. 선택 기능을 사용하기 전에 capability를 확인하세요. 오래된 백엔드는 미지원 필드를 거부할 수 있습니다. 요청의 오류율은 점수 임계값이며 정답 보장이 아닙니다. 알 수 없는 필드는 거부합니다. 어댑터는 HTTP 한도인 판단 128개, 미디어 4개, 디코딩된 이미지당 8 MiB, 본문당 44 MiB를 확인합니다. 검증은 구조적입니다. 이미지 디코딩, 모델·토큰·컨텍스트 사전 검증, 백엔드별 제한은 런타임이 필요합니다. 템플릿 이미지의 자리표시는 실제 표준 base64 바이트로 교체해야 합니다.

일반적인 흐름은 가이드·예제 읽기, 요청 설계, 검증, 백엔드 capability 확인, `l2s1_decide` 호출입니다. 보류한 결과도 `status: "abstained"`와 null 선택값을 가진 성공한 도구 실행입니다. 전송, 검증, 백엔드 실패는 MCP 도구 오류가 됩니다. 백엔드 HTTP 상태와 구조화된 오류 본문은 오류 메시지에 유지됩니다. 시간 제한 후에도 추론이 실행 중일 수 있으며 어댑터는 자동 재전송하지 않습니다.

<a id="verification"></a>
## 검증

```sh
python /path/to/L2S1/mcp/server.py --help
python -m unittest discover -s mcp -p 'test_*.py' -v
python /path/to/L2S1/mcp/smoke_client.py
```

가상환경의 Python으로 실행하세요. 테스트는 HTTP 픽스처와 subprocess stdio의 공식 SDK 클라이언트를 사용합니다. 검색, 스키마·리소스, 검증 실패, 근거·판단 보류의 무변경 반환, 백엔드 오류와 프롬프트, 선택적 정책·추론 필드, 재시도 없는 시간 제한을 검증합니다. CI는 모델 가중치 없이 Python 3.10과 3.14에서 같은 테스트를 실행하도록 구성되어 있습니다. 스모크 클라이언트는 실제 HTTP 백엔드가 필요하며 실제 MCP → HTTP → 모델 경로로 창고 요청 하나를 검증합니다. 특정 모델의 답을 가정하지 않고 선택과 보류를 보고합니다.
