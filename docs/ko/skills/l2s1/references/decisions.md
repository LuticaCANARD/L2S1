<a id="requests-and-result-handling"></a>
# 요청과 결과 처리

[English](../../../../en/skills/l2s1/references/decisions.md) · [한국어](decisions.md) · [日本語](../../../../ja/skills/l2s1/references/decisions.md)

[English index](../../../../en/README.md) · [한국어 색인](../../../README.md) · [日本語索引](../../../../ja/README.md)

아래 텍스트 요청은 Rust 코어, CLI stdin, HTTP v1에서 사용할 수 있습니다.

```json
{
  "state": {"storage_requirement": "chilled", "hours_until_dispatch": 4},
  "decisions": [
    {
      "id": "storage_zone",
      "instruction": "Select the zone matching storage_requirement.",
      "kind": {
        "type": "choice",
        "options": [
          {"id": "ambient", "criterion": "Ambient storage is required."},
          {"id": "chilled", "criterion": "Chilled storage is required."},
          {"id": "frozen", "criterion": "Frozen storage is required."}
        ]
      }
    },
    {
      "id": "cold_chain",
      "instruction": "Chilled and frozen shipments require temperature control. Does this shipment require it?",
      "kind": {
        "type": "binary",
        "false_label": "Temperature control is not required.",
        "true_label": "Temperature control is required."
      }
    },
    {
      "id": "priority",
      "instruction": "Choose priority from hours_until_dispatch using the exact thresholds.",
      "kind": {
        "type": "ordinal",
        "levels": [
          {"id": "low", "criterion": "More than 24 hours remain.", "value": 0},
          {"id": "medium", "criterion": "More than 6 and at most 24 hours remain.", "value": 1},
          {"id": "high", "criterion": "At most 6 hours remain.", "value": 2}
        ]
      }
    }
  ]
}
```

ID와 기준에는 공백 외의 텍스트가 있어야 합니다. 판단 ID는 요청 전체에서, 후보 ID는 각 판단 안에서 고유합니다. Choice·ordinal은 최소 후보 두 개가 필요합니다. 후보가 26개 이하면 단일 문자 답변 코드, 더 많으면 토너먼트가 아닌 고정 길이 전체 코드 시퀀스를 사용합니다. 실제 모델이 토큰화와 컨텍스트 수용 여부를 판단합니다.

<a id="local-cli-and-rust-results"></a>
## 로컬 CLI와 Rust 결과

응답은 `backend`, `policy`, `results`를 포함합니다. 각 결과는 `id`, `value`, `scores`, `candidate_mass`, `top_option_probability`, `abstention_reasons`, `scoring_method`, 토큰 사용량을 포함합니다.

- Binary: `value = {"type":"binary","value":false,"p_true":...}`도 선택된 값일 수 있습니다. `value.value == null`은 판단 보류입니다.
- Choice: `value.selected`는 앱의 후보 ID 또는 `null`입니다.
- Ordinal: `value.selected`는 수준 ID 또는 `null`입니다. `expected_value`는 점수 기반 추정값이며 선택한 수준을 대체하지 않습니다.

보류 이유에는 `low_top_probability`, `low_candidate_mass`, `tied_candidates`가 있습니다. 결과와 함께 유지하세요. 실험을 기록할 때 선택된 라벨만 저장하지 말고 원시 근거를 보존합니다.

<a id="http-and-mcp-results"></a>
## HTTP와 MCP 결과

HTTP는 `api_version`, `request_id`, 백엔드 래퍼, 각 결과의 `status`, `evidence`, `usage`를 추가합니다. MCP `l2s1_decide`는 이 래퍼를 바꾸지 않고 구조화된 내용으로 반환하며 JSON 텍스트 표현도 제공합니다. 로컬 점수는 `evidence.type == "model_scored"`이고 점수·추정값은 `evidence` 아래(`evidence.estimate.p_true` 또는 `.expected_value`)에 있습니다. OpenRouter는 `evidence.type == "selection_only"`, `policy == null`입니다.

`status == "abstained"`는 MCP 도구 오류나 HTTP 전송·백엔드 오류와 구분되는 정상 결과로 처리합니다. 시간 제한은 모델 작업 취소를 입증하지 않습니다. 어댑터는 자동 재시도하지 않습니다.

이미지 HTTP는 `{"type":"image","id":"photo","data_base64":"..."}` 형태의 `media`와 판단별 `media_ids`도 받습니다. `media_ids`를 생략하면 요청 이미지 전체를, `[]`는 텍스트 전용 판단을 선택합니다. 로컬 백엔드는 판단당 최대 이미지 한 장을 받습니다. 어댑터는 안정된 `state`, `decisions`, `media` 필드와 판단 최대 128개, 미디어 4개, 디코딩된 이미지당 8 MiB, 인코딩된 HTTP 본문당 44 MiB를 지원합니다. 디코딩 지원과 모델별 한도는 백엔드가 적용합니다.

선택적 HTTP 필드는 `reasoning`(`mode: "direct" | "thinking"`, `max_tokens: 1..1024`, 기본 128), `policy`(`min_top_probability`, `min_candidate_mass` 모두 `[0,1]`), `[0,1]`의 `target_error_rate`, `failure_reasons`입니다. 사용 전에 capability를 확인하세요. 오래된 서버는 미지원 필드를 거부하며 선택 결과만 있는 근거에는 점수 정책을 적용할 수 없습니다. `target_error_rate`는 후보 질량 검사를 유지하며 최상위 기준을 `1 - rate`로 바꾸지만 정답률을 보장하지 않습니다. 실패 문구의 키는 `low_top_probability`, `low_candidate_mass`, `tied_candidates`, `reasoning_limit`, `native_failure`이며 메시지는 공백이 아닌 텍스트로 최대 UTF-8 512바이트입니다. 이 필드는 HTTP 전용이며 코어 `DecisionRequest`와 CLI 텍스트 JSON은 받지 않습니다.
