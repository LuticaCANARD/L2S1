# Requests and result handling

[English](../../../docs/en/skills/l2s1/references/decisions.md) · [한국어](../../../docs/ko/skills/l2s1/references/decisions.md) · [日本語](../../../docs/ja/skills/l2s1/references/decisions.md)

[English index](../../../docs/en/README.md) · [한국어 색인](../../../docs/ko/README.md) · [日本語索引](../../../docs/ja/README.md)

The text request below works with core Rust, CLI stdin and HTTP v1:

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

IDs and criteria must contain non-whitespace text. Decision IDs are unique across
the request; candidate IDs are unique within each decision. Choice/ordinal require
at least two candidates. Up to 26 candidates use single-letter answer codes; wider
sets use complete fixed-width code sequences, not a tournament. The actual model
still determines tokenization and context fit.

## Local CLI and Rust results

The response contains `backend`, `policy` and `results`. Each result includes
`id`, `value`, `scores`, `candidate_mass`, `top_option_probability`,
`abstention_reasons`, `scoring_method`, and token usage.

- Binary: `value = {"type":"binary","value":false,"p_true":...}` can be selected.
  `value.value == null` is abstention.
- Choice: `value.selected` is the application candidate ID or `null`.
- Ordinal: `value.selected` is a level ID or `null`; `expected_value` is a score
  estimate and does not replace a selected level.

Abstention reasons include `low_top_probability`, `low_candidate_mass` and
`tied_candidates`. Keep them with the result. Preserve raw evidence when recording
an experiment instead of saving only the selected label.

## HTTP and MCP results

HTTP adds `api_version`, `request_id`, a backend envelope, and each result's
`status`, `evidence`, and `usage`. The MCP `l2s1_decide` tool returns that envelope
unchanged in structured content; MCP also provides its JSON text representation.
For local scores, `evidence.type == "model_scored"`, and scores and estimates live
under `evidence` (`evidence.estimate.p_true` or `.expected_value`). For OpenRouter,
`evidence.type == "selection_only"` and `policy == null`.

Handle `status == "abstained"` as a normal result, distinct from an MCP tool error
or HTTP transport/backend error. A timeout does not establish that model work was
cancelled; the adapter never retries automatically.

For images, HTTP additionally accepts `media` with
`{"type":"image","id":"photo","data_base64":"..."}` and decision `media_ids`.
Omitting `media_ids` selects all request images; `[]` makes a decision text only.
Local backends accept at most one image per decision. The adapter supports the
stable `state`, `decisions`, `media` fields, up to 128 decisions, 4 media items,
8 MiB per decoded image, and 44 MiB per encoded HTTP body. Decode support and
model-dependent limits are enforced by the backend.

Optional HTTP fields are `reasoning` (`mode: "direct" | "thinking"`,
`max_tokens: 1..1024`, default 128), `policy` (both `min_top_probability` and
`min_candidate_mass` in `[0,1]`), `target_error_rate` in `[0,1]`, and
`failure_reasons`. Inspect capabilities before using them: an older server can
reject fields it does not support, and selection-only evidence cannot use a scored
policy. `target_error_rate` overrides the top threshold to `1 - rate`, retaining
the candidate-mass check; it does not guarantee accuracy. Failure message keys are
`low_top_probability`, `low_candidate_mass`, `tied_candidates`, `reasoning_limit`,
or `native_failure`, with nonblank messages of at most 512 UTF-8 bytes. These fields
are HTTP-only; core `DecisionRequest` and CLI text JSON do not accept them.
