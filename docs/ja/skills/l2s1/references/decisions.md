<a id="requests-and-result-handling"></a>
# リクエストと結果の処理

[English](../../../../en/skills/l2s1/references/decisions.md) · [한국어](../../../../ko/skills/l2s1/references/decisions.md) · [日本語](decisions.md)

[English index](../../../../en/README.md) · [한국어 색인](../../../../ko/README.md) · [日本語索引](../../../README.md)

以下のテキストリクエストは、Rust コア、CLI stdin、HTTP v1 で使えます。

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

ID と基準には空白以外のテキストが必要です。判断 ID はリクエスト全体、候補 ID は各判断内で一意です。Choice・ordinal は最低 2 候補が必要です。26 候補までは単一文字の回答コード、それより多い場合はトーナメントでなく固定長の完全なコード列を使います。実モデルがトークン化とコンテキストへの収まりを決めます。

<a id="local-cli-and-rust-results"></a>
## ローカル CLI と Rust の結果

応答は `backend`、`policy`、`results` を含みます。各結果には `id`、`value`、`scores`、`candidate_mass`、`top_option_probability`、`abstention_reasons`、`scoring_method`、トークン使用量があります。

- Binary: `value = {"type":"binary","value":false,"p_true":...}` は選択された値です。`value.value == null` が保留です。
- Choice: `value.selected` はアプリの候補 ID または `null` です。
- Ordinal: `value.selected` はレベル ID または `null`。`expected_value` はスコアに基づく推定で、選択したレベルを置き換えません。

保留理由には `low_top_probability`、`low_candidate_mass`、`tied_candidates` があります。結果と一緒に保持します。実験の記録では選択ラベルだけでなく生の根拠を残します。

<a id="http-and-mcp-results"></a>
## HTTP と MCP の結果

HTTP は `api_version`、`request_id`、バックエンドのエンベロープ、各結果の `status`、`evidence`、`usage` を追加します。MCP `l2s1_decide` は変更せず構造化された内容として返し、JSON テキストも提供します。ローカルでは `evidence.type == "model_scored"` で、スコア・推定は `evidence`（`evidence.estimate.p_true` または `.expected_value`）の下にあります。OpenRouter では `evidence.type == "selection_only"`、`policy == null` です。

`status == "abstained"` は、MCP ツールエラーや HTTP 転送・バックエンドエラーとは異なる正常な結果です。タイムアウトはモデル処理の取消を証明しません。アダプターは自動再試行しません。

画像の HTTP は `{"type":"image","id":"photo","data_base64":"..."}` の `media` と判断の `media_ids` を受け付けます。`media_ids` の省略は全画像を、`[]` はテキストのみを選びます。ローカルバックエンドは判断ごとに最大 1 画像です。アダプターは安定した `state`、`decisions`、`media` と、最大 128 判断、4 メディア、復号画像 8 MiB、エンコード済み HTTP 本文 44 MiB に対応します。復号対応とモデル固有の制限はバックエンドが適用します。

任意の HTTP フィールドは `reasoning`（`mode: "direct" | "thinking"`、`max_tokens: 1..1024`、デフォルト 128）、`policy`（`min_top_probability`、`min_candidate_mass` ともに `[0,1]`）、`[0,1]` の `target_error_rate`、`failure_reasons` です。使用前に capability を確認してください。古いサーバーは未対応フィールドを拒否し、選択のみの根拠はスコアポリシーを使えません。`target_error_rate` は候補質量検査を維持し、最上位しきい値を `1 - rate` に変えますが正解率を保証しません。失敗文言のキーは `low_top_probability`、`low_candidate_mass`、`tied_candidates`、`reasoning_limit`、`native_failure` で、空白のみではない最大 512 UTF-8 バイトの文言です。HTTP 専用で、コア `DecisionRequest` と CLI テキスト JSON は受け付けません。
