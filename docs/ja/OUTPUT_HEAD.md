<a id="frozen-deployment-output-heads"></a>
# モデル本体を固定する配布用出力ヘッド

[English](../en/OUTPUT_HEAD.md) · [한국어](../ko/OUTPUT_HEAD.md) · [日本語](OUTPUT_HEAD.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

オプションの `--output-head <JSON>` は、明示的に識別した 1 つの choice タスクを小さな学習済み分類器で評価します。GGUF 本体は固定します。要求しなければアダプターを読み込まず、通常の判断は元のスコア経路を維持します。

2 種類のヘッドに対応します。

- `logit_affine`: 意味を表す選択肢 ID に対応付けた基本候補 logits 上で行列とバイアスを学習します。単位行列で temperature のみの校正になります。
- `hidden`: Gemma4 の最終出力正規化後の隠れ状態上で行列とバイアスを学習します。E2B の特徴は 1,536 次元、3 クラスのヘッドは 4,611 パラメーターです。

いずれも `softmax((W x + b) / temperature)` を計算します。成果物の行は意味を表す選択肢 ID を使うため、順序を変えるとコードとクラスの対応が正しく変わります。プロンプトの並べ替えは元のモデルの特徴にも影響するため、別途測定します。

`candidate_mass` は元の全語彙 LM 候補質量を維持します。保持する独立した互換性基準であり、**新しい分類器の提供する確率ではありません。** Temperature は候補分布だけを校正します。応答は `calibration_id` でヘッドを識別し、`backend.output_head_path` にパスを記録し、`learned_hidden_softmax_with_base_mass_v1` または `learned_logit_affine_softmax_with_base_mass_v1` を使います。`scores[].raw_logit` は temperature 適用前のヘッドスコア、`option_probability` は適用後です。

<a id="runtime-contract"></a>
## ランタイムの契約

成果物は正確な GGUF SHA-256、デバイス記述、計算オプション、プロンプト版、タスク ID、指示、意味を表す選択肢 ID・基準に紐付きます。選択肢の並べ替えは許可します。他のタスク ID は基本モデルを使います。学習済み ID に変更した指示や選択肢の意味を使うと明示的に失敗します。タスク ID は呼び出し側の宣言であり、その ID で別ドメインのテキストを渡しても自動では検出しません。

ヘッドを読み込むと fresh 実行のみ対応します。LoRA と出力ヘッドは併用できません。読み込み後に変更したプロンプト・実行設定は推論前に再検査します。隠れ特徴の出力は現在 Gemma4 のみです。同じ固定 llama.cpp リビジョンを使い、ランタイム変更後は再検証・再学習してください。JSON は共有ライブラリや GPU ドライバーの指紋を記録しません。

ネイティブブリッジは固定 llama.cpp の staging API `llama_set_embeddings_nextn(..., true, false)` と `llama_get_embeddings_nextn_ith` で Gemma4 の post-norm 特徴を読みます。マスクしない抽出は元のグラフ形状を保持し、最後の decode バッチから隠れ行を転送します。Rust は最後の行だけを受け取ります。通常の embedding モードは全入力トークンの語彙出力を強制するため、意図的に使いません。保持する質量基準には全語彙投影が必要で、LM ヘッドの除去や Jev/RLCD のレイテンシ再現は主張しません。

<a id="reproduce-the-pilot"></a>
## 試験の再現

同梱の llama.cpp コミット `3d82ef62d47fd74e18f36c5eccbdcf965b617b17` と CUDA feature でビルドします。

```bash
cargo build --release --offline --features llama-cuda --examples --bin l2s1
cargo build --release --offline -p l2s1-tools

target/release/l2s1-tools prepare-output-head \
  --source results/kaggle-airline-20260922 \
  --previous results/finetune-20260923/data \
  --output results/output-head-20260923/data

mkdir -p results/output-head-20260923/features
for split in train dev calibration test probe; do
  target/release/examples/export_decision_features \
    --model models/gemma-4-E2B-it-Q8_0.gguf --cuda \
    --input "results/output-head-20260923/data/$split.jsonl" \
    --output "results/output-head-20260923/features/$split.jsonl"
done

# Python environment with PyTorch (CPU training) and the standard library.
python3 scripts/train_output_head.py \
  --data results/output-head-20260923/data \
  --features results/output-head-20260923/features \
  --output results/output-head-20260923/heads

target/release/examples/evaluate_jsonl \
  --model models/gemma-4-E2B-it-Q8_0.gguf --cuda --warmup \
  --output-head results/output-head-20260923/heads/selected.json \
  --input results/output-head-20260923/data/test.jsonl \
  --output results/output-head-20260923/selected-test.jsonl
```

選択した成果物のメイン CLI の呼び出し例です。

```bash
target/release/l2s1 --model models/gemma-4-E2B-it-Q8_0.gguf \
  --device cuda --output-head results/output-head-20260923/heads/selected.json \
  --input results/output-head-20260923/example-request.json
```

メイン CLI は `--device cuda --output-head ...` で同じ成果物を受け付けます。入力は JSONL ベンチマークのラッパーでなく `DecisionRequest` です。`LlamaBackend::extract_features` は fresh の基本リクエスト 1 つを通常の判断応答とともに出力し、`load_output_head` は指定した成果物を有効にします。

この試験は過去の 2 つの航空会社実験で使った正規化済みの完全一致テキストをすべて除外します。学習 900 テキスト（3 つの循環順序 = 2,700 呼び出し）、開発 300、校正 400、テスト 400 を固定します。テスト 60 テキストの順序検証は 180 呼び出しであり、180 独立ケースではありません。分割は均等で、元の母集団分布ではありません。意味的な近似重複は除外しません。

特徴の標準化は学習データのみを使います。Float64 CPU LBFGS は交差エントロピーに `L2/2 * ||W||²` を加えて学習します。固定グリッドは 0.001、0.01、0.1、1 です。各ヘッドのペナルティと採用ヘッドは、校正前の最小開発 NLL で選びます。Temperature は校正データのみです。標準化は出力する重みとバイアスに組み込みます。最終テストラベルは、ハイパーパラメーター、ヘッド、temperature の選択に使いません。生成される `selection.json` と `heads/training.json` に選択と学習の詳細を記録します。

モデルの重み、生データ、特徴、学習済み成果物は無視対象の `models/` と `results/` に置きます。データセットとモデルの利用条件は、リポジトリのソースコードライセンスと別です。
