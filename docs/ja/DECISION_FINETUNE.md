<a id="supervised-decision-lora-pilot"></a>
# 監修済み 判断 LoRA パイロット

[English](../en/DECISION_FINETUNE.md) · [한국어](../ko/DECISION_FINETUNE.md) · [日本語](DECISION_FINETUNE.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

この実験では、既存の航空会社センチメント 判断 インターフェイスに合わせて Gemma 4 E2B IT をトレーニングします。これは、型付きの確率的判断という公的目標に従います。 TypeSafe 独自の RLCD または Jev アーキテクチャを再現するものではありません。新しい並列アテンション アーキテクチャは追加されず、一般的な 判断 機能は主張されません。

<a id="frozen-protocol"></a>
## 凍結されたプロトコル

- 出典: `target/release/l2s1-tools kaggle-airline` が作成した、固定された Kaggle Twitter US Airlines Sentiment アーカイブ。データセットのテキストと重みは無視された `results/` のままです。このリポジトリによって再配布されることはありません。
- 正規化されたテキストによるすべての 800 以前の 校正/評価例を除外します。正確に正規化された重複および競合するラベルのテキスト グループを削除します。
- シード 20260923: 900 固有のトレーニング ツイート (クラスごとの 300)、 400 新しい 校正 ツイート、および 400 新しいテスト ツイート。分割は正規化されたテキストによって切り離されます。これは意図的にバランスが取れた単一ドメインのサンプルであり、自然なクラスの蔓延ではありません。
- 各トレーニング ツイートは、3 つの循環オプション注文すべてに表示されます (2,700 トレーニング入力)。固定の 60 ケースのテスト サブセットは、3 つの順序すべてで評価されます (180 診断呼び出し。これらは追加の独立したケースではありません)。
- `export_decision_tokens` は、実際の Rust/GGUF プロンプト レンダラとトークナイザーを使用します。トランスフォーマーは、これらの正確な入力および候補トークン ID を使用して、本番環境 プロンプトの再実装を回避します。ラベルは別々に保管されます。
- 基本モデル: `google/gemma-4-E2B-it`、リビジョン `3e22461f65e89153144f8adb70e3b8c2cc9845a7`。トレーニングでは、bf16 コンピューティングを備えた NF4 ベース、bf16 および fp32 正規化/アダプターで凍結された大きな行列を使用します。
- LoRA ランク 8、アルファ 16、ドロップアウト 0、言語モデル アテンション Q/V プロジェクションのみ。 1 エポック、マイクロバッチ 1、累積 12、ゼロに直線的に減衰する学習率 0.0001、重み減衰ゼロの AdamW、1 で勾配ノルム クリッピング。
- 判断 ポジションでの損失: 候補クロスエントロピーに 0.1 を乗じた負の対数 全語彙 候補の確率質量。プロンプト トークンや生成されたテキストに損失はありません。
- 最終的な チェックポイント を修正しました。テストベースの チェックポイント/ハイパーパラメーターの選択はありませんでした。トレーニングのみの スモーク 実行では、最初に GPU/メモリ/勾配の互換性がチェックされます。そのアダプターは破棄され、実際の実行は元のベースから開始されます。
- 各 チェックポイント の温度を新しい 校正 分割にのみ適合させます。生のトップ - 1、NLL、Brier、ECE、採用率、および 採用された判断の正解率 をしきい値 0.6–1.0 でレポートします。 候補の確率質量 は変更されず、その 0.05 ゲートはアクティブのままです。
- 同じ実行時間/精度でベース チェックポイントとトレーニング済みチェックポイントを比較します。 Transformers NF4 と llama.cpp Q8 を別個の比較として扱います。コンバージョン効果はトレーニング効果ではありません。このパイロットではアプリケーションのデフォルトは変更されません。
- GGUF ベースとアダプターに対する追加のドメイン外回帰チェックとして、以前の AG News 400 ケース開発ベンチマークを実行します。これは新しい未使用のテスト セットではなく、この実験のアダプターや温度には適合しません。

トレーニング前の公開公開や重複に近いテキストは除外されません。単一のトレーニング シードでは、ランダムな初期化全体での再現性を確立できません。新しいテスト データは同じ履歴データセットから取得され、ドメイン外または 本番環境 正解率 を確立しません。現在の API 校正 はオフライン アーティファクトであり、アプリケーションの応答にサイレントに適用されません。

<a id="entry-points"></a>
## エントリーポイント

1. `target/release/l2s1-tools prepare-decision-finetune` はデータとプロトコルをフリーズします。
2. `examples/export_decision_tokens.rs` は、本番環境 入力/候補 ID をエクスポートします。
3. `scripts/train_decision_lora.py` は、固定されたモデルをダウンロードし、スモーク テストまたは完全なペアの実験を実行し、アダプターと生の測定値を保存します。
4. `target/release/l2s1-tools report-decision-finetune` は、校正 のみの温度に適合し、保持されたメトリクスとオプション注文診断をレポートします。

CLI および JSONL エバリュエーターは、`--lora path/to/adapter.gguf` を受け入れます。対応する llama.cpp `convert_lora_to_gguf.py --base <config-dir>` ツールを使用して PEFT アダプターを変換します。 1 スケールでは、バックエンド ごとに 1 つのアダプターがサポートされます。並列実行によってコンテキストのサイズが変更されるたびに再アタッチされます。応答メタデータ レコードは `lora_path` です。このオプションを使用しない場合、既存のベースモデルのパスは変更されません。アダプターをロードしても、温度 校正 は自動的に適用されません。

生成されたアーティファクト、依存関係ロック、および実行された正確なスクリプトは、無視されるローカルの `results/` ディレクトリに保持されます。

新しいアーティファクト ディレクトリの場合 (これは、新しいテスト セットではなく、同じ分割を再現します):

```sh
cargo build --release --locked -p l2s1-tools
target/release/l2s1-tools prepare-decision-finetune \
  --source results/kaggle-airline-20260922 --output results/decision-pilot-new/data
for split in train calibration test probe; do
  target/release/examples/export_decision_tokens \
    --model models/gemma-4-E2B-it-Q8_0.gguf \
    --input "results/decision-pilot-new/data/$split.jsonl" \
    --output "results/decision-pilot-new/data/$split-tokens.jsonl"
done
target/release/l2s1-tools prepare-decision-finetune \
  --output results/decision-pilot-new/data --seal-tokens
# Run in the locked training environment on the GPU host:
python scripts/train_decision_lora.py --data results/decision-pilot-new/data \
  --output results/decision-pilot-new/pilot --cache results/decision-pilot-new/hf-cache
target/release/l2s1-tools report-decision-finetune --data results/decision-pilot-new/data \
  --run results/decision-pilot-new/pilot
```

<a id="frozen-synthetic-accuracy-study"></a>
## 凍結合成 正解率 研究

新しい研究は、航空会社のパイロットや以前の倉庫設備とは別のものです。 6 つのドメインにわたって、480 train、 120 dev、 120 校正、および 180 テスト論理ケースを生成します。しきい値と文言テンプレート ファミリはどちらも分割間で互いに素です。各ケースには、自然言語と記号の 判断基準 がペアになっています。これらは 1 つのケースを 2 つ表現したものであり、独立したサンプルではありません。選択、バイナリ、および 順序付き ターゲットは、正確な境界と隣接する整数/小数境界を含めてバランスが取れています。 順序付き 値はソートされた順序を保持します。

次のワークフローでは、構成済みの ネイティブ ビルドと、上記の固定リビジョンの既存のローカル Gemma 4 E2B IT チェックポイント が必要です。トレーニングには、互換性のある PyTorch、Transformers、bitsandbytes、PEFT を備えた GPU Python 環境も必要です。実験とともに依存関係のバージョンを記録します。すべてのデータ、トークンのエクスポート、教師の応答、アダプター、およびレポートを無視された `results/` の下に保持します。基本の GGUF 重みを無視された `models/` の下に維持します。新しいスクリプトは、出力ファイル/ディレクトリの上書きを拒否し、チェックポイントをダウンロードしません。

スタディを準備し、**dev のみ** でプロンプト構成を比較します。

```sh
study=results/accuracy-study-new
model=models/gemma-4-E2B-it-Q8_0.gguf
checkpoint=/path/to/local/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7
variant=natural
mkdir -p "$study"
target/release/l2s1-tools prepare-accuracy-study --output "$study/data"
target/release/l2s1-tools prepare-accuracy-study --output "$study/data" --verify
cargo build --release --locked --features llama-cuda \
  --example evaluate_accuracy --example export_decision_tokens

target/release/examples/evaluate_accuracy --model "$model" --cuda \
  --input "$study/data/dev-$variant-requests.jsonl" \
  --output "$study/dev-$variant-predictions.jsonl" \
  --prompt-details minimal,typed,typed-examples --layouts legacy,state-first \
  --all-rotations
target/release/l2s1-tools report-accuracy-study --data "$study/data" \
  --predictions "$study/dev-$variant-predictions.jsonl" \
  --split dev --variant "$variant" --output "$study/dev-$variant-report.json" \
  --select "$study/dev-$variant-selection.json"
```

ペア表現実験には `symbolic` を使用します。両方のバリアントを比較する場合は、校正/test 予測を読み取る前に、両方の開発レポートを終了し、勝ったバリアントをフリーズします。選択では、生のトップが 1 にランク付けされ、次にすべてのケースで受け入れられた正しい部分がランク付けされ、次に、計算パスのレイテンシの中央値が決定的な最終タイ ブレークでランク付けされます。スプリット全体をカバーする構成のみが対象となります。ローテーション 2 は、3 つのオプションのタスクのみをカバーし、フルスプリット構成には勝てません。 `--select` は 校正 を拒否し、分割をテストします。タイミングにはモデルの読み込みと出力のシリアル化は含まれません。これらはエンドツーエンドのサービス遅延ではありません。

凍結された選択肢を読み取り、コード ローテーションを含む **training リクエストのみ** をエクスポートします。エクスポーターは 本番環境 GGUF トークナイザーを使用します。トレーナーの `--detail` ではアンダースコアが使用されています。 ネイティブ CLI 列挙値にはハイフンが使用されます。

```sh
selection="$study/dev-$variant-selection.json"
detail=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["setting"]["prompt_detail"])' "$selection")
detail_cli=$(python3 -c 'import sys; print(sys.argv[1].replace("_", "-"))' "$detail")
layout=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["setting"]["prompt_layout"].replace("_", "-"))' "$selection")
target/release/examples/export_decision_tokens --model "$model" \
  --input "$study/data/train-$variant-requests.jsonl" \
  --output "$study/train-tokens.jsonl" --all-rotations \
  --prompt-detail "$detail_cli" --prompt-layout "$layout"

python3 - "$study" "$variant" "$detail" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
seal = dict(schema_version=1, split='train', variant=sys.argv[2], detail=sys.argv[3],
            manifest_sha256=digest(root/'data/manifest.json'),
            token_sha256=digest(root/'train-tokens.jsonl'),
            hf_model='google/gemma-4-E2B-it',
            hf_revision='3e22461f65e89153144f8adb70e3b8c2cc9845a7')
with (root/'train-token-seal.json').open('x') as out:
    json.dump(seal, out, sort_keys=True, indent=2)
    out.write('\n')
PY
```

思考教師の回答を生成し、破棄された スモーク チェックを実行して、新しいアダプターをトレーニングします。独立した **train** ラベルと一致する教師の最終回答のみが許可されます。生成された根拠は記録されますが、生徒の目標として使用されることはありません。教師の回答が欠落していたり​​間違っていたりしても、オラクルのラベルに置き換えられることはありません。同じ受け入れられた例とプロトコルが、スモーク と最終トレーニングをバインドする必要があります。

`--teacher-batch-size` は、左パディングされた教師の生成を有効にします (デフォルトは 1)。各回答は、検証前の最初の EOS で個別にトリミングされます。別の制限されたパイロットを使用して、GPU メモリのサイズを設定します。教師レポートには、ピークの CUDA 割り当てとバッチ タイミングが記録されます。不完全なパイロットはトレーニングのインプットとして受け入れられません。

```sh
train_stage() {
  python3 scripts/train_accuracy_lora.py "$@" \
    --data "$study/data" --variant "$variant" --detail "$detail" \
    --tokens "$study/train-tokens.jsonl" --token-seal "$study/train-token-seal.json" \
    --checkpoint "$checkpoint"
}
train_stage teacher --output "$study/teacher" --max-new-tokens 384
train_stage smoke --teacher "$study/teacher" --output "$study/smoke"
train_stage train --teacher "$study/teacher" \
  --smoke-report "$study/smoke/complete.json" --output "$study/train"
python3 "$LLAMA_CPP_DIR/convert_lora_to_gguf.py" "$study/train/adapter" \
  --base "$checkpoint" --outfile "$study/adapter.gguf" --outtype f16
```

この変換コマンドには、`LLAMA_CPP_DIR` での完全な llama.cpp チェックアウトが必要です。バンドルされている ネイティブ ビルド スナップショットには変換ツールが省略されています。推論ビルドでは、`L2S1_LLAMA_CPP_SOURCE` または従来の `LLAMA_CPP_DIR` オーバーライドが設定されていない限り、バンドルされたソースが使用されます。

トレーニング損失が成功した場合、スモーク チェックまたは変換では、ネイティブ 正解率 の改善は確立されません。同じ GGUF、コンピューティング構成、および凍結されたプロンプト設定をベースとして、変換されたアダプターを評価します。たとえば:

```sh
target/release/examples/evaluate_accuracy --model "$model" --cuda \
  --lora "$study/adapter.gguf" \
  --input "$study/data/test-$variant-requests.jsonl" \
  --output "$study/test-$variant-adapter.jsonl" \
  --prompt-details "$detail_cli" --layouts "$layout" --all-rotations
target/release/l2s1-tools report-accuracy-study --data "$study/data" \
  --predictions "$study/test-$variant-adapter.jsonl" \
  --split test --variant "$variant" --output "$study/test-$variant-adapter-report.json"
```

`--lora` を使用せずに、同一の ネイティブ 評価を別の基本出力ファイルに実行します。以前に凍結された単一の回転またはアンサンブルのみを比較します。追加のローテーション行は診断であ​​り、テストで新たに選択する機会ではありません。 校正 または 判断保留 しきい値に適合する場合は、最終テスト評価の前に 校正 分割を使用してその作業を終了します。記者自体はどちらにも当てはまらない。生のトップ - 1、NLL/Brier、採用率、採用された判断の正解率、accepted-correct/all、および回転感度を一緒にレポートします。 NF4 教師/トレーニングの結果と GGUF アダプターの結果には、実行時間/精度の境界が異なります。このワークフロー自体では、測定された 正解率 やパフォーマンスに関する主張はありません。
