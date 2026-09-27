<a id="state-first-decisions-and-request-local-prefix-reuse"></a>
# 状態優先の判断とリクエストローカルのプレフィックスの再利用

[English](../en/SEMIF_ALGORITHM.md) · [한국어](../ko/SEMIF_ALGORITHM.md) · [日本語](SEMIF_ALGORITHM.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

The prefix-reuse mode applies SemIf's evidence-first prompt and serial prefix-cache ideas to the existing Rust/libllama 判断 バックエンド.スコアリング コントラクトは、全語彙 候補の確率質量、型指定された結果、および明示的な 判断保留 を使用した単一トークンの条件付きソフトマックスのままです。

<a id="reference-and-scope"></a>
## 参照と範囲

[SemIf](https://github.com/TheoLeeCJ/SemIf) ソースを 2026-09-21 で確認しました。

- [`core.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/core.py): 判断基準 とオプションの前の証拠。 Git BLOB `6e93b16dcd4ab56c7a859c9046c48c6731d7180c` をレビューしました。
- [`serial.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/serial.py): 正確なプレフィックス検証、独立したサフィックス評価、および候補質量診断。 Git BLOB `015023b7e9d616e6ace0a600548e6fe33ad7f98f` をレビューしました。
- [`shared.py`](https://github.com/TheoLeeCJ/SemIf/blob/master/src/semif_phase1/shared.py): 並列共有状態ブランチ レイアウト。 Git BLOB `6e6870305d1b7693cead70b30637c0fec2684354` をレビューしました。

SemIf は MIT ライセンスを取得しており、著作権は 2026 TheoLeeCJ にあります。このアップデートは、これらのアルゴリズムのアイデアを独立した Rust/C++ で実装したものです。 SemIf コード、モデルの重み、データセットは提供しません。公開されているパフォーマンスと 正解率 は、このプロジェクトの測定値ではありません。この実装はサフィックスをシリアルに評価します。並列実行については、[PARALLEL_EXECUTION.md](PARALLEL_EXECUTION.md) で別途説明されています。

<a id="algorithm"></a>
## アルゴリズム

1. オプトイン `state-first` の場合、型付きペイロードを固定順序 `state`、`instruction`、`options` でシリアル化します。 Rust 構造体は、serde_json マップ機能が異なる場合でもその順序を保証します。構造化された JSON を保持し、特別なトークン解析を無効にしてすべてのリクエスト データをトークン化します。
2. 既存のモデル固有のチャット テンプレートとアシスタント境界を適用します。 Qwen3 non- Thinking および GPT-OSS Harmony 最終プリフィルは引き続きサポートされます。 A ～ Z の各応答コードが安定した単一トークンの継続であることを検証します。
3. `fresh` の場合は、メモリをクリアし、プロンプト全体を評価します。オプトイン `prefix-reuse` の場合、完全なトークン化されたプロンプトを以前の 判断 のトークンと比較します。状態ハッシュや生のテキストの長さから等価性を推測しないでください。
4. 正確な共通プレフィックスを切り捨てて、完全なオリジナルのプレフィル バッチを作成します。同一のプロンプトであっても、評価用に少なくとも 1 つの最終トークンを保持してください。これにより、新しい推論と同じサフィックス バッチ境界が保持されます。 `--batch` より短い共有プレフィックスは再利用されません。
5. 前のサフィックス `llama_memory_seq_rm` を削除し、残りのトークンを元の絶対位置で評価します。反復/ハイブリッド モデルまたはサフィックスの削除が失敗した場合は、メモリをクリアして新しく評価します。すべてのリクエスト境界および失敗後に、ネイティブ キャッシュ メタデータと KV メモリの両方をクリアします。
6. 最終的な 全語彙 logits を読み取り、変更されていないスコアリングと 判断保留 ポリシーを適用します。 `input_tokens` は論理トークンをカウントします。 `reused_prefix_tokens` は、実際に再利用されたトークンをカウントします。それらの差が評価される数値になります。 `backend.execution_mode` は、再利用がフレッシュにフォールバックする場合を含め、要求されたモードを記録します。

デフォルトは `--prompt-layout legacy --execution-mode fresh` のままで、元のプロンプトの動作が維持されます。バージョン管理された v2 プロンプトを使用するには、`--prompt-layout state-first` を個別に選択します。どちらのレイアウトもどちらの実行モードもサポートしていますが、従来のプロンプトでは再利用可能な証拠があまりありません。以前の v1 ベンチマーク出力を v2 の結果として再ラベル付けしないでください。証拠を移動すると、キャッシュの再利用とは関係なく、モデルの予測が変更されます。スコアは未調整のままであり、高速化してもより良いセマンティック 正解率 は確立されません。

<a id="reproduce"></a>
## 再現する

README に記載されている固定された sys 依存関係ビルドを使用します。

```sh
cargo test --locked --offline
cargo test --release --locked --offline --features llama

SKID_MODEL=models/Qwen3-0.6B-Q8_0.gguf SKID_CUDA=0 \
  SKID_REUSE_OUTPUT=/tmp/qwen3-reuse.json \
  cargo test --release --locked --offline --features llama \
  --test prefix_reuse -- --ignored --nocapture
```

CUDA では `--features llama-cuda` と `SKID_CUDA=1` を使い、GPU へのアクセスが必要です。小さい prefill バッチを確認するには `SKID_BATCH=32` を設定します（デフォルト 256）。任意実行のネイティブテストは、ラベル付き合成要求 12 件（判断 36 件）、長い状態の 6 判断要求 1 件、長い状態で同一プロンプトの 3 判断要求 1 件を使います。確率・候補質量・生の top-1・ポリシーが採用した選択を比較し、エラーと要求の分離を確認します。既存の確率・質量許容誤差 0.02 は回帰検査であり、正解率の保証ではありません。top-1 または採用した選択が変わっても失敗します。モードごとにウォームアップ要求を 1 件実行し、測定順序はモード間で交互にします。各要求を各モードで 1 回測るため、時間は安定したパーセンタイルではなく動作確認用の測定です。

ラベル付き 正解率、採用率、レイテンシー分布、および繰り返し実行の場合:

```sh
target/release/l2s1-tools benchmark-models --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode fresh --output results/v2-fresh
target/release/l2s1-tools benchmark-models --model qwen3 --device cpu \
  --prompt-layout state-first --execution-mode prefix-reuse --output results/v2-reuse
```

個別の出力ディレクトリを使用し、モデル ハッシュ、プロンプト バージョン、デバイス、バッチ、コンテキスト、およびポリシーを保存します。ランナーは、再利用および評価されたトークン数を記録します。論理入力トークンのスループットだけでは、実際に節約されたコンピューティングを測定することはできません。

<a id="fixed-schema-sessions"></a>
## 変化する状態を評価する固定スキーマセッション

`LlamaBackend::shared_decision(decision)` は `shared_state(state)` の逆の用途である。指示・候補スキーマを固定して変化する状態を評価する。呼び出し間で実際のデコーダ KV を保持するが、通常の `decide()` はリクエスト境界で KV を消去する。Python/TypeScript SDK の `PreparedDecision` は定義のスナップショットであり、このネイティブセッションを有効にしない。

```rust
backend.set_execution_mode(l2s1::ExecutionMode::PrefixReuse);
backend.set_prompt_layout(l2s1::PromptLayout::Legacy);
let mut session = backend.shared_decision(decision)?;
let first = session.decide(first_state)?;
let next = session.decide(next_state)?;
println!("reused: {}", next.results[0].reused_prefix_tokens);
```

既定のJSONマップ順序ではLegacy/minimal は指示・候補を可変状態の前に置く。依存関係で`serde_json/preserve_order`を有効にするとlegacyの順序が変わり得るため、実際の再利用トークンを確認する。State-first は固定状態と可変質問という逆の用途に適する。セッションがバックエンドを排他的に借用し、モデル・ポリシー・スキーマを固定する。生成・失敗・終了時に KV を消去し、前の状態の接尾部を置換する。回答はキャッシュしない。Recurrent/hybrid モデルと出力ヘッドは拒否する。保持するのは一つのシーケンスであり、自動的な複数スキーマ HTTP キャッシュではない。

バッチ境界を維持する。共通接頭辞が180トークンなら、バッチ256で0、128で128、64でも128トークンを再利用する。小さいバッチは prefill の処理量を下げ得る。各バッチで fresh/reuse を比較し、最速の fresh 基準とも比較する。再利用カウンタだけから実際の高速化を推定しない。

`examples/benchmark_schema_reuse.rs` は `decision-rules-v1` のラベル付き36決定を、完全に同じ指示・候補順・スキーマの10グループに分けて再生する。両経路とも呼び出し当たり一決定とし、順序を交代して fresh とセッションを比較する。確率、候補質量、argmax、受理回答、棄権理由を確認する。このグループ処理量は元の三決定リクエストの遅延とは別測定。正答、プロンプト、候補順は変更しない。

```sh
mkdir -p results
cargo build --release --locked --features llama --example benchmark_schema_reuse
cargo run --release --locked --features llama --example benchmark_schema_reuse -- \
  --model models/Qwen3-0.6B-Q8_0.gguf --device cpu --batch 64 --rounds 3 \
  --output results/schema-reuse-b64.json
```

バッチ256でも別の出力パスで繰り返す。macOS は `llama-metal` と `--device metal`、CUDA は `llama-cuda` と `--device cuda` を使う。セッション生成・破棄、トークン準備、採点を計時に含む。ロードとウォームアップは除外し、準備キャッシュは無効。結果一致は独立データでの精度を証明しない。

### CPU実測検証（2026-09-27）

Linux/WSL2、i9-9900Kと表示される論理CPU 4個、4スレッドでQwen3-0.6B-Q8_0を実行した。Context 2048、full evidence、準備キャッシュ無効。各セルは36判断全体をグループ処理した3回の中央値である。各経路を別途ウォームアップし、ペア内のfresh/reuse順序を交互にした。**全パスの秒数**であり、従来のリクエストp50ではない。

| Batch | Fresh | Shared decision | Reused / input tokens |
| --- | ---: | ---: | ---: |
| 64 | 33.311 s | 22.068 s | 1,984 / 4,983 |
| 256 | 31.691 s | 31.589 s | 0 / 4,983 |

バッチ64の再利用は同じバッチのfreshより1.51倍、**測定した最速のfresh設定（バッチ256）より1.44倍速い**。論理入力長は120–165トークンなので、完全なバッチ単位の規則では256で再利用できない。ネイティブ呼出しが時間を占める。バッチ64の最初のfreshパスは全体33.311秒、うちネイティブ32.806秒、準備0.039秒だった。

同一バッチの全ペア、反復パス、バッチ間の比較で、確率・候補質量・raw top-1・受理回答・棄権の差はすべて0だった。Raw top-1は13/36、受理30件、うち正解10件で変わらない。高速化は精度改善ではない。実際のネイティブ推論で失敗後の回復とリクエスト・セッション分離も通過した。M5 Max/Metal、jv.py、同時HTTP、メモリ使用量や一般精度の測定ではない。[縮約証拠](../../benchmarks/schema-reuse-20260927/summary.json)と[出所](../../benchmarks/schema-reuse-20260927/provenance.json)に時間、初回スコア、ハッシュと範囲を記録した。完全な応答・ログはローカルのignoredファイルである。

別のワークロードは`--input requests.jsonl --context 16384`で実行する。入力は`evaluate_jsonl`と同じ、正解を含まない`{ "id": "unique-case", "request": { "state": ..., "decisions": [...] } }`行である。グループ化でも指示・候補順序を維持する。外部入力レポートはfreshとセッションの結果一致を検査し、精度は分離した正解データで別途採点する。

[RTX 3080 JevBench 231問の検証](JEVBENCH.md#rtx3080-rerun-20260927)ではbatch 64の再利用は入力トークンの6.48–7.20%でした。同一batchの出力は保持しましたが、Qwen3 0.6B・Gemma 4 E2Bともbatch 256 freshより遅く、batch変更自体で一部のtop-1が変化しました。CPUルールフィクスチャの効果はワークロードに依存し、最適化にはbatch処理量と正解変化の確認も必要です。
