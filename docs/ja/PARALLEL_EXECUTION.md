<a id="parallel-question-execution"></a>
# 質問の並列実行

[English](../en/PARALLEL_EXECUTION.md) · [한국어](../ko/PARALLEL_EXECUTION.md) · [日本語](PARALLEL_EXECUTION.md)

[English index](../en/README.md) · [한국어 색인](../ko/README.md) · [日本語索引](README.md)

サポートされている `parallel` モードは、個別の llama.cpp シーケンス ID を使用して独立した質問を評価します。ロードされたモデルを共有し、正確な共通プロンプト接頭辞を Wave ごとに 1 回計算し、複数の質問からの接尾辞トークンを各デコード バッチに入れます。 1 つの Lama コンテキストを同時に変更する複数のスレッドは開始されません。

```sh
cargo run --release --locked --features llama-cuda -- \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --input examples/warehouse.json --device cuda \
  --prompt-layout state-first --execution-mode parallel --parallel-width 4
```

デフォルトの実行モードは引き続き `fresh` です。0.2.0 以降、テキストの `parallel` 実行は `--prompt-layout`（`set_prompt_layout`）で明示しない限り `state-first` レイアウトを使います。その他のすべてのモードと、ビジョンプロジェクターを読み込んだバックエンドは `legacy` のままです。プロンプトのレイアウトと実行モードは引き続き個別の選択です。状態を移動するとプロンプトが変わり、並列処理は実行スケジュールを変えます。同じレイアウトですべての実行モードを比較してください。

<a id="implementation"></a>
## 実装

- Rust は、各質問のトークンと A ～ Z の単一トークンの継続を準備して検証します。ユーザー状態と質問データは特別なトークン解析保護を保持します。
- 各ウェーブには最大で `parallel_width` の質問が入力されます (デフォルトは 4、許可される 1–32)。最後の波はもっと小さくなる可能性があります。結果には、異なるデコード バッチでプロンプトがいつ終了したかを含め、リクエストの順序と ID が保持されます。
- 共有プレフィックスは、質問ごとに少なくとも 1 つのサフィックストークンを残しつつ、正確なトークン ID 上の木構造を作ります。ウェーブ全体が共通プレフィックスを共有し、より長く一致する質問どうしは別の質問が先に分岐した後も共有を続けます。デフォルトでは各共有区間を完全なオリジナルのプレフィルバッチ単位に切り捨てます。1 つのシーケンスが各区間を評価し、`llama_memory_seq_cp` がその KV エントリを同じグループの他のシーケンスと共有します。
- サフィックス トークンには、独立したシーケンス ID と元の絶対位置があります。彼らの注意は、別の質問の接尾辞ではなく、自分自身のシーケンスと共有接頭辞に注目します。最終 全語彙 logits は、関連するデコード バッチ内の各質問の最終トークン インデックスを使用してコピーされます。
- メモリは API 呼び出しの境界と失敗後にクリアされます。1 回の呼び出しのウェーブ間では、前のウェーブの最初のシーケンスのルート共有プレフィックスだけを、次のウェーブの最初のプロンプトが正確に同じトークンで始まる場合に限り保持します。呼び出しをまたぐ保持には明示的な `ParallelPrefixSession` が必要です。このモードでは、リカレント/ハイブリッドモデルは明示的にサポートされていません。
- モデルは一度ロードされたままになります。デフォルトでは、シーケンス容量が変更されると、公称トークン容量 `context * width` でコンテキストが遅延して再作成されます。 `context` は質問ごとの入力制限のままです。幅を大きくすると、KV/アテンション メモリが増加し、割り当てに失敗する可能性があります。サイレント CPU フォールバックや入力切り捨てはありません。
- `backend.parallel_width` は、設定された波の制限を記録します (シリアル モードは 1 をレポートします)。`reused_prefix_tokens` は、その呼び出しで質問が自分で評価しなかったトークン数です。共有プレフィックスを評価した質問は 0、それをコピーした質問は共有トークン数になります。保持されたプレフィックスはすべての質問で再利用として数えられるため、後のウェーブやセッション呼び出しの最初の質問も 0 でないことがあります。`input_tokens - reused_prefix_tokens` の合計が実際に評価したトークン数です。0 が 1 つあることを「共有の失敗」と読まないでください。

これは、固定された llama.cpp [batch、シーケンス、メモリ コピー、およびトークンごとの logits API](https://github.com/ggml-org/llama.cpp/blob/3d82ef62d47fd74e18f36c5eccbdcf965b617b17/include/llama.h) を使用します。これはシリアル ウェーブによる独立したシーケンス バッチ処理であり、置換モデル アーキテクチャや校正済みの 判断 トレーニングではありません。

<a id="prefix-sharing-controls-020"></a>
### プレフィックス共有の設定 (0.2.0)

これらの設定は、どの正確なトークンプレフィックスを共有するかだけを変え、プロンプトの内容は変えません（上記のデフォルトレイアウトを除く）。

- **レイアウト.** `legacy` は `instruction` と `options` を `state` より前に直列化し、typed detail は `decision_kind` を先頭に置きます。そのため質問が混在するウェーブは数トークンで分岐し、通常は何も共有できません。`state-first` は状態を先頭に置きます。明示的に選んだ `legacy` レイアウトが複数質問の並列呼び出しで 1 トークンも共有できない場合、バックエンドは stderr に警告を 1 回出力します。レイアウトを変えると予測が変わります。精度を再評価し、校正を新しい `prompt_version` に結び付けてください。
- **固定された JSON 順序.** どちらのレイアウトも、ペイロードのフィールドと `state`・`shared` のすべての入れ子オブジェクトを、serde_json の `preserve_order` 機能に関係なくキーを並べ替えて直列化します。Cargo は依存グラフ全体で機能を統合するため、以前はどれか 1 つの依存が `preserve_order` を有効にするだけで `legacy` プロンプト（ひいては判断と校正）が気付かないうちに変わり得ました。バイト列は以前のデフォルトビルドの出力と同じです。
- **共有入力.** `DecisionRequest.shared`（JSON `"shared"`）は、知識や例など多くのリクエストや質問に共通する根拠を保持します。どちらのレイアウトでも常にデータ区間の最初のフィールドになるため、`state` 内のキー名の順序を利用した工夫は不要です。省略または `null` の場合、プロンプトは変わりません。ONNX バックエンドはこれを拒否します。
- **ウェーブ順序.** `--parallel-wave-order prefix`（デフォルト、`ParallelWaveOrder::Prefix`）は、ウェーブを作る前に準備済みプロンプトをトークン列で並べ替え、長い共通プレフィックスを持つ質問を同じウェーブに入れます。`request` はリクエスト順の連続したウェーブに戻します。結果は常にリクエストと判断の順序を保ちますが、デコードバッチが変わるためスコアがわずかに変わることがあります。
- **アライメント.** `--parallel-prefix-alignment batch`（デフォルト）はすべての共有区間を完全なプレフィルバッチ単位に切り捨て、共有 KV をシリアルなプレフィルと同じデコード境界で計算します。各質問は区間ごとに最大 `batch - 1` 個の共通トークンを再評価します。`token` はすべての共通トークンを共有します。長い共通プレフィックスでは高速ですが、スコアがわずかに変わることがあるため、先に判断結果を検証してください。トークン単位で再利用トークンが 16 未満の入れ子区間は、別のデコードに分割しません。
- **呼び出しをまたぐ保持.** `backend.parallel_prefix_session()` は `ParallelPrefixSession` を返し、その `decide` / `decide_batch` は最新のルート共有プレフィックスを呼び出し間でシーケンス 0 に保持します。後の呼び出しの最初のプロンプト（ウェーブの並べ替え後）が同じトークンで始まる場合、その部分は再評価しません。セッションは `parallel` モードを必要とし、設定された幅をシーケンス容量として保ち、作成時・失敗時・drop 時にネイティブ KV をクリアします。レスポンスは `backend.parallel_prefix_retained` を報告します。同じ呼び出し順序は同じスコアを生みます。

ローカル測定（Qwen3-0.6B Q8_0、CPU、4 スレッド、batch 512、幅 8、約 1,300 トークンの `shared` 知識フィールドを共有する 8 リクエストの呼び出しを 3 回、明示的な `legacy` レイアウト、一部の区間で別のテストジョブと CPU を共有した 1 回の実行のため時間は参考値で再利用トークン数は正確、一般的な高速化の主張ではありません）：fresh 243.9 秒、parallel `batch` 64.6 秒（入力 33,110 トークン中 21,504 を再利用）、parallel `token` 16.1 秒（28,728）、セッション `batch` 51.7 秒（24,576）、セッション `token` 1.6 秒（32,832）。batch が 512 トークンの場合、batch アライメントは質問ごとに最大 511 個の共通トークンを再評価し、ここでは残りの作業の大半を占めました。

`backend.parallel_prefix_alignment` と `backend.parallel_wave_order` はデフォルト以外の設定を報告します。フィールドがなければ `batch` とリクエスト順（0.2.0 以前の動作）を意味します。

<a id="dynamic-parallel-kv-context"></a>
### 動的並列 KV コンテキスト

`--parallel-context-dynamic` は、`parallel` モードのオプトイン メモリ設定です。各 Wave をトークン化した後、従来の `context * width` 予約に上限を設けて、最大 `sum(input_tokens) + batch` KV スロットを予約します。 llama.cpp は、要求されたサイズをパディングする場合があります。合計では共有プレフィックスが複数回カウントされるため、これは控えめな値になります。 `--context` は引き続き個々の質問を検証します。入力は切り捨てられません。

ネイティブ コンテキストは、後のウェーブでより多くのスロットが必要になると増加します。コンテキストの再構築を繰り返すことを避けるために、同じ有効質問数で後のウェーブでもその高い容量を維持します。デフォルト モードと動的モードを切り替えるとコンテキストが再作成され、どちらのモードでも推論後にリクエスト ローカル KV がクリアされます。 `backend.parallel_context_dynamic` はオプトイン結果を識別します。 `backend.parallel_context_tokens` は、動的モードで現在割り当てられているパディングされたコンテキストを報告します。モデルの重みはロードされたままになります。

これにより、llama.cpp コンテキスト サイズが変更され、logits または判断が変更される可能性があります。判断にメモリ設定を使用する前に、正確なモデル、プロンプト、ポリシー、および入力セットをデフォルトの並列モードと比較します。これはメモリの最適化です。スループットの向上を約束するものではありません。

<a id="independent-request-batches"></a>
### 独立したリクエストバッチ

`backend.decide_batch(&requests)` は、状態を結合したり質問プロンプトを変更したりせずに、複数のリクエストを評価します。パラレル モードは、質問を分離されたシーケンスに平坦化し、境界のあるウェーブを処理し、元のリクエスト/判断 グループ化を復元します。異なるリクエストでの 判断 ID の繰り返しは許可されます。この明示的な呼び出し内で共有できるのは、正確なトークン プレフィックスのみです。 KV は通話を存続しません。失敗すると、部分的な応答結果はなく、バッチ全体のエラーが返されます。空のバッチは空の結果リストを返します。

JSONL エバリュエーターは、`--execution-mode parallel --parallel-width 16 --request-batch-size 16` に加えて、オプションの `--warmup` バッチをサポートします。これにより、同じ独立した記事のリクエストについて公平な比較が可能になります。各アーティクルの完全なバッチ完了レイテンシー、バッチ ID、償却コンピューティング時間を個別に記録します。バッチの経過時間をそのサイズで割ると、個々の応答待ち時間ではなく、償却コストが測定されます。

<a id="independent-image-batches"></a>
### 独立した画像バッチ

対応するビジョン プロジェクターがロードされているため、`backend.decide_vision_batch(&requests, &images)` は、独自の状態と 1 つの元の画像から各リクエストをスコアリングします。 `parallel` モードでは、判断は有界の ネイティブ ウェーブに入ります。 `fresh` モードはそれらをシリアルに評価します。各 判断 が `media_ids` 経由で 1 つのイメージを参照し、リスナーが `--execution-mode parallel --parallel-width 4` を使用する場合、HTTP は同じパスを公開します。 4 つの独立した判断を持つ 4 つのリクエスト メディア アイテムは、単に HTTP 応答を共有するのではなく、ネイティブ 実行を共有します。

画像のデコードとマルチモーダル トークン化では、信頼できるプロンプトの制御トークンの境界が保持されます。互換性のあるプロジェクター チャンクは llama.cpp mtmd のバッチ エンコーダーに入ります。プロジェクターの形状とトークンの制限により、これらがより小さなエンコーダー バッチに分割される場合があります。すべての Wave の質問は、独立したデコーダー シーケンス ID とシーケンスごとの画像位置を使用し、独自の 全語彙 最終 logits またはコンパクト候補スコアと 全語彙 ノーマライザーを使用します。現在、Vision はプロンプト プレフィックス KV を共有していないため、`reused_prefix_tokens` はゼロのままです。非因果的なイメージ チャンクは、構成されたトークン バッチとマイクロバッチに完全に適合する必要があります。サポートされていないレイアウトまたはリカレント モデルの場合は、シリアル実行に切り替わらずにエラーが返されます。 26 を超えるオプションでは、依然として `fresh` ビジョンの実行が必要です。

画像ウェーブは別の KV ストリームを使用します。動的なコンテキストのサイジングでは、質問ごとの制限によって制限される、最長のマルチモーダル入力に加えて、各ストリームのヘッドルームの 1 トークン バッチが予約されます。トークン数とポジション範囲の両方がその制限に対してチェックされます。各質問は `--context` によって制限されたままです。メモリと画像の埋め込みは、バッチまたはエラーの後にクリアされます。最後の部分ウェーブはリクエストと 判断 順序を保持します。バッチが失敗した場合、部分的な応答は返されません。

`backend.vision_batch_metrics()` は、最新の ネイティブ ウェーブのカウンターを公開します。並列イメージ HTTP 応答には、`backend.details.vision_batch` の下に `scope: "last_native_wave"` というラベルが付けられた次のものも含まれます: `projector_encode_calls`、`projector_batch_max`、`decoder_calls`、`decoder_batch_max_sequences`、 `projector_reused_chunks`、`kv_clear_calls`、および `kv_clear_skipped`。 2 つのクリア カウンタには `kv_clear_scope: "since_last_native_vision_start"` があります。次の ネイティブ ビジョン コールでリセットされる前に、後続のクリーンアップまたは構成の呼び出しでこれらの値がインクリメントされる可能性があります。準備キャッシュ カウンターには、構成以降の バックエンド ライフタイムをカバーする別の `preparation_cache_scope` があります。これらのカウンターは、実際のエンコーダーとデコーダーのバッチ形状を確立します。 HTTP リクエスト内の画像の数だけでは異なります。合計バッチ完了レイテンシーとイメージごとの償却ミリ秒を個別に測定し、同じイメージ上のスコア、上位の選択肢、判断保留、タスク 正解率 と `fresh` を比較します。

[120 イメージ TrashNet 測定 ](benchmarks/trashnet-vision-20260925/REPORT.md#native-four-image-batching-2026-09-26) は、RTX 3080 上の 4 つの ネイティブ デコーダー シーケンスを検証し、償却処理時間は短くなりましたが、3 つの CUDA はすべて検証されました。チェックポイントは既存の数値等価性 判断基準 に失敗しました。選択された値または生のランキングが変更されました。画像のバッチ処理は、モデルとレイアウトの制限内でサポートされます。 ネイティブ バッチ カウンターと分離テストでは、スコアの同等性やタスク 正解率 は確立されません。

<a id="optimization-components"></a>
### 最適化コンポーネント

最適化された 4 つのイメージ プロファイルには、正確なプロンプト準備キャッシュ、コンパクトな証拠転送、ダーティのみの物理的 KV クリアが含まれます。これらのコンポーネントは、変更されていないコンピューティング構成で既存のセッターを使用して個別にテストされます。それらのスコアの同等性は、フレッシュとパラレルの同等性を確立しません。シリアル キャッシュとコンパクトの比較では高速化は確立されませんでした。これらのコンポーネントは `--vision-optimized` に含まれています。

新しいイメージの HTTP 応答は、バックエンド ライフタイム キャッシュ スナップショットを含む `backend.details.vision_preparation` と、最後の ネイティブ 呼び出しの `vision_kv_clear` を公開します。すべてのシリアル メディア グループは同じ最終スナップショットを受信するため、共通の応答メタデータの一貫性が保たれます。これらのフィールドは、イメージまたは KV の再利用をアサートしません。

`--vision-projector-reuse` は、パラレル モードでの明示的なプロジェクターの再利用でサポートされている設定です。一意のチャンクをそれぞれ個別にエンコードし、1 つの Wave 内で同一のチャンクを再利用します。シリアル デコーダの数値実行は保存されません。

<a id="combined-vision-optimizations"></a>
### 視覚の最適化を組み合わせる

`--vision-optimized` には、一致するプロジェクターと、明示的に選択された CUDA または Metal デバイスが必要です。並列幅 4、ストリームごとの動的コンテキスト、トークン バッチ/マイクロバッチ 1024、Flash Attention オン、コンパクトな証拠、8 MiB 準備キャッシュ、および同一画像プロジェクターの再利用を選択します。質問ごとのコンテキストのデフォルトは 4096 です。 Rust 呼び出し元は、`ComputeOptions::vision_optimized()` を使用して バックエンド を構築し、プロジェクターをロードしてから、`enable_vision_optimizations()` を呼び出します。

ネイティブ メモリは、部分的な書き込み後に失敗する操作を含む、すべてのデコードまたはスナップショットの復元に入る前に書き込みを追跡します。 `sd_clear` は論理的にキャッシュされたトークンを常に無効にしますが、ダーティな場合は物理的な KV をゼロにするだけです。成功したビジョン呼び出しは、ネイティブ クリーンアップを所有します。 Rust ガードは、検証/準備/スコアリング エラーおよびアンワインド時にクリアされます。これにより、リクエストとセッションの分離を維持しながら、重複したクリアがスキップされます。

準備キャッシュには、正確なレンダリング状態/判断 プロンプトと応答境界マッピングが保持されます。画像バイトまたは KV は保存されません。同一のバイトを持ち、順序付けられたチャンク/位置メタデータが一致する画像は、1 つの Wave 内で不変のプロジェクター エンベディングを共有できます。この再利用モードでは、一意の各チャンクが単一チャンクのエンコーダー バッチを使用するため、別のイメージを変更してもエンコーダー バッチの形状やスロットを変更できません。これにより、プロジェクターのバッチ処理の代替手段が選択されます。デコーダのバッチ処理は保持されます。デコーダ KV ストリームと画像位置は独立したままになります。 `backend.vision_projector_reuse` は有効な設定を報告し、`projector_reused_chunks` は実際に再利用されたチャンクを報告します。これは、グローバル イメージとタイルを作成するプロジェクターのイメージ数を超える可能性があります。

コンパクト転送は、各 判断 の候補とその完全な語彙ノーマライザーのみを Rust にコピーします。 llama.cpp は依然として完全な語彙を計算し、その出力をホストに転送します。最適化されたプロファイルは、テキストのみの KV プレフィックス セッションやスナップショットの復元と独立したビジョン ストリームを組み合わせません。これらは代替実行モードです。ハイブリッド/リカレント モデルとワイド応答コードは、このプロファイルによって明示的に拒否されます。

並列 attention とプロジェクターのバッチ形状はいずれも数値結果を変え得ます。fresh をデフォルトにし、元の入力で統合プロファイルと fresh を比較します。同じ計算設定での全・簡略化の同等性は、fresh・最適化の同等性とは別に検証します。Metal 実行・性能には引き続き実ハードウェアテストが必要です。

<a id="validation-and-measurement"></a>
## 検証と測定

視覚分離と数値的等価性は別個のテストです。 `SKID_VISION_MODEL`、`SKID_VISION_MMPROJ`、およびオプションで `SKID_CUDA=1` を設定し、次を実行します。

```sh
cargo test --release --locked --features llama-cuda --test vision \
  real_vision_batch_preserves_image_state_order_and_recovers_after_errors \
  -- --ignored --test-threads=1
cargo test --release --locked --features llama-cuda --test vision \
  real_vision_batch_equivalence_on_color_fixture -- --ignored --test-threads=1
python3 benchmarks/trashnet-vision-20260925/evaluate_batch.py --help
```

分離テストが成功した場合でも、等価性テストは CUDA で失敗する可能性があります。評価者は、許容範囲を増やしたり、バッチ処理された結果をシリアル推論に置き換えたりする代わりに、そのような失敗を比較の際に保持します。

```sh
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  cargo test --release --locked --offline --features llama-cuda \
  --test parallel real_model_parallel_contract -- --ignored --nocapture
SKID_MODEL=models/gemma-4-E2B-it-Q8_0.gguf SKID_CUDA=1 \
  SKID_PARALLEL_WIDTH=4 SKID_PARALLEL_OUTPUT=/tmp/parallel.json \
  cargo test --release --locked --offline --features llama-cuda \
  --test parallel real_model_parallel_measurement -- --ignored --nocapture
```

The contract test covers mixed typed questions, output IDs/order, unequal prompt lengths, a partial wave, shared-prefix reuse, A–Z candidate mapping, untrusted token-like text, cross-question isolation, changed-state request isolation, error recovery after an earlier wave completed, invalid widths, and width-one equivalence to fresh execution.

測定では、短期および長期の倉庫状態に関する 1、 4、 16、および 32 の質問を使用します。質問は 3 つのウェアハウス 判断基準 を繰り返してスケーリングを測定します。これらは、32 の個別のグラウンドトゥルース タスクではありません。各モードには、1 回の時間制限なしのウォームアップ (コンテキストの割り当てを含む) と 3 回の時間制限のある繰り返しがあります。モードの順序は構成間で交互になります。詳細な JSON は、すべてのスコアとレイテンシーを保持します。 It reports serial-versus-parallel differences instead of asserting that distinct batch shapes are numerically identical; 0.02 remains the existing probability/mass comparison threshold and any changed top-1 or accepted selection fails the reported equivalence 判断基準.

These measurements exclude model/context startup from steady-state latency, do not include a remote API/network, and do not establish Jev performance parity or calibrated confidence.

<a id="numerical-limitations"></a>
## 数値的制限

並列実行ではバッチの形状が変化し、確率、上位の選択肢、受け入れられた判断が変化する可能性があります。以前のラベル付き フィクスチャー チェックでは、新しい 判断保留 が間違って受け入れられた回答になることが観察されました。より広い フィクスチャー は、既存の 0.02 確率/質量許容誤差を満たしていません。このモードはサポートされており、`--execution-mode parallel` で明示的に選択されます。同じポリシーと同等のしきい値を維持しながら、正確な チェックポイント およびワークロードでの新たな実行と比較します。
