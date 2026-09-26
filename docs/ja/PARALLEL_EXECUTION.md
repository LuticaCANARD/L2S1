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

デフォルトは、`legacy` プロンプトと `fresh` 実行のままです。プロンプトのレイアウトと実行モードは個別に選択できます。状態を移動するとプロンプトが変わります。並列処理により実行スケジュールが変更されます。同じレイアウトですべての実行モードを比較します。

<a id="implementation"></a>
## 実装

- Rust は、各質問のトークンと A ～ Z の単一トークンの継続を準備して検証します。ユーザー状態と質問データは特別なトークン解析保護を保持します。
- 各ウェーブには最大で `parallel_width` の質問が入力されます (デフォルトは 4、許可される 1–32)。最後の波はもっと小さくなる可能性があります。結果には、異なるデコード バッチでプロンプトがいつ終了したかを含め、リクエストの順序と ID が保持されます。
- 共通のプレフィックスは、質問ごとに少なくとも 1 つのサフィックス トークンを保持しながら、Wave 全体で正確なトークン ID を比較することによって見つかります。これは、完全なオリジナルのプレフィル バッチに切り捨てられます。最初のシーケンスはそれを評価します。 `llama_memory_seq_cp` は、その KV エントリを他のシーケンスと共有します。
- サフィックス トークンには、独立したシーケンス ID と元の絶対位置があります。彼らの注意は、別の質問の接尾辞ではなく、自分自身のシーケンスと共有接頭辞に注目します。最終 全語彙 logits は、関連するデコード バッチ内の各質問の最終トークン インデックスを使用してコピーされます。
- メモリはウェーブ間、API 呼び出し間、および失敗後にクリアされます。永続的なクロスリクエスト キャッシュはありません。このモードでは、リカレント/ハイブリッド モデルは明示的にサポートされていません。
- モデルは一度ロードされたままになります。デフォルトでは、シーケンス容量が変更されると、公称トークン容量 `context * width` でコンテキストが遅延して再作成されます。 `context` は質問ごとの入力制限のままです。幅を大きくすると、KV/アテンション メモリが増加し、割り当てに失敗する可能性があります。サイレント CPU フォールバックや入力切り捨てはありません。
- `backend.parallel_width` は、設定された波の制限を記録します (シリアル モードは 1 をレポートします)。 Wave の最初の質問では、`reused_prefix_tokens` はゼロであり、共有プレフィックスとそのフォロワーの共有トークン数の支払いが行われます。合計された `input_tokens - reused_prefix_tokens` は、実際に送信されたトークンを表します。

これは、固定された llama.cpp [batch、シーケンス、メモリ コピー、およびトークンごとの logits API](https://github.com/ggml-org/llama.cpp/blob/3d82ef62d47fd74e18f36c5eccbdcf965b617b17/include/llama.h) を使用します。これはシリアル ウェーブによる独立したシーケンス バッチ処理であり、置換モデル アーキテクチャや校正済みの 判断 トレーニングではありません。

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
