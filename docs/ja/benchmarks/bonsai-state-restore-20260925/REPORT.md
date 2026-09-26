<a id="bonsai-request-local-state-restoration-on-rtx-3060"></a>
# Bonsai リクエスト - RTX でのローカル状態の復元 3060

[English](../../../en/benchmarks/bonsai-state-restore-20260925/REPORT.md) · [한국어](../../../ko/benchmarks/bonsai-state-restore-20260925/REPORT.md) · [日本語](REPORT.md)

[English index](../../../en/README.md) · [한국어 색인](../../../ko/README.md) · [日本語索引](../../README.md)

`lucatagpu`、ドライバー 595.71.05 を備えた NVIDIA GeForce RTX 3060 12 GiB 上の 2026-09-25 で測定。ハイブリッド チェックポイント は、`prefix-reuse` モードでの新規実行にフォールバックします。明示的な `state-restore` モードは、共通プレフィックスの完全な llama.cpp シーケンス状態を保存し、その状態で最初のサフィックスを実行し、その後の各サフィックスに対してそれを復元します。この状態は、1 つの ネイティブ リクエスト中にのみ存在します。スナップショットが使用できない場合、実行は新しい状態に戻ります。

<a id="setup"></a>
## セットアップ

- L2S1 はコミット `a12a99b` からのビルドを測定しました (`7be9a29` としてリベースされた同じ実装)。 CUDA バイナリ SHA-256 `edeb9b1e61afb72a0db94d71201f6d0baf63e4cfd3eb3c60bb7cf3c1e4edbdde`。 llama.cpp ソース リビジョン `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`。
- Ternary Bonsai 2 27B Q1_0 GGUF SHA-256 `17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0`、CUDA に完全にオフロードされます。
- 固定の [16-判断 ウェアハウス リクエスト ](../../../../benchmarks/bonsai-state-restore-20260925/request-16.json) とその最初の 判断 のみ。 フィクスチャー は、共有状態プレフィックスと選択判断を使用します。フラグ: `--device cuda --context 4096 --batch 256 --ubatch 256 --threads 8 --prompt-layout state-first`; 2 番目のパスは `--execution-mode state-restore` を追加します。どちらも完全な証拠の転送を使用します。
- カウントおよびモードごとに 1 つのウォームアップ リクエストと 3 つの測定されたループバック HTTP リクエスト (モードごとに 1 つの常駐モデル)。タイミングには HTTP および JSON の処理が含まれますが、起動とモデルの読み込みは含まれません。モードは順番に実行されました。 GPU メモリは実行ごとにサンプリングされました。 [raw の概要 ](../../../../benchmarks/bonsai-state-restore-20260925/summary.json) には、測定されたすべての経過時間と結果の比較が含まれています。

<a id="results"></a>
## 結果

| モード | 1 判断 p50 | 16 判断 p50 | 16 判断の高速化 | 16 判断の再利用プレフィックストークン | 最大 GPU メモリ |
| --- | ---: | ---: | ---: | ---: | ---: |
| fresh/フル | 757.0 ms | 12,207.9 ms | 1.00× | 0 | 4,309 MiB |
| 状態復元/完全 | 766.1 ms | 6,976.4 ms | **1.75×** | 3,840 | 4,309 MiB |

3 つの 16 ～ 判断 クライアント時間は、新規の場合は 12,156.0/12,207.9/12,226.3 ms、状態復元の場合は 6,973.4/6,976.4/6,981.5 ms でした。復元されたリクエストは、173,678,124 バイトの状態スナップショットを保存し、後の 15 判断のために 15 回ロードしました。最初の 判断 は、元の事前入力状態を使用しました。その診断タイミングは、338.7 ms プレフィル、 257.2 ms 保存、 1,225.7 ms リストア、および 5,158.5 ms サフィックス実行でした。 one-判断 リクエストには再利用されたトークンがありませんでした。

測定されたすべてのリクエストについて、復元されたパスには、**zero** の変更された選択と生の上位選択肢があり、オプションの確率とfresh状態に対する 候補の確率質量 の最大差は **zero** でした。すべての 16 の生の上位選択肢は、フィクスチャー の単純なルールに一致しました。デフォルトのポリシーは両方のパスで 15/16 判断を受け入れました。リアルモデル回帰では、オプションごとのトークン ID、logits、確率、候補の確率質量、値、および 判断保留 の理由も比較されました。このチェックポイントを渡しました。ゼロバイトのスナップショット制限により、`snapshot_memory_budget` が報告され、再利用トークンなしで新たに実行され、同じ結果が得られました。通常の制限に戻すと、再度再利用が可能になります。

別個の [three-判断 CUDA 適合結果 ](../../../../benchmarks/bonsai-state-restore-20260925/conformance.json) は、盆栽に関するバイナリ、選択、および 順序付き の判断をカバーしています。状態復元では、167,384,364 バイトのスナップショットを 2 回ロードしました。オプション確率と候補質量の差はゼロで、最上位の選択肢や受け入れられた値と新しい値は変更されませんでした。ゼロバイト制限により、再び `snapshot_memory_budget` が生成され、リストアはゼロになりました。これにより、別のプロンプト形状がチェックされます。これはラベル付きの 正解率 セットではありません。

<a id="reproduction-and-limits"></a>
## 生殖と限界

同じ GGUF を持つ CUDA ホストで、`L2S1_BONSAI_MODEL=/path/to/Bonsai-27B-Q1_0.gguf cargo test --release --locked --features llama-cuda --test state_restore_bonsai -- --ignored --nocapture` を使用して無視された回帰を実行します。 `L2S1_CONFORMANCE_MODELS=/path/to/Bonsai-27B-Q1_0.gguf SKID_CUDA=1 cargo test --release --locked --features llama-cuda --test conformance -- --ignored --nocapture` で他のコントラクトを実行します。 CLI 診断の場合は、リンクされたリクエストを指す `--input` を含む上記のフラグを使用し、`--diagnostics` を追加します。完全なベンチマーク出力とテスト ログは、測定対象ホストの `~/personal/skid/state-restore-stable-20260925/results/` に残ります。

これは、1 つの GGUF および GPU 上の 1 つの合成 フィクスチャー の実行同等性とレイテンシです。ラベル付きタスクの品質、他のハイブリッド チェックポイント、クロスリクエストの再利用、または 本番環境 レイテンシの分布は測定されません。スナップショット制限は、合計プロセス メモリではなく、スナップショット バッファを制御します。 `state-restore` はオプトインのままです。 `prefix-reuse` は依然として Bonsai でのハイブリッド フォールバックを報告しています。
