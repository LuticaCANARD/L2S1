# ドキュメント索引

[English](../en/README.md) · [한국어](../ko/README.md) · [日本語](../ja/README.md)

ビルドと最初の判断の実行は、[日本語 README](../../README.ja.md)から始めてください。この索引は、公開されているすべての補足ガイド、パッケージのドキュメント、ベンチマークのレポートを案内します。すべてのガイド、パッケージ文書、ベンチマークレポートに英語・韓国語・日本語の本文があります。文書の言語リンクで、同じ文書の別の言語版に移動できます。コード・コマンド・識別子・測定値は原文の値を保持します。

## 利用と統合

| ドキュメント | 内容 |
| --- | --- |
| [利用ガイド](GUIDE.md) | ビルド、スキーマ、スコア、CLI、Rust、HTTP、画像、バックエンド |
| [TypeScript ライブラリ](typescript/README.md) | 常駐モデルプロセス、HTTP クライアント、配布、検証 |
| [Python SDK](python/README.md) | 型付き非同期呼び出し、固定定義再利用、wheel、共通 Rust ランタイム |
| [バッチ API の検討](BATCHING_API_REVIEW.md) | 反復 state 入力、native batch の境界と HTTP 接続案 |
| [配布パイプライン](RELEASE_PIPELINE.md) | GitHub Release・npm・PyPI・native Cargo 公開 |
| [AI エージェント統合](AGENT_INTEGRATION.md) | 環境をまたいで使えるスキル、stdio MCP、リクエスト検証、常駐推論 |
| [ブラウザー WebGPU デモ](WEBGPU_DEMO.md) | ローカル Qwen3 ONNX 推論、WebGPU、上限付きの思考、ポリシー |
| [画像とテキストのデモ](IMAGE_DEMO.md) | 記録済みの応答、ポリシー、失敗の説明、ローカルサーバー |
| [Pages デプロイ](PAGES_DEPLOYMENT.md) | Cloudflare Pages と n2s1.luticalab.net の DNS |
| [モデルの交換](MODEL_INTERCHANGEABILITY.md) | モデル識別、事前検証、校正、診断、ワーカー、メモリ |
| [検証](VERIFICATION.md) | ビルドとモデル別の検証コマンド |
| [ライセンス](LICENSING.md) | ソース、依存関係、モデルのライセンスの区分 |

## 実行と特化

| ドキュメント | 内容 |
| --- | --- |
| [推論モード](REASONING.md) | 直接回答・ネイティブの思考、トークン上限、ランタイムの対応範囲 |
| [並列実行](PARALLEL_EXECUTION.md) | テキスト・画像のバッチ処理、動的コンテキスト、プロジェクター再利用、ビジョンプロファイル |
| [プレフィックスアルゴリズム](SEMIF_ALGORITHM.md) | プレフィックスの準備と再利用 |
| [判断のファインチューニング](DECISION_FINETUNE.md) | LoRA の手順と記録済みの評価 |
| [出力ヘッド](OUTPUT_HEAD.md) | タスク別のヘッドと成果物の紐付け |

## 評価手法と結果

| ドキュメント | 内容 |
| --- | --- |
| [モデルの測定結果](MODEL_RESULTS.md) | チェックポイントの比較、範囲、測定上の限界 |
| [合成ベンチマーク](BENCHMARK.md) | ルールに基づくテストデータの正解、判断保留、一貫性、レイテンシ |
| [JevBench](JEVBENCH.md) | 公開タスクの対応付け、元の採点方法、採用指標 |
| [Ollaya 比較](OLLAYA_COMPARISON.md) | 根拠に基づく違いと、残る品質・製品面の課題 |
| [typed-decisions](TYPED_DECISIONS_BENCHMARK.md) | 2,000 判断の全テスト、校正、測定手順 |
| [意図分類](INTENT_BENCHMARK.md) | 多候補の回答コード、BANKING77、MASSIVE 韓国語 |
| [AG News](KAGGLE_BENCHMARK.md) | 固定された分類手順 |
| [Laya/Jev タスクとキャッシュ](LAYA_BENCHMARK.md) | タスクの変換と CPU・GPU キャッシュの検証 |
| [画像ベンチマーク](VISION_BENCHMARK.md) | 画像の直接推論と時間測定の範囲 |

## タスク・ハードウェアのレポート

| ドキュメント | 内容 |
| --- | --- |
| [Caltech-101](benchmarks/caltech101-vision-20260924/README.md) | 静止画像の分類 |
| [猫と犬](benchmarks/cats-dogs-vision-20260924/REPORT.md) | 二値の画像分類 |
| [TrashNet と画像処理のスループット](benchmarks/trashnet-vision-20260925/REPORT.md) | 画像分類と実行モード |
| [汎用 GGUF CUDA スモークテスト](benchmarks/gguf-cuda-20260925/README.md) | CUDA モデルのスモーク測定 |
| [Bonsai 状態復元](benchmarks/bonsai-state-restore-20260925/REPORT.md) | 状態復元の測定 |
| [共有状態キャッシュ](benchmarks/shared-state-cache-20260925/REPORT.md) | キャッシュの測定 |
| [ビルドキャッシュ](benchmarks/build-cache-20260925/REPORT.md) | ネイティブビルドキャッシュの測定 |
| [typed-decisions 成果物](benchmarks/typed-decisions-20260926/README.md) | 保存済みの要約、記録、再現用メタデータ |

## ソースとツール

| ドキュメント | 内容 |
| --- | --- |
| [アーキテクチャと構成要素](GUIDE.md#architecture) | ソースの構成 |
| [ネイティブ llama.cpp 依存関係](crates/l2s1-llama-sys/README.md) | 固定されたネイティブランタイムとブリッジ |
| [データセット・レポートツール](crates/l2s1-tools/README.md) | データセットの準備、評価、再集計 |
| [倉庫のリクエスト例](../../examples/warehouse.json) | 二値・選択・順序付きのリクエスト |
| [ドキュメントサイト](web/README.md) | Svelte サイト、デモ、公開ドキュメントの出力 |
| [npm 公開](typescript/PUBLISHING.md) | プラットフォームランタイムの配布、公開条件、検証範囲 |
| [実験ツールの表記](crates/l2s1-tools/NOTICE.md) | JevBench・Laya の出典とデータセットの範囲 |
| [L2S1 スキル](skills/l2s1/SKILL.md) | 統合の指示と根拠の処理 |
| [スキルのリクエスト参照](skills/l2s1/references/decisions.md) | リクエスト例、型付き結果、HTTP・MCP フィールド |
| [スキルのインターフェース参照](skills/l2s1/references/interfaces.md) | ビルド、CLI、HTTP、MCP、Rust の統合 |

ソース・依存関係の表記: [LICENSE](../../LICENSE)、[THIRD_PARTY_LICENSES.txt](../../THIRD_PARTY_LICENSES.txt)。モデルの重みは別途用意し、それぞれの条件に従います。レポートは記載された範囲での実験記録です。正解率、採用された判断の正解率、採用率を区別してください。
