# Documentation / 문서 / ドキュメント

| Language / 언어 / 言語 | Introduction / 소개 / 紹介 | Index / 색인 / 索引 |
| --- | --- | --- |
| English | [README](../README.md) | [Documentation index](en/README.md) |
| 한국어 | [README](../README.ko.md) | [문서 색인](ko/README.md) |
| 日本語 | [README](../README.ja.md) | [ドキュメント索引](ja/README.md) |

```text
docs/
├── README.md           # Language directory
├── translations.json  # Public document inventory
├── en/                # English index and complete bodies
├── ko/                # Korean index and complete bodies
├── ja/                # Japanese index and complete bodies
└── *.md               # Existing document paths retained for compatibility
```

Each language directory contains an index and complete bodies for all public guides, package documentation, benchmark reports, and skill references. Every document links to the same document in all three languages and to all three indexes. Guides use the original filename; package, benchmark, and skill documents retain their source hierarchy inside each language directory. Original heading anchors are preserved across translations. Existing document paths remain available for repository links, MCP resources, and published references. Keep all three bodies aligned when updating a document; `translations.json` records their source paths and translation revisions.

각 언어 디렉토리는 모든 공개 가이드·패키지 문서·벤치마크 보고서·스킬 참조의 색인과 전체 본문을 제공합니다. 각 문서는 같은 문서의 세 언어판과 세 언어 색인에 연결됩니다. 가이드는 원래 파일명, 패키지·벤치마크·스킬 문서는 원래 하위 디렉토리 구성을 유지합니다. 번역에도 원문 제목 앵커를 유지합니다. 기존 문서 경로는 저장소·MCP·외부 참조와의 호환성을 위해 남깁니다. 문서 갱신 시 세 본문을 함께 맞추고, `translations.json`에서 원본 경로와 번역 기준 리비전을 확인하세요.

各言語ディレクトリに、全公開ガイド、パッケージ文書、ベンチマークレポート、スキル参照の索引と完全な本文があります。各文書は同じ文書の 3 言語版と各索引にリンクします。ガイドは元のファイル名、パッケージ・ベンチマーク・スキルは元の階層を保持します。翻訳でも元の見出しのアンカーを保持します。既存パスはリポジトリ・MCP・外部参照との互換性のため残します。更新時は 3 本文を揃え、`translations.json` で原本パスと翻訳の基準リビジョンを確認してください。
