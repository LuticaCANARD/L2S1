<a id="l2s1--llm-to-system-1-website"></a>
# L2S1 — LLM to System 1 website

[English](README.md) · [한국어](../../ko/web/README.md) · [日本語](../../ja/web/README.md)

[English index](../README.md) · [한국어 색인](../../ko/README.md) · [日本語索引](../../ja/README.md)

The English SvelteKit introduction is prerendered with `adapter-static`. Its contract illustration uses labeled synthetic probabilities; the performance table contains recorded model-test summaries. `/webgpu` offers real Qwen3 0.6B browser inference in English, Korean and Japanese, with an explicit larger local GGUF server option. Browser-mode inputs remain in the worker; local-mode inputs are sent only when analysis is requested. Models download on demand and are not bundled with the site. See the [execution guide](../WEBGPU_DEMO.md).

```sh
npm ci
npm run check
npm run lint
npm run build
npm run dev
```

Deploy the generated `build/` directory to a static host. No deployment has been performed by this setup.

For browser checks:

```sh
npx playwright install chromium
npm run build
npm test
```

`PLAYWRIGHT_CHROMIUM_EXECUTABLE` can select an already installed compatible Chromium. Browser tests cover prerendering without JavaScript, keyboard-controlled abstention, all decision types, clipboard feedback, document downloads, and desktop/mobile overflow. Screenshots are saved under `test-results/`.

`scripts/sync-content.mjs` mirrors the root English/Korean/Japanese READMEs, licenses, top-level Markdown files in `docs/`, the complete `docs/en/`, `docs/ko/`, and `docs/ja/` language directories, public package and website READMEs, benchmark README/report files, and warehouse JSON examples into `static/docs/`. Repository-relative documentation links retain their paths. Each language directory contains an index and complete public document bodies; same-document language links and source heading anchors are preserved. It does not copy model directories or original model outputs. An explicit five-file allowlist publishes the typed-decisions summaries, 4,000 scored records and a portable manifest under `/benchmarks/typed-decisions-20260926/`. Summaries and scored JSONL preserve the checked-in bytes; the manifest embeds both model run records and labels its absolute-path normalization. The records exclude original states and generated thought text. `src/lib/benchmarks.json` is a selected, path-free summary of actual measurements; update it only from newly completed performance runs.

The site source follows the repository's MIT license. Framework and dependency notices are recorded separately in `THIRD_PARTY_LICENSES.txt` and copied into the static build. Build tools and tests are not part of the shipped client runtime.
