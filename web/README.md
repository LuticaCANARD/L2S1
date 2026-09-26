# L2S1 — LLM to System 1 website

[English](../docs/en/web/README.md) · [한국어](../docs/ko/web/README.md) · [日本語](../docs/ja/web/README.md)

[English index](../docs/en/README.md) · [한국어 색인](../docs/ko/README.md) · [日本語索引](../docs/ja/README.md)

The English SvelteKit introduction is prerendered with `adapter-static`. Its contract illustration uses labeled synthetic probabilities; the performance table contains recorded model-test summaries. `/webgpu` offers real Qwen3 0.6B browser inference in English, Korean and Japanese, with an explicit larger local GGUF server option. Browser-mode inputs remain in the worker; local-mode inputs are sent only when analysis is requested. Models download on demand and are not bundled with the site. See the [execution guide](../docs/en/WEBGPU_DEMO.md).

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

The image/text demo is at `/demo`; see [IMAGE_DEMO.md](../docs/IMAGE_DEMO.md) for the request contract, direct/thinking modes, policy controls, and local server setup. Recorded observations are served from `static/demo/`; the existing landing-page playground continues to use labeled synthetic scores.

The `/webgpu` page runs an actual Qwen3 ONNX model in a browser worker using WebGPU. Loading is opt-in; inputs are not submitted to inference servers. The model and pinned runtime are downloaded remotely, and the dedicated cache can be cleared. See [WEBGPU_DEMO.md](../docs/WEBGPU_DEMO.md) for direct/thinking scoring, download sizes, adapter requirements, and differences from native GGUF measurements.

## Language and appearance

All three pages (`/`, `/demo`, `/webgpu`) offer Korean, English and Japanese in
one shared display-preferences bar. `?lang=ko`, `?lang=en` and `?lang=ja` take
priority over the saved choice; otherwise the browser's supported language is
used, with English as the fallback. Choices use the browser-local
`l2s1-locale` and `l2s1-theme` keys. Display controls still work if storage is
unavailable, although a choice cannot then persist across a full reload.

Appearance supports light, dark and system. The system option follows OS color
scheme changes; saved appearance is applied before rendering the page. Language
changes update labels, document language and page metadata without rerunning
inference. User-edited input and failure messages, model prompts, recorded raw
responses and benchmark values retain their original data.

Translation catalogs live under `src/lib/i18n/`; `src/theme.css` supplies shared
appearance tokens. Preview an explicit language with `/webgpu?lang=ja`.
