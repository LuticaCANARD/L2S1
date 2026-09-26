# L2S1 — LLM to System 1 website

An English SvelteKit introduction site, prerendered with `adapter-static`. It runs no inference and distributes no model weights. The interactive example uses labeled synthetic probabilities. The performance table contains measured summaries from the repository's opt-in model tests.

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

`scripts/sync-content.mjs` mirrors the root English/Korean READMEs, licenses, Markdown files in `docs/`, benchmark README/report files, and warehouse JSON examples into `static/docs/`. Repository-relative documentation links retain their paths. It does not copy model directories or raw benchmark observations. `src/lib/benchmarks.json` is a selected, path-free summary of actual measurements; update it only from newly completed performance runs.

The site source follows the repository's MIT license. Framework and dependency notices are recorded separately in `THIRD_PARTY_LICENSES.txt` and copied into the static build. Build tools and tests are not part of the shipped client runtime.
