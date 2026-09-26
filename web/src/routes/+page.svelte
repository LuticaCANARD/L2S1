<script lang="ts">
  import benchmarks from '$lib/benchmarks.json';
  import typedDecisions from '$lib/typed-decisions.json';
  import rtx3060 from '$lib/rtx3060.json';
  import sharedState from '$lib/shared-state-highlight.json';
  import { locale, translate } from '$lib/i18n';
  import { messages } from '$lib/i18n/landing';
  const t = (key: keyof typeof messages.en, params?: Record<string, string | number>) => translate($locale, messages, key, params);
  const installs = [
    { name: 'Python 3.11+', command: 'pip install l2s1-sdk==0.1.1', guide: 'python/README.md' },
    { name: 'TypeScript / Node.js 22+', command: 'npm install @l2s1/node@0.1.1', guide: 'typescript/README.md' },
    { name: 'Rust', command: 'cargo add l2s1@0.1.1 --features llama', guide: 'GUIDE.md' }
  ];

  type Kind = 'choice' | 'binary' | 'ordinal';
  type BenchmarkRow = { model: string; cpuP50: number; cudaP50: number; cudaDecisionsPerSecond: number; cudaAbstentionRate: number; nativeCpu: boolean; nativeCuda: boolean };
  type TypedRow = { model: string; accuracy: number; coverage: number; acceptedAccuracy: number | null; correctAll: number; p50Ms: number; p95Ms: number };
  const typedRows = typedDecisions.models as TypedRow[];
  const measurements = benchmarks.rows as BenchmarkRow[];
  const nativeMeasurements = measurements.filter((row) => row.nativeCuda);
  const latencyScale = Math.ceil(Math.max(1, ...nativeMeasurements.map((row) => Math.max(row.cpuP50, row.cudaP50))) / 1000) * 1000;
  const compact = rtx3060.rows.find((row) => row.id === 'Qwen3-0.6B-Q8_0')!;
  const best3060 = rtx3060.rows.reduce((best, row) => row.rawAccuracy > best.rawAccuracy ? row : best);
  const quantized9b = rtx3060.rows.find((row) => row.id === 'Qwen3.5-9B-Q4_K_M')!;
  const q8_9b = rtx3060.rows.find((row) => row.id === 'Qwen3.5-9B-Q8_0')!;
  const memorySaved = (1 - quantized9b.peakGpuMiB / q8_9b.peakGpuMiB) * 100;
  let benchmarkView = $state<'rtx3060' | 'warehouse'>('rtx3060');
  function typedFor(model: string): TypedRow | undefined {
    const key = model.toLowerCase().replace(/[^a-z0-9]/g, '');
    return typedRows.find((row) => row.model.toLowerCase().replace(/[^a-z0-9]/g, '').includes(key));
  }
  const additionalTypedRows = typedRows.filter((row) => !measurements.some((measurement) => typedFor(measurement.model) === row));
  const typedPercent = (value: number | null | undefined) => value === null || value === undefined ? '—' : `${value.toFixed(2)}%`;
  const fileGiB = (bytes: number) => `${(bytes / 1024 ** 3).toFixed(2)} GiB`;
  const modelSize = (id: string) => rtx3060.rows.find((row) => row.id === id)!.fileSizeBytes;
  const typedMs = (value: number | undefined) => value === undefined ? '—' : `${value.toFixed(1)} ms`;
  let kind = $state<Kind>('choice');
  let copied = $state(false);
  let copyError = $state(false);
  const choices = {
    choice: [{ code: 'A', labelKey: 'ambient', probability: 0.02 }, { code: 'B', labelKey: 'chilled', probability: 0.96 }, { code: 'C', labelKey: 'frozen', probability: 0.02 }],
    binary: [{ code: 'A', labelKey: 'no', probability: 0.08 }, { code: 'B', labelKey: 'yes', probability: 0.92 }],
    ordinal: [{ code: 'A', labelKey: 'low', probability: 0.03 }, { code: 'B', labelKey: 'medium', probability: 0.09 }, { code: 'C', labelKey: 'high', probability: 0.88 }]
  } as const;
  const questions = { choice: 'questionChoice', binary: 'questionBinary', ordinal: 'questionOrdinal' } as const;
  const typeLabels = { choice: 'choiceType', binary: 'binaryType', ordinal: 'ordinalType' } as const;
  let options = $derived(choices[kind]);
  let result = $derived(kind === 'choice'
    ? { type: 'choice', selected: 'chilled' }
    : kind === 'binary'
      ? { type: 'binary', p_true: 0.92, value: true }
      : { type: 'ordinal', expected_value: 1.85, selected: 'high' });
  let resultJson = $derived(JSON.stringify({ value: result }, null, 2));
  const command = `cargo run --release --features llama -- \\\n  --model /path/to/chat-model.gguf \\\n  --input examples/warehouse.json \\\n  --device cpu`;
  async function copyCommand() {
    try {
      await navigator.clipboard.writeText(command);
      copied = true;
      copyError = false;
      window.setTimeout(() => copied = false, 2500);
    } catch {
      copyError = true;
    }
  }
  const models = [
    { name: 'Gemma 4', id: 'gemma-4-E2B-it-Q8_0', detailKey: 'gemma4Detail', url: 'https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF' },
    { name: 'Gemma 3', id: 'gemma-3-1b-it-Q8_0', detailKey: 'gemma3Detail', url: 'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF' },
    { name: 'Qwen3', id: 'Qwen3-0.6B-Q8_0', detailKey: 'qwenDetail', url: 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF' }
  ] as const;
</script>

<svelte:head>
  <title>{t('metaTitle')}</title>
  <meta name="description" content={t('metaDescription')} />
  <meta property="og:title" content={t('metaTitle')} />
  <meta property="og:description" content={t('ogDescription')} />
  <meta property="og:type" content="website" />
</svelte:head>

<a class="skip-link" href="#main">{t('skip')}</a>
<div class="page-shell">

  <main id="main">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow"><span class="status-dot"></span> {t('eyebrowHero')} <span class="version">v0.1.1</span></p>
        <h1>{t('heroTitle')}<br /><em>{t('heroTitleEmphasis')}</em></h1>
        <p class="hero-description">{t('heroDescription')}<br />{t('heroDescriptionAnswer')}</p>
        <p class="hero-detail">{t('heroDetail')}</p>
        <div class="hero-actions"><a class="button primary" href="#get-started">{t('getStarted')} <span aria-hidden="true">↗</span></a><a class="text-link" href="#playground">{t('exploreContract')} <span aria-hidden="true">↓</span></a></div>
        <div class="hero-tags"><span>{t('mitCode')}</span><span>{t('devices')}</span><span>{t('ownModel')}</span><span>llama.cpp · GGUF</span></div>
      </div>
      <div class="decision-window" aria-label={t('illustrationAria')}>
        <div class="window-bar"><span><i></i><i></i><i></i></span><span>decision / storage_zone</span><span>01</span></div>
        <div class="window-content">
          <div class="terminal-label"><span>{t('inputState')}</span><span>JSON</span></div>
          <pre class="input-preview"><span class="code-muted">&#123;</span>
  <span class="code-key">"storage_requirement"</span>: <span class="code-string">"chilled"</span>,
  <span class="code-key">"hours_until_dispatch"</span>: <span class="code-number">4</span>
<span class="code-muted">&#125;</span></pre>
          <div class="flow-line"><span></span><b>↓</b><span>{t('localModel')}</span></div>
          <div class="option-stack"><div><span class="letter">A</span><span>{t('ambient')}</span><span class="option-mark">—</span></div><div class="selected"><span class="letter">B</span><span>{t('chilled')}</span><span class="option-mark">↗</span></div><div><span class="letter">C</span><span>{t('frozen')}</span><span class="option-mark">—</span></div></div>
          <div class="flow-line small"><span></span><b>↓</b><span>{t('typedResult')}</span></div>
          <div class="result-preview"><span class="code-muted">&#123; </span><span>"selected"</span>: <strong>"chilled"</strong><span class="code-muted"> &#125;</span></div>
          <p class="illustration-note">{t('illustrationNote')}</p>
        </div>
        <span class="window-offset" aria-hidden="true">{t('oneQuestion')}</span>
      </div>
    </section>

    <div class="principles"><span><b>01</b> {t('principleData')}</span><span><b>02</b> {t('principleScores')}</span><span><b>03</b> {t('principleUncertainty')}</span></div>

    <section id="how-it-works" class="section contract-section">
      <div class="section-heading"><p class="eyebrow">{t('contractEyebrow')}</p><h2>{t('contractTitle')}<br /><em>{t('contractTitleEmphasis')}</em></h2><p>{t('contractDescription')}</p></div>
      <div class="contract-grid">
        <article><span class="card-index">A → B → C</span><h3>{t('choiceTitle')}</h3><p>{t('choiceDescription')}</p><code>{'{ "selected": "chilled" }'}</code><span class="card-type">{t('choiceCard')} <span>↗</span></span></article>
        <article><span class="card-index">0 / 1</span><h3>{t('binaryTitle')}</h3><p>{t('binaryDescription')}</p><code>{'{ "value": true, "p_true": 0.92 }'}</code><span class="card-type">{t('binaryCard')} <span>↗</span></span></article>
        <article><span class="card-index">▂ ▄ ▆</span><h3>{t('ordinalTitle')}</h3><p>{t('ordinalDescription')}</p><code>{'{ "selected": "high" }'}</code><span class="card-type">{t('ordinalCard')} <span>↗</span></span></article>
      </div>
    </section>

    <section id="playground" class="playground-section">
      <div class="playground-copy"><p class="eyebrow">{t('playgroundEyebrow')}</p><h2>{t('playgroundTitle')}<br /><em>{t('playgroundTitleEmphasis')}</em></h2><p>{t('playgroundDescription')}</p><div class="demo-disclosure"><span>ⓘ</span><p><strong>{t('demoTitle')}</strong></p></div></div>
      <div class="playground-panel">
        <div class="segmented-control" aria-label={t('decisionType')}>{#each ['choice', 'binary', 'ordinal'] as type (type)}<button class:active={kind === type} aria-pressed={kind === type} onclick={() => kind = type as Kind}>{t(typeLabels[type as Kind])}</button>{/each}</div>
        <div class="question-line"><h3>{t(questions[kind])}</h3><span class="decision-status">{t('selected')}</span></div>
        <div class="probability-list">{#each options as option (option.code)}<div class="probability-row"><span>{option.code}</span><span>{t(option.labelKey)}</span><div class="bar-track"><div style:width={`${option.probability * 100}%`}></div></div><strong>{Math.round(option.probability * 100)}%</strong></div>{/each}</div>
        <div class="demo-result" aria-live="polite"><span class="terminal-label">RESULT.JSON</span><pre>{resultJson}</pre></div>
      </div>
    </section>

    <section id="models" class="section model-section">
      <div class="section-heading horizontal"><div><p class="eyebrow">{t('modelsEyebrow')}</p><h2>{t('modelsTitle')}<br /><em>{t('modelsTitleEmphasis')}</em></h2></div><p>{t('modelsDescription')}</p></div>
      <div class="compatibility-highlight"><div><strong>{t('llamaCompatibility')}</strong><p>{t('llamaCompatibilityDetail')}</p></div><a href={`/docs/docs/${$locale}/RTX3060_BENCHMARK.md`} rel="external">{t('llamaMeasured')} ↗</a></div>
      <div class="model-list">{#each models as model (model.name)}<a href={model.url} target="_blank" rel="noreferrer external"><span class="model-dot" aria-hidden="true"></span><strong>{model.name}</strong><span>{t(model.detailKey)}<br /><b class="model-size">{fileGiB(modelSize(model.id))}</b> · GGUF</span><span class="model-arrow" aria-hidden="true">↗</span></a>{/each}</div>
      <p class="section-note">{t('modelsNote')} <a href={`/docs/docs/${$locale}/VERIFICATION.md`} rel="external">{t('verification')}</a></p>
    </section>

    <section id="performance" class="section performance-section" aria-labelledby="performance-heading">
      <div class="performance-hero">
        <div class="performance-heading"><p class="eyebrow">{t('performanceEyebrow')}</p><h2 id="performance-heading">{t('performanceTitle')}<br /><em>{t('performanceTitleEmphasis')}</em></h2><p>{t('performanceDescription')}</p><span class="performance-hardware">RTX 3060 · 12 GB <span aria-hidden="true">↗</span> CUDA</span></div>
        <div class="performance-spotlight"><p class="metric-label">{t('performanceLatency')}</p><div class="performance-value"><strong>{compact.p50Ms.toFixed(1)}</strong><span>ms</span></div><div class="metric-rule" aria-hidden="true"></div><p class="metric-model">{compact.model}</p><p class="metric-scope">{t('rtxSpotlightScope', { accuracy: compact.rawAccuracy.toFixed(2) })}</p></div>
      </div>
      <div class="performance-stats">
        <div class="performance-stat"><span class="metric-label">{t('sampledGpuMemory')}</span><div class="stat-value">{(compact.peakGpuMiB / 1024).toFixed(2)}<span>GiB</span></div><p>{compact.model} · RTX 3060</p></div>
        <div class="performance-stat"><span class="metric-label">{t('jevRawAccuracy')}</span><div class="stat-value">{best3060.rawAccuracy.toFixed(2)}<span>%</span></div><p>{best3060.model} · JevBench</p></div>
        <div class="performance-stat"><span class="metric-label">{t('sharedStateSpeedup')}</span><div class="stat-value">{sharedState.speedup.toFixed(2)}<span>×</span></div><p>{t('sharedStateScope')}</p><a class="metric-evidence" href="/benchmarks/shared-state-cache-20260925/highlight.json" download rel="external">{t('sharedStateEvidence')} ↗</a></div>
        <div class="performance-stat"><span class="metric-label">{t('inferenceApiFee')}</span><div class="stat-value">0<span>{t('apiFeeUnit')}</span></div><p>{t('localExecutionCost')}</p></div>
      </div>
      <div class="capacity-highlight"><div><span class="metric-label">{t('modelDownloadSize')}</span><strong>{(compact.fileSizeBytes / 1024 ** 2).toFixed(1)} <small>MiB</small></strong></div><p>{t('compactCapacity')}</p><a href="/benchmarks/rtx3060-20260926/manifest.json" download rel="external">{t('capacityEvidence')} ↗</a></div>
      <p class="quantization-highlight"><strong>{memorySaved.toFixed(1)}% ↓</strong><span>{t('quantizationSaving', { accuracy: quantized9b.rawAccuracy.toFixed(2) })}</span></p>
      <div class="benchmark-tabs" role="group" aria-label={t('benchmarkDataset')}>
        <button class:active={benchmarkView === 'rtx3060'} aria-pressed={benchmarkView === 'rtx3060'} onclick={() => benchmarkView = 'rtx3060'}>RTX 3060 · JevBench</button>
        <button class:active={benchmarkView === 'warehouse'} aria-pressed={benchmarkView === 'warehouse'} onclick={() => benchmarkView = 'warehouse'}>RTX 3080 · Warehouse / Typed-decisions</button>
      </div>
      {#if benchmarkView === 'warehouse' && nativeMeasurements.length}
      <figure class="performance-chart">
        <figcaption><div><h3>{t('performanceChartTitle')}</h3><p>{t('performanceChartScope', { date: benchmarks.date })}</p></div><div class="chart-legend"><span><i class="cpu-key" aria-hidden="true"></i>CPU</span><span><i class="cuda-key" aria-hidden="true"></i>CUDA</span></div></figcaption>
        {#each nativeMeasurements as row (row.model)}<div class="latency-model"><p>{row.model}</p><div class="latency-lines"><div class="latency-row"><span>CPU</span><div class="latency-track" aria-hidden="true"><div class="cpu-bar" style:width={`${row.cpuP50 / latencyScale * 100}%`}></div></div><strong>{row.cpuP50.toFixed(1)} <span>ms</span></strong></div><div class="latency-row"><span>CUDA</span><div class="latency-track" aria-hidden="true"><div class="cuda-bar" style:width={`${row.cudaP50 / latencyScale * 100}%`}></div></div><strong>{row.cudaP50.toFixed(1)} <span>ms</span></strong></div></div></div>{/each}
        <div class="latency-axis" aria-hidden="true"><span>0</span><span>{(latencyScale / 2).toLocaleString($locale)}</span><span>{latencyScale.toLocaleString($locale)} ms</span></div>
      </figure>
      {/if}
      <div class="performance-details">
      <div class="performance-details-heading"><h3>{t('performanceDetails')}</h3><span>{t('performanceDetailsBadge')}</span></div>
      {#if benchmarkView === 'rtx3060'}
        <div class="benchmark-meta"><span class="status-dot"></span><strong>RTX 3060 · 12 GB</strong><span>{t('rtxDatasetScope', { models: rtx3060.models, decisions: rtx3060.decisions, date: rtx3060.date })}</span></div>
        <div class="table-scroll"><table><caption class="sr-only">{t('rtxTableCaption')}</caption><thead><tr><th>{t('checkpoint')}</th><th>{t('jevRawAccuracy')}</th><th>{t('policyCoverage')}</th><th>{t('policyAccuracy')}</th><th>{t('policyCorrectAll')}</th><th>p50 / p95</th><th>{t('sampledGpuMemory')}</th><th>{t('modelDownloadSize')}</th></tr></thead><tbody>{#each rtx3060.rows as row (row.id)}<tr><td>{row.model}</td><td>{typedPercent(row.rawAccuracy)}</td><td>{typedPercent(row.coverage)}</td><td>{typedPercent(row.acceptedAccuracy)}</td><td>{typedPercent(row.correctAll)}</td><td>{typedMs(row.p50Ms)} / {typedMs(row.p95Ms)}</td><td>{(row.peakGpuMiB / 1024).toFixed(2)} GiB</td><td>{fileGiB(row.fileSizeBytes)}</td></tr>{/each}</tbody></table></div>
        <p class="section-note">{t('capacityScope')} {t('rtxMethodology', { context: rtx3060.context })} GPT-OSS: <code>GGML_CUDA_DISABLE_GRAPHS=1</code>. <a href={`/docs/docs/${$locale}/RTX3060_BENCHMARK.md`} rel="external">{t('protocolReport')}</a></p>
        <p class="section-note">{t('downloadEvidence')} <a href="/benchmarks/rtx3060-20260926/summary.json" download rel="external">{t('rtxSummary')}</a> · <a href="/benchmarks/rtx3060-20260926/manifest.json" download rel="external">{t('manifest')}</a></p>
      {:else if measurements.length}
        <div class="benchmark-meta"><span class="status-dot"></span><strong>{t('localMeasurements')}</strong><span>{t('measurementMeta', { samples: benchmarks.samples })}</span></div>
        <div class="table-scroll"><table><caption class="sr-only">{t('tableCaption')}</caption><thead><tr><th>{t('checkpoint')}</th><th>{t('cpuP50')}</th><th>{t('cudaP50')}</th><th>{t('cudaThroughput')}</th><th>{t('cudaAbstentions')}</th>{#if typedRows.length}<th>{t('typedRaw')}</th><th>{t('typedCoverage')}</th><th>{t('typedAccepted')}</th><th>{t('typedCorrectAll')}</th><th>{t('typedLatency')}</th>{/if}</tr></thead><tbody>{#each measurements as row (row.model)}<tr><td>{row.model}</td><td>{row.cpuP50.toFixed(1)} <span>ms</span></td><td>{row.cudaP50.toFixed(1)} <span>ms</span></td><td>{row.cudaDecisionsPerSecond.toFixed(2)}</td><td>{Math.round(row.cudaAbstentionRate * 100)}%</td>{#if typedRows.length}{@const typed = typedFor(row.model)}<td>{typedPercent(typed?.accuracy)}</td><td>{typedPercent(typed?.coverage)}</td><td>{typedPercent(typed?.acceptedAccuracy)}</td><td>{typedPercent(typed?.correctAll)}</td><td>{typedMs(typed?.p50Ms)} / {typedMs(typed?.p95Ms)}</td>{/if}</tr>{/each}{#each additionalTypedRows as typed (typed.model)}<tr><td>{typed.model}</td><td>—</td><td>—</td><td>—</td><td>—</td><td>{typedPercent(typed.accuracy)}</td><td>{typedPercent(typed.coverage)}</td><td>{typedPercent(typed.acceptedAccuracy)}</td><td>{typedPercent(typed.correctAll)}</td><td>{typedMs(typed.p50Ms)} / {typedMs(typed.p95Ms)}</td></tr>{/each}</tbody></table></div>
        <p class="section-note">{t('warehouseNote', { date: benchmarks.date })}</p>
      {/if}
      {#if typedRows.length}<p class="section-note">{t('typedNote', { cases: typedDecisions.cases, decisions: typedDecisions.decisions, gpu: typedDecisions.gpu, date: typedDecisions.date })} <a href={`/docs/docs/${$locale}/TYPED_DECISIONS_BENCHMARK.md`} rel="external">{t('protocolReport')}</a></p><p class="section-note">{t('downloadEvidence')} <a href="/benchmarks/typed-decisions-20260926/gemma4-e2b-summary.json" download rel="external">{t('gemmaSummary')}</a> · <a href="/benchmarks/typed-decisions-20260926/qwen3-06b-summary.json" download rel="external">{t('qwenSummary')}</a> · <a href="/benchmarks/typed-decisions-20260926/gemma4-e2b-scored.jsonl" download rel="external">{t('gemmaScored')}</a> · <a href="/benchmarks/typed-decisions-20260926/qwen3-06b-scored.jsonl" download rel="external">{t('qwenScored')}</a> · <a href="/benchmarks/typed-decisions-20260926/manifest.json" download rel="external">{t('manifest')}</a></p>{/if}
      <p class="section-note"><a href={`/docs/docs/${$locale}/MODEL_AUDIT_20260926.md`} rel="external">{t('modelAudit')} ↗</a></p>
      <a class="benchmark-link" href={`/docs/docs/${$locale}/BENCHMARK.md`} rel="external">{t('runBenchmark')} <span aria-hidden="true">↗</span></a>
      </div>
    </section>

    <section id="get-started" class="start-section">
      <div><p class="eyebrow">{t('startEyebrow')}</p><h2>{t('startTitle')}<br /><em>{t('startTitleEmphasis')}</em></h2><p>{t('startDescription')}</p><a class="text-link" href="/docs/warehouse.json" rel="external" download>{t('downloadExample')} <span aria-hidden="true">↓</span></a></div>
      <div class="command-panel"><div class="command-header"><span>{t('terminalQuickStart')}</span><button onclick={copyCommand}>{t(copied ? 'copied' : 'copyCommand')}</button></div><pre><code>{command}</code></pre>{#if copyError}<p role="status">{t('clipboardError')}</p>{/if}<div class="command-footer"><span>{t('localInference')}</span><span>{t('noDownloads')}</span></div></div>
      <div class="install-panel" aria-labelledby="install-heading">
        <div class="install-heading"><h3 id="install-heading">{t('installPackages')}</h3><a href="https://github.com/LuticaCANARD/L2S1/releases/tag/v0.1.1" rel="external">{t('releaseFiles')} ↗</a></div>
        <div class="install-grid">{#each installs as install (install.name)}<article><h4>{install.name}</h4><pre><code>{install.command}</code></pre><a href={`/docs/docs/${$locale}/${install.guide}`} rel="external">{t('packageGuide')} ↗</a></article>{/each}</div>
        <p class="install-note">{t('installRequirements')}</p>
      </div>
    </section>

    <section class="license-strip"><div class="license-symbol" aria-hidden="true">↗</div><div><h3>{t('licenseTitle')}</h3><p>{t('licenseDescription')}</p></div><a href={`/docs/docs/${$locale}/LICENSING.md`} rel="external">{t('licenseDetails')}</a></section>
  </main>
  <footer><a class="brand" href="#main"><span class="brand-symbol" aria-hidden="true">s1<span>↗</span></span><span>L2S1</span></a><p>{t('footerTagline')}</p><div><a href="/docs/docs/en/README.md" rel="external" lang="en">{t('englishDocs')}</a><a href="/docs/docs/ko/README.md" rel="external" lang="ko">{t('koreanDocs')}</a><a href="/docs/docs/ja/README.md" rel="external" lang="ja">{t('japaneseDocs')}</a><a href="/docs/LICENSE" rel="external">{t('mitLicense')}</a><a href="/docs/WEB_THIRD_PARTY_LICENSES.txt" rel="external">{t('thirdParty')}</a></div><span>{t('footerBuilt')}</span></footer>
</div>

<style>
  :global(*) {
    box-sizing: border-box;
  }
  :global(html) {
    scroll-behavior: smooth;
    scroll-padding-top: 32px;
  }
  :global(body) {
    margin: 0;
    background: var(--theme-page, #f5f4ed);
    color: var(--theme-text, #232b24);
    font-family: Arial, Helvetica, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  :global(a) {
    color: inherit;
    text-decoration: none;
  }
  :global(button),
  :global(input) {
    font: inherit;
  }
  :global(button),
  :global(a),
  :global(input) {
    -webkit-tap-highlight-color: transparent;
  }
  :global(button:focus-visible),
  :global(a:focus-visible),
  :global(input:focus-visible) {
    outline: 3px solid var(--site-focus, #638930);
    outline-offset: 5px;
  }
  :global(::selection) {
    background: var(--theme-chart, #d1f77b);
    color: var(--theme-terminal, #202820);
  }
  .page-shell {
    max-width: 1440px;
    margin: auto;
    padding: 0 70px;
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 11px;
    font-weight: 700;
    font-size: 21px;
    letter-spacing: -0.8px;
  }
  .brand-symbol {
    width: 36px;
    height: 36px;
    background: var(--theme-terminal, #263327);
    color: var(--theme-accent, #d1f77b);
    display: flex;
    align-items: center;
    justify-content: center;
    font: 700 16px monospace;
    position: relative;
    border-radius: 7px;
  }
  .brand-symbol span {
    font: 14px monospace;
    position: absolute;
    right: 2px;
    top: 0;
  }

  .text-link:hover,
  footer a:hover {
    text-decoration: underline;
    text-underline-offset: 5px;
  }

  .hero {
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 60px;
    align-items: center;
    padding: 91px 0 97px;
  }
  .eyebrow {
    display: flex;
    align-items: center;
    gap: 10px;
    font: 11px/1.5 monospace;
    letter-spacing: 1.1px;
    margin: 0 0 29px;
  }
  .status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--theme-chart, #648f43);
    display: inline-block;
    flex-shrink: 0;
  }
  .version {
    font-size: 10px;
    border: 1px solid var(--theme-border, #c9cec0);
    padding: 2px 5px;
    border-radius: 3px;
    letter-spacing: 0;
    margin-left: 6px;
  }
  h1 {
    font-size: clamp(48px, 5.4vw, 78px);
    letter-spacing: -4.1px;
    line-height: 1.04;
    font-weight: 500;
    margin: 0 0 28px;
  }
  h1 em,
  h2 em {
    font-family: Georgia, "Times New Roman", serif;
    font-weight: 400;
  }
  h1 em {
    color: var(--theme-muted, #506346);
  }
  .hero-description {
    font-size: 22px;
    line-height: 1.45;
    letter-spacing: -0.5px;
    margin: 0 0 18px;
  }
  .hero-detail {
    font-size: 15px;
    line-height: 1.75;
    color: var(--theme-muted, #687063);
    max-width: 390px;
    margin: 0;
  }
  .hero-actions {
    display: flex;
    align-items: center;
    gap: 24px;
    margin-top: 31px;
  }
  .button {
    display: inline-flex;
    align-items: center;
    gap: 29px;
    padding: 16px 21px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 600;
  }
  .primary {
    background: var(--theme-terminal, #283a2b);
    color: var(--theme-on-primary, #fff);
  }
  .primary:hover {
    background: var(--theme-terminal, #3f553d);
  }
  .text-link {
    font-size: 13px;
    font-weight: 500;
  }
  .hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    margin-top: 27px;
    font: 10px monospace;
    color: var(--theme-muted, #65705e);
  }
  .hero-tags span + span::before {
    content: "·";
    margin-right: 15px;
  }
  .decision-window {
    --theme-terminal: #071522;
    --theme-muted: #c1d6e7;
    --theme-accent: #a4d9ff;
    --theme-chart: #91cdef;
    --theme-border: #65849c;
    background: var(--theme-terminal);
    color: var(--theme-on-primary, #e7ebe0);
    border: 1px solid var(--theme-border);
    border-radius: 9px;
    position: relative;
    box-shadow: 0 16px 40px -25px var(--theme-shadow, #202e2350);
    transform: rotate(1deg);
  }
  .window-bar {
    height: 45px;
    background: #030c14;
    border-bottom: 1px solid #415e76;
    border-radius: 8px 8px 0 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 22px;
    font: 10px monospace;
    color: var(--theme-muted, #a3ad9e);
  }
  .window-bar > span:first-child {
    display: flex;
    gap: 5px;
  }
  .window-bar i {
    display: block;
    width: 6px;
    height: 6px;
    background: var(--theme-chart, #65705e);
    border-radius: 50%;
  }
  .window-bar i:first-child {
    background: var(--theme-chart, #bad98a);
  }
  .window-content {
    padding: 25px 28px 22px;
  }
  .terminal-label {
    display: flex;
    justify-content: space-between;
    font: 9px monospace;
    letter-spacing: 1.4px;
    color: var(--theme-muted, #a5b299);
  }
  .input-preview {
    font: 12px/1.9 monospace;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin: 14px 0;
  }
  .code-muted {
    color: var(--theme-muted, #87917f);
  }
  .code-key {
    color: #f0f7ff;
    font-weight: 600;
  }
  .code-string,
  .code-number {
    color: var(--theme-accent, #d1f77b);
  }
  .flow-line {
    display: flex;
    align-items: center;
    gap: 10px;
    font: 8px monospace;
    letter-spacing: 1.2px;
    color: var(--theme-muted, #97a790);
    margin: 23px 0 16px;
  }
  .flow-line > span:first-child {
    width: 19px;
    height: 1px;
    background: var(--theme-chart, #61755c);
  }
  .flow-line b {
    font: 20px monospace;
    color: var(--theme-muted, #abc295);
    margin-left: -4px;
  }
  .option-stack {
    display: flex;
    flex-direction: column;
    gap: 7px;
  }
  .option-stack > div {
    display: flex;
    align-items: center;
    gap: 12px;
    border: 1px solid var(--theme-border, #485440);
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 12px;
    color: var(--theme-muted, #b6c1b0);
  }
  .letter {
    display: flex;
    align-items: center;
    justify-content: center;
    font: 10px monospace;
    border: 1px solid var(--theme-border, #596652);
    border-radius: 3px;
    width: 22px;
    height: 22px;
  }
  .option-mark {
    margin-left: auto;
    font: 16px monospace;
  }
  .option-stack .selected {
    background: var(--theme-chart, #d1f77b);
    color: var(--theme-terminal, #24351e);
    border-color: var(--theme-border, #d1f77b);
    font-weight: 600;
  }
  .selected .letter {
    border-color: #315c77;
  }
  .flow-line.small {
    margin: 17px 0 12px;
  }
  .result-preview {
    padding: 14px 12px;
    background: #030c14;
    border: 1px solid var(--theme-border, #364731);
    border-radius: 4px;
    font: 12px/1.6 monospace;
  }
  .result-preview strong {
    color: var(--theme-accent, #d1f77b);
    font-weight: 400;
  }
  .illustration-note {
    font: 9px/1.6 monospace;
    color: var(--theme-muted, #a9b59f);
    margin: 13px 0 0;
  }
  .window-offset {
    position: absolute;
    bottom: -27px;
    right: 1px;
    color: var(--theme-muted, #7b8573);
    font: 8px monospace;
    letter-spacing: 1.5px;
  }
  .principles {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 22px 0;
    border-top: 1px solid var(--theme-border, #d6d9ce);
    border-bottom: 1px solid var(--theme-border, #d6d9ce);
    font-size: 12px;
    color: var(--theme-muted, #626d5c);
  }
  .principles b {
    font: 10px monospace;
    color: var(--theme-muted, #9aa18f);
    margin-right: 11px;
  }
  .section {
    padding: 92px 0;
  }
  .section-heading {
    max-width: 660px;
    margin-bottom: 38px;
  }
  .section-heading .eyebrow {
    margin-bottom: 23px;
  }
  h2 {
    font-size: 44px;
    font-weight: 500;
    line-height: 1.15;
    letter-spacing: -1.8px;
    margin: 0 0 22px;
  }
  h2 em {
    color: var(--theme-muted, #65745c);
  }
  .section-heading > p:last-child,
  .section-heading.horizontal > p {
    font-size: 15px;
    line-height: 1.8;
    color: var(--theme-muted, #65705f);
    max-width: 475px;
    margin-bottom: 0;
  }
  .contract-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }
  .contract-grid article {
    border: 1px solid var(--theme-border, #d2d7c8);
    border-radius: 5px;
    padding: 27px 24px 20px;
    display: flex;
    flex-direction: column;
    background: var(--theme-panel, #ffffff);
    box-shadow: var(--theme-panel-shadow);
  }
  .card-index {
    font: 14px monospace;
    color: var(--theme-muted, #697e57);
    height: 42px;
  }
  h3 {
    font-size: 19px;
    letter-spacing: -0.5px;
    font-weight: 500;
    margin: 15px 0 10px;
  }
  .contract-grid p {
    font-size: 13px;
    line-height: 1.8;
    color: var(--theme-muted, #677060);
    min-height: 70px;
    margin: 0 0 21px;
  }
  .contract-grid code {
    font-size: 10px;
    line-height: 1.6;
    background: var(--theme-tint, #eaf0e0);
    padding: 11px 9px;
    display: block;
    border-radius: 3px;
    overflow-wrap: anywhere;
    white-space: normal;
    margin-top: auto;
  }
  .card-type {
    display: flex;
    justify-content: space-between;
    font: 9px monospace;
    letter-spacing: 1.2px;
    margin-top: 26px;
    color: var(--theme-muted, #68785c);
  }
  .playground-section {
    background: var(--theme-tint, #e9eddf);
    margin: 0 -70px;
    padding: 73px 70px;
    display: grid;
    grid-template-columns: 0.9fr 1.1fr;
    gap: 95px;
    align-items: center;
    border-top: 1px solid var(--theme-border, #d9decd);
    border-bottom: 1px solid var(--theme-border, #d9decd);
  }
  .playground-copy > p:not(.eyebrow) {
    font-size: 15px;
    line-height: 1.9;
    color: var(--theme-muted, #5e6b54);
  }
  .demo-disclosure {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    margin-top: 30px;
    border-top: 1px solid var(--theme-border, #cbd2be);
    padding-top: 20px;
    color: var(--theme-muted, #5e6a53);
  }
  .demo-disclosure > span {
    font-size: 17px;
  }
  .demo-disclosure p {
    font-size: 12px;
    line-height: 1.8;
    margin: 0;
  }
  .demo-disclosure strong {
    font-weight: 600;
  }
  .playground-panel {
    background: var(--theme-panel, #fcfcf7);
    border: 1px solid var(--theme-border, #cdd6be);
    border-radius: 7px;
    padding: 24px;
    box-shadow: var(--theme-panel-shadow);
  }
  .segmented-control {
    display: flex;
    padding: 4px;
    border-radius: 4px;
    background: var(--theme-tint, #eaf0e1);
    gap: 3px;
  }
  .segmented-control button {
    flex: 1;
    border: 0;
    background: transparent;
    padding: 9px;
    cursor: pointer;
    font: 11px monospace;
    text-transform: uppercase;
    color: var(--theme-muted, #566748);
    border-radius: 3px;
  }
  .segmented-control button.active {
    background: var(--theme-panel, #fcfcf7);
    box-shadow: 0 1px 4px var(--theme-shadow, #334f1f1a);
    color: var(--theme-accent, #233718);
  }
  .question-line {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
    margin: 22px 0;
  }
  .question-line h3 {
    font-size: 15px;
    margin: 0;
    line-height: 1.4;
  }
  .decision-status {
    padding: 5px 8px;
    background: var(--theme-tint, #e3efcc);
    color: var(--theme-accent, #38531e);
    font: 10px monospace;
    border-radius: 3px;
  }
  .probability-list {
    display: flex;
    flex-direction: column;
    gap: 15px;
    margin: 25px 0 27px;
  }
  .probability-row {
    display: grid;
    grid-template-columns: 18px 64px 1fr 35px;
    gap: 10px;
    align-items: center;
    font-size: 11px;
  }
  .probability-row > span:first-child {
    font-family: monospace;
    color: var(--theme-muted, #667e52);
  }
  .probability-row strong {
    font: 10px monospace;
    text-align: right;
  }
  .bar-track {
    height: 5px;
    border-radius: 3px;
    background: var(--theme-tint, #e6eadf);
    overflow: hidden;
  }
  .bar-track > div {
    background: var(--theme-chart, #83a458);
    height: 100%;
    border-radius: 3px;
    transition: width 0.2s;
  }
  .demo-result {
    margin-top: 23px;
    background: var(--theme-terminal, #253125);
    color: var(--theme-on-primary, #e6ebdc);
    border-radius: 4px;
    padding: 15px;
  }
  .demo-result .terminal-label {
    font-size: 8px;
  }
  .demo-result pre {
    font: 10px/1.7 monospace;
    margin: 13px 0 0;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    min-height: 122px;
    color: var(--theme-accent, #d4e7bb);
  }
  .horizontal {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 55px;
    max-width: none;
  }
  .horizontal > p {
    max-width: 385px !important;
    margin: 0 0 23px !important;
  }
  .compatibility-highlight, .capacity-highlight {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 18px;
    border: 1px solid var(--theme-border);
    background: var(--theme-panel);
    padding: 24px;
    margin-bottom: 24px;
  }
  .compatibility-highlight strong { font-size: clamp(22px, 3vw, 32px); }
  .compatibility-highlight p, .capacity-highlight p { color: var(--theme-muted); font-size: 12px; line-height: 1.8; margin: 8px 0 0; }
  .compatibility-highlight a, .capacity-highlight a { color: var(--theme-text); font-size: 11px; line-height: 1.8; }
  .capacity-highlight { margin: 24px 0; background: var(--theme-terminal); color: #f0f7ff; }
  .capacity-highlight .metric-label, .capacity-highlight p { color: #c1d6e7; }
  .capacity-highlight a { color: #a4d9ff; }
  .capacity-highlight strong { display: block; font-size: clamp(32px, 5vw, 48px); font-weight: 500; }
  .capacity-highlight small { font-size: 18px; }
  .model-size { font-size: 14px; font-weight: 600; color: var(--theme-text); }
  .model-list {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    border: 1px solid var(--theme-border, #d2d7c8);
    border-radius: 5px;
    overflow: hidden;
  }
  .model-list > a {
    padding: 25px 19px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    background: var(--theme-tint, #f9f9f3);
    position: relative;
    min-width: 0;
  }
  .model-list > a + a {
    border-left: 1px solid var(--theme-border, #d2d7c8);
  }
  .model-list > a:hover {
    background: var(--theme-tint, #eaf0df);
  }
  .model-dot {
    width: 9px;
    height: 9px;
    background: var(--theme-chart, #82976c);
    border-radius: 2px;
    margin-bottom: 21px;
    transform: rotate(45deg);
  }
  .model-list strong {
    font-size: 15px;
    font-weight: 500;
  }
  .model-list > a > span:not(.model-dot):not(.model-arrow) {
    font: 9px/1.6 monospace;
    color: var(--theme-muted, #718063);
  }
  .model-arrow {
    position: absolute;
    right: 16px;
    top: 23px;
    font-size: 16px;
    color: var(--theme-muted, #78915f);
  }
  .section-note {
    font-size: 11px;
    line-height: 1.9;
    color: var(--theme-muted, #6b7762);
    max-width: 850px;
    margin-top: 20px;
  }
  .section-note a {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .benchmark-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11px;
    color: var(--theme-muted, #748167);
    margin-bottom: 19px;
    flex-wrap: wrap;
  }
  .benchmark-meta strong {
    color: var(--theme-muted, #536448);
    font-weight: 500;
  }
  .benchmark-meta > span:last-child {
    margin-left: auto;
  }
  .table-scroll {
    overflow-x: auto;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    font-size: 12px;
    text-align: left;
  }
  th {
    font: 10px monospace;
    color: var(--theme-muted, #66795a);
    padding: 15px 12px;
    border-bottom: 1px solid var(--theme-border, #c8d1bb);
    white-space: nowrap;
  }
  td {
    padding: 19px 12px;
    border-bottom: 1px solid var(--theme-border, #dde2d3);
    font-family: monospace;
    white-space: nowrap;
  }
  td:first-child {
    font:
      13px Arial,
      sans-serif;
  }
  td span {
    color: var(--theme-muted, #8a9580);
    font-size: 10px;
  }
  th:not(:first-child),
  td:not(:first-child) {
    text-align: right;
  }
  .benchmark-link {
    display: flex;
    justify-content: space-between;
    padding: 17px 0;
    border-bottom: 1px solid var(--theme-border, #d2d9c8);
    font-size: 12px;
    margin-top: 30px;
  }
  .benchmark-link:hover {
    color: var(--theme-accent, #527032);
  }
  .start-section {
    padding: 61px 0 80px;
    border-top: 1px solid var(--theme-border, #d6d9ce);
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 74px;
    align-items: center;
  }
  .start-section > div > p:not(.eyebrow) {
    color: var(--theme-muted, #67765a);
    font-size: 14px;
    line-height: 1.8;
    max-width: 350px;
  }
  .start-section .text-link {
    display: inline-block;
    margin-top: 15px;
  }
  .install-panel { grid-column: 1 / -1; min-width: 0; }
  .install-heading { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 18px; }
  .install-heading h3 { margin: 0; font-size: 20px; }
  .install-heading a, .install-grid a { font-size: 12px; text-decoration: underline; text-underline-offset: 3px; }
  .install-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 280px), 1fr)); gap: 16px; }
  .install-grid article { min-width: 0; padding: 22px; border: 1px solid var(--theme-border, #d6d9ce); border-radius: 5px; }
  .install-grid h4 { margin: 0 0 15px; font-size: 14px; }
  .install-grid pre { margin: 0 0 18px; font: 12px/1.8 monospace; white-space: pre-wrap; overflow-wrap: anywhere; }
  .start-section > .install-panel > .install-note { max-width: none; margin: 18px 0 0; }
  .command-panel {
    background: var(--theme-terminal, #222e24);
    border-radius: 5px;
    color: var(--theme-on-primary, #e0e9d5);
    overflow: hidden;
  }
  .command-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 17px 22px;
    border-bottom: 1px solid var(--theme-border, #46553b);
    font: 9px monospace;
    letter-spacing: 0.7px;
  }
  .command-header button {
    color: var(--theme-on-primary, #e4eed9);
    border: 1px solid var(--theme-border, #6f7d61);
    background: transparent;
    padding: 7px 9px;
    font: 9px monospace;
    cursor: pointer;
    border-radius: 3px;
    white-space: nowrap;
  }
  .command-panel pre {
    font: 10px/1.9 monospace;
    overflow: auto;
    margin: 0;
    padding: 25px 22px;
  }
  .command-panel > p {
    font-size: 11px;
    line-height: 1.6;
    margin: 0 22px 15px;
  }
  .command-footer {
    display: flex;
    justify-content: space-between;
    font: 9px monospace;
    color: var(--theme-muted, #9cb08b);
    padding: 15px 22px;
    background: var(--theme-terminal, #1c261f);
    border-top: 1px solid var(--theme-border, #36462d);
  }
  .license-strip {
    border: 1px solid var(--theme-border, #d3d9c7);
    border-radius: 5px;
    display: flex;
    align-items: center;
    gap: 25px;
    padding: 26px;
    margin-bottom: 70px;
    background: var(--theme-tint, #eff1e6);
  }
  .license-symbol {
    font:
      28px Georgia,
      serif;
    background: var(--theme-tint, #e1e9d1);
    width: 50px;
    height: 50px;
    display: grid;
    place-items: center;
    flex-shrink: 0;
    border-radius: 5px;
  }
  .license-strip h3 {
    font-size: 15px;
    margin: 0 0 8px;
  }
  .license-strip p {
    font-size: 11px;
    line-height: 1.8;
    margin: 0;
    color: var(--theme-muted, #647356);
    max-width: 640px;
  }
  .license-strip > a {
    font-size: 11px;
    white-space: nowrap;
    margin-left: auto;
  }
  footer {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 20px;
    border-top: 1px solid var(--theme-border, #d5dbca);
    padding: 34px 0 25px;
  }
  footer .brand {
    font-size: 17px;
  }
  footer .brand-symbol {
    width: 29px;
    height: 29px;
    font-size: 13px;
  }
  footer > p {
    font-size: 11px;
    color: var(--theme-muted, #748166);
  }
  footer > div {
    display: flex;
    gap: 21px;
    margin-left: auto;
    font-size: 10px;
  }
  footer > span {
    flex-basis: 100%;
    font: 9px monospace;
    color: var(--theme-muted, #7e8a71);
    margin-top: 13px;
  }
  .skip-link {
    position: absolute;
    left: 20px;
    top: -60px;
    background: var(--theme-chart, #d1f77b);
    color: var(--theme-terminal, #232b24);
    padding: 12px;
    z-index: 20;
  }
  .skip-link:focus {
    top: 16px;
  }
  .sr-only {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
  @media (min-width: 1400px) {
    .hero {
      gap: 95px;
    }
    .hero h1 {
      font-size: 78px;
    }
  }
  @media (max-width: 1050px) {
    .page-shell {
      padding: 0 35px;
    }
    .hero {
      gap: 32px;
      padding: 65px 0 75px;
    }
    h1 {
      font-size: 54px;
      letter-spacing: -2.8px;
    }
    .hero-description {
      font-size: 19px;
    }
    .hero-detail {
      font-size: 13px;
    }
    .hero-actions {
      gap: 16px;
    }
    .hero-tags {
      gap: 9px;
      font-size: 9px;
    }
    .hero-tags span + span::before {
      margin-right: 9px;
    }
    .window-content {
      padding: 22px;
    }
    .input-preview {
      font-size: 10px;
    }
    .hero .text-link {
      font-size: 11px;
    }
    .playground-section {
      margin: 0 -35px;
      padding: 60px 35px;
      gap: 40px;
    }
    .contract-grid {
      gap: 12px;
    }
    .contract-grid article {
      padding: 24px 17px;
    }
    .contract-grid p {
      min-height: 94px;
    }
    .contract-grid code {
      font-size: 9px;
    }
    .model-list > a {
      padding: 21px 12px;
    }
    .model-list strong {
      font-size: 13px;
    }
    .model-list > a > span:not(.model-dot):not(.model-arrow) {
      font-size: 8px;
    }
    .horizontal {
      gap: 30px;
    }
    .horizontal > p {
      max-width: 320px !important;
    }
    .start-section {
      gap: 30px;
    }
    h2 {
      font-size: 39px;
    }
    .license-strip {
      gap: 17px;
    }
  }
  @media (max-width: 760px) {
    .page-shell {
      padding: 0 24px;
    }


    .brand {
      font-size: 19px;
    }

    .hero {
      grid-template-columns: 1fr;
      padding: 52px 0 70px;
      gap: 45px;
    }
    .hero-copy {
      max-width: 560px;
    }
    .eyebrow {
      font-size: 10px;
      margin-bottom: 24px;
    }
    h1 {
      font-size: 61px;
      letter-spacing: -3px;
    }
    .hero-description {
      font-size: 21px;
    }
    .hero-detail {
      font-size: 14px;
      max-width: 430px;
    }
    .hero-actions {
      gap: 26px;
    }
    .hero .text-link {
      font-size: 12px;
    }
    .hero-tags {
      font-size: 10px;
      gap: 13px;
    }
    .decision-window {
      width: 100%;
      max-width: 510px;
      justify-self: center;
      transform: none;
    }
    .window-content {
      padding: 23px 26px;
    }
    .input-preview {
      font-size: 12px;
    }
    .principles {
      flex-direction: column;
      gap: 16px;
      padding: 22px 0;
      font-size: 12px;
    }
    .section {
      padding: 61px 0;
    }
    h2 {
      font-size: 38px;
    }
    .section-heading > p:last-child {
      font-size: 14px;
    }
    .contract-grid {
      grid-template-columns: 1fr;
      gap: 12px;
    }
    .contract-grid article {
      padding: 25px;
    }
    .card-index {
      height: 28px;
    }
    .contract-grid p {
      min-height: 0;
      max-width: 450px;
    }
    .contract-grid code {
      font-size: 12px;
    }
    .card-type {
      margin-top: 20px;
    }
    .playground-section {
      grid-template-columns: 1fr;
      margin: 0 -24px;
      padding: 53px 24px;
      gap: 26px;
    }
    .playground-copy > p:not(.eyebrow) {
      font-size: 14px;
      max-width: 440px;
    }
    .demo-disclosure {
      margin-top: 19px;
    }
    .playground-panel {
      max-width: 510px;
      width: 100%;
      justify-self: center;
    }
    .horizontal {
      display: block;
    }
    .horizontal > p {
      max-width: 480px !important;
      margin: 0 !important;
    }
    .model-list {
      grid-template-columns: 1fr;
    }
    .model-list > a {
      display: grid;
      grid-template-columns: 9px 95px 1fr 14px;
      gap: 15px;
      align-items: center;
      padding: 22px 18px;
    }
    .model-list > a + a {
      border-left: 0;
      border-top: 1px solid var(--theme-border, #d2d7c8);
    }
    .model-dot {
      margin: 0;
      width: 7px;
      height: 7px;
    }
    .model-list strong {
      font-size: 14px;
    }
    .model-list > a > span:not(.model-dot):not(.model-arrow) {
      font-size: 9px;
    }
    .model-arrow {
      position: static;
      font-size: 16px;
    }
    .benchmark-meta > span:last-child {
      margin-left: 0;
      line-height: 1.7;
      flex-basis: 100%;
    }
    th {
      padding: 13px 9px;
      font-size: 9px;
    }
    td {
      padding: 18px 9px;
      font-size: 11px;
    }
    td:first-child {
      font-size: 12px;
    }
    .start-section {
      grid-template-columns: 1fr;
      padding: 51px 0;
      gap: 28px;
    }
    .command-panel pre {
      font-size: 10px;
      padding: 22px 17px;
    }
    .command-header,
    .command-footer {
      padding-left: 17px;
      padding-right: 17px;
    }
    .license-strip {
      flex-wrap: wrap;
      padding: 22px;
      gap: 16px;
      margin-bottom: 42px;
    }
    .license-strip > div:nth-child(2) {
      flex: 1;
      min-width: 180px;
    }
    .license-strip > a {
      margin-left: 66px;
    }
    .license-strip p {
      font-size: 11px;
    }
    footer {
      gap: 17px;
      padding-top: 27px;
    }
    footer > p {
      margin-left: auto;
      font-size: 10px;
    }
    footer > div {
      margin-left: 0;
      flex-basis: 100%;
      flex-wrap: wrap;
    }
    footer > span {
      line-height: 1.8;
      margin-top: 0;
    }
  }
  @media (max-width: 390px) {
    .page-shell {
      padding: 0 18px;
    }
    h1 {
      font-size: 53px;
      letter-spacing: -2.6px;
    }
    .hero-tags {
      font-size: 9px;
      gap: 9px;
    }
    .hero-tags span + span::before {
      margin-right: 9px;
    }
    .playground-section {
      margin: 0 -18px;
      padding-left: 18px;
      padding-right: 18px;
    }
    .playground-panel {
      padding: 18px;
    }
    .hero-actions {
      gap: 20px;
    }
    .input-preview {
      font-size: 10px;
    }
    .model-list > a {
      grid-template-columns: 7px 85px 1fr 12px;
      gap: 10px;
    }
    .window-content {
      padding: 22px;
    }
    .brand {
      font-size: 17px;
    }

  }
  @media (prefers-reduced-motion: reduce) {
    :global(html) {
      scroll-behavior: auto;
    }
    .bar-track > div {
      transition: none;
    }
  }

  .performance-section {
    --performance-inset: 70px;
    --theme-text: #edf7ff;
    --theme-muted: #abc8db;
    --theme-accent: #a5ddff;
    --theme-border: #355d79;
    --theme-chart: #79bde5;
    --site-focus: #b8e5ff;
    color: var(--theme-text);
    margin-inline: calc(-1 * var(--performance-inset));
    padding: 80px var(--performance-inset) 68px;
    border-top: 0;
    background: radial-gradient(ellipse at 85% 0%, #195d824d, transparent 52%), #003153;
    position: relative;
    isolation: isolate;
  }
  :global(html[data-theme='dark']) .performance-section {
    background: radial-gradient(ellipse at 85% 0%, #195d8238, transparent 52%), #07151f;
  }
  .performance-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 64px;
    align-items: center;
    margin-bottom: 54px;
  }
  .performance-heading .eyebrow { color: #9ccfea; margin: 0 0 28px; }
  .performance-heading h2 { font-size: clamp(48px, 5.7vw, 80px); line-height: 1.02; letter-spacing: -3px; margin-bottom: 25px; }
  .performance-heading h2 em { color: #a5ddff; }
  .performance-heading > p:not(.eyebrow) { font-size: 14px; line-height: 1.8; color: var(--theme-muted); max-width: 380px; }
  .performance-hardware { display: inline-flex; align-items: center; gap: 13px; font: 10px monospace; letter-spacing: 1.4px; margin-top: 18px; color: #c6e5f7; }
  .performance-hardware > span { color: #79bde5; font-size: 18px; }
  .performance-spotlight { border-left: 1px solid var(--theme-border); padding-left: 48px; min-width: 0; }
  .metric-label { font: 10px/1.7 monospace; letter-spacing: 1px; color: #abc8db; margin: 0; }
  .performance-value { display: flex; align-items: baseline; gap: 15px; margin: 15px 0 25px; white-space: nowrap; }
  .performance-value strong { font-size: clamp(86px, 10vw, 144px); font-weight: 500; letter-spacing: -8px; line-height: 1; font-variant-numeric: tabular-nums; }
  .performance-value > span { font-size: 32px; color: #a5ddff; letter-spacing: -1px; }
  .metric-rule { height: 3px; background: linear-gradient(90deg, #a5ddff, #4b8aac55); position: relative; margin-bottom: 24px; }
  .metric-rule::after { content: ''; position: absolute; width: 7px; height: 7px; border: 1px solid #a5ddff; background: #003153; right: 0; top: -3px; border-radius: 50%; }
  .metric-model { font: 12px/1.6 monospace; margin: 0 0 6px; }
  .metric-scope { color: var(--theme-muted); font-size: 11px; line-height: 1.8; margin: 0; }
  .performance-stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); border-top: 1px solid var(--theme-border); border-bottom: 1px solid var(--theme-border); margin-bottom: 43px; }
  .performance-stat { padding: 27px 24px 27px 0; min-width: 0; }
  .performance-stat + .performance-stat { border-left: 1px solid var(--theme-border); padding-left: 27px; }
  .stat-value { font-size: clamp(30px, 3.4vw, 48px); letter-spacing: -2px; line-height: 1.2; margin: 15px 0 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .stat-value > span { font-size: 14px; letter-spacing: 0; margin-left: 8px; color: #a5ddff; }
  .performance-stat p { font: 10px/1.8 monospace; color: var(--theme-muted); margin: 0; overflow-wrap: anywhere; }
  .metric-evidence { display: inline-block; margin-top: 8px; font: 10px monospace; color: #a5ddff; text-decoration: underline; text-underline-offset: 3px; }
  .quantization-highlight { margin: -15px 0 32px; display: flex; align-items: center; gap: 18px; font: 11px/1.8 monospace; color: #c6e5f7; }
  .quantization-highlight strong { font-size: 24px; color: #a5ddff; white-space: nowrap; }
  .benchmark-tabs { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 26px; }
  .benchmark-tabs button { cursor: pointer; border: 1px solid #477593; border-radius: 6px; color: #c6e5f7; background: #001e334d; padding: 12px 15px; font: 11px/1.6 monospace; text-align: left; }
  .benchmark-tabs button.active { color: #003153; border-color: #a5ddff; background: #a5ddff; }
  .performance-details { padding: 26px; border: 1px solid #47759366; border-radius: 8px; background: #001e333b; min-width: 0; }
  .performance-details-heading { display: flex; justify-content: space-between; align-items: center; gap: 18px; padding-bottom: 24px; }
  .performance-details-heading h3 { font-size: 19px; letter-spacing: -.4px; font-weight: 500; margin: 0; }
  .performance-details-heading > span { font: 9px/1.5 monospace; letter-spacing: 1px; color: #abc8db; text-align: right; }
  .performance-details .benchmark-meta { font-size: 10px; }
  .performance-details th { font-size: 9px; padding-inline: 9px; }
  .performance-details td { font-size: 11px; padding-inline: 9px; font-variant-numeric: tabular-nums; }
  .performance-details td:first-child { font-size: 12px; }
  .performance-details tbody tr { transition: background .15s; }
  .performance-details tbody tr:hover { background: #89c8ec0c; }
  .performance-details td:first-child { border-left: 2px solid transparent; }
  .performance-details tbody tr:hover td:first-child { border-left-color: #a5ddff; }
  .performance-details .section-note { font-size: 10px; margin-top: 16px; max-width: none; }
  .performance-details .benchmark-link { margin-top: 20px; padding-bottom: 3px; border-bottom: 0; }
  @media (max-width: 1050px) {
    .performance-section { --performance-inset: 35px; }
    .performance-hero { gap: 32px; }
    .performance-spotlight { padding-left: 30px; }
    .performance-value strong { font-size: clamp(82px, 11vw, 122px); letter-spacing: -6px; }
    .performance-stat { padding-right: 15px; }
    .performance-stat + .performance-stat { padding-left: 18px; }
  }
  @media (max-width: 760px) {
    .performance-section { --performance-inset: 24px; padding-block: 55px 42px; }
    .performance-hero { grid-template-columns: 1fr; gap: 36px; margin-bottom: 32px; }
    .performance-heading h2 { font-size: 56px; letter-spacing: -2px; }
    .performance-heading > p:not(.eyebrow) { font-size: 13px; }
    .performance-spotlight { border-left: 0; border-top: 1px solid var(--theme-border); padding: 26px 0 0; }
    .performance-value strong { font-size: 112px; }
    .performance-value > span { font-size: 27px; }
    .performance-stats { grid-template-columns: 1fr; margin-bottom: 30px; }
    .performance-stat { padding: 24px 0; display: grid; grid-template-columns: minmax(0, 1fr) auto; column-gap: 15px; align-items: center; }
    .performance-stat + .performance-stat { border-left: 0; border-top: 1px solid var(--theme-border); padding: 24px 0; }
    .performance-stat .metric-label { grid-column: 1; }
    .stat-value { grid-column: 2; grid-row: 1 / 3; font-size: 34px; margin: 0; letter-spacing: -1px; }
    .stat-value > span { font-size: 10px; margin-left: 4px; }
    .performance-stat p { grid-column: 1; font-size: 9px; }
    .performance-details { padding: 20px 14px; }
    .performance-details-heading { gap: 12px; align-items: flex-start; }
    .performance-details-heading h3 { font-size: 16px; }
    .performance-details-heading > span { font-size: 8px; }
  }
  @media (max-width: 390px) {
    .performance-section { --performance-inset: 18px; }
    .performance-heading h2 { font-size: 51px; }
    .stat-value { font-size: 29px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .performance-details tbody tr { transition: none; }
  }

  .performance-chart { margin: 0 0 40px; }
  .performance-chart figcaption { display: flex; align-items: center; justify-content: space-between; gap: 20px; margin-bottom: 30px; }
  .performance-chart h3 { font-size: 20px; font-weight: 500; margin: 0 0 9px; letter-spacing: -.4px; }
  .performance-chart figcaption p { font: 10px/1.8 monospace; color: var(--theme-muted); margin: 0; }
  .chart-legend { display: flex; gap: 20px; font: 10px monospace; color: #c6e5f7; }
  .chart-legend > span { display: inline-flex; align-items: center; gap: 7px; }
  .chart-legend i { height: 6px; width: 15px; border-radius: 3px; }
  .cpu-key, .cpu-bar { background: #8faabd; }
  .cuda-key, .cuda-bar { background: #7ad5ff; }
  .latency-model { display: grid; grid-template-columns: 190px minmax(0, 1fr); gap: 25px; align-items: center; margin: 0 0 25px; }
  .latency-model > p { font: 11px/1.7 monospace; margin: 0; }
  .latency-lines { display: grid; gap: 15px; }
  .latency-row { display: grid; grid-template-columns: 40px minmax(0, 1fr) 104px; align-items: center; gap: 14px; font: 10px monospace; }
  .latency-row > span { color: #abc8db; }
  .latency-track { height: 7px; border-radius: 3px; background: #47759324; }
  .latency-track > div { height: 100%; border-radius: 3px; }
  .latency-row strong { font-weight: 400; text-align: right; font-variant-numeric: tabular-nums; }
  .latency-row strong > span { color: #abc8db; font-size: 9px; }
  .latency-axis { margin: 10px 118px 0 269px; padding-top: 9px; border-top: 1px solid #47759366; display: flex; justify-content: space-between; font: 9px monospace; color: #abc8db; }
  @media (max-width: 760px) {
    .performance-chart figcaption { align-items: flex-start; flex-direction: column; gap: 14px; margin-bottom: 25px; }
    .performance-chart h3 { font-size: 18px; }
    .performance-chart figcaption p { font-size: 9px; }
    .latency-model { grid-template-columns: 1fr; gap: 14px; margin-bottom: 26px; }
    .latency-model > p { font-size: 10px; }
    .latency-row { grid-template-columns: 32px minmax(0, 1fr) 81px; gap: 9px; font-size: 9px; }
    .latency-axis { margin-left: 41px; margin-right: 90px; font-size: 8px; }
  }
</style>
