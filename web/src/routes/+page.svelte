<script lang="ts">
  import benchmarks from '$lib/benchmarks.json';

  type Kind = 'choice' | 'binary' | 'ordinal';
  type BenchmarkRow = { model: string; cpuP50: number; cudaP50: number; cudaDecisionsPerSecond: number; cudaAbstentionRate: number; nativeCpu: boolean; nativeCuda: boolean };
  const measurements = benchmarks.rows as BenchmarkRow[];
  let kind = $state<Kind>('choice');
  let threshold = $state(80);
  let copied = $state(false);
  let copyError = $state(false);
  const choices = {
    choice: [{ code: 'A', label: 'Ambient', probability: 0.02 }, { code: 'B', label: 'Chilled', probability: 0.96 }, { code: 'C', label: 'Frozen', probability: 0.02 }],
    binary: [{ code: 'A', label: 'No', probability: 0.08 }, { code: 'B', label: 'Yes', probability: 0.92 }],
    ordinal: [{ code: 'A', label: 'Low', probability: 0.03 }, { code: 'B', label: 'Medium', probability: 0.09 }, { code: 'C', label: 'High', probability: 0.88 }]
  };
  const questions = { choice: 'Which storage zone?', binary: 'Is a cold chain required?', ordinal: 'What is the dispatch priority?' };
  let options = $derived(choices[kind]);
  let top = $derived(Math.max(...options.map((option) => option.probability)));
  let abstained = $derived(top < threshold / 100);
  let result = $derived(kind === 'choice'
    ? { type: 'choice', selected: abstained ? null : 'chilled' }
    : kind === 'binary'
      ? { type: 'binary', p_true: 0.92, value: abstained ? null : true }
      : { type: 'ordinal', expected_value: 1.85, selected: abstained ? null : 'high' });
  let resultJson = $derived(JSON.stringify({ value: result, abstention_reasons: abstained ? ['low_top_probability'] : [] }, null, 2));
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
    { name: 'Gemma 4', detail: 'E2B Instruct · Q8_0', url: 'https://huggingface.co/ggml-org/gemma-4-E2B-it-GGUF' },
    { name: 'Gemma 3', detail: '1B IT · Q8_0', url: 'https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF' },
    { name: 'Qwen3', detail: '0.6B · Q8_0', url: 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF' },
    { name: 'SmolLM2', detail: '135M Instruct · Q8_0', url: 'https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF' },
    { name: 'TinyLlama', detail: '1.1B Chat · Q4_K_M', url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF' }
  ];
</script>

<svelte:head>
  <title>L2S1 — Local models. Typed decisions.</title>
  <meta name="description" content="L2S1 (LLM to System 1): a Rust library for typed decisions from local GGUF models. Get choices, booleans, and ordinal scores with an explicit abstention policy." />
  <meta property="og:title" content="L2S1 — Local models. Typed decisions." />
  <meta property="og:description" content="L2S1 — LLM to System 1. Give your model a question. Give your code a typed answer. Local inference with Rust and llama.cpp." />
  <meta property="og:type" content="website" />
</svelte:head>

<a class="skip-link" href="#main">Skip to content</a>
<div class="page-shell">
  <header class="site-header">
    <a class="brand" href="#main" aria-label="L2S1 home"><span class="brand-symbol" aria-hidden="true">s1<span>↗</span></span><span>L2S1</span></a>
    <nav aria-label="Main navigation">
      <a href="#how-it-works">How it works</a><a href="#models">Models</a><a href="#performance">Performance</a>
    </nav>
    <a class="header-cta" href="#get-started">Start building <span aria-hidden="true">↗</span></a>
  </header>

  <main id="main">
    <section class="hero">
      <div class="hero-copy">
        <p class="eyebrow"><span class="status-dot"></span> LLM TO SYSTEM 1 <span class="version">v0.1</span></p>
        <h1>A small decision.<br /><em>A clear result.</em></h1>
        <p class="hero-description">Give your model a question.<br />Give your code a typed answer.</p>
        <p class="hero-detail">Turn local GGUF model scores into choices, booleans, and ordered values—with an explicit option to abstain.</p>
        <div class="hero-actions"><a class="button primary" href="#get-started">Get started <span aria-hidden="true">↗</span></a><a class="text-link" href="#playground">Explore the contract <span aria-hidden="true">↓</span></a></div>
        <div class="hero-tags"><span>MIT application code</span><span>CPU & CUDA</span><span>Bring your own model</span></div>
      </div>
      <div class="decision-window" aria-label="Illustration of a typed decision from warehouse input">
        <div class="window-bar"><span><i></i><i></i><i></i></span><span>decision / storage_zone</span><span>01</span></div>
        <div class="window-content">
          <div class="terminal-label"><span>INPUT STATE</span><span>JSON</span></div>
          <pre class="input-preview"><span class="code-muted">&#123;</span>
  <span class="code-key">"storage_requirement"</span>: <span class="code-string">"chilled"</span>,
  <span class="code-key">"hours_until_dispatch"</span>: <span class="code-number">4</span>
<span class="code-muted">&#125;</span></pre>
          <div class="flow-line"><span></span><b>↓</b><span>LOCAL GGUF MODEL</span></div>
          <div class="option-stack"><div><span class="letter">A</span><span>Ambient</span><span class="option-mark">—</span></div><div class="selected"><span class="letter">B</span><span>Chilled</span><span class="option-mark">↗</span></div><div><span class="letter">C</span><span>Frozen</span><span class="option-mark">—</span></div></div>
          <div class="flow-line small"><span></span><b>↓</b><span>TYPED RESULT</span></div>
          <div class="result-preview"><span class="code-muted">&#123; </span><span>"selected"</span>: <strong>"chilled"</strong><span class="code-muted"> &#125;</span></div>
          <p class="illustration-note">Illustrative output · no model runs in this browser</p>
        </div>
        <span class="window-offset" aria-hidden="true">ONE QUESTION. ONE STRUCTURED RESULT.</span>
      </div>
    </section>

    <div class="principles"><span><b>01</b> Your data stays with your runtime</span><span><b>02</b> Scores you can inspect</span><span><b>03</b> Uncertainty you can handle</span></div>

    <section id="how-it-works" class="section contract-section">
      <div class="section-heading"><p class="eyebrow">01 / THE CONTRACT</p><h2>Small outputs.<br /><em>Useful building blocks.</em></h2><p>Define the options. Read their next-token scores. Apply a policy. Return a value your application understands.</p></div>
      <div class="contract-grid">
        <article><span class="card-index">A → B → C</span><h3>Choose an option.</h3><p>Route a shipment, select a category, or pick the next action. Your semantic IDs stay yours.</p><code>{'{ "selected": "chilled" }'}</code><span class="card-type">CHOICE <span>↗</span></span></article>
        <article><span class="card-index">0 / 1</span><h3>Answer a condition.</h3><p>Turn a two-option question into a boolean, with the candidate-relative probability alongside it.</p><code>{'{ "value": true, "p_true": 0.92 }'}</code><span class="card-type">BINARY <span>↗</span></span></article>
        <article><span class="card-index">▂ ▄ ▆</span><h3>Find the level.</h3><p>Use your own ordered levels and numeric values for priority, severity, or another defined scale.</p><code>{'{ "selected": "high" }'}</code><span class="card-type">ORDINAL <span>↗</span></span></article>
      </div>
    </section>

    <section id="playground" class="playground-section">
      <div class="playground-copy"><p class="eyebrow">02 / MAKE ROOM FOR UNCERTAINTY</p><h2>A result can<br /><em>also be “wait.”</em></h2><p>Move the acceptance threshold to see a typed decision become an abstention. Your application decides what happens next.</p><div class="demo-disclosure"><span>ⓘ</span><p><strong>Illustrative scoring demo.</strong> These fixed probabilities are examples, not model inference or accuracy estimates. Candidate mass is fixed at 0.98, above the 0.05 policy threshold.</p></div></div>
      <div class="playground-panel">
        <div class="segmented-control" aria-label="Decision type">{#each ['choice', 'binary', 'ordinal'] as type (type)}<button class:active={kind === type} aria-pressed={kind === type} onclick={() => kind = type as Kind}>{type}</button>{/each}</div>
        <div class="question-line"><h3>{questions[kind]}</h3><span class:waiting={abstained} class="decision-status">{abstained ? 'Abstained' : 'Accepted'}</span></div>
        <div class="probability-list">{#each options as option (option.code)}<div class="probability-row"><span>{option.code}</span><span>{option.label}</span><div class="bar-track"><div style:width={`${option.probability * 100}%`}></div></div><strong>{Math.round(option.probability * 100)}%</strong></div>{/each}</div>
        <label class="threshold-label" for="threshold"><span>Minimum top probability</span><output for="threshold">{threshold}%</output></label>
        <input id="threshold" type="range" min="50" max="99" step="1" bind:value={threshold} />
        <div class="slider-labels"><span>More accepting</span><span>More selective</span></div>
        <div class="demo-result" aria-live="polite"><span class="terminal-label">RESULT.JSON</span><pre>{resultJson}</pre></div>
        <p class="panel-footnote">Candidate-relative probability is not a probability of correctness.</p>
      </div>
    </section>

    <section id="models" class="section model-section">
      <div class="section-heading horizontal"><div><p class="eyebrow">03 / YOUR MODEL, YOUR MACHINE</p><h2>A common contract.<br /><em>Different models.</em></h2></div><p>Use a compatible local chat GGUF. Embedded Jinja templates handle model-specific formatting; Qwen3 keeps its non-thinking profile.</p></div>
      <div class="model-list">{#each models as model (model.name)}<a href={model.url} target="_blank" rel="noreferrer external"><span class="model-dot" aria-hidden="true"></span><strong>{model.name}</strong><span>{model.detail}</span><span class="model-arrow" aria-hidden="true">↗</span></a>{/each}</div>
      <p class="section-note">Specific checkpoints tested locally. Compatibility still depends on the template, tokenizer, libllama build, and available memory. <a href="/docs/docs/en/VERIFICATION.md" rel="external">Read the verification record ↗</a></p>
    </section>

    <section id="performance" class="section performance-section">
      <div class="section-heading horizontal"><div><p class="eyebrow">04 / MEASURE THE WORK</p><h2>Performance,<br /><em>with the context attached.</em></h2></div><p>A local inference test measures loading, first-request latency, steady-state latency, and throughput. Use your own checkpoint and hardware.</p></div>
      {#if measurements.length}
        <div class="benchmark-meta"><span class="status-dot"></span><strong>Local measurements</strong><span>RTX 3080 · release build · {benchmarks.samples} samples · 3 decisions/request</span></div>
        <div class="table-scroll"><table><caption class="sr-only">Warehouse request latency and throughput by model</caption><thead><tr><th>Checkpoint</th><th>CPU p50 / request</th><th>CUDA p50 / request</th><th>CUDA decisions / sec</th><th>CUDA abstentions</th></tr></thead><tbody>{#each measurements as row (row.model)}<tr><td>{row.model}{#if !row.nativeCuda}<span aria-label="CUDA batch consistency limitation"> †</span>{/if}</td><td>{row.cpuP50.toFixed(1)} <span>ms</span></td><td>{row.cudaP50.toFixed(1)} <span>ms</span></td><td>{row.cudaDecisionsPerSecond.toFixed(2)}</td><td>{Math.round(row.cudaAbstentionRate * 100)}%</td></tr>{/each}</tbody></table></div>
        <p class="section-note">English warehouse fixture · {benchmarks.date}. Five samples are a smoke measurement, not a broad benchmark. Inference timings include prompt processing and scoring, not generated text. Hardware, quantization, and build differences matter. Throughput includes abstentions. † Gemma 3 and TinyLlama exceed the CUDA batch-consistency tolerance; these timings use a fixed batch of 256.</p>
      {/if}
      <a class="benchmark-link" href="/docs/docs/en/BENCHMARK.md" rel="external">Run the performance test on your machine <span aria-hidden="true">↗</span></a>
    </section>

    <section id="get-started" class="start-section">
      <div><p class="eyebrow">05 / START LOCAL</p><h2>Your next decision<br /><em>starts here.</em></h2><p>Bring a GGUF checkpoint and a matching llama.cpp build. Define a question, run the CLI, and inspect the JSON.</p><a class="text-link" href="/docs/warehouse.json" rel="external" download>Download the English example <span aria-hidden="true">↓</span></a></div>
      <div class="command-panel"><div class="command-header"><span>TERMINAL / QUICK START</span><button onclick={copyCommand}>{copied ? 'Copied ✓' : 'Copy command'}</button></div><pre><code>{command}</code></pre>{#if copyError}<p role="status">Clipboard access is unavailable. Select and copy the command above.</p>{/if}<div class="command-footer"><span>Local inference</span><span>No automatic downloads</span></div></div>
    </section>

    <section class="license-strip"><div class="license-symbol" aria-hidden="true">↗</div><div><h3>Open source code. Separate model licenses.</h3><p>The application is MIT-licensed. Model weights are never bundled. Gemma 4 uses Apache 2.0; Gemma 3 has its own terms. Preserve applicable third-party notices when distributing.</p></div><a href="/docs/docs/en/LICENSING.md" rel="external">License details ↗</a></section>
  </main>
  <footer><a class="brand" href="#main"><span class="brand-symbol" aria-hidden="true">s1<span>↗</span></span><span>L2S1</span></a><p>Local models. Typed decisions.</p><div><a href="/docs/docs/en/README.md" rel="external">English docs</a><a href="/docs/docs/ko/README.md" rel="external">한국어 문서</a><a href="/docs/docs/ja/README.md" rel="external">日本語ドキュメント</a><a href="/docs/LICENSE" rel="external">MIT license</a><a href="/docs/WEB_THIRD_PARTY_LICENSES.txt" rel="external">Third-party notices</a></div><span>Built with Rust. Presented with Svelte.</span></footer>
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
    background: #f5f4ed;
    color: #232b24;
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
    outline: 3px solid #638930;
    outline-offset: 5px;
  }
  :global(::selection) {
    background: #d1f77b;
    color: #202820;
  }
  .page-shell {
    max-width: 1440px;
    margin: auto;
    padding: 0 70px;
  }
  .site-header {
    height: 104px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 1px solid #d6d9ce;
    gap: 24px;
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
    background: #263327;
    color: #d1f77b;
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
  .site-header nav {
    display: flex;
    gap: 30px;
    font-size: 13px;
  }
  .site-header nav a:hover,
  .text-link:hover,
  footer a:hover {
    text-decoration: underline;
    text-underline-offset: 5px;
  }
  .header-cta {
    font-size: 13px;
    border: 1px solid #344735;
    padding: 12px 16px;
    border-radius: 4px;
    display: flex;
    gap: 23px;
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
    background: #648f43;
    display: inline-block;
    flex-shrink: 0;
  }
  .version {
    font-size: 10px;
    border: 1px solid #c9cec0;
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
    color: #506346;
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
    color: #687063;
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
    background: #283a2b;
    color: #fff;
  }
  .primary:hover {
    background: #3f553d;
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
    color: #65705e;
  }
  .hero-tags span + span::before {
    content: "·";
    margin-right: 15px;
  }
  .decision-window {
    background: #212c24;
    color: #e7ebe0;
    border-radius: 9px;
    position: relative;
    box-shadow: 0 16px 40px -25px #202e2350;
    transform: rotate(1deg);
  }
  .window-bar {
    height: 45px;
    border-bottom: 1px solid #ffffff16;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 22px;
    font: 10px monospace;
    color: #a3ad9e;
  }
  .window-bar > span:first-child {
    display: flex;
    gap: 5px;
  }
  .window-bar i {
    display: block;
    width: 6px;
    height: 6px;
    background: #65705e;
    border-radius: 50%;
  }
  .window-bar i:first-child {
    background: #bad98a;
  }
  .window-content {
    padding: 25px 28px 22px;
  }
  .terminal-label {
    display: flex;
    justify-content: space-between;
    font: 9px monospace;
    letter-spacing: 1.4px;
    color: #a5b299;
  }
  .input-preview {
    font: 12px/1.9 monospace;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    margin: 14px 0;
  }
  .code-muted {
    color: #87917f;
  }
  .code-key {
    color: #ced6c6;
  }
  .code-string,
  .code-number {
    color: #d1f77b;
  }
  .flow-line {
    display: flex;
    align-items: center;
    gap: 10px;
    font: 8px monospace;
    letter-spacing: 1.2px;
    color: #97a790;
    margin: 23px 0 16px;
  }
  .flow-line > span:first-child {
    width: 19px;
    height: 1px;
    background: #61755c;
  }
  .flow-line b {
    font: 20px monospace;
    color: #abc295;
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
    border: 1px solid #485440;
    border-radius: 4px;
    padding: 10px 12px;
    font-size: 12px;
    color: #b6c1b0;
  }
  .letter {
    display: flex;
    align-items: center;
    justify-content: center;
    font: 10px monospace;
    border: 1px solid #596652;
    border-radius: 3px;
    width: 22px;
    height: 22px;
  }
  .option-mark {
    margin-left: auto;
    font: 16px monospace;
  }
  .option-stack .selected {
    background: #d1f77b;
    color: #24351e;
    border-color: #d1f77b;
    font-weight: 600;
  }
  .selected .letter {
    border-color: #90ab59;
  }
  .flow-line.small {
    margin: 17px 0 12px;
  }
  .result-preview {
    padding: 14px 12px;
    background: #141e18;
    border: 1px solid #364731;
    border-radius: 4px;
    font: 12px/1.6 monospace;
  }
  .result-preview strong {
    color: #d1f77b;
    font-weight: 400;
  }
  .illustration-note {
    font: 9px/1.6 monospace;
    color: #a9b59f;
    margin: 13px 0 0;
  }
  .window-offset {
    position: absolute;
    bottom: -27px;
    right: 1px;
    color: #7b8573;
    font: 8px monospace;
    letter-spacing: 1.5px;
  }
  .principles {
    display: flex;
    justify-content: space-between;
    gap: 18px;
    padding: 22px 0;
    border-top: 1px solid #d6d9ce;
    border-bottom: 1px solid #d6d9ce;
    font-size: 12px;
    color: #626d5c;
  }
  .principles b {
    font: 10px monospace;
    color: #9aa18f;
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
    color: #65745c;
  }
  .section-heading > p:last-child,
  .section-heading.horizontal > p {
    font-size: 15px;
    line-height: 1.8;
    color: #65705f;
    max-width: 475px;
    margin-bottom: 0;
  }
  .contract-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }
  .contract-grid article {
    border: 1px solid #d2d7c8;
    border-radius: 5px;
    padding: 27px 24px 20px;
    display: flex;
    flex-direction: column;
    background: #f9f9f3;
  }
  .card-index {
    font: 14px monospace;
    color: #697e57;
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
    color: #677060;
    min-height: 70px;
    margin: 0 0 21px;
  }
  .contract-grid code {
    font-size: 10px;
    line-height: 1.6;
    background: #eaf0e0;
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
    color: #68785c;
  }
  .playground-section {
    background: #e9eddf;
    margin: 0 -70px;
    padding: 73px 70px;
    display: grid;
    grid-template-columns: 0.9fr 1.1fr;
    gap: 95px;
    align-items: center;
    border-top: 1px solid #d9decd;
    border-bottom: 1px solid #d9decd;
  }
  .playground-copy > p:not(.eyebrow) {
    font-size: 15px;
    line-height: 1.9;
    color: #5e6b54;
  }
  .demo-disclosure {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    margin-top: 30px;
    border-top: 1px solid #cbd2be;
    padding-top: 20px;
    color: #5e6a53;
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
    background: #fcfcf7;
    border: 1px solid #cdd6be;
    border-radius: 7px;
    padding: 24px;
    box-shadow: 0 10px 30px -25px #243e16;
  }
  .segmented-control {
    display: flex;
    padding: 4px;
    border-radius: 4px;
    background: #eaf0e1;
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
    color: #566748;
    border-radius: 3px;
  }
  .segmented-control button.active {
    background: #fcfcf7;
    box-shadow: 0 1px 4px #334f1f1a;
    color: #233718;
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
    background: #e3efcc;
    color: #38531e;
    font: 10px monospace;
    border-radius: 3px;
  }
  .decision-status.waiting {
    background: #f4e6ca;
    color: #795315;
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
    color: #667e52;
  }
  .probability-row strong {
    font: 10px monospace;
    text-align: right;
  }
  .bar-track {
    height: 5px;
    border-radius: 3px;
    background: #e6eadf;
    overflow: hidden;
  }
  .bar-track > div {
    background: #83a458;
    height: 100%;
    border-radius: 3px;
    transition: width 0.2s;
  }
  .threshold-label {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #55634b;
    margin-bottom: 10px;
  }
  .threshold-label output {
    font: 12px monospace;
    color: #293d20;
  }
  input[type="range"] {
    width: 100%;
    accent-color: #405d2e;
    cursor: pointer;
    margin: 0;
    height: 20px;
  }
  .slider-labels {
    display: flex;
    justify-content: space-between;
    color: #7a8472;
    font-size: 9px;
    margin-top: 4px;
  }
  .demo-result {
    margin-top: 23px;
    background: #253125;
    color: #e6ebdc;
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
    color: #d4e7bb;
  }
  .panel-footnote {
    font-size: 10px;
    line-height: 1.6;
    color: #6a775e;
    margin: 12px 0 0;
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
  .model-list {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    border: 1px solid #d2d7c8;
    border-radius: 5px;
    overflow: hidden;
  }
  .model-list > a {
    padding: 25px 19px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    background: #f9f9f3;
    position: relative;
    min-width: 0;
  }
  .model-list > a + a {
    border-left: 1px solid #d2d7c8;
  }
  .model-list > a:hover {
    background: #eaf0df;
  }
  .model-dot {
    width: 9px;
    height: 9px;
    background: #82976c;
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
    color: #718063;
  }
  .model-arrow {
    position: absolute;
    right: 16px;
    top: 23px;
    font-size: 16px;
    color: #78915f;
  }
  .section-note {
    font-size: 11px;
    line-height: 1.9;
    color: #6b7762;
    max-width: 850px;
    margin-top: 20px;
  }
  .section-note a {
    text-decoration: underline;
    text-underline-offset: 3px;
  }
  .performance-section {
    border-top: 1px solid #d6d9ce;
    padding-top: 69px;
  }
  .benchmark-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 11px;
    color: #748167;
    margin-bottom: 19px;
    flex-wrap: wrap;
  }
  .benchmark-meta strong {
    color: #536448;
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
    color: #66795a;
    padding: 15px 12px;
    border-bottom: 1px solid #c8d1bb;
    white-space: nowrap;
  }
  td {
    padding: 19px 12px;
    border-bottom: 1px solid #dde2d3;
    font-family: monospace;
    white-space: nowrap;
  }
  td:first-child {
    font:
      13px Arial,
      sans-serif;
  }
  td span {
    color: #8a9580;
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
    border-bottom: 1px solid #d2d9c8;
    font-size: 12px;
    margin-top: 30px;
  }
  .benchmark-link:hover {
    color: #527032;
  }
  .start-section {
    padding: 61px 0 80px;
    border-top: 1px solid #d6d9ce;
    display: grid;
    grid-template-columns: 1fr 1.2fr;
    gap: 74px;
    align-items: center;
  }
  .start-section > div > p:not(.eyebrow) {
    color: #67765a;
    font-size: 14px;
    line-height: 1.8;
    max-width: 350px;
  }
  .start-section .text-link {
    display: inline-block;
    margin-top: 15px;
  }
  .command-panel {
    background: #222e24;
    border-radius: 5px;
    color: #e0e9d5;
    overflow: hidden;
  }
  .command-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    padding: 17px 22px;
    border-bottom: 1px solid #46553b;
    font: 9px monospace;
    letter-spacing: 0.7px;
  }
  .command-header button {
    color: #e4eed9;
    border: 1px solid #6f7d61;
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
    color: #9cb08b;
    padding: 15px 22px;
    background: #1c261f;
    border-top: 1px solid #36462d;
  }
  .license-strip {
    border: 1px solid #d3d9c7;
    border-radius: 5px;
    display: flex;
    align-items: center;
    gap: 25px;
    padding: 26px;
    margin-bottom: 70px;
    background: #eff1e6;
  }
  .license-symbol {
    font:
      28px Georgia,
      serif;
    background: #e1e9d1;
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
    color: #647356;
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
    border-top: 1px solid #d5dbca;
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
    color: #748166;
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
    color: #7e8a71;
    margin-top: 13px;
  }
  .skip-link {
    position: absolute;
    left: 20px;
    top: -60px;
    background: #d1f77b;
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
    .site-header {
      height: 80px;
    }
    .site-header nav {
      display: none;
    }
    .brand {
      font-size: 19px;
    }
    .header-cta {
      padding: 10px 12px;
      font-size: 11px;
      gap: 13px;
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
      border-top: 1px solid #d2d7c8;
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
    .header-cta {
      font-size: 10px;
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
</style>
