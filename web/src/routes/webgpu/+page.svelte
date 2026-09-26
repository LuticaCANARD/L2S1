<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve } from '$app/paths';
  import { locale, translate } from '$lib/i18n';
  import { messages, WebgpuMessageError, type WebgpuMessageKey, type MessageSpec, type MessageParams } from '$lib/i18n/webgpu';
  const t = (key: WebgpuMessageKey, params?: MessageParams) => translate($locale, messages, key, params);
  import warehouse from '$lib/demo/text-request.json';
  import { percent, reasonLabel, selection } from '$lib/demo/types';
  import { CACHE_KEY, MODEL_ID, validateBrowserRequest, type AdapterInfo, type BrowserRequest, type BrowserResponse, type Dtype, type WorkerOutput } from '$lib/webgpu/contract';
  const cat = { state: { text: 'The animal is a cat.' }, decisions: [{ id: 'cat_present', instruction: 'Does the text explicitly name a cat?', kind: { type: 'binary', false_label: 'No', true_label: 'Yes' } }] };
  let stateText = $state(JSON.stringify(cat.state, null, 2));
  let decisionsText = $state(JSON.stringify(cat.decisions, null, 2));
  let phase = $state<'checking' | 'idle' | 'unsupported' | 'loading' | 'ready' | 'running'>('checking');
  let loaded = $state(false);
  let capabilityFailure = $state<MessageSpec | string | null>(null);
  let support = $state<AdapterInfo | null>(null);
  let dtype = $state<Dtype>('q4f16');
  let reasoningMode = $state<'direct' | 'thinking'>('direct');
  let reasoningTokens = $state(128);
  let minTop = $state(0.8);
  let minMass = $state(0.05);
  let targetErrorRate = $state(20);
  function initialFailureMessages() { return JSON.stringify({ low_top_probability: t('failureLowTop'), low_candidate_mass: t('failureLowMass'), tied_candidates: t('failureTied'), reasoning_limit: t('failureLimit'), reasoning_incomplete: t('failureIncomplete'), unsupported_thinking: t('failureUnsupported'), native_failure: t('failureNative') }, null, 2); }
  let failureText = $state(initialFailureMessages());
  let lastDefaultReasons = initialFailureMessages();
  $effect(() => {
    const next = initialFailureMessages();
    if (next !== lastDefaultReasons) {
      if (failureText === lastDefaultReasons) failureText = next;
      lastDefaultReasons = next;
    }
  });
  let statusMessage = $state<MessageSpec | string>({ key: 'statusChecking' });
  let status = $derived(typeof statusMessage === 'string' ? statusMessage : t(statusMessage.key, statusMessage.params));
  let errorMessage = $state<{ key?: WebgpuMessageKey; params?: MessageParams; raw?: string; code?: string } | null>(null);
  let error = $derived(errorMessage ? `${errorMessage.code ? `[${errorMessage.code}] ` : ''}${errorMessage.key ? t(errorMessage.key, errorMessage.params) : errorMessage.raw ?? ''}` : '');
  function showError(cause: unknown) { errorMessage = cause instanceof WebgpuMessageError ? { key: cause.message_key, params: cause.message_params } : { raw: cause instanceof Error ? cause.message : String(cause) }; }
  let userReason = $state('');
  let response = $state<BrowserResponse | null>(null);
  let files = $state<Record<string, { progress: number; loaded: number; total: number }>>({});
  let currentFile = $state('');
  let worker: Worker | undefined;
  let busy = $derived(phase === 'loading' || phase === 'running');
  let loadedBytes = $derived(Object.values(files).reduce((sum, file) => sum + file.loaded, 0));
  let activeProgress = $derived(files[currentFile]?.progress ?? 0);
  let blockedMessage = $derived(phase === 'checking' ? t('statusChecking')
    : capabilityFailure ? typeof capabilityFailure === 'string' ? capabilityFailure : t(capabilityFailure.key, capabilityFailure.params)
    : !loaded && error ? error
    : phase === 'loading' ? status
    : phase === 'running' ? t('blockedRunning')
    : !loaded ? t('blockedModel') : null);
  const size = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
  function clearResult() { response = null; errorMessage = null; userReason = ''; }
  function example(kind: 'cat' | 'warehouse') {
    const fixture = kind === 'cat' ? cat : warehouse;
    stateText = JSON.stringify(fixture.state, null, 2); decisionsText = JSON.stringify(fixture.decisions, null, 2); clearResult();
  }
  function stopWorker() {
    worker?.terminate(); worker = undefined; loaded = false; phase = support ? 'idle' : 'unsupported';
    files = {}; currentFile = ''; response = null;
  }
  function stop() { stopWorker(); statusMessage = { key: 'statusStopped' }; }
  async function clearCache() {
    stopWorker(); clearResult();
    try { if ('caches' in window) await caches.delete(CACHE_KEY); statusMessage = { key: 'statusCacheCleared' }; }
    catch (cause) { errorMessage = { key: 'errorCache', params: { error: (cause as Error).message } }; }
  }
  function failureMessage(code: string): string { try { return JSON.parse(failureText)[code] ?? ''; } catch { return ''; } }
  function startLoad() {
    if (!support || busy) return;
    clearResult(); files = {}; currentFile = ''; phase = 'loading'; statusMessage = { key: 'statusDownloading' };
    worker?.terminate(); worker = new Worker(new URL('../../lib/webgpu/worker.ts', import.meta.url), { type: 'module' });
    worker.onmessage = (event: MessageEvent<WorkerOutput>) => {
      const message = event.data;
      if (message.type === 'progress') {
        currentFile = message.file;
        files = { ...files, [message.file]: { progress: message.progress, loaded: message.loaded ?? files[message.file]?.loaded ?? 0, total: message.total ?? files[message.file]?.total ?? 0 } };
      } else if (message.type === 'status') statusMessage = message.message_key ? { key: message.message_key, params: message.message_params } : message.message;
      else if (message.type === 'ready') { loaded = true; support = message.adapter; phase = 'ready'; statusMessage = { key: 'statusReady' }; }
      else if (message.type === 'result') { response = message.response; phase = 'ready'; statusMessage = { key: 'statusCompleted' }; }
      else if (message.type === 'released') { stopWorker(); statusMessage = { key: 'statusReleased' }; }
      else { errorMessage = { key: message.message_key, params: message.message_params, raw: message.message, code: message.code }; userReason = message.user_reason ?? failureMessage(message.code); phase = loaded ? 'ready' : support ? 'idle' : 'unsupported'; }
    };
    worker.onerror = (event) => { errorMessage = { key: 'errorWorker', params: { error: event.message } }; stopWorker(); statusMessage = { key: 'statusReload' }; };
    worker.postMessage({ type: 'load', dtype, locale: $locale });
  }
  function release() { if (!worker || busy) return; phase = 'loading'; statusMessage = { key: 'statusReleasing' }; worker.postMessage({ type: 'release', locale: $locale }); }
  function analyze() {
    if (!loaded || !worker || busy) return;
    clearResult();
    try {
      const request = { state: JSON.parse(stateText), decisions: JSON.parse(decisionsText), reasoning: { mode: reasoningMode, max_tokens: reasoningTokens }, policy: { min_top_probability: minTop, min_candidate_mass: minMass }, failure_reasons: JSON.parse(failureText) } as BrowserRequest;
      validateBrowserRequest(request, $locale);
      if (!Number.isFinite(targetErrorRate) || targetErrorRate < 0 || targetErrorRate > 100) throw new WebgpuMessageError('errorRateRange', {}, $locale);
      phase = 'running'; statusMessage = { key: 'statusAnalyzing' }; worker.postMessage({ type: 'analyze', request, locale: $locale });
    } catch (cause) { if (cause instanceof SyntaxError) errorMessage = { key: 'errorJson' }; else showError(cause); }
  }
  onMount(() => {
    let active = true;
    async function probe() {
      try {
        if (!window.isSecureContext) throw new WebgpuMessageError('errorInsecureContext', {}, $locale);
        const gpu = (navigator as unknown as { gpu?: GPU }).gpu;
        if (!gpu) throw new WebgpuMessageError('errorNoWebgpu', {}, $locale);
        const adapter = await gpu.requestAdapter();
        if (!adapter) throw new WebgpuMessageError('errorNoAdapter', {}, $locale);
        const info = adapter.info ?? { description: '', vendor: '', architecture: '' };
        if (active) {
          support = { description: info.description || '', vendor: info.vendor, architecture: info.architecture, shaderF16: adapter.features.has('shader-f16'), software: ('isFallbackAdapter' in adapter && adapter.isFallbackAdapter === true) || /swiftshader|software|llvmpipe/i.test(`${info.description} ${info.vendor} ${info.architecture}`), hardwareVerified: false };
          dtype = support.shaderF16 ? 'q4f16' : 'q4'; phase = 'idle'; statusMessage = { key: 'statusAdapterReady' };
        }
      } catch (cause) { if (active) { phase = 'unsupported'; capabilityFailure = cause instanceof WebgpuMessageError ? { key: cause.message_key, params: cause.message_params } : { key: 'errorAdapterProbe', params: { error: (cause as Error).message } }; statusMessage = capabilityFailure; } }
    }
    void probe(); return () => { active = false; worker?.terminate(); };
  });
</script>

<svelte:head><title>{t('title')}</title><meta name="description" content={t('description')} /></svelte:head>
<div class="shell" lang={$locale}>
  <main><p class="eyebrow">{t('eyebrow')}</p><h1>{t('heroStart')}<br /><em>{t('heroEnd')}</em></h1><p class="lead">{t('leadFirst')}<br /> {t('leadSecond')}</p>
    <section class="load-panel" aria-labelledby="model-heading"><div><h2 id="model-heading">{t('prepare')}</h2><p><a href="https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX" target="_blank" rel="noreferrer external">{MODEL_ID} ↗</a> · Apache 2.0</p></div><div class="model-controls"><label for="dtype">{t('dtype')}</label><select id="dtype" bind:value={dtype} disabled={busy || loaded}><option value="q4f16" disabled={!support?.shaderF16}>q4f16 · fp16 GPU</option><option value="q4">q4 · fp32 GPU</option></select><div class="actions"><button class="primary" onclick={startLoad} disabled={!support || busy || loaded}>{t('load')}</button><button onclick={release} disabled={!loaded || busy}>{t('release')}</button><button onclick={clearCache} disabled={busy}>{t('clearCache')}</button>{#if busy}<button onclick={stop}>{t('stop')}</button>{/if}</div></div>
      <div class="support" role="status"><strong>{phase === 'unsupported' ? t('unsupported') : support?.software ? t('softwareAdapter') : support ? t('adapterConfirmed') : t('checkingSupport')}</strong><p>{status}</p>{#if support}<p>{support.description || t('adapterUndisclosed')} · fp16 {support.shaderF16 ? t('supported') : t('notSupported')}</p>{/if}</div>
      {#if phase === 'loading' && currentFile}<div class="progress"><label for="download-progress">{currentFile} · {activeProgress.toFixed(1)}%</label><progress id="download-progress" max="100" value={activeProgress}></progress><p class="hint">{t('downloadTotal', { size: size(loadedBytes) })}</p></div>{/if}
    </section>
    <div class="workspace"><section class="panel"><h2>{t('inputHeading')}</h2><div class="actions"><button onclick={() => example('cat')} disabled={busy}>{t('catExample')}</button><button onclick={() => example('warehouse')} disabled={busy}>{t('warehouseExample')}</button></div><label for="state">{t('stateJson')}</label><textarea id="state" rows="5" bind:value={stateText} oninput={clearResult} disabled={busy} spellcheck="false"></textarea><details><summary>{t('editQuestions')}</summary><p class="hint">{t('questionsHint')}</p><label class="sr-only" for="questions">{t('questionsJson')}</label><textarea id="questions" rows="18" bind:value={decisionsText} oninput={clearResult} disabled={busy} spellcheck="false"></textarea></details>
      <label for="reasoning">{t('reasoning')}</label><select id="reasoning" bind:value={reasoningMode} onchange={clearResult} disabled={busy}><option value="direct">{t('direct')}</option><option value="thinking">{t('thinking')}</option></select>{#if reasoningMode === 'thinking'}<label for="max-thought">{t('thoughtLimit')}</label><input id="max-thought" type="number" min="1" max="256" bind:value={reasoningTokens} oninput={clearResult} disabled={busy} /><p class="hint">{t('thoughtHint')}</p>{/if}
      <label for="error-rate">{t('errorRate')}</label><input id="error-rate" type="number" min="0" max="100" step="1" bind:value={targetErrorRate} oninput={(event) => { minTop = Number((1-Number(event.currentTarget.value)/100).toFixed(4)); clearResult(); }} disabled={busy} /><p class="hint">{t('errorRateHint')}</p><div class="policy-grid"><div><label for="min-top">{t('minTop')}</label><input id="min-top" type="number" min="0" max="1" step="0.01" bind:value={minTop} oninput={(event) => { targetErrorRate = Number(((1-Number(event.currentTarget.value))*100).toFixed(2)); clearResult(); }} disabled={busy} /></div><div><label for="min-mass">{t('minMass')}</label><input id="min-mass" type="number" min="0" max="1" step="0.01" bind:value={minMass} oninput={clearResult} disabled={busy} /></div></div>
      <details><summary>{t('customFailures')}</summary><label class="sr-only" for="failure">{t('failureJson')}</label><textarea id="failure" rows="10" bind:value={failureText} oninput={clearResult} disabled={busy} spellcheck="false"></textarea><p class="hint">{t('failureHint')}</p></details><button class="primary run" onclick={analyze} disabled={!loaded || busy} aria-describedby={blockedMessage ? 'analysis-blocked' : undefined}>{phase === 'running' ? t('analyzing') : t('analyze')}</button>{#if blockedMessage}<p id="analysis-blocked" class="hint" role="status" aria-live="polite">{blockedMessage}</p>{/if}{#if error}<div class="error" role="alert"><strong>{error}</strong>{#if userReason}<p>{t('customFailure', { reason: userReason })}</p>{/if}</div>{/if}
    </section><section class="panel output" aria-busy={phase === 'running'}><h2>{t('resultsHeading')}</h2><div aria-live="polite">{#if response}<div class="evidence-note"><strong>{t('actualOnnx')}</strong><p>{t('inferenceTiming', { dtype: response.backend.dtype, seconds: (response.elapsed_ms/1000).toFixed(2) })}</p><p>{response.backend.adapter.software ? t('softwareRun') : response.backend.adapter.description || t('adapterUndisclosed')}</p></div>{#each response.results as result (result.id)}<article class="result"><div class="result-head"><h3>{result.id}</h3><span class:abstained={result.status === 'abstained'}>{result.value.type} · {result.status === 'abstained' ? t('abstained') : t('selected')}</span></div><p class="value">{selection(result, $locale)}</p>{#if result.usage?.reasoning}<p class="hint">{t('usage', { mode: result.usage.reasoning.mode, tokens: result.usage.reasoning.generated_tokens, completed: result.usage.reasoning.completed ? t('yes') : t('no') })}</p>{/if}{#each result.evidence.scores ?? [] as score (score.id)}<div class="score"><span>{score.id}</span><div class="track"><div style:width={`${score.option_probability*100}%`}></div></div><strong>{percent(score.option_probability, $locale)}</strong></div>{/each}<dl><div><dt>{t('topProbability')}</dt><dd>{percent(result.evidence.top_option_probability, $locale)}</dd></div><div><dt>{t('candidateMass')}</dt><dd>{percent(result.evidence.candidate_mass, $locale)}</dd></div>{#if result.evidence.estimate?.expected_value !== undefined}<div><dt>{t('expectedValue')}</dt><dd>{result.evidence.estimate.expected_value.toFixed(3)}</dd></div>{/if}</dl>{#if result.abstention_reasons.length}<ul>{#each result.abstention_reasons as reason (reason)}<li>{reasonLabel(reason, $locale)} <code>{reason}</code></li>{/each}{#each result.reason_messages ?? [] as message (message.code)}<li>{message.message}</li>{/each}</ul>{/if}</article>{/each}<details><summary>{t('fullJson')}</summary><pre>{JSON.stringify(response,null,2)}</pre></details>{:else}<div class="empty"><span aria-hidden="true">↗</span><h3>{phase === 'running' ? t('waitingScores') : t('loadBeforeAnalysis')}</h3></div>{/if}</div></section></div>
  </main><footer><a href={resolve('/')}>{t('back')}</a><a href="/docs/WEB_THIRD_PARTY_LICENSES.txt" rel="external">{t('licenses')}</a></footer>
</div>
<style>
  :global(body){margin:0;background:var(--theme-page, #f6f5f0);color:var(--theme-text, #27352a);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}:global(*){box-sizing:border-box}a{color:inherit}button,input,select,textarea{font:inherit}:global(button),:global(a),:global(input),:global(select),:global(textarea),summary{outline-offset:5px}.shell{max-width:1380px;margin:auto;padding:0 42px}footer{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--theme-border, #d8dfd1);padding:25px 0;gap:15px}main{padding-top:35px}.eyebrow{font-size:11px;color:var(--theme-muted, #6a8045);letter-spacing:2px;font-weight:700}h1{font-size:clamp(35px,4vw,53px);line-height:1.2;letter-spacing:-1.5px;margin:18px 0}em{color:var(--theme-muted, #6a8045);font-style:normal}.lead{font-size:16px;line-height:1.8;margin:20px 0 35px}h2{font-size:18px;margin:0 0 16px}.load-panel,.panel{background:var(--theme-panel, white);border:1px solid var(--theme-border, #dce2d5);border-radius:10px;padding:23px;min-width:0}.load-panel{display:grid;grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:22px;margin-bottom:25px}.load-panel>div>p{font-size:13px;line-height:1.8;margin:8px 0}.hint{font-size:12px!important;line-height:1.8!important;color:var(--theme-muted, #707f66);overflow-wrap:anywhere}.actions{display:flex;gap:9px;flex-wrap:wrap;align-items:center;margin:12px 0}button{border:1px solid var(--theme-border, #d1dbc8);background:var(--theme-panel, white);color:var(--theme-muted, #344a2e);border-radius:5px;padding:11px 13px;font-size:12px;cursor:pointer;font-weight:600}button.primary{background:var(--theme-primary, #36532b);color:var(--theme-on-primary, white);border-color:var(--theme-border, #36532b)}button:disabled{opacity:.45;cursor:default}.model-controls label{margin-top:0}label{display:block;font-size:12px;font-weight:600;margin:20px 0 9px}select,input{border:1px solid var(--theme-border, #d4decb);background:var(--theme-panel, #fafbf7);border-radius:5px;padding:10px 12px;font-size:13px;color:var(--theme-muted, #344a2e);max-width:100%}.support{grid-column:1/-1;background:var(--theme-tint, #eff3e9);border:1px solid var(--theme-border, #dce5d1);border-radius:6px;padding:13px;font-size:12px}.support p{margin:7px 0 0;line-height:1.7;overflow-wrap:anywhere}.progress{grid-column:1/-1}.progress label{font-size:11px;overflow-wrap:anywhere;margin:0 0 8px}progress{width:100%;height:13px;accent-color:var(--theme-chart, #758d51)}.workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.03fr);gap:25px;align-items:start}textarea{width:100%;max-width:100%;resize:vertical;border:1px solid var(--theme-border, #d4decb);background:var(--theme-panel, #fafbf7);border-radius:5px;padding:12px;color:var(--theme-muted, #344a2e);font:12px/1.75 ui-monospace,SFMono-Regular,monospace}details{border-top:1px solid var(--theme-border, #e1e8da);margin-top:20px;padding-top:14px;min-width:0}summary{font-size:12px;cursor:pointer;line-height:1.8}.policy-grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}.policy-grid input{width:100%}.run{width:100%;margin-top:25px}.error{background:var(--theme-error-bg, #fff0e7);border:1px solid var(--theme-error-border, #e8c7b2);border-radius:6px;padding:15px;margin-top:17px;font-size:12px;line-height:1.8;color:var(--theme-error-text, #8a4d31);overflow-wrap:anywhere}.error p{margin-bottom:0}.evidence-note{background:var(--theme-tint, #eff3e9);border:1px solid var(--theme-border, #dce5d1);border-radius:6px;padding:14px;font-size:12px;color:var(--theme-muted, #5c7245)}.evidence-note p{margin:7px 0 0;line-height:1.8}.result{border:1px solid var(--theme-border, #dee6d5);border-radius:6px;padding:18px;margin:18px 0}.result-head{display:flex;align-items:center;justify-content:space-between;gap:15px}.result h3{font-size:16px;margin:0;overflow-wrap:anywhere}.result-head span{font-size:10px;background:var(--theme-tint, #eef4e7);color:var(--theme-accent, #547135);padding:6px;border-radius:4px;white-space:nowrap}.result-head span.abstained{background:var(--theme-warning-bg, #fff0d9);color:var(--theme-warning-text, #8c672e)}.value{font-size:25px;font-weight:700;margin:17px 0;overflow-wrap:anywhere}.score{display:grid;grid-template-columns:minmax(60px,1fr) minmax(50px,1.2fr) 70px;gap:10px;font-size:11px;align-items:center;margin:12px 0}.score span{overflow-wrap:anywhere}.score strong{text-align:right;font-variant-numeric:tabular-nums}.track{background:var(--theme-tint, #edf1e7);border-radius:3px;height:6px;overflow:hidden}.track>div{height:100%;background:var(--theme-chart, #819858)}dl{display:flex;gap:20px;flex-wrap:wrap;border-top:1px solid var(--theme-border, #e2e8db);margin:20px 0 0;padding-top:13px}dt{font-size:10px;color:var(--theme-muted, #7a8870)}dd{margin:5px 0 0;font-size:15px;font-weight:600}.result ul{padding-left:18px;color:var(--theme-warning-text, #7f6734);font-size:12px;line-height:1.9;overflow-wrap:anywhere}.result code{font-size:10px}pre{background:var(--theme-tint, #f1f4eb);border:1px solid var(--theme-border, #dce5d1);border-radius:5px;padding:12px;max-width:100%;overflow:auto;font:11px/1.75 ui-monospace,SFMono-Regular,monospace}.empty{min-height:350px;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:30px;text-align:center;color:var(--theme-muted, #7a896d)}.empty>span{font-size:38px}.empty h3{font-size:17px;line-height:1.8}footer{border-bottom:0;border-top:1px solid var(--theme-border, #d8dfd1);font-size:12px;margin-bottom:10px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:750px){.shell{padding:0 18px}main{padding-top:25px}.lead{font-size:14px}.load-panel{grid-template-columns:1fr;gap:12px;padding:18px}.support,.progress{grid-column:1}.workspace{grid-template-columns:1fr;gap:18px}.panel{padding:18px}h1{letter-spacing:-1px}.policy-grid{gap:10px}.policy-grid label{font-size:11px}.empty{min-height:250px}}
</style>
