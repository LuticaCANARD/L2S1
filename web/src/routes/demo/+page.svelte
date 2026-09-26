<script lang="ts">
  import { resolve } from '$app/paths';
  import { onMount } from 'svelte';
  import { locale, translate } from '$lib/i18n';
  import { messages } from '$lib/i18n/demo';
  type MessageKey = keyof typeof messages.en;
  const t = (key: MessageKey, params?: Record<string, string | number>) => translate($locale, messages, key, params);
  let errorKey = $state<MessageKey | null>(null);
  let errorParams = $state<Record<string, string | number>>({});
  const error = $derived(errorKey ? t(errorKey, errorParams) : '');
  function showError(cause: unknown, fallback?: MessageKey) { errorKey = cause instanceof UiError ? cause.key : fallback ?? 'networkFailure'; errorParams = cause instanceof UiError ? cause.params ?? {} : { detail: (cause as Error).message }; }
  function defaultReasons() { return JSON.stringify({ low_top_probability: t('defaultLowTop'), low_candidate_mass: t('defaultLowMass'), tied_candidates: t('defaultTie'), reasoning_limit: t('defaultLimit'), native_failure: t('defaultFailure') }, null, 2); }
  let lastDefaultReasons = defaultReasons();
  import defaults from '$lib/demo/request.json';
  import textDefaults from '$lib/demo/text-request.json';
  import { DemoError as UiError, MAX_IMAGE_BYTES, MAX_REQUEST_BYTES, percent, reasonLabel, selection, validateRequest } from '$lib/demo/types';
  import type { AnalysisResponse, Decision, Recording } from '$lib/demo/types';
  let recording = $state<Recording | null>(null);
  let photoRecording = $state<Recording | null>(null);
  let textDirectRecording = $state<Recording | null>(null);
  let textThinkingRecording = $state<Recording | null>(null);
  let inputMode = $state<'photo' | 'text'>('photo');
  let targetErrorRate = $state(20);
  let minTopProbability = $state(0.8);
  let minCandidateMass = $state(0.05);
  let failureReasonsText = $state(lastDefaultReasons);
  $effect(() => { const next = defaultReasons(); if (failureReasonsText === lastDefaultReasons) failureReasonsText = next; lastDefaultReasons = next; });
  let recordingError = $state<MessageKey | null>(null);
  let stateText = $state(JSON.stringify(defaults.state, null, 2));
  let decisionsText = $state(JSON.stringify(defaults.decisions, null, 2));
  let imageUrl = $state('');
  let imageAlt = $state('');
  let imageFile = $state<File | null>(null);
  let imageName = $state('');
  let response = $state<AnalysisResponse | null>(null);
  let evaluatedDecisions = $state<Decision[]>([]);
  let mode = $state<'recorded' | 'live' | null>(null);
  let busy = $state(false);
  let elapsedMs = $state<number | undefined>(undefined);
  let reasoningMode = $state<'direct' | 'thinking'>('direct');
  let reasoningTokens = $state(128);
  let abort: AbortController | undefined;
  let uploadInput = $state<HTMLInputElement>();
  const displayNames: Record<string, MessageKey> = { material: 'material', single_object: 'single_object', transparency: 'transparency', storage_zone: 'storage_zone', cold_chain_required: 'cold_chain_required', dispatch_priority: 'dispatch_priority' };
  function clearResult() { response = null; mode = null; errorKey = null; }
  function replaceImage(url: string) { if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl); imageUrl = url; }
  function loadRecording() {
    if (!recording || busy) return;
    if (recording.image_url) replaceImage(recording.image_url); else replaceImage('');
    imageAlt = recording.image_alt ?? ''; imageName = recording.title; imageFile = null;
    reasoningMode = recording.request.reasoning?.mode ?? 'direct'; reasoningTokens = recording.request.reasoning?.max_tokens ?? 128;
    minTopProbability = recording.response.policy?.min_top_probability ?? 0.8; minCandidateMass = recording.response.policy?.min_candidate_mass ?? 0.05; targetErrorRate = Number(((1-minTopProbability)*100).toFixed(2));
    if (uploadInput) uploadInput.value = '';
    stateText = JSON.stringify(recording.request.state, null, 2); decisionsText = JSON.stringify(recording.request.decisions, null, 2);
    evaluatedDecisions = recording.request.decisions; response = recording.response; elapsedMs = recording.elapsed_ms; mode = 'recorded'; errorKey = null;
  }
  function changeInput(next: 'photo' | 'text') {
    inputMode = next; clearResult(); reasoningMode = 'direct';
    recording = next === 'photo' ? photoRecording : textDirectRecording;
    if (recording) loadRecording();
    else if (next === 'text') { stateText = JSON.stringify(textDefaults.state, null, 2); decisionsText = JSON.stringify(textDefaults.decisions, null, 2); }
  }
  function showThinkingRecord() { if (textThinkingRecording) { recording = textThinkingRecording; loadRecording(); } }
  function changeErrorRate(event: Event) { const rate = Number((event.target as HTMLInputElement).value); minTopProbability = Number((1-rate/100).toFixed(4)); clearResult(); }
  function upload(event: Event) {
    const file = (event.target as HTMLInputElement).files?.[0]; if (!file) return; clearResult(); imageFile = null; imageName = ''; replaceImage('');
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { errorKey = 'imageType'; return; }
    if (file.size === 0 || file.size > MAX_IMAGE_BYTES) { errorKey = 'imageSize'; return; }
    imageFile = file; imageName = file.name; imageAlt = ''; replaceImage(URL.createObjectURL(file));
  }
  async function readImageBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1]);
      reader.onerror = () => reject(new UiError('readImage')); reader.readAsDataURL(blob);
    });
  }
  async function analyze() {
    if (busy) return; clearResult();
    if (inputMode === 'photo' && !imageUrl) { errorKey = 'chooseImage'; return; }
    let request;
    try {
      request = validateRequest(stateText, decisionsText, $locale);
      if (!Number.isFinite(targetErrorRate) || targetErrorRate < 0 || targetErrorRate > 100) throw new UiError('rateInvalid');
      if (![minTopProbability, minCandidateMass].every((v) => Number.isFinite(v) && v >= 0 && v <= 1)) throw new UiError('thresholdInvalid');
      if (!Number.isInteger(reasoningTokens) || reasoningTokens < 1 || reasoningTokens > 1024) throw new UiError('tokensInvalid');
    } catch (cause) { showError(cause, cause instanceof SyntaxError ? 'jsonInvalid' : undefined); return; }
    busy = true; abort = new AbortController(); const timeout = window.setTimeout(() => abort?.abort(), 180_000); const start = performance.now();
    try {
      const customReasons = JSON.parse(failureReasonsText);
      if (!customReasons || Array.isArray(customReasons) || typeof customReasons !== 'object' || Object.values(customReasons).some((value) => typeof value !== 'string')) throw new UiError('reasonsInvalid');
      let media: { id: string; type: string; data_base64: string }[] = [];
      if (inputMode === 'photo') {
      let blob: Blob;
      if (imageFile) blob = imageFile;
      else { const result = await fetch(imageUrl, { signal: abort.signal }); if (!result.ok) throw new UiError('sampleRead'); blob = await result.blob(); }
      if (blob.size > MAX_IMAGE_BYTES) throw new UiError('imageTooLarge');
      media = [{ id: 'photo', type: 'image', data_base64: await readImageBase64(blob) }];
      } else request.decisions = request.decisions.map((decision) => ({ ...decision, media_ids: [] }));
      const body = JSON.stringify({ ...request, target_error_rate: targetErrorRate / 100, policy: { min_top_probability: minTopProbability, min_candidate_mass: minCandidateMass }, failure_reasons: customReasons, reasoning: { mode: reasoningMode, max_tokens: reasoningTokens }, media });
      if (new TextEncoder().encode(body).byteLength > MAX_REQUEST_BYTES) throw new UiError('requestTooLarge');
      const result = await fetch(inputMode === 'photo' ? '/inference/v1/decisions' : '/text-inference/v1/decisions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body, signal: abort.signal });
      const text = await result.text(); let json;
      try { json = JSON.parse(text); } catch { throw new UiError('connection'); }
      if (!result.ok) { const detail = typeof json.error === 'string' ? json.error : json.error?.message ?? json.message ?? `HTTP ${result.status}`; const custom = json.error?.user_reason ?? ''; const code = json.error?.code ? `[${json.error.code}] ` : ''; throw new UiError('serverRejected', { detail: `${code}${detail}`, custom }); }
      if (!Array.isArray(json.results) || !json.backend) throw new UiError('contract');
      evaluatedDecisions = request.decisions; response = json; elapsedMs = performance.now() - start; mode = 'live';
    } catch (cause) { showError(cause, cause instanceof SyntaxError ? 'reasonsJson' : (cause as Error).name === 'AbortError' ? 'aborted' : undefined); }
    finally { window.clearTimeout(timeout); busy = false; abort = undefined; }
  }
  onMount(() => {
    let mounted = true;
    fetch('/demo/recorded.json').then(async (result) => {
      if (!result.ok) throw new UiError('recordingLoad'); const data = await result.json() as Recording;
      if (!data.response?.results?.length || !data.request?.decisions?.length || !data.image_url?.startsWith('/demo/')) throw new UiError('recordingFormat');
      if (mounted) { photoRecording = data; if (inputMode === 'photo') { recording = data; loadRecording(); } }
    }).catch((cause) => { if (mounted) recordingError = cause instanceof UiError ? cause.key : 'recordingLoad'; });
    for (const mode of ['direct', 'thinking'] as const) fetch(`/demo/text-${mode}.json`).then(async (result) => { if (!result.ok) return; const data = await result.json() as Recording; if (!data.response?.results?.length) return; if (mounted) { if (mode === 'direct') textDirectRecording = data; else textThinkingRecording = data; } }).catch(() => {});
    return () => { mounted = false; abort?.abort(); if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl); };
  });
</script>

<svelte:head><title>{t('title')}</title><meta name="description" content={t('description')} /></svelte:head>
<a class="skip" href="#main">{t('skip')}</a>
<div class="shell" lang={$locale}>
  <main id="main">
    <div class="intro"><p class="eyebrow">{t('visionBadge')}</p><h1>{t('headline1')}<br /><em>{t('headline2')}</em></h1><p>{t('lead1')}<br /> {t('lead2')}</p></div>
    <div class="input-tabs" aria-label={t('inputType')}><button class:active={inputMode === 'photo'} aria-pressed={inputMode === 'photo'} onclick={() => changeInput('photo')} disabled={busy}>{t('photo')}</button><button class:active={inputMode === 'text'} aria-pressed={inputMode === 'text'} onclick={() => changeInput('text')} disabled={busy}>{t('text')}</button>{#if response}<a class="results-link" href="#output-heading">{t('results')}</a>{/if}</div><div class="workspace">
      <section class="input-panel" aria-labelledby="input-heading">
        <div class="panel-heading"><h2 id="input-heading">{t('inputHeading')}</h2><span>{t('inputBadge')}</span></div>
        {#if inputMode === 'photo'}<div class="photo">{#if imageUrl}<img src={imageUrl} alt={imageFile ? t('fileAlt', { name: imageFile.name }) : imageAlt && photoRecording?.image_url === imageUrl ? t('photoRecordAlt') : imageAlt || t('uploadedAlt')} />{:else}<div class="empty-photo"><span aria-hidden="true">↗</span><p>{t('choosePhoto')}</p></div>{/if}</div>
        {#if imageName}<p class="image-name">{imageFile ? imageName : photoRecording?.title === imageName ? t('photoRecordTitle') : imageName}</p>{/if}
        <div class="image-actions"><label class="upload-button" for="photo-upload">{t('upload')}</label><input bind:this={uploadInput} id="photo-upload" type="file" accept="image/jpeg,image/png,image/webp" onchange={upload} disabled={busy} /><button class="secondary" onclick={loadRecording} disabled={!recording || busy}>{t('showRecording')}</button></div>
        <p class="hint">{t('imageHint')}</p>
        {:else}<div class="image-actions"><button class="secondary" onclick={() => { recording = textDirectRecording; loadRecording(); }} disabled={!textDirectRecording || busy}>{t('directRecording')}</button><button class="secondary" onclick={showThinkingRecord} disabled={!textThinkingRecording || busy}>{t('thinkingRecording')}</button></div>{/if}
        {#if recordingError && inputMode === 'photo'}<p class="hint">{t(recordingError)} {t('recordingFallback')}</p>{/if}
        <label class="field-label" for="state">{t('state')} <span>JSON</span></label><textarea id="state" bind:value={stateText} oninput={clearResult} disabled={busy} rows="5" spellcheck="false"></textarea>
        <details class="questions"><summary>{t('editQuestions')} <span>choice / binary / ordinal</span></summary><p class="hint">{t('questionHint')}</p><label class="sr-only" for="decisions">{t('questionsJson')}</label><textarea id="decisions" bind:value={decisionsText} oninput={clearResult} disabled={busy} rows="22" spellcheck="false"></textarea></details>
        <div class="reasoning"><label for="reasoning">{t('reasoning')}</label><select id="reasoning" bind:value={reasoningMode} onchange={clearResult} disabled={busy}><option value="direct">{t('direct')}</option><option value="thinking" disabled={inputMode === 'photo'}>{t('thinking')}</option></select>{#if reasoningMode === 'thinking'}<label for="reasoning-tokens">{t('maxTokens')}</label><input id="reasoning-tokens" type="number" min="1" max="1024" step="1" bind:value={reasoningTokens} oninput={clearResult} disabled={busy} />{/if}<p class="hint">{t('thinkingHint')}</p></div>
        <details class="questions" open><summary>{t('policyHeading')}</summary><label class="field-label" for="error-rate">{t('errorRate')}</label><input class="numeric" id="error-rate" type="number" min="0" max="100" step="1" bind:value={targetErrorRate} oninput={changeErrorRate} disabled={busy} /><p class="hint">{t('errorRateHint')}</p><label class="field-label" for="min-top">{t('minTop')}</label><input class="numeric" id="min-top" type="number" min="0" max="1" step="0.01" bind:value={minTopProbability} oninput={(event) => { targetErrorRate = Number(((1-Number(event.currentTarget.value))*100).toFixed(2)); clearResult(); }} disabled={busy} /><label class="field-label" for="min-mass">{t('minMass')}</label><input class="numeric" id="min-mass" type="number" min="0" max="1" step="0.01" bind:value={minCandidateMass} oninput={clearResult} disabled={busy} /><label class="field-label" for="failure-reasons">{t('customReasons')} <span>JSON</span></label><textarea id="failure-reasons" rows="9" spellcheck="false" bind:value={failureReasonsText} oninput={clearResult} disabled={busy}></textarea></details>
        <div class="run-actions"><button class="primary" onclick={analyze} disabled={(inputMode === 'photo' && !imageUrl) || busy}>{busy ? t('analyzing') : t('run')}</button>{#if busy}<button class="secondary" onclick={() => abort?.abort()}>{t('stop')}</button>{/if}</div>
        <p class="hint">{t('endpointHint')}</p>{#if error}<div class="error" role="alert">{error}</div>{/if}
      </section>
      <section class="output-panel" aria-labelledby="output-heading" aria-busy={busy}>
        <div class="panel-heading"><h2 id="output-heading">{t('outputHeading')}</h2><span>{t('outputBadge')}</span></div>
        <div aria-live="polite">{#if response}
          <div class="provenance"><strong>{mode === 'recorded' ? t('recorded') : t('live')}</strong>{#if mode === 'recorded' && recording}<p><time datetime={recording.recorded_at}>{recording.recorded_at}</time> · {recording.model} · {recording.runtime}</p>{:else}<p>{response.backend.runtime} · {response.backend.model.split('/').at(-1)}</p>{/if}{#if elapsedMs !== undefined}<p>{(elapsedMs / 1000).toFixed(2)}{t('seconds')} · {mode === 'live' ? t('liveTime') : t('recordedTime')}</p>{/if}</div>
          {#each response.results as result (result.id)}
            {@const decision = evaluatedDecisions.find((item) => item.id === result.id)}
            <article class="result-card"><div class="result-heading"><div><span class="type">{result.value.type.toUpperCase()}</span><h3>{displayNames[result.id] ? t(displayNames[result.id]) : result.id}</h3></div><span class="status" class:abstained={result.status === 'abstained'}>{result.status === 'abstained' ? t('abstained') : t('selected')}</span></div><p class="instruction">{decision?.instruction}</p><p class="selected-value">{selection(result, $locale)}</p>{#if result.usage?.reasoning}<p class="thinking-info">{t('reasoningUsage', { mode: result.usage.reasoning.mode, count: result.usage.reasoning.generated_tokens, status: t(result.usage.reasoning.completed ? 'completed' : 'incomplete') })}</p>{/if}
              {#if result.reason_messages?.length}<ul class="reasons">{#each result.reason_messages as message, index (index)}<li>{message.message} <code>{message.code}</code></li>{/each}</ul>{/if}
              {#if result.abstention_reasons?.length}<ul class="reasons">{#each result.abstention_reasons as reason (reason)}<li>{reasonLabel(reason, $locale)} <code>{reason}</code></li>{/each}</ul>{/if}
              {#if result.evidence.scores?.length}<div class="score-list">{#each result.evidence.scores as score (score.id)}<div class="score-row"><span>{score.id}</span><div class="track"><div style:width={`${Math.max(0, Math.min(100, score.option_probability * 100))}%`}></div></div><strong>{percent(score.option_probability, $locale)}</strong></div>{/each}</div>{/if}
              <dl class="metrics"><div><dt>{t('topProbability')}</dt><dd>{percent(result.evidence.top_option_probability, $locale)}</dd></div><div><dt>{t('candidateMass')}</dt><dd>{percent(result.evidence.candidate_mass, $locale)}</dd></div>{#if result.evidence.estimate?.expected_value !== undefined}<div><dt>{t('expectedValue')}</dt><dd>{result.evidence.estimate.expected_value.toFixed(3)}</dd></div>{/if}</dl>
              <details class="evidence"><summary>{t('evidenceJson')}</summary><pre>{JSON.stringify(result, null, 2)}</pre></details>
            </article>
          {/each}
          {#if response.policy}<p class="policy">{t('serverPolicy', { top: percent(response.policy.min_top_probability, $locale), mass: percent(response.policy.min_candidate_mass, $locale) })}</p>{/if}
          {#if response.error_budget}<p class="policy">{t('budget', { rate: percent(response.error_budget.requested_rate, $locale), top: percent(response.error_budget.min_top_probability, $locale) })}</p>{/if}
          <details class="raw"><summary>{t('responseJson')}</summary><pre>{JSON.stringify(response, null, 2)}</pre></details>
        {:else}<div class="empty-output"><span aria-hidden="true">{busy ? '◌' : '↗'}</span><h3>{busy ? t('waiting') : t('empty')}</h3><p>{busy ? t('waitingHint') : t('emptyHint')}</p></div>{/if}</div>
      </section>
    </div>
    <aside class="setup"><h2>{t('setup')}</h2><pre><code>./target/release/l2s1 --model /path/to/vision-model.gguf \
  --mmproj /path/to/mmproj.gguf --device cuda \
  --listen 127.0.0.1:8080

cd web
npm ci
npm run dev</code></pre><a href={`/docs/docs/${$locale}/IMAGE_DEMO.md`} rel="external">{t('fullGuide')}</a></aside>
    {#if recording?.source}<p class="attribution">{t('samplePhoto')} <a href={recording.source.url} rel="noreferrer external" target="_blank">{recording.source.dataset}</a>{#if recording.source.license} · {recording.source.license}{/if}</p>{/if}
  </main><footer><span>{t('footer')}</span><a href={resolve('/')}>{t('back')}</a></footer>
</div>

<style>
  :global(body){margin:0;background:var(--theme-page, #f6f5f0);color:var(--theme-text, #263229);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif} :global(*){box-sizing:border-box} :global(button),:global(textarea){font:inherit} :global(button),:global(a),:global(input),:global(textarea),summary,.upload-button{outline-offset:5px}a{color:inherit}.shell{max-width:1440px;margin:auto;padding:0 48px}footer{display:flex;flex-wrap:wrap;gap:15px;align-items:center;justify-content:space-between;padding:27px 0;border-bottom:1px solid var(--theme-border, #d8ddd4)}.skip{position:absolute;left:15px;top:-60px;background:var(--theme-panel, white);padding:12px;z-index:10}.skip:focus{top:10px}.intro{padding:55px 0 38px}.eyebrow{color:var(--theme-accent, #65773f);font-size:12px;letter-spacing:2px;font-weight:700}h1{font-size:clamp(34px,4vw,55px);line-height:1.25;letter-spacing:-2px;margin:18px 0 23px}em{font-style:normal;color:var(--theme-muted, #6a7e46)}.intro>p:not(.eyebrow){font-size:16px;line-height:1.8;margin:7px 0}.input-tabs{display:flex;flex-wrap:wrap;gap:10px;align-items:center}.results-link{font-size:12px;color:var(--theme-muted, #5c7342);margin-left:auto;white-space:nowrap}.input-tabs button{border:1px solid var(--theme-border, #d5ddcf);background:var(--theme-panel, white);color:var(--theme-text, #334229)}.input-tabs button.active{background:var(--theme-terminal, #314f2b);color:var(--theme-on-primary, white)}.workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.1fr);gap:32px;align-items:start}.input-panel,.output-panel{min-width:0;background:var(--theme-panel, #fff);border:1px solid var(--theme-border, #d8ddd4);border-radius:12px;padding:24px}.panel-heading{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}.panel-heading h2{font-size:18px;margin:0}.panel-heading>span{font-size:10px;letter-spacing:2px;color:var(--theme-muted, #778270)}.photo{background:var(--theme-tint, #f1f3ec);border:1px solid var(--theme-border, #e2e7dc);border-radius:7px;height:295px;display:flex;align-items:center;justify-content:center;overflow:hidden}.photo img{width:100%;height:100%;object-fit:contain}.empty-photo{text-align:center;color:var(--theme-muted, #738367)}.empty-photo>span{font-size:36px}.image-name{font-size:12px;color:var(--theme-muted, #65715e);overflow-wrap:anywhere}.image-actions,.run-actions{display:flex;gap:10px;margin:16px 0 9px;flex-wrap:wrap}button,.upload-button{border-radius:6px;padding:12px 15px;font-size:13px;font-weight:600;cursor:pointer;line-height:1.4}.upload-button,.secondary{border:1px solid var(--theme-border, #d5ddcf);background:var(--theme-panel, white);color:var(--theme-text, #334229)}.primary{border:1px solid var(--theme-border, #314f2b);background:var(--theme-terminal, #314f2b);color:var(--theme-on-primary, white);flex:1}.primary:hover{background:var(--theme-primary, #243e1f)}button:disabled{opacity:.5;cursor:default}#photo-upload{max-width:100%;font-size:11px;align-self:center}input[type=file]::file-selector-button{display:none}.hint{font-size:12px;color:var(--theme-muted, #687563);line-height:1.75;margin:10px 0 18px}.field-label{font-size:13px;font-weight:600;display:flex;justify-content:space-between;margin:24px 0 10px}.field-label span{font-size:11px;color:var(--theme-muted, #748369)}textarea{width:100%;max-width:100%;resize:vertical;padding:13px;border:1px solid var(--theme-border, #d8ddd4);border-radius:6px;background:var(--theme-panel, #fafbf7);font-family:ui-monospace,SFMono-Regular,monospace;font-size:12px;line-height:1.7;color:var(--theme-muted, #33422f)}details{min-width:0}summary{cursor:pointer;font-size:13px;line-height:1.6}summary>span{float:right;font-size:10px;color:var(--theme-muted, #738367)}.questions{margin-top:15px;border:1px solid var(--theme-border, #e0e5da);border-radius:6px;padding:12px}.reasoning{margin-top:22px}.reasoning>label{display:block;font-size:12px;margin:13px 0 7px}.numeric,.reasoning select,.reasoning input{max-width:100%;padding:10px 12px;border:1px solid var(--theme-border, #d8ddd4);border-radius:5px;background:var(--theme-panel, #fafbf7);color:var(--theme-muted, #33422f)}.provenance{background:var(--theme-tint, #eef3e7);border:1px solid var(--theme-border, #dbe6ce);border-radius:7px;padding:15px;margin-bottom:20px}.provenance strong{font-size:13px;color:var(--theme-accent, #476433)}.provenance p{font-size:12px;line-height:1.65;color:var(--theme-muted, #64735b);margin:6px 0 0;overflow-wrap:anywhere}.result-card{border:1px solid var(--theme-border, #dfe5d9);border-radius:8px;margin:16px 0;padding:18px}.result-heading{display:flex;align-items:center;justify-content:space-between;gap:12px}.type{font-size:9px;letter-spacing:1.4px;color:var(--theme-muted, #788970)}h3{font-size:17px;margin:6px 0}.status{background:var(--theme-tint, #eef4e7);color:var(--theme-accent, #46662c);font-size:11px;border-radius:4px;padding:6px 10px}.status.abstained{background:var(--theme-warning-bg, #fbf0dc);color:var(--theme-warning-text, #876029)}.thinking-info{font-size:12px;color:var(--theme-muted, #64735b);line-height:1.8}.instruction{font-size:12px;color:var(--theme-muted, #75816e);line-height:1.65;margin:12px 0;overflow-wrap:anywhere}.selected-value{font-size:24px;font-weight:700;letter-spacing:-.5px;margin:12px 0 20px;overflow-wrap:anywhere}.score-row{display:grid;grid-template-columns:minmax(70px,1fr) minmax(50px,1.4fr) 72px;gap:12px;align-items:center;font-size:11px;margin:10px 0}.score-row>span{overflow-wrap:anywhere}.score-row strong{text-align:right;font-variant-numeric:tabular-nums;font-size:11px}.track{height:6px;background:var(--theme-tint, #ecf0e7);border-radius:3px;overflow:hidden}.track>div{height:100%;background:var(--theme-chart, #7e925b);border-radius:3px}.metrics{display:flex;gap:24px;flex-wrap:wrap;border-top:1px solid var(--theme-border, #e6eadd);margin:20px 0 12px;padding-top:15px}.metrics dt{font-size:10px;color:var(--theme-muted, #79866f)}.metrics dd{margin:5px 0 0;font-size:15px;font-weight:600;font-variant-numeric:tabular-nums}.evidence summary,.raw summary{font-size:11px;color:var(--theme-muted, #6e7d61)}.evidence{margin-top:15px}.reasons{background:var(--theme-warning-bg, #fff8ea);border-radius:5px;padding:12px 12px 12px 28px;font-size:12px;line-height:1.8;color:var(--theme-warning-text, #7c622f)}.reasons code{font-size:10px;display:block;overflow-wrap:anywhere}.policy{font-size:11px;line-height:1.8;color:var(--theme-muted, #6d7c62)}pre{overflow:auto;max-width:100%;background:var(--theme-tint, #f3f5ee);border:1px solid var(--theme-border, #e1e7d9);border-radius:5px;padding:15px;font-size:11px;line-height:1.75;margin-top:12px}pre code{font-size:inherit}.empty-output{min-height:420px;display:flex;flex-direction:column;justify-content:center;align-items:center;text-align:center;padding:35px;color:var(--theme-muted, #728265)}.empty-output>span{font-size:40px}.empty-output h3{font-size:16px;margin-top:25px;line-height:1.7}.empty-output p{font-size:13px;line-height:1.8}.error{font-size:13px;line-height:1.8;border:1px solid var(--theme-error-border, #e8b3a7);background:var(--theme-error-bg, #fff1ed);color:var(--theme-error-text, #8d3b2b);padding:15px;border-radius:6px;overflow-wrap:anywhere}.setup{border-top:1px solid var(--theme-border, #d8ddd4);margin:42px 0 22px;padding:26px 0}.setup h2{font-size:20px}.setup pre{max-width:850px}.setup a{font-size:13px}.attribution{font-size:11px;line-height:1.8;color:var(--theme-muted, #6e7b63);margin-bottom:35px}footer{border-bottom:0;border-top:1px solid var(--theme-border, #d8ddd4);font-size:12px;color:var(--theme-muted, #6b7862);gap:18px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:900px){.shell{padding:0 24px}.workspace{gap:18px}.input-panel,.output-panel{padding:18px}.score-row{gap:8px}.metrics{gap:15px}}@media(max-width:700px){.shell{padding:0 18px}.intro{padding:35px 0 25px}h1{letter-spacing:-1px}.workspace{grid-template-columns:1fr}.input-panel,.output-panel{padding:18px}.photo{height:260px}.image-actions{gap:8px}.image-actions button,.upload-button{font-size:12px;padding:10px 12px}.empty-output{min-height:240px}summary>span{float:none;display:block}.setup pre{font-size:10px}footer{flex-wrap:wrap}.score-row{grid-template-columns:minmax(60px,1fr) minmax(40px,1fr) 68px;gap:8px}}
</style>
