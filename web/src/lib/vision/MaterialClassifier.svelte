<script lang="ts">
  import { onMount } from 'svelte';
  import { asset } from '$app/paths';
  import { locale } from '$lib/i18n';
  import type { Sample } from './types';
  let { sample }: { sample?: Sample } = $props();
  const say = (ko: string, en: string, ja: string) => $locale === 'ko' ? ko : $locale === 'ja' ? ja : en;
  const material = (id: string) => ({ cardboard: say('골판지','Cardboard','段ボール'), glass: say('유리','Glass','ガラス'), metal: say('금속','Metal','金属'), paper: say('종이','Paper','紙'), plastic: say('플라스틱','Plastic','プラスチック'), trash: say('일반 쓰레기','Trash','その他ごみ') }[id] ?? id);
  let available = $state(false);
  let busy = $state(false);
  let error = $state('');
  let preview = $state('');
  let filename = $state('');
  let result = $state<{ selected: string; latency_ms: number; scores: { id: string; option_probability: number }[] }>();
  let controller: AbortController | undefined;
  function release() { if (preview.startsWith('blob:')) URL.revokeObjectURL(preview); preview = ''; }
  function cancel() { controller?.abort(); controller = undefined; busy = false; }
  async function classify(blob: Blob, name: string) {
    cancel(); release(); result = undefined; error = ''; filename = name;
    if (!['image/jpeg','image/png','image/webp'].includes(blob.type) || blob.size > 8 * 1024 * 1024) { error = say('JPEG·PNG·WebP, 최대 8 MiB입니다.','Choose JPEG, PNG or WebP, up to 8 MiB.','JPEG・PNG・WebP、最大8 MiBです。'); return; }
    preview = URL.createObjectURL(blob); busy = true;
    const active = new AbortController(); controller = active;
    const timer = setTimeout(() => active.abort(), 60_000);
    try {
      const response = await fetch('/vision-inference/classify', { method: 'POST', headers: { 'Content-Type': blob.type }, body: blob, signal: active.signal });
      if (!response.ok) throw new Error((await response.json()).detail ?? `HTTP ${response.status}`);
      const payload = await response.json();
      if (controller === active) result = payload;
    } catch (cause) {
      if (controller === active) error = active.signal.aborted ? say('실행 시간이 초과되었습니다.','The request timed out.','時間制限を超えました。') : String(cause);
    } finally { clearTimeout(timer); if (controller === active) { busy = false; controller = undefined; } }
  }
  async function runSample() {
    if (!sample) return;
    error = '';
    try {
      const response = await fetch(asset(sample.image_url));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      await classify(await response.blob(), sample.name.split('/').at(-1) ?? sample.name);
    } catch (cause) { error = String(cause); }
  }
  onMount(() => {
    let mounted = true;
    const health = new AbortController();
    fetch('/vision-inference/health', { signal: health.signal }).then(async response => { if (response.ok && mounted) available = (await response.json()).ready === true; }).catch(() => {});
    return () => { mounted = false; health.abort(); cancel(); release(); };
  });
</script>
<section class="panel live-classifier" aria-label={say('개선 모델 직접 실행','Try the trained classifier','学習済みモデルを実行')}>
  <h2>{say('개선 모델에 새 사진 넣기','Try a new photo','新しい写真で試す')}</h2>
  <p class="muted">{say('SigLIP2 + 학습한 재질 분류기 · 이 컴퓨터의 모델로 실제 실행합니다. 사진 전체의 대표 분류이며, 여러 물체 각각의 재질을 판정하지 않습니다.','SigLIP2 + trained material classifier, running on this computer. It classifies the main item in the whole image, not each object separately.','SigLIP2と学習済み素材分類器をこのコンピューターで実行します。画像全体の主な物体を分類します。')}</p>
  {#if available}
    <div class="controls">
      <label>{say('내 사진 선택','Upload a photo for classification','分類する写真を選択')}<input type="file" accept="image/jpeg,image/png,image/webp" disabled={busy} onchange={event => { const file = event.currentTarget.files?.[0]; event.currentTarget.value = ''; if (file) void classify(file, file.name); }} /></label>
      {#if sample}<button disabled={busy} onclick={runSample}>{say('선택한 예제로 실행','Classify selected sample','選択したサンプルを分類')}</button>{/if}
      {#if busy}<button onclick={cancel}>{say('중지','Stop','停止')}</button>{/if}
    </div>
  {:else}<p class="muted">{say('새 사진 실행은 로컬 분류 서버가 연결되면 사용할 수 있습니다. 위의 비교 결과는 서버 없이도 볼 수 있습니다.','New photos require the local classifier server. Recorded comparisons remain available without it.','新しい写真の分類にはローカルサーバーが必要です。記録の比較はサーバーなしで表示できます。')}</p>{/if}
  {#if preview}<img class="upload-preview" src={preview} alt={filename} />{/if}
  {#if busy}<p role="status">{say('실제 모델로 분류 중…','Running the classifier…','分類中…')}</p>{/if}
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  {#if result}<div class="live-result"><h3>{material(result.selected)}</h3><p class="muted">{filename} · {result.latency_ms.toFixed(1)} ms</p>
    {#each [...result.scores].sort((a,b) => b.option_probability - a.option_probability) as score (score.id)}<div class="bar-row"><span>{material(score.id)}</span><div class="track"><span style:width={`${score.option_probability * 100}%`}></span></div><strong>{(score.option_probability * 100).toFixed(1)}%</strong></div>{/each}
    <p class="muted">{say('보류 없이 가장 높은 점수를 선택합니다. 점수는 정답일 확률이 아닙니다.','Forced top-1; scores are not probabilities of correctness.','保留なしで首位を選びます。スコアは正解確率ではありません。')}</p>
  </div>{/if}
</section>
<style>.live-classifier{margin-top:24px}.upload-preview{width:100%;max-width:360px;max-height:260px;object-fit:contain;border-radius:8px}.live-result{max-width:520px}</style>
