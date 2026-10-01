<script lang="ts">
  import { onMount } from 'svelte';
  import { resolve, asset } from '$app/paths';
  import { locale } from '$lib/i18n';
  import { visibleDetections, type Detection, type DetectionRecording, type DetectorOutput, type Gallery, type Sample, type Improvement } from '$lib/vision/types';
  import baseConfig from '$lib/vision/detector.json';
  import openConfig from '$lib/vision/open-detector.json';
  import MaterialClassifier from '$lib/vision/MaterialClassifier.svelte';
  import '$lib/vision/visual.css';
  const say = (ko: string, en: string, ja: string) => $locale === 'ko' ? ko : $locale === 'ja' ? ja : en;
  let recording = $state<DetectionRecording>();
  let recordings = $state<Partial<Record<'detr' | 'owlvit', DetectionRecording>>>({});
  let engine = $state<'detr' | 'owlvit'>('detr');
  const config = $derived(engine === 'detr' ? baseConfig : openConfig);
  let improvement = $state<Improvement>();
  let samples = $state<Sample[]>([]);
  let current = $state<Sample>();
  const materialResult = $derived(current && improvement?.model.observations[current.name]);
  let imageUrl = $state('');
  let imageName = $state('');
  let imageWidth = $state(512);
  let imageHeight = $state(384);
  let detections = $state<Detection[]>([]);
  let mode = $state<'empty' | 'recorded' | 'live'>('empty');
  let threshold = $state(0.5);
  let showBoxes = $state(true);
  let activeBox = $state(-1);
  let busy = $state(false);
  let loading = $state(true);
  let phase = $state<'loading' | 'running'>('loading');
  let progress = $state('');
  let error = $state('');
  let elapsedMs = $state<number>();
  let worker: Worker | undefined;
  let timeout: ReturnType<typeof setTimeout> | undefined;
  let uploadVersion = 0;
  const visible = $derived(visibleDetections(detections, threshold));
  const recordedSample = $derived(recording?.samples.find((sample) => sample.name === current?.name));
  const objectName = (label: string) => ({ bottle: say('병','bottle','ボトル'), bowl: say('그릇','bowl','ボウル'), vase: say('꽃병','vase','花瓶'), cup: say('컵','cup','カップ'), backpack: say('배낭','backpack','リュック'), refrigerator: say('냉장고','refrigerator','冷蔵庫'), banana: say('바나나','banana','バナナ'), cake: say('케이크','cake','ケーキ'), book: say('책','book','本') }[label] ?? label);
  function clear() { detections = []; mode = 'empty'; elapsedMs = undefined; error = ''; activeBox = -1; }
  function changeEngine() { stop(); recording = recordings[engine]; threshold = engine === 'detr' ? 0.5 : openConfig.display_score; if (current) choose(current); else clear(); }
  function releaseUrl() { if (imageUrl.startsWith('blob:')) URL.revokeObjectURL(imageUrl); }
  function choose(sample: Sample) {
    uploadVersion++; releaseUrl(); clear(); current = sample;
    imageName = sample.name.split('/').at(-1) ?? sample.name; imageUrl = asset(sample.image_url); imageWidth = 512; imageHeight = 384;
    const saved = recording?.samples.find((item) => item.name === sample.name);
    if (saved) { detections = saved.detections; elapsedMs = saved.elapsed_ms; imageWidth = saved.width; imageHeight = saved.height; mode = 'recorded'; }
  }
  async function upload(event: Event) {
    const input = event.currentTarget as HTMLInputElement;
    const file = input.files?.[0]; input.value = '';
    if (!file) return;
    const version = ++uploadVersion;
    if (!['image/jpeg','image/png','image/webp'].includes(file.type) || file.size > 8 * 1024 * 1024) {
      error = say('JPEG·PNG·WebP 사진을 8 MiB 이하로 선택하세요.','Choose a JPEG, PNG or WebP image up to 8 MiB.','8 MiB以下のJPEG・PNG・WebP画像を選んでください。'); return;
    }
    const url = URL.createObjectURL(file);
    try {
      const image = new Image(); image.src = url; await image.decode();
      if (version !== uploadVersion) { URL.revokeObjectURL(url); return; }
      if (image.naturalWidth * image.naturalHeight > 24_000_000) throw new Error('size');
      releaseUrl(); clear(); current = undefined; imageUrl = url; imageName = file.name;
      imageWidth = image.naturalWidth; imageHeight = image.naturalHeight;
    } catch {
      URL.revokeObjectURL(url);
      if (version === uploadVersion) error = say('이미지를 읽을 수 없거나 2,400만 화소를 초과합니다.','The image cannot be decoded or exceeds 24 megapixels.','画像を読み込めないか、2400万画素を超えています。');
    }
  }
  function stop() { worker?.terminate(); worker = undefined; clearTimeout(timeout); busy = false; progress = ''; }
  function run() {
    if (!imageUrl || busy || engine !== 'detr') return;
    clear(); busy = true; phase = 'loading'; progress = '';
    try {
      worker ??= new Worker(new URL('../../lib/vision/worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (event: MessageEvent<DetectorOutput>) => {
        const message = event.data;
        if (message.type === 'progress') progress = `${message.file} · ${Math.round(message.progress)}%`;
        if (message.type === 'ready') { phase = 'running'; progress = ''; }
        if (message.type === 'result') { detections = message.detections; elapsedMs = message.elapsed_ms; mode = 'live'; busy = false; clearTimeout(timeout); }
        if (message.type === 'error') { stop(); error = say('탐지 실행 실패: ','Detection failed: ','検出に失敗しました：') + message.message; }
      };
      worker.onerror = () => { stop(); error = say('탐지 모델을 시작하지 못했습니다. 네트워크를 확인하고 다시 실행하세요.','Could not start the detector. Check your network and retry.','検出モデルを起動できませんでした。ネットワークを確認して再試行してください。'); };
      timeout = setTimeout(() => { stop(); error = say('3분 제한을 초과했습니다. 다시 실행하거나 기록 예제를 선택하세요.','The 3-minute limit was reached. Retry or select a recorded example.','3分の制限を超えました。再試行するか記録サンプルを選択してください。'); }, 180_000);
      worker.postMessage({ type: 'detect', image: imageUrl });
    } catch (cause) { stop(); error = String(cause); }
  }
  function download() {
    const blob = new Blob([JSON.stringify({ model: config.model, revision: config.revision, mode, image: imageName, width: imageWidth, height: imageHeight, threshold, coordinates: 'normalized_xyxy', detections: visible }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob); const link = document.createElement('a'); link.href = url; link.download = 'l2s1-detect.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  onMount(() => {
    let mounted = true;
    Promise.all([fetch(asset('/trashnet/detections.json')), fetch(asset('/trashnet/recorded.json')), fetch(asset('/trashnet/open-detections.json')), fetch(asset('/trashnet/improvement.json'))]).then(async ([detectionResponse, galleryResponse, openResponse, improvedResponse]) => {
      if (!detectionResponse.ok || !galleryResponse.ok || !openResponse.ok || !improvedResponse.ok) throw new Error('recordings');
      const saved: DetectionRecording = await detectionResponse.json(); const gallery: Gallery = await galleryResponse.json();
      const openSaved: DetectionRecording = await openResponse.json(); const improved: Improvement = await improvedResponse.json();
      if (!mounted) return;
      recordings = { detr: saved, owlvit: openSaved }; recording = saved; improvement = improved; samples = [...gallery.source.records, ...improved.test_records];
      const requested = new URL(window.location.href).searchParams.get('sample') ?? 'glass330.jpg';
      // Only resolve a filename against the bundled manifest; never fetch an arbitrary query URL.
      const initial = samples.find((sample) => sample.name.split('/').at(-1) === requested) ?? samples[0];
      choose(initial);
    }).catch(() => { if (mounted) error = say('기록 예제를 불러오지 못했습니다. 사진을 업로드해 탐지할 수 있습니다.','Recorded examples could not load. You can still upload a photo and run detection.','記録サンプルを読み込めませんでした。写真のアップロードで検出できます。'); }).finally(() => { if (mounted) loading = false; });
    return () => { mounted = false; uploadVersion++; stop(); releaseUrl(); };
  });
</script>
<svelte:head><title>Detect · {say('사진 속 객체 탐지','Visual object detection','写真の物体検出')} · L2S1</title><meta name="description" content="Inspect actual DETR object detections on TrashNet images with bounding boxes, or detect objects in your own photos locally in the browser." /></svelte:head>
<main class="visual-page">
  <p class="eyebrow">VISION LAB / 02 — DETECT</p>
  <h1>{say('물체가 있는 곳을, 박스로.','Objects, located.','物体の位置を、ボックスで。')}</h1>
  <p class="lead">{say('사진 위에 실제 탐지 위치와 물체 이름을 표시합니다. 예제를 살펴보거나 내 사진을 업로드해 브라우저에서 직접 탐지해 보세요.','See actual object locations and labels on the image. Explore an example or upload your own photo and run detection in your browser.','写真上に実際の検出位置と物体名を表示します。サンプルを確認したり、自分の写真をブラウザーで検出できます。')}</p>
  <nav class="tabs" aria-label={say('시각 예제','Visual examples','視覚サンプル')}><a href={resolve('/trashnet')}>TrashNet</a><a href={resolve('/detect')} aria-current="page">Detect</a><a href={resolve('/demo')}>{say('이미지 판단 실험','Image playground','画像判断')}</a></nav>
  <div class="provenance">{say('DETR 객체 탐지 모델이 COCO의 물체 종류와 위치를 찾습니다. L2S1의 재질 분류와는 별도 예제입니다. TrashNet에는 박스 정답이 없어 위치 정확도를 평가하지 않으며, 높은 점수의 오탐도 있습니다.','The DETR detector finds COCO object categories and locations. This is separate from L2S1 material classification. TrashNet has no bounding-box annotations here, so localization accuracy is not evaluated; high-scoring false detections can occur.','DETRがCOCOの物体カテゴリと位置を検出します。L2S1の素材分類とは別のサンプルです。TrashNetに正解ボックスはないため位置精度は評価せず、高スコアの誤検出もあります。')}</div>
  <div class="controls"><label>{say('탐지 모델 비교','Detection model','検出モデル')}<select aria-label={say('탐지 모델 비교','Detection model','検出モデル')} bind:value={engine} onchange={changeEngine} disabled={busy || loading}><option value="detr">DETR · COCO</option><option value="owlvit">OWL-ViT · {say('쓰레기 어휘 · 기록 비교','waste vocabulary · recordings','ごみ語彙・記録比較')}</option></select></label></div>
  {#if engine === 'owlvit'}<p class="provenance">{say('15개 쓰레기 관련 표현으로 실제 탐지하고 겹치는 박스를 줄인 실험입니다. 오탐이 남아 있으며 위치 정확도 향상을 입증한 결과는 아닙니다. 점수는 DETR과 직접 비교할 수 없습니다. 브라우저 실행 검증을 통과하지 않아 CPU 실행 기록만 제공합니다.','An experiment using 15 waste-related prompts and overlap suppression. False detections remain; this does not establish improved localization accuracy. Scores are not directly comparable with DETR. Browser execution did not pass verification, so only CPU recordings are available.','15個のごみ関連表現と重複抑制による実験です。誤検出が残り、位置精度の改善を実証していません。スコアはDETRと直接比較できません。ブラウザー実行は検証に通らなかったためCPUの記録のみ提供します。')}</p>{/if}
  {#if materialResult}<div class="panel material-decision"><h2>{say('개선 모델의 사진 전체 재질 판단','Trained classifier: whole-image material','学習済み分類器：画像全体の素材')}</h2><p><strong>{materialResult.selected}</strong> · {say('실행 기록','Recorded inference','実行記録')} · SigLIP2</p><p class="muted">{say('사진 전체를 분류한 결과입니다. 아래 각 박스의 재질을 판정한 결과는 아닙니다.','This classifies the whole photo; it does not assign a material to each box below.','写真全体の分類です。下の各ボックスの素材を判定した結果ではありません。')}</p></div>{/if}
  <div class="split">
    <section class="panel">
      <div class="controls image-controls">
        <label class="sample-select">{say('TrashNet 사진','TrashNet photo','TrashNetの写真')}<select aria-label={say('TrashNet 사진','TrashNet photo','TrashNetの写真')} value={current?.name ?? ''} disabled={busy || loading || !samples.length} onchange={(event) => { const sample = samples.find((item) => item.name === event.currentTarget.value); if (sample) choose(sample); }}><option value="" disabled>{say('사진 선택','Choose a photo','写真を選択')}</option>{#each samples as sample (sample.name)}<option value={sample.name}>{sample.name.split('/').at(-1)} · {sample.label}{recording?.samples.some((item) => item.name === sample.name) ? ' ●' : ''}</option>{/each}</select></label>
        <label class="upload-label">{say('내 사진 업로드','Upload a photo','写真をアップロード')}<input type="file" accept="image/jpeg,image/png,image/webp" onchange={upload} disabled={busy || loading} /></label>
      </div>
      <div class="canvas" aria-label={say('탐지 박스가 표시된 사진','Photo with detection boxes','検出ボックス付きの写真')}>
        {#if imageUrl}<img src={imageUrl} alt={imageName} width={imageWidth} height={imageHeight} />
          {#if showBoxes}<svg class="overlay" viewBox="0 0 100 100" preserveAspectRatio="none" role="img" aria-label={say(`${visible.length}개 탐지 박스`,`${visible.length} detection boxes`,`${visible.length}個の検出ボックス`)}>
            {#each visible as item, index (index)}<rect x={item.box.xmin * 100} y={item.box.ymin * 100} width={(item.box.xmax - item.box.xmin) * 100} height={(item.box.ymax - item.box.ymin) * 100} class:highlighted={activeBox === index} />{/each}
          </svg>
          {#each visible as item, index (index)}<span class="box-label" class:highlighted={activeBox === index} style:left={`${Math.min(75, item.box.xmin * 100)}%`} style:top={`${Math.min(92, item.box.ymin * 100)}%`}>{index + 1}. {objectName(item.label)} {(item.score * 100).toFixed(0)}%</span>{/each}{/if}
        {:else}<div class="empty">{loading ? say('예제를 불러오는 중…','Loading examples…','サンプルを読み込み中…') : say('사진을 선택하세요.','Choose a photo.','写真を選択してください。')}</div>{/if}
      </div>
      <p class="muted filename">{imageName} {#if imageUrl}· {imageWidth} × {imageHeight}{/if}{#if current} · {say('데이터셋 재질','Dataset material','データセットの素材')}: {current.label}{/if}</p>
      <div class="display-controls"><label><input type="checkbox" bind:checked={showBoxes} /> {say('박스 표시','Show boxes','ボックス表示')}</label><label>{say('표시할 최소 점수','Minimum display score','表示する最小スコア')} <strong>{threshold.toFixed(2)}</strong><input aria-label={say('표시할 최소 점수','Minimum display score','表示する最小スコア')} type="range" min={config.minimum_score} max="0.99" step="0.01" bind:value={threshold} oninput={() => activeBox = -1} /></label></div>
      <p class="muted">{say('점수 조절은 현재 탐지 결과에서 표시할 박스만 걸러냅니다. 모델을 다시 실행하지 않습니다.','The slider filters boxes from the current detections. It does not rerun the model.','スライダーは検出済みのボックスを絞り込みます。モデルは再実行しません。')}</p>
      <div class="run-controls"><button class="primary" onclick={run} disabled={!imageUrl || busy || loading || engine !== 'detr'}>{busy ? say('탐지 중…','Detecting…','検出中…') : say('이 사진 직접 탐지','Run detection on this photo','この写真を検出')}</button>{#if busy}<button onclick={stop}>{say('중지','Stop','停止')}</button>{:else if recordedSample}<button onclick={() => { if (current) choose(current); }}>{say('실제 실행 기록 보기','Show recorded inference','実行記録を表示')}</button>{/if}</div>
      <p class="muted">{say('DETR 탐지는 이 브라우저에서 실행됩니다. 첫 모델 다운로드 약 43 MB + 실행 파일. OWL-ViT는 기록 비교만 제공합니다. JPEG·PNG·WebP, 최대 8 MiB.','DETR runs in this browser; first download ~43 MB plus runtime. OWL-ViT is available as recorded comparisons only. JPEG, PNG or WebP, up to 8 MiB.','DETRはブラウザーで実行します（初回約43 MBとランタイム）。OWL-ViTは記録比較のみです。JPEG・PNG・WebP、最大8 MiB。')}</p>
      {#if error}<p class="error" role="alert">{error}</p>{/if}
    </section>
    <section class="panel results" aria-busy={busy}>
      <h2>{say('탐지된 물체','Detected objects','検出された物体')} <span class="badge">{visible.length}</span></h2>
      <div aria-live="polite">
        {#if busy}<p role="status">{phase === 'loading' ? say('모델 다운로드·준비 중…','Downloading / preparing the model…','モデルをダウンロード・準備中…') : say('사진에서 물체를 찾고 있습니다…','Finding objects in the photo…','写真から物体を検出中…')}</p><p class="muted progress">{progress}</p>
        {:else if mode !== 'empty'}<p class="badge">{mode === 'recorded' ? say('실제 모델 실행 기록','Recorded model inference','実際のモデル実行記録') : say('방금 실행한 브라우저 추론','Just run in this browser','このブラウザーで実行した推論')}</p><p class="muted">{mode === 'recorded' ? recording?.runtime : 'Transformers.js / ONNX Runtime / WASM'}<br />{mode === 'recorded' ? recording?.recorded_at : ''}{#if elapsedMs !== undefined} · {(elapsedMs / 1000).toFixed(2)} s{/if}</p>
          {#each visible as item, index (index)}<button class="object-row" class:active={activeBox === index} aria-pressed={activeBox === index} onmouseenter={() => activeBox = index} onmouseleave={() => activeBox = -1} onfocus={() => activeBox = index} onblur={() => activeBox = -1} onclick={() => activeBox = index}><span class="object-index">{index + 1}</span><span><strong>{objectName(item.label)}</strong><small>{item.label} · x {Math.round(item.box.xmin * imageWidth)}–{Math.round(item.box.xmax * imageWidth)} / y {Math.round(item.box.ymin * imageHeight)}–{Math.round(item.box.ymax * imageHeight)} px</small></span><b>{(item.score * 100).toFixed(1)}%</b></button>
          {:else}<p class="empty">{say('이 점수 기준에서 탐지된 물체가 없습니다. 물체가 없다는 뜻은 아닙니다.','No objects detected at this threshold. This does not establish that the image contains no objects.','このスコア基準で検出された物体はありません。物体が存在しないという意味ではありません。')}</p>{/each}
          <p class="muted">{say('점수는 탐지 모델의 점수이며, 정답 확률이나 TrashNet 재질 분류 점수가 아닙니다.','Scores come from the detector, not calibrated correctness probabilities or TrashNet material scores.','スコアは検出モデルの値であり、正解確率やTrashNetの素材スコアではありません。')}</p>
          <button onclick={download}>{say('박스 JSON 저장','Download boxes as JSON','ボックスJSONを保存')}</button><details><summary>{say('좌표와 원본 결과','Coordinates and raw results','座標と元の結果')}</summary><pre>{JSON.stringify({ coordinates: 'normalized_xyxy', detections: visible }, null, 2)}</pre></details>
        {:else}<p class="empty">{say('이 사진의 탐지 기록은 없습니다. DETR 모델을 선택하면 직접 탐지할 수 있습니다.','No detection is recorded for this photo. Select DETR to run detection.','この写真の検出記録はありません。DETRを選択すると検出できます。')}</p>{/if}
      </div>
      {#if recording}<h3 class="example-heading">{say('바로 볼 수 있는 예제','Ready-to-view examples','すぐに見られるサンプル')}</h3><div class="sample-grid">{#each recording.samples as sample (sample.name)}<button onclick={() => choose(sample)} disabled={busy} aria-label={sample.name.split('/').at(-1)} aria-pressed={current?.name === sample.name}><img src={asset(sample.image_url)} alt={sample.label} width="512" height="384" loading="lazy" /><span>{sample.name.split('/').at(-1)}</span></button>{/each}</div>{/if}
    </section>
  </div>
  <MaterialClassifier sample={current} />
  <footer class="footer"><a href={`https://huggingface.co/${config.model}/tree/${config.revision}`} target="_blank" rel="noreferrer">{config.model}</a> · <a href={asset('/trashnet/THIRD_PARTY_NOTICE.txt')} rel="external">TrashNet / {say('이미지 출처','Image credits','画像の出典')}</a> · <a href={asset(engine === 'detr' ? '/trashnet/detections.json' : '/trashnet/open-detections.json')} download>{say('탐지 실행 기록','Detection recording','検出実行記録')}</a><br />{say('각 모델의 12장 예제는 고정 버전 Q8 CPU 추론 결과입니다. 새 사진은 WASM으로 실행되므로 기기와 실행 환경에 따라 점수와 시간이 달라질 수 있습니다.','Each model has 12 examples recorded with pinned Q8 CPU inference. New photos run with WASM; scores and timings may differ across devices and runtimes.','各モデルの12枚のサンプルは固定版Q8のCPU推論記録です。新しい写真はWASMで実行するため、機器や環境でスコアと時間が変わる場合があります。')}</footer>
</main>
<style>
  .material-decision{margin-bottom:20px}.image-controls{margin-top:0!important}.sample-select{flex:1;min-width:180px}.sample-select select{width:100%}.upload-label{max-width:100%}.upload-label input{display:block;max-width:240px;margin-top:8px;font-size:12px}.canvas{position:relative;background:#f5f5f3;border-radius:8px;overflow:hidden;isolation:isolate}.canvas>img{display:block;width:100%;height:auto}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.overlay rect{fill:#1cbb9910;stroke:#007b65;stroke-width:2.5;vector-effect:non-scaling-stroke}.overlay rect.highlighted{fill:#f6b93430;stroke:#c37500;stroke-width:4}.box-label{position:absolute;max-width:95%;background:#005f4f;color:white;font-size:11px;font-weight:700;padding:4px 6px;border-radius:3px;pointer-events:none;white-space:nowrap}.box-label.highlighted{background:#7a4800}.filename{overflow-wrap:anywhere}.display-controls{display:flex;flex-wrap:wrap;gap:20px;align-items:center;margin:22px 0 12px}.display-controls>label:last-child{flex:1}.display-controls input[type=range]{display:block;width:100%;margin-top:10px;accent-color:var(--theme-accent)}.display-controls input[type=checkbox]{accent-color:var(--theme-accent)}.display-controls strong{float:right;margin-left:10px}.run-controls{display:flex;flex-wrap:wrap;gap:10px}.results>h2{display:flex;justify-content:space-between;align-items:center}.object-row{width:100%;display:grid;grid-template-columns:26px 1fr auto;gap:12px;align-items:center;text-align:left;margin:10px 0;padding:13px!important}.object-row.active{outline:2px solid var(--theme-chart)}.object-index{font-size:12px;font-weight:bold}.object-row strong{font-size:15px}.object-row small{display:block;font-size:10px;margin-top:6px;color:var(--theme-muted)}.object-row b{font-size:12px}.example-heading{border-top:1px solid var(--theme-border);padding-top:22px;margin-top:25px}.sample-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:9px}.sample-grid button{padding:0!important;overflow:hidden}.sample-grid button[aria-pressed=true]{outline:2px solid var(--theme-chart);outline-offset:2px}.sample-grid img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:cover}.sample-grid span{display:block;font-size:9px;margin:7px 4px;overflow-wrap:anywhere}.progress{overflow-wrap:anywhere}@media(max-width:500px){.box-label{font-size:9px;padding:3px}.object-row{gap:7px}}
</style>
