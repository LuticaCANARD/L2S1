<script lang="ts">
  import { onMount, tick } from 'svelte';
  import { resolve, asset } from '$app/paths';
  import { locale } from '$lib/i18n';
  import { outcome, type Gallery, type Sample, type Improvement } from '$lib/vision/types';
  import MaterialClassifier from '$lib/vision/MaterialClassifier.svelte';
  import '$lib/vision/visual.css';
  const say = (ko: string, en: string, ja: string) => $locale === 'ko' ? ko : $locale === 'ja' ? ja : en;
  const material = (id: string | null) => id === null ? say('보류', 'Abstained', '保留') : ({ cardboard: say('골판지','Cardboard','段ボール'), glass: say('유리','Glass','ガラス'), metal: say('금속','Metal','金属'), paper: say('종이','Paper','紙'), plastic: say('플라스틱','Plastic','プラスチック'), trash: say('일반 쓰레기','Trash','その他ごみ') }[id] ?? id);
  const statusLabel = (status: string) => status === 'correct' ? say('정답','Correct','正解') : status === 'wrong' ? say('오답','Wrong','不正解') : say('보류','Abstained','保留');
  let data = $state<Gallery>();
  let failed = $state(false);
  let modelId = $state('siglip2-trained');
  let improvement = $state<Improvement>();
  let dataset = $state('gallery');
  let category = $state('all');
  let status = $state('all');
  let selected = $state<Sample>();
  let pending = $state(true);
  const model = $derived(data?.models.find((item) => item.id === modelId));
  const displayedSamples = $derived(dataset === 'test' ? improvement?.test_records ?? [] : data?.source.records ?? []);
  const total = $derived(displayedSamples.length);
  const rows = $derived(displayedSamples.map(s => model?.observations[s.name]).filter((r): r is NonNullable<typeof r> => !!r));
  const accepted = $derived(rows.filter((row) => row.selected !== null));
  const correct = $derived(rows.filter((row) => outcome(row) === 'correct').length);
  const rawCorrect = $derived(rows.filter((row) => row.raw_top1 === row.ground_truth).length);
  const filtered = $derived(displayedSamples.filter((sample) => (category === 'all' || category === sample.label) && (status === 'all' || (model && outcome(model.observations[sample.name]) === status))));
  const detail = $derived(selected && model?.observations[selected.name]);
  const fixed = $derived(data?.source.records.filter(sample => {
    const old = data?.models.find(m => m.id === 'qwen3vl')?.observations[sample.name];
    const next = improvement?.model.observations[sample.name];
    return old && next && old.raw_top1 !== sample.label && next.raw_top1 === sample.label;
  }).length ?? 0);
  async function choose(sample: Sample) { selected = sample; await tick(); document.getElementById('photo-detail')?.scrollIntoView({ block: 'start' }); }
  function updateFilters() { selected = undefined; }
  function changeDataset() { if (dataset === 'test') modelId = 'siglip2-trained'; selected = undefined; category = 'all'; status = 'all'; }
  async function load() {
    pending = true; failed = false;
    try {
      const [result, updated] = await Promise.all([fetch(asset('/trashnet/recorded.json')), fetch(asset('/trashnet/improvement.json'))]);
      if (!result.ok || !updated.ok) throw new Error('recording');
      data = await result.json() as Gallery;
      improvement = await updated.json() as Improvement;
      data.models.unshift(improvement.model);
      selected = data.source.records[0];
    } catch { failed = true; } finally { pending = false; }
  }
  onMount(() => { void load(); });
</script>
<svelte:head><title>TrashNet · {say('사진으로 보는 분류','Visual classification','写真で見る分類')} · L2S1</title><meta name="description" content="Explore 120 TrashNet photos with real L2S1 classification results, abstentions, and per-class comparisons across three models." /></svelte:head>
<main class="visual-page">
  <p class="eyebrow">VISION LAB / 01 — TRASHNET</p>
  <h1>{say('사진을 보고, 판단을 비교하세요.','See the image. Compare the decision.','写真を見て、判断を比べる。')}</h1>
  <p class="lead">{say('유리병부터 구겨진 종이까지. 120장의 실제 사진에서 모델이 무엇을 맞히고, 틀리고, 보류했는지 직접 살펴보세요.','From glass bottles to crumpled paper. Explore what each model gets right, gets wrong, or abstains on across 120 real photographs.','ガラス瓶から丸めた紙まで。120枚の実際の写真で、各モデルの正解・誤り・保留を確認できます。')}</p>
  <nav class="tabs" aria-label={say('시각 예제','Visual examples','視覚サンプル')}><a href={resolve('/trashnet')} aria-current="page">TrashNet</a><a href={resolve('/detect')}>Detect ↗</a><a href={resolve('/demo')}>{say('이미지 판단 실험','Image playground','画像判断')}</a></nav>
  {#if pending}<p role="status">{say('사진과 실행 기록을 불러오는 중…','Loading photos and recorded results…','写真と実行記録を読み込み中…')}</p>
  {:else if failed}<div class="error" role="alert">{say('기록을 불러오지 못했습니다.','Could not load the recording.','記録を読み込めませんでした。')} <button onclick={load}>{say('다시 시도','Retry','再試行')}</button></div>
  {:else if data && model}
    {#if improvement}
      <section class="panel improvement" aria-label={say('정확도 개선 결과','Accuracy improvement','精度改善結果')}>
        <h2>{say('같은 사진, 개선된 판단','Same photos. Better decisions.','同じ写真、改善された判断。')}</h2>
        <div class="stats">
          <div class="stat"><span>{say('기존 Qwen · 보류 전 1순위','Original Qwen · raw top-1','従来Qwen・首位正解')}</span><strong>79.2%</strong><span>95 / 120</span></div>
          <div class="stat"><span>{say('학습한 분류기 · 같은 120장','Trained classifier · same 120','学習済み分類器・同じ120枚')}</span><strong>{(improvement.report.evaluations.gallery.accuracy * 100).toFixed(1)}%</strong><span>{improvement.report.evaluations.gallery.correct} / 120 · +{((improvement.report.evaluations.gallery.accuracy - 95/120)*100).toFixed(1)}%p</span></div>
          <div class="stat"><span>{say('별도 평가 사진 · 학습 제외','Separate test set · never trained on','独立評価・学習対象外')}</span><strong>{(improvement.report.evaluations.test.accuracy * 100).toFixed(1)}%</strong><span>{improvement.report.evaluations.test.correct} / {improvement.report.evaluations.test.total}</span></div>
          <div class="stat"><span>{say('일반 쓰레기 정답 · 같은 20장','Trash correct · same 20 photos','その他ごみ正解・同じ20枚')}</span><strong>0 → {improvement.report.evaluations.gallery.confusion_matrix[5][5]}</strong><span>{say('전체에서 고친 오답','Previously wrong, now correct','全体の修正した誤答')}: {fixed}</span></div>
        </div>
        <p class="muted">{say('SigLIP2의 이미지 특징에 재질 분류기를 학습했습니다. L2S1 기본 모델과는 별도 모델이며, 학습한 모델과 무학습 모델의 비교입니다. 개선 모델은 보류 없이 1순위를 선택합니다.','A material classifier trained on SigLIP2 image features. This is a separate supervised model compared with zero-shot L2S1 baselines. The trained model always selects its top choice.','SigLIP2の画像特徴で素材分類器を学習しました。L2S1のゼロショットモデルとは別の教師ありモデルで、保留せず首位を選びます。')}</p>
        <details><summary>{say('학습·평가 분리와 재질별 성능','Training separation and per-class results','学習・評価の分離とクラス別結果')}</summary>
          <p class="muted">{say('학습 1,441장 / 검증 481장 / 최종 평가 481장. 기존 예제 120장과 유사 이미지 4장은 학습·모델 선택에서 제외했습니다. 8개 설정 중 검증 재질별 평균 정확도로 선택했습니다. 동일 데이터셋 내부 평가이며 실제 현장 성능은 별도 검증이 필요합니다.','1,441 training / 481 validation / 481 final test images. The 120 historical examples and 4 similar images were excluded from training and selection. Eight candidates were selected by validation balanced accuracy. This is an internal dataset evaluation, not field performance.','学習1,441枚／検証481枚／最終評価481枚。従来の120枚と類似画像4枚は学習と選択から除外。8候補から検証のクラス平均精度で選択しました。実環境の精度とは異なります。')}</p>
          <div class="table-scroll"><table><thead><tr><th>{say('재질','Material','素材')}</th><th>Qwen / 20</th><th>SigLIP2 / 20</th><th>{say('별도 평가 정답','Separate test correct','独立評価の正解')}</th></tr></thead><tbody>{#each data.source.classes as label, i (label)}<tr><th>{material(label)}</th><td>{Object.values(data.models.find(m => m.id === 'qwen3vl')?.observations ?? {}).filter(r => r.ground_truth === label && r.raw_top1 === label).length}</td><td>{improvement.report.evaluations.gallery.confusion_matrix[i][i]}</td><td>{improvement.report.evaluations.test.confusion_matrix[i][i]} / {improvement.report.evaluations.test.confusion_matrix[i].reduce((a,b) => a+b,0)}</td></tr>{/each}</tbody></table></div>
          <a href={asset('/trashnet/improvement.json')} download>{say('개선 결과와 원본 점수 내려받기','Download results and scores','結果とスコアをダウンロード')}</a>
        </details>
      </section>
    {/if}
    <div class="controls">
      <label>{say('사진 묶음','Photo set','写真セット')}<select aria-label={say('사진 묶음','Photo set','写真セット')} bind:value={dataset} onchange={changeDataset}><option value="gallery">{say('기존 비교 120장','Historical comparison · 120','従来の比較・120枚')}</option><option value="test">{say('독립 평가 481장','Independent test · 481','独立評価・481枚')}</option></select></label>
      <label>{say('비교할 모델','Model','モデル')}<select aria-label={say('비교할 모델','Model','モデル')} bind:value={modelId}>{#each data.models.filter(item => dataset === 'gallery' || item.supervised) as item (item.id)}<option value={item.id}>{item.name}</option>{/each}</select></label>
      <label>{say('정답 재질','Ground-truth material','正解の素材')}<select aria-label={say('정답 재질','Ground-truth material','正解の素材')} bind:value={category} onchange={updateFilters}><option value="all">{say('전체 재질','All materials','すべての素材')}</option>{#each data.source.classes as id (id)}<option value={id}>{material(id)}</option>{/each}</select></label>
      <label>{say('판단 결과','Outcome','判断結果')}<select aria-label={say('판단 결과','Outcome','判断結果')} bind:value={status} onchange={updateFilters}><option value="all">{say('전체 결과','All outcomes','すべての結果')}</option>{#each ['correct','wrong','abstained'] as id (id)}<option value={id}>{statusLabel(id)}</option>{/each}</select></label>
      <span class="muted count">{filtered.length} / {total} {say('장','photos','枚')}</span>
    </div>
    <div class="stats gallery-stats" aria-label={say('선택한 사진 묶음 전체 집계','All photos in selected set','選択した写真セットの集計')}>
      <div class="stat"><span>{say('전체 중 수락 정답','Accepted correct / all','全体の受理正解')}</span><strong>{correct}/{total}</strong><span>{(correct / total * 100).toFixed(1)}%</span></div>
      <div class="stat"><span>{say('응답 수락률','Coverage','回答受理率')}</span><strong>{accepted.length}/{total}</strong><span>{(accepted.length / total * 100).toFixed(1)}%</span></div>
      <div class="stat"><span>{say('수락한 답 중 정답','Correct / accepted','受理した回答の正解')}</span><strong>{accepted.length ? `${(correct / accepted.length * 100).toFixed(1)}%` : '—'}</strong><span>{correct}/{accepted.length}</span></div>
      <div class="stat"><span>{say('보류 전 1순위 정답','Raw top-1 correct','保留前の首位正解')}</span><strong>{rawCorrect}/{total}</strong><span>{say('보류도 포함한 후보 순위','Includes abstained rankings','保留した順位も含む')}</span></div>
    </div>
    <p class="provenance"><strong>{say('실제 모델 실행 기록','Recorded model inference','実際のモデル実行記録')}</strong> · {model.recorded_at ?? data.recorded_at} · {model.runtime ?? data.runtime}<br />{say('집계는 필터와 관계없이 선택한 사진 묶음 전체 기준입니다. 사진 선택은 저장된 결과를 표시하며 모델을 새로 실행하지 않습니다.','Metrics cover the full selected photo set regardless of filters. Selecting a photo displays saved results; it does not run inference.','集計は選択した写真セット全体が対象です。写真の選択は記録の表示であり、再推論ではありません。')}</p>
    {#if selected && detail}
      <section id="photo-detail" class="split detail" aria-label={say('선택한 사진 상세','Selected photo details','選択した写真の詳細')}>
        <div class="panel photo-panel"><img class="hero-photo" src={asset(selected.image_url)} alt={`${material(selected.label)} · ${selected.name.split('/').at(-1)}`} width="512" height="384" /><div class="photo-caption"><span>{selected.name.split('/').at(-1)}</span><a href={resolve(`/detect?sample=${encodeURIComponent(selected.name.split('/').at(-1) ?? '')}`)}>{say('이 사진에서 객체 탐지','Detect objects in this photo','この写真の物体を検出')} ↗</a></div></div>
        <div class="panel"><span class={`badge ${outcome(detail)}`}>{statusLabel(outcome(detail))}</span><h2 class="decision">{material(detail.selected)}</h2><p>{say('데이터셋 정답','Dataset label','データセットの正解')} <strong>{material(selected.label)}</strong></p><p class="muted">{model.name}{#if detail.latency_ms !== null} · {detail.latency_ms.toFixed(1)} ms · {say('기록된 처리 시간','recorded latency','記録した処理時間')}{/if}</p>
          {#each [...detail.scores].sort((a,b) => b.option_probability - a.option_probability) as score (score.id)}<div class="bar-row"><span>{material(score.id)}</span><div class="track"><span style:width={`${score.option_probability * 100}%`}></span></div><strong class="number">{(score.option_probability * 100).toFixed(1)}%</strong></div>{/each}
          <p class="muted">{say('후보 내 상대 점수이며 정답일 확률은 아닙니다.','Relative scores among candidates, not probabilities of correctness.','候補間の相対スコアであり、正解確率ではありません。')}</p>
          {#if detail.selected === null}<p class="badge abstained">{say('판단 보류','Decision abstained','判断を保留')} · {detail.abstention_reasons.join(', ')}</p>{/if}
          {#if detail.candidate_mass !== null}<p class="muted">Candidate mass: {detail.candidate_mass.toPrecision(5)}<br />{say('수락 기준','Acceptance thresholds','受理基準')}: top ≥ 0.8 · mass ≥ 0.05</p>{:else}<p class="muted">{say('지도학습 분류기 · 보류 없이 1순위 선택 · candidate mass 해당 없음','Supervised classifier · forced top-1 · candidate mass does not apply','教師あり分類器・保留なしの首位選択・candidate mass対象外')}</p>{/if}
          <div class="comparison">{#each data.models as peer (peer.id)}{@const row = peer.observations[selected.name]}{#if row}<div><span>{peer.name}</span><strong class={`badge ${outcome(row)}`}>{material(row.selected)}</strong></div>{/if}{/each}</div>
          <details><summary>{say('원본 점수와 기록 보기','View original scores and record','元のスコアと記録を見る')}</summary><pre>{JSON.stringify(detail, null, 2)}</pre></details>
        </div>
      </section>
    {/if}
    <MaterialClassifier sample={selected} />
    <h2 class="gallery-heading">{say('사진 둘러보기','Explore the photographs','写真を探す')} <span>{filtered.length}</span></h2>
    <div class="gallery">
      {#each filtered as sample (sample.name)}{@const row = model.observations[sample.name]}
        <button class="photo-card" class:chosen={selected?.name === sample.name} aria-pressed={selected?.name === sample.name} onclick={() => choose(sample)} aria-label={`${sample.name.split('/').at(-1)} · ${material(sample.label)} · ${statusLabel(outcome(row))}`}>
          <img src={asset(sample.image_url)} alt={material(sample.label)} loading="lazy" width="512" height="384" />
          <div class="card-body"><span class={`badge ${outcome(row)}`}>{statusLabel(outcome(row))}</span><strong>{material(sample.label)} <span aria-hidden="true">→</span> {material(row.selected)}</strong><small>{sample.name.split('/').at(-1)}</small></div>
        </button>
      {:else}<p class="empty">{say('이 조건에 맞는 사진이 없습니다.','No photos match these filters.','この条件に合う写真はありません。')}</p>{/each}
    </div>
    <details class="panel matrix"><summary>{say('재질별 혼동 행렬 보기','Show confusion matrix by material','素材別の混同行列を見る')}</summary><div class="table-scroll"><table><caption>{say('행: 정답 재질 / 열: 모델의 수락 답변 또는 보류','Rows: ground truth / columns: accepted answer or abstention','行：正解素材／列：受理した回答または保留')}</caption><thead><tr><th scope="col">{say('정답 ↓ / 결과 →','Label ↓ / Result →','正解 ↓ / 結果 →')}</th>{#each [...data.source.classes, 'abstained'] as label (label)}<th scope="col">{material(label === 'abstained' ? null : label)}</th>{/each}</tr></thead><tbody>{#each data.source.classes as truth (truth)}<tr><th scope="row">{material(truth)}</th>{#each [...data.source.classes, 'abstained'] as predicted (predicted)}{@const count = rows.filter((row) => row.ground_truth === truth && (row.selected ?? 'abstained') === predicted).length}<td class:diagonal={truth === predicted} class:populated={count > 0}>{count}</td>{/each}</tr>{/each}</tbody></table></div></details>
    <footer class="footer">{say('기존 비교는 재질별 20장, 독립 평가는 별도 481장입니다. 전체 데이터셋 성능이나 실사용 정확도를 뜻하지 않습니다.','Historical comparison: 20 photos per material. Independent test: 481 separate photos. These are not full-dataset or deployment accuracy.','従来比較は各素材20枚、独立評価は別の481枚です。全データセットや実運用の精度ではありません。')}<br /><a href={`https://github.com/garythung/trashnet/tree/${data.source.source_commit}`} target="_blank" rel="noreferrer">TrashNet · Gary Thung / Mindy Yang</a> · <a href={asset('/trashnet/THIRD_PARTY_NOTICE.txt')} rel="external">MIT / {say('이미지 출처','Image credits','画像の出典')}</a> · <a href={asset('/trashnet/recorded.json')} download>{say('원본 기록 내려받기','Download records','記録をダウンロード')}</a></footer>
  {/if}
</main>
<style>
  .improvement{margin-bottom:24px}.gallery-heading{margin-top:28px!important}.count{margin-left:auto}.detail{margin-bottom:32px}.photo-panel{padding:12px!important}.hero-photo{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:contain;background:#f5f5f3;border-radius:8px}.photo-caption{display:flex;flex-wrap:wrap;justify-content:space-between;gap:12px;font-size:12px;padding:14px 8px 6px}.decision{font-size:34px!important;margin:15px 0!important}.comparison{border-top:1px solid var(--theme-border);padding-top:12px}.comparison>div{display:flex;align-items:center;justify-content:space-between;gap:10px;margin:9px 0;font-size:12px}.gallery-heading{display:flex;gap:12px;align-items:center}.gallery-heading span{font-size:12px;color:var(--theme-muted)}.gallery{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}.photo-card{padding:0!important;text-align:left;overflow:hidden;border-radius:10px!important}.photo-card.chosen{outline:3px solid var(--theme-chart);outline-offset:2px}.photo-card img{display:block;width:100%;height:auto;aspect-ratio:4/3;object-fit:cover;background:white}.card-body{padding:13px}.card-body strong,.card-body small{display:block;margin-top:10px;font-size:12px}.card-body small{color:var(--theme-muted);font-size:10px}.matrix{margin-top:28px}.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:12px;text-align:center;margin-top:16px}caption{font-size:12px;text-align:left;padding:15px 0;color:var(--theme-muted)}th,td{padding:12px;border:1px solid var(--theme-border);white-space:nowrap}td.populated{background:var(--theme-warning-bg)}td.diagonal{background:var(--theme-tint);font-weight:bold}@media(max-width:1000px){.gallery{grid-template-columns:repeat(3,minmax(0,1fr))}}@media(max-width:650px){.gallery{grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.count{margin-left:0}.card-body{padding:10px}}
</style>
