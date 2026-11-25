<script lang="ts">
  import { locale } from '$lib/i18n';
  import type { Decision } from '$lib/demo/types';

  let { source, disabled, onchange }: {
    source: string;
    disabled: boolean;
    onchange: (source: string) => void;
  } = $props();

  const labels = {
    en: { title: 'Decision builder', add: 'Add decision', remove: 'Remove decision', id: 'Decision ID', type: 'Type', rule: 'Instruction', choice: 'Choice', binary: 'Binary', ordinal: 'Ordinal', false: 'False criterion', true: 'True criterion', candidate: 'Option ID', criterion: 'Criterion', value: 'Value', addOption: 'Add option', removeOption: 'Remove option', invalid: 'Fix the decision JSON to use the builder.' },
    ko: { title: '판단 편집기', add: '판단 추가', remove: '판단 삭제', id: '판단 ID', type: '유형', rule: '판단 지침', choice: '선택형', binary: '이진형', ordinal: '순서형', false: '거짓 기준', true: '참 기준', candidate: '옵션 ID', criterion: '기준', value: '값', addOption: '옵션 추가', removeOption: '옵션 삭제', invalid: '판단 JSON을 수정하면 편집기를 사용할 수 있어요.' },
    ja: { title: '判断エディター', add: '判断を追加', remove: '判断を削除', id: '判断 ID', type: '種類', rule: '判断の指示', choice: '選択', binary: '二値', ordinal: '順序', false: '偽の条件', true: '真の条件', candidate: '選択肢 ID', criterion: '条件', value: '値', addOption: '選択肢を追加', removeOption: '選択肢を削除', invalid: '判断の JSON を修正するとエディターを使えます。' }
  };

  type Kind = Decision['kind']['type'];
  type Option = { id: string; criterion: string; value?: number | null };
  type Draft = Omit<Decision, 'kind'> & {
    kind: Omit<Decision['kind'], 'levels'> & { levels?: Option[] };
  };

  let copy = $derived(labels[$locale]);
  let decisions = $derived(read(source));

  function read(json: string): Draft[] | null {
    try {
      const value = JSON.parse(json);
      if (!Array.isArray(value) || !value.every((item) => item && typeof item.id === 'string'
        && typeof item.instruction === 'string' && ['choice', 'ordinal', 'binary'].includes(item.kind?.type)
        && (item.kind.type === 'binary' || Array.isArray(item.kind[item.kind.type === 'choice' ? 'options' : 'levels'])
          && item.kind[item.kind.type === 'choice' ? 'options' : 'levels'].every(
            (option: Option) => option && typeof option.id === 'string' && typeof option.criterion === 'string'
          )))) return null;
      return value;
    } catch {
      return null;
    }
  }

  function edit(update: (draft: Draft[]) => void) {
    const draft = read(source);
    if (!draft) return;
    update(draft);
    onchange(JSON.stringify(draft, null, 2));
  }

  function newId(items: { id: string }[], prefix: string) {
    let index = items.length + 1;
    while (items.some((item) => item.id === `${prefix}_${index}`)) index++;
    return `${prefix}_${index}`;
  }

  function kind(type: Kind): Draft['kind'] {
    const options = [{ id: 'option_1', criterion: '' }, { id: 'option_2', criterion: '' }];
    if (type === 'choice') return { type, options };
    if (type === 'ordinal') return { type, levels: options.map((option, value) => ({ ...option, value })) };
    return { type, false_label: '', true_label: '' };
  }

  function optionsFor(item: Draft): Option[] {
    return item.kind.type === 'ordinal' ? item.kind.levels! : item.kind.options!;
  }
</script>

<div class="builder">
  <div class="builder-heading">
    <h3>{copy.title}</h3>
    {#if decisions}<button type="button" disabled={disabled || decisions.length >= 8} onclick={() => edit((items) => items.push({ id: newId(items, 'decision'), instruction: '', kind: kind('binary') }))}>{copy.add}</button>{/if}
  </div>
  {#if decisions}
    {#each decisions as decision, index}
      <fieldset {disabled}>
        <legend>{decision.id || `#${index + 1}`}</legend>
        <div class="identity">
          <label>{copy.id}<input value={decision.id} oninput={(event) => edit((items) => { items[index].id = event.currentTarget.value; })} /></label>
          <label>{copy.type}<select value={decision.kind.type} onchange={(event) => edit((items) => { items[index].kind = kind(event.currentTarget.value as Kind); })}>
            <option value="binary">{copy.binary}</option><option value="choice">{copy.choice}</option><option value="ordinal">{copy.ordinal}</option>
          </select></label>
        </div>
        <label>{copy.rule}<textarea rows="2" value={decision.instruction} oninput={(event) => edit((items) => { items[index].instruction = event.currentTarget.value; })}></textarea></label>
        {#if decision.kind.type === 'binary'}
          {#each ['false_label', 'true_label'] as key, labelIndex}
            {@const field = key as 'false_label' | 'true_label'}
            <label>{labelIndex === 0 ? copy.false : copy.true}<input value={decision.kind[field] ?? ''} oninput={(event) => edit((items) => { items[index].kind[field] = event.currentTarget.value; })} /></label>
          {/each}
        {:else}
          {@const options = optionsFor(decision)}
          {#each options as option, optionIndex}
            <div class="option">
              <div class="identity">
                <label>{copy.candidate}<input value={option.id} oninput={(event) => edit((items) => { optionsFor(items[index])[optionIndex].id = event.currentTarget.value; })} /></label>
                {#if decision.kind.type === 'ordinal'}
                  <label>{copy.value}<input type="number" step="any" value={option.value ?? undefined} oninput={(event) => edit((items) => { optionsFor(items[index])[optionIndex].value = Number.isFinite(event.currentTarget.valueAsNumber) ? event.currentTarget.valueAsNumber : null; })} /></label>
                {/if}
              </div>
              <label>{copy.criterion}<input value={option.criterion} oninput={(event) => edit((items) => { optionsFor(items[index])[optionIndex].criterion = event.currentTarget.value; })} /></label>
              <button type="button" disabled={options.length <= 2} onclick={() => edit((items) => optionsFor(items[index]).splice(optionIndex, 1))}>{copy.removeOption}</button>
            </div>
          {/each}
          <button type="button" disabled={options.length >= 26} onclick={() => edit((items) => {
            const item = items[index];
            const candidates = optionsFor(item);
            const next = { id: newId(candidates, 'option'), criterion: '' };
            if (item.kind.type === 'ordinal') {
              const last = candidates.at(-1)?.value ?? 0;
              candidates.push({ ...next, value: last + Math.max(1, Math.abs(last) * Number.EPSILON) });
            } else candidates.push(next);
          })}>{copy.addOption}</button>
        {/if}
        <button class="remove" type="button" disabled={decisions.length <= 1} onclick={() => edit((items) => items.splice(index, 1))}>{copy.remove}</button>
      </fieldset>
    {/each}
  {:else}
    <p role="status">{copy.invalid}</p>
  {/if}
</div>

<style>
  .builder { margin-block: 1.5rem 1rem; }
  .builder-heading, .identity { display: flex; align-items: center; gap: .75rem; }
  .builder-heading { justify-content: space-between; flex-wrap: wrap; margin-bottom: .75rem; }
  h3 { margin: 0; font-size: 1rem; }
  fieldset { min-width: 0; margin: 0 0 1rem; padding: .75rem; border: 1px solid var(--theme-border); border-radius: 8px; }
  legend { max-width: 100%; padding-inline: .35rem; color: var(--theme-accent); font: 600 .85rem ui-monospace, monospace; overflow-wrap: anywhere; }
  .identity > label { flex: 1; min-width: 0; }
  label { display: grid; gap: .25rem; margin-bottom: .65rem; font-size: .875rem; }
  input, select, textarea, button { font: inherit; color: var(--theme-text); background: var(--theme-panel); border: 1px solid var(--theme-border); border-radius: 5px; padding: .55rem .65rem; }
  input, select, textarea { width: 100%; }
  textarea { resize: vertical; }
  button { font-size: .875rem; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .option { background: var(--theme-tint); border-radius: 6px; padding: .65rem; margin-bottom: .65rem; }
  fieldset > .remove { margin-inline-start: .35rem; }
</style>
