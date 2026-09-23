#!/usr/bin/env python3
"""Independently audit and summarize saved intent tournament runs."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    position = q * (len(values)-1)
    i = int(position)
    return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(position-i)


def audit(root, directory):
    data = load(root/'prepared/datasets.json')
    gold = {r['id']: r for r in rows(root/'prepared/gold.jsonl')}
    first_list = rows(directory/'stage1-predictions.jsonl')
    final_list = rows(directory/'final-predictions.jsonl')
    first = {r['id']: r for r in first_list}
    final = {r['id']: r for r in final_list}
    assert len(first) == len(first_list) == 1200
    assert len(final) == len(final_list) == 400
    assert set(first) == {key + f':g{i}' for key in gold for i in range(3)}
    assert set(final) == {key + ':final' for key in gold}
    winners, selections = {}, {}
    for p in first_list + final_list:
        assert 'error' not in p
        r = p['response']['results'][0]
        assert not r['truncated']
        scores = {s['id']: s['option_probability'] for s in r['scores']}
        assert len(scores) == len(r['scores']) and abs(sum(scores.values())-1) < 1e-9
        best = sorted(scores, key=lambda k: (-scores[k], k))[0]
        winners[p['id']] = best
        expected_selection = best if scores[best] >= 0.8 and r['candidate_mass'] >= 0.05 else None
        assert r['value']['selected'] == expected_selection
        selections[p['id']] = expected_selection
    tallies = {key: collections.Counter() for key in data}
    timings = {key: [] for key in data}
    for key, item in gold.items():
        group_keys = [key+f':g{i}' for i in range(3)]
        finalist_ids = [s['id'] for s in final[key+':final']['response']['results'][0]['scores']]
        assert finalist_ids == [winners[g] for g in group_keys]
        for i, group in enumerate(group_keys):
            ids = [s['id'] for s in first[group]['response']['results'][0]['scores']]
            assert ids == data[item['dataset']]['groups'][i]
        picked = winners[key+':final']
        winning_group = next(g for g in group_keys if winners[g] == picked)
        accepted = selections[winning_group] == picked and selections[key+':final'] == picked
        counter = tallies[item['dataset']]
        counter['total'] += 1
        counter['correct'] += picked == item['expected']
        counter['accepted'] += accepted
        counter['accepted_correct'] += accepted and picked == item['expected']
        counter['gold_reached_final'] += item['expected'] in finalist_ids
        timings[item['dataset']].append(sum(first[g]['elapsed_ms'] for g in group_keys) + final[key+':final']['elapsed_ms'])
    summary = load(directory/'summary.json')
    for name, counter in tallies.items():
        for metric, value in counter.items():
            assert summary[name][metric] == value, (name, metric)
        for q, key in [(0.5, 'p50_ms'), (0.95, 'p95_ms')]:
            assert abs(summary[name][key]-quantile(timings[name], q)) < 1e-8
    manifest = load(directory/'manifest.json')
    assert manifest['prepared_manifest_sha256'] == digest(root/'prepared/manifest.json')
    return dict(id=directory.name, manifest=manifest, datasets=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    prepared_manifest = load(root/'prepared/manifest.json')
    for name, expected in prepared_manifest['prepared_sha256'].items():
        assert digest(root/'prepared'/name) == expected
    plan = load(root/'plan.json')
    results = [audit(root, root/'runs'/r['id']) for r in plan]
    assert len({r['manifest']['evaluator_sha256'] for r in results}) == 1
    assert len({r['manifest']['script_sha256'] for r in results}) == 1
    for expected, measured in zip(plan, results):
        assert expected == measured['manifest']['model']
        assert measured['manifest']['script_sha256'] == digest(root/'evaluate_intents.py')
    datasets = load(root/'prepared/datasets.json')
    data = dict(independently_audited=True, configurations=len(results),
                examples_per_model=400, total_decisions=1200, total_native_calls=4800, results=results)
    (root/'REPORT.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    text = ['# BANKING77 English and MASSIVE Korean: 200 examples each', '',
            'Date: 2026-09-23. Host: `100.66.64.91`, NVIDIA RTX 3060 12 GiB.', '',
            'Gemma 4 E2B Q8_0, E4B Q8_0 and E4B Q4_K_M use the previously rebuilt L2S1 evaluator. Each model evaluated the same 400 frozen test examples. All 1,200 final classifications and 4,800 underlying inference calls completed without errors or truncation.', '',
            '**These are zero-shot, two-stage tournament results.** L2S1 supports at most 26 candidates per call. All 77 BANKING77 labels or 60 MASSIVE labels are included in three fixed groups; each group nominates its argmax label, then all three nominees compete in a final call. A correct label eliminated in the first stage remains an end-to-end error. These scores are not from a single full-label softmax and are not directly comparable to a trained benchmark leaderboard.', '',
            '| Model | Dataset | Correct / 200 | Accuracy | Wilson 95% interval | p50 ms | p95 ms |',
            '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for result in results:
        for name, s in result['datasets'].items():
            low, high = s['wilson95']
            text.append(f"| {result['id']} | {name} | {s['correct']} | {s['accuracy']:.1%} | {low:.1%}–{high:.1%} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} |")
    text += ['', '## Abstention and candidate routing', '',
             'Raw accuracy always uses the final argmax. A classification is accepted only when both the winning first-stage branch and the final decision pass the unchanged default policy (top candidate probability ≥0.8, full-vocabulary candidate mass ≥0.05). Losing branches can abstain without blocking the winning branch. This is an explicitly defined composed policy, not a native global 77/60-class confidence. Stage probabilities are conditional on their respective candidate sets; no global calibrated probabilities are claimed.', '',
             '| Model | Dataset | Accepted | Accepted correct | Accepted wrong | Accepted accuracy | Coverage | Abstained | Gold reached final |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for result in results:
        for name, s in result['datasets'].items():
            accepted_accuracy = f"{s['accepted_accuracy']:.1%}" if s['accepted_accuracy'] is not None else 'N/A'
            text.append(f"| {result['id']} | {name} | {s['accepted']} | {s['accepted_correct']} | {s['accepted_wrong']} | {accepted_accuracy} | {s['coverage']:.1%} | {s['abstained']} | {s['gold_reached_final']}/200 |")
    text += ['', '## Frozen data and source attribution', '',
             '- [BANKING77 / PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets): CC BY 4.0; test split 3,080 rows, 77 official labels. See Casanueva et al., *Efficient Intent Detection with Dual Sentence Encoders* (2020).',
             '- [MASSIVE / Amazon](https://huggingface.co/datasets/AmazonScience/massive): CC BY 4.0; Korean `ko-KR` test split 2,974 rows, 60 official intent labels. The Parquet is pinned to `6e31162aba58a715666d3791566f42afdcfa62b2`; label IDs are decoded from its embedded Hugging Face schema. See FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset* (2022).',
             '- Uniform sampling without replacement, Python random seed `20260923` independently per dataset; sampled indices sorted back into source order. Sampling, prompts and grouping were fixed before inference. No training, label-based shortlist, test-driven prompt tuning, retrieval or adapters.',
             '- Only the raw utterance enters the state. No scenario, annotated utterance, judgment, expected label or other gold metadata enters inference. Official label names with underscores replaced by spaces serve as criteria. BANKING77 instructions are English; MASSIVE instructions are Korean with original English label names.',
             '- Alphabetically sorted labels are distributed round-robin into three groups. BANKING77 sizes: 26/26/25; MASSIVE: 20/20/20. Final candidate order follows group order.', '']
    for name, d in datasets.items():
        text.append(f"- `{name}`: 200 examples contain {len(d['sampled_class_counts'])} distinct gold classes. Official labels absent from the full test split: {d['labels_absent_from_test']}. All official labels remain candidates.")
    text += ['', '## Runtime, validation and limits', '',
             '- Same binary as the preceding Gemma-only rebuild: `'+results[0]['manifest']['evaluator_sha256']+'`.',
             '- CUDA, context 8192, batch/microbatch 256, four threads, fresh execution, legacy/minimal prompt, FlashAttention off, full evidence, no preparation cache, no LoRA/output head/calibration. One warmup per evaluator process.',
             '- Timing sums all four serial Rust inference-call durations for each example. Model loading, warmups, Python routing, process startup and network transfer are excluded. It is neither single-call latency nor measured online wall-clock latency.',
             '- The two stages run in separate evaluator processes; phase logs, exact commands, original probabilities, shortlist requests and per-example summed timings are retained. No repeated timing trial was used.',
             '- A separate report script verified group coverage, routing, top-1 outcomes, default policy decisions, totals and latency quantiles from raw predictions. All configurations share source sample and evaluator hashes. Four adapter contract tests cover complete candidate coverage, gold isolation, routing through abstention and deterministic ties.',
             '- The 200-row samples estimate performance on these particular test subsets, not the entire datasets or production traffic. Wilson intervals describe sampling uncertainty; model pretraining contamination is not audited. Korean results evaluate intent classification, not slot extraction.', '',
             '## Reproduction and artifacts', '',
             '`prepared/manifest.json` records raw and prepared file hashes, source URLs and the seed. `prepared/gold.jsonl` is kept separate from unlabeled samples and requests. `runs/<model>/` contains raw stage predictions, final requests, score details, summaries, commands and timing logs. `REPORT.json` includes all metrics and frequent confusions.', '',
             'Use `scripts/evaluate_intents.py prepare --root RESULTS` with pyarrow to prepare the downloaded files, then `scripts/evaluate_intents.py run --root RESULTS --evaluator BINARY --plan PLAN`. Preparation and run output directories must be fresh. Recount with `scripts/report_intents.py RESULTS`.', '']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({r['id']: {name: s['correct'] for name,s in r['datasets'].items()} for r in results}))


if __name__ == '__main__':
    main()
