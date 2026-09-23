#!/usr/bin/env python3
"""Frozen BANKING77/MASSIVE intent evaluation using a 3-group tournament.

The existing engine supports at most 26 candidates. Every label is included in
stage one; three group winners compete in stage two. This is a routed classifier,
not a single 77/60-way softmax. Gold is read only when scoring saved predictions.
"""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time

SEED = 20260923
EVALUATOR_SHA256 = 'f7fdc30802e4295c3423a7c04037e8e4ad246405e18b1479eb521cd611b5a19e'


def load(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def save_rows(path, values):
    path.write_text(''.join(json.dumps(v, ensure_ascii=False, allow_nan=False) + '\n' for v in values))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def groups(labels):
    assert len(labels) in (60, 77) and len(set(labels)) == len(labels)
    return [sorted(labels)[i::3] for i in range(3)]


def request(item, labels, suffix):
    instruction = ('Choose the intent that best matches the customer utterance. '
                   'Select exactly one of the listed intent labels.' if item['dataset'] == 'banking77-en'
                   else '사용자 발화에 가장 잘 맞는 의도를 선택하세요. 나열된 의도 라벨 중 정확히 하나를 선택하세요.')
    return dict(id=item['id'] + suffix, request=dict(state={'utterance': item['text']}, decisions=[
        dict(id='intent', instruction=instruction, kind=dict(type='choice', options=[
            dict(id=label, criterion=label.replace('_', ' ')) for label in labels]))]))


def prepare(root):
    import pyarrow.parquet as pq
    out = root / 'prepared'
    out.mkdir(exist_ok=False)
    banking = root / 'data/banking77'
    massive = root / 'data/massive'
    bank_labels = load(banking / 'categories.json')
    with (banking / 'test.csv').open(newline='') as stream:
        bank = [dict(id=f'banking77-en:{i}', dataset='banking77-en', source_id=i,
                     text=r['text'], expected=r['category']) for i, r in enumerate(csv.DictReader(stream))]
    table = pq.read_table(massive / 'ko-KR-test.parquet')
    features = json.loads(table.schema.metadata[b'huggingface'])['info']['features']
    mass_labels = features['intent']['names']
    mass = []
    for r in table.to_pylist():
        assert r['locale'] == 'ko-KR' and r['partition'] == 'test'
        mass.append(dict(id='massive-ko:' + r['id'], dataset='massive-ko', source_id=r['id'],
                         text=r['utt'], expected=mass_labels[r['intent']]))
    assert len(bank) == 3080 and len(mass) == 2974
    assert len(bank_labels) == 77 and len(mass_labels) == 60
    samples, datasets = [], {}
    for name, population, labels in [('banking77-en', bank, bank_labels), ('massive-ko', mass, mass_labels)]:
        assert set(r['expected'] for r in population) <= set(labels)
        indices = sorted(random.Random(SEED).sample(range(len(population)), 200))
        selected = [population[i] for i in indices]
        samples.extend(selected)
        datasets[name] = dict(test_population=len(population), sample_size=200, labels=sorted(labels),
                              groups=groups(labels), sampled_indices=indices,
                              sampled_class_counts=dict(collections.Counter(r['expected'] for r in selected)),
                              labels_absent_from_test=sorted(set(labels)-{r['expected'] for r in population}))
    # Gold and source annotations never enter the evaluator input or routing.
    unlabeled = [{k: r[k] for k in ('id', 'dataset', 'text')} for r in samples]
    save_rows(out / 'samples.jsonl', unlabeled)
    save_rows(out / 'gold.jsonl', samples)
    save_rows(out / 'stage1-requests.jsonl', [request(r, g, f':g{i}') for r in unlabeled
        for i, g in enumerate(datasets[r['dataset']]['groups'])])
    save(out / 'datasets.json', datasets)
    save(out / 'manifest.json', dict(seed=SEED, sampling='Uniform without replacement, original test row order; independent seeded RNG per dataset',
        grouping='Alphabetically sorted official labels, round-robin into three fixed groups',
        method='3 group argmax winners, then final argmax among all 3; no gold-based shortlist',
        sources={
            'banking77': 'https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data',
            'massive': 'https://huggingface.co/datasets/AmazonScience/massive/blob/6e31162aba58a715666d3791566f42afdcfa62b2/ko-KR/massive-test.parquet'},
        licenses={'banking77': 'CC BY 4.0, PolyAI', 'massive': 'CC BY 4.0, Amazon'},
        raw_sha256={str(p.relative_to(root)): sha(p) for p in [banking/'test.csv', banking/'categories.json', massive/'ko-KR-test.parquet']},
        prepared_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}))
    print(json.dumps({k: dict(n=v['sample_size'], classes_in_sample=len(v['sampled_class_counts']),
                             group_sizes=list(map(len, v['groups']))) for k, v in datasets.items()}))


def validated(predictions, requests):
    expected = {r['id']: r for r in requests}
    assert len(expected) == len(requests)
    assert len(predictions) == len(requests)
    result = {}
    for p in predictions:
        key = p['id']
        assert key in expected and key not in result
        assert not p.get('error'), (key, p.get('error'))
        backend = p['response']['backend']
        assert 'RTX 3060' in backend['offload_device'] and backend['offload_requested']
        assert backend['execution_mode'] == 'fresh' and backend['prompt_layout'] == 'legacy'
        assert backend['compute'] == dict(batch=256, context=8192, flash_attention='off', threads=4, ubatch=256)
        assert len(p['response']['results']) == 1
        r = p['response']['results'][0]
        assert r['id'] == 'intent' and not r['truncated'] and r['reused_prefix_tokens'] == 0
        assert [s['id'] for s in r['scores']] == [o['id'] for o in expected[key]['request']['decisions'][0]['kind']['options']]
        probs = [s['option_probability'] for s in r['scores']]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in probs) and abs(sum(probs) - 1) < 1e-9
        assert math.isfinite(p['elapsed_ms']) and p['elapsed_ms'] >= 0
        assert p['response']['policy'] == dict(min_top_probability=0.8, min_candidate_mass=0.05)
        result[key] = p
    return result


def top(prediction):
    return min(prediction['response']['results'][0]['scores'], key=lambda s: (-s['option_probability'], s['id']))['id']


def route(samples, first):
    return [request(r, [top(first[r['id'] + f':g{i}']) for i in range(3)], ':final') for r in samples]


def quantile(values, q):
    values = sorted(values)
    point = (len(values) - 1) * q
    low, high = math.floor(point), math.ceil(point)
    return values[low] + (values[high] - values[low]) * (point - low)


def score(root, out):
    prepared = root / 'prepared'
    first = validated(rows(out/'stage1-predictions.jsonl'), rows(prepared/'stage1-requests.jsonl'))
    final_requests = rows(out/'final-requests.jsonl')
    assert final_requests == route(rows(prepared/'samples.jsonl'), first), 'Shortlist is not reproduced by saved group predictions'
    final = validated(rows(out/'final-predictions.jsonl'), final_requests)
    gold = rows(prepared/'gold.jsonl')
    details = []
    for item in gold:
        group_predictions = [first[item['id'] + f':g{i}'] for i in range(3)]
        p = final[item['id'] + ':final']
        label = top(p)
        winner = next(g for g in group_predictions if top(g) == label)
        selected = lambda record: record['response']['results'][0]['value']['selected']
        accepted = selected(p) == label and selected(winner) == label
        details.append(dict(id=item['id'], dataset=item['dataset'], expected=item['expected'], predicted=label,
                            correct=label == item['expected'], accepted=accepted,
                            gold_reached_final=item['expected'] in [top(g) for g in group_predictions],
                            elapsed_ms=p['elapsed_ms'] + sum(g['elapsed_ms'] for g in group_predictions),
                            max_input_tokens=max(g['response']['results'][0]['input_tokens'] for g in [*group_predictions, p])))
    save_rows(out/'scored.jsonl', details)
    summaries = {}
    for name in ('banking77-en', 'massive-ko'):
        subset = [r for r in details if r['dataset'] == name]
        assert len(subset) == 200
        correct = sum(r['correct'] for r in subset)
        accepted = [r for r in subset if r['accepted']]
        correct_accepted = sum(r['correct'] for r in accepted)
        latencies = [r['elapsed_ms'] for r in subset]
        n, z = len(subset), 1.959963984540054
        acc = correct / n
        center = (acc + z*z/(2*n)) / (1+z*z/n)
        half = z*math.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / (1+z*z/n)
        confusions = collections.Counter((r['expected'], r['predicted']) for r in subset if not r['correct'])
        summaries[name] = dict(total=n, correct=correct, accuracy=acc, wilson95=[center-half, center+half],
            accepted=len(accepted), accepted_correct=correct_accepted, accepted_wrong=len(accepted)-correct_accepted,
            accepted_accuracy=correct_accepted/len(accepted) if accepted else None,
            coverage=len(accepted)/n, abstained=n-len(accepted),
            gold_reached_final=sum(r['gold_reached_final'] for r in subset), errors=0, truncated=0,
            p50_ms=quantile(latencies, 0.5), p95_ms=quantile(latencies, 0.95),
            max_input_tokens=max(r['max_input_tokens'] for r in subset),
            top_confusions=[dict(expected=a, predicted=b, count=c) for (a,b),c in confusions.most_common(15)])
    save(out/'summary.json', summaries)
    return summaries


def evaluate(evaluator, model, input_path, output_path, log):
    command = [str(evaluator), '--model', model, '--input', str(input_path), '--output', str(output_path),
               '--context', '8192', '--batch', '256', '--ubatch', '256', '--threads', '4',
               '--execution-mode', 'fresh', '--prompt-layout', 'legacy', '--warmup', '--cuda']
    save(log.with_suffix('.command.json'), command)
    with log.open('x') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=3600)


def run(root, evaluator, plan):
    assert sha(evaluator) == EVALUATOR_SHA256, 'Evaluator differs from the verified Gemma rebuild'
    for p, digest in load(root/'prepared/manifest.json')['prepared_sha256'].items():
        assert sha(root/'prepared'/p) == digest
    samples = rows(root/'prepared/samples.jsonl')
    for model in load(plan):
        name = model['id']
        out = root / 'runs' / name
        out.mkdir(parents=True, exist_ok=False)
        started = time.time()
        print('START', name, flush=True)
        digest = sha(Path(model['path']))
        assert digest == model['sha256']
        save(out/'manifest.json', dict(model=model, evaluator_sha256=sha(evaluator),
            script_sha256=sha(Path(__file__)), prepared_manifest_sha256=sha(root/'prepared/manifest.json'),
            method='3 groups plus final; four fresh calls per example', start_unix=started))
        evaluate(evaluator, model['path'], root/'prepared/stage1-requests.jsonl', out/'stage1-predictions.jsonl', out/'stage1.log')
        first = validated(rows(out/'stage1-predictions.jsonl'), rows(root/'prepared/stage1-requests.jsonl'))
        save_rows(out/'final-requests.jsonl', route(samples, first))
        print('FINAL_STAGE', name, flush=True)
        evaluate(evaluator, model['path'], out/'final-requests.jsonl', out/'final-predictions.jsonl', out/'final.log')
        summaries = score(root, out)
        save(out/'complete.json', dict(elapsed_s=time.time()-started, inference_calls=1600, examples=400))
        print('COMPLETE', name, json.dumps({k: v['correct'] for k,v in summaries.items()}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'run', 'score'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evaluator', type=Path)
    parser.add_argument('--plan', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == 'prepare':
        prepare(root)
    elif args.mode == 'run':
        run(root, args.evaluator.resolve(), args.plan)
    else:
        for directory in sorted((root/'runs').iterdir()):
            print(directory.name, json.dumps(score(root, directory)))


if __name__ == '__main__':
    main()
