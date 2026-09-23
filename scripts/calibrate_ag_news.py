#!/usr/bin/env python3
"""AG News-only candidate temperature experiment. Never changes full-vocabulary mass."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import random
import zipfile
from kaggle_ag_news import ARCHIVE_SHA256, CLASSES, DESCRIPTIONS, identity, sha256


def prepare(archive, template, output):
    if sha256(archive) != ARCHIVE_SHA256:
        raise ValueError('Unexpected dataset archive')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'selection.json').exists():
        raise ValueError('Selection already exists')
    with zipfile.ZipFile(archive) as z:
        train, test = [list(csv.DictReader(io.StringIO(z.read(f).decode('utf-8-sig')))) for f in ['train.csv', 'test.csv']]
    seen = {identity(row) for row in test}
    groups = {label: [] for label in CLASSES}
    for index, row in enumerate(train, 1):
        key = identity(row)
        if key in seen or not row['Title'].strip() or not row['Description'].strip():
            continue
        seen.add(key)
        groups[row['Class Index']].append((index, row))
    rng = random.Random(20260922)
    splits = {'fit': [], 'validation': []}
    for group in groups.values():
        chosen = rng.sample(group, 200)
        splits['fit'].extend(chosen[:100])
        splits['validation'].extend(chosen[100:])
    decision = json.loads(template.read_text().splitlines()[0])['request']['decisions']
    manifest = dict(seed=20260922, archive_sha256=ARCHIVE_SHA256, labels={}, splits={},
                    scope='400 fit and 400 held-out validation from train.csv, deduplicated against all test.csv and each other. Fit NLL only, fixed acceptance 0.8, unchanged mass threshold 0.05. No prompt tuning.')
    for split, selected in splits.items():
        rng.shuffle(selected)
        records = []
        for index, row in selected:
            case_id = f'ag-news-train-{index:06d}'
            manifest['labels'][case_id] = CLASSES[row['Class Index']]
            records.append(dict(id=case_id, request=dict(state=dict(title=row['Title'], description=row['Description']), decisions=decision)))
        path = output / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
        manifest['splits'][split] = dict(ids=[r['id'] for r in records], request_sha256=sha256(path))
    (output / 'selection.json').write_text(json.dumps(manifest, indent=2)+'\n')


def probabilities(logits, temperature):
    if not math.isfinite(temperature) or temperature <= 0 or not logits or not all(math.isfinite(z) for z in logits):
        raise ValueError('Finite logits and positive finite temperature required')
    largest = max(logits)
    shifted = [(z-largest)/temperature for z in logits]
    lse = math.log(sum(math.exp(z) for z in shifted))
    logs = [z-lse for z in shifted]
    return [math.exp(z) for z in logs], logs


def metrics(rows, labels, temperature):
    nll = brier = confidence = 0.0
    correct = accepted = accepted_correct = 0
    bins = [[] for _ in range(10)]
    for row in rows:
        result = row['response']['results'][0]
        assert not result['truncated']
        scores = result['scores']
        p, logp = probabilities([s['raw_logit'] for s in scores], temperature)
        gold = next(i for i, s in enumerate(scores) if s['id'] == labels[row['id']])
        best = max(range(len(p)), key=p.__getitem__)
        hit = best == gold
        tied = sum(abs(v-p[best]) < 1e-12 for v in p) > 1
        hit = hit and not tied
        nll -= logp[gold]
        brier += sum((v-(i == gold))**2 for i, v in enumerate(p))
        confidence += p[best]
        correct += hit
        bins[min(9, int(p[best]*10))].append((p[best], hit))
        if p[best] >= .8 and result['candidate_mass'] >= .05 and not tied:
            accepted += 1
            accepted_correct += hit
    n = len(rows)
    ece = sum(abs(sum(p for p, _ in b)-sum(hit for _, hit in b)) for b in bins)/n
    return dict(n=n, nll=nll/n, brier=brier/n, ece_10_equal_width=ece, mean_confidence=confidence/n,
                raw_top1=correct/n, accepted=accepted, accepted_correct=accepted_correct,
                accepted_accuracy=accepted_correct/accepted if accepted else None, coverage=accepted/n,
                correct_all=accepted_correct/n)


def fit_temperature(rows, labels):
    # Deterministic bounded golden-section search on log(T); fit labels only.
    lo, hi = math.log(.05), math.log(100.0)
    ratio = (math.sqrt(5)-1)/2
    loss = lambda t: metrics(rows, labels, math.exp(t))['nll']
    c, d = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fc, fd = loss(c), loss(d)
    for _ in range(70):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi-ratio*(hi-lo)
            fc = loss(c)
        else:
            lo, c, fc = c, d, fd
            d = lo+ratio*(hi-lo)
            fd = loss(d)
    return math.exp((lo+hi)/2)


def fit(selection, fit_file, validation_file, output):
    manifest = json.loads(selection.read_text())
    rows = {name: [json.loads(s) for s in path.read_text().splitlines()] for name, path in [('fit', fit_file), ('validation', validation_file)]}
    backend = rows['fit'][0]['response']['backend']
    for name, records in rows.items():
        ids = [r['id'] for r in records]
        assert len(ids) == len(set(ids)) and set(ids) == set(manifest['splits'][name]['ids'])
        assert all('response' in r for r in records)
        for row in records:
            response = row['response']
            assert response['backend'] == backend, 'Do not mix compute/model/prompt configurations'
            assert response['policy'] == {'min_top_probability': .8, 'min_candidate_mass': .05}
            assert len(response['results']) == 1
            result = response['results'][0]
            assert result['id'] == 'news_topic' and result['calibration_id'] is None
            assert [s['id'] for s in result['scores']] == list(DESCRIPTIONS)
    assert set(manifest['splits']['fit']['ids']).isdisjoint(manifest['splits']['validation']['ids'])
    labels = manifest['labels']
    temperature = fit_temperature(rows['fit'], labels)
    result = dict(temperature=temperature, fit_objective='NLL', temperature_bounds=[.05, 100.0], scope='AG News candidate-only temperature. Does not calibrate candidate_mass or retrain the model; argmax unchanged. Validation labels not used for fitting.',
                  provenance={str(p): sha256(p) for p in [selection, fit_file, validation_file]},
                  backend=backend,
                  evaluation={name: {'before': metrics(r, labels, 1.0), 'after': metrics(r, labels, temperature)} for name, r in rows.items()})
    with output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    for arg in ['archive', 'template', 'output']:
        a.add_argument('--'+arg, type=Path, required=True)
    a = sub.add_parser('fit')
    for arg in ['selection', 'fit-file', 'validation-file', 'output']:
        a.add_argument('--'+arg, type=Path, required=True)
    args = vars(p.parse_args())
    command = args.pop('command')
    (prepare if command == 'prepare' else fit)(**args)


if __name__ == '__main__':
    main()
