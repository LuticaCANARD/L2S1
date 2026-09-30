#!/usr/bin/env python3
"""Freeze independent Jev data splits or reproduce the pinned specialist pilot."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from .common import digest, option_specs, read_jsonl, require, write_json

REVISION = 'c76749ec58bd8c3d2ea706b31c333a9059c38f90'
SEED = 20260927

PROTOCOL = dict(epochs=1, rank=8, alpha=16, dropout=0, learning_rate=1e-4,
                      gradient_accumulation=12, mass_loss_weight=0.1,
                      objective='soft-target candidate cross entropy + candidate mass loss',
                      rotation='one SHA256-selected code rotation per decision, independent of labels',
                      selection='fixed final epoch; no development/test selection or calibration',
                      prompt_layout='legacy', prompt_detail='minimal')
# Exporter/evaluator flag values. Train on the prompt the application deploys: an adapter
# only helps prompts rendered the way it was trained.
PROMPT_LAYOUTS = ('legacy', 'state-first')
PROMPT_DETAILS = ('minimal', 'typed', 'typed-examples')


def protocol(layout='legacy', detail='minimal'):
    require(layout in PROMPT_LAYOUTS and detail in PROMPT_DETAILS, 'Unknown prompt layout/detail')
    return dict(PROTOCOL, prompt_layout=layout, prompt_detail=detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def distribution(value, ids):
    import math
    require(set(value) == set(ids), 'Probability labels differ from criteria')
    p = [float(value[k]) for k in ids]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in p), 'Invalid probability')
    require(abs(sum(p)-1) < 1e-4, 'Distribution must sum to one (rounding tolerance 1e-4)')
    return [v/sum(p) for v in p]


def decision(qid, q):
    criteria = q.get('criteria')
    kind = q['type']
    if kind == 'noul':
        criteria = criteria or {'false': 'No', 'true': 'Yes'}
        require(set(criteria) == {'false', 'true'}, 'Invalid Noul criteria')
        native = dict(type='binary', false_label=criteria['false'], true_label=criteria['true'])
    elif kind == 'choice':
        require(isinstance(criteria, dict) and 2 <= len(criteria) <= 26, 'Invalid Choice criteria')
        native = dict(type='choice', options=[dict(id=k, criterion=v) for k, v in criteria.items()])
    elif kind == 'score':
        # An ordered array labels levels "0", "1", ...; an ordered object keeps the
        # application's own level IDs (visible to the model in typed prompts).
        items = list(criteria.items()) if isinstance(criteria, dict) else list(enumerate(criteria or []))
        require(isinstance(criteria, (list, dict)) and 2 <= len(items) <= 10, 'Invalid Score criteria')
        native = dict(type='ordinal', levels=[dict(id=str(k), criterion=v, value=i)
                                             for i, (k, v) in enumerate(items)])
    else:
        raise ValueError('Unknown Jev type')
    require(all(isinstance(o['id'], str) and o['id'].strip() and
                isinstance(o['criterion'], str) and o['criterion'].strip()
                for o in option_specs(dict(kind=native))), 'Invalid criterion ID/text')
    require(isinstance(q['instructions'], str) and q['instructions'].strip(), 'Missing instruction')
    return dict(id=qid, instruction=q['instructions'], kind=native)


def convert(row, *, encoded=True):
    state, questions, gold = [json.loads(row[k]) if encoded else row[k]
                              for k in ('state', 'questions', 'gold')]
    require(isinstance(row['id'], str) and row['id'].strip() and '/' not in row['id'], 'Invalid case ID')
    require(isinstance(row['workflow'], str) and row['workflow'].strip(), 'Missing workflow')
    require(isinstance(questions, dict) and isinstance(gold, dict), 'Questions/gold must be objects')
    require(all(isinstance(k, str) and k.strip() and '/' not in k for k in questions), 'Invalid question ID')
    require(questions and set(questions) == set(gold), 'Question/gold mismatch')
    decisions = [decision(k, q) for k, q in questions.items()]
    for d in decisions:
        g = gold[d['id']]
        ids = [o['id'] for o in option_specs(d)]
        distribution(g['probabilities'], ids)
        require(g['label'] in ids and g['type'] == questions[d['id']]['type'], 'Invalid gold')
    return dict(id=row['id'], workflow=row['workflow'], request=dict(state=state, decisions=decisions), gold=gold)


def state_key(case):
    return hashlib.sha256(canonical(case['request']['state']).encode()).hexdigest()


def check_disjoint(splits):
    ids, states = set(), set()
    for name, rows in splits.items():
        local_ids = {r['id'] for r in rows}
        local_states = {state_key(r) for r in rows}
        require(len(local_ids) == len(rows) == len(local_states), f'Duplicate ID/state in {name}')
        require(not (ids & local_ids or states & local_states), f'Cross-split ID/state leakage in {name}')
        ids.update(local_ids)
        states.update(local_states)


def jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')


def prepare(source, output, layout='legacy', detail='minimal'):
    import pyarrow.parquet as pq
    train = [convert(r) for r in pq.read_table(source/'train.parquet').to_pylist()]
    test = [convert(r) for r in pq.read_table(source/'test.parquet').to_pylist()]
    require(len(train) == 1200 and len(test) == 400, 'Pinned split size changed')
    require(digest(source/'train.parquet') == '46a58d63edfd86e23229c78afe8b72307bb4ca9fb0e8df180cabb3c67ec9dcd5',
            'Pinned train source changed')
    require(digest(source/'test.parquet') == '4f294f218ea1da27f3efef936359389c62ea4d3973a41457732990f1d31b647c',
            'Pinned test source changed')
    check_disjoint(dict(source_train=train, test=test))
    grouped = collections.defaultdict(list)
    for row in train:
        grouped[row['workflow']].append(row)
    require(len(grouped) == 4 and all(len(v) == 300 for v in grouped.values()), 'Workflow allocation changed')
    splits = dict(train=[], development=[], unused=[], test=test)
    for workflow in sorted(grouped):
        rows = sorted(grouped[workflow], key=lambda r: hashlib.sha256(f'{SEED}:{r["id"]}'.encode()).hexdigest())
        splits['train'].extend(rows[:30])
        splits['development'].extend(rows[30:50])
        splits['unused'].extend(rows[50:])
    check_disjoint(splits)
    write_dataset(splits, output, 'LocalLLaMA/typed-decisions', REVISION,
        {p.name: digest(p) for p in source.iterdir() if p.is_file()},
        'Synthetic teacher agreement; four seen workflows. Exact canonical state/ID disjointness only; no near-duplicate or pretraining exclusion.',
        protocol(layout, detail))


def prepare_custom(train, development, test, output, layout='legacy', detail='minimal'):
    paths = dict(train=train, development=development, test=test)
    splits = {name: [convert(r, encoded=False) for r in read_jsonl(path)] for name, path in paths.items()}
    require(all(splits.values()), 'Train, development and test splits must be nonempty')
    check_disjoint(splits)
    write_dataset(splits, output, 'custom-jev-jsonl', None,
        {name: digest(path) for name, path in paths.items()},
        'Agreement with supplied labels. Exact state/ID separation only; label quality and near duplicates require dataset review.',
        protocol(layout, detail))


def write_dataset(splits, output, dataset, revision, source_hashes, scope, protocol):
    check_disjoint(splits)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, rows in splits.items():
        labeled = output/f'{name}.jsonl'
        requests = output/f'{name}-requests.jsonl'
        jsonl(labeled, rows)
        jsonl(requests, [dict(id=r['id'], request=r['request']) for r in rows])
        files[name] = dict(cases=len(rows), decisions=sum(len(r['gold']) for r in rows),
                           ids=[r['id'] for r in rows], sha256=digest(labeled), requests_sha256=digest(requests))
    flat = [dict(id=f'{r["id"]}/{d["id"]}', request=dict(state=r['request']['state'], decisions=[d]))
            for r in splits['train'] for d in r['request']['decisions']]
    jsonl(output/'train-token-requests.jsonl', flat)
    write_json(output/'manifest.json', dict(schema_version=1, seed=SEED, mode='specialist',
        dataset=dataset, revision=revision, source_sha256=source_hashes,
        splits=files, token_requests_sha256=digest(output/'train-token-requests.jsonl'),
        protocol=protocol,
        scope=scope))


def validate_dataset(data):
    """Verify both labels and inference payloads before exporting/training/evaluation."""
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['schema_version'] == 1 and manifest['seed'] == SEED, 'Unsupported dataset schema/seed')
    stated = manifest['protocol']
    require(isinstance(stated, dict) and stated == protocol(stated.get('prompt_layout'), stated.get('prompt_detail')),
            'Unsupported training protocol; prepare data with this package')
    require({'train', 'development', 'test'} <= set(manifest['splits']) <= {'train', 'development', 'test', 'unused'},
            'Invalid split names')
    splits = {}
    for name, meta in manifest['splits'].items():
        labeled, requests = data/f'{name}.jsonl', data/f'{name}-requests.jsonl'
        require(digest(labeled) == meta['sha256'] and digest(requests) == meta['requests_sha256'], 'Dataset hash mismatch')
        rows = read_jsonl(labeled)
        require([r['id'] for r in rows] == meta['ids'] and len(rows) == meta['cases'], 'Dataset IDs/count mismatch')
        require(read_jsonl(requests) == [dict(id=r['id'], request=r['request']) for r in rows],
                'Inference payload/label request mismatch')
        require(sum(len(r['gold']) for r in rows) == meta['decisions'], 'Decision count mismatch')
        splits[name] = rows
    require(all(splits[k] for k in ('train', 'development', 'test')), 'Empty required split')
    check_disjoint(splits)
    flat = [dict(id=f'{r["id"]}/{d["id"]}', request=dict(state=r['request']['state'], decisions=[d]))
            for r in splits['train'] for d in r['request']['decisions']]
    require(digest(data/'train-token-requests.jsonl') == manifest['token_requests_sha256'] and
            read_jsonl(data/'train-token-requests.jsonl') == flat, 'Training request binding mismatch')
    return manifest


def main():
    p = argparse.ArgumentParser(prog="l2s1-train prepare", description=__doc__)
    p.add_argument('--source', type=Path, help='Pinned typed-decisions parquet source (pilot reproduction)')
    p.add_argument('--train', type=Path, help='Custom Jev JSONL training split')
    p.add_argument('--development', type=Path, help='Separate development JSONL split')
    p.add_argument('--test', type=Path, help='Separate final test JSONL split')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt-layout', choices=PROMPT_LAYOUTS, default='legacy',
                   help='Prompt layout the application deploys (recorded in the protocol)')
    p.add_argument('--prompt-detail', choices=PROMPT_DETAILS, default='minimal',
                   help='Prompt detail the application deploys (recorded in the protocol)')
    a = p.parse_args()
    prompt = dict(layout=a.prompt_layout, detail=a.prompt_detail)
    if a.source:
        if any((a.train, a.development, a.test)):
            p.error('--source cannot be combined with custom splits')
        prepare(a.source, a.output, **prompt)
    else:
        if not all((a.train, a.development, a.test)):
            p.error('Provide --source or all of --train, --development, --test')
        prepare_custom(a.train, a.development, a.test, a.output, **prompt)


if __name__ == '__main__':
    main()
