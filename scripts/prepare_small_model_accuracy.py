#!/usr/bin/env python3
"""Prepare an exact-state-disjoint audit split; does not establish training disjointness.

Requires pyarrow. --prior is results/accuracy-20260929, --test-requests is
results/ollaya-comparison-20260928/requests.jsonl. Inputs retain source order.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def state_hash(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def convert(state, questions):
    decisions = []
    for key, question in questions.items():
        criteria = question.get('criteria')
        match question['type']:
            case 'choice':
                kind = dict(type='choice', options=[dict(id=k, criterion=v) for k,v in criteria.items()])
            case 'noul':
                criteria = criteria or {}
                kind = dict(type='binary', false_label=criteria.get('false', 'false'),
                            true_label=criteria.get('true', 'true'))
            case 'score':
                assert isinstance(criteria, list)
                kind = dict(type='ordinal', levels=[dict(id=str(i), criterion=v, value=i)
                                                    for i,v in enumerate(criteria)])
            case _:
                raise ValueError(question['type'])
        decisions.append(dict(id=key, instruction=question.get('instructions') or '', kind=kind))
    return dict(state=state, decisions=decisions)


def main():
    import pyarrow.parquet as pq
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior', type=Path, required=True)
    parser.add_argument('--test-requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dev = read(args.prior/'dev-requests.jsonl')
    excluded = dev + read(args.prior/'calibration-requests.jsonl') + read(args.test_requests)
    seen = {state_hash(r['l2s1']['state']) for r in excluded}
    groups = defaultdict(list)
    for row in sorted(pq.read_table(args.prior/'train.parquet').to_pylist(),
                      key=lambda r: hashlib.sha256(('small-accuracy-20261001/'+r['id']).encode()).hexdigest()):
        h = state_hash(json.loads(row['state']))
        if h in seen:
            continue
        seen.add(h)
        groups[row['workflow']].append(row)
    selected = []
    for workflow, candidates in sorted(groups.items()):
        assert len(candidates) >= 50, workflow
        selected.extend(candidates[:50])
    datasets = {
        'dev-requests': [dict(id=r['id'], request=r['l2s1']) for r in dev],
        'dev-gold': read(args.prior/'dev-gold.jsonl'),
        'holdout-requests': [dict(id=r['id'], request=convert(json.loads(r['state']), json.loads(r['questions']))) for r in selected],
        'holdout-gold': [dict(id=r['id'], gold=json.loads(r['gold'])) for r in selected],
    }
    for name, records in datasets.items():
        with (args.output/(name+'.jsonl')).open('x') as out:
            for record in records:
                out.write(json.dumps(record, ensure_ascii=False)+'\n')
    print({name: len(records) for name, records in datasets.items()})


if __name__ == '__main__':
    main()
