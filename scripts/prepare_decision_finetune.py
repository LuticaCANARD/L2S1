#!/usr/bin/env python3
"""Freeze a disjoint airline decision pilot before any training/evaluation."""
import argparse
import collections
import copy
import csv
import io
import json
from pathlib import Path
import random
import zipfile
from kaggle_airline import ARCHIVE_SHA256, INSTRUCTION, OPTIONS, identity
from kaggle_ag_news import sha256


def split_rows(rows, excluded_texts, seed=20260923):
    groups = collections.defaultdict(list)
    for index, row in enumerate(rows, 1):
        groups[identity(row['text'])].append((index, row))
    available = {label: [] for label in OPTIONS}
    for key, copies in groups.items():
        if not key or key in excluded_texts or len({r['airline_sentiment'] for _, r in copies}) != 1:
            continue
        index, row = copies[0]
        available[row['airline_sentiment']].append((index, row))
    rng = random.Random(seed)
    splits = {'train': [], 'calibration': [], 'test': []}
    for label, heldout_n in zip(OPTIONS, [134, 133, 133]):
        chosen = rng.sample(available[label], 300 + heldout_n * 2)
        splits['train'].extend(chosen[:300])
        splits['calibration'].extend(chosen[300:300+heldout_n])
        splits['test'].extend(chosen[300+heldout_n:])
    for selected in splits.values():
        rng.shuffle(selected)
    return splits


def prepare(source, output):
    output.mkdir(parents=True, exist_ok=False)
    archive = source/'dataset.zip'
    assert sha256(archive) == ARCHIVE_SHA256
    old = [json.loads(line) for name in ['fit', 'validation'] for line in (source/(name+'.jsonl')).read_text().splitlines()]
    excluded = {identity(c['request']['state']['tweet']) for c in old}
    with zipfile.ZipFile(archive) as z:
        rows = list(csv.DictReader(io.StringIO(z.read('Tweets.csv').decode('utf-8-sig'))))
    splits = split_rows(rows, excluded)
    manifest = dict(seed=20260923, archive_sha256=ARCHIVE_SHA256, excluded_prior_ids=[c['id'] for c in old],
                    labels={}, splits={}, protocol=dict(train_unique=900, train_rotations=3, epochs=1,
                    learning_rate=0.0001, rank=8, alpha=16, dropout=0.0, gradient_accumulation=12,
                    mass_loss_weight=0.1, checkpoint_selection='fixed final epoch; no test-based selection',
                    calibration='fit temperature only on 400 new calibration cases',
                    scope='Single-domain supervised pilot, not RLCD reproduction. Balanced own split; exact normalized deduplication only.'))
    for split, selected in splits.items():
        records=[]
        for rotation in (range(3) if split == 'train' else range(1)):
            for index, row in selected:
                base=f'airline-row-{index:05d}'
                cid=base+f'-r{rotation}'
                labels=list(OPTIONS)
                labels=labels[rotation:]+labels[:rotation]
                records.append(dict(id=cid, request=dict(state={'tweet':row['text']}, decisions=[dict(
                    id='airline_sentiment', instruction=INSTRUCTION, kind=dict(type='choice',options=[dict(id=k,criterion=OPTIONS[k]) for k in labels]))])))
                manifest['labels'][cid]=row['airline_sentiment']
        path=output/(split+'.jsonl')
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        manifest['splits'][split]=dict(ids=[r['id'] for r in records], unique_ids=[f'airline-row-{i:05d}' for i,_ in selected],
            sha256=sha256(path), counts=dict(collections.Counter(row['airline_sentiment'] for _,row in selected)))
    # All three cyclic orders for a fixed 60-case test subset, chosen without model scores.
    test=[json.loads(s) for s in (output/'test.jsonl').read_text().splitlines()]
    probe=[]
    for label in OPTIONS:
        chosen=[c for c in test if manifest['labels'][c['id']]==label][:20]
        for case in chosen:
            for rotation in range(3):
                row=copy.deepcopy(case)
                row['id']=case['id'].removesuffix('-r0')+f'-probe-r{rotation}'
                opts=row['request']['decisions'][0]['kind']['options']
                row['request']['decisions'][0]['kind']['options']=opts[rotation:]+opts[:rotation]
                manifest['labels'][row['id']]=label
                probe.append(row)
    path=output/'probe.jsonl'
    path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probe))
    manifest['splits']['probe']=dict(ids=[r['id'] for r in probe],sha256=sha256(path),scope='60 test cases x 3 orders; not 180 independent cases')
    (output/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:dict(records=len(v['ids']),counts=v.get('counts')) for k,v in manifest['splits'].items()},indent=2))


def seal_tokens(output):
    manifest=json.loads((output/'selection.json').read_text())
    for name,split in manifest['splits'].items():
        tokens_path=output/(name+'-tokens.jsonl')
        rows=[json.loads(s) for s in tokens_path.read_text().splitlines()]
        requests=[json.loads(s) for s in (output/(name+'.jsonl')).read_text().splitlines()]
        assert sha256(output/(name+'.jsonl'))==split['sha256']
        assert [r['id'] for r in rows]==split['ids']
        for row,request in zip(rows,requests):
            options=request['request']['decisions'][0]['kind']['options']
            assert row['option_ids']==[o['id'] for o in options]
            assert manifest['labels'][row['id']] in row['option_ids']
            assert len(set(row['candidate_ids']))==len(options)
            assert 0<len(row['input_ids'])<=2048
        digest=sha256(tokens_path)
        if 'token_sha256' in split:assert digest==split['token_sha256'], 'Refusing to reseal changed tokens'
        split['token_sha256']=digest
        split['max_input_tokens']=max(len(r['input_ids']) for r in rows)
    (output/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seal-tokens',action='store_true');a=p.parse_args()
    if a.seal_tokens:seal_tokens(a.output)
    elif a.source:prepare(a.source,a.output)
    else:p.error('--source is required when preparing a new split')
