#!/usr/bin/env python3
"""Evaluate all official intent labels with automatically sized answer codes."""
import argparse
import collections
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from evaluate_intents import load, rows, save, save_rows, sha, request, quantile


def prepare(root, base):
    source = base/'prepared'
    for name, expected in load(source/'manifest.json')['prepared_sha256'].items():
        assert sha(source/name) == expected
    out = root/'prepared'
    out.mkdir(exist_ok=False)
    for name in ['samples.jsonl', 'gold.jsonl', 'datasets.json']:
        shutil.copy2(source/name, out/name)
    samples, datasets = rows(out/'samples.jsonl'), load(out/'datasets.json')
    requests = [request(item, datasets[item['dataset']]['labels'], '') for item in samples]
    assert len(requests) == 400
    save_rows(out/'requests.jsonl', requests)
    save(out/'manifest.json', dict(base_manifest=load(source/'manifest.json'),
        method='All 77/60 labels in one decision; fixed-width codes, complete canonical token-sequence likelihoods',
        prepared_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}))


def code(index, count):
    width = 1
    while count > 26**width:
        width += 1
    chars = ['A']*width
    for pos in range(width-1, -1, -1):
        chars[pos] = chr(ord('A') + index % 26)
        index //= 26
    return ''.join(chars)


def score(root, out):
    gold = {r['id']: r for r in rows(root/'prepared/gold.jsonl')}
    requests = {r['id']: r for r in rows(root/'prepared/requests.jsonl')}
    predictions = rows(out/'predictions.jsonl')
    assert len(predictions) == len(gold) == 400 and {p['id'] for p in predictions} == set(gold)
    details = []
    for p in predictions:
        assert not p.get('error'), (p['id'], p.get('error'))
        r = p['response']['results'][0]
        backend = p['response']['backend']
        assert 'RTX 3060' in backend['offload_device'] and backend['offload_requested']
        assert backend['execution_mode'] == 'fresh' and backend['prompt_layout'] == 'legacy'
        assert backend['compute'] == dict(batch=256, ubatch=256, threads=4, context=8192, flash_attention='off')
        assert backend['prompt_version'].endswith('/fixed-width-code-sequences-v1')
        assert r['scoring_method'] == 'code_sequence_conditional_softmax_v1' and not r['truncated']
        item = gold[p['id']]
        scores = r['scores']
        expected_labels = [o['id'] for o in requests[p['id']]['request']['decisions'][0]['kind']['options']]
        assert [s['id'] for s in scores] == expected_labels
        assert len(scores) == (77 if item['dataset'] == 'banking77-en' else 60)
        probabilities = [s['option_probability'] for s in scores]
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities) and abs(sum(probabilities)-1)<1e-8
        assert all(s['code'] == code(i,len(scores)) for i,s in enumerate(scores))
        assert all(s['token_ids'] and s['token_id'] == (s['token_ids'][0] if len(s['token_ids'])==1 else -1) for s in scores)
        maximum = max(s['raw_logit'] for s in scores)
        norm = sum(math.exp(s['raw_logit']-maximum) for s in scores)
        for s in scores:
            assert abs(s['option_probability']-math.exp(s['raw_logit']-maximum)/norm)<1e-9
        assert abs(r['candidate_mass']-sum(math.exp(s['raw_logit']) for s in scores))<1e-8
        picked = min(scores, key=lambda s: (-s['option_probability'],s['id']))['id']
        accepted = r['value']['selected'] is not None
        assert not accepted or r['value']['selected'] == picked
        policy = p['response']['policy']
        assert policy == dict(min_top_probability=0.8,min_candidate_mass=0.05)
        should_accept = max(probabilities)>=0.8 and r['candidate_mass']>=0.05 and sum(abs(v-max(probabilities))<1e-12 for v in probabilities)==1
        assert accepted == should_accept
        details.append(dict(id=p['id'], dataset=item['dataset'], expected=item['expected'], predicted=picked,
            correct=picked==item['expected'], accepted=accepted, confidence=max(probabilities),
            brier=sum((s['option_probability']-(s['id']==item['expected']))**2 for s in scores),
            elapsed_ms=p['elapsed_ms'], input_tokens=r['input_tokens'],
            code_prefix_evaluations=r['code_prefix_evaluations'],
            token_lengths=sorted(set(len(s['token_ids']) for s in scores))))
    save_rows(out/'scored.jsonl', details)
    summary = {}
    for name in ('banking77-en','massive-ko'):
        records = [r for r in details if r['dataset']==name]
        correct=sum(r['correct'] for r in records)
        accepted=[r for r in records if r['accepted']]
        accepted_correct=sum(r['correct'] for r in accepted)
        bins=collections.defaultdict(list)
        for r in records: bins[min(int(r['confidence']*10),9)].append(r)
        n=len(records); assert n==200
        acc=correct/n; z=1.959963984540054
        center=(acc+z*z/(2*n))/(1+z*z/n); half=z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n))/(1+z*z/n)
        summary[name]=dict(total=n,correct=correct,accuracy=acc,wilson95=[center-half,center+half],
            accepted=len(accepted),accepted_correct=accepted_correct,accepted_wrong=len(accepted)-accepted_correct,
            accepted_accuracy=accepted_correct/len(accepted) if accepted else None,coverage=len(accepted)/n,
            abstained=n-len(accepted), errors=0,truncated=0,
            brier=sum(r['brier'] for r in records)/n,
            ece=sum(abs(sum(r['confidence']-r['correct'] for r in group)) for group in bins.values())/n,
            p50_ms=quantile([r['elapsed_ms'] for r in records],.5),p95_ms=quantile([r['elapsed_ms'] for r in records],.95),
            max_input_tokens=max(r['input_tokens'] for r in records),
            prefix_evaluations=sorted(set(r['code_prefix_evaluations'] for r in records)),
            candidate_token_lengths=sorted({v for r in records for v in r['token_lengths']}))
    save(out/'summary.json',summary)
    return summary


def run(root, evaluator, plan):
    for name, expected in load(root/'prepared/manifest.json')['prepared_sha256'].items():
        assert sha(root/'prepared'/name)==expected
    for model in load(plan):
        out=root/'runs'/model['id']; out.mkdir(parents=True,exist_ok=False)
        assert sha(Path(model['path']))==model['sha256']
        command=[str(evaluator),'--model',model['path'],'--input',str(root/'prepared/requests.jsonl'),
            '--output',str(out/'predictions.jsonl'),'--context','8192','--batch','256','--ubatch','256',
            '--threads','4','--execution-mode','fresh','--prompt-layout','legacy','--warmup','--cuda']
        save(out/'manifest.json',dict(model=model,evaluator_sha256=sha(evaluator),
            requests_sha256=sha(root/'prepared/requests.jsonl'),prepared_manifest_sha256=sha(root/'prepared/manifest.json'),
            script_sha256=sha(Path(__file__)),command=command))
        print('START',model['id'],flush=True); started=time.time()
        with (out/'inference.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
        summary=score(root,out)
        save(out/'complete.json',dict(elapsed_s=time.time()-started,examples=400))
        print('COMPLETE',model['id'],json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','run','score']); p.add_argument('--root',type=Path,required=True)
    p.add_argument('--base',type=Path); p.add_argument('--evaluator',type=Path); p.add_argument('--plan',type=Path)
    args=p.parse_args(); root=args.root.resolve()
    if args.mode=='prepare': prepare(root,args.base.resolve())
    elif args.mode=='run': run(root,args.evaluator.resolve(),args.plan)
    else:
        for out in sorted((root/'runs').iterdir()): print(out.name,json.dumps(score(root,out)))
