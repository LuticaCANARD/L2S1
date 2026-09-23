#!/usr/bin/env python3
"""Freeze new train/dev/calibration/test texts before fitting a deployment head."""
import argparse, collections, copy, csv, io, json, random, zipfile
from pathlib import Path
from kaggle_airline import ARCHIVE_SHA256, INSTRUCTION, OPTIONS, identity, SOURCE
from kaggle_ag_news import sha256


def prepare(source, previous, output):
    archive=source/'dataset.zip'
    assert sha256(archive)==ARCHIVE_SHA256
    excluded={identity(json.loads(s)['request']['state']['tweet']) for folder,names in
              [(source,['fit','validation']),(previous,['train','calibration','test'])]
              for name in names for s in (folder/(name+'.jsonl')).read_text().splitlines()}
    with zipfile.ZipFile(archive) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('Tweets.csv').decode('utf-8-sig'))))
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows,1): groups[identity(r['text'])].append((i,r))
    available={k:[] for k in OPTIONS}
    for key,copies in groups.items():
        if key and key not in excluded and len({r['airline_sentiment'] for _,r in copies})==1:
            available[copies[0][1]['airline_sentiment']].append(copies[0])
    rng=random.Random(20260924);splits={k:[] for k in ['train','dev','calibration','test']}
    for label,n in zip(OPTIONS,[134,133,133]):
        chosen=rng.sample(available[label],400+2*n)
        for name,lo,hi in [('train',0,300),('dev',300,400),('calibration',400,400+n),('test',400+n,400+2*n)]:
            splits[name].extend(chosen[lo:hi])
    output.mkdir(parents=True,exist_ok=False)
    manifest=dict(seed=20260924,source=SOURCE,archive_sha256=ARCHIVE_SHA256,excluded_prior_unique=len(excluded),
        labels={},splits={},protocol=dict(model='gemma-4-E2B-it-Q8_0.gguf',
        model_sha256='996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a',
        methods=['base_temperature','logit_affine','hidden'],l2_grid=[.001,.01,.1,1.],
        objective='mean cross entropy + L2/2 * squared standardized weights; CPU float64 LBFGS max_iter=200',
        standardization='train only; std clamp 1e-6; fold into exported weights',
        hyperparameter_selection='lowest dev NLL before temperature; never test',
        calibration='temperature fit only on independent 400 calibration cases',
        deployment_selection='lowest dev NLL across trained heads; explicit opt-in',
        scope='Balanced task-specific head. Normalized exact dedup, not semantic dedup. Not RLCD. Base candidate mass gate preserved.'))
    records={}
    for name,selected in splits.items():
        rng.shuffle(selected);records[name]=[]
        for rot in (range(3) if name=='train' else range(1)):
            for i,r in selected:
                cid=f'airline-row-{i:05d}-r{rot}';opts=list(OPTIONS);opts=opts[rot:]+opts[:rot]
                records[name].append(dict(id=cid,request=dict(state={'tweet':r['text']},decisions=[dict(
                    id='airline_sentiment',instruction=INSTRUCTION,kind=dict(type='choice',options=[dict(id=k,criterion=OPTIONS[k]) for k in opts]))])))
                manifest['labels'][cid]=r['airline_sentiment']
    records['probe']=[]
    for label in OPTIONS:
        for case in [c for c in records['test'] if manifest['labels'][c['id']]==label][:20]:
            for rot in range(3):
                c=copy.deepcopy(case);c['id']=case['id'].removesuffix('-r0')+f'-probe-r{rot}'
                opts=c['request']['decisions'][0]['kind']['options'];c['request']['decisions'][0]['kind']['options']=opts[rot:]+opts[:rot]
                records['probe'].append(c);manifest['labels'][c['id']]=label
    for name,items in records.items():
        path=output/(name+'.jsonl');path.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in items))
        manifest['splits'][name]=dict(ids=[c['id'] for c in items],sha256=sha256(path))
    assert all({identity(r['text']) for _,r in splits[a]}.isdisjoint(identity(r['text']) for _,r in splits[b])
               for a in splits for b in splits if a!=b)
    (output/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print({k:len(v) for k,v in records.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.source,a.previous,a.output)
