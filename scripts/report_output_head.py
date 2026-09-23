#!/usr/bin/env python3
"""Report the sealed holdout after all head choices; independently check runtime scores."""
import argparse, collections, copy, json, math, statistics
from pathlib import Path
from train_output_head import read, matrix, scored
from calibrate_ag_news import metrics, probabilities
from kaggle_airline import threshold_counts
from kaggle_ag_news import sha256
from report_decision_finetune import prediction
import torch


def report(data,features,heads,runtime,output):
    manifest=json.loads((data/'selection.json').read_text());labels=manifest['labels']
    rows={k:read(features/(k+'.jsonl')) for k in ['test','probe']}
    for name,rs in rows.items():
        assert [r['id'] for r in rs]==manifest['splits'][name]['ids']
        assert sha256(data/(name+'.jsonl'))==manifest['splits'][name]['sha256']
    result={};preds={}
    for kind in ['base_temperature','logit_affine','hidden']:
        h=json.loads((heads/(kind+'.json')).read_text());opts=[o['id'] for o in h['options']]
        w=torch.tensor(h['weights'],dtype=torch.float64);b=torch.tensor(h['bias'],dtype=torch.float64)
        mapped={k:scored(rs,matrix(rs,h['feature_kind'],opts)@w.T+b,opts) for k,rs in rows.items()}
        checks={}
        for name,rs in mapped.items():
            path=runtime/(kind+'-'+name+'.jsonl')
            if path.exists():
                live=read(path);assert [r['id'] for r in live]==[r['id'] for r in rs]
                errors=[];prob_errors=[]
                for a,e in zip(live,rs):
                    actual=a['response']['results'][0];expected=e['response']['results'][0]
                    assert actual['calibration_id']==h['id']
                    assert actual['candidate_mass']==expected['candidate_mass']
                    p,_=probabilities([s['raw_logit'] for s in expected['scores']],h['temperature'])
                    for sa,se,pe in zip(actual['scores'],expected['scores'],p):
                        assert sa['id']==se['id'];errors.append(abs(sa['raw_logit']-se['raw_logit']));prob_errors.append(abs(sa['option_probability']-pe))
                assert max(errors)<1e-5 and max(prob_errors)<1e-6
                checks[name]=dict(max_logit_error=max(errors),max_probability_error=max(prob_errors))
                assert sum(not r['response']['results'][0]['abstention_reasons'] for r in live)==metrics(rs,labels,h['temperature'])['accepted']
        tests=mapped['test'];preds[kind]={r['id']:prediction(r) for r in tests}
        grouped=collections.defaultdict(list)
        for r in mapped['probe']: grouped[r['id'].rsplit('-probe-r',1)[0]].append(prediction(r))
        ranked=sorted(tests,key=lambda r:max(probabilities([s['raw_logit'] for s in r['response']['results'][0]['scores']],h['temperature'])[0]),reverse=True)
        result[kind]=dict(temperature=h['temperature'],raw=metrics(tests,labels,1),calibrated=metrics(tests,labels,h['temperature']),
            thresholds=[threshold_counts(tests,labels,h['temperature'],t) for t in [.6,.7,.8,.9]],
            probe=dict(unique_cases=len(grouped),consistent_cases=sum(None not in v and len(set(v))==1 for v in grouped.values()),
                correct=sum(prediction(r)==labels[r['id']] for r in mapped['probe']),calls=len(mapped['probe'])),
            fixed_coverage_diagnostic={str(f):metrics(ranked[:round(len(tests)*f)],labels,h['temperature'])['raw_top1'] for f in [.6,.8]},
            runtime_checks=checks)
    base=preds['base_temperature']
    for kind in ['logit_affine','hidden']:
        wins=sum(base[i]!=labels[i] and p==labels[i] for i,p in preds[kind].items())
        losses=sum(base[i]==labels[i] and p!=labels[i] for i,p in preds[kind].items());n=wins+losses
        pvalue=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/(2**n)) if n else 1.
        result[kind]['paired_vs_base']=dict(wins=wins,losses=losses,mcnemar_exact_p=pvalue)
    result['provenance']=dict(selection_sha256=sha256(data/'selection.json'),head_sha256={k:sha256(heads/(k+'.json')) for k in preds},
        test_features_sha256=sha256(features/'test.jsonl'))
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['data','features','heads','runtime','output']: p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();report(a.data,a.features,a.heads,a.runtime,a.output)
