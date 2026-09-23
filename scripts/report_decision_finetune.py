#!/usr/bin/env python3
"""Evaluate fixed LoRA artifacts; temperature fitting accesses calibration only."""
import argparse
import collections
import json
from pathlib import Path
import statistics
from calibrate_ag_news import fit_temperature, metrics, probabilities
from kaggle_airline import threshold_counts


def prediction(row):
    scores=row['response']['results'][0]['scores']
    best=max(s['raw_logit'] for s in scores)
    winners=[s['id'] for s in scores if abs(s['raw_logit']-best)<1e-12]
    return winners[0] if len(winners)==1 else None


def report(data, run, prefixes):
    manifest=json.loads((data/'selection.json').read_text());labels=manifest['labels']
    summary={}
    for prefix in prefixes:
        rows={name:[json.loads(s) for s in (run/(prefix+'-'+name+'.jsonl')).read_text().splitlines()]
              for name in ['calibration','test','probe']}
        for name,values in rows.items():
            assert len(values)==len({r['id'] for r in values})==len(manifest['splits'][name]['ids'])
            assert {r['id'] for r in values}==set(manifest['splits'][name]['ids'])
            assert all(not r['response']['results'][0]['truncated'] for r in values)
        temperature=fit_temperature(rows['calibration'],labels)
        tests=rows['test'];groups=collections.defaultdict(list)
        for row in rows['probe']:
            groups[row['id'].rsplit('-probe-r',1)[0]].append(prediction(row))
        probe_correct=sum(prediction(r)==labels[r['id']] for r in rows['probe'])
        coverage=[]
        ranked=sorted(tests,key=lambda r:max(probabilities([s['raw_logit'] for s in r['response']['results'][0]['scores']],temperature)[0]),reverse=True)
        for fraction in [.6,.8,1.]:
            subset=ranked[:round(len(tests)*fraction)]
            coverage.append(dict(fraction=fraction,accuracy=metrics(subset,labels,temperature)['raw_top1'],n=len(subset),
                                 scope='test confidence ranking, diagnostic only, mass gate not applied'))
        timings=[r.get('elapsed_ms',r.get('batch_elapsed_ms')) for r in tests]
        summary[prefix]=dict(temperature=temperature,raw=metrics(tests,labels,1),calibrated=metrics(tests,labels,temperature),
            thresholds=[threshold_counts(tests,labels,temperature,t) for t in [.6,.7,.8,.9,1]],
            fixed_coverage_diagnostic=coverage,probe=dict(unique_cases=len(groups),calls=len(rows['probe']),correct=probe_correct,
                consistent_cases=sum(None not in x and len(set(x))==1 for x in groups.values()),
                tied_calls=sum(prediction(r) is None for r in rows['probe'])),
            test_ties=sum(prediction(r) is None for r in tests),
            timing=dict(total_ms=sum(timings),p50_ms=statistics.median(timings),scope='forward timing; excludes startup, warmup, IO'))
    (run/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--prefixes',nargs='+',default=['base','lora']);a=p.parse_args();report(a.data,a.run,a.prefixes)
