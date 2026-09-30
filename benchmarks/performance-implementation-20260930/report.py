"""Export compact, auditable measured evidence from completed study directories."""
import argparse
import json
from pathlib import Path
import random
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from benchmark_decision_performance import read_rows, digest, metrics, outcomes, percentile, write_json


def wilson(k,n):
    if not n:return None
    p=k/n;z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
    return [center-half,center+half]


def paired(reference,candidate,gold):
    a,b=outcomes(reference,gold),outcomes(candidate,gold)
    deltas=[int(b[k]['top']==b[k]['expected'])-int(v['top']==v['expected']) for k,v in a.items()]
    rng=random.Random(73003)
    estimates=sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(5000))
    return {'fixed':deltas.count(1),'regressed':deltas.count(-1),'delta':sum(deltas)/len(deltas),
            'item_bootstrap95':[percentile(estimates,.025),percentile(estimates,.975)],
            'scope':'unique synthetic instances or public regression items, not independent task templates'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True)
    args=parser.parse_args()
    study=Path(__file__).resolve().parent
    summary={}; provenance={}; records=[]
    for model in ('e2b','12b'):
        summary[model]={};provenance[model]={}
        for phase in ('accuracy-dev','compute','schema','shared','holdout','jevbench'):
            folder=args.results/model/phase
            if not (folder/'summary.json').exists():continue
            plan=json.loads((folder/'plan.json').read_text())
            runs=json.loads((folder/'runs.json').read_text())
            raw_summary=json.loads((folder/'summary.json').read_text())
            # Follow the actual recorded invocation, including per-model frozen inputs.
            command=runs[0]['command']
            input_path=Path(command[command.index('--input')+1])
            if phase=='jevbench':gold_path=study/'jevbench-gold.json'
            else:gold_path=study/({'accuracy-dev':'dev','compute':'compute-dev','holdout':'holdout'}.get(phase,phase)+'-gold.json')
            gold=json.loads(gold_path.read_text())
            if digest(input_path)!=plan['hashes']['input'] or digest(gold_path)!=plan['hashes']['gold']:
                raise ValueError('dataset changed since execution')
            configs={}
            for config in plan['configs']:
                name=config['name']; group=[r for r in runs if r['config']==name]
                stats=[];peaks=[];cold=[];warm=[];rss=[];hashes={}
                for run in group:
                    path=folder/f"{name}-{run['repeat']}.jsonl"
                    if 'error' in run:raise ValueError(f'incomplete run: {path}')
                    rows=read_rows(path);stats.append(metrics(rows,gold));hashes[path.name]=digest(path)
                    if run.get('sampled_rss_mib') is not None:rss.append(run['sampled_rss_mib'])
                    for sample in json.loads((folder/f"{name}-{run['repeat']}-gpu.json").read_text()):
                        for line in sample['gpus'].splitlines():
                            index,memory,_=line.split(',')
                            if int(index.strip())==0:peaks.append(float(memory.strip()))
                    batches={r['batch_index']:r['batch_elapsed_ms'] for r in rows}
                    cold.append(batches[0]);warm.append(percentile(list(batches.values())[1:],.5) if len(batches)>1 else None)
                    for row in rows:
                        copy={k:v for k,v in row.items() if k in ('id','batch_index','batch_elapsed_ms','batch_profile','resident','facts_sha256','preprocess_ms','inference_ms')}
                        copy['response']={'results':[{k:v for k,v in item.items() if k in ('id','scores','value','abstention_reasons','candidate_mass','input_tokens','reused_prefix_tokens','truncated')} for item in row['response']['results']]}
                        # Raw logits are unnecessary for reproducing these reported metrics.
                        for item in copy['response']['results']:
                            item['scores']=[{k:v for k,v in s.items() if k in ('id','option_probability')} for s in item['scores']]
                        records.append(dict(model=model,phase=phase,config=name,repeat=run['repeat'],record=copy))
                stable=all(all(s[k]==stats[0][k] for k in ('raw_correct','accepted','accepted_correct','wrong_accepted')) for s in stats)
                configs[name]={'args':config['args'],'counts_first_run':{k:stats[0][k] for k in ('cases','decisions','raw_correct','accepted','accepted_correct','wrong_accepted','abstained')},
                               'stable_counts':stable,'runs':stats,'median_p50_ms':percentile([s['p50_ms'] for s in stats],.5),
                               'p50_range_ms':[min(s['p50_ms'] for s in stats),max(s['p50_ms'] for s in stats)],
                               'worst_p95_ms':max(s['p95_ms'] for s in stats),'gpu0_sampled_max_mib':max(peaks) if peaks else None,
                               'process_sampled_max_rss_mib':max(rss) if rss else None,'first_call_ms':cold,'subsequent_p50_ms':warm,
                               'raw_wilson95':wilson(stats[0]['raw_correct'],stats[0]['decisions']),
                               'accepted_wilson95':wilson(stats[0]['accepted_correct'],stats[0]['accepted']),
                               'comparisons':raw_summary[name].get('comparisons'),'source_sha256':hashes}
            section={'configs':configs,'input_sha256':plan['hashes']['input'],'gold_sha256':plan['hashes']['gold']}
            if phase in ('holdout','jevbench') and len(plan['configs'])>1:
                names=[c['name'] for c in plan['configs']]
                section['paired']=paired(read_rows(folder/(names[0]+'-0.jsonl')),read_rows(folder/(names[1]+'-0.jsonl')),gold)
            if (folder/'selection.json').exists():section['selection']=json.loads((folder/'selection.json').read_text())
            if phase in ('accuracy-dev','holdout'):
                section['families']={}
                for family in ('numeric','temporal','graph'):
                    labels={k:v for k,v in gold.items() if '-'+family+'-' in k}
                    section['families'][family]={c['name']:metrics([r for r in read_rows(folder/(c['name']+'-0.jsonl')) if r['id'] in labels],labels) for c in plan['configs']}
            summary[model][phase]=section
            plan['platform'].pop('node',None)
            plan['raw_plan_sha256']=digest(folder/'plan.json')
            provenance[model][phase]=plan
    write_json(study/'summary.json',summary)
    write_json(study/'provenance.json',provenance)
    (study/'records.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
    write_json(study/'records-sha256.json',{'records.jsonl':digest(study/'records.jsonl')})
    print('Exported',len(records),'records')

if __name__=='__main__':main()
