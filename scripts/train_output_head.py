#!/usr/bin/env python3
"""Fit linear heads using frozen Q8 features; test labels never select a model."""
import argparse, copy, json, time
from pathlib import Path
import torch
from calibrate_ag_news import fit_temperature, metrics
from kaggle_ag_news import sha256

def read(path): return [json.loads(s) for s in path.read_text().splitlines()]
def matrix(rows,kind,options):
    if kind=='hidden': x=[r['hidden'] for r in rows]
    else: x=[[next(s['raw_logit'] for s in r['response']['results'][0]['scores'] if s['id']==o) for o in options] for r in rows]
    result=torch.tensor(x,dtype=torch.float64)
    assert torch.isfinite(result).all()
    return result

def scored(rows,logits,options):
    out=[]
    for r,z in zip(rows,logits.tolist()):
        c=copy.deepcopy({k:v for k,v in r.items() if k!='hidden'})
        for s in c['response']['results'][0]['scores']: s['raw_logit']=z[options.index(s['id'])]
        out.append(c)
    return out

def train(data,features,output):
    torch.set_num_threads(4);torch.manual_seed(20260924)
    manifest=json.loads((data/'selection.json').read_text());labels=manifest['labels']
    rows={k:read(features/(k+'.jsonl')) for k in ['train','dev','calibration']}
    requests=read(data/'train.jsonl');decision=requests[0]['request']['decisions'][0]
    options=[o['id'] for o in decision['kind']['options']]
    for name,rs in rows.items():
        assert [r['id'] for r in rs]==manifest['splits'][name]['ids']
        assert sha256(data/(name+'.jsonl'))==manifest['splits'][name]['sha256']
        assert all(r['response']['backend']==rows['train'][0]['response']['backend'] for r in rs)
    y={k:torch.tensor([options.index(labels[r['id']]) for r in rs]) for k,rs in rows.items()}
    backend=rows['train'][0]['response']['backend'];output.mkdir(parents=True,exist_ok=False)
    summary={};start=time.monotonic()
    # Identity affine head delivers baseline temperature through the same opt-in API.
    for kind in ['base_temperature','logit_affine','hidden']:
        feature_kind='logit_affine' if kind=='base_temperature' else kind
        x={k:matrix(rs,feature_kind,options) for k,rs in rows.items()}
        grid=[]
        if kind=='base_temperature':
            best_w=torch.eye(len(options),dtype=torch.float64);best_b=torch.zeros(len(options),dtype=torch.float64)
            dev_loss=torch.nn.functional.cross_entropy(x['dev'],y['dev']).item()
        else:
            mean=x['train'].mean(0);std=x['train'].std(0,unbiased=False).clamp_min(1e-6)
            normalized=(x['train']-mean)/std;best=float('inf')
            for l2 in manifest['protocol']['l2_grid']:
                w=torch.zeros((len(options),normalized.shape[1]),dtype=torch.float64,requires_grad=True)
                b=torch.zeros(len(options),dtype=torch.float64,requires_grad=True)
                opt=torch.optim.LBFGS([w,b],lr=1.,max_iter=200,tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn='strong_wolfe')
                def closure():
                    opt.zero_grad();loss=torch.nn.functional.cross_entropy(normalized@w.T+b,y['train'])+l2/2*w.square().sum();loss.backward();return loss
                opt.step(closure)
                folded_w=w.detach()/std;folded_b=b.detach()-folded_w@mean
                loss=torch.nn.functional.cross_entropy(x['dev']@folded_w.T+folded_b,y['dev']).item()
                grid.append(dict(l2=l2,dev_nll=loss,iterations=opt.state[w]['n_iter']))
                if loss<best: best=loss;best_w=folded_w.clone();best_b=folded_b.clone();chosen_l2=l2
            dev_loss=best
        cal=scored(rows['calibration'],x['calibration']@best_w.T+best_b,options)
        temp=fit_temperature(cal,labels)
        artifact=dict(version=1,id='airline-q8-output-20260923-'+kind,feature_kind=feature_kind,
            model_sha256=manifest['protocol']['model_sha256'],device=backend.get('offload_device','CPU'),compute=backend['compute'],prompt_version=backend['prompt_version'],
            decision_id=decision['id'],instruction=decision['instruction'],options=decision['kind']['options'],weights=best_w.tolist(),bias=best_b.tolist(),temperature=temp)
        (output/(kind+'.json')).write_text(json.dumps(artifact,indent=2)+'\n')
        summary[kind]=dict(dev_nll=dev_loss,temperature=temp,grid=grid,parameters=best_w.numel()+best_b.numel(),
            calibration=metrics(cal,labels,temp),chosen_l2=None if kind=='base_temperature' else chosen_l2)
        print(kind,summary[kind],flush=True)
    chosen=min(['logit_affine','hidden'],key=lambda k:summary[k]['dev_nll'])
    summary['selection']=dict(method=chosen,criterion='lowest development NLL before temperature',training_seconds=time.monotonic()-start,
        data_manifest_sha256=sha256(data/'selection.json'),feature_sha256={k:sha256(features/(k+'.jsonl')) for k in rows},
        script_sha256=sha256(Path(__file__)))
    (output/'training.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'selected.json').write_bytes((output/(chosen+'.json')).read_bytes())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--features',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();train(a.data,a.features,a.output)
