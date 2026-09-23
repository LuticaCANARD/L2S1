#!/usr/bin/env python3
"""Small post-hoc label-order diagnostic; not an additional accuracy benchmark."""
import copy
import json
import os
from pathlib import Path
import subprocess
from kaggle_airline import MODELS, OPTIONS, ROOT
from kaggle_ag_news import sha256


def main():
    folder=ROOT/'results/kaggle-airline-20260922'
    out=folder/'option-order-probe'
    out.mkdir(exist_ok=False)
    selection=json.loads((folder/'selection.json').read_text())
    cases=[json.loads(s) for s in (folder/'validation.jsonl').read_text().splitlines()]
    chosen=[]
    for label in OPTIONS:
        chosen.extend([c for c in cases if selection['labels'][c['id']]==label][:4])
    requests=[]
    labels={}
    for rotation in range(3):
        for case in chosen:
            row=copy.deepcopy(case)
            row['id']=case['id']+f'-rotation-{rotation}'
            options=row['request']['decisions'][0]['kind']['options']
            row['request']['decisions'][0]['kind']['options']=options[rotation:]+options[:rotation]
            labels[row['id']]=selection['labels'][case['id']]
            requests.append(row)
    inputs=out/'requests.jsonl'
    inputs.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in requests))
    (out/'selection.json').write_text(json.dumps(dict(labels=labels,selected_ids=[r['id'] for r in chosen],
        request_sha256=sha256(inputs),scope='Post-hoc diagnostic: first four frozen validation cases per class, all three cyclic label orders; no calibration or model selection.'),indent=2)+'\n')
    binary=ROOT/'results/tuning-20260922/bin/evaluate_jsonl-optimized'
    env=os.environ.copy()
    env['LD_LIBRARY_PATH']='/home/lutica/personal/Openweight-Test/llama.cpp/build-cuda/bin'
    for key in ['GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_GRAPH_OPT']:
        env.pop(key,None)
    summary={}
    for model in ['gemma3','qwen3-0.6b']:
        output=out/(model+'.jsonl')
        cmd=[str(binary),'--model',str(ROOT/'models'/MODELS[model]),'--input',str(inputs),'--output',str(output),'--cuda','--warmup']
        with (out/(model+'.log')).open('w') as f:
            subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=120,check=True)
        rows=[json.loads(s) for s in output.read_text().splitlines()]
        assert len(rows)==36 and {r['id'] for r in rows}==set(labels)
        summary[model]=[]
        for rotation in range(3):
            subset=[r for r in rows if r['id'].endswith(f'rotation-{rotation}')]
            first_code=correct=0
            for row in subset:
                result=row['response']['results'][0]
                assert not result['truncated']
                top=max(result['scores'],key=lambda s:s['option_probability'])
                unique=sum(abs(s['option_probability']-top['option_probability'])<1e-12 for s in result['scores'])==1
                first_code+=unique and top['code']=='A'
                correct+=unique and top['id']==labels[row['id']]
            summary[model].append(dict(rotation=rotation,first_label=list(OPTIONS)[rotation],cases=12,first_code_predictions=first_code,correct=correct))
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
