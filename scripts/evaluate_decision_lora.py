#!/usr/bin/env python3
"""Paired local Q8 deployment check with an explicit F32 GGUF LoRA adapter."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from kaggle_ag_news import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('models/gemma-4-E2B-it-Q8_0.gguf'))
    p.add_argument('--binary',type=Path,default=Path('target/release/examples/evaluate_jsonl'))
    p.add_argument('--runtime',type=Path,default=Path('/home/lutica/personal/Openweight-Test/llama.cpp/build-cuda/bin'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env['LD_LIBRARY_PATH']=str(a.runtime.resolve())
    for key in ['GGML_CUDA_CUBLAS_COMPUTE_TYPE','GGML_CUDA_DISABLE_GRAPHS','GGML_CUDA_DISABLE_FUSION','GGML_CUDA_GRAPH_OPT']:env.pop(key,None)
    provenance=dict(model_sha256=sha256(a.model),adapter_sha256=sha256(a.adapter),binary_sha256=sha256(a.binary),
                    script_sha256=sha256(Path(__file__)),runtime={f.name:sha256(f) for f in a.runtime.glob('*.so')},runs=[])
    def run(name,request,adapter):
        output=a.output/(name+'.jsonl')
        if output.exists():raise FileExistsError(output)
        cmd=[str(a.binary.resolve()),'--model',str(a.model.resolve()),'--input',str(request.resolve()),'--output',str(output.resolve()),
             '--cuda','--context','2048','--batch','256','--ubatch','256','--threads','4','--flash-attention','off',
             '--execution-mode','fresh','--request-batch-size','1','--warmup']
        if adapter:cmd+=['--lora',str(a.adapter.resolve())]
        print('START',name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        provenance['runs'].append(dict(name=name,command=cmd,exit_code=result.returncode,request_sha256=sha256(request)))
        (a.output/'adapter-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
        result.check_returncode()
        rows=[json.loads(s) for s in output.read_text().splitlines()]
        assert all('response' in row and not row['response']['results'][0]['truncated'] for row in rows)
        assert all(bool(row['response']['backend'].get('lora_path'))==adapter for row in rows)
        return rows
    for split in ['calibration','test','probe']:
        run('lora-'+split,a.data/(split+'.jsonl'),True)
    run('lora-ag-news',Path('results/kaggle-ag-news/requests.jsonl'),True)
    timings={'base':[],'lora':[]};reference={}
    for rep in range(1,4):
        for kind in (['base','lora'] if rep%2 else ['lora','base']):
            rows=run(f'timing-{kind}-{rep}',a.data/'test.jsonl',kind=='lora')
            scores={r['id']:[s['raw_logit'] for s in r['response']['results'][0]['scores']] for r in rows}
            if kind in reference:assert scores==reference[kind], 'Scores drifted between repeated measurements'
            reference[kind]=scores
            elapsed=[r['batch_elapsed_ms'] for r in rows]
            timings[kind].append(dict(total_ms=sum(elapsed),p50_ms=statistics.median(elapsed)))
    (a.output/'timings.json').write_text(json.dumps(timings,indent=2)+'\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
