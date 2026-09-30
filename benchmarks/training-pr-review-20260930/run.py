"""Run the frozen evidence-preservation comparison on two local CUDA models."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo/'scripts'))
from benchmark_decision_performance import digest, metrics, read_rows

p = argparse.ArgumentParser()
p.add_argument('--evaluator', type=Path, required=True)
p.add_argument('--models', type=Path, required=True)
p.add_argument('--llama-source', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
root = Path(__file__).resolve().parent
protocol = json.loads((root/'protocol.json').read_text())
for name, sha in protocol['files'].items():
    assert digest(root/name) == sha, name
models = {'e2b': a.models/'gemma-4-E2B-it-Q8_0.gguf', '12b': a.models/'gemma-4-12b-it-qat-q4_0.gguf'}
summary = dict(protocol_sha256=digest(root/'protocol.json'), evaluator_sha256=digest(a.evaluator),
               models={k: dict(path=v.name, sha256=digest(v)) for k,v in models.items()}, runs=[])
summary['source_sha256'] = {str(f): digest(repo/f) for f in map(Path, [
    'src/llama.rs', 'examples/evaluate_jsonl.rs', 'examples/export_decision_tokens.rs',
    'training/src/l2s1_training/prepare_jev_data.py', 'training/src/l2s1_training/common.py',
    'training/src/l2s1_training/train_jev_lora.py', 'benchmarks/training-pr-review-20260930/run.py'])}
summary['llama_cpp_commit'] = subprocess.check_output(['git','-C',str(a.llama_source),'rev-parse','HEAD'],text=True).strip()
summary['cpu'] = next(s.split(':',1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines() if s.startswith('model name'))
summary['hardware'] = subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'], text=True).strip()
records = []
for phase in ('development', 'test'):
    for name, model in models.items():
        for arm in ('before', 'after'):
            source = root/f'{phase}-before.jsonl' if arm == 'before' else root/'data'/f'{phase}-requests.jsonl'
            target = a.output/f'{name}-{phase}-{arm}.jsonl'
            command = [str(a.evaluator), '--model', str(model), '--input', str(source), '--output', str(target),
                       '--cuda', '--threads','4','--context','4096','--batch','256','--ubatch','256',
                       '--flash-attention','off','--execution-mode','fresh','--prompt-layout','state-first',
                       '--prompt-detail','typed','--request-batch-size','1','--warmup']
            rss_kib = gpu_mib = 0
            with target.with_suffix('.log').open('w') as log:
                started=time.monotonic()
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
                while process.poll() is None:
                    try:
                        status=Path(f'/proc/{process.pid}/status').read_text()
                        for line in status.splitlines():
                            if line.startswith(('VmRSS:', 'VmHWM:')):
                                rss_kib=max(rss_kib,int(line.split()[1]))
                        gpu_mib=max(gpu_mib, int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()))
                    except (FileNotFoundError, ProcessLookupError):
                        pass
                    time.sleep(.25)
                assert process.returncode == 0, target.with_suffix('.log')
            rows=read_rows(target)
            result=dict(model=name,phase=phase,arm=arm,command=command,requests_sha256=digest(source),
                        predictions_sha256=digest(target),wall_seconds=time.monotonic()-started,
                        process_rss_hwm_kib=rss_kib,whole_board_gpu_max_mib=gpu_mib,
                        metrics=metrics(rows,json.loads((root/f'{phase}-gold.json').read_text())))
            summary['runs'].append(result)
            for r in rows:
                results=[]
                for item in r['response']['results']:
                    keep=['id','truncated','candidate_mass','value','abstention_reasons','input_tokens','reused_prefix_tokens']
                    results.append(dict({k:item[k] for k in keep},scores=[{k:s[k] for k in ('id','option_probability')} for s in item['scores']]))
                compact={k:r[k] for k in ('id','batch_elapsed_ms','batch_index','batch_profile')}
                compact['response']=dict(results=results)
                records.append(dict(model=name,phase=phase,arm=arm,record=compact))
            (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            print(name,phase,arm,json.dumps(result['metrics']),flush=True)
(a.output/'records.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
