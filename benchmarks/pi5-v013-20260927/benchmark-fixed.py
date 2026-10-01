import asyncio,datetime,hashlib,json,math,os,platform,statistics,subprocess,time
from pathlib import Path
from l2s1 import L2S1,LoadOptions,DecisionRequest,__version__
ROOT=Path(__file__).resolve().parent
MODEL=Path('/home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf')
SUITE=ROOT/'decision_benchmark.json'
def write(name,value): (ROOT/name).write_text(json.dumps(value,indent=2)+'\n')
def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def sensor(pid):
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    status=dict(line.split(':',1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
    throttle=subprocess.check_output(['vcgencmd','get_throttled'],text=True).strip()
    return dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),temperature_c=int(Path('/sys/class/thermal/thermal_zone0/temp').read_text())/1000,throttled=throttle,rss_kib=int(status.get('VmRSS','0 kB').split()[0]),hwm_kib=int(status.get('VmHWM','0 kB').split()[0]),swap_kib=int(mem['SwapTotal'].split()[0])-int(mem['SwapFree'].split()[0]))
async def monitor(pid,stop,records,mode):
    with (ROOT/f'{mode}-telemetry.jsonl').open('x') as log:
        while not stop.is_set():
            try:
                sample=await asyncio.to_thread(sensor,pid);records.append(sample);log.write(json.dumps(sample)+'\n');log.flush()
            except FileNotFoundError:break
            try:await asyncio.wait_for(stop.wait(),timeout=1)
            except TimeoutError:pass
async def run(mode,cases):
    samples=[];records=[];responses=[];warmup=[];stop=asyncio.Event()
    with (ROOT/f'{mode}-stderr.log').open('x') as stderr:
        started=time.perf_counter()
        engine=await L2S1.load(LoadOptions(model=str(MODEL),runtime_dir=ROOT/'runtime/package',context=2048,batch=256,ubatch=256,threads=4,execution_mode='fresh' if mode=='fresh' else 'prefix-reuse',fixed_schema=mode=='fixed',on_stderr=lambda s:stderr.write(s+'\n')))
        load_ms=(time.perf_counter()-started)*1000
        task=asyncio.create_task(monitor(engine._backend._child.pid,stop,records,mode))
        try:
            # One complete warmup pass, then three measured passes over changing states.
            for pass_id in range(-1,2):
                for case in (cases[:1] if pass_id < 0 else cases):
                    start=time.perf_counter();result=await engine.decide(DecisionRequest.model_validate(case['request']));elapsed=(time.perf_counter()-start)*1000
                    row=dict(pass_id=pass_id,case_id=case['id'],elapsed_ms=elapsed,response=result.model_dump(mode='json'))
                    if pass_id<0:warmup.append(row)
                    else:responses.append(row);samples.append(elapsed)
                print(json.dumps(dict(mode=mode,completed_pass=pass_id,requests=len(responses))),flush=True)
            records.append(await asyncio.to_thread(sensor,engine._backend._child.pid))
        finally:
            stop.set();await task;await engine.close()
    write(f'{mode}-responses.json',responses);write(f'{mode}-warmup.json',warmup)
    expected={c['id']:c['expected'] for c in cases}
    n=correct=accepted=accepted_correct=reused=total=0;baseline={};changed=0
    for row in responses:
        for result in row['response']['results']:
            scores=result['evidence']['scores'];top=max(scores,key=lambda s:s['option_probability'])['id'];good=top==expected[row['case_id']][result['id']]
            n+=1;correct+=good;ok=result['status']=='selected';accepted+=ok;accepted_correct+=ok and good
            reused+=result['usage'].get('reused_prefix_tokens') or 0;total+=result['usage'].get('input_tokens') or 0
            key=(row['case_id'],result['id'])
            if key in baseline:changed+=baseline[key]!=result['evidence']
            else:baseline[key]=result['evidence']
    values=sorted(samples)
    summary=dict(mode=mode,load_ms=load_ms,measured_requests=len(samples),decisions=n,p50_ms=statistics.median(values),p95_ms=values[math.ceil(len(values)*.95)-1],raw_top1=correct/n,coverage=accepted/n,accepted_accuracy=accepted_correct/accepted if accepted else None,reused_prefix_tokens=reused,input_tokens=total,repeat_evidence_changes=changed,peak_native_rss_gib=max(s['hwm_kib'] for s in records)/1024**2,max_temperature_c=max(s['temperature_c'] for s in records),max_swap_mib=max(s['swap_kib'] for s in records)/1024,active_undervoltage_samples=sum(bool(int(s['throttled'].split('=')[1],16)&1) for s in records),active_throttle_samples=sum(bool(int(s['throttled'].split('=')[1],16)&4) for s in records),telemetry_samples=len(records))
    write(f'{mode}-summary.json',summary);print(json.dumps(summary),flush=True)
    return summary
async def main():
    assert __version__=='0.1.3'
    identity=dict(sdk_version=__version__,model=MODEL.name,model_sha256=digest(MODEL),suite_sha256=digest(SUITE),runtime=json.loads((ROOT/'runtime/package/runtime-manifest.json').read_text()),host=platform.uname()._asdict(),device=Path('/proc/device-tree/model').read_text().rstrip('\0'),scope='Physical Pi, official published binary, 3 warehouse cases x 2 passes per mode; repeated passes are not independent quality evidence. Latency includes local Python stdio round trip; loading and one warmup request excluded. Mode order fresh then fixed; not a randomized comparison. SIMD enabled in both modes; no isolated SIMD speedup claim.')
    write('identity.json',identity)
    cases=json.loads(SUITE.read_text())['cases'][:3]
    summaries=[await run(mode,cases) for mode in ('fixed',)]
    write('summary.json',dict(runs=summaries,scope='fixed mode only; combine with preserved fresh responses using report.py'))
asyncio.run(main())
