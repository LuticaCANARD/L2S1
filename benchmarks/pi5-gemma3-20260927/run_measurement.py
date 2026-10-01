"""Measure the native rules suite on the Pi and retain power/RAM observations."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

ROOT=Path.home()/'l2s1-pi-20260927'
SOURCE=ROOT/'source'
OUT=ROOT/'runs'/'gemma3-q8-rules'
MODEL=ROOT/'models/gemma-3-1b-it-Q8_0.gguf'
EXPECTED='b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5'

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None

def command(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=5)
    return r.stdout.strip()

def sensor(pid=None):
    mem=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    temp=read('/sys/class/thermal/thermal_zone0/temp')
    info=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        temperature_c=int(temp)/1000 if temp else None,
        available_kib=int(mem['MemAvailable'].split()[0]),
        swap_used_kib=int(mem['SwapTotal'].split()[0])-int(mem['SwapFree'].split()[0]),
        frequency_khz=read('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq'),
        throttled=command(['vcgencmd','get_throttled']))
    if pid:
        status=read(f'/proc/{pid}/status') or ''
        values=dict(line.split(':',1) for line in status.splitlines() if ':' in line)
        for name in ('VmRSS','VmHWM','VmSwap'):
            info[name+'_kib']=int(values.get(name,'0 kB').split()[0])
    return info

def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')

OUT.mkdir(exist_ok=False)
identity=dict(repository_revision='7b071fd9c8c5c06e9a567c59ac587ee74166be64',
    llama_cpp_revision='3d82ef62d47fd74e18f36c5eccbdcf965b617b17',
    model=MODEL.name,model_bytes=MODEL.stat().st_size,model_sha256=digest(MODEL),
    device_model=read('/proc/device-tree/model').strip('\x00'),os=read('/etc/os-release'),
    architecture=command(['uname','-m']),kernel=command(['uname','-r']),
    adapter=None,quantization='Q8_0',scope='Base model on physical Pi 5. Synthetic rule suite; no fine-tuning or general quality claim.')
write(OUT/'identity.json',identity)
if identity['model_sha256']!=EXPECTED:
    raise RuntimeError('Transferred model checksum mismatch')
artifacts=[json.loads(line) for line in (ROOT/'benchmark-build.jsonl').read_text().splitlines() if line.startswith('{')]
bins=[r['executable'] for r in artifacts if r.get('reason')=='compiler-artifact' and r.get('target',{}).get('name')=='benchmark' and r.get('executable')]
if len(bins)!=1:
    raise RuntimeError(f'Expected one native benchmark executable, got {bins}')
binary=Path(bins[0])
identity.update(benchmark_executable_sha256=digest(binary),suite_sha256=digest(SOURCE/'tests/fixtures/decision_benchmark.json'))
write(OUT/'identity.json',identity)
env=dict(os.environ,SKID_MODEL=str(MODEL),SKID_CUDA='0',SKID_BENCH_ITERATIONS='3',
    SKID_BENCH_WARMUP='1',SKID_CONTEXT='2048',SKID_BATCH='256',SKID_THREADS='4',
    SKID_PROMPT_LAYOUT='legacy',SKID_EXECUTION_MODE='fresh',SKID_BENCH_OUTPUT=str(OUT/'benchmark.json'))
cmd=[str(binary),'--exact','native::model_decision_benchmark','--ignored','--nocapture']
write(OUT/'command.json',dict(command=cmd,cwd=str(SOURCE),settings={k:v for k,v in env.items() if k.startswith('SKID_')},timeout_seconds=3600))
observations=[sensor()]
started=time.monotonic()
result=dict(status='running',started_utc=observations[0]['utc'])
write(OUT/'run.json',result)
with (OUT/'benchmark.log').open('x') as log, (OUT/'telemetry.jsonl').open('x') as telemetry:
    process=subprocess.Popen(cmd,cwd=SOURCE,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:
        while True:
            pid,status,usage=os.wait4(process.pid,os.WNOHANG)
            if pid:
                process.returncode=os.waitstatus_to_exitcode(status)
                result.update(status='ok' if process.returncode==0 else 'failed',exit_code=process.returncode,
                              peak_process_rss_kib=usage.ru_maxrss,user_cpu_s=usage.ru_utime,system_cpu_s=usage.ru_stime)
                break
            sample=sensor(process.pid)
            observations.append(sample)
            telemetry.write(json.dumps(sample)+'\n');telemetry.flush()
            if time.monotonic()-started>3600:
                os.killpg(process.pid,signal.SIGKILL)
                pid,status,usage=os.wait4(process.pid,0)
                process.returncode=os.waitstatus_to_exitcode(status)
                result.update(status='timeout',exit_code=process.returncode,peak_process_rss_kib=usage.ru_maxrss)
                break
            time.sleep(1)
    finally:
        if process.returncode is None:
            os.killpg(process.pid,signal.SIGKILL)
            _,status,_=os.wait4(process.pid,0)
            process.returncode=os.waitstatus_to_exitcode(status)
            result.update(status='interrupted',exit_code=process.returncode)
        observations.append(sensor())
        result.update(wall_seconds_including_load_warmup=time.monotonic()-started,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            temperature_max_c=max(s['temperature_c'] for s in observations if s['temperature_c'] is not None),
            available_memory_min_kib=min(s['available_kib'] for s in observations),
            swap_used_max_kib=max(s['swap_used_kib'] for s in observations),
            throttled_values=sorted(set(s['throttled'] for s in observations)),
            current_throttle_flags_observed=any(int(s['throttled'].split('=')[1],16)&0xffff for s in observations),
            telemetry_samples=len(observations),
            telemetry_scope='One-second samples including loading/warmup; brief events between samples may be missed. Historical throttle bits were already set before this run.')
        write(OUT/'run.json',result)
print(json.dumps(result,indent=2),flush=True)
raise SystemExit(0 if result['status']=='ok' else 1)
