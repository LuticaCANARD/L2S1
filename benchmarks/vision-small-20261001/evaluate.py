#!/usr/bin/env python3
"""Run one frozen TrashNet pass over real loopback HTTP; omit policy overrides."""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.used',
                                    '--format=csv,noheader,nounits'], text=True).strip()


def memory(pid):
    return {k: int(v.split()[0]) for line in Path(f'/proc/{pid}/status').read_text().splitlines()
            for k, v in [line.split(':', 1)] if k in ('VmRSS', 'VmHWM')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('binary', 'model', 'mmproj', 'archive', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    legacy = root / 'benchmarks/trashnet-vision-20260925'
    spec = importlib.util.spec_from_file_location('frozen_trashnet', legacy / 'evaluate.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    selection = json.loads((legacy / 'selection.json').read_text())
    assert digest(args.archive) == selection['source_sha256']
    records = selection['records']
    assert len(records) == 120
    payloads = []
    with ZipFile(args.archive) as archive:
        for record in records:
            image = archive.read(record['name'])
            assert hashlib.sha256(image).hexdigest() == record['sha256']
            payloads.append(json.dumps({
                'state': {}, 'decisions': [{'id': 'material', 'instruction': frozen.PROMPTS['baseline'],
                    'kind': {'type': 'choice', 'options': [
                        {'id': label, 'criterion': frozen.DESCRIPTIONS[label]} for label in frozen.CLASSES]}}],
                'media': [{'id': 'image', 'type': 'image', 'data_base64': base64.b64encode(image).decode()}]
            }, separators=(',', ':')).encode())
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    command = [str(args.binary.resolve()), '--model', str(args.model.resolve()), '--mmproj',
               str(args.mmproj.resolve()), '--device', 'cuda', '--execution-mode', 'fresh',
               '--context', '8192', '--batch', '256', '--ubatch', '256', '--threads', '4',
               '--flash-attention', 'off', '--preparation-cache-bytes', '0', '--model-load-mode',
               'read', '--listen', f'127.0.0.1:{port}']
    protocol = {'command': command, 'source_commit': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'binary_sha256': digest(args.binary), 'model_sha256': digest(args.model),
        'projector_sha256': digest(args.mmproj), 'selection_sha256': digest(legacy / 'selection.json'),
        'archive_sha256': digest(args.archive), 'runner_sha256': digest(__file__),
        'warmup': 1, 'measured': len(records), 'policy': 'omitted: validate default 0/0',
        'prompt': frozen.PROMPTS['baseline'], 'options': frozen.DESCRIPTIONS,
        'gpu_before': gpu(), 'platform': list(os.uname())}
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    rows = []
    started = time.perf_counter()
    with (args.output / 'server.log').open('wb') as log:
        server = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(1200):
                if server.poll() is not None:
                    raise RuntimeError(f'server exited: {server.returncode}; see {args.output}/server.log')
                try:
                    with urlopen(url + '/healthz', timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(.25)
            else:
                raise RuntimeError('health check timeout')
            startup = time.perf_counter() - started
            with (args.output / 'responses.jsonl').open('w') as raw:
                for index in [-1] + list(range(len(records))):
                    sample = max(index, 0)
                    request = Request(url + '/v1/decisions', payloads[sample], {'Content-Type': 'application/json'})
                    start = time.perf_counter()
                    with urlopen(request, timeout=180) as response:
                        body = json.load(response)
                    elapsed = (time.perf_counter() - start) * 1000
                    raw.write(json.dumps({'index': index, 'latency_ms': elapsed, 'response': body}) + '\n')
                    raw.flush()
                    assert body['policy'] == {'min_candidate_mass': 0.0, 'min_top_probability': 0.0}
                    assert body['backend']['details']['offload_requested']
                    result = body['results'][0]
                    assert not result['evidence']['truncated']
                    scores = result['evidence']['scores']
                    top = max(scores, key=lambda s: s['option_probability'])['id']
                    selected = (result.get('value') or {}).get('selected')
                    assert selected == top, result
                    row = dict(records[sample], selected=selected, raw_top1=top,
                               correct=selected == records[sample]['label'], latency_ms=elapsed,
                               memory_kib=memory(server.pid), gpu=gpu())
                    if index >= 0:
                        rows.append(row)
                    if index == -1 or (index + 1) % 10 == 0:
                        print(f'{args.model.name}: {index + 1}/120 correct={sum(r["correct"] for r in rows)} latency={elapsed:.1f}ms', flush=True)
            loaded_memory = memory(server.pid)
            loaded_gpu = gpu()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    n = len(rows)
    k = sum(r['correct'] for r in rows)
    p = k / n
    z = 1.959963984540054
    centre = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    latencies = sorted(r['latency_ms'] for r in rows)
    summary = {'model': args.model.name, 'n': n, 'correct': k, 'raw_top1_accuracy': p,
               'accepted': n, 'abstained': 0, 'coverage': 1.0, 'accepted_accuracy': p,
               'correct_over_all': p, 'wilson_95': [centre-half, centre+half],
               'latency_ms': {'p50': statistics.median(latencies), 'p95': latencies[math.ceil(.95*n)-1]},
               'startup_seconds': startup, 'process_memory_kib': loaded_memory, 'gpu_after': loaded_gpu,
               'per_class': {c: {'n': sum(r['label'] == c for r in rows),
                   'correct': sum(r['label'] == c and r['correct'] for r in rows)} for c in frozen.CLASSES},
               'confusion': {c: {d: sum(r['label'] == c and r['selected'] == d for r in rows)
                                for d in frozen.CLASSES} for c in frozen.CLASSES},
               'raw_responses_sha256': digest(args.output / 'responses.jsonl'), 'observations': rows}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'observations'}), flush=True)


if __name__ == '__main__':
    main()
