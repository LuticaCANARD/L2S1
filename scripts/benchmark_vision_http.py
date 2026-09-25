#!/usr/bin/env python3
"""Measure serial loopback HTTP vision latency for one loaded L2S1 model."""

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_sample():
    try:
        output = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,driver_version,memory.used,temperature.gpu', '--format=csv,noheader,nounits'],
            text=True, timeout=5)
        return output.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def rss_kib(pid):
    try:
        lines = Path(f'/proc/{pid}/status').read_text().splitlines()
    except OSError:
        return None
    return {name: int(value.split()[0]) for line in lines
            if (parts := line.split(':', 1)) and len(parts) == 2
            for name, value in [parts] if name in ('VmRSS', 'VmHWM')}


def nearest_rank(values, percentile):
    ordered = sorted(values)
    return ordered[math.ceil(len(ordered) * percentile) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'mmproj', 'red-image', 'blue-image', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--listen', default='127.0.0.1:18765')
    parser.add_argument('--warmup', type=int, default=4)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--context', type=int, default=2048)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 2:
        parser.error('warmup must be nonnegative and iterations at least 2')
    if args.output.exists():
        parser.error('output already exists')
    for path in (args.binary, args.model, args.mmproj, args.red_image, args.blue_image):
        if not path.is_file():
            parser.error(f'file unavailable: {path}')
    base = {'state': {}, 'decisions': [{'id': 'color', 'instruction': 'Which color fills the image?',
             'kind': {'type': 'choice', 'options': [
                 {'id': 'red', 'criterion': 'The image is red.'},
                 {'id': 'blue', 'criterion': 'The image is blue.'}]}}]}
    requests = {}
    for color, path in (('red', args.red_image), ('blue', args.blue_image)):
        payload = dict(base)
        payload['media'] = [{'id': 'image', 'type': 'image', 'data_base64': base64.b64encode(path.read_bytes()).decode('ascii')}]
        requests[color] = json.dumps(payload, separators=(',', ':')).encode()
    url = 'http://' + args.listen
    command = [str(args.binary.resolve()), '--model', str(args.model.resolve()), '--mmproj',
               str(args.mmproj.resolve()), '--device', args.device, '--context', str(args.context),
               '--batch', str(args.batch), '--threads', str(args.threads),
               '--model-load-mode', 'read', '--listen', args.listen]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.output.with_suffix('.server.log')
    if log_path.exists():
        parser.error('server log already exists')
    baseline_gpu = gpu_sample()
    started = time.perf_counter_ns()
    with log_path.open('wb') as log:
        server = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(480):
                if server.poll() is not None:
                    raise RuntimeError(f'server exited before health check: {server.returncode}; see {log_path}')
                try:
                    with urlopen(url + '/healthz', timeout=1) as response:
                        if response.status == 200:
                            break
                except (URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise RuntimeError(f'server did not become ready; see {log_path}')
            startup_ms = (time.perf_counter_ns() - started) / 1_000_000
            observations = []
            for iteration in range(args.warmup + args.iterations):
                color = 'red' if iteration % 2 == 0 else 'blue'
                req = Request(url + '/v1/decisions', requests[color], {'Content-Type': 'application/json'})
                start = time.perf_counter_ns()
                with urlopen(req, timeout=180) as response:
                    body = response.read()
                    status = response.status
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                result = json.loads(body)
                selected = result['results'][0]['value']['selected']
                backend = result['backend']['details']
                if status != 200 or selected != color:
                    raise RuntimeError(f'{color} request failed: status={status}, selected={selected}')
                if (args.device == 'cuda') != bool(backend['offload_requested']):
                    raise RuntimeError('backend offload setting differs from requested device')
                observations.append({'phase': 'warmup' if iteration < args.warmup else 'measured',
                                     'color': color, 'latency_ms': round(elapsed_ms, 6),
                                     'http_status': status, 'selected': selected,
                                     'input_tokens': result['results'][0]['usage']['input_tokens'],
                                     'offload_device': backend['offload_device']})
            loaded_gpu = gpu_sample()
            memory = rss_kib(server.pid)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    measured = [item['latency_ms'] for item in observations if item['phase'] == 'measured']
    report = {'schema': 'l2s1-vision-http-benchmark-v1',
              'scope': 'serial loopback HTTP requests; one loaded model; synthetic solid-color 64x64 PNGs; timings include HTTP transfer and inference, exclude startup and warmup',
              'host': socket.gethostname(), 'device': args.device,
              'binary_sha256': sha256(args.binary),
              'model_sha256': sha256(args.model), 'projector_sha256': sha256(args.mmproj),
              'images': {color: {'sha256': sha256(path), 'bytes': path.stat().st_size}
                         for color, path in (('red', args.red_image), ('blue', args.blue_image))},
              'config': {'context': args.context, 'batch': args.batch, 'threads': args.threads,
                         'model_load_mode': 'read', 'warmup': args.warmup, 'iterations': args.iterations},
              'startup_to_health_ms': round(startup_ms, 3),
              'latency_ms': {'mean': round(statistics.mean(measured), 3),
                             'p50': round(statistics.median(measured), 3),
                             'p95_nearest_rank': round(nearest_rank(measured, 0.95), 3),
                             'min': round(min(measured), 3), 'max': round(max(measured), 3)},
              'gpu_before': baseline_gpu, 'gpu_after_warmup_and_measurement': loaded_gpu,
              'process_memory_kib': memory, 'observations': observations}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('device', 'startup_to_health_ms', 'latency_ms',
                                                    'gpu_after_warmup_and_measurement', 'process_memory_kib')},
                     ensure_ascii=False))
    print(f'raw report: {args.output}', file=sys.stderr)


if __name__ == '__main__':
    main()
