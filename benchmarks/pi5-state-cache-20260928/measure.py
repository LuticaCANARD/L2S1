"""Matched resident diagnostic transport; raw predictions, copy costs and Pi telemetry."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Probe:
    def __init__(self, args, version, directory):
        self.directory, self.phase = directory, 'loading'
        directory.mkdir(parents=True)
        self.done = threading.Event()
        self.sensor = args.sensor
        package = (args.root.parent / 'installed/kernels/node_modules/@l2s1/runtime-linux-arm64'
                   if version == 'before' else args.root / 'installed/node_modules/@l2s1/runtime-linux-arm64')
        binary = args.root / version / 'probe'
        command = [str(binary), '--model', str(args.model), '--batch', '256']
        env = {**os.environ, 'LD_LIBRARY_PATH': str(package / 'bin'), 'OMP_NUM_THREADS': '4'}
        env.pop('GGML_BACKEND_PATH', None)
        env.pop('LD_PRELOAD', None)
        self.log = (directory / 'stderr.log').open('w')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1, env=env)
        self.monitor = threading.Thread(target=self.sample, daemon=True)
        self.monitor.start()
        inspection = json.loads(self.process.stdout.readline())
        (directory / 'inspection.json').write_text(json.dumps(inspection, indent=2)+'\n')
        maps = Path(f'/proc/{self.process.pid}/maps').read_text()
        paths = sorted({line.split()[-1] for line in maps.splitlines()
                        if any('/lib'+name in line for name in ['llama', 'ggml', 'mtmd'])})
        assert paths and all(str(package / 'bin')+'/' in path for path in paths)
        (directory / 'identity.json').write_text(json.dumps({
            'command': command, 'binary_sha256': digest(binary), 'library_path': env['LD_LIBRARY_PATH'],
            'libraries': {path: digest(Path(path)) for path in paths},
            'runtime_manifest': json.loads((package / 'runtime-manifest.json').read_text()),
        }, indent=2)+'\n')

    def sample(self):
        with (self.directory / 'telemetry.jsonl').open('w') as stream:
            while not self.done.is_set():
                try:
                    row = self.sensor(self.process.pid)
                except FileNotFoundError:
                    return
                row['phase'] = self.phase
                stream.write(json.dumps(row)+'\n')
                stream.flush()
                self.done.wait(1)

    def call(self, payload):
        self.process.stdin.write(json.dumps(payload)+'\n')
        self.process.stdin.flush()
        reply = json.loads(self.process.stdout.readline())
        assert 'Err' not in reply, reply
        return reply.get('Ok', reply)

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.done.set()
        self.monitor.join()
        self.log.close()
        assert self.process.returncode == 0


def block(args, name, version, mode, cases, warmup, shared=False):
    directory = args.root / 'results' / name
    # Same ten-second idle for every block. Temperature is observed, not controlled.
    idle, started = [], time.monotonic()
    while True:
        sample = args.sensor()
        idle.append(sample)
        if time.monotonic()-started >= 10:
            break
        time.sleep(2)
    probe = Probe(args, version, directory)
    (directory / 'idle.json').write_text(json.dumps(idle, indent=2)+'\n')
    try:
        with (directory / 'responses.jsonl').open('w') as stream:
            for phase, requests in [('warmup', warmup), ('measured', cases)]:
                for index, case in enumerate(requests):
                    payload = {'request': case['request'], 'mode': mode, 'layout': case.get('layout', 'state_first'),
                               'shared_session': shared, 'snapshot_limit': case.get('snapshot_limit')}
                    probe.phase = f'{phase}:{case["id"]}'
                    start = time.monotonic()
                    result = probe.call(payload)
                    end = time.monotonic()
                    row = {'case_id': case['id'], 'expected': case['expected'], 'index': index,
                           'phase': phase, 'version': version, 'mode': mode, 'shared_session': shared,
                           'start': start, 'end': end, 'elapsed_ms': (end-start)*1000,
                           'payload': payload, 'result': result}
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                    reused = sum(r['reused_prefix_tokens'] for r in result['response']['results'])
                    if mode == 'fresh':
                        assert reused == 0
                    print(name, phase, case['id'], round(row['elapsed_ms']), 'reuse', reused,
                          'restore', result['state_restore'], flush=True)
    finally:
        probe.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--part', choices=['smoke', 'performance', 'quality', 'session'], required=True)
    args = parser.parse_args()
    # Reuse the previous study's unchanged sensor implementation.
    spec = importlib.util.spec_from_file_location('sensors', args.root.parent / 'measure.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.sensor = module.sensor
    suite = json.loads(args.suite.read_text())['cases']
    long = copy.deepcopy(suite[0])
    long['id'] = 'warehouse-01-long-state'
    long['request']['state']['irrelevant_packing_notes'] = 'packing ' * 300
    limited = copy.deepcopy(long)
    limited['id'] = 'warehouse-01-zero-budget'
    limited['snapshot_limit'] = 0
    cases, warmup = suite[:3]+[long, limited], [suite[0], long]
    (args.root / 'inputs.json').write_text(json.dumps({'suite_sha256': digest(args.suite),
        'model_sha256': digest(args.model), 'performance_cases': cases}, indent=2)+'\n')
    configurations = [('after', 'fresh'), ('after', 'prefix_reuse'),
                      ('before', 'state_restore'), ('after', 'state_restore')]
    if args.part == 'smoke':
        for version, mode in configurations:
            block(args, f'smoke-{version}-{mode}', version, mode, [long, limited], [])
    elif args.part == 'performance':
        for index, order in enumerate([[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]):
            for choice in order:
                version, mode = configurations[choice]
                block(args, f'performance-{index}-{version}-{mode}', version, mode, cases, warmup)
    elif args.part == 'quality':
        for version, mode in configurations:
            block(args, f'quality-{version}-{mode}', version, mode, suite, [suite[0]])
    else:
        for round_id in range(4):
            for shared in ([False, True] if round_id % 2 == 0 else [True, False]):
                mode = 'prefix_reuse' if shared else 'fresh'
                block(args, f'session-{round_id}-{shared}', 'after', mode, [suite[0], long], [suite[0]], shared)


if __name__ == '__main__':
    main()
