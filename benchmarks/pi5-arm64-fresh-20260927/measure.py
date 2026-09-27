"""Run installed packages over identical stdio; preserve every response and sensor sample."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import threading
import time


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def sensor(pid=None):
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'monotonic': time.monotonic()}
    result['temperature_c'] = int(Path('/sys/class/thermal/thermal_zone0/temp').read_text()) / 1000
    result['throttled'] = int(subprocess.check_output(['vcgencmd', 'get_throttled'], text=True).strip().split('=')[1], 16)
    result['arm_clock_hz'] = int(subprocess.check_output(['vcgencmd', 'measure_clock', 'arm'], text=True).strip().split('=')[1])
    result['scaling_cur_freq_khz'] = int(Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq').read_text())
    mem = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    result['swap_kib'] = int(mem['SwapTotal'].split()[0]) - int(mem['SwapFree'].split()[0])
    if pid:
        status = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/status').read_text().splitlines() if ':' in line)
        for key in ('VmRSS', 'VmHWM'):
            result[key + '_kib'] = int(status.get(key, '0 kB').split()[0])
    return result


class Engine:
    def __init__(self, package, model, mode, directory, preload=None):
        self.directory = directory
        directory.mkdir(parents=True)
        self.phase = 'loading'
        self.stop = threading.Event()
        self.args = [str(package / 'bin/l2s1'), '--model', str(model), '--device', 'cpu',
                     '--context', '2048', '--batch', '256', '--ubatch', '256', '--threads', '4',
                     '--execution-mode', 'fresh' if mode == 'fresh' else 'prefix-reuse', '--stdio']
        if mode == 'fixed':
            self.args.append('--fixed-schema')
        self.env = {**os.environ, 'LD_LIBRARY_PATH': str(package / 'bin'), 'OMP_NUM_THREADS': '4'}
        self.env.pop('GGML_BACKEND_PATH', None)
        if preload:
            self.env['LD_PRELOAD'] = str(preload)
        self.stderr = (directory / 'stderr.log').open('w')
        self.process = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr, text=True, bufsize=1, env=self.env)
        (directory / 'command.json').write_text(json.dumps({'args': self.args, 'LD_LIBRARY_PATH': self.env['LD_LIBRARY_PATH'],
                                                          'OMP_NUM_THREADS': '4', 'LD_PRELOAD': preload}, indent=2) + '\n')
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()
        self.count = 0
        self.call('health')
        (directory / 'capabilities.json').write_text(json.dumps(self.call('capabilities'), indent=2) + '\n')
        maps = Path(f'/proc/{self.process.pid}/maps').read_text()
        (directory / 'maps.txt').write_text(maps)
        libraries = sorted({line.split()[-1] for line in maps.splitlines()
                            if any('/lib'+name in line for name in ['llama', 'ggml', 'mtmd'])})
        assert libraries and all(str(package / 'bin') + '/' in path for path in libraries), libraries
        (directory / 'loaded-libraries.json').write_text(json.dumps(
            {path: digest(Path(path)) for path in libraries}, indent=2) + '\n')

    def monitor(self):
        with (self.directory / 'telemetry.jsonl').open('w') as f:
            while not self.stop.is_set():
                try:
                    row = sensor(self.process.pid)
                    row['phase'] = self.phase
                    f.write(json.dumps(row) + '\n')
                    f.flush()
                except FileNotFoundError:
                    break
                self.stop.wait(1)

    def call(self, op, body=None):
        self.count += 1
        self.process.stdin.write(json.dumps({'id': str(self.count), 'op': op, 'body': body}) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'child exited {self.process.poll()}; see {self.directory}')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(reply)
        return reply['result']

    def close(self):
        self.stop.set()
        self.thread.join()
        self.process.stdin.close()
        self.process.wait(timeout=30)
        self.stderr.close()
        if self.process.returncode:
            raise RuntimeError(f'child exited {self.process.returncode}')


def block(root, variant, mode, cases, warmup, model, name, preload=None):
    directory = root / 'results' / name
    # Same precondition for all variants; failure to cool is recorded, never hidden.
    idle = []
    start = time.monotonic()
    while True:
        sample = sensor()
        idle.append(sample)
        if sample['temperature_c'] <= 60 or time.monotonic() - start > 120:
            break
        time.sleep(2)
    engine = Engine(root / 'installed' / variant / 'node_modules/@l2s1/runtime-linux-arm64', model, mode, directory, preload)
    (directory / 'idle.json').write_text(json.dumps(idle, indent=2) + '\n')
    try:
        with (directory / 'responses.jsonl').open('w') as f:
            for phase, items in [('warmup', warmup), ('measured', cases)]:
                for index, case in enumerate(items):
                    engine.phase = f'{phase}:{index}:{case["id"]}'
                    before = time.monotonic()
                    response = engine.call('decide', case['request'])
                    after = time.monotonic()
                    record = {'variant': variant, 'mode': mode, 'phase': phase, 'case_id': case['id'],
                              'index': index, 'start': before, 'end': after, 'elapsed_ms': (after-before)*1000,
                              'expected': case['expected'], 'response': response}
                    f.write(json.dumps(record) + '\n')
                    f.flush()
                    if mode == 'fresh':
                        assert response['backend']['details']['execution_mode'] == 'fresh'
                        assert all(r['usage'].get('reused_prefix_tokens', 0) == 0 for r in response['results'])
                    print(name, phase, case['id'], round(record['elapsed_ms']), flush=True)
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--part', choices=['factorial', 'quality', 'fallback'], default='factorial')
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text())['cases']
    short = suite[:3]
    identity = {'model_sha256': digest(args.model), 'suite_sha256': digest(args.suite), 'host': platform.uname()._asdict(),
                'settings': {'context': 2048, 'batch': 256, 'ubatch': 256, 'threads': 4, 'transport': 'direct stdio',
                             'warmup': 'all warehouse-01/02/03 once per block', 'cooling': '60 C or timeout 120s'},
                'cpuinfo': '\n'.join(line for line in Path('/proc/cpuinfo').read_text().splitlines() if not line.startswith('Serial'))}
    assert identity['model_sha256'] == 'b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5'
    (args.root / f'identity-{args.part}.json').write_text(json.dumps(identity, indent=2) + '\n')
    if args.part == 'factorial':
        # Balanced Latin square: every variant appears in every position once;
        # each ordered pair is adjacent once (carryover balanced).
        order = [['baseline', 'kernels', 'combined', 'openmp'],
                 ['kernels', 'openmp', 'baseline', 'combined'],
                 ['openmp', 'combined', 'kernels', 'baseline'],
                 ['combined', 'baseline', 'openmp', 'kernels']]
        for round_id, variants in enumerate(order):
            for variant in variants:
                block(args.root, variant, 'fresh', short, short, args.model, f'factorial-{round_id}-{variant}')
    elif args.part == 'quality':
        for variant in ['baseline', 'kernels', 'openmp', 'combined']:
            block(args.root, variant, 'fresh', suite, short, args.model, f'quality-{variant}-fresh')
        for mode in ['prefix', 'fixed']:
            block(args.root, 'kernels', mode, suite * 2, short, args.model, f'quality-kernels-{mode}')
    else:
        block(args.root, 'kernels', 'fresh', short, short, args.model, 'fallback', str(args.root / 'mask-features.so'))


if __name__ == '__main__':
    main()
