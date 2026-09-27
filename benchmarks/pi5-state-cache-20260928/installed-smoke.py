"""Exercise the actual npm-installed l2s1 binary, independently of the probe."""
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
model = '/home/lutica/l2s1-pi-20260927/models/gemma-3-1b-it-Q8_0.gguf'
package = root / 'installed/node_modules/@l2s1/runtime-linux-arm64'
request = json.loads((root / 'inputs.json').read_text())['performance_cases'][3]['request']
output = []
for mode, limit, layout in [('fresh', None, 'state-first'), ('state-restore', None, 'state-first'),
                            ('state-restore', 0, 'state-first'), ('state-restore', None, 'legacy')]:
    args = [str(package / 'bin/l2s1'), '--model', model, '--context', '2048', '--batch', '256',
            '--ubatch', '256', '--threads', '4', '--device', 'cpu', '--prompt-layout', layout,
            '--execution-mode', mode, '--stdio']
    if limit is not None:
        args += ['--snapshot-limit-bytes', str(limit)]
    env = {**os.environ, 'LD_LIBRARY_PATH': str(package / 'bin')}
    env.pop('GGML_BACKEND_PATH', None)
    env.pop('LD_PRELOAD', None)
    log_path = root / f'installed-{mode}-{limit}-{layout}.log'
    with log_path.open('w') as log:
        process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   env=env, text=True, bufsize=1)
        def call(body):
            process.stdin.write(json.dumps({'id': 'smoke', 'op': 'decide', 'body': body})+'\n')
            process.stdin.flush()
            return json.loads(process.stdout.readline())
        cold, warm = call(request), call(request)
        assert 'error' not in cold and 'error' not in warm
        evidence = lambda response: [r['evidence'] for r in response['result']['results']]
        assert evidence(cold) == evidence(warm)
        reused = sum(r['usage'].get('reused_prefix_tokens', 0) for r in warm['result']['results'])
        if mode == 'fresh' or limit == 0 or layout == 'legacy':
            assert reused == 0
        else:
            assert reused > 0
        bad = {**request, 'state': {'history': 'overlong ' * 4000}}
        assert 'error' in call(bad)
        recovered = call(request)
        assert evidence(recovered) == evidence(cold)
        process.stdin.close()
        process.wait(timeout=30)
        assert process.returncode == 0
        if layout == 'state-first' and output:
            assert evidence(cold) == evidence(output[0]['cold'])
        output.append({'args': args, 'cold': cold, 'warm': warm, 'recovered': recovered,
                       'reused_tokens': reused, 'error_recovery': True})
(root / 'installed-smoke.json').write_text(json.dumps(output, indent=2)+'\n')
print('Installed fresh/restore/zero-budget/legacy, cold/warm and error recovery passed')
