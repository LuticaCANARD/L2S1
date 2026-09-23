#!/usr/bin/env python3
"""Run an explicit compute matrix; preserve inputs, commands, runtime hashes and outcomes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from kaggle_ag_news import score, sha256


def compare(reference, rows):
    baseline = {r['id']: r['response']['results'][0] for r in reference}
    changed = top_changed = 0
    delta = mass_delta = 0.0
    for row in rows:
        a, b = baseline[row['id']], row['response']['results'][0]
        assert a['input_tokens'] == b['input_tokens']
        assert [s['token_id'] for s in a['scores']] == [s['token_id'] for s in b['scores']]
        changed += a['value'] != b['value']
        top_changed += max(a['scores'], key=lambda s: s['option_probability'])['id'] != max(b['scores'], key=lambda s: s['option_probability'])['id']
        delta = max(delta, *(abs(x['option_probability'] - y['option_probability']) for x, y in zip(a['scores'], b['scores'])))
        mass_delta = max(mass_delta, abs(a['candidate_mass'] - b['candidate_mass']))
    return dict(changed_selection=changed, changed_top1=top_changed, max_probability_delta=delta,
                max_candidate_mass_delta=mass_delta, equivalent=changed == 0 and top_changed == 0 and delta <= .02 and mass_delta <= .02)


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    for name in ['binary', 'model', 'input', 'selection', 'matrix', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    manifest = a.out / 'summary.json'
    if manifest.exists():
        raise ValueError('Use a new output directory')
    labels = json.loads(a.selection.read_text())['labels']
    ids = [r['id'] for r in read(a.input)]
    assert len(ids) == len(set(ids))
    labels = {i: labels[i] for i in ids}
    reference = read(a.reference) if a.reference else None
    summary = dict(binary_sha256=sha256(a.binary), model_sha256=sha256(a.model),
                   input_sha256=sha256(a.input), selection_sha256=sha256(a.selection),
                   runtime={f.name: sha256(f) for f in a.runtime.glob('*.so')}, runs=[])
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = str(a.runtime.resolve())
    runtime_hashes = {str(a.runtime.resolve()): summary['runtime']}
    binary_hashes = {str(a.binary.resolve()): summary['binary_sha256']}
    for config in json.loads(a.matrix.read_text()):
        name = config['name']
        output = a.out / (name + '.jsonl')
        assert not output.exists()
        binary = Path(config.get('binary', a.binary)).resolve()
        runtime = Path(config.get('runtime', a.runtime)).resolve()
        if str(binary) not in binary_hashes:
            binary_hashes[str(binary)] = sha256(binary)
        if str(runtime) not in runtime_hashes:
            runtime_hashes[str(runtime)] = {f.name: sha256(f) for f in runtime.glob('*.so')}
        cmd = [str(binary), '--model', str(a.model.resolve()), '--input', str(a.input.resolve()),
               '--output', str(output.resolve()), '--warmup'] + ([] if a.cpu else ['--cuda']) + config['args']
        run_env = env.copy()
        run_env['LD_LIBRARY_PATH'] = str(runtime)
        for key in ['GGML_CUDA_CUBLAS_COMPUTE_TYPE', 'GGML_CUDA_DISABLE_GRAPHS', 'GGML_CUDA_DISABLE_FUSION', 'GGML_CUDA_GRAPH_OPT']:
            run_env.pop(key, None)
        run_env.update(config.get('env', {}))
        print('START', name, flush=True)
        start = time.monotonic()
        with (a.out / (name + '.log')).open('w') as log:
            try:
                code = subprocess.run(cmd, env=run_env, stdout=log, stderr=subprocess.STDOUT, timeout=config.get('timeout', 600)).returncode
            except subprocess.TimeoutExpired:
                code = 124
        rows = read(output) if output.exists() else []
        metrics = score(labels, rows)
        batches = {}
        for r in rows:
            batches.setdefault(r['batch_index'], r)
        ms = sum(r['batch_elapsed_ms'] for r in batches.values())
        profile = {key: sum(r.get('batch_profile', {}).get(key, 0) for r in batches.values()) for key in ['prepare_ms', 'native_ms', 'score_ms', 'decisions']}
        result = dict(config=config, command=cmd, binary_sha256=binary_hashes[str(binary)], runtime=runtime_hashes[str(runtime)], exit_code=code, wall_seconds=time.monotonic()-start,
                      metrics=metrics, measured_ms=ms, articles_per_second=len(rows)*1000/ms if ms else None, profile=profile)
        if code == 0 and not metrics['counts']['errors'] and not metrics['counts']['missing']:
            if reference:
                result['comparison'] = compare(reference, rows)
        summary['runs'].append(result)
        manifest.write_text(json.dumps(summary, indent=2)+'\n')
        print('END', name, json.dumps({k: result[k] for k in ['exit_code', 'measured_ms', 'comparison'] if k in result}), flush=True)


if __name__ == '__main__':
    main()
