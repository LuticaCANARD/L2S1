#!/usr/bin/env python3
"""Tune evaluate_jsonl on development data with complete, paired measurements.

Gold is {case_id: {decision_id: option_id}} and is read only by the scorer.
A matrix is [{"name": "baseline", "args": ["--batch", "256", ...]}, ...].
The first config is the reference. Models run sequentially; processes are cold,
with one excluded evaluator warmup. Resident sessions start cold after warmup.
"""
import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def outcomes(rows, gold):
    """Reject incomplete/invalid outputs instead of scoring the surviving rows."""
    if len(rows) != len(gold) or {r['id'] for r in rows} != set(gold):
        raise ValueError('missing, duplicate, or unexpected case IDs')
    result = {}
    for row in rows:
        if 'error' in row:
            raise ValueError(f"inference error: {row['id']}: {row['error']}")
        items = row['response']['results']
        if len(items) != len(gold[row['id']]) or {x['id'] for x in items} != set(gold[row['id']]):
            raise ValueError('decision schema differs from gold')
        for item in items:
            if item.get('truncated'):
                raise ValueError('truncated inference')
            scores = item['scores']
            probabilities = [s['option_probability'] for s in scores]
            if not scores or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError('invalid probabilities')
            if len({s['id'] for s in scores}) != len(scores):
                raise ValueError('duplicate option ID')
            if gold[row['id']][item['id']] not in {s['id'] for s in scores}:
                raise ValueError('gold option missing from scores')
            mass = item['candidate_mass']
            if not math.isfinite(mass) or not 0 <= mass <= 1:
                raise ValueError('invalid candidate mass')
            top = max(scores, key=lambda s: s['option_probability'])['id']
            value = item['value']
            selected = value.get('selected')
            if 'value' in value and isinstance(value['value'], bool):
                selected = str(value['value']).lower()
            if selected is not None and selected not in {s['id'] for s in scores}:
                raise ValueError('unknown selected option')
            if bool(item['abstention_reasons']) != (selected is None):
                raise ValueError('selection and abstention disagree')
            result[(row['id'], item['id'])] = dict(top=top, selected=selected,
                expected=gold[row['id']][item['id']], mass=mass,
                scores={s['id']: s['option_probability'] for s in scores},
                tokens=item['input_tokens'], reused=item['reused_prefix_tokens'])
    return result


def metrics(rows, gold):
    pairs = outcomes(rows, gold)
    values = list(pairs.values())
    n = len(values)
    accepted = sum(v['selected'] is not None for v in values)
    correct = sum(v['selected'] == v['expected'] for v in values)
    batches = {}
    for row in rows:
        elapsed = row['batch_elapsed_ms']
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError('invalid latency')
        index = row['batch_index']
        if index in batches and batches[index]['batch_elapsed_ms'] != elapsed:
            raise ValueError('inconsistent batch timing')
        batches[index] = row
    times = [r['batch_elapsed_ms'] for r in batches.values()]
    profile = {key: sum(r['batch_profile'][key] for r in batches.values())
               for key in ('prepare_ms', 'native_ms', 'score_ms')}
    return dict(decisions=n, cases=len(rows), raw_correct=sum(v['top'] == v['expected'] for v in values),
                accepted=accepted, accepted_correct=correct, wrong_accepted=accepted-correct,
                abstained=n-accepted, coverage=accepted/n, accepted_accuracy=correct/accepted if accepted else None,
                correct_all=correct/n, p50_ms=percentile(times, .5), p95_ms=percentile(times, .95),
                measured_ms=sum(times), decisions_per_second=n*1000/sum(times),
                input_tokens=sum(v['tokens'] for v in values), reused_prefix_tokens=sum(v['reused'] for v in values),
                profile=profile)


def compare(reference, candidate, gold):
    a, b = outcomes(reference, gold), outcomes(candidate, gold)
    if a.keys() != b.keys():
        raise ValueError('unpaired predictions')
    delta = mass = 0.0
    top = selected = token_changes = 0
    for key, left in a.items():
        right = b[key]
        if left['scores'].keys() != right['scores'].keys():
            raise ValueError('candidate schema changed')
        delta = max(delta, *(abs(p-right['scores'][k]) for k, p in left['scores'].items()))
        mass = max(mass, abs(left['mass']-right['mass']))
        top += left['top'] != right['top']
        selected += left['selected'] != right['selected']
        token_changes += left['tokens'] != right['tokens']
    return dict(changed_top1=top, changed_selection=selected, changed_input_tokens=token_changes,
                max_probability_delta=delta, max_mass_delta=mass)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'input', 'gold', 'matrix', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--evaluation-only', action='store_true', help='Do not select configurations on held-out data')
    args = parser.parse_args()
    if args.repeats < 1 or (args.repeats < 3 and not args.evaluation_only):
        parser.error('tuning requires at least three repeats')
    args.out.mkdir(parents=True, exist_ok=False)
    gold = json.loads(args.gold.read_text())
    inputs = read_rows(args.input)
    if len(inputs) != len(gold) or {r['id'] for r in inputs} != set(gold) or not inputs:
        raise ValueError('input and gold IDs must be identical and unique')
    configs = json.loads(args.matrix.read_text())
    names = [c['name'] for c in configs]
    if not names or len(names) != len(set(names)) or any(not n or any(not(c.isascii() and (c.isalnum() or c in '-_')) for c in n) for n in names):
        raise ValueError('matrix requires unique safe names')
    forbidden = {'--model', '--input', '--output', '--warmup', '--cuda', '--metal'}
    for config in configs:
        if not isinstance(config['args'], list) or any(not isinstance(v, str) or v.split('=')[0] in forbidden for v in config['args']):
            raise ValueError('matrix overrides a controlled argument')
    env = dict(os.environ)
    if args.runtime:
        env['LD_LIBRARY_PATH'] = str(args.runtime.resolve())
    plan = dict(version=1, hashes={k:digest(getattr(args,k)) for k in ('binary','model','input','gold','matrix')},
                configs=configs, repeats=args.repeats, evaluation_only=args.evaluation_only, platform=platform.uname()._asdict(),
                gate={'top1_changes':0, 'selection_changes':0, 'input_token_changes':0,
                      'max_probability_delta':.02, 'max_mass_delta':.02, 'p50_ratio':.95, 'p95_ratio':1.05},
                environment={k:v for k,v in env.items() if k.startswith('GGML_') or k in ('CUDA_VISIBLE_DEVICES',)},
                runtime={p.name:digest(p) for p in args.runtime.glob('*.so*') if p.is_file()} if args.runtime else None)
    write_json(args.out/'plan.json', plan)
    runs = []
    for repeat in range(args.repeats):
        # Interleave order to reduce a systematic warm-board advantage.
        ordered = configs if repeat % 2 == 0 else list(reversed(configs))
        for config in ordered:
            stem = f"{config['name']}-{repeat}"
            output = args.out/(stem+'.jsonl')
            cmd = [str(args.binary.resolve()), '--model', str(args.model.resolve()), '--input', str(args.input.resolve()),
                   '--output', str(output.resolve()), '--warmup']
            if not args.cpu: cmd += ['--cuda']
            cmd += config['args']
            started = time.monotonic()
            samples = []
            telemetry_error = None
            peak_rss_kib = None
            with (args.out/(stem+'.log')).open('w') as log:
                child = subprocess.Popen(cmd, stdout=log, stderr=log, env=env)
                while child.poll() is None:
                    try:
                        status = Path(f'/proc/{child.pid}/status').read_text()
                        for line in status.splitlines():
                            if line.startswith(('VmRSS:', 'VmHWM:')):
                                peak_rss_kib = max(peak_rss_kib or 0, int(line.split()[1]))
                    except (OSError, ValueError):
                        pass
                    if time.monotonic()-started > args.timeout:
                        child.kill(); child.wait(); break
                    if not args.cpu and telemetry_error is None:
                        try:
                            sample = subprocess.run(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                                                    capture_output=True, text=True, timeout=2, check=True).stdout.strip()
                            samples.append({'seconds':time.monotonic()-started,'gpus':sample})
                        except (OSError, subprocess.SubprocessError) as exc:
                            telemetry_error = str(exc)
                    time.sleep(.1)
            run = dict(config=config['name'], repeat=repeat, command=cmd, exit_code=child.returncode,
                       wall_seconds=time.monotonic()-started, telemetry_error=telemetry_error, sampled_rss_mib=peak_rss_kib/1024 if peak_rss_kib is not None else None)
            write_json(args.out/(stem+'-gpu.json'), samples)
            try:
                if child.returncode != 0: raise ValueError('evaluator exited unsuccessfully')
                rows = read_rows(output)
                run['metrics'] = metrics(rows, gold)
            except (ValueError, KeyError, OSError) as exc:
                run['error'] = str(exc)
            runs.append(run)
            write_json(args.out/'runs.json', runs)
            print(stem, run.get('metrics', run.get('error')), flush=True)
    summary = {}
    for config in configs:
        name = config['name']
        group = [r for r in runs if r['config'] == name]
        valid = len(group) == args.repeats and all('error' not in r for r in group)
        entry = {'complete':valid, 'runs':group}
        if valid:
            reference_path = args.out/(names[0]+'-0.jsonl')
            try:
                reference = read_rows(reference_path)
                entry['comparisons'] = [compare(reference, read_rows(args.out/f'{name}-{i}.jsonl'), gold) for i in range(args.repeats)]
                entry['p50_ms'] = percentile([r['metrics']['p50_ms'] for r in group], .5)
                entry['p95_ms'] = max(r['metrics']['p95_ms'] for r in group)
                entry['equivalent'] = all(c['changed_top1']==c['changed_selection']==c['changed_input_tokens']==0 and
                                          c['max_probability_delta'] <= .02 and c['max_mass_delta'] <= .02 for c in entry['comparisons'])
            except (ValueError, KeyError, OSError) as exc:
                entry.update(complete=False, error=str(exc))
        summary[name] = entry
    write_json(args.out/'summary.json', summary)
    if not args.evaluation_only:
        baseline = summary[names[0]]
        if not baseline.get('complete') or not baseline.get('equivalent'):
            raise ValueError('baseline failed or was unstable; no selection written')
        eligible = [c for c in configs[1:] if summary[c['name']].get('complete') and summary[c['name']].get('equivalent') and
                    summary[c['name']]['p50_ms'] <= baseline['p50_ms']*.95 and summary[c['name']]['p95_ms'] <= baseline['p95_ms']*1.05]
        selected = min(eligible, key=lambda c:summary[c['name']]['p50_ms']) if eligible else configs[0]
        write_json(args.out/'selection.json', {'development_hashes':plan['hashes'],'selected':selected,
                                              'reason':'passed fixed development gates' if eligible else 'no candidate passed; retain baseline'})
    if any('error' in r for r in runs):
        raise SystemExit('one or more runs failed; inspect runs.json')


if __name__ == '__main__':
    main()
