#!/usr/bin/env python3
"""Paired native prompt evaluation. Gold is read only after inference exits.

Inputs: <phase>-requests.jsonl ({id, request}) and <phase>-gold.jsonl
({id, gold: {decision_id: {label}}}). No training or threshold fitting.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def measure(inputs, gold, predictions):
    assert [r['id'] for r in inputs] == [r['id'] for r in predictions]
    assert len({r['id'] for r in inputs}) == len(inputs)
    labels = {r['id']: r['gold'] for r in gold}
    assert len(labels) == len(gold)
    assert set(labels) == {r['id'] for r in inputs}
    hits, accepted_hits, confs, losses, gated, times = [], [], [], [], [], []
    by_type, by_state = {}, []
    tokens = reused = 0
    for request, output in zip(inputs, predictions):
        assert 'error' not in output, output
        results = output['response']['results']
        decisions = request['request']['decisions']
        assert set(labels[request['id']]) == {d['id'] for d in decisions}
        assert [d['id'] for d in decisions] == [d['id'] for d in results]
        state_hits = []
        times.append(output['elapsed_ms'])
        for decision, result in zip(decisions, results):
            assert not result['truncated']
            scores = result['scores']
            assert len({s['id'] for s in scores}) == len(scores)
            p = {s['id']: s['option_probability'] for s in scores}
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
            assert abs(sum(p.values()) - 1) < 1e-7
            label = str(labels[request['id']][decision['id']]['label'])
            assert label in p
            best = max(p, key=p.get)
            hit = int(best == label)
            hits.append(hit)
            state_hits.append(hit)
            confs.append(p[best])
            losses.append(-math.log(max(p[label], 1e-30)))
            value = result['value']
            selected = value.get('selected')
            if value['type'] == 'binary':
                selected = value.get('value')
                if selected is not None:
                    selected = str(selected).lower()
            if selected is not None:
                selected_id = selected['id'] if isinstance(selected, dict) else selected
                accepted_hits.append(int(selected_id == label))
            gated.append(p[best] >= .8 and result['candidate_mass'] >= .05
                         and sum(v == p[best] for v in p.values()) == 1)
            by_type.setdefault(decision['kind']['type'], []).append(hit)
            tokens += result['input_tokens']
            reused += result['reused_prefix_tokens']
        by_state.append(state_hits)
    n = len(hits)
    ece = 0
    for i in range(15):
        idx = [j for j, c in enumerate(confs) if i / 15 < c <= (i + 1) / 15]
        if idx:
            ece += len(idx) / n * abs(statistics.mean(confs[j] for j in idx)
                                    - statistics.mean(hits[j] for j in idx))
    return dict(n=n, correct=sum(hits), raw_accuracy=sum(hits)/n,
                returned=len(accepted_hits), coverage=len(accepted_hits)/n,
                accepted_correct=sum(accepted_hits),
                accepted_accuracy=statistics.mean(accepted_hits) if accepted_hits else None,
                accepted_correct_all=sum(accepted_hits)/n,
                abstained=n-len(accepted_hits), nll=statistics.mean(losses), ece15=ece,
                historical_gate=dict(accepted=sum(gated), correct=sum(h*g for h,g in zip(hits,gated))),
                by_type={k: dict(n=len(v), correct=sum(v), accuracy=statistics.mean(v))
                         for k,v in by_type.items()},
                latency=dict(p50_ms=statistics.median(times),
                             p95_ms=sorted(times)[math.ceil(len(times)*.95)-1],
                             total_ms=sum(times)), input_tokens=tokens,
                reused_prefix_tokens=reused, state_hits=by_state)


def run(args):
    output = args.results / args.phase / args.model_name / args.profile
    output.mkdir(parents=True, exist_ok=True)
    input_path = args.results / (args.phase + '-requests.jsonl')
    prediction_path = output / 'predictions.jsonl'
    detail = args.profile.removeprefix('model-') if args.profile.startswith('model-') else 'minimal'
    profile = 'model' if args.profile.startswith('model-') else args.profile
    command = [str(args.binary), '--model', str(args.model), '--input', str(input_path),
               '--output', str(prediction_path), '--cuda', '--context', '8192',
               '--batch', '256', '--ubatch', '256', '--threads', '4',
               '--flash-attention', 'off', '--execution-mode', 'fresh',
               '--prompt-layout', 'legacy', '--prompt-detail', detail,
               '--prompt-profile', profile, '--min-top-probability', '0',
               '--min-candidate-mass', '0', '--warmup']
    manifest = dict(command=command, started=time.time(), binary_sha256=digest(args.binary),
                    model_sha256=digest(args.model), input_sha256=digest(input_path),
                    status='running', warmup_requests=1,
                    transport='native CUDA, per-request latency; excludes load and warmup')
    if args.phase in ('holdout', 'typed-test'):
        selection = args.results / 'selection.json'
        manifest['selection_sha256'] = digest(selection)
        chosen = json.loads(selection.read_text())[args.model_name]['profile']
        assert args.profile in ('model', chosen), 'holdout configuration was not frozen'
    save(output / 'manifest.json', manifest)
    with (output / 'run.log').open('w') as log, (output / 'gpu.csv').open('w') as gpu:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,power.draw,temperature.gpu', '--format=csv', '-lms', '200'], stdout=gpu)
        try:
            completed = subprocess.run(['/usr/bin/time', '-v', *command], stdout=log, stderr=log)
        finally:
            monitor.terminate()
            monitor.wait()
    assert completed.returncode == 0, str(output / 'run.log')
    result = measure(rows(input_path), rows(args.results / (args.phase + '-gold.jsonl')),
                     rows(prediction_path))
    save(output / 'summary.json', result)
    manifest.update(status='complete', finished=time.time(), predictions_sha256=digest(prediction_path))
    save(output / 'manifest.json', manifest)
    print(args.phase, args.model_name, args.profile, result['correct'], '/', result['n'], result['latency'], flush=True)


def freeze(root):
    plan = json.loads((root / 'plan.json').read_text())
    selection = {}
    for model, profiles in plan['candidates'].items():
        scores = [(p, json.loads((root/'dev'/model/p/'summary.json').read_text())) for p in profiles]
        chosen, summary = max(scores, key=lambda x: (x[1]['correct'], x[0] == 'model'))
        selection[model] = dict(profile=chosen, correct=summary['correct'], n=summary['n'],
                                candidates={p:s['correct'] for p,s in scores}, frozen_at=time.time())
    path = root / 'selection.json'
    assert not path.exists(), 'selection already frozen'
    save(path, selection)
    print(json.dumps(selection, indent=2))


def compare(root, phase='holdout'):
    selected = json.loads((root/'selection.json').read_text())
    report = {}
    for model, selection in selected.items():
        baseline = json.loads((root/phase/model/'model'/'summary.json').read_text())
        improved = json.loads((root/phase/model/selection['profile']/'summary.json').read_text())
        a, b = baseline.pop('state_hits'), improved.pop('state_hits')
        assert len(a) == len(b) and all(len(x) == len(y) for x,y in zip(a,b))
        rng = random.Random(20261001)
        deltas = []
        for _ in range(5000):
            idx = rng.choices(range(len(a)), k=len(a))
            deltas.append(sum(sum(b[i])-sum(a[i]) for i in idx)/sum(len(a[i]) for i in idx))
        deltas.sort()
        report[model] = dict(profile=selection['profile'], baseline=baseline, improved=improved,
                             delta=improved['raw_accuracy']-baseline['raw_accuracy'],
                             state_bootstrap_95=[deltas[125], deltas[4874]],
                             corrected=sum(not x and y for aa,bb in zip(a,b) for x,y in zip(aa,bb)),
                             regressed=sum(x and not y for aa,bb in zip(a,b) for x,y in zip(aa,bb)))
    save(root/('comparison-'+phase+'.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['run', 'freeze', 'compare'])
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--binary', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--model-name')
    parser.add_argument('--phase', choices=['dev', 'holdout', 'typed-test', 'regression'])
    parser.add_argument('--profile', choices=['model', 'model-typed', 'model-typed-examples', 'winnow', 'gemma4-decision'])
    args = parser.parse_args()
    if args.operation == 'run':
        run(args)
    elif args.operation == 'freeze':
        freeze(args.results)
    else:
        compare(args.results, args.phase or 'holdout')
