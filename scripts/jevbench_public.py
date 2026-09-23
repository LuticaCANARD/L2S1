#!/usr/bin/env python3
"""Evaluate a frozen public JevBench split with the local GGUF JSONL evaluator."""
import argparse
import collections
import hashlib
import json
import math
import pathlib
import subprocess
import sys

JEVBENCH_REVISION = 'f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84'
SPLITS = {'easy': 48, 'original': 72, 'hard': 111}


def sha256(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    pathlib.Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def write_jsonl(path, values):
    with pathlib.Path(path).open('x') as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def to_request(task):
    """Allowlist inference fields: never send expected, rationale or gold_probs."""
    question, labels = task['question'], task['labels']
    criteria = question['criteria']
    if not 2 <= len(labels) <= 26 or len(set(labels)) != len(labels):
        raise ValueError('Invalid candidate set')
    if question['type'] == 'noul':
        if labels != ['no', 'yes'] or set(criteria) != {'false', 'true'}:
            raise ValueError('Noncanonical binary task')
        kind = {'type': 'binary', 'false_label': criteria['false'], 'true_label': criteria['true']}
    elif question['type'] == 'score':
        if labels != [str(i) for i in range(len(criteria))] or not isinstance(criteria, list):
            raise ValueError('Noncanonical ordinal task')
        kind = {'type': 'ordinal', 'levels': [
            {'id': label, 'criterion': criteria[i], 'value': i} for i, label in enumerate(labels)]}
    elif question['type'] == 'choice':
        if set(criteria) != set(labels):
            raise ValueError('Choice rubric differs from labels')
        kind = {'type': 'choice', 'options': [
            {'id': label, 'criterion': criteria[label]} for label in labels]}
    else:
        raise ValueError('Unsupported question type')
    return {'id': task['id'], 'request': {'state': task['state'], 'decisions': [
        {'id': 'decision', 'instruction': question['instructions'], 'kind': kind}]}}


def exact_probabilities(task, result):
    """Map native candidate softmax to canonical labels without calibration."""
    mapping = {'false': 'no', 'true': 'yes'} if task.question['type'] == 'noul' else {}
    scores = result['scores']
    labels = [mapping.get(score['id'], score['id']) for score in scores]
    if labels != task.labels:
        raise ValueError('Returned candidate order/set differs from the request')
    return {label: score['option_probability'] for label, score in zip(labels, scores)}


def score_predictions(tasks, predictions, score_task, model_name):
    by_id = {task.id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError('Duplicate task IDs')
    seen, records, selective = set(), [], []
    for prediction in predictions:
        tid = prediction['id']
        if tid not in by_id or tid in seen:
            raise ValueError('Duplicate or unknown prediction ID')
        seen.add(tid)
        task = by_id[tid]
        error, probabilities, result = prediction.get('error'), None, None
        elapsed = prediction['elapsed_ms']
        if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Invalid elapsed time')
        if not error:
            try:
                results = prediction['response']['results']
                if len(results) != 1 or results[0]['id'] != 'decision':
                    raise ValueError('Expected exactly one decision')
                result = results[0]
                if result['truncated']:
                    raise ValueError('Truncated input')
                probabilities = exact_probabilities(task, result)
            except (KeyError, TypeError, ValueError) as exc:
                error = str(exc)
        scored = score_task(probabilities or {}, task)
        ok = not error and scored['valid']
        records.append({
            'task_id': tid, 'family': task.family, 'split': task.split, 'group': task.group,
            'status': 'ok' if ok else 'failed', 'ok': ok, **scored,
            'probs_as_returned': probabilities, 'probs_source': 'native_candidate_softmax',
            'model': model_name, 'error': error or scored.get('error'),
            'latency_s': elapsed / 1000, 'cost_usd': None,
            'cost_basis': 'local_gpu_no_provider_tariff',
            'usage': {'input_tokens': result['input_tokens'], 'output_tokens': 0} if result else {},
        })
        accepted = False
        selected = None
        if ok:
            value = result['value']
            if value['type'] == 'binary':
                selected = {False: 'no', True: 'yes', None: None}[value['value']]
            else:
                selected = value['selected']
            accepted = selected is not None
        selective.append({'id': tid, 'accepted': accepted,
                          'correct': accepted and str(selected) == str(task.expected),
                          'abstained': ok and not accepted, 'error': not ok,
                          'candidate_mass': result.get('candidate_mass') if result else None,
                          'abstention_reasons': result.get('abstention_reasons') if result else None})
    if seen != set(by_id):
        raise ValueError(f'Missing predictions: {len(set(by_id) - seen)}')
    return records, selective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jevbench', type=pathlib.Path, required=True)
    parser.add_argument('--evaluator', type=pathlib.Path, required=True)
    parser.add_argument('--model', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    upstream = args.jevbench.resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != JEVBENCH_REVISION:
        parser.error('Upstream revision differs from the reviewed frozen version')
    if subprocess.check_output(['git', '-C', str(upstream), 'status', '--porcelain'], text=True).strip():
        parser.error('Upstream checkout must be clean')
    sys.path.insert(0, str(upstream))
    from jevbench.tasks import Task, dataset_hash
    from jevbench.scoring import score_task
    from jevbench.summarize import summarize
    from jevbench.composite_v13 import tvd

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    tasks, tiers, source_files = [], {}, {}
    for split, count in SPLITS.items():
        path = upstream / 'datasets/public' / (split + '.jsonl')
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != count:
            raise ValueError('Frozen split size changed')
        for row in rows:
            task = Task.from_dict(row)
            if task.id in tiers or task.split != 'public':
                raise ValueError('Repeated ID or nonpublic task')
            tiers[task.id] = split
            tasks.append(task)
        source_files[path.name] = sha256(path)
    request_path = out / 'requests.jsonl'
    write_jsonl(request_path, [to_request(json.loads(task.to_json())) for task in tasks])
    write_jsonl(out / 'tasks-with-gold.jsonl', [json.loads(task.to_json()) for task in tasks])
    predictions_path = out / 'predictions.jsonl'
    command = [str(args.evaluator.resolve()), '--model', str(args.model.resolve()),
               '--input', str(request_path), '--output', str(predictions_path),
               '--context', str(args.context), '--batch', '256', '--threads', '4',
               '--execution-mode', 'fresh', '--prompt-layout', 'legacy', '--warmup']
    if args.device == 'cuda':
        command.append('--cuda')
    manifest = {'jevbench_revision': revision, 'dataset_hash': dataset_hash(tasks),
                'source_sha256': source_files, 'requests_sha256': sha256(request_path),
                'evaluator_sha256': sha256(args.evaluator), 'model_sha256': sha256(args.model),
                'model_name': args.model.name, 'device': args.device,
                'adapter_sha256': sha256(__file__), 'command': command,
                'planned': len(tasks), 'tier_counts': SPLITS,
                'label_order': 'canonical task.labels; binary false/true mapped to no/yes',
                'probability_origin': 'native candidate-token softmax, not learned calibration',
                'adapters': {'lora': None, 'output_head': None, 'calibration': None},
                'latency_scope': 'Rust decide_batch serial calls, batch size 1; no HTTP; load and one warmup excluded',
                'scope': 'Public 231 items only; no official full-suite score or rank; no training or tuning'}
    write_json(out / 'manifest.json', manifest)
    with (out / 'inference.stderr.log').open('w') as stderr, (out / 'gpu-memory.csv').open('w') as gpu_log:
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits', '--loop-ms=200'],
                                   stdout=gpu_log, stderr=subprocess.DEVNULL)
        try:
            result = subprocess.run(command, stderr=stderr, timeout=args.timeout)
        finally:
            monitor.terminate()
            monitor.wait()
    manifest['inference_exit_code'] = result.returncode
    write_json(out / 'manifest.json', manifest)
    if result.returncode:
        raise RuntimeError('Inference failed; retained partial evidence, no complete-result claim')
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines()]
    records, selective = score_predictions(tasks, predictions, score_task, args.model.name + '/l2s1')
    backends = [p['response']['backend'] for p in predictions if 'response' in p]
    if args.device == 'cuda' and any(not b['offload_requested'] or '3060' not in b['offload_device'] for b in backends):
        raise ValueError('Run did not use the expected RTX 3060')
    summary = summarize(tasks, records)
    summary['backend'] = backends[0] if backends else None
    samples = [float(line.split(',')[1]) for line in (out / 'gpu-memory.csv').read_text().splitlines()
               if len(line.split(',')) == 3]
    summary['gpu_memory'] = {'peak_board_used_mib': max(samples) if samples else None,
                             'sampling_ms': 200, 'scope': 'whole GPU, includes baseline and other processes'}
    summary['max_input_tokens'] = max((r['usage'].get('input_tokens', 0) for r in records), default=0)
    summary['tiers'] = {tier: summarize([t for t in tasks if tiers[t.id] == tier],
                                      [r for r in records if tiers[r['task_id']] == tier]) for tier in SPLITS}
    totals = collections.Counter()
    for row in selective:
        for key in ('accepted', 'correct', 'abstained', 'error'):
            totals[key] += int(row[key])
    totals['wrong_accepted'] = totals['accepted'] - totals['correct']
    summary['selective_policy'] = dict(totals, total=len(tasks),
        coverage=totals['accepted'] / len(tasks),
        accepted_accuracy=totals['correct'] / totals['accepted'] if totals['accepted'] else None)
    by_id = {r['task_id']: r for r in records}
    distances = [tvd(by_id[t.id]['probs'], t.provenance['gold_probs'], t.labels)
                 for t in tasks if t.provenance.get('gold_probs') and by_id[t.id]['valid']]
    summary['hard_gold_distribution_tvd'] = {'n': len(distances), 'mean': sum(distances) / len(distances) if distances else None}
    summary['scope'] = manifest['scope']
    summary['latency_scope'] = manifest['latency_scope']
    write_jsonl(out / 'jevbench-records.jsonl', records)
    write_jsonl(out / 'selective-policy.jsonl', selective)
    write_json(out / 'summary.json', summary)
    for tier, item in summary['tiers'].items():
        print(f"{tier}: {item['n_correct']}/{item['n_scorable']} correct, ECE={item['ece']['ece'] if item['ece'] else None}")
    print(f"ALL: {summary['n_correct']}/{summary['n_scorable']}; errors={totals['error']}; output={out}")
    return int(totals['error'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
