#!/usr/bin/env python3
"""Independently recount public JevBench predictions and report a model matrix."""
import argparse
import collections
import json
import math
import pathlib


def load(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def audit(directory):
    summary = load(directory / 'summary.json')
    manifest = load(directory / 'manifest.json')
    tasks = {t['id']: t for t in lines(directory / 'tasks-with-gold.jsonl')}
    predictions = lines(directory / 'predictions.jsonl')
    assert len(tasks) == len(predictions) == 231
    assert {p['id'] for p in predictions} == set(tasks)
    correct = errors = 0
    brier, confidences, bins = [], [], collections.defaultdict(list)
    for prediction in predictions:
        if prediction.get('error'):
            errors += 1
            continue
        task = tasks[prediction['id']]
        result = prediction['response']['results'][0]
        assert not result['truncated']
        mapping = {'false': 'no', 'true': 'yes'} if task['question']['type'] == 'noul' else {}
        labels = [mapping.get(s['id'], s['id']) for s in result['scores']]
        assert labels == task['labels']
        probabilities = dict(zip(labels, [s['option_probability'] for s in result['scores']]))
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values())
        assert abs(sum(probabilities.values()) - 1) <= 0.001
        chosen = min(probabilities, key=lambda label: (-probabilities[label], label))
        hit = chosen == str(task['expected'])
        correct += hit
        confidence = max(probabilities.values())
        bins[min(int(confidence * 10), 9)].append((confidence, hit))
        if confidence >= 0.9:
            confidences.append((confidence, hit))
        brier.append(sum((p - (label == str(task['expected']))) ** 2 for label, p in probabilities.items()))
    assert correct == summary['n_correct']
    assert errors == summary['selective_policy']['error']
    if brier:
        measured_brier = sum(brier) / len(brier)
        measured_ece = sum(abs(sum(p - hit for p, hit in group)) for group in bins.values()) / len(brier)
        assert abs(measured_brier - summary['brier_mean']) < 1e-10
        assert abs(measured_ece - summary['ece']['ece']) < 1e-10
    assert summary['model_identities'] == [manifest['model_name'] + '/l2s1']
    return dict(model=manifest['model_name'], correct=correct, total=len(tasks), errors=errors,
                accuracy=summary['accuracy'], hard_accuracy=summary['tiers']['hard']['accuracy'],
                easy_correct=summary['tiers']['easy']['n_correct'],
                original_correct=summary['tiers']['original']['n_correct'],
                hard_correct=summary['tiers']['hard']['n_correct'],
                brier=summary['brier_mean'], ece=summary['ece']['ece'] if summary['ece'] else None,
                p50_ms=summary['latency']['p50_s'] * 1000,
                p95_ms=summary['latency']['p95_s'] * 1000,
                peak_gpu_mib=summary['gpu_memory']['peak_board_used_mib'],
                selective=summary['selective_policy'],
                confidence_ge_90=dict(n=len(confidences),
                    accuracy=sum(hit for _, hit in confidences)/len(confidences) if confidences else None),
                requests_sha256=manifest['requests_sha256'], evaluator_sha256=manifest['evaluator_sha256'],
                independently_verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('matrix', type=pathlib.Path)
    parser.add_argument('--report', type=pathlib.Path, required=True)
    args = parser.parse_args()
    plan = load(args.matrix / 'plan.json')
    statuses = {r['id']: r for r in load(args.matrix / 'matrix-status.json')}
    completed, failures, pending = [], [], []
    for row in plan:
        directory = args.matrix / row['id']
        if (directory / 'summary.json').exists():
            completed.append(audit(directory))
        elif row['id'] in statuses:
            logs = [directory / 'inference.stderr.log', args.matrix / (row['id'] + '.log')]
            failures.append(dict(model=row['id'], status=statuses[row['id']]['status'],
                                 log_tail='\n'.join(p.read_text()[-3000:] for p in logs if p.exists())))
        else:
            pending.append(row['id'])
    assert len({r['requests_sha256'] for r in completed}) <= 1
    assert len({r['evaluator_sha256'] for r in completed}) <= 1
    completed.sort(key=lambda r: (-r['accuracy'], -r['hard_accuracy'], r['p50_ms']))
    data = dict(planned=len(plan), completed=completed, failed=failures, pending=pending)
    args.report.with_suffix('.json').write_text(json.dumps(data, indent=2) + '\n')
    text = ['# JevBench multi-model measurements', '',
            'Host: `100.66.64.91`, RTX 3060 12 GiB. Public JevBench: 231 items (48 Easy, 72 Original, 111 Hard).', '',
            f"Planned configurations: {len(plan)}. Scored: {len(completed)}. Failed before complete scoring: {len(failures)}. Pending: {len(pending)}.", '',
            '| Model | Correct / 231 | Accuracy | Hard | ECE ↓ | p50 ms | p95 ms | Peak GPU MiB | Accepted wrong | Abstained | Errors |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in completed:
        s = r['selective']
        text.append(f"| {r['model']} | {r['correct']} | {r['accuracy']:.2%} | {r['hard_accuracy']:.2%} | {r['ece']:.4f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['peak_gpu_mib']:.0f} | {s['wrong_accepted']} | {s['abstained']} | {r['errors']} |")
    text += ['', '## Method and limits', '',
             '- Same frozen public dataset, Rust evaluator binary, legacy layout, fresh requests, context 8192, batch/ubatch 256, four threads, FlashAttention off, one warmup, and default abstention thresholds.',
             '- Embedded model templates and the existing Auto prompt profile are used. No generation of reasoning tokens, LoRA, learned head, calibration, or benchmark-based prompt tuning.',
             '- Accuracy is argmax over candidate probabilities before abstention. Accepted errors, abstentions, and probability calibration remain separate.',
             '- Native predictions were independently recounted against gold labels; Brier and ECE were recomputed. Requests and binary hashes match across scored models.',
             '- Latency is serial local inference, excluding loading and warmup. Downloads can overlap runs; these single-run timings are not an SLA or directly comparable to HTTP leaderboard timings.',
             '- Peak GPU memory is whole-board usage sampled every 200 ms, including baseline; short peaks may be missed.',
             '- This is a selected hardware/runtime feasibility matrix, not every published model or every quantization. Small models may run beyond their training context; logs retain warnings.',
             '- Public subset only: no claim of the official full-suite score or rank. Ranking on this dataset is model selection evidence, not an independent production validation.', '']
    if failures:
        text += ['## Failed configurations', '']
        for f in failures:
            text += [f"### {f['model']}", '', '```text', f['log_tail'], '```', '']
    if pending:
        text += ['## Pending', '', *['- ' + x for x in pending], '']
    args.report.write_text('\n'.join(text))
    print(json.dumps({'scored': len(completed), 'failed': len(failures), 'pending': len(pending)}))


if __name__ == '__main__':
    main()
