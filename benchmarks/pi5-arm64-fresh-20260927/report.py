"""Rebuild summaries from saved native responses and telemetry; no inference."""
import argparse
import json
import math
from pathlib import Path
import statistics


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def top(result):
    return max(result['evidence']['scores'], key=lambda score: score['option_probability'])['id']


def quantile(values, q):
    return sorted(values)[max(0, math.ceil(len(values)*q)-1)]


def quality(rows):
    # Unique cases only, no inflated accuracy denominators from repeated inputs.
    unique = {row['case_id']: row for row in rows}
    decisions = [(row, result) for row in unique.values() for result in row['response']['results']]
    accepted = [(row, r) for row, r in decisions if r['status'] == 'selected']
    correct = sum(top(r) == row['expected'][r['id']] for row, r in decisions)
    accepted_correct = sum(top(r) == row['expected'][r['id']] for row, r in accepted)
    return {'unique_cases': len(unique), 'decisions': len(decisions), 'raw_top1_correct': correct,
            'raw_top1': correct/len(decisions), 'accepted': len(accepted), 'coverage': len(accepted)/len(decisions),
            'accepted_correct': accepted_correct, 'accepted_accuracy': accepted_correct/len(accepted) if accepted else None,
            'correct_accepted_fraction': accepted_correct/len(decisions)}


def compare(left, right):
    a = {(row['case_id'], r['id']): r for row in left for r in row['response']['results']}
    b = {(row['case_id'], r['id']): r for row in right for r in row['response']['results']}
    keys = sorted(a.keys() & b.keys())
    result = {'decisions': len(keys), 'top1_changes': 0, 'status_changes': 0, 'value_changes': 0, 'wire_value_changes': 0,
              'max_probability_delta': 0, 'max_candidate_mass_delta': 0, 'max_raw_logit_delta': 0,
              'exact_evidence_matches': 0, 'changed_decisions': []}
    for key in keys:
        x, y = a[key], b[key]
        scores_x = {s['id']: s for s in x['evidence']['scores']}
        scores_y = {s['id']: s for s in y['evidence']['scores']}
        assert scores_x.keys() == scores_y.keys()
        pd = max(abs(scores_x[s]['option_probability'] - scores_y[s]['option_probability']) for s in scores_x)
        ld = max(abs(scores_x[s]['raw_logit'] - scores_y[s]['raw_logit']) for s in scores_x)
        md = abs(x['evidence']['candidate_mass'] - y['evidence']['candidate_mass'])
        def selected_value(r):
            if r['status'] != 'selected':
                return None
            value = r['value']
            return value.get('selected', value.get('value'))
        tc, sc, vc = top(x) != top(y), x['status'] != y['status'], selected_value(x) != selected_value(y)
        result['wire_value_changes'] += x['value'] != y['value']
        result['top1_changes'] += tc
        result['status_changes'] += sc
        result['value_changes'] += vc
        result['exact_evidence_matches'] += x['evidence'] == y['evidence']
        result['max_probability_delta'] = max(pd, result['max_probability_delta'])
        result['max_candidate_mass_delta'] = max(md, result['max_candidate_mass_delta'])
        result['max_raw_logit_delta'] = max(ld, result['max_raw_logit_delta'])
        if tc or sc or pd > .02 or md > .02:
            result['changed_decisions'].append({'case_id': key[0], 'decision_id': key[1], 'top1': [top(x), top(y)],
                                                'status': [x['status'], y['status']], 'probability_delta': pd, 'mass_delta': md})
    return result


def summarize(rows, samples):
    latency = [r['elapsed_ms'] for r in rows]
    active = [s for s in samples if any(r['start'] <= s['monotonic'] <= r['end'] for r in rows)]
    phases = {s['phase'] for s in active}
    result = {**quality(rows), 'requests': len(rows), 'p50_ms': statistics.median(latency), 'p95_ms': quantile(latency, .95),
              'min_ms': min(latency), 'max_ms': max(latency),
              'reused_prefix_tokens': sum(r['usage'].get('reused_prefix_tokens', 0) for row in rows for r in row['response']['results']),
              'input_tokens': sum(r['usage']['input_tokens'] for row in rows for r in row['response']['results']),
              'peak_process_rss_gib': max(s.get('VmHWM_kib', 0) for s in samples)/1024**2,
              'measured_telemetry_samples': len(active), 'measured_telemetry_phases': sorted(phases)}
    if active:
        result.update({'temperature_c_min': min(s['temperature_c'] for s in active),
                       'temperature_c_max': max(s['temperature_c'] for s in active),
                       'arm_clock_hz_min': min(s['arm_clock_hz'] for s in active),
                       'arm_clock_hz_max': max(s['arm_clock_hz'] for s in active),
                       'arm_clock_hz_p50': statistics.median(s['arm_clock_hz'] for s in active),
                       'active_undervoltage_samples': sum(bool(s['throttled'] & 1) for s in active),
                       'active_throttling_samples': sum(bool(s['throttled'] & 4) for s in active),
                       'active_frequency_capped_samples': sum(bool(s['throttled'] & 2) for s in active),
                       'active_soft_temperature_limit_samples': sum(bool(s['throttled'] & 8) for s in active),
                       'historical_undervoltage_samples': sum(bool(s['throttled'] & (1 << 16)) for s in active),
                       'max_swap_mib': max(s['swap_kib'] for s in samples)/1024})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    all_rows, all_samples, blocks = [], [], {}
    for path in sorted((args.root / 'results').glob('*/responses.jsonl')):
        rows = [r for r in read_lines(path) if r['phase'] == 'measured']
        samples = read_lines(path.parent / 'telemetry.jsonl')
        all_rows.extend((path.parent.name, row) for row in rows)
        all_samples.extend(samples)
        blocks[path.parent.name] = summarize(rows, samples)
    summary = {'blocks': blocks, 'factorial': {}, 'quality': {}, 'comparisons': {}}
    for variant in ['baseline', 'kernels', 'openmp', 'combined']:
        rows = [r for name, r in all_rows if name.startswith('factorial-') and r['variant'] == variant]
        if rows:
            samples = [s for name in blocks if name.startswith('factorial-') and name.endswith('-'+variant)
                       for s in read_lines(args.root / 'results' / name / 'telemetry.jsonl')]
            summary['factorial'][variant] = summarize(rows, samples)
        name = f'quality-{variant}-fresh'
        if name in blocks:
            summary['quality'][variant] = blocks[name]
    for left, right in [('baseline', 'kernels'), ('baseline', 'openmp'), ('kernels', 'combined')]:
        x = [r for name, r in all_rows if name == f'quality-{left}-fresh']
        y = [r for name, r in all_rows if name == f'quality-{right}-fresh']
        if x and y:
            summary['comparisons'][left+'-vs-'+right] = compare(x, y)
    fresh = [r for name, r in all_rows if name == 'quality-kernels-fresh']
    for mode in ['prefix', 'fixed']:
        rows = [r for name, r in all_rows if name == 'quality-kernels-'+mode]
        if rows and fresh:
            summary['comparisons']['kernels-fresh-vs-'+mode] = compare(fresh, rows)
            half = len(rows)//2
            summary['comparisons']['kernels-'+mode+'-repeat'] = compare(rows[:half], rows[half:])
    for variant in summary['factorial']:
        rows = [r for name, r in all_rows if name.startswith('factorial-') and r['variant'] == variant]
        summary['comparisons'][variant+'-fresh-repeat'] = [compare(rows[:3], rows[i:i+3]) for i in range(3, len(rows), 3)]
    fallback = [r for name, r in all_rows if name == 'fallback']
    baseline = [r for name, r in all_rows if name == 'factorial-0-baseline']
    if fallback and baseline:
        summary['comparisons']['baseline-vs-masked-fallback'] = compare(baseline, fallback)
    summary['round_ratios'] = []
    for round_id in range(4):
        names = {v: f'factorial-{round_id}-{v}' for v in ['baseline', 'kernels', 'openmp', 'combined']}
        if all(name in blocks for name in names.values()):
            medians = {v: blocks[name]['p50_ms'] for v, name in names.items()}
            summary['round_ratios'].append({'round': round_id, 'medians_ms': medians,
                'baseline_over_kernels': medians['baseline']/medians['kernels'],
                'baseline_over_openmp': medians['baseline']/medians['openmp'],
                'kernels_over_combined': medians['kernels']/medians['combined']})
    historical_path = args.root / 'historical-benchmark.json'
    if historical_path.exists():
        historical = json.loads(historical_path.read_text())
        history = [{'case_id': row['case'], 'expected': row['expected'], 'response': {'results': [
            {'id': r['id'], 'value': r['value'], 'status': 'abstained' if r['abstention_reasons'] else 'selected',
             'evidence': r} for r in row['results']]}} for row in historical['samples'] if row['pass'] == 0]
        summary['historical_quality'] = quality(history)
        for variant in ['baseline', 'kernels']:
            current = [r for name, r in all_rows if name == f'quality-{variant}-fresh']
            if current:
                summary['comparisons']['historical-vs-'+variant] = compare(history, current)
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'blocks'}, indent=2))


if __name__ == '__main__':
    main()
