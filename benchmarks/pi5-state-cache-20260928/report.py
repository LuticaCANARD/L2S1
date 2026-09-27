"""Derive performance, copy overhead and numerical comparisons from raw evidence."""
import json
import math
from pathlib import Path
import statistics
import sys


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def top(result):
    return max(result['scores'], key=lambda score: score['option_probability'])['id']


def quality(rows):
    unique = {r['case_id']: r for r in rows}
    pairs = [(r, d) for r in unique.values() for d in r['result']['response']['results']]
    accepted = [(r, d) for r, d in pairs if not d['abstention_reasons']]
    correct = sum(top(d) == r['expected'][d['id']] for r, d in pairs)
    accepted_correct = sum(top(d) == r['expected'][d['id']] for r, d in accepted)
    return {'decisions': len(pairs), 'raw_top1_correct': correct, 'accepted': len(accepted),
            'accepted_correct': accepted_correct, 'raw_top1': correct / len(pairs),
            'coverage': len(accepted) / len(pairs),
            'accepted_accuracy': accepted_correct / len(accepted) if accepted else None,
            'correct_accepted_fraction': accepted_correct / len(pairs)}


def compare(left, right):
    a = {(r['case_id'], d['id']): d for r in left for d in r['result']['response']['results']}
    b = {(r['case_id'], d['id']): d for r in right for d in r['result']['response']['results']}
    assert a.keys() == b.keys()
    result = {'decisions': len(a), 'top1_changes': 0, 'acceptance_changes': 0, 'value_changes': 0,
              'max_probability_delta': 0, 'max_candidate_mass_delta': 0, 'max_raw_logit_delta': 0}
    for key in a:
        x, y = a[key], b[key]
        assert x['input_tokens'] == y['input_tokens']
        result['top1_changes'] += top(x) != top(y)
        result['acceptance_changes'] += bool(x['abstention_reasons']) != bool(y['abstention_reasons'])
        result['value_changes'] += x['value'] != y['value']
        result['max_candidate_mass_delta'] = max(result['max_candidate_mass_delta'], abs(x['candidate_mass'] - y['candidate_mass']))
        for sx, sy in zip(x['scores'], y['scores'], strict=True):
            assert sx['id'] == sy['id'] and sx['token_id'] == sy['token_id']
            result['max_probability_delta'] = max(result['max_probability_delta'], abs(sx['option_probability'] - sy['option_probability']))
            result['max_raw_logit_delta'] = max(result['max_raw_logit_delta'], abs(sx['raw_logit'] - sy['raw_logit']))
    return result


def stats(rows, samples):
    latency = [r['elapsed_ms'] for r in rows]
    active = [s for s in samples if any(r['start'] <= s['monotonic'] <= r['end'] for r in rows)]
    restores = [r['result']['state_restore'] for r in rows if r['result']['state_restore']]
    result = {'requests': len(rows), 'p50_ms': statistics.median(latency),
              'p95_ms': sorted(latency)[math.ceil(len(latency)*.95)-1],
              'input_tokens': sum(d['input_tokens'] for r in rows for d in r['result']['response']['results']),
              'reused_tokens': sum(d['reused_prefix_tokens'] for r in rows for d in r['result']['response']['results']),
              'peak_rss_gib': max(s.get('VmHWM_kib', 0) for s in samples)/1024**2,
              'active_samples': len(active),
              'undervoltage_samples': sum(bool(s['throttled'] & 1) for s in active),
              'throttling_samples': sum(bool(s['throttled'] & 4) for s in active),
              'temperature_range_c': [min(s['temperature_c'] for s in active), max(s['temperature_c'] for s in active)],
              'clock_range_hz': [min(s['arm_clock_hz'] for s in active), max(s['arm_clock_hz'] for s in active)],
              'max_swap_mib': max(s['swap_kib'] for s in samples)/1024}
    if restores:
        result['restore_metrics'] = {key: statistics.median(r[key] for r in restores)
                                     for key in ['prefill_ms', 'save_ms', 'restore_ms', 'suffix_ms', 'snapshot_bytes', 'restores']}
        result['fallback_reasons'] = sorted({r['fallback_reason'] or 'none' for r in restores})
    return result


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
blocks = {}
for path in sorted((root / 'results').glob('*/responses.jsonl')):
    rows = [r for r in lines(path) if r['phase'] == 'measured']
    expected = 5 if path.parent.name.startswith('performance-') else 12 if path.parent.name.startswith('quality-') else 2
    if len(rows) < expected:
        continue
    if rows:
        blocks[path.parent.name] = (rows, lines(path.parent / 'telemetry.jsonl'))
report = {'performance': {}, 'quality': {}, 'comparisons': {}, 'session': {}, 'rounds': {}}
configs = ['after-fresh', 'after-prefix_reuse', 'before-state_restore', 'after-state_restore']
for config in configs:
    matching = [(rows, samples) for name, (rows, samples) in blocks.items()
                if name.startswith('performance-') and name.endswith('-'+config)]
    if matching:
        rows = [r for rs, _ in matching for r in rs]
        samples = [s for _, ss in matching for s in ss]
        report['performance'][config] = {}
        for category in ['short', 'long-state', 'zero-budget']:
            selected = [r for r in rows if (r['case_id'].endswith(category) if category != 'short'
                                           else not r['case_id'].endswith(('long-state', 'zero-budget')))]
            if selected:
                report['performance'][config][category] = stats(selected, samples)
        for index, (rs, _) in enumerate(matching[1:], 1):
            report['comparisons'][f'{config}-repeat-{index}'] = compare(matching[0][0], rs)
    name = 'quality-'+config
    if name in blocks:
        report['quality'][config] = quality(blocks[name][0])
        if 'quality-after-fresh' in blocks:
            report['comparisons']['quality-fresh-vs-'+config] = compare(blocks['quality-after-fresh'][0], blocks[name][0])
for category in ['performance', 'quality']:
    left = [r for name, (rows, _) in blocks.items() if name.startswith(category+'-') and name.endswith('-before-state_restore') for r in rows]
    right = [r for name, (rows, _) in blocks.items() if name.startswith(category+'-') and name.endswith('-after-state_restore') for r in rows]
    if left and right:
        report['comparisons'][category+'-restore-before-vs-after'] = compare(left, right)
for round_id in range(4):
    names = [f'performance-{round_id}-{c}' for c in configs]
    if all(name in blocks for name in names):
        report['rounds'][str(round_id)] = {c: {r['case_id']: r['elapsed_ms'] for r in blocks[n][0]} for c, n in zip(configs, names)}
for shared in [False, True]:
    rows = [r for name, (rs, _) in blocks.items() if name.startswith('session-') and name.endswith('-'+str(shared)) for r in rs]
    samples = [s for name, (_, ss) in blocks.items() if name.startswith('session-') and name.endswith('-'+str(shared)) for s in ss]
    if rows:
        report['session'][str(shared)] = {case: stats([r for r in rows if r['case_id'] == case], samples) for case in sorted({r['case_id'] for r in rows})}
        if shared:
            fresh = [r for name, (rs, _) in blocks.items() if name.startswith('session-') and name.endswith('-False') for r in rs]
            report['comparisons']['fresh-vs-shared-session'] = compare(fresh, rows)
(root / 'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
