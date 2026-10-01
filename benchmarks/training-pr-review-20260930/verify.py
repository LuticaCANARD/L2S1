"""Recompute reported metrics and paired case-bootstrap intervals from compact records."""
import collections
import json
from pathlib import Path
import random
import sys
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root.parents[1]/'scripts'))
from benchmark_decision_performance import digest, metrics, outcomes, compare

summary = json.loads((root/'summary.json').read_text())
protocol = json.loads((root/'protocol.json').read_text())
assert summary['protocol_sha256'] == digest(root/'protocol.json')
for name, sha in protocol['files'].items():
    assert digest(root/name) == sha, name
assert digest(root/'records.jsonl') == summary['records_sha256']
records = collections.defaultdict(list)
for line in (root/'records.jsonl').read_text().splitlines():
    r = json.loads(line)
    records[(r['model'], r['phase'], r['arm'])].append(r['record'])
assert set(records) == {(m,p,a) for m in ('e2b','12b') for p in ('development','test') for a in ('before','after')}
assert len(summary['runs']) == len(records)
assert {(r['model'], r['phase'], r['arm']) for r in summary['runs']} == set(records)
for run in summary['runs']:
    key = (run['model'], run['phase'], run['arm'])
    gold = json.loads((root/f'{run["phase"]}-gold.json').read_text())
    assert metrics(records[key],gold) == run['metrics'], key
for model in ('e2b','12b'):
    gold = json.loads((root/'test-gold.json').read_text())
    before = outcomes(records[(model,'test','before')], gold)
    after = outcomes(records[(model,'test','after')], gold)
    deltas = []
    for case, answers in gold.items():
        def correct(which):
            return sum(which[(case,d)]['top'] == expected for d,expected in answers.items())
        deltas.append((correct(after)-correct(before))/len(answers))
    rng = random.Random(20260930091)
    draws = sorted(sum(rng.choices(deltas,k=len(deltas)))/len(deltas) for _ in range(10000))
    print(model, 'test raw delta', sum(deltas)/len(deltas), 'case-bootstrap 95% interval', draws[249], draws[9749])
assert sum(map(len, records.values())) == 320
assert sum(len(r['response']['results']) for group in records.values() for r in group) == 960
print('Verified 8 complete paired runs, 320 cases / 960 decision predictions.')

parallel = json.loads((root/'parallel-validation.json').read_text())
assert digest(root/'parallel-records.jsonl') == parallel['records_sha256']
parallel_records = collections.defaultdict(list)
for line in (root/'parallel-records.jsonl').read_text().splitlines():
    r = json.loads(line)
    parallel_records[r['model']].append(r['record'])
assert set(parallel_records) == {'e2b', '12b'}
assert len(parallel['runs']) == 2
assert {r['model'] for r in parallel['runs']} == set(parallel_records)
gold = json.loads((root/'test-gold.json').read_text())
for run in parallel['runs']:
    rows = parallel_records[run['model']]
    assert metrics(rows, gold) == run['metrics']
    assert compare(records[(run['model'], 'test', 'after')], rows, gold) == run['comparison']
print('Verified parallel inference checks: 128 case predictions / 384 decision predictions.')
