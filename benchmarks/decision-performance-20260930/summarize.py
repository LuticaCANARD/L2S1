"""Recompute this public-item review; no inference or threshold selection."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
records = [json.loads(line) for line in (ROOT / 'records.jsonl').read_text().splitlines()]
provenance = json.loads((ROOT / 'provenance.json').read_text())
assert hashlib.sha256((ROOT / 'records.jsonl').read_bytes()).hexdigest() == provenance['records_sha256']
models = {name: [r for r in records if r['model'] == name] for name in ('gemma4-e2b', 'gemma4-12b', 'laya')}
ids = {r['task_id'] for r in models['gemma4-e2b']}
assert len(ids) == 231
for rows in models.values():
    assert len(rows) == 231 and {r['task_id'] for r in rows} == ids

def metrics(rows):
    accepted = [r for r in rows if r['accepted']]
    return dict(n=len(rows), correct=sum(r['correct'] for r in rows),
                accepted=len(accepted), accepted_correct=sum(r['correct'] for r in accepted),
                accepted_wrong=sum(not r['correct'] for r in accepted),
                abstained=len(rows)-len(accepted),
                high_confidence_wrong_099=sum(not r['correct'] and max(r['probs'].values()) >= .99 for r in rows))

summary = {'models': {name: metrics(rows) for name, rows in models.items()}}
for name, rows in models.items():
    summary['models'][name]['families'] = {
        family: metrics([r for r in rows if r['family'] == family])
        for family in sorted({r['family'] for r in rows})}
large = {r['task_id']: r for r in models['gemma4-12b']}
cascade = [r if r['accepted'] else large[r['task_id']] for r in models['gemma4-e2b']]
summary['cascade_replay'] = metrics(cascade)
summary['cascade_replay'].update(routed_to_12b=18,
    rule='Keep E2B when its existing 0.8/0.05 policy accepts; otherwise use 12B with its own policy',
    latency='Not measured: no model co-residency or routing overhead included',
    scope='Exploratory replay on public evaluation items; not held-out validation')
assert summary['cascade_replay']['correct'] == 166
assert summary['cascade_replay']['accepted_correct'] == 159
assert summary['models']['gemma4-12b']['correct'] == 194
assert summary['models']['laya']['correct'] == 133
(ROOT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('Verified 693 records, 231 identical item IDs per model, and fixed-policy cascade replay.')
