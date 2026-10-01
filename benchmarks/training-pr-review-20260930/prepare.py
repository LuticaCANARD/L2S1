"""Frozen synthetic evidence-loss regression, not a general reasoning benchmark."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training/src'))
from l2s1_training.prepare_jev_data import prepare_custom

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
root = parser.parse_args().output
root.mkdir(parents=True, exist_ok=True)
rng = random.Random(20260930091)
teams = ['amber', 'cobalt', 'jade', 'violet']
levels = ['low', 'medium', 'high']
questions = {
    'team': dict(type='choice', instructions='Read the shared directory entry matching state.record_id. Which team is assigned?',
                 criteria={k: f'The assigned team is {k}.' for k in teams}),
    'approved': dict(type='noul', instructions='Read the shared directory entry matching state.record_id. Is approved true?'),
    'priority': dict(type='score', instructions='Read the shared directory entry matching state.record_id. What is its priority?',
                    criteria={k: f'The priority is {k}.' for k in levels})}
for split, count in [('train', 16), ('development', 16), ('test', 64)]:
    rows = []
    for i in range(count):
        directory = {}
        keys = [hashlib.sha256(f'{split}:{i}:{j}:directory'.encode()).hexdigest()[:12] for j in range(4)]
        for key in keys:
            directory[key] = dict(team=rng.choice(teams), approved=rng.choice([False, True]), priority=rng.choice(levels))
        key = rng.choice(keys)
        answer = directory[key]
        gold = {}
        for name, ids, label in [('team', teams, answer['team']), ('approved', ['false', 'true'], str(answer['approved']).lower()),
                                 ('priority', levels, answer['priority'])]:
            gold[name] = dict(type=questions[name]['type'], label=label, probabilities={k: float(k == label) for k in ids})
        rows.append(dict(id=f'{split}-{i:03}', workflow='shared-directory', state={'record_id': key},
                         shared={'directory': directory}, questions=questions, gold=gold))
    (root/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
prepare_custom(root/'train.jsonl', root/'development.jsonl', root/'test.jsonl', root/'data', layout='state-first', detail='typed')
# The old preparation path silently discarded only shared; keep every other input identical.
for split in ('development', 'test'):
    requests = [json.loads(line) for line in (root/'data'/f'{split}-requests.jsonl').read_text().splitlines()]
    old = copy.deepcopy(requests)
    for r in old:
        r['request'].pop('shared')
    (root/f'{split}-before.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in old))
    (root/f'{split}-gold.json').write_text(json.dumps({r['id']: {k: v['label'] for k,v in r['gold'].items()}
        for r in map(json.loads, (root/f'{split}.jsonl').read_text().splitlines())}, indent=2)+'\n')
files = [p for p in root.rglob('*') if p.suffix in ('.json', '.jsonl')]
(root/'protocol.json').write_text(json.dumps(dict(seed=20260930091,
    selection='No search or tuning: restore shared evidence with the same state-first/typed prompt in both arms.',
    scope='Synthetic directory lookup with three decisions per case. This tests evidence preservation, not JevBench or general accuracy. Train split is preparation-only; no adapter is trained.',
    execution='fresh, CUDA, 4 threads, context4096, batch256, ubatch256, flash_attention off, one warmup, one measured pass',
    models=['Gemma 4 E2B Q8_0', 'Gemma 4 12B QAT Q4_0'],
    denominators=dict(development_cases=16, development_decisions=48, test_cases=64, test_decisions=192),
    files={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}), indent=2)+'\n')
