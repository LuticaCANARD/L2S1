"""Recompute every checked-in metric from compact records; no model required."""
from collections import defaultdict
import json
import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent/'measured-source/scripts'))

root=Path(__file__).resolve().parent
# Authenticate the archived metric implementation before importing it.
for file, sha in json.loads((root/'implementation-hashes.json').read_text()).items():
    assert hashlib.sha256((root/'measured-source'/file).read_bytes()).hexdigest() == sha, file+' measured source snapshot changed'
from benchmark_decision_performance import digest, metrics, read_rows

summary=json.loads((root/'summary.json').read_text())
assert set(summary)=={'e2b','12b'}
for model in ('e2b','12b'):
    phases={'accuracy-dev','schema','shared','holdout','jevbench'} | ({'compute'} if model=='12b' else set())
    assert set(summary[model])==phases

assert digest(root/'records.jsonl')==json.loads((root/'records-sha256.json').read_text())['records.jsonl']
for model in ('e2b','12b'):
    freeze=json.loads((root/(model+'-accuracy-selection.json')).read_text())
    assert digest(root/'protocol.json')==freeze['protocol_sha256']
    assert digest(root/'dev.jsonl')==freeze['development_input_sha256']
    assert digest(root/'holdout.jsonl')==freeze['holdout_input_sha256']
records=defaultdict(list)
for row in read_rows(root/'records.jsonl'):
    records[(row['model'],row['phase'],row['config'],row['repeat'])].append(row['record'])
for (model,phase,config,repeat),rows in records.items():
    dataset={'accuracy-dev':'dev','compute':'compute-dev'}.get(phase,phase)
    gold=json.loads((root/(dataset+'-gold.json')).read_text())
    actual=metrics(rows,gold)
    expected=summary[model][phase]['configs'][config]['runs'][repeat]
    assert actual==expected,(model,phase,config,repeat)
for model,phases in summary.items():
    for phase,section in phases.items():
        for config,values in section['configs'].items():
            for repeat in range(len(values['runs'])):
                assert (model,phase,config,repeat) in records
assert len(records)==96
assert sum(map(len,records.values()))==5364
print(f'Verified {len(records)} complete runs from checked-in records.')
