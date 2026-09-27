"""Validate completed evidence without rerunning inference or relaxing tolerances."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
report = json.loads((root / 'summary.json').read_text())
inputs = json.loads((root / 'inputs.json').read_text())
assert inputs['model_sha256'] == 'b205840c5dcef55078e37d344677869a714ffd42a4ae448c48dcfb52e4bb10d5'
assert len(report['rounds']) == 4 and len(report['performance']) == 4
assert len(report['quality']) == 4 and all(q['decisions'] == 36 for q in report['quality'].values())
for name, config in report['performance'].items():
    assert config['short']['requests'] == 12
    assert config['long-state']['requests'] == config['zero-budget']['requests'] == 4
    assert config['short']['reused_tokens'] == 0
    if name == 'after-fresh':
        assert all(run['reused_tokens'] == 0 for run in config.values())
    else:
        assert config['long-state']['reused_tokens'] == 4 * 512
    if name.endswith('state_restore'):
        long = config['long-state']
        assert long['restore_metrics']['restores'] == 2
        assert long['restore_metrics']['snapshot_bytes'] > 0
        assert long['fallback_reasons'] == ['none']
        assert config['zero-budget']['reused_tokens'] == 0
        assert config['zero-budget']['fallback_reasons'] == ['snapshot_memory_budget']
assert report['performance']['after-state_restore']['zero-budget']['restore_metrics']['prefill_ms'] == 0
assert report['performance']['before-state_restore']['zero-budget']['restore_metrics']['prefill_ms'] > 0
for name, comparison in report['comparisons'].items():
    assert comparison['top1_changes'] == comparison['acceptance_changes'] == comparison['value_changes'] == 0, name
    assert comparison['max_probability_delta'] < .02, (name, comparison)
    assert comparison['max_candidate_mass_delta'] < .02, (name, comparison)
assert report['comparisons']['quality-restore-before-vs-after']['decisions'] == 36
assert report['comparisons']['fresh-vs-shared-session']['decisions'] == 6
for session in report['session'].values():
    assert all(run['requests'] == 4 for run in session.values())
for name, tests in [('native-tests.log', 2), ('force-clear-tests.log', 1)]:
    log = (root / name).read_text()
    assert log.count('test result: ok. 1 passed') == tests and 'test result: FAILED' not in log
identity = json.loads((root / 'identity.json').read_text())
assert identity['native_shared_libraries_identical']
assert identity['upstream_commit'] == '3d82ef62d47fd74e18f36c5eccbdcf965b617b17'
for name, expected in identity['source_sha256'].items():
    source = root / 'measured-probe.rs' if name == 'examples/state_cache_probe.rs' else root.parent.parent / name
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, name
smoke = json.loads((root / 'installed-smoke.json').read_text())
assert len(smoke) == 4 and all(run['error_recovery'] for run in smoke)
assert [r['reused_tokens'] for r in smoke] == [0, 512, 0, 0]
print('Complete measurement coverage, score equivalence, real cache use, budget fallback and installed-runtime checks passed')
