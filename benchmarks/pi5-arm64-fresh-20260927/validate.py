"""Check saved measurement coverage and regression boundaries without inference."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
summary = json.loads((root / 'summary.json').read_text())
for variant, run in summary['factorial'].items():
    assert run['requests'] == 12 and run['reused_prefix_tokens'] == 0
    assert summary['quality'][variant]['decisions'] == 36
    for comparison in summary['comparisons'][variant + '-fresh-repeat']:
        assert comparison['decisions'] == 9
        assert comparison['top1_changes'] == comparison['status_changes'] == 0
        assert comparison['max_probability_delta'] == comparison['max_candidate_mass_delta'] == 0

for name in ['baseline-vs-openmp', 'kernels-vs-combined', 'kernels-fresh-vs-prefix',
             'kernels-prefix-repeat', 'kernels-fixed-repeat', 'historical-vs-kernels',
             'baseline-vs-masked-fallback']:
    comparison = summary['comparisons'][name]
    assert comparison['decisions'] == (9 if name.endswith('fallback') else 36)
    assert comparison['top1_changes'] == comparison['status_changes'] == 0
    assert comparison['max_probability_delta'] == comparison['max_candidate_mass_delta'] == 0

# Cross-kernel and fresh-versus-fixed changes are reported, not erased by a tolerance.
reuse = json.loads((root / 'prefix-equivalence.json').read_text())
assert reuse['comparisons'] == 45 and reuse['reused_prefix_tokens'] > 0
assert reuse['changed_top1'] == reuse['changed_selected'] == 0
assert reuse['max_probability_delta'] < .02 and reuse['max_candidate_mass_delta'] < .02
sdk = json.loads((root / 'sdk-smoke.json').read_text())
for run in sdk['runs']:
    assert [r['evidence'] for r in run['cold']['results']] == [r['evidence'] for r in run['warm']['results']]
    assert (run['warm_reused_tokens'] == 0) == (run['execution_override'] == 'fresh')
for filename in ['rust-tests.log', 'native-equivalence.log']:
    log = (root / filename).read_text()
    assert 'test result: ok.' in log and 'test result: FAILED' not in log
print('Saved fresh, request-local reuse, fixed cold/warm, fallback and SDK checks passed')
