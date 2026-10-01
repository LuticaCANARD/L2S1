import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from prepare_jev_data import check_disjoint, decision, distribution
from report_jev import answer, native_selection, summarize
from jev_model_profiles import DEFAULT_PROFILES, read_profile
from report_jev_rule_regression import unique_top1


class JevContractTests(unittest.TestCase):
    def result(self, ps):
        return dict(id='q', scores=[dict(id=k, option_probability=v) for k,v in ps.items()])

    def test_noul_is_probability_not_boolean(self):
        d = decision('q', dict(type='noul', instructions='True?'))
        self.assertEqual(answer(d, self.result({'false': .3, 'true': .7})), dict(type='noul', noul=.7))

    def test_score_is_expectation_in_rubric_order(self):
        d = decision('q', dict(type='score', instructions='Risk?', criteria=['low', 'medium', 'high']))
        a = answer(d, self.result({'2': .2, '0': .3, '1': .5}))
        self.assertAlmostEqual(a['score'], .9)
        self.assertEqual(a['legend'], {'0':'low', '1':'medium', '2':'high'})

    def test_choice_uniform_confidence_and_first_option_tie(self):
        d = decision('q', dict(type='choice', instructions='Route?', criteria={'b':'Billing','a':'Other'}))
        a = answer(d, self.result({'a': .5, 'b': .5}))
        self.assertEqual(a['choice'], 'b')
        self.assertAlmostEqual(a['confidence'], 0)

    def test_bad_probability_rejected(self):
        for p in ({'a':.8,'b':.8}, {'a':float('nan'),'b':.5}, {'a':1}, {'a':-.1,'b':1.1}):
            with self.assertRaises(ValueError):
                distribution(p, ['a','b'])

    def test_ollaya_confidence_is_not_entropy(self):
        d = decision('q', dict(type='choice', instructions='Route?', criteria={'a':'A','b':'B','c':'C'}))
        a = answer(d, self.result({'a':.61, 'b':.35, 'c':.04}))
        self.assertEqual(a['confidence'], .415)

    def test_wire_rounding_preserves_unrounded_winner(self):
        d = decision('q', dict(type='choice', instructions='Route?', criteria={'a':'A','b':'B'}))
        a = answer(d, self.result({'a':.499999, 'b':.500001}))
        self.assertEqual(a['choice'], 'b')
        self.assertEqual(a['probabilities'], {'a':.5,'b':.5})

    def test_duplicate_or_missing_score_rejected(self):
        d = decision('q', dict(type='noul', instructions='True?'))
        r = self.result({'false':.5,'true':.5})
        r['scores'][1]['id'] = 'false'
        with self.assertRaises(ValueError):
            answer(d, r)

    def test_state_leakage_ignores_object_key_order(self):
        a = dict(id='train', request=dict(state={'a':1,'b':2}))
        b = dict(id='test', request=dict(state={'b':2,'a':1}))
        with self.assertRaises(ValueError):
            check_disjoint(dict(train=[a], test=[b]))

    def test_false_is_accepted_and_null_is_abstention(self):
        self.assertEqual(native_selection(dict(value=dict(type='binary',value=False))), 'false')
        self.assertIsNone(native_selection(dict(value=dict(type='choice',selected=None))))

    def test_failures_stay_in_denominator(self):
        s = summarize([], 5)
        self.assertEqual((s['failed'],s['raw_accuracy'],s['coverage']), (5,0,0))
        self.assertIsNone(s['accepted_accuracy'])

    def test_regression_near_ties_match_native_benchmark_tolerance(self):
        self.assertIsNone(unique_top1(['a','b'], [.5-1e-13, .5+1e-13]))
        self.assertEqual(unique_top1(['a','b'], [.6,.4]), 'a')

    def test_all_model_matrix_retains_missing_checkpoint_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root/'matrix'
            run = subprocess.run([sys.executable, str(Path(__file__).with_name('run_jev_lora.py')),
                'run', '--models', 'all', '--checkpoint-root', str(root/'missing'),
                '--data', str(root/'data'), '--converter', str(root/'converter.py'),
                '--output', str(output)], capture_output=True, text=True)
            self.assertNotEqual(run.returncode, 0)
            rows = json.loads((output/'summary.json').read_text())['runs']
            self.assertEqual({r['model'] for r in rows}, set(json.loads(DEFAULT_PROFILES.read_text())['models']))
            self.assertTrue(all(r['status'] == 'failed' and r['error'] for r in rows))

    def test_legacy_registry_matches_packaged_registry(self):
        self.assertEqual(json.loads(Path(__file__).with_name('jev_model_profiles.json').read_text()),
                         json.loads(DEFAULT_PROFILES.read_text()))

    def test_model_profiles_pin_revision_and_reject_unknown_model(self):
        for name in ('smollm2', 'qwen3', 'gemma3', 'tinyllama', 'gemma4'):
            self.assertEqual(len(read_profile(name)['revision']), 40)
        with self.assertRaises(ValueError):
            read_profile('not-registered')


if __name__ == '__main__':
    unittest.main()
