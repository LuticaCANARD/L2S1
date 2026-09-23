import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from laya_benchmark import (auc, compare, decision, distribution, metrics, options, phish_cases,
                            probe_cases, save, score, score_decision, sha, typed_cases, write_rows)


def typed_row():
    return dict(state='{"text":"Input only"}', workflow='test', questions=json.dumps({
        'binary': dict(type='noul', instructions='Is this true?'),
        'choice': dict(type='choice', instructions='Choose.', criteria={'z': 'Z', 'a': 'A'}),
        'score': dict(type='score', instructions='Rate.', criteria=['low', 'medium', 'high'])}),
        gold=json.dumps({'binary': {'label': 'true', 'noul': .8},
                         'choice': {'label': 'z', 'probabilities': {'z': .7, 'a': .3}},
                         'score': {'label': '1', 'score': 1.2}}))


def result(d, p):
    labels = options(d)
    value = dict(type=d['kind']['type'], selected=None)
    if value['type'] == 'binary':
        value = dict(type='binary', value=None, p_true=p[1])
    if value['type'] == 'ordinal':
        value['expected_value'] = sum(i * v for i, v in enumerate(p))
    return dict(id=d['id'], scores=[dict(id=k, option_probability=v) for k, v in zip(labels, p)],
                truncated=False, value=value, entropy_confidence=.1,
                candidate_mass=.9, input_tokens=300, reused_prefix_tokens=0)


def prediction(case, batch=0, batch_size=1):
    cache = {kind: {f: 0 for f in ['hits', 'misses', 'insertions', 'evictions', 'skipped']}
             for kind in ['prompts', 'candidates']}
    after = copy.deepcopy(cache); after['prompts']['hits'] = 2
    return dict(id=case['id'], response=dict(results=[result(d, [1 / len(options(d))] * len(options(d)))
             for d in case['request']['decisions']]), elapsed_ms=100, batch_index=batch, batch_size=batch_size,
             batch_elapsed_ms=100, batch_profile=dict(prepare_ms=1, native_ms=98, score_ms=1),
             preparation_cache_before=cache, preparation_cache_after=after)


class AdapterTests(unittest.TestCase):
    def test_gold_cannot_change_request(self):
        row = typed_row(); a = typed_cases([row])[0]
        g = json.loads(row['gold']); g['choice']['label'] = 'a'; g['choice']['rationale'] = 'SECRET'
        row['gold'] = json.dumps(g); row['workflow'] = 'SECRET'
        b = typed_cases([row])[0]
        self.assertEqual(a['request'], b['request'])
        self.assertNotIn('SECRET', json.dumps(b['request']))

    def test_all_question_types_and_order(self):
        ds = typed_cases([typed_row()])[0]['request']['decisions']
        self.assertEqual([d['id'] for d in ds], ['binary', 'choice', 'score'])
        self.assertEqual(options(ds[0]), ['false', 'true'])
        self.assertEqual(options(ds[1]), ['z', 'a'])
        self.assertEqual([x['value'] for x in ds[2]['kind']['levels']], [0, 1, 2])

    def test_invalid_criteria_and_missing_gold(self):
        for q in [dict(type='choice', instructions='q', criteria={'a': 'x'}),
                  dict(type='score', instructions='q', criteria={'0': 'x', '1': 'y'}),
                  dict(type='noul', instructions='q', criteria={'yes': 'y', 'no': 'n'})]:
            with self.assertRaises(ValueError): decision('q', q)
        row = typed_row(); row['gold'] = '{}'
        with self.assertRaises(ValueError): typed_cases([row])

    def test_phishing_plain_text_and_json_keep_state(self):
        cases = phish_cases([dict(email_content='hello', phish_label=0),
                             dict(email_content='{"body":"payload"}', phish_label=1)])
        self.assertEqual(cases[0]['request']['state'], 'hello')
        self.assertEqual(cases[1]['request']['state'], {'body': 'payload'})
        self.assertEqual(cases[1]['gold']['is_phishing']['label'], 'true')
        with self.assertRaises(ValueError): phish_cases([dict(email_content='x', phish_label=2)])

    def test_auroc_ties_and_absent_class(self):
        self.assertEqual(auc([0, 1], [.5, .5]), .5)
        self.assertEqual(auc([0, 1, 0, 1], [.1, .8, .2, .9]), 1)
        self.assertEqual(auc([0, 1], [.9, .1]), 0)
        self.assertIsNone(auc([1, 1], [.1, .2]))

    def test_binary_polarity_tie_and_abstention_do_not_hide_accuracy(self):
        d = decision('q', dict(type='noul', instructions='q'))
        r = score_decision(d, result(d, [.5, .5]), dict(label='true', noul=1))
        self.assertTrue(r['correct']); self.assertFalse(r['accepted'])
        self.assertEqual(r['brier_hard'], .5)
        self.assertEqual(metrics([r], 2)['accuracy'], .5)

    def test_soft_targets_and_ordinal_expected_value(self):
        ds = typed_cases([typed_row()])[0]['request']['decisions']
        r = score_decision(ds[1], result(ds[1], [.7, .3]), dict(label='z', probabilities={'z': .7, 'a': .300001}))
        self.assertLess(r['brier_soft'], 1e-10)
        r = score_decision(ds[2], result(ds[2], [.2, .6, .2]), dict(label='1', score=1.2))
        self.assertTrue(r['correct']); self.assertAlmostEqual(r['score_mae'], .2)
        self.assertNotIn('brier_soft', r)

    def test_invalid_model_evidence_rejected(self):
        d = decision('q', dict(type='noul', instructions='q'))
        for p in [[math.nan, .5], [1, 1], [-.1, 1.1]]:
            with self.assertRaises(ValueError): distribution(d, result(d, p))
        r = result(d, [.2, .8]); r['scores'].reverse()
        with self.assertRaises(ValueError): distribution(d, r)
        r = result(d, [.2, .8]); r['truncated'] = True
        with self.assertRaises(ValueError): distribution(d, r)

    def make_prepared(self, root, count=2):
        cases = typed_cases([typed_row()] * count)
        for c in cases: c.update(base_id=c['id'], repeat=0)
        write_rows(root / 'cases-with-gold.jsonl', cases)
        write_rows(root / 'requests.jsonl', [dict(id=c['id'], request=c['request']) for c in cases])
        save(root / 'manifest.json', dict(suite='typed', logical_cases=count, decisions=count * 3, repeats=1,
             limited=True, files={p.name: sha(p) for p in root.iterdir()}))
        return cases

    def test_missing_predictions_are_failures_not_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases = self.make_prepared(root)
            write_rows(root / 'predictions.jsonl', [prediction(cases[0])])
            summary, details = score(root, root / 'predictions.jsonl')
            m = summary['per_repeat']['0']
            self.assertEqual(m['attempted'], 6); self.assertEqual(m['valid'], 3)
            self.assertEqual(len(summary['errors']), 1); self.assertEqual(len(details), 3)
            self.assertEqual(m['accuracy'], m['correct'] / 6)

    def test_batch_latency_and_cache_counters_counted_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases = self.make_prepared(root)
            write_rows(root / 'predictions.jsonl', [prediction(c, batch_size=2) for c in cases])
            summary, _ = score(root, root / 'predictions.jsonl')
            self.assertEqual(summary['unique_batch_elapsed_ms'], 100)
            self.assertEqual(summary['profile']['native_ms'], 98)
            self.assertEqual(summary['cache']['prompts']['hits'], 2)
            self.assertEqual(summary['per_repeat']['0']['case_latency_ms']['p50'], 100)

    def test_duplicate_ids_and_hash_changes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); cases = self.make_prepared(root)
            p = prediction(cases[0]); write_rows(root / 'predictions.jsonl', [p, p])
            with self.assertRaises(ValueError): score(root, root / 'predictions.jsonl')
            (root / 'requests.jsonl').write_text('tampered')
            with self.assertRaises(ValueError): score(root, root / 'predictions.jsonl')

    def test_probe_reader_does_not_execute_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / 'probe.py'
            path.write_text('TICKET = __import__("os").system("false")')
            with self.assertRaises(ValueError): probe_cases(path)

    def comparison_fixture(self, root):
        cases = self.make_prepared(root)
        preds = [prediction(c, batch=i) for i, c in enumerate(cases)]
        for p in preds:
            p['response'].update(backend={'prompt_layout': 'state-first', 'compute': {'batch': 256}}, policy={})
        for name in ['base', 'candidate']:
            out = root / name; out.mkdir()
            save(out / 'run.json', dict(model_sha256='model', evaluator_sha256='evaluator',
                                       prepared_manifest_sha256=sha(root / 'manifest.json')))
            write_rows(out / 'predictions.jsonl', preds)
        return preds

    def test_compare_requires_identical_prompt_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); preds = self.comparison_fixture(root)
            report = compare(root, root / 'base', root / 'candidate')
            self.assertTrue(report['exact_probabilities']); self.assertEqual(report['total_inference_speedup'], 1)
            preds[0]['response']['backend']['prompt_layout'] = 'legacy'
            (root / 'candidate/predictions.jsonl').write_text('\n'.join(map(json.dumps, preds)) + '\n')
            with self.assertRaisesRegex(ValueError, 'execution-only'):
                compare(root, root / 'base', root / 'candidate')

    def test_compare_detects_selection_changes_even_with_small_probability_delta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); preds = self.comparison_fixture(root)
            scores = preds[0]['response']['results'][0]['scores']
            scores[0]['option_probability'] = .501; scores[1]['option_probability'] = .499
            preds[0]['response']['results'][0]['value']['p_true'] = .499
            (root / 'candidate/predictions.jsonl').write_text('\n'.join(map(json.dumps, preds)) + '\n')
            report = compare(root, root / 'base', root / 'candidate')
            self.assertLess(report['max_probability_delta'], .02)
            self.assertEqual(report['raw_top1_changes'], 1); self.assertFalse(report['within_existing_tolerance'])


if __name__ == '__main__':
    unittest.main()
