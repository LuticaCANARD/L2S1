import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from prepare_accuracy_study import DEFAULT_COUNTS, json_bytes, prepare
from report_accuracy_study import build_report, metrics, option_ids, select_configuration, write_outputs


def prediction(source, rotation=0, kind='single', detail='minimal', latency=4, correct=True, tied=False):
    decision = source['request']['decisions'][0]
    ids = option_ids(decision)
    gold = source['expected'][decision['id']]
    winner = gold if correct else next(label for label in ids if label != gold)
    probabilities = {label: (1 / len(ids) if tied else .8 if label == winner else .2 / (len(ids) - 1))
                     for label in ids}
    selected = None if tied else winner
    def identity_for_rotation(index):
        suffix = {'minimal': 'Minimal', 'typed': 'Typed', 'typed_examples': 'TypedExamples'}[detail]
        version = 'fixture-v1' if detail == 'minimal' and index == 0 else f'fixture-v1/detail-{suffix}-v1/rotation-{index}'
        return dict(weights_sha256='a' * 64, runtime_build_sha256='b' * 64, prompt_version=version)
    identity = identity_for_rotation(rotation)
    if decision['kind']['type'] == 'binary':
        value = dict(type='binary', p_true=probabilities['true'],
                     value=None if selected is None else selected == 'true')
    elif decision['kind']['type'] == 'ordinal':
        value = dict(type='ordinal', selected=selected,
                     expected_value=sum(probabilities[level['id']] * level['value']
                                        for level in decision['kind']['levels']))
    else:
        value = dict(type='choice', selected=selected)
    observation = dict(id=decision['id'], scores=[dict(id=label, code=chr(65 + (i + rotation) % len(ids)),
                                                      token_id=100 + (i + rotation) % len(ids),
                                                      raw_logit=math.log(probabilities[label]),
                                                      option_probability=probabilities[label])
                                               for i, label in enumerate(ids)],
                       candidate_mass=.7, top_option_probability=max(probabilities.values()),
                       value=value, abstention_reasons=['tied_candidates'] if tied else [],
                       scoring_method='fixture', calibration_id=None, input_tokens=80, truncated=False)
    row = dict(id=source['id'], group=source['group'],
               setting=dict(prompt_detail=detail, prompt_layout='legacy'), kind=kind,
               elapsed_ms=latency,
               response=dict(policy=dict(min_top_probability=.5, min_candidate_mass=.05), results=[observation]))
    if kind == 'single':
        row.update(rotation=rotation, model_identity=identity)
    else:
        row.update(rotations=list(range(len(ids))),
                   model_identities=[identity_for_rotation(i) for i in range(len(ids))])
    return row


class AccuracyReportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.data = self.directory / 'data'
        self.manifest = prepare(self.data, dict.fromkeys(DEFAULT_COUNTS, 12))
        self.sources = {}
        for split in DEFAULT_COUNTS:
            path = self.manifest['splits'][split]['variants']['natural']['path']
            self.sources[split] = [json.loads(line) for line in (self.data / path).read_text().splitlines()]

    def tearDown(self):
        self.temporary.cleanup()

    def report(self, rows, split='dev'):
        path = self.directory / 'predictions.jsonl'
        rows = copy.deepcopy(rows)
        for row in rows:
            row.setdefault('input_sha256', self.manifest['splits'][split]['variants']['natural']['requests_sha256'])
        path.write_bytes(b''.join(json_bytes(row) for row in rows))
        return build_report(self.data, [path], split, 'natural')

    def test_recounts_metrics_from_gold_and_semantic_probabilities(self):
        rows = [prediction(source, latency=i + 1) for i, source in enumerate(self.sources['dev'])]
        rows[0] = prediction(self.sources['dev'][0], correct=False, latency=1)
        rows[1] = prediction(self.sources['dev'][1], tied=True, latency=2)
        report = self.report(rows)
        config = report['configurations'][0]
        measured = config['metrics']
        self.assertEqual(measured['count'], 12)
        self.assertEqual(measured['raw_correct'], 10)
        self.assertEqual(measured['ties'], 1)
        self.assertEqual(measured['accepted'], 11)
        self.assertEqual(measured['accepted_wrong'], 1)
        self.assertEqual(measured['accepted_accuracy'], 10 / 11)
        self.assertEqual(measured['coverage'], 11 / 12)
        self.assertEqual(measured['accepted_correct_over_all'], 10 / 12)
        self.assertEqual(measured['latency_ms'], dict(total=78, p50=6.5, p95=12))
        expected_nll, expected_brier = [], []
        for source, row in zip(self.sources['dev'], rows):
            expected = next(iter(source['expected'].values()))
            scores = row['response']['results'][0]['scores']
            probabilities = {score['id']: score['option_probability'] for score in scores}
            expected_nll.append(-math.log(probabilities[expected]))
            expected_brier.append(sum((p - (label == expected)) ** 2 for label, p in probabilities.items()))
        self.assertAlmostEqual(measured['nll_mean'], sum(expected_nll) / 12)
        self.assertAlmostEqual(measured['brier_mean'], sum(expected_brier) / 12)
        self.assertEqual(set(config['by_decision_kind']), {'binary', 'choice', 'ordinal'})
        self.assertEqual(len(config['errors']), 2)
        self.assertEqual(report['logical_cases'], 12)
        self.assertEqual(report['manifest_sha256'], hashlib.sha256((self.data / 'manifest.json').read_bytes()).hexdigest())

    def test_ties_incorrect_and_zero_acceptance_has_null_accuracy(self):
        report = self.report([prediction(source, tied=True) for source in self.sources['dev']])
        measured = report['configurations'][0]['metrics']
        self.assertEqual(measured['ties'], 12)
        self.assertEqual(measured['raw_top1'], 0)
        self.assertEqual(measured['coverage'], 0)
        self.assertIsNone(measured['accepted_accuracy'])
        self.assertEqual(measured['accepted_correct_over_all'], 0)
        self.assertIsNone(metrics([])['raw_top1'])

    def test_runtime_failures_remain_in_accuracy_coverage_denominators(self):
        rows = [prediction(source) for source in self.sources['dev']]
        rows[0].pop('response')
        rows[0]['error'] = 'fixture inference failure'
        report = self.report(rows)
        measured = report['configurations'][0]['metrics']
        self.assertEqual(measured['runtime_errors'], 1)
        self.assertEqual(measured['abstained'], 0)
        self.assertEqual(measured['not_accepted'], 1)
        self.assertEqual(measured['probability_evaluated'], 11)
        self.assertEqual(measured['raw_top1'], 11 / 12)
        self.assertEqual(measured['coverage'], 11 / 12)
        self.assertEqual(measured['accepted_accuracy'], 1)
        self.assertAlmostEqual(measured['nll_mean'], -math.log(.8))

    def test_complete_rotation_probe_handles_binary_and_ternary_subsets(self):
        rows = []
        for index, source in enumerate(self.sources['dev']):
            count = len(option_ids(source['request']['decisions'][0]))
            for rotation in range(count):
                rows.append(prediction(source, rotation=rotation, correct=not (index == 0 and rotation == 1)))
            rows.append(prediction(source, kind='ensemble', latency=count * 4))
        report = self.report(rows)
        self.assertEqual(len(report['configurations']), 4)
        r2 = next(config for config in report['configurations'] if config['rotation'] == 2)
        self.assertEqual(r2['metrics']['count'], 8)
        self.assertFalse(r2['covers_full_split'])
        self.assertNotIn('binary', r2['by_decision_kind'])
        bias = report['rotation_consistency'][0]
        self.assertEqual(bias['compared_cases'], 12)
        self.assertEqual(bias['changed_top1_cases'], 1)
        self.assertEqual(bias['changed_top1_rate'], 1 / 12)
        self.assertGreater(bias['probability_deviation']['maximum'], .5)
        self.assertLess(bias['probability_deviation']['mean'], .1)

    def test_incomplete_or_duplicate_rotation_probes_are_rejected(self):
        rows = [prediction(source, rotation=rotation) for source in self.sources['dev'] for rotation in (0, 1)]
        with self.assertRaisesRegex(ValueError, 'missing a required case/rotation'):
            self.report(rows)
        rows = [prediction(source) for source in self.sources['dev']]
        with self.assertRaisesRegex(ValueError, 'duplicate logical ID'):
            self.report(rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, 'incomplete logical-ID coverage'):
            self.report(rows[:-1])
        altered = copy.deepcopy(rows)
        altered[0]['id'] = self.sources['test'][0]['id']
        with self.assertRaisesRegex(ValueError, 'outside the declared split'):
            self.report(altered)
        altered = copy.deepcopy(rows)
        altered[0]['group'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'group mismatch'):
            self.report(altered)

    def test_semantic_mapping_finite_probabilities_identity_and_policy_are_checked(self):
        base = [prediction(source) for source in self.sources['dev']]
        mutations = [
            lambda row: row.update(elapsed_ms=float('nan')),
            lambda row: row['response']['results'][0]['scores'][0].update(id='unknown'),
            lambda row: row['response']['results'][0]['scores'][0].update(option_probability=-.1),
            lambda row: row['response']['results'][0]['scores'][0].update(option_probability=.12345),
            lambda row: row['response']['results'][0].update(candidate_mass=1.5),
            lambda row: row['response']['results'][0].update(top_option_probability=.3),
            lambda row: row['response']['results'][0].update(id='wrong-decision'),
            lambda row: row['response']['results'][0].update(abstention_reasons=['low_candidate_mass']),
            lambda row: row['response']['policy'].update(min_candidate_mass=.9),
            lambda row: row['model_identity'].update(weights_sha256='different-model'),
            lambda row: row['model_identity'].update(prompt_version='fixture-v1/detail-Minimal-v1/rotation-1'),
            lambda row: row['model_identity'].update(prompt_version='fixture-v1/detail-Typed-v1/rotation-0'),
            lambda row: row['model_identity'].update(prompt_version='fixture-v1/detail-Minimal-v1/rotation-01'),
            lambda row: row.update(input_sha256='wrong-split-or-variant'),
        ]
        for mutate in mutations:
            rows = copy.deepcopy(base)
            mutate(rows[0])
            with self.assertRaises((ValueError, KeyError), msg=str(mutate)):
                self.report(rows)
        binary = next(i for i, source in enumerate(self.sources['dev'])
                      if source['request']['decisions'][0]['kind']['type'] == 'binary')
        ordinal = next(i for i, source in enumerate(self.sources['dev'])
                       if source['request']['decisions'][0]['kind']['type'] == 'ordinal')
        for index, update in [(binary, {'p_true': .33}), (ordinal, {'expected_value': 999})]:
            rows = copy.deepcopy(base)
            rows[index]['response']['results'][0]['value'].update(update)
            with self.assertRaises(ValueError):
                self.report(rows)

    def test_ensemble_requires_all_distinct_rotations_and_same_model(self):
        base = [prediction(source, kind='ensemble') for source in self.sources['dev']]
        report = self.report(base)
        self.assertEqual(report['configurations'][0]['kind'], 'ensemble')
        for mutation in ('duplicate', 'missing', 'model'):
            rows = copy.deepcopy(base)
            if mutation == 'duplicate':
                rows[0]['rotations'][1] = 0
            elif mutation == 'missing':
                rows[0]['model_identities'].pop()
            else:
                rows[0]['model_identities'][1]['weights_sha256'] = 'another-model'
            with self.assertRaises(ValueError):
                self.report(rows)

    def test_selection_dev_only_full_coverage_ranked_and_frozen_without_overwrite(self):
        rows = []
        for source in self.sources['dev']:
            for rotation in range(len(option_ids(source['request']['decisions'][0]))):
                rows.append(prediction(source, rotation=rotation, correct=rotation == 2))
            rows.append(prediction(source, detail='typed', latency=3))
            rows.append(prediction(source, detail='typed_examples', latency=5))
        report = self.report(rows)
        chosen = select_configuration(report, 'report-digest')
        self.assertEqual(chosen['selected']['setting']['prompt_detail'], 'typed')
        self.assertEqual(chosen['selected']['rotation'], 0)
        self.assertEqual(chosen['variant'], 'natural')
        self.assertEqual(chosen['report_sha256'], 'report-digest')
        output, selection = self.directory / 'report.json', self.directory / 'selection.json'
        write_outputs(report, output, selection)
        saved = json.loads(selection.read_text())
        self.assertEqual(saved['report_sha256'], hashlib.sha256(output.read_bytes()).hexdigest())
        before = output.read_bytes()
        with self.assertRaises(FileExistsError):
            write_outputs(report, output, selection)
        self.assertEqual(output.read_bytes(), before)
        for split in ('test', 'calibration'):
            held_out = dict(report, split=split)
            with self.assertRaisesRegex(ValueError, 'dev only'):
                select_configuration(held_out, 'hash')
            forbidden = self.directory / (split + '.json')
            with self.assertRaisesRegex(ValueError, 'dev only'):
                write_outputs(held_out, forbidden, self.directory / (split + '-selection.json'))
            self.assertFalse(forbidden.exists())

    def test_selection_does_not_compare_a_perfect_rotation_two_subset_with_full_split(self):
        rows = [prediction(source, rotation=rotation, correct=rotation == 2)
                for source in self.sources['dev']
                for rotation in range(len(option_ids(source['request']['decisions'][0])))]
        report = self.report(rows)
        selected = select_configuration(report, 'hash')['selected']
        self.assertIn(selected['rotation'], (0, 1))
        self.assertNotEqual(selected['rotation'], 2)
        # A report containing only the eligible rotation-two subset cannot select.
        only_subset = [row for row in rows if row['rotation'] == 2]
        partial = self.report(only_subset)
        with self.assertRaisesRegex(ValueError, 'no full-split'):
            select_configuration(partial, 'hash')

    def test_prediction_provenance_and_frozen_data_tampering_are_checked(self):
        rows = [prediction(source) for source in self.sources['dev']]
        report = self.report(rows)
        path = self.directory / 'predictions.jsonl'
        self.assertEqual(report['predictions'], [dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())])
        labels = self.data / 'labels.json'
        labels.write_bytes(labels.read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'labels hash mismatch'):
            build_report(self.data, [path], 'dev', 'natural')


if __name__ == '__main__':
    unittest.main()
