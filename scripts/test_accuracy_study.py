import collections
from decimal import Decimal
import itertools
import json
from pathlib import Path
import re
import tempfile
import unittest

from prepare_accuracy_study import DEFAULT_COUNTS, THRESHOLDS, prepare, validate_dataset


def reference_band(value, lower, upper):
    """Independent interval-membership oracle, not generator's branch ordering."""
    memberships = [value < lower, lower <= value < upper, upper <= value]
    if sum(memberships) != 1:
        raise AssertionError('intervals must partition the number line')
    return memberships.index(True)


def reference_binary(value, cutoff, true_if_ge):
    # Complement of strict-below is inclusive-at-or-above, including equality.
    below = value < cutoff
    truth = not below if true_if_ge else below
    return ('false', 'true')[truth]


def evaluate_symbolic(text, state):
    comparisons = re.findall(r'([a-z_]+)\s*(>=|<)\s*(-?\d+(?:\.\d+)?)', text)
    if not comparisons:
        raise AssertionError(f'missing symbolic comparison: {text}')
    values = []
    for field, operator, constant in comparisons:
        value, boundary = Decimal(str(state[field])), Decimal(constant)
        values.append(value >= boundary if operator == '>=' else value < boundary)
    return all(values)


class AccuracyStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temporary.name) / 'study'
        cls.manifest = prepare(cls.output)
        cls.rows = {
            split: {
                variant: [json.loads(line) for line in (cls.output / data['path']).read_text().splitlines()]
                for variant, data in metadata['variants'].items()
            }
            for split, metadata in cls.manifest['splits'].items()
        }

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_exact_default_counts_and_balanced_kinds_and_semantic_bands(self):
        for split, expected in DEFAULT_COUNTS.items():
            rows = self.rows[split]['natural']
            self.assertEqual(len(rows), expected)
            kinds = collections.Counter(row['request']['decisions'][0]['kind']['type'] for row in rows)
            self.assertEqual(kinds, dict.fromkeys(('choice', 'binary', 'ordinal'), expected // 3))
            targets = collections.defaultdict(collections.Counter)
            for row in rows:
                case = self.manifest['cases'][row['id']]
                label = next(iter(row['expected'].values()))
                targets[case['kind']][case['semantic_ids'].index(label)] += 1
            for counts in targets.values():
                self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
            choice_positions = collections.Counter()
            for row in rows:
                decision = row['request']['decisions'][0]
                if decision['kind']['type'] == 'choice':
                    ids = [option['id'] for option in decision['kind']['options']]
                    choice_positions[ids.index(row['expected'][decision['id']])] += 1
            self.assertLessEqual(max(choice_positions.values()) - min(choice_positions.values()), 1)

    def test_thresholds_templates_groups_and_logical_cases_are_split_disjoint(self):
        for a, b in itertools.combinations(DEFAULT_COUNTS, 2):
            left, right = self.manifest['splits'][a], self.manifest['splits'][b]
            self.assertFalse(set(left['ids']) & set(right['ids']))
            self.assertFalse(set(left['groups']) & set(right['groups']))
            thresholds_a = {Decimal(v) for pair in THRESHOLDS[a] for v in pair}
            thresholds_b = {Decimal(v) for pair in THRESHOLDS[b] for v in pair}
            self.assertFalse(thresholds_a & thresholds_b)
            for variant in ('intro', 'symbolic_intro', 'low', 'mid', 'high', 'ge', 'lt', 'symbolic'):
                templates_a = {self.manifest['templates'][t][variant]
                               for t in self.manifest['template_allocations'][a]}
                templates_b = {self.manifest['templates'][t][variant]
                               for t in self.manifest['template_allocations'][b]}
                self.assertFalse(templates_a & templates_b)
        for split in DEFAULT_COUNTS:
            logical = []
            for row in self.rows[split]['natural']:
                design = self.manifest['cases'][row['id']]
                self.assertIn(design['template_id'], self.manifest['template_allocations'][split])
                logical.append(tuple(sorted((key, str(value)) for key, value in design.items()
                                            if key not in ('semantic_ids', 'step'))))
                self.assertEqual(row['group'], design['threshold_id'] + '::' + design['template_id'])
            self.assertEqual(len(logical), len(set(logical)))
            values = {Decimal(v) for pair in THRESHOLDS[split] for v in pair}
            self.assertFalse(values & {Decimal(6), Decimal(24)})

    def test_independent_numeric_oracle_and_rendered_symbolic_criteria(self):
        for split in DEFAULT_COUNTS:
            boundary_coverage = set()
            numeric_types = set()
            domains = set()
            for natural, symbolic in zip(self.rows[split]['natural'], self.rows[split]['symbolic']):
                design = self.manifest['cases'][natural['id']]
                state = natural['request']['state']
                field = design['field']
                value = Decimal(str(state[field]))
                lower, upper = Decimal(design['lower']), Decimal(design['upper'])
                boundary_coverage.add(design['boundary'])
                domains.add(design['domain'])
                numeric_types.add(lower == lower.to_integral_value())
                self.assertNotIn('expected', state)
                self.assertNotIn('label', state)
                self.assertNotIn('storage_requirement', state)
                self.assertNotIn('hours_until_dispatch', state)
                self.assertGreater(len(state), 3)
                endpoint, relation = design['boundary'].split('-')
                anchor = lower if endpoint == 'lower' else upper
                expected_offset = {'below': -Decimal(design['step']), 'exact': Decimal(0),
                                   'above': Decimal(design['step'])}[relation]
                self.assertEqual(value - anchor, expected_offset)
                if design['kind'] == 'binary':
                    expected = reference_binary(value, Decimal(design['cutoff']), design['true_if_ge'])
                else:
                    expected = design['semantic_ids'][reference_band(value, lower, upper)]
                for row in (natural, symbolic):
                    decision = row['request']['decisions'][0]
                    self.assertEqual(row['expected'], {decision['id']: expected})
                decision = symbolic['request']['decisions'][0]
                kind = decision['kind']
                if kind['type'] == 'binary':
                    false_matches = evaluate_symbolic(kind['false_label'], state)
                    true_matches = evaluate_symbolic(kind['true_label'], state)
                    self.assertNotEqual(false_matches, true_matches)
                    actual = 'true' if true_matches else 'false'
                else:
                    options = kind.get('options', kind.get('levels'))
                    satisfied = [option['id'] for option in options
                                 if evaluate_symbolic(option['criterion'], state)]
                    self.assertEqual(len(satisfied), 1)
                    actual = satisfied[0]
                self.assertEqual(actual, expected)
            self.assertEqual(boundary_coverage, {a + '-' + b for a in ('lower', 'upper')
                                               for b in ('below', 'exact', 'above')})
            self.assertEqual(numeric_types, {True, False})
            self.assertEqual(len(domains), 6)

    def test_pairs_preserve_state_and_semantics_and_never_double_count(self):
        labels = json.loads((self.output / 'labels.json').read_text())
        self.assertEqual(len(labels), sum(DEFAULT_COUNTS.values()))
        choice_orders = set()
        semantic_sets = set()
        for split in DEFAULT_COUNTS:
            for a, b in zip(self.rows[split]['natural'], self.rows[split]['symbolic']):
                self.assertEqual(set(a), {'id', 'group', 'request', 'expected'})
                for field in ('id', 'group', 'expected'):
                    self.assertEqual(a[field], b[field])
                self.assertEqual(a['request']['state'], b['request']['state'])
                da, db = a['request']['decisions'][0], b['request']['decisions'][0]
                self.assertNotEqual(da['instruction'], db['instruction'])
                self.assertEqual(da['kind']['type'], db['kind']['type'])
                if da['kind']['type'] != 'binary':
                    ka = da['kind'].get('options', da['kind'].get('levels'))
                    kb = db['kind'].get('options', db['kind'].get('levels'))
                    self.assertEqual([o['id'] for o in ka], [o['id'] for o in kb])
                    semantic_sets.add(tuple(o['id'] for o in ka))
                    self.assertEqual(len({o['id'] for o in ka}), 3)
                    if da['kind']['type'] == 'ordinal':
                        values = [o['value'] for o in ka]
                        self.assertEqual(values, sorted(set(values)))
                    else:
                        semantic = self.manifest['cases'][a['id']]['semantic_ids']
                        choice_orders.add(tuple(semantic.index(o['id']) for o in ka))
        self.assertEqual(len(choice_orders), 6)
        self.assertEqual(len(semantic_sets), sum(DEFAULT_COUNTS.values()) * 2 // 3)

    def test_unlabeled_files_exclude_gold_and_manifest_verifies(self):
        self.assertEqual(validate_dataset(self.output), self.manifest)
        for split in DEFAULT_COUNTS:
            for variant in ('natural', 'symbolic'):
                info = self.manifest['splits'][split]['variants'][variant]
                requests = [json.loads(line) for line in (self.output / info['requests_path']).read_text().splitlines()]
                self.assertEqual(requests, [{key: row[key] for key in ('id', 'group', 'request')}
                                           for row in self.rows[split][variant]])
                self.assertTrue(all(set(row) == {'id', 'group', 'request'} for row in requests))

    def test_deterministic_bytes_no_overwrite_and_invalid_counts_leave_no_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            counts = dict.fromkeys(DEFAULT_COUNTS, 12)
            prepare(a, counts, seed=17)
            prepare(b, counts, seed=17)
            self.assertEqual({p.name: p.read_bytes() for p in a.iterdir()},
                             {p.name: p.read_bytes() for p in b.iterdir()})
            before = (a / 'manifest.json').read_bytes()
            with self.assertRaises(FileExistsError):
                prepare(a, counts, seed=99)
            self.assertEqual(before, (a / 'manifest.json').read_bytes())
            invalid = Path(directory) / 'invalid'
            with self.assertRaises(ValueError):
                prepare(invalid, dict(counts, train=7))
            self.assertFalse(invalid.exists())
            with self.assertRaises(ValueError):
                prepare(invalid, dict(counts, train=999996))
            self.assertFalse(invalid.exists())

    def test_tampering_any_frozen_request_or_labels_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'data'
            manifest = prepare(output, dict.fromkeys(DEFAULT_COUNTS, 6))
            targets = [output / 'labels.json']
            for split in DEFAULT_COUNTS:
                for variant in ('natural', 'symbolic'):
                    info = manifest['splits'][split]['variants'][variant]
                    targets.extend(output / info[key] for key in ('path', 'requests_path'))
            for path in targets:
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                    validate_dataset(output)
                path.write_bytes(original)
            validate_dataset(output)


if __name__ == '__main__':
    unittest.main()
