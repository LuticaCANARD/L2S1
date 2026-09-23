import copy
from types import SimpleNamespace
import unittest

from jevbench_public import exact_probabilities, score_predictions, to_request


class JevBenchMappingTests(unittest.TestCase):
    def choice(self):
        return {'id': 'case', 'state': {'text': 'payload'}, 'labels': ['zebra', 'apple'],
                'question': {'type': 'choice', 'instructions': 'Choose.',
                             'criteria': {'apple': 'A', 'zebra': 'Z'}},
                'expected': 'zebra', 'provenance': {'rationale': 'SECRET', 'gold_probs': {'zebra': 1}}}

    def test_gold_and_metadata_cannot_change_inference_request(self):
        task = self.choice()
        changed = copy.deepcopy(task)
        changed['expected'] = 'apple'
        changed['provenance'] = {'rationale': 'OTHER SECRET'}
        self.assertEqual(to_request(task), to_request(changed))
        self.assertNotIn('SECRET', str(to_request(task)))

    def test_canonical_order_overrides_dictionary_order(self):
        options = to_request(self.choice())['request']['decisions'][0]['kind']['options']
        self.assertEqual(options, [{'id': 'zebra', 'criterion': 'Z'}, {'id': 'apple', 'criterion': 'A'}])

    def test_binary_polarity_is_not_reversed(self):
        task = SimpleNamespace(question={'type': 'noul'}, labels=['no', 'yes'])
        result = {'scores': [{'id': 'false', 'option_probability': 0.1},
                             {'id': 'true', 'option_probability': 0.9}]}
        self.assertEqual(exact_probabilities(task, result), {'no': 0.1, 'yes': 0.9})
        result['scores'].reverse()
        with self.assertRaises(ValueError):
            exact_probabilities(task, result)

    def test_ordinal_uses_numeric_label_order(self):
        task = self.choice()
        task['labels'] = ['0', '1', '2']
        task['question'] = {'type': 'score', 'instructions': 'Rate.', 'criteria': ['low', 'mid', 'high']}
        levels = to_request(task)['request']['decisions'][0]['kind']['levels']
        self.assertEqual([(x['id'], x['value']) for x in levels], [('0', 0), ('1', 1), ('2', 2)])

    def test_missing_or_repeated_labels_fail_closed(self):
        task = self.choice()
        task['labels'] = ['zebra', 'missing']
        with self.assertRaises(ValueError):
            to_request(task)
        task['labels'] = ['zebra', 'zebra']
        with self.assertRaises(ValueError):
            to_request(task)
        result = {'scores': [{'id': 'zebra', 'option_probability': 0.5}] * 2}
        with self.assertRaises(ValueError):
            exact_probabilities(SimpleNamespace(question={'type': 'choice'}, labels=['zebra', 'apple']), result)

    def test_model_identity_and_failed_predictions_are_preserved(self):
        task = SimpleNamespace(id='case', family='test', split='public', group=None,
                               expected='yes', question={'type': 'noul'}, labels=['no', 'yes'])
        prediction = {'id': 'case', 'elapsed_ms': 1.0, 'error': 'context allocation failed'}
        scorer = lambda probabilities, task: {'valid': False, 'correct': False}
        records, selective = score_predictions([task], [prediction], scorer, 'alternative.gguf/l2s1')
        self.assertEqual(records[0]['model'], 'alternative.gguf/l2s1')
        self.assertEqual(records[0]['status'], 'failed')
        self.assertEqual(records[0]['error'], prediction['error'])
        self.assertTrue(selective[0]['error'])
        self.assertFalse(selective[0]['abstained'])
        with self.assertRaisesRegex(ValueError, 'Missing predictions'):
            score_predictions([task], [], scorer, 'alternative.gguf/l2s1')
        with self.assertRaisesRegex(ValueError, 'Duplicate or unknown'):
            score_predictions([task], [prediction, prediction], scorer, 'alternative.gguf/l2s1')


if __name__ == '__main__':
    unittest.main()
