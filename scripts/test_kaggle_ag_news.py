import unittest
from kaggle_ag_news import DESCRIPTIONS, score, select_rows, wilson


def prediction(case_id, selected=None, error=None):
    if error:
        return {'id': case_id, 'error': error}
    return {'id': case_id, 'elapsed_ms': 10, 'response': {'results': [{
        'id': 'news_topic', 'truncated': False,
        'value': {'type': 'choice', 'selected': selected},
        'scores': [{'id': k, 'option_probability': .7 if k == 'world' else .1} for k in DESCRIPTIONS],
        'abstention_reasons': [] if selected else ['low_top_probability'],
    }]}}


class EvaluationTests(unittest.TestCase):
    def test_denominators_include_abstention_errors_and_missing(self):
        labels = {'a': 'world', 'b': 'sports', 'c': 'business', 'd': 'world', 'e': 'sports'}
        result = score(labels, [prediction('a', 'world'), prediction('b', 'world'), prediction('c'), prediction('d', error='context limit')])
        self.assertEqual(result['counts'], {'total': 5, 'accepted': 2, 'correct': 1, 'wrong': 1, 'abstained': 1, 'errors': 1, 'top1_correct': 1, 'missing': 1})
        self.assertEqual(result['correct_all'], .2)
        self.assertEqual(result['accepted_accuracy'], .5)
        self.assertEqual(result['coverage'], .4)
        self.assertEqual(result['confusion']['sports']['missing'], 1)

    def test_all_abstained_has_no_accepted_accuracy(self):
        result = score({'a': 'world'}, [prediction('a')])
        self.assertIsNone(result['accepted_accuracy'])
        self.assertIsNone(result['accepted_accuracy_wilson95'])
        self.assertEqual(result['raw_top1'], 1)
        self.assertEqual(result['correct_all'], 0)

    def test_duplicate_and_unknown_predictions_are_rejected(self):
        with self.assertRaises(ValueError):
            score({'a': 'world'}, [prediction('a'), prediction('a')])
        with self.assertRaises(ValueError):
            score({'a': 'world'}, [prediction('unknown')])

    def test_split_overlap_and_duplicates_are_removed_before_sampling(self):
        row = lambda label, text: {'Class Index': label, 'Title': text, 'Description': 'Body'}
        train = [row('1', 'already seen')]
        test = [row('1', ' ALREADY   SEEN '), row('1', 'one'), row('1', 'ONE'), row('2', 'two'), row('3', 'three'), row('4', 'four')]
        first, excluded = select_rows(train, test, 1, 42)
        self.assertEqual(excluded, {'train_overlap': 1, 'test_duplicate': 1})
        self.assertEqual(len(first), 4)
        self.assertEqual(first, select_rows(train, test, 1, 42)[0])
        self.assertEqual({r['Class Index'] for _, r in first}, {'1', '2', '3', '4'})

    def test_wilson_small_samples_and_extremes(self):
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(50, 100)[0], .40383153)
        self.assertAlmostEqual(wilson(50, 100)[1], .59616847)
        self.assertLess(wilson(1, 1)[0], .21)
        self.assertGreater(wilson(0, 1)[1], .79)


if __name__ == '__main__':
    unittest.main()
