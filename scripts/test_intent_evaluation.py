import unittest

from evaluate_intents import groups, request, route, top


def prediction(a, b, probability=0.9):
    return {'response': {'results': [{'scores': [
        {'id': a, 'option_probability': probability},
        {'id': b, 'option_probability': 1-probability}],
        'value': {'selected': None}}]}}


class IntentEvaluationTests(unittest.TestCase):
    def test_all_official_candidates_survive_partition_once(self):
        for size in (60, 77):
            labels = [f'intent-{i}' for i in range(size)]
            partition = groups(labels)
            self.assertEqual(sorted(x for g in partition for x in g), sorted(labels))
            self.assertTrue(all(2 <= len(g) <= 26 for g in partition))

    def test_request_allowlist_does_not_expose_gold_or_annotations(self):
        sample = dict(id='test', dataset='massive-ko', text='불 꺼줘', expected='SECRET',
                      scenario='SECRET', annot_utt='SECRET', judgment='SECRET')
        result = request(sample, ['iot_hue_lightoff', 'alarm_set'], ':g0')
        self.assertNotIn('SECRET', str(result))
        self.assertEqual(result['request']['state'], {'utterance': '불 꺼줘'})

    def test_routing_uses_argmax_even_when_stage_policy_abstains(self):
        item = dict(id='test', dataset='banking77-en', text='message')
        first = {'test:g0': prediction('a', 'b'), 'test:g1': prediction('c', 'd'),
                 'test:g2': prediction('e', 'f')}
        finalists = route([item], first)[0]['request']['decisions'][0]['kind']['options']
        self.assertEqual([r['id'] for r in finalists], ['a', 'c', 'e'])
        # Scoring gold cannot rescue a label eliminated by stage one.
        item['expected'] = 'b'
        self.assertEqual(route([item], first)[0]['request']['decisions'][0]['kind']['options'], finalists)

    def test_ties_are_independent_of_score_order(self):
        self.assertEqual(top(prediction('z', 'a', 0.5)), 'a')


if __name__ == '__main__':
    unittest.main()
