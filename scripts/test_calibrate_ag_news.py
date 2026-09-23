import math
import unittest
from calibrate_ag_news import fit_temperature, metrics, probabilities


class CalibrationTests(unittest.TestCase):
    def test_stable_and_ranking_preserved(self):
        for t in [.05, 1, 5, 100]:
            p, logs = probabilities([10000, 9998, -10000], t)
            self.assertAlmostEqual(sum(p), 1)
            self.assertTrue(p[0] > p[1] >= p[2])
            self.assertTrue(all(math.isfinite(x) for x in logs))

    def test_fit_known_binary_confidence_without_changing_top1(self):
        rows = []
        labels = {}
        for i in range(100):
            labels[str(i)] = 'a' if i < 80 else 'b'
            rows.append({'id':str(i), 'response':{'results':[{'truncated':False,'candidate_mass':1,
                'scores':[{'id':'a','raw_logit':8.0},{'id':'b','raw_logit':0.0}]}]}})
        t = fit_temperature(rows, labels)
        self.assertAlmostEqual(t, 8 / math.log(4), places=5)
        before, after = [metrics(rows, labels, v) for v in [1,t]]
        self.assertEqual(before['raw_top1'], after['raw_top1'])
        self.assertLess(after['nll'], before['nll'])
        self.assertLess(after['ece_10_equal_width'], 1e-6)


if __name__ == '__main__':
    unittest.main()
