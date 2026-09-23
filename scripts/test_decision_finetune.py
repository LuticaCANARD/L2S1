import itertools
import unittest
from prepare_decision_finetune import split_rows
from kaggle_airline import OPTIONS, identity
from report_decision_finetune import prediction


class FineTuneSplitTests(unittest.TestCase):
    def test_no_prior_examples_duplicates_or_cross_split_leakage(self):
        rows=[dict(text=f'{label} example {i}',airline_sentiment=label) for label in OPTIONS for i in range(700)]
        rows += [dict(text='NEGATIVE   EXAMPLE 0',airline_sentiment='negative'),
                 dict(text='conflict',airline_sentiment='positive'),dict(text='CONFLICT',airline_sentiment='negative')]
        excluded={f'{label} example 0' for label in OPTIONS}
        splits=split_rows(rows,excluded)
        texts={s:{identity(r['text']) for _,r in values} for s,values in splits.items()}
        self.assertEqual({s:len(t) for s,t in texts.items()},{'train':900,'calibration':400,'test':400})
        for a,b in itertools.combinations(texts,2):self.assertFalse(texts[a]&texts[b])
        for values in texts.values():
            self.assertFalse(values & excluded)
            self.assertNotIn('conflict',values)
        self.assertEqual(splits,split_rows(rows,excluded))

    def test_order_audit_does_not_treat_a_tie_as_the_first_label(self):
        row={'response':{'results':[{'scores':[{'id':'negative','raw_logit':2.0},
                    {'id':'positive','raw_logit':2.0},{'id':'neutral','raw_logit':-1.0}]}]}}
        self.assertIsNone(prediction(row))
        row['response']['results'][0]['scores'].reverse()
        self.assertIsNone(prediction(row))


if __name__=='__main__':unittest.main()
