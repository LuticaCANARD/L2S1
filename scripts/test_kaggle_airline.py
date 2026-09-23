import unittest
from kaggle_airline import OPTIONS, identity, select, threshold_counts


class AirlineBenchmarkTests(unittest.TestCase):
    def test_split_excludes_duplicates_and_conflicting_labels(self):
        rows=[{'text':f'{label} item {i}', 'airline_sentiment':label} for label in OPTIONS for i in range(280)]
        rows.extend([
            {'text':'NEGATIVE   ITEM 0','airline_sentiment':'negative'},
            {'text':'conflict','airline_sentiment':'negative'},
            {'text':'CONFLICT','airline_sentiment':'positive'},
            {'text':'   ','airline_sentiment':'neutral'},
        ])
        splits,excluded,_=select(rows)
        self.assertEqual(excluded,{'duplicate_rows':1,'conflicting_label_rows':2,'empty_rows':1})
        texts={name:{identity(r['text']) for _,r in data} for name,data in splits.items()}
        self.assertEqual(len(texts['fit']),400)
        self.assertEqual(len(texts['validation']),400)
        self.assertFalse(texts['fit'] & texts['validation'])
        self.assertNotIn('conflict',texts['fit'] | texts['validation'])
        self.assertEqual(select(rows),select(rows))

    def test_lower_threshold_keeps_mass_and_tie_abstentions(self):
        rows=[]
        labels={}
        for name,logits,mass in [('accepted',[3,0,-1],1),('tie',[0,0,-1],1),('low_mass',[3,0,-1],.01)]:
            labels[name]='negative'
            rows.append({'id':name,'response':{'results':[{'candidate_mass':mass,'scores':[
                {'id':label,'raw_logit':logit} for label,logit in zip(OPTIONS,logits)]}]}})
        result=threshold_counts(rows,labels,1,0)
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['abstained'],2)
        self.assertEqual(result['wrong'],0)


if __name__=='__main__':
    unittest.main()
