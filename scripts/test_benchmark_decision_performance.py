import copy
import unittest
import os
import json
from pathlib import Path
import subprocess
import tempfile
from benchmark_decision_performance import compare, metrics

class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.gold = {'x': {'d': 'yes'}}
        self.rows = [{'id':'x','batch_index':0,'batch_elapsed_ms':12.,'batch_profile':{'prepare_ms':1.,'native_ms':10.,'score_ms':1.},
            'response':{'results':[{'id':'d','scores':[{'id':'yes','option_probability':.9},{'id':'no','option_probability':.1}],
            'value':{'selected':'yes'},'abstention_reasons':[], 'candidate_mass':.5,'input_tokens':200,'reused_prefix_tokens':128}]}}]
    def test_metrics_and_pairing(self):
        m = metrics(self.rows,self.gold)
        self.assertEqual((m['raw_correct'],m['accepted_correct'],m['reused_prefix_tokens']),(1,1,128))
        self.assertEqual(compare(self.rows,self.rows,self.gold)['max_probability_delta'],0)
        for rows in ([],self.rows*2,[dict(self.rows[0],id='unknown')],[{'id':'x','error':'OOM'}]):
            with self.assertRaises(ValueError): metrics(rows,self.gold)
    def test_invalid_evidence_and_abstention(self):
        for field,value in [('truncated',True),('candidate_mass',float('nan'))]:
            rows=copy.deepcopy(self.rows);rows[0]['response']['results'][0][field]=value
            with self.assertRaises(ValueError): metrics(rows,self.gold)
        rows=copy.deepcopy(self.rows); item=rows[0]['response']['results'][0]
        item['value']['selected']=None; item['abstention_reasons']=['low_top_probability']
        m=metrics(rows,self.gold)
        self.assertEqual((m['raw_correct'],m['accepted'],m['accepted_accuracy']),(1,0,None))

class HarnessContracts(unittest.TestCase):
    def test_selection_requires_complete_equivalent_faster_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            fake=root/'evaluate'
            fake.write_text("""#!/usr/bin/env python3
import json,sys
from pathlib import Path
args=sys.argv
speed=float(args[args.index('--speed')+1])
rows=[]
for line in Path(args[args.index('--input')+1]).read_text().splitlines():
    r=json.loads(line)
    rows.append({'id':r['id'],'batch_index':len(rows),'batch_elapsed_ms':speed,'batch_profile':{'prepare_ms':1.,'native_ms':speed-2.,'score_ms':1.},'response':{'results':[{'id':'d','scores':[{'id':'yes','option_probability':.9},{'id':'no','option_probability':.1}],'value':{'selected':'yes'},'abstention_reasons':[],'candidate_mass':.5,'input_tokens':200,'reused_prefix_tokens':0}]}})
if '--incomplete' in args: rows=[]
Path(args[args.index('--output')+1]).write_text(''.join(json.dumps(r)+'\\n' for r in rows))
""")
            fake.chmod(0o755)
            (root/'model').write_text('fake model')
            (root/'input').write_text(json.dumps({'id':'x','request':{}})+'\n')
            (root/'gold').write_text(json.dumps({'x':{'d':'yes'}}))
            matrix=[{'name':'baseline','args':['--speed','50']},{'name':'candidate','args':['--speed','20']}]
            (root/'matrix').write_text(json.dumps(matrix))
            command=[os.sys.executable,str(Path(__file__).with_name('benchmark_decision_performance.py')),
                     '--cpu','--binary',str(fake),'--model',str(root/'model'),'--input',str(root/'input'),
                     '--gold',str(root/'gold'),'--matrix',str(root/'matrix')]
            good=subprocess.run(command+['--out',str(root/'good')],capture_output=True,text=True)
            self.assertEqual(good.returncode,0,good.stderr)
            self.assertEqual(json.loads((root/'good'/'selection.json').read_text())['selected']['name'],'candidate')
            matrix[0]['args'].append('--incomplete');(root/'matrix').write_text(json.dumps(matrix))
            bad=subprocess.run(command+['--out',str(root/'bad')],capture_output=True,text=True)
            self.assertNotEqual(bad.returncode,0)
            self.assertFalse((root/'bad'/'selection.json').exists())

@unittest.skipUnless(os.environ.get('L2S1_EVALUATOR'), 'set L2S1_EVALUATOR for executable contract checks')
class EvaluatorContracts(unittest.TestCase):
    def test_resident_rejects_incompatible_inputs_before_model_load(self):
        decision={'id':'d','instruction':'Choose one','kind':{'type':'choice','options':[{'id':'a','criterion':'a'},{'id':'b','criterion':'b'}]}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            rows=[{'id':'a','request':{'state':{},'decisions':[decision]}}, {'id':'b','request':{'state':{},'decisions':[dict(decision,instruction='Different schema')]}}]
            (root/'input.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            base=[os.environ['L2S1_EVALUATOR'],'--model','missing.gguf','--input',str(root/'input.jsonl'),'--output',str(root/'output.jsonl')]
            for args,message in [(['--resident','fixed-schema'],'requires'),
                                 (['--resident','fixed-schema','--execution-mode','prefix-reuse'],'identical decision'),
                                 (['--resident','shared-prefix'],'requires')]:
                result=subprocess.run(base+args,capture_output=True,text=True)
                self.assertNotEqual(result.returncode,0)
                self.assertIn(message,result.stderr)
                self.assertFalse((root/'output.jsonl').exists())

if __name__ == '__main__': unittest.main()
