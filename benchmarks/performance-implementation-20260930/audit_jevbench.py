"""Check L2S1's saved JevBench scores against the pinned upstream Python scorer."""
import argparse
import json
import math
from pathlib import Path
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream',type=Path,required=True)
parser.add_argument('--results',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.upstream))
from jevbench.tasks import Task
from jevbench.scoring import score_task
from jevbench.summarize import summarize
read=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
tasks=[Task.from_dict(r) for r in rows(args.results/'jevbench-prepared/tasks-with-gold.jsonl')]
by={t.id:t for t in tasks}
assert len(tasks)==len(by)==231
output=[]
for model in ('e2b','12b'):
 for config in ('baseline','flash'):
  folder=args.results/model/'jevbench'
  official=folder/('official-'+config)
  summary=read(official/'summary.json')
  predictions=rows(folder/(config+'-0.jsonl'))
  saved={r['task_id']:r for r in rows(official/'jevbench-records.jsonl')}
  assert len(predictions)==len(saved)==231
  checked=[]
  for prediction in predictions:
   assert 'error' not in prediction
   task=by[prediction['id']]
   result=prediction['response']['results'][0]
   backend=prediction['response']['backend']
   assert backend['offload_requested'] and 'RTX 3080' in backend['offload_device']
   assert backend['execution_mode']=='fresh' and backend['prompt_layout']=='legacy'
   assert backend['compute']['flash_attention']==('off' if config=='baseline' else 'on')
   assert not result['truncated']
   mapping={'false':'no','true':'yes'} if task.question['type']=='noul' else {}
   probabilities={mapping.get(s['id'],s['id']):s['option_probability'] for s in result['scores']}
   actual=score_task(probabilities,task)
   expected=saved[task.id]
   for key in ('correct','predicted','valid','strict_valid','renormalized','probs'):
    assert actual[key]==expected[key],(model,config,task.id,key)
   checked.append(dict(expected,**actual))
  independent=summarize(tasks,checked)
  for key in ('n_correct','n_scorable','n_valid','n_attempted','n_planned','accuracy','brier_mean','macro_accuracy'):
   assert math.isclose(independent[key],summary[key],abs_tol=1e-10),(model,config,key)
  for key in ('p50_s','p95_s'):
   assert math.isclose(independent['latency'][key],summary['latency'][key],abs_tol=1e-10)
  assert math.isclose(independent['ece']['ece'],summary['ece']['ece'],abs_tol=1e-10)
  compact=lambda value:{k:value[k] for k in ('n_correct','n_scorable','n_valid','n_planned','accuracy','latency')}
  output.append({'model':model,'config':config,'correct':summary['n_correct'],'total':231,
                 'upstream_scorer_matches':True,'selective_policy':summary['selective_policy'],
                 'per_family':{k:compact(v) for k,v in summary['per_family'].items()},
                 'tiers':{k:compact(v) for k,v in summary['tiers'].items()},
                 'brier_mean':summary['brier_mean'],'ece':summary['ece']['ece'],
                 'latency_first_run':summary['latency']})
Path(__file__).with_name('jevbench-audit.json').write_text(json.dumps(output,indent=2)+'\n')
print('All four 231-item runs agree with pinned upstream scoring and summarization.')
