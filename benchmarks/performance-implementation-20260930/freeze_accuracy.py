import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from benchmark_decision_performance import read_rows,metrics,percentile,write_json,digest
root=Path(__file__).resolve().parents[2]
study=root/'benchmarks/performance-implementation-20260930'
import argparse
parser=argparse.ArgumentParser(description='Freeze model/family choices before running holdout')
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--selection-dir',type=Path,default=study)
args=parser.parse_args()
out=args.results
args.selection_dir.mkdir(parents=True,exist_ok=True)
for model in ('e2b','12b'):
 run=out/model/'accuracy-dev'
 gold=json.loads((study/'dev-gold.json').read_text())
 selected=[];families={}
 for family in ('numeric','temporal','graph'):
  labels={k:v for k,v in gold.items() if '-'+family+'-' in k}
  stats={config:[metrics([r for r in read_rows(run/f'{config}-{i}.jsonl') if r['id'] in labels],labels) for i in range(3)] for config in ('raw','facts')}
  # Require the same outcome on every repeat; timing gate uses worst p95.
  raw=stats['raw'];facts=stats['facts']
  enable=min(s['raw_correct'] for s in facts)>max(s['raw_correct'] for s in raw) and max(s['wrong_accepted'] for s in facts)<=min(s['wrong_accepted'] for s in raw) and max(s['p95_ms'] for s in facts)<=2*max(s['p95_ms'] for s in raw)
  if enable:selected.append(family)
  families[family]={'enabled':enable,'metrics':stats}
 freeze={'model':model,'selected_families':selected,'families':families,
         'protocol_sha256':digest(study/'protocol.json'),'development_input_sha256':digest(study/'dev.jsonl'),
         'development_outputs':{p.name:digest(p) for p in sorted(run.glob('*.jsonl'))},
         'holdout_input_sha256':digest(study/'holdout.jsonl'), 'holdout_outcomes_examined':False}
 dest=args.selection_dir/(model+'-accuracy-selection.json')
 if dest.exists():raise SystemExit('selection already frozen')
 write_json(dest,freeze)
 rows=read_rows(study/'holdout.jsonl')
 for row in rows:
  family=row['id'].split('-')[1]
  if family not in selected:row['facts']=[]
 prepared=out/model/'selected-holdout.jsonl'
 prepared.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
 print(model,selected,flush=True)
