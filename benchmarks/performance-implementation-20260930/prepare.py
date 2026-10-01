"""Deterministic synthetic development/holdout inputs. No model labels used.

Holdout uses disjoint number ranges and graph node namespaces. This measures
structured arithmetic/graph contracts, not natural-language JevBench accuracy.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]

def write(name, rows, labels):
    (ROOT/(name+'.jsonl')).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    (ROOT/(name+'-gold.json')).write_text(json.dumps(labels,indent=2)+'\n')


def accuracy(split, count, seed, minimum):
    rng=random.Random(seed)
    rows=[];labels={}
    for family in ('numeric','temporal','graph'):
        for i in range(count):
            key=f'{split}-{family}-{i:03d}'
            if family=='numeric':
                a=rng.randrange(minimum,minimum*10);b=rng.randrange(minimum,minimum*10)
                if i%3==0:b=-b
                state={'a':a,'b':b};answer=a+b
                facts=[dict(name='sum',op='add',left='/a',right='/b')]
                instruction='What is a + b? Use the exact integer sum, including its sign.'
                choices=[answer,answer+1,answer-1,answer+10]
            elif family=='temporal':
                start=rng.randrange(minimum*100,minimum*100+100000)
                answer=rng.randrange(-90000,90000)
                state={'start_utc_seconds':start,'end_utc_seconds':start+answer}
                facts=[dict(name='elapsed',op='elapsed_seconds',start='/start_utc_seconds',end='/end_utc_seconds')]
                instruction='How many signed seconds elapsed from start_utc_seconds to end_utc_seconds? Compute end minus start.'
                choices=[answer,answer+60,answer-60,-answer if answer else 1]
            else:
                hops=i%6
                nodes=[f'{split}_{i}_node_{j}' for j in range(9)]
                edges=[[nodes[j],nodes[j+1]] for j in range(hops)]
                edges += [[nodes[7],nodes[8]],[nodes[8],nodes[7]]]
                if hops>1:edges += [[nodes[1],nodes[0]]]
                rng.shuffle(edges)
                unreachable=i%4==0
                state={'edges':edges,'from':nodes[0],'to':nodes[8] if unreachable else nodes[hops]}
                answer='unreachable' if unreachable else str(hops)
                facts=[dict(name='route',op='shortest_path',edges='/edges',**{'from':'/from','to':'/to'})]
                instruction='What is the minimum number of directed edges from from to to? A node reaches itself in zero hops. Select unreachable if no path exists.'
                choices=list(map(str,range(6)))+['unreachable']
            rng.shuffle(choices)
            options=[{'id':f'v{j}','criterion':str(value)} for j,value in enumerate(choices)]
            expected=options[choices.index(answer)]['id']
            decision={'id':'answer','instruction':instruction,'kind':{'type':'choice','options':options}}
            rows.append({'id':key,'request':{'state':state,'decisions':[decision]},'facts':facts})
            labels[key]={'answer':expected}
    write(split,rows,labels)


def main():
    accuracy('dev',12,73001,1000)
    accuracy('holdout',40,73002,1000000)
    fixture=json.loads((REPO/'tests/fixtures/decision_benchmark.json').read_text())['cases']
    rows=[];labels={}
    for i,case in enumerate(fixture):
        request=case['request']
        # Vary prefill length without introducing new task instructions.
        request['state']['audit_history']='Parcel scanned at depot. '*(i%4*40)
        rows.append({'id':case['id'],'request':request});labels[case['id']]=case['expected']
    write('compute-dev',rows,labels)
    decision=fixture[0]['request']['decisions'][0]
    # Long fixed schema makes batch-aligned reuse observable with batch=128.
    decision['instruction'] += ' '+('Use the current storage_requirement field; historical scans do not determine the storage zone. '*22)
    fixed=[];labels={}
    for i in range(12):
        zone=['ambient','chilled','frozen'][i%3]
        key=f'schema-{i}'
        fixed.append({'id':key,'request':{'state':{'storage_requirement':zone,'parcel':i},'decisions':[decision]}})
        labels[key]={'storage_zone':zone}
    write('schema',fixed,labels)
    for row in fixed:
        row['request']['decisions'][0]=fixture[1]['request']['decisions'][0]
        row['request']['shared']={'manual':'Ambient shipments go to ambient storage. Chilled shipments go to chilled storage. Frozen shipments go to frozen storage. '*24}
    write('shared',fixed,labels)
    base=['--context','4096','--batch','256','--threads','4','--prompt-layout','legacy']
    configs=[{'name':'baseline','args':base},
             {'name':'flash','args':base+['--flash-attention','on']},
             {'name':'flash-ub128','args':base+['--flash-attention','on','--ubatch','128']}]
    small=['--context','4096','--batch','128','--threads','4','--prompt-layout','legacy','--flash-attention','on']
    configs.append({'name':'flash-b128','args':small})
    for width in (1,2,3):
        configs.append({'name':f'parallel-dynamic-{width}','args':small+['--execution-mode','parallel','--parallel-width',str(width),'--parallel-context-dynamic']})
    configs.append({'name':'parallel-static-3','args':small+['--execution-mode','parallel','--parallel-width','3']})
    (ROOT/'compute-matrix.json').write_text(json.dumps(configs,indent=2)+'\n')
    for kind in ('schema','shared'):
        mode='prefix-reuse' if kind=='schema' else 'parallel'
        resident='fixed-schema' if kind=='schema' else 'shared-prefix'
        args=small+(['--request-batch-size','2','--parallel-width','2','--parallel-context-dynamic'] if kind=='shared' else [])
        matrix=[{'name':'fresh','args':args}, {'name':'split-cold','args':args+['--execution-mode',mode]},
                {'name':'split-resident','args':args+['--execution-mode',mode,'--resident',resident]}]
        (ROOT/(kind+'-matrix.json')).write_text(json.dumps(matrix,indent=2)+'\n')
    matrix=[{'name':'raw','args':base}, {'name':'facts','args':base+['--derive-facts']}]
    (ROOT/'accuracy-matrix.json').write_text(json.dumps(matrix,indent=2)+'\n')

if __name__=='__main__':main()
