import json,statistics,math
from pathlib import Path
root=Path(__file__).resolve().parent
expected={c['id']:c['expected'] for c in json.loads((root/'decision_benchmark.json').read_text())['cases']}
reports=[];by_mode={}
def top(result):return max(result['evidence']['scores'],key=lambda s:s['option_probability'])['id']
for mode in ('fresh','fixed'):
    rows=json.loads((root/f'{mode}-responses.json').read_text())
    telemetry=[json.loads(line) for line in (root/f'{mode}-telemetry.jsonl').read_text().splitlines()]
    samples=sorted(row['elapsed_ms'] for row in rows)
    accepted=correct=raw_correct=reused=total=n=changed=comparisons=0;first={}
    for row in rows:
        for result in row['response']['results']:
            good=top(result)==expected[row['case_id']][result['id']];ok=result['status']=='selected'
            n+=1;raw_correct+=good;accepted+=ok;correct+=ok and good
            reused+=result['usage'].get('reused_prefix_tokens') or 0;total+=result['usage'].get('input_tokens') or 0
            key=(row['case_id'],result['id'])
            if key in first:changed+=first[key]['evidence']!=result['evidence'];comparisons+=1
            else:first[key]=result
    summary=dict(mode=mode,measured_requests=len(rows),unique_cases=len(set(r['case_id'] for r in rows)),decisions=n,p50_ms=statistics.median(samples),p95_ms=samples[math.ceil(len(samples)*.95)-1],raw_top1=raw_correct/n,coverage=accepted/n,accepted_accuracy=correct/accepted if accepted else None,accepted_decisions=accepted,accepted_correct=correct,correct_accepted_fraction=correct/n,reused_prefix_tokens=reused,input_tokens=total,repeat_evidence_changes=changed,repeat_evidence_comparisons=comparisons,peak_native_rss_gib=max(s['hwm_kib'] for s in telemetry)/1024**2,max_temperature_c=max(s['temperature_c'] for s in telemetry),max_swap_mib=max(s['swap_kib'] for s in telemetry)/1024,active_undervoltage_samples=sum(bool(int(s['throttled'].split('=')[1],16)&1) for s in telemetry),active_throttle_samples=sum(bool(int(s['throttled'].split('=')[1],16)&4) for s in telemetry),telemetry_samples=len(telemetry))
    reports.append(summary);by_mode[mode]=first
keys=by_mode['fresh'].keys()
report=dict(runs=reports,p50_speedup=reports[0]['p50_ms']/reports[1]['p50_ms'],cross_mode_unique_decisions=len(keys),cross_mode_top1_changes=sum(top(by_mode['fresh'][k])!=top(by_mode['fixed'][k]) for k in keys),cross_mode_evidence_exact_matches=sum(by_mode['fresh'][k]['evidence']==by_mode['fixed'][k]['evidence'] for k in keys),scope='Published v0.1.3, physical Pi5, 3 warehouse cases x2 measured passes per mode. Median includes Python stdio round trip. One warmup request excluded. Sequential fresh then fixed order. Both use SIMD. Small smoke benchmark, not the complete rules suite. Summary reconstructed from preserved responses and telemetry after fixing score id field in the original reporter. Native predictions unchanged.')
(root/'verified-summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
