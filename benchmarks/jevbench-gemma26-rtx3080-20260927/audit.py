"""Recount native results independently; requires the ignored full run directory."""
import argparse
from collections import Counter
import hashlib
import json
import math
import itertools
import statistics
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    k = (len(values) - 1) * fraction
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] if lo == hi else values[lo] * (hi - k) + values[hi] * (k - lo)


def evidence(task, result):
    assert not result['truncated']
    binary = task['question']['type'] == 'noul'
    label = lambda key: {'false': 'no', 'true': 'yes'}.get(key, key) if binary else key
    p = {label(s['id']): s['option_probability'] for s in result['scores']}
    assert set(p) == set(task['labels'])
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in p.values())
    assert abs(sum(p.values()) - 1) < 1e-3
    # The pinned JevBench scorer resolves raw ties by lexical label order.
    top = min(p, key=lambda key: (-p[key], key))
    value = result['value']
    if value['type'] == 'binary':
        selected = None if value['value'] is None else ('yes' if value['value'] else 'no')
    else:
        selected = value['selected']
    expected = str(task['expected'])
    return dict(probabilities=p, predicted=top, selected=selected, expected=expected,
                correct=top == expected, accepted_correct=selected == expected,
                candidate_mass=result['candidate_mass'], input_tokens=result['input_tokens'],
                reused_prefix_tokens=result['reused_prefix_tokens'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tasks = {t['id']: t for t in read_rows(args.runs/'prepared/tasks-with-gold.jsonl')}
    assert len(tasks) == 231
    assert all(t['expected'] is not None and not t['provenance'].get('exclude_reason') for t in tasks.values())
    schemas = Counter(json.dumps(d, sort_keys=True) for row in read_rows(args.runs/'prepared/requests.jsonl') for d in row['request']['decisions'])
    assert len(schemas) == 117 and sum(n == 1 for n in schemas.values()) == 96
    runs, exported, baselines = [], [], {}
    for model in ['gemma26']:
        folder = args.runs/model
        report = json.loads((folder/'summary.json').read_text())
        manifest = json.loads((folder/'manifest.json').read_text())
        raw = read_rows(folder/'predictions.jsonl')
        assert manifest['model_sha256'] == 'f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f'
        assert len(raw) == 231 and {r['id'] for r in raw} == set(tasks)
        rows = []
        for r in raw:
            assert not r.get('error') and len(r['response']['results']) == 1
            backend = r['response']['backend']
            assert backend['offload_requested'] and 'RTX 3080' in backend['offload_device']
            assert backend['execution_mode'] == 'fresh' and backend['compute']['batch'] == 256
            assert backend['compute']['context'] == 8192 and backend['compute']['cpu_moe_layers'] == 22
            item = evidence(tasks[r['id']], r['response']['results'][0])
            item.update(model=model, id=r['id'], elapsed_ms=r['elapsed_ms'])
            rows.append(item)
        exported.extend(rows)
        baselines[model] = {r['id']:r for r in rows}
        correct = sum(r['correct'] for r in rows)
        accepted = sum(r['selected'] is not None for r in rows)
        accepted_correct = sum(r['accepted_correct'] for r in rows)
        brier = statistics.mean(sum((p - (key == r['expected']))**2 for key,p in r['probabilities'].items()) for r in rows)
        bins = [[] for _ in range(10)]
        for r in rows:
            confidence = max(r['probabilities'].values())
            bins[min(int(confidence*10),9)].append((confidence,r['correct']))
        ece = sum(abs(sum(c for c,_ in b)-sum(ok for _,ok in b))/231 for b in bins)
        times = [r['elapsed_ms'] for r in rows]
        assert report['n_valid'] == report['n_scorable'] == 231 and report['n_correct'] == correct
        assert abs(report['brier_mean']-brier) < 1e-12 and abs(report['ece']['ece']-ece) < 1e-12
        policy = report['selective_policy']
        assert (policy['accepted'],policy['correct'],policy['wrong_accepted'],policy['error']) == (accepted,accepted_correct,accepted-accepted_correct,0)
        assert abs(report['latency']['p50_s']*1000-percentile(times,.5)) < 1e-9
        assert abs(report['latency']['p95_s']*1000-percentile(times,.95)) < 1e-9
        tiers = {tier:{'correct':sum(r['correct'] for r in rows if r['id'].startswith(tier+'-')),
                       'total':sum(r['id'].startswith(tier+'-') for r in rows)} for tier in ['easy','original','hard']}
        for tier, counts in tiers.items():
            assert report['tiers'][tier]['n_correct'] == counts['correct']
        runs.append(dict(model=model, checkpoint=manifest['model_name'], total=231, correct=correct,
                         accuracy=correct/231, accepted=accepted, accepted_correct=accepted_correct,
                         wrong_accepted=accepted-accepted_correct, abstained=231-accepted, coverage=accepted/231,
                         accepted_accuracy=accepted_correct/accepted if accepted else None,
                         p50_ms=percentile(times,.5),p95_ms=percentile(times,.95),total_ms=sum(times),
                         brier=brier,ece=ece,tiers=tiers,backend=report['backend'],
                         input_tokens=sum(r['input_tokens'] for r in rows),
                         reused_prefix_tokens=sum(r['reused_prefix_tokens'] for r in rows),
                         risk_coverage_at_error_budget=report['risk_coverage']['at_error_budget'],
                         gpu_memory=report['gpu_memory'],
                         identity={k:manifest[k] for k in ['model_sha256','evaluator_sha256','requests_sha256','adapter_sha256']},
                         predictions_sha256=sha(folder/'predictions.jsonl'),summary_sha256=sha(folder/'summary.json')))
    diagnostic_ids = {row['id'] for row in read_rows(args.runs/'repeated-schema-requests.jsonl')}
    assert len(diagnostic_ids) == 135
    expected_ids = {row['id'] for row in read_rows(args.runs/'prepared/requests.jsonl')
                    if schemas[json.dumps(row['request']['decisions'][0], sort_keys=True)] > 1}
    assert diagnostic_ids == expected_ids
    diagnostics = []
    for model, batch in itertools.product(['gemma26'], [64, 256]):
        path = args.runs/f'{model}-schema-b{batch}.json'
        data = json.loads(path.read_text())
        assert data['identity']['compute']['context'] == 8192
        assert data['identity']['compute']['cpu_moe_layers'] == 22
        assert data['identity']['compute']['batch'] == batch
        assert data['unique_decisions'] == 135 and data['schemas'] == 21
        assert len(data['cases']) == 135 and {c['case'] for c in data['cases']} == diagnostic_ids
        assert len(data['records']) == 2
        pair = {}
        for record in data['records']:
            assert len(record['responses']) == len(data['cases'])
            rows = {case['case']:evidence(tasks[case['case']],response['results'][0])
                    for case,response in zip(data['cases'],record['responses'])}
            assert len(rows) == 135
            pair[record['mode']] = (record,rows)
        fresh, reuse = pair['fresh'],pair['shared_decision']
        delta = max(abs(fresh[1][key]['probabilities'][label]-reuse[1][key]['probabilities'][label])
                    for key in diagnostic_ids for label in tasks[key]['labels'])
        mass_delta = max(abs(fresh[1][key]['candidate_mass']-reuse[1][key]['candidate_mass']) for key in diagnostic_ids)
        changed = {field:sum(fresh[1][key][field] != reuse[1][key][field] for key in diagnostic_ids) for field in ['predicted','selected']}
        assert abs(delta-fresh[0]['difference']['max_probability_delta']) < 1e-12
        assert changed['predicted'] == fresh[0]['difference']['changed_top1']
        diagnostics.append(dict(model=model, batch=batch, rounds=1, schemas=21, decisions=135, fresh_ms=fresh[0]['elapsed_ms'],
                                reused_ms=reuse[0]['elapsed_ms'], speed_ratio=fresh[0]['elapsed_ms']/reuse[0]['elapsed_ms'],
                                input_tokens=reuse[0]['input_tokens'], reused_prefix_tokens=reuse[0]['reused_prefix_tokens'],
                                probability_delta=delta,mass_delta=mass_delta,changed=changed,
                                equivalence_passed=data['equivalence_passed'],
                                correct_fresh=sum(r['correct'] for r in fresh[1].values()),
                                correct_reuse=sum(r['correct'] for r in reuse[1].values()),
                                vs_standard256_changed_top1=sum(fresh[1][k]['predicted'] != baselines[model][k]['predicted'] for k in diagnostic_ids),
                                vs_standard256_changed_selection=sum(fresh[1][k]['selected'] != baselines[model][k]['selected'] for k in diagnostic_ids),
                                vs_standard256_probability_delta=max(abs(fresh[1][k]['probabilities'][label]-baselines[model][k]['probabilities'][label]) for k in diagnostic_ids for label in tasks[k]['labels']),
                                vs_standard256_changed_cases=[dict(id=k, expected=fresh[1][k]['expected'],
                                    predicted=fresh[1][k]['predicted'], standard_predicted=baselines[model][k]['predicted'],
                                    selected=fresh[1][k]['selected'], standard_selected=baselines[model][k]['selected'])
                                    for k in sorted(diagnostic_ids) if any(fresh[1][k][field] != baselines[model][k][field] for field in ['predicted','selected'])],
                                raw_report_sha256=sha(path)))
    args.output.mkdir(parents=True,exist_ok=True)
    result = dict(date='2026-09-27',device='NVIDIA GeForce RTX 3080 10 GiB',runs=runs,schema_diagnostics=diagnostics,
                  input_manifest=json.loads((args.runs/'prepared/manifest.json').read_text()),
                  source=json.loads((args.runs/'source-provenance.json').read_text()),
                  audit_sha256=sha(Path(__file__)),
                  diagnostic_requests_sha256=sha(args.runs/'repeated-schema-requests.jsonl'),
                  schema_counts=dict(total=len(schemas), singletons=sum(n == 1 for n in schemas.values()),
                                     repeated=sum(n > 1 for n in schemas.values()), post_first_calls=sum(n - 1 for n in schemas.values())),
                  scope='Public 231 items per model, not full 534 or official scores. One fresh 26B run with CPU expert offload. Schema diagnostics cover only the 135 items in 21 repeated schemas (96 singleton schemas excluded), with one paired grouped pass after per-path warmups; not stable percentiles or an accuracy gain. Latency comparisons across previous hardware/builds are uncontrolled. Full native responses and logs remain local ignored artifacts.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'predictions.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in exported))
    for r in runs:
        print(r['model'],r['correct'],r['accepted'],r['accepted_correct'],round(r['p50_ms'],3))
    print(json.dumps(diagnostics,indent=2))


if __name__ == '__main__':
    main()
