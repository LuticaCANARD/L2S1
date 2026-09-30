#!/usr/bin/env python3
"""Convert native probabilities to Jev-shaped answers and score a frozen holdout.

This is a local file adapter, not a TypeSafe API implementation. Answer rendering
follows Ollaya f9e2d11fee1d01235878bfa6cfa1eb1e42bbbaea, answer.rs: normalized
top-probability confidence and four-decimal wire rounding. Metrics stay unrounded.
Native abstention is retained separately and never turns into an accepted action.
"""
import argparse
import collections
import json
import math
from pathlib import Path

from .prepare_jev_data import distribution, jsonl
from .common import digest, option_specs, read_jsonl, require, write_json


def answer(decision, result):
    ids = [o['id'] for o in option_specs(decision)]
    require(result['id'] == decision['id'] and not result.get('truncated'), 'Invalid/truncated result')
    require(len(result['scores']) == len(ids), 'Missing or duplicate scores')
    values = {s['id']: s['option_probability'] for s in result['scores']}
    p = dict(zip(ids, distribution(values, ids)))
    confidence = round(max(0., min(1., (len(p)*max(p.values())-1)/(len(p)-1))), 4)
    wire_p = {k:round(v, 4) for k,v in p.items()}
    kind = decision['kind']['type']
    if kind == 'binary':
        return dict(type='noul', noul=round(p['true'], 4))
    if kind == 'choice':
        return dict(type='choice', choice=max(p, key=p.get), probabilities=wire_p, confidence=confidence)
    levels = decision['kind']['levels']
    require([o['value'] for o in levels] == list(range(len(levels))), 'Jev Score requires zero-based levels')
    value = {o['id']: o['value'] for o in levels}
    return dict(type='score', score=round(sum(value[k]*v for k, v in p.items()), 4),
                probabilities=wire_p, confidence=confidence, legend={o['id']: o['criterion'] for o in levels})


def native_selection(result):
    value = result['value']
    selected = value.get('value') if value['type'] == 'binary' else value.get('selected')
    if isinstance(selected, bool):
        return str(selected).lower()
    return selected


def score_case(case, prediction):
    require('response' in prediction, prediction.get('error', 'Missing response'))
    response = prediction['response']
    decisions = case['request']['decisions']
    require(len(response['results']) == len(decisions), 'Wrong result count')
    results = {r['id']: r for r in response['results']}
    require(set(results) == {d['id'] for d in decisions}, 'Wrong result IDs')
    answers, policies, metrics = {}, {}, []
    for d in decisions:
        r, g = results[d['id']], case['gold'][d['id']]
        a = answer(d, r)
        answers[d['id']] = a
        ids = [o['id'] for o in option_specs(d)]
        pv = dict(zip(ids, distribution({s['id']: s['option_probability'] for s in r['scores']}, ids)))
        gv = dict(zip(ids, distribution(g['probabilities'], ids)))
        # Match the existing typed-decisions protocol: binary ties true, other ties first option.
        predicted = ('true' if pv['true'] >= .5 else 'false') if a['type'] == 'noul' else max(pv, key=pv.get)
        selected = native_selection(r)
        require(selected is None or selected in ids, 'Invalid native selection')
        accepted = selected is not None and not r.get('abstention_reasons')
        policies[d['id']] = dict(accepted=accepted, selected=selected,
            abstention_reasons=r.get('abstention_reasons', []), candidate_mass=r['candidate_mass'],
            top_option_probability=max(pv.values()), entropy_confidence=r.get('entropy_confidence'))
        row = dict(type=a['type'], workflow=case['workflow'], correct=predicted == g['label'],
            accepted=accepted, accepted_correct=accepted and selected == g['label'],
            confidence=max(pv.values()), hard_nll=-math.log(max(pv[g['label']], 1e-12)),
            hard_brier=sum((pv[k]-float(k == g['label']))**2 for k in ids),
            soft_kl=sum(v*math.log(v/max(pv[k], 1e-12)) for k, v in gv.items() if v > 0),
            soft_brier=sum((pv[k]-gv[k])**2 for k in ids))
        if a['type'] == 'score':
            value = {o['id']: o['value'] for o in d['kind']['levels']}
            target_score = float(g['score']) if 'score' in g else sum(value[k]*v for k,v in gv.items())
            require(math.isfinite(target_score) and 0 <= target_score <= len(ids)-1, 'Invalid gold Score expectation')
            row['score_mae'] = abs(sum(value[k]*v for k,v in pv.items())-target_score)
        metrics.append(row)
    return dict(id=case['id'], model=response['backend']['model_description'], answers=answers,
                l2s1_policy=dict(thresholds=response['policy'], decisions=policies),
                usage=dict(input_tokens=sum(r['input_tokens'] for r in results.values()), output_tokens=0),
                confidence_definition='Ollaya: (K * max_probability - 1) / (K - 1); uncalibrated',
                elapsed_ms=prediction['elapsed_ms']), metrics


def summarize(rows, planned):
    accepted = sum(r['accepted'] for r in rows)
    correct = sum(r['correct'] for r in rows)
    accepted_correct = sum(r['accepted_correct'] for r in rows)
    out = dict(decisions=planned, valid=len(rows), failed=planned-len(rows), raw_correct=correct,
               raw_accuracy=correct/planned, accepted=accepted, accepted_correct=accepted_correct,
               coverage=accepted/planned, accepted_accuracy=accepted_correct/accepted if accepted else None,
               correct_all=accepted_correct/planned)
    for key in ('hard_nll', 'hard_brier', 'soft_kl', 'soft_brier', 'score_mae'):
        values = [r[key] for r in rows if key in r]
        out[key] = sum(values)/len(values) if values else None
    bins = collections.defaultdict(list)
    for r in rows:
        bins[min(int(r['confidence']*15), 14)].append(r)
    out['hard_ece_15'] = sum(abs(sum(r['confidence']-r['correct'] for r in b)) for b in bins.values())/len(rows) if rows else None
    return out


def report(data, predictions, output):
    cases = read_jsonl(data/'test.jsonl')
    manifest = json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    require(digest(data/'test.jsonl') == manifest['splits']['test']['sha256'], 'Test hash mismatch')
    rows = read_jsonl(predictions)
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate prediction IDs')
    indexed = {r['id']: r for r in rows}
    require(set(indexed) <= {r['id'] for r in cases}, 'Unexpected prediction IDs')
    answers, metrics, failures = [], [], []
    planned_types, planned_workflows = collections.Counter(), collections.Counter()
    for case in cases:
        planned_types.update(g['type'] for g in case['gold'].values())
        planned_workflows[case['workflow']] += len(case['gold'])
        try:
            converted, case_metrics = score_case(case, indexed.get(case['id'], {}))
            answers.append(converted)
            metrics.extend(case_metrics)
        except (KeyError, TypeError, ValueError) as error:
            failures.append(dict(id=case['id'], error=str(error)))
            answers.append(dict(id=case['id'], error=str(error)))
    times = sorted(r['elapsed_ms'] for r in rows)
    require(all(math.isfinite(t) and t > 0 for t in times), 'Invalid latency')
    output.mkdir(parents=True, exist_ok=False)
    jsonl(output/'jev-answers.jsonl', answers)
    summary = dict(schema_version=1, mode='specialist',
        manifest_sha256=digest(data/'manifest.json'), predictions_sha256=digest(predictions),
        protocol=manifest['protocol'], overall=summarize(metrics, sum(planned_types.values())),
        by_type={k:summarize([r for r in metrics if r['type'] == k], n) for k,n in planned_types.items()},
        by_workflow={k:summarize([r for r in metrics if r['workflow'] == k], n) for k,n in planned_workflows.items()},
        failures=failures, latency_ms=dict(samples=len(times), p50=times[math.ceil(.5*len(times))-1] if times else None,
            p95=times[math.ceil(.95*len(times))-1] if times else None),
        scope=manifest['scope']+' Failure decisions remain in accuracy/coverage denominators. Latency is per request; request sizes may vary. See evaluator command for load/warmup settings.')
    write_json(output/'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(prog="l2s1-train report", description=__doc__)
    for name in ('data', 'predictions', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if report(a.data, a.predictions, a.output)['failures']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
