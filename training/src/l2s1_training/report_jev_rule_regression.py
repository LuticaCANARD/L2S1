#!/usr/bin/env python3
"""Check the existing decision-rules suite after specialization; ties are wrong."""
import argparse
import json
from pathlib import Path

from .prepare_jev_data import distribution, jsonl
from .report_jev import native_selection
from .common import digest, option_specs, read_jsonl, require, write_json


def unique_top1(ids, probabilities):
    best = max(probabilities)
    winners = [k for k,v in zip(ids, probabilities) if abs(v-best) < 1e-12]
    return winners[0] if len(winners) == 1 else None


def main():
    p = argparse.ArgumentParser(prog="l2s1-train regression", description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--requests-output', type=Path)
    p.add_argument('--predictions', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    suite = json.loads(a.fixture.read_text(encoding='utf-8'))
    cases = suite['cases']
    if a.requests_output:
        jsonl(a.requests_output, [dict(id=c['id'], request=c['request']) for c in cases])
        return
    if a.predictions is None or a.output is None:
        p.error('Scoring requires --predictions and --output')
    rows = read_jsonl(a.predictions)
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate predictions')
    indexed = {r['id']:r for r in rows}
    require(set(indexed) == {r['id'] for r in cases}, 'Incomplete regression')
    outcomes = []
    for c in cases:
        results = indexed[c['id']]['response']['results']
        require(len(results) == len(c['request']['decisions']), 'Missing regression results')
        by_id = {r['id']:r for r in results}
        require(len(by_id) == len(results), 'Duplicate regression results')
        for d in c['request']['decisions']:
            r = by_id[d['id']]
            require(not r['truncated'], 'Truncated regression')
            ids = [o['id'] for o in option_specs(d)]
            ps = distribution({s['id']:s['option_probability'] for s in r['scores']}, ids)
            winner = unique_top1(ids, ps)
            selected = native_selection(r)
            gold = c['expected'][d['id']]
            accepted = selected is not None and not r['abstention_reasons']
            outcomes.append(dict(case=c['id'], decision=d['id'], expected=gold,
                raw_correct=winner == gold,
                accepted=accepted, accepted_correct=accepted and selected == gold,
                selected=selected, probabilities=dict(zip(ids, ps))))
    n = len(outcomes)
    accepted = sum(r['accepted'] for r in outcomes)
    correct = sum(r['accepted_correct'] for r in outcomes)
    write_json(a.output, dict(suite=suite['id'], fixture_sha256=digest(a.fixture),
        predictions_sha256=digest(a.predictions), decisions=n, accepted=accepted,
        accepted_correct=correct, accepted_wrong=accepted-correct,
        raw_accuracy=sum(r['raw_correct'] for r in outcomes)/n,
        coverage=accepted/n, accepted_accuracy=correct/accepted if accepted else None,
        correct_all=correct/n, outcomes=outcomes,
        scope='Previously seen regression suite, single pass; not a new holdout or the Windows RTX 5090 run.'))


if __name__ == '__main__':
    main()
