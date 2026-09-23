#!/usr/bin/env python3
"""Check repeated output equality and summarize isolated paired GPU timings."""
import argparse, json, statistics
from pathlib import Path
from kaggle_ag_news import sha256

def read(p):return [json.loads(s) for s in p.read_text().splitlines()]
def results(rows):return [(r['id'],r['response']['results']) for r in rows]
def summarize(root,previous):
    timing={}
    for kind in ['base','hidden']:
        runs=[read(root/'timing'/f'{kind}-{n}.jsonl') for n in range(3)]
        assert all(len(r)==400 for r in runs)
        assert all(results(r)==results(runs[0]) for r in runs)
        totals=[sum(r['elapsed_ms'] for r in rows) for rows in runs]
        timing[kind]=dict(total_ms=totals,median_total_ms=statistics.median(totals),
            p50_ms=[statistics.median(r['elapsed_ms'] for r in rows) for rows in runs],
            p95_ms=[sorted(r['elapsed_ms'] for r in rows)[379] for rows in runs],
            load_ms=[rows[0]['load_ms'] for rows in runs],identical_repeated_results=True)
    timing['overhead_percent']=(timing['hidden']['median_total_ms']/timing['base']['median_total_ms']-1)*100
    timing['scope']='Three isolated paired runs, 400 requests each, one warmup. Pair order base/hidden, hidden/base, base/hidden. Excludes model load, GGUF hash, warmup and serialization.'
    timing['median_request_ms']={k:timing[k]['median_total_ms']/400 for k in ['base','hidden']}
    base=read(root/'runtime/ag-news-base.jsonl');head=read(root/'runtime/ag-news-selected.jsonl')
    assert len(base)==400 and results(base)==results(head)
    old=read(previous);assert results(old)==results(base)
    timing['unrelated_task']=dict(cases=400,identical_results_with_head=True,identical_to_previous_binary_results=True)
    # The feature-enabled path must preserve exact baseline scores across all 400 cases.
    feature_rows=read(root/'features/test.jsonl');plain=read(root/'timing/base-0.jsonl')
    assert results(feature_rows)==results(plain)
    timing['feature_extraction']=dict(cases=400,identical_base_results=True)
    (root/'timings.json').write_text(json.dumps(timing,indent=2)+'\n');print(json.dumps(timing,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();summarize(a.root,a.previous)
