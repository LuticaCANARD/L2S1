#!/usr/bin/env python3
"""Independent raw-prediction audit and report for full-label intent evaluation."""
import argparse
import collections
import json
import math
from pathlib import Path
import re

from evaluate_intents import load, rows, save, sha, quantile


def audit(root, out):
    tasks = {r['id']: r for r in rows(root/'prepared/gold.jsonl')}
    labels = load(root/'prepared/datasets.json')
    manifest = load(out/'manifest.json')
    assert manifest['requests_sha256'] == sha(root/'prepared/requests.jsonl')
    assert manifest['prepared_manifest_sha256'] == sha(root/'prepared/manifest.json')
    predictions = rows(out/'predictions.jsonl')
    assert len(predictions)==len({p['id'] for p in predictions})==400
    assert {p['id'] for p in predictions}==set(tasks)
    measured=collections.defaultdict(list)
    for p in predictions:
        assert not p.get('error')
        t=tasks[p['id']]; r=p['response']['results'][0]
        assert not r['truncated'] and r['scoring_method']=='code_sequence_conditional_softmax_v1'
        assert [s['id'] for s in r['scores']]==labels[t['dataset']]['labels']
        probabilities={s['id']:s['option_probability'] for s in r['scores']}
        assert all(math.isfinite(v) and 0<=v<=1 for v in probabilities.values())
        assert abs(sum(probabilities.values())-1)<1e-9
        pick=sorted(probabilities,key=lambda k:(-probabilities[k],k))[0]
        confidence=probabilities[pick]
        tied=sum(abs(v-confidence)<1e-12 for v in probabilities.values())>1
        accepted=confidence>=0.8 and r['candidate_mass']>=0.05 and not tied
        assert r['value']['selected']==(pick if accepted else None)
        measured[t['dataset']].append(dict(correct=pick==t['expected'],accepted=accepted,confidence=confidence,
            brier=sum((v-(k==t['expected']))**2 for k,v in probabilities.items()),latency=p['elapsed_ms']))
    summary=load(out/'summary.json')
    for name,records in measured.items():
        n=len(records); assert n==200
        s=summary[name]
        assert s['correct']==sum(r['correct'] for r in records)
        assert s['accepted']==sum(r['accepted'] for r in records)
        assert s['accepted_correct']==sum(r['accepted'] and r['correct'] for r in records)
        bins=collections.defaultdict(list)
        for r in records: bins[min(int(r['confidence']*10),9)].append(r)
        ece=sum(abs(sum(r['confidence']-r['correct'] for r in group)) for group in bins.values())/n
        assert abs(s['ece']-ece)<1e-10
        assert abs(s['brier']-sum(r['brier'] for r in records)/n)<1e-10
        for key,q in [('p50_ms',.5),('p95_ms',.95)]:
            assert abs(s[key]-quantile([r['latency'] for r in records],q))<1e-9
    return dict(id=out.name,manifest=manifest,datasets=summary)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    p.add_argument('--grouped',type=Path,required=True); p.add_argument('--legacy',type=Path,required=True)
    args=p.parse_args(); root=args.root
    for name,h in load(root/'prepared/manifest.json')['prepared_sha256'].items():
        assert sha(root/'prepared'/name)==h
    assert sha(root/'prepared/gold.jsonl')==sha(args.grouped/'prepared/gold.jsonl')
    results=[audit(root,root/'runs'/m['id']) for m in load(root/'plan.json')]
    assert results and len({r['manifest']['evaluator_sha256'] for r in results})==1
    for result,model in zip(results,load(root/'plan.json')):
        assert result['manifest']['model']==model
    before={r['id']:r for r in rows(args.legacy/'predictions.jsonl')}
    after={r['id']:r for r in rows(root/'legacy-regression/predictions.jsonl')}
    assert len(before)==len(after)==231 and set(before)==set(after)
    delta=max(abs(a['option_probability']-b['option_probability']) for key in before
        for a,b in zip(before[key]['response']['results'][0]['scores'],after[key]['response']['results'][0]['scores']))
    result_changes=sum(before[k]['response']['results']!=after[k]['response']['results'] for k in before)
    assert delta==0 and result_changes==0
    tests={}
    for name in ['default-tests','tests','native-test','three-letter-native']:
        log=(root/(name+'.log')).read_text()
        stats=[tuple(map(int,r)) for r in re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;',log)]
        assert stats and 'FAILED' not in log
        tests[name]=dict(zip(['passed','failed','ignored'],map(sum,zip(*stats))))
    assert tests['native-test']['passed']==tests['three-letter-native']['passed']==1
    assert (root/'verification.exit-code').read_text().strip()=='0'
    verification=[l for l in (root/'three-letter-native.log').read_text().splitlines() if '677 candidates verified' in l]
    assert len(verification)==1
    report=dict(results=results,tests=tests,legacy_regression=dict(examples=231,changed_results=result_changes,max_probability_delta=delta),
        three_letter_verification=verification[0],independently_audited=True)
    save(root/'REPORT.json',report)
    text=['# Full-label BANKING77 and MASSIVE evaluation', '',
        '2026-09-23 · `100.66.64.91` · NVIDIA RTX 3060 12 GiB', '',
        f'The rebuilt engine sizes uppercase answer codes to the candidate count: A–Z, AA–ZZ, AAA–ZZZ, and onward. All 77 BANKING77 labels and all 60 MASSIVE labels are compared in one decision per example. No group tournament or candidate shortlist is used. All {len(results)*400:,} measured classifications completed without inference errors or truncation.', '',
        '| Model | Dataset | Correct / 200 | Accuracy | p50 ms | p95 ms | Accepted correct / accepted | Accepted wrong | Abstained |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        for name,s in r['datasets'].items():
            text.append(f"| {r['id']} | {name} | {s['correct']} | {s['accuracy']:.1%} | {s['p50_ms']:.2f} | {s['p95_ms']:.2f} | {s['accepted_correct']}/{s['accepted']} | {s['accepted_wrong']} | {s['abstained']} |")
    text+=['','## Comparison with the earlier grouped baseline','',
        'The same sampled utterances, gold labels and model checkpoints were used. The old baseline used three groups plus a final vote (four inference calls per example); the new path includes the full label set in a single decision. Changes include the prompt and code mapping, so the difference is not an isolated estimate of routing loss.', '',
        '| Model | Dataset | Grouped correct | Full-label correct | Grouped p50 ms | Full-label p50 ms |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for r in results:
        if not (args.grouped/'runs'/r['id']/'summary.json').exists():
            continue
        old=load(args.grouped/'runs'/r['id']/'summary.json')
        for name,s in r['datasets'].items():
            text.append(f"| {r['id']} | {name} | {old[name]['correct']}/200 | {s['correct']}/200 | {old[name]['p50_ms']:.2f} | {s['p50_ms']:.2f} |")
    text+=['','## Correctness evidence','']
    for name,s in tests.items(): text.append(f"- {name}: {s['passed']} passed, {s['failed']} failed, {s['ignored']} ignored.")
    text+=['- '+verification[0],
        '- A complete 231-item Gemma E2B JevBench replay preserved every existing A–Z result and probability exactly: zero changed result objects and maximum probability delta 0.',
        '- A separate report implementation independently recounted accuracy, acceptance, Brier, ECE and latency from raw predictions. Candidate/request/model hashes and complete ID coverage were checked.', '',
        '## Data, settings and limits','',
        '- [BANKING77 / PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets), CC BY 4.0: 200 English test rows sampled from 3,080; all 77 labels remain candidates.',
        '- [MASSIVE / Amazon](https://huggingface.co/datasets/AmazonScience/massive), CC BY 4.0: 200 `ko-KR` test rows sampled from 2,974; all 60 labels remain candidates. Korean Parquet revision: `6e31162aba58a715666d3791566f42afdcfa62b2`.',
        '- Seed `20260923`, uniform sampling without replacement independently per dataset, fixed before inference. The English sample contains 73 gold classes; the Korean sample contains 50. No training, retrieval, adapters, calibration fitting or prompt tuning on these samples.',
        '- State contains only the raw utterance. Instructions are English for BANKING77 and Korean for MASSIVE; option criteria use official English label names with underscores replaced by spaces. Scenario, gold intent, annotated utterance and human judgments are never inference inputs.',
        '- CUDA; context 8192; batch/microbatch 256; four threads; fresh execution; legacy/minimal layout; full evidence; FlashAttention off; one warmup per model. Whole-code token paths are validated at the actual assistant boundary.',
        '- In these Gemma checkpoints, all AA–CY and AA–CH candidate codes are single tokens, so each benchmark decision needs one native prefix evaluation. The separate 677-candidate GPU test exercises genuinely multi-token three-letter codes.',
        '- Accuracy uses argmax before abstention. Acceptance retains the native 0.8 top-probability and 0.05 full-vocabulary mass thresholds, with ties rejected. High confidence is not treated as calibrated correctness; accepted errors are reported explicitly.',
        '- Latency measures a local Rust decision call, including preparation and complete-code scoring. Model loading, warmup and network transport are excluded. Single-run timing is descriptive.',
        '- These are 200-row test-subset measurements, not complete-dataset leaderboard scores. Wilson 95% intervals, Brier and ECE are available in REPORT.json. Pretraining contamination was not audited.',
        '- Candidate width has no alphabet-derived ceiling. Real requests remain limited by context, memory and tokenizer compatibility. Multi-letter scoring currently requires fresh/full mode without output heads, scalar calibration or feature export.', '',
        '## Artifacts','',
        '- `prepared/`: identical frozen gold/sample files, complete-label requests and source hashes.',
        '- `runs/`: model and evaluator hashes, commands, raw predictions, scored records and summaries.',
        '- `source/`, `source-manifest.json`, `source.tar.gz`: frozen evaluated runtime source. `tests/answer_codes_native.rs` was added afterward for supplementary GPU validation without changing runtime source.',
        '- Build/test logs, native 677-candidate proof, and `legacy-regression/`: compatibility replay evidence.',
        '- Evaluator SHA256: `'+results[0]['manifest']['evaluator_sha256']+'`.', '']
    (root/'REPORT.md').write_text('\n'.join(text))
    print(json.dumps({r['id']:{name:s['correct'] for name,s in r['datasets'].items()} for r in results}))


if __name__=='__main__': main()
