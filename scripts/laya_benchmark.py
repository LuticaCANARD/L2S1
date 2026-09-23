#!/usr/bin/env python3
"""Prepare and evaluate the public Luni/Laya tasks with L2S1 (no Laya dependency)."""
import argparse
import ast
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import urllib.request

SOURCES = {
    'benchmark': ('Luni/laya-jev-benchmark', 'd75081b2a4b2ad772793d6a7f5f5b4fdca00d557'),
    'typed': ('LocalLLaMA/typed-decisions', 'c76749ec58bd8c3d2ea706b31c333a9059c38f90'),
    'phish': ('AreLit/PhishNChips', '89afcc39610084298c4679159cb2e27d9ffffa46'),
}
FILES = {
    'probes': [('benchmark', 'bench/probe.py')],
    'typed': [('typed', 'all/test-00000-of-00001.parquet'), ('benchmark', 'bench/eval.py')],
    'phish': [('phish', 'core_emails.csv'), ('benchmark', 'bench/bench_phish.py')],
}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_rows(path, values):
    with Path(path).open('x') as f:
        for value in values:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')


def fetch(suite, output):
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for source, name in FILES[suite]:
        repo, revision = SOURCES[source]
        url = f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}'
        dest = output / Path(name).name
        with urllib.request.urlopen(url, timeout=120) as response, dest.open('xb') as f:
            shutil.copyfileobj(response, f)
        files.append(dict(repo=repo, revision=revision, path=name, local=dest.name,
                          url=url, sha256=sha(dest)))
    save(output / 'sources.json', dict(suite=suite, files=files))


def decoded(value):
    return json.loads(value) if isinstance(value, str) else value


def nonempty(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Expected nonempty text')
    return value


def decision(qid, question):
    """Allowlist question fields; keep option insertion order and original rubric."""
    qid = nonempty(qid)
    instruction = nonempty(question['instructions'])
    criteria = question.get('criteria')
    if question['type'] == 'noul':
        if criteria is not None and (not isinstance(criteria, dict) or set(criteria) != {'false', 'true'}):
            raise ValueError('Invalid binary criteria')
        criteria = criteria or {'false': 'No', 'true': 'Yes'}
        kind = dict(type='binary', false_label=nonempty(criteria['false']),
                    true_label=nonempty(criteria['true']))
    elif question['type'] == 'choice':
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError('Invalid choice criteria')
        kind = dict(type='choice', options=[dict(id=nonempty(k), criterion=nonempty(v))
                                           for k, v in criteria.items()])
    elif question['type'] == 'score':
        if not isinstance(criteria, list) or len(criteria) < 2:
            raise ValueError('Invalid ordinal criteria')
        kind = dict(type='ordinal', levels=[dict(id=str(i), criterion=nonempty(v), value=i)
                                            for i, v in enumerate(criteria)])
    else:
        raise ValueError('Unknown question type')
    return dict(id=qid, instruction=instruction, kind=kind)


def options(d):
    kind = d['kind']
    if kind['type'] == 'binary':
        return ['false', 'true']
    return [x['id'] for x in kind.get('options', kind.get('levels'))]


def typed_cases(rows):
    cases = []
    for i, row in enumerate(rows):
        state, questions, gold = (decoded(row[k]) for k in ('state', 'questions', 'gold'))
        if not questions or set(questions) != set(gold):
            raise ValueError('Question/gold IDs differ')
        cases.append(dict(id=f'typed:{i}', request=dict(state=state, decisions=[
            decision(k, q) for k, q in questions.items()]), gold=gold,
            workflow=row.get('workflow', 'unknown'), suite='typed'))
    return cases


def phish_cases(rows):
    cases = []
    q = decision('is_phishing', dict(type='noul', instructions='Is this email a phishing or scam attempt?'))
    for i, row in enumerate(rows):
        state = row['email_content']
        try:
            state = json.loads(state)
        except (ValueError, TypeError):
            pass
        y = int(row['phish_label'])
        if y not in (0, 1):
            raise ValueError('Invalid phishing label')
        cases.append(dict(id=f'phish:{i}', request=dict(state=state, decisions=[q]),
                          gold={'is_phishing': {'label': 'true' if y else 'false', 'noul': y}},
                          workflow='phishing', suite='phish'))
    return cases


def probe_cases(path):
    """Read only literal probe definitions. Never import/execute downloaded code."""
    definitions = {}

    def literal(node):
        if isinstance(node, ast.Name) and node.id in definitions:
            return definitions[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [literal(x) for x in node.elts]
        if isinstance(node, ast.Dict):
            return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
        return ast.literal_eval(node)

    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in {'TICKET', 'PHISH', 'CONTRADICTIONS', 'STABILITY', 'GROUNDING'}:
                definitions[name] = literal(node.value)
    cases = []
    for i, (state, a, b) in enumerate(definitions['CONTRADICTIONS']):
        cases.append(dict(id=f'contradiction:{i}', request=dict(state=state, decisions=[
            decision(k, dict(type='noul', instructions=q)) for k, q in [('a', a), ('b', b)]]),
            gold={}, probe='contradiction', suite='probes'))
    for i, (state, _, variants) in enumerate(definitions['STABILITY']):
        cases.append(dict(id=f'stability:{i}', request=dict(state=state, decisions=[
            decision(str(j), dict(type='choice', instructions=q, criteria=c))
            for j, (q, c, _) in enumerate(variants)]),
            gold={str(j): {'label': expected} for j, (_, _, expected) in enumerate(variants)},
            probe='stability', suite='probes'))
    for i, (state, q, expected) in enumerate(definitions['GROUNDING']):
        cases.append(dict(id=f'grounding:{i}', request=dict(state=state, decisions=[
            decision('q', dict(type='noul', instructions=q))]),
            gold={'q': {'label': 'true' if expected else 'false', 'noul': int(expected)}},
            probe='grounding', suite='probes'))
    if len(cases) != 9 or sum(len(c['request']['decisions']) for c in cases) != 14:
        raise ValueError('Pinned probe definitions changed')
    return cases


def prepare(suite, source, output, limit=0, repeats=1):
    provenance = load(source / 'sources.json')
    if provenance['suite'] != suite:
        raise ValueError('Source suite mismatch')
    for item in provenance['files']:
        if sha(source / item['local']) != item['sha256']:
            raise ValueError('Source hash mismatch')
    if suite == 'probes':
        cases = probe_cases(source / 'probe.py')
    elif suite == 'typed':
        import pyarrow.parquet as pq  # Only data preparation needs this optional dependency.
        cases = typed_cases(pq.read_table(source / 'test-00000-of-00001.parquet').to_pylist())
        if len(cases) != 400 or sum(len(c['request']['decisions']) for c in cases) != 2000:
            raise ValueError('Pinned typed-decisions test split changed')
    else:
        csv.field_size_limit(16 * 1024 * 1024)
        with (source / 'core_emails.csv').open(newline='') as f:
            cases = phish_cases(csv.DictReader(f))
        if len(cases) != 2000 or sum(c['gold']['is_phishing']['noul'] for c in cases) != 1000:
            raise ValueError('Pinned balanced phishing core split changed')
    if limit < 0 or repeats < 1:
        raise ValueError('Limit must be nonnegative and repeats positive')
    total_cases = len(cases)
    cases = cases[:limit] if limit else cases
    expanded = []
    for case in cases:
        for repeat in range(repeats):
            expanded.append({**case, 'id': f"{case['id']}/r{repeat}", 'base_id': case['id'], 'repeat': repeat})
    output.mkdir(parents=True, exist_ok=False)
    write_rows(output / 'requests.jsonl', [dict(id=c['id'], request=c['request']) for c in expanded])
    write_rows(output / 'cases-with-gold.jsonl', expanded)
    save(output / 'manifest.json', dict(suite=suite, sources=provenance, available_cases=total_cases,
        logical_cases=len(cases), repeats=repeats, measured_cases=len(expanded),
        decisions=sum(len(c['request']['decisions']) for c in cases), limited=bool(limit and limit < total_cases),
        files={f: sha(output / f) for f in ['requests.jsonl', 'cases-with-gold.jsonl']},
        adapter_sha256=sha(Path(__file__)), ordering='source order, immediate repeats per case',
        protocol='Only id/state/instruction/criteria enter inference. No gold, rationale or workflow metadata. '
                 'Repeated identical cases measure preparation caching, not a persistent KV or answer cache. '
                 'Probe grounding outputs also supply overconfidence checks; no second inference for the same check.'))


def quantile(values, q):
    if not values:
        return None
    a = sorted(values); position = (len(a) - 1) * q; i = int(position)
    return a[i] + (a[min(i + 1, len(a) - 1)] - a[i]) * (position - i)


def mean(values):
    return sum(values) / len(values) if values else None


def auc(labels, probabilities):
    """Pairwise-equivalent rank AUROC, including half credit for tied scores."""
    pairs = sorted(zip(probabilities, labels)); n1 = sum(labels); n0 = len(labels) - n1
    if not n0 or not n1:
        return None
    rank_sum = 0.; i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(y for _, y in pairs[i:j]); i = j
    return (rank_sum - n1 * (n1 + 1) / 2) / (n0 * n1)


def distribution(d, result):
    scores = result['scores']; labels = options(d)
    if result['id'] != d['id'] or result['truncated'] or [s['id'] for s in scores] != labels:
        raise ValueError('Mismatched labels/decision or truncated input')
    p = [s['option_probability'] for s in scores]
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p) - 1) > 1e-6:
        raise ValueError('Invalid probability distribution')
    count, reused = result['input_tokens'], result['reused_prefix_tokens']
    if not isinstance(count, int) or not isinstance(reused, int) or not 0 <= reused < count:
        raise ValueError('Invalid token accounting')
    for key in ['candidate_mass', 'entropy_confidence']:
        if not math.isfinite(result[key]) or not 0 <= result[key] <= 1 + 1e-6:
            raise ValueError('Invalid evidence: ' + key)
    return dict(zip(labels, p))


def score_decision(d, result, gold):
    p = distribution(d, result); labels = list(p)
    # Preserve upstream binary threshold (true at exactly .5), ordered argmax otherwise.
    selected = ('true' if p['true'] >= .5 else 'false') if d['kind']['type'] == 'binary' else max(p, key=p.get)
    value = result['value']; accepted = value.get('value') is not None if value['type'] == 'binary' else value.get('selected') is not None
    if value['type'] != d['kind']['type']:
        raise ValueError('Typed value does not match the decision kind')
    if value['type'] == 'binary':
        if not math.isfinite(value['p_true']) or abs(value['p_true'] - p['true']) > 1e-6 or value['value'] not in (None, True, False):
            raise ValueError('Invalid binary value')
    elif value['selected'] is not None and value['selected'] not in p:
        raise ValueError('Selected label outside candidate set')
    row = dict(id=d['id'], probabilities=p, predicted=selected, confidence=max(p.values()),
               entropy_confidence=result['entropy_confidence'], candidate_mass=result['candidate_mass'],
               accepted=accepted, input_tokens=result['input_tokens'], reused_prefix_tokens=result['reused_prefix_tokens'])
    if gold is None:
        return row
    expected = str(gold['label']).lower() if d['kind']['type'] == 'binary' else str(gold['label'])
    if expected not in p:
        raise ValueError('Gold label outside candidate set')
    row.update(expected=expected, correct=selected == expected)
    gp = gold.get('probabilities')
    if d['kind']['type'] == 'binary':
        v = gold.get('noul', (gp or {}).get('true'))
        if v is not None:
            gp = {'false': 1 - v, 'true': v}
    row['brier_hard'] = sum((p[k] - (k == expected)) ** 2 for k in labels)
    if gp is not None and d['kind']['type'] != 'ordinal':
        if set(gp) != set(labels) or any(not math.isfinite(v) or v < 0 for v in gp.values()) or abs(sum(gp.values()) - 1) > 1e-4:
            raise ValueError('Invalid gold distribution')
        # Public soft targets are rounded to six decimals; normalize as upstream does.
        total = sum(gp.values()); gp = {k: v / total for k, v in gp.items()}
        row.update(soft_accuracy=sum(p[k] * gp[k] for k in labels),
                   brier_soft=sum((p[k] - gp[k]) ** 2 for k in labels),
                   tvd=sum(abs(p[k] - gp[k]) for k in labels) / 2,
                   kl=sum(gp[k] * math.log(max(1e-12, min(1e4, gp[k] / max(p[k], 1e-300)))) for k in labels if gp[k]))
    if d['kind']['type'] == 'ordinal':
        expected_value = sum(float(k) * p[k] for k in labels)
        if not math.isfinite(value['expected_value']) or abs(value['expected_value'] - expected_value) > 1e-6:
            raise ValueError('Ordinal expectation differs from probabilities')
        row['score_mae'] = abs(expected_value - gold['score'])
        row['within_one'] = row['score_mae'] <= 1
    return row


def metrics(rows, attempted):
    labeled = [r for r in rows if 'correct' in r]
    bins = collections.defaultdict(list)
    for r in labeled:
        bins[min(int(r['confidence'] * 10), 9)].append(r)
    accepted = [r for r in labeled if r['accepted']]
    m = dict(attempted=attempted, valid=len(labeled), correct=sum(r['correct'] for r in labeled),
             accuracy=sum(r['correct'] for r in labeled) / attempted if attempted else None,
             valid_accuracy=mean([r['correct'] for r in labeled]),
             ece_hard=sum(abs(sum(r['confidence'] - r['correct'] for r in group)) for group in bins.values()) / len(labeled) if labeled else None,
             accepted=len(accepted), accepted_accuracy=mean([r['correct'] for r in accepted]),
             wrong_accepted=sum(not r['correct'] for r in accepted),
             coverage=len(accepted) / attempted if attempted else None)
    for key in ['soft_accuracy', 'brier_soft', 'brier_hard', 'kl', 'tvd', 'score_mae', 'within_one']:
        m[key] = mean([r[key] for r in labeled if key in r])
    return m


def score(prepared, predictions):
    manifest = load(prepared / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha(prepared / name) != expected:
            raise ValueError('Prepared data hash mismatch')
    cases = read_rows(prepared / 'cases-with-gold.jsonl'); raw = read_rows(predictions)
    expected_ids = {c['id'] for c in cases}; by_id = {}
    for p in raw:
        if p['id'] in by_id or p['id'] not in expected_ids:
            raise ValueError('Duplicate or unknown prediction ID')
        by_id[p['id']] = p
    details, errors, checks, batches = [], [], [], {}
    valid_case_ids = set()
    for c in cases:
        p = by_id.get(c['id'])
        if p is None or p.get('error'):
            errors.append(dict(id=c['id'], error=p.get('error') if p else 'Missing prediction'))
            continue
        try:
            if not math.isfinite(p['elapsed_ms']) or p['elapsed_ms'] < 0:
                raise ValueError('Invalid latency')
            result = p['response']['results']; decisions = c['request']['decisions']
            if len(result) != len(decisions):
                raise ValueError('Decision count mismatch')
            scored = [score_decision(d, r, c['gold'].get(d['id'])) for d, r in zip(decisions, result)]
            batch = {k: p[k] for k in ['batch_size', 'batch_elapsed_ms', 'batch_profile',
                                      'preparation_cache_before', 'preparation_cache_after']}
            if p['batch_index'] in batches and batches[p['batch_index']] != batch:
                raise ValueError('Inconsistent batch metadata')
            batches[p['batch_index']] = batch
        except (KeyError, TypeError, ValueError) as error:
            errors.append(dict(id=c['id'], error=str(error))); continue
        valid_case_ids.add(c['id'])
        for r in scored:
            details.append({**r, 'case_id': c['id'], 'base_id': c['base_id'], 'repeat': c['repeat'],
                            'workflow': c.get('workflow'), 'case_elapsed_ms': p['elapsed_ms']})
        probe = c.get('probe')
        if probe == 'contradiction':
            total = sum(r['probabilities']['true'] for r in scored)
            checks.append(dict(id=c['id'], kind=probe, failed=abs(total - 1) > .35, sum_probability=total, repeat=c['repeat']))
        elif probe == 'stability':
            checks.append(dict(id=c['id'], kind=probe, failed=not all(r['correct'] for r in scored), repeat=c['repeat']))
        elif probe == 'grounding':
            r = scored[0]
            checks.extend([dict(id=c['id'], kind='grounding', failed=not r['correct'], repeat=c['repeat']),
                           dict(id=c['id'], kind='overconfidence', failed=not r['correct'] and r['entropy_confidence'] >= .8, repeat=c['repeat'])])
    summaries = {}
    for repeat in range(manifest['repeats']):
        chosen = [r for r in details if r['repeat'] == repeat]
        subset = [c for c in cases if c['repeat'] == repeat]
        planned = sum(len(c['gold']) for c in subset)
        m = metrics(chosen, planned)
        m['case_errors'] = sum(e['id'] in {c['id'] for c in subset} for e in errors)
        latencies = [by_id[c['id']]['elapsed_ms'] for c in subset if c['id'] in valid_case_ids]
        m['case_latency_ms'] = dict(p50=quantile(latencies, .5), p95=quantile(latencies, .95))
        m['by_workflow'] = {wf: metrics([r for r in chosen if r['workflow'] == wf],
            sum(len(c['gold']) for c in subset if c.get('workflow') == wf)) for wf in sorted({c['workflow'] for c in subset if 'workflow' in c})}
        if manifest['suite'] == 'phish':
            y = [int(r['expected'] == 'true') for r in chosen]; probs = [r['probabilities']['true'] for r in chosen]
            m.update(auroc=auc(y, probs), recall=mean([r['predicted'] == 'true' for r in chosen if r['expected'] == 'true']),
                     precision=mean([r['expected'] == 'true' for r in chosen if r['predicted'] == 'true']))
        m['probe_checks'] = [x for x in checks if x['repeat'] == repeat]
        m['probe_failures'] = sum(x['failed'] for x in m['probe_checks'])
        summaries[str(repeat)] = m
    cache = {}
    for kind in ['prompts', 'candidates']:
        cache[kind] = {field: sum(b['preparation_cache_after'][kind][field] - b['preparation_cache_before'][kind][field]
                                  for b in batches.values()) for field in ['hits', 'misses', 'insertions', 'evictions', 'skipped']}
    summary = dict(suite=manifest['suite'], limited=manifest['limited'], logical_cases=manifest['logical_cases'],
        decisions=manifest['decisions'], per_repeat=summaries, errors=errors, cache=cache,
        profile={key: sum(b['batch_profile'][key] for b in batches.values()) for key in ['prepare_ms', 'native_ms', 'score_ms']},
        unique_batch_elapsed_ms=sum(b['batch_elapsed_ms'] for b in batches.values()),
        logical_input_tokens=sum(r['input_tokens'] for r in details),
        reused_prefix_tokens=sum(r['reused_prefix_tokens'] for r in details),
        evaluated_input_tokens=sum(r['input_tokens'] - r['reused_prefix_tokens'] for r in details),
        scope='Local L2S1 inference, no official JevBench composite score. Per-case latency is full batch completion, '
              'not divided by questions/batch width. Repeat 0 alone is the dataset quality result; later repeats are cache diagnostics. '
              'Hard ECE uses ten equal-width bins against argmax labels, not soft-target calibration. '
              'Soft metrics cover binary/choice only, as upstream; ordinal MAE uses the probability-weighted level. '
              'AUROC uses tie-aware ranks. No Platt fitting or evaluation-set training. '
              'Probe thresholds are upstream heuristics, not logical guarantees; stability also changes rubric wording. '
              'Overconfidence uses L2S1 entropy confidence and is not assumed identical to Laya confidence.')
    summary['boundary_and_other_ms'] = max(0., summary['unique_batch_elapsed_ms'] - sum(summary['profile'].values()))
    return summary, details


def run(args):
    prepared = args.prepared.resolve(); out = args.output.resolve()
    manifest = load(prepared / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha(prepared / name) != expected:
            raise ValueError('Prepared hash mismatch')
    out.mkdir(parents=True, exist_ok=False)
    evaluator = args.evaluator.resolve(); model = args.model.resolve()
    command = [str(evaluator), '--model', str(model), '--input', str(prepared / 'requests.jsonl'),
        '--output', str(out / 'predictions.jsonl'), '--context', str(args.context), '--batch', str(args.batch),
        '--threads', str(args.threads), '--execution-mode', args.execution_mode, '--prompt-layout', args.prompt_layout,
        '--parallel-width', str(args.parallel_width), '--request-batch-size', str(args.request_batch_size),
        '--model-load-mode', args.model_load_mode, '--preparation-cache-bytes', str(args.cache_bytes),
        '--preparation-cache-entries', str(args.cache_entries), '--warmup']
    if args.cuda:
        command.append('--cuda')
    if args.gpu_layers is not None:
        command += ['--gpu-layers', str(args.gpu_layers)]
    if args.cpu_moe_layers:
        command += ['--cpu-moe-layers', str(args.cpu_moe_layers)]
    save(out / 'run.json', dict(command=command, evaluator_sha256=sha(evaluator), model_sha256=sha(model),
        prepared_manifest_sha256=sha(prepared / 'manifest.json'), adapter_sha256=sha(Path(__file__)),
        timing='One resident model, untimed warmup then preparation-cache clear; load excluded. No answer cache.'))
    with (out / 'inference.log').open('x') as log:
        try:
            p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = dict(exit_code=p.returncode, timeout=False)
        except subprocess.TimeoutExpired:
            status = dict(exit_code=None, timeout=True)
    save(out / 'exit.json', status)
    if not (out / 'predictions.jsonl').exists():
        raise RuntimeError('Evaluator failed before creating predictions; inspect inference.log')
    summary, details = score(prepared, out / 'predictions.jsonl')
    save(out / 'summary.json', summary); write_rows(out / 'scored.jsonl', details)
    if status['exit_code'] != 0 or summary['errors']:
        raise RuntimeError('Incomplete evaluation; failures retained in summary.json')
    print(json.dumps(summary, indent=2))


def compare(prepared, baseline, candidate):
    """Compare execution changes only, never silently compare different prompts."""
    a, ar = score(prepared, baseline / 'predictions.jsonl')
    b, br = score(prepared, candidate / 'predictions.jsonl')
    if a['errors'] or b['errors']:
        raise ValueError('Cannot certify an incomplete run')
    ma, mb = load(baseline / 'run.json'), load(candidate / 'run.json')
    for key in ['model_sha256', 'evaluator_sha256', 'prepared_manifest_sha256']:
        if ma[key] != mb[key]:
            raise ValueError('Comparison identity mismatch: ' + key)
    pa = read_rows(baseline / 'predictions.jsonl'); pb = read_rows(candidate / 'predictions.jsonl')
    if [p['id'] for p in pa] != [p['id'] for p in pb]:
        raise ValueError('Run order differs')
    maximum_mass = 0.; accepted_changes = 0
    for x, y in zip(pa, pb):
        for key in ['prompt_layout', 'prompt_version', 'prompt_profile', 'compute', 'lora_path', 'output_head_path']:
            if x['response']['backend'].get(key) != y['response']['backend'].get(key):
                raise ValueError('Not an execution-only comparison: ' + key)
        if x['response']['policy'] != y['response']['policy']:
            raise ValueError('Policy differs')
        for rx, ry in zip(x['response']['results'], y['response']['results']):
            vx, vy = rx['value'], ry['value']
            accepted_changes += vx.get('selected', vx.get('value')) != vy.get('selected', vy.get('value'))
            maximum_mass = max(maximum_mass, abs(rx['candidate_mass'] - ry['candidate_mass']))
    probability_delta = max(abs(x['probabilities'][k] - y['probabilities'][k])
                            for x, y in zip(ar, br) for k in x['probabilities'])
    flips = sum(x['predicted'] != y['predicted'] for x, y in zip(ar, br))
    return dict(decisions_compared=len(ar), max_probability_delta=probability_delta,
        max_candidate_mass_delta=maximum_mass, raw_top1_changes=flips, accepted_selection_changes=accepted_changes,
        exact_probabilities=probability_delta == maximum_mass == 0,
        within_existing_tolerance=probability_delta <= .02 and maximum_mass <= .02 and flips == accepted_changes == 0,
        total_inference_speedup=a['unique_batch_elapsed_ms'] / b['unique_batch_elapsed_ms'],
        baseline_profile=a['profile'], candidate_profile=b['profile'],
        candidate_cache=b['cache'], reused_tokens=b['reused_prefix_tokens'],
        scope='Same-model, same-evaluator, same-input, same-layout/compute/policy execution comparison; '
              'one local run with immediate repeats, not a service latency guarantee.')


def main():
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='mode', required=True)
    f = sub.add_parser('fetch'); f.add_argument('--suite', choices=FILES, required=True); f.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('prepare'); a.add_argument('--suite', choices=FILES, required=True)
    a.add_argument('--source', type=Path, required=True); a.add_argument('--output', type=Path, required=True)
    a.add_argument('--limit', type=int, default=0); a.add_argument('--repeats', type=int, default=1)
    s = sub.add_parser('score'); s.add_argument('--prepared', type=Path, required=True)
    s.add_argument('--predictions', type=Path, required=True); s.add_argument('--output', type=Path, required=True)
    c = sub.add_parser('compare'); c.add_argument('--prepared', type=Path, required=True)
    c.add_argument('--baseline', type=Path, required=True); c.add_argument('--candidate', type=Path, required=True)
    c.add_argument('--output', type=Path, required=True)
    r = sub.add_parser('run'); r.add_argument('--prepared', type=Path, required=True); r.add_argument('--output', type=Path, required=True)
    r.add_argument('--evaluator', type=Path, required=True); r.add_argument('--model', type=Path, required=True)
    r.add_argument('--cuda', action='store_true'); r.add_argument('--gpu-layers', type=int); r.add_argument('--cpu-moe-layers', type=int, default=0)
    r.add_argument('--context', type=int, default=8192); r.add_argument('--batch', type=int, default=256); r.add_argument('--threads', type=int, default=8)
    r.add_argument('--model-load-mode', choices=['auto', 'read'], default='auto')
    r.add_argument('--execution-mode', choices=['fresh', 'prefix-reuse', 'parallel', 'state-restore'], default='fresh')
    r.add_argument('--prompt-layout', choices=['legacy', 'state-first'], default='legacy')
    r.add_argument('--parallel-width', type=int, default=4); r.add_argument('--request-batch-size', type=int, default=1)
    r.add_argument('--cache-bytes', type=int, default=0); r.add_argument('--cache-entries', type=int, default=128)
    r.add_argument('--timeout', type=float, default=14400)
    args = p.parse_args()
    if args.mode == 'fetch':
        fetch(args.suite, args.output)
    elif args.mode == 'prepare':
        prepare(args.suite, args.source, args.output, args.limit, args.repeats)
    elif args.mode == 'run':
        run(args)
    elif args.mode == 'compare':
        report = compare(args.prepared, args.baseline, args.candidate)
        save(args.output, report); print(json.dumps(report, indent=2))
    else:
        summary, _ = score(args.prepared, args.predictions); save(args.output, summary); print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
