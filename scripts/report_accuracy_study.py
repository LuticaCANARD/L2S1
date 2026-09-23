#!/usr/bin/env python3
"""Audit frozen accuracy-study predictions; select configurations on dev only."""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path
import statistics

from prepare_accuracy_study import json_bytes, sha256, validate_dataset

DETAILS = {'minimal', 'typed', 'typed_examples'}
LAYOUTS = {'legacy', 'state_first'}
TIE_TOLERANCE = 1e-12
NLL_FLOOR = 1e-15


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite prediction value')
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def read_json(line):
    def invalid_constant(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    value = json.loads(line, parse_constant=invalid_constant)
    finite_tree(value)
    return value


def finite_number(value, name, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric')
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f'{name} outside valid range')
    return value


def option_ids(decision):
    kind = decision['kind']
    if kind['type'] == 'binary':
        return ['false', 'true']
    return [option['id'] for option in kind.get('options', kind.get('levels', []))]


def argmax_unique(scores):
    maximum = max(scores.values())
    winners = [label for label, score in scores.items() if abs(score - maximum) <= TIE_TOLERANCE]
    return winners[0] if len(winners) == 1 else None


def config_key(row):
    setting = row.get('setting', {})
    detail, layout = setting.get('prompt_detail'), setting.get('prompt_layout')
    if detail not in DETAILS or layout not in LAYOUTS:
        raise ValueError('unknown prompt detail/layout')
    kind = row.get('kind')
    rotation = row.get('rotation') if kind == 'single' else None
    if kind not in ('single', 'ensemble'):
        raise ValueError('unknown prediction kind')
    if kind == 'single' and (isinstance(rotation, bool) or not isinstance(rotation, int) or rotation < 0):
        raise ValueError('single predictions require a nonnegative integer rotation')
    return detail, layout, kind, rotation


def describe(key):
    detail, layout, kind, rotation = key
    return dict(setting=dict(prompt_detail=detail, prompt_layout=layout), kind=kind, rotation=rotation)


def identity_digest(row, count):
    """Require one stable model/configuration, allowing only ensemble code rotation."""
    if row['kind'] == 'single':
        identities = [row.get('model_identity')]
        rotations = [row['rotation']]
    else:
        identities = row.get('model_identities')
        rotations = row.get('rotations')
        if rotations != list(range(count)) or not isinstance(identities, list) or len(identities) != count:
            raise ValueError('ensemble must contain every distinct code rotation exactly once')
    normalized = []
    for identity, rotation in zip(identities, rotations):
        if not isinstance(identity, dict) or not identity:
            raise ValueError('missing model identity')
        detail = {'minimal': 'Minimal', 'typed': 'Typed', 'typed_examples': 'TypedExamples'}[row['setting']['prompt_detail']]
        version = identity.get('prompt_version')
        if not isinstance(version, str) or not version:
            raise ValueError('missing model prompt version')
        match = re.fullmatch(r'(.+)/detail-(Minimal|Typed|TypedExamples)-v1/rotation-(0|[1-9][0-9]*)', version)
        if match:
            base, actual_detail, actual_rotation = match.groups()
            if actual_detail != detail or int(actual_rotation) != rotation:
                raise ValueError('model identity prompt detail/rotation mismatch')
        else:
            if detail != 'Minimal' or rotation != 0:
                raise ValueError('only minimal rotation zero may omit model prompt suffix')
            base = version
        if '/detail-' in base or '/rotation-' in base:
            raise ValueError('malformed model prompt identity suffix')
        normalized.append(dict(identity, prompt_version=f'{base}/detail-{detail}-v1'))
    if any(identity != normalized[0] for identity in normalized):
        raise ValueError('ensemble changes model/configuration between rotations')
    return hashlib.sha256(json_bytes(normalized[0])).hexdigest()


def validate_response(row, source, gold):
    decision = source['request']['decisions'][0]
    semantic_ids = option_ids(decision)
    result = dict(id=row['id'], expected=gold, decision_kind=decision['kind']['type'],
                  elapsed_ms=finite_number(row.get('elapsed_ms'), 'elapsed_ms', 0),
                  raw_top1=None, selected=None, probabilities=None, tied=False,
                  abstention_reasons=[], runtime_error=None)
    if row.get('error') is not None:
        if 'response' in row or not row['error']:
            raise ValueError('runtime error row must have a nonempty error and no response')
        result['runtime_error'] = row['error']
        return result
    response = row.get('response', {})
    results = response.get('results', [])
    if len(results) != 1 or results[0].get('id') != decision['id']:
        raise ValueError('response must match exactly one requested decision')
    observation = results[0]
    scores = observation.get('scores', [])
    if len(scores) != len(semantic_ids) or {score.get('id') for score in scores} != set(semantic_ids):
        raise ValueError('missing, duplicate or unknown semantic option IDs')
    logits, probabilities = {}, {}
    for score in scores:
        logits[score['id']] = finite_number(score.get('raw_logit'), 'raw_logit')
        probabilities[score['id']] = finite_number(score.get('option_probability'), 'option_probability', 0, 1)
    if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6, rel_tol=0):
        raise ValueError('candidate probabilities do not sum to one')
    mass = finite_number(observation.get('candidate_mass'), 'candidate_mass', 0, 1)
    top = max(probabilities.values())
    if not math.isclose(finite_number(observation.get('top_option_probability'), 'top probability', 0, 1),
                        top, abs_tol=1e-8, rel_tol=0):
        raise ValueError('top probability does not match candidate scores')
    value = observation.get('value', {})
    if value.get('type') != decision['kind']['type']:
        raise ValueError('response decision kind mismatch')
    if value['type'] == 'binary':
        selected = value.get('value')
        if selected is not None and not isinstance(selected, bool):
            raise ValueError('binary selected value must be bool or null')
        selected = None if selected is None else ('true' if selected else 'false')
        if not math.isclose(finite_number(value.get('p_true'), 'p_true', 0, 1), probabilities['true'],
                            abs_tol=1e-8, rel_tol=0):
            raise ValueError('binary probability mapping mismatch')
    else:
        selected = value.get('selected')
        if value['type'] == 'ordinal':
            expectation = sum(probabilities[level['id']] * level['value'] for level in decision['kind']['levels'])
            if not math.isclose(finite_number(value.get('expected_value'), 'ordinal expected value'), expectation,
                                abs_tol=1e-7, rel_tol=1e-8):
                raise ValueError('ordinal expectation does not match semantic probabilities')
    if selected is not None and selected not in semantic_ids:
        raise ValueError('selected unknown semantic option ID')
    reasons = observation.get('abstention_reasons')
    if not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
        raise ValueError('invalid abstention reasons')
    policy = response.get('policy', {})
    minimum_probability = finite_number(policy.get('min_top_probability'), 'probability policy', 0, 1)
    minimum_mass = finite_number(policy.get('min_candidate_mass'), 'mass policy', 0, 1)
    if selected is not None:
        if reasons or selected != argmax_unique(probabilities) or mass < minimum_mass or top < minimum_probability:
            raise ValueError('accepted result violates its declared score/abstention policy')
    elif not reasons:
        raise ValueError('abstained result lacks a reason')
    result.update(raw_top1=argmax_unique(logits), selected=selected, probabilities=probabilities,
                  tied=argmax_unique(logits) is None, abstention_reasons=reasons)
    return result


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def metrics(rows):
    count = len(rows)
    scored = [row for row in rows if row['probabilities'] is not None]
    accepted = [row for row in rows if row['selected'] is not None]
    raw_correct = sum(row['raw_top1'] == row['expected'] for row in rows)
    accepted_correct = sum(row['selected'] == row['expected'] for row in accepted)
    nll = [-math.log(max(row['probabilities'][row['expected']], NLL_FLOOR)) for row in scored]
    brier = [sum((probability - (label == row['expected'])) ** 2
                 for label, probability in row['probabilities'].items()) for row in scored]
    latencies = [row['elapsed_ms'] for row in rows]
    return dict(count=count, raw_correct=raw_correct, raw_top1=raw_correct / count if count else None,
                ties=sum(row['tied'] for row in rows), accepted=len(accepted), accepted_correct=accepted_correct,
                accepted_wrong=len(accepted) - accepted_correct, abstained=len(scored) - len(accepted),
                not_accepted=count - len(accepted),
                coverage=len(accepted) / count if count else None,
                accepted_accuracy=accepted_correct / len(accepted) if accepted else None,
                accepted_correct_over_all=accepted_correct / count if count else None,
                runtime_errors=sum(row['runtime_error'] is not None for row in rows),
                probability_evaluated=len(scored), nll_mean=statistics.mean(nll) if nll else None,
                brier_mean=statistics.mean(brier) if brier else None,
                latency_ms=dict(total=sum(latencies), p50=statistics.median(latencies) if latencies else None,
                                p95=percentile(latencies, 0.95)))


def rotation_metrics(grouped, sources):
    by_setting = collections.defaultdict(dict)
    for key, rows in grouped.items():
        if key[2] == 'single':
            by_setting[key[:2]][key[3]] = rows
    result = []
    for setting, rotations in sorted(by_setting.items()):
        if len(rotations) < 2:
            result.append(dict(setting=dict(prompt_detail=setting[0], prompt_layout=setting[1]),
                               status='not_measured; fewer than two rotation configurations'))
            continue
        changed, deviations, failures = 0, [], 0
        for case_id, source in sources.items():
            count = len(option_ids(source['request']['decisions'][0]))
            if any(rotation not in rotations or case_id not in rotations[rotation] for rotation in range(count)):
                raise ValueError('rotation probe is missing a required case/rotation; partial probes cannot be compared')
            observations = [rotations[rotation][case_id] for rotation in range(count)]
            if any(row['runtime_error'] is not None for row in observations):
                failures += 1
                continue
            changed += len({row['raw_top1'] for row in observations}) > 1
            labels = observations[0]['probabilities']
            deviations.append(max(max(row['probabilities'][label] for row in observations) -
                                  min(row['probabilities'][label] for row in observations) for label in labels))
        result.append(dict(setting=dict(prompt_detail=setting[0], prompt_layout=setting[1]),
                           status='complete_distinct_rotations', logical_cases=len(sources),
                           compared_cases=len(deviations), runtime_error_cases=failures,
                           changed_top1_cases=changed, changed_top1_rate=changed / len(deviations) if deviations else None,
                           probability_deviation=dict(mean=statistics.mean(deviations) if deviations else None,
                                                      p95=percentile(deviations, 0.95),
                                                      maximum=max(deviations) if deviations else None)))
    return result


def build_report(data, predictions, split, variant):
    if split not in ('dev', 'calibration', 'test') or variant not in ('natural', 'symbolic'):
        raise ValueError('invalid evaluation split/variant')
    data = Path(data)
    manifest = validate_dataset(data)
    selected = manifest['splits'][split]
    source_file = data / selected['variants'][variant]['path']
    sources = {row['id']: row for row in map(read_json, source_file.read_text().splitlines())}
    labels = json.loads((data / manifest['labels']['path']).read_text())
    grouped = collections.defaultdict(dict)
    identity_by_config = {}
    policies = {}
    provenance = []
    for path in map(Path, predictions):
        provenance.append(dict(path=str(path), sha256=sha256(path)))
        for line in path.read_text().splitlines():
            if not line.strip():
                raise ValueError('blank prediction row')
            row = read_json(line)
            key = config_key(row)
            if row.get('input_sha256') != selected['variants'][variant]['requests_sha256']:
                raise ValueError('prediction input hash does not match frozen split/variant requests')
            case_id = row.get('id')
            if case_id not in sources:
                raise ValueError('prediction ID is outside the declared split')
            if case_id in grouped[key]:
                raise ValueError('duplicate logical ID within prediction configuration')
            source = sources[case_id]
            if row.get('group') != source['group']:
                raise ValueError('prediction group mismatch')
            decision = source['request']['decisions'][0]
            count = len(option_ids(decision))
            if key[2] == 'single' and key[3] >= count:
                raise ValueError('rotation outside the distinct candidate code assignments')
            identity = identity_digest(row, count)
            if key in identity_by_config and identity_by_config[key] != identity:
                raise ValueError('model identity changes within a prediction configuration')
            identity_by_config[key] = identity
            if row.get('error') is None:
                policy = row['response']['policy']
                if key in policies and policies[key] != policy:
                    raise ValueError('decision policy changes within a prediction configuration')
                policies[key] = policy
            grouped[key][case_id] = validate_response(row, source, labels[case_id][decision['id']])
    if not grouped:
        raise ValueError('no predictions')
    identities_by_setting = collections.defaultdict(set)
    for key, identity in identity_by_config.items():
        identities_by_setting[key[:2]].add(identity)
    if any(len(identities) != 1 for identities in identities_by_setting.values()):
        raise ValueError('model identity changes across rotations or ensemble')
    configurations = []
    for key, rows in sorted(grouped.items(), key=lambda pair: (pair[0][:3], -1 if pair[0][3] is None else pair[0][3])):
        required = {case_id for case_id, source in sources.items()
                    if key[2] == 'ensemble' or key[3] < len(option_ids(source['request']['decisions'][0]))}
        if set(rows) != required:
            raise ValueError(f'incomplete logical-ID coverage for configuration {key}')
        ordered = [rows[case_id] for case_id in selected['ids'] if case_id in rows]
        by_kind = {kind: metrics([row for row in ordered if row['decision_kind'] == kind])
                   for kind in sorted({row['decision_kind'] for row in ordered})}
        errors = [{name: row[name] for name in ('id', 'expected', 'raw_top1', 'selected',
                                               'abstention_reasons', 'runtime_error')}
                  for row in ordered if row['raw_top1'] != row['expected'] or row['selected'] != row['expected']]
        configurations.append(dict(describe(key), covers_full_split=set(rows) == set(sources),
                                   coverage_scope='all logical cases' if set(rows) == set(sources) else
                                   'only cases whose option_count exceeds rotation; ineligible for selection',
                                   identity_sha256=identity_by_config[key], policy=policies.get(key),
                                   metrics=metrics(ordered), by_decision_kind=by_kind, errors=errors))
    return dict(schema_version=1, manifest_sha256=sha256(data / 'manifest.json'),
                labels_sha256=manifest['labels']['sha256'], split=split, variant=variant,
                logical_cases=len(sources), source_sha256=selected['variants'][variant]['sha256'],
                predictions=provenance, configurations=configurations,
                rotation_consistency=rotation_metrics(grouped, sources),
                definitions={
                    'raw_top1': 'unique raw_logit argmax before abstention; ties within 1e-12 count as incorrect',
                    'probabilities': 'candidate-normalized option_probability; NLL/Brier exclude runtime failures and report denominator',
                    'nll_floor': NLL_FLOOR, 'brier': 'sum of squared class-probability errors, no half factor',
                    'coverage': 'accepted/all logical cases; runtime failures remain in denominator',
                    'accepted_accuracy': 'correct/accepted, null when no predictions are accepted',
                    'latency': 'compute-path inference latency per logical request, not end-to-end request latency; ensemble elapsed_ms sums constituent passes; p95 nearest rank',
                    'rotation_probe': 'each logical case compared over all of its distinct rotations; incomplete probes rejected',
                    'selection': 'dev only and full-split configurations only; held-out test must never drive selection',
                })


def select_configuration(report, report_sha256):
    if report['split'] != 'dev':
        raise ValueError('configuration selection is allowed on dev only')
    eligible = [configuration for configuration in report['configurations']
                if configuration['covers_full_split']]
    if not eligible:
        raise ValueError('no full-split configurations eligible for selection')
    def key(configuration):
        measured = configuration['metrics']
        return (-measured['raw_top1'], -measured['accepted_correct_over_all'],
                measured['latency_ms']['p50'], json.dumps(describe((
                    configuration['setting']['prompt_detail'], configuration['setting']['prompt_layout'],
                    configuration['kind'], configuration['rotation'])), sort_keys=True))
    winner = min(eligible, key=key)
    return dict(schema_version=1, split='dev', variant=report['variant'],
                manifest_sha256=report['manifest_sha256'], labels_sha256=report['labels_sha256'],
                report_sha256=report_sha256, predictions=report['predictions'],
                selected={name: winner[name] for name in ('setting', 'kind', 'rotation', 'identity_sha256', 'policy')},
                metrics=winner['metrics'],
                criterion='max raw_top1, max accepted_correct_over_all, min p50 latency, lexical configuration key',
                protocol='dev selection only; freeze this file before accessing calibration/test predictions')


def write_outputs(report, output, selection=None):
    output = Path(output)
    selection = Path(selection) if selection is not None else None
    if selection is not None and report['split'] != 'dev':
        raise ValueError('configuration selection is allowed on dev only')
    if output.exists() or selection is not None and (selection.exists() or selection == output):
        raise FileExistsError('refusing to overwrite a report or selection file')
    encoded = json_bytes(report)
    chosen = select_configuration(report, hashlib.sha256(encoded).hexdigest()) if selection is not None else None
    with output.open('xb') as stream:
        stream.write(encoded)
    if selection is not None:
        with selection.open('xb') as stream:
            stream.write(json_bytes(chosen))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--predictions', required=True, action='append', type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'calibration', 'test'))
    parser.add_argument('--variant', required=True, choices=('natural', 'symbolic'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--select', type=Path, help='write a frozen configuration choice; dev only')
    args = parser.parse_args()
    if args.select is not None and args.split != 'dev':
        parser.error('--select requires --split dev')
    report = build_report(args.data, args.predictions, args.split, args.variant)
    write_outputs(report, args.output, args.select)
    print(json.dumps(dict(configurations=len(report['configurations']), logical_cases=report['logical_cases']), sort_keys=True))


if __name__ == '__main__':
    main()
