#!/usr/bin/env python3
"""Freeze a paired, grouped synthetic decision study without model-based selection.

Generated data are local experiment artifacts, not source-controlled fixtures.
Natural and symbolic rows share a logical ID and must never be counted as two
independent examples. All split assignments are frozen before inference.
"""
import argparse
import collections
import hashlib
import itertools
import json
from decimal import Decimal
from pathlib import Path
import random

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260926
DEFAULT_COUNTS = {'train': 480, 'dev': 120, 'calibration': 120, 'test': 180}
KINDS = ('choice', 'binary', 'ordinal')
VARIANTS = ('natural', 'symbolic')
DOMAINS = (
    ('finance', 'net_balance', ('reserve', 'maintain', 'invest')),
    ('sensors', 'signal_delta', ('weak', 'nominal', 'strong')),
    ('ecology', 'habitat_index', ('restore', 'observe', 'protect')),
    ('manufacturing', 'tolerance_score', ('adjust', 'retain', 'inspect')),
    ('networking', 'capacity_margin', ('restrict', 'normal', 'expand')),
    ('education', 'progress_index', ('support', 'practice', 'advance')),
)
# Threshold values themselves, not merely threshold-pair IDs, are split-disjoint.
# These rules never use the earlier warehouse thresholds 6 and 24.
THRESHOLDS = {
    'train': [('11', '19'), ('31.5', '43.5'), ('-23', '-13'), ('71.25', '83.75'),
              ('101', '119'), ('-51.5', '-37.5'), ('151.25', '163.75'), ('211', '229')],
    'dev': [('17.25', '27.75'), ('47', '59'), ('-33.5', '-21.5'), ('89.25', '97.75'),
            ('127', '139'), ('-67', '-55'), ('179.5', '193.5'), ('241.25', '257.75')],
    'calibration': [('37', '49'), ('61.25', '73.75'), ('-43.25', '-29.75'), ('107.5', '121.5'),
                    ('167', '181'), ('-87.25', '-69.75'), ('223.5', '237.5'), ('281', '299')],
    'test': [('53.25', '67.75'), ('79', '91'), ('-63.5', '-47.5'), ('137.25', '149.75'),
             ('197', '209'), ('-109', '-93'), ('263.25', '277.75'), ('317.5', '331.5')],
}
# Entire wording families are allocated to one split. Repeated math operators
# and schema keys are necessarily shared grammar, not independent held-out text.
TEMPLATES = {
    'train-a': {
        'intro': 'Use the supplied {domain} measurement and choose the matching rule.',
        'low': '{field} is less than {lo}.',
        'mid': '{field} is at least {lo} and less than {hi}.',
        'high': '{field} is at least {hi}.',
        'ge': '{field} is at least {cutoff}.', 'lt': '{field} is less than {cutoff}.',
        'symbolic_intro': 'Evaluate the {domain} rule expressions against the state.',
        'symbolic': 'rule: {expression}',
    },
    'train-b': {
        'intro': 'Classify the {domain} reading by the numerical conditions listed below.',
        'low': 'The value of {field} falls below {lo}.',
        'mid': 'The value of {field} reaches {lo} but remains below {hi}.',
        'high': 'The value of {field} reaches or exceeds {hi}.',
        'ge': 'The value of {field} reaches or exceeds {cutoff}.',
        'lt': 'The value of {field} falls below {cutoff}.',
        'symbolic_intro': 'Select the satisfied {domain} Boolean condition.',
        'symbolic': 'condition({expression})',
    },
    'dev-a': {
        'intro': 'Assign this {domain} observation to its stated numerical category.',
        'low': 'For {field}, the required range ends strictly before {lo}.',
        'mid': 'For {field}, the range starts inclusively at {lo} and ends exclusively at {hi}.',
        'high': 'For {field}, the required range begins inclusively at {hi}.',
        'ge': 'For {field}, the range begins inclusively at {cutoff}.',
        'lt': 'For {field}, the range ends strictly before {cutoff}.',
        'symbolic_intro': 'Find the true predicate for this {domain} observation.',
        'symbolic': 'predicate[{expression}]',
    },
    'dev-b': {
        'intro': 'Resolve the {domain} decision using the comparison criteria.',
        'low': 'Accept when {field} is smaller than {lo}.',
        'mid': 'Accept when {field} is not smaller than {lo} but is smaller than {hi}.',
        'high': 'Accept when {field} is not smaller than {hi}.',
        'ge': 'Accept when {field} is not smaller than {cutoff}.',
        'lt': 'Accept when {field} is smaller than {cutoff}.',
        'symbolic_intro': 'Resolve the {domain} decision by testing each expression.',
        'symbolic': 'accept iff ({expression})',
    },
    'calibration-a': {
        'intro': 'Determine which listed interval contains the {domain} measurement.',
        'low': '{field} belongs below the excluded endpoint {lo}.',
        'mid': '{field} lies between included endpoint {lo} and excluded endpoint {hi}.',
        'high': '{field} belongs at or above the included endpoint {hi}.',
        'ge': '{field} belongs at or above the included endpoint {cutoff}.',
        'lt': '{field} belongs below the excluded endpoint {cutoff}.',
        'symbolic_intro': 'Determine the matching formal guard for the {domain} value.',
        'symbolic': 'guard = {expression}',
    },
    'calibration-b': {
        'intro': 'Apply the specified numerical bands to this {domain} record.',
        'low': 'A {field} reading under {lo} qualifies.',
        'mid': 'A {field} reading from {lo} inclusive to {hi} exclusive qualifies.',
        'high': 'A {field} reading of {hi} or greater qualifies.',
        'ge': 'A {field} reading of {cutoff} or greater qualifies.',
        'lt': 'A {field} reading under {cutoff} qualifies.',
        'symbolic_intro': 'Apply the listed comparison formulas to the {domain} record.',
        'symbolic': 'formula: ({expression})',
    },
    'test-a': {
        'intro': 'Locate the {domain} input within the criteria-defined ranges.',
        'low': 'The permitted {field} values are strictly lower than {lo}.',
        'mid': 'The permitted {field} values include {lo} and extend up to, but not including, {hi}.',
        'high': 'The permitted {field} values include {hi} and every greater value.',
        'ge': 'The permitted {field} values include {cutoff} and every greater value.',
        'lt': 'The permitted {field} values are strictly lower than {cutoff}.',
        'symbolic_intro': 'Locate the satisfied logical test for the {domain} input.',
        'symbolic': 'logical test: {expression}',
    },
    'test-b': {
        'intro': 'Match the {domain} quantity to the applicable comparison statement.',
        'low': 'This statement holds only while {field} stays short of {lo}.',
        'mid': 'This statement holds from {field} equal to {lo} until it reaches {hi}, which is excluded.',
        'high': 'This statement holds once {field} meets the minimum {hi}.',
        'ge': 'This statement holds once {field} meets the minimum {cutoff}.',
        'lt': 'This statement holds only while {field} stays short of {cutoff}.',
        'symbolic_intro': 'Match the {domain} quantity to a true formal statement.',
        'symbolic': 'statement: [{expression}]',
    },
}
TEMPLATE_ALLOCATION = {split: [split + '-a', split + '-b'] for split in DEFAULT_COUNTS}


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    """All fractions are eighths/quarters/halves, exactly representable in JSON floats."""
    value = Decimal(value)
    return int(value) if value == value.to_integral_value() else float(value)


def band_index(value, lower, upper):
    """Generation oracle; independently checked by interval enumeration in tests."""
    return 0 if value < lower else 1 if value < upper else 2


def binary_label(value, cutoff, true_if_ge):
    return 'true' if (value >= cutoff) == true_if_ge else 'false'


def candidates(split, kind):
    for threshold_index, (lo_text, hi_text) in enumerate(THRESHOLDS[split]):
        lo, hi = Decimal(lo_text), Decimal(hi_text)
        for template_id, domain_index, step_text in itertools.product(
                TEMPLATE_ALLOCATION[split], range(len(DOMAINS)), ('0.125', '1')):
            step = Decimal(step_text)
            common = dict(threshold_id=f'{split}-threshold-{threshold_index}',
                          template_id=template_id, domain_index=domain_index,
                          lower=lo_text, upper=hi_text, kind=kind, step=step_text)
            if kind == 'binary':
                for cutoff_name, true_if_ge in itertools.product(('lower', 'upper'), (False, True)):
                    # Explicit loops keep each generated boundary recorded for auditing.
                    cutoff = lo if cutoff_name == 'lower' else hi
                    for relation, offset in [('below', -step), ('exact', Decimal(0)), ('above', step)]:
                        value = cutoff + offset
                        yield dict(common, value=str(value), cutoff=str(cutoff), true_if_ge=true_if_ge,
                                   boundary=f'{cutoff_name}-{relation}',
                                   label=binary_label(value, cutoff, true_if_ge))
            else:
                for boundary, value in [('lower-below', lo-step), ('lower-exact', lo),
                                        ('lower-above', lo+step), ('upper-below', hi-step),
                                        ('upper-exact', hi), ('upper-above', hi+step)]:
                    yield dict(common, value=str(value), boundary=boundary,
                               label=band_index(value, lo, hi))


def choose_designs(split, count, rng):
    if count <= 0 or count % 6:
        raise ValueError('each split count must be a positive multiple of six')
    selected = []
    for kind in KINDS:
        strata = collections.defaultdict(list)
        # Exact boundary cases with different step sizes are the same logical input.
        seen = set()
        for design in candidates(split, kind):
            key = tuple((k, v) for k, v in design.items() if k != 'step')
            if key in seen:
                continue
            seen.add(key)
            strata[design['label']].append(design)
        labels = ('false', 'true') if kind == 'binary' else (0, 1, 2)
        per_kind = count // len(KINDS)
        for i, label in enumerate(labels):
            required = per_kind // len(labels) + (i < per_kind % len(labels))
            pool = strata[label]
            if required > len(pool):
                raise ValueError(f'{split} {kind}: requested count exceeds unique rule pool')
            rng.shuffle(pool)
            selected.extend(pool[:required])
    rng.shuffle(selected)
    return selected


def render(design, case_id, rng, choice_position):
    domain, field, base_ids = DOMAINS[design['domain_index']]
    template = TEMPLATES[design['template_id']]
    group = f"{design['threshold_id']}::{design['template_id']}"
    decision_id = f'{domain}_{design["kind"]}_{case_id[-8:]}'
    state = {
        field: number(design['value']),
        'record_tag': hashlib.sha256((case_id + '-tag').encode()).hexdigest()[:10],
        'unrelated_counter': rng.randrange(-400, 401),
        'operator_note': rng.choice(('Routine audit entry.', 'Sensor label was refreshed.',
                                     'Historical annotation; ignore for this decision.')),
        'auxiliary': {'revision': rng.randrange(1, 10), 'enabled': bool(rng.randrange(2))},
    }
    semantic_ids = [base + '-' + hashlib.sha256((case_id + base).encode()).hexdigest()[:6]
                    for base in base_ids]
    ordinal_values = rng.choice(([-3, 0, 7], [-1.5, 2.25, 9], [10, 20, 40]))
    order = list(range(3))
    rng.shuffle(order)
    if design['kind'] == 'choice':
        # Balance the correct answer's code position as well as semantic bands.
        # The order of the other two options remains randomized.
        current = order.index(design['label'])
        order[current], order[choice_position] = order[choice_position], order[current]
    params = dict(domain=domain, field=field, lo=design['lower'], hi=design['upper'],
                  cutoff=design.get('cutoff'))
    expressions = [f"{field} < {design['lower']}",
                   f"({field} >= {design['lower']}) AND ({field} < {design['upper']})",
                   f"{field} >= {design['upper']}"]
    outputs = {}
    if design['kind'] == 'binary':
        expected = design['label']
        semantic_ids = ['false', 'true']
    else:
        expected = semantic_ids[design['label']]
    for variant in VARIANTS:
        def criterion(name, expression):
            return (template[name].format(**params) if variant == 'natural' else
                    template['symbolic'].format(expression=expression))
        if design['kind'] == 'binary':
            positive = 'ge' if design['true_if_ge'] else 'lt'
            negative = 'lt' if design['true_if_ge'] else 'ge'
            expression = {'ge': f"{field} >= {design['cutoff']}", 'lt': f"{field} < {design['cutoff']}"}
            kind = dict(type='binary', false_label=criterion(negative, expression[negative]),
                        true_label=criterion(positive, expression[positive]))
        else:
            options = [dict(id=semantic_ids[i], criterion=criterion(name, expressions[i]))
                       for i, name in enumerate(('low', 'mid', 'high'))]
            if design['kind'] == 'choice':
                kind = dict(type='choice', options=[options[i] for i in order])
            else:
                kind = dict(type='ordinal', levels=[dict(option, value=value)
                                                   for option, value in zip(options, ordinal_values)])
        instruction = template['intro' if variant == 'natural' else 'symbolic_intro'].format(**params)
        instruction += f' Use only {field}; other state fields are irrelevant.'
        outputs[variant] = dict(id=case_id, group=group,
                               request=dict(state=state, decisions=[dict(id=decision_id, instruction=instruction, kind=kind)]),
                               expected={decision_id: expected})
    metadata = dict(design, domain=domain, field=field, semantic_ids=semantic_ids)
    metadata.pop('label')
    metadata.pop('domain_index')
    return outputs, metadata


def prepare(output, counts=None, seed=DEFAULT_SEED):
    output = Path(output)
    counts = dict(DEFAULT_COUNTS if counts is None else counts)
    if set(counts) != set(DEFAULT_COUNTS):
        raise ValueError('exactly train/dev/calibration/test splits are required')
    # Build everything before creating output; no partially written study on bad counts.
    rng = random.Random(seed)
    study = {}
    cases = {}
    labels = {}
    for split in DEFAULT_COUNTS:
        rows = {variant: [] for variant in VARIANTS}
        choice_count = 0
        for index, design in enumerate(choose_designs(split, counts[split], rng)):
            identity = json_bytes(dict(seed=seed, split=split, design=design, index=index))
            case_id = 'rule-' + hashlib.sha256(identity).hexdigest()[:20]
            paired, metadata = render(design, case_id, rng, choice_count % 3)
            if design['kind'] == 'choice':
                choice_count += 1
            cases[case_id] = metadata
            labels[case_id] = paired['natural']['expected']
            for variant in VARIANTS:
                rows[variant].append(paired[variant])
        study[split] = rows
    output.mkdir(parents=True, exist_ok=False)
    labels_path = output / 'labels.json'
    labels_path.write_bytes(json_bytes(labels))
    manifest = dict(schema_version=SCHEMA_VERSION, seed=seed, labels=dict(path='labels.json', sha256=sha256(labels_path)),
                    protocol={
                        'study': 'frozen_synthetic_threshold_rules_v1',
                        'independent_unit': 'logical case; natural/symbolic variants are paired, never independent',
                        'split_policy': 'disjoint threshold values and disjoint wording-template families, not random row splitting',
                        'selection': 'freeze all splits before model inference; no prompt, hyperparameter or checkpoint selection on test',
                        'train': 'training only; optional teacher labeling may access training requests only',
                        'dev': 'prompt, hyperparameter and checkpoint selection only',
                        'calibration': 'fit score calibration and abstention policy after model selection',
                        'test': 'final held-out evaluation only; do not fit or select using these rows',
                        'reference': 'exact Decimal comparisons; eighth/quarter/half fractions; below/exact/above boundaries',
                        'scope': 'synthetic arithmetic rule-following across six domains; not production accuracy or a real-world generalization claim',
                        'prior_cases': 'new generated rules; excludes warehouse fields and thresholds 6/24',
                        'code_rotations': 'none generated here; any training rotations retain source logical ID and group',
                    }, threshold_allocations={split: [list(pair) for pair in pairs] for split, pairs in THRESHOLDS.items()},
                    template_allocations=TEMPLATE_ALLOCATION,
                    templates=TEMPLATES, splits={}, cases=cases)
    for split, rows in study.items():
        base = rows['natural']
        entry = dict(count=len(base), ids=[r['id'] for r in base], groups=sorted({r['group'] for r in base}),
                     kinds=dict(collections.Counter(cases[r['id']]['kind'] for r in base)), variants={})
        for variant, records in rows.items():
            path = output / f'{split}-{variant}.jsonl'
            path.write_bytes(b''.join(json_bytes(row) for row in records))
            request_path = output / f'{split}-{variant}-requests.jsonl'
            request_path.write_bytes(b''.join(json_bytes({k: row[k] for k in ('id', 'group', 'request')})
                                             for row in records))
            entry['variants'][variant] = dict(path=path.name, sha256=sha256(path),
                                              requests_path=request_path.name, requests_sha256=sha256(request_path))
        manifest['splits'][split] = entry
    (output / 'manifest.json').write_bytes(json_bytes(manifest))
    return manifest


def validate_dataset(output):
    """Verify frozen bytes and paired/disjoint IDs, groups and request/label separation."""
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_bytes())
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported study schema')
    labels_meta = manifest['labels']
    if sha256(output / labels_meta['path']) != labels_meta['sha256']:
        raise ValueError('labels hash mismatch')
    labels = json.loads((output / labels_meta['path']).read_bytes())
    if set(manifest['splits']) != set(DEFAULT_COUNTS):
        raise ValueError('missing or unknown study split')
    seen_ids, seen_groups = set(), set()
    for split, entry in manifest['splits'].items():
        ids, groups = set(entry['ids']), set(entry['groups'])
        if len(ids) != entry['count'] or ids & seen_ids or groups & seen_groups:
            raise ValueError('duplicate or cross-split logical IDs/groups')
        seen_ids.update(ids)
        seen_groups.update(groups)
        paired = []
        for variant in VARIANTS:
            metadata = entry['variants'][variant]
            for file_key, hash_key in [('path', 'sha256'), ('requests_path', 'requests_sha256')]:
                if sha256(output / metadata[file_key]) != metadata[hash_key]:
                    raise ValueError(f'{split}/{variant} hash mismatch')
            rows = [json.loads(line) for line in (output / metadata['path']).read_text().splitlines()]
            requests = [json.loads(line) for line in (output / metadata['requests_path']).read_text().splitlines()]
            if [r['id'] for r in rows] != entry['ids'] or {r['group'] for r in rows} != groups:
                raise ValueError('manifest row identity mismatch')
            if requests != [{k: r[k] for k in ('id', 'group', 'request')} for r in rows]:
                raise ValueError('unlabeled request mismatch')
            if any(row['expected'] != labels[row['id']] for row in rows):
                raise ValueError('label mapping mismatch')
            for row in rows:
                design = manifest['cases'][row['id']]
                allocation = {
                    f'{split}-threshold-{i}': pair
                    for i, pair in enumerate(manifest['threshold_allocations'][split])
                }
                if (allocation.get(design['threshold_id']) != [design['lower'], design['upper']]
                        or design['template_id'] not in manifest['template_allocations'][split]
                        or row['group'] != design['threshold_id'] + '::' + design['template_id']):
                    raise ValueError('case threshold/template allocation mismatch')
                decisions = row['request']['decisions']
                if len(decisions) != 1 or set(row['expected']) != {decisions[0]['id']}:
                    raise ValueError('expected exactly one labeled decision')
            paired.append(rows)
        for natural, symbolic in zip(*paired):
            if any(natural[k] != symbolic[k] for k in ('id', 'group', 'expected')):
                raise ValueError('variant pairing mismatch')
            if natural['request']['state'] != symbolic['request']['state']:
                raise ValueError('variant state mismatch')
    if set(labels) != seen_ids or set(manifest['cases']) != seen_ids:
        raise ValueError('label/case manifest coverage mismatch')
    for first, second in itertools.combinations(manifest['splits'], 2):
        a = {Decimal(v) for pair in manifest['threshold_allocations'][first] for v in pair}
        b = {Decimal(v) for pair in manifest['threshold_allocations'][second] for v in pair}
        if a & b or set(manifest['template_allocations'][first]) & set(manifest['template_allocations'][second]):
            raise ValueError('threshold/template leakage')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--verify', action='store_true', help='verify existing frozen files without modifying them')
    for split, count in DEFAULT_COUNTS.items():
        parser.add_argument('--' + split, type=int, default=count)
    args = parser.parse_args()
    if args.verify:
        manifest = validate_dataset(args.output)
    else:
        manifest = prepare(args.output, {split: getattr(args, split) for split in DEFAULT_COUNTS}, args.seed)
        validate_dataset(args.output)
    print(json.dumps({split: entry['count'] for split, entry in manifest['splits'].items()}, sort_keys=True))


if __name__ == '__main__':
    main()
