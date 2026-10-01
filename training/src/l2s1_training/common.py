"""Shared file, token and decision contracts; no ML dependencies at import time."""
import hashlib
import json
import re
from pathlib import Path

def require(condition, message):
    if not condition:
        raise ValueError(message)

def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write('\n')

def option_specs(decision):
    kind = decision['kind']
    if kind['type'] == 'binary':
        return [{'id': 'false', 'criterion': kind['false_label']},
                {'id': 'true', 'criterion': kind['true_label']}]
    require(kind['type'] in ('choice', 'ordinal'), 'Unsupported decision kind')
    return kind['options'] if kind['type'] == 'choice' else kind['levels']

def normalized_token_identity(identity, rotation, detail, layout=None):
    """Remove only an authenticated rotation suffix, preserving prompt detail."""
    variants = {'minimal': 'Minimal', 'typed': 'Typed', 'typed_examples': 'TypedExamples'}
    require(detail in variants, 'Unknown prompt detail')
    version = identity.get('prompt_version')
    require(isinstance(version, str) and version, 'Missing prompt version')
    match = re.fullmatch(r'(.+)/detail-(Minimal|Typed|TypedExamples)-v1/rotation-(0|[1-9][0-9]*)', version)
    if match:
        base, actual_detail, actual_rotation = match.groups()
        require(actual_detail == variants[detail] and int(actual_rotation) == rotation,
                'Prompt detail/rotation identity mismatch')
    else:
        require(detail == 'minimal' and rotation == 0,
                'Only minimal rotation zero may omit prompt suffix')
        base = version
    require('/detail-' not in base and '/rotation-' not in base, 'Malformed prompt identity suffix')
    if layout is not None:
        require(layout in ('legacy', 'state-first'), 'Unknown prompt layout')
        require(('-state-first-' in base) == (layout == 'state-first'),
                'Prompt layout identity mismatch')
    return dict(identity, prompt_version=f'{base}/detail-{variants[detail]}-v1')

def validate_tokenizer(tokenizer, rows):
    size = len(tokenizer)
    for row in rows:
        require(all(i < size for i in row['input_ids'] + row['candidate_ids']), 'HF vocabulary mismatch')
        actual = [tokenizer.decode([i], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False) for i in row['candidate_ids']]
        require(actual == row['candidate_codes'], 'HF/native answer-token mapping mismatch')


def save_status(path, value):
    """Atomically replace a live status file; preserve the last complete JSON."""
    import os
    import tempfile
    path = Path(path)
    descriptor, tmp = tempfile.mkstemp(prefix=path.name+'.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as out:
            json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)
