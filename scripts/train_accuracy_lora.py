#!/usr/bin/env python3
"""Offline Gemma4 thinking-teacher distillation into production single-token decisions.

Teacher, discarded smoke, and final training are separate invocations. Only verified
train teachers enter optimization, including all sealed code rotations for an
accepted logical case. No generated rationale or evaluation label enters the loss.
"""
import argparse
import collections
from collections.abc import Mapping
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path

MODEL = 'google/gemma-4-E2B-it'
REVISION = '3e22461f65e89153144f8adb70e3b8c2cc9845a7'
SEED = 20260923
PROTOCOL = dict(epochs=1, rank=8, alpha=16, dropout=0.0,
                gradient_accumulation=12, learning_rate=1e-4,
                mass_loss_weight=0.1, seed=SEED,
                checkpoint_selection='fixed final epoch; no evaluation-label selection')


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
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, ensure_ascii=False)
        out.write('\n')


def checked_file(root, name, expected):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Manifest file must be inside dataset')
    require(digest(path) == expected, f'Hash mismatch: {name}')
    return path


def option_specs(decision):
    kind = decision['kind']
    if kind['type'] == 'binary':
        return [{'id': 'false', 'criterion': kind['false_label']},
                {'id': 'true', 'criterion': kind['true_label']}]
    require(kind['type'] in ('choice', 'ordinal'), 'Unsupported decision kind')
    return kind['options'] if kind['type'] == 'choice' else kind['levels']


def normalized_token_identity(identity, rotation, detail):
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
    return dict(identity, prompt_version=f'{base}/detail-{variants[detail]}-v1')


def load_data(data, variant, tokens_path, seal_path, detail):
    """Verify the frozen dataset, leakage boundary and native-token provenance."""
    data = Path(data)
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    require(manifest['schema_version'] == 1, 'Unknown dataset schema')
    labels = json.loads(checked_file(data, manifest['labels']['path'],
                                    manifest['labels']['sha256']).read_text())
    seen_ids, seen_groups = set(), set()
    train_cases = None
    for split in ('train', 'dev', 'calibration', 'test'):
        config = manifest['splits'][split]
        ids, groups = set(config['ids']), set(config['groups'])
        require(len(ids) == config['count'] == len(config['ids']), 'Duplicate split IDs')
        require(not ids & seen_ids and not groups & seen_groups, 'Cross-split ID/group leakage')
        seen_ids.update(ids)
        seen_groups.update(groups)
        for name, files in config['variants'].items():
            rows = read_jsonl(checked_file(data, files['path'], files['sha256']))
            requests = read_jsonl(checked_file(data, files['requests_path'], files['requests_sha256']))
            require([r['id'] for r in rows] == config['ids'], 'Labeled ID order mismatch')
            require([r['id'] for r in requests] == config['ids'], 'Request ID order mismatch')
            require({r['group'] for r in rows} == groups, 'Group allocation mismatch')
            for row, request in zip(rows, requests):
                require(set(request) == {'id', 'group', 'request'}, 'Unlabeled request contains extra fields')
                require({k: row[k] for k in request} == request, 'Labeled/request disagreement')
                require(row['expected'] == labels[row['id']], 'Independent label mismatch')
                decisions = row['request']['decisions']
                require(decisions and len({d['id'] for d in decisions}) == len(decisions),
                        'Invalid or duplicate decision IDs')
                require(set(row['expected']) == {d['id'] for d in decisions}, 'Label task mismatch')
                for decision in decisions:
                    options = option_specs(decision)
                    require(2 <= len(options) <= 26 and len({o['id'] for o in options}) == len(options),
                            'Invalid semantic options')
                    require(row['expected'][decision['id']] in {o['id'] for o in options},
                            'Label outside semantic options')
            if split == 'train' and name == variant:
                train_cases = {r['id']: r for r in rows}
    require(set(labels) == seen_ids, 'Labels outside frozen split IDs')
    require(train_cases is not None, 'Requested train variant missing')
    seal = json.loads(Path(seal_path).read_text())
    require(seal['schema_version'] == 1 and seal['split'] == 'train', 'Only sealed train tokens accepted')
    require(seal['manifest_sha256'] == digest(manifest_path), 'Token seal dataset mismatch')
    require(seal['token_sha256'] == digest(tokens_path), 'Production token hash mismatch')
    require(seal['variant'] == variant and seal['detail'] == detail, 'Token prompt configuration mismatch')
    require(seal['hf_model'] == MODEL and seal['hf_revision'] == REVISION, 'Token source revision mismatch')
    rows = read_jsonl(tokens_path)
    require(rows, 'Empty production token export')
    keys, identities = set(), set()
    covered = set()
    for row in rows:
        require(row['id'] in train_cases, 'Non-training ID in tokens')
        case = train_cases[row['id']]
        decisions = {d['id']: d for d in case['request']['decisions']}
        require(row['decision_id'] in decisions, 'Unknown token decision ID')
        rotation = row.get('code_rotation', 0)
        require(type(rotation) is int and rotation >= 0, 'Invalid code rotation')
        key = (row['id'], row['decision_id'], rotation)
        require(key not in keys, 'Duplicate token row')
        keys.add(key)
        covered.add(key[:2])
        options = option_specs(decisions[row['decision_id']])
        semantic_ids = [o['id'] for o in options]
        width = len(semantic_ids)
        require(len(row['option_ids']) == width and set(row['option_ids']) == set(semantic_ids),
                'Token semantic mapping mismatch')
        require(len(row['candidate_ids']) == width and len(set(row['candidate_ids'])) == width,
                'Duplicate/missing candidate tokens')
        require(len(row['candidate_codes']) == width and
                set(row['candidate_codes']) == {chr(65+i) for i in range(width)}, 'Invalid candidate codes')
        require(0 < len(row['input_ids']) <= 2048, 'Production input outside 1..2048 token context')
        require(all(type(i) is int and i >= 0 for i in row['input_ids'] + row['candidate_ids']),
                'Invalid production token ID')
        identity = row['model_identity']
        require(isinstance(identity, dict) and
                all(isinstance(identity.get(k), str) and re.fullmatch('[0-9a-f]{64}', identity[k])
                    for k in ('weights_sha256', 'template_sha256', 'runtime_build_sha256')),
                'Missing production model identity')
        require(not identity.get('adapter_sha256') and not identity.get('head_sha256'),
                'Student tokens must come from a base model without adapter/head')
        identities.add(json.dumps(normalized_token_identity(identity, rotation, detail), sort_keys=True))
    expected = {(cid, d['id']) for cid, case in train_cases.items()
                for d in case['request']['decisions']}
    require(covered == expected, 'Incomplete production training token export')
    require(len(identities) == 1, 'Mixed production model identities')
    provenance = dict(manifest_sha256=digest(manifest_path), token_sha256=digest(tokens_path),
                      token_seal_sha256=digest(seal_path), variant=variant, detail=detail,
                      hf_model=MODEL, hf_revision=REVISION)
    return train_cases, rows, provenance


def checkpoint_identity(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    require(checkpoint.is_dir(), 'Checkpoint must be an existing offline directory')
    config = json.loads((checkpoint / 'config.json').read_text())
    require(checkpoint.name == REVISION or config.get('_commit_hash') == REVISION,
            'Checkpoint must identify the exact pinned HF revision')
    require(config.get('model_type') == 'gemma4', 'Expected Gemma4 checkpoint')
    files = sorted(p for p in checkpoint.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.jinja') or p.name == 'tokenizer.model'))
    require(any(p.suffix == '.safetensors' for p in files), 'Checkpoint weights missing')
    hashes = {p.name: digest(p) for p in files}
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def validate_tokenizer(tokenizer, rows):
    size = len(tokenizer)
    for row in rows:
        require(all(i < size for i in row['input_ids'] + row['candidate_ids']), 'HF vocabulary mismatch')
        actual = [tokenizer.decode([i], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False) for i in row['candidate_ids']]
        require(actual == row['candidate_codes'], 'HF/native answer-token mapping mismatch')


def teacher_messages(case, row):
    decision = next(d for d in case['request']['decisions'] if d['id'] == row['decision_id'])
    options = {o['id']: o for o in option_specs(decision)}
    mapped = [dict(code=code, **options[sid])
              for sid, code in zip(row['option_ids'], row['candidate_codes'])]
    content = dict(state=case['request']['state'], instruction=decision['instruction'],
                   kind=decision['kind']['type'], candidates=mapped)
    return [dict(role='user', content=(
        'Determine the correct decision from the state and the exact candidate criteria. '
        'For ordinal candidates apply their thresholds exactly. Think carefully in the thought channel. '
        'After ending the thought channel, output exactly one candidate code and nothing else.\n'
        + json.dumps(content, ensure_ascii=False, sort_keys=True)))]


def teacher_inputs(tokenizer, messages, device):
    # Transformers 5 defaults to BatchEncoding; request the mapping explicitly
    # and pass its tensors (including the attention mask) into generation.
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, return_tensors='pt', return_dict=True,
        add_generation_prompt=True, enable_thinking=True)
    if isinstance(encoded, Mapping):
        require('input_ids' in encoded, 'Teacher tokenizer returned no input IDs')
        inputs = {key: value.to(device) for key, value in encoded.items()
                  if key in ('input_ids', 'attention_mask')}
    else:
        # Compatible with tokenizer implementations that still return a tensor.
        inputs = {'input_ids': encoded.to(device)}
    require(len(inputs['input_ids'].shape) == 2 and inputs['input_ids'].shape[0] == 1,
            'Teacher requires exactly one tokenized prompt')
    return inputs


def parse_teacher(text, codes, reached_budget):
    """Never treat a code mentioned inside reasoning as the final answer."""
    if '<channel|>' not in text:
        return ('thought_budget_exhausted' if reached_budget else 'missing_final'), None
    thought, final = text.rsplit('<channel|>', 1)
    if '<|channel>thought' not in thought:
        return 'missing_thought_channel', None
    match = re.fullmatch(r'\s*([A-Z])\s*(?:(?:<turn\|>|<eos>)\s*)*', final)
    if not match or match.group(1) not in codes:
        return 'invalid_final', None
    return 'parsed', match.group(1)


def seed_everything(torch):
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)


def load_model(checkpoint, training):
    import torch
    from transformers import BitsAndBytesConfig, Gemma4ForConditionalGeneration
    seed_everything(torch)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        checkpoint, local_files_only=True, quantization_config=quant,
        device_map={'': 'cuda:0'}, dtype=torch.bfloat16, attn_implementation='sdpa')
    for _, param in model.named_parameters():
        param.requires_grad_(False)
        if param.is_floating_point() and param.ndim < 2:
            param.data = param.data.float()
    model.config.use_cache = not training
    if training:
        from peft import LoraConfig, get_peft_model
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(
            r=8, lora_alpha=16, lora_dropout=0.0, bias='none', task_type='CAUSAL_LM',
            target_modules=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)'))
        for name, param in model.named_parameters():
            if param.requires_grad:
                require('lora_' in name, 'Unexpected non-adapter trainable parameter')
                param.data = param.data.float()
    return model


def run_teacher(args, cases, rows, provenance, tokenizer):
    import torch
    model = load_model(args.checkpoint, training=False)
    model.eval()
    selected_ids = sorted(cases)[:args.max_teacher_cases or None]
    by_task = {}
    for row in rows:
        if row['id'] in selected_ids:
            key = row['id'], row['decision_id']
            if key not in by_task or row.get('code_rotation', 0) < by_task[key].get('code_rotation', 0):
                by_task[key] = row
    counts = collections.Counter()
    with torch.inference_mode(), (args.output / 'teacher.jsonl').open('x') as output:
        for key, row in sorted(by_task.items()):
            case = cases[row['id']]
            messages = teacher_messages(case, row)
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
            require('<|think|>' in prompt, 'Checkpoint template did not enable thinking')
            inputs = teacher_inputs(tokenizer, messages, 'cuda')
            input_length = inputs['input_ids'].shape[-1]
            require(input_length <= 2048, 'Teacher input exceeds budget')
            started = time.monotonic()
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, use_cache=True)
            continuation = generated[0, input_length:].tolist()
            text = tokenizer.decode(continuation, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
            budget = len(continuation) >= args.max_new_tokens
            status, code = parse_teacher(text, row['candidate_codes'], budget)
            predicted = (row['option_ids'][row['candidate_codes'].index(code)]
                         if code is not None else None)
            if status == 'parsed':
                status = 'correct' if predicted == case['expected'][row['decision_id']] else 'wrong'
            record = dict(id=row['id'], decision_id=row['decision_id'], group=case['group'],
                          status=status, code=code, predicted=predicted,
                          candidate_codes=row['candidate_codes'], option_ids=row['option_ids'],
                          generated_text=text, generated_tokens=len(continuation), reached_budget=budget,
                          elapsed_s=time.monotonic()-started)
            output.write(json.dumps(record, ensure_ascii=False)+'\n')
            output.flush()
            counts[status] += 1
            print('TEACHER', len(by_task), sum(counts.values()), status, flush=True)
    report = dict(provenance, complete=True, teacher_sha256=digest(args.output/'teacher.jsonl'),
                  checkpoint_sha256=provenance['checkpoint_sha256'], statuses=dict(counts),
                  selected_logical_cases=len(selected_ids), teacher_tasks=len(by_task),
                  max_new_tokens=args.max_new_tokens, max_teacher_cases=args.max_teacher_cases,
                  enable_thinking=True, teacher_precision='NF4 base, bf16 matrices, fp32 norms; SDPA',
                  student_supervision='verified final semantic answer only; no rationale loss')
    write_json(args.output/'teacher-report.json', report)


def accepted_training(cases, rows, provenance, teacher_dir, max_cases=0):
    report = json.loads((teacher_dir/'teacher-report.json').read_text())
    require(report.get('complete') is True and report.get('enable_thinking') is True,
            'Teacher run incomplete or thinking disabled')
    for key, value in provenance.items():
        require(report.get(key) == value, f'Teacher provenance mismatch: {key}')
    teacher_path = teacher_dir/'teacher.jsonl'
    require(digest(teacher_path) == report['teacher_sha256'], 'Teacher audit changed')
    teachers = read_jsonl(teacher_path)
    representative = {}
    for token_row in rows:
        key = token_row['id'], token_row['decision_id']
        if key not in representative or token_row.get('code_rotation', 0) < representative[key].get('code_rotation', 0):
            representative[key] = token_row
    accepted, seen = {}, set()
    for row in teachers:
        key = row['id'], row['decision_id']
        require(key not in seen and row['id'] in cases, 'Duplicate or held-out teacher')
        seen.add(key)
        case = cases[row['id']]
        require(row['group'] == case['group'] and row['decision_id'] in case['expected'],
                'Teacher group/task mismatch')
        require(key in representative and all(row[field] == representative[key][field]
                for field in ('option_ids', 'candidate_codes')), 'Teacher/export mapping mismatch')
        status, code = parse_teacher(row['generated_text'], row['candidate_codes'], row['reached_budget'])
        if row['status'] != 'correct':
            continue
        require(status == 'parsed' and code == row['code'], 'Accepted teacher final parse mismatch')
        require(len(row['option_ids']) == len(row['candidate_codes']) and
                len(set(row['candidate_codes'])) == len(row['candidate_codes']), 'Invalid teacher mapping')
        semantic = row['option_ids'][row['candidate_codes'].index(code)]
        require(semantic == row['predicted'] == case['expected'][row['decision_id']],
                'Accepted teacher disagrees with independent train label')
        accepted[key] = semantic
    require(dict(collections.Counter(r['status'] for r in teachers)) == report['statuses'],
            'Teacher status totals disagree')
    ids = sorted({key[0] for key in accepted})[:max_cases or None]
    selected = []
    for row in rows:
        key = row['id'], row['decision_id']
        if key in accepted and row['id'] in ids:
            selected.append(dict(row, teacher_target=accepted[key]))
    require(selected, 'No verified teacher examples; refusing oracle substitution')
    return selected, dict(teacher_sha256=report['teacher_sha256'],
                          accepted_tasks=len(accepted), accepted_logical_cases=len({k[0] for k in accepted}),
                          selected_logical_cases=len(ids), selected_rotated_examples=len(selected),
                          teacher_statuses=report['statuses'], max_train_cases=max_cases)


def run_training(args, cases, rows, provenance, tokenizer):
    import torch
    import transformers
    import peft
    selected, teacher = accepted_training(cases, rows, provenance, args.teacher, args.max_train_cases)
    binding = dict(provenance, **teacher, protocol=PROTOCOL)
    smoke = args.command == 'smoke'
    if not smoke:
        require(args.smoke_report is not None, 'Final training requires a discarded smoke report')
        report = json.loads(args.smoke_report.read_text())
        require(report.get('complete') is True and report.get('smoke') is True and
                report.get('adapter_saved') is False, 'Smoke must complete without retaining an adapter')
        require(report.get('binding') == binding, 'Smoke used different data/teacher/checkpoint/protocol')
    model = load_model(args.checkpoint, training=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    require(trainable, 'No LoRA parameters matched pinned model')
    model.train()
    order = list(selected)
    random.Random(SEED).shuffle(order)
    if smoke:
        order = order[:12]
    accumulation = PROTOCOL['gradient_accumulation']
    steps = math.ceil(len(order)/accumulation)
    optimizer = torch.optim.AdamW(trainable, lr=PROTOCOL['learning_rate'], weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    write_json(args.output/'metadata.json', dict(binding=binding, smoke=smoke,
               torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
               gpu=torch.cuda.get_device_name(), script_sha256=digest(Path(__file__)),
               trainable_parameters=sum(p.numel() for p in trainable),
               precision='NF4 base, bf16 frozen matrices, fp32 adapters/norms; SDPA',
               objective='candidate CE + 0.1 * negative log full-vocabulary candidate mass',
               training_examples=len(order), optimization_steps=steps))
    sums = [0.0, 0.0]
    with (args.output/'training.jsonl').open('x') as log:
        for i, row in enumerate(order):
            inputs = torch.tensor([row['input_ids']], device='cuda')
            logits = model(input_ids=inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            candidate = logits[row['candidate_ids']]
            target = row['option_ids'].index(row['teacher_target'])
            ce = torch.nn.functional.cross_entropy(candidate[None], torch.tensor([target], device='cuda'))
            mass = torch.logsumexp(logits, 0)-torch.logsumexp(candidate, 0)
            loss = ce+PROTOCOL['mass_loss_weight']*mass
            require(torch.isfinite(loss).item(), 'Nonfinite training loss')
            size = min(accumulation, len(order)-(i//accumulation)*accumulation)
            (loss/size).backward()
            sums[0] += ce.item()
            sums[1] += mass.item()
            if (i+1) % accumulation == 0 or i+1 == len(order):
                grad = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = i//accumulation+1
                optimizer.param_groups[0]['lr'] = PROTOCOL['learning_rate']*(1-step/steps)
                info = dict(step=step, examples=i+1, ce=sums[0]/size, mass_loss=sums[1]/size,
                            grad_norm=grad.item(), elapsed_s=time.monotonic()-started,
                            peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
                log.write(json.dumps(info)+'\n')
                log.flush()
                print('TRAIN', json.dumps(info), flush=True)
                sums = [0.0, 0.0]
    require(any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if 'lora_B' in n),
            'LoRA did not update')
    if not smoke:
        model.save_pretrained(args.output/'adapter')
        tokenizer.save_pretrained(args.output/'adapter')
    write_json(args.output/'complete.json', dict(complete=True, smoke=smoke, binding=binding,
               adapter_saved=not smoke, optimization_steps=steps, training_examples=len(order),
               elapsed_s=time.monotonic()-started,
               peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['teacher', 'smoke', 'train'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--variant', choices=['natural', 'symbolic'], required=True)
    parser.add_argument('--detail', choices=['minimal', 'typed', 'typed_examples'], required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--token-seal', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Existing pinned HF revision; offline only')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--teacher', type=Path, help='Completed teacher output for smoke/train')
    parser.add_argument('--smoke-report', type=Path, help='Discarded smoke complete.json for final train')
    parser.add_argument('--max-teacher-cases', type=int, default=0, help='0 means all train logical cases')
    parser.add_argument('--max-new-tokens', type=int, default=384)
    parser.add_argument('--max-train-cases', type=int, default=0, help='0 means all accepted logical cases and rotations')
    args = parser.parse_args()
    require(args.max_teacher_cases >= 0 and args.max_train_cases >= 0 and args.max_new_tokens > 0,
            'Limits must be nonnegative; token budget must be positive')
    require(args.command == 'teacher' or args.teacher is not None, 'Training requires --teacher')
    from prepare_accuracy_study import validate_dataset
    validate_dataset(args.data)  # Also enforce frozen threshold/template allocation and pairing.
    cases, rows, provenance = load_data(args.data, args.variant, args.tokens, args.token_seal, args.detail)
    provenance['trainer_sha256'] = digest(Path(__file__))
    provenance['checkpoint_sha256'] = checkpoint_identity(args.checkpoint)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    validate_tokenizer(tokenizer, rows)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.command == 'teacher':
        run_teacher(args, cases, rows, provenance, tokenizer)
    else:
        run_training(args, cases, rows, provenance, tokenizer)


if __name__ == '__main__':
    main()
