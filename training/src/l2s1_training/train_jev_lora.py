#!/usr/bin/env python3
"""Train any registered local decision model on frozen Jev-type probability targets."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

from .prepare_jev_data import SEED, check_disjoint, distribution, validate_dataset
from .jev_model_profiles import checkpoint_identity, load_model, read_profile
from .common import (digest,
                                 normalized_token_identity, option_specs, read_jsonl,
                                 require, validate_tokenizer, write_json)


def training_rows(data, tokens):
    manifest = validate_dataset(data)
    require(manifest['schema_version'] == 1 and manifest['seed'] == SEED, 'Invalid dataset manifest')
    seal = json.loads(tokens.with_suffix('.seal.json').read_text(encoding='utf-8'))
    require(seal['tokens_sha256'] == digest(tokens) and seal['manifest_sha256'] == digest(data/'manifest.json'),
            'Token seal mismatch')
    require(seal['requests_sha256'] == digest(data/'train-token-requests.jsonl') == manifest['token_requests_sha256'],
            'Exported request binding mismatch')
    protocol = manifest['protocol']
    require(seal.get('prompt_layout') == protocol['prompt_layout'] and
            seal.get('prompt_detail') == protocol['prompt_detail'], 'Token prompt settings differ from the protocol')
    # Seals written before the exporter took --context used the default 2048.
    context = seal.get('context', 2048)
    require(type(context) is int and 0 < context <= 2**31-1, 'Invalid token context')
    splits = {}
    for name, config in manifest['splits'].items():
        path = data/f'{name}.jsonl'
        require(digest(path) == config['sha256'], 'Dataset hash mismatch')
        splits[name] = read_jsonl(path)
        require([r['id'] for r in splits[name]] == config['ids'], 'Dataset IDs mismatch')
    check_disjoint(splits)
    expected = {}
    for case in splits['train']:
        for d in case['request']['decisions']:
            key = f'{case["id"]}/{d["id"]}'
            expected[key] = (d, case['gold'][d['id']])
    rows = read_jsonl(tokens)
    seen, selected, identities = set(), [], set()
    for row in rows:
        require(row['id'] in expected, 'Non-training token row')
        d, gold = expected[row['id']]
        ids = [o['id'] for o in option_specs(d)]
        width, rotation = len(ids), row['code_rotation']
        require(row['decision_id'] == d['id'] and row['option_ids'] == ids, 'Token semantic mapping mismatch')
        require(type(rotation) is int and 0 <= rotation < width, 'Invalid rotation')
        require((row['id'], rotation) not in seen, 'Duplicate rotation')
        seen.add((row['id'], rotation))
        require(row['candidate_codes'] == [chr(65+(i+width-rotation)%width) for i in range(width)], 'Code mapping mismatch')
        require(len(row['candidate_ids']) == len(set(row['candidate_ids'])) == width and
                0 < len(row['input_ids']) <= context, 'Invalid token count')
        require(all(type(i) is int and i >= 0 for i in row['input_ids']+row['candidate_ids']), 'Invalid token ID')
        identity = row['model_identity']
        require(identity.get('adapter_sha256') is None and identity.get('head_sha256') is None, 'Export must use base')
        require(identity['weights_sha256'] == seal['model_sha256'], 'Token model hash mismatch')
        identities.add(json.dumps(normalized_token_identity(identity, rotation,
            protocol['prompt_detail'].replace('-', '_'), protocol['prompt_layout']), sort_keys=True))
        chosen = int(hashlib.sha256(f'{SEED}:{row["id"]}'.encode()).hexdigest(), 16) % width
        if rotation == chosen:
            selected.append(dict(row, target=distribution(gold['probabilities'], ids)))
    require(seen == {(k, r) for k, (d, _) in expected.items() for r in range(len(option_specs(d)))}, 'Missing exported rotations')
    require(len(identities) == 1 and len(selected) == len(expected), 'Incomplete or mixed token export')
    return selected, dict(manifest_sha256=digest(data/'manifest.json'), tokens_sha256=digest(tokens),
                         token_seal=seal, native_identity=json.loads(next(iter(identities))), protocol=manifest['protocol'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['smoke', 'train'])
    p.add_argument('--model', required=True, help='Registered model key (or custom profile key)')
    p.add_argument('--profiles', type=Path, help='Optional custom model profile registry')
    for name in ('data', 'tokens', 'checkpoint', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--smoke-report', type=Path)
    args = p.parse_args()
    profile = read_profile(args.model, args.profiles)
    rows, binding = training_rows(args.data, args.tokens)
    binding.update(checkpoint_sha256=checkpoint_identity(args.checkpoint, profile), model_profile=profile,
                   trainer_sha256=digest(Path(__file__)),
                   loader_sha256=digest(Path(__file__).with_name('jev_model_profiles.py')),
                   data_code_sha256=digest(Path(__file__).with_name('prepare_jev_data.py')),
                   common_code_sha256=digest(Path(__file__).with_name('common.py')))
    smoke = args.command == 'smoke'
    if not smoke:
        require(args.smoke_report is not None, 'Discarded smoke required')
        report = json.loads(args.smoke_report.read_text(encoding='utf-8'))
        require(report['complete'] and report['smoke'] and not report['adapter_saved'] and report['binding'] == binding,
                'Smoke binding mismatch')
    import torch
    import transformers
    import peft
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    validate_tokenizer(tokenizer, rows)
    args.output.mkdir(parents=True, exist_ok=False)
    model = load_model(args.checkpoint, profile, binding['protocol'])
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    require(params, 'No trainable parameters')
    random.Random(SEED).shuffle(rows)
    if smoke:
        rows = rows[:12]
    protocol = binding['protocol']
    accumulation, lr = protocol['gradient_accumulation'], protocol['learning_rate']
    steps = math.ceil(len(rows)/accumulation)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    write_json(args.output/'metadata.json', dict(binding=binding, smoke=smoke, seed=SEED,
        adapter_initialization_seed=SEED, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, transformers=transformers.__version__, peft=peft.__version__,
        training_examples=len(rows), trainable_parameters=sum(p.numel() for p in params)))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    sums = [0., 0.]
    with (args.output/'training.jsonl').open('x') as log:
        for i, row in enumerate(rows):
            logits = model(input_ids=torch.tensor([row['input_ids']], device='cuda'),
                           use_cache=False, logits_to_keep=1).logits[0, -1].float()
            candidate = logits[row['candidate_ids']]
            target = torch.tensor(row['target'], device='cuda')
            ce = -(target * torch.log_softmax(candidate, dim=0)).sum()
            mass = torch.logsumexp(logits, 0)-torch.logsumexp(candidate, 0)
            loss = ce + protocol['mass_loss_weight']*mass
            require(torch.isfinite(loss).item(), 'Nonfinite loss')
            size = min(accumulation, len(rows)-(i//accumulation)*accumulation)
            (loss/size).backward()
            sums[0] += ce.item()
            sums[1] += mass.item()
            if (i+1) % accumulation == 0 or i+1 == len(rows):
                grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step = i//accumulation+1
                optimizer.param_groups[0]['lr'] = lr*(1-step/steps)
                info = dict(step=step, examples=i+1, soft_ce=sums[0]/size, mass_loss=sums[1]/size,
                            grad_norm=grad.item(), elapsed_s=time.monotonic()-started,
                            peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3)
                log.write(json.dumps(info)+'\n')
                log.flush()
                print(json.dumps(info), flush=True)
                sums = [0., 0.]
    require(any(torch.count_nonzero(p).item() for n, p in model.named_parameters() if 'lora_B' in n), 'No LoRA update')
    if not smoke:
        model.save_pretrained(args.output/'adapter')
        tokenizer.save_pretrained(args.output/'adapter')
    write_json(args.output/'complete.json', dict(complete=True, smoke=smoke, adapter_saved=not smoke,
        binding=binding, steps=steps, examples=len(rows), elapsed_s=time.monotonic()-started,
        peak_cuda_gib=torch.cuda.max_memory_allocated()/1024**3))


if __name__ == '__main__':
    main()
