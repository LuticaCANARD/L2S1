#!/usr/bin/env python3
"""Apple Silicon (MLX) variant of train_jev_lora: same sealed tokens, loss and PEFT-format adapter."""
import argparse
import json
import math
import random
import time
from pathlib import Path

from .common import digest, require, validate_tokenizer, write_json
from .jev_model_profiles import checkpoint_identity, read_profile
from .prepare_jev_data import SEED
from .train_jev_lora import training_rows


def lora_keys(block, suffixes):
    keys = {name for name, _ in block.named_modules() if name.rsplit('.', 1)[-1] in suffixes}
    require({k.rsplit('.', 1)[-1] for k in keys} == set(suffixes), 'Some adapter target modules are missing')
    return keys


def checkpoint_modules(checkpoint):
    """Language-model module names from the HF safetensors headers (no weights read)."""
    names = set()
    for path in Path(checkpoint).glob('*.safetensors'):
        with path.open('rb') as f:
            header = json.loads(f.read(int.from_bytes(f.read(8), 'little')))
        names |= {k.removesuffix('.weight') for k in header if k.endswith('.weight')}
    return {n for n in names if not any(part in n.lower() for part in ('vision', 'audio', 'projector'))}


def hf_module(module, hf_modules):
    """mlx-lm renames prefixes (e.g. Gemma 4 language_model.model.layers vs HF
    model.language_model.layers); match on the layers.N.<path> suffix instead."""
    suffix = module[module.rindex('layers.'):]
    matches = [n for n in hf_modules if n == suffix or n.endswith('.'+suffix)]
    require(len(matches) == 1, f'No unique checkpoint tensor for {module}')
    return matches[0]


def peft_adapter(model, protocol, targets, hf_modules):
    """MLX lora_a (in, r) / lora_b (r, out) at scale alpha/r -> PEFT lora_A (r, in) / lora_B (out, r)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    tensors = {}
    for name, value in tree_flatten(model.trainable_parameters()):
        module, kind = name.rsplit('.', 1)
        require(kind in ('lora_a', 'lora_b'), f'Unexpected trainable parameter: {name}')
        module = hf_module(module, hf_modules)
        tensors[f'base_model.model.{module}.lora_{kind[-1].upper()}.weight'] = value.T.astype(mx.float32)
    config = dict(peft_type='LORA', task_type='CAUSAL_LM', r=protocol['rank'], lora_alpha=protocol['alpha'],
                  lora_dropout=protocol['dropout'], bias='none', target_modules=sorted(targets))
    return tensors, config


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
                   trainer_sha256=digest(Path(__file__)), backend='mlx',
                   data_code_sha256=digest(Path(__file__).with_name('prepare_jev_data.py')),
                   common_code_sha256=digest(Path(__file__).with_name('common.py')))
    smoke = args.command == 'smoke'
    if not smoke:
        require(args.smoke_report is not None, 'Discarded smoke required')
        report = json.loads(args.smoke_report.read_text(encoding='utf-8'))
        require(report['complete'] and report['smoke'] and not report['adapter_saved'] and report['binding'] == binding,
                'Smoke binding mismatch')
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx_lm
    from mlx_lm.tuner.trainer import grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx.utils import tree_flatten, tree_map
    protocol = binding['protocol']
    # Lazy load: bf16 weights stream through quantization instead of all being resident.
    model, tokenizer = mlx_lm.load(str(args.checkpoint), lazy=True)
    validate_tokenizer(getattr(tokenizer, '_tokenizer', tokenizer), rows)
    args.output.mkdir(parents=True, exist_ok=False)
    mx.random.seed(SEED)
    # 4-bit base like the CUDA NF4 path; only the float LoRA matrices train.
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda _, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    mx.eval(model.parameters())
    model.freeze()
    grad_checkpoint(model.layers[0])
    keys = lora_keys(model.layers[0], profile['target_modules'])
    linear_to_lora_layers(model, len(model.layers), dict(rank=protocol['rank'], dropout=protocol['dropout'],
                          scale=protocol['alpha']/protocol['rank'], keys=keys))
    targets = {k.rsplit('.', 1)[-1] for k in keys}
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    require(trainable, 'No trainable parameters')
    model.train()
    random.Random(SEED).shuffle(rows)
    if smoke:
        rows = rows[:12]
    accumulation, lr = protocol['gradient_accumulation'], protocol['learning_rate']
    steps = math.ceil(len(rows)/accumulation)
    optimizer = optim.AdamW(learning_rate=lr, weight_decay=0)

    def loss_fn(model, input_ids, candidate_ids, target):
        logits = model(input_ids[None])[0, -1].astype(mx.float32)
        candidate = logits[candidate_ids]
        ce = -(target * (candidate - mx.logsumexp(candidate))).sum()
        mass = mx.logsumexp(logits) - mx.logsumexp(candidate)
        return ce + protocol['mass_loss_weight']*mass, (ce, mass)

    step_grad = nn.value_and_grad(model, loss_fn)
    write_json(args.output/'metadata.json', dict(binding=binding, smoke=smoke, seed=SEED, device=str(mx.default_device()),
        mlx=mx.__version__, mlx_lm=mlx_lm.__version__, training_examples=len(rows), trainable_parameters=trainable))
    started = time.monotonic()
    sums, grads = [0., 0.], None
    with (args.output/'training.jsonl').open('x') as log:
        for i, row in enumerate(rows):
            batch = (mx.array(row['input_ids']), mx.array(row['candidate_ids']), mx.array(row['target'], dtype=mx.float32))
            (loss, (ce, mass)), grad = step_grad(model, *batch)
            require(math.isfinite(loss.item()), 'Nonfinite loss')
            size = min(accumulation, len(rows)-(i//accumulation)*accumulation)
            grad = tree_map(lambda g: g/size, grad)
            grads = grad if grads is None else tree_map(mx.add, grads, grad)
            sums[0] += ce.item()
            sums[1] += mass.item()
            if (i+1) % accumulation == 0 or i+1 == len(rows):
                grads, norm = optim.clip_grad_norm(grads, 1.)
                require(math.isfinite(norm.item()), 'Nonfinite gradient')
                optimizer.update(model, grads)
                mx.eval(model.trainable_parameters(), optimizer.state)
                grads = None
                step = i//accumulation+1
                optimizer.learning_rate = lr*(1-step/steps)
                info = dict(step=step, examples=i+1, soft_ce=sums[0]/size, mass_loss=sums[1]/size,
                            grad_norm=norm.item(), elapsed_s=time.monotonic()-started,
                            peak_metal_gib=mx.get_peak_memory()/1024**3)
                log.write(json.dumps(info)+'\n')
                log.flush()
                print(json.dumps(info), flush=True)
                sums = [0., 0.]
    tensors, config = peft_adapter(model, protocol, targets, checkpoint_modules(args.checkpoint))
    require(any(mx.any(v != 0).item() for k, v in tensors.items() if 'lora_B' in k), 'No LoRA update')
    if not smoke:
        (args.output/'adapter').mkdir()
        mx.save_safetensors(str(args.output/'adapter/adapter_model.safetensors'), tensors)
        write_json(args.output/'adapter/adapter_config.json', dict(config, base_model_name_or_path=str(args.checkpoint)))
    write_json(args.output/'complete.json', dict(complete=True, smoke=smoke, adapter_saved=not smoke,
        binding=binding, steps=steps, examples=len(rows), elapsed_s=time.monotonic()-started,
        peak_metal_gib=mx.get_peak_memory()/1024**3))


if __name__ == '__main__':
    main()
