"""Offline model loading shared by all Jev LoRA profiles; no remote Python code."""
import hashlib
import json
import re
from pathlib import Path

from .common import digest, require

DEFAULT_PROFILES = Path(__file__).with_name('jev_model_profiles.json')


def read_profile(name, path=None):
    registry = json.loads((path or DEFAULT_PROFILES).read_text(encoding='utf-8'))
    require(registry.get('schema_version') == 1, 'Unsupported profile registry schema')
    require(re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name), 'Invalid profile name')
    require(name in registry['models'], f'Unknown model profile: {name}')
    profile = dict(registry['models'][name], name=name)
    require(re.fullmatch('[0-9a-f]{40}', profile['revision']), 'Pin a full HF commit revision')
    require(profile['loader'] in ('AutoModelForCausalLM', 'AutoModelForImageTextToText'), 'Unsupported safe Auto loader')
    require(profile['target_modules'] and all(isinstance(s, str) and s for s in profile['target_modules']), 'Missing adapter targets')
    require(isinstance(profile['gguf'], str) and profile['gguf'].endswith('.gguf') and
            '/' not in profile['gguf'] and '\\' not in profile['gguf'] and ':' not in profile['gguf'],
            'GGUF must be a filename within gguf-root')
    return profile


def checkpoint_identity(checkpoint, profile):
    checkpoint = Path(checkpoint).resolve()
    config = json.loads((checkpoint/'config.json').read_text(encoding='utf-8'))
    require(checkpoint.name == profile['revision'] or config.get('_commit_hash') == profile['revision'],
            'Checkpoint directory/config must identify the pinned HF revision')
    require(config['model_type'] == profile['model_type'], 'Checkpoint/profile architecture mismatch')
    files = sorted(p for p in checkpoint.iterdir() if p.is_file() and
                   (p.suffix in ('.json', '.safetensors', '.jinja') or p.name == 'tokenizer.model'))
    require(any(p.suffix == '.safetensors' for p in files), 'Offline checkpoint weights missing')
    return hashlib.sha256(json.dumps({p.name:digest(p) for p in files}, sort_keys=True).encode()).hexdigest()


def target_names(model, profile):
    suffixes = set(profile['target_modules'])
    names = [name for name, module in model.named_modules()
             if name.rsplit('.', 1)[-1] in suffixes and hasattr(module, 'weight')
             and not any(part in name.lower() for part in ('vision', 'audio', 'projector'))]
    require(names, 'No language-model adapter targets matched this profile')
    require({name.rsplit('.', 1)[-1] for name in names} == suffixes, 'Some adapter target modules are missing')
    return names


def attach_adapter(model, profile, protocol):
    import torch
    from peft import LoraConfig, get_peft_model
    for param in model.parameters():
        param.requires_grad_(False)
        if param.is_floating_point() and param.ndim < 2:
            param.data = param.data.float()
    model.config.use_cache = False
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(r=protocol['rank'], lora_alpha=protocol['alpha'],
        lora_dropout=protocol['dropout'], bias='none', task_type='CAUSAL_LM',
        target_modules=target_names(model, profile)))
    for name, param in model.named_parameters():
        if param.requires_grad:
            require('lora_' in name, 'Unexpected trainable base weight')
            param.data = param.data.float()
    return model


def load_model(checkpoint, profile, protocol):
    import torch
    import transformers
    from .prepare_jev_data import SEED
    require(torch.cuda.is_available(), 'This NF4 training path requires a CUDA GPU')
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(4)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    loader = getattr(transformers, profile['loader'])
    model = loader.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False,
              quantization_config=quant, device_map={'': 'cuda:0'}, dtype=dtype, attn_implementation='sdpa')
    return attach_adapter(model, profile, protocol)
