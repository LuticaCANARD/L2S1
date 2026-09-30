"""Supported command line entry point; ML dependencies are loaded on demand."""
import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from . import __version__
from .jev_model_profiles import DEFAULT_PROFILES, read_profile

COMMANDS = {
    'prepare': ('prepare_jev_data', 'Freeze Jev JSONL splits or reproduce the pinned pilot'),
    'fetch': ('run_jev_lora', 'Download pinned checkpoints explicitly'),
    'run': ('run_jev_lora', 'Export, smoke, train, convert and evaluate one model or a matrix'),
    'report': ('report_jev', 'Render Jev answers and score held-out native predictions'),
    'regression': ('report_jev_rule_regression', 'Prepare or score the decision-rules regression'),
    'verify-ollaya': ('verify_jev_ollaya', 'Check answer parity against pinned Ollaya source'),
    'profiles': (None, 'List the bundled or custom pinned model profiles'),
    'doctor': (None, 'Check optional dependencies, CUDA/MLX and native tool capabilities'),
}


def doctor(argv):
    import subprocess
    p = argparse.ArgumentParser(prog='l2s1-train doctor')
    p.add_argument('--cuda', action='store_true', help='Require the complete CUDA training stack')
    p.add_argument('--mlx', action='store_true', help='Require the Apple Silicon MLX training stack')
    p.add_argument('--exporter', type=Path)
    p.add_argument('--evaluator', type=Path)
    p.add_argument('--converter', type=Path)
    a = p.parse_args(argv)
    checks = []
    cuda_stack = ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate')
    for package in cuda_stack + ('mlx', 'mlx-lm', 'huggingface-hub', 'pyarrow'):
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = None
        checks.append(dict(name=package, ok=version is not None, version=version,
                           required=(a.cuda and package in cuda_stack + ('huggingface-hub',))
                                    or (a.mlx and package in ('mlx', 'mlx-lm'))))
    if a.mlx:
        try:
            import mlx.core as mx
            from mlx_lm.tuner.utils import linear_to_lora_layers
            checks.append(dict(name='metal', ok=mx.metal.is_available(), required=True))
        except Exception as error:
            checks.append(dict(name='training-imports', ok=False, required=True, error=str(error)))
    if a.cuda:
        try:
            import torch
            ok = torch.cuda.is_available()
            checks.append(dict(name='cuda', ok=ok, required=True,
                               device=torch.cuda.get_device_name() if ok else None))
            # Import errors and incompatible architectures are actionable before a long export.
            import transformers
            import peft
            import bitsandbytes
            checks.append(dict(name='training-imports', ok=True, required=True))
        except Exception as error:
            checks.append(dict(name='training-imports', ok=False, required=True, error=str(error)))
    for name, path, flags in (
        ('exporter', a.exporter, ['--all-rotations', '--prompt-detail']),
        ('evaluator', a.evaluator, ['--lora', '--request-batch-size', '--model-load-mode', '--parallel-prefix-alignment']),
    ):
        if path:
            try:
                result = subprocess.run([str(path.resolve()), '--help'], capture_output=True, text=True, timeout=15)
                missing = [flag for flag in flags if flag not in result.stdout]
                checks.append(dict(name=name, ok=result.returncode == 0 and not missing,
                                   required=True, missing_flags=missing, exit_code=result.returncode))
            except (OSError, subprocess.TimeoutExpired) as error:
                checks.append(dict(name=name, ok=False, required=True, error=str(error)))
    if a.converter:
        checks.append(dict(name='converter', ok=a.converter.is_file(), required=True))
    ok = all(c['ok'] for c in checks if c['required'])
    print(json.dumps(dict(schema_version=1, version=__version__, ok=ok, checks=checks,
        scope='Environment/capability checks only; no pretrained-model inference or conversion validation.'), indent=2))
    return 0 if ok else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog='l2s1-train', description='L2S1 Jev-type training and evaluation')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command', required=True)
    for name, (_, description) in COMMANDS.items():
        sub.add_parser(name, help=description, add_help=False)
    # Each command owns its argument parser, including --help.
    if not argv or argv[0] not in COMMANDS:
        p.parse_args(argv)
        return
    command, rest = argv[0], argv[1:]
    try:
        if command == 'doctor':
            raise SystemExit(doctor(rest))
        if command == 'profiles':
            pp = argparse.ArgumentParser(prog='l2s1-train profiles')
            pp.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
            a = pp.parse_args(rest)
            registry = json.loads(a.profiles.read_text(encoding='utf-8'))
            print(json.dumps(dict(schema_version=1, models=[read_profile(n, a.profiles) for n in registry['models']]), indent=2))
            return
        module = importlib.import_module('.'+COMMANDS[command][0], __package__)
        previous = sys.argv
        try:
            sys.argv = ['l2s1-train '+command] + rest
            if command in ('fetch', 'run'):
                module.main(command=command)
            else:
                module.main()
        finally:
            sys.argv = previous
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        print(f'l2s1-train: {error}', file=sys.stderr)
        raise SystemExit(1) from None
