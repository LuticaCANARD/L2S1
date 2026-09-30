#!/usr/bin/env python3
"""Fetch pinned checkpoints or run the same Jev LoRA pipeline for one/all models.

No downloads occur in run mode. Each model has its own output, hashes, smoke,
adapter, conversion, paired native evaluation and failure record. GPUs run serially.
"""
import argparse
import datetime
import json
import os
import re
import signal
import math
import sys
from pathlib import Path

from .jev_model_profiles import DEFAULT_PROFILES, read_profile
from .common import digest, write_json, save_status
from .execution import execute_stage
from .prepare_jev_data import validate_dataset
from . import __version__


def seal_tokens(data, tokens, model, exporter):
    write_json(tokens.with_suffix('.seal.json'), dict(schema_version=1, split='train',
        manifest_sha256=digest(data/'manifest.json'), tokens_sha256=digest(tokens),
        requests_sha256=digest(data/'train-token-requests.jsonl'), model_sha256=digest(model),
        exporter_sha256=digest(exporter), prompt_layout='legacy', prompt_detail='minimal'))


def main(command=None):
    p = argparse.ArgumentParser(prog="l2s1-train "+command if command else None, description=__doc__)
    if command is None:
        p.add_argument('command', choices=['fetch', 'run'])
    else:
        p.set_defaults(command=command)
    p.add_argument('--models', nargs='+', default=['gemma4'], help='Profile names, or all')
    p.add_argument('--profiles', type=Path, default=DEFAULT_PROFILES)
    p.add_argument('--checkpoint-root', type=Path, required=True, help='Contains pinned HF revision directories')
    p.add_argument('--data', type=Path)
    p.add_argument('--gguf-root', type=Path, default=Path('models'))
    p.add_argument('--output', type=Path, required=True, help='New run directory; never overwritten')
    p.add_argument('--exporter', type=Path, default=Path('target/release/examples/export_decision_tokens'))
    p.add_argument('--evaluator', type=Path, default=Path('target/release/examples/evaluate_jsonl'))
    p.add_argument('--converter', type=Path, help='Matching llama.cpp convert_lora_to_gguf.py')
    p.add_argument('--library-path', type=Path, help='Matching llama.cpp shared libraries')
    p.add_argument('--stage-timeout', type=float, default=7200, help='Maximum seconds per stage (default: 7200)')
    p.add_argument('--backend', choices=['cuda', 'mlx'], default='cuda',
                   help='Training/evaluation device: CUDA (NF4 + PEFT) or Apple Silicon (MLX + Metal)')
    p.add_argument('--eval-execution', choices=['fresh', 'parallel'], default='fresh',
                   help='fresh reproduces the pilot; parallel batches each case\'s decisions and shares '
                        'their prompt prefix (faster with state-first prompts; probabilities can differ slightly)')
    p.add_argument('--rules-fixture', type=Path, help='Optional existing decision-rules regression fixture')
    a = p.parse_args()
    if not math.isfinite(a.stage_timeout) or a.stage_timeout <= 0:
        p.error('--stage-timeout must be finite and positive')
    names = list(json.loads(a.profiles.read_text(encoding='utf-8'))['models']) if a.models == ['all'] else a.models
    if not names or any(re.fullmatch(r'[a-z0-9][a-z0-9_-]*', n) is None for n in names):
        p.error('Model profile keys must be safe lowercase names without path separators')
    profiles = [read_profile(n, a.profiles) for n in names]
    if len(set(names)) != len(names):
        p.error('Model names must be unique')
    if a.command == 'run' and (a.data is None or a.converter is None):
        p.error('run requires --data and --converter')
    a.output.mkdir(parents=True, exist_ok=False)
    scripts = Path(__file__).resolve().parent
    env = dict(os.environ)
    env['PYTHONPATH'] = str(scripts.parent) + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    def cancelled(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, cancelled)
    if a.library_path:
        library_variable = 'PATH' if os.name == 'nt' else 'LD_LIBRARY_PATH'
        env[library_variable] = str(a.library_path.resolve()) + (os.pathsep+env[library_variable] if env.get(library_variable) else '')
    statuses = [dict(model=p['name'], profile=p, status='not_started', stages=[]) for p in profiles]
    for item in statuses:
        (a.output/item['model']).mkdir()
    def persist():
        for item in statuses:
            save_status(a.output/item['model']/'run.json', item)
        save_status(a.output/'summary.json', dict(schema_version=1, package_version=__version__,
            command=a.command, stage_timeout_s=a.stage_timeout, runs=statuses,
            scope='ok means pipeline completed, not accuracy improved. Only requested models were run.'))
    persist()
    interrupted = False
    for profile, status in zip(profiles, statuses):
        out = a.output/profile['name']
        checkpoint = (a.checkpoint_root/profile['revision']).resolve()
        status.update(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='running')
        persist()

        def stage(name, cmd):
            execute_stage(name, cmd, out, env, status, persist, a.stage_timeout)
            print(profile['name'], name, 'ok', flush=True)

        try:
            if a.command == 'fetch':
                stage('fetch', [sys.executable, '-m', 'l2s1_training.fetch',
                      '--model', profile['name'], '--profiles', a.profiles, '--checkpoint', checkpoint])
                status['checkpoint'] = str(checkpoint)
            else:
                model = (a.gguf_root/profile['gguf']).resolve()
                for path in (checkpoint/'config.json', model, a.exporter, a.evaluator, a.converter):
                    if not path.is_file():
                        raise FileNotFoundError(f'Missing local prerequisite: {path}')
                validate_dataset(a.data)
                status['artifacts'] = {k:digest(v) for k,v in dict(model=model, exporter=a.exporter,
                    evaluator=a.evaluator, converter=a.converter, manifest=a.data/'manifest.json').items()}
                tokens = out/'train-tokens.jsonl'
                stage('export', [a.exporter.resolve(), '--model', model, '--input', a.data/'train-token-requests.jsonl',
                      '--output', tokens, '--all-rotations', '--prompt-layout', 'legacy', '--prompt-detail', 'minimal'])
                seal_tokens(a.data, tokens, model, a.exporter)
                shared = ['--model', profile['name'], '--profiles', a.profiles, '--data', a.data,
                          '--tokens', tokens, '--checkpoint', checkpoint]
                trainer = 'l2s1_training.train_jev_mlx' if a.backend == 'mlx' else 'l2s1_training.train_jev_lora'
                stage('smoke', [sys.executable, '-m', trainer, 'smoke', *shared, '--output', out/'smoke'])
                stage('train', [sys.executable, '-m', trainer, 'train', *shared, '--output', out/'train',
                               '--smoke-report', out/'smoke/complete.json'])
                stage('convert', [sys.executable, a.converter.resolve(), out/'train/adapter', '--base', checkpoint,
                                  '--outfile', out/'adapter.gguf', '--outtype', 'f16'])
                status['artifacts']['adapter'] = digest(out/'adapter.gguf')
                if a.rules_fixture:
                    stage('prepare-regression', [sys.executable, '-m', 'l2s1_training.report_jev_rule_regression',
                        '--fixture', a.rules_fixture, '--requests-output', out/'rules-requests.jsonl'])
                layout = validate_dataset(a.data)['protocol']['prompt_layout']
                execution = ['--execution-mode', 'fresh'] if a.eval_execution == 'fresh' else [
                    '--execution-mode', 'parallel', '--parallel-width', '8', '--parallel-context-dynamic',
                    # Token sharing pays off only when prompts share the state; legacy leads with the question.
                    '--parallel-prefix-alignment', 'token' if layout == 'state-first' else 'batch']
                for variant in ('base', 'adapter'):
                    pred = out/f'{variant}-predictions.jsonl'
                    cmd = [a.evaluator.resolve(), '--model', model, '--input', a.data/'test-requests.jsonl', '--output', pred,
                           '--'+('metal' if a.backend == 'mlx' else 'cuda'), '--context', '8192', '--batch', '256', '--threads', '4', *execution,
                           '--prompt-layout', layout, '--prompt-detail', 'minimal', '--request-batch-size', '1',
                           '--model-load-mode', 'read', '--warmup']
                    if variant == 'adapter':
                        cmd += ['--lora', out/'adapter.gguf']
                    stage(variant, cmd)
                    stage(variant+'-report', [sys.executable, '-m', 'l2s1_training.report_jev', '--data', a.data,
                        '--predictions', pred, '--output', out/(variant+'-report')])
                    summary = json.loads((out/(variant+'-report')/'summary.json').read_text(encoding='utf-8'))
                    status.setdefault('metrics', {})[variant] = dict(summary['overall'], latency_ms=summary['latency_ms'],
                                                                    by_type=summary['by_type'])
                    if summary['failures']:
                        raise RuntimeError(f'{variant}: incomplete evaluation; failures retained in report')
                    if a.rules_fixture:
                        regression = out/f'{variant}-rules-predictions.jsonl'
                        cmd[cmd.index('--input')+1] = out/'rules-requests.jsonl'
                        cmd[cmd.index('--output')+1] = regression
                        stage(variant+'-rules', cmd)
                        stage(variant+'-rules-report', [sys.executable, '-m', 'l2s1_training.report_jev_rule_regression',
                            '--fixture', a.rules_fixture, '--predictions', regression,
                            '--output', out/f'{variant}-rules-report.json'])
                status['delta_after_minus_before'] = {k: status['metrics']['adapter'][k]-status['metrics']['base'][k]
                    for k in ('raw_accuracy', 'coverage', 'accepted_accuracy', 'correct_all', 'soft_kl', 'soft_brier', 'score_mae')
                    if status['metrics']['adapter'][k] is not None and status['metrics']['base'][k] is not None}
            status['status'] = 'ok'
        except KeyboardInterrupt:
            status.update(status='cancelled', error='Interrupted; partial artifacts retained')
            interrupted = True
        except Exception as error:
            status.update(status='timeout' if isinstance(error, TimeoutError) else 'failed', error=str(error))
            print(profile['name'], 'failed:', error, flush=True)
        status['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        persist()
        if interrupted:
            break
    if interrupted:
        sys.exit(130)
    if any(s['status'] != 'ok' for s in statuses):
        sys.exit(1)


if __name__ == '__main__':
    main()
