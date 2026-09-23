#!/usr/bin/env python3
"""Download a pinned model plan or run it serially on the benchmark host."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.parse


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def download(row):
    path = pathlib.Path(row['path'])
    if 'repo' not in row:
        return
    ready = path.with_suffix('.ready.json')
    failed = path.with_suffix('.failed.json')
    if ready.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix('.part')
    url = 'https://huggingface.co/{}/resolve/{}/{}?download=true'.format(
        row['repo'], row['revision'], urllib.parse.quote(row['file']))
    print('DOWNLOAD', path.name, flush=True)
    try:
        with path.with_suffix('.download.log').open('w') as log:
            subprocess.run(['curl', '-fL', '--http1.1', '--retry', '5', '--retry-all-errors', '--retry-delay', '3',
                            '--connect-timeout', '30', '--max-time', '14400',
                            '-C', '-', '-o', str(part), url], stderr=log, check=True)
        with part.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if part.stat().st_size != row['size'] or digest != row['sha256']:
            raise ValueError('Downloaded checkpoint size or SHA256 differs from pinned LFS metadata')
        part.replace(path)
        save(ready, row)
        print('READY', path.name, flush=True)
    except Exception as exc:
        save(failed, dict(row, error=str(exc)))
        print('DOWNLOAD_FAILED', path.name, str(exc), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['download', 'run'])
    parser.add_argument('--plan', type=pathlib.Path, required=True)
    parser.add_argument('--source', type=pathlib.Path, required=True)
    parser.add_argument('--upstream', type=pathlib.Path, required=True)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.plan.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'download':
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(download, rows))
        return
    with (args.output / 'matrix.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        results_path = args.output / 'matrix-status.json'
        results = json.loads(results_path.read_text()) if results_path.exists() else []
        done = {r['id'] for r in results}
        pending = [r for r in rows if r['id'] not in done]
        deadline = time.monotonic() + 21600
        while pending:
            progress = False
            for row in pending[:]:
                model = pathlib.Path(row['path'])
                if 'repo' in row and not model.with_suffix('.ready.json').exists():
                    if model.with_suffix('.failed.json').exists():
                        results.append(dict(row, status='download_failed'))
                        pending.remove(row)
                        save(results_path, results)
                    continue
                if not model.exists():
                    continue
                out = args.output / row['id']
                command = ['python3', '-u', str(args.source / 'scripts/jevbench_public.py'),
                           '--jevbench', str(args.upstream),
                           '--evaluator', str(args.source / 'target/release/examples/evaluate_jsonl'),
                           '--model', str(model), '--output', str(out), '--context', '8192']
                print('RUN', row['id'], flush=True)
                started = time.time()
                with (args.output / (row['id'] + '.log')).open('x') as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                            env={**os.environ, **row.get('environment', {})})
                manifest_path = out / 'manifest.json'
                if manifest_path.exists():
                    manifest = json.loads(manifest_path.read_text())
                    manifest['runtime_environment_overrides'] = row.get('environment', {})
                    save(manifest_path, manifest)
                record = dict(row, status='complete' if result.returncode == 0 else 'failed',
                              exit_code=result.returncode, elapsed_s=time.time() - started,
                              started_unix=started, command=command)
                if (out / 'summary.json').exists():
                    summary = json.loads((out / 'summary.json').read_text())
                    record['n_correct'] = summary['n_correct']
                    record['n_scorable'] = summary['n_scorable']
                    record['errors'] = summary['selective_policy']['error']
                results.append(record)
                save(results_path, results)
                pending.remove(row)
                progress = True
                print('RESULT', json.dumps(record), flush=True)
            if not progress and pending:
                if time.monotonic() > deadline:
                    raise TimeoutError('Model downloads did not finish within six hours')
                time.sleep(5)


if __name__ == '__main__':
    main()
