#!/usr/bin/env python3
"""Audit and export the independently evaluated classifier into the visual demo."""
import argparse
import hashlib
import json
import shutil
import time
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--live-url', help='Optional loopback server: verify every gallery prediction and record actual HTTP latency')
    args = p.parse_args()
    report = json.loads((args.run / 'report.json').read_text())
    split = json.loads((args.run / 'split.json').read_text())
    assert hashlib.sha256(args.archive.read_bytes()).hexdigest() == split['source_sha256']
    groups = {}
    hashes = {}
    for row in split['records']:
        partition = 'gallery' if row['split'] == 'excluded_gallery_neighbor' else row['split']
        for table, key in [(groups, row['group']), (hashes, row['sha256'])]:
            assert key not in table or table[key] == partition, 'Split leakage'
            table[key] = partition
    observations = {}
    for partition in ['validation', 'test', 'gallery']:
        rows = json.loads((args.run / f'{partition}-observations.json').read_text())
        expected = {r['name']: r for r in split['records'] if r['split'] == partition}
        assert len(rows) == len(expected)
        assert {r['image'] for r in rows} == set(expected)
        matrix = [[0] * 6 for _ in range(6)]
        for row in rows:
            assert row['image_sha256'] == expected[row['image']]['sha256']
            assert row['ground_truth'] == expected[row['image']]['label']
            assert max(row['scores'], key=lambda s: s['option_probability'])['id'] == row['selected']
            matrix[report['classes'].index(row['ground_truth'])][report['classes'].index(row['selected'])] += 1
            row['latency_ms'] = None  # Batched feature throughput is not per-request latency.
        assert matrix == report['evaluations'][partition]['confusion_matrix']
        assert sum(matrix[i][i] for i in range(6)) == report['evaluations'][partition]['correct']
        if partition != 'validation':
            observations.update({r['image']: r for r in rows})
    latencies = []
    if args.live_url:
        assert args.live_url.startswith(('http://127.0.0.1:', 'http://localhost:'))
        with zipfile.ZipFile(args.archive) as archive:
            for r in split['records']:
                if r['split'] != 'gallery':
                    continue
                data = archive.read(r['name'])
                started = time.perf_counter()
                req = urllib.request.Request(args.live_url + '/classify', data=data, headers={'Content-Type': 'image/jpeg'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    live = json.load(response)
                elapsed = (time.perf_counter() - started) * 1000
                row = observations[r['name']]
                assert live['classifier_sha256'] == report['classifier_sha256']
                assert live['image_sha256'] == r['sha256']
                assert live['selected'] == row['selected'], r['name']
                row['latency_ms'] = elapsed
                latencies.append(elapsed)
        latencies.sort()
        report['http_verification'] = {'images': len(latencies), 'matching_predictions': len(latencies), 'p50_ms': latencies[len(latencies)//2], 'p95_ms': latencies[int(len(latencies)*0.95)], 'includes': 'HTTP + decoding + preprocessing + two image views + classifier; model load excluded; first request included'}
    out = ROOT / 'web/static/trashnet'
    test_records = []
    with zipfile.ZipFile(args.archive) as archive:
        for r in split['records']:
            if r['split'] == 'test':
                data = archive.read(r['name'])
                assert hashlib.sha256(data).hexdigest() == r['sha256']
                target = out / 'test-images' / Path(r['name']).name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                test_records.append({**r, 'image_url': '/trashnet/test-images/' + target.name})
    payload = {'model': {'id': 'siglip2-trained', 'name': 'SigLIP2 + trained classifier', 'recorded_at': '2026-09-27', 'runtime': report['runtime'], 'supervised': True, 'observations': observations}, 'report': report, 'test_records': test_records}
    (out / 'improvement.json').write_text(json.dumps(payload, separators=(',', ':')) + '\n')
    notice = out / 'THIRD_PARTY_NOTICE.txt'
    existing = notice.read_text()
    notice.write_text('The 120 historical images in images/ and 481 held-out images in test-images/ are from Gary Thung\'s TrashNet.\nPaths and SHA-256 hashes are in recorded.json and improvement.json.\n' + existing[existing.index('Source:'):])
    evidence = ROOT / 'benchmarks/trashnet-trained-20260927'
    evidence.mkdir(exist_ok=True)
    for name in ['report.json', 'selection.json', 'split.json', 'validation-observations.json', 'test-observations.json', 'gallery-observations.json']:
        shutil.copyfile(args.run / name, evidence / name)
    (evidence / 'serving-verification.json').write_text(json.dumps(report.get('http_verification', {}), indent=2) + '\n')
    print(json.dumps({'exported': len(observations), 'splits': dict(Counter(r['split'] for r in split['records'])), 'http': report.get('http_verification')}, indent=2))


if __name__ == '__main__':
    main()
