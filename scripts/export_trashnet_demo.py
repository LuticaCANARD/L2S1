#!/usr/bin/env python3
"""Export verified original images and baseline observations for the visual demo."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / 'benchmarks/trashnet-vision-20260925'
OUT = ROOT / 'web/static/trashnet'


def export(archive: Path):
    selection = json.loads((BENCH / 'selection.json').read_text())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != selection['source_sha256']:
        raise ValueError('TrashNet archive SHA-256 differs from the frozen benchmark')
    models = []
    for key, label in [('qwen3vl', 'Qwen3-VL 2B'), ('gemma4', 'Gemma 4 E2B'), ('smolvlm', 'SmolVLM 256M')]:
        rows = [json.loads(line) for line in (BENCH / key / 'observations.jsonl').read_text().splitlines()]
        assert len(rows) == len(selection['records']) == 120
        by_name = {row['image']: row for row in rows}
        assert len(by_name) == 120
        for item in selection['records']:
            row = by_name[item['name']]
            assert row['image_sha256'] == item['sha256'] and row['ground_truth'] == item['label']
        models.append({'id': key, 'name': label, 'summary': json.loads((BENCH / key / 'summary.json').read_text()), 'observations': by_name})
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for item in selection['records']:
            data = source.read(item['name'])
            assert hashlib.sha256(data).hexdigest() == item['sha256']
            name = Path(item['name']).name
            (OUT / 'images' / name).write_bytes(data)
            item['image_url'] = '/trashnet/images/' + name
    payload = {'recorded_at': '2026-09-25', 'runtime': 'RTX 3080 / CUDA / Q8_0', 'source': selection, 'models': models,
               'policy': {'min_top_probability': 0.8, 'min_candidate_mass': 0.05},
               'prompt': json.loads((BENCH / 'prompts.json').read_text())['prompts']['baseline']}
    (OUT / 'recorded.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
    notice = (ROOT / 'web/static/demo/THIRD_PARTY_NOTICE.txt').read_text()
    (OUT / 'THIRD_PARTY_NOTICE.txt').write_text('The 120 original images in images/ are from Gary Thung\'s TrashNet.\nPaths and per-image SHA-256 hashes are in recorded.json (source.records).\n' + notice[notice.index('Source:'):].replace('SHA-256: a0251609850efa59b9ba88efd0c02178ce2901df8dc5cdf0c960d38aadfd6a2a\n', 'Archive SHA-256: ' + selection['source_sha256'] + '\n'))
    print(f'Exported {len(selection["records"])} verified images and {len(models) * 120} observations to {OUT}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    export(parser.parse_args().archive)
