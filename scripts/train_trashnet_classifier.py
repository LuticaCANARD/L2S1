#!/usr/bin/env python3
"""Train a task-specific classifier without training on the visual demo or holdout.

Frozen SigLIP2 features + validation-selected sklearn classifier. No TrashNet-
finetuned public checkpoint is used. All split memberships and predictions are
exported; the historical 120-image gallery is separate from the fresh test set.
"""
import argparse
import hashlib
import io
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'google/siglip2-base-patch16-224'
REVISION = '75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2'
SEED = 20260927
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(archive):
    baseline = json.loads((ROOT / 'benchmarks/trashnet-vision-20260925/selection.json').read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == baseline['source_sha256']
    demo = {r['name'] for r in baseline['records']}
    records, hashes = [], []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            parts = Path(name).parts
            if len(parts) != 3 or parts[0] != 'dataset-resized' or parts[1] not in CLASSES or not name.endswith('.jpg'):
                continue
            data = z.read(name)
            image = Image.open(io.BytesIO(data)).convert('L').resize((32, 32))
            low = dctn(np.asarray(image, dtype=float), norm='ortho')[:8, :8].flatten()[1:]
            bits = low > np.median(low)
            phash = sum(int(bit) << i for i, bit in enumerate(bits))
            hashes.append(phash)
            records.append({'name': name, 'label': parts[1], 'sha256': hashlib.sha256(data).hexdigest(), 'phash': f'{phash:016x}'})
    assert len(records) == 2527
    # Conservative perceptual grouping prevents near-identical images crossing splits.
    parents = list(range(len(records)))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    for i in range(len(records)):
        for j in range(i):
            if (hashes[i] ^ hashes[j]).bit_count() <= 6 or records[i]['sha256'] == records[j]['sha256']:
                parents[root(i)] = root(j)
    groups = np.array([root(i) for i in range(len(records))])
    reserved = {groups[i] for i, r in enumerate(records) if r['name'] in demo}
    eligible = np.array([i for i, g in enumerate(groups) if g not in reserved])
    y = np.array([CLASSES.index(r['label']) for r in records])
    remain, test = next(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(eligible, y[eligible], groups[eligible]))
    test_ids, remain_ids = eligible[test], eligible[remain]
    train, val = next(StratifiedGroupKFold(4, shuffle=True, random_state=SEED + 1).split(remain_ids, y[remain_ids], groups[remain_ids]))
    splits = {'train': remain_ids[train], 'validation': remain_ids[val], 'test': test_ids}
    for i, r in enumerate(records):
        r['group'] = int(groups[i])
        r['split'] = 'gallery' if r['name'] in demo else 'excluded_gallery_neighbor'
    for name, ids in splits.items():
        for i in ids:
            records[i]['split'] = name
    return records, baseline


def extract(archive, records, cache):
    import torch
    from transformers import SiglipImageProcessor, SiglipVisionModel
    signature = hashlib.sha256(json.dumps({'model': MODEL, 'revision': REVISION, 'images': [r['sha256'] for r in records], 'views': ['original', 'horizontal_flip']}, sort_keys=True).encode()).hexdigest()
    if cache.exists():
        saved = np.load(cache)
        assert str(saved['signature']) == signature
        return saved['features'], float(saved['ms_per_image'])
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), 'This recorded protocol requires CUDA'
    processor = SiglipImageProcessor.from_pretrained(MODEL, revision=REVISION)
    model = SiglipVisionModel.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32).cuda().eval()
    vectors = []
    started = time.perf_counter()
    with zipfile.ZipFile(archive) as z, torch.inference_mode():
        for offset in range(0, len(records), 24):
            images = [Image.open(io.BytesIO(z.read(r['name']))).convert('RGB') for r in records[offset:offset + 24]]
            views = images + [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in images]
            inputs = processor(images=views, return_tensors='pt').to('cuda')
            outputs = model(**inputs).pooler_output.float()
            a, b = outputs.chunk(2)
            features = (a + b) / 2
            features = torch.nn.functional.normalize(features, dim=-1)
            vectors.append(features.cpu().numpy())
            if offset % 240 == 0:
                print(f'features {offset + len(images)}/{len(records)}', flush=True)
    features = np.concatenate(vectors)
    elapsed = (time.perf_counter() - started) * 1000 / len(records)
    np.savez(cache, signature=signature, features=features, ms_per_image=elapsed)
    return features, elapsed


def metrics(y, pred):
    return {'correct': int((y == pred).sum()), 'total': len(y), 'accuracy': float(accuracy_score(y, pred)), 'balanced_accuracy': float(balanced_accuracy_score(y, pred)), 'confusion_matrix': confusion_matrix(y, pred, labels=range(6)).tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'report.json').exists():
        raise SystemExit('Completed run exists; choose a new output to preserve evidence')
    records, baseline = prepare(args.archive)
    write(args.output / 'split.json', {'seed': SEED, 'source_sha256': baseline['source_sha256'], 'phash_max_distance': 6, 'records': records})
    print('splits', dict(Counter(r['split'] for r in records)), flush=True)
    x, ms = extract(args.archive, records, args.output / 'features.npz')
    y = np.array([CLASSES.index(r['label']) for r in records])
    indices = {split: np.array([i for i, r in enumerate(records) if r['split'] == split]) for split in ['train', 'validation', 'test', 'gallery']}
    train, val = indices['train'], indices['validation']
    candidates = []
    best, best_score, best_name = None, -1, None
    # This fixed grid is selected using validation balanced accuracy only.
    for kind in ['logistic', 'rbf_svm']:
        for c in [0.1, 1, 10, 100]:
            clf = LogisticRegression(C=c, max_iter=2000, class_weight='balanced', random_state=SEED) if kind == 'logistic' else SVC(C=c, kernel='rbf', gamma='scale', class_weight='balanced', probability=True, random_state=SEED)
            clf.fit(x[train], y[train])
            result = metrics(y[val], clf.predict_proba(x[val]).argmax(axis=1))
            name = f'{kind}_C{c}'
            candidates.append({'name': name, **result})
            print(name, result['accuracy'], result['balanced_accuracy'], flush=True)
            if result['balanced_accuracy'] > best_score:
                best, best_score, best_name = clf, result['balanced_accuracy'], name
    # Freeze the choice before accessing test/gallery predictions. Do not refit.
    write(args.output / 'selection.json', {'selected': best_name, 'criterion': 'validation balanced accuracy', 'candidates': candidates})
    joblib.dump(best, args.output / 'classifier.joblib')
    checkpoint_hash = hashlib.sha256((args.output / 'classifier.joblib').read_bytes()).hexdigest()
    report = {'model': MODEL, 'revision': REVISION, 'classifier': best_name, 'classifier_sha256': checkpoint_hash, 'classes': CLASSES, 'seed': SEED, 'feature_views': ['original', 'horizontal_flip'], 'feature_ms_per_image': ms, 'runtime': 'RTX 3080 / CUDA / float32; frozen SigLIP2 + supervised classifier', 'split_counts': dict(Counter(r['split'] for r in records)), 'selection': candidates, 'policy': 'forced top-1, no abstention; scores are not correctness probabilities', 'evaluations': {}}
    for split in ['validation', 'test', 'gallery']:
        ids = indices[split]
        probabilities = best.predict_proba(x[ids])
        # Public results use probability argmax consistently (SVC.predict may differ).
        predicted = probabilities.argmax(axis=1)
        result = metrics(y[ids], predicted)
        report['evaluations'][split] = result
        rows = []
        for i, prob, pred in zip(ids, probabilities, predicted):
            r = records[i]
            rows.append({'image': r['name'], 'image_sha256': r['sha256'], 'ground_truth': r['label'], 'selected': CLASSES[pred], 'raw_top1': CLASSES[pred], 'candidate_mass': None, 'top_option_probability': float(prob[pred]), 'abstention_reasons': [], 'latency_ms': None, 'scores': [{'id': label, 'option_probability': float(score)} for label, score in zip(CLASSES, prob)]})
        write(args.output / f'{split}-observations.json', rows)
        print(split, result, flush=True)
    write(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
