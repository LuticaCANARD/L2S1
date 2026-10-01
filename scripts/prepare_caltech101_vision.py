#!/usr/bin/env python3
"""Freeze a balanced Caltech-101 subset before any model inference."""

import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import zipfile


SEED = 20260924


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path,
                        help="Kaggle mirror archive downloaded as dataset.zip")
    parser.add_argument("--output", required=True, type=Path,
                        help="New directory for sample-30x5.zip and selection.json")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample_path = args.output / "sample-30x5.zip"
    selection_path = args.output / "selection.json"
    if sample_path.exists() or selection_path.exists():
        raise SystemExit("sample or selection already exists")
    with zipfile.ZipFile(args.source) as source:
        files = collections.defaultdict(list)
        for name in source.namelist():
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                files[name.split("/")[1]].append(name)
        assert len(files) == 102 and all(len(v) >= 5 for v in files.values())
        classes = sorted(set(files) - {"BACKGROUND_Google"})
        assert len(classes) == 101
        rng = random.Random(SEED)
        chosen = sorted(rng.sample(classes, 30))
        records = []
        with zipfile.ZipFile(sample_path, "w", compression=zipfile.ZIP_STORED) as sample:
            for label in chosen:
                for name in sorted(rng.sample(sorted(files[label]), 5)):
                    image = source.read(name)
                    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_STORED
                    info.external_attr = 0o644 << 16
                    sample.writestr(info, image)
                    records.append({"name": name, "label": label, "sha256": sha256(image)})
    selection = {
        "source": "https://www.kaggle.com/datasets/imbikramsaha/caltech-101",
        "source_archive_sha256": sha256(args.source.read_bytes()),
        "seed": SEED,
        "population": "all 101 object classes; exclude only BACKGROUND_Google",
        "sampling": "Python random.Random(seed): sample 30 sorted class names, then 5 sorted image names per class",
        "class_count": len(chosen), "images_per_class": 5,
        "classes": chosen, "records": records,
        "sample_archive_sha256": sha256(sample_path.read_bytes()),
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"classes": chosen, "images": len(records),
                      "sample_archive_sha256": selection["sample_archive_sha256"]}))


if __name__ == "__main__":
    main()
