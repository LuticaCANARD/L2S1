#!/usr/bin/env python3
"""Freeze and evaluate a balanced TrashNet sample with local L2S1 vision."""

import argparse
import base64
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from zipfile import ZipFile


SOURCE_COMMIT = "6fa2b878c6c1b4304b91109070ce0edf9279bb31"
SOURCE_SHA256 = "0bf472790f8b20e5c950d5b5012a9d38af0d3392efd65f8ce171334fc16b07c2"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
DESCRIPTIONS = {
    "cardboard": "A piece of corrugated cardboard or cardboard packaging.",
    "glass": "A glass object, such as a bottle or jar.",
    "metal": "A metal object, such as a can or metal container.",
    "paper": "A sheet of paper, newspaper, or paper packaging that is not cardboard.",
    "plastic": "A plastic object, such as a bottle or plastic container.",
    "trash": "Miscellaneous waste that is not primarily cardboard, glass, metal, paper, or plastic.",
}
PROMPTS = {
    "baseline": "Choose the primary material of the main waste item in the photo. Use the image only.",
    "traits_v1": (
        "Choose the primary material of the main waste item in the photo. "
        "Use visible material cues, not the background, printed words, or presumed recyclability. "
        "Cardboard: thick, fairly stiff layered paperboard, often a carton or box panel; "
        "corrugation may be visible at an edge. "
        "Glass: hard, rigid material with a smooth reflective surface, often transparent "
        "or translucent with thick edges, as in a bottle or jar. "
        "Metal: opaque metallic material, often a can, lid, or foil; it may be rigid "
        "or crumpled and show a metallic sheen or formed rim. "
        "Paper: thin, flexible fibrous sheets, pages, leaflets, or paper wrapping, "
        "even when printed or folded; distinguish it from thick box board. "
        "Plastic: molded polymer containers or flexible film and wrappers; it may be "
        "clear or opaque, rigid or bendable, so transparency alone does not prove glass. "
        "Trash: residual, visibly soiled, mixed-material, or other waste when none of "
        "the five named materials is clearly the primary material. "
        "For composite objects choose the dominant visible material; use trash if none dominates."
    ),
    "traits_v2": (
        "Classify the primary visible material of the main discarded object. First identify the "
        "foreground object; ignore the white background, shadows, printed text, brand, presumed "
        "contents, and whether an item is recyclable. Compare several visible physical cues "
        "rather than one color, shape, or highlight. The six category guides are: "
        "CARDBOARD is thick, relatively stiff paper fiber formed into a box, carton, or packaging "
        "panel. A cut edge may reveal layers or corrugation; a printed surface can still be "
        "cardboard. Distinguish it from a thin sheet that bends or folds easily. "
        "GLASS is hard and rigid, often with thick transparent or colored translucent walls, "
        "sharp specular reflections, and a solid bottle or jar rim. A bottle silhouette or "
        "transparency by itself does not establish glass. "
        "METAL includes cans, lids, sheet metal, and foil. Look for a metallic surface, formed "
        "rims or seams, or crumpled reflective foil; paint and printing can hide the sheen. "
        "PAPER is a relatively thin, flexible fibrous sheet such as a page, leaflet, newspaper, "
        "or paper wrapper. Printing, gloss, and folding do not turn a thin sheet into cardboard. "
        "PLASTIC includes molded bottles, tubs, caps, and thin film or wrappers. It can be clear "
        "or opaque, stiff or flexible; thin walls, molded seams, and film folds can distinguish "
        "it from rigid glass. "
        "TRASH is residual waste, visibly soiled material, a mixed or composite item with no "
        "clear primary material among the five above, or an object made mainly from another "
        "material. Do not use trash merely because a clearly identifiable paper, plastic, "
        "metal, glass, or cardboard item is broken or discarded. "
        "If several materials appear, choose the material occupying most of the main object; "
        "choose trash when none clearly dominates. Return one of the six named classes."
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(args):
    if args.count < 1:
        raise ValueError("count must be positive")
    if file_digest(args.source) != SOURCE_SHA256:
        raise ValueError("source archive SHA-256 differs from the pinned download")
    records = []
    with ZipFile(args.source) as archive:
        images = {name: [] for name in CLASSES}
        for name in archive.namelist():
            parts = name.split("/")
            if (
                len(parts) == 3
                and parts[0] == "dataset-resized"
                and parts[1] in images
                and parts[2].lower().endswith(".jpg")
            ):
                images[parts[1]].append(name)
        expected = {"cardboard": 403, "glass": 501, "metal": 410,
                    "paper": 594, "plastic": 482, "trash": 137}
        if {key: len(value) for key, value in images.items()} != expected:
            raise ValueError("source archive class counts differ from the published dataset")
        rng = random.Random(args.seed)
        for label in CLASSES:
            for name in sorted(rng.sample(sorted(images[label]), args.count)):
                data = archive.read(name)
                records.append({"name": name, "label": label, "sha256": digest(data)})
    rng.shuffle(records)
    selection = {
        "dataset": "garythung/trashnet dataset-resized",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "seed": args.seed,
        "images_per_class": args.count,
        "classes": list(CLASSES),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(selection, stream, indent=2)
        stream.write("\n")
    print(f"selected {len(records)} original JPEGs: {args.output}")


def nearest_rank(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def run(args):
    if not (0 <= args.min_top_probability <= 1 and 0 <= args.min_candidate_mass <= 1):
        raise ValueError("decision thresholds must be in [0, 1]")
    selection = json.loads(args.selection.read_text())
    if selection["source_sha256"] != file_digest(args.source):
        raise ValueError("selected archive SHA-256 mismatch")
    if selection["classes"] != list(CLASSES):
        raise ValueError("class schema mismatch")
    if len(selection["records"]) != selection["images_per_class"] * len(CLASSES):
        raise ValueError("sample count mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    options = [{"id": label, "criterion": DESCRIPTIONS[label]} for label in CLASSES]
    instruction = PROMPTS[args.prompt_mode]
    command = [str(args.binary.resolve()), "--model", str(args.model.resolve()),
               "--mmproj", str(args.mmproj.resolve()), "--device", "cuda",
               "--context", "4096", "--batch", "256", "--threads", "4",
               "--model-load-mode", "read", "--listen", args.listen,
               "--min-top-probability", str(args.min_top_probability),
               "--min-candidate-mass", str(args.min_candidate_mass)]
    url = "http://" + args.listen
    observations = []
    started = time.perf_counter_ns()
    with ZipFile(args.source) as archive, (args.output / "server.log").open("wb") as log:
        server = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}; see server.log")
                try:
                    with urllib.request.urlopen(url + "/healthz", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.25)
            else:
                raise TimeoutError("server readiness timeout")
            startup_ms = (time.perf_counter_ns() - started) / 1e6
            with (args.output / "observations.jsonl").open("x") as output:
                for index, item in enumerate(selection["records"], 1):
                    image = archive.read(item["name"])
                    if digest(image) != item["sha256"]:
                        raise ValueError(f"sample image hash mismatch: {item['name']}")
                    payload = {
                        "state": {},
                        "decisions": [{
                            "id": "material",
                            "instruction": instruction,
                            "kind": {"type": "choice", "options": options},
                        }],
                        "image_base64": base64.b64encode(image).decode("ascii"),
                    }
                    request = urllib.request.Request(
                        url + "/v1/decisions",
                        json.dumps(payload, separators=(",", ":")).encode(),
                        {"Content-Type": "application/json"},
                    )
                    requested = time.perf_counter_ns()
                    try:
                        with urllib.request.urlopen(request, timeout=180) as response:
                            status, body = response.status, response.read()
                    except urllib.error.HTTPError as exc:
                        status, body = exc.code, exc.read()
                    latency_ms = (time.perf_counter_ns() - requested) / 1e6
                    if status != 200:
                        raise RuntimeError(f"image {index}: HTTP {status}: {body[:300]!r}")
                    result = json.loads(body)
                    backend = result["backend"]
                    if not backend["offload_requested"] or "3080" not in backend["offload_device"]:
                        raise RuntimeError("CUDA offload did not match the RTX 3080")
                    decision = result["results"][0]
                    ranked = sorted(decision["scores"], key=lambda row: row["option_probability"], reverse=True)
                    if len(ranked) != len(CLASSES):
                        raise RuntimeError("response option count mismatch")
                    raw_top1 = None if ranked[0]["option_probability"] == ranked[1]["option_probability"] else ranked[0]["id"]
                    record = {
                        "index": index, "image": item["name"], "image_sha256": item["sha256"],
                        "ground_truth": item["label"],
                        "selected": decision["value"]["selected"], "raw_top1": raw_top1,
                        "candidate_mass": decision["candidate_mass"],
                        "top_option_probability": decision["top_option_probability"],
                        "abstention_reasons": decision["abstention_reasons"],
                        "scores": decision["scores"], "input_tokens": decision["input_tokens"],
                        "latency_ms": latency_ms,
                    }
                    observations.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    if index % 20 == 0:
                        print(f"completed {index}/{len(selection['records'])}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    if len(observations) != len(selection["records"]):
        raise RuntimeError("incomplete image evaluation")
    accepted = [row for row in observations if row["selected"] in CLASSES]
    correct_accepted = sum(row["selected"] == row["ground_truth"] for row in accepted)
    correct_raw = sum(row["raw_top1"] == row["ground_truth"] for row in observations)
    by_class = {}
    for label in CLASSES:
        rows = [row for row in observations if row["ground_truth"] == label]
        by_class[label] = {
            "total": len(rows),
            "accepted": sum(row["selected"] in CLASSES for row in rows),
            "accepted_correct": sum(row["selected"] == label for row in rows),
            "raw_top1_correct": sum(row["raw_top1"] == label for row in rows),
            "predictions": dict(Counter(row["selected"] or "abstain" for row in rows)),
        }
    latencies = [row["latency_ms"] for row in observations]
    summary = {
        "dataset": selection["dataset"], "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256, "selection_sha256": file_digest(args.selection),
        "sample_size": len(observations), "images_per_class": selection["images_per_class"],
        "binary_sha256": file_digest(args.binary), "model_sha256": file_digest(args.model),
        "mmproj_sha256": file_digest(args.mmproj), "model": str(args.model),
        "prompt_mode": args.prompt_mode, "instruction": instruction,
        "instruction_sha256": digest(instruction.encode("utf-8")),
        "options": options, "min_top_probability": args.min_top_probability,
        "min_candidate_mass": args.min_candidate_mass,
        "startup_to_health_ms": startup_ms,
        "protocol": "One original JPEG per serial local HTTP request; fixed six-choice material question; label and filename excluded; no warmup; RTX 3080 CUDA; loaded-model latency includes first request",
        "accepted": len(accepted), "abstained": len(observations) - len(accepted),
        "accepted_correct": correct_accepted,
        "correct_over_all": correct_accepted / len(observations),
        "accepted_accuracy": correct_accepted / len(accepted) if accepted else None,
        "coverage": len(accepted) / len(observations),
        "raw_top1_correct": correct_raw, "raw_top1_accuracy": correct_raw / len(observations),
        "latency_ms": {"mean": statistics.mean(latencies),
                       "p50": statistics.median(latencies), "p95_nearest_rank": nearest_rank(latencies, 0.95)},
        "by_class": by_class,
    }
    with (args.output / "summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    print(json.dumps({key: summary[key] for key in
                      ("sample_size", "accepted", "accepted_correct", "raw_top1_correct", "latency_ms")}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    selection = sub.add_parser("prepare")
    selection.add_argument("--source", required=True, type=Path)
    selection.add_argument("--output", required=True, type=Path)
    selection.add_argument("--count", type=int, default=20)
    selection.add_argument("--seed", type=int, default=20260925)
    evaluate = sub.add_parser("run")
    for name in ("source", "selection", "binary", "model", "mmproj", "output"):
        evaluate.add_argument("--" + name, required=True, type=Path)
    evaluate.add_argument("--listen", default="127.0.0.1:18777")
    evaluate.add_argument("--prompt-mode", choices=tuple(PROMPTS), default="baseline")
    evaluate.add_argument("--min-top-probability", type=float, default=0.8)
    evaluate.add_argument("--min-candidate-mass", type=float, default=0.05)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
