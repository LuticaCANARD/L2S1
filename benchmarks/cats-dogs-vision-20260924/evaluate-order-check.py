#!/usr/bin/env python3
"""Evaluate Kaggle Cats and Dogs validation images through L2S1 vision HTTP."""

import argparse
import base64
import collections
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def percentile(values, fraction):
    return sorted(values)[math.ceil(len(values) * fraction) - 1]


def request(url, payload, timeout=120):
    data = json.dumps(payload, separators=(",", ":")).encode()
    req = urllib.request.Request(url + "/v1/decisions", data,
                                 {"Content-Type": "application/json"})
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    return status, json.loads(body), (time.perf_counter_ns() - started) / 1e6


def main():
    parser = argparse.ArgumentParser()
    for name in ("archive", "binary", "model", "mmproj", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--listen", default="127.0.0.1:18766")
    parser.add_argument("--order", choices=("cat-dog", "dog-cat"), default="cat-dog")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    archive_bytes = args.archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        csv_bytes = archive.read("val.csv")
        rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
        expected = {"0": "cat", "1": "dog"}
        assert len(rows) == 70
        assert collections.Counter(row["category"] for row in rows) == {"0": 24, "1": 46}
        for row in rows:
            assert f"/{expected[row['category']]}/" in row["image:FILE"]
            assert row["image:FILE"] in archive.namelist()

        command = [str(args.binary), "--model", str(args.model), "--mmproj", str(args.mmproj),
                   "--device", "cuda", "--context", "2048", "--batch", "256",
                   "--threads", "4", "--model-load-mode", "read", "--listen", args.listen]
        url = "http://" + args.listen
        started = time.perf_counter_ns()
        with (args.output / "server.log").open("wb") as log:
            server = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                for _ in range(600):
                    if server.poll() is not None:
                        raise RuntimeError(f"server exited {server.returncode}; inspect server.log")
                    try:
                        with urllib.request.urlopen(url + "/healthz", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.25)
                else:
                    raise RuntimeError("server readiness timeout")
                load_ms = (time.perf_counter_ns() - started) / 1e6
                print(f"ready after {load_ms:.1f} ms", flush=True)
                observations = []
                with (args.output / "observations.jsonl").open("w") as output:
                    for i, row in enumerate(rows, 1):
                        image_name = row["image:FILE"]
                        image = archive.read(image_name)
                        ground_truth = expected[row["category"]]
                        options = [
                            {"id": "cat", "criterion": "The main subject is a cat."},
                            {"id": "dog", "criterion": "The main subject is a dog."},
                        ]
                        if args.order == "dog-cat":
                            options.reverse()
                        payload = {
                            "state": {},
                            "decisions": [{
                                "id": "animal",
                                "instruction": "Which animal is the main subject of this image? Select cat or dog based only on the visible image.",
                                "kind": {"type": "choice", "options": options},
                            }],
                            "image_base64": base64.b64encode(image).decode("ascii"),
                        }
                        try:
                            status, response, latency_ms = request(url, payload)
                            result = response.get("results", [{}])[0] if status == 200 else {}
                            scores = result.get("scores", [])
                            ranked = sorted(scores, key=lambda item: item["option_probability"], reverse=True)
                            tie = len(ranked) > 1 and ranked[0]["option_probability"] == ranked[1]["option_probability"]
                            raw_top1 = ranked[0]["id"] if ranked and not tie else None
                            selected = result.get("value", {}).get("selected")
                            if status == 200 and not response.get("backend", {}).get("offload_requested"):
                                raise RuntimeError("CUDA offload not requested")
                            record = {"index": i, "image": image_name, "image_sha256": digest(image),
                                      "ground_truth": ground_truth, "http_status": status,
                                      "latency_ms": round(latency_ms, 6), "selected": selected,
                                      "raw_top1": raw_top1, "tie": tie,
                                      "abstention_reasons": result.get("abstention_reasons", []),
                                      "top_option_probability": result.get("top_option_probability"),
                                      "candidate_mass": result.get("candidate_mass"),
                                      "scores": scores, "input_tokens": result.get("input_tokens"),
                                      "error": response.get("error")}
                        except Exception as exc:
                            record = {"index": i, "image": image_name, "image_sha256": digest(image),
                                      "ground_truth": ground_truth, "http_status": None,
                                      "selected": None, "raw_top1": None, "error": repr(exc)}
                        observations.append(record)
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output.flush()
                        if i % 10 == 0 or i == len(rows):
                            print(f"completed {i}/{len(rows)}", flush=True)
                backend = response.get("backend", {}) if status == 200 else {}
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()

    total = len(observations)
    accepted = [row for row in observations if row.get("selected") in ("cat", "dog")]
    valid = [row for row in observations if row.get("http_status") == 200]
    raw = [row for row in observations if row.get("raw_top1") in ("cat", "dog")]
    latencies = [row["latency_ms"] for row in valid]
    summary = {
        "dataset": "https://www.kaggle.com/datasets/marquis03/cats-and-dogs",
        "archive_sha256": digest(archive_bytes), "val_csv_sha256": digest(csv_bytes),
        "rows": total, "label_counts": dict(collections.Counter(row["ground_truth"] for row in observations)),
        "protocol": "validation split only; original JPEG bytes; one serial loopback HTTP request per image; no examples or labels in prompts; default abstention policy; no warmup; loaded-model latency includes first request, excludes startup",
        "option_order": args.order,
        "binary_sha256": digest(args.binary.read_bytes()),
        "model_sha256": digest(args.model.read_bytes()),
        "mmproj_sha256": digest(args.mmproj.read_bytes()),
        "backend": backend, "startup_to_health_ms": round(load_ms, 3),
        "http_200": len(valid), "errors": total - len(valid),
        "accepted": len(accepted), "abstained": len(valid) - len(accepted),
        "accepted_correct": sum(row["selected"] == row["ground_truth"] for row in accepted),
        "correct_over_all": sum(row["selected"] == row["ground_truth"] for row in accepted) / total,
        "coverage": len(accepted) / total,
        "accepted_accuracy": (sum(row["selected"] == row["ground_truth"] for row in accepted) / len(accepted)) if accepted else None,
        "raw_top1_correct": sum(row["raw_top1"] == row["ground_truth"] for row in raw),
        "raw_top1_accuracy_over_all": sum(row["raw_top1"] == row["ground_truth"] for row in raw) / total,
        "confusion": {label: dict(collections.Counter(row.get("selected") or ("error" if row.get("http_status") != 200 else "abstain")
                                                    for row in observations if row["ground_truth"] == label))
                      for label in ("cat", "dog")},
        "latency_ms": {"mean": round(statistics.mean(latencies), 3),
                       "p50": round(statistics.median(latencies), 3),
                       "p95_nearest_rank": round(percentile(latencies, 0.95), 3),
                       "min": round(min(latencies), 3), "max": round(max(latencies), 3)} if latencies else None,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("rows", "http_200", "accepted", "abstained", "accepted_correct", "raw_top1_correct", "confusion", "latency_ms")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
