#!/usr/bin/env python3
"""Run frozen 30-class Caltech-101 images through L2S1 vision HTTP."""

import argparse
import base64
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request
import zipfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def p95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def gpu_status():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser()
    for name in ("selection", "sample", "binary", "model", "mmproj", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--listen", default="127.0.0.1:18768")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    selection = json.loads(args.selection.read_text())
    assert selection["class_count"] == 30 and len(selection["records"]) == 150
    assert sha256(args.sample.read_bytes()) == selection["sample_archive_sha256"]
    classes = selection["classes"]
    friendly = {name: name.replace("_", " ").lower() for name in classes}
    friendly.update({"Faces_easy": "a human face", "bass": "a bass fish",
                     "flamingo_head": "a close-up of a flamingo head",
                     "hawksbill": "a hawksbill sea turtle",
                     "gerenuk": "a gerenuk antelope",
                     "snoopy": "the Snoopy cartoon character"})
    options = [{"id": name, "criterion": "Main visible subject category: " + friendly[name] + "."}
               for name in classes]
    command = [str(args.binary), "--model", str(args.model), "--mmproj", str(args.mmproj),
               "--device", "cuda", "--context", "4096", "--batch", "256", "--threads", "4",
               "--model-load-mode", "read", "--listen", args.listen]
    url = "http://" + args.listen
    started = time.perf_counter_ns()
    with zipfile.ZipFile(args.sample) as sample, (args.output / "server.log").open("wb") as log:
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
            startup_ms = (time.perf_counter_ns() - started) / 1e6
            print(f"ready after {startup_ms:.1f} ms", flush=True)
            observations = []
            backend = None
            policy = None
            with (args.output / "observations.jsonl").open("w") as output:
                for index, item in enumerate(selection["records"], 1):
                    image = sample.read(item["name"])
                    assert sha256(image) == item["sha256"]
                    payload = {
                        "state": {},
                        "decisions": [{"id": "object",
                                       "instruction": "Choose the single best object category for the main visible subject in this image. Use only the image, not any filename or metadata.",
                                       "kind": {"type": "choice", "options": options}}],
                        "image_base64": base64.b64encode(image).decode("ascii"),
                    }
                    req = urllib.request.Request(
                        url + "/v1/decisions", json.dumps(payload, separators=(",", ":")).encode(),
                        {"Content-Type": "application/json"})
                    started = time.perf_counter_ns()
                    try:
                        with urllib.request.urlopen(req, timeout=180) as response:
                            body = response.read()
                            status = response.status
                    except urllib.error.HTTPError as exc:
                        status = exc.code
                        body = exc.read()
                    latency_ms = (time.perf_counter_ns() - started) / 1e6
                    if status != 200:
                        raise RuntimeError(f"image {index}: HTTP {status}: {body[:300]!r}")
                    result = json.loads(body)
                    backend = result["backend"]
                    policy = result["policy"]
                    if not backend["offload_requested"] or "3060" not in backend["offload_device"]:
                        raise RuntimeError("GPU offload does not match the intended RTX 3060")
                    decision = result["results"][0]
                    scores = decision["scores"]
                    assert len(scores) == len(classes)
                    ranked = sorted(scores, key=lambda score: score["option_probability"], reverse=True)
                    raw_top1 = None if ranked[0]["option_probability"] == ranked[1]["option_probability"] else ranked[0]["id"]
                    record = {
                        "index": index, "image": item["name"], "image_sha256": item["sha256"],
                        "ground_truth": item["label"], "selected": decision["value"]["selected"],
                        "raw_top1": raw_top1, "latency_ms": round(latency_ms, 6),
                        "candidate_mass": decision["candidate_mass"],
                        "top_option_probability": decision["top_option_probability"],
                        "abstention_reasons": decision["abstention_reasons"],
                        "input_tokens": decision["input_tokens"],
                        "code_prefix_evaluations": decision["code_prefix_evaluations"],
                        "code_evaluated_tokens": decision["code_evaluated_tokens"],
                        "scoring_method": decision["scoring_method"], "scores": scores,
                    }
                    observations.append(record)
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output.flush()
                    if index % 10 == 0:
                        print(f"completed {index}/150", flush=True)
            loaded_gpu = gpu_status()
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    assert len(observations) == 150
    accepted = [row for row in observations if row["selected"] in classes]
    latencies = [row["latency_ms"] for row in observations]
    summary = {
        "schema": "l2s1-caltech101-30class-vision-http-v1",
        "dataset": selection["source"],
        "source_archive_sha256": selection["source_archive_sha256"],
        "sample_archive_sha256": selection["sample_archive_sha256"],
        "seed": selection["seed"], "classes": classes, "images_per_class": 5,
        "rows": len(observations),
        "protocol": "Balanced 30-class subset sampled before inference; original JPEG bytes; one serial local HTTP request per image; no image name or label in prompt; fixed class order; default policy; no warmup; loaded-model latency includes first request and excludes startup",
        "model_sha256": sha256(args.model.read_bytes()),
        "mmproj_sha256": sha256(args.mmproj.read_bytes()),
        "binary_sha256": sha256(args.binary.read_bytes()),
        "backend": backend, "policy": policy,
        "startup_to_health_ms": round(startup_ms, 3), "gpu_during_run": loaded_gpu,
        "accepted": len(accepted), "abstained": 150 - len(accepted),
        "accepted_correct": sum(row["selected"] == row["ground_truth"] for row in accepted),
        "correct_over_all": sum(row["selected"] == row["ground_truth"] for row in accepted) / 150,
        "accepted_accuracy": (sum(row["selected"] == row["ground_truth"] for row in accepted) / len(accepted)) if accepted else None,
        "coverage": len(accepted) / 150,
        "raw_top1_correct": sum(row["raw_top1"] == row["ground_truth"] for row in observations),
        "raw_top1_accuracy": sum(row["raw_top1"] == row["ground_truth"] for row in observations) / 150,
        "per_class": {label: {"accepted_correct": sum(row["selected"] == label for row in observations if row["ground_truth"] == label),
                              "accepted": sum(row["selected"] in classes for row in observations if row["ground_truth"] == label),
                              "raw_top1_correct": sum(row["raw_top1"] == label for row in observations if row["ground_truth"] == label)}
                      for label in classes},
        "latency_ms": {"mean": round(statistics.mean(latencies), 3),
                       "p50": round(statistics.median(latencies), 3),
                       "p95_nearest_rank": round(p95(latencies), 3),
                       "min": round(min(latencies), 3), "max": round(max(latencies), 3)},
        "prefix_evaluations": dict(collections.Counter(row["code_prefix_evaluations"] for row in observations)),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("rows", "accepted", "abstained", "accepted_correct", "raw_top1_correct", "latency_ms", "prefix_evaluations")}), flush=True)


if __name__ == "__main__":
    main()
