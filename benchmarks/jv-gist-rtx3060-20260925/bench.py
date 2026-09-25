#!/usr/bin/env python3
"""Bounded same-GGUF CUDA execution-mode benchmark for the MLX jv comparison."""
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BINARY = ROOT / "source" / "target" / "release" / "l2s1"
MODEL_KIND = os.environ.get("BENCH_MODEL", "bonsai")
if MODEL_KIND not in ("bonsai", "gemma"):
    raise ValueError("BENCH_MODEL must be bonsai or gemma")
MODEL = (Path.home() / "l2s1-bonsai-vision-test" / "Bonsai-27B-Q1_0.gguf"
    if MODEL_KIND == "bonsai" else Path.home() / "personal/skid/gemma26-offload-20260923-211926/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")
OUT = ROOT / "results" / ("bonsai-q1-shared-state" if MODEL_KIND == "bonsai" else "gemma26-q4-shared-state")
PORT = 18733
URL = f"http://127.0.0.1:{PORT}"
MODES = [
    ("fresh_full", "fresh", "full"),
    ("prefix_full", "prefix-reuse", "full"),
    ("prefix_compact", "prefix-reuse", "compact"),
]
if MODEL_KIND == "gemma":
    MODES.insert(2, ("parallel_width2_full", "parallel", "full"))
COUNTS = (1, 16)
WARMUPS = 1
REPEATS = 3


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_request():
    zones = ("cold_room", "ambient", "hazmat_cage")
    items = []
    decisions = []
    for index in range(16):
        sku = f"SKU-{index + 1:02d}"
        condition = ("chilled", "room_temperature", "flammable")[index % 3]
        items.append({"sku": sku, "condition": condition, "batch": 100 + index})
        decisions.append({
            "id": f"zone_{index + 1:02d}",
            "instruction": f"Choose the correct storage zone for {sku}. Chilled items use cold_room; flammable items use hazmat_cage; all other items use ambient.",
            "kind": {"type": "choice", "options": [
                {"id": zone, "criterion": zone.replace("_", " ")} for zone in zones
            ]},
        })
    return {"state": {"warehouse": "North", "items": items}, "decisions": decisions}


def fetch(path, payload=None, timeout=300):
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(URL + path, data=data,
        headers={"Content-Type": "application/json"} if data is not None else {})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def gpu_state():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], text=True)
    used, temp, power = text.strip().split(",")
    return {"memory_mib": float(used), "temperature_c": float(temp), "power_w": float(power)}


def percentile(values, ratio):
    values = sorted(values)
    return values[max(0, math.ceil(ratio * len(values)) - 1)]


def record_response(response):
    return [{
        "id": row["id"], "selected": row["value"]["selected"],
        "status": row["status"],
        "input_tokens": row["usage"]["input_tokens"],
        "reused_prefix_tokens": row["usage"].get("reused_prefix_tokens", 0),
        "candidate_mass": row["evidence"]["candidate_mass"],
        "scores": [{"id": s["id"], "option_probability": s["option_probability"]} for s in row["evidence"]["scores"]],
    } for row in response["results"]]


def compare(reference, other):
    if [x["id"] for x in reference] != [x["id"] for x in other]:
        raise RuntimeError("decision IDs differ")
    delta_p = delta_mass = 0.0
    changed_selected = changed_top1 = 0
    for a, b in zip(reference, other):
        delta_mass = max(delta_mass, abs(a["candidate_mass"] - b["candidate_mass"]))
        changed_selected += a["selected"] != b["selected"]
        aa = {x["id"]: x["option_probability"] for x in a["scores"]}
        bb = {x["id"]: x["option_probability"] for x in b["scores"]}
        if aa.keys() != bb.keys():
            raise RuntimeError("option IDs differ")
        delta_p = max(delta_p, *(abs(aa[key] - bb[key]) for key in aa))
        changed_top1 += max(aa, key=aa.get) != max(bb, key=bb.get)
    return {"max_option_probability_delta": delta_p, "max_candidate_mass_delta": delta_mass,
        "changed_selected": changed_selected, "changed_raw_top1": changed_top1}


def one_mode(name, execution_mode, evidence_transfer, payloads, output):
    command = [str(BINARY), "--model", str(MODEL), "--device", "cuda",
        "--context", "4096", "--batch", "256", "--ubatch", "256", "--threads", "8",
        "--prompt-layout", "state-first", "--execution-mode", execution_mode,
        "--parallel-width", "2" if MODEL_KIND == "gemma" else "4", "--evidence-transfer", evidence_transfer,
        "--listen", f"127.0.0.1:{PORT}"]
    if MODEL_KIND == "gemma":
        command += ["--cpu-moe-layers", "18"]
    log = (OUT / f"{name}.server.log").open("w")
    started = time.perf_counter()
    server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT / "source")
    stop = threading.Event()
    gpu_samples = []
    def sample():
        while not stop.wait(0.2):
            try:
                gpu_samples.append(gpu_state())
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        for _ in range(1200):
            if server.poll() is not None:
                raise RuntimeError(f"{name} server exited {server.returncode}; see {name}.server.log")
            try:
                fetch("/healthz", timeout=0.5)
                break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(0.1)
        else:
            raise RuntimeError(f"{name} readiness timeout")
        startup_ms = (time.perf_counter() - started) * 1000
        capabilities = fetch("/v1/capabilities")
        rows = []
        first_responses = {}
        backend_details = None
        for count in COUNTS:
            payload = payloads[count]
            for iteration in range(WARMUPS + REPEATS):
                before = time.perf_counter_ns()
                result = fetch("/v1/decisions", payload)
                elapsed_ms = (time.perf_counter_ns() - before) / 1e6
                if result.get("api_version") != 1 or len(result["results"]) != count:
                    raise RuntimeError("invalid API response")
                details = result["backend"]["details"]
                if backend_details is None:
                    backend_details = details
                if not details["offload_requested"] or "3060" not in str(details["offload_device"]):
                    raise RuntimeError("RTX 3060 offload was not verified")
                if iteration == WARMUPS:
                    first_responses[count] = record_response(result)
                rows.append({"mode": name, "count": count, "phase": "warmup" if iteration < WARMUPS else "measured",
                    "iteration": iteration, "elapsed_ms": elapsed_ms,
                    "accepted": sum(row["status"] == "selected" for row in result["results"]),
                    "input_tokens": sum(row["usage"]["input_tokens"] for row in result["results"]),
                    "reused_prefix_tokens": sum(row["usage"].get("reused_prefix_tokens", 0) for row in result["results"]),
                    "request_id": result["request_id"]})
                print(f"{name} {count} {iteration} {elapsed_ms:.1f}ms", flush=True)
        (OUT / f"{name}.json").write_text(json.dumps({"command": command, "startup_ms": startup_ms,
            "capabilities": capabilities, "backend_details": backend_details, "rows": rows, "first_responses": first_responses,
            "gpu_samples": gpu_samples}, indent=2) + "\n")
        return rows, first_responses, startup_ms, gpu_samples, backend_details
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        stop.set()
        sampler.join(timeout=2)
        log.close()


def main():
    resume = os.environ.get("BENCH_RESUME") == "1"
    OUT.mkdir(parents=True, exist_ok=resume)
    base = build_request()
    payloads = {count: {"state": base["state"], "decisions": base["decisions"][:count]} for count in COUNTS}
    for count, payload in payloads.items():
        (OUT / f"request-{count}.json").write_text(json.dumps(payload, indent=2) + "\n")
    model_sha256 = digest(MODEL)
    print("model_sha256", model_sha256, flush=True)
    report = {"source_commit": "c41925a92dc59feae30537fd5cc1f71215ba6c5e",
        "gist_sha256": "8a44930e4233a13795258793cd1a879db0a08df8b25e2a462a8bcce011c102e5",
        "model_kind": MODEL_KIND, "model": str(MODEL), "model_sha256": model_sha256, "binary_sha256": digest(BINARY),
        "host": platform.node(), "gpu_info": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], text=True).strip(),
        "gpu": gpu_state(), "warmups": WARMUPS, "repeats": REPEATS,
        "request_sha256": {str(count): hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            for count, payload in payloads.items()}, "modes": {}}
    first = {}
    if resume:
        previous = json.loads((OUT / "summary.json").read_text())
        if previous["model_sha256"] != report["model_sha256"] or previous["binary_sha256"] != report["binary_sha256"] or previous["request_sha256"] != report["request_sha256"]:
            raise RuntimeError("resume artifact identity differs")
        report = previous
        for name in report["modes"]:
            saved = json.loads((OUT / f"{name}.json").read_text())
            first[name] = {int(key): value for key, value in saved["first_responses"].items()}
    for name, mode, evidence in MODES:
        if name in report["modes"]:
            continue
        rows, responses, startup_ms, gpu_samples, backend_details = one_mode(name, mode, evidence, payloads, OUT)
        first[name] = responses
        statistics_by_count = {}
        for count in COUNTS:
            measured = [row for row in rows if row["count"] == count and row["phase"] == "measured"]
            times = [row["elapsed_ms"] for row in measured]
            statistics_by_count[str(count)] = {"latency_ms": {"mean": statistics.mean(times),
                "median": statistics.median(times), "p95_nearest_rank": percentile(times, .95),
                "min": min(times), "max": max(times)},
                "decisions_per_s_at_median": count / (statistics.median(times) / 1000),
                "accepted": [row["accepted"] for row in measured],
                "input_tokens": [row["input_tokens"] for row in measured],
                "reused_prefix_tokens": [row["reused_prefix_tokens"] for row in measured]}
        report["modes"][name] = {"startup_ms": startup_ms, "backend_details": backend_details,
            "peak_gpu_memory_mib": max((x["memory_mib"] for x in gpu_samples), default=None),
            "gpu_temperature_c_range": [min((x["temperature_c"] for x in gpu_samples), default=None),
                max((x["temperature_c"] for x in gpu_samples), default=None)],
            "counts": statistics_by_count,
            "parity_vs_fresh": {str(count): compare(first["fresh_full"][count], responses[count])
                for count in COUNTS} if "fresh_full" in first else None}
        (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["modes"], indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr, flush=True)
        raise
