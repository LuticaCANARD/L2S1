#!/usr/bin/env python3
"""Compare fresh and native parallel vision using identical four-image HTTP groups."""
import argparse
import base64
import json
import math
import os
import shutil
from pathlib import Path
import statistics
import subprocess
import time
import urllib.error
import urllib.request
from zipfile import ZipFile

import evaluate as ev


def post(url, payload):
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    request = urllib.request.Request(url + "/v1/decisions", encoded,
                                     {"Content-Type": "application/json"})
    started = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, body = exc.code, exc.read()
    latency = (time.perf_counter_ns() - started) / 1e6
    if status != 200:
        raise RuntimeError(f"HTTP {status}: {body[:1000]!r}")
    return json.loads(body), latency, ev.digest(encoded)


def measure(args, mode, groups):
    directory = args.output / mode
    directory.mkdir(exist_ok=False)
    binary = args.baseline_binary if mode == "fresh" and args.baseline_binary else args.binary
    execution = "fresh" if mode == "fresh" else args.candidate_execution_mode
    command = [str(binary.resolve()), "--model", str(args.model.resolve()),
               "--mmproj", str(args.mmproj.resolve()), "--device", "cuda",
               "--context", "4096", "--threads", "4",
               "--model-load-mode", "read", "--listen", args.listen,
               "--min-top-probability", "0.8", "--min-candidate-mass", "0.05"]
    if mode == "parallel" and args.candidate_vision_optimized:
        command += ["--vision-optimized"]
    else:
        batch = 256 if mode == "fresh" else args.candidate_batch
        command += ["--execution-mode", execution, "--parallel-width", "4",
                    "--batch", str(batch), "--ubatch", str(256 if mode == "fresh" else args.candidate_ubatch or batch),
                    "--flash-attention", "off" if mode == "fresh" else args.candidate_flash_attention,
                    "--evidence-transfer", "full" if mode == "fresh" else args.candidate_evidence_transfer,
                    "--preparation-cache-bytes", str(0 if mode == "fresh" else args.candidate_cache_bytes)]
        if mode == "parallel" and args.candidate_dynamic_context:
            command += ["--parallel-context-dynamic"]
        if mode == "parallel" and args.candidate_projector_reuse:
            command += ["--vision-projector-reuse"]
    environment = os.environ.copy()
    force_clear = args.baseline_force_kv_clear if mode == "fresh" else args.candidate_force_kv_clear
    environment["L2S1_FORCE_KV_CLEAR"] = "1" if force_clear else "0"
    if mode == "fresh" and args.baseline_runtime_dir:
        environment["LD_LIBRARY_PATH"] = str(args.baseline_runtime_dir.resolve())
    url = "http://" + args.listen
    samples = []
    with (directory / "server.log").open("wb") as log:
        started = time.perf_counter_ns()
        server = subprocess.Popen(command, stdout=log, stderr=log, env=environment)
        try:
            for _ in range(600):
                if server.poll() is not None:
                    raise RuntimeError(f"server exited {server.returncode}; see {directory}/server.log")
                try:
                    with urllib.request.urlopen(url + "/healthz", timeout=1) as response:
                        if response.status == 200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(.25)
            else:
                raise TimeoutError("server readiness timeout")
            startup_ms = (time.perf_counter_ns() - started) / 1e6
            # The first complete group warms graph/context allocation. Preserve
            # its response and latency, but report timed full passes separately.
            warmup, warmup_ms, _ = post(url, groups[0]["payload"])
            (directory / "warmup.json").write_text(json.dumps({"latency_ms": warmup_ms, "response": warmup}, indent=2) + "\n")
            with (directory / "batches.jsonl").open("x") as stream:
                for repetition in range(args.repetitions):
                    for batch_index, group in enumerate(groups):
                        result, latency_ms, request_hash = post(url, group["payload"])
                        backend = result["backend"]["details"]
                        if not backend["offload_requested"] or "3080" not in backend["offload_device"]:
                            raise RuntimeError("expected RTX 3080 CUDA offload")
                        if mode == "parallel" and execution == "parallel":
                            metrics = backend.get("vision_batch")
                            if not metrics or metrics["decoder_batch_max_sequences"] < 2:
                                raise RuntimeError("native multi-sequence image batching was not observed")
                        if [row["id"] for row in result["results"]] != [f"material{i}" for i in range(4)]:
                            raise RuntimeError("result count/order mismatch")
                        sample = {"repetition": repetition + 1, "batch_index": batch_index + 1,
                                  "image_indices": group["indices"], "latency_ms": latency_ms,
                                  "request_sha256": request_hash, "response": result}
                        stream.write(json.dumps(sample) + "\n")
                        stream.flush()
                        samples.append(sample)
                    print(f"{mode}: completed pass {repetition + 1}/{args.repetitions}", flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    latencies = [sample["latency_ms"] for sample in samples]
    total = sum(latencies)
    summary = {"mode": mode, "images_per_request": 4, "request_count": len(samples),
               "execution_mode": execution, "command": command, "binary_sha256": ev.file_digest(binary),
               "force_kv_clear": force_clear,
               "runtime_override": str(args.baseline_runtime_dir) if mode == "fresh" and args.baseline_runtime_dir else None,
               "image_evaluations": len(samples) * 4, "startup_to_health_ms": startup_ms,
               "warmup_group_ms": warmup_ms,
               "group_latency_ms": {"mean": statistics.mean(latencies), "p50": statistics.median(latencies),
                                    "p95_nearest_rank": ev.nearest_rank(latencies, .95)},
               "effective_ms_per_image": total / (4 * len(samples)),
               "images_per_second": 4000 * len(samples) / total,
               "http_latency_sum_ms": total,
               "pass_http_latency_sum_ms": [sum(sample["latency_ms"] for sample in samples if sample["repetition"] == r)
                                             for r in range(1, args.repetitions + 1)]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return samples, summary


def observations(samples, records):
    rows = []
    for sample in samples:
        for index, result in zip(sample["image_indices"], sample["response"]["results"]):
            ranked = sorted(result["evidence"]["scores"], key=lambda s: s["option_probability"], reverse=True)
            if len(ranked) != len(ev.CLASSES):
                raise RuntimeError("option schema mismatch")
            rows.append({"index": index, "repetition": sample["repetition"], "label": records[index - 1]["label"],
                         "selected": result["value"]["selected"],
                         "raw_top1": None if ranked[0]["option_probability"] == ranked[1]["option_probability"] else ranked[0]["id"],
                         "ranking": [score["id"] for score in ranked],
                         "candidate_mass": result["evidence"]["candidate_mass"],
                         "scores": result["evidence"]["scores"], "input_tokens": result["usage"]["input_tokens"],
                         "abstention_reasons": result["abstention_reasons"]})
    return rows


def main():
    parser = argparse.ArgumentParser()
    for name in ("source", "selection", "binary", "model", "mmproj", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--listen", default="127.0.0.1:18778")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mode-order", choices=("fresh-first", "parallel-first"), default="fresh-first")
    parser.add_argument("--baseline-binary", type=Path)
    parser.add_argument("--baseline-runtime-dir", type=Path)
    parser.add_argument("--baseline-force-kv-clear", action="store_true")
    parser.add_argument("--candidate-force-kv-clear", action="store_true")
    parser.add_argument("--baseline-recorded-run", type=Path,
                        help="Reuse a verified earlier fresh run on the same frozen inputs; record its provenance")
    parser.add_argument("--candidate-vision-optimized", action="store_true")
    parser.add_argument("--candidate-execution-mode", choices=("fresh", "parallel"), default="parallel")
    parser.add_argument("--candidate-batch", type=int, default=256)
    parser.add_argument("--candidate-ubatch", type=int)
    parser.add_argument("--candidate-flash-attention", choices=("off", "on", "auto"), default="off")
    parser.add_argument("--candidate-evidence-transfer", choices=("full", "compact"), default="full")
    parser.add_argument("--candidate-cache-bytes", type=int, default=0)
    parser.add_argument("--candidate-dynamic-context", action="store_true")
    parser.add_argument("--candidate-projector-reuse", action="store_true")
    args = parser.parse_args()
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if args.candidate_vision_optimized and args.candidate_execution_mode != "parallel":
        raise ValueError("optimized vision requires parallel execution")
    selection = json.loads(args.selection.read_text())
    if ev.file_digest(args.source) != selection["source_sha256"] or selection["source_sha256"] != ev.SOURCE_SHA256:
        raise ValueError("source archive identity mismatch")
    records = selection["records"]
    if len(records) != 120 or selection["classes"] != list(ev.CLASSES):
        raise ValueError("expected the frozen 120-image six-class selection")
    args.output.mkdir(parents=True, exist_ok=False)
    options = [{"id": label, "criterion": ev.DESCRIPTIONS[label]} for label in ev.CLASSES]
    instruction = ev.PROMPTS["baseline"]
    groups = []
    with ZipFile(args.source) as archive:
        for offset in range(0, len(records), 4):
            payload = {"state": {}, "decisions": [], "media": []}
            for slot, item in enumerate(records[offset:offset + 4]):
                data = archive.read(item["name"])
                if ev.digest(data) != item["sha256"]:
                    raise ValueError("image identity mismatch")
                payload["media"].append({"id": f"image{slot}", "type": "image", "data_base64": base64.b64encode(data).decode("ascii")})
                payload["decisions"].append({"id": f"material{slot}", "instruction": instruction,
                    "kind": {"type": "choice", "options": options}, "media_ids": [f"image{slot}"]})
            groups.append({"payload": payload, "indices": list(range(offset + 1, offset + 5))})
    modes = ("fresh", "parallel") if args.mode_order == "fresh-first" else ("parallel", "fresh")
    runs = {}
    if args.baseline_recorded_run:
        recorded = json.loads((args.baseline_recorded_run / "summary.json").read_text())
        if recorded["timing"]["fresh"]["execution_mode"] != "fresh":
            raise ValueError("recorded baseline must use fresh execution")
        if recorded["timing"]["fresh"].get("force_kv_clear", False) != args.baseline_force_kv_clear:
            raise ValueError("recorded baseline clear diagnostic mismatch")
        if args.baseline_binary and recorded["timing"]["fresh"]["binary_sha256"] != ev.file_digest(args.baseline_binary):
            raise ValueError("recorded baseline executable identity mismatch")
        for key, path in (("source_sha256", args.source), ("selection_sha256", args.selection),
                          ("model_sha256", args.model), ("mmproj_sha256", args.mmproj)):
            if recorded[key] != ev.file_digest(path):
                raise ValueError(f"recorded baseline {key} mismatch")
        if recorded["repetitions"] != args.repetitions or recorded["instruction"] != instruction or recorded["options"] != options:
            raise ValueError("recorded baseline protocol mismatch")
        if recorded["min_top_probability"] != .8 or recorded["min_candidate_mass"] != .05:
            raise ValueError("recorded baseline policy mismatch")
        samples = [json.loads(line) for line in (args.baseline_recorded_run / "fresh/batches.jsonl").read_text().splitlines()]
        if len(samples) != len(groups) * args.repetitions:
            raise ValueError("recorded baseline sample count mismatch")
        for position, sample in enumerate(samples):
            group = groups[position % len(groups)]
            expected_hash = ev.digest(json.dumps(group["payload"], separators=(",", ":")).encode())
            if (sample["request_sha256"] != expected_hash or sample["image_indices"] != group["indices"]
                    or sample["repetition"] != position // len(groups) + 1
                    or sample["batch_index"] != position % len(groups) + 1
                    or [row["id"] for row in sample["response"]["results"]] != [f"material{i}" for i in range(4)]):
                raise ValueError("recorded baseline ordered request identity mismatch")
            backend = sample["response"]["backend"]["details"]
            expected_compute = {"context": 4096, "batch": 256, "ubatch": 256,
                                "threads": 4, "flash_attention": "off", "model_load_mode": "read"}
            if (any(backend["compute"].get(key) != value for key, value in expected_compute.items())
                    or backend["execution_mode"] != "fresh"
                    or backend.get("evidence_transfer", "full") != "full"
                    or not backend["offload_requested"] or "3080" not in backend["offload_device"]):
                raise ValueError("recorded baseline runtime configuration mismatch")
        timing = recorded["timing"]["fresh"]
        latencies = [sample["latency_ms"] for sample in samples]
        total = sum(latencies)
        if (timing["request_count"] != len(samples) or timing["image_evaluations"] != 4 * len(samples)
                or any(not math.isclose(timing[key], value, rel_tol=1e-12) for key, value in
                       (("http_latency_sum_ms", total), ("effective_ms_per_image", total / (4 * len(samples))),
                        ("images_per_second", 4000 * len(samples) / total)))
                or any(not math.isclose(timing["group_latency_ms"][key], value, rel_tol=1e-12) for key, value in
                       (("mean", statistics.mean(latencies)), ("p50", statistics.median(latencies)),
                        ("p95_nearest_rank", ev.nearest_rank(latencies, .95))))
                or timing["pass_http_latency_sum_ms"] != [
                    sum(sample["latency_ms"] for sample in samples if sample["repetition"] == r)
                    for r in range(1, args.repetitions + 1)]):
            raise ValueError("recorded baseline timing does not match its samples")
        shutil.copytree(args.baseline_recorded_run / "fresh", args.output / "fresh")
        runs["fresh"] = (samples, recorded["timing"]["fresh"])
    for mode in modes:
        if mode not in runs:
            runs[mode] = measure(args, mode, groups)
    rows = {mode: observations(runs[mode][0], records) for mode in modes}
    comparisons = {"changed_selected": 0, "changed_top1": 0, "changed_abstention": 0,
                   "changed_ranking": 0, "changed_input_tokens": 0, "max_raw_logit_delta": 0.,
                   "max_probability_delta": 0., "max_candidate_mass_delta": 0.}
    changed_indices = {"selected": set(), "raw_top1": set(), "abstention_reasons": set(), "ranking": set()}
    for serial, batch in zip(rows["fresh"], rows["parallel"]):
        for field, key in (("selected", "changed_selected"), ("raw_top1", "changed_top1"),
                           ("abstention_reasons", "changed_abstention"), ("ranking", "changed_ranking"),
                           ("input_tokens", "changed_input_tokens")):
            comparisons[key] += int(serial[field] != batch[field])
            if field in changed_indices and serial[field] != batch[field]:
                changed_indices[field].add(serial["index"])
        comparisons["max_candidate_mass_delta"] = max(comparisons["max_candidate_mass_delta"], abs(serial["candidate_mass"] - batch["candidate_mass"]))
        for a, b in zip(serial["scores"], batch["scores"]):
            if (a["id"], a["token_id"]) != (b["id"], b["token_id"]):
                raise RuntimeError("answer code identity changed")
            comparisons["max_probability_delta"] = max(comparisons["max_probability_delta"], abs(a["option_probability"] - b["option_probability"]))
            comparisons["max_raw_logit_delta"] = max(comparisons["max_raw_logit_delta"], abs(a["raw_logit"] - b["raw_logit"]))
    comparisons["image_evaluations"] = len(rows["fresh"])
    comparisons["unique_changed_image_indices"] = {key: sorted(value) for key, value in changed_indices.items()}
    comparisons["within_existing_equivalence_tolerance"] = (
        comparisons["max_probability_delta"] < .02 and comparisons["max_candidate_mass_delta"] < .02
        and comparisons["changed_selected"] == 0 and comparisons["changed_top1"] == 0
        and comparisons["changed_abstention"] == 0 and comparisons["changed_input_tokens"] == 0)
    comparisons["exact_evidence_match"] = (
        comparisons["within_existing_equivalence_tolerance"] and comparisons["changed_ranking"] == 0
        and comparisons["max_raw_logit_delta"] == 0 and comparisons["max_probability_delta"] == 0
        and comparisons["max_candidate_mass_delta"] == 0)
    quality = {}
    for mode in modes:
        quality[mode] = []
        for repetition in range(1, args.repetitions + 1):
            current = [row for row in rows[mode] if row["repetition"] == repetition]
            accepted = [row for row in current if row["selected"] in ev.CLASSES]
            correct = sum(row["selected"] == row["label"] for row in accepted)
            quality[mode].append({"repetition": repetition, "accepted": len(accepted), "abstained": 120 - len(accepted),
                "accepted_correct": correct, "correct_over_all": correct / 120,
                "accepted_accuracy": correct / len(accepted) if accepted else None,
                "raw_top1_correct": sum(row["raw_top1"] == row["label"] for row in current)})
    summary = {"protocol": "Identical 4-image HTTP groups, one independent decision per image; RTX 3080 CUDA, context4096, threads4, width4; baseline fresh/full/batch256/flashoff, candidate settings recorded in commands; one warmup group then repeated full120 passes per mode. Filenames and labels excluded.",
        "sample_size": 120, "repetitions": args.repetitions, "mode_order": list(modes),
        "baseline_recorded_run": str(args.baseline_recorded_run) if args.baseline_recorded_run else None,
        "source_sha256": ev.file_digest(args.source), "selection_sha256": ev.file_digest(args.selection),
        "binary_sha256": ev.file_digest(args.binary), "model_sha256": ev.file_digest(args.model),
        "mmproj_sha256": ev.file_digest(args.mmproj), "model": str(args.model),
        "instruction": instruction, "options": options, "min_top_probability": .8, "min_candidate_mass": .05,
        "timing": {mode: runs[mode][1] for mode in modes}, "quality": quality, "comparison": comparisons,
        "parallel_http_time_change_percent": (runs["parallel"][1]["http_latency_sum_ms"] / runs["fresh"][1]["http_latency_sum_ms"] - 1) * 100,
        "parallel_wave_metrics": [sample["response"]["backend"]["details"].get("vision_batch") for sample in runs["parallel"][0]]}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for mode in modes:
        (args.output / mode / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows[mode]))
    print(json.dumps({key: summary[key] for key in ("timing", "quality", "comparison", "parallel_http_time_change_percent")}, indent=2))


if __name__ == "__main__":
    main()
