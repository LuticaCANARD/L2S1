#!/usr/bin/env python3
"""Run the Rust decision benchmark sequentially against existing local GGUF files."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return value


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be a probability in [0, 1]")
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def revision(path):
    if path is None:
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except OSError:
        return None


def llama_revision():
    source = os.environ.get("L2S1_LLAMA_CPP_SOURCE") or os.environ.get("LLAMA_CPP_DIR")
    if source:
        return revision(source)
    marker = ROOT / "crates/l2s1-llama-sys/UPSTREAM_COMMIT"
    return marker.read_text().strip() if marker.is_file() else None


def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator and key.strip() == "model name":
                return value.strip()
    except OSError:
        pass
    return platform.processor() or None


def read_models(manifest, selected):
    models = json.loads(manifest.read_text())["models"]
    if not isinstance(models, list) or not models:
        raise ValueError("manifest.models must be a nonempty list")
    ids = set()
    for model in models:
        name = model["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or name in ids:
            raise ValueError("model IDs must be unique lowercase names without path separators")
        ids.add(name)
        path = Path(model["path"])
        model["path"] = str((ROOT / path).resolve())
    if selected:
        unknown = set(selected) - ids
        if unknown:
            raise ValueError(f"unknown model IDs: {', '.join(sorted(unknown))}")
        models = [m for m in models if m["id"] in selected]
    return models


def build(output, cuda=False):
    feature = "llama-cuda" if cuda else "llama"
    command = ["cargo", "test", "--release", "--locked", "--offline", "--features", feature,
               "--test", "benchmark", "--no-run", "--message-format=json"]
    with (output / "build.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=log, text=True)
        log.write(result.stdout)
    if result.returncode:
        raise RuntimeError(f"release build failed; see {output / 'build.log'}")
    for line in result.stdout.splitlines():
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == "benchmark"
                and message.get("executable")):
            return Path(message["executable"])
    raise RuntimeError("Cargo did not return a benchmark test executable")


def percentage(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def save_summary(output, summary):
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    lines = ["# Local decision benchmark", "", f"Suite: `{summary['suite']}`. Sequential runs; load and warmups excluded from request timings.", "",
             "| Model | Device | Status | Coverage | Accepted accuracy | Correct / all | Raw top-1 | p50 ms/request | p95 ms/request | Decisions/s | Correct accepted/s |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in summary["runs"]:
        if run["status"] != "ok":
            lines.append(f"| {run['model']} | {run['device']} | {run['status']} | — | — | — | — | — | — | — | — |")
            continue
        q, latency = run["quality"], run["latency_ms"]
        lines.append(f"| {run['model']} | {run['device']} | ok | {percentage(q['coverage'])} | {percentage(q['accepted_accuracy'])} | {percentage(q['correct_fraction'])} | {percentage(q['top1_accuracy_before_abstention'])} | {latency['p50']:.1f} | {latency['p95']:.1f} | {run['decisions_per_second']:.2f} | {run['accepted_correct_decisions_per_second']:.2f} |")
    lines.extend(["", "Each request contains three decisions. Accuracy denominators include repeated passes; repeats are not independent examples. Accepted accuracy is n/a when every answer abstains. Raw top-1 ignores policy abstention, with ties counted incorrect. Decisions/s includes abstentions. These synthetic rule tasks do not establish general model quality. Status 'ok' means execution completed, not that every answer was correct or the native consistency suite passed.", "",
                  "Per-run JSON includes all labels, predictions, scores, group metrics, timings, and repeat-consistency counts. Failures and timeouts remain in summary.json and per-run logs. No model weights are copied.", ""])
    (output / "summary.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/fixtures/benchmark_models.json")
    parser.add_argument("--model", action="append", help="select a manifest ID; repeat to select several")
    parser.add_argument("--device", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
    parser.add_argument("--execution-mode", choices=["fresh", "prefix-reuse"], default="fresh")
    parser.add_argument("--prompt-layout", choices=["legacy", "state-first"], default="legacy")
    parser.add_argument("--iterations", type=positive, default=3)
    parser.add_argument("--warmup", type=nonnegative, default=1, help="full warmup passes, excluding first request")
    parser.add_argument("--context", type=positive, default=2048)
    parser.add_argument("--batch", type=positive, default=256)
    parser.add_argument("--threads", type=positive, default=4)
    parser.add_argument("--min-top-probability", type=probability, default=0.8)
    parser.add_argument("--min-candidate-mass", type=probability, default=0.05)
    parser.add_argument("--timeout", type=positive, default=1800, help="seconds per model/device, including warmup")
    parser.add_argument("--output", type=Path, help="new output directory; existing directories are rejected")
    args = parser.parse_args()
    try:
        models = read_models(args.manifest, args.model)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = (args.output or ROOT / "results/benchmark" / timestamp).resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output already exists: {output}; choose a new directory")
    print(f"Reports: {output}", flush=True)
    summary = {
        "schema_version": 1, "created_at": timestamp, "suite": "decision-rules-v1",
        "suite_sha256": sha256(ROOT / "tests/fixtures/decision_benchmark.json"),
        "host": {"platform": platform.platform(), "cpu": cpu_model()},
        "repository_revision": revision(ROOT),
        "llama_cpp_revision": llama_revision(),
        "settings": {key: value for key, value in vars(args).items() if key not in ("manifest", "output")},
        "runs": [],
    }
    save_summary(output, summary)
    try:
        executable = build(output, "cuda" in args.device)
        summary["benchmark_executable_sha256"] = sha256(executable)
    except (RuntimeError, OSError) as error:
        summary["build_error"] = str(error)
        save_summary(output, summary)
        print(error, file=sys.stderr)
        return 1
    for model in models:
        path = Path(model["path"])
        model_hash = None
        preflight_error = None
        try:
            with path.open("rb") as source:
                if source.read(4) != b"GGUF":
                    raise ValueError("model does not have a GGUF header")
            print(f"Hashing {model['id']}…", flush=True)
            model_hash = sha256(path)
        except (OSError, ValueError) as error:
            preflight_error = str(error)
        for device in dict.fromkeys(args.device):
            name = f"{model['id']}.{device}"
            report = output / f"{name}.json"
            log_path = output / f"{name}.log"
            run = {"model": model["id"], "device": device, "sha256": model_hash,
                   "model_file": path.name, "status": "failed", "log": log_path.name}
            if preflight_error:
                run["error"] = preflight_error
                log_path.write_text(preflight_error + "\n")
            else:
                print(f"Running {name}: {args.iterations} measured passes, {args.warmup} warmup passes", flush=True)
                env = os.environ | {
                    "SKID_MODEL": str(path), "SKID_CUDA": "1" if device == "cuda" else "0",
                    "SKID_BENCH_OUTPUT": str(report), "SKID_BENCH_ITERATIONS": str(args.iterations),
                    "SKID_BENCH_WARMUP": str(args.warmup), "SKID_CONTEXT": str(args.context),
                    "SKID_BATCH": str(args.batch), "SKID_THREADS": str(args.threads),
                    "SKID_MIN_TOP_PROBABILITY": str(args.min_top_probability),
                    "SKID_MIN_CANDIDATE_MASS": str(args.min_candidate_mass),
                    "SKID_EXECUTION_MODE": args.execution_mode,
                    "SKID_PROMPT_LAYOUT": args.prompt_layout,
                }
                try:
                    with log_path.open("w") as log:
                        result = subprocess.run(
                            [str(executable), "native::model_decision_benchmark", "--exact", "--ignored", "--nocapture"],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout,
                        )
                    if result.returncode:
                        raise RuntimeError(f"test exited with status {result.returncode}; see {log_path.name}")
                    data = json.loads(report.read_text())
                    if data["suite"] != summary["suite"] or data["iterations"] != args.iterations:
                        raise ValueError("benchmark report does not match requested run")
                    run.update({key: data[key] for key in ("quality", "latency_ms", "decisions_per_second", "accepted_correct_decisions_per_second", "repeat_consistency")})
                    run.update(status="ok", report=report.name)
                except subprocess.TimeoutExpired:
                    run.update(status="timeout", error=f"exceeded {args.timeout} seconds")
                except (RuntimeError, ValueError, OSError, KeyError) as error:
                    run["error"] = str(error)
            summary["runs"].append(run)
            save_summary(output, summary)
            print(f"{name}: {run['status']}", flush=True)
    print(f"Comparison: {output / 'summary.md'}", flush=True)
    return int(any(run["status"] != "ok" for run in summary["runs"]))


if __name__ == "__main__":
    sys.exit(main())
