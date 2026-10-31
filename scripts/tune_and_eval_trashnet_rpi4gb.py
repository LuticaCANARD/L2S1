#!/usr/bin/env python3
"""Tune and evaluate TrashNet vision classification and 4GB Raspberry Pi CPU profile via l2s1 --tune."""

import argparse
import base64
import gzip
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
DEFAULT_MODELS = ("qwen3vl-2b", "qwen35-2b", "qwen35-4b", "gemma4-e4b")


def argmax(values):
    return max(range(len(values)), key=lambda i: values[i])


def apply_affine_head(logits, head):
    if not head:
        return logits
    weights = head["weights"]
    bias = head["bias"]
    return [
        sum(w * x for w, x in zip(row, logits)) + b
        for row, b in zip(weights, bias)
    ]


def load_split_examples(responses_gz: Path, records: list[dict], variant: str) -> list[dict]:
    by_idx = {}
    with gzip.open(responses_gz, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            idx = row.get("index", -1)
            if idx >= 0 and row.get("variant") == variant:
                result = row["response"]["results"][0]
                evidence = result["evidence"]
                by_idx[idx] = (
                    [s["raw_logit"] for s in evidence["scores"]],
                    evidence["candidate_mass"],
                )
    examples = []
    for idx, rec in enumerate(records):
        logits, mass = by_idx[idx]
        examples.append(
            {
                "state": {"image": rec["name"]},
                "expected": rec["label"],
                "group": f"grp-{rec['group']}",
                "raw_logits": logits,
                "candidate_mass": mass,
            }
        )
    return examples


def per_class_breakdown(held_out: list[dict], head: dict | None, use_head: bool) -> dict:
    counts = {c: {"total": 0, "before": 0, "after": 0} for c in CLASSES}
    for ex in held_out:
        label = ex["expected"]
        raw = ex["raw_logits"]
        tuned = apply_affine_head(raw, head) if use_head and head else raw
        counts[label]["total"] += 1
        if CLASSES[argmax(raw)] == label:
            counts[label]["before"] += 1
        if CLASSES[argmax(tuned)] == label:
            counts[label]["after"] += 1
    before_balanced = sum(
        counts[c]["before"] / counts[c]["total"] for c in CLASSES
    ) / len(CLASSES)
    after_balanced = sum(
        counts[c]["after"] / counts[c]["total"] for c in CLASSES
    ) / len(CLASSES)
    return {
        "by_class": counts,
        "balanced_accuracy_before": before_balanced,
        "balanced_accuracy_after": after_balanced,
    }


def run_benchmark_tuning(
    binary: Path,
    helper_model: Path,
    benchmark_dir: Path,
    variant: str,
    models: list[str],
    target_accepted_accuracy: float,
    min_coverage: float,
    l2_regularization: float,
) -> dict:
    dev_records = json.loads((benchmark_dir / "development.json").read_text())["records"]
    test_records = json.loads((benchmark_dir / "test.json").read_text())["records"]
    prompts = json.loads((benchmark_dir / "prompts.json").read_text())
    prompt_spec = prompts[variant]

    decision = {
        "id": "material",
        "instruction": prompt_spec["instruction"],
        "kind": {
            "type": "choice",
            "options": prompt_spec["options"],
        },
    }

    report = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for model in models:
            dev_ex = load_split_examples(
                benchmark_dir / "develop" / model / "responses.jsonl.gz",
                dev_records,
                variant,
            )
            test_ex = load_split_examples(
                benchmark_dir / "evaluate" / model / "responses.jsonl.gz",
                test_records,
                variant,
            )
            model_results = {}
            for mode in ("policy-and-temperature", "logit-affine", "auto"):
                req = {
                    "decision": decision,
                    "examples": dev_ex,
                    "held_out": test_ex,
                    "mode": mode.replace("-", "_"),
                    "target_accepted_accuracy": target_accepted_accuracy,
                    "min_coverage": min_coverage,
                    "l2_regularization": l2_regularization,
                    "apply": False,
                }
                req_path = tmp / f"trashnet_{model}_{mode}.json"
                req_path.write_text(json.dumps(req))
                proc = subprocess.run(
                    [
                        str(binary),
                        "--model",
                        str(helper_model),
                        "--tune",
                        "--tune-mode",
                        mode,
                        "--input",
                        str(req_path),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                res = json.loads(proc.stdout)
                use_head = res["recommended_method"] == "logit_affine_head"
                breakdown = per_class_breakdown(test_ex, res.get("output_head"), use_head)
                model_results[mode] = {
                    "response": res,
                    "breakdown": breakdown,
                }
            report[model] = model_results
    return {"decision": decision, "models": report}


def export_rpi4gb_head(
    binary: Path,
    model_path: Path,
    mmproj_path: Path,
    tuned_head: dict,
    output_head_path: Path,
    context: int = 2048,
    batch: int = 256,
    threads: int = 4,
) -> dict:
    inspect_proc = subprocess.run(
        [
            str(binary),
            "--model",
            str(model_path),
            "--mmproj",
            str(mmproj_path),
            "--device",
            "cpu",
            "--context",
            str(context),
            "--batch",
            str(batch),
            "--ubatch",
            str(batch),
            "--threads",
            str(threads),
            "--inspect",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    identity = json.loads(inspect_proc.stdout)["identity"]
    head = dict(tuned_head)
    head["id"] = f"trashnet-{model_path.stem.lower()}-rpi4gb-cpu"
    head["model_sha256"] = identity["weights_sha256"]
    head["device"] = identity["device"]
    head["compute"] = identity["compute"]
    head["prompt_version"] = identity["prompt_version"]
    output_head_path.write_text(json.dumps(head, indent=2) + "\n")
    return head


def evaluate_live_cpu_images(
    binary: Path,
    model_path: Path,
    mmproj_path: Path,
    output_head_path: Path,
    decision: dict,
    test_records: list[dict],
    images_dir: Path,
    limit: int,
    context: int = 2048,
    batch: int = 256,
    threads: int = 4,
) -> dict:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    cmd = [
        str(binary),
        "--model",
        str(model_path),
        "--mmproj",
        str(mmproj_path),
        "--output-head",
        str(output_head_path),
        "--device",
        "cpu",
        "--execution-mode",
        "fresh",
        "--context",
        str(context),
        "--batch",
        str(batch),
        "--ubatch",
        str(batch),
        "--threads",
        str(threads),
        "--listen",
        f"127.0.0.1:{port}",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(360):
            if proc.poll() is not None:
                raise RuntimeError(f"server exited ({proc.returncode}): {proc.stderr.read()}")
            try:
                with urlopen(url + "/healthz", timeout=1) as resp:
                    if resp.status == 200:
                        break
            except (URLError, TimeoutError):
                time.sleep(0.25)
        else:
            raise RuntimeError("server readiness timeout")

        subset = test_records[:limit] if limit > 0 else test_records
        correct = 0
        latencies = []
        items = []
        for idx, rec in enumerate(subset):
            img_path = images_dir / Path(rec["name"]).name
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            payload = {
                "state": {},
                "decisions": [decision],
                "media": [{"id": "image", "type": "image", "data_base64": b64}],
            }
            t0 = time.perf_counter()
            req = Request(
                url + "/v1/decisions",
                json.dumps(payload).encode("utf-8"),
                {"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=180) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)
            result = body["results"][0]
            pred = result["value"]["selected"]
            ok = pred == rec["label"]
            if ok:
                correct += 1
            items.append(
                {
                    "index": idx,
                    "image": Path(rec["name"]).name,
                    "label": rec["label"],
                    "predicted": pred,
                    "correct": ok,
                    "top_probability": result["evidence"]["top_option_probability"],
                    "calibration_id": result["evidence"]["calibration_id"],
                    "latency_ms": elapsed_ms,
                }
            )
            print(
                f"[{idx + 1:3d}/{len(subset)}] label={rec['label']:10s} -> pred={pred:10s} "
                f"({'OK' if ok else 'ERR'}) | {elapsed_ms:7.1f} ms | "
                f"prob={result['evidence']['top_option_probability']:.4f}"
            )
        return {
            "evaluated": len(subset),
            "correct": correct,
            "accuracy": correct / len(subset) if subset else 0.0,
            "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
            "items": items,
        }
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("target/debug/l2s1"))
    parser.add_argument(
        "--helper-model",
        type=Path,
        default=Path("models/SmolLM2-135M-Instruct-Q8_0.gguf"),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("benchmarks/trashnet-task-prompt-20261001"),
    )
    parser.add_argument("--variant", default="task_options")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--target-accepted-accuracy", type=float, default=0.93)
    parser.add_argument("--min-coverage", type=float, default=0.70)
    parser.add_argument("--l2", type=float, default=0.05)
    parser.add_argument("--rpi-model", type=Path)
    parser.add_argument("--rpi-mmproj", type=Path)
    parser.add_argument("--export-head", type=Path)
    parser.add_argument(
        "--live-images-dir",
        type=Path,
        default=Path("web/static/trashnet/test-images"),
    )
    parser.add_argument("--live-limit", type=int, default=0)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    summary = run_benchmark_tuning(
        binary=args.binary,
        helper_model=args.helper_model,
        benchmark_dir=args.benchmark_dir,
        variant=args.variant,
        models=args.models,
        target_accepted_accuracy=args.target_accepted_accuracy,
        min_coverage=args.min_coverage,
        l2_regularization=args.l2,
    )

    for model, modes in summary["models"].items():
        print(f"\n=== {model} ===")
        for mode, entry in modes.items():
            res = entry["response"]
            bd = entry["breakdown"]
            u_ho = res["uncalibrated_held_out"]
            t_ho = res["tuned_held_out"]
            print(
                f"  [{mode:22s}] method={res['recommended_method']:22s} | "
                f"Test RawTop1: {u_ho['raw_top_accuracy']*100:5.2f}% -> {t_ho['raw_top_accuracy']*100:5.2f}% | "
                f"Balanced: {bd['balanced_accuracy_before']*100:5.2f}% -> {bd['balanced_accuracy_after']*100:5.2f}% | "
                f"Accepted: {t_ho['accepted_accuracy']*100:5.2f}% @ {t_ho['coverage']*100:5.1f}% cov | "
                f"NLL: {u_ho['nll']:.4f} -> {t_ho['nll']:.4f}"
            )

    if args.rpi_model and args.rpi_mmproj and args.export_head:
        qwen_head = summary["models"]["qwen3vl-2b"]["auto"]["response"]["output_head"]
        export_rpi4gb_head(
            binary=args.binary,
            model_path=args.rpi_model,
            mmproj_path=args.rpi_mmproj,
            tuned_head=qwen_head,
            output_head_path=args.export_head,
        )
        print(f"\nExported 4GB Raspberry Pi CPU OutputHead to {args.export_head}")
        if args.live_limit > 0:
            test_records = json.loads((args.benchmark_dir / "test.json").read_text())["records"]
            live = evaluate_live_cpu_images(
                binary=args.binary,
                model_path=args.rpi_model,
                mmproj_path=args.rpi_mmproj,
                output_head_path=args.export_head,
                decision=summary["decision"],
                test_records=test_records,
                images_dir=args.live_images_dir,
                limit=args.live_limit,
            )
            summary["live_cpu_evaluation"] = live

    if args.output_json:
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
