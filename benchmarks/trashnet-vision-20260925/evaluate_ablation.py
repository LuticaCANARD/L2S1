#!/usr/bin/env python3
"""Run the frozen vision optimization matrix for one local model/projector pair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "selection", "binary", "model", "mmproj", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("baseline-binary", "baseline-runtime-dir", "baseline-recorded-run"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--listen", default="127.0.0.1:18778")
    parser.add_argument("--repetitions", type=int, default=1,
                        help="One pass screens equivalence; repeated finalist runs measure latency")
    args = parser.parse_args()
    configuration = json.loads(Path(__file__).with_name("ablation-config.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for variant in configuration["variants"]:
        destination = args.output / variant["id"]
        command = [sys.executable, str(Path(__file__).with_name("evaluate_batch.py")),
                   "--output", str(destination), "--listen", args.listen,
                   "--repetitions", str(args.repetitions)]
        for name in ("source", "selection", "binary", "model", "mmproj",
                     "baseline_binary", "baseline_runtime_dir", "baseline_recorded_run"):
            value = getattr(args, name)
            if value is not None:
                command += ["--" + name.replace("_", "-"), str(value)]
        command += variant["flags"]
        with (args.output / (variant["id"] + ".log")).open("w") as log:
            result = subprocess.run(command, env=dict(os.environ, L2S1_LOG="warn"),
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{variant['id']} failed ({result.returncode}); inspect its log")
        summary = json.loads((destination / "summary.json").read_text())
        summaries[variant["id"]] = summary
        (args.output / "ablation-summary.json").write_text(
            json.dumps({"configuration": configuration, "variants": summaries}, indent=2) + "\n")
        print(variant["id"], "exact:", summary["comparison"]["exact_evidence_match"],
              "ms/image:", round(summary["timing"]["parallel"]["effective_ms_per_image"], 2),
              flush=True)


if __name__ == "__main__":
    main()
