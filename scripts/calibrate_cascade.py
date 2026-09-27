#!/usr/bin/env python3
"""Fit per-schema thresholds on calibration rows, validate on a disjoint held-out set.
JSONL: {"id":"unique","request":{...},"expected":{"decision-id":"option-id"}}.
Endpoints must already be running. This produces empirical evidence, not an error guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

def api(base, path, body=None):
    request = urllib.request.Request(base.rstrip("/") + path, data=None if body is None else json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

def dataset(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError("dataset must contain unique IDs")
    return rows

def score_rows(base, rows):
    result = {}
    for row in rows:
        output = api(base, "/v1/decisions", row["request"])
        for decision, answer in zip(row["request"]["decisions"], output["results"], strict=True):
            # Use the Rust serializer's schema identity exposed by the backend.
            schema = answer["evidence"]["schema_sha256"]
            scores = answer["evidence"]["scores"]
            top = max(scores, key=lambda score: score["option_probability"])
            expected = row["expected"][decision["id"]]
            if expected not in [s["id"] for s in scores]:
                raise ValueError("expected label is not a candidate ID")
            result.setdefault(schema, []).append((top["option_probability"], top["id"] != expected, answer["status"] == "selected"))
    return result

def fit(calibration, validation, max_error):
    rules = []
    for schema, rows in calibration.items():
        thresholds = sorted({p for p, _, selected in rows if selected})
        chosen = None
        for threshold in thresholds:
            accepted = [wrong for p, wrong, selected in rows if selected and p >= threshold]
            if accepted and sum(accepted) / len(accepted) <= max_error:
                chosen = threshold
                break
        if chosen is None or schema not in validation:
            continue
        heldout = validation[schema]
        accepted = [wrong for p, wrong, selected in heldout if selected and p >= chosen]
        # Never retune using validation labels. Reject the entire failing rule.
        if accepted and sum(accepted) / len(accepted) <= max_error:
            rules.append(dict(schema_sha256=schema, min_probability=chosen,
                validation_items=len(heldout), accepted=len(accepted), errors=sum(accepted), max_empirical_error=max_error))
    return rules

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fast-url", "slow-url"):
        parser.add_argument("--" + name, required=True)
    for name in ("calibration", "validation", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--max-error", type=float, default=0.05)
    args = parser.parse_args()
    if not 0 <= args.max_error <= 1:
        raise ValueError("max-error must be in [0,1]")
    calibration, validation = dataset(args.calibration), dataset(args.validation)
    if {r["id"] for r in calibration} & {r["id"] for r in validation} or {canonical(r["request"]["state"]) for r in calibration} & {canonical(r["request"]["state"]) for r in validation}:
        raise ValueError("calibration and validation must have disjoint IDs and exact states")
    fast, slow = api(args.fast_url, "/v1/capabilities"), api(args.slow_url, "/v1/capabilities")
    rules = fit(score_rows(args.fast_url, calibration), score_rows(args.fast_url, validation), args.max_error)
    for rule in rules:
        for name in ("calibration", "validation"):
            rule[name + "_dataset_sha256"] = hashlib.sha256(getattr(args, name).read_bytes()).hexdigest()
    policy = dict(version=1, fast_artifact_id=fast["artifact_id"], slow_artifact_id=slow["artifact_id"], rules=rules)
    args.output.write_text(json.dumps(policy, indent=2) + "\n")
    print(f"Wrote {len(rules)} validated rules; unmatched schemas use the slow model. No production error guarantee.")

if __name__ == "__main__":
    main()
