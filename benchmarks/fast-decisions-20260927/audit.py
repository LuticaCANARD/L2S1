#!/usr/bin/env python3
"""Recompute parity from the complete compressed score and input records."""
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def audit(device):
    suffix = "cuda-final" if device == "cuda" else "cpu"
    ours = json.loads(gzip.decompress((ROOT / f"l2s1-{suffix}.json.gz").read_bytes()))
    reference = json.loads(gzip.decompress((ROOT / f"ollaya-{suffix}.json.gz").read_bytes()))
    count = changes = 0
    delta = 0.0
    for a, b in zip(ours["outputs"], reference["outputs"], strict=True):
        assert a["id"] == b["id"]
        for inputs, result, ref in zip(a["inputs"], a["response"]["results"], b["items"], strict=True):
            assert inputs["id"] == result["id"] == ref["id"]
            assert inputs["ids"] == ref["ids"]
            assert inputs["markers"] == ref["markers"]
            scores = result["evidence"]["scores"]
            logits = [v["raw_logit"] for v in scores]
            probabilities = [v["option_probability"] for v in scores]
            assert logits == ref["logits"]
            delta = max(delta, *(abs(x-y) for x, y in zip(probabilities, ref["probabilities"], strict=True)))
            changes += max(range(len(logits)), key=logits.__getitem__) != max(range(len(ref["logits"])), key=ref["logits"].__getitem__)
            count += 1
    assert delta < 1e-12 and changes == 0
    return dict(judgments=count, input_tokens_equal=True, markers_equal=True, raw_logits_equal=True,
        max_probability_delta=delta, top1_changes=changes,
        l2s1={k:ours[k] for k in ["n", "p50_ms", "p95_ms", "scope"]},
        ollaya={k:reference[k] for k in ["n", "p50_ms", "p95_ms", "scope"]})

if __name__ == "__main__":
    print(json.dumps({device:audit(device) for device in ["cpu", "cuda"]}, indent=2))
