#!/usr/bin/env python3
"""Download the pinned English Laya fp32 graph and author weights; verify every digest.
No inference or dependency installation. Use --output to choose a model directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

DERIVED = "https://huggingface.co/ollaya-dev/laya/resolve/32fdc8cc3d76e77c28ad8ad9187c0cf3bd8cc0a2/en/"
AUTHOR = "https://huggingface.co/convaiinnovations/laya/resolve/aa8c91ca088ec597df95a0d1c76b3063cb2ae5e8/"
FILES = {
    "model.onnx": (DERIVED + "model-fp32.onnx", "9dc0bb17f5f7e389cc4b99cdd14f76a1043a2a60ed91daf2358524b80ab35711"),
    "model.safetensors": (AUTHOR + "model.safetensors", "891102d372688fc2a094dac56a384bc537b87c63f21f9f3dac0be2b7cbc8d86c"),
    "tokenizer.json": (AUTHOR + "tokenizer/tokenizer.json", "6c8aaa9a542084f2457eab775d4eeb51f92a70c0fd9de28d5edb0ddec3c08d30"),
    "decision.json": (DERIVED + "decision.json", "dff4b9fc6cb4285be70fa781cb9d1876a8061915808df41b63ffe7ddb13dbd2c"),
    "calibration.json": (DERIVED + "calibration.json", "a7af5a5fafb5b418420a63511194038faed0aade75af68d210144e67fd792426"),
}

def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    if args.precision == "fp16":
        FILES["model.onnx"] = (DERIVED + "model-fp16.onnx", "32b6a61761499a64d8f20406ed1baa5fe6f0e860a65717c03f804b15bc958886")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, (url, expected) in FILES.items():
        path = args.output / name
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".partial")
            print(f"Downloading {name}", flush=True)
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            if digest(temporary) != expected:
                raise ValueError(f"SHA-256 mismatch: {name}")
            temporary.replace(path)
        if digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
    # The unmodified derived graph uses a content-addressed relative weight filename.
    alias = args.output / ("sha256-" + FILES["model.safetensors"][1])
    if not alias.exists():
        alias.hardlink_to(args.output / "model.safetensors")
    if digest(alias) != FILES["model.safetensors"][1]:
        raise ValueError("External ONNX weights differ from the verified weights")
    (args.output / "provenance.json").write_text(json.dumps({"files": FILES, "license": "Apache-2.0", "author": "Convai Innovations", "derived_graph": "Ollaya"}, indent=2) + "\n")
    print(f"Verified Laya {args.precision} model; model weights are licensed Apache-2.0.")

if __name__ == "__main__":
    main()
