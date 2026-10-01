"""Pinned checkpoint download subprocess, bounded by the matrix runner."""
import argparse
from pathlib import Path
from .jev_model_profiles import read_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--profiles', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    a = p.parse_args()
    profile = read_profile(a.model, a.profiles)
    from huggingface_hub import snapshot_download
    snapshot_download(profile['hf_model'], revision=profile['revision'], local_dir=a.checkpoint,
        allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.model', 'LICENSE*', 'NOTICE*'])


if __name__ == '__main__':
    main()
