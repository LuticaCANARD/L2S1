"""Build the explicitly native Cargo distribution without changing the checkout."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import shutil
import tomllib

ROOT = Path(__file__).resolve().parents[1]
WGPU_DEPENDENCIES = ('rullama-engine', 'wgpu', 'pollster', 'async-trait', 'image', 'half')


def prepare(root: Path, output: Path) -> None:
    if output.exists():
        raise ValueError('Cargo staging directory must be new')
    text = (root / 'Cargo.toml').read_text()
    text = text.replace('members = [".", "crates/l2s1-llama-sys", "crates/l2s1-tools"]',
                        'members = [".", "crates/l2s1-llama-sys"]')
    text = re.sub(r'^wgpu = .*\n', '', text, flags=re.M)
    for name in WGPU_DEPENDENCIES:
        text = re.sub(r'^' + re.escape(name) + r' = .*\n', '', text, flags=re.M)
    text = re.sub(r'\n\[\[bin\]\]\nname = "l2s1-wgpu".*?(?=\n\[|\Z)', '', text, flags=re.S)
    parsed = tomllib.loads(text)
    assert set(parsed['workspace']['members']) == {'.', 'crates/l2s1-llama-sys'}
    assert 'wgpu' not in parsed['features']
    assert not any(isinstance(dep, dict) and 'git' in dep for dep in parsed['dependencies'].values())
    output.mkdir(parents=True)
    (output / 'Cargo.toml').write_text(text)
    for name in ('src', 'examples', 'crates/l2s1-llama-sys'):
        shutil.copytree(root / name, output / name)
    # wgpu remains source-checkout only; do not trigger unknown-feature warnings.
    for path in (output / 'src').rglob('*.rs'):
        source = path.read_text()
        source = source.replace('feature = "wgpu"', 'any()')
        path.write_text(source)
    (output / 'src/bin/l2s1-wgpu.rs').unlink(missing_ok=True)
    for name in ('build.rs', 'LICENSE', 'THIRD_PARTY_LICENSES.txt', 'README.md'):
        shutil.copyfile(root / name, output / name)
    # The staged source is immutable after packaging and all publication jobs use it.
    (output / 'README.md').write_text((output / 'README.md').read_text() +
        '\n\nCargo registry distribution: native CPU/CUDA/Metal and OpenRouter only. '
        'WGPU requires the source checkout because its engine dependency is unpublished.\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(ROOT, args.output.resolve())
