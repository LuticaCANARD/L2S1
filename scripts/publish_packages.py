"""Publish verified assets; existing versions must have identical checksums."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def registry(url: str) -> dict | None:
    try:
        with urlopen(Request(url, headers={'User-Agent': 'L2S1-release-pipeline'}), timeout=30) as stream:
            return json.load(stream)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise  # Auth, server and network failures must not become 'missing'.


def digest(path: Path, algorithm: str = 'sha256') -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def publish_npm(assets: Path, version: str) -> None:
    paths = sorted(assets.glob(f'l2s1-runtime-*-{version}.tgz')) + [assets / f'l2s1-node-{version}.tgz']
    for path in paths:
        package = '@l2s1/' + path.name.removesuffix(f'-{version}.tgz').removeprefix('l2s1-')
        metadata = registry(f'https://registry.npmjs.org/{quote(package, safe="")}/{version}')
        if metadata is not None:
            expected = 'sha512-' + base64.b64encode(bytes.fromhex(digest(path, 'sha512'))).decode()
            if metadata.get('dist', {}).get('integrity') != expected:
                raise ValueError(f'Existing npm version differs: {package}@{version}')
            print(f'Already published with matching checksum: {package}@{version}', flush=True)
            continue
        subprocess.run(['npm', 'publish', str(path.resolve()), '--access', 'public', '--ignore-scripts', '--provenance'], check=True)


def check_pypi(assets: Path, version: str) -> None:
    metadata = registry(f'https://pypi.org/pypi/l2s1/{version}/json')
    if metadata is not None:
        expected = {path.name: digest(path) for path in assets.iterdir() if path.suffix == '.whl' or path.name.endswith('.tar.gz')}
        for item in metadata['urls']:
            if expected.get(item['filename']) != item['digests']['sha256']:
                raise ValueError('Existing PyPI version has different or unexpected files')


def publish_cargo(assets: Path, version: str, source: Path) -> None:
    for crate in ('l2s1-llama-sys', 'l2s1'):
        path = assets / f'{crate}-{version}.crate'
        metadata = registry(f'https://crates.io/api/v1/crates/{crate}/{version}')
        if metadata is not None:
            if metadata['version']['checksum'] != digest(path):
                raise ValueError(f'Existing crates.io version differs: {crate}@{version}')
            print(f'Already published with matching checksum: {crate}@{version}', flush=True)
            continue
        command = ['cargo', 'publish', '--locked', '--package', crate]
        subprocess.run([*command, '--dry-run'], cwd=source, check=True)
        packaged = source / 'target/package' / path.name
        if digest(packaged) != digest(path):
            raise ValueError('Cargo repackaged archive differs from verified release artifact')
        subprocess.run(command, cwd=source, check=True)
        # Cargo waits for the registry index to expose this version before returning.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registry', choices=('npm', 'cargo', 'pypi-check'))
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--version', required=True)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.assets / 'release.json').read_text())
    if manifest['tag'] != 'v' + args.version:
        raise ValueError('Artifact version differs from publication version')
    for name, checksum in manifest['files'].items():
        if digest(args.assets / name) != checksum:
            raise ValueError(f'Artifact checksum changed: {name}')
    if args.registry == 'npm':
        publish_npm(args.assets, args.version)
    elif args.registry == 'pypi-check':
        check_pypi(args.assets, args.version)
    else:
        if args.source is None:
            parser.error('--source required for Cargo')
        publish_cargo(args.assets, args.version, args.source)
