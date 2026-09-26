"""Validate one release's versions and archives before any registry write."""
from __future__ import annotations
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import tarfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ('linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64')
CRATES = ('l2s1-llama-sys', 'l2s1')


def versions(root: Path, tag: str) -> str:
    if not re.fullmatch(r'v(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)', tag):
        raise ValueError('Release tag must be vMAJOR.MINOR.PATCH (stable versions only)')
    version = tag[1:]
    for path, key in [('Cargo.toml', 'package'), ('crates/l2s1-llama-sys/Cargo.toml', 'package'),
                      ('python/pyproject.toml', 'project')]:
        if tomllib.loads((root / path).read_text())[key]['version'] != version:
            raise ValueError(f'Version mismatch in {path}')
    sdk = json.loads((root / 'typescript/package.json').read_text())
    lock = json.loads((root / 'typescript/package-lock.json').read_text())
    if sdk['version'] != version or lock['version'] != version or lock['packages']['']['version'] != version:
        raise ValueError('TypeScript version/lock mismatch')
    expected = {f'@l2s1/runtime-{platform}': version for platform in PLATFORMS}
    if sdk['optionalDependencies'] != expected or lock['packages']['']['optionalDependencies'] != expected:
        raise ValueError('Native runtime versions do not match wrapper')
    native = (root / 'python/src/l2s1/native.py').read_text()
    exported = (root / 'python/src/l2s1/__init__.py').read_text()
    if f'VERSION = "{version}"' not in native or f'__version__ = "{version}"' not in exported:
        raise ValueError('Python runtime/exported version mismatch')
    return version


def archive_json(archive: tarfile.TarFile, name: str) -> dict:
    stream = archive.extractfile(name)
    if stream is None:
        raise ValueError(f'Missing archive file: {name}')
    return json.load(stream)


def validate_npm(path: Path, name: str, version: str, platform: str | None) -> None:
    with tarfile.open(path, 'r:gz') as archive:
        package = archive_json(archive, 'package/package.json')
        if package['name'] != name or package['version'] != version or package.get('scripts', {}).get('install'):
            raise ValueError(f'Invalid npm package identity or install script: {path.name}')
        if platform:
            manifest = archive_json(archive, 'package/runtime-manifest.json')
            if manifest['platform'] != platform or manifest['version'] != version:
                raise ValueError(f'Runtime identity mismatch: {path.name}')
            executable = 'l2s1.exe' if platform.startswith('win32') else 'l2s1'
            if not any(item['path'] == f'bin/{executable}' for item in manifest['files']):
                raise ValueError('Runtime executable absent')
            for item in manifest['files']:
                if not re.fullmatch(r'bin/[^/]+', item['path']):
                    raise ValueError('Invalid runtime path')
                stream = archive.extractfile('package/' + item['path'])
                if stream is None or archive.getmember('package/' + item['path']).size != item['bytes']:
                    raise ValueError('Missing runtime file or size mismatch')
                if hashlib.file_digest(stream, 'sha256').hexdigest() != item['sha256']:
                    raise ValueError('Runtime checksum mismatch')
        elif 'package/dist/stdio.js' not in archive.getnames():
            raise ValueError('Wrapper lacks compiled stdio SDK')


def validate_assets(directory: Path, version: str) -> list[Path]:
    names = [f'l2s1-node-{version}.tgz', *(f'l2s1-runtime-{p}-{version}.tgz' for p in PLATFORMS),
             f'l2s1-{version}-py3-none-any.whl', f'l2s1-{version}.tar.gz',
             *(f'{crate}-{version}.crate' for crate in CRATES)]
    expected = set(names)
    actual = {path.name for path in directory.iterdir() if path.is_file()} - {'SHA256SUMS', 'release.json'}
    if actual != expected:
        raise ValueError(f'Release asset mismatch: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}')
    validate_npm(directory / names[0], '@l2s1/node', version, None)
    for platform in PLATFORMS:
        validate_npm(directory / f'l2s1-runtime-{platform}-{version}.tgz', f'@l2s1/runtime-{platform}', version, platform)
    wheel = directory / f'l2s1-{version}-py3-none-any.whl'
    with zipfile.ZipFile(wheel) as archive:
        metadata = Parser().parsestr(archive.read(f'l2s1-{version}.dist-info/METADATA').decode())
        if metadata['Name'] != 'l2s1' or metadata['Version'] != version:
            raise ValueError('Wheel identity mismatch')
        for name in ('py.typed', 'stdio.py', 'wire.py'):
            archive.getinfo('l2s1/' + name)
    with tarfile.open(directory / f'l2s1-{version}.tar.gz') as archive:
        stream = archive.extractfile(f'l2s1-{version}/PKG-INFO')
        assert stream is not None
        metadata = Parser().parsestr(stream.read().decode())
        if metadata['Name'] != 'l2s1' or metadata['Version'] != version:
            raise ValueError('Sdist identity mismatch')
    for crate in CRATES:
        with tarfile.open(directory / f'{crate}-{version}.crate') as archive:
            stream = archive.extractfile(f'{crate}-{version}/Cargo.toml')
            assert stream is not None
            manifest = tomllib.loads(stream.read().decode())
            if manifest['package']['name'] != crate or manifest['package']['version'] != version:
                raise ValueError('Cargo archive identity mismatch')
            if any(isinstance(dep, dict) and ('git' in dep or 'path' in dep) for dep in manifest.get('dependencies', {}).values()):
                raise ValueError('Cargo archive contains unpublished source dependencies')
    return [directory / name for name in sorted(names)]


def seal(directory: Path, files: list[Path], tag: str, commit: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Commit must be a full Git SHA')
    hashes = {}
    for path in files:
        with path.open('rb') as stream:
            hashes[path.name] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (directory / 'SHA256SUMS').write_text(''.join(f'{digest}  {name}\n' for name, digest in hashes.items()))
    (directory / 'release.json').write_text(json.dumps({'tag': tag, 'commit': commit, 'files': hashes,
        'cargo_features': ['llama', 'llama-cuda', 'llama-metal', 'openrouter'], 'wgpu': 'source checkout only'}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--assets', type=Path)
    parser.add_argument('--commit')
    args = parser.parse_args()
    version = versions(ROOT, args.tag)
    if args.assets:
        files = validate_assets(args.assets, version)
        if not args.commit:
            parser.error('--commit required when sealing assets')
        seal(args.assets, files, args.tag, args.commit)
    print(version)
