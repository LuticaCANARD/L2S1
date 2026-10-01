"""Capture final artifact identities after timing, without mixing native libraries."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
source = root.parent / 'source'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


packages = {
    'before': root.parent / 'installed/kernels/node_modules/@l2s1/runtime-linux-arm64',
    'after': root / 'installed/node_modules/@l2s1/runtime-linux-arm64',
}
report = {'base_commit': '4f6c02fa84cfc192a4f7a49e3a5e14e5eef8c649',
          'upstream_commit': (source / 'crates/l2s1-llama-sys/UPSTREAM_COMMIT').read_text().strip(),
          'source_sha256': {path: digest(source / path) for path in [
              'crates/l2s1-llama-sys/src/text.rs', 'crates/l2s1-llama-sys/src/bridge.rs',
              'crates/l2s1-llama-sys/build.rs', 'crates/l2s1-llama-sys/native/exception.cpp',
              'examples/state_cache_probe.rs', 'Cargo.lock']}, 'artifacts': {}}
for name, package in packages.items():
    manifest = json.loads((package / 'runtime-manifest.json').read_text())
    for entry in manifest['files']:
        assert entry['sha256'] == digest(package / entry['path'])
    report['artifacts'][name] = {'manifest': manifest, 'probe_sha256': digest(root / name / 'probe')}
libraries = [{entry['path']: entry['sha256'] for entry in report['artifacts'][name]['manifest']['files']
              if entry['path'].startswith('bin/lib')} for name in ['before', 'after']]
assert libraries[0] == libraries[1]
report['native_shared_libraries_identical'] = True
report['uname'] = subprocess.check_output(['uname', '-a'], text=True).strip()
report['compiler'] = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
report['scope'] = 'Rust state-restore changes only; both native library sets are byte-identical, pinned and independently manifest-verified. Neither package is a new published release.'
(root / 'identity.json').write_text(json.dumps(report, indent=2)+'\n')
print('Both package manifests verified; native shared libraries are byte-identical')
