"""Reject a missing fallback and a dependency missing only from a dlopen module."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

root = Path(sys.argv[1])
original = root / 'builds/kernels/package'
node = root / 'node/bin/node'
verifier = root / 'source/sdks/typescript/scripts/verify-runtime.mjs'
results = []
with tempfile.TemporaryDirectory(prefix='l2s1-package-negative-') as temp:
    temp = Path(temp)
    for kind in ['missing-fallback', 'module-only-missing-dependency']:
        package = temp / kind
        shutil.copytree(original, package)
        manifest = json.loads((package / 'runtime-manifest.json').read_text())
        if kind == 'missing-fallback':
            name = 'bin/libggml-cpu-armv8.0_1.so'
            (package / name).unlink()
            manifest['files'] = [f for f in manifest['files'] if f['path'] != name]
            expected = 'missing the ARMv8 fallback'
        else:
            (temp / 'dependency.c').write_text('void l2s1_test_dependency(void) {}\n')
            (temp / 'module.c').write_text('extern void l2s1_test_dependency(void); void probe(void) { l2s1_test_dependency(); }\n')
            subprocess.run(['cc', '-shared', '-fPIC', str(temp/'dependency.c'), '-o', str(temp/'libl2s1-test-dependency.so')], check=True)
            name = 'bin/libggml-cpu-test.so'
            subprocess.run(['cc', '-shared', '-fPIC', str(temp/'module.c'), '-L'+str(temp), '-ll2s1-test-dependency', '-o', str(package/name)], check=True)
            (temp / 'libl2s1-test-dependency.so').unlink()
            content = (package / name).read_bytes()
            manifest['files'].append({'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()})
            expected = 'Missing dependency in bin/libggml-cpu-test.so'
        (package/'runtime-manifest.json').write_text(json.dumps(manifest))
        result = subprocess.run([str(node), str(verifier), str(package)], text=True, capture_output=True)
        assert result.returncode and expected in result.stderr, result.stderr
        results.append({'case': kind, 'exit_code': result.returncode, 'matched_diagnostic': expected})
(root/'package-negative-tests.json').write_text(json.dumps(results, indent=2)+'\n')
print(json.dumps(results, indent=2))
