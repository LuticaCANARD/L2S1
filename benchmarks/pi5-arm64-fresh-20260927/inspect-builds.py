"""Bind installed bytes to their CMake source/flags, and inspect real backend selection."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(sys.argv[1])

def run(args, env=None):
    return subprocess.check_output(args, env=env, text=True, stderr=subprocess.STDOUT).strip()

def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

if len(sys.argv) > 2 and sys.argv[2] == 'probe':
    base = ROOT
    ctypes.CDLL(str(base / 'libggml.so')).ggml_backend_load_all_from_path(ctypes.c_char_p(os.fsencode(base)))
    llama = ctypes.CDLL(str(base / 'libllama.so'))
    llama.llama_print_system_info.restype = ctypes.c_char_p
    scores = {}
    for path in sorted(base.glob('libggml-cpu-*.so')):
        scores[path.name] = ctypes.CDLL(str(path)).ggml_backend_score()
    print(json.dumps({'system_info': llama.llama_print_system_info().decode(), 'scores': scores}))
    sys.exit()

report = {}
for variant in ['baseline', 'kernels', 'openmp', 'combined']:
    package = ROOT / 'installed' / variant / 'node_modules/@l2s1/runtime-linux-arm64'
    manifest = json.loads((package / 'runtime-manifest.json').read_text())
    for file in manifest['files']:
        assert digest(package / file['path']) == file['sha256']
    binary = package / 'bin/l2s1'
    dynamic = run(['readelf', '-d', str(binary)])
    paths = re.search(r'\((?:RUNPATH|RPATH)\).*\[(.*?)\]', dynamic).group(1).split(':')
    lib = next(Path(p) for p in paths if p.startswith(str(ROOT)) and Path(p).name == 'lib')
    build = lib.parent / 'build'
    cache = (build / 'CMakeCache.txt').read_text()
    source = Path((build / 'l2s1-llama-source.txt').read_text())
    files = sorted(p for folder in ['src', 'include', 'ggml', 'common', 'tools/mtmd'] for p in (source / folder).rglob('*') if p.is_file())
    source_digest = hashlib.sha256()
    for p in files:
        source_digest.update(str(p.relative_to(source)).encode() + b'\0' + p.read_bytes())
    flags = {str(p.relative_to(build)): p.read_text() for p in build.glob('_deps/llama_cpp-build/ggml/src/**/flags.make')}
    env = {**os.environ, 'LD_LIBRARY_PATH': str(package / 'bin')}
    env.pop('GGML_BACKEND_PATH', None)
    ldd = {file['path']: run(['ldd', str(package / file['path'])], env) for file in manifest['files']}
    for output in ldd.values():
        assert 'not found' not in output
        for line in output.splitlines():
            if re.search(r'lib(?:llama|ggml|mtmd)', line):
                assert str(package / 'bin') in line
    instructions = {}
    for path in sorted((package / 'bin').glob('libggml-cpu*.so')):
        asm = run(['objdump', '-d', str(path)])
        instructions[path.name] = {op: len(re.findall(r'\b'+op+r'\b', asm)) for op in ['sdot', 'udot', 'smmla', 'ummla']}
    probe = run([sys.executable, __file__, str(package / 'bin'), 'probe'], env)
    report[variant] = {'runtime_manifest': manifest, 'native_install_libdir': str(lib), 'source': str(source),
                       'source_tree_sha256': source_digest.hexdigest(),
                       'cache': [line for line in cache.splitlines() if re.match(r'(GGML_|LLAMA_BUILD_COMMIT|CMAKE_(C|CXX)_(COMPILER|FLAGS)|CMAKE_BUILD_TYPE|CMAKE_INSTALL_(LIBDIR|BINDIR|RPATH))[^=]*=', line)],
                       'cpu_flags': flags, 'ldd': ldd, 'instructions': instructions, 'probe': probe}
(ROOT / 'build-identity.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: {'source_tree_sha256': v['source_tree_sha256'], 'instructions': v['instructions'], 'probe': v['probe']} for k,v in report.items()}, indent=2))
