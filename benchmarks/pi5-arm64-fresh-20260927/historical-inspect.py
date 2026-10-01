import hashlib,json,subprocess
from pathlib import Path
root=Path('/home/lutica/l2s1-arm64-fresh-20260927')
old=Path('/home/lutica/l2s1-pi-20260927')
published=Path('/home/lutica/l2s1-v013-validation-20260927')
def digest(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT)
exe=old/'source/target/release/deps/benchmark-1634f2fff2589970'
ldd=run(['ldd',str(exe)])
libs={line.split('=>')[0].strip():line.split('=>')[1].split()[0] for line in ldd.splitlines() if '=>' in line and any('lib'+x in line for x in ['llama','ggml','mtmd'])}
source=next((old/'source/target/release/build').glob('l2s1-llama-sys-*/out/llama/*/build/_deps/llama_cpp-src'))
h=hashlib.sha256()
for p in sorted(p for folder in ['src','include','ggml','common','tools/mtmd'] for p in (source/folder).rglob('*') if p.is_file()):
 h.update(str(p.relative_to(source)).encode()+b'\0'+p.read_bytes())
manifest=json.loads((published/'runtime/package/runtime-manifest.json').read_text())
for file in manifest['files']:assert digest(published/'runtime/package'/file['path'])==file['sha256']
build=source.parent.parent
result={'historical_executable_sha256':digest(exe),'historical_ldd':ldd,'historical_libraries':{k:{'path':v,'sha256':digest(Path(v))} for k,v in libs.items()},
 'historical_source_tree_sha256':h.hexdigest(),'historical_upstream_commit':(old/'source/crates/l2s1-llama-sys/UPSTREAM_COMMIT').read_text().strip(),
 'historical_cmake_cache':(build/'CMakeCache.txt').read_text(),
 'historical_cpu_flags':{str(p.relative_to(build)):p.read_text() for p in build.glob('_deps/llama_cpp-build/ggml/src/**/flags.make')},
 'published_runtime_manifest_verified':True,'published_archive_sha256':digest(published/'l2s1-runtime-linux-arm64-0.1.3.tgz'),
 'published_release':json.loads((published/'release.json').read_text()),
 'toolchain':{'gcc':run(['g++','--version']),'rustc':run([str(old/'cargo/bin/rustc'),'--version']),'ldd':run(['ldd','--version'])}}
(root/'historical-live-identity.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k in ['historical_executable_sha256','historical_source_tree_sha256','historical_upstream_commit','published_runtime_manifest_verified','published_archive_sha256']},indent=2))
