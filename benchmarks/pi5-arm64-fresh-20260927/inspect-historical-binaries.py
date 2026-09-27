"""Run after timing: inspect actual historical/released libraries, not build intent."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1])
historical=json.loads((root/'historical-live-identity.json').read_text())
def run(args,env=None):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,env=env)
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path(historical['historical_libraries']['libggml-cpu.so.0']['path']).parent
released=Path('/home/lutica/l2s1-v013-validation-20260927/runtime/package/bin')
new=root/'installed/kernels/node_modules/@l2s1/runtime-linux-arm64/bin'
report={}
for name,directory in [('historical_native',old),('released_v013',released),('candidate_kernels',new)]:
 env={**os.environ,'LD_LIBRARY_PATH':str(directory)}
 env.pop('GGML_BACKEND_PATH',None)
 probe=run([sys.executable,str(root/'inspect-builds.py'),str(directory),'probe'],env)
 instructions={}
 for p in directory.glob('libggml-cpu*.so'):
  asm=run(['objdump','-d',str(p)])
  instructions[p.name]={'sha256':digest(p),'sdot':len(re.findall(r'\bsdot\b',asm)),'smmla':len(re.findall(r'\bsmmla\b',asm))}
 versions=set()
 for p in directory.glob('*.so'):
  versions.update(re.findall(r'Name: ((?:GLIBC|GLIBCXX|CXXABI)_[0-9.]+)',run(['readelf','--version-info',str(p)])))
 if (directory/'l2s1').exists():
  versions.update(re.findall(r'Name: ((?:GLIBC|GLIBCXX|CXXABI)_[0-9.]+)',run(['readelf','--version-info',str(directory/'l2s1')])))
 report[name]={'directory':str(directory),'probe':probe,'instructions':instructions,'symbol_versions':sorted(versions)}
report['released_ldd']=run(['ldd',str(released/'l2s1')],{**os.environ,'LD_LIBRARY_PATH':str(released)})
for line in report['released_ldd'].splitlines():
 if any('lib'+name in line for name in ['llama','ggml','mtmd']):assert str(released) in line
report['openmp_system_package']=run(['dpkg-query','-W','libgomp1:arm64'])
gomp=Path('/usr/lib/aarch64-linux-gnu/libgomp.so.1').resolve()
report['openmp_system_library']={'path':str(gomp),'sha256':digest(gomp)}
(root/'historical-binary-inspection.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({name:{'probe':r['probe'],'instructions':r['instructions']} for name,r in report.items() if isinstance(r,dict) and 'probe' in r},indent=2))
