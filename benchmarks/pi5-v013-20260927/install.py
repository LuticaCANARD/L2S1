import hashlib,json,pathlib,tarfile,urllib.request
root=pathlib.Path(__file__).resolve().parent
base='https://github.com/LuticaCANARD/L2S1/releases/download/v0.1.3/'
def download(name):
    path=root/name
    with urllib.request.urlopen(base+name,timeout=90) as src,path.open('xb') as dst:
        while chunk:=src.read(1024*1024): dst.write(chunk)
    return path
manifest_path=download('release.json')
manifest=json.loads(manifest_path.read_text())
assert manifest['tag']=='v0.1.3'
name='l2s1-runtime-linux-arm64-0.1.3.tgz'
archive=download(name)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['files'][name]
with tarfile.open(archive) as tf: tf.extractall(root/'runtime',filter='data')
print(json.dumps({'tag':manifest['tag'],'commit':manifest['commit'],'runtime_sha256':manifest['files'][name]}))
