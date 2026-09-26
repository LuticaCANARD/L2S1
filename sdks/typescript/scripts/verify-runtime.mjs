import { readFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { resolve, join, delimiter } from 'node:path';

const directory = resolve(process.argv[2]);
const manifest = JSON.parse(await readFile(join(directory, 'runtime-manifest.json'), 'utf8'));
for (const file of manifest.files) {
  const data = await readFile(join(directory, file.path));
  if (data.length !== file.bytes || createHash('sha256').update(data).digest('hex') !== file.sha256) throw new Error(`Artifact checksum mismatch: ${file.path}`);
}
const bin = join(directory, 'bin');
const env = { ...process.env };
const key = process.platform === 'win32' ? Object.keys(env).find((name) => name.toLowerCase() === 'path') ?? 'PATH'
  : process.platform === 'darwin' ? 'DYLD_LIBRARY_PATH' : 'LD_LIBRARY_PATH';
env[key] = [bin, env[key]].filter(Boolean).join(delimiter);
const executable = join(bin, process.platform === 'win32' ? 'l2s1.exe' : 'l2s1');
const version = execFileSync(executable, ['--version'], { env, encoding: 'utf8' });
if (!version.startsWith(`l2s1 ${manifest.version}`)) throw new Error(`Native version mismatch: ${version}`);
if (process.platform === 'linux') {
  const dependencies = execFileSync('ldd', [executable], { env, encoding: 'utf8' });
  for (const line of dependencies.split('\n')) {
    if (line.includes('not found')) throw new Error(`Missing dependency: ${line}`);
    if (/lib(?:llama|mtmd|ggml)/.test(line) && !line.includes(bin)) throw new Error(`Dependency escaped bundle: ${line}`);
  }
}
console.log(`Verified ${manifest.platform}: ${manifest.files.length} bundled files, ${version.trim()}`);
