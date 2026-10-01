import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { execFileSync, spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { resolve, join, delimiter } from 'node:path';

const directory = resolve(process.argv[2]);
const manifest = JSON.parse(await readFile(join(directory, 'runtime-manifest.json'), 'utf8'));
for (const file of manifest.files) {
  const data = await readFile(join(directory, file.path));
  if (data.length !== file.bytes || createHash('sha256').update(data).digest('hex') !== file.sha256) throw new Error(`Artifact checksum mismatch: ${file.path}`);
}
const cpuVariants = manifest.files.filter((file) => /^bin\/libggml-cpu-.*\.so$/.test(file.path));
if (manifest.platform === 'linux-arm64' && cpuVariants.length &&
    !cpuVariants.some((file) => file.path === 'bin/libggml-cpu-armv8.0_1.so')) {
  throw new Error('ARM64 dispatch bundle is missing the ARMv8 fallback');
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
  // CPU variants are dlopen modules, so ldd on the executable cannot see them.
  for (const file of manifest.files.filter((file) => file.path === 'bin/l2s1' || /\.so(?:\.|$)/.test(file.path))) {
    const dependencies = execFileSync('ldd', [join(directory, file.path)], { env, encoding: 'utf8' });
    for (const line of dependencies.split('\n')) {
      if (line.includes('not found')) throw new Error(`Missing dependency in ${file.path}: ${line}`);
      if (/lib(?:llama|mtmd|ggml)/.test(line) && !line.includes(bin)) throw new Error(`Dependency escaped bundle: ${line}`);
    }
  }
}
if (manifest.platform === 'linux-arm64' && cpuVariants.length) {
  // A nonexistent model exits before backend initialization. Use an existing
  // invalid GGUF so this checks dlopen/dispatch without downloading a model.
  const probeDirectory = await mkdtemp(join(tmpdir(), 'l2s1-backend-probe-'));
  try {
    const model = join(probeDirectory, 'invalid.gguf');
    await writeFile(model, 'invalid GGUF');
    const probe = spawnSync(executable, ['--model', model, '--execution-mode', 'fresh'], {
      env: { ...env, L2S1_LOG: 'info' }, encoding: 'utf8', timeout: 30000,
    });
    if (probe.error || probe.status !== 1 ||
        !cpuVariants.some((file) => probe.stderr.includes(`loaded CPU backend from ${join(directory, file.path)}`)) ||
        !probe.stderr.includes('model load failed')) {
      throw new Error(`ARM64 CPU dispatch failed: ${probe.error ?? probe.stderr}`);
    }
  } finally { await rm(probeDirectory, { recursive: true, force: true }); }
}
console.log(`Verified ${manifest.platform}: ${manifest.files.length} bundled files, ${version.trim()}`);
