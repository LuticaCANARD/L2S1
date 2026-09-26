import { readFile, writeFile, mkdir, readdir, copyFile, chmod, stat } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const { values } = parseArgs({ options: {
  'cargo-log': { type: 'string' }, binary: { type: 'string' }, 'library-dir': { type: 'string' },
  devices: { type: 'string', default: 'cpu' }, output: { type: 'string' },
} });
const platform = `${process.platform}-${process.arch}`;
if (!['linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64'].includes(platform)) throw new Error(`Unsupported package platform ${platform}`);
let binary = values.binary;
let libraryDirectory = values['library-dir'];
if (values['cargo-log']) {
  const messages = (await readFile(values['cargo-log'], 'utf8')).split('\n').filter(Boolean).map((line) => JSON.parse(line));
  binary ??= messages.findLast((message) => message.reason === 'compiler-artifact' && message.target.name === 'l2s1' && message.executable)?.executable;
  const build = messages.findLast((message) => message.reason === 'build-script-executed' && message.package_id.includes('l2s1-llama-sys'));
  if (build && !libraryDirectory) {
    const output = await readFile(join(dirname(build.out_dir), 'output'), 'utf8');
    libraryDirectory = /^cargo::metadata=runtime_libdir=(.+)$/m.exec(output)?.[1]?.trim();
    // Older Unix builds publish libdir only.
    libraryDirectory ??= /^cargo::metadata=libdir=(.+)$/m.exec(output)?.[1]?.trim();
  }
}
if (!binary || !libraryDirectory) throw new Error('Supply --cargo-log build.jsonl or both --binary and --library-dir');
const devices = values.devices.split(',');
if (!devices.includes('cpu') || devices.some((device) => !['cpu', 'metal'].includes(device)) || (devices.includes('metal') && process.platform !== 'darwin')) {
  throw new Error('Published runtime devices must be cpu, or cpu,metal on macOS');
}
const manifest = JSON.parse(await readFile(join(root, 'sdks/typescript/package.json'), 'utf8'));
const directory = resolve(values.output ?? join(root, 'sdks/typescript/runtime-packages', platform));
try { await stat(directory); throw new Error(`Output already exists: ${directory}; use a fresh --output directory`); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const bin = join(directory, 'bin');
await mkdir(bin, { recursive: true });
const executableName = process.platform === 'win32' ? 'l2s1.exe' : 'l2s1';
await copyFile(binary, join(bin, executableName));
await chmod(join(bin, executableName), 0o755);
const pattern = process.platform === 'win32' ? /^(?:lib)?(?:llama|mtmd|ggml).*\.dll$/i
  : process.platform === 'darwin' ? /^lib(?:llama|mtmd|ggml).*\.dylib$/ : /^lib(?:llama|mtmd|ggml).*\.so(?:\..*)?$/;
const libraries = (await readdir(libraryDirectory)).filter((name) => pattern.test(name)).sort();
for (const required of ['llama', 'mtmd', 'ggml', 'ggml-base', 'ggml-cpu']) {
  if (!libraries.some((name) => name.startsWith(`lib${required}.`) || name.startsWith(`${required}.`))) {
    throw new Error(`Missing runtime library ${required} in ${libraryDirectory}`);
  }
}
for (const name of libraries) await copyFile(join(libraryDirectory, name), join(bin, name));
const files = [];
for (const name of [executableName, ...libraries]) {
  const content = await readFile(join(bin, name));
  files.push({ path: `bin/${name}`, bytes: content.length, sha256: createHash('sha256').update(content).digest('hex') });
}
await writeFile(join(directory, 'package.json'), JSON.stringify({
  name: `@l2s1/runtime-${platform}`, version: manifest.version,
  description: `Prebuilt L2S1 Rust inference engine for ${platform}`,
  license: 'MIT', os: [process.platform], cpu: [process.arch],
  publishConfig: manifest.publishConfig,
  ...(process.platform === 'linux' ? { libc: ['glibc'] } : {}),
  files: ['bin', 'runtime-manifest.json', 'LICENSE', 'THIRD_PARTY_LICENSES.txt', 'README.md'],
  repository: manifest.repository,
  l2s1: { devices },
}, null, 2) + '\n');
await writeFile(join(directory, 'runtime-manifest.json'), JSON.stringify({ platform, version: manifest.version, devices, files }, null, 2) + '\n');
for (const name of ['LICENSE', 'THIRD_PARTY_LICENSES.txt']) await copyFile(join(root, name), join(directory, name));
await writeFile(join(directory, 'README.md'), `# L2S1 ${platform} runtime\n\nPrebuilt Rust executable and matching llama.cpp/GGML libraries for @l2s1/node ${manifest.version}. Devices: ${devices.join(', ')}. Model weights are separate. This package has no install scripts.\n\n${process.platform === 'linux' ? 'Requires glibc and the system C++ runtime; musl/Alpine is not supported.' : process.platform === 'win32' ? 'Requires the Microsoft Visual C++ 2015-2022 x64 runtime.' : 'Uses the system macOS C++ runtime.'}\n`);
console.log(directory);
