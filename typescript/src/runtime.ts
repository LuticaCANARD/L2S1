import { createRequire } from 'node:module';
import { readFileSync, existsSync } from 'node:fs';
import { dirname, join, delimiter } from 'node:path';
import { L2S1Error } from './http.js';

const require = createRequire(import.meta.url);
const packageVersion: string = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).version;
const supported = new Set(['linux-x64', 'linux-arm64', 'darwin-x64', 'darwin-arm64', 'win32-x64']);

export function resolveRuntime(binaryPath?: string, device: 'cpu' | 'cuda' | 'metal' = 'cpu'): { binaryPath: string; env: NodeJS.ProcessEnv } {
  if (binaryPath !== undefined) {
    if (!binaryPath.trim()) throw new TypeError('binaryPath must not be empty');
    return { binaryPath, env: { ...process.env } };
  }
  const platform = `${process.platform}-${process.arch}`;
  if (!supported.has(platform)) {
    throw new L2S1Error(`No bundled runtime for ${platform}; supply binaryPath for a custom Rust build`, 'unsupported_platform');
  }
  const name = `@l2s1/runtime-${platform}`;
  let manifestPath: string;
  try { manifestPath = require.resolve(`${name}/package.json`); }
  catch (cause) {
    throw new L2S1Error(`Install ${name}@${packageVersion} or supply binaryPath. Keep npm optional dependencies enabled.`,
      'runtime_not_installed', undefined, undefined, undefined, { cause });
  }
  const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
  if (manifest.version !== packageVersion || manifest.name !== name) {
    throw new L2S1Error(`Runtime version mismatch: expected ${name}@${packageVersion}`, 'runtime_version_mismatch');
  }
  if (!Array.isArray(manifest.l2s1?.devices) || !manifest.l2s1.devices.includes(device)) {
    throw new L2S1Error(`Bundled ${platform} runtime does not support ${device}; supply binaryPath for that device`, 'unsupported_device');
  }
  const directory = join(dirname(manifestPath), 'bin');
  const executable = join(directory, process.platform === 'win32' ? 'l2s1.exe' : 'l2s1');
  if (!existsSync(executable)) throw new L2S1Error(`Bundled executable is missing: ${executable}`, 'runtime_not_installed');
  const env = { ...process.env };
  const variable = process.platform === 'win32' ? 'PATH' : process.platform === 'darwin' ? 'DYLD_LIBRARY_PATH' : 'LD_LIBRARY_PATH';
  // Windows environment variable names are case insensitive; do not emit both Path and PATH.
  const existingKey = process.platform === 'win32' ? Object.keys(env).find((key) => key.toLowerCase() === 'path') ?? variable : variable;
  env[existingKey] = [directory, env[existingKey]].filter(Boolean).join(delimiter);
  return { binaryPath: executable, env };
}
