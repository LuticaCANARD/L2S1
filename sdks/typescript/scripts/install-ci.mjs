/** Install locked build tools without resolving the runtimes this checkout builds. */
import { spawnSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export function installCi(directory) {
  const manifestPath = resolve(directory, 'package.json');
  const lockPath = resolve(directory, 'package-lock.json');
  const manifestBytes = readFileSync(manifestPath);
  const lockBytes = readFileSync(lockPath);
  const manifest = JSON.parse(manifestBytes);
  const lock = JSON.parse(lockBytes);
  const runtimes = Object.keys(manifest.optionalDependencies ?? {});
  if (runtimes.some(name => !/^@l2s1\/runtime-(linux|darwin|win32)-(x64|arm64)$/.test(name))) {
    throw new Error('Unexpected optional dependency; review the CI installation scope');
  }
  delete manifest.optionalDependencies;
  delete lock.packages[''].optionalDependencies;
  for (const name of runtimes) delete lock.packages[`node_modules/${name}`];
  try {
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
    writeFileSync(lockPath, JSON.stringify(lock, null, 2) + '\n');
    // npm.cmd requires the command shell on Windows; no user input enters this command.
    const result = spawnSync('npm ci --ignore-scripts --no-audit --no-fund', {
      cwd: directory, shell: true, stdio: 'inherit',
    });
    if (result.error) throw result.error;
    if (result.status !== 0) throw new Error(`Locked build-tool installation failed (${result.status ?? result.signal})`);
  } finally {
    writeFileSync(manifestPath, manifestBytes);
    writeFileSync(lockPath, lockBytes);
  }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  installCi(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
}
