import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, existsSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { spawnSync } from 'node:child_process';
import { installCi } from '../scripts/install-ci.mjs';

test('CI installs locked tools with unavailable or published runtimes and restores inputs on failure', () => {
  const root = mkdtempSync(join(tmpdir(), 'l2s1-ci-install-'));
  try {
    const tool = join(root, 'tool'); mkdirSync(tool);
    writeFileSync(join(tool, 'package.json'), JSON.stringify({ name: 'fixture-build-tool', version: '1.0.0' }));
    const packed = spawnSync('npm pack --ignore-scripts --json', { cwd: tool, shell: true, encoding: 'utf8' });
    assert.equal(packed.status, 0, packed.stderr);
    const tarball = join(tool, JSON.parse(packed.stdout)[0].filename).replaceAll('\\', '/');
    const runtime = '@l2s1/runtime-linux-x64';
    const manifest = { name: 'ci-fixture', version: '1.0.0', devDependencies: { 'fixture-build-tool': `file:${tarball}` }, optionalDependencies: { [runtime]: '1.0.0' } };
    const project = join(root, 'project'); mkdirSync(project);
    const manifestPath = join(project, 'package.json');
    const lockPath = join(project, 'package-lock.json');
    // Generate only the real tool lock locally, without contacting a registry.
    writeFileSync(manifestPath, JSON.stringify({ ...manifest, optionalDependencies: {} }));
    const locked = spawnSync('npm install --package-lock-only --ignore-scripts --offline --no-audit --no-fund', { cwd: project, shell: true, encoding: 'utf8' });
    assert.equal(locked.status, 0, locked.stderr);
    const base = JSON.parse(readFileSync(lockPath));
    base.packages[''].optionalDependencies = manifest.optionalDependencies;
    // An unreachable tarball proves that a resolved runtime is not fetched either.
    for (const runtimeEntry of [{ optional: true }, { optional: true, version: '1.0.0', resolved: 'http://127.0.0.1:1/runtime.tgz' }]) {
      const lock = structuredClone(base);
      lock.packages[`node_modules/${runtime}`] = runtimeEntry;
      const manifestText = JSON.stringify(manifest, null, 4) + '\n';
      const lockText = JSON.stringify(lock, null, 4) + '\n';
      writeFileSync(manifestPath, manifestText); writeFileSync(lockPath, lockText);
      installCi(project);
      assert.equal(JSON.parse(readFileSync(join(project, 'node_modules/fixture-build-tool/package.json'))).version, '1.0.0');
      assert.equal(existsSync(join(project, 'node_modules/@l2s1')), false);
      assert.equal(readFileSync(manifestPath, 'utf8'), manifestText);
      assert.equal(readFileSync(lockPath, 'utf8'), lockText);
    }
    // A real dependency failure must propagate and preserve both files byte-for-byte.
    const broken = JSON.parse(readFileSync(manifestPath));
    broken.devDependencies['missing-tool'] = 'file:/missing/l2s1-fixture.tgz';
    writeFileSync(manifestPath, JSON.stringify(broken));
    const before = [readFileSync(manifestPath, 'utf8'), readFileSync(lockPath, 'utf8')];
    assert.throws(() => installCi(project), /installation failed/);
    assert.deepEqual([readFileSync(manifestPath, 'utf8'), readFileSync(lockPath, 'utf8')], before);
  } finally { rmSync(root, { recursive: true, force: true }); }
});
