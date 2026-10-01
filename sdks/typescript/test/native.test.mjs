import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { L2S1 } from '../dist/index.js';

test('load rejects missing model, reserved flags, invalid timeout and pre-aborted startup', async () => {
  await assert.rejects(L2S1.load({ model: '' }), /model is required/);
  await assert.rejects(L2S1.load({ model: 'fixture', extraArgs: ['--listen=0.0.0.0:80'] }), /managed/);
  await assert.rejects(L2S1.load({ model: 'fixture', startupTimeoutMs: 0 }), RangeError);
  const abort = new AbortController(); abort.abort(new Error('cancelled'));
  await assert.rejects(L2S1.load({ model: 'fixture', signal: abort.signal }), /cancelled/);
});
test('missing Rust executable reports spawn failure and cleans up', async () => {
  await assert.rejects(L2S1.load({ model: 'fixture', binaryPath: join(tmpdir(), 'l2s1-nonexistent-executable') }), { code: 'spawn_failed' });
});
test('startup failure, deadline and caller abort reap owned processes', { skip: process.platform === 'win32' }, async () => {
  const dir = await mkdtemp(join(tmpdir(), 'l2s1 process '));
  const binary = join(dir, 'test executable');
  const pidPath = join(dir, 'pid');
  try {
    await writeFile(binary, `#!/usr/bin/env node\nimport { writeFileSync } from 'node:fs';\nwriteFileSync(${JSON.stringify(pidPath)}, String(process.pid));\nif (process.argv.includes('failure')) { console.error('fixture model load failed'); process.exit(7); }\nsetInterval(() => {}, 1000);\n`, { mode: 0o755 });
    await assert.rejects(L2S1.load({ model: 'failure', binaryPath: binary }), (error) => ['startup_failed', 'process_closed'].includes(error.code));
    await assert.rejects(L2S1.load({ model: 'hang', binaryPath: binary, startupTimeoutMs: 100 }), { code: 'startup_timeout' });
    const { readFile } = await import('node:fs/promises');
    const pid = Number(await readFile(pidPath, 'utf8'));
    assert.throws(() => process.kill(pid, 0), { code: 'ESRCH' });
    const abort = new AbortController();
    const pending = L2S1.load({ model: 'hang', binaryPath: binary, signal: abort.signal });
    setTimeout(() => abort.abort(new Error('cancel startup')), 100);
    await assert.rejects(pending, /cancel startup/);
    const cancelledPid = Number(await readFile(pidPath, 'utf8'));
    assert.throws(() => process.kill(cancelledPid, 0), { code: 'ESRCH' });
  } finally { await rm(dir, { recursive: true, force: true }); }
});

// Probe the subprocess boundary: omission inherits the runtime's model-aware
// choice, while an explicit false must survive serialization as a fresh override.
test('resident reuse defaults and explicit overrides reach the native process', { skip: process.platform === 'win32' }, async () => {
  const dir = await mkdtemp(join(tmpdir(), 'l2s1 default reuse '));
  const binary = join(dir, 'probe');
  const record = join(dir, 'args.json');
  try {
    await writeFile(binary, `#!/usr/bin/env node\nimport { writeFileSync } from 'node:fs';\nwriteFileSync(${JSON.stringify(record)}, JSON.stringify(process.argv.slice(2)));\nprocess.exit(1);\n`, { mode: 0o755 });
    for (const [options, expected, required] of [
      [{}, undefined, false],
      [{ fixedSchema: false }, 'fresh', false],
      [{ fixedSchema: true }, 'prefix-reuse', true],
      [{ fixedSchema: false, executionMode: 'parallel' }, 'parallel', false],
    ]) {
      await assert.rejects(L2S1.load({ model: 'fixture', binaryPath: binary, promptProfile: 'winnow', ...options }));
      const args = JSON.parse(await readFile(record, 'utf8'));
      assert.equal(args[args.indexOf('--prompt-profile') + 1], 'winnow');
      const index = args.indexOf('--execution-mode');
      assert.equal(index < 0 ? undefined : args[index + 1], expected);
      assert.equal(args.includes('--fixed-schema'), required);
      assert.equal(args.filter(arg => arg === '--execution-mode').length, expected ? 1 : 0);
    }
  } finally { await rm(dir, { recursive: true, force: true }); }
});
