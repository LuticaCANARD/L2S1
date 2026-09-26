import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { execFileSync } from 'node:child_process';

const [wrapper, runtime] = process.argv.slice(2).map((path) => resolve(path));
if (!wrapper || !runtime) throw new Error('Usage: node test-installed-runtime.mjs wrapper.tgz runtime.tgz');
const directory = await mkdtemp(join(tmpdir(), 'l2s1-installed-'));
try {
  await writeFile(join(directory, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  // npm_execpath works across Windows and Unix without invoking a shell.
  const npm = process.env.npm_execpath;
  if (!npm) throw new Error('Run through npm: npm run test:package -- wrapper.tgz runtime.tgz');
  execFileSync(process.execPath, [npm, 'install', '--offline', '--ignore-scripts', '--no-audit', '--no-fund', '--cache', join(directory, '.npm-cache'), wrapper, runtime], { cwd: directory, stdio: 'inherit' });
  const code = `
    import assert from 'node:assert/strict';
    import { L2S1 } from '@l2s1/node';
    import { L2S1Client } from '@l2s1/node/http';
    assert.equal(typeof L2S1Client, 'function');
    if (process.env.L2S1_MODEL) {
      const engine = await L2S1.load({ model: process.env.L2S1_MODEL, device: 'cpu', threads: 2 });
      try {
        assert.equal((await engine.capabilities()).evidence, 'model_scored');
        const response = await engine.decide({ state: { x: 1 }, decisions: [{ id: 'positive', instruction: 'Is x positive?', kind: { type: 'binary', false_label: 'x <= 0', true_label: 'x > 0' } }] });
        assert.equal(response.results[0].evidence.type, 'model_scored');
        console.log('Installed bundle real-model result:', JSON.stringify(response.results[0].value));
      } finally { await engine.close(); }
    } else {
      await assert.rejects(L2S1.load({ model: './does-not-exist.gguf' }), (error) => {
        assert.equal(error.code, 'startup_failed');
        assert.match(error.message, /model load failed|failed to load model/i);
        return true;
      });
      console.log('Installed bundle resolved and executed; invalid-model failure confirmed');
    }
  `;
  execFileSync(process.execPath, ['--input-type=module', '--eval', code], { cwd: directory, stdio: 'inherit', env: process.env });
} finally { await rm(directory, { recursive: true, force: true }); }
