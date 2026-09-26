import assert from 'node:assert/strict';
import { test } from 'node:test';
import { resolveRuntime } from '../dist/runtime.js';

test('explicit Rust builds keep caller environment and accept GPU devices', () => {
  const runtime = resolveRuntime('/custom/path/l2s1', 'cuda');
  assert.equal(runtime.binaryPath, '/custom/path/l2s1');
  assert.ok(Object.entries(process.env).every(([key, value]) => runtime.env[key] === value));
  assert.notEqual(runtime.env, process.env);
  assert.throws(() => resolveRuntime('  '), TypeError);
});
test('automatic resolution selects a prebuilt package or reports an actionable missing-runtime error', () => {
  try {
    const runtime = resolveRuntime();
    assert.match(runtime.binaryPath, /runtime-[^/\\]+[/\\]bin[/\\]l2s1(?:\.exe)?$/);
  } catch (error) {
    assert.ok(['runtime_not_installed', 'unsupported_platform'].includes(error.code));
    assert.match(error.message, /binaryPath/);
  }
});
