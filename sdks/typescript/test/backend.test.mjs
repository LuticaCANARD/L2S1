import assert from 'node:assert/strict';
import { test } from 'node:test';
import { L2S1 } from '../dist/index.js';

test('custom backends share the facade and are closed exactly once', async () => {
  const request = { state: null, decisions: [] };
  const options = { timeoutMs: 100 };
  let closes = 0;
  const backend = {
    async decide(received, callOptions) { assert.equal(received, request); assert.equal(callOptions, options); return 'custom result'; },
    async capabilities() { return 'custom capabilities'; },
    async close() { closes++; },
  };
  const engine = L2S1.fromBackend(backend);
  assert.equal(await engine.decide(request, options), 'custom result');
  assert.equal(await engine.capabilities(), 'custom capabilities');
  await Promise.all([engine.close(), engine.close(), engine[Symbol.asyncDispose]()]);
  assert.equal(closes, 1);
  assert.throws(() => engine.decide(request), { code: 'backend_closed' });
});
test('HTTP connect shares decide without spawning a process or closing a shared server', async () => {
  const engine = L2S1.connect({ baseUrl: 'https://fixture.invalid', fetch: async () => Response.json({ status: 'ok' }) });
  // A mocked fetch returns the wrong envelope; the same transport check still runs.
  await assert.rejects(engine.decide({ state: null, decisions: [] }), { code: 'invalid_response' });
  await engine.close();
  assert.throws(() => engine.capabilities(), { code: 'backend_closed' });
});
test('a throwing custom close hook remains idempotent', async () => {
  let closes = 0;
  const engine = L2S1.fromBackend({ decide: async () => {}, capabilities: async () => {}, close() { closes++; throw new Error('close failed'); } });
  await assert.rejects(engine.close(), /close failed/);
  await assert.rejects(engine.close(), /close failed/);
  assert.equal(closes, 1);
});
