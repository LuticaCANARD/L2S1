import assert from 'node:assert/strict';
import { test } from 'node:test';
import { L2S1 } from '../dist/index.js';

const decisions = () => [{ id: 'cold', instruction: 'Is temperature_c below 10?', kind: {
  type: 'binary', false_label: 'At least 10.', true_label: 'Below 10.',
} }];

test('prepared calls reuse fixed definitions with independent state and snapshot caller mutations', async () => {
  const seen = [];
  let batches = 0;
  const engine = L2S1.fromBackend({ async decide(request) { seen.push(structuredClone(request)); return request.state; }, async decideBatch(requests) { batches++; seen.push(...structuredClone(requests)); return requests.map((request) => request.state); }, async capabilities() {} });
  const input = decisions();
  const plan = engine.prepare(input);
  input[0].instruction = 'changed';
  assert.deepEqual(await plan.decide({ temperature_c: 6 }), { temperature_c: 6 });
  assert.deepEqual(await plan.decideBatch([{ temperature_c: 15 }, { temperature_c: 2 }]), [{ temperature_c: 15 }, { temperature_c: 2 }]);
  assert.ok(seen.every((request) => request.decisions[0].instruction === 'Is temperature_c below 10?'));
  assert.equal(seen.length, 3);
  assert.equal(batches, 1);
  assert.deepEqual(await plan.decideBatch([]), []);
  await engine.close();
  assert.throws(() => plan.decide({ temperature_c: 1 }), { code: 'backend_closed' });
});

test('native batch snapshots inputs and never falls back after failure or absent support', async () => {
  let calls = 0;
  const engine = L2S1.fromBackend({
    async decide() { assert.fail('must not call individual decide'); },
    async decideBatch(requests) {
      calls++;
      await new Promise((resolve) => setTimeout(resolve, 1));
      assert.deepEqual(requests.map((request) => request.state), [1, 2, 3]);
      throw new Error('native inference failed');
    }, async capabilities() {},
  });
  const input = [1, 2, 3].map((state) => ({ state, decisions: decisions() }));
  const pending = engine.decideBatch(input);
  input[1].state = 99;
  await assert.rejects(pending, /native inference failed/);
  assert.equal(calls, 1);
  await engine.close();
  const unsupported = L2S1.fromBackend({ async decide() { assert.fail('no serial fallback'); }, async capabilities() {} });
  await assert.rejects(unsupported.decideBatch(input), { code: 'batch_unsupported' });
  await unsupported.close();
});
