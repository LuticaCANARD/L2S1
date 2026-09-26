import assert from 'node:assert/strict';
import { test } from 'node:test';
import { resolve } from 'node:path';
import { L2S1 } from '../dist/index.js';

const binaryPath = resolve('../target/debug/examples/typescript_fixture' + (process.platform === 'win32' ? '.exe' : ''));
const request = { state: { x: 1 }, decisions: [
  { id: 'binary', instruction: 'Is x positive?', kind: { type: 'binary', false_label: 'No', true_label: 'Yes' } },
  { id: 'choice', instruction: 'Select a class.', kind: { type: 'choice', options: [{ id: 'a', criterion: 'A' }, { id: 'b', criterion: 'B' }] } },
  { id: 'ordinal', instruction: 'Rate x.', kind: { type: 'ordinal', levels: [{ id: 'low', criterion: 'Low', value: 10 }, { id: 'high', criterion: 'High', value: 30 }] } },
] };

test('Node managed lifecycle crosses the real Rust HTTP/scoring boundary', async () => {
  let logs = '';
  const engine = await L2S1.load({ model: 'fixture path with spaces', binaryPath, onStderr: (chunk) => { logs += chunk; } });
  try {
    assert.match(logs, /l2s1 HTTP listening on 127\.0\.0\.1:\d+/);
    const capabilities = await engine.capabilities();
    assert.equal(capabilities.backend.runtime, 'rust-fixture');
    const address = /l2s1 HTTP listening on (127\.0\.0\.1:\d+)/.exec(logs)[1];
    const remote = L2S1.connect({ baseUrl: `http://${address}` });
    try { assert.equal((await remote.decide(request)).results[0].value.value, true); }
    finally { await remote.close(); }
    // Remote close leaves the shared Rust server available to its owner.
    const response = await engine.decide(request);
    assert.deepEqual(response.results.map((result) => result.value), [
      { type: 'binary', value: true }, { type: 'choice', selected: 'b' }, { type: 'ordinal', selected: 'high' },
    ]);
    for (const result of response.results) {
      assert.equal(result.evidence.type, 'model_scored');
      assert.equal(result.usage.input_tokens, 17);
      assert.ok(Math.abs(result.evidence.candidate_mass - (1 + Math.exp(4)) / (2 + Math.exp(4))) < 1e-12);
      assert.ok(Math.abs(result.evidence.top_option_probability - Math.exp(4) / (1 + Math.exp(4))) < 1e-12);
    }
    assert.ok(Math.abs(response.results[2].evidence.estimate.expected_value - (10 + 30 * Math.exp(4)) / (1 + Math.exp(4))) < 1e-12);
    if (capabilities.request_policy?.supported) {
      const abstained = await engine.decide({ ...request, policy: { min_top_probability: 1, min_candidate_mass: 1 }, failure_reasons: { low_top_probability: '검토 필요' } });
      assert.ok(abstained.results.every((result) => result.status === 'abstained'));
      assert.equal(abstained.results[0].value.value, null);
      assert.equal(abstained.results[0].reason_messages[0].message, '검토 필요');
      const budget = await engine.decide({ ...request, target_error_rate: 0.01 });
      assert.equal(budget.error_budget.guaranteed, false);
      assert.equal(budget.results[0].status, 'abstained');
    } else {
      // The main-branch v1 server rejects optional extensions it cannot honor.
      await assert.rejects(engine.decide({ ...request, policy: { min_top_probability: 1, min_candidate_mass: 1 } }), { status: 400, code: 'invalid_request' });
    }
    await assert.rejects(engine.decide({ ...request, decisions: [request.decisions[0], request.decisions[0]] }), { status: 400, code: 'invalid_request' });
    await assert.rejects(engine.decide({ ...request, reasoning: { mode: 'thinking' } }), { status: 400, code: 'invalid_request' });
    // Invalid input must leave the resident process usable.
    const responses = await Promise.all([engine.decide(request), engine.decide(request)]);
    assert.notEqual(responses[0].request_id, responses[1].request_id);
  } finally { await Promise.all([engine.close(), engine.close()]); }
  assert.throws(() => engine.decide(request), { code: 'backend_closed' });
  await engine[Symbol.asyncDispose]();
});
