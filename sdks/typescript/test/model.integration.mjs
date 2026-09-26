import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { L2S1 } from '../dist/index.js';

test('real local GGUF model returns all warehouse decision kinds across repeated calls', {
  skip: !process.env.L2S1_MODEL,
  timeout: 180_000,
}, async () => {
  const request = JSON.parse(await readFile(new URL('../../../examples/warehouse.json', import.meta.url), 'utf8'));
  const engine = await L2S1.load({
    binaryPath: process.env.L2S1_BINARY ?? 'l2s1', model: process.env.L2S1_MODEL,
    device: 'cpu', context: 2048, threads: 2,
  });
  try {
    assert.equal((await engine.capabilities()).evidence, 'model_scored');
    await assert.rejects(engine.decideBatch([request]), { code: 'batch_not_enabled' });
    for (let iteration = 0; iteration < 2; iteration++) {
      const response = await engine.decide(request);
      assert.equal(response.results.length, request.decisions.length);
      for (let i = 0; i < response.results.length; i++) {
        const result = response.results[i];
        assert.equal(result.id, request.decisions[i].id);
        assert.equal(result.value.type, request.decisions[i].kind.type);
        assert.equal(result.evidence.type, 'model_scored');
        assert.ok(result.evidence.candidate_mass >= 0 && result.evidence.candidate_mass <= 1);
        assert.ok(Math.abs(result.evidence.scores.reduce((sum, score) => sum + score.option_probability, 0) - 1) < 1e-8);
        assert.ok(result.usage.input_tokens > 0);
      }
    }
  } finally { await engine.close(); }
});

test('compiled stdio sends independent states to native parallel waves with shared prefix usage', {
  skip: !process.env.L2S1_MODEL,
  timeout: 180_000,
}, async () => {
  const request = JSON.parse(await readFile(new URL('../../../examples/warehouse.json', import.meta.url), 'utf8'));
  const fetch = globalThis.fetch;
  globalThis.fetch = () => { throw new Error('native stdio must not call HTTP'); };
  let engine;
  try {
    engine = await L2S1.load({
      binaryPath: process.env.L2S1_BINARY ?? 'l2s1', model: process.env.L2S1_MODEL,
      executionMode: 'parallel', parallelWidth: 2, context: 2048, batch: 32, threads: 2,
    });
    const capabilities = await engine.capabilities();
    assert.equal(capabilities.batch.enabled, true);
    assert.equal(capabilities.batch.execution, 'native_parallel');
    const plan = engine.prepare(request.decisions);
    const states = [request.state, { temperature_c: 20 }, { temperature_c: 2 }];
    const responses = await plan.decideBatch(states);
    assert.equal(responses.length, states.length);
    assert.equal(new Set(responses.map((response) => response.request_id)).size, states.length);
    let reused = 0;
    for (let index = 0; index < responses.length; index++) {
      const response = responses[index];
      assert.ok(response.request_id.endsWith(`/${index}`));
      assert.equal(response.backend.details.execution_mode, 'parallel');
      assert.deepEqual(response.results.map((result) => result.id), request.decisions.map((decision) => decision.id));
      for (const result of response.results) {
        assert.equal(result.evidence.type, 'model_scored');
        assert.ok(Math.abs(result.evidence.scores.reduce((sum, score) => sum + score.option_probability, 0) - 1) < 1e-8);
        reused += result.usage.reused_prefix_tokens;
      }
    }
    assert.ok(reused > 0, 'native wave should report actual shared prefix tokens');
    console.log(JSON.stringify({ transport: 'stdio', execution: 'native_parallel', requests: responses.length, reused_prefix_tokens: reused }));
    await engine.decide(request);
  } finally { if (engine) await engine.close(); globalThis.fetch = fetch; }
});
