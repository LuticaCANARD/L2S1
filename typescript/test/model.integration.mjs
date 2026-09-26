import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { L2S1 } from '../dist/index.js';

test('real local GGUF model returns all warehouse decision kinds across repeated calls', {
  skip: !process.env.L2S1_MODEL,
  timeout: 180_000,
}, async () => {
  const request = JSON.parse(await readFile(new URL('../../examples/warehouse.json', import.meta.url), 'utf8'));
  const engine = await L2S1.load({
    binaryPath: process.env.L2S1_BINARY ?? 'l2s1', model: process.env.L2S1_MODEL,
    device: 'cpu', context: 2048, threads: 2,
  });
  try {
    assert.equal((await engine.capabilities()).evidence, 'model_scored');
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
