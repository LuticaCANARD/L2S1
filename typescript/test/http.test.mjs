import assert from 'node:assert/strict';
import { test } from 'node:test';
import { L2S1Client, L2S1Error } from '../dist/http.js';

const request = { state: { x: 1 }, decisions: [{ id: 'positive', instruction: 'Is x positive?', kind: { type: 'binary', false_label: 'No', true_label: 'Yes' } }] };
const output = () => ({ api_version: 1, request_id: 'req-1', backend: { runtime: 'fixture', model: 'fixture', details: null }, policy: null,
  results: [{ id: 'positive', value: { type: 'binary', value: true }, status: 'selected', abstention_reasons: [],
    evidence: { type: 'selection_only', selected_code: 'B', provider_model: null }, usage: { input_tokens: null, output_tokens: null } }] });
function client(fetch, options = {}) { return new L2S1Client({ baseUrl: 'http://localhost:8080/proxy/', fetch, ...options }); }

test('HTTP subpath imports without Node runtime dependencies', async () => {
  const { readFile } = await import('node:fs/promises');
  const code = await readFile(new URL('../dist/http.js', import.meta.url), 'utf8');
  assert.doesNotMatch(code, /node:|\.\/native/);
});
test('sends the exact Rust wire payload and caller headers through a proxy path', async () => {
  const payload = { ...request, media: [{ id: 'photo', type: 'image', data_base64: 'aGVsbG8=' }],
    policy: { min_top_probability: 0.9, min_candidate_mass: 0.1 }, reasoning: { mode: 'direct' },
    target_error_rate: 0.1, failure_reasons: { low_candidate_mass: '검토 필요' } };
  const response = await client(async (url, init) => {
    assert.equal(url, 'http://localhost:8080/proxy/v1/decisions');
    assert.equal(init.method, 'POST');
    assert.equal(init.headers.get('content-type'), 'application/json');
    assert.equal(init.headers.get('authorization'), 'Bearer fixture');
    assert.equal(init.redirect, 'error');
    assert.deepEqual(JSON.parse(init.body), payload);
    return Response.json(output());
  }, { headers: { authorization: 'Bearer fixture' } }).decide(payload);
  assert.equal(response.results[0].value.value, true);
  assert.equal(response.results[0].evidence.type, 'selection_only');
});
test('health and capabilities use GET', async () => {
  const api = client(async (url, init) => {
    assert.equal(init.method, 'GET');
    assert.equal(init.body, undefined);
    return Response.json(url.endsWith('/healthz') ? { status: 'ok' } : {
      api_version: 1, backend: { runtime: 'fixture', model: 'fixture' }, decision_types: ['binary'], evidence: 'selection_only', media: {}, limits: {},
    });
  });
  assert.deepEqual(await api.health(), { status: 'ok' });
  assert.equal((await api.capabilities()).api_version, 1);
});
test('Rust errors retain HTTP status, code, request ID and user explanation', async () => {
  const api = client(async () => Response.json({ error: { code: 'reasoning_limit', message: 'limit exceeded', request_id: 'req-2', user_reason: '검토 필요' } }, { status: 422 }));
  await assert.rejects(api.decide(request), (error) => {
    assert.ok(error instanceof L2S1Error);
    assert.equal(error.status, 422); assert.equal(error.code, 'reasoning_limit');
    assert.equal(error.requestId, 'req-2'); assert.equal(error.userReason, '검토 필요');
    return true;
  });
});
test('never retries inference on 503', async () => {
  let calls = 0;
  await assert.rejects(client(async () => { calls++; return Response.json({}, { status: 503 }); }).decide(request), { code: 'http_error' });
  assert.equal(calls, 1);
});
test('invalid JSON and mismatched response envelopes fail explicitly', async () => {
  await assert.rejects(client(async () => new Response('<html>')).decide(request), { code: 'invalid_response' });
  for (const mutate of [
    (v) => { v.api_version = 2; }, (v) => { v.results = []; },
    (v) => { v.results[0].id = 'other'; }, (v) => { v.results[0].value.type = 'choice'; },
    (v) => { v.results[0].evidence.type = 'generated_probability'; },
  ]) {
    const value = output(); mutate(value);
    await assert.rejects(client(async () => Response.json(value)).decide(request), { code: 'invalid_response' });
  }
});
test('request aborts and deadlines reach the transport', async () => {
  const fetch = async (_url, { signal }) => {
    signal.throwIfAborted();
    return new Promise((_resolve, reject) => {
      // Keep the test event loop alive; AbortSignal.timeout itself is unref'ed.
      const timer = setTimeout(() => reject(new Error('signal did not abort')), 1000);
      signal.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason); }, { once: true });
    });
  };
  await assert.rejects(client(fetch).decide(request, { timeoutMs: 10 }), { name: 'TimeoutError' });
  const abort = new AbortController(); abort.abort(new Error('caller cancelled'));
  await assert.rejects(client(fetch).decide(request, { signal: abort.signal }), /caller cancelled/);
});
test('rejects invalid timeouts and non-HTTP base URLs', () => {
  for (const timeoutMs of [0, -1, 0.5, Infinity, 2 ** 31]) assert.throws(() => client(globalThis.fetch, { timeoutMs }), RangeError);
  for (const baseUrl of ['file:///tmp/socket', 'http://user:pass@localhost', 'http://localhost?x=1', 'http://localhost/#hash']) {
    assert.throws(() => client(globalThis.fetch, { baseUrl }), TypeError);
  }
});
test('closing the HTTP client cancels local calls without a server shutdown request', async () => {
  let calls = 0;
  const api = client(async (_url, { signal }) => {
    calls++;
    return new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }));
  });
  const pending = api.decide(request);
  api.close();
  await assert.rejects(pending, { code: 'backend_closed' });
  await assert.rejects(api.health(), { code: 'backend_closed' });
  assert.equal(calls, 1);
});
