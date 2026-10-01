import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

export const rulesStudy = 'decision-rules-windows-20260926';
const root = new URL('../../', import.meta.url);
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const close = (actual, expected, label) => {
  assert.ok(Number.isFinite(actual) && Number.isFinite(expected), `${label}: finite numbers required`);
  assert.ok(Math.abs(actual - expected) <= 1e-10 * Math.max(1, Math.abs(expected)), `${label}: ${actual} != ${expected}`);
};

// Audit aggregates supplied by the user. This does not rerun inference or verify unavailable logs.
export async function verifyDecisionRules(summaryBytes, fixtureBytes, provenance) {
  const summary = JSON.parse(summaryBytes.toString());
  const fixture = JSON.parse(fixtureBytes.toString());
  assert.equal(summary.schema_version, 1);
  assert.equal(summary.suite, 'decision-rules-v1');
  assert.equal(fixture.id, summary.suite);
  assert.equal(summary.settings.iterations, 3);
  assert.equal(fixture.cases.length, 12);
  assert.ok(fixture.cases.every((item) => item.request.decisions.length === 3));
  for (const key of ['suite_sha256', 'benchmark_executable_sha256']) assert.match(summary[key], /^[a-f0-9]{64}$/);
  for (const key of ['repository_revision', 'llama_cpp_revision']) assert.match(summary[key], /^[a-f0-9]{40}$/);
  const lf = fixtureBytes.toString().replace(/\r\n/g, '\n');
  const fixtureLfSha256 = hash(lf);
  const fixtureCrlfSha256 = hash(lf.replace(/\n/g, '\r\n'));
  assert.ok([fixtureLfSha256, fixtureCrlfSha256].includes(summary.suite_sha256), 'Fixture differs beyond line endings');
  assert.equal(summary.runs.length, 10);
  const seen = new Set();
  const models = new Map();
  const audited = [];
  for (const run of summary.runs) {
    const key = `${run.model}.${run.device}`;
    assert.ok(!seen.has(key), `Duplicate run ${key}`);
    seen.add(key);
    assert.ok(['cpu', 'cuda'].includes(run.device));
    assert.match(run.sha256, /^[a-f0-9]{64}$/);
    assert.equal(run.log, `${key}.log`);
    const checkpoint = `${run.model_file}:${run.sha256}`;
    if (models.has(run.model)) assert.equal(models.get(run.model), checkpoint);
    models.set(run.model, checkpoint);
    if (run.status !== 'ok') {
      assert.equal(run.status, 'timeout');
      assert.equal(key, 'gemma4.cpu');
      assert.equal(run.error, `timeout: exceeded ${summary.settings.timeout} seconds`);
      assert.equal(run.quality, undefined, 'Timeout must not contain completed quality metrics');
      assert.equal(run.latency_ms, undefined, 'Timeout must not contain completed timings');
      audited.push({ model: run.model, device: run.device, status: run.status });
      continue;
    }
    assert.equal(run.report, `${key}.json`);
    const q = run.quality;
    const c = q.counts;
    for (const count of Object.values(c)) assert.ok(Number.isSafeInteger(count) && count >= 0);
    assert.equal(c.decisions, 108);
    assert.equal(c.accepted + c.abstained, c.decisions);
    assert.equal(c.accepted_correct + c.accepted_wrong, c.accepted);
    assert.ok(c.top1_correct >= c.accepted_correct && c.top1_correct <= c.decisions);
    close(q.coverage, c.accepted / c.decisions, `${key} coverage`);
    close(q.abstention_rate, c.abstained / c.decisions, `${key} abstentions`);
    close(q.correct_fraction, c.accepted_correct / c.decisions, `${key} correct/all`);
    close(q.top1_accuracy_before_abstention, c.top1_correct / c.decisions, `${key} raw top-1`);
    if (c.accepted === 0) assert.equal(q.accepted_accuracy, null);
    else close(q.accepted_accuracy, c.accepted_correct / c.accepted, `${key} accepted accuracy`);
    const samples = run.latency_ms.samples;
    assert.equal(samples.length, 36);
    assert.ok(samples.every((value) => Number.isFinite(value) && value > 0));
    const sorted = [...samples].sort((a, b) => a - b);
    const elapsedMs = samples.reduce((sum, value) => sum + value, 0);
    for (const [metric, expected] of Object.entries({ min: sorted[0], max: sorted.at(-1), mean: elapsedMs / samples.length, p50: sorted[Math.ceil(samples.length * 0.5) - 1], p95: sorted[Math.ceil(samples.length * 0.95) - 1] })) {
      close(run.latency_ms[metric], expected, `${key} ${metric}`);
    }
    close(run.decisions_per_second, c.decisions * 1000 / elapsedMs, `${key} decisions/s`);
    close(run.accepted_correct_decisions_per_second, c.accepted_correct * 1000 / elapsedMs, `${key} correct accepted/s`);
    assert.equal(run.repeat_consistency.comparisons, 72);
    assert.equal(run.repeat_consistency.changed_outputs, 0);
    // Dividing counts is justified here by the reported unchanged outputs, not additional independent samples.
    for (const count of Object.values(c)) assert.equal(count % summary.settings.iterations, 0);
    audited.push({ model: run.model, device: run.device, status: run.status, decisionsPerPass: 36,
      acceptedCorrectPerPass: c.accepted_correct / 3, acceptedWrongPerPass: c.accepted_wrong / 3,
      abstainedPerPass: c.abstained / 3, rawCorrectPerPass: c.top1_correct / 3 });
  }
  assert.equal(models.size, 5);
  for (const model of models.keys()) for (const device of ['cpu', 'cuda']) assert.ok(seen.has(`${model}.${device}`));
  return { schema_version: 1, scope: 'Aggregate arithmetic and fixture identity only; no inference rerun or per-case audit',
    summary_sha256: hash(summaryBytes), provenance_canonical_json_sha256: hash(JSON.stringify(provenance)),
    fixture_lf_sha256: fixtureLfSha256, fixture_crlf_sha256: fixtureCrlfSha256,
    reported_fixture_line_endings: summary.suite_sha256 === fixtureCrlfSha256 ? 'CRLF' : 'LF',
    completed_runs: audited.filter((run) => run.status === 'ok').length,
    timeout_runs: audited.filter((run) => run.status === 'timeout').length,
    timing_samples_checked: 324, independent_decisions_per_run: 36,
    gpu: provenance.gpu, gpu_evidence: provenance.gpu_evidence, runs: audited };
}

export async function loadDecisionRules() {
  const [bytes, fixture, provenance] = await Promise.all([
    readFile(new URL(`benchmarks/${rulesStudy}/summary.json`, root)),
    readFile(new URL('tests/fixtures/decision_benchmark.json', root)),
    readFile(new URL(`benchmarks/${rulesStudy}/provenance.json`, root), 'utf8').then(JSON.parse),
  ]);
  return { bytes, summary: JSON.parse(bytes.toString()), provenance, audit: await verifyDecisionRules(bytes, fixture, provenance) };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const { audit } = await loadDecisionRules();
  process.stdout.write(JSON.stringify(audit, null, 2) + '\n');
}
