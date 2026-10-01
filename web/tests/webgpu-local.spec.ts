import { expect, test } from '@playwright/test';
import { messages } from '../src/lib/i18n/webgpu';
import warehouse from '../src/lib/demo/text-request.json' with { type: 'json' };

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => { localStorage.setItem('l2s1-locale', 'ko'); Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }); });
});

test('local inference works without WebGPU and preserves input and acceptance policy', async ({ page }) => {
  const calls: string[] = [];
  await page.route('**/quality-inference/**', async route => {
    calls.push(route.request().url());
    if (route.request().url().endsWith('/v1/capabilities')) await route.fulfill({ json: { api_version: 1, backend: { model: '/models/large.gguf', runtime: 'local-libllama' }, evidence: 'model_scored', request_policy: { supported: true }, reasoning: { modes: ['direct'] } } });
    else {
      const request = route.request().postDataJSON();
      expect(request.state).toEqual(warehouse.state);
      expect(request.decisions).toEqual(warehouse.decisions);
      expect(request.policy).toEqual({ min_top_probability: 0.8, min_candidate_mass: 0.05 });
      expect(Object.keys(request.failure_reasons).sort()).toEqual(['low_candidate_mass', 'low_top_probability', 'native_failure', 'reasoning_limit', 'tied_candidates']);
      expect(request.failure_reasons.low_candidate_mass).toBe(messages.ko.failureLowMass);
      await route.fulfill({ json: { backend: { model: '/models/large.gguf', runtime: 'local-libllama' }, policy: request.policy, results: request.decisions.map((decision: { id: string; kind: { type: string } }) => ({ id: decision.id, status: 'abstained', value: { type: decision.kind.type, selected: null }, abstention_reasons: ['low_top_probability'], evidence: { type: 'model_scored', candidate_mass: 0.99, top_option_probability: 0.7 } })) } });
    }
  });
  await page.goto('/webgpu');
  await page.getByRole('button', { name: messages.ko.warehouseExample }).click();
  expect(calls).toEqual([]);
  await page.locator('#execution').selectOption('local');
  expect(calls).toEqual([]);
  const guide = page.getByRole('link', { name: `${messages.ko.guide} ↗`, exact: true });
  await expect(guide).toHaveAttribute('href', '/docs/docs/ko/WEBGPU_DEMO.md');
  expect((await page.request.get(await guide.getAttribute('href') ?? '')).status()).toBe(200);
  await expect(page.locator('#state')).toHaveValue(JSON.stringify(warehouse.state, null, 2));
  await expect(page.getByRole('button', { name: messages.ko.localAnalyze })).toBeDisabled();
  await page.getByRole('button', { name: messages.ko.localConnect }).click();
  await expect(page.getByText('연결됨: large.gguf · local-libllama')).toBeVisible();
  await expect(page.locator('#reasoning option[value="thinking"]')).toBeDisabled();
  await page.getByRole('button', { name: messages.ko.localAnalyze }).click();
  await expect(page.getByText(messages.ko.actualLocal, { exact: true })).toBeVisible();
  await expect(page.locator('.result')).toHaveCount(3);
  await expect(page.locator('.result .value').first()).toContainText('판단 보류');
  await page.locator('#min-top').fill('0.9');
  await expect(page.locator('.result')).toHaveCount(0);
  await page.locator('#execution').selectOption('browser');
  await expect(page.getByRole('button', { name: messages.ko.analyze })).toBeDisabled();
  expect(calls).toHaveLength(2);
});

test('a missing local server keeps analysis blocked with a localized error', async ({ page }) => {
  await page.route('**/quality-inference/**', route => route.fulfill({ status: 503, json: { error: { code: 'model_unavailable', message: 'Start the local server.' } } }));
  await page.goto('/webgpu?lang=ja');
  await page.locator('#execution').selectOption('local');
  await page.getByRole('button', { name: messages.ja.localConnect }).click();
  await expect(page.getByRole('alert')).toContainText('ローカルモデルの接続または推論に失敗');
  await expect(page.getByRole('alert')).toContainText('model_unavailable');
  await expect(page.getByRole('button', { name: messages.ja.localAnalyze })).toBeDisabled();
});

test('results are rejected if the server lowers the requested acceptance policy', async ({ page }) => {
  await page.route('**/quality-inference/**', route => route.fulfill({ json: route.request().url().endsWith('/v1/capabilities')
    ? { api_version: 1, backend: { model: 'large.gguf', runtime: 'local-libllama' }, evidence: 'model_scored' }
    : { backend: { model: 'large.gguf', runtime: 'local-libllama' }, policy: { min_top_probability: 0, min_candidate_mass: 0 }, results: [] } }));
  await page.goto('/webgpu');
  await page.locator('#execution').selectOption('local');
  await page.getByRole('button', { name: messages.ko.localConnect }).click();
  await expect(page.getByRole('button', { name: messages.ko.localAnalyze })).toBeEnabled();
  await page.getByRole('button', { name: messages.ko.localAnalyze }).click();
  await expect(page.getByRole('alert')).toContainText('acceptance policy');
  await expect(page.locator('.result')).toHaveCount(0);
});


test('current native HTTP contract accepts only state and decisions and retains its policy', async ({ page }) => {
  await page.route('**/quality-inference/**', async route => {
    if (route.request().url().endsWith('/v1/capabilities')) {
      await route.fulfill({ json: { api_version: 1, backend: { model: 'current-main.gguf', runtime: 'local-libllama' }, evidence: 'model_scored' } });
    } else {
      const request = route.request().postDataJSON();
      expect(Object.keys(request).sort()).toEqual(['decisions', 'state']);
      await route.fulfill({ json: { backend: { model: 'current-main.gguf', runtime: 'local-libllama' }, policy: { min_top_probability: 0.8, min_candidate_mass: 0.05 }, results: request.decisions.map((decision: { id: string; kind: { type: string } }) => ({ id: decision.id, status: 'abstained', value: { type: decision.kind.type, selected: null }, abstention_reasons: ['low_top_probability'], evidence: { type: 'model_scored' } })) } });
    }
  });
  await page.goto('/webgpu');
  await page.locator('#execution').selectOption('local');
  await page.getByRole('button', { name: messages.ko.localConnect }).click();
  await page.getByRole('button', { name: messages.ko.localAnalyze }).click();
  await expect(page.getByText(messages.ko.actualLocal, { exact: true })).toBeVisible();
  await expect(page.locator('.result')).toContainText(messages.ko.failureLowTop);
});
