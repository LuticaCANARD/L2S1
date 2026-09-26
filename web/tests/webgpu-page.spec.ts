import { expect, test } from '@playwright/test';
import { messages } from '../src/lib/i18n/webgpu';
import { validateBrowserRequest, type BrowserRequest } from '../src/lib/webgpu/contract';
import { scoreDecision } from '../src/lib/webgpu/scoring';

test.beforeEach(async ({ page }) => { await page.addInitScript(() => localStorage.setItem('l2s1-locale', 'ko')); });

test('unsupported WebGPU blocks loading without downloading model or runtime', async ({ page }) => {
  const remote: string[] = [];
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }));
  page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net/.test(request.url())) remote.push(request.url()); });
  await page.goto('/webgpu');
  await expect(page.getByText('WebGPU 실행 불가', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '모델 다운로드·로드' })).toBeDisabled();
  await expect(page.getByRole('button', { name: '이 브라우저에서 분석' })).toBeDisabled();
  await expect(page.locator('.support')).toContainText(messages.ko.errorNoWebgpu);
  await expect(page.locator('#analysis-blocked')).toHaveText(messages.ko.errorNoWebgpu);
  await expect(page.getByRole('button', { name: messages.ko.analyze })).toHaveAttribute('aria-describedby', 'analysis-blocked');
  expect(remote).toEqual([]);
});

test('capability probe does not load models and policy inputs stay synchronized', async ({ page }) => {
  const remote: string[] = [];
  const errors: string[] = [];
  page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net/.test(request.url())) remote.push(request.url()); });
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/webgpu');
  await expect(page.getByRole('heading', { name: '01 · 모델 준비' })).toBeVisible();
  await page.locator('#error-rate').fill('15');
  await expect(page.locator('#min-top')).toHaveValue('0.85');
  await page.locator('#min-top').fill('0.9');
  await expect(page.locator('#error-rate')).toHaveValue('10');
  await page.getByRole('button', { name: '물류 세 질문' }).click();
  expect(JSON.parse(await page.locator('#state').inputValue()).storage_requirement).toBe('chilled');
  await page.locator('#reasoning').selectOption('thinking');
  await expect(page.locator('#max-thought')).toHaveValue('128');
  await expect(page.locator('.output')).toContainText(messages.ko.loadBeforeAnalysis);
  await expect(page.locator('#analysis-blocked')).toBeVisible();
  expect(remote).toEqual([]);
  expect(errors).toEqual([]);
});

for (const viewport of [{ name: 'desktop', width: 1440, height: 1000 }, { name: 'mobile', width: 375, height: 900 }]) {
  test(`WebGPU ${viewport.name} page fits without inference`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.goto('/webgpu');
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await page.screenshot({ path: testInfo.outputPath(`webgpu-${viewport.name}.png`), fullPage: true });
  });
}


for (const language of ['en', 'ko', 'ja'] as const) {
  test(`WebGPU ${language} localizes unsupported state without starting inference`, async ({ page }) => {
    const remote: string[] = [];
    await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }));
    page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net/.test(request.url())) remote.push(request.url()); });
    await page.goto(`/webgpu?lang=${language}`);
    await expect(page.getByRole('heading', { name: messages[language].prepare })).toBeVisible();
    await expect(page.locator('.support')).toContainText(messages[language].errorNoWebgpu);
    await expect(page.locator('#analysis-blocked')).toHaveText(messages[language].errorNoWebgpu);
    await expect(page.getByRole('button', { name: messages[language].load })).toBeDisabled();
    await expect(page).toHaveTitle(messages[language].title);
    await expect(page.locator('.shell')).toHaveAttribute('lang', language);
    await page.locator('summary').filter({ hasText: messages[language].customFailures }).click();
    await expect.poll(async () => JSON.parse(await page.locator('#failure').inputValue()).reasoning_limit).toBe(messages[language].failureLimit);
    expect(remote).toEqual([]);
  });
}

test('language switches translate current status and preserve editable inputs', async ({ page }) => {
  const remote: string[] = [];
  await page.addInitScript(() => Object.defineProperty(navigator, 'gpu', { value: undefined, configurable: true }));
  page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net/.test(request.url())) remote.push(request.url()); });
  await page.goto('/webgpu?lang=ko');
  await expect(page.locator('.support')).toContainText(messages.ko.errorNoWebgpu);
  await page.locator('#state').fill('{"text":"my unchanged state"}');
  await page.locator('summary').filter({ hasText: messages.ko.editQuestions }).click();
  const questions = await page.locator('#questions').inputValue();
  await page.locator('summary').filter({ hasText: messages.ko.customFailures }).click();
  await expect.poll(async () => JSON.parse(await page.locator('#failure').inputValue()).reasoning_limit).toBe(messages.ko.failureLimit);
  await page.locator('#site-language').selectOption('ja');
  await expect.poll(async () => JSON.parse(await page.locator('#failure').inputValue()).reasoning_limit).toBe(messages.ja.failureLimit);
  await page.locator('#site-language').selectOption('ko');
  await page.locator('#failure').fill('{"reasoning_limit":"Keep my own message 그대로"}');
  for (const language of ['ja', 'en', 'ko'] as const) {
    await page.locator('#site-language').selectOption(language);
    await expect(page.locator('.support')).toContainText(messages[language].errorNoWebgpu);
    await expect(page.locator('#analysis-blocked')).toHaveText(messages[language].errorNoWebgpu);
    await expect(page.locator('#state')).toHaveValue('{"text":"my unchanged state"}');
    await expect(page.locator('#questions')).toHaveValue(questions);
    await expect(page.locator('#failure')).toHaveValue('{"reasoning_limit":"Keep my own message 그대로"}');
  }
  expect(remote).toEqual([]);
});

test('localized validation changes messages while model evidence and custom messages remain identical', () => {
  const request: BrowserRequest = { state: { text: 'The animal is a cat.' }, decisions: [{ id: 'cat', instruction: 'Does it name a cat?', kind: { type: 'binary', false_label: 'No cat', true_label: 'Names a cat' } }], reasoning: { mode: 'direct', max_tokens: 128 }, policy: { min_top_probability: 0.9, min_candidate_mass: 0.8 }, failure_reasons: { low_top_probability: 'Preserve this message 그대로' } };
  const baseline = scoreDecision(request.decisions[0], [0, 1], [0, 1], request.policy, request.failure_reasons);
  for (const language of ['en', 'ko', 'ja'] as const) {
    expect(() => validateBrowserRequest({ ...request, decisions: [] }, language)).toThrow(messages[language].validateQuestions);
    expect(() => validateBrowserRequest({ ...request, reasoning: { mode: 'thinking', max_tokens: 0 } }, language)).toThrow(messages[language].validateReasoning);
    expect(scoreDecision(request.decisions[0], [0, 1], [0, 1], request.policy, request.failure_reasons, language)).toEqual(baseline);
    expect(() => scoreDecision(request.decisions[0], [NaN, 1], [0, 1], request.policy, request.failure_reasons, language)).toThrow(messages[language].errorFiniteLogits);
  }
});


test('insecure context explains the blocked analyze button before any adapter or worker starts', async ({ page }) => {
  const remote: string[] = []; let workers = 0;
  await page.addInitScript(() => {
    Object.defineProperty(window, 'isSecureContext', { value: false, configurable: true });
    Object.defineProperty(navigator, 'gpu', { value: { requestAdapter: () => { throw new Error('Adapter must not be queried outside a secure context'); } }, configurable: true });
  });
  page.on('worker', () => workers++);
  page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net/.test(request.url())) remote.push(request.url()); });
  await page.goto('/webgpu?lang=en');
  await expect(page.locator('#analysis-blocked')).toHaveText(messages.en.errorInsecureContext);
  await expect(page.getByRole('button', { name: messages.en.analyze })).toBeDisabled();
  await expect(page.getByRole('button', { name: messages.en.load })).toBeDisabled();
  expect(remote).toEqual([]); expect(workers).toBe(0);
});

for (const adapterMode of ['absent', 'rejected', 'available'] as const) {
  test(`adapter ${adapterMode} fixture shows the actual pre-click block reason without loading`, async ({ page }) => {
    const remote: string[] = []; let workers = 0;
    await page.addInitScript((mode) => {
      Object.defineProperty(navigator, 'gpu', { value: { requestAdapter: async () => {
        if (mode === 'rejected') throw new Error('adapter rejected fixture');
        if (mode === 'absent') return null;
        return { info: { description: 'fixture adapter', vendor: '', architecture: '' }, features: new Set<string>() };
      } }, configurable: true });
    }, adapterMode);
    page.on('worker', () => workers++);
    page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net/.test(request.url())) remote.push(request.url()); });
    await page.goto('/webgpu?lang=ja');
    const expected = adapterMode === 'absent' ? messages.ja.errorNoAdapter : adapterMode === 'rejected' ? messages.ja.errorAdapterProbe.replace('{error}', 'adapter rejected fixture') : messages.ja.blockedModel;
    await expect(page.locator('#analysis-blocked')).toHaveText(expected);
    await expect(page.getByRole('button', { name: messages.ja.analyze })).toBeDisabled();
    if (adapterMode === 'available') await expect(page.getByRole('button', { name: messages.ja.load })).toBeEnabled();
    else await expect(page.getByRole('button', { name: messages.ja.load })).toBeDisabled();
    expect(remote).toEqual([]); expect(workers).toBe(0);
  });
}

test('a model-load failure remains the analyze block reason instead of the unloaded-model hint', async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator, 'gpu', { value: { requestAdapter: async () => ({ info: { description: 'fixture adapter', vendor: '', architecture: '' }, features: new Set<string>() }) }, configurable: true });
    Object.defineProperty(window, 'Worker', { value: class {
      onmessage?: (event: { data: { type: string; code: string; message: string } }) => void;
      postMessage(message: { type: string }) {
        if (message.type === 'load') setTimeout(() => this.onmessage?.({ data: { type: 'error', code: 'native_failure', message: 'fixture engine initialization failed' } }), 1000);
      }
      terminate() {}
    }, configurable: true });
  });
  await page.goto('/webgpu?lang=en');
  await page.getByRole('button', { name: messages.en.load }).click();
  await expect(page.locator('#analysis-blocked')).toHaveText(messages.en.statusDownloading);
  await expect(page.locator('#analysis-blocked')).toHaveText('[native_failure] fixture engine initialization failed');
  await expect(page.getByRole('alert')).toContainText('fixture engine initialization failed');
  await expect(page.getByRole('button', { name: messages.en.analyze })).toBeDisabled();
});
