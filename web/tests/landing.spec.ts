import { expect, test } from '@playwright/test';
import { messages } from '../src/lib/i18n/landing';

test('landing catalogs cover the same keys and preserve interpolation parameters', () => {
  const keys = Object.keys(messages.en);
  for (const language of ['ko', 'ja'] as const) {
    expect(Object.keys(messages[language])).toEqual(keys);
    for (const key of keys as (keyof typeof messages.en)[]) {
      expect(messages[language][key].trim()).not.toBe('');
      const parameters = (text: string) => [...text.matchAll(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g)].map((match) => match[1]).sort();
      expect(parameters(messages[language][key])).toEqual(parameters(messages.en[key]));
    }
  }
});

test('switching all three languages updates text and metadata without resetting the decision or measured cells', async ({ page }) => {
  // A translated query title confirms hydration before interacting with the SSR controls.
  await page.goto('/?lang=ja');
  await expect(page).toHaveTitle(messages.ja.metaTitle);
  await page.locator('#site-language').selectOption('en');
  await expect(page).toHaveTitle(messages.en.metaTitle);
  const playground = page.locator('#playground');
  await playground.getByRole('button', { name: 'ordinal', exact: true }).click();
  await expect(playground.getByRole('button', { name: 'ordinal', exact: true })).toHaveAttribute('aria-pressed', 'true');
  const originalResult = await playground.locator('.demo-result pre').textContent();
  const originalCells = await page.locator('#performance tbody td').allTextContents();
  const originalCommand = await page.locator('.command-panel pre').textContent();
  for (const language of ['ko', 'ja', 'en'] as const) {
    await page.locator('#site-language').selectOption(language);
    await expect(page).toHaveTitle(messages[language].metaTitle);
    await expect(page.locator('html')).toHaveAttribute('lang', language);
    await expect(page.locator('meta[name="description"]')).toHaveAttribute('content', messages[language].metaDescription);
    await expect(page.locator('meta[property="og:description"]')).toHaveAttribute('content', messages[language].ogDescription);
    await expect(page.getByRole('navigation', { name: messages[language].mainNavigation })).toBeVisible();
    await expect(playground.locator('h3')).toHaveText(messages[language].questionOrdinal);
    await expect(playground.locator('.decision-status')).toHaveText(messages[language].selected);
    expect(await playground.locator('.demo-result pre').textContent()).toBe(originalResult);
    expect(await page.locator('#performance tbody td').allTextContents()).toEqual(originalCells);
    expect(await page.locator('.command-panel pre').textContent()).toBe(originalCommand);
    await expect(page.locator('.model-list a').first()).toContainText(messages[language].gemma4Detail);
    expect(await page.evaluate(() => localStorage.getItem('l2s1-locale'))).toBe(language);
  }
});

for (const language of ['ko', 'ja'] as const) {
  test(`${language} query localizes mobile landing and clipboard errors`, async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 900 });
    await page.addInitScript(() => {
      Object.defineProperty(navigator.clipboard, 'writeText', { value: async () => { throw new Error('clipboard denied'); } });
    });
    await page.goto(`/?lang=${language}`);
    await expect(page).toHaveTitle(messages[language].metaTitle);
    await expect(page.getByRole('heading', { level: 1 })).toContainText(messages[language].heroTitle);
    await expect(page.getByRole('link', { name: messages[language].gemmaSummary, exact: true })).toBeVisible();
    await page.getByRole('button', { name: messages[language].copyCommand, exact: true }).click();
    await expect(page.locator('.command-panel [role="status"]')).toHaveText(messages[language].clipboardError);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await expect(page.locator('.input-preview')).toContainText('"storage_requirement"');
    await expect(page.locator('.result-preview')).toContainText('"chilled"');
  });
}

test('prerendered introduction works without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  await page.goto('/');
  await expect(page).toHaveTitle('L2S1 — Local models. Typed decisions.');
  await expect(page.getByRole('heading', { level: 1 })).toContainText('A small decision.');
  await expect(page.getByRole('table')).toBeVisible();
  const docs = await page.request.get('/docs/LICENSE');
  expect(docs.ok()).toBeTruthy();
  expect(await docs.text()).toContain('MIT License');
  await context.close();
});

test('decision types demonstrate clear typed output examples', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  await expect(page.locator('#site-language')).toBeEnabled();
  const playground = page.locator('#playground');
  await expect(playground.getByText('Output example', { exact: true })).toBeVisible();
  await expect(playground.locator('.decision-status')).toHaveText('Selected');
  await expect(playground.locator('.demo-result')).toContainText('"selected": "chilled"');
  await playground.getByRole('button', { name: 'binary', exact: true }).click();
  await expect(playground.locator('.demo-result')).toContainText('"value": true');
  await playground.getByRole('button', { name: 'ordinal', exact: true }).click();
  await expect(playground.locator('.demo-result')).toContainText('"selected": "high"');
  expect(errors).toEqual([]);
});

test('quick start, model links, and downloads are usable', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/');
  await expect(page.locator('#site-language')).toBeEnabled();
  await page.getByRole('link', { name: 'Get started', exact: false }).first().click();
  await expect(page).toHaveURL(/#get-started$/);
  await page.getByRole('button', { name: 'Copy command' }).click();
  await expect(page.getByRole('button', { name: 'Copied ✓' })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('examples/warehouse.json');
  for (const link of await page.locator('a[href^="/docs/"]').all()) {
    const response = await page.request.get((await link.getAttribute('href'))!);
    expect(response.ok()).toBeTruthy();
  }
  await expect(page.locator('.model-list a')).toHaveCount(3);
  await expect(page.locator('.model-list')).not.toContainText('SmolLM2');
  await expect(page.locator('.model-list')).not.toContainText('TinyLlama');
  await expect(page.locator('#performance tbody')).not.toContainText('SmolLM2');
  await expect(page.locator('#performance tbody')).not.toContainText('TinyLlama');
  expect(await page.locator('.model-list a').first().getAttribute('href')).toContain('huggingface.co');
});

for (const viewport of [{ name: 'desktop', width: 1440, height: 1000 }, { name: 'mobile', width: 375, height: 900 }]) {
  test(`${viewport.name} layout stays within the viewport`, async ({ page }, testInfo) => {
    await page.setViewportSize(viewport);
    await page.goto('/');
    await expect(page.getByRole('heading', { level: 1 })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await page.screenshot({ path: testInfo.outputPath(`landing-${viewport.name}.png`), fullPage: true });
    await page.screenshot({ path: testInfo.outputPath(`landing-${viewport.name}-fold.png`) });
    await page.locator('#performance').screenshot({ path: testInfo.outputPath(`performance-${viewport.name}.png`) });
  });
}

test('public typed-decision downloads preserve measured records and portable provenance', async ({ page }) => {
  const { readFile } = await import('node:fs/promises');
  const { createHash } = await import('node:crypto');
  await page.goto('/');
  for (const name of ['gemma4-e2b-summary.json', 'qwen3-06b-summary.json', 'gemma4-e2b-scored.jsonl', 'qwen3-06b-scored.jsonl']) {
    const response = await page.request.get(`/benchmarks/typed-decisions-20260926/${name}`);
    expect(response.ok()).toBeTruthy();
    const published = await response.body();
    const source = await readFile(`../benchmarks/typed-decisions-20260926/${name}`);
    expect(createHash('sha256').update(published).digest('hex')).toBe(createHash('sha256').update(source).digest('hex'));
    if (name.endsWith('.jsonl')) {
      const records = published.toString('utf8').trim().split('\n').map((line) => JSON.parse(line));
      expect(records).toHaveLength(2000);
      expect(records.every((record) => !('state' in record) && !('reasoning_text' in record))).toBeTruthy();
    }
  }
  const manifestResponse = await page.request.get('/benchmarks/typed-decisions-20260926/manifest.json');
  expect(manifestResponse.ok()).toBeTruthy();
  const manifest = await manifestResponse.json();
  expect(manifest.libraries_original_path).toBe('lib');
  expect(manifest.public_export.path_normalization).toContain('basenames');
  expect(manifest.runs['gemma4-e2b'].model_sha256).toHaveLength(64);
  expect(manifest.runs['qwen3-06b'].command).toContain('--cuda');
  const report = await page.request.get('/docs/docs/TYPED_DECISIONS_BENCHMARK.md');
  expect(await report.text()).toContain('(/benchmarks/typed-decisions-20260926/gemma4-e2b-scored.jsonl)');
  await expect(page.getByRole('link', { name: 'Gemma summary JSON', exact: true })).toBeVisible();
});

test('RTX 3060 evidence and dataset selection preserve separate measurement scopes', async ({ page }) => {
  const { readFile } = await import('node:fs/promises');
  const { createHash } = await import('node:crypto');
  await page.goto('/?lang=en');
  await expect(page.locator('#site-language')).toBeEnabled();
  const response = await page.request.get('/benchmarks/rtx3060-20260926/summary.json');
  expect(response.ok()).toBeTruthy();
  const published = await response.body();
  const summary = JSON.parse(published.toString());
  const source = await readFile('../benchmarks/rtx3060-20260926/summary.json');
  expect(published.equals(source)).toBeTruthy();
  expect(summary.rows).toHaveLength(20);
  expect(summary.decisions).toBe(4620);
  expect(summary.rows.every((row: { model: string; valid: number; errors: number }) => !/SmolLM2|TinyLlama/.test(row.model) && row.valid === 231 && row.errors === 0)).toBeTruthy();
  for (const row of summary.rows) {
    expect(row.rawAccuracy).toBeCloseTo(row.counts.rawCorrect / row.valid * 100, 10);
    expect(row.coverage).toBeCloseTo(row.counts.accepted / row.valid * 100, 10);
    expect(row.acceptedAccuracy).toBeCloseTo(row.counts.acceptedCorrect / row.counts.accepted * 100, 10);
  }
  const manifestResponse = await page.request.get('/benchmarks/rtx3060-20260926/manifest.json');
  expect(manifestResponse.ok()).toBeTruthy();
  const manifest = await manifestResponse.json();
  expect(manifest.summarySha256).toBe(createHash('sha256').update(published).digest('hex'));
  expect(JSON.stringify(manifest)).not.toMatch(/\/home\/|\/tmp\//);
  for (const row of summary.rows) {
    expect(Number.isSafeInteger(row.fileSizeBytes) && row.fileSizeBytes > 0).toBeTruthy();
    const checkpoint = manifest.models.find((model: { id: string }) => model.id === row.id);
    expect(checkpoint.fileSizeBytes).toBe(row.fileSizeBytes);
    expect(checkpoint.modelSha256).toHaveLength(64);
    expect(checkpoint.fileSizeEvidence.method).toMatch(/metadata|SHA-256/);
    const displayed = page.locator('#performance tbody tr').filter({ has: page.locator('td').getByText(row.model, { exact: true }) });
    await expect(displayed.locator('td').last()).toHaveText(`${(row.fileSizeBytes / 1024 ** 3).toFixed(2)} GiB`);
  }
  expect(summary.rows.find((row: { id: string }) => row.id === 'Qwen3-0.6B-Q8_0').fileSizeBytes).toBe(639446688);
  await expect(page.locator('.capacity-highlight')).toContainText('609.8');
  await expect(page.locator('.compatibility-highlight')).toContainText('llama.cpp compatible.');
  await expect(page.locator('.compatibility-highlight')).toContainText('Llama 3.2 3B Q8_0');

  expect(summary.rows.find((row: { model: string }) => row.model.startsWith('gpt-oss')).config.runtimeEnvironmentOverrides).toEqual({ GGML_CUDA_DISABLE_GRAPHS: '1' });
  const table = page.locator('#performance table');
  await expect(table.locator('tbody tr')).toHaveCount(20);
  await expect(table.locator('tbody tr').first()).toContainText('79.65%');
  await expect(page.locator('.performance-value')).toContainText('34.0');
  await expect(page.locator('.performance-stats')).toContainText('1.81');
  await expect(page.locator('.performance-stats')).toContainText('79.65');
  await expect(page.locator('.performance-stats')).toContainText('2.78');
  const q4 = summary.rows.find((row: { id: string }) => row.id === 'Qwen3.5-9B-Q4_K_M');
  const q8 = summary.rows.find((row: { id: string }) => row.id === 'Qwen3.5-9B-Q8_0');
  expect(q4.counts.rawCorrect).toBe(q8.counts.rawCorrect);
  expect((1 - q4.peakGpuMiB / q8.peakGpuMiB) * 100).toBeCloseTo(36.07, 2);
  await expect(page.locator('.quantization-highlight')).toContainText('36.1%');
  const sharedResponse = await page.request.get('/benchmarks/shared-state-cache-20260925/highlight.json');
  const shared = await sharedResponse.json();
  expect(shared.speedup).toBeCloseTo(shared.freshMedianMs / shared.sharedMedianMs, 10);
  expect(shared.questions).toBe(16);
  expect(shared.gpu).toBe('NVIDIA GeForce RTX 3080');
  expect(shared.differences.every((delta: Record<string, number>) => Object.values(delta).every((value) => value === 0))).toBeTruthy();
  await page.getByRole('button', { name: 'RTX 3080 · Warehouse / Typed-decisions', exact: true }).click();
  await expect(table.locator('tbody tr')).toHaveCount(3);
  await expect(table).toContainText('54.30%');
  await expect(page.locator('.performance-chart')).toBeVisible();
  await page.locator('#site-language').selectOption('ko');
  await expect(page.getByRole('button', { name: 'RTX 3080 · Warehouse / Typed-decisions', exact: true })).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: 'RTX 3060 · JevBench', exact: true }).click();
  await expect(table.locator('tbody tr')).toHaveCount(20);
  await expect(page.locator('.performance-chart')).toHaveCount(0);
  const report = await page.request.get('/docs/docs/ko/RTX3060_BENCHMARK.md');
  expect(await report.text()).toContain('(/benchmarks/rtx3060-20260926/summary.json)');
});
