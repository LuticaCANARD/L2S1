import { expect, test } from '@playwright/test';
import { messages } from '../src/lib/i18n/demo';

test.beforeEach(async ({ page }) => { await page.addInitScript(() => localStorage.setItem('l2s1-locale', 'ko')); });

test('actual image recording loads without calling inference and edits clear old results', async ({ page }) => {
  let inferenceCalls = 0;
  page.on('request', (request) => { if (request.url().includes('/inference/v1/decisions')) inferenceCalls++; });
  await page.goto('/demo');
  await expect(page.getByText('기록된 결과', { exact: true })).toBeVisible();
  await expect(page.locator('.result-card')).toHaveCount(3);
  await expect(page.locator('.result-card').first().locator('.selected-value')).toHaveText('metal');
  await expect(page.locator('.verification')).toHaveCount(0);
  await expect(page.getByRole('heading', { name: '주요 재질' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '하나의 주요 물체' })).toBeVisible();
  await expect(page.getByRole('heading', { name: '투명도' })).toBeVisible();
  expect(inferenceCalls).toBe(0);
  await page.locator('#state').fill('{"task":"edited input"}');
  await expect(page.locator('.result-card')).toHaveCount(0);
  await page.getByRole('button', { name: '기록된 결과 보기' }).click();
  await expect(page.locator('.result-card')).toHaveCount(3);
  await page.locator('#error-rate').fill('10');
  await expect(page.locator('#min-top')).toHaveValue('0.9');
  await expect(page.locator('.result-card')).toHaveCount(0);
});

test('image upload replaces recording and malformed input fails before inference', async ({ page }) => {
  await page.goto('/demo');
  await expect(page.getByText('기록된 결과', { exact: true })).toBeVisible();
  await page.locator('#photo-upload').setInputFiles('static/demo/sample.jpg');
  await expect(page.locator('.result-card')).toHaveCount(0);
  await expect(page.locator('.photo img')).toHaveAttribute('src', /^blob:/);
  await page.locator('#state').fill('{broken');
  await page.getByRole('button', { name: '내 로컬 모델로 분석' }).click();
  await expect(page.getByRole('alert')).toContainText(messages.ko.jsonInvalid);
  await expect(page.getByRole('button', { name: '내 로컬 모델로 분석' })).toBeEnabled();
});

test('recorded text thinking exposes generated-token evidence and photo thinking stays unavailable', async ({ page }) => {
  await page.goto('/demo');
  await expect(page.getByText('기록된 결과', { exact: true })).toBeVisible();
  expect(await page.locator('#reasoning option[value="thinking"]').isDisabled()).toBeTruthy();
  await page.getByRole('button', { name: '텍스트 판단', exact: true }).click();
  await expect(page.getByRole('button', { name: 'thinking 기록' })).toBeEnabled();
  await page.getByRole('button', { name: 'thinking 기록' }).click();
  await expect(page.locator('#reasoning')).toHaveValue('thinking');
  await expect(page.locator('.thinking-info').first()).toContainText('생성된 생각 토큰');
});

for (const viewport of [{ name: 'desktop', width: 1440, height: 1000 }, { name: 'mobile', width: 375, height: 900 }]) {
  test(`demo ${viewport.name} layout fits and recording is inspectable`, async ({ page }, testInfo) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.setViewportSize(viewport);
    await page.goto('/demo');
    await expect(page.getByText('기록된 결과', { exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await page.screenshot({ path: testInfo.outputPath(`demo-${viewport.name}.png`), fullPage: true });
    await page.screenshot({ path: testInfo.outputPath(`demo-${viewport.name}-fold.png`) });
    expect(errors).toEqual([]);
  });
}

test('language switches translate the recording while preserving evidence and edited failure messages', async ({ page }) => {
  let inferenceCalls = 0;
  page.on('request', (request) => { if (/\/(?:text-)?inference\/v1\/decisions/.test(request.url())) inferenceCalls++; });
  await page.goto('/demo?lang=en');
  await expect(page.getByText('Recorded results', { exact: true })).toBeVisible();
  const state = await page.locator('#state').inputValue();
  const questions = await page.locator('#decisions').inputValue();
  const evidence = await page.locator('.raw pre').textContent();
  await expect(page.locator('#failure-reasons')).toHaveValue(/human review is needed/);
  await page.locator('#site-language').selectOption('ja');
  await expect(page.getByRole('heading', { name: '主な素材', exact: true })).toBeVisible();
  await expect(page.locator('.result-card').first().locator('.selected-value')).toHaveText('metal');
  await expect(page.locator('#failure-reasons')).toHaveValue(/人の確認が必要/);
  await expect(page.locator('#state')).toHaveValue(state);
  await expect(page.locator('#decisions')).toHaveValue(questions);
  expect(await page.locator('.raw pre').textContent()).toBe(evidence);
  await expect(page).toHaveTitle('L2S1 — 画像分析デモ');
  await page.locator('#failure-reasons').fill('{"low_top_probability":"My own review instruction"}');
  await page.locator('#site-language').selectOption('en');
  await expect(page.locator('#failure-reasons')).toHaveValue('{"low_top_probability":"My own review instruction"}');
  await expect(page.locator('#state')).toHaveValue(state);
  expect(inferenceCalls).toBe(0);
});

test('a validation error changes language without repeating inference or changing input', async ({ page }) => {
  await page.goto('/demo?lang=en');
  await expect(page.getByText('Recorded results', { exact: true })).toBeVisible();
  await page.locator('#state').fill('{broken');
  await page.getByRole('button', { name: 'Analyze with my local model' }).click();
  await expect(page.getByRole('alert')).toContainText('Invalid input JSON');
  await page.locator('#site-language').selectOption('ja');
  await expect(page.getByRole('alert')).toContainText('入力JSON形式が正しくありません');
  await expect(page.locator('#state')).toHaveValue('{broken');
});

for (const language of ['en', 'ja']) {
  test(`demo ${language} mobile layout supports the shared dark theme controls`, async ({ page }, testInfo) => {
    await page.setViewportSize({ width: 375, height: 900 });
    await page.goto(`/demo?lang=${language}`);
    await expect(page.locator('.result-card')).toHaveCount(3);
    await page.locator('#site-theme').selectOption('dark');
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await page.screenshot({ path: testInfo.outputPath(`demo-${language}-dark-mobile.png`), fullPage: true });
  });
}
