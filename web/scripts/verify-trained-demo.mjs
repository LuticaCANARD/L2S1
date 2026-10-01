// End-to-end check against the running local UI and real classifier server.
import { chromium } from '@playwright/test';
import assert from 'node:assert/strict';
import { readFile, mkdir } from 'node:fs/promises';
const browser = await chromium.launch({ executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE || undefined });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } });
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto('http://127.0.0.1:5173/trashnet?lang=ko');
  await page.getByText('94.2%', { exact: true }).first().waitFor();
  await page.getByRole('button', { name: '선택한 예제로 실행', exact: true }).waitFor();
  await page.getByRole('button', { name: '선택한 예제로 실행', exact: true }).click();
  await page.locator('.live-result').waitFor();
  assert.match(await page.locator('.live-result h3').innerText(), /유리/);
  // Uploaded filename deliberately contradicts the pixels: only image bytes matter.
  await page.locator('.live-classifier input[type=file]').setInputFiles({ name: 'paper.jpg', mimeType: 'image/jpeg', buffer: await readFile('static/trashnet/images/glass330.jpg') });
  await page.waitForFunction(() => document.querySelector('.live-result')?.textContent?.includes('paper.jpg'));
  assert.match(await page.locator('.live-result h3').innerText(), /유리/);
  await page.getByLabel('정답 재질', { exact: true }).selectOption('trash');
  await page.locator('.photo-card').filter({ hasText: 'trash6.jpg' }).click();
  await page.evaluate(() => window.scrollTo(0, 0));
  await mkdir('../results/vision-improvement-20260927/screenshots', { recursive: true });
  await page.evaluate(async () => { await Promise.all([...document.images].map(image => { image.loading = 'eager'; return image.decode().catch(() => {}); })); });
  await page.screenshot({ path: '../results/vision-improvement-20260927/screenshots/comparison-overview.png' });
  await page.screenshot({ path: '../results/vision-improvement-20260927/screenshots/comparison-desktop.png', fullPage: false, animations: 'disabled' });
  await page.getByLabel('사진 묶음', { exact: true }).selectOption('test');
  await page.getByLabel('판단 결과', { exact: true }).selectOption('wrong');
  assert.equal(await page.locator('.photo-card').count(), 15);
  await page.screenshot({ path: '../results/vision-improvement-20260927/screenshots/holdout-errors.png', fullPage: false, animations: 'disabled' });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByLabel('화면 모드').selectOption('dark');
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: '../results/vision-improvement-20260927/screenshots/comparison-mobile.png', fullPage: false, animations: 'disabled' });
  assert.deepEqual(errors, []);
  console.log('PASS: real sample inference, uploaded pixels with misleading filename, 15 holdout errors, mobile dark mode, no page errors');
} finally { await browser.close(); }
