import { expect, test } from '@playwright/test';

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

test('threshold and decision types demonstrate explicit abstention', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/');
  const playground = page.locator('#playground');
  await expect(playground.getByText('Illustrative scoring demo.', { exact: true })).toBeVisible();
  await expect(playground.locator('.decision-status')).toHaveText('Accepted');
  await page.locator('#threshold').focus();
  await page.keyboard.press('End');
  await expect(playground.locator('.decision-status')).toHaveText('Abstained');
  await expect(playground.locator('.demo-result')).toContainText('"selected": null');
  await expect(playground.locator('.demo-result')).toContainText('low_top_probability');
  await playground.getByRole('button', { name: 'binary', exact: true }).click();
  await page.locator('#threshold').focus();
  await page.keyboard.press('Home');
  await expect(playground.locator('.demo-result')).toContainText('"value": true');
  await playground.getByRole('button', { name: 'ordinal', exact: true }).click();
  await expect(playground.locator('.demo-result')).toContainText('"selected": "high"');
  expect(errors).toEqual([]);
});

test('quick start, model links, and downloads are usable', async ({ page, context }) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/');
  await page.getByRole('link', { name: 'Get started', exact: false }).first().click();
  await expect(page).toHaveURL(/#get-started$/);
  await page.getByRole('button', { name: 'Copy command' }).click();
  await expect(page.getByRole('button', { name: 'Copied ✓' })).toBeVisible();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('examples/warehouse.json');
  for (const link of await page.locator('a[href^="/docs/"]').all()) {
    const response = await page.request.get((await link.getAttribute('href'))!);
    expect(response.ok()).toBeTruthy();
  }
  await expect(page.locator('.model-list a')).toHaveCount(5);
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
  });
}
