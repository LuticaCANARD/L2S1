import { expect, test } from '@playwright/test';
import { messages } from '../src/lib/i18n/common';

const destinations = ['/#how-it-works', '/webgpu', '/demo', '/#models', '/#performance'];

for (const language of ['ko', 'en', 'ja'] as const) {
  test(`${language} shares one accessible header and navigation across routes`, async ({ page }) => {
    for (const route of ['/', '/demo', '/webgpu']) {
      await page.goto(`${route}?lang=${language}`);
      await expect(page.locator('#site-language')).toBeEnabled();
      const header = page.getByRole('banner');
      await expect(header).toHaveCount(1);
      const navigation = header.getByRole('navigation', { name: messages[language].mainNavigation });
      await expect(navigation).toBeVisible();
      expect(await navigation.getByRole('link').evaluateAll((links) => links.map((link) => link.getAttribute('href')))).toEqual(destinations);
      await expect(header.locator('#site-language')).toHaveCount(1);
      await expect(header.locator('#site-theme')).toHaveCount(1);
      await expect(header.getByRole('link', { name: messages[language].home })).toHaveAttribute('href', '/');
      await expect(header.locator('a[href="/#get-started"]')).toBeVisible();
      if (route !== '/') await expect(navigation.locator(`a[href="${route}"]`)).toHaveAttribute('aria-current', 'page');
    }
  });
}

test('mobile shared navigation opens home sections and demos while preserving preferences', async ({ page }) => {
  const errors: string[] = [];
  const remote: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => {
    if (/huggingface\.co|cdn\.jsdelivr\.net|\/inference\/|\/v1\/decisions/.test(request.url())) remote.push(request.url());
  });
  await page.setViewportSize({ width: 375, height: 900 });
  await page.goto('/demo?lang=en');
  await page.locator('#site-language').selectOption('ja');
  await page.locator('#site-theme').selectOption('dark');

  for (const anchor of ['how-it-works', 'models', 'performance', 'get-started']) {
    await page.goto('/webgpu');
    const header = page.getByRole('banner');
    await expect(header.getByRole('navigation')).toBeVisible();
    await header.locator(`a[href="/#${anchor}"]`).click();
    await expect(page).toHaveURL(new RegExp(`/#${anchor}$`));
    await expect(page.locator(`#${anchor}`)).toBeInViewport();
    await expect(page.locator('#site-language')).toHaveValue('ja');
    await expect(page.locator('#site-theme')).toHaveValue('dark');
  }

  for (const route of ['/demo', '/webgpu']) {
    await page.getByRole('banner').locator(`a[href="${route}"]`).click();
    await expect(page).toHaveURL(new RegExp(`${route}$`));
    await expect(page.locator('html')).toHaveAttribute('lang', 'ja');
    await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
  }
  await page.getByRole('banner').getByRole('link', { name: messages.ja.home }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole('banner')).toHaveCount(1);
  expect(errors).toEqual([]);
  expect(remote).toEqual([]);
});
