import { expect, test } from '@playwright/test';
const languages = ['ko', 'en', 'ja'] as const;
for (const language of languages) {
  test(`${language} covers all routes, both themes, metadata and mobile width`, async ({ page }, testInfo) => {
    const errors: string[] = []; const remote: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    page.on('request', (request) => { if (/huggingface\.co|cdn\.jsdelivr\.net|\/inference\/|\/v1\/decisions/.test(request.url())) remote.push(request.url()); });
    for (const route of ['/', '/demo', '/webgpu']) {
      await page.goto(`${route}?lang=${language}`);
      await expect(page.locator('#site-language')).toBeEnabled();
      await expect(page.locator('html')).toHaveAttribute('lang', language);
      await expect(page.locator('#site-language')).toHaveValue(language);
      for (const appearance of ['light', 'dark']) {
        await page.locator('#site-theme').selectOption(appearance);
        await expect(page.locator('html')).toHaveAttribute('data-theme', appearance);
        const color = await page.locator('body').evaluate((body) => getComputedStyle(body).backgroundColor);
        if (appearance === 'dark') expect(color).toBe('rgb(5, 7, 10)');
        else expect(color).toBe('rgb(231, 238, 244)');
        await page.setViewportSize({ width: 375, height: 900 });
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
        await page.screenshot({ path: testInfo.outputPath(`${route.replaceAll('/', '') || 'home'}-${language}-${appearance}-mobile.png`), fullPage: true });
        await page.setViewportSize({ width: 1440, height: 1000 });
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
      }
      const heading = await page.locator('h1').textContent();
      const title = await page.title();
      const description = await page.locator('meta[name="description"]').getAttribute('content');
      if (language === 'en') { expect(heading).not.toMatch(/[가-힣ぁ-んァ-ヶ]/); expect(title).not.toMatch(/[가-힣ぁ-んァ-ヶ]/); expect(description).not.toMatch(/[가-힣ぁ-んァ-ヶ]/); }
      else if (language === 'ko') { expect(heading).toMatch(/[가-힣]/); expect(title).toMatch(/[가-힣]/); expect(description).toMatch(/[가-힣]/); }
      else { expect(heading).toMatch(/[ぁ-んァ-ヶ一-龯]/); expect(title).toMatch(/[ぁ-んァ-ヶ一-龯]/); expect(description).toMatch(/[ぁ-んァ-ヶ一-龯]/); }
    }
    expect(errors).toEqual([]); expect(remote).toEqual([]);
  });
}
test('language and theme persist through navigation and reload; explicit URL takes priority', async ({ page }) => {
  await page.goto('/?lang=en');
  await page.locator('#site-language').selectOption('ja');
  await page.locator('#site-theme').selectOption('dark');
  await expect(page).toHaveURL(/lang=ja/);
  await page.locator('a[href="/webgpu"]').first().click();
  await expect(page).toHaveURL(/\/webgpu/);
  await expect(page.locator('html')).toHaveAttribute('lang', 'ja');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.reload();
  await expect(page.locator('#site-language')).toHaveValue('ja');
  await expect(page.locator('#site-theme')).toHaveValue('dark');
  await page.goto('/demo?lang=ko');
  await expect(page.locator('#site-language')).toHaveValue('ko');
  await expect(page.locator('#site-theme')).toHaveValue('dark');
});
test('locale and theme changes preserve editable input, policy and recorded raw evidence', async ({ page }) => {
  await page.goto('/demo?lang=ko');
  await page.locator('.result-card').first().waitFor();
  const recorded = await page.locator('.raw pre').textContent();
  await page.locator('#site-language').selectOption('en');
  await page.locator('#site-theme').selectOption('dark');
  expect(await page.locator('.raw pre').textContent()).toBe(recorded);
  await page.goto('/webgpu?lang=ko');
  await expect(page.locator('#site-language')).toBeEnabled();
  const state = '{"text":"My edited input"}';
  await page.locator('#state').fill(state);
  await page.locator('#error-rate').fill('12');
  await page.getByText('사용자 지정 보류·추론 실패 이유', { exact: true }).click();
  const failure = '{"reasoning_limit":"My own error message"}';
  await page.locator('#failure').fill(failure);
  await page.locator('#site-language').selectOption('ja');
  await page.locator('#site-theme').selectOption('light');
  await expect(page.locator('#state')).toHaveValue(state);
  await expect(page.locator('#error-rate')).toHaveValue('12');
  await expect(page.locator('#min-top')).toHaveValue('0.88');
  await expect(page.locator('#failure')).toHaveValue(failure);
});
test('system appearance follows OS changes and explicit choice overrides it', async ({ page }) => {
  await page.emulateMedia({ colorScheme: 'dark' }); await page.goto('/?lang=en');
  await expect(page.locator('#site-theme')).toHaveValue('system');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.emulateMedia({ colorScheme: 'light' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await page.locator('#site-theme').selectOption('dark'); await page.emulateMedia({ colorScheme: 'light' });
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});
test('preferences work when browser storage is unavailable', async ({ page }) => {
  await page.addInitScript(() => { Object.defineProperty(Storage.prototype, 'getItem', { value: () => { throw new Error('Storage disabled'); } }); Object.defineProperty(Storage.prototype, 'setItem', { value: () => { throw new Error('Storage disabled'); } }); });
  await page.goto('/webgpu?lang=ja');
  await expect(page.locator('html')).toHaveAttribute('lang', 'ja');
  await page.locator('#site-language').selectOption('ko'); await page.locator('#site-theme').selectOption('dark');
  await expect(page.locator('html')).toHaveAttribute('lang', 'ko'); await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
});
test('browser language is selected when URL and stored preference are absent', async ({ browser }) => {
  const context = await browser.newContext({ locale: 'ja-JP' });
  const page = await context.newPage(); await page.goto('/');
  await expect(page.locator('#site-language')).toHaveValue('ja'); await expect(page.locator('html')).toHaveAttribute('lang', 'ja');
  await context.close();
});
