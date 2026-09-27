import { expect, test } from '@playwright/test';
import { installMessages } from '../src/lib/i18n/install';

for (const lang of ['en', 'ko', 'ja'] as const) {
  test(`${lang}: installation guides provide copyable commands and matching Python runtimes`, async ({ page, context }) => {
    await context.grantPermissions(['clipboard-read', 'clipboard-write']);
    await page.goto(`/?lang=${lang}#get-started`);
    await expect(page.locator('#site-language')).toBeEnabled();
    const guide = page.getByTestId('install-guide');
    const text = installMessages[lang];
    await expect(guide.getByRole('heading', { name: text.title })).toBeVisible();
    for (const [language, command] of [
      ['TypeScript', 'npm install @l2s1/node@0.1.3'],
      ['Python', 'python -m pip install l2s1-sdk==0.1.3'],
      ['Rust', 'cargo add l2s1@0.1.3 --features llama'],
    ]) {
      await guide.getByRole('button', { name: language, exact: true }).click();
      await expect(guide.getByRole('button', { name: language, exact: true })).toHaveAttribute('aria-pressed', 'true');
      await guide.getByRole('button', { name: `${text.copy}: ${text.install}`, exact: true }).click();
      await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(command);
      await expect(guide.getByRole('status')).toHaveText(text.copied);
      const docs = await page.request.get((await guide.getByRole('link', { name: text.exampleLink, exact: false }).getAttribute('href'))!);
      expect(docs.ok()).toBeTruthy();
    }
    await guide.getByRole('button', { name: 'Python', exact: true }).click();
    for (const platform of ['linux-x64', 'linux-arm64', 'darwin-arm64', 'darwin-x64', 'win32-x64']) {
      await guide.getByLabel(text.platform).selectOption(platform);
      const archive = `l2s1-runtime-${platform}-0.1.3.tgz`;
      await expect(guide.getByRole('link', { name: text.download, exact: false })).toHaveAttribute('href', `https://github.com/LuticaCANARD/L2S1/releases/download/v0.1.3/${archive}`);
      await guide.getByRole('button', { name: `${text.copy}: ${text.extract}`, exact: true }).click();
      await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(`tar -xzf ${archive}`);
    }
  });
}

test('mobile installation handles clipboard denial and keyboard selection without overflow', async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 900 });
  await page.addInitScript(() => {
    Object.defineProperty(navigator.clipboard, 'writeText', { value: async () => { throw new Error('denied'); } });
  });
  await page.goto('/?lang=ko#get-started');
  await expect(page.locator('#site-language')).toBeEnabled();
  const guide = page.getByTestId('install-guide');
  const python = guide.getByRole('button', { name: 'Python', exact: true });
  await python.focus(); await page.keyboard.press('Enter');
  await expect(python).toHaveAttribute('aria-pressed', 'true');
  await guide.getByRole('button', { name: '복사: 패키지 설치', exact: true }).click();
  await expect(guide.getByRole('status')).toHaveText(installMessages.ko.copyError);
  await expect(guide.locator('pre').first()).toHaveText('python -m pip install l2s1-sdk==0.1.3');
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBeTruthy();
});
