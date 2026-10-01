import { expect, test } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';

test('RTX 5090 rules evidence preserves counts, timeout, abstentions and language scope', async ({ page }, testInfo) => {
  await page.goto('/?lang=en');
  await expect(page.locator('#site-language')).toBeEnabled();
  await expect(page.locator('.rules-highlight')).toContainText('94.3%');
  await expect(page.locator('.rules-highlight')).toContainText('36 synthetic rule decisions');
  await page.getByRole('button', { name: 'RTX 5090 · Decision rules', exact: true }).click();
  const table = page.locator('#performance table');
  await expect(table.locator('tbody tr')).toHaveCount(10);
  const cells = await table.locator('tbody td').allTextContents();
  const gemmaCuda = table.locator('tbody tr').filter({ hasText: 'gemma-4-E2B-it-Q8_0.gguf' }).filter({ hasText: 'CUDA' });
  expect(await gemmaCuda.locator('td').allTextContents()).toEqual([
    'gemma-4-E2B-it-Q8_0.gguf', 'CUDA', 'ok', '97.2%', '94.3%', '91.7%', '94.4%', '67.6 / 80.7', '43.90', '40.24',
  ]);
  const timeout = table.locator('tbody tr').filter({ hasText: 'timeout' });
  expect(await timeout.locator('td').allTextContents()).toEqual([
    'gemma-4-E2B-it-Q8_0.gguf', 'CPU', 'timeout', '—', '—', '—', '—', '—', '—', '—',
  ]);
  const tinyRows = table.locator('tbody tr').filter({ hasText: 'tinyllama' });
  await expect(tinyRows).toHaveCount(2);
  for (const row of await tinyRows.all()) {
    await expect(row.locator('td').nth(4)).toHaveText('n/a');
    await expect(row.locator('td').nth(9)).toHaveText('0.00');
  }
  const path = '/benchmarks/decision-rules-windows-20260926/';
  const published = await page.request.get(path + 'summary.json');
  expect(published.ok()).toBeTruthy();
  const bytes = await published.body();
  expect(bytes.equals(await readFile('../benchmarks/decision-rules-windows-20260926/summary.json'))).toBeTruthy();
  const audit = await (await page.request.get(path + 'audit.json')).json();
  expect(audit.summary_sha256).toBe(createHash('sha256').update(bytes).digest('hex'));
  expect(audit).toMatchObject({ completed_runs: 9, timeout_runs: 1, timing_samples_checked: 324, independent_decisions_per_run: 36, reported_fixture_line_endings: 'CRLF', gpu: 'NVIDIA GeForce RTX 5090' });
  expect(audit.runs.at(-1)).toMatchObject({ acceptedCorrectPerPass: 33, acceptedWrongPerPass: 2, abstainedPerPass: 1 });
  expect(audit).toEqual(JSON.parse(await readFile('../benchmarks/decision-rules-windows-20260926/audit.json', 'utf8')));
  const provenance = await (await page.request.get(path + 'provenance.json')).json();
  expect(provenance).toMatchObject({ gpu: 'NVIDIA GeForce RTX 5090', original_reports_available: false, inference_rerun: false });
  for (const reportPath of [path + 'README.md', '/docs/benchmarks/decision-rules-windows-20260926/README.md']) {
    const report = await page.request.get(reportPath);
    expect(report.ok()).toBeTruthy();
    expect(await report.text()).toContain(`(${path}summary.json)`);
    expect(await report.text()).toContain('(/docs/docs/en/BENCHMARK.md)');
  }
  for (const language of ['ko', 'ja', 'en']) {
    await page.locator('#site-language').selectOption(language);
    expect(await table.locator('tbody td').allTextContents()).toEqual(cells);
    await expect(page.getByRole('button', { name: 'RTX 5090 · Decision rules', exact: true })).toHaveAttribute('aria-pressed', 'true');
    const doc = await page.request.get(`/docs/docs/${language}/BENCHMARK.md`);
    expect(doc.ok()).toBeTruthy();
    expect(await doc.text()).toContain('<a id="recorded-windows-rtx-5090-results"></a>');
    expect(await doc.text()).toContain(`(${path}summary.json)`);
    const readmeName = language === 'en' ? 'README.md' : `README.${language}.md`;
    const readme = await page.request.get(`/docs/${readmeName}`);
    expect(await readme.text()).toContain('94.3%');
    expect(await readme.text()).toContain(`(${path}summary.json)`);
  }
  for (const viewport of [{ width: 1440, height: 1100 }, { width: 390, height: 844 }]) {
    await page.setViewportSize(viewport);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBeTruthy();
    await page.locator('#performance').screenshot({ path: testInfo.outputPath(`decision-rules-${viewport.width}.png`) });
  }
});
