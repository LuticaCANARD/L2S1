<script lang="ts">
  import { resolve } from '$app/paths';
  import { page } from '$app/state';
  import { locale, theme, translate } from '$lib/i18n';
  import { messages } from '$lib/i18n/common';

  let { ready, onlocalechange, onthemechange }: {
    ready: boolean;
    onlocalechange: (event: Event) => void;
    onthemechange: (event: Event) => void;
  } = $props();
  const t = (key: keyof typeof messages.en) => translate($locale, messages, key);
</script>

<header class="site-header">
  <div class="header-inner">
    <a class="brand" href={resolve('/')} aria-label={t('home')} aria-current={page.url.pathname === resolve('/') ? 'page' : undefined}>
      <span class="brand-symbol" aria-hidden="true">s1<span>↗</span></span><span>L2S1</span>
    </a>
    <nav aria-label={t('mainNavigation')}>
      <a href={resolve('/#how-it-works')}>{t('howItWorks')}</a>
      <a href={resolve('/webgpu')} aria-current={page.url.pathname === resolve('/webgpu') ? 'page' : undefined}>{t('webgpuDemo')}</a>
      <a href={resolve('/demo')} aria-current={page.url.pathname === resolve('/demo') ? 'page' : undefined}>{t('imageDemo')}</a>
      <a href={resolve('/#models')}>{t('models')}</a>
      <a href={resolve('/#performance')}>{t('performance')}</a>
    </nav>
    <a class="header-cta" href={resolve('/#get-started')}>{t('startBuilding')} <span aria-hidden="true">↗</span></a>
    <div class="preferences" role="group" aria-label={t('preferences')}>
      <div class="preference-field">
        <label for="site-language">{t('language')}</label>
        <select id="site-language" value={$locale} onchange={onlocalechange} disabled={!ready}>
          <option value="ko">한국어</option><option value="en">English</option><option value="ja">日本語</option>
        </select>
      </div>
      <div class="preference-field">
        <label for="site-theme">{t('theme')}</label>
        <select id="site-theme" value={$theme} onchange={onthemechange} disabled={!ready}>
          <option value="system">{t('system')}</option><option value="light">{t('light')}</option><option value="dark">{t('dark')}</option>
        </select>
      </div>
    </div>
  </div>
</header>

<style>
  .site-header { background: var(--site-background); color: var(--site-text); border-bottom: 1px solid var(--site-border); font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  .header-inner { max-width: 1380px; margin: auto; padding: 22px 42px 16px; display: grid; grid-template-columns: auto minmax(0, 1fr) auto; gap: 16px 28px; align-items: center; box-sizing: border-box; }
  .brand { display: inline-flex; align-items: center; gap: 11px; font-size: 24px; font-weight: 800; letter-spacing: -.8px; text-decoration: none; color: inherit; }
  .brand-symbol { display: inline-flex; align-items: center; justify-content: center; width: 40px; height: 40px; background: var(--theme-terminal); color: var(--theme-on-primary); font-size: 22px; border-radius: 8px; letter-spacing: -2px; }
  .brand-symbol span { font-size: 15px; margin-left: 1px; align-self: flex-start; margin-top: 4px; }
  nav { display: flex; justify-content: center; flex-wrap: wrap; gap: 12px 22px; min-width: 0; }
  nav a { color: inherit; text-decoration: none; font-size: 13px; line-height: 1.5; white-space: nowrap; }
  nav a:hover, nav a[aria-current='page'] { color: var(--theme-accent); text-decoration: underline; text-underline-offset: 5px; }
  .header-cta { border: 1px solid var(--site-border); border-radius: 5px; padding: 10px 13px; font-size: 12px; font-weight: 600; text-decoration: none; color: inherit; white-space: nowrap; }
  .header-cta span { margin-left: 12px; }
  .preferences { grid-column: 2 / -1; display: flex; justify-content: flex-end; flex-wrap: wrap; gap: 12px 18px; font-size: 12px; }
  .preference-field { display: flex; align-items: center; gap: 8px; }
  label { font-size: 12px; }
  select { font: inherit; color: var(--site-text); background: var(--site-control); border: 1px solid var(--site-border); border-radius: 5px; padding: 6px 9px; max-width: 160px; }
  select:disabled { opacity: .6; }
  @media (max-width: 900px) {
    .header-inner { padding: 18px 24px 14px; grid-template-columns: auto minmax(0, 1fr); gap: 16px; }
    nav { grid-column: 1 / -1; grid-row: 2; justify-content: flex-start; gap: 10px 18px; }
    .header-cta { justify-self: end; }
    .preferences { grid-column: 1 / -1; justify-content: flex-start; }
  }
  @media (max-width: 500px) {
    .header-inner { padding: 16px 18px 14px; gap: 14px; }
    .brand { font-size: 22px; gap: 9px; }
    .brand-symbol { width: 34px; height: 34px; font-size: 20px; }
    nav { gap: 9px 16px; }
    nav a { font-size: 12px; }
    .preferences { gap: 9px 14px; }
    .preference-field { gap: 6px; }
    .header-cta { padding: 9px 11px; }
  }
</style>
