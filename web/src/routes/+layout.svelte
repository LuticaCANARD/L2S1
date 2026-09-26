<script lang="ts">
  import { onMount } from 'svelte';
  import { afterNavigate, replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import { locale, theme, isLocale, isTheme, preferenceKeys, type Locale, type Theme } from '$lib/i18n';
  import SiteHeader from '$lib/components/SiteHeader.svelte';
  import '../theme.css';
  let { children } = $props();
  let ready = $state(false);
  function readPreference(key: string): string | null { try { return localStorage.getItem(key); } catch { return null; } }
  function writePreference(key: string, value: string) { try { localStorage.setItem(key, value); } catch { /* Preferences still work without storage. */ } }
  function applyLocale(value: Locale) { locale.set(value); document.documentElement.lang = value; }
  function applyTheme(value: Theme) {
    theme.set(value);
    const resolved = value === 'system' ? window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light' : value;
    document.documentElement.dataset.theme = resolved;
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', resolved === 'dark' ? '#05070a' : '#e7eef4');
  }
  function chooseLocale(event: Event) {
    const value = (event.currentTarget as HTMLSelectElement).value;
    if (!isLocale(value)) return;
    applyLocale(value); writePreference(preferenceKeys.locale, value);
    const url = new URL(window.location.href); url.searchParams.set('lang', value);
    // The current absolute URL already includes the configured base path.
    // eslint-disable-next-line svelte/no-navigation-without-resolve
    replaceState(url, page.state);
  }
  function chooseTheme(event: Event) {
    const value = (event.currentTarget as HTMLSelectElement).value;
    if (!isTheme(value)) return;
    applyTheme(value); writePreference(preferenceKeys.theme, value);
  }
  afterNavigate(() => {
    const requested = page.url.searchParams.get('lang');
    if (isLocale(requested)) { applyLocale(requested); writePreference(preferenceKeys.locale, requested); }
  });
  onMount(() => {
    const query = new URL(window.location.href).searchParams.get('lang');
    const stored = readPreference(preferenceKeys.locale);
    const browser = navigator.languages.map((language) => language.split('-')[0]).find(isLocale);
    applyLocale(isLocale(query) ? query : isLocale(stored) ? stored : browser ?? 'en');
    if (isLocale(query)) writePreference(preferenceKeys.locale, query);
    const storedTheme = readPreference(preferenceKeys.theme); applyTheme(isTheme(storedTheme) ? storedTheme : 'system');
    ready = true;
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const update = () => { if ($theme === 'system') applyTheme('system'); };
    const popstate = () => { const next = new URL(window.location.href).searchParams.get('lang'); if (isLocale(next)) applyLocale(next); };
    media.addEventListener('change', update); window.addEventListener('popstate', popstate);
    return () => { media.removeEventListener('change', update); window.removeEventListener('popstate', popstate); };
  });
</script>
<SiteHeader {ready} onlocalechange={chooseLocale} onthemechange={chooseTheme} />
{@render children()}
