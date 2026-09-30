<script lang="ts">
  import { locale, translate } from '$lib/i18n';
  import { installMessages } from '$lib/i18n/install';
  const version = '0.2.1';
  const release = `https://github.com/LuticaCANARD/L2S1/releases/tag/v${version}`;
  const t = (key: keyof typeof installMessages.en) => translate($locale, installMessages, key, { version });
  const languages = ['TypeScript', 'Python', 'Rust'] as const;
  type Language = typeof languages[number];
  let selected = $state<Language>('TypeScript');
  let platform = $state('linux-x64');
  let copied = $state('');
  let copyError = $state(false);
  const packages = {
    TypeScript: { command: `npm install @l2s1/node@${version}`, verify: 'npm ls @l2s1/node', guide: 'typescript/README.md', requirement: 'typescriptRequirement' },
    Python: { command: `python -m pip install l2s1-sdk==${version}`, verify: 'python -c "import l2s1; print(l2s1.__version__)"', guide: 'python/README.md', requirement: 'pythonRequirement' },
    Rust: { command: `cargo add l2s1@${version} --features llama`, verify: 'cargo tree -i l2s1', guide: 'GUIDE.md', requirement: 'rustRequirement' },
  } as const;
  const platforms = [
    { id: 'linux-x64', label: 'Linux · x64 (Intel / AMD)' },
    { id: 'linux-arm64', label: 'Linux · ARM64' },
    { id: 'darwin-arm64', label: 'macOS · Apple Silicon' },
    { id: 'darwin-x64', label: 'macOS · Intel' },
    { id: 'win32-x64', label: 'Windows · x64 (Intel / AMD)' },
  ];
  const current = $derived(packages[selected]);
  const archive = $derived(`l2s1-runtime-${platform}-${version}.tgz`);
  async function copy(command: string, id: string) {
    copied = ''; copyError = false;
    try { await navigator.clipboard.writeText(command); copied = id; }
    catch { copyError = true; }
  }
  function select(language: Language) { selected = language; copied = ''; copyError = false; }
</script>

<section class="install-panel" aria-labelledby="install-heading" data-testid="install-guide">
  <div class="heading"><div><span class="version">v{version}</span><h3 id="install-heading">{t('title')}</h3></div><a href={release} rel="external">{t('release')} ↗</a></div>
  <p class="intro">{t('intro')}</p>
  <div class="languages" role="group" aria-label={t('language')}>
    {#each languages as language (language)}<button type="button" aria-pressed={selected === language} onclick={() => select(language)}>{language}</button>{/each}
  </div>
  <p class="requirement">{t(current.requirement)}</p>
  <!-- eslint-disable-next-line svelte/no-at-html-tags -- Static Cloudflare marker; no user HTML. -->
  {@html '<!--email_off-->'}
  <ol>
    <li><div class="step"><span class="number" aria-hidden="true">1</span><h4>{t('install')}</h4></div>
      <div class="command"><pre><code>{current.command}</code></pre><button type="button" onclick={() => copy(current.command, 'install')} aria-label={`${t('copy')}: ${t('install')}`}>{t(copied === 'install' ? 'copied' : 'copy')}</button></div>
      {#if selected === 'TypeScript'}<p class="hint">{t('runtimeIncluded')}</p>{/if}
    </li>
    <li><div class="step"><span class="number" aria-hidden="true">2</span><h4>{t('verify')}</h4></div>
      <div class="command"><pre><code>{current.verify}</code></pre><button type="button" onclick={() => copy(current.verify, 'verify')} aria-label={`${t('copy')}: ${t('verify')}`}>{t(copied === 'verify' ? 'copied' : 'copy')}</button></div>
      <p class="hint">{t(selected === 'Rust' ? 'rustVerifyNote' : 'verifyNote')}</p>
    </li>
    {#if selected === 'Python'}
      <li><div class="step"><span class="number" aria-hidden="true">3</span><h4>{t('runtimeTitle')}</h4></div>
        <p>{t('runtimeNote')}</p>
        <div class="runtime"><label for="runtime-platform">{t('platform')}<select id="runtime-platform" bind:value={platform} onchange={() => { copied = ''; copyError = false; }}>{#each platforms as item (item.id)}<option value={item.id}>{item.label}</option>{/each}</select></label><a class="download" href={`https://github.com/LuticaCANARD/L2S1/releases/download/v${version}/${archive}`}>{t('download')} ↓</a></div>
        <div class="command"><pre><code>tar -xzf {archive}</code></pre><button type="button" onclick={() => copy(`tar -xzf ${archive}`, 'extract')} aria-label={`${t('copy')}: ${t('extract')}`}>{t(copied === 'extract' ? 'copied' : 'copy')}</button></div>
        <p class="hint">{t('runtimePath')}</p>
      </li>
    {/if}
    <li class="next"><div class="step"><span class="number" aria-hidden="true">{selected === 'Python' ? '4' : '3'}</span><h4>{t('modelTitle')}</h4></div>
      <p>{t('modelNote')}</p><div class="links"><a href="#models">{t('modelLink')} ↓</a><a class="example" href={`/docs/docs/${$locale}/${current.guide}`} rel="external">{t('exampleLink')} ↗</a></div>
    </li>
  </ol>
  <!-- eslint-disable-next-line svelte/no-at-html-tags -- Static Cloudflare marker; no user HTML. -->
  {@html '<!--/email_off-->'}
  <p class="feedback" role="status">{copyError ? t('copyError') : copied ? t('copied') : ''}</p>
  <noscript><p>{t('noJs')}</p>{#each languages as language (language)}<a href={`/docs/docs/${$locale}/${packages[language].guide}`} rel="external">{language} ↗ </a>{/each}</noscript>
</section>

<style>
  .install-panel { grid-column: 1 / -1; min-width: 0; padding: clamp(20px, 4vw, 40px); border: 1px solid var(--theme-border, #d6d9ce); border-radius: 12px; }
  .heading, .heading > div, .step, .links, .runtime { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
  .heading { justify-content: space-between; }
  h3 { margin: 0; font-size: clamp(20px, 3vw, 26px); } h4 { margin: 0; font-size: 16px; }
  p { font-size: 14px; line-height: 1.7; } a { color: inherit; text-underline-offset: 4px; font-size: 13px; }
  .version { font: 12px monospace; padding: 6px 9px; border: 1px solid var(--theme-border, #d6d9ce); border-radius: 20px; }
  .intro { margin: 14px 0 22px; } .languages { display: flex; gap: 8px; flex-wrap: wrap; }
  button { cursor: pointer; font: inherit; } .languages button { padding: 12px 22px; border: 1px solid var(--theme-border, #d6d9ce); background: transparent; color: inherit; border-radius: 7px; font-size: 14px; min-height: 44px; }
  .languages button[aria-pressed='true'], .example { background: var(--theme-terminal, #222e24); color: var(--theme-on-primary, #e0e9d5); }
  button:focus-visible, a:focus-visible, select:focus-visible { outline: 3px solid #7b9c55; outline-offset: 3px; }
  .requirement { margin: 18px 0 26px; } ol { list-style: none; padding: 0; margin: 0; display: grid; gap: 26px; }
  .number { border: 1px solid var(--theme-border, #d6d9ce); border-radius: 50%; width: 28px; height: 28px; display: grid; place-items: center; font: 12px monospace; flex: none; }
  .command { margin-top: 12px; display: flex; align-items: flex-start; gap: 12px; background: var(--theme-terminal, #222e24); color: var(--theme-on-primary, #e0e9d5); padding: 14px; border-radius: 7px; }
  pre { flex: 1; min-width: 0; margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.9 monospace; align-self: center; }
  .command button { border: 1px solid currentColor; border-radius: 5px; padding: 10px 13px; background: transparent; color: inherit; min-height: 44px; font-size: 12px; flex: none; }
  .hint { font-size: 12px; margin: 10px 0 0; } .next { border-top: 1px solid var(--theme-border, #d6d9ce); padding-top: 24px; }
  .example, .download { padding: 12px 16px; border-radius: 6px; text-decoration: none; border: 1px solid var(--theme-border, #d6d9ce); min-height: 44px; box-sizing: border-box; }
  .links { gap: 20px; } .runtime { align-items: end; margin: 16px 0; } label { font-size: 12px; flex: 1; min-width: 0; }
  select { display: block; margin-top: 8px; width: 100%; min-height: 44px; padding: 10px; color: inherit; background: var(--theme-surface, #f6f7f0); border: 1px solid var(--theme-border, #d6d9ce); border-radius: 5px; }
  .feedback { min-height: 24px; margin: 18px 0 0; }
  @media (max-width: 480px) { .languages { display: grid; grid-template-columns: repeat(3, 1fr); } .languages button { padding: 12px 7px; } .runtime { display: block; } .download { display: inline-block; margin-top: 12px; } }
</style>
