import { mkdir, copyFile, readdir, readFile, writeFile, rm } from 'node:fs/promises';
const root = new URL('../../', import.meta.url);
const target = new URL('../static/docs/', import.meta.url);
const publicStudy = 'typed-decisions-20260926';
// This directory is generated; remove obsolete paths after documentation moves.
await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });
for (const name of [
  'README.md', 'README.ko.md', 'README.ja.md', 'LICENSE', 'THIRD_PARTY_LICENSES.txt',
]) {
  await copyFile(new URL(name, root), new URL(name, target));
}

// Mirror documentation paths so README language links and guide links resolve.
const documents = new URL('docs/', root);
const documentTarget = new URL('docs/', target);
await mkdir(documentTarget, { recursive: true });
await copyFile(new URL('translations.json', documents), new URL('translations.json', documentTarget));
function publicEvidenceLinks(markdown) {
  for (const study of [publicStudy, 'rtx3060-20260926']) {
    markdown = markdown.replace(new RegExp(`(?:\\.\\./)+benchmarks/${study}/([^\\s)]+\\.jsonl?)`, 'g'), `/benchmarks/${study}/$1`);
  }
  return markdown;
}
for (const entry of await readdir(documents, { withFileTypes: true })) {
  if (!entry.isFile() || !entry.name.endsWith('.md')) continue;
  const markdown = await readFile(new URL(entry.name, documents), 'utf8');
  // Repository links stay relative; deployed links point to explicit public files.
  const publicMarkdown = publicEvidenceLinks(markdown);
  await writeFile(new URL(entry.name, documentTarget), publicMarkdown);
}

// Publish only the declared language directories, not arbitrary local studies.
async function mirrorLocalizedMarkdown(source, destination) {
  await mkdir(destination, { recursive: true });
  for (const entry of await readdir(source, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      await mirrorLocalizedMarkdown(new URL(`${entry.name}/`, source), new URL(`${entry.name}/`, destination));
    } else if (entry.isFile() && entry.name.endsWith('.md')) {
      const markdown = await readFile(new URL(entry.name, source), 'utf8');
      await writeFile(new URL(entry.name, destination), publicEvidenceLinks(markdown));
    }
  }
}
for (const language of ['en', 'ko', 'ja']) {
  await mirrorLocalizedMarkdown(new URL(`${language}/`, documents), new URL(`${language}/`, documentTarget));
}

// The language indexes also link to public package and website documentation.
for (const path of [
  'typescript/README.md', 'python/README.md', 'crates/l2s1-llama-sys/README.md',
  'crates/l2s1-tools/README.md', 'crates/l2s1-tools/NOTICE.md',
  'typescript/PUBLISHING.md', 'web/README.md',
  'skills/l2s1/SKILL.md', 'skills/l2s1/references/decisions.md',
  'skills/l2s1/references/interfaces.md',
]) {
  const destination = new URL(path, target);
  await mkdir(new URL('./', destination), { recursive: true });
  await copyFile(new URL(path, root), destination);
}

// Publish reports; observations are excluded except for the explicit scored-record allowlist below.
const benchmarks = new URL('benchmarks/', root);
for (const entry of await readdir(benchmarks, { withFileTypes: true })) {
  if (!entry.isDirectory()) continue;
  const source = new URL(`${entry.name}/`, benchmarks);
  const destination = new URL(`benchmarks/${entry.name}/`, target);
  for (const report of await readdir(source, { withFileTypes: true })) {
    if (!report.isFile() || !/^(README|REPORT)\.md$/.test(report.name)) continue;
    await mkdir(destination, { recursive: true });
    await copyFile(new URL(report.name, source), new URL(report.name, destination));
  }
}
await copyFile(new URL('examples/warehouse.json', root), new URL('warehouse.json', target));

// Public example and attribution referenced by the translated image guide.
for (const path of ['examples/image-analysis.json', 'web/static/demo/THIRD_PARTY_NOTICE.txt']) {
  const destination = new URL(path, target);
  await mkdir(new URL('./', destination), { recursive: true });
  await copyFile(new URL(path, root), destination);
}

const examples = new URL('examples/', root);
const exampleTarget = new URL('examples/', target);
await mkdir(exampleTarget, { recursive: true });
for (const name of await readdir(examples)) {
  if (!/^warehouse(?:\.[a-z0-9]+\.(?:cpu|cuda)\.output)?\.json$/.test(name)) continue;
  const data = JSON.parse(await readFile(new URL(name, examples), 'utf8'));
  // Publish portable model identifiers, not local filesystem paths.
  if (data.backend?.model_path) {
    data.backend.model_path = `models/${data.backend.model_path.split(/[\\/]/).at(-1)}`;
  }
  await writeFile(new URL(name, exampleTarget), JSON.stringify(data, null, 2) + '\n');
}

await copyFile(new URL('../THIRD_PARTY_LICENSES.txt', import.meta.url), new URL('WEB_THIRD_PARTY_LICENSES.txt', target));

// Deliberate public evidence export: never scan results/, models/, or arbitrary JSONL.
const publicEvidence = new URL(`../static/benchmarks/${publicStudy}/`, import.meta.url);
await rm(publicEvidence, { recursive: true, force: true });
await mkdir(publicEvidence, { recursive: true });
for (const name of [
  'manifest.json',
  'gemma4-e2b-summary.json', 'qwen3-06b-summary.json',
  'gemma4-e2b-scored.jsonl', 'qwen3-06b-scored.jsonl',
]) {
  const source = new URL(`benchmarks/${publicStudy}/${name}`, root);
  const destination = new URL(name, publicEvidence);
  if (name === 'manifest.json') {
    const manifest = JSON.parse(await readFile(source, 'utf8'));
    manifest.libraries_original_path = manifest.libraries_original_path.split(/[\\/]/).at(-1);
    manifest.runs = {};
    for (const model of ['gemma4-e2b', 'qwen3-06b']) {
      const run = JSON.parse(await readFile(new URL(`benchmarks/${publicStudy}/${model}-run.json`, root), 'utf8'));
      run.command = run.command.map((arg) => /^[\\/]/.test(arg) ? arg.split(/[\\/]/).at(-1) : arg);
      manifest.runs[model] = run;
    }
    manifest.public_export = { path_normalization: 'Absolute native-library and command paths are reduced to basenames. Runs are added from checked-in model run metadata. Measurements, revisions, command flags and hashes are unchanged.' };
    await writeFile(destination, JSON.stringify(manifest, null, 2) + '\n');
  } else {
    await copyFile(source, destination);
  }
}

// Fixed aggregate exports: source records and local environments stay outside this allowlist.
const rtxStudy = 'rtx3060-20260926';
const rtxTarget = new URL(`../static/benchmarks/${rtxStudy}/`, import.meta.url);
await rm(rtxTarget, { recursive: true, force: true });
await mkdir(rtxTarget, { recursive: true });
for (const name of ['summary.json', 'manifest.json', 'README.md']) {
  await copyFile(new URL(`benchmarks/${rtxStudy}/${name}`, root), new URL(name, rtxTarget));
}
const sharedTarget = new URL('../static/benchmarks/shared-state-cache-20260925/', import.meta.url);
await mkdir(sharedTarget, { recursive: true });
await copyFile(new URL('../src/lib/shared-state-highlight.json', import.meta.url), new URL('highlight.json', sharedTarget));
