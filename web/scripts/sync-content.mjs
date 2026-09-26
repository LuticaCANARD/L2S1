import { mkdir, copyFile, readdir, readFile, writeFile, rm } from 'node:fs/promises';
const root = new URL('../../', import.meta.url);
const target = new URL('../static/docs/', import.meta.url);
// This directory is generated; remove obsolete paths after documentation moves.
await rm(target, { recursive: true, force: true });
await mkdir(target, { recursive: true });
for (const name of [
  'README.md', 'README.ko.md', 'LICENSE', 'THIRD_PARTY_LICENSES.txt',
]) {
  await copyFile(new URL(name, root), new URL(name, target));
}

// Mirror documentation paths so README language links and guide links resolve.
const documents = new URL('docs/', root);
const documentTarget = new URL('docs/', target);
await mkdir(documentTarget, { recursive: true });
for (const entry of await readdir(documents, { withFileTypes: true })) {
  if (!entry.isFile() || !entry.name.endsWith('.md')) continue;
  await copyFile(new URL(entry.name, documents), new URL(entry.name, documentTarget));
}

// Publish reports without copying raw observations, local results, or model files.
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
