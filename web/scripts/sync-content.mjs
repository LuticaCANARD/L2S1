import { mkdir, copyFile, readdir, readFile, writeFile } from 'node:fs/promises';
const root = new URL('../../', import.meta.url);
const target = new URL('../static/docs/', import.meta.url);
await mkdir(target, { recursive: true });
for (const name of ['README.md', 'BENCHMARK.md', 'SEMIF_ALGORITHM.md', 'VERIFICATION.md', 'LICENSING.md', 'LICENSE', 'THIRD_PARTY_LICENSES.txt']) {
  await copyFile(new URL(name, root), new URL(name, target));
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
