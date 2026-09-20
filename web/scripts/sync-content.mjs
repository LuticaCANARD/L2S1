import { mkdir, copyFile } from 'node:fs/promises';
const root = new URL('../../', import.meta.url);
const target = new URL('../static/docs/', import.meta.url);
await mkdir(target, { recursive: true });
for (const name of ['README.md', 'VERIFICATION.md', 'LICENSING.md', 'LICENSE', 'THIRD_PARTY_LICENSES.txt']) {
  await copyFile(new URL(name, root), new URL(name, target));
}
await copyFile(new URL('examples/warehouse.json', root), new URL('warehouse.json', target));
