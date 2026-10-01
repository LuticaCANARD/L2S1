import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { test } from 'node:test';
import { StdioClient } from '../dist/stdio.js';

// Controlled RPC peer verifies cancellation correlation, not native inference.
test('stdio timeout discards the late reply without retrying, then close rejects pending work', async () => {
  const code = `
    const readline = require('node:readline');
    readline.createInterface({ input: process.stdin }).on('line', (line) => {
      const call = JSON.parse(line);
      setTimeout(() => process.stdout.write(JSON.stringify({ id: call.id, result: { status: 'ok' } }) + '\\n'), 50);
    });
  `;
  const child = spawn(process.execPath, ['-e', code], { stdio: ['pipe', 'pipe', 'pipe'] });
  const exit = once(child, 'close');
  const client = new StdioClient(child, 5000);
  try {
    await assert.rejects(client.health({ timeoutMs: 5 }), { name: 'TimeoutError' });
    await client.health();
    const pending = client.health();
    const rejection = assert.rejects(pending, { code: 'backend_closed' });
    client.close();
    await rejection;
    await assert.rejects(client.health(), { code: 'backend_closed' });
  } finally { client.close(); child.kill(); await exit; }
});
