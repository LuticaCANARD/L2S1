import { L2S1 } from '../../typescript/dist/index.js';

let input = '';
for await (const chunk of process.stdin) input += chunk;
const stdio = process.argv[2] === '--stdio';
const engine = stdio ? await L2S1.load({ binaryPath: process.argv[3], model: 'fixture' })
  : L2S1.connect({ baseUrl: process.argv[2] });
try { process.stdout.write(JSON.stringify(stdio ? await engine.decideBatch(JSON.parse(input)) : await engine.decide(JSON.parse(input)))); }
finally { await engine.close(); }
