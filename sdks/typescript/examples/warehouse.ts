import { L2S1, type DecisionRequest } from '@l2s1/node';

const request: DecisionRequest = {
  state: { storage_requirement: 'chilled' },
  decisions: [{
    id: 'storage_zone', instruction: 'Select the storage zone matching storage_requirement.',
    kind: { type: 'choice', options: [
      { id: 'ambient', criterion: 'Ambient storage is required.' },
      { id: 'chilled', criterion: 'Chilled storage is required.' },
      { id: 'frozen', criterion: 'Frozen storage is required.' },
    ] },
  }],
};
const model = process.argv[2];
if (!model) throw new Error('Usage: node examples/warehouse.ts /path/to/model.gguf');
const engine = await L2S1.load({ model, ...(process.env.L2S1_BINARY ? { binaryPath: process.env.L2S1_BINARY } : {}) });
try { console.log(JSON.stringify(await engine.decide(request), null, 2)); }
finally { await engine.close(); }
