import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { decisionResponse } from '../dist/http.js';
test('real Laya evidence preserves scores without invented mass or token IDs', async () => {
 const raw=JSON.parse(await readFile(new URL('../../fixtures/discriminative_response.json',import.meta.url),'utf8'));
 const request={state:'fixture',decisions:raw.results.map(r=>({id:r.id,kind:{type:r.value.type}}))};
 const parsed=decisionResponse(raw,request);
 for(const r of parsed.results){assert.equal(r.evidence.type,'discriminative');assert.equal('candidate_mass' in r.evidence,false);assert.equal('token_id' in r.evidence.scores[0],false);assert.equal(r.evidence.schema_sha256.length,64);}
});
