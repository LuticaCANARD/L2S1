import { L2S1Client } from '@l2s1/node/http';

const client = new L2S1Client({ baseUrl: 'http://127.0.0.1:8080' });
const response = await client.decide({
  state: { temperature_c: 6 },
  decisions: [{ id: 'cold', instruction: 'Is temperature_c below 10?', kind: { type: 'binary', false_label: 'Temperature is at least 10.', true_label: 'Temperature is below 10.' } }],
});
for (const result of response.results) {
  console.log(result.id, result.value, result.status);
  if (result.evidence.type === 'model_scored') console.log(result.evidence.candidate_mass);
}
