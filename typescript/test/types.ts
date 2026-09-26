import { L2S1, L2S1Client, type DecisionBackend, type DecisionRequest, type DecisionResponse } from '../src/index.js';
const request: DecisionRequest = { state: { x: 1 }, decisions: [
  { id: 'q', instruction: 'Is x positive?', kind: { type: 'binary', false_label: 'No', true_label: 'Yes' } },
] };
const http = new L2S1Client({ baseUrl: 'http://127.0.0.1:8080' });
const response: Promise<DecisionResponse> = http.decide(request);
void response;
async function managed() {
  await using engine = await L2S1.load({ model: 'model.gguf', device: 'cpu' });
  const result = (await engine.decide(request)).results[0]!;
  if (result.value.type === 'binary') { const value: boolean | null = result.value.value; void value; }
  if (result.evidence.type === 'model_scored') { const mass: number = result.evidence.candidate_mass; void mass; }
  else {
    // @ts-expect-error Provider selections have no candidate probabilities.
    void result.evidence.candidate_mass;
  }
}
void managed;
const backend: DecisionBackend = http;
const custom = L2S1.fromBackend(backend);
const remote = L2S1.connect({ baseUrl: 'https://example.com/l2s1' });
void custom.decide(request);
void remote.decide(request);
// @ts-expect-error Unknown decision kinds are rejected by TypeScript.
const invalid: DecisionRequest = { state: null, decisions: [{ id: 'q', instruction: 'q', kind: { type: 'free_text' } }] };
void invalid;
