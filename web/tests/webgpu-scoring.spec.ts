import { expect, test } from '@playwright/test';
import { scoreDecision } from '../src/lib/webgpu/scoring';
import type { Decision } from '../src/lib/demo/types';

const choice: Decision = { id: 'material', instruction: 'Select the material.', kind: { type: 'choice', options: [{ id: 'glass', criterion: 'Glass' }, { id: 'wood', criterion: 'Wood' }] } };
const permissive = { min_top_probability: 0, min_candidate_mass: 0 };

test('candidate mass includes every vocabulary token while option probabilities condition on the candidates', () => {
  // Full-vocabulary probabilities are 1/10, 2/10, 3/10, 4/10.
  const result = scoreDecision(choice, Float32Array.from([0, Math.log(2), Math.log(3), Math.log(4)]), [1, 3], permissive);
  expect(result.value.selected).toBe('wood');
  expect(result.evidence.candidate_mass).toBeCloseTo(0.6, 7);
  expect(result.evidence.scores?.[0].option_probability).toBeCloseTo(1 / 3, 7);
  expect(result.evidence.scores?.[1].option_probability).toBeCloseTo(2 / 3, 7);
  expect(result.evidence.top_option_probability).toBeCloseTo(2 / 3, 7);
});

test('candidate normalization survives underflow in full-vocabulary mass', () => {
  const result = scoreDecision(choice, Float64Array.from([1000, 0, 2]), [1, 2], permissive);
  expect(result.evidence.candidate_mass).toBe(0);
  expect(result.evidence.scores?.[1].option_probability).toBeCloseTo(1 / (1 + Math.exp(-2)), 14);
  expect(result.evidence.scores?.reduce((sum, item) => sum + item.option_probability, 0)).toBeCloseTo(1, 14);
  expect(result.value.selected).toBe('wood');
});

test('a policy change only changes acceptance and custom messages, preserving scored evidence', () => {
  const logits = [0, Math.log(2), Math.log(3), Math.log(4)];
  const accepted = scoreDecision(choice, logits, [1, 3], permissive);
  const rejected = scoreDecision(choice, logits, [1, 3], { min_top_probability: 0.9, min_candidate_mass: 0.8 }, { low_candidate_mass: 'Candidate support is insufficient.', low_top_probability: 'No option is clear.' });
  expect(rejected.evidence).toEqual(accepted.evidence);
  expect(rejected.status).toBe('abstained');
  expect(rejected.value.selected).toBeNull();
  expect(rejected.abstention_reasons).toEqual(['low_candidate_mass', 'low_top_probability']);
  expect(rejected.reason_messages).toEqual([
    { code: 'low_candidate_mass', message: 'Candidate support is insufficient.', user_defined: true },
    { code: 'low_top_probability', message: 'No option is clear.', user_defined: true }
  ]);
  expect(scoreDecision(choice, logits, [1, 3], permissive)).toEqual(accepted);
});

test('exactly reaching a threshold is accepted, while a tie remains explicit abstention', () => {
  const result = scoreDecision(choice, [Math.log(1), Math.log(3)], [0, 1], permissive);
  const boundary = scoreDecision(choice, [Math.log(1), Math.log(3)], [0, 1], { min_top_probability: result.evidence.top_option_probability!, min_candidate_mass: result.evidence.candidate_mass! });
  expect(boundary.status).toBe('selected');
  const tie = scoreDecision(choice, [0, 0], [0, 1], permissive);
  expect(tie.status).toBe('abstained');
  expect(tie.value.selected).toBeNull();
  expect(tie.abstention_reasons).toEqual(['tied_candidates']);
  expect(tie.evidence.scores?.map((item) => item.option_probability)).toEqual([0.5, 0.5]);
});

test('binary candidates keep false/true order and retain p_true after abstention', () => {
  const binary: Decision = { id: 'cat', instruction: 'Does it name a cat?', kind: { type: 'binary', false_label: 'No cat', true_label: 'Names a cat' } };
  const selected = scoreDecision(binary, [Math.log(1), Math.log(3)], [0, 1], permissive);
  expect(selected.value.value).toBe(true);
  expect(selected.evidence.scores?.map((item) => [item.id, item.code])).toEqual([['false', 'A'], ['true', 'B']]);
  expect(selected.evidence.estimate?.p_true).toBeCloseTo(0.75, 14);
  const abstained = scoreDecision(binary, [Math.log(1), Math.log(3)], [0, 1], { ...permissive, min_top_probability: 0.9 });
  expect(abstained.value.value).toBeNull();
  expect(abstained.evidence).toEqual(selected.evidence);
});

test('ordinal expectation uses numeric level values rather than rank or code', () => {
  const ordinal: Decision = { id: 'risk', instruction: 'Assess risk.', kind: { type: 'ordinal', levels: [{ id: 'low', criterion: 'Low', value: -2 }, { id: 'high', criterion: 'High', value: 10 }] } };
  const result = scoreDecision(ordinal, [Math.log(1), Math.log(3)], [0, 1], permissive);
  expect(result.value.selected).toBe('high');
  expect(result.evidence.estimate?.expected_value).toBeCloseTo(7, 14);
});

test('duplicate, missing, out-of-vocabulary codes and nonfinite logits fail before selection', () => {
  for (const tokens of [[0], [0, 0], [-1, 1], [0, 2], [0, 0.5]]) {
    expect(() => scoreDecision(choice, [0, 1], tokens, permissive)).toThrow();
  }
  for (const logits of [[NaN, 1], [0, Infinity], [0, -Infinity], [0, 1, NaN]]) {
    expect(() => scoreDecision(choice, logits, [0, 1], permissive)).toThrow();
  }
});
