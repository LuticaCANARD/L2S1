import { WebgpuMessageError, type WebgpuLocale } from '../i18n/webgpu';
import type { AnalysisResult, Decision } from '$lib/demo/types';
import { candidates, type BrowserRequest } from './contract';

// Separate normalizers preserve relative scores even when full-vocabulary mass underflows.
function logSumExp(values: ArrayLike<number>, locale: WebgpuLocale): number {
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new WebgpuMessageError('errorFiniteLogits', {}, locale);
    max = Math.max(max, values[i]);
  }
  if (!values.length) throw new WebgpuMessageError('errorEmptyLogits', {}, locale);
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += Math.exp(values[i] - max);
  return max + Math.log(sum);
}
export function scoreDecision(decision: Decision, logits: ArrayLike<number>, tokenIds: number[], policy: BrowserRequest['policy'], failureReasons: Record<string, string> = {}, locale: WebgpuLocale = 'ko'): AnalysisResult {
  if (![policy.min_top_probability, policy.min_candidate_mass].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) throw new WebgpuMessageError('errorScorePolicy', {}, locale);
  const options = candidates(decision);
  if (options.length < 2) throw new WebgpuMessageError('errorMinCandidates', {}, locale);
  if (options.length !== tokenIds.length || new Set(tokenIds).size !== tokenIds.length || tokenIds.some((id) => !Number.isInteger(id) || id < 0 || id >= logits.length)) throw new WebgpuMessageError('errorCodeTokens', {}, locale);
  const selectedLogits = tokenIds.map((id) => logits[id]);
  const candidateLse = logSumExp(selectedLogits, locale);
  const fullLse = logSumExp(logits, locale);
  const probabilities = selectedLogits.map((logit) => Math.exp(logit - candidateLse));
  const mass = Math.min(1, Math.max(0, Math.exp(candidateLse - fullLse)));
  let best = 0;
  for (let i = 1; i < probabilities.length; i++) if (probabilities[i] > probabilities[best]) best = i;
  const top = probabilities[best];
  const reasons: string[] = [];
  if (mass < policy.min_candidate_mass) reasons.push('low_candidate_mass');
  if (top < policy.min_top_probability) reasons.push('low_top_probability');
  if (probabilities.filter((value) => Math.abs(value - top) < 1e-12).length > 1) reasons.push('tied_candidates');
  const accepted = reasons.length === 0;
  const value = decision.kind.type === 'binary' ? { type: 'binary', value: accepted ? best === 1 : null } : { type: decision.kind.type, selected: accepted ? options[best].id : null };
  const estimate = decision.kind.type === 'binary' ? { p_true: probabilities[1] } : decision.kind.type === 'ordinal' ? { expected_value: probabilities.reduce((sum, probability, index) => sum + probability * (options[index].numericValue ?? 0), 0) } : {};
  return { id: decision.id, status: accepted ? 'selected' : 'abstained', value, abstention_reasons: reasons,
    reason_messages: reasons.filter((code) => failureReasons[code]).map((code) => ({ code, message: failureReasons[code], user_defined: true })),
    evidence: { type: 'model_scored', scores: options.map((option, index) => ({ id: option.id, code: option.code, option_probability: probabilities[index], raw_logit: selectedLogits[index], token_id: tokenIds[index] })), candidate_mass: mass, top_option_probability: top, scoring_method: 'browser_onnx_single_token_conditional_softmax_v1', estimate } };
}
