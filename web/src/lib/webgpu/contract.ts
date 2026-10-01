import { WebgpuMessageError, type WebgpuLocale, type WebgpuMessageKey, type MessageParams } from '../i18n/webgpu';
import type { AnalysisResult, Decision } from '$lib/demo/types';
export const MODEL_ID = 'onnx-community/Qwen3-0.6B-ONNX';
export const MODEL_REVISION = 'da1453100cf3ff33ef56d17983fc7a8648706db6';
export const RUNTIME_VERSION = '4.3.0';
export const CACHE_KEY = 'l2s1-webgpu-qwen3-da145310';
export const MAX_INPUT_TOKENS = 1024;
export type Dtype = 'q4f16' | 'q4';
export type AdapterInfo = { description: string; vendor: string; architecture: string; shaderF16: boolean; software: boolean; hardwareVerified: false };
export type BrowserRequest = {
  state: unknown; decisions: Decision[]; reasoning: { mode: 'direct' | 'thinking'; max_tokens: number };
  policy: { min_top_probability: number; min_candidate_mass: number };
  failure_reasons: Record<string, string>;
};
export type BrowserResponse = {
  backend: { runtime: string; model: string; revision: string; dtype: Dtype; device: 'webgpu'; adapter: AdapterInfo; wasm_fallback: false };
  policy: BrowserRequest['policy']; reasoning: BrowserRequest['reasoning']; results: AnalysisResult[]; elapsed_ms: number;
};
export type WorkerInput = ({ type: 'load'; dtype: Dtype } | { type: 'analyze'; request: BrowserRequest } | { type: 'release' }) & { locale?: WebgpuLocale };
export type WorkerOutput =
  | { type: 'progress'; file: string; progress: number; loaded?: number; total?: number }
  | { type: 'status'; message: string; message_key?: WebgpuMessageKey; message_params?: MessageParams }
  | { type: 'ready'; adapter: AdapterInfo; dtype: Dtype }
  | { type: 'result'; response: BrowserResponse }
  | { type: 'error'; code: string; message: string; user_reason?: string; message_key?: WebgpuMessageKey; message_params?: MessageParams }
  | { type: 'released' };
export type Candidate = { id: string; code: string; criterion: string; numericValue?: number };
export function candidates(decision: Decision): Candidate[] {
  if (decision.kind.type === 'binary') return [{ id: 'false', code: 'A', criterion: decision.kind.false_label ?? '' }, { id: 'true', code: 'B', criterion: decision.kind.true_label ?? '' }];
  const items = decision.kind.type === 'choice' ? decision.kind.options : decision.kind.levels;
  return (items ?? []).map((item, index) => ({ id: item.id, code: String.fromCharCode(65 + index), criterion: item.criterion, numericValue: 'value' in item && typeof item.value === 'number' ? item.value : undefined }));
}
export function validateBrowserRequest(request: BrowserRequest, locale: WebgpuLocale = 'ko'): void {
  if (!request || !Array.isArray(request.decisions) || request.decisions.length < 1 || request.decisions.length > 8) throw new WebgpuMessageError('validateQuestions', {}, locale);
  if (new TextEncoder().encode(JSON.stringify(request.state)).byteLength > 16_384) throw new WebgpuMessageError('validateState', {}, locale);
  if (!['direct', 'thinking'].includes(request.reasoning.mode) || !Number.isInteger(request.reasoning.max_tokens) || request.reasoning.max_tokens < 1 || request.reasoning.max_tokens > 256) throw new WebgpuMessageError('validateReasoning', {}, locale);
  if (![request.policy.min_top_probability, request.policy.min_candidate_mass].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) throw new WebgpuMessageError('validatePolicy', {}, locale);
  if (!request.failure_reasons || Array.isArray(request.failure_reasons) || typeof request.failure_reasons !== 'object' || Object.values(request.failure_reasons).some((value) => typeof value !== 'string' || value.length > 512)) throw new WebgpuMessageError('validateFailures', {}, locale);
  const decisionIds = new Set<string>();
  for (const decision of request.decisions) {
    if (typeof decision.id !== 'string' || !decision.id.trim() || decisionIds.has(decision.id) || typeof decision.instruction !== 'string' || !decision.instruction.trim() || !['binary', 'choice', 'ordinal'].includes(decision.kind?.type)) throw new WebgpuMessageError('validateDecision', {}, locale);
    decisionIds.add(decision.id);
    const options = candidates(decision);
    if (options.length < 2 || options.length > 26) throw new WebgpuMessageError('validateCandidates', {}, locale);
    const ids = new Set<string>();
    for (const option of options) {
      if (typeof option.id !== 'string' || !option.id.trim() || ids.has(option.id) || typeof option.criterion !== 'string' || !option.criterion.trim()) throw new WebgpuMessageError('validateCandidate', {}, locale);
      if (decision.kind.type === 'ordinal' && !Number.isFinite(option.numericValue)) throw new WebgpuMessageError('validateOrdinal', {}, locale);
      ids.add(option.id);
    }
  }
}
