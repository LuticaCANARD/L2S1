import type { AnalysisResponse } from '../demo/types';
import type { BrowserRequest } from './contract';

export const LOCAL_ENDPOINT = '/quality-inference';
export type LocalBackend = { model: string; runtime: string; reasoning_modes: string[]; request_policy_supported: boolean };
export type LocalResponse = AnalysisResponse & { elapsed_ms: number };
const LOCAL_FAILURE_CODES = new Set(['low_top_probability', 'low_candidate_mass', 'tied_candidates', 'reasoning_limit', 'native_failure']);

async function readResponse(response: Response) {
  const data = await response.json();
  if (!response.ok) {
    const error = data.error;
    throw new Error(typeof error === 'string' ? error : `${error?.code ?? response.status}: ${error?.message ?? response.statusText}${error?.user_reason ? ` · ${error.user_reason}` : ''}`);
  }
  return data;
}

export async function connectLocal(signal?: AbortSignal): Promise<LocalBackend> {
  const data = await readResponse(await fetch(`${LOCAL_ENDPOINT}/v1/capabilities`, { signal }));
  if (data.api_version !== 1 || typeof data.backend?.model !== 'string' || typeof data.backend?.runtime !== 'string' || data.evidence !== 'model_scored') throw new Error('Invalid local model capabilities');
  return { model: data.backend.model, runtime: data.backend.runtime, reasoning_modes: data.reasoning?.modes ?? ['direct'], request_policy_supported: data.request_policy?.supported === true };
}

export async function analyzeLocal(request: BrowserRequest, signal?: AbortSignal, requestPolicySupported = false): Promise<LocalResponse> {
  const start = performance.now();
  // Browser-only thinking errors are not accepted by the version 1 HTTP contract.
  const failure_reasons = Object.fromEntries(Object.entries(request.failure_reasons).filter(([code, message]) => LOCAL_FAILURE_CODES.has(code) && message.trim()));
  if (!requestPolicySupported && request.reasoning.mode !== 'direct') throw new Error('Local server does not support request reasoning');
  const payload = requestPolicySupported ? { ...request, failure_reasons } : { state: request.state, decisions: request.decisions };
  const data = await readResponse(await fetch(`${LOCAL_ENDPOINT}/v1/decisions`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload), signal }));
  if (data.policy?.min_top_probability !== request.policy.min_top_probability || data.policy?.min_candidate_mass !== request.policy.min_candidate_mass) throw new Error('Local server did not preserve the requested acceptance policy');
  if (typeof data.backend?.model !== 'string' || typeof data.backend?.runtime !== 'string' || !Array.isArray(data.results) || data.results.length !== request.decisions.length || data.results.some((result: AnalysisResponse['results'][number], index: number) => result.id !== request.decisions[index].id || !['selected', 'abstained'].includes(result.status) || !result.value || result.evidence?.type !== 'model_scored' || !Array.isArray(result.abstention_reasons))) throw new Error('Invalid local decision response');
  if (!requestPolicySupported) {
    for (const result of data.results) {
      result.reason_messages = result.abstention_reasons.filter((code: string) => failure_reasons[code]).map((code: string) => ({ code, message: failure_reasons[code], user_defined: true }));
    }
  }
  return { ...data, elapsed_ms: performance.now() - start };
}
