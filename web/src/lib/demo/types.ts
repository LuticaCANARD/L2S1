import { translate, type Locale } from '$lib/i18n';
import { messages } from '$lib/i18n/demo';
export class DemoError extends Error {
  constructor(public key: keyof typeof messages.en, public params?: Record<string, string | number>, language: Locale = 'ko') { super(translate(language, messages, key, params)); }
}
export type Decision = {
  id: string;
  instruction: string;
  kind: {
    type: 'choice' | 'binary' | 'ordinal';
    options?: { id: string; criterion: string }[];
    levels?: { id: string; criterion: string; value: number }[];
    false_label?: string;
    true_label?: string;
  };
  media_ids?: string[];
};
export type AnalysisRequest = { state: unknown; decisions: Decision[]; reasoning?: { mode: 'direct' | 'thinking'; max_tokens: number }; policy?: { min_top_probability: number; min_candidate_mass: number }; failure_reasons?: Record<string, string>; target_error_rate?: number };
export type AnalysisResult = {
  id: string;
  status: 'selected' | 'abstained';
  value: { type: string; value?: boolean | null; selected?: string | null };
  abstention_reasons: string[];
  reason_messages?: { code: string; message: string; user_defined: boolean }[];
  evidence: {
    type: string;
    scores?: { id: string; code: string; option_probability: number; raw_logit?: number; token_id?: number }[];
    candidate_mass?: number;
    top_option_probability?: number;
    scoring_method?: string;
    estimate?: { p_true?: number; expected_value?: number };
  };
  usage?: { input_tokens: number; scoring_input_tokens?: number; reasoning?: { mode: string; generated_tokens: number; completed: boolean } };
};
export type AnalysisResponse = {
  backend: { runtime: string; model: string };
  policy: { min_top_probability: number; min_candidate_mass: number } | null;
  results: AnalysisResult[];
  reasoning?: { mode: string; max_tokens: number };
  error_budget?: { requested_rate: number; guaranteed: boolean; interpretation: string; min_top_probability: number };
};
export type Recording = {
  title: string;
  image_url: string;
  image_alt: string;
  recorded_at: string;
  model: string;
  runtime: string;
  source?: { dataset: string; url: string; license?: string };
  request: AnalysisRequest;
  response: AnalysisResponse;
  elapsed_ms?: number;
  verification?: { source_label: string; model_material_prediction: string; material_correct: boolean; sample_selection: string; scope: string };
};
export const MAX_IMAGE_BYTES = 8 * 1024 * 1024;
export const MAX_REQUEST_BYTES = 44 * 1024 * 1024;
export const MAX_DECISIONS = 128;
export function validateRequest(stateText: string, decisionsText: string, language: Locale = 'ko'): AnalysisRequest {
  const state: unknown = JSON.parse(stateText);
  const decisions: unknown = JSON.parse(decisionsText);
  if (!Array.isArray(decisions) || !decisions.length || decisions.length > MAX_DECISIONS) {
    throw new DemoError('questionsCount', { max: MAX_DECISIONS }, language);
  }
  const ids = new Set<string>();
  for (const decision of decisions) {
    if (!decision || typeof decision.id !== 'string' || !decision.id.trim() || ids.has(decision.id)
      || typeof decision.instruction !== 'string' || !decision.instruction.trim()
      || !['choice', 'binary', 'ordinal'].includes(decision.kind?.type)) {
      throw new DemoError('questionInvalid', undefined, language);
    }
    ids.add(decision.id);
  }
  return { state, decisions: decisions.map((decision) => ({ ...decision, media_ids: ['photo'] })) };
}
export function percent(value: number | undefined, language: Locale = 'ko'): string {
  if (typeof value === 'number' && value > 0 && value < 0.00005) return '<0.01%';
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : translate(language, messages, 'notProvided');
}
export function selection(result: AnalysisResult, language: Locale = 'ko'): string {
  if (result.status === 'abstained') return translate(language, messages, 'selectionAbstained');
  if (result.value.type === 'binary') return result.value.value === true ? translate(language, messages, 'yes') : result.value.value === false ? translate(language, messages, 'no') : 'null';
  return result.value.selected ?? 'null';
}
export function reasonLabel(reason: string, language: Locale = 'ko'): string {
  return ['low_top_probability', 'low_candidate_mass', 'tied_candidates'].includes(reason)
    ? translate(language, messages, reason as 'low_top_probability' | 'low_candidate_mass' | 'tied_candidates') : reason;
}
