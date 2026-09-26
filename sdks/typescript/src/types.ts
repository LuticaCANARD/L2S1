/** JSON field names deliberately match the Rust HTTP v1 contract. */
export type JsonValue = null | boolean | number | string | JsonValue[] | { [key: string]: JsonValue };
export interface OptionSpec { id: string; criterion: string }
export interface Level extends OptionSpec { value: number }
export type DecisionKind =
  | { type: 'binary'; false_label: string; true_label: string }
  | { type: 'choice'; options: OptionSpec[] }
  | { type: 'ordinal'; levels: Level[] };
export interface Decision {
  id: string;
  instruction: string;
  kind: DecisionKind;
  /** Omitted uses all request media; [] means text only. */
  media_ids?: string[];
}
export interface DecisionPolicy { min_top_probability: number; min_candidate_mass: number }
export interface ReasoningOptions { mode?: 'direct' | 'thinking'; max_tokens?: number }
export type FailureReasonCode = 'low_candidate_mass' | 'low_top_probability' | 'tied_candidates' | 'reasoning_limit' | 'native_failure';
export interface DecisionRequest {
  state: JsonValue;
  decisions: Decision[];
  media?: { type: 'image'; id: string; data_base64: string }[];
  policy?: DecisionPolicy;
  reasoning?: ReasoningOptions;
  /** A model score threshold, not a guaranteed error rate. */
  target_error_rate?: number;
  failure_reasons?: Partial<Record<FailureReasonCode, string>>;
}
export type DecisionValue =
  | { type: 'binary'; value: boolean | null }
  | { type: 'choice'; selected: string | null }
  | { type: 'ordinal'; selected: string | null; level_value?: number | null };
export interface OptionScore {
  id: string; code: string; token_id: number; token_ids?: number[];
  raw_logit: number; option_probability: number;
}
export interface ModelScoredEvidence {
  type: 'model_scored'; scores: OptionScore[];
  candidate_mass: number; top_option_probability: number; entropy_confidence: number;
  scoring_method: string; calibration_id: string | null; truncated: boolean;
  code_prefix_evaluations: number; code_evaluated_tokens: number;
  estimate: { p_true?: number; expected_value?: number };
}
export interface SelectionOnlyEvidence {
  type: 'selection_only'; selected_code: string | null; provider_model: string | null;
}
export interface DecisionResult {
  id: string; value: DecisionValue; status: 'selected' | 'abstained';
  abstention_reasons: string[];
  reason_messages?: { code: string; message: string; user_defined: true }[];
  evidence: ModelScoredEvidence | SelectionOnlyEvidence;
  usage: {
    input_tokens: number | null; reused_prefix_tokens?: number; output_tokens?: number | null;
    reasoning?: { mode: 'direct' | 'thinking'; generated_tokens: number; completed: boolean };
  };
}
export interface DecisionResponse {
  api_version: 1; request_id: string;
  backend: { runtime: string; model: string; details: JsonValue };
  policy: DecisionPolicy | null; results: DecisionResult[];
  reasoning?: ReasoningOptions;
  error_budget?: { requested_rate: number; guaranteed: false; interpretation: 'model_score_threshold'; min_top_probability: number };
}
export interface Capabilities {
  api_version: 1; backend: { runtime: string; model: string };
  decision_types: ('binary' | 'choice' | 'ordinal')[];
  evidence: 'model_scored' | 'selection_only';
  media: { image: { supported: boolean; max_per_decision: number; max_bytes_each: number; [key: string]: JsonValue } };
  limits: { [key: string]: number };
  reasoning?: { modes: string[]; thinking_supported: boolean; [key: string]: JsonValue };
  request_policy?: { supported: boolean; target_error_rate: string };
  batch?: {
    supported: boolean; enabled: boolean; execution: 'native_parallel';
    max_requests: number; max_decisions: number; max_decisions_per_wave: number;
    text: boolean; image: boolean; mixed_media: boolean; reasoning_modes: string[];
  };
}
