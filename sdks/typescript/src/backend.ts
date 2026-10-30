import type { CallOptions } from './http.js';
import type { Capabilities, DecisionRequest, DecisionResponse, TuneRequest, TuneResponse } from './types.js';

/** A transport or runtime supplying the shared decision contract. */
export interface DecisionBackend {
  decide(request: DecisionRequest, options?: CallOptions): Promise<DecisionResponse>;
  /** Optional explicit native batch boundary. There is no serial fallback. */
  decideBatch?(requests: readonly DecisionRequest[], options?: CallOptions): Promise<DecisionResponse[]>;
  /** Optional decision accuracy tuning boundary. */
  tune?(request: TuneRequest, options?: CallOptions): Promise<TuneResponse>;
  capabilities(options?: CallOptions): Promise<Capabilities>;
  close?(): void | Promise<void>;
}
