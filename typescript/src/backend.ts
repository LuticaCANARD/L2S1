import type { CallOptions } from './http.js';
import type { Capabilities, DecisionRequest, DecisionResponse } from './types.js';

/** A transport or runtime supplying the shared decision contract. */
export interface DecisionBackend {
  decide(request: DecisionRequest, options?: CallOptions): Promise<DecisionResponse>;
  capabilities(options?: CallOptions): Promise<Capabilities>;
  close?(): void | Promise<void>;
}
