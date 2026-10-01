import type { DecisionBackend } from './backend.js';
import { L2S1Client, L2S1Error } from './http.js';
import type { CallOptions, ClientOptions } from './http.js';
import { RustProcessBackend } from './native.js';
import type { LoadOptions } from './native.js';
import type { Capabilities, Decision, DecisionRequest, DecisionResponse, JsonValue } from './types.js';

/** One application API for a bundled engine, HTTP server, or custom backend. */
export class L2S1 implements AsyncDisposable {
  private closing: Promise<void> | undefined;
  private constructor(private readonly backend: DecisionBackend) {}
  static async load(options: LoadOptions): Promise<L2S1> {
    return new L2S1(await RustProcessBackend.load(options));
  }
  static connect(options: ClientOptions): L2S1 {
    return new L2S1(new L2S1Client(options));
  }
  /** Ownership is transferred: close() calls the backend's optional close hook. */
  static fromBackend(backend: DecisionBackend): L2S1 {
    return new L2S1(backend);
  }
  private checkOpen(): void {
    if (this.closing) throw new L2S1Error('Backend is closed', 'backend_closed');
  }
  decide(request: DecisionRequest, options: CallOptions = {}): Promise<DecisionResponse> {
    this.checkOpen();
    return this.backend.decide(request, options);
  }
  /** One native batch call. Timeout applies to the entire batch; no replay. */
  async decideBatch(requests: readonly DecisionRequest[], options: CallOptions = {}): Promise<DecisionResponse[]> {
    this.checkOpen();
    const snapshots = structuredClone(requests);
    if (!snapshots.length) return [];
    if (!this.backend.decideBatch) throw new L2S1Error('Backend does not support native batching', 'batch_unsupported');
    return this.backend.decideBatch(snapshots, options);
  }
  /** Snapshot fixed questions; each invocation supplies independent state.
   * This is an application template, not model token compilation or KV reuse.
   */
  prepare<State extends JsonValue = JsonValue>(decisions: readonly Decision[]): PreparedDecision<State> {
    this.checkOpen();
    return new PreparedDecision<State>(this, decisions);
  }
  capabilities(options: CallOptions = {}): Promise<Capabilities> {
    this.checkOpen();
    return this.backend.capabilities(options);
  }
  close(): Promise<void> {
    // Set state before invoking a user close hook, including a synchronous throw.
    this.closing ??= Promise.resolve().then(() => this.backend.close?.());
    return this.closing;
  }
  [Symbol.asyncDispose](): Promise<void> { return this.close(); }
}
export class PreparedDecision<State extends JsonValue = JsonValue> {
  private readonly decisions: Decision[];
  constructor(private readonly engine: L2S1, decisions: readonly Decision[]) {
    this.decisions = structuredClone([...decisions]);
  }
  decide(state: State, options: CallOptions = {}): Promise<DecisionResponse> {
    return this.engine.decide({ state: structuredClone(state), decisions: structuredClone(this.decisions) }, options);
  }
  decideBatch(states: readonly State[], options: CallOptions = {}): Promise<DecisionResponse[]> {
    return this.engine.decideBatch(states.map((state) => ({ state, decisions: this.decisions })), options);
  }
}
