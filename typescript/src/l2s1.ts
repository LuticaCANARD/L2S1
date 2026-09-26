import type { DecisionBackend } from './backend.js';
import { L2S1Client, L2S1Error } from './http.js';
import type { CallOptions, ClientOptions } from './http.js';
import { RustProcessBackend } from './native.js';
import type { LoadOptions } from './native.js';
import type { Capabilities, DecisionRequest, DecisionResponse } from './types.js';

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
