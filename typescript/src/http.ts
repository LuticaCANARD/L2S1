export type * from './types.js';
import type { Capabilities, DecisionRequest, DecisionResponse } from './types.js';

export class L2S1Error extends Error {
  constructor(message: string, readonly code: string, readonly status?: number,
    readonly requestId?: string, readonly userReason?: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'L2S1Error';
  }
}
export interface ClientOptions {
  baseUrl: string;
  /** Default 180 seconds. No automatic retries: inference may already have run. */
  timeoutMs?: number;
  headers?: HeadersInit;
  fetch?: typeof globalThis.fetch;
}
export interface CallOptions { signal?: AbortSignal; timeoutMs?: number }
export function positiveTimeout(value: number): number {
  if (!Number.isSafeInteger(value) || value < 1 || value > 2_147_483_647) {
    throw new RangeError('timeoutMs must be an integer in 1..2147483647');
  }
  return value;
}
function object(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Browser-compatible transport; validation and inference stay in Rust. */
export class L2S1Client {
  readonly baseUrl: string;
  private readonly timeoutMs: number;
  private readonly headers: Headers;
  private readonly fetcher: typeof globalThis.fetch;
  private readonly lifetime = new AbortController();
  constructor(options: ClientOptions) {
    const url = new URL(options.baseUrl);
    if (!['http:', 'https:'].includes(url.protocol) || url.search || url.hash || url.username || url.password) {
      throw new TypeError('baseUrl must be an HTTP(S) URL without credentials, query or fragment');
    }
    this.baseUrl = url.href.replace(/\/$/, '');
    this.timeoutMs = positiveTimeout(options.timeoutMs ?? 180_000);
    this.headers = new Headers(options.headers);
    this.fetcher = options.fetch ?? globalThis.fetch;
  }
  private async request(path: string, body: unknown, options: CallOptions): Promise<unknown> {
    this.lifetime.signal.throwIfAborted();
    const timeout = AbortSignal.timeout(positiveTimeout(options.timeoutMs ?? this.timeoutMs));
    const signal = AbortSignal.any([this.lifetime.signal, timeout, ...(options.signal ? [options.signal] : [])]);
    const headers = new Headers(this.headers);
    headers.set('accept', 'application/json');
    if (body !== undefined) headers.set('content-type', 'application/json');
    const response = await this.fetcher(`${this.baseUrl}${path}`, {
      method: body === undefined ? 'GET' : 'POST', headers, signal,
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
      redirect: 'error',
    });
    let value: unknown;
    try { value = await response.json(); }
    catch (cause) {
      if (signal.aborted) throw signal.reason;
      throw new L2S1Error('Server returned invalid JSON', 'invalid_response', response.status, undefined, undefined, { cause });
    }
    if (!response.ok) {
      const error = object(value) && object(value.error) ? value.error : {};
      throw new L2S1Error(
        typeof error.message === 'string' ? error.message : `HTTP ${response.status}`,
        typeof error.code === 'string' ? error.code : 'http_error', response.status,
        typeof error.request_id === 'string' ? error.request_id : undefined,
        typeof error.user_reason === 'string' ? error.user_reason : undefined,
      );
    }
    return value;
  }
  async health(options: CallOptions = {}): Promise<{ status: 'ok' }> {
    const value = await this.request('/healthz', undefined, options);
    if (!object(value) || value.status !== 'ok') throw new L2S1Error('Invalid health response', 'invalid_response');
    return { status: 'ok' };
  }
  async capabilities(options: CallOptions = {}): Promise<Capabilities> {
    const value = await this.request('/v1/capabilities', undefined, options);
    if (!object(value) || value.api_version !== 1 || !object(value.backend)
      || typeof value.backend.runtime !== 'string' || typeof value.backend.model !== 'string'
      || !Array.isArray(value.decision_types) || !['model_scored', 'selection_only'].includes(String(value.evidence))
      || !object(value.media) || !object(value.limits)) {
      throw new L2S1Error('Invalid v1 capabilities response', 'invalid_response');
    }
    return value as unknown as Capabilities;
  }
  async decide(request: DecisionRequest, options: CallOptions = {}): Promise<DecisionResponse> {
    const value = await this.request('/v1/decisions', request, options);
    if (!object(value) || value.api_version !== 1 || typeof value.request_id !== 'string'
      || !object(value.backend) || typeof value.backend.runtime !== 'string' || typeof value.backend.model !== 'string'
      || !('policy' in value) || !Array.isArray(value.results) || value.results.length !== request.decisions.length
      || value.results.some((result, index) => !object(result) || result.id !== request.decisions[index]?.id
        || !object(result.value) || result.value.type !== request.decisions[index]?.kind.type
        || !['selected', 'abstained'].includes(String(result.status)) || !Array.isArray(result.abstention_reasons)
        || !object(result.evidence) || !['model_scored', 'selection_only'].includes(String(result.evidence.type))
        || !object(result.usage))) {
      throw new L2S1Error('Server response does not match the v1 decision contract', 'invalid_response');
    }
    return value as unknown as DecisionResponse;
  }
  /** Cancels this client's HTTP calls; the remote server remains running. */
  close(): void {
    this.lifetime.abort(new L2S1Error('HTTP client is closed', 'backend_closed'));
  }
}
