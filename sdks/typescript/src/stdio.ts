import type { ChildProcess } from 'node:child_process';
import { L2S1Error, decisionResponse, positiveTimeout } from './http.js';
import type { CallOptions } from './http.js';
import type { Capabilities, DecisionRequest, DecisionResponse } from './types.js';

interface Pending { resolve(value: unknown): void; reject(error: unknown): void }
const object = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);

/** JSON-line RPC directly to a compiled Rust child. No HTTP or sockets. */
export class StdioClient {
  private counter = 0;
  private closed = false;
  private pending = new Map<string, Pending>();
  private outstanding = new Set<string>();
  constructor(private readonly child: ChildProcess, private readonly timeoutMs: number) {
    let buffer = '';
    child.stdout!.setEncoding('utf8');
    child.stdout!.on('data', (chunk: string) => {
      buffer += chunk;
      if (buffer.length > 64 * 1024 * 1024) { this.fail(new L2S1Error('stdio response exceeds limit', 'invalid_response')); return; }
      let newline: number;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline); buffer = buffer.slice(newline + 1);
        try {
          const value: unknown = JSON.parse(line);
          if (!object(value) || typeof value.id !== 'string' || !this.outstanding.delete(value.id)) {
            throw new L2S1Error('Invalid stdio response ID', 'invalid_response');
          }
          const pending = this.pending.get(value.id); this.pending.delete(value.id);
          if (!pending) continue; // A timed-out or cancelled call is never replayed.
          if (object(value.error)) {
            pending.reject(new L2S1Error(String(value.error.message), String(value.error.code), undefined, value.id,
              typeof value.error.user_reason === 'string' ? value.error.user_reason : undefined));
          } else if ('result' in value) pending.resolve(value.result);
          else pending.reject(new L2S1Error('Invalid stdio envelope', 'invalid_response'));
        } catch (error) { this.fail(error); }
      }
    });
    child.stdin!.on('error', (error) => this.fail(error));
    child.once('close', () => this.fail(new L2S1Error('Rust process closed', 'process_closed')));
  }
  private fail(error: unknown): void {
    this.closed = true;
    for (const pending of this.pending.values()) pending.reject(error);
    this.pending.clear(); this.outstanding.clear();
  }
  private request(op: string, body: unknown, options: CallOptions): Promise<unknown> {
    if (this.closed) return Promise.reject(new L2S1Error('stdio client is closed', 'backend_closed'));
    if (this.outstanding.size >= 16) return Promise.reject(new L2S1Error('stdio inference queue is full', 'busy'));
    const signal = AbortSignal.any([AbortSignal.timeout(positiveTimeout(options.timeoutMs ?? this.timeoutMs)), ...(options.signal ? [options.signal] : [])]);
    signal.throwIfAborted();
    const id = `local-${++this.counter}`;
    const line = JSON.stringify({ id, op, body });
    if (Buffer.byteLength(line) > 44 * 1024 * 1024 + 1024) return Promise.reject(new L2S1Error('stdio request exceeds limit', 'invalid_request'));
    return new Promise((resolve, reject) => {
      const cleanup = () => signal.removeEventListener('abort', onAbort);
      const onAbort = () => { this.pending.delete(id); cleanup(); reject(signal.reason); };
      this.pending.set(id, {
        resolve(value) { cleanup(); resolve(value); },
        reject(error) { cleanup(); reject(error); },
      });
      this.outstanding.add(id);
      signal.addEventListener('abort', onAbort, { once: true });
      this.child.stdin!.write(line + '\n', (error) => { if (error) this.fail(error); });
    });
  }
  async health(options: CallOptions = {}): Promise<void> {
    const value = await this.request('health', null, options);
    if (!object(value) || value.status !== 'ok') throw new L2S1Error('Invalid stdio health response', 'invalid_response');
  }
  async capabilities(options: CallOptions = {}): Promise<Capabilities> {
    const value = await this.request('capabilities', null, options);
    if (!object(value) || value.api_version !== 1 || !object(value.backend)
      || typeof value.backend.runtime !== 'string' || typeof value.backend.model !== 'string'
      || !Array.isArray(value.decision_types) || !['model_scored', 'selection_only'].includes(String(value.evidence))
      || !object(value.media) || !object(value.limits)) {
      throw new L2S1Error('Invalid stdio capabilities', 'invalid_response');
    }
    return value as unknown as Capabilities;
  }
  async decide(request: DecisionRequest, options: CallOptions = {}): Promise<DecisionResponse> {
    const snapshot = structuredClone(request);
    return decisionResponse(await this.request('decide', snapshot, options), snapshot);
  }
  async decideBatch(requests: readonly DecisionRequest[], options: CallOptions = {}): Promise<DecisionResponse[]> {
    if (this.closed) throw new L2S1Error('stdio client is closed', 'backend_closed');
    const snapshots = structuredClone(requests);
    if (!snapshots.length) return [];
    const value = await this.request('decide_batch', { requests: snapshots }, options);
    if (!object(value) || value.api_version !== 1 || typeof value.request_id !== 'string'
      || value.execution !== 'native_parallel' || !Array.isArray(value.responses) || value.responses.length !== snapshots.length) {
      throw new L2S1Error('Invalid native stdio batch response', 'invalid_response');
    }
    return value.responses.map((response, index) => decisionResponse(response, snapshots[index]!));
  }
  close(): void { this.fail(new L2S1Error('stdio client is closed', 'backend_closed')); }
}
