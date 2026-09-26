import { spawn, type ChildProcess } from 'node:child_process';
import { L2S1Client, L2S1Error, positiveTimeout } from './http.js';
import type { CallOptions } from './http.js';
import type { Capabilities, DecisionPolicy, DecisionRequest, DecisionResponse } from './types.js';
import { resolveRuntime } from './runtime.js';

export interface LoadOptions {
  model: string;
  /** Override the platform package with a custom Rust executable or a name on PATH. */
  binaryPath?: string;
  device?: 'cpu' | 'cuda' | 'metal';
  mmproj?: string;
  lora?: string;
  context?: number;
  batch?: number;
  ubatch?: number;
  threads?: number;
  gpuLayers?: number;
  executionMode?: 'fresh' | 'prefix-reuse' | 'state-restore' | 'parallel';
  parallelWidth?: number;
  promptLayout?: 'legacy' | 'state-first';
  promptDetail?: 'minimal' | 'typed' | 'typed-examples';
  policy?: DecisionPolicy;
  /** Model startup timeout; defaults to 120 seconds. */
  startupTimeoutMs?: number;
  /** Per HTTP request timeout; defaults to 180 seconds. */
  timeoutMs?: number;
  signal?: AbortSignal;
  /** Additional Rust CLI flags. --listen is reserved for process ownership. */
  extraArgs?: string[];
  /** Native diagnostic chunks. Kept draining even when no callback is supplied. */
  onStderr?: (chunk: string) => void;
}

function argumentsFor(options: LoadOptions): string[] {
  if (!options.model?.trim()) throw new TypeError('model is required');
  const extra = options.extraArgs ?? [];
  if (extra.some((arg) => arg === '--listen' || arg.startsWith('--listen='))) {
    throw new TypeError('--listen is managed by L2S1.load');
  }
  const args = ['--model', options.model];
  const flags: [string, string | number | undefined][] = [
    ['device', options.device], ['mmproj', options.mmproj], ['lora', options.lora],
    ['context', options.context], ['batch', options.batch], ['ubatch', options.ubatch],
    ['threads', options.threads], ['gpu-layers', options.gpuLayers], ['execution-mode', options.executionMode],
    ['parallel-width', options.parallelWidth], ['prompt-layout', options.promptLayout], ['prompt-detail', options.promptDetail],
    ['min-top-probability', options.policy?.min_top_probability], ['min-candidate-mass', options.policy?.min_candidate_mass],
  ];
  for (const [key, value] of flags) if (value !== undefined) args.push(`--${key}`, String(value));
  // Port 0 lets Rust bind an available port atomically, without a port reservation race.
  args.push(...extra, '--listen', '127.0.0.1:0');
  return args;
}

/** Owns one resident Rust model process. Requests use the existing HTTP v1 API. */
export class RustProcessBackend implements AsyncDisposable {
  private closing: Promise<void> | undefined;
  private exited = false;
  private readonly exit: Promise<void>;
  private readonly lifetime = new AbortController();
  private client: L2S1Client | undefined;
  private constructor(private readonly child: ChildProcess) {
    this.exit = new Promise((resolve) => {
      child.once('close', () => {
        this.exited = true;
        this.lifetime.abort(new L2S1Error('Rust process closed', 'process_closed'));
        resolve();
      });
    });
  }
  static async load(options: LoadOptions): Promise<RustProcessBackend> {
    options.signal?.throwIfAborted();
    const args = argumentsFor(options);
    const startupTimeout = positiveTimeout(options.startupTimeoutMs ?? 120_000);
    positiveTimeout(options.timeoutMs ?? 180_000);
    const runtime = resolveRuntime(options.binaryPath, options.device);
    const child = spawn(runtime.binaryPath, args, { env: runtime.env, stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true, shell: false });
    const engine = new RustProcessBackend(child);
    let diagnosticTail = '';
    let pending = '';
    let reportReady: (url: string) => void = () => {};
    let reportFailure: (error: unknown) => void = () => {};
    let announced = false;
    const ready = new Promise<string>((resolve, reject) => { reportReady = resolve; reportFailure = reject; });
    child.once('error', (cause) => reportFailure(new L2S1Error(`Could not start Rust executable: ${cause.message}`, 'spawn_failed', undefined, undefined, undefined, { cause })));
    child.once('close', (code, signal) => reportFailure(new L2S1Error(
      `Rust process exited during startup (${code ?? signal}): ${diagnosticTail}`, 'startup_failed')));
    child.stderr!.setEncoding('utf8');
    child.stderr!.on('data', (chunk: string) => {
      diagnosticTail = (diagnosticTail + chunk).slice(-8192);
      if (options.onStderr) {
        try { options.onStderr(chunk); } catch (cause) { reportFailure(cause); }
      }
      if (announced) return;
      pending += chunk;
      const lines = pending.split('\n');
      pending = lines.pop()!.slice(-8192);
      for (const line of lines) {
        const match = /^l2s1 HTTP listening on 127\.0\.0\.1:(\d+)\s*$/.exec(line);
        if (match && Number(match[1]) > 0 && Number(match[1]) <= 65535) {
          announced = true;
          reportReady(`http://127.0.0.1:${match[1]}`);
          break;
        }
      }
    });
    const startup = new AbortController();
    const timer = setTimeout(() => startup.abort(new L2S1Error('Rust model startup timed out', 'startup_timeout')), startupTimeout);
    const signal = options.signal ? AbortSignal.any([options.signal, startup.signal]) : startup.signal;
    const onAbort = () => reportFailure(signal.reason);
    signal.addEventListener('abort', onAbort, { once: true });
    try {
      const baseUrl = await ready;
      engine.client = new L2S1Client({ baseUrl, timeoutMs: options.timeoutMs ?? 180_000 });
      // A listening announcement follows bind; health confirms the API responds.
      await engine.client.health({ signal: AbortSignal.any([signal, engine.lifetime.signal]) });
      signal.throwIfAborted();
      return engine;
    } catch (error) {
      await engine.close();
      throw error;
    } finally {
      clearTimeout(timer);
      signal.removeEventListener('abort', onAbort);
    }
  }
  private callOptions(options: CallOptions): CallOptions {
    if (this.closing || this.exited || !this.client) throw new L2S1Error('Rust process is closed', 'process_closed');
    return { ...options, signal: options.signal ? AbortSignal.any([options.signal, this.lifetime.signal]) : this.lifetime.signal };
  }
  decide(request: DecisionRequest, options: CallOptions = {}): Promise<DecisionResponse> {
    return this.client!.decide(request, this.callOptions(options));
  }
  capabilities(options: CallOptions = {}): Promise<Capabilities> {
    return this.client!.capabilities(this.callOptions(options));
  }
  /** Idempotent. Aborts local HTTP calls, terminates the owned child and waits for exit. */
  close(): Promise<void> {
    if (!this.closing) {
      this.lifetime.abort(new L2S1Error('Rust process closed', 'process_closed'));
      this.closing = (async () => {
        if (this.exited) return;
        this.child.kill('SIGTERM');
        const timer = setTimeout(() => { this.child.kill('SIGKILL'); }, 2000);
        try { await this.exit; } finally { clearTimeout(timer); }
      })();
    }
    return this.closing;
  }
  [Symbol.asyncDispose](): Promise<void> { return this.close(); }
}
