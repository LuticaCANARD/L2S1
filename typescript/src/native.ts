import { spawn, type ChildProcess } from 'node:child_process';
import { L2S1Client, L2S1Error, positiveTimeout } from './http.js';
import type { CallOptions } from './http.js';
import type { Capabilities, DecisionPolicy, DecisionRequest, DecisionResponse } from './types.js';
import { resolveRuntime } from './runtime.js';
import { StdioClient } from './stdio.js';

export interface LoadOptions {
  model: string;
  /** Local compiled process RPC. Default stdio; HTTP is an explicit option. */
  transport?: 'stdio' | 'http';
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
  /** Per call timeout, including the whole batch; defaults to 180 seconds. */
  timeoutMs?: number;
  signal?: AbortSignal;
  /** Additional Rust CLI flags. --stdio and --listen are reserved. */
  extraArgs?: string[];
  /** Native diagnostic chunks. Kept draining even when no callback is supplied. */
  onStderr?: (chunk: string) => void;
}

function argumentsFor(options: LoadOptions): string[] {
  if (!options.model?.trim()) throw new TypeError('model is required');
  const extra = options.extraArgs ?? [];
  if (extra.some((arg) => arg === '--stdio' || arg.startsWith('--stdio=') || arg === '--listen' || arg.startsWith('--listen='))) {
    throw new TypeError('--stdio/--listen are managed by L2S1.load');
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
  if (options.transport !== undefined && !['stdio', 'http'].includes(options.transport)) throw new TypeError('Invalid transport');
  args.push(...extra, ...(options.transport === 'http' ? ['--listen', '127.0.0.1:0'] : ['--stdio']));
  return args;
}

/** Owns one resident compiled Rust model process with typed JSON requests. */
export class RustProcessBackend implements AsyncDisposable {
  private closing: Promise<void> | undefined;
  private exited = false;
  private readonly exit: Promise<void>;
  private readonly lifetime = new AbortController();
  private client: L2S1Client | StdioClient | undefined;
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
    const stdio = options.transport !== 'http';
    const child = spawn(runtime.binaryPath, args, { env: runtime.env, stdio: stdio ? ['pipe', 'pipe', 'pipe'] : ['ignore', 'ignore', 'pipe'], windowsHide: true, shell: false });
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
        if (stdio && line.trim() === 'l2s1 stdio ready') {
          announced = true;
          reportReady('stdio');
          break;
        }
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
      engine.client = stdio ? new StdioClient(child, options.timeoutMs ?? 180_000)
        : new L2S1Client({ baseUrl, timeoutMs: options.timeoutMs ?? 180_000 });
      // Health confirms that the selected transport responds after startup.
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
  decideBatch(requests: readonly DecisionRequest[], options: CallOptions = {}): Promise<DecisionResponse[]> {
    return this.client!.decideBatch(requests, this.callOptions(options));
  }
  capabilities(options: CallOptions = {}): Promise<Capabilities> {
    return this.client!.capabilities(this.callOptions(options));
  }
  /** Idempotent. Aborts local calls, terminates the owned child and waits for exit. */
  close(): Promise<void> {
    if (!this.closing) {
      this.lifetime.abort(new L2S1Error('Rust process closed', 'process_closed'));
      this.client?.close();
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
