# Reproducible decision tuning and structured evidence

These features are opt-in. Default model loading, prompt layout and acceptance
policy remain unchanged. Start with the [measured study](../benchmarks/performance-implementation-20260930/README.md).

## Tune compute on development requests

Build `evaluate_jsonl` with the backend you use, then measure a matrix on
representative **development** inputs. Each input row is `{ "id": "case", "request":
{...} }`; the separate gold file is `{ "case": { "decision_id": "option_id" } }`.
The evaluator never opens gold. Matrix entries contain a name and evaluator CLI
arguments; the first entry is the reference. See the study's `compute-matrix.json`
for FlashAttention, batch/ubatch, parallel width and dynamic context examples.

```sh
cargo build --release --features llama-cuda --example evaluate_jsonl
python3 scripts/benchmark_decision_performance.py \
  --binary target/release/examples/evaluate_jsonl --model models/model.gguf \
  --input development.jsonl --gold development-gold.json \
  --matrix compute-matrix.json --out results/compute-study --repeats 3
```

Each model runs in its own process; one batch warms the model before measurement.
The harness alternates configuration order, retains every response and error,
records stage timings and samples whole-board GPU memory/utilization (including
other processes), plus Linux process RSS when available. Supply `--runtime /path/to/libs` to pin and hash dynamically
loaded libraries, or `--cpu` for CPU runs. Without `--runtime`, ambient library
resolution is not a pinned runtime. Loading is recorded separately from measured
request latency. GPU samples include loading and are sampled maxima, not allocator
peaks; telemetry failure is explicit. Run only one GPU study at a time.

A development recommendation requires three complete runs, no raw/accepted
selection changes or token-count changes, maximum probability and candidate-mass
drift <=0.02, median run p50 <=95% of the reference and worst run p95 <=105% of the
reference. If no candidate passes, `selection.json` retains the reference. An
unstable or incomplete reference produces no selection. This is a conservative
workload-specific gate, not a statistical guarantee. Freeze selection before
running independent evaluation with `--evaluation-only`; do not change universal
defaults based on this small study. An inference failure retains partial evidence
but makes the harness exit unsuccessfully.

## Keep native prefixes between calls

The evaluator now exposes existing scoped Rust sessions:

```sh
# Every row has one identical decision, varying state, and no shared evidence.
target/release/examples/evaluate_jsonl --model models/model.gguf --cuda \
  --input fixed-schema.jsonl --output fixed-schema-results.jsonl --warmup \
  --execution-mode prefix-reuse --resident fixed-schema --batch 128 \
  --flash-attention on --prompt-layout legacy

# Requests carry identical leading evidence in their shared field.
target/release/examples/evaluate_jsonl --model models/model.gguf --cuda \
  --input shared-prefix.jsonl --output shared-prefix-results.jsonl --warmup \
  --execution-mode parallel --resident shared-prefix --parallel-width 2 \
  --request-batch-size 2 --parallel-context-dynamic --batch 128 \
  --flash-attention on --prompt-layout legacy
```

Fixed-schema mode checks the complete input schema before loading the model,
requires request batch size one, and rejects output heads. Shared-prefix mode
requires parallel execution. Sessions start **after** ordinary warmup, so the
first measured call is cold; later calls report actual `reused_prefix_tokens`.
A failed native call clears retained KV, and dropping a session isolates the next
caller. Short prefixes can reuse zero tokens. Batch alignment remains the default;
no answer or state suffix is reused as a prediction. Recurrent/hybrid models are
rejected by the existing session implementation.

The shown configuration is the measured FA-on/batch-128 profile. Do not assume
fresh and parallel are numerically equivalent for every model: the existing E2B
FA-off/batch-256/width-4 regression fixture exceeded its 0.05 probability-drift
limit (observed 0.1143). That failure is retained in the study; its tolerance was
not relaxed. Evaluate probabilities and accepted decisions for your own prompts
before adopting a profile, even when raw top-1 remains unchanged.

## Add exact facts to structured state

`derive_facts(&state, &specs)` is a backend-independent Rust helper. It returns a
new object preserving input fields and adding `_l2s1_facts`. It performs only
explicit computations; it never reads gold or picks an option. Numeric operands
are signed 64-bit integers. Use scaled integers for amounts and normalize units
in the application; fractions, strings and overflow are rejected. Times must
already be UTC Unix seconds, not local date strings. Graphs contain directed
pairs of nonempty node IDs, and shortest paths are computed with cycle-safe BFS.

```json
{
  "id": "elapsed-example",
  "request": {
    "state": {"start": 86399, "end": 86401},
    "decisions": [{
      "id": "elapsed",
      "instruction": "How many signed seconds elapsed from start to end?",
      "kind": {"type": "choice", "options": [
        {"id": "two", "criterion": "2 seconds"},
        {"id": "sixty", "criterion": "60 seconds"}
      ]}
    }]
  },
  "facts": [{"name": "elapsed", "op": "elapsed_seconds", "start": "/start", "end": "/end"}]
}
```

Pass `--derive-facts` to `evaluate_jsonl` to enable per-row specs. Without it,
original requests are evaluated. Supported operations: `add`, `subtract`,
`compare` (operands `left`/`right`); `elapsed_seconds` (`start`/`end`); and
`shortest_path` (`edges`/`from`/`to`). Operands are JSON pointers into the original
state, never preceding facts. A path from a node to itself has zero hops even if
that node does not occur in the edge list. An unreachable path has `hops: null`.
Reserved-key collisions, duplicate fact names, unknown fields/operations, missing
operands and malformed edges fail before model loading. Limits are 64 facts,
4096 graph edges, 256 bytes per node ID, and 128 bytes per fact name.

Latency includes cloning/preprocessing immediately before inference. Outputs
include preprocessing time, inference time and the SHA-256 of serialized fact
specs. Hash the input file as well to identify operand values. Prompt content
changes: validate accuracy, latency and calibration for each task/model. These
facts are computational evidence from caller-supplied data, not verification of
that data's truth. This is not a parser for arbitrary prose, a JevBench-specific
answer lookup, or a replacement for independent evaluation.
