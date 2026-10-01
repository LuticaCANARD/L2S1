"""Render the human report from the exported measurements."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
s=json.loads((root/'summary.json').read_text())
lines=[]
def add(text=''):lines.append(text)
def row(*values):add('| '+' | '.join(map(str,values))+' |')
def percent(n,d):return f'{100*n/d:.2f}%'
add('''# Decision performance implementation — 2026-09-30

Implements the first three items from the [performance review](../../docs/PERFORMANCE_REVIEW.md): a gated compute tuner, resident evaluator sessions, and opt-in structured arithmetic/time/graph facts. [Usage](../../docs/DECISION_PERFORMANCE.md).

Base: main `6b2fcfa`; exact measured source, binary, model, input and runtime hashes are retained in the artifacts below. No model weights, universal compute defaults, prompt defaults, or acceptance thresholds were changed.

## Conditions and reporting

RTX 3080 10 GiB, i9-9900K, four inference threads, WSL2, CUDA 12.4.131, driver 596.21. Pinned llama.cpp `3d82ef62d47fd74e18f36c5eccbdcf965b617b17`. Native Rust in-process, one model process at a time, one ordinary untimed warmup batch, three timed passes in alternating configuration order. Context 4096 throughout measured studies. Models: Gemma 4 E2B Q8_0 and Gemma 4 12B QAT Q4_0.

Native placement logs confirm E2B 36/36 layers and 12B 49/49 layers offloaded. CUDA model buffers: 2342.30 / 6637.69 MiB; CPU-mapped model buffers: 2788.00 / 787.50 MiB respectively. Layer offload is not a claim that every model byte resides on GPU. Reported GPU maxima sample the whole board, including desktop and loading; RSS/HWM samples include loading. They are separate from timed inference and are not allocator-exact peaks.

Tables use median of run p50s and worst run nearest-rank p95. Full ranges, stage timings, throughput, memory and probability drift are in `summary.json`. Repeats assess timing/stability: they do not multiply the accuracy denominator. Policy: top probability >=0.8, candidate mass >=0.05, no top tie. Scores are not guarantees of correctness.

## 1. Compute tuning

Development fixture: 12 requests / 36 unique rule decisions, with varied audit-history lengths. All eight profiles retained 36/36 raw and accepted-correct decisions in all three runs. Layout was explicitly Legacy in every profile, so compute comparisons did not change prompt order.
''')
row('Profile','p50 ms','worst p95 ms','GPU sampled max MiB')
row('---','---:','---:','---:')
for name,v in s['12b']['compute']['configs'].items():row(name,f"{v['median_p50_ms']:.2f}",f"{v['worst_p95_ms']:.2f}",v['gpu0_sampled_max_mib'])
compute=s['12b']['compute'];base=compute['configs']['baseline'];flash=compute['configs']['flash']
add(f"\nFlashAttention alone reduced median p50 by {100*(1-flash['median_p50_ms']/base['median_p50_ms']):.2f}%, below the predeclared 5% gate. The tuner selected **{compute['selection']['selected']['name']}**. Probability/mass drift stayed below 0.02 on this small fixture. This does not establish equivalence on other tasks; JevBench below demonstrates that limit.")
add('\nDynamic allocation greatly reduced the fixed width-3 parallel slowdown, but parallel still did not beat fresh on this fixture. High board memory and the static-context slowdown were observed together; no page-migration trace was collected to establish its internal cause.')
add('''
## 2. Actual native prefix reuse

Each profile evaluates 12 changing states, all 12/12 raw and accepted-correct. The schema/manual is deliberately long enough for batch-aligned reuse; short unrelated production prompts can reuse zero tokens. Batch/ubatch 128, FA on, Legacy. Fixed schema uses one request per call; shared evidence uses two requests per call, width 2 and dynamic context. Shared-evidence latency is **completion latency for the two-request call**, not divided by two.

The resident scope opens after ordinary warmup, so its first measured call is cold. `split-cold` uses the same execution mode without retention across calls. Counts below are actual reused tokens per pass, including within-call sharing where applicable.
''')
row('Model / fixture','Mode','p50 ms','worst p95 ms','Reused tokens/pass','Max probability drift vs fresh')
row('---','---','---:','---:','---:','---:')
for model in ('e2b','12b'):
 for phase in ('schema','shared'):
  for name,v in s[model][phase]['configs'].items():
   delta=max(c['max_probability_delta'] for c in v['comparisons'])
   row(model+' / '+phase,name,f"{v['median_p50_ms']:.2f}",f"{v['worst_p95_ms']:.2f}",v['runs'][0]['reused_prefix_tokens'],f'{delta:.6f}')
add('''
Both models passed fixed-schema and measured FA-on resident-parallel error recovery and post-drop isolation tests. Dynamic allocation was primed for the recovery reference: allocation growth itself clears KV and can legitimately alter reuse counts. Cold-first and subsequent-call timings remain separate in the JSON evidence.

## 3. Structured-input accuracy

The helper performs checked i64 arithmetic, signed UTC-second differences and directed shortest paths using explicit JSON pointers. It preserves source fields and adds auditable facts. The model still chooses an option and applies its policy. This measures the value of supplied computational evidence; it is not a new general reasoning capability or a result on natural-language JevBench.

Development: 36 instances (12 per family). Freeze rule: enable a model/family only if raw correct increases, wrong accepted does not increase, and p95 is at most twice baseline. E2B enabled numeric/time and rejected graph (wrong accepted 1→4 on development). 12B enabled all three. Selection files were frozen before holdout evaluation.

Holdout: **120 unique instances per model**, 40 per family, disjoint numeric ranges and graph node namespaces. Same templates are reused, so these results do not establish cross-domain generalization. Selection was not changed after observing holdout.
''')
row('Model / mode','Raw correct / 120','Accepted correct / accepted','Wrong accepted','Abstained','p50 / worst p95 ms')
row('---','---:','---:','---:','---:','---:')
for model in ('e2b','12b'):
 for name,v in s[model]['holdout']['configs'].items():
  c=v['counts_first_run'];row(model+' / '+name,f"{c['raw_correct']}/120 ({percent(c['raw_correct'],120)})",f"{c['accepted_correct']}/{c['accepted']}",c['wrong_accepted'],c['abstained'],f"{v['median_p50_ms']:.2f} / {v['worst_p95_ms']:.2f}")
for model in ('e2b','12b'):
 p=s[model]['holdout']['paired'];lo,hi=p['item_bootstrap95']
 add(f"\n{model}: {p['fixed']} fixed, {p['regressed']} regressed; paired gain {p['delta']*100:.2f} percentage points, instance-bootstrap 95% interval [{lo*100:.2f}, {hi*100:.2f}]. Shared templates limit this interval's scope. Timing includes preprocessing; model loading and upfront input validation are excluded.")
add('''
## JevBench public regression

Pinned revision `f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84`: 48 easy + 72 original + 111 hard = **231 unique items**, not the full 534-item suite. This public set had already been examined; it is a regression set, not the untouched selection holdout. Original requests were retained and structured facts were disabled. Fresh/Legacy, context 4096, batch/ubatch 256, four threads. The off/on comparison was fixed before reading new public outcomes.
''')
row('Model / FA','Raw correct / 231','Accepted correct / accepted','Wrong accepted','Abstained','p50 / worst p95 ms')
row('---','---:','---:','---:','---:','---:')
drift_notes=[]
for model in ('e2b','12b'):
 for name,v in s[model]['jevbench']['configs'].items():
  c=v['counts_first_run'];row(model+' / '+('off' if name=='baseline' else 'on'),f"{c['raw_correct']}/231 ({percent(c['raw_correct'],231)})",f"{c['accepted_correct']}/{c['accepted']}",c['wrong_accepted'],c['abstained'],f"{v['median_p50_ms']:.2f} / {v['worst_p95_ms']:.2f}")
 for name,v in s[model]['jevbench']['configs'].items():assert v['stable_counts'],(model,name)
 c=s[model]['jevbench']['configs']['flash']['comparisons'][0]
 drift_notes.append(f"\n{model} FA-on vs off: {c['changed_top1']} top-1 changes, {c['changed_selection']} accepted-selection changes, max probability drift {c['max_probability_delta']:.6f}. Counts were stable in all three repeats. Speed does not justify silently changing compute/calibration identity or acceptance behavior.")
for note in drift_notes:add(note)
add('''
The Rust adapter's first run for all four model/config pairs was independently checked against pinned upstream `score_task` and `summarize`, including accuracy, calibration metrics and latency. No truncation or failed predictions occurred. Baseline counts reproduce the earlier 159/231 E2B and 194/231 12B results.

## Validation and known failing profile

New deterministic-fact unit tests, evaluator input-contract tests, tuner failure/selection tests, native resident recovery/isolation and fixed-layout compute-resize tests passed. The compute test now explicitly fixes Legacy layout, avoiding an invalid comparison between different prompts. CI also rebuilds the evaluator with the native CPU backend and replays the committed evidence.

**Known failed existing test:** `tests/parallel.rs::real_model_prefix_sharing` on E2B, FA off, batch 256, width 4, static context 2048, Legacy: probability drift versus fresh was **0.1142807672**, exceeding its existing **0.05** limit. The limit was not relaxed and this profile is not claimed equivalent. The measured resident tests instead isolate cross-call retention within the same FA-on profile. This PR adds tooling around existing native sessions; it does not claim to fix that legacy fresh/parallel drift.

## Evidence and reproduction

- [Summary](summary.json): all 3-pass metrics, family counts, drift, uncertainty and memory.
- [Records](records.jsonl), [record hash](records-sha256.json), [provenance](provenance.json).
- [Protocol](protocol.json), [JevBench protocol](jevbench-protocol.json), [E2B selection](e2b-accuracy-selection.json), [12B selection](12b-accuracy-selection.json).
- `hardware.json`, `implementation-hashes.json`, `validation.json` and `jevbench-audit.json` record the environment and checks. JevBench data is MIT licensed; see [license](JEVBENCH-LICENSE).

From the repository root, run `python3 benchmarks/performance-implementation-20260930/verify.py` to recompute every reported run without a model. `prepare.py` deterministically recreates synthetic inputs and gold. Build the evaluator with `cargo build --release --locked --features llama-cuda --example evaluate_jsonl`.

Use `scripts/benchmark_decision_performance.py` with the checked-in input/gold/matrix files, model GGUF, native library directory, a new output directory and `--repeats 3`. Use `--evaluation-only` for schema/shared/accuracy/JevBench studies. Compute tuning omits that flag and writes a gated `selection.json`. Run E2B and 12B sequentially.

After separate `accuracy-dev` runs under `<results>/e2b` and `<results>/12b`, run `freeze_accuracy.py --results <results> --selection-dir <results>/selections` before evaluating the generated per-model `selected-holdout.jsonl` with `accuracy-matrix.json` and `holdout-gold.json`.

For JevBench, use `l2s1-tools jevbench-public prepare` with the pinned upstream checkout, evaluate its unchanged `requests.jsonl` with `jevbench-matrix.json` and `jevbench-gold.json`, then use `jevbench-public score`. `audit_jevbench.py --upstream <checkout> --results <results>` checks upstream parity. `report.py --results <results>` exports compact records and metrics; `render.py` generates this report. All scripts require only the Python standard library except the pinned upstream scorer's own dependencies.
''')
(root/'README.md').write_text('\n'.join(lines).rstrip()+'\n')
