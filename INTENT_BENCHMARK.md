# Dynamic answer codes and intent evaluation

Candidate count determines one shared code width for the entire decision:

| Candidate count | Codes | Width |
| --- | --- | ---: |
| 2–26 | A, B, …, Z | 1 |
| 27–676 | AA, AB, …, ZZ | 2 |
| 677–17,576 | AAA, AAB, …, ZZZ | 3 |
| 17,577 and above | AAAA, AAAB, …; extend again as needed | 4 or more |

Only as many codes as there are candidates are used. For example, 77 candidates
use AA–CY; 60 use AA–CH. There is no alphabet-derived candidate-count ceiling.
The code-width arithmetic avoids overflowing when the count approaches `usize`
limits. Real requests remain bounded by addressable memory and the model's
context, including any answer prefix that must be evaluated.

## Probability contract

The A–Z path retains its exact prompt and single-token scoring behavior. Above
26 candidates, all codes in the decision have the same character width. The
model template instructs that width explicitly. Candidate names, order and
semantic result IDs remain unchanged; code rotation operates modulo the actual
candidate count.

The tokenizer may encode AA as one token or multiple tokens. The backend checks
each complete continuation at the real assistant boundary and rejects boundary
retokenization, duplicate token paths, and any token path that prefixes another.
It scores each canonical token path autoregressively:

`log P(code | prompt) = sum(log P(next token | prompt, preceding code tokens))`

Each term uses the full vocabulary denominator. The sum of those disjoint code
probabilities is `candidate_mass`; normalizing over the complete candidate set
produces `option_probability`. This is neither the first-character probability
nor an average of token probabilities. It does not sum alternative tokenizations
of the same text, and it does not claim calibrated correctness probabilities.

Each distinct code-token prefix is evaluated once. Exact complete prefill batches
may be reused inside this decision; each branch starts from the original prompt
plus its own code prefix. No generated branch is allowed to contaminate another.
KV state is cleared before and after sequence scoring, including failures.
No candidate shortlist or group tournament is used on this path.

Sequence responses use `scoring_method: code_sequence_conditional_softmax_v1`.
Each score includes `token_ids`; `token_id` is the only token for a one-token
path and `-1` for a multi-token path. Here `raw_logit` stores the joint log
probability. `code_prefix_evaluations` and `code_evaluated_tokens` report native
work across all branches. `input_tokens` still describes the original prompt.
The response prompt version includes `/fixed-width-code-sequences-v1`.

Multi-letter codes support fresh or prefix-reuse execution with full evidence without
output heads, scalar calibration or hidden-feature export. Unsupported
combinations fail explicitly. `encode_decision_sequences()` and `preflight()`
expose complete candidate token paths; the old scalar token export rejects
wide candidates. The single-token calibration, training/export and consensus
workflows retain their existing evidence contracts.

## Preparation and KV caching

Enable `PreparationCacheConfig` on a resident backend to cache exact prompt
preparation and candidate token paths, including `AA` and `AAA` codes. All code
widths share the existing byte/entry budgets. Changing state or schema causes a
prompt miss but can still reuse a compatible answer-boundary mapping. Cache
entries contain tokens, not predictions; failed preparations are not stored.

Select `ExecutionMode::PrefixReuse` and use `backend.shared_state(state)` to
reuse exact, complete prefill batches across calls on one immutable state. A
session may receive different decision schemas. Ordinary `decide()` calls remain
KV-isolated; session creation, errors and drop clear KV. Branch scoring always
uses the original prompt plus the branch's own prefix. `reused_prefix_tokens`
reports root-prompt reuse; `code_evaluated_tokens` includes all branch work.

The cache comparison harness uses the same frozen 400 full-label requests,
one first call and three immediate repeats per case, with two global warmups:

```sh
cargo run --release --offline --features llama --example measure_intent_cache -- \
  --model models/gemma-4-E2B-it-Q8_0.gguf \
  --input results/intent-wide-20260923/prepared/requests.jsonl \
  --output results/cache-session --mode session-cached
```

Use a new output directory for each mode: `fresh-uncached`, `fresh-cached`, or
`session-cached`. The last mode creates one session per case. Reports record
actual preparation hits and native reuse separately; a hot repeat alone is not
proof of a cache hit. Timing excludes model loading, serialization and session
construction. This measures repeated identical calls, not novel-utterance speed.

## BANKING77 and MASSIVE evaluation

The frozen test samples contain 200 English BANKING77 utterances and 200 Korean
MASSIVE utterances. The same samples are used by Gemma 4 E2B Q8_0, E4B Q8_0 and
E4B Q4_K_M, plus Qwen3-8B Q8_0, on `100.66.64.91` (RTX 3060 12 GiB). All official 77 or 60 labels are
present in every respective request. Only raw utterance text enters the state;
gold labels, scenarios, annotations and judgments remain outside inference.

Sampling is uniform without replacement with seed `20260923`, independently per
dataset, then restored to source order. BANKING77's sample represents 73 gold
classes; MASSIVE's sample represents 50. The full Korean test split has no
`cooking_query` example, but that official class remains a candidate. Instructions
are English for BANKING77 and Korean for MASSIVE. Criteria use the original
English label names with underscores replaced by spaces. No training, retrieval,
gold-based candidate selection, or test-driven prompt tuning is performed.

Sources: [BANKING77 / PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets)
and [MASSIVE / Amazon](https://huggingface.co/datasets/AmazonScience/massive), both
CC BY 4.0. MASSIVE's Korean Parquet is pinned to revision
`6e31162aba58a715666d3791566f42afdcfa62b2`. Raw file hashes, selected row indices,
labels and inference requests are retained in the local result manifests.

`scripts/evaluate_intents_wide.py` prepares, runs and recounts the complete-label
experiment. The earlier `scripts/evaluate_intents.py` preserves the separate
26-candidate-engine tournament baseline. Its results must not be relabeled as
single-decision full-label inference.

Results, exact commands, build/test logs and the frozen source are retained in
`results/intent-wide-20260923/`; the earlier grouped baseline is retained in
`results/intent200-20260923/`. Each accuracy uses all 200 examples before
abstention. Accepted accuracy, coverage, wrong accepted answers and latency are
reported separately. These samples are not full-dataset leaderboard results.

## Measured results

All four configurations completed both 200-row subsets without inference errors
or truncation. Every request included the entire official label set.

| Model | BANKING77 English | MASSIVE Korean | English / Korean p50 ms |
| --- | ---: | ---: | ---: |
| Gemma 4 E2B Q8_0 | 123/200 (61.5%) | 103/200 (51.5%) | 214.15 / 157.52 |
| Gemma 4 E4B Q8_0 | 130/200 (65.0%) | 143/200 (71.5%) | 371.72 / 277.05 |
| Gemma 4 E4B Q4_K_M | 130/200 (65.0%) | 138/200 (69.0%) | 383.61 / 287.00 |
| Qwen3-8B Q8_0 | 111/200 (55.5%) | 98/200 (49.0%) | 877.64 / 593.45 |

These Gemma checkpoints encode every used two-letter code as one token. Qwen3-8B
uses two tokens for four BANKING77 codes and two MASSIVE codes, requiring three
and two native prefix evaluations respectively. A separate 677-candidate Gemma
GPU test scored three-letter codes, including the genuinely two-token `AAB`,
with a joint log-probability difference of zero against fresh teacher forcing.
The 231-item legacy A–Z JevBench replay preserved every result and probability.
Default tests passed 53/53; the llama feature suite passed 63 with 19 ignored;
two targeted real-model CUDA tests were then executed and passed.

The full report (local artifact: `results/intent-wide-20260923/REPORT.md`, not committed) records raw accuracy,
accepted errors, coverage, confidence metrics, latency and immutable evidence.

## Qwen diagnosis

An exploratory paired control retained all 200 English inputs and label meanings
but rotated display order and code assignment by 38. Accuracy changed from
111/200 to 107/200, while **83/200 semantic predictions changed**: 23 correct
answers became wrong and 19 wrong answers became correct. This establishes
substantial display/code sensitivity, without isolating order from code priors.
The rotated score is a diagnostic and does not replace the original benchmark.

At candidate confidence of at least 90%, the original Qwen run had 58 errors among
163 English predictions and 68 among 160 Korean predictions. Two-token gold codes
occurred in only 14 English examples; most of the 89 English errors also occurred
with single-token gold codes. Token length alone cannot explain the errors.
The current label-name-only, zero-shot setup leaves fine-grained label semantics,
presentation effects and confidence calibration as separate improvement targets.
Any revised prompt or calibration should be chosen on development/calibration
data and evaluated on a separate untouched test set.

The diagnostic report (local artifact: `results/intent-wide-20260923/diagnostics/REPORT.md`, not committed) includes
per-example paired predictions and token-length transitions.
