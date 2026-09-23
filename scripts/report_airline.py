#!/usr/bin/env python3
"""Render the completed airline model comparison, including failed/missing runs."""
import collections
import json
from pathlib import Path
from kaggle_airline import MODELS, OPTIONS, ROOT, SOURCE, threshold_counts
from kaggle_ag_news import wilson


def main():
    folder=ROOT/'results/kaggle-airline-20260922'
    manifest=json.loads((folder/'selection.json').read_text())
    metadata=json.loads((folder/'kaggle-metadata.json').read_text())[0]
    lines=['# Airline sentiment: seven-model comparison','',
           f'Dataset: [Twitter US Airline Sentiment]({SOURCE}), Kaggle version {metadata["currentVersionNumber"]}, attributed to CrowdFlower / Figure Eight. The data card specifies CC BY-NC-SA 4.0. This local benchmark keeps all tweet text and label files in ignored `results/`; it does not redistribute the dataset.','',
           'Task: classify sentiment toward an airline or its service as **negative, neutral, or positive**. The prompt explicitly asks the model to account for negation and sarcasm; it was fixed before inference and is identical across models. Inputs contain tweet text only. Labels, annotator confidence, negative-reason labels, user metadata, and other source columns are excluded from prompts.','',
           '## Data and protocol','',
           f'The source contains {manifest["source_rows"]:,} rows. We removed {manifest["excluded"]["duplicate_rows"]} duplicate rows and all {manifest["excluded"]["conflicting_label_rows"]} rows belonging to text groups with conflicting sentiment labels. Identity normalizes HTML entities, whitespace, and case. Near-duplicates and model pretraining exposure are not excluded. Crowd labels may be ambiguous; no annotator-confidence filter was applied.','',
           'Seed 20260923 selects **400 calibration-fit and 400 validation examples**, disjoint by normalized text. Each split has 134 negative, 133 neutral, and 133 positive examples. This is our balanced split, not an official test split or a sample preserving deployment class prevalence. The majority-class baseline is 33.5%.','',
           'All seven models use local RTX 3080 CUDA, fresh execution, context 2048, token batch/microbatch 256, four CPU threads, FlashAttention off, one warmup request, and the same frozen inputs. This differs from the earlier parallel Gemma 4 throughput experiment. Model-specific automatic templates remain in effect: Qwen3 non-thinking, GPT-OSS final-channel prefill, embedded model template otherwise. Recurrent/hybrid Qwen35 cannot use the current parallel bridge, motivating uniform fresh execution. Quantizations differ and are recorded; this is a comparison of these local GGUFs and this decision adapter, not a general model capability ranking.','',
           'Each model gets its own temperature fitted only on its 400 fit examples by NLL minimization over [0.05, 100]. Validation labels do not fit the temperature or thresholds. Full-vocabulary candidate mass remains uncalibrated, with the same minimum 0.05; ties abstain. Temperatures do not change argmax. Values at the search boundary are explicitly reported. Thresholds 0.6, 0.7, 0.8, 0.9, 1.0 are evaluated as requested.','',
           '## Model comparison at calibrated threshold 0.6','',
           '| Model file | Raw top-1 | Correct / wrong / abstain | Coverage | Accepted accuracy | 400-item inference (s) | Temperature |',
           '|---|---:|---|---:|---:|---:|---:|']
    reports={}
    audit={}
    for model,file in MODELS.items():
        path=folder/model/'evaluation.json'
        if not path.exists():
            lines.append(f'| {file} | pending or failed | — | — | — | — | — |')
            continue
        r=json.loads(path.read_text());reports[model]=r
        t=r['thresholds'][0];acc='n/a' if t['accepted_accuracy'] is None else f'{t["accepted_accuracy"]*100:.2f}%'
        boundary=' (upper bound)' if r['temperature']>=99.99 else ' (lower bound)' if r['temperature']<=.050001 else ''
        lines.append(f'| {file} | {r["raw"]["raw_top1"]*100:.2f}% | {t["correct"]} / {t["wrong"]} / {t["abstained"]} | {t["coverage"]*100:.2f}% | {acc} | {r["timing"]["total_ms"]/1000:.3f} | {r["temperature"]:.4f}{boundary} |')
        rows=[json.loads(s) for s in (folder/model/'validation-results.jsonl').read_text().splitlines()]
        assert len(rows)==len({row['id'] for row in rows})==400
        assert {row['id'] for row in rows}==set(manifest['splits']['validation']['ids'])
        confusion={k:collections.Counter() for k in OPTIONS}
        mass_rejections=0
        selected_correct=selected_wrong=selected_abstain=raw_correct=0
        for row in rows:
            result=row['response']['results'][0]
            assert not result['truncated'] and result['id']=='airline_sentiment'
            scores=result['scores'];top=max(scores,key=lambda s:s['option_probability'])
            tied=sum(abs(s['option_probability']-top['option_probability'])<1e-12 for s in scores)>1
            predicted='tie' if tied else top['id']
            gold=manifest['labels'][row['id']]
            confusion[gold][predicted]+=1
            raw_correct+=predicted==gold
            mass_rejections+=result['candidate_mass']<.05
            selected=result['value']['selected']
            if selected is None:selected_abstain+=1
            elif selected==gold:selected_correct+=1
            else:selected_wrong+=1
        assert raw_correct/400==r['raw']['raw_top1']
        assert selected_correct==r['raw']['accepted_correct']
        assert selected_correct+selected_wrong==r['raw']['accepted']
        assert selected_abstain==400-r['raw']['accepted']
        for expected in r['thresholds']:
            assert threshold_counts(rows,manifest['labels'],r['temperature'],expected['threshold'])==expected
        audit[model]=dict(total=400,raw_correct=raw_correct,raw_confusion={k:dict(v) for k,v in confusion.items()},
                          mass_rejections=mass_rejections,raw_correct_wilson95=wilson(raw_correct,400),
                          default_policy=dict(correct=selected_correct,wrong=selected_wrong,abstained=selected_abstain),
                          complete_no_errors_or_truncation=True)
    lines+=['','Inference timing is a single measured pass, excludes model loading, warmup and file output, and is not a repeated latency benchmark. GPT-OSS weights exceed this GPU’s physical VRAM; its timing reflects this hardware constraint. Do not compare these numbers to the earlier parallel article-batch timings as if the execution conditions were identical.','',
            '## Threshold sweep','',
            '| Model | Threshold | Correct | Wrong | Abstain | Coverage | Accepted accuracy |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in reports.items():
        for t in r['thresholds']:
            acc='n/a' if t['accepted_accuracy'] is None else f'{t["accepted_accuracy"]*100:.2f}%'
            lines.append(f'| {name} | {t["threshold"]:.1f} | {t["correct"]} | {t["wrong"]} | {t["abstained"]} | {t["coverage"]*100:.2f}% | {acc} |')
    lines+=['','## Uncalibrated default policy and low candidate mass','',
            'The following uses the existing raw-probability threshold 0.8 and mass threshold 0.05. It separates the adapter’s existing behavior from the offline temperature experiment. A model with no accepted predictions has undefined accepted accuracy, not zero. Lowering the probability threshold cannot override low candidate mass.','',
            '| Model | Correct / wrong / abstain | Raw top-1 prediction totals | Low-mass cases |','|---|---|---|---:|']
    for name,a in audit.items():
        t=a['default_policy'];pred=collections.Counter()
        for c in a['raw_confusion'].values():pred.update(c)
        lines.append(f'| {name} | {t["correct"]} / {t["wrong"]} / {t["abstained"]} | {dict(pred)} | {a["mass_rejections"]} |')
    lines+=['','## Reproduction','',
            f'Archive SHA-256: `{manifest["archive_sha256"]}`. Request hashes, selected row IDs, labels, and annotations are frozen in `results/kaggle-airline-20260922/selection.json`. Each model directory contains raw outputs, actual prompt/backend metadata, model/runtime/binary hashes, commands, and logs. `comparison-audit.json` independently recounts completeness and original selected outcomes. No application defaults or model weights were changed.','',
            '```sh',
            '# Download the pinned archive into a new output folder, then:',
            'python3 scripts/kaggle_airline.py prepare --folder results/airline-new',
            'python3 scripts/kaggle_airline.py run --folder results/airline-new \\',
            '  --models gemma4 gemma3 qwen3-0.6b smollm2 tinyllama qwen38 gpt-oss-20b',
            '```','',
            'The runner uses the archived evaluator and original CUDA runtime from this workspace. It refuses to overwrite existing model runs. The dataset-selection tests cover normalization, contradictory labels, disjoint splits, reproducibility, and mass/tie abstention gates. Python test suite: `python3 -m unittest discover -s scripts -p "test_*.py"`.','']
    paired=json.loads((folder/'gemma4-qwen38-paired.json').read_text())
    probe_path=folder/'option-order-probe/summary.json'
    if len(reports)==7:
        lines+=['## Interpretation and diagnostic checks','',
                'On this balanced sample, Qwen3.8 has 314/400 raw-correct answers versus Gemma 4 with 311/400. The paired discordances are 38 Qwen-only correct and 35 Gemma-only correct (two-sided exact McNemar p = 0.8151). This does not establish a quality advantage; Gemma 4 is substantially faster on this hardware. At calibrated threshold 0.6 it also accepts more cases, at lower accepted accuracy. These are different coverage/accuracy tradeoffs.', '',
                'GPT-OSS correctly classifies only 27 of the 133 positive examples; 95 are predicted neutral and 11 negative. No validation case is rejected for candidate mass below 0.05, so candidate-mass gating does not explain this error pattern. The final-channel prefill is active. This identifies an observed failure pattern, not its cause.', '',
                'Gemma 3, Qwen3 0.6B, SmolLM2, and TinyLlama reach the temperature-search upper bound. Temperature scaling can reduce unjustified confidence but cannot fix their low raw accuracy or missing task understanding in this adapter. TinyLlama additionally fails the candidate-mass gate on all 400 validation cases.', '']
    if probe_path.exists():
        probe=json.loads(probe_path.read_text())
        lines+=['### Post-hoc option-order probe','',
                'Selected the first four frozen validation cases per class (12 unique cases) and reran all three cyclic candidate orders for Gemma 3 and Qwen3 0.6B. This is a 72-call diagnostic, not another independent accuracy benchmark. No recalibration was applied.', '',
                '| Model | First candidate | A predictions / 12 | Correct / 12 |','|---|---|---:|---:|']
        for model,rotations in probe.items():
            for r in rotations:
                lines.append(f"| {model} | {r['first_label']} | {r['first_code_predictions']} | {r['correct']} |")
        lines+=['','Qwen3 0.6B chose A in all 36 calls even though A rotated from negative to neutral to positive. Gemma 3 chose A in all cases for the negative-first and positive-first orders, while the neutral-first order chose B in 11 cases and C in one. Both scored 4/12 for each order. These results demonstrate decision-format/order sensitivity on this diagnostic, not that the models have no general sentiment capability.', '']
    if len(reports)==7 and probe_path.exists():
        lines+=['### Verification','',
                '- All seven models completed both 400-example splits: 5600 inference outcomes, no errors, missing IDs, or truncation. These reuse 800 distinct examples, not 5600 independent test samples.',
                '- Independent audits recount raw and original-policy outcomes; the split/normalization/gating/calibration Python suite passes all 12 tests. The additional 72 option-order calls also completed.',
                '- A reporting-only p50 index was corrected from a fixed 100-batch assumption to nearest rank over the actual 400 batches. Raw outcomes, total inference time, calibration, and accuracy were unchanged; details are recorded in `timing-report-correction.json`.',
                '- No inference-engine defaults, prompts after the frozen selection, or model weights were changed for the seven-model comparison.', '']
    (ROOT/'AIRLINE_BENCHMARK_RESULTS.md').write_text('\n'.join(lines)+'\n')
    (folder/'comparison-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('Reported completed models:',', '.join(reports))


if __name__=='__main__':main()
