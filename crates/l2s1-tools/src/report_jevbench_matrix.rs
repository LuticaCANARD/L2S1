use crate::common::{read_json, read_jsonl, write_json};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(ClapArgs)]
pub struct Args {
    matrix: PathBuf,
    #[arg(long)]
    report: PathBuf,
}

fn audit(dir: &Path) -> Result<Value> {
    let summary = read_json(&dir.join("summary.json"))?;
    let manifest = read_json(&dir.join("manifest.json"))?;
    let tasks = read_jsonl(&dir.join("tasks-with-gold.jsonl"))?
        .into_iter()
        .map(|t| Ok((t["id"].as_str().context("task ID")?.to_owned(), t)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let predictions = read_jsonl(&dir.join("predictions.jsonl"))?;
    ensure!(
        tasks.len() == 231 && predictions.len() == 231,
        "expected exactly 231 tasks and predictions"
    );
    let mut seen = HashSet::new();
    let mut correct = 0;
    let mut errors = 0;
    let mut brier = Vec::new();
    let mut bins = vec![Vec::<(f64, bool)>::new(); 10];
    let mut high = Vec::new();
    for prediction in predictions {
        let id = prediction["id"].as_str().context("prediction ID")?;
        ensure!(seen.insert(id.to_owned()), "duplicate prediction ID");
        let task = tasks.get(id).context("unknown prediction ID")?;
        if prediction
            .get("error")
            .is_some_and(|v| !v.is_null() && v != false && v != "")
        {
            errors += 1;
            continue;
        }
        let result = &prediction["response"]["results"][0];
        ensure!(result["truncated"] == false, "truncated inference");
        let scores = result["scores"].as_array().context("missing scores")?;
        let binary = task["question"]["type"] == "noul";
        let labels = task["labels"].as_array().context("missing labels")?;
        ensure!(scores.len() == labels.len(), "label count mismatch");
        let mut probs = BTreeMap::new();
        for (score, label) in scores.iter().zip(labels) {
            let raw = score["id"].as_str().context("missing option ID")?;
            let mapped = if binary {
                match raw {
                    "false" => "no",
                    "true" => "yes",
                    _ => raw,
                }
            } else {
                raw
            };
            ensure!(
                mapped == label.as_str().context("invalid label")?,
                "option order changed"
            );
            let p = score["option_probability"]
                .as_f64()
                .context("missing probability")?;
            ensure!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "invalid probability"
            );
            probs.insert(mapped.to_owned(), p);
        }
        ensure!(
            (probs.values().sum::<f64>() - 1.0).abs() <= 0.001,
            "probabilities do not sum to one"
        );
        let chosen = probs
            .iter()
            .min_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)))
            .context("empty options")?
            .0;
        let gold = task["expected"]
            .as_str()
            .map(str::to_owned)
            .unwrap_or_else(|| task["expected"].to_string());
        let hit = chosen == &gold;
        correct += usize::from(hit);
        let confidence = probs.values().copied().fold(f64::NEG_INFINITY, f64::max);
        bins[((confidence * 10.0) as usize).min(9)].push((confidence, hit));
        if confidence >= 0.9 {
            high.push(hit);
        }
        brier.push(
            probs
                .iter()
                .map(|(label, p)| (p - f64::from(label == &gold)).powi(2))
                .sum::<f64>(),
        );
    }
    ensure!(seen.len() == tasks.len(), "missing prediction IDs");
    ensure!(
        correct
            == summary["n_correct"]
                .as_u64()
                .context("missing correct count")? as usize,
        "correct count mismatch"
    );
    ensure!(
        errors
            == summary["selective_policy"]["error"]
                .as_u64()
                .context("missing error count")? as usize,
        "error count mismatch"
    );
    if !brier.is_empty() {
        let measured = brier.iter().sum::<f64>() / brier.len() as f64;
        let ece = bins
            .iter()
            .map(|bin| {
                (bin.iter().map(|(p, _)| p).sum::<f64>()
                    - bin.iter().filter(|(_, hit)| *hit).count() as f64)
                    .abs()
            })
            .sum::<f64>()
            / brier.len() as f64;
        ensure!(
            (measured - summary["brier_mean"].as_f64().context("missing Brier")?).abs() < 1e-10,
            "Brier mismatch"
        );
        ensure!(
            (ece - summary["ece"]["ece"].as_f64().context("missing ECE")?).abs() < 1e-10,
            "ECE mismatch"
        );
    }
    let model = manifest["model_name"]
        .as_str()
        .context("missing model name")?;
    ensure!(
        summary["model_identities"] == json!([format!("{model}/l2s1")]),
        "model identity mismatch"
    );
    Ok(
        json!({"model":model,"correct":correct,"total":tasks.len(),"errors":errors,"accuracy":summary["accuracy"],
        "hard_accuracy":summary["tiers"]["hard"]["accuracy"],"easy_correct":summary["tiers"]["easy"]["n_correct"],
        "original_correct":summary["tiers"]["original"]["n_correct"],"hard_correct":summary["tiers"]["hard"]["n_correct"],
        "brier":summary["brier_mean"],"ece":summary["ece"]["ece"],
        "p50_ms":summary["latency"]["p50_s"].as_f64().context("missing p50")?*1000.0,
        "p95_ms":summary["latency"]["p95_s"].as_f64().context("missing p95")?*1000.0,
        "peak_gpu_mib":summary["gpu_memory"]["peak_board_used_mib"],"selective":summary["selective_policy"],
        "confidence_ge_90":{"n":high.len(),"accuracy":(!high.is_empty()).then(||high.iter().filter(|hit|**hit).count() as f64/high.len() as f64)},
        "requests_sha256":manifest["requests_sha256"],"evaluator_sha256":manifest["evaluator_sha256"],"independently_verified":true}),
    )
}

pub fn run(args: Args) -> Result<()> {
    let plan = read_json(&args.matrix.join("plan.json"))?;
    let rows = plan.as_array().context("invalid plan")?;
    let status = read_json(&args.matrix.join("matrix-status.json"))?;
    let statuses = status
        .as_array()
        .context("invalid status")?
        .iter()
        .map(|r| Ok((r["id"].as_str().context("status ID")?.to_owned(), r)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let mut completed = Vec::new();
    let mut failures = Vec::new();
    let mut pending = Vec::new();
    for row in rows {
        let id = row["id"].as_str().context("plan ID")?;
        let dir = args.matrix.join(id);
        if dir.join("summary.json").exists() {
            let manifest = read_json(&dir.join("manifest.json"))?;
            if let Some(hash) = row.get("sha256") {
                ensure!(
                    manifest["model_sha256"] == *hash,
                    "model differs from pinned download"
                );
            }
            ensure!(
                manifest["model_name"]
                    == Path::new(row["path"].as_str().context("model path")?)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .as_ref(),
                "model name mismatch"
            );
            ensure!(
                manifest
                    .get("runtime_environment_overrides")
                    .unwrap_or(&json!({}))
                    == row.get("environment").unwrap_or(&json!({})),
                "runtime environment mismatch"
            );
            let mut measured = audit(&dir)?;
            measured["configuration_id"] = json!(id);
            measured["runtime_environment_overrides"] =
                row.get("environment").cloned().unwrap_or_else(|| json!({}));
            completed.push(measured);
        } else if let Some(state) = statuses.get(id) {
            let log = [
                dir.join("inference.stderr.log"),
                args.matrix.join(format!("{id}.log")),
            ]
            .iter()
            .filter_map(|p| fs::read_to_string(p).ok())
            .map(|v| {
                v.chars()
                    .rev()
                    .take(3000)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
            failures.push(json!({"model":id,"status":state["status"],"log_tail":log}));
        } else {
            pending.push(id.to_owned());
        }
    }
    ensure!(
        completed
            .iter()
            .map(|r| r["requests_sha256"].as_str().unwrap_or(""))
            .collect::<HashSet<_>>()
            .len()
            <= 1,
        "request hashes differ"
    );
    ensure!(
        completed
            .iter()
            .map(|r| r["evaluator_sha256"].as_str().unwrap_or(""))
            .collect::<HashSet<_>>()
            .len()
            <= 1,
        "evaluator hashes differ"
    );
    completed.sort_by(|a, b| {
        b["accuracy"]
            .as_f64()
            .unwrap_or(0.0)
            .total_cmp(&a["accuracy"].as_f64().unwrap_or(0.0))
            .then_with(|| {
                b["hard_accuracy"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .total_cmp(&a["hard_accuracy"].as_f64().unwrap_or(0.0))
            })
            .then_with(|| {
                a["p50_ms"]
                    .as_f64()
                    .unwrap_or(f64::INFINITY)
                    .total_cmp(&b["p50_ms"].as_f64().unwrap_or(f64::INFINITY))
            })
    });
    let confirmation_path = args.matrix.join("idle-confirmation.json");
    let confirmation = if confirmation_path.exists() {
        read_json(&confirmation_path)?
    } else {
        json!([])
    };
    let data = json!({"planned":rows.len(),"distinct_checkpoints":rows.iter().map(|r|r["path"].as_str().unwrap_or("")).collect::<HashSet<_>>().len(),
        "completed":completed,"failed":failures,"pending":pending,"idle_confirmation":confirmation});
    write_json(&args.report.with_extension("json"), &data)?;
    let mut lines=vec!["# JevBench multi-model measurements".to_owned(),String::new(),
        "Host: `100.66.64.91`, RTX 3060 12 GiB. Public JevBench: 231 items (48 Easy, 72 Original, 111 Hard).".to_owned(),String::new(),
        format!("Distinct GGUF checkpoints: {}. Runtime configurations attempted: {}. Scored: {}. Failed before complete scoring: {}. Pending: {}.",
            data["distinct_checkpoints"],rows.len(),completed.len(),failures.len(),pending.len()),String::new(),
        "| Model | Correct / 231 | Accuracy | Hard | ECE ↓ | p50 ms | p95 ms | Peak GPU MiB | Accepted wrong | Abstained | Errors |".to_owned(),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned()];
    for r in &completed {
        let label = format!(
            "{}{}",
            r["model"].as_str().unwrap_or("?"),
            if r["runtime_environment_overrides"]
                .get("GGML_CUDA_DISABLE_GRAPHS")
                .is_some()
            {
                " (CUDA Graphs off)"
            } else {
                ""
            }
        );
        lines.push(format!(
            "| {label} | {} | {:.2}% | {:.2}% | {:.4} | {:.2} | {:.2} | {:.0} | {} | {} | {} |",
            r["correct"],
            100.0 * r["accuracy"].as_f64().unwrap_or(0.0),
            100.0 * r["hard_accuracy"].as_f64().unwrap_or(0.0),
            r["ece"].as_f64().unwrap_or(0.0),
            r["p50_ms"].as_f64().unwrap_or(0.0),
            r["p95_ms"].as_f64().unwrap_or(0.0),
            r["peak_gpu_mib"].as_f64().unwrap_or(0.0),
            r["selective"]["wrong_accepted"],
            r["selective"]["abstained"],
            r["errors"]
        ));
    }
    lines.extend([String::new(),"## Method and limits".to_owned(),String::new(),
        "- Same frozen public dataset, Rust evaluator binary, legacy layout, fresh requests, context 8192, batch/ubatch 256, four threads, FlashAttention off, one warmup, and default abstention thresholds.".to_owned(),
        "- Embedded model templates and the existing Auto prompt profile are used. No generation of reasoning tokens, LoRA, learned head, calibration, or benchmark-based prompt tuning.".to_owned(),
        "- Any CUDA Graph override is named in the table and saved in the model manifest. Its latency belongs to that explicitly different runtime setting.".to_owned(),
        "- Accuracy is argmax over candidate probabilities before abstention. Accepted errors, abstentions, and probability calibration remain separate.".to_owned(),
        "- Native predictions were independently recounted against gold labels; Brier and ECE were recomputed. Requests and binary hashes match across scored models.".to_owned(),
        "- Latency is serial local inference, excluding loading and warmup. Downloads can overlap runs; these single-run timings are not an SLA or directly comparable to HTTP leaderboard timings.".to_owned(),
        "- Peak GPU memory is whole-board usage sampled every 200 ms, including baseline; short peaks may be missed.".to_owned(),
        "- This is a selected hardware/runtime feasibility matrix, not every published model or every quantization. Small models may run beyond their training context; logs retain warnings.".to_owned(),
        "- Public subset only: no claim of the official full-suite score or rank. Ranking on this dataset is model selection evidence, not an independent production validation.".to_owned(),String::new()]);
    let recovery = args.matrix.join("gpt-oss-graphs-comparison.json");
    if recovery.exists() {
        let item = read_json(&recovery)?;
        lines.extend(["## GPT-OSS memory recovery".to_owned(),String::new(),
            "The default GPT-OSS run ran out of GPU memory in `cudaGraphInstantiate` after 129 completed predictions. That partial run remains a failed configuration and is not assigned a full-suite score. The separate `GGML_CUDA_DISABLE_GRAPHS=1` run uses the same model, evaluator, requests, context and batch settings.".to_owned(),String::new(),
            format!("The {} shared predictions had {} argmax changes and a maximum candidate-probability delta of {}. This comparison covers the shared prefix only.",
                item["shared_predictions"],item["argmax_changes"],item["max_probability_delta"]),String::new()]);
    }
    if let Some(confirmations) = confirmation.as_array()
        && !confirmations.is_empty()
    {
        lines.extend(["## Confirmation after model downloads finished".to_owned(),String::new(),
            "Separate complete reruns of the leading checkpoint and the Gemma baseline; these timings are not substituted into the original matrix.".to_owned(),String::new(),
            "| Model | Correct / 231 | p50 ms | p95 ms | Max probability delta from matrix |".to_owned(),
            "| --- | ---: | ---: | ---: | ---: |".to_owned()]);
        for item in confirmations {
            lines.push(format!(
                "| {} | {} | {:.2} | {:.2} | {} |",
                item["model"].as_str().unwrap_or("?"),
                item["correct"],
                item["p50_ms"].as_f64().unwrap_or(0.0),
                item["p95_ms"].as_f64().unwrap_or(0.0),
                item["max_probability_delta"]
            ));
        }
        lines.push(String::new());
    }
    if !failures.is_empty() {
        lines.push("## Failed configurations".to_owned());
        lines.push(String::new());
        for f in &failures {
            lines.extend([
                format!("### {}", f["model"].as_str().unwrap_or("?")),
                String::new(),
                "```text".to_owned(),
                f["log_tail"].as_str().unwrap_or("").to_owned(),
                "```".to_owned(),
                String::new(),
            ]);
        }
    }
    if !pending.is_empty() {
        lines.push("## Pending".to_owned());
        lines.push(String::new());
        for id in &pending {
            lines.push(format!("- {id}"));
        }
        lines.push(String::new());
    }
    fs::write(&args.report, lines.join("\n"))?;
    println!(
        "scored={} failed={} pending={}",
        completed.len(),
        failures.len(),
        pending.len()
    );
    Ok(())
}
