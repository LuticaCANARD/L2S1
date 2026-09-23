use crate::{
    common::{read_json, read_jsonl, write_json},
    kaggle_ag_news::wilson,
    kaggle_airline::{LABELS, MODELS, threshold_counts},
};
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
    #[arg(long)]
    folder: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    audit: Option<PathBuf>,
}

pub fn run(root: &Path, args: Args) -> Result<()> {
    let folder = args
        .folder
        .unwrap_or_else(|| root.join("results/kaggle-airline-20260922"));
    let output = args
        .output
        .unwrap_or_else(|| root.join("AIRLINE_BENCHMARK_RESULTS.md"));
    let audit_path = args
        .audit
        .unwrap_or_else(|| folder.join("comparison-audit.json"));
    let manifest = read_json(&folder.join("selection.json"))?;
    let metadata = read_json(&folder.join("kaggle-metadata.json"))?;
    let version = &metadata[0]["currentVersionNumber"];
    let mut lines=vec!["# Airline sentiment: seven-model comparison".to_owned(),String::new(),
        format!("Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment), Kaggle version {version}, attributed to CrowdFlower / Figure Eight. The data card specifies CC BY-NC-SA 4.0. Dataset text remains in local results."),String::new(),
        "Task: classify sentiment toward an airline or its service as negative, neutral, or positive. The fixed prompt uses tweet text only; labels and annotator metadata stay separate.".to_owned(),String::new(),
        "## Data and protocol".to_owned(),String::new(),
        format!("The source contains {} rows. Excluded duplicate rows: {}; conflicting-label rows: {}. The balanced split has 400 fit and 400 validation examples, with 134 negative, 133 neutral, and 133 positive examples in each.",
            manifest["source_rows"],manifest["excluded"]["duplicate_rows"],manifest["excluded"]["conflicting_label_rows"]),String::new(),
        "The seven local GGUF models use the same frozen requests, fresh CUDA execution, context 2048, batch 256, four CPU threads, FlashAttention off, and one warmup. Candidate-only temperatures are fitted on the fit split; validation does not select the temperature. Candidate mass remains uncalibrated.".to_owned(),String::new(),
        "## Model comparison at calibrated threshold 0.6".to_owned(),String::new(),
        "| Model file | Raw top-1 | Correct / wrong / abstain | Coverage | Accepted accuracy | 400-item inference (s) | Temperature |".to_owned(),
        "|---|---:|---|---:|---:|---:|---:|".to_owned()];
    let mut reports = BTreeMap::new();
    let mut audits = serde_json::Map::new();
    for (name, file) in MODELS {
        let path = folder.join(name).join("evaluation.json");
        if !path.exists() {
            lines.push(format!(
                "| {file} | pending or failed | — | — | — | — | — |"
            ));
            continue;
        }
        let report = read_json(&path)?;
        let threshold = &report["thresholds"][0];
        let percent = |v: &Value| {
            v.as_f64()
                .map_or_else(|| "n/a".to_owned(), |v| format!("{:.2}%", v * 100.0))
        };
        let temperature = report["temperature"]
            .as_f64()
            .context("missing temperature")?;
        let boundary = if temperature >= 99.99 {
            " (upper bound)"
        } else if temperature <= 0.050001 {
            " (lower bound)"
        } else {
            ""
        };
        lines.push(format!(
            "| {file} | {} | {} / {} / {} | {} | {} | {:.3} | {:.4}{boundary} |",
            percent(&report["raw"]["raw_top1"]),
            threshold["correct"],
            threshold["wrong"],
            threshold["abstained"],
            percent(&threshold["coverage"]),
            percent(&threshold["accepted_accuracy"]),
            report["timing"]["total_ms"]
                .as_f64()
                .context("missing total_ms")?
                / 1000.0,
            temperature
        ));
        let rows = read_jsonl(&folder.join(name).join("validation-results.jsonl"))?;
        let ids = rows
            .iter()
            .map(|r| r["id"].as_str().context("missing ID"))
            .collect::<Result<HashSet<_>>>()?;
        let expected = manifest["splits"]["validation"]["ids"]
            .as_array()
            .context("missing IDs")?
            .iter()
            .map(|v| v.as_str().context("invalid ID"))
            .collect::<Result<HashSet<_>>>()?;
        ensure!(
            rows.len() == 400 && ids.len() == 400 && ids == expected,
            "validation IDs differ"
        );
        let mut confusion = LABELS
            .iter()
            .map(|label| ((*label).to_owned(), BTreeMap::<String, usize>::new()))
            .collect::<BTreeMap<_, _>>();
        let (
            mut mass_rejections,
            mut selected_correct,
            mut selected_wrong,
            mut selected_abstain,
            mut raw_correct,
        ) = (0, 0, 0, 0, 0);
        for row in &rows {
            let result = &row["response"]["results"][0];
            ensure!(
                result["truncated"] == false && result["id"] == "airline_sentiment",
                "invalid result"
            );
            let scores = result["scores"].as_array().context("missing scores")?;
            let top = scores
                .iter()
                .max_by(|a, b| {
                    a["option_probability"]
                        .as_f64()
                        .unwrap_or(f64::NAN)
                        .total_cmp(&b["option_probability"].as_f64().unwrap_or(f64::NAN))
                })
                .context("empty scores")?;
            let best = top["option_probability"]
                .as_f64()
                .context("missing probability")?;
            let predicted = if scores
                .iter()
                .filter(|s| {
                    (s["option_probability"].as_f64().unwrap_or(f64::NAN) - best).abs() < 1e-12
                })
                .count()
                > 1
            {
                "tie"
            } else {
                top["id"].as_str().context("missing option ID")?
            };
            let id = row["id"].as_str().context("missing ID")?;
            let gold = manifest["labels"][id]
                .as_str()
                .context("missing gold label")?;
            *confusion
                .get_mut(gold)
                .context("unknown gold label")?
                .entry(predicted.to_owned())
                .or_default() += 1;
            raw_correct += usize::from(predicted == gold);
            mass_rejections += usize::from(
                result["candidate_mass"]
                    .as_f64()
                    .context("missing candidate_mass")?
                    < 0.05,
            );
            let selected = result["value"]["selected"].as_str();
            match selected {
                None => selected_abstain += 1,
                Some(value) if value == gold => selected_correct += 1,
                Some(_) => selected_wrong += 1,
            }
        }
        ensure!(
            raw_correct as f64 / 400.0
                == report["raw"]["raw_top1"]
                    .as_f64()
                    .context("missing raw top1")?,
            "raw correct mismatch"
        );
        ensure!(
            selected_correct
                == report["raw"]["accepted_correct"]
                    .as_u64()
                    .context("missing accepted correct")? as usize,
            "accepted correct mismatch"
        );
        ensure!(
            selected_correct + selected_wrong
                == report["raw"]["accepted"]
                    .as_u64()
                    .context("missing accepted")? as usize,
            "accepted count mismatch"
        );
        ensure!(
            selected_abstain == 400 - selected_correct - selected_wrong,
            "abstention mismatch"
        );
        for expected in report["thresholds"]
            .as_array()
            .context("missing thresholds")?
        {
            let actual = threshold_counts(
                &rows,
                &manifest["labels"],
                temperature,
                expected["threshold"]
                    .as_f64()
                    .context("missing threshold")?,
            )?;
            ensure!(
                ["correct", "wrong", "abstained"]
                    .iter()
                    .all(|key| actual[*key] == expected[*key])
                    && ["threshold", "coverage", "accepted_accuracy"]
                        .iter()
                        .all(
                            |key| match (actual[*key].as_f64(), expected[*key].as_f64()) {
                                (None, None) => true,
                                (Some(a), Some(b)) => (a - b).abs() < 1e-12,
                                _ => false,
                            }
                        ),
                "threshold recount mismatch for {name}: actual={actual} expected={expected}"
            );
        }
        audits.insert(name.to_owned(),json!({"total":400,"raw_correct":raw_correct,"raw_confusion":confusion,
            "mass_rejections":mass_rejections,"raw_correct_wilson95":wilson(raw_correct,400),
            "default_policy":{"correct":selected_correct,"wrong":selected_wrong,"abstained":selected_abstain},
            "complete_no_errors_or_truncation":true}));
        reports.insert(name, report);
    }
    lines.extend([String::new(),"Inference timing is one measured pass, excluding model load, warmup and file output. Different quantizations and memory pressure limit direct speed comparisons.".to_owned(),String::new(),
        "## Threshold sweep".to_owned(),String::new(),"| Model | Threshold | Correct | Wrong | Abstain | Coverage | Accepted accuracy |".to_owned(),
        "|---|---:|---:|---:|---:|---:|---:|".to_owned()]);
    for (name, report) in &reports {
        for threshold in report["thresholds"]
            .as_array()
            .context("missing thresholds")?
        {
            let acc = threshold["accepted_accuracy"]
                .as_f64()
                .map_or_else(|| "n/a".to_owned(), |v| format!("{:.2}%", v * 100.0));
            lines.push(format!(
                "| {name} | {:.1} | {} | {} | {} | {:.2}% | {acc} |",
                threshold["threshold"].as_f64().unwrap_or(0.0),
                threshold["correct"],
                threshold["wrong"],
                threshold["abstained"],
                threshold["coverage"].as_f64().unwrap_or(0.0) * 100.0
            ));
        }
    }
    lines.extend([String::new(),"## Provenance and scope".to_owned(),String::new(),
        format!("- Archive SHA256: `{}`. Frozen selection and labels: `selection.json`.",manifest["archive_sha256"].as_str().unwrap_or("?")),
        "- The comparison uses a balanced local split, not an official test split. Exact normalized duplicates and conflicting labels were excluded; near duplicates and pretraining overlap remain possible.".to_owned(),
        "- `comparison-audit.json` independently recounts raw predictions, default-policy outcomes, and calibrated thresholds.".to_owned(),String::new()]);
    write_json(&audit_path, &Value::Object(audits))?;
    fs::write(&output, lines.join("\n") + "\n")?;
    println!("reported {} completed models", reports.len());
    Ok(())
}
