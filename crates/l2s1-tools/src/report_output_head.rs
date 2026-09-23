use crate::{
    calibrate_ag_news::{metrics, probabilities},
    common::{read_json, read_jsonl, sha256, write_json},
    kaggle_airline::threshold_counts,
    report_decision_finetune::prediction,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    path::PathBuf,
};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    features: PathBuf,
    #[arg(long)]
    heads: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

fn transform(rows: &[Value], head: &Value, options: &[String]) -> Result<Vec<Value>> {
    let weights = head["weights"].as_array().context("missing weights")?;
    let bias = head["bias"].as_array().context("missing bias")?;
    ensure!(
        weights.len() == options.len() && bias.len() == options.len(),
        "head output dimensions differ"
    );
    let kind = head["feature_kind"]
        .as_str()
        .context("missing feature kind")?;
    rows.iter()
        .map(|row| {
            let vector = if kind == "hidden" {
                row["hidden"]
                    .as_array()
                    .context("missing hidden features")?
                    .iter()
                    .map(|v| v.as_f64().context("nonfinite hidden feature"))
                    .collect::<Result<Vec<_>>>()?
            } else {
                let scores = row["response"]["results"][0]["scores"]
                    .as_array()
                    .context("missing scores")?;
                options
                    .iter()
                    .map(|id| {
                        scores
                            .iter()
                            .find(|s| s["id"] == id.as_str())
                            .and_then(|s| s["raw_logit"].as_f64())
                            .context("missing option logit")
                    })
                    .collect::<Result<Vec<_>>>()?
            };
            ensure!(vector.iter().all(|v| v.is_finite()), "nonfinite feature");
            let logits = weights
                .iter()
                .zip(bias)
                .map(|(weights, bias)| {
                    let weights = weights.as_array().context("invalid weights")?;
                    ensure!(
                        weights.len() == vector.len(),
                        "head input dimension differs"
                    );
                    Ok(weights
                        .iter()
                        .zip(&vector)
                        .map(|(w, x)| w.as_f64().context("nonfinite weight").map(|w| w * x))
                        .collect::<Result<Vec<_>>>()?
                        .iter()
                        .sum::<f64>()
                        + bias.as_f64().context("invalid bias")?)
                })
                .collect::<Result<Vec<f64>>>()?;
            let mut copy = row.clone();
            copy.as_object_mut()
                .context("invalid row")?
                .remove("hidden");
            let scores = copy["response"]["results"][0]["scores"]
                .as_array_mut()
                .context("missing scores")?;
            for score in scores {
                let id = score["id"].as_str().context("missing option ID")?;
                let index = options
                    .iter()
                    .position(|name| name == id)
                    .context("unknown option")?;
                score["raw_logit"] = json!(logits[index]);
            }
            Ok(copy)
        })
        .collect()
}

fn ranked_accuracy(
    rows: &[Value],
    labels: &Value,
    temperature: f64,
    fraction: f64,
) -> Result<Value> {
    let mut ranked = rows
        .iter()
        .map(|row| {
            let logits = row["response"]["results"][0]["scores"]
                .as_array()
                .context("missing scores")?
                .iter()
                .map(|s| s["raw_logit"].as_f64().context("missing logit"))
                .collect::<Result<Vec<_>>>()?;
            let confidence = probabilities(&logits, temperature)?
                .0
                .into_iter()
                .fold(f64::NEG_INFINITY, f64::max);
            Ok((confidence, row.clone()))
        })
        .collect::<Result<Vec<(f64, Value)>>>()?;
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let count = (rows.len() as f64 * fraction).round() as usize;
    Ok(metrics(
        &ranked[..count]
            .iter()
            .map(|(_, row)| row.clone())
            .collect::<Vec<_>>(),
        labels,
        temperature,
    )?["raw_top1"]
        .clone())
}

fn exact_mcnemar(wins: usize, losses: usize) -> f64 {
    let n = wins + losses;
    if n == 0 {
        return 1.0;
    }
    let limit = wins.min(losses);
    let mut term = 2.0_f64.powi(-(n as i32));
    let mut sum = term;
    for j in 1..=limit {
        term *= ((n - j + 1) as f64) / (j as f64);
        sum += term;
    }
    (2.0 * sum).min(1.0)
}

pub fn run(args: Args) -> Result<()> {
    ensure!(!args.output.exists(), "output exists");
    let manifest = read_json(&args.data.join("selection.json"))?;
    let labels = &manifest["labels"];
    let mut rows = BTreeMap::new();
    for name in ["test", "probe"] {
        let path = args.data.join(format!("{name}.jsonl"));
        ensure!(
            sha256(&path)?
                == manifest["splits"][name]["sha256"]
                    .as_str()
                    .context("missing split hash")?,
            "frozen request changed"
        );
        let values = read_jsonl(&args.features.join(format!("{name}.jsonl")))?;
        let expected = manifest["splits"][name]["ids"]
            .as_array()
            .context("missing IDs")?;
        ensure!(
            values.iter().map(|v| &v["id"]).eq(expected.iter()),
            "feature ID order differs"
        );
        rows.insert(name, values);
    }
    let mut result = serde_json::Map::new();
    let mut predictions = BTreeMap::<String, BTreeMap<String, Option<String>>>::new();
    for kind in ["base_temperature", "logit_affine", "hidden"] {
        let head = read_json(&args.heads.join(format!("{kind}.json")))?;
        let options = head["options"]
            .as_array()
            .context("missing head options")?
            .iter()
            .map(|o| {
                o["id"]
                    .as_str()
                    .context("missing option ID")
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut mapped = BTreeMap::new();
        for (name, values) in &rows {
            mapped.insert(*name, transform(values, &head, &options)?);
        }
        let mut checks = serde_json::Map::new();
        for (name, expected) in &mapped {
            let path = args.runtime.join(format!("{kind}-{name}.jsonl"));
            if !path.exists() {
                continue;
            }
            let live = read_jsonl(&path)?;
            ensure!(
                live.iter()
                    .map(|r| &r["id"])
                    .eq(expected.iter().map(|r| &r["id"])),
                "runtime ID order differs"
            );
            let mut max_logit = 0.0_f64;
            let mut max_probability = 0.0_f64;
            for (actual, expected) in live.iter().zip(expected) {
                let actual = &actual["response"]["results"][0];
                let expected = &expected["response"]["results"][0];
                ensure!(
                    actual["calibration_id"] == head["id"]
                        && actual["candidate_mass"] == expected["candidate_mass"],
                    "runtime head identity or candidate mass differs"
                );
                let scores = expected["scores"].as_array().context("missing scores")?;
                let logits = scores
                    .iter()
                    .map(|s| s["raw_logit"].as_f64().context("missing logit"))
                    .collect::<Result<Vec<_>>>()?;
                let (p, _) = probabilities(
                    &logits,
                    head["temperature"]
                        .as_f64()
                        .context("missing temperature")?,
                )?;
                let live_scores = actual["scores"]
                    .as_array()
                    .context("missing runtime scores")?;
                for ((a, e), probability) in live_scores.iter().zip(scores).zip(p) {
                    ensure!(a["id"] == e["id"], "runtime option order differs");
                    max_logit = max_logit.max(
                        (a["raw_logit"].as_f64().context("runtime logit")?
                            - e["raw_logit"].as_f64().context("expected logit")?)
                        .abs(),
                    );
                    max_probability = max_probability.max(
                        (a["option_probability"]
                            .as_f64()
                            .context("runtime probability")?
                            - probability)
                            .abs(),
                    );
                }
            }
            ensure!(
                max_logit < 1e-5 && max_probability < 1e-6,
                "runtime head scores differ from Rust calculation"
            );
            checks.insert(
                (*name).to_owned(),
                json!({"max_logit_error":max_logit,"max_probability_error":max_probability}),
            );
        }
        let tests = &mapped["test"];
        let probes = &mapped["probe"];
        predictions.insert(
            kind.to_owned(),
            tests
                .iter()
                .map(|r| Ok((r["id"].as_str().context("ID")?.to_owned(), prediction(r)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        );
        let mut groups = BTreeMap::<String, Vec<Option<String>>>::new();
        let mut correct = 0;
        for row in probes {
            let id = row["id"].as_str().context("probe ID")?;
            let answer = prediction(row)?;
            correct += usize::from(answer.as_deref() == labels[id].as_str());
            groups
                .entry(
                    id.rsplit_once("-probe-r")
                        .context("invalid probe ID")?
                        .0
                        .to_owned(),
                )
                .or_default()
                .push(answer);
        }
        let temperature = head["temperature"]
            .as_f64()
            .context("missing temperature")?;
        let thresholds = [0.6, 0.7, 0.8, 0.9]
            .iter()
            .map(|v| threshold_counts(tests, labels, temperature, *v))
            .collect::<Result<Vec<_>>>()?;
        result.insert(kind.to_owned(),json!({"temperature":temperature,"raw":metrics(tests,labels,1.0)?,"calibrated":metrics(tests,labels,temperature)?,
            "thresholds":thresholds,"probe":{"unique_cases":groups.len(),"consistent_cases":groups.values().filter(|v|v.iter().all(Option::is_some)&&v.iter().collect::<HashSet<_>>().len()==1).count(),
                "correct":correct,"calls":probes.len()},
            "fixed_coverage_diagnostic":{"0.6":ranked_accuracy(tests,labels,temperature,0.6)?,"0.8":ranked_accuracy(tests,labels,temperature,0.8)?},
            "runtime_checks":checks}));
    }
    let base = &predictions["base_temperature"];
    for kind in ["logit_affine", "hidden"] {
        let mut wins = 0;
        let mut losses = 0;
        for (id, answer) in &predictions[kind] {
            let gold = labels[id].as_str().context("missing label")?;
            if base[id].as_deref() != Some(gold) && answer.as_deref() == Some(gold) {
                wins += 1;
            }
            if base[id].as_deref() == Some(gold) && answer.as_deref() != Some(gold) {
                losses += 1;
            }
        }
        result[kind]["paired_vs_base"] =
            json!({"wins":wins,"losses":losses,"mcnemar_exact_p":exact_mcnemar(wins,losses)});
    }
    let head_hashes = ["base_temperature", "logit_affine", "hidden"]
        .iter()
        .map(|kind| {
            Ok((
                (*kind).to_owned(),
                sha256(&args.heads.join(format!("{kind}.json")))?,
            ))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    result.insert("provenance".to_owned(),json!({"selection_sha256":sha256(&args.data.join("selection.json"))?,"head_sha256":head_hashes,
        "test_features_sha256":sha256(&args.features.join("test.jsonl"))?}));
    let result = Value::Object(result);
    write_json(&args.output, &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
