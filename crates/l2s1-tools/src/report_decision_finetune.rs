use crate::{
    calibrate_ag_news::{fit_temperature, metrics, probabilities},
    common::{read_json, read_jsonl, write_json},
    kaggle_airline::threshold_counts,
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
    run: PathBuf,
    #[arg(long, num_args=1.., default_values_t=vec!["base".to_owned(),"lora".to_owned()])]
    prefixes: Vec<String>,
}

pub fn prediction(row: &Value) -> Result<Option<String>> {
    let scores = row["response"]["results"][0]["scores"]
        .as_array()
        .context("missing scores")?;
    let best = scores
        .iter()
        .map(|s| s["raw_logit"].as_f64().context("missing logit"))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .fold(f64::NEG_INFINITY, f64::max);
    let winners = scores
        .iter()
        .filter(|s| (s["raw_logit"].as_f64().unwrap_or(f64::NAN) - best).abs() < 1e-12)
        .collect::<Vec<_>>();
    Ok(if winners.len() == 1 {
        Some(
            winners[0]["id"]
                .as_str()
                .context("missing option ID")?
                .to_owned(),
        )
    } else {
        None
    })
}

pub fn run(args: Args) -> Result<()> {
    let manifest = read_json(&args.data.join("selection.json"))?;
    let labels = &manifest["labels"];
    let mut summary = serde_json::Map::new();
    for prefix in args.prefixes {
        let mut rows = BTreeMap::new();
        for name in ["calibration", "test", "probe"] {
            let values = read_jsonl(&args.run.join(format!("{prefix}-{name}.jsonl")))?;
            let ids = values
                .iter()
                .map(|r| r["id"].as_str().context("missing ID"))
                .collect::<Result<HashSet<_>>>()?;
            let expected = manifest["splits"][name]["ids"]
                .as_array()
                .context("missing split IDs")?
                .iter()
                .map(|v| v.as_str().context("invalid ID"))
                .collect::<Result<HashSet<_>>>()?;
            ensure!(
                values.len() == ids.len() && ids == expected,
                "prediction IDs differ from frozen selection"
            );
            ensure!(
                values
                    .iter()
                    .all(|r| r["response"]["results"][0]["truncated"] == false),
                "truncated predictions"
            );
            rows.insert(name, values);
        }
        let calibration = &rows["calibration"];
        let tests = &rows["test"];
        let probes = &rows["probe"];
        let temperature = fit_temperature(calibration, labels)?;
        let mut groups = BTreeMap::<String, Vec<Option<String>>>::new();
        let mut probe_correct = 0;
        let mut tied_calls = 0;
        for row in probes {
            let id = row["id"].as_str().context("missing probe ID")?;
            let answer = prediction(row)?;
            probe_correct += usize::from(answer.as_deref() == labels[id].as_str());
            tied_calls += usize::from(answer.is_none());
            let base = id.rsplit_once("-probe-r").context("invalid probe ID")?.0;
            groups.entry(base.to_owned()).or_default().push(answer);
        }
        let mut ranked = tests.iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| {
            let confidence = |row: &Value| -> f64 {
                let scores = row["response"]["results"][0]["scores"].as_array().unwrap();
                let logits = scores
                    .iter()
                    .map(|s| s["raw_logit"].as_f64().unwrap())
                    .collect::<Vec<_>>();
                probabilities(&logits, temperature)
                    .unwrap()
                    .0
                    .into_iter()
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            confidence(b).total_cmp(&confidence(a))
        });
        let mut coverage = Vec::new();
        for fraction in [0.6, 0.8, 1.0] {
            let count = (tests.len() as f64 * fraction).round() as usize;
            let subset = ranked[..count]
                .iter()
                .map(|r| (*r).clone())
                .collect::<Vec<_>>();
            coverage.push(json!({"fraction":fraction,"accuracy":metrics(&subset,labels,temperature)?["raw_top1"],"n":count,
                "scope":"test confidence ranking, diagnostic only, mass gate not applied"}));
        }
        let mut times = tests
            .iter()
            .map(|r| {
                r.get("elapsed_ms")
                    .or_else(|| r.get("batch_elapsed_ms"))
                    .and_then(Value::as_f64)
                    .context("missing timing")
            })
            .collect::<Result<Vec<_>>>()?;
        times.sort_by(f64::total_cmp);
        let median = if times.len() % 2 == 0 {
            (times[times.len() / 2 - 1] + times[times.len() / 2]) / 2.0
        } else {
            times[times.len() / 2]
        };
        let thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
            .iter()
            .map(|v| threshold_counts(tests, labels, temperature, *v))
            .collect::<Result<Vec<_>>>()?;
        summary.insert(prefix,json!({"temperature":temperature,"raw":metrics(tests,labels,1.0)?,"calibrated":metrics(tests,labels,temperature)?,
            "thresholds":thresholds,
            "fixed_coverage_diagnostic":coverage,"probe":{"unique_cases":groups.len(),"calls":probes.len(),"correct":probe_correct,
                "consistent_cases":groups.values().filter(|v|v.iter().all(Option::is_some) && v.iter().collect::<HashSet<_>>().len()==1).count(),"tied_calls":tied_calls},
            "test_ties":tests.iter().map(prediction).collect::<Result<Vec<_>>>()?.iter().filter(|v|v.is_none()).count(),
            "timing":{"total_ms":times.iter().sum::<f64>(),"p50_ms":median,"scope":"forward timing; excludes startup, warmup, IO"}}));
    }
    let summary = Value::Object(summary);
    write_json(&args.run.join("comparison.json"), &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
