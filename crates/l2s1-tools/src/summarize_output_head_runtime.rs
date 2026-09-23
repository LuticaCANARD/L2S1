use crate::common::{read_jsonl, write_json};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    previous: PathBuf,
}

fn result_pairs(rows: &[Value]) -> Vec<(Value, Value)> {
    rows.iter()
        .map(|r| (r["id"].clone(), r["response"]["results"].clone()))
        .collect()
}
fn median(v: &mut [f64]) -> Result<f64> {
    ensure!(!v.is_empty(), "empty timing sample");
    v.sort_by(f64::total_cmp);
    Ok(if v.len().is_multiple_of(2) {
        (v[v.len() / 2 - 1] + v[v.len() / 2]) / 2.0
    } else {
        v[v.len() / 2]
    })
}
pub fn run(args: Args) -> Result<()> {
    let mut timing = serde_json::Map::new();
    for kind in ["base", "hidden"] {
        let runs = (0..3)
            .map(|n| read_jsonl(&args.root.join(format!("timing/{kind}-{n}.jsonl"))))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            runs.iter().all(|r| r.len() == 400),
            "expected 400 rows in each run"
        );
        let baseline = result_pairs(&runs[0]);
        ensure!(
            runs.iter().all(|r| result_pairs(r) == baseline),
            "repeated output differs"
        );
        let mut totals = Vec::new();
        let mut p50 = Vec::new();
        let mut p95 = Vec::new();
        let mut load = Vec::new();
        for rows in &runs {
            let mut values = rows
                .iter()
                .map(|r| r["elapsed_ms"].as_f64().context("missing elapsed_ms"))
                .collect::<Result<Vec<_>>>()?;
            totals.push(values.iter().sum::<f64>());
            p50.push(median(&mut values)?);
            p95.push(values[379]);
            load.push(rows[0]["load_ms"].clone());
        }
        let mut sorted = totals.clone();
        let middle = median(&mut sorted)?;
        timing.insert(
            kind.to_owned(),
            json!({"total_ms":totals,"median_total_ms":middle,"p50_ms":p50,"p95_ms":p95,
            "load_ms":load,"identical_repeated_results":true}),
        );
    }
    timing.insert(
        "overhead_percent".to_owned(),
        json!(
            (timing["hidden"]["median_total_ms"].as_f64().unwrap()
                / timing["base"]["median_total_ms"].as_f64().unwrap()
                - 1.0)
                * 100.0
        ),
    );
    timing.insert("scope".to_owned(),json!("Three isolated paired runs, 400 requests each, one warmup. Pair order base/hidden, hidden/base, base/hidden. Excludes model load, GGUF hash, warmup and serialization."));
    timing.insert(
        "median_request_ms".to_owned(),
        json!({"base":timing["base"]["median_total_ms"].as_f64().unwrap()/400.0,
        "hidden":timing["hidden"]["median_total_ms"].as_f64().unwrap()/400.0}),
    );
    let base = read_jsonl(&args.root.join("runtime/ag-news-base.jsonl"))?;
    let head = read_jsonl(&args.root.join("runtime/ag-news-selected.jsonl"))?;
    ensure!(
        base.len() == 400 && result_pairs(&base) == result_pairs(&head),
        "output-head run differs from base"
    );
    let old = read_jsonl(&args.previous)?;
    ensure!(
        result_pairs(&old) == result_pairs(&base),
        "previous binary results differ"
    );
    timing.insert("unrelated_task".to_owned(),json!({"cases":400,"identical_results_with_head":true,"identical_to_previous_binary_results":true}));
    let features = read_jsonl(&args.root.join("features/test.jsonl"))?;
    let plain = read_jsonl(&args.root.join("timing/base-0.jsonl"))?;
    ensure!(
        result_pairs(&features) == result_pairs(&plain),
        "feature extraction changed base predictions"
    );
    timing.insert(
        "feature_extraction".to_owned(),
        json!({"cases":400,"identical_base_results":true}),
    );
    let result = Value::Object(timing);
    write_json(&args.root.join("timings.json"), &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
