use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    evaluate_intents::top,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(ClapArgs)]
pub struct Args {
    root: PathBuf,
}

fn quantile(values: &[f64], fraction: f64) -> Result<f64> {
    ensure!(!values.is_empty(), "empty timings");
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let position = fraction * (values.len() - 1) as f64;
    let i = position.floor() as usize;
    Ok(values[i] + (values[(i + 1).min(values.len() - 1)] - values[i]) * (position - i as f64))
}

fn audit(root: &Path, dir: &Path) -> Result<Value> {
    let data = read_json(&root.join("prepared/datasets.json"))?;
    let gold = read_jsonl(&root.join("prepared/gold.jsonl"))?
        .into_iter()
        .map(|r| Ok((r["id"].as_str().context("gold ID")?.to_owned(), r)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let first_rows = read_jsonl(&dir.join("stage1-predictions.jsonl"))?;
    let final_rows = read_jsonl(&dir.join("final-predictions.jsonl"))?;
    let first = first_rows
        .iter()
        .map(|r| Ok((r["id"].as_str().context("stage ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    let final_rows_by_id = final_rows
        .iter()
        .map(|r| Ok((r["id"].as_str().context("final ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(
        first_rows.len() == 1200
            && first.len() == 1200
            && final_rows.len() == 400
            && final_rows_by_id.len() == 400,
        "prediction count mismatch"
    );
    let expected_first = gold
        .keys()
        .flat_map(|id| (0..3).map(move |i| format!("{id}:g{i}")))
        .collect::<HashSet<_>>();
    let expected_final = gold
        .keys()
        .map(|id| format!("{id}:final"))
        .collect::<HashSet<_>>();
    ensure!(
        first.keys().collect::<HashSet<_>>() == expected_first.iter().collect()
            && final_rows_by_id.keys().collect::<HashSet<_>>() == expected_final.iter().collect(),
        "prediction IDs differ"
    );
    let mut winners = HashMap::<String, String>::new();
    let mut selections = HashMap::<String, Option<String>>::new();
    for row in first_rows.iter().chain(&final_rows) {
        ensure!(row.get("error").is_none(), "runtime error");
        let result = &row["response"]["results"][0];
        ensure!(result["truncated"] == false, "truncated result");
        let scores = result["scores"].as_array().context("missing scores")?;
        let mut probabilities = BTreeMap::new();
        for score in scores {
            let id = score["id"].as_str().context("option ID")?;
            let p = score["option_probability"]
                .as_f64()
                .context("probability")?;
            ensure!(
                probabilities.insert(id.to_owned(), p).is_none(),
                "duplicate option"
            );
        }
        ensure!(
            (probabilities.values().sum::<f64>() - 1.0).abs() < 1e-9,
            "invalid probabilities"
        );
        let best = top(row)?.to_owned();
        let probability = probabilities[&best];
        let selected =
            if probability >= 0.8 && result["candidate_mass"].as_f64().context("mass")? >= 0.05 {
                Some(best.clone())
            } else {
                None
            };
        ensure!(
            result["value"]["selected"].as_str() == selected.as_deref(),
            "declared selection differs from policy"
        );
        let id = row["id"].as_str().unwrap().to_owned();
        winners.insert(id.clone(), best);
        selections.insert(id, selected);
    }
    let mut tallies = BTreeMap::<String, BTreeMap<&str, usize>>::new();
    let mut timings = BTreeMap::<String, Vec<f64>>::new();
    for (id, item) in &gold {
        let name = item["dataset"].as_str().context("dataset")?;
        let keys = (0..3).map(|i| format!("{id}:g{i}")).collect::<Vec<_>>();
        let final_id = format!("{id}:final");
        let row = final_rows_by_id
            .get(&final_id)
            .context("missing final result")?;
        let finalist_ids = row["response"]["results"][0]["scores"]
            .as_array()
            .context("final scores")?
            .iter()
            .map(|s| s["id"].as_str().unwrap_or(""))
            .collect::<Vec<_>>();
        ensure!(
            finalist_ids
                == keys
                    .iter()
                    .map(|key| winners[key].as_str())
                    .collect::<Vec<_>>(),
            "routed finalists differ"
        );
        for (i, key) in keys.iter().enumerate() {
            let scores = first[key]["response"]["results"][0]["scores"]
                .as_array()
                .context("group scores")?;
            ensure!(
                scores.iter().map(|s| &s["id"]).eq(data[name]["groups"][i]
                    .as_array()
                    .context("group labels")?
                    .iter()),
                "group label order differs"
            );
        }
        let picked = &winners[&final_id];
        let winning_group = keys
            .iter()
            .find(|key| winners[*key] == *picked)
            .context("winning group absent")?;
        let accepted = selections[winning_group].as_ref() == Some(picked)
            && selections[&final_id].as_ref() == Some(picked);
        let expected = item["expected"].as_str().context("expected label")?;
        let count = tallies.entry(name.to_owned()).or_default();
        for (key, value) in [
            ("total", true),
            ("correct", picked == expected),
            ("accepted", accepted),
            ("accepted_correct", accepted && picked == expected),
            ("gold_reached_final", finalist_ids.contains(&expected)),
        ] {
            *count.entry(key).or_default() += usize::from(value);
        }
        let elapsed = keys
            .iter()
            .map(|key| first[key]["elapsed_ms"].as_f64().unwrap_or(0.0))
            .sum::<f64>()
            + row["elapsed_ms"].as_f64().context("elapsed_ms")?;
        timings.entry(name.to_owned()).or_default().push(elapsed);
    }
    let summary = read_json(&dir.join("summary.json"))?;
    for (name, counts) in tallies {
        for (key, value) in counts {
            ensure!(
                summary[&name][key] == value,
                "summary tally differs: {name}/{key}"
            );
        }
        for (fraction, key) in [(0.5, "p50_ms"), (0.95, "p95_ms")] {
            ensure!(
                (summary[&name][key].as_f64().context("summary quantile")?
                    - quantile(&timings[&name], fraction)?)
                .abs()
                    < 1e-8,
                "latency quantile differs"
            );
        }
    }
    let manifest = read_json(&dir.join("manifest.json"))?;
    ensure!(
        manifest["prepared_manifest_sha256"] == sha256(&root.join("prepared/manifest.json"))?,
        "prepared manifest changed"
    );
    Ok(
        json!({"id":dir.file_name().context("run directory name")?.to_string_lossy(),"manifest":manifest,"datasets":summary}),
    )
}

pub fn run(args: Args) -> Result<()> {
    let root = args.root.canonicalize()?;
    let prepared = read_json(&root.join("prepared/manifest.json"))?;
    for (name, expected) in prepared["prepared_sha256"]
        .as_object()
        .context("prepared hashes")?
    {
        ensure!(
            sha256(&root.join("prepared").join(name))? == expected.as_str().context("hash")?,
            "prepared data changed"
        );
    }
    let plan = read_json(&root.join("plan.json"))?;
    let mut results = Vec::new();
    for row in plan.as_array().context("plan must be list")? {
        let id = row["id"].as_str().context("plan ID")?;
        let result = audit(&root, &root.join("runs").join(id))?;
        ensure!(
            result["manifest"]["model"] == *row,
            "run model identity differs from plan"
        );
        results.push(result);
    }
    ensure!(
        results
            .iter()
            .map(|r| r["manifest"]["evaluator_sha256"].as_str().unwrap_or(""))
            .collect::<HashSet<_>>()
            .len()
            == 1,
        "evaluator hash differs"
    );
    let script_hashes = results
        .iter()
        .filter_map(|r| r["manifest"]["script_sha256"].as_str())
        .collect::<HashSet<_>>();
    ensure!(script_hashes.len() <= 1, "legacy script hash differs");
    if let Some(hash) = script_hashes.iter().next() {
        let legacy = root.join("evaluate_intents.py");
        if legacy.exists() {
            ensure!(sha256(&legacy)? == **hash, "legacy run script changed");
        }
    }
    let data = json!({"independently_audited":true,"configurations":results.len(),"examples_per_model":400,
        "total_decisions":1200,"total_native_calls":4800,"results":results});
    write_json(&root.join("REPORT.json"), &data)?;
    let mut text=vec!["# BANKING77 English and MASSIVE Korean: 200 examples each".to_owned(),String::new(),
        "Zero-shot, two-stage tournament: every intent label appears in one of three fixed groups; their winners compete in a final call. Gold labels are separate from inference requests.".to_owned(),String::new(),
        "| Model | Dataset | Correct / 200 | Accuracy | Wilson 95% interval | p50 ms | p95 ms |".to_owned(),
        "| --- | --- | ---: | ---: | --- | ---: | ---: |".to_owned()];
    for result in data["results"].as_array().unwrap() {
        for name in ["banking77-en", "massive-ko"] {
            let s = &result["datasets"][name];
            let interval = s["wilson95"].as_array().context("Wilson interval")?;
            text.push(format!(
                "| {} | {name} | {} | {:.1}% | {:.1}%–{:.1}% | {:.2} | {:.2} |",
                result["id"].as_str().unwrap_or("?"),
                s["correct"],
                s["accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                interval[0].as_f64().unwrap_or(0.0) * 100.0,
                interval[1].as_f64().unwrap_or(0.0) * 100.0,
                s["p50_ms"].as_f64().unwrap_or(0.0),
                s["p95_ms"].as_f64().unwrap_or(0.0)
            ));
        }
    }
    text.extend([String::new(),"## Abstention and routing".to_owned(),String::new(),
        "Acceptance requires both the winning first-stage branch and final decision to pass the unchanged top-probability and candidate-mass policy. Stage probabilities are conditional on their candidate groups.".to_owned(),String::new(),
        "| Model | Dataset | Accepted | Accepted correct | Accepted wrong | Accepted accuracy | Coverage | Abstained | Gold reached final |".to_owned(),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned()]);
    for result in data["results"].as_array().unwrap() {
        for name in ["banking77-en", "massive-ko"] {
            let s = &result["datasets"][name];
            let accepted = s["accepted_accuracy"]
                .as_f64()
                .map_or_else(|| "N/A".to_owned(), |v| format!("{:.1}%", v * 100.0));
            text.push(format!(
                "| {} | {name} | {} | {} | {} | {accepted} | {:.1}% | {} | {}/200 |",
                result["id"].as_str().unwrap_or("?"),
                s["accepted"],
                s["accepted_correct"],
                s["accepted_wrong"],
                s["coverage"].as_f64().unwrap_or(0.0) * 100.0,
                s["abstained"],
                s["gold_reached_final"]
            ));
        }
    }
    text.extend([String::new(),"## Scope".to_owned(),String::new(),
        "- The frozen BANKING77 English and MASSIVE Korean public test subsets contain 200 sampled examples each. This report does not claim a full-label softmax or an official leaderboard score.".to_owned(),
        "- Timings sum four serial local Rust inference calls per example, excluding model loading and warmups. They are not online latency measurements.".to_owned(),
        "- `REPORT.json` retains the audited per-model counts, confusion pairs, provenance, and latency quantiles.".to_owned(),String::new()]);
    fs::write(root.join("REPORT.md"), text.join("\n"))?;
    println!("audited {} configurations", results.len());
    Ok(())
}
