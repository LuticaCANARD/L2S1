use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    evaluate_intents::quantile,
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
    #[arg(long)]
    grouped: PathBuf,
    #[arg(long)]
    legacy: PathBuf,
}

fn audit(root: &Path, out: &Path) -> Result<Value> {
    let tasks = read_jsonl(&root.join("prepared/gold.jsonl"))?
        .into_iter()
        .map(|r| Ok((r["id"].as_str().context("task ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    let labels = read_json(&root.join("prepared/datasets.json"))?;
    let manifest = read_json(&out.join("manifest.json"))?;
    ensure!(
        manifest["requests_sha256"] == sha256(&root.join("prepared/requests.jsonl"))?
            && manifest["prepared_manifest_sha256"]
                == sha256(&root.join("prepared/manifest.json"))?,
        "frozen input changed"
    );
    let predictions = read_jsonl(&out.join("predictions.jsonl"))?;
    ensure!(
        predictions.len() == 400
            && predictions
                .iter()
                .map(|r| r["id"].as_str().unwrap_or(""))
                .collect::<HashSet<_>>()
                .len()
                == 400,
        "prediction IDs differ"
    );
    let mut measured = BTreeMap::<String, Vec<Value>>::new();
    for row in predictions {
        ensure!(row.get("error").is_none(), "runtime error");
        let id = row["id"].as_str().context("prediction ID")?;
        let task = tasks.get(id).context("unknown prediction ID")?;
        let name = task["dataset"].as_str().context("dataset")?;
        let result = &row["response"]["results"][0];
        ensure!(
            result["truncated"] == false
                && result["scoring_method"] == "code_sequence_conditional_softmax_v1",
            "invalid result"
        );
        let scores = result["scores"].as_array().context("missing scores")?;
        let candidates = labels[name]["labels"]
            .as_array()
            .context("missing labels")?;
        ensure!(
            scores.iter().map(|s| &s["id"]).eq(candidates.iter()),
            "label order changed"
        );
        let mut probs = BTreeMap::new();
        for score in scores {
            let id = score["id"].as_str().context("option ID")?;
            let p = score["option_probability"]
                .as_f64()
                .context("probability")?;
            ensure!(
                p.is_finite() && (0.0..=1.0).contains(&p),
                "invalid probability"
            );
            ensure!(probs.insert(id.to_owned(), p).is_none(), "duplicate option");
        }
        ensure!(
            (probs.values().sum::<f64>() - 1.0).abs() < 1e-9,
            "probability sum differs"
        );
        let picked = probs
            .iter()
            .min_by(|a, b| b.1.total_cmp(a.1).then_with(|| a.0.cmp(b.0)))
            .context("empty options")?;
        let confidence = *picked.1;
        let tied = probs
            .values()
            .filter(|p| (**p - confidence).abs() < 1e-12)
            .count()
            > 1;
        let accepted = confidence >= 0.8
            && result["candidate_mass"].as_f64().context("mass")? >= 0.05
            && !tied;
        ensure!(
            result["value"]["selected"].as_str()
                == if accepted {
                    Some(picked.0.as_str())
                } else {
                    None
                },
            "selected option violates policy"
        );
        let gold = task["expected"].as_str().context("gold label")?;
        let brier = probs
            .iter()
            .map(|(id, p)| (p - f64::from(id == gold)).powi(2))
            .sum::<f64>();
        measured.entry(name.to_owned()).or_default().push(
            json!({"correct":picked.0==gold,"accepted":accepted,"confidence":confidence,
            "brier":brier,"latency":row["elapsed_ms"]}),
        );
    }
    let summary = read_json(&out.join("summary.json"))?;
    for (name, records) in &measured {
        ensure!(records.len() == 200, "dataset count differs");
        let s = &summary[name];
        for (key, count) in [
            (
                "correct",
                records.iter().filter(|r| r["correct"] == true).count(),
            ),
            (
                "accepted",
                records.iter().filter(|r| r["accepted"] == true).count(),
            ),
            (
                "accepted_correct",
                records
                    .iter()
                    .filter(|r| r["accepted"] == true && r["correct"] == true)
                    .count(),
            ),
        ] {
            ensure!(s[key] == count, "summary differs: {name}/{key}");
        }
        let mut bins = vec![Vec::<&Value>::new(); 10];
        for r in records {
            bins[((r["confidence"].as_f64().unwrap() * 10.0) as usize).min(9)].push(r);
        }
        let ece = bins
            .iter()
            .map(|bin| {
                bin.iter()
                    .map(|r| r["confidence"].as_f64().unwrap() - f64::from(r["correct"] == true))
                    .sum::<f64>()
                    .abs()
            })
            .sum::<f64>()
            / 200.0;
        let brier = records
            .iter()
            .map(|r| r["brier"].as_f64().unwrap())
            .sum::<f64>()
            / 200.0;
        ensure!(
            (s["ece"].as_f64().context("summary ECE")? - ece).abs() < 1e-10
                && (s["brier"].as_f64().context("summary Brier")? - brier).abs() < 1e-10,
            "calibration metrics differ"
        );
        let times = records
            .iter()
            .map(|r| r["latency"].as_f64().unwrap())
            .collect::<Vec<_>>();
        for (key, q) in [("p50_ms", 0.5), ("p95_ms", 0.95)] {
            ensure!(
                (s[key].as_f64().context("summary quantile")? - quantile(&times, q)?).abs() < 1e-9,
                "latency differs"
            );
        }
    }
    Ok(
        json!({"id":out.file_name().context("run name")?.to_string_lossy(),"manifest":manifest,"datasets":summary}),
    )
}

fn parse_test_log(log: &str) -> Result<Value> {
    ensure!(!log.contains("FAILED"), "failed test log");
    let mut count = [0_usize; 3];
    let mut found = false;
    for line in log.lines() {
        if let Some(part) = line.split_once("test result: ok. ").map(|(_, part)| part) {
            let parts = part.split(';').collect::<Vec<_>>();
            if parts.len() < 3 {
                continue;
            }
            for (i, word) in ["passed", "failed", "ignored"].iter().enumerate() {
                let text = parts[i].trim();
                let value = text
                    .split_whitespace()
                    .next()
                    .context("test count")?
                    .parse::<usize>()?;
                ensure!(text.ends_with(word), "test log field order changed");
                count[i] += value;
            }
            found = true;
        }
    }
    ensure!(found, "no test result in log");
    Ok(json!({"passed":count[0],"failed":count[1],"ignored":count[2]}))
}

pub fn run(args: Args) -> Result<()> {
    let root = args.root.canonicalize()?;
    let manifest = read_json(&root.join("prepared/manifest.json"))?;
    for (name, hash) in manifest["prepared_sha256"]
        .as_object()
        .context("prepared hashes")?
    {
        ensure!(
            sha256(&root.join("prepared").join(name))? == hash.as_str().context("hash")?,
            "prepared file changed"
        );
    }
    ensure!(
        sha256(&root.join("prepared/gold.jsonl"))?
            == sha256(&args.grouped.join("prepared/gold.jsonl"))?,
        "grouped gold differs"
    );
    let plan = read_json(&root.join("plan.json"))?;
    let mut results = Vec::new();
    for model in plan.as_array().context("plan must be list")? {
        let id = model["id"].as_str().context("model ID")?;
        let result = audit(&root, &root.join("runs").join(id))?;
        ensure!(
            result["manifest"]["model"] == *model,
            "model identity differs"
        );
        results.push(result);
    }
    ensure!(
        !results.is_empty()
            && results
                .iter()
                .map(|r| r["manifest"]["evaluator_sha256"].as_str().unwrap_or(""))
                .collect::<HashSet<_>>()
                .len()
                == 1,
        "evaluator hashes differ"
    );
    let before = read_jsonl(&args.legacy.join("predictions.jsonl"))?
        .into_iter()
        .map(|r| Ok((r["id"].as_str().context("legacy ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    let after = read_jsonl(&root.join("legacy-regression/predictions.jsonl"))?
        .into_iter()
        .map(|r| Ok((r["id"].as_str().context("replay ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(
        before.len() == 231
            && after.len() == 231
            && before.keys().collect::<HashSet<_>>() == after.keys().collect(),
        "legacy replay coverage differs"
    );
    let mut delta = 0.0_f64;
    let mut changes = 0;
    for (id, a) in &before {
        let b = &after[id];
        let aa = &a["response"]["results"];
        let bb = &b["response"]["results"];
        changes += usize::from(aa != bb);
        let scores_a = aa[0]["scores"].as_array().context("scores")?;
        let scores_b = bb[0]["scores"].as_array().context("scores")?;
        ensure!(
            scores_a.len() == scores_b.len(),
            "legacy candidate count differs"
        );
        for (x, y) in scores_a.iter().zip(scores_b) {
            delta = delta.max(
                (x["option_probability"].as_f64().unwrap()
                    - y["option_probability"].as_f64().unwrap())
                .abs(),
            );
        }
    }
    ensure!(changes == 0 && delta == 0.0, "legacy replay changed");
    let mut tests = serde_json::Map::new();
    for name in [
        "default-tests",
        "tests",
        "native-test",
        "three-letter-native",
    ] {
        tests.insert(
            name.to_owned(),
            parse_test_log(&fs::read_to_string(root.join(format!("{name}.log")))?)?,
        );
    }
    ensure!(
        tests["native-test"]["passed"] == 1 && tests["three-letter-native"]["passed"] == 1,
        "native tests missing"
    );
    ensure!(
        fs::read_to_string(root.join("verification.exit-code"))?.trim() == "0",
        "verification exit code nonzero"
    );
    let proof = fs::read_to_string(root.join("three-letter-native.log"))?
        .lines()
        .filter(|line| line.contains("677 candidates verified"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    ensure!(proof.len() == 1, "three-letter proof missing");
    let report = json!({"results":results,"tests":tests,"legacy_regression":{"examples":231,"changed_results":changes,"max_probability_delta":delta},
        "three_letter_verification":proof[0],"independently_audited":true});
    write_json(&root.join("REPORT.json"), &report)?;
    let mut lines=vec!["# Full-label BANKING77 and MASSIVE evaluation".to_owned(),String::new(),
        "All 77 BANKING77 and 60 MASSIVE labels are compared in one decision per example using fixed-width answer codes. Gold labels remain separate from inference inputs.".to_owned(),String::new(),
        "| Model | Dataset | Correct / 200 | Accuracy | p50 ms | p95 ms | Accepted correct / accepted | Accepted wrong | Abstained |".to_owned(),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned()];
    for result in &results {
        for name in ["banking77-en", "massive-ko"] {
            let s = &result["datasets"][name];
            lines.push(format!(
                "| {} | {name} | {} | {:.1}% | {:.2} | {:.2} | {}/{} | {} | {} |",
                result["id"].as_str().unwrap_or("?"),
                s["correct"],
                s["accuracy"].as_f64().unwrap_or(0.0) * 100.0,
                s["p50_ms"].as_f64().unwrap_or(0.0),
                s["p95_ms"].as_f64().unwrap_or(0.0),
                s["accepted_correct"],
                s["accepted"],
                s["accepted_wrong"],
                s["abstained"]
            ));
        }
    }
    lines.extend([String::new(),"## Comparison with grouped baseline".to_owned(),String::new(),
        "The same sampled utterances and gold labels were used. The grouped baseline needed four calls per example; complete-label scoring uses one. Prompt and code mapping also changed, so this comparison does not isolate routing loss.".to_owned(),String::new(),
        "| Model | Dataset | Grouped correct | Full-label correct | Grouped p50 ms | Full-label p50 ms |".to_owned(),
        "| --- | --- | ---: | ---: | ---: | ---: |".to_owned()]);
    for result in &results {
        let id = result["id"].as_str().unwrap_or("");
        let old = args.grouped.join("runs").join(id).join("summary.json");
        if !old.exists() {
            continue;
        }
        let old = read_json(&old)?;
        for name in ["banking77-en", "massive-ko"] {
            let s = &result["datasets"][name];
            lines.push(format!(
                "| {id} | {name} | {}/200 | {}/200 | {:.2} | {:.2} |",
                old[name]["correct"],
                s["correct"],
                old[name]["p50_ms"].as_f64().unwrap_or(0.0),
                s["p50_ms"].as_f64().unwrap_or(0.0)
            ));
        }
    }
    lines.extend([String::new(),"## Verification".to_owned(),String::new(),
        format!("- 231-item legacy replay: {changes} changed results; maximum probability delta {delta}."),
        format!("- {}",proof[0]),
        "- `REPORT.json` retains the independently audited calibration, accuracy, acceptance and latency metrics, along with test logs and model provenance.".to_owned(),String::new()]);
    fs::write(root.join("REPORT.md"), lines.join("\n"))?;
    println!("audited {} configurations", results.len());
    Ok(())
}
