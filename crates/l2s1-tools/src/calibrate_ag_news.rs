use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    python_random::PythonRandom,
};
use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, Subcommand};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};
use zip::ZipArchive;

const ARCHIVE_SHA256: &str = "6dce0e4e9d48d02fc63649d853dd906ba031ce9a38113bc01ad12e20ad04e23a";
const CLASSES: [&str; 4] = ["world", "sports", "business", "science_technology"];

#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    Prepare {
        #[arg(long)]
        archive: PathBuf,
        #[arg(long)]
        template: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Fit {
        #[arg(long)]
        selection: PathBuf,
        #[arg(long)]
        fit_file: PathBuf,
        #[arg(long)]
        validation_file: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}

pub fn probabilities(logits: &[f64], temperature: f64) -> Result<(Vec<f64>, Vec<f64>)> {
    ensure!(
        temperature.is_finite()
            && temperature > 0.0
            && !logits.is_empty()
            && logits.iter().all(|v| v.is_finite()),
        "finite logits and positive finite temperature required"
    );
    let largest = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let shifted = logits
        .iter()
        .map(|v| (v - largest) / temperature)
        .collect::<Vec<_>>();
    let log_sum = shifted.iter().map(|v| v.exp()).sum::<f64>().ln();
    let logs = shifted.iter().map(|v| v - log_sum).collect::<Vec<_>>();
    Ok((logs.iter().map(|v| v.exp()).collect(), logs))
}

pub fn metrics(rows: &[Value], labels: &Value, temperature: f64) -> Result<Value> {
    ensure!(!rows.is_empty(), "empty predictions");
    let mut nll = 0.0;
    let mut brier = 0.0;
    let mut confidence = 0.0;
    let mut correct = 0;
    let mut accepted = 0;
    let mut accepted_correct = 0;
    let mut bins = vec![Vec::<(f64, bool)>::new(); 10];
    for row in rows {
        let result = &row["response"]["results"][0];
        ensure!(result["truncated"] == false, "truncated prediction");
        let scores = result["scores"].as_array().context("missing scores")?;
        let logits = scores
            .iter()
            .map(|s| s["raw_logit"].as_f64().context("missing raw_logit"))
            .collect::<Result<Vec<_>>>()?;
        let (p, logs) = probabilities(&logits, temperature)?;
        let id = row["id"].as_str().context("missing prediction ID")?;
        let label = labels[id].as_str().context("missing gold label")?;
        let gold = scores
            .iter()
            .position(|s| s["id"] == label)
            .context("gold option absent")?;
        let best = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        let hit = best == gold && p.iter().filter(|v| (*v - p[best]).abs() < 1e-12).count() == 1;
        let tied = p.iter().filter(|v| (*v - p[best]).abs() < 1e-12).count() > 1;
        nll -= logs[gold];
        brier += p
            .iter()
            .enumerate()
            .map(|(i, v)| (v - f64::from(i == gold)).powi(2))
            .sum::<f64>();
        confidence += p[best];
        correct += usize::from(hit);
        bins[((p[best] * 10.0) as usize).min(9)].push((p[best], hit));
        if p[best] >= 0.8
            && result["candidate_mass"]
                .as_f64()
                .context("missing candidate_mass")?
                >= 0.05
            && !tied
        {
            accepted += 1;
            accepted_correct += usize::from(hit);
        }
    }
    let n = rows.len() as f64;
    let ece = bins
        .iter()
        .map(|bin| {
            (bin.iter().map(|(p, _)| p).sum::<f64>()
                - bin.iter().filter(|(_, hit)| *hit).count() as f64)
                .abs()
        })
        .sum::<f64>()
        / n;
    Ok(
        json!({"n": rows.len(), "nll": nll/n, "brier": brier/n, "ece_10_equal_width": ece,
        "mean_confidence": confidence/n, "raw_top1": correct as f64/n, "accepted": accepted,
        "accepted_correct": accepted_correct, "accepted_accuracy": (accepted>0).then(|| accepted_correct as f64/accepted as f64),
        "coverage": accepted as f64/n, "correct_all": accepted_correct as f64/n}),
    )
}

pub fn fit_temperature(rows: &[Value], labels: &Value) -> Result<f64> {
    let (mut lo, mut hi) = (0.05_f64.ln(), 100.0_f64.ln());
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let loss = |t: f64| -> Result<f64> {
        metrics(rows, labels, t.exp())?["nll"]
            .as_f64()
            .context("missing NLL")
    };
    let (mut c, mut d) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fc, mut fd) = (loss(c)?, loss(d)?);
    for _ in 0..70 {
        if fc < fd {
            hi = d;
            d = c;
            fd = fc;
            c = hi - ratio * (hi - lo);
            fc = loss(c)?;
        } else {
            lo = c;
            c = d;
            fc = fd;
            d = lo + ratio * (hi - lo);
            fd = loss(d)?;
        }
    }
    Ok(((lo + hi) / 2.0).exp())
}

fn read_csv(archive: &mut ZipArchive<File>, name: &str) -> Result<Vec<BTreeMap<String, String>>> {
    let mut bytes = Vec::new();
    archive.by_name(name)?.read_to_end(&mut bytes)?;
    let text = String::from_utf8(bytes)?
        .trim_start_matches('\u{feff}')
        .to_owned();
    let mut reader = csv::Reader::from_reader(text.as_bytes());
    let names = reader
        .headers()?
        .iter()
        .map(str::to_owned)
        .collect::<Vec<_>>();
    reader
        .records()
        .map(|record| {
            Ok(names
                .iter()
                .cloned()
                .zip(record?.iter().map(str::to_owned))
                .collect())
        })
        .collect()
}

fn identity(row: &BTreeMap<String, String>) -> Result<String> {
    Ok(format!(
        "{}\n{}",
        row.get("Title")
            .context("missing Title")?
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" "),
        row.get("Description")
            .context("missing Description")?
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    )
    .to_lowercase())
}

fn prepare(archive: &Path, template: &Path, output: &Path) -> Result<()> {
    ensure!(
        sha256(archive)? == ARCHIVE_SHA256,
        "unexpected dataset archive"
    );
    fs::create_dir_all(output)?;
    ensure!(
        !output.join("selection.json").exists(),
        "selection already exists"
    );
    let mut zip = ZipArchive::new(File::open(archive)?)?;
    let train = read_csv(&mut zip, "train.csv")?;
    let test = read_csv(&mut zip, "test.csv")?;
    let mut seen = test.iter().map(identity).collect::<Result<HashSet<_>>>()?;
    let mut groups = vec![Vec::<(usize, BTreeMap<String, String>)>::new(); 4];
    for (i, row) in train.into_iter().enumerate() {
        let key = identity(&row)?;
        if seen.contains(&key)
            || row["Title"].trim().is_empty()
            || row["Description"].trim().is_empty()
        {
            continue;
        }
        seen.insert(key);
        let class = row["Class Index"].parse::<usize>()?;
        ensure!((1..=4).contains(&class), "unknown class");
        groups[class - 1].push((i + 1, row));
    }
    let mut rng = PythonRandom::new(20260922);
    let mut fit = Vec::new();
    let mut validation = Vec::new();
    for group in groups {
        ensure!(group.len() >= 200, "too few eligible records");
        let chosen = rng.sample_indices(group.len(), 200);
        fit.extend(chosen[..100].iter().map(|i| group[*i].clone()));
        validation.extend(chosen[100..].iter().map(|i| group[*i].clone()));
    }
    let template_text = fs::read_to_string(template)?;
    let line = template_text.lines().next().context("empty template")?;
    let (_, tail) = line
        .split_once("\"decisions\": ")
        .context("template must contain Python JSON decisions")?;
    let decisions = tail
        .strip_suffix("}}")
        .context("decisions must be the final request field")?;
    let parsed: Value = serde_json::from_str(line)?;
    ensure!(
        parsed["request"]["decisions"].is_array(),
        "invalid template decisions"
    );
    let mut labels = serde_json::Map::new();
    let mut splits = serde_json::Map::new();
    for (name, mut selected) in [("fit", fit), ("validation", validation)] {
        rng.shuffle(&mut selected);
        let path = output.join(format!("{name}.jsonl"));
        let mut file = File::create(&path)?;
        let mut ids = Vec::new();
        for (index, row) in selected {
            let id = format!("ag-news-train-{index:06}");
            labels.insert(
                id.clone(),
                json!(CLASSES[row["Class Index"].parse::<usize>()? - 1]),
            );
            writeln!(
                file,
                "{{\"id\": {}, \"request\": {{\"state\": {{\"title\": {}, \"description\": {}}}, \"decisions\": {}}}}}",
                serde_json::to_string(&id)?,
                serde_json::to_string(&row["Title"])?,
                serde_json::to_string(&row["Description"])?,
                decisions
            )?;
            ids.push(id);
        }
        file.flush()?;
        splits.insert(
            name.to_owned(),
            json!({"ids":ids, "request_sha256":sha256(&path)?}),
        );
    }
    write_json(
        &output.join("selection.json"),
        &json!({"seed":20260922,"archive_sha256":ARCHIVE_SHA256,
        "labels":labels,"splits":splits,"scope":"400 fit and 400 held-out validation from train.csv, deduplicated against all test.csv and each other. Fit NLL only, fixed acceptance 0.8, unchanged mass threshold 0.05. No prompt tuning."}),
    )
}

fn fit(selection: &Path, fit_file: &Path, validation_file: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "output exists: {}", output.display());
    let manifest = read_json(selection)?;
    let rows = [
        ("fit", read_jsonl(fit_file)?),
        ("validation", read_jsonl(validation_file)?),
    ];
    let backend = &rows[0].1.first().context("empty fit predictions")?["response"]["backend"];
    for (name, records) in &rows {
        let ids = records
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
            ids.len() == records.len() && ids == expected,
            "prediction IDs differ from frozen selection"
        );
        for row in records {
            let response = &row["response"];
            ensure!(
                response["backend"] == *backend
                    && response["policy"]
                        == json!({"min_top_probability":0.8,"min_candidate_mass":0.05}),
                "mixed backend or policy"
            );
            let results = response["results"].as_array().context("missing results")?;
            ensure!(
                results.len() == 1
                    && results[0]["id"] == "news_topic"
                    && results[0]["calibration_id"].is_null(),
                "invalid result"
            );
            let ids = results[0]["scores"]
                .as_array()
                .context("missing scores")?
                .iter()
                .map(|s| s["id"].as_str().unwrap_or(""))
                .collect::<Vec<_>>();
            ensure!(ids == CLASSES, "option order changed");
        }
    }
    let labels = &manifest["labels"];
    let temperature = fit_temperature(&rows[0].1, labels)?;
    let result = json!({"temperature":temperature,"fit_objective":"NLL","temperature_bounds":[0.05,100.0],
        "scope":"AG News candidate-only temperature. Does not calibrate candidate_mass or retrain the model; argmax unchanged. Validation labels not used for fitting.",
        "provenance": {selection.to_string_lossy().to_string():sha256(selection)?, fit_file.to_string_lossy().to_string():sha256(fit_file)?, validation_file.to_string_lossy().to_string():sha256(validation_file)?},
        "backend":backend,"evaluation": {"fit":{"before":metrics(&rows[0].1,labels,1.0)?,"after":metrics(&rows[0].1,labels,temperature)?},
            "validation":{"before":metrics(&rows[1].1,labels,1.0)?,"after":metrics(&rows[1].1,labels,temperature)?}}});
    write_json(output, &result)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}

pub fn run(args: Args) -> Result<()> {
    match args.command {
        Action::Prepare {
            archive,
            template,
            output,
        } => prepare(&archive, &template, &output),
        Action::Fit {
            selection,
            fit_file,
            validation_file,
            output,
        } => fit(&selection, &fit_file, &validation_file, &output),
    }
}
