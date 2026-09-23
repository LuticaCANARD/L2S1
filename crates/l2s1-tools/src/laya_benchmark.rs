use crate::{
    common::{digest_bytes, read_json, read_jsonl, sha256, write_json, write_jsonl},
    parquet_data,
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args as ClapArgs, Subcommand};
use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

const PROBE_SHA: &str = "296623eb401e4f66c8e8fec450359c37a25eb65493e4e6bb1f1c4d002a00fb27";
const PROTOCOL: &str = "Only id/state/instruction/criteria enter inference. No gold, rationale or workflow metadata. Repeated identical cases measure preparation caching, not a persistent KV or answer cache. Probe grounding outputs also supply overconfidence checks; no second inference for the same check.";
const SCOPE: &str = "Local L2S1 inference, no official JevBench composite score. Per-case latency is full batch completion, not divided by questions/batch width. Repeat 0 alone is the dataset quality result; later repeats are cache diagnostics. Hard ECE uses ten equal-width bins against argmax labels, not soft-target calibration. Soft metrics cover binary/choice only, as upstream; ordinal MAE uses the probability-weighted level. AUROC uses tie-aware ranks. No Platt fitting or evaluation-set training. Probe thresholds are upstream heuristics, not logical guarantees; stability also changes rubric wording. Overconfidence uses L2S1 entropy confidence and is not assumed identical to Laya confidence.";
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Fetch {
        #[arg(long)]
        suite: String,
        #[arg(long)]
        output: PathBuf,
    },
    Prepare {
        #[arg(long)]
        suite: String,
        #[arg(long)]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 0)]
        limit: usize,
        #[arg(long, default_value_t = 1)]
        repeats: usize,
    },
    Score {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        predictions: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Compare {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        baseline: PathBuf,
        #[arg(long)]
        candidate: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Run {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        evaluator: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long)]
        gpu_layers: Option<usize>,
        #[arg(long, default_value_t = 0)]
        cpu_moe_layers: usize,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value_t = 256)]
        batch: usize,
        #[arg(long, default_value_t = 8)]
        threads: usize,
        #[arg(long, default_value = "auto")]
        model_load_mode: String,
        #[arg(long, default_value = "fresh")]
        execution_mode: String,
        #[arg(long, default_value = "legacy")]
        prompt_layout: String,
        #[arg(long, default_value_t = 4)]
        parallel_width: usize,
        #[arg(long, default_value_t = 1)]
        request_batch_size: usize,
        #[arg(long, default_value_t = 0)]
        cache_bytes: usize,
        #[arg(long, default_value_t = 128)]
        cache_entries: usize,
        #[arg(long, default_value_t = 14400)]
        timeout: u64,
    },
}
fn source_specs(suite: &str) -> Result<Vec<(&'static str, &'static str, &'static str)>> {
    let benchmark = (
        "Luni/laya-jev-benchmark",
        "d75081b2a4b2ad772793d6a7f5f5b4fdca00d557",
    );
    let typed = (
        "LocalLLaMA/typed-decisions",
        "c76749ec58bd8c3d2ea706b31c333a9059c38f90",
    );
    let phish = (
        "AreLit/PhishNChips",
        "89afcc39610084298c4679159cb2e27d9ffffa46",
    );
    Ok(match suite {
        "probes" => vec![(benchmark.0, benchmark.1, "bench/probe.py")],
        "typed" => vec![
            (typed.0, typed.1, "all/test-00000-of-00001.parquet"),
            (benchmark.0, benchmark.1, "bench/eval.py"),
        ],
        "phish" => vec![
            (phish.0, phish.1, "core_emails.csv"),
            (benchmark.0, benchmark.1, "bench/bench_phish.py"),
        ],
        _ => bail!("unknown Laya suite {suite}"),
    })
}
fn fetch(suite: &str, output: &Path) -> Result<()> {
    let specs = source_specs(suite)?;
    fs::create_dir(output)?;
    let mut files = Vec::new();
    for (repo, revision, name) in specs {
        let local = Path::new(name).file_name().unwrap().to_str().unwrap();
        let dest = output.join(local);
        let url = format!("https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}");
        let status = Command::new("curl")
            .args([
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--max-time",
                "120",
                "--output",
            ])
            .arg(&dest)
            .arg(&url)
            .status()?;
        ensure!(status.success(), "download failed: {url}");
        files.push(json!({"repo":repo,"revision":revision,"path":name,"local":local,"url":url,"sha256":sha256(&dest)?}));
    }
    write_json(
        &output.join("sources.json"),
        &json!({"suite":suite,"files":files}),
    )
}
fn decoded(value: &Value) -> Result<Value> {
    if let Some(s) = value.as_str() {
        Ok(serde_json::from_str(s)?)
    } else {
        Ok(value.clone())
    }
}
fn nonempty(value: &Value) -> Result<&str> {
    let text = value.as_str().context("Expected nonempty text")?;
    ensure!(!text.trim().is_empty(), "Expected nonempty text");
    Ok(text)
}
#[derive(Deserialize)]
struct Question {
    #[serde(rename = "type")]
    kind: String,
    instructions: String,
    criteria: Option<Criteria>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Criteria {
    Map(IndexMap<String, String>),
    List(Vec<String>),
}
fn decision(id: &str, q: &Question) -> Result<Value> {
    ensure!(
        !id.trim().is_empty() && !q.instructions.trim().is_empty(),
        "Expected nonempty text"
    );
    let kind = match (&*q.kind, &q.criteria) {
        ("noul", None) => json!({"type":"binary","false_label":"No","true_label":"Yes"}),
        ("noul", Some(Criteria::Map(criteria))) => {
            ensure!(
                criteria.len() == 2
                    && criteria.contains_key("false")
                    && criteria.contains_key("true"),
                "Invalid binary criteria"
            );
            json!({"type":"binary","false_label":criteria["false"],"true_label":criteria["true"]})
        }
        ("choice", Some(Criteria::Map(criteria))) => {
            ensure!(criteria.len() >= 2, "Invalid choice criteria");
            let options = criteria
                .iter()
                .map(|(id, text)| json!({"id":id,"criterion":text}))
                .collect::<Vec<_>>();
            json!({"type":"choice","options":options})
        }
        ("score", Some(Criteria::List(criteria))) => {
            ensure!(criteria.len() >= 2, "Invalid ordinal criteria");
            let levels = criteria
                .iter()
                .enumerate()
                .map(|(i, text)| json!({"id":i.to_string(),"criterion":text,"value":i}))
                .collect::<Vec<_>>();
            json!({"type":"ordinal","levels":levels})
        }
        _ => bail!("Unknown or malformed question type"),
    };
    Ok(json!({"id":id,"instruction":q.instructions,"kind":kind}))
}
fn typed_cases(rows: &[Value]) -> Result<Vec<Value>> {
    rows.iter().enumerate().map(|(i,row)|{
        let state=decoded(&row["state"])?;
        let questions=row["questions"].as_str().context("questions JSON")?;
        let questions:IndexMap<String,Question>=serde_json::from_str(questions)?;
        let gold=decoded(&row["gold"])?;
        ensure!(!questions.is_empty() && questions.keys().collect::<HashSet<_>>()==gold.as_object().context("gold map")?.keys().collect(),"Question/gold IDs differ");
        let decisions=questions.iter().map(|(id,q)|decision(id,q)).collect::<Result<Vec<_>>>()?;
        Ok(json!({"id":format!("typed:{i}"),"request":{"state":state,"decisions":decisions},
            "gold":gold,"workflow":row.get("workflow").cloned().unwrap_or(json!("unknown")),"suite":"typed"}))
    }).collect()
}
fn phish_cases(path: &Path) -> Result<Vec<Value>> {
    let mut reader = csv::ReaderBuilder::new().from_path(path)?;
    let mut cases = Vec::new();
    let question = Question {
        kind: "noul".to_owned(),
        instructions: "Is this email a phishing or scam attempt?".to_owned(),
        criteria: None,
    };
    let d = decision("is_phishing", &question)?;
    for (i, row) in reader.deserialize::<HashMap<String, String>>().enumerate() {
        let row = row?;
        let email = row.get("email_content").context("email_content")?;
        let state = serde_json::from_str::<Value>(email).unwrap_or_else(|_| json!(email));
        let label = row
            .get("phish_label")
            .context("phish_label")?
            .parse::<u8>()?;
        ensure!(label <= 1, "Invalid phishing label");
        cases.push(
            json!({"id":format!("phish:{i}"),"request":{"state":state,"decisions":[d]},
            "gold":{"is_phishing":{"label":if label==1{"true"}else{"false"},"noul":label}},
            "workflow":"phishing","suite":"phish"}),
        );
    }
    Ok(cases)
}
fn prepare(suite: &str, source: &Path, output: &Path, limit: usize, repeats: usize) -> Result<()> {
    source_specs(suite)?;
    ensure!(repeats >= 1, "Repeats must be positive");
    let provenance = read_json(&source.join("sources.json"))?;
    ensure!(provenance["suite"] == suite, "Source suite mismatch");
    for item in provenance["files"].as_array().context("source files")? {
        ensure!(
            sha256(&source.join(nonempty(&item["local"])?))? == item["sha256"],
            "Source hash mismatch"
        );
    }
    let cases = match suite {
        "probes" => {
            ensure!(
                sha256(&source.join("probe.py"))? == PROBE_SHA,
                "Pinned probe definitions changed"
            );
            serde_json::from_str::<Vec<Value>>(include_str!("laya_probes.json"))?
        }
        "typed" => {
            let cases = typed_cases(&parquet_data::rows(
                &source.join("test-00000-of-00001.parquet"),
            )?)?;
            ensure!(
                cases.len() == 400
                    && cases
                        .iter()
                        .map(|c| c["request"]["decisions"].as_array().unwrap().len())
                        .sum::<usize>()
                        == 2000,
                "Pinned typed-decisions test split changed"
            );
            cases
        }
        "phish" => {
            let cases = phish_cases(&source.join("core_emails.csv"))?;
            ensure!(
                cases.len() == 2000
                    && cases
                        .iter()
                        .map(|c| c["gold"]["is_phishing"]["noul"].as_u64().unwrap() as usize)
                        .sum::<usize>()
                        == 1000,
                "Pinned balanced phishing core split changed"
            );
            cases
        }
        _ => unreachable!(),
    };
    let total = cases.len();
    let chosen = if limit > 0 {
        &cases[..limit.min(total)]
    } else {
        &cases[..]
    };
    let mut expanded = Vec::new();
    for case in chosen {
        for repeat in 0..repeats {
            let mut row = case.clone();
            let id = nonempty(&case["id"])?;
            row["base_id"] = json!(id);
            row["id"] = json!(format!("{id}/r{repeat}"));
            row["repeat"] = json!(repeat);
            expanded.push(row);
        }
    }
    fs::create_dir(output)?;
    write_jsonl(
        &output.join("requests.jsonl"),
        &expanded
            .iter()
            .map(|c| json!({"id":c["id"],"request":c["request"]}))
            .collect::<Vec<_>>(),
    )?;
    write_jsonl(&output.join("cases-with-gold.jsonl"), &expanded)?;
    let manifest = json!({"suite":suite,"sources":provenance,"available_cases":total,"logical_cases":chosen.len(),"repeats":repeats,
        "measured_cases":expanded.len(),"decisions":chosen.iter().map(|c|c["request"]["decisions"].as_array().unwrap().len()).sum::<usize>(),
        "limited":limit>0 && limit<total,"files":{"requests.jsonl":sha256(&output.join("requests.jsonl"))?,
            "cases-with-gold.jsonl":sha256(&output.join("cases-with-gold.jsonl"))?},
        "adapter_sha256":digest_bytes(include_bytes!("laya_benchmark.rs")),"ordering":"source order, immediate repeats per case","protocol":PROTOCOL});
    write_json(&output.join("manifest.json"), &manifest)
}
fn options(decision: &Value) -> Result<Vec<String>> {
    let kind = &decision["kind"];
    if kind["type"] == "binary" {
        return Ok(vec!["false".to_owned(), "true".to_owned()]);
    }
    let list = if kind["type"] == "choice" {
        &kind["options"]
    } else {
        &kind["levels"]
    };
    list.as_array()
        .context("options")?
        .iter()
        .map(|x| Ok(nonempty(&x["id"])?.to_owned()))
        .collect()
}
fn quantile(values: &[f64], q: f64) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let pos = (values.len() - 1) as f64 * q;
    let i = pos.floor() as usize;
    let frac = pos - i as f64;
    json!(values[i] + (values[(i + 1).min(values.len() - 1)] - values[i]) * frac)
}
fn mean(values: &[f64]) -> Value {
    if values.is_empty() {
        Value::Null
    } else {
        json!(values.iter().sum::<f64>() / values.len() as f64)
    }
}
fn distribution(decision: &Value, result: &Value) -> Result<IndexMap<String, f64>> {
    let labels = options(decision)?;
    let scores = result["scores"].as_array().context("scores")?;
    ensure!(
        result["id"] == decision["id"]
            && result["truncated"] == false
            && scores.len() == labels.len(),
        "Mismatched labels/decision or truncated input"
    );
    let mut p = IndexMap::new();
    for (score, label) in scores.iter().zip(labels) {
        ensure!(
            score["id"] == label,
            "Mismatched labels/decision or truncated input"
        );
        let value = score["option_probability"]
            .as_f64()
            .context("probability")?;
        ensure!(
            value.is_finite() && (0. ..=1.).contains(&value),
            "Invalid probability distribution"
        );
        p.insert(label, value);
    }
    ensure!(
        (p.values().sum::<f64>() - 1.).abs() <= 1e-6,
        "Invalid probability distribution"
    );
    let count = result["input_tokens"].as_u64().context("input_tokens")?;
    let reused = result["reused_prefix_tokens"]
        .as_u64()
        .context("reused_prefix_tokens")?;
    ensure!(reused < count, "Invalid token accounting");
    for field in ["candidate_mass", "entropy_confidence"] {
        let value = result[field].as_f64().context("evidence")?;
        ensure!(
            value.is_finite() && (0. ..=1. + 1e-6).contains(&value),
            "Invalid evidence: {field}"
        );
    }
    Ok(p)
}
fn score_decision(decision: &Value, result: &Value, gold: Option<&Value>) -> Result<Value> {
    let p = distribution(decision, result)?;
    let labels = p.keys().cloned().collect::<Vec<_>>();
    let kind = nonempty(&decision["kind"]["type"])?;
    let selected = if kind == "binary" {
        if p["true"] >= 0.5 {
            "true".to_owned()
        } else {
            "false".to_owned()
        }
    } else {
        p.iter()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
            .clone()
    };
    let value = &result["value"];
    ensure!(
        value["type"] == decision["kind"]["type"],
        "Typed value does not match the decision kind"
    );
    let accepted = if kind == "binary" {
        let true_prob = value["p_true"].as_f64().context("p_true")?;
        ensure!(
            true_prob.is_finite()
                && (true_prob - p["true"]).abs() <= 1e-6
                && (value["value"].is_null() || value["value"].is_boolean()),
            "Invalid binary value"
        );
        !value["value"].is_null()
    } else {
        ensure!(
            value["selected"].is_null() || p.contains_key(nonempty(&value["selected"])?),
            "Selected label outside candidate set"
        );
        !value["selected"].is_null()
    };
    let confidence = p.values().copied().fold(0., f64::max);
    let mut row = json!({"id":decision["id"],"probabilities":p,"predicted":selected,"confidence":confidence,
        "entropy_confidence":result["entropy_confidence"],"candidate_mass":result["candidate_mass"],
        "accepted":accepted,"input_tokens":result["input_tokens"],"reused_prefix_tokens":result["reused_prefix_tokens"]});
    let Some(gold) = gold else { return Ok(row) };
    let expected = if kind == "binary" {
        nonempty(&gold["label"])?.to_lowercase()
    } else {
        nonempty(&gold["label"])?.to_owned()
    };
    ensure!(
        p.contains_key(&expected),
        "Gold label outside candidate set"
    );
    row["expected"] = json!(expected);
    row["correct"] = json!(selected == expected);
    let mut gp = gold.get("probabilities").filter(|v| !v.is_null()).cloned();
    if kind == "binary" {
        let v = gold
            .get("noul")
            .or_else(|| gp.as_ref().and_then(|x| x.get("true")))
            .and_then(Value::as_f64);
        if let Some(v) = v {
            gp = Some(json!({"false":1.-v,"true":v}));
        }
    }
    row["brier_hard"] = json!(
        labels
            .iter()
            .map(|label| {
                let delta = p[label] - f64::from(label == &expected);
                delta * delta
            })
            .sum::<f64>()
    );
    if let Some(gp) = gp.filter(|_| kind != "ordinal") {
        let gp = gp.as_object().context("gold distribution")?;
        ensure!(
            gp.len() == labels.len() && labels.iter().all(|label| gp.contains_key(label)),
            "Invalid gold distribution"
        );
        let values = labels
            .iter()
            .map(|label| gp[label].as_f64().context("gold probability"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            values.iter().all(|v| v.is_finite() && *v >= 0.)
                && (values.iter().sum::<f64>() - 1.).abs() <= 1e-4,
            "Invalid gold distribution"
        );
        let total = values.iter().sum::<f64>();
        let targets = values.iter().map(|v| v / total).collect::<Vec<_>>();
        row["soft_accuracy"] = json!(
            labels
                .iter()
                .enumerate()
                .map(|(i, label)| p[label] * targets[i])
                .sum::<f64>()
        );
        row["brier_soft"] = json!(
            labels
                .iter()
                .enumerate()
                .map(|(i, label)| (p[label] - targets[i]).powi(2))
                .sum::<f64>()
        );
        row["tvd"] = json!(
            labels
                .iter()
                .enumerate()
                .map(|(i, label)| (p[label] - targets[i]).abs())
                .sum::<f64>()
                / 2.
        );
        row["kl"] = json!(
            labels
                .iter()
                .enumerate()
                .filter(|(i, _)| targets[*i] > 0.)
                .map(|(i, label)| targets[i]
                    * (targets[i] / p[label].max(1e-300)).clamp(1e-12, 1e4).ln())
                .sum::<f64>()
        );
    }
    if kind == "ordinal" {
        let ev = labels
            .iter()
            .map(|label| label.parse::<f64>().unwrap() * p[label])
            .sum::<f64>();
        let reported = value["expected_value"]
            .as_f64()
            .context("ordinal expected value")?;
        ensure!(
            reported.is_finite() && (reported - ev).abs() <= 1e-6,
            "Ordinal expectation differs from probabilities"
        );
        let mae = (ev - gold["score"].as_f64().context("gold score")?).abs();
        row["score_mae"] = json!(mae);
        row["within_one"] = json!(mae <= 1.);
    }
    Ok(row)
}
fn auroc(labels: &[bool], probabilities: &[f64]) -> Value {
    let n1 = labels.iter().filter(|v| **v).count();
    let n0 = labels.len() - n1;
    if n1 == 0 || n0 == 0 {
        return Value::Null;
    }
    let mut pairs = probabilities
        .iter()
        .copied()
        .zip(labels.iter().copied())
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut i = 0;
    let mut rank_sum = 0.;
    while i < pairs.len() {
        let mut j = i + 1;
        while j < pairs.len() && pairs[j].0 == pairs[i].0 {
            j += 1;
        }
        rank_sum += (i + j + 1) as f64 / 2. * pairs[i..j].iter().filter(|x| x.1).count() as f64;
        i = j;
    }
    json!((rank_sum - (n1 * (n1 + 1)) as f64 / 2.) / (n0 * n1) as f64)
}
fn metrics(rows: &[Value], attempted: usize) -> Value {
    let labeled = rows
        .iter()
        .filter(|r| r.get("correct").is_some())
        .collect::<Vec<_>>();
    let accepted = labeled
        .iter()
        .filter(|r| r["accepted"] == true)
        .collect::<Vec<_>>();
    let correct = labeled.iter().filter(|r| r["correct"] == true).count();
    let mut bins = BTreeMap::<usize, Vec<&Value>>::new();
    for row in &labeled {
        let bucket = (row["confidence"].as_f64().unwrap() * 10.) as usize;
        bins.entry(bucket.min(9)).or_default().push(row);
    }
    let ece = if labeled.is_empty() {
        Value::Null
    } else {
        json!(
            bins.values()
                .map(|group| group
                    .iter()
                    .map(|r| r["confidence"].as_f64().unwrap() - f64::from(r["correct"] == true))
                    .sum::<f64>()
                    .abs())
                .sum::<f64>()
                / labeled.len() as f64
        )
    };
    let mut m = json!({"attempted":attempted,"valid":labeled.len(),"correct":correct,
        "accuracy":if attempted==0{Value::Null}else{json!(correct as f64/attempted as f64)},
        "valid_accuracy":mean(&labeled.iter().map(|r|f64::from(r["correct"]==true)).collect::<Vec<_>>()),
        "ece_hard":ece,"accepted":accepted.len(),
        "accepted_accuracy":mean(&accepted.iter().map(|r|f64::from(r["correct"]==true)).collect::<Vec<_>>()),
        "wrong_accepted":accepted.iter().filter(|r|r["correct"]==false).count(),
        "coverage":if attempted==0{Value::Null}else{json!(accepted.len() as f64/attempted as f64)}});
    for field in [
        "soft_accuracy",
        "brier_soft",
        "brier_hard",
        "kl",
        "tvd",
        "score_mae",
        "within_one",
    ] {
        let values = labeled
            .iter()
            .filter_map(|r| r.get(field))
            .map(|v| {
                if v.is_boolean() {
                    f64::from(v == true)
                } else {
                    v.as_f64().unwrap()
                }
            })
            .collect::<Vec<_>>();
        m[field] = mean(&values);
    }
    m
}
fn score(prepared: &Path, predictions: &Path) -> Result<(Value, Vec<Value>)> {
    let manifest = read_json(&prepared.join("manifest.json"))?;
    for (name, digest) in manifest["files"].as_object().context("manifest files")? {
        ensure!(
            sha256(&prepared.join(name))? == *digest,
            "Prepared data hash mismatch"
        );
    }
    let cases = read_jsonl(&prepared.join("cases-with-gold.jsonl"))?;
    let raw = read_jsonl(predictions)?;
    let expected_ids = cases
        .iter()
        .map(|c| nonempty(&c["id"]).unwrap().to_owned())
        .collect::<HashSet<_>>();
    let mut by_id = HashMap::new();
    for row in raw {
        let id = nonempty(&row["id"])?.to_owned();
        ensure!(
            expected_ids.contains(&id) && by_id.insert(id, row).is_none(),
            "Duplicate or unknown prediction ID"
        );
    }
    let mut details = Vec::new();
    let mut errors = Vec::new();
    let mut checks = Vec::new();
    let mut batches = HashMap::<u64, Value>::new();
    let mut valid_cases = HashSet::new();
    for case in &cases {
        let id = nonempty(&case["id"])?;
        let Some(prediction) = by_id.get(id) else {
            errors.push(json!({"id":id,"error":"Missing prediction"}));
            continue;
        };
        if !prediction["error"].is_null() {
            errors.push(json!({"id":id,"error":prediction["error"]}));
            continue;
        }
        let attempt = (|| -> Result<(Vec<Value>, Value)> {
            let latency = prediction["elapsed_ms"]
                .as_f64()
                .context("Invalid latency")?;
            ensure!(latency.is_finite() && latency >= 0., "Invalid latency");
            let results = prediction["response"]["results"]
                .as_array()
                .context("results")?;
            let decisions = case["request"]["decisions"]
                .as_array()
                .context("decisions")?;
            ensure!(results.len() == decisions.len(), "Decision count mismatch");
            let scored = decisions
                .iter()
                .zip(results)
                .map(|(d, r)| score_decision(d, r, case["gold"].get(nonempty(&d["id"])?)))
                .collect::<Result<Vec<_>>>()?;
            let batch = json!({"batch_size":prediction["batch_size"],"batch_elapsed_ms":prediction["batch_elapsed_ms"],
                "batch_profile":prediction["batch_profile"],"preparation_cache_before":prediction["preparation_cache_before"],
                "preparation_cache_after":prediction["preparation_cache_after"]});
            Ok((scored, batch))
        })();
        let (mut scored, batch) = match attempt {
            Ok(v) => v,
            Err(error) => {
                errors.push(json!({"id":id,"error":error.to_string()}));
                continue;
            }
        };
        let batch_index = prediction["batch_index"].as_u64().context("batch_index")?;
        if let Some(previous) = batches.insert(batch_index, batch.clone()) {
            ensure!(previous == batch, "Inconsistent batch metadata");
        }
        valid_cases.insert(id.to_owned());
        for row in &mut scored {
            row["case_id"] = json!(id);
            row["base_id"] = case["base_id"].clone();
            row["repeat"] = case["repeat"].clone();
            row["workflow"] = case.get("workflow").cloned().unwrap_or(Value::Null);
            row["case_elapsed_ms"] = prediction["elapsed_ms"].clone();
        }
        details.extend(scored.clone());
        let repeat = &case["repeat"];
        match case["probe"].as_str(){
            Some("contradiction")=>{
                let total=scored.iter().map(|r|r["probabilities"]["true"].as_f64().unwrap()).sum::<f64>();
                checks.push(json!({"id":id,"kind":"contradiction","failed":(total-1.).abs()>0.35,"sum_probability":total,"repeat":repeat}));
            },
            Some("stability")=>checks.push(json!({"id":id,"kind":"stability","failed":scored.iter().any(|r|r["correct"]!=true),"repeat":repeat})),
            Some("grounding")=>{
                let row=&scored[0];let wrong=row["correct"]!=true;
                checks.push(json!({"id":id,"kind":"grounding","failed":wrong,"repeat":repeat}));
                checks.push(json!({"id":id,"kind":"overconfidence","failed":wrong && row["entropy_confidence"].as_f64().unwrap()>=0.8,"repeat":repeat}));
            },_=>{}
        }
    }
    let mut per_repeat = Map::new();
    for repeat in 0..manifest["repeats"].as_u64().context("repeats")? {
        let chosen = details
            .iter()
            .filter(|r| r["repeat"] == repeat)
            .cloned()
            .collect::<Vec<_>>();
        let subset = cases
            .iter()
            .filter(|c| c["repeat"] == repeat)
            .collect::<Vec<_>>();
        let planned = subset
            .iter()
            .map(|c| c["gold"].as_object().map_or(0, Map::len))
            .sum();
        let mut measured = metrics(&chosen, planned);
        let ids = subset
            .iter()
            .map(|c| nonempty(&c["id"]).unwrap())
            .collect::<HashSet<_>>();
        measured["case_errors"] = json!(
            errors
                .iter()
                .filter(|e| ids.contains(e["id"].as_str().unwrap()))
                .count()
        );
        let latencies = subset
            .iter()
            .filter(|c| valid_cases.contains(nonempty(&c["id"]).unwrap()))
            .map(|c| {
                by_id[nonempty(&c["id"]).unwrap()]["elapsed_ms"]
                    .as_f64()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        measured["case_latency_ms"] =
            json!({"p50":quantile(&latencies,0.5),"p95":quantile(&latencies,0.95)});
        let workflows = subset
            .iter()
            .filter_map(|c| c.get("workflow").and_then(Value::as_str))
            .collect::<HashSet<_>>();
        let mut by_workflow = Map::new();
        for workflow in workflows {
            let rows = chosen
                .iter()
                .filter(|r| r["workflow"] == workflow)
                .cloned()
                .collect::<Vec<_>>();
            let total = subset
                .iter()
                .filter(|c| c["workflow"] == workflow)
                .map(|c| c["gold"].as_object().map_or(0, Map::len))
                .sum();
            by_workflow.insert(workflow.to_owned(), metrics(&rows, total));
        }
        measured["by_workflow"] = json!(by_workflow);
        if manifest["suite"] == "phish" {
            let y = chosen
                .iter()
                .map(|r| r["expected"] == "true")
                .collect::<Vec<_>>();
            let probabilities = chosen
                .iter()
                .map(|r| r["probabilities"]["true"].as_f64().unwrap())
                .collect::<Vec<_>>();
            measured["auroc"] = auroc(&y, &probabilities);
            measured["recall"] = mean(
                &chosen
                    .iter()
                    .filter(|r| r["expected"] == "true")
                    .map(|r| f64::from(r["predicted"] == "true"))
                    .collect::<Vec<_>>(),
            );
            measured["precision"] = mean(
                &chosen
                    .iter()
                    .filter(|r| r["predicted"] == "true")
                    .map(|r| f64::from(r["expected"] == "true"))
                    .collect::<Vec<_>>(),
            );
        }
        let probe_checks = checks
            .iter()
            .filter(|c| c["repeat"] == repeat)
            .cloned()
            .collect::<Vec<_>>();
        measured["probe_failures"] =
            json!(probe_checks.iter().filter(|c| c["failed"] == true).count());
        measured["probe_checks"] = json!(probe_checks);
        per_repeat.insert(repeat.to_string(), measured);
    }
    let mut cache = Map::new();
    for kind in ["prompts", "candidates"] {
        let mut fields = Map::new();
        for field in ["hits", "misses", "insertions", "evictions", "skipped"] {
            let sum = batches
                .values()
                .map(|b| {
                    b["preparation_cache_after"][kind][field].as_i64().unwrap()
                        - b["preparation_cache_before"][kind][field].as_i64().unwrap()
                })
                .sum::<i64>();
            fields.insert(field.to_owned(), json!(sum));
        }
        cache.insert(kind.to_owned(), Value::Object(fields));
    }
    let mut profile = Map::new();
    for key in ["prepare_ms", "native_ms", "score_ms"] {
        profile.insert(
            key.to_owned(),
            json!(
                batches
                    .values()
                    .map(|b| b["batch_profile"][key].as_f64().unwrap())
                    .sum::<f64>()
            ),
        );
    }
    let elapsed = batches
        .values()
        .map(|b| b["batch_elapsed_ms"].as_f64().unwrap())
        .sum::<f64>();
    let profile_total = profile.values().map(|v| v.as_f64().unwrap()).sum::<f64>();
    let input_tokens = details
        .iter()
        .map(|r| r["input_tokens"].as_u64().unwrap())
        .sum::<u64>();
    let reused = details
        .iter()
        .map(|r| r["reused_prefix_tokens"].as_u64().unwrap())
        .sum::<u64>();
    let summary = json!({"suite":manifest["suite"],"limited":manifest["limited"],"logical_cases":manifest["logical_cases"],
        "decisions":manifest["decisions"],"per_repeat":per_repeat,"errors":errors,"cache":cache,"profile":profile,
        "unique_batch_elapsed_ms":elapsed,"logical_input_tokens":input_tokens,"reused_prefix_tokens":reused,
        "evaluated_input_tokens":input_tokens-reused,"scope":SCOPE,"boundary_and_other_ms":(elapsed-profile_total).max(0.)});
    Ok((summary, details))
}
fn compare(prepared: &Path, baseline: &Path, candidate: &Path) -> Result<Value> {
    let (a, ar) = score(prepared, &baseline.join("predictions.jsonl"))?;
    let (b, br) = score(prepared, &candidate.join("predictions.jsonl"))?;
    ensure!(
        a["errors"].as_array().unwrap().is_empty() && b["errors"].as_array().unwrap().is_empty(),
        "Cannot certify an incomplete run"
    );
    let ma = read_json(&baseline.join("run.json"))?;
    let mb = read_json(&candidate.join("run.json"))?;
    for key in [
        "model_sha256",
        "evaluator_sha256",
        "prepared_manifest_sha256",
    ] {
        ensure!(ma[key] == mb[key], "Comparison identity mismatch: {key}");
    }
    let pa = read_jsonl(&baseline.join("predictions.jsonl"))?;
    let pb = read_jsonl(&candidate.join("predictions.jsonl"))?;
    ensure!(
        pa.len() == pb.len() && pa.iter().zip(&pb).all(|(x, y)| x["id"] == y["id"]),
        "Run order differs"
    );
    let mut mass = 0_f64;
    let mut accepted_changes = 0;
    for (x, y) in pa.iter().zip(&pb) {
        for key in [
            "prompt_layout",
            "prompt_version",
            "prompt_profile",
            "compute",
            "lora_path",
            "output_head_path",
        ] {
            ensure!(
                x["response"]["backend"][key] == y["response"]["backend"][key],
                "Not an execution-only comparison: {key}"
            );
        }
        ensure!(
            x["response"]["policy"] == y["response"]["policy"],
            "Policy differs"
        );
        let rx = x["response"]["results"].as_array().context("results")?;
        let ry = y["response"]["results"].as_array().context("results")?;
        ensure!(rx.len() == ry.len(), "Decision count differs");
        for (x, y) in rx.iter().zip(ry) {
            let sx = x["value"].get("selected").unwrap_or(&x["value"]["value"]);
            let sy = y["value"].get("selected").unwrap_or(&y["value"]["value"]);
            accepted_changes += usize::from(sx != sy);
            mass = mass.max(
                (x["candidate_mass"].as_f64().unwrap() - y["candidate_mass"].as_f64().unwrap())
                    .abs(),
            );
        }
    }
    ensure!(ar.len() == br.len(), "scored decision count differs");
    let mut probability_delta = 0_f64;
    let mut flips = 0;
    for (x, y) in ar.iter().zip(&br) {
        ensure!(x["id"] == y["id"], "scored decision order differs");
        flips += usize::from(x["predicted"] != y["predicted"]);
        let px = x["probabilities"].as_object().context("probabilities")?;
        let py = y["probabilities"].as_object().context("probabilities")?;
        ensure!(px.keys().eq(py.keys()), "candidate keys differ");
        for key in px.keys() {
            probability_delta = probability_delta
                .max((px[key].as_f64().unwrap() - py[key].as_f64().unwrap()).abs());
        }
    }
    Ok(
        json!({"decisions_compared":ar.len(),"max_probability_delta":probability_delta,
        "max_candidate_mass_delta":mass,"raw_top1_changes":flips,"accepted_selection_changes":accepted_changes,
        "exact_probabilities":probability_delta==0. && mass==0.,
        "within_existing_tolerance":probability_delta<=0.02 && mass<=0.02 && flips==0 && accepted_changes==0,
        "total_inference_speedup":a["unique_batch_elapsed_ms"].as_f64().unwrap()/b["unique_batch_elapsed_ms"].as_f64().unwrap(),
        "baseline_profile":a["profile"],"candidate_profile":b["profile"],"candidate_cache":b["cache"],
        "reused_tokens":b["reused_prefix_tokens"],
        "scope":"Same-model, same-evaluator, same-input, same-layout/compute/policy execution comparison; one local run with immediate repeats, not a service latency guarantee."}),
    )
}
#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    prepared: &Path,
    output: &Path,
    evaluator: &Path,
    model: &Path,
    cuda: bool,
    gpu_layers: Option<usize>,
    cpu_moe_layers: usize,
    context: usize,
    batch: usize,
    threads: usize,
    model_load_mode: &str,
    execution_mode: &str,
    prompt_layout: &str,
    parallel_width: usize,
    request_batch_size: usize,
    cache_bytes: usize,
    cache_entries: usize,
    timeout: u64,
) -> Result<()> {
    let manifest = read_json(&prepared.join("manifest.json"))?;
    for (name, expected) in manifest["files"].as_object().context("manifest files")? {
        ensure!(
            sha256(&prepared.join(name))? == *expected,
            "Prepared hash mismatch"
        );
    }
    fs::create_dir(output)?;
    let evaluator = evaluator.canonicalize()?;
    let model = model.canonicalize()?;
    let mut command = vec![
        "--model".to_owned(),
        model.to_string_lossy().into_owned(),
        "--input".to_owned(),
        prepared
            .join("requests.jsonl")
            .canonicalize()?
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        output
            .join("predictions.jsonl")
            .to_string_lossy()
            .into_owned(),
        "--context".to_owned(),
        context.to_string(),
        "--batch".to_owned(),
        batch.to_string(),
        "--threads".to_owned(),
        threads.to_string(),
        "--execution-mode".to_owned(),
        execution_mode.to_owned(),
        "--prompt-layout".to_owned(),
        prompt_layout.to_owned(),
        "--parallel-width".to_owned(),
        parallel_width.to_string(),
        "--request-batch-size".to_owned(),
        request_batch_size.to_string(),
        "--model-load-mode".to_owned(),
        model_load_mode.to_owned(),
        "--preparation-cache-bytes".to_owned(),
        cache_bytes.to_string(),
        "--preparation-cache-entries".to_owned(),
        cache_entries.to_string(),
        "--warmup".to_owned(),
    ];
    if cuda {
        command.push("--cuda".to_owned());
    }
    if let Some(n) = gpu_layers {
        command.extend(["--gpu-layers".to_owned(), n.to_string()]);
    }
    if cpu_moe_layers > 0 {
        command.extend(["--cpu-moe-layers".to_owned(), cpu_moe_layers.to_string()]);
    }
    let mut recorded = vec![evaluator.to_string_lossy().into_owned()];
    recorded.extend(command.clone());
    write_json(
        &output.join("run.json"),
        &json!({"command":recorded,"evaluator_sha256":sha256(&evaluator)?,
        "model_sha256":sha256(&model)?,"prepared_manifest_sha256":sha256(&prepared.join("manifest.json"))?,
        "adapter_sha256":digest_bytes(include_bytes!("laya_benchmark.rs")),
        "timing":"One resident model, untimed warmup then preparation-cache clear; load excluded. No answer cache."}),
    )?;
    let log = File::create(output.join("inference.log"))?;
    let mut child = Command::new(&evaluator)
        .args(&command)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()?;
    let started = std::time::Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break json!({"exit_code":status.code(),"timeout":false});
        }
        if started.elapsed() > Duration::from_secs(timeout) {
            child.kill()?;
            child.wait()?;
            break json!({"exit_code":null,"timeout":true});
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    write_json(&output.join("exit.json"), &status)?;
    ensure!(
        output.join("predictions.jsonl").exists(),
        "Evaluator failed before creating predictions; inspect inference.log"
    );
    let (summary, details) = score(prepared, &output.join("predictions.jsonl"))?;
    write_json(&output.join("summary.json"), &summary)?;
    write_jsonl(&output.join("scored.jsonl"), &details)?;
    ensure!(
        status["exit_code"] == 0 && summary["errors"].as_array().unwrap().is_empty(),
        "Incomplete evaluation; failures retained in summary.json"
    );
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
pub fn run(args: Args) -> Result<()> {
    match args.command {
        Action::Fetch { suite, output } => fetch(&suite, &output),
        Action::Prepare {
            suite,
            source,
            output,
            limit,
            repeats,
        } => prepare(&suite, &source, &output, limit, repeats),
        Action::Score {
            prepared,
            predictions,
            output,
        } => {
            let (summary, _) = score(&prepared, &predictions)?;
            write_json(&output, &summary)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        Action::Compare {
            prepared,
            baseline,
            candidate,
            output,
        } => {
            let report = compare(&prepared, &baseline, &candidate)?;
            write_json(&output, &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Action::Run {
            prepared,
            output,
            evaluator,
            model,
            cuda,
            gpu_layers,
            cpu_moe_layers,
            context,
            batch,
            threads,
            model_load_mode,
            execution_mode,
            prompt_layout,
            parallel_width,
            request_batch_size,
            cache_bytes,
            cache_entries,
            timeout,
        } => run_benchmark(
            &prepared,
            &output,
            &evaluator,
            &model,
            cuda,
            gpu_layers,
            cpu_moe_layers,
            context,
            batch,
            threads,
            &model_load_mode,
            &execution_mode,
            &prompt_layout,
            parallel_width,
            request_batch_size,
            cache_bytes,
            cache_entries,
            timeout,
        ),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_question_order_and_gold_boundary() {
        let row = json!({"state":"{\"text\":\"Input only\"}","questions":"{\"route\":{\"type\":\"choice\",\"instructions\":\"Choose.\",\"criteria\":{\"z\":\"Z\",\"a\":\"A\"}}}",
            "gold":"{\"route\":{\"label\":\"z\",\"rationale\":\"SECRET\"}}","workflow":"SECRET"});
        let case = &typed_cases(&[row]).unwrap()[0];
        let request = &case["request"];
        assert_eq!(request["decisions"][0]["kind"]["options"][0]["id"], "z");
        assert!(!serde_json::to_string(request).unwrap().contains("SECRET"));
    }
    #[test]
    fn binary_tie_is_true_before_abstention() {
        let question = Question {
            kind: "noul".to_owned(),
            instructions: "Is it true?".to_owned(),
            criteria: None,
        };
        let decision = decision("q", &question).unwrap();
        let result = json!({"id":"q","scores":[{"id":"false","option_probability":0.5},{"id":"true","option_probability":0.5}],
            "truncated":false,"value":{"type":"binary","value":null,"p_true":0.5},"entropy_confidence":0.1,
            "candidate_mass":0.9,"input_tokens":10,"reused_prefix_tokens":0});
        let measured =
            score_decision(&decision, &result, Some(&json!({"label":"true","noul":1}))).unwrap();
        assert_eq!(measured["predicted"], "true");
        assert_eq!(measured["correct"], true);
        assert_eq!(measured["accepted"], false);
    }
}
