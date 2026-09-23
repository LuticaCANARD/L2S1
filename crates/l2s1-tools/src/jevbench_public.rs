use crate::common::{digest_bytes, read_json, read_jsonl, sha256, write_json, write_jsonl};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args as ClapArgs, Subcommand};
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
const REVISION: &str = "f79a1cab94ab9a5879383b7ef9ee1805b9dc2d84";
const SPLITS: [(&str, usize); 3] = [("easy", 48), ("original", 72), ("hard", 111)];
const SOURCE: &str = "native_candidate_softmax";
#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    Prepare {
        #[arg(long)]
        jevbench: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Score {
        #[arg(long)]
        prepared: PathBuf,
        #[arg(long)]
        predictions: PathBuf,
        #[arg(long)]
        model_name: String,
        #[arg(long)]
        output: PathBuf,
    },
    Run {
        #[arg(long)]
        jevbench: PathBuf,
        #[arg(long)]
        evaluator: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 8192)]
        context: usize,
        #[arg(long, default_value_t = 4)]
        threads: usize,
        #[arg(long)]
        gpu_layers: Option<usize>,
        #[arg(long, default_value_t = 0)]
        cpu_moe_layers: usize,
        #[arg(long, default_value = "auto")]
        model_load_mode: String,
        #[arg(long, default_value = "cuda")]
        device: String,
        #[arg(long)]
        expected_gpu: Option<String>,
        #[arg(long, default_value_t = 1800)]
        timeout: u64,
    },
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key].as_str().with_context(|| format!("missing {key}"))
}
fn canonical_python_json(value: &Value) -> Result<String> {
    let compact = serde_json::to_string(value)?;
    let mut expanded = String::with_capacity(compact.len() + compact.len() / 10);
    let mut quoted = false;
    let mut escaped = false;
    for character in compact.chars() {
        if quoted {
            expanded.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
        } else if character == '"' {
            expanded.push(character);
            quoted = true;
        } else if character == ',' || character == ':' {
            expanded.push(character);
            expanded.push(' ');
        } else {
            expanded.push(character);
        }
    }
    Ok(expanded)
}
fn validate_task(task: &Value) -> Result<()> {
    let id = text(task, "id")?;
    let kind = text(&task["question"], "type")?;
    ensure!(
        ["noul", "choice", "score"].contains(&kind),
        "{id}: bad question type"
    );
    ensure!(
        ["public", "private"].contains(&text(task, "split")?),
        "{id}: bad split"
    );
    let labels = task["labels"].as_array().context("labels")?;
    ensure!(!labels.is_empty(), "{id}: empty labels");
    let expected = &task["expected"];
    if !expected.is_null() {
        if kind == "score" {
            ensure!(
                expected.is_i64()
                    && labels.contains(&json!(expected.as_i64().unwrap().to_string())),
                "{id}: score expected outside labels"
            );
        } else {
            ensure!(labels.contains(expected), "{id}: expected outside labels");
        }
    }
    if let Some(state) = task["state"].as_object() {
        for banned in ["expected", "label", "ground_truth", "answer_key"] {
            ensure!(
                !state.contains_key(banned),
                "{id}: state contains banned key {banned}"
            );
        }
    }
    Ok(())
}
fn request(task: &Value) -> Result<Value> {
    validate_task(task)?;
    let question = &task["question"];
    let labels = task["labels"].as_array().unwrap();
    ensure!(
        (2..=26).contains(&labels.len())
            && labels.iter().collect::<HashSet<_>>().len() == labels.len(),
        "Invalid candidate set"
    );
    let criteria = &question["criteria"];
    let kind = match text(question, "type")? {
        "noul" => {
            ensure!(
                labels == &vec![json!("no"), json!("yes")]
                    && criteria.as_object().is_some_and(|c| c.len() == 2
                        && c.contains_key("false")
                        && c.contains_key("true")),
                "Noncanonical binary task"
            );
            json!({"type":"binary","false_label":criteria["false"],"true_label":criteria["true"]})
        }
        "score" => {
            let levels = criteria.as_array().context("score criteria")?;
            ensure!(
                labels
                    .iter()
                    .enumerate()
                    .all(|(i, v)| *v == json!(i.to_string()))
                    && labels.len() == levels.len(),
                "Noncanonical ordinal task"
            );
            json!({"type":"ordinal","levels":labels.iter().enumerate().map(|(i,label)|json!({"id":label,"criterion":levels[i],"value":i})).collect::<Vec<_>>()})
        }
        "choice" => {
            let criteria = criteria.as_object().context("choice criteria")?;
            ensure!(
                criteria.len() == labels.len()
                    && labels
                        .iter()
                        .all(|label| criteria.contains_key(label.as_str().unwrap_or(""))),
                "Choice rubric differs from labels"
            );
            json!({"type":"choice","options":labels.iter().map(|label|json!({"id":label,"criterion":criteria[label.as_str().unwrap()]})).collect::<Vec<_>>()})
        }
        _ => bail!("Unsupported question type"),
    };
    Ok(
        json!({"id":task["id"],"request":{"state":task["state"],"decisions":[{"id":"decision","instruction":question["instructions"],"kind":kind}]}}),
    )
}
fn source_tasks(jevbench: &Path) -> Result<(Vec<Value>, Map<String, Value>, String)> {
    let revision = Command::new("git")
        .args([
            "-C",
            jevbench.to_str().context("jevbench path")?,
            "rev-parse",
            "HEAD",
        ])
        .output()?;
    ensure!(
        revision.status.success() && String::from_utf8_lossy(&revision.stdout).trim() == REVISION,
        "Upstream revision differs from the reviewed frozen version"
    );
    let status = Command::new("git")
        .args(["-C", jevbench.to_str().unwrap(), "status", "--porcelain"])
        .output()?;
    ensure!(
        status.status.success() && status.stdout.is_empty(),
        "Upstream checkout must be clean"
    );
    let mut tasks = Vec::new();
    let mut hashes = Map::new();
    let mut ids = HashSet::new();
    for (split, count) in SPLITS {
        let path = jevbench
            .join("datasets/public")
            .join(format!("{split}.jsonl"));
        let rows = read_jsonl(&path)?;
        ensure!(rows.len() == count, "Frozen split size changed");
        for row in rows {
            validate_task(&row)?;
            ensure!(
                row["split"] == "public" && ids.insert(text(&row, "id")?.to_owned()),
                "Repeated ID or nonpublic task"
            );
            tasks.push(row);
        }
        hashes.insert(format!("{split}.jsonl"), json!(sha256(&path)?));
    }
    let mut blobs = tasks
        .iter()
        .map(canonical_python_json)
        .collect::<Result<Vec<_>>>()?;
    blobs.sort();
    let mut payload = blobs.join("\n");
    payload.push('\n');
    Ok((tasks, hashes, digest_bytes(payload.as_bytes())))
}
fn prepare(jevbench: &Path, output: &Path) -> Result<Value> {
    let (tasks, hashes, dataset_hash) = source_tasks(jevbench)?;
    fs::create_dir(output)?;
    let requests = tasks.iter().map(request).collect::<Result<Vec<_>>>()?;
    write_jsonl(&output.join("requests.jsonl"), &requests)?;
    write_jsonl(&output.join("tasks-with-gold.jsonl"), &tasks)?;
    let manifest = json!({"jevbench_revision":REVISION,"dataset_hash":dataset_hash,"source_sha256":hashes,
        "requests_sha256":sha256(&output.join("requests.jsonl"))?,"planned":tasks.len(),
        "tier_counts":SPLITS.iter().map(|(n,c)|(n.to_string(),json!(c))).collect::<Map<_,_>>(),
        "label_order":"canonical task.labels; binary false/true mapped to no/yes",
        "probability_origin":"native candidate-token softmax, not learned calibration",
        "scope":"Public 231 items only; no official full-suite score or rank; no training or tuning"});
    write_json(&output.join("manifest.json"), &manifest)?;
    Ok(manifest)
}
fn score_task(task: &Value, probabilities: Option<&Map<String, Value>>) -> Value {
    let labels = task["labels"].as_array().unwrap();
    let failed = |error: String| json!({"valid":false,"strict_valid":false,"renormalized":false,"error":error,"correct":false,"predicted":null});
    let Some(probabilities) = probabilities else {
        return failed("probs is not a dict".to_owned());
    };
    let wanted = labels
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect::<HashSet<_>>();
    let got = probabilities.keys().cloned().collect::<HashSet<_>>();
    if wanted != got {
        let mut missing = wanted.difference(&got).cloned().collect::<Vec<_>>();
        let mut extra = got.difference(&wanted).cloned().collect::<Vec<_>>();
        missing.sort();
        extra.sort();
        return failed(format!(
            "label keys mismatch: missing={missing:?} extra={extra:?}"
        ));
    }
    let mut clean = BTreeMap::new();
    let mut sum = 0.;
    for (key, value) in probabilities {
        let Some(v) = value.as_f64() else {
            return failed(format!("prob[{key:?}] is not a number"));
        };
        if !v.is_finite() {
            return failed(format!("prob[{key:?}] is not finite"));
        }
        if !(0. ..=1.).contains(&v) {
            return failed(format!("prob[{key:?}] out of [0,1]: {v}"));
        }
        clean.insert(key.clone(), v);
        sum += v;
    }
    let strict = (sum - 1_f64).abs() <= 1e-3;
    if (sum - 1_f64).abs() > 0.02 {
        return failed(format!("probs sum to {sum}, tolerance 0.001"));
    }
    if !strict {
        if sum <= 0. {
            return failed("probabilities sum to zero".to_owned());
        }
        for value in clean.values_mut() {
            *value /= sum;
        }
    }
    let mut result =
        json!({"valid":true,"strict_valid":strict,"renormalized":!strict,"probs":clean});
    let kind = text(&task["question"], "type").unwrap();
    let predicted = if kind == "score" && task["expected"].is_null() {
        Value::Null
    } else {
        json!(
            clean
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .unwrap()
                .0
        )
    };
    result["predicted"] = predicted.clone();
    if kind == "score" {
        result["ordinal_ev"] = json!(
            clean
                .iter()
                .map(|(key, value)| key.parse::<f64>().unwrap() * value)
                .sum::<f64>()
        );
    }
    result["correct"] = if task["expected"].is_null() {
        Value::Null
    } else if kind == "score" {
        json!(predicted == json!(task["expected"].as_i64().unwrap().to_string()))
    } else {
        json!(predicted == task["expected"])
    };
    result
}
fn exact_probabilities(task: &Value, result: &Value) -> Result<Map<String, Value>> {
    let labels = task["labels"].as_array().context("labels")?;
    let scores = result["scores"].as_array().context("scores")?;
    ensure!(
        scores.len() == labels.len(),
        "Returned candidate order/set differs from the request"
    );
    let mut output = Map::new();
    for (score, label) in scores.iter().zip(labels) {
        let mapped = if task["question"]["type"] == "noul" {
            match text(score, "id")? {
                "false" => "no",
                "true" => "yes",
                other => other,
            }
        } else {
            text(score, "id")?
        };
        ensure!(
            label == mapped,
            "Returned candidate order/set differs from the request"
        );
        output.insert(mapped.to_owned(), score["option_probability"].clone());
    }
    Ok(output)
}
fn score_predictions(
    tasks: &[Value],
    predictions: &[Value],
    model_name: &str,
) -> Result<(Vec<Value>, Vec<Value>)> {
    let by_id = tasks
        .iter()
        .map(|t| Ok((text(t, "id")?.to_owned(), t)))
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(by_id.len() == tasks.len(), "Duplicate task IDs");
    let mut seen = HashSet::new();
    let mut records = Vec::new();
    let mut selective = Vec::new();
    for prediction in predictions {
        let id = text(prediction, "id")?;
        let task = by_id
            .get(id)
            .context("Duplicate or unknown prediction ID")?;
        ensure!(
            seen.insert(id.to_owned()),
            "Duplicate or unknown prediction ID"
        );
        let elapsed = prediction["elapsed_ms"]
            .as_f64()
            .context("Invalid elapsed time")?;
        ensure!(elapsed.is_finite() && elapsed >= 0., "Invalid elapsed time");
        let mut error = prediction.get("error").filter(|v| !v.is_null()).cloned();
        let mut probabilities = None;
        let mut result = None;
        if error.is_none() {
            let inspected = (|| -> Result<(Map<String, Value>, Value)> {
                let results = prediction["response"]["results"]
                    .as_array()
                    .context("results")?;
                ensure!(
                    results.len() == 1 && results[0]["id"] == "decision",
                    "Expected exactly one decision"
                );
                ensure!(results[0]["truncated"] == false, "Truncated input");
                let p = exact_probabilities(task, &results[0])?;
                Ok((p, results[0].clone()))
            })();
            match inspected {
                Ok((p, r)) => {
                    probabilities = Some(p);
                    result = Some(r);
                }
                Err(e) => error = Some(json!(e.to_string())),
            }
        }
        let scored = score_task(task, probabilities.as_ref());
        let ok = error.is_none() && scored["valid"] == true;
        let mut record = json!({"task_id":id,"family":task["family"],"split":task["split"],"group":task["group"],
            "status":if ok{"ok"}else{"failed"},"ok":ok,
            "probs_as_returned":probabilities,"probs_source":SOURCE,"model":model_name,
            "error":error.clone().unwrap_or_else(||scored.get("error").cloned().unwrap_or(Value::Null)),
            "latency_s":elapsed/1000.,"cost_usd":null,"cost_basis":"local_gpu_no_provider_tariff",
            "usage":result.as_ref().map(|r|json!({"input_tokens":r["input_tokens"],"output_tokens":0})).unwrap_or(json!({}))});
        record
            .as_object_mut()
            .unwrap()
            .extend(scored.as_object().unwrap().clone());
        let mut accepted = false;
        let mut selected = Value::Null;
        if ok {
            let value = &result.as_ref().unwrap()["value"];
            selected = if value["type"] == "binary" {
                match value["value"].as_bool() {
                    Some(true) => json!("yes"),
                    Some(false) => json!("no"),
                    None => Value::Null,
                }
            } else {
                value["selected"].clone()
            };
            accepted = !selected.is_null();
        }
        let expected = if task["expected"].is_string() {
            task["expected"].clone()
        } else if let Some(level) = task["expected"].as_i64() {
            json!(level.to_string())
        } else {
            Value::Null
        };
        selective.push(json!({"id":id,"accepted":accepted,"correct":accepted && selected==expected,
            "abstained":ok && !accepted,"error":!ok,"candidate_mass":result.as_ref().map(|r|r["candidate_mass"].clone()),
            "abstention_reasons":result.as_ref().map(|r|r["abstention_reasons"].clone())}));
        records.push(record);
    }
    ensure!(
        seen.len() == tasks.len(),
        "Missing predictions: {}",
        tasks.len() - seen.len()
    );
    Ok((records, selective))
}
fn optional_rate(n: usize, d: usize) -> Value {
    if d == 0 {
        Value::Null
    } else {
        json!(n as f64 / d as f64)
    }
}
fn mean(values: &[f64]) -> Value {
    if values.is_empty() {
        Value::Null
    } else {
        json!(values.iter().sum::<f64>() / values.len() as f64)
    }
}
fn percentile(values: &[f64], fraction: f64) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let k = (values.len() - 1) as f64 * fraction;
    let floor = k.floor() as usize;
    let ceil = k.ceil() as usize;
    if floor == ceil {
        json!(values[floor])
    } else {
        json!(values[floor] * (ceil as f64 - k) + values[ceil] * (k - floor as f64))
    }
}
fn latency_summary(values: &[f64]) -> Value {
    json!({"n":values.len(),"p50_s":percentile(values,0.5),"p95_s":percentile(values,0.95)})
}
fn ece(pairs: &[(f64, bool)]) -> Value {
    let mut bins = vec![(0usize, 0_f64, 0usize); 10];
    for (confidence, correct) in pairs {
        let confidence = confidence.clamp(0., 1.);
        let index = ((confidence * 10.) as usize).min(9);
        bins[index].0 += 1;
        bins[index].1 += confidence;
        bins[index].2 += usize::from(*correct);
    }
    let n = pairs.len();
    let mut total = 0.;
    let mut out = Vec::new();
    for (i, (count, confidence, correct)) in bins.iter().enumerate() {
        if *count > 0 {
            total += (*count as f64 / n as f64)
                * (*correct as f64 / (*count as f64) - confidence / (*count as f64)).abs();
        }
        out.push(json!({"lo":i as f64/10.,"hi":(i+1) as f64/10.,"n":count,
            "mean_confidence":if *count>0{json!(confidence/(*count as f64))}else{Value::Null},
            "accuracy":optional_rate(*correct,*count)}));
    }
    json!({"ece":total,"n":n,"bins":out})
}
fn metric(tasks: &[&Value], records: &[Value]) -> Result<Value> {
    let by_id = tasks
        .iter()
        .map(|t| (text(t, "id").unwrap(), *t))
        .collect::<HashMap<_, _>>();
    let rs = records
        .iter()
        .filter(|r| by_id.contains_key(text(r, "task_id").unwrap()))
        .collect::<Vec<_>>();
    let valid = rs.iter().filter(|r| r["valid"] == true).count();
    let scorable = rs
        .iter()
        .filter(|r| {
            let task = by_id[text(r, "task_id").unwrap()];
            !task["expected"].is_null() && task["provenance"]["exclude_reason"].is_null()
        })
        .copied()
        .collect::<Vec<_>>();
    let probs = scorable
        .iter()
        .filter(|r| r["probs"].is_object())
        .collect::<Vec<_>>();
    let mut brier = Vec::new();
    let mut pairs = Vec::new();
    let mut mae = Vec::new();
    for row in &probs {
        let task = by_id[text(row, "task_id")?];
        let p = row["probs"].as_object().context("probs")?;
        let gold = if task["expected"].is_string() {
            text(task, "expected")?.to_owned()
        } else {
            task["expected"].as_i64().unwrap().to_string()
        };
        let labels = task["labels"].as_array().unwrap();
        brier.push(
            labels
                .iter()
                .map(|label| {
                    let label = label.as_str().unwrap();
                    let delta = p[label].as_f64().unwrap() - f64::from(label == gold);
                    delta * delta
                })
                .sum::<f64>(),
        );
        let top = p.values().map(|v| v.as_f64().unwrap()).fold(0., f64::max);
        let predicted = p
            .iter()
            .max_by(|a, b| {
                a.1.as_f64()
                    .unwrap()
                    .total_cmp(&b.1.as_f64().unwrap())
                    .then_with(|| b.0.cmp(a.0))
            })
            .unwrap()
            .0;
        pairs.push((top, predicted == &gold));
        if task["question"]["type"] == "score" {
            let ev = p
                .iter()
                .map(|(key, value)| key.parse::<f64>().unwrap() * value.as_f64().unwrap())
                .sum::<f64>();
            mae.push((ev - task["expected"].as_f64().unwrap()).abs());
        }
    }
    let latencies = rs
        .iter()
        .filter_map(|r| r["latency_s"].as_f64())
        .collect::<Vec<_>>();
    let failures = rs
        .iter()
        .filter(|r| r["ok"] == false)
        .filter_map(|r| r["latency_s"].as_f64())
        .collect::<Vec<_>>();
    let correct = scorable.iter().filter(|r| r["correct"] == true).count();
    let cost_basis = rs
        .iter()
        .map(|r| r["cost_basis"].as_str().unwrap_or("unknown").to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    Ok(
        json!({"n_planned":tasks.len(),"n_attempted":rs.len(),"n_scorable":scorable.len(),
        "n_valid":valid,"n_correct":correct,"accuracy":optional_rate(correct,scorable.len()),
        "coverage":optional_rate(rs.len(),tasks.len()),"schema_validity":optional_rate(valid,rs.len()),
        "schema_validity_strict":optional_rate(rs.iter().filter(|r|r["strict_valid"]==true).count(),rs.len()),
        "n_renormalized":rs.iter().filter(|r|r["renormalized"]==true).count(),
        "operational_success":optional_rate(rs.iter().filter(|r|r["ok"]==true).count(),rs.len()),
        "calibration_n":probs.len(),"brier_mean":mean(&brier),"ece":if pairs.is_empty(){Value::Null}else{ece(&pairs)},
        "ordinal_mae":mean(&mae),"latency":latency_summary(&latencies),"latency_failures":latency_summary(&failures),
        "price_per_1000_decisions_usd":null,"cost_basis":cost_basis}),
    )
}
fn agreement(tasks: &[Value], records: &[Value]) -> Value {
    let by_id = records
        .iter()
        .map(|r| (text(r, "task_id").unwrap(), r))
        .collect::<HashMap<_, _>>();
    let mut groups = BTreeMap::<String, Vec<&Value>>::new();
    for task in tasks {
        if let Some(group) = task["group"].as_str()
            && !group.is_empty()
        {
            groups.entry(group.to_owned()).or_default().push(task);
        }
    }
    let pairs = groups
        .values()
        .filter(|rows| rows.len() == 2)
        .collect::<Vec<_>>();
    let mut both_valid = 0;
    let mut agree = 0;
    let mut both_correct = 0;
    for pair in &pairs {
        let a = by_id.get(text(pair[0], "id").unwrap());
        let b = by_id.get(text(pair[1], "id").unwrap());
        if let (Some(a), Some(b)) = (a, b)
            && a["valid"] == true
            && b["valid"] == true
        {
            both_valid += 1;
            agree += usize::from(a["predicted"] == b["predicted"]);
            both_correct += usize::from(a["correct"] == true && b["correct"] == true);
        }
    }
    json!({"pairs":pairs.len(),"both_valid":both_valid,"agree":agree,
        "agreement":optional_rate(agree,both_valid),"both_correct_rate_all_pairs":optional_rate(both_correct,pairs.len())})
}
fn summarize(tasks: &[Value], records: &[Value]) -> Result<Value> {
    ensure!(
        records
            .iter()
            .map(|r| text(r, "task_id").unwrap())
            .collect::<HashSet<_>>()
            .len()
            == records.len(),
        "Duplicate per-item records"
    );
    let ids = tasks
        .iter()
        .map(|t| text(t, "id").unwrap())
        .collect::<HashSet<_>>();
    ensure!(
        records
            .iter()
            .all(|r| ids.contains(text(r, "task_id").unwrap())),
        "Unknown task in run"
    );
    let all = tasks.iter().collect::<Vec<_>>();
    let mut summary = metric(&all, records)?;
    let families = tasks
        .iter()
        .map(|t| text(t, "family").unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let mut per_family = Map::new();
    for family in families {
        let subset = tasks
            .iter()
            .filter(|t| t["family"] == family)
            .collect::<Vec<_>>();
        per_family.insert(family.to_owned(), metric(&subset, records)?);
    }
    let accuracies = per_family
        .values()
        .filter_map(|v| v["accuracy"].as_f64())
        .collect::<Vec<_>>();
    let splits = tasks
        .iter()
        .map(|t| text(t, "split").unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    let mut by_split = Map::new();
    for split in splits {
        let subset = tasks
            .iter()
            .filter(|t| t["split"] == split)
            .collect::<Vec<_>>();
        by_split.insert(split.to_owned(), metric(&subset, records)?);
    }
    let models = records
        .iter()
        .map(|r| r["model"].as_str().unwrap_or("unknown").to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let sources = records
        .iter()
        .map(|r| r["probs_source"].as_str().unwrap_or("unknown").to_owned())
        .collect::<std::collections::BTreeSet<_>>();
    let extra = json!({"version":"jevbench-v1","macro_accuracy":mean(&accuracies),"per_family":per_family,
        "complete":records.len()==tasks.len(),"paraphrase_consistency":agreement(tasks,records),
        "model_identities":models,"probability_sources":sources,"splits":by_split,"ledger_charged_usd":null});
    summary
        .as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    Ok(summary)
}
fn add_public_fields(
    summary: &mut Value,
    tasks: &[Value],
    records: &[Value],
    selective: &[Value],
    manifest: &Value,
    gpu_memory: Option<f64>,
) -> Result<()> {
    let by_id = records
        .iter()
        .map(|r| (text(r, "task_id").unwrap(), r))
        .collect::<HashMap<_, _>>();
    let mut tiers = Map::new();
    for (tier, _) in SPLITS {
        let subset = tasks
            .iter()
            .filter(|t| text(t, "id").unwrap().starts_with(&format!("{tier}-")))
            .cloned()
            .collect::<Vec<_>>();
        let ids = subset
            .iter()
            .map(|t| text(t, "id").unwrap())
            .collect::<HashSet<_>>();
        let records = records
            .iter()
            .filter(|r| ids.contains(text(r, "task_id").unwrap()))
            .cloned()
            .collect::<Vec<_>>();
        tiers.insert(tier.to_owned(), summarize(&subset, &records)?);
    }
    let accepted = selective.iter().filter(|r| r["accepted"] == true).count();
    let correct = selective.iter().filter(|r| r["correct"] == true).count();
    let abstained = selective.iter().filter(|r| r["abstained"] == true).count();
    let errors = selective.iter().filter(|r| r["error"] == true).count();
    let distances = tasks
        .iter()
        .filter_map(|task| {
            let gold = task["provenance"]["gold_probs"].as_object()?;
            let row = *by_id.get(text(task, "id").ok()?)?;
            if row["valid"] != true {
                return None;
            }
            let p = row["probs"].as_object()?;
            Some(
                task["labels"]
                    .as_array()?
                    .iter()
                    .map(|label| {
                        let key = label.as_str().unwrap();
                        (p.get(key).and_then(Value::as_f64).unwrap_or(0.)
                            - gold.get(key).and_then(Value::as_f64).unwrap_or(0.))
                        .abs()
                    })
                    .sum::<f64>()
                    / 2.,
            )
        })
        .collect::<Vec<_>>();
    summary["backend"] = records
        .first()
        .and_then(|_| Value::as_object(&manifest["backend"]))
        .map(|_| manifest["backend"].clone())
        .unwrap_or(Value::Null);
    summary["gpu_memory"] = json!({"peak_board_used_mib":gpu_memory,"sampling_ms":200,"scope":"whole GPU, includes baseline and other processes"});
    summary["max_input_tokens"] = json!(
        records
            .iter()
            .filter_map(|r| r["usage"]["input_tokens"].as_u64())
            .max()
            .unwrap_or(0)
    );
    summary["tiers"] = json!(tiers);
    summary["selective_policy"] = json!({"accepted":accepted,"correct":correct,"abstained":abstained,"error":errors,
        "wrong_accepted":accepted-correct,"total":tasks.len(),"coverage":accepted as f64/tasks.len() as f64,
        "accepted_accuracy":optional_rate(correct,accepted)});
    summary["hard_gold_distribution_tvd"] = json!({"n":distances.len(),"mean":mean(&distances)});
    summary["scope"] = manifest["scope"].clone();
    summary["latency_scope"] = manifest["latency_scope"].clone();
    Ok(())
}
fn score_existing(
    prepared: &Path,
    predictions: &Path,
    model_name: &str,
    output: &Path,
) -> Result<Value> {
    let tasks = read_jsonl(&prepared.join("tasks-with-gold.jsonl"))?;
    ensure!(tasks.len() == 231, "Frozen task count changed");
    let raw = read_jsonl(predictions)?;
    let (records, selective) = score_predictions(&tasks, &raw, model_name)?;
    let mut manifest = read_json(&prepared.join("manifest.json"))?;
    manifest["backend"] = raw
        .iter()
        .find_map(|r| r.get("response").map(|v| v["backend"].clone()))
        .unwrap_or(Value::Null);
    if manifest["latency_scope"].is_null() {
        manifest["latency_scope"] = json!(
            "Rust decide_batch serial calls, batch size 1; no HTTP; load and one warmup excluded"
        );
    }
    let memory = if prepared.join("gpu-memory.csv").exists() {
        fs::read_to_string(prepared.join("gpu-memory.csv"))?
            .lines()
            .filter_map(|line| {
                let parts = line.split(',').collect::<Vec<_>>();
                if parts.len() == 3 {
                    parts[1].trim().parse::<f64>().ok()
                } else {
                    None
                }
            })
            .reduce(f64::max)
    } else {
        None
    };
    let mut summary = summarize(&tasks, &records)?;
    add_public_fields(
        &mut summary,
        &tasks,
        &records,
        &selective,
        &manifest,
        memory,
    )?;
    fs::create_dir_all(output)?;
    write_jsonl(&output.join("jevbench-records.jsonl"), &records)?;
    write_jsonl(&output.join("selective-policy.jsonl"), &selective)?;
    write_json(&output.join("summary.json"), &summary)?;
    Ok(summary)
}
#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    jevbench: &Path,
    evaluator: &Path,
    model: &Path,
    output: &Path,
    context: usize,
    threads: usize,
    gpu_layers: Option<usize>,
    cpu_moe_layers: usize,
    model_load_mode: &str,
    device: &str,
    expected_gpu: Option<&str>,
    timeout: u64,
) -> Result<()> {
    ensure!(
        threads >= 1
            && ["cuda", "cpu"].contains(&device)
            && ["auto", "read"].contains(&model_load_mode),
        "invalid placement or runtime argument"
    );
    ensure!(
        device != "cpu" || (cpu_moe_layers == 0 && gpu_layers.is_none_or(|n| n == 0)),
        "CPU loading does not accept CUDA placement requests"
    );
    let mut manifest = prepare(jevbench, output)?;
    let evaluator = evaluator.canonicalize()?;
    let model = model.canonicalize()?;
    let prediction = output
        .join("predictions.jsonl")
        .canonicalize()
        .unwrap_or(output.join("predictions.jsonl"));
    let mut command = vec![
        "--model".to_owned(),
        model.to_string_lossy().into_owned(),
        "--input".to_owned(),
        output
            .join("requests.jsonl")
            .canonicalize()?
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        prediction.to_string_lossy().into_owned(),
        "--context".to_owned(),
        context.to_string(),
        "--batch".to_owned(),
        "256".to_owned(),
        "--threads".to_owned(),
        threads.to_string(),
        "--execution-mode".to_owned(),
        "fresh".to_owned(),
        "--prompt-layout".to_owned(),
        "legacy".to_owned(),
        "--warmup".to_owned(),
    ];
    if device == "cuda" {
        command.push("--cuda".to_owned());
    }
    if let Some(layers) = gpu_layers {
        command.extend(["--gpu-layers".to_owned(), layers.to_string()]);
    }
    if cpu_moe_layers > 0 {
        command.extend(["--cpu-moe-layers".to_owned(), cpu_moe_layers.to_string()]);
    }
    if model_load_mode != "auto" {
        command.extend(["--model-load-mode".to_owned(), model_load_mode.to_owned()]);
    }
    let mut recorded = vec![evaluator.to_string_lossy().into_owned()];
    recorded.extend(command.clone());
    manifest["evaluator_sha256"] = json!(sha256(&evaluator)?);
    manifest["model_sha256"] = json!(sha256(&model)?);
    manifest["model_name"] = json!(model.file_name().unwrap().to_string_lossy());
    manifest["device"] = json!(device);
    manifest["expected_gpu"] = json!(expected_gpu);
    manifest["placement"] = json!({"gpu_layers":gpu_layers,"cpu_moe_layers":cpu_moe_layers});
    manifest["threads"] = json!(threads);
    manifest["model_load_mode"] = json!(model_load_mode);
    manifest["adapter_sha256"] = json!(digest_bytes(include_bytes!("jevbench_public.rs")));
    manifest["command"] = json!(recorded);
    manifest["adapters"] = json!({"lora":null,"output_head":null,"calibration":null});
    manifest["latency_scope"] = json!(
        "Rust decide_batch serial calls, batch size 1; no HTTP; load and one warmup excluded"
    );
    write_json(&output.join("manifest.json"), &manifest)?;
    let stderr = File::create(output.join("inference.stderr.log"))?;
    let mut monitor = if device == "cuda" {
        let gpu_log = File::create(output.join("gpu-memory.csv"))?;
        Command::new("nvidia-smi")
            .args([
                "--query-gpu=timestamp,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
                "--loop-ms=200",
            ])
            .stdout(Stdio::from(gpu_log))
            .stderr(Stdio::null())
            .spawn()
            .ok()
    } else {
        None
    };
    let mut child = Command::new(&evaluator)
        .args(&command)
        .stderr(Stdio::from(stderr))
        .spawn()?;
    let start = Instant::now();
    let mut timed_out = false;
    let result = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if start.elapsed() > Duration::from_secs(timeout) {
            child.kill()?;
            timed_out = true;
            break child.wait()?;
        }
        std::thread::sleep(Duration::from_millis(100));
    };
    if let Some(mut process) = monitor.take() {
        let _ = process.kill();
        let _ = process.wait();
    }
    manifest["inference_exit_code"] = json!(result.code());
    write_json(&output.join("manifest.json"), &manifest)?;
    ensure!(!timed_out, "Inference timed out; retained partial evidence");
    ensure!(
        result.success(),
        "Inference failed; retained partial evidence, no complete-result claim"
    );
    let raw = read_jsonl(&prediction)?;
    for row in &raw {
        let Some(backend) = row.get("response").map(|r| &r["backend"]) else {
            continue;
        };
        ensure!(
            backend["compute"]["model_load_mode"]
                .as_str()
                .unwrap_or("auto")
                == model_load_mode,
            "Reported model loading mode differs from requested configuration"
        );
        ensure!(
            backend["compute"]["gpu_layers"] == json!(gpu_layers)
                && backend["compute"]["cpu_moe_layers"].as_u64().unwrap_or(0)
                    == cpu_moe_layers as u64,
            "Reported CPU/GPU placement differs from requested configuration"
        );
        if device == "cuda" {
            ensure!(
                backend["offload_requested"] == true
                    && backend["offload_device"].as_str().is_some_and(
                        |s| !s.is_empty() && expected_gpu.is_none_or(|e| s.contains(e))
                    ),
                "Run did not use CUDA or expected GPU"
            );
        }
    }
    let model_name = format!("{}/l2s1", model.file_name().unwrap().to_string_lossy());
    let summary = score_existing(output, &prediction, &model_name, output)?;
    let errors = summary["selective_policy"]["error"].as_u64().unwrap_or(0);
    println!(
        "ALL: {}/{}; errors={errors}; output={}",
        summary["n_correct"],
        summary["n_scorable"],
        output.display()
    );
    ensure!(errors == 0, "Public run contains failed predictions");
    Ok(())
}
pub fn run(args: Args) -> Result<()> {
    match args.command {
        Action::Prepare { jevbench, output } => {
            let manifest = prepare(&jevbench, &output)?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
            Ok(())
        }
        Action::Score {
            prepared,
            predictions,
            model_name,
            output,
        } => {
            let summary = score_existing(&prepared, &predictions, &model_name, &output)?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            Ok(())
        }
        Action::Run {
            jevbench,
            evaluator,
            model,
            output,
            context,
            threads,
            gpu_layers,
            cpu_moe_layers,
            model_load_mode,
            device,
            expected_gpu,
            timeout,
        } => run_benchmark(
            &jevbench,
            &evaluator,
            &model,
            &output,
            context,
            threads,
            gpu_layers,
            cpu_moe_layers,
            &model_load_mode,
            &device,
            expected_gpu.as_deref(),
            timeout,
        ),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn request_excludes_gold_and_preserves_canonical_label_order() {
        let task = json!({"id":"case","family":"intent","split":"public","state":{"text":"payload"},
            "question":{"type":"choice","instructions":"Choose.","criteria":{"apple":"A","zebra":"Z"}},
            "labels":["zebra","apple"],"expected":"zebra","provenance":{"rationale":"SECRET"}});
        let output = request(&task).unwrap();
        assert_eq!(
            output["request"]["decisions"][0]["kind"]["options"][0]["id"],
            "zebra"
        );
        assert!(!serde_json::to_string(&output).unwrap().contains("SECRET"));
        assert!(!serde_json::to_string(&output).unwrap().contains("expected"));
    }
    #[test]
    fn rounded_distribution_is_renormalized_and_scored() {
        let task = json!({"id":"score","family":"ordinal","split":"public","state":"x",
            "question":{"type":"score","instructions":"Rate","criteria":["low","high"]},
            "labels":["0","1"],"expected":1});
        let probs = json!({"0":0.499,"1":0.501});
        let result = score_task(&task, Some(probs.as_object().unwrap()));
        assert_eq!(result["valid"], true);
        assert_eq!(result["strict_valid"], true);
        assert_eq!(result["correct"], true);
        let rounded = json!({"0":0.495,"1":0.5});
        let result = score_task(&task, Some(rounded.as_object().unwrap()));
        assert_eq!(result["renormalized"], true);
        assert!((result["probs"]["1"].as_f64().unwrap() - 0.5 / 0.995).abs() < 1e-12);
    }
}
