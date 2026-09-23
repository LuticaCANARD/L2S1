use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    parquet_data,
    python_random::PythonRandom,
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args as ClapArgs, Subcommand};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const SEED: u64 = 20260923;
const EVALUATOR_SHA256: &str = "f7fdc30802e4295c3423a7c04037e8e4ad246405e18b1479eb521cd611b5a19e";

#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    action: Action,
    #[arg(long, global = true)]
    study: Option<PathBuf>,
    #[arg(long, global = true)]
    evaluator: Option<PathBuf>,
    #[arg(long, global = true)]
    plan: Option<PathBuf>,
}
#[derive(Subcommand)]
enum Action {
    Prepare,
    Run,
    Score,
}

fn labels(value: &Value) -> Result<Vec<String>> {
    value
        .as_array()
        .context("labels must be a list")?
        .iter()
        .map(|v| v.as_str().context("label must be text").map(str::to_owned))
        .collect()
}
fn groups(labels: &[String]) -> Result<Vec<Vec<String>>> {
    ensure!(
        [60, 77].contains(&labels.len())
            && labels.iter().collect::<HashSet<_>>().len() == labels.len(),
        "invalid intent labels"
    );
    let mut sorted = labels.to_vec();
    sorted.sort();
    Ok((0..3)
        .map(|group| sorted.iter().skip(group).step_by(3).cloned().collect())
        .collect())
}
pub fn request(item: &Value, labels: &[String], suffix: &str) -> Result<String> {
    let dataset = item["dataset"].as_str().context("dataset")?;
    let instruction = if dataset == "banking77-en" {
        "Choose the intent that best matches the customer utterance. Select exactly one of the listed intent labels."
    } else {
        "사용자 발화에 가장 잘 맞는 의도를 선택하세요. 나열된 의도 라벨 중 정확히 하나를 선택하세요."
    };
    let id = format!("{}{}", item["id"].as_str().context("ID")?, suffix);
    let options = labels
        .iter()
        .map(|label| {
            Ok(format!(
                "{{\"id\": {}, \"criterion\": {}}}",
                serde_json::to_string(label)?,
                serde_json::to_string(&label.replace('_', " "))?
            ))
        })
        .collect::<Result<Vec<_>>>()?
        .join(", ");
    Ok(format!(
        "{{\"id\": {}, \"request\": {{\"state\": {{\"utterance\": {}}}, \"decisions\": [{{\"id\": \"intent\", \"instruction\": {}, \"kind\": {{\"type\": \"choice\", \"options\": [{}]}}}}]}}}}",
        serde_json::to_string(&id)?,
        serde_json::to_string(item["text"].as_str().context("text")?)?,
        serde_json::to_string(instruction)?,
        options
    ))
}
pub fn write_lines(path: &Path, lines: impl IntoIterator<Item = String>) -> Result<()> {
    let mut file = File::create(path)?;
    for line in lines {
        writeln!(file, "{line}")?;
    }
    file.flush()?;
    Ok(())
}
fn prepare(root: &Path) -> Result<()> {
    let out = root.join("prepared");
    fs::create_dir(&out)?;
    let banking = root.join("data/banking77");
    let massive = root.join("data/massive");
    let bank_labels = labels(&read_json(&banking.join("categories.json"))?)?;
    let mut reader = csv::Reader::from_path(banking.join("test.csv"))?;
    let headers = reader.headers()?.clone();
    let text = headers
        .iter()
        .position(|v| v == "text")
        .context("BANKING77 text column")?;
    let category = headers
        .iter()
        .position(|v| v == "category")
        .context("BANKING77 category column")?;
    let bank=reader.records().enumerate().map(|(i,row)|{let row=row?;Ok(json!({"id":format!("banking77-en:{i}"),"dataset":"banking77-en","source_id":i,
        "text":row.get(text).context("missing text")?,"expected":row.get(category).context("missing category")?}))}).collect::<Result<Vec<_>>>()?;
    let parquet = massive.join("ko-KR-test.parquet");
    let metadata = parquet_data::metadata(&parquet, "huggingface")?;
    let mass_labels = labels(&metadata["info"]["features"]["intent"]["names"])?;
    let mass = parquet_data::rows(&parquet)?
        .into_iter()
        .map(|row| {
            ensure!(
                row["locale"] == "ko-KR" && row["partition"] == "test",
                "unexpected MASSIVE row"
            );
            let id = row["id"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| row["id"].to_string());
            let intent = row["intent"].as_u64().context("invalid intent index")? as usize;
            Ok(
                json!({"id":format!("massive-ko:{id}"),"dataset":"massive-ko","source_id":row["id"],
            "text":row["utt"],"expected":mass_labels.get(intent).context("unknown intent index")?}),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        bank.len() == 3080
            && mass.len() == 2974
            && bank_labels.len() == 77
            && mass_labels.len() == 60,
        "frozen test sizes changed"
    );
    let mut samples = Vec::new();
    let mut datasets = serde_json::Map::new();
    for (name, population, labels) in [
        ("banking77-en", bank, bank_labels),
        ("massive-ko", mass, mass_labels),
    ] {
        let label_set = labels.iter().map(String::as_str).collect::<HashSet<_>>();
        ensure!(
            population.iter().all(|r| r["expected"]
                .as_str()
                .is_some_and(|v| label_set.contains(v))),
            "unknown gold label"
        );
        let mut rng = PythonRandom::new(SEED);
        let mut indices = rng.sample_indices(population.len(), 200);
        indices.sort();
        let selected = indices
            .iter()
            .map(|i| population[*i].clone())
            .collect::<Vec<_>>();
        let mut counts = BTreeMap::new();
        for row in &selected {
            *counts
                .entry(row["expected"].as_str().unwrap().to_owned())
                .or_insert(0usize) += 1;
        }
        let present = population
            .iter()
            .map(|r| r["expected"].as_str().unwrap())
            .collect::<HashSet<_>>();
        let mut absent = labels
            .iter()
            .filter(|label| !present.contains(label.as_str()))
            .cloned()
            .collect::<Vec<_>>();
        absent.sort();
        let mut sorted_labels = labels.clone();
        sorted_labels.sort();
        datasets.insert(name.to_owned(),json!({"test_population":population.len(),"sample_size":200,"labels":sorted_labels,
            "groups":groups(&labels)?,"sampled_indices":indices,"sampled_class_counts":counts,"labels_absent_from_test":absent}));
        samples.extend(selected);
    }
    write_lines(
        &out.join("samples.jsonl"),
        samples.iter().map(|r| {
            format!(
                "{{\"id\": {}, \"dataset\": {}, \"text\": {}}}",
                serde_json::to_string(&r["id"]).unwrap(),
                serde_json::to_string(&r["dataset"]).unwrap(),
                serde_json::to_string(&r["text"]).unwrap()
            )
        }),
    )?;
    write_lines(&out.join("gold.jsonl"),samples.iter().map(|r|format!("{{\"id\": {}, \"dataset\": {}, \"source_id\": {}, \"text\": {}, \"expected\": {}}}",
        serde_json::to_string(&r["id"]).unwrap(),serde_json::to_string(&r["dataset"]).unwrap(),serde_json::to_string(&r["source_id"]).unwrap(),
        serde_json::to_string(&r["text"]).unwrap(),serde_json::to_string(&r["expected"]).unwrap())))?;
    let mut requests = Vec::new();
    for row in &samples {
        for (i, group) in datasets[row["dataset"].as_str().unwrap()]["groups"]
            .as_array()
            .context("missing groups")?
            .iter()
            .enumerate()
        {
            requests.push(request(row, &labels(group)?, &format!(":g{i}"))?);
        }
    }
    write_lines(&out.join("stage1-requests.jsonl"), requests)?;
    write_json(&out.join("datasets.json"), &Value::Object(datasets))?;
    let mut hashes = BTreeMap::new();
    for (key, path) in [
        ("data/banking77/test.csv", banking.join("test.csv")),
        (
            "data/banking77/categories.json",
            banking.join("categories.json"),
        ),
        ("data/massive/ko-KR-test.parquet", parquet),
    ] {
        hashes.insert(key, sha256(&path)?);
    }
    let mut prepared = BTreeMap::new();
    for name in [
        "samples.jsonl",
        "gold.jsonl",
        "stage1-requests.jsonl",
        "datasets.json",
    ] {
        prepared.insert(name, sha256(&out.join(name))?);
    }
    write_json(
        &out.join("manifest.json"),
        &json!({"seed":SEED,"sampling":"Uniform without replacement, original test row order; independent seeded RNG per dataset",
        "grouping":"Alphabetically sorted official labels, round-robin into three fixed groups",
        "method":"3 group argmax winners, then final argmax among all 3; no gold-based shortlist",
        "sources":{"banking77":"https://github.com/PolyAI-LDN/task-specific-datasets/tree/master/banking_data",
            "massive":"https://huggingface.co/datasets/AmazonScience/massive/blob/6e31162aba58a715666d3791566f42afdcfa62b2/ko-KR/massive-test.parquet"},
        "licenses":{"banking77":"CC BY 4.0, PolyAI","massive":"CC BY 4.0, Amazon"},"raw_sha256":hashes,"prepared_sha256":prepared}),
    )?;
    println!("prepared 400 intent examples and 1200 stage-one requests");
    Ok(())
}

fn validated(predictions: Vec<Value>, requests: &[Value]) -> Result<HashMap<String, Value>> {
    let expected = requests
        .iter()
        .map(|r| Ok((r["id"].as_str().context("request ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(
        expected.len() == requests.len() && predictions.len() == requests.len(),
        "prediction count mismatch"
    );
    let mut output = HashMap::new();
    for prediction in predictions {
        let id = prediction["id"]
            .as_str()
            .context("prediction ID")?
            .to_owned();
        let request = expected.get(&id).context("unknown prediction ID")?;
        ensure!(
            !output.contains_key(&id) && prediction.get("error").is_none(),
            "duplicate or error prediction"
        );
        let backend = &prediction["response"]["backend"];
        ensure!(
            backend["offload_device"]
                .as_str()
                .is_some_and(|v| v.contains("RTX 3060"))
                && backend["offload_requested"] == true,
            "unexpected GPU"
        );
        ensure!(
            backend["execution_mode"] == "fresh" && backend["prompt_layout"] == "legacy",
            "unexpected execution mode"
        );
        ensure!(
            backend["compute"]
                == json!({"batch":256,"context":8192,"flash_attention":"off","threads":4,"ubatch":256}),
            "compute settings changed"
        );
        let results = prediction["response"]["results"]
            .as_array()
            .context("missing results")?;
        ensure!(results.len() == 1, "wrong result count");
        let result = &results[0];
        ensure!(
            result["id"] == "intent"
                && result["truncated"] == false
                && result["reused_prefix_tokens"] == 0,
            "invalid result"
        );
        let scores = result["scores"].as_array().context("missing scores")?;
        let options = request["request"]["decisions"][0]["kind"]["options"]
            .as_array()
            .context("missing options")?;
        ensure!(
            scores
                .iter()
                .map(|s| &s["id"])
                .eq(options.iter().map(|o| &o["id"])),
            "option order changed"
        );
        let probabilities = scores
            .iter()
            .map(|s| {
                s["option_probability"]
                    .as_f64()
                    .context("missing probability")
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            probabilities
                .iter()
                .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
                && (probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9,
            "invalid probability distribution"
        );
        ensure!(
            prediction["elapsed_ms"]
                .as_f64()
                .is_some_and(|v| v.is_finite() && v >= 0.0),
            "invalid elapsed_ms"
        );
        ensure!(
            prediction["response"]["policy"]
                == json!({"min_top_probability":0.8,"min_candidate_mass":0.05}),
            "policy changed"
        );
        output.insert(id, prediction);
    }
    Ok(output)
}

pub fn top(prediction: &Value) -> Result<&str> {
    let scores = prediction["response"]["results"][0]["scores"]
        .as_array()
        .context("missing scores")?;
    scores
        .iter()
        .min_by(|a, b| {
            b["option_probability"]
                .as_f64()
                .unwrap_or(f64::NAN)
                .total_cmp(&a["option_probability"].as_f64().unwrap_or(f64::NAN))
                .then_with(|| {
                    a["id"]
                        .as_str()
                        .unwrap_or("")
                        .cmp(b["id"].as_str().unwrap_or(""))
                })
        })
        .and_then(|s| s["id"].as_str())
        .context("empty scores")
}

fn route(samples: &[Value], first: &HashMap<String, Value>) -> Result<Vec<String>> {
    samples
        .iter()
        .map(|item| {
            let id = item["id"].as_str().context("sample ID")?;
            let winners = (0..3)
                .map(|i| {
                    top(first
                        .get(&format!("{id}:g{i}"))
                        .context("missing group prediction")?)
                    .map(str::to_owned)
                })
                .collect::<Result<Vec<_>>>()?;
            request(item, &winners, ":final")
        })
        .collect()
}

pub fn quantile(values: &[f64], fraction: f64) -> Result<f64> {
    ensure!(!values.is_empty(), "empty values");
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let point = (values.len() - 1) as f64 * fraction;
    let low = point.floor() as usize;
    let high = point.ceil() as usize;
    Ok(values[low] + (values[high] - values[low]) * (point - low as f64))
}

fn selected(row: &Value) -> Option<&str> {
    row["response"]["results"][0]["value"]["selected"].as_str()
}

fn score(root: &Path, out: &Path) -> Result<Value> {
    let prepared = root.join("prepared");
    let first = validated(
        read_jsonl(&out.join("stage1-predictions.jsonl"))?,
        &read_jsonl(&prepared.join("stage1-requests.jsonl"))?,
    )?;
    let samples = read_jsonl(&prepared.join("samples.jsonl"))?;
    let expected = route(&samples, &first)?;
    let final_requests = read_jsonl(&out.join("final-requests.jsonl"))?;
    ensure!(
        final_requests
            == expected
                .iter()
                .map(|line| serde_json::from_str::<Value>(line))
                .collect::<std::result::Result<Vec<_>, _>>()?,
        "final shortlist not reproduced"
    );
    let final_predictions = validated(
        read_jsonl(&out.join("final-predictions.jsonl"))?,
        &final_requests,
    )?;
    let gold = read_jsonl(&prepared.join("gold.jsonl"))?;
    let mut details = Vec::new();
    for item in &gold {
        let id = item["id"].as_str().context("gold ID")?;
        let groups = (0..3)
            .map(|i| {
                first
                    .get(&format!("{id}:g{i}"))
                    .context("missing group result")
            })
            .collect::<Result<Vec<_>>>()?;
        let final_row = final_predictions
            .get(&format!("{id}:final"))
            .context("missing final result")?;
        let label = top(final_row)?;
        let winner = groups
            .iter()
            .find(|group| top(group).ok() == Some(label))
            .context("finalist not from group")?;
        let accepted = selected(final_row) == Some(label) && selected(winner) == Some(label);
        let expected = item["expected"].as_str().context("gold label")?;
        let elapsed = final_row["elapsed_ms"]
            .as_f64()
            .context("final elapsed_ms")?
            + groups
                .iter()
                .map(|g| g["elapsed_ms"].as_f64().unwrap_or(0.0))
                .sum::<f64>();
        let max_tokens = groups
            .iter()
            .copied()
            .chain([final_row])
            .filter_map(|g| g["response"]["results"][0]["input_tokens"].as_u64())
            .max()
            .context("missing input tokens")?;
        let winners = groups.iter().map(|g| top(g)).collect::<Result<Vec<_>>>()?;
        details.push(json!({"id":id,"dataset":item["dataset"],"expected":expected,"predicted":label,"correct":label==expected,
            "accepted":accepted,"gold_reached_final":winners.contains(&expected),"elapsed_ms":elapsed,"max_input_tokens":max_tokens}));
    }
    write_lines(
        &out.join("scored.jsonl"),
        details
            .iter()
            .map(|row| serde_json::to_string(row).unwrap()),
    )?;
    let mut summaries = serde_json::Map::new();
    for name in ["banking77-en", "massive-ko"] {
        let subset = details
            .iter()
            .filter(|r| r["dataset"] == name)
            .collect::<Vec<_>>();
        ensure!(subset.len() == 200, "missing scored cases");
        let correct = subset.iter().filter(|r| r["correct"] == true).count();
        let accepted = subset
            .iter()
            .filter(|r| r["accepted"] == true)
            .collect::<Vec<_>>();
        let accepted_correct = accepted.iter().filter(|r| r["correct"] == true).count();
        let latencies = subset
            .iter()
            .map(|r| r["elapsed_ms"].as_f64().unwrap())
            .collect::<Vec<_>>();
        let accuracy = correct as f64 / 200.0;
        let z = 1.959963984540054;
        let denominator = 1.0 + z * z / 200.0;
        let center = (accuracy + z * z / 400.0) / denominator;
        let half = z * (accuracy * (1.0 - accuracy) / 200.0 + z * z / (4.0 * 200.0 * 200.0)).sqrt()
            / denominator;
        let mut counts = HashMap::<(String, String), usize>::new();
        let mut order = Vec::new();
        for row in subset.iter().filter(|r| r["correct"] == false) {
            let key = (
                row["expected"].as_str().unwrap().to_owned(),
                row["predicted"].as_str().unwrap().to_owned(),
            );
            if !counts.contains_key(&key) {
                order.push(key.clone());
            }
            *counts.entry(key).or_default() += 1;
        }
        order.sort_by(|a, b| counts[b].cmp(&counts[a]));
        let top_confusions = order
            .into_iter()
            .take(15)
            .map(|(a, b)| json!({"expected":a,"predicted":b,"count":counts[&(a,b)]}))
            .collect::<Vec<_>>();
        summaries.insert(name.to_owned(),json!({"total":200,"correct":correct,"accuracy":accuracy,"wilson95":[center-half,center+half],
            "accepted":accepted.len(),"accepted_correct":accepted_correct,"accepted_wrong":accepted.len()-accepted_correct,
            "accepted_accuracy":(!accepted.is_empty()).then(||accepted_correct as f64/accepted.len() as f64),"coverage":accepted.len() as f64/200.0,
            "abstained":200-accepted.len(),"gold_reached_final":subset.iter().filter(|r|r["gold_reached_final"]==true).count(),
            "errors":0,"truncated":0,"p50_ms":quantile(&latencies,0.5)?,"p95_ms":quantile(&latencies,0.95)?,
            "max_input_tokens":subset.iter().filter_map(|r|r["max_input_tokens"].as_u64()).max(),"top_confusions":top_confusions}));
    }
    let result = Value::Object(summaries);
    write_json(&out.join("summary.json"), &result)?;
    Ok(result)
}

fn evaluate(evaluator: &Path, model: &Path, input: &Path, output: &Path, log: &Path) -> Result<()> {
    let args = [
        "--model".to_owned(),
        model.display().to_string(),
        "--input".to_owned(),
        input.display().to_string(),
        "--output".to_owned(),
        output.display().to_string(),
        "--context".to_owned(),
        "8192".to_owned(),
        "--batch".to_owned(),
        "256".to_owned(),
        "--ubatch".to_owned(),
        "256".to_owned(),
        "--threads".to_owned(),
        "4".to_owned(),
        "--execution-mode".to_owned(),
        "fresh".to_owned(),
        "--prompt-layout".to_owned(),
        "legacy".to_owned(),
        "--warmup".to_owned(),
        "--cuda".to_owned(),
    ];
    write_json(
        &log.with_extension("command.json"),
        &json!(
            std::iter::once(evaluator.display().to_string())
                .chain(args.iter().cloned())
                .collect::<Vec<_>>()
        ),
    )?;
    let file = File::create(log)?;
    let mut child = Command::new(evaluator)
        .args(&args)
        .stdout(Stdio::from(file.try_clone()?))
        .stderr(Stdio::from(file))
        .spawn()?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            ensure!(status.success(), "evaluator failed: {status}");
            break;
        }
        if start.elapsed() > Duration::from_secs(3600) {
            child.kill()?;
            child.wait()?;
            bail!("evaluator timed out")
        }
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

fn run_models(root: &Path, evaluator: &Path, plan: &Path) -> Result<()> {
    ensure!(
        sha256(evaluator)? == EVALUATOR_SHA256,
        "evaluator differs from verified Gemma rebuild"
    );
    let prepared = root.join("prepared");
    let frozen = read_json(&prepared.join("manifest.json"))?;
    for (name, hash) in frozen["prepared_sha256"]
        .as_object()
        .context("missing prepared hashes")?
    {
        ensure!(
            sha256(&prepared.join(name))? == hash.as_str().context("invalid digest")?,
            "prepared data changed"
        );
    }
    let samples = read_jsonl(&prepared.join("samples.jsonl"))?;
    for model in read_json(plan)?.as_array().context("plan must be a list")? {
        let name = model["id"].as_str().context("model ID")?;
        let path = PathBuf::from(model["path"].as_str().context("model path")?);
        ensure!(
            sha256(&path)? == model["sha256"].as_str().context("model digest")?,
            "model digest changed"
        );
        let out = root.join("runs").join(name);
        fs::create_dir(&out)?;
        let start = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        write_json(
            &out.join("manifest.json"),
            &json!({"model":model,"evaluator_sha256":sha256(evaluator)?,
            "prepared_manifest_sha256":sha256(&prepared.join("manifest.json"))?,"method":"3 groups plus final; four fresh calls per example","start_unix":start}),
        )?;
        evaluate(
            evaluator,
            &path,
            &prepared.join("stage1-requests.jsonl"),
            &out.join("stage1-predictions.jsonl"),
            &out.join("stage1.log"),
        )?;
        let first = validated(
            read_jsonl(&out.join("stage1-predictions.jsonl"))?,
            &read_jsonl(&prepared.join("stage1-requests.jsonl"))?,
        )?;
        write_lines(&out.join("final-requests.jsonl"), route(&samples, &first)?)?;
        evaluate(
            evaluator,
            &path,
            &out.join("final-requests.jsonl"),
            &out.join("final-predictions.jsonl"),
            &out.join("final.log"),
        )?;
        score(root, &out)?;
        write_json(
            &out.join("complete.json"),
            &json!({"elapsed_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64()-start,"inference_calls":1600,"examples":400}),
        )?;
    }
    Ok(())
}

pub fn run(args: Args) -> Result<()> {
    let study = args.study.as_deref().context("--study required")?;
    match args.action {
        Action::Prepare => prepare(study),
        Action::Run => run_models(
            study,
            args.evaluator.as_deref().context("--evaluator required")?,
            args.plan.as_deref().context("--plan required")?,
        ),
        Action::Score => {
            for dir in fs::read_dir(study.join("runs"))? {
                let path = dir?.path();
                if path.is_dir() {
                    println!("{}: {}", path.display(), score(study, &path)?);
                }
            }
            Ok(())
        }
    }
}
