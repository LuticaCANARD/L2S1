use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    evaluate_intents::{quantile, request, write_lines},
};
use anyhow::{Context, Result, bail, ensure};
use clap::{Args as ClapArgs, Subcommand};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[derive(ClapArgs)]
pub struct Args {
    #[command(subcommand)]
    action: Action,
    #[arg(long, global = true)]
    study: Option<PathBuf>,
    #[arg(long, global = true)]
    base: Option<PathBuf>,
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

pub fn code(mut index: usize, count: usize) -> String {
    let mut width = 1;
    while count > 26_usize.pow(width) {
        width += 1;
    }
    let mut chars = vec!['A'; width as usize];
    for position in (0..width as usize).rev() {
        chars[position] = char::from(b'A' + (index % 26) as u8);
        index /= 26;
    }
    chars.into_iter().collect()
}

fn prepare(root: &Path, base: &Path) -> Result<()> {
    let source = base.join("prepared");
    let manifest = read_json(&source.join("manifest.json"))?;
    for (name, hash) in manifest["prepared_sha256"]
        .as_object()
        .context("missing hashes")?
    {
        ensure!(
            sha256(&source.join(name))? == hash.as_str().context("invalid hash")?,
            "source prepared file changed"
        );
    }
    let out = root.join("prepared");
    fs::create_dir(&out)?;
    for name in ["samples.jsonl", "gold.jsonl", "datasets.json"] {
        fs::copy(source.join(name), out.join(name))?;
    }
    let samples = read_jsonl(&out.join("samples.jsonl"))?;
    let datasets = read_json(&out.join("datasets.json"))?;
    ensure!(samples.len() == 400, "expected 400 samples");
    let requests = samples
        .iter()
        .map(|item| {
            let labels = datasets[item["dataset"].as_str().context("dataset")?]["labels"]
                .as_array()
                .context("missing labels")?
                .iter()
                .map(|v| v.as_str().context("invalid label").map(str::to_owned))
                .collect::<Result<Vec<_>>>()?;
            request(item, &labels, "")
        })
        .collect::<Result<Vec<_>>>()?;
    write_lines(&out.join("requests.jsonl"), requests)?;
    let mut hashes = BTreeMap::new();
    for name in [
        "samples.jsonl",
        "gold.jsonl",
        "datasets.json",
        "requests.jsonl",
    ] {
        hashes.insert(name, sha256(&out.join(name))?);
    }
    write_json(
        &out.join("manifest.json"),
        &json!({"base_manifest":manifest,
        "method":"All 77/60 labels in one decision; fixed-width codes, complete canonical token-sequence likelihoods","prepared_sha256":hashes}),
    )
}

fn score(root: &Path, out: &Path) -> Result<Value> {
    let gold = read_jsonl(&root.join("prepared/gold.jsonl"))?
        .into_iter()
        .map(|r| Ok((r["id"].as_str().context("gold ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    let requests = read_jsonl(&root.join("prepared/requests.jsonl"))?
        .into_iter()
        .map(|r| Ok((r["id"].as_str().context("request ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    let predictions = read_jsonl(&out.join("predictions.jsonl"))?;
    ensure!(
        gold.len() == 400
            && predictions.len() == 400
            && predictions
                .iter()
                .map(|p| p["id"].as_str().unwrap_or(""))
                .collect::<HashSet<_>>()
                .len()
                == 400,
        "prediction count mismatch"
    );
    let mut details = Vec::new();
    for p in predictions {
        let id = p["id"].as_str().context("prediction ID")?;
        ensure!(p.get("error").is_none(), "runtime error");
        let item = gold.get(id).context("unknown prediction ID")?;
        let original = requests.get(id).context("unknown request ID")?;
        let backend = &p["response"]["backend"];
        let result = &p["response"]["results"][0];
        ensure!(
            backend["offload_device"]
                .as_str()
                .is_some_and(|v| v.contains("RTX 3060"))
                && backend["offload_requested"] == true,
            "unexpected GPU"
        );
        ensure!(
            backend["execution_mode"] == "fresh" && backend["prompt_layout"] == "legacy",
            "execution mode changed"
        );
        ensure!(
            backend["compute"]
                == json!({"batch":256,"ubatch":256,"threads":4,"context":8192,"flash_attention":"off"}),
            "compute settings changed"
        );
        ensure!(
            backend["prompt_version"]
                .as_str()
                .is_some_and(|v| v.ends_with("/fixed-width-code-sequences-v1")),
            "prompt version changed"
        );
        ensure!(
            result["scoring_method"] == "code_sequence_conditional_softmax_v1"
                && result["truncated"] == false,
            "invalid scoring result"
        );
        let scores = result["scores"].as_array().context("missing scores")?;
        let options = original["request"]["decisions"][0]["kind"]["options"]
            .as_array()
            .context("missing options")?;
        ensure!(
            scores
                .iter()
                .map(|s| &s["id"])
                .eq(options.iter().map(|o| &o["id"])),
            "candidate order changed"
        );
        ensure!(
            scores.len()
                == if item["dataset"] == "banking77-en" {
                    77
                } else {
                    60
                },
            "candidate count changed"
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
                .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
                && (probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-8,
            "invalid probabilities"
        );
        for (i, s) in scores.iter().enumerate() {
            let token_ids = s["token_ids"].as_array().context("missing token IDs")?;
            ensure!(
                s["code"] == code(i, scores.len()) && !token_ids.is_empty(),
                "code assignment differs"
            );
            let first = token_ids[0].as_i64().context("token ID")?;
            ensure!(
                s["token_id"] == if token_ids.len() == 1 { first } else { -1 },
                "token identity differs"
            );
        }
        let maximum = scores
            .iter()
            .filter_map(|s| s["raw_logit"].as_f64())
            .fold(f64::NEG_INFINITY, f64::max);
        let norm = scores
            .iter()
            .map(|s| (s["raw_logit"].as_f64().unwrap() - maximum).exp())
            .sum::<f64>();
        for s in scores {
            ensure!(
                (s["option_probability"].as_f64().unwrap()
                    - (s["raw_logit"].as_f64().unwrap() - maximum).exp() / norm)
                    .abs()
                    < 1e-9,
                "softmax mismatch"
            );
        }
        ensure!(
            (result["candidate_mass"]
                .as_f64()
                .context("candidate mass")?
                - scores
                    .iter()
                    .map(|s| s["raw_logit"].as_f64().unwrap().exp())
                    .sum::<f64>())
            .abs()
                < 1e-8,
            "candidate mass mismatch"
        );
        let picked = scores
            .iter()
            .min_by(|a, b| {
                b["option_probability"]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&a["option_probability"].as_f64().unwrap())
                    .then_with(|| a["id"].as_str().unwrap().cmp(b["id"].as_str().unwrap()))
            })
            .context("empty scores")?["id"]
            .as_str()
            .unwrap();
        let accepted = !result["value"]["selected"].is_null();
        ensure!(
            !accepted || result["value"]["selected"] == picked,
            "selected option differs"
        );
        ensure!(
            p["response"]["policy"] == json!({"min_top_probability":0.8,"min_candidate_mass":0.05}),
            "policy changed"
        );
        let best = probabilities
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let should_accept = best >= 0.8
            && result["candidate_mass"].as_f64().unwrap() >= 0.05
            && probabilities
                .iter()
                .filter(|v| (**v - best).abs() < 1e-12)
                .count()
                == 1;
        ensure!(accepted == should_accept, "acceptance differs");
        let expected = item["expected"].as_str().context("gold label")?;
        let brier = scores
            .iter()
            .map(|s| {
                (s["option_probability"].as_f64().unwrap() - f64::from(s["id"] == expected)).powi(2)
            })
            .sum::<f64>();
        let mut token_lengths = scores
            .iter()
            .map(|s| s["token_ids"].as_array().unwrap().len())
            .collect::<Vec<_>>();
        token_lengths.sort();
        token_lengths.dedup();
        details.push(json!({"id":id,"dataset":item["dataset"],"expected":expected,"predicted":picked,"correct":picked==expected,
            "accepted":accepted,"confidence":best,"brier":brier,"elapsed_ms":p["elapsed_ms"],"input_tokens":result["input_tokens"],
            "code_prefix_evaluations":result["code_prefix_evaluations"],"token_lengths":token_lengths}));
    }
    write_lines(
        &out.join("scored.jsonl"),
        details
            .iter()
            .map(|row| serde_json::to_string(row).unwrap()),
    )?;
    let mut summaries = serde_json::Map::new();
    for name in ["banking77-en", "massive-ko"] {
        let rows = details
            .iter()
            .filter(|r| r["dataset"] == name)
            .collect::<Vec<_>>();
        ensure!(rows.len() == 200, "dataset count mismatch");
        let correct = rows.iter().filter(|r| r["correct"] == true).count();
        let accepted = rows
            .iter()
            .filter(|r| r["accepted"] == true)
            .collect::<Vec<_>>();
        let accepted_correct = accepted.iter().filter(|r| r["correct"] == true).count();
        let mut bins = vec![Vec::<&Value>::new(); 10];
        for row in &rows {
            bins[((row["confidence"].as_f64().unwrap() * 10.0) as usize).min(9)].push(row);
        }
        let accuracy = correct as f64 / 200.0;
        let z = 1.959963984540054;
        let denom = 1.0 + z * z / 200.0;
        let center = (accuracy + z * z / 400.0) / denom;
        let half = z * (accuracy * (1.0 - accuracy) / 200.0 + z * z / (4.0 * 200.0 * 200.0)).sqrt()
            / denom;
        let latencies = rows
            .iter()
            .map(|r| r["elapsed_ms"].as_f64().unwrap())
            .collect::<Vec<_>>();
        let mut prefixes = rows
            .iter()
            .filter_map(|r| r["code_prefix_evaluations"].as_u64())
            .collect::<Vec<_>>();
        prefixes.sort();
        prefixes.dedup();
        let mut lengths = rows
            .iter()
            .flat_map(|r| {
                r["token_lengths"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter_map(Value::as_u64)
            })
            .collect::<Vec<_>>();
        lengths.sort();
        lengths.dedup();
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
        summaries.insert(name.to_owned(),json!({"total":200,"correct":correct,"accuracy":accuracy,"wilson95":[center-half,center+half],
            "accepted":accepted.len(),"accepted_correct":accepted_correct,"accepted_wrong":accepted.len()-accepted_correct,
            "accepted_accuracy":(!accepted.is_empty()).then(||accepted_correct as f64/accepted.len() as f64),"coverage":accepted.len() as f64/200.0,
            "abstained":200-accepted.len(),"errors":0,"truncated":0,"brier":rows.iter().map(|r|r["brier"].as_f64().unwrap()).sum::<f64>()/200.0,
            "ece":ece,"p50_ms":quantile(&latencies,0.5)?,"p95_ms":quantile(&latencies,0.95)?,
            "max_input_tokens":rows.iter().filter_map(|r|r["input_tokens"].as_u64()).max(),"prefix_evaluations":prefixes,"candidate_token_lengths":lengths}));
    }
    let value = Value::Object(summaries);
    write_json(&out.join("summary.json"), &value)?;
    Ok(value)
}

fn run_models(root: &Path, evaluator: &Path, plan: &Path) -> Result<()> {
    let prepared = root.join("prepared");
    let manifest = read_json(&prepared.join("manifest.json"))?;
    for (name, hash) in manifest["prepared_sha256"]
        .as_object()
        .context("prepared hashes")?
    {
        ensure!(
            sha256(&prepared.join(name))? == hash.as_str().context("invalid hash")?,
            "prepared file changed"
        );
    }
    for model in read_json(plan)?.as_array().context("plan must be list")? {
        let id = model["id"].as_str().context("model ID")?;
        let path = PathBuf::from(model["path"].as_str().context("model path")?);
        ensure!(
            sha256(&path)? == model["sha256"].as_str().context("model hash")?,
            "model changed"
        );
        let out = root.join("runs").join(id);
        fs::create_dir(&out)?;
        let args = [
            "--model".to_owned(),
            path.display().to_string(),
            "--input".to_owned(),
            prepared.join("requests.jsonl").display().to_string(),
            "--output".to_owned(),
            out.join("predictions.jsonl").display().to_string(),
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
            &out.join("manifest.json"),
            &json!({"model":model,"evaluator_sha256":sha256(evaluator)?,"requests_sha256":sha256(&prepared.join("requests.jsonl"))?,
            "prepared_manifest_sha256":sha256(&prepared.join("manifest.json"))?,"command":std::iter::once(evaluator.display().to_string()).chain(args.iter().cloned()).collect::<Vec<_>>()}),
        )?;
        let start = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
        let log = File::create(out.join("inference.log"))?;
        let mut child = Command::new(evaluator)
            .args(&args)
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()?;
        let timer = Instant::now();
        loop {
            if let Some(status) = child.try_wait()? {
                ensure!(status.success(), "evaluator failed");
                break;
            }
            if timer.elapsed() > Duration::from_secs(7200) {
                child.kill()?;
                child.wait()?;
                bail!("evaluator timed out")
            }
            thread::sleep(Duration::from_millis(100));
        }
        score(root, &out)?;
        write_json(
            &out.join("complete.json"),
            &json!({"elapsed_s":SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64()-start,"examples":400}),
        )?;
    }
    Ok(())
}

pub fn run(args: Args) -> Result<()> {
    let study = args.study.as_deref().context("--study required")?;
    match args.action {
        Action::Prepare => prepare(study, args.base.as_deref().context("--base required")?),
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
