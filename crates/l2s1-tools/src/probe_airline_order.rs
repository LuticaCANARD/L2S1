use crate::{
    common::{read_json, read_jsonl, sha256, write_json, write_jsonl},
    kaggle_airline::{LABELS, MODELS},
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Map, Value, json};
use std::{
    collections::HashSet,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    folder: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    binary: Option<PathBuf>,
    #[arg(long)]
    runtime: Option<PathBuf>,
}
pub fn run(root: &Path, args: Args) -> Result<()> {
    let folder = args
        .folder
        .unwrap_or_else(|| root.join("results/kaggle-airline-20260922"));
    let output = args
        .output
        .unwrap_or_else(|| folder.join("option-order-probe"));
    let selection = read_json(&folder.join("selection.json"))?;
    let cases = read_jsonl(&folder.join("validation.jsonl"))?;
    let mut chosen = Vec::new();
    for label in LABELS {
        chosen.extend(
            cases
                .iter()
                .filter(|c| selection["labels"][c["id"].as_str().unwrap()] == label)
                .take(4)
                .cloned(),
        );
    }
    ensure!(
        chosen.len() == 12,
        "expected four frozen validation cases per class"
    );
    let mut requests = Vec::new();
    let mut labels = Map::new();
    for rotation in 0..3 {
        for case in &chosen {
            let mut row = case.clone();
            let id = format!(
                "{}-rotation-{rotation}",
                case["id"].as_str().context("case ID")?
            );
            row["id"] = json!(id);
            let options = row["request"]["decisions"][0]["kind"]["options"]
                .as_array_mut()
                .context("choice options")?;
            ensure!(options.len() == 3, "expected three options");
            options.rotate_left(rotation);
            labels.insert(
                id,
                selection["labels"][case["id"].as_str().unwrap()].clone(),
            );
            requests.push(row);
        }
    }
    fs::create_dir(&output)?;
    let input = output.join("requests.jsonl");
    write_jsonl(&input, &requests)?;
    let metadata = json!({"labels":labels,"selected_ids":chosen.iter().map(|r|r["id"].clone()).collect::<Vec<_>>(),
        "request_sha256":sha256(&input)?,
        "scope":"Post-hoc diagnostic: first four frozen validation cases per class, all three cyclic label orders; no calibration or model selection."});
    write_json(&output.join("selection.json"), &metadata)?;
    let binary = args
        .binary
        .unwrap_or_else(|| root.join("results/tuning-20260922/bin/evaluate_jsonl-optimized"))
        .canonicalize()?;
    let runtime = args.runtime.unwrap_or_else(|| {
        PathBuf::from("/home/lutica/personal/Openweight-Test/llama.cpp/build-cuda/bin")
    });
    let mut summary = Map::new();
    for model_name in ["gemma3", "qwen3-0.6b"] {
        let filename = MODELS
            .iter()
            .find(|(name, _)| *name == model_name)
            .unwrap()
            .1;
        let model = root.join("models").join(filename);
        let predictions = output.join(format!("{model_name}.jsonl"));
        let log = File::create(output.join(format!("{model_name}.log")))?;
        let mut command = Command::new(&binary);
        command
            .args([
                "--model",
                model.to_str().context("model path")?,
                "--input",
                input.to_str().context("input path")?,
                "--output",
                predictions.to_str().context("output path")?,
                "--cuda",
                "--warmup",
            ])
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .env("LD_LIBRARY_PATH", &runtime);
        for key in [
            "GGML_CUDA_CUBLAS_COMPUTE_TYPE",
            "GGML_CUDA_DISABLE_GRAPHS",
            "GGML_CUDA_DISABLE_FUSION",
            "GGML_CUDA_GRAPH_OPT",
        ] {
            command.env_remove(key);
        }
        let mut process = command.spawn()?;
        let started = Instant::now();
        loop {
            if let Some(status) = process.try_wait()? {
                ensure!(status.success(), "{model_name} evaluator failed");
                break;
            }
            if started.elapsed() > Duration::from_secs(120) {
                process.kill()?;
                process.wait()?;
                ensure!(false, "{model_name} evaluator timed out");
            }
            std::thread::sleep(Duration::from_millis(100));
        }
        let rows = read_jsonl(&predictions)?;
        let ids = rows
            .iter()
            .filter_map(|r| r["id"].as_str())
            .collect::<HashSet<_>>();
        ensure!(
            rows.len() == 36 && ids.len() == 36 && ids.iter().all(|id| labels.contains_key(*id)),
            "prediction ID coverage mismatch"
        );
        let mut parts = Vec::new();
        for (rotation, first_label) in LABELS.iter().enumerate() {
            let suffix = format!("rotation-{rotation}");
            let subset = rows
                .iter()
                .filter(|r| r["id"].as_str().unwrap().ends_with(&suffix))
                .collect::<Vec<_>>();
            ensure!(subset.len() == 12, "rotation coverage mismatch");
            let mut first_code = 0;
            let mut correct = 0;
            for row in subset {
                let result = &row["response"]["results"][0];
                ensure!(result["truncated"] == false, "truncated result");
                let scores = result["scores"].as_array().context("scores")?;
                let top = scores
                    .iter()
                    .max_by(|a, b| {
                        a["option_probability"]
                            .as_f64()
                            .unwrap()
                            .total_cmp(&b["option_probability"].as_f64().unwrap())
                    })
                    .context("empty scores")?;
                let probability = top["option_probability"].as_f64().unwrap();
                let unique = scores
                    .iter()
                    .filter(|score| {
                        (score["option_probability"].as_f64().unwrap() - probability).abs() < 1e-12
                    })
                    .count()
                    == 1;
                first_code += usize::from(unique && top["code"] == "A");
                correct += usize::from(unique && top["id"] == labels[row["id"].as_str().unwrap()]);
            }
            parts.push(
                json!({"rotation":rotation,"first_label":first_label,"cases":12,
                "first_code_predictions":first_code,"correct":correct}),
            );
        }
        summary.insert(model_name.to_owned(), json!(parts));
    }
    let summary = Value::Object(summary);
    write_json(&output.join("summary.json"), &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
