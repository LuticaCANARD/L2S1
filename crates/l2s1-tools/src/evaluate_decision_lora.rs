use crate::common::{digest_bytes, read_jsonl, sha256, write_json};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::HashMap,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};
#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    adapter: PathBuf,
    #[arg(long)]
    model: Option<PathBuf>,
    #[arg(long)]
    binary: Option<PathBuf>,
    #[arg(long)]
    runtime: Option<PathBuf>,
}
fn median(values: &[f64]) -> Result<f64> {
    ensure!(!values.is_empty(), "empty timing run");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    Ok(if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.
    })
}
#[allow(clippy::too_many_arguments)]
fn run_one(
    name: &str,
    request: &Path,
    adapter: bool,
    output: &Path,
    model: &Path,
    binary: &Path,
    lora: &Path,
    runtime: &Path,
    provenance: &mut Value,
) -> Result<Vec<Value>> {
    let path = output.join(format!("{name}.jsonl"));
    ensure!(!path.exists(), "output exists: {}", path.display());
    let mut args = vec![
        "--model".to_owned(),
        model.to_string_lossy().into_owned(),
        "--input".to_owned(),
        request.canonicalize()?.to_string_lossy().into_owned(),
        "--output".to_owned(),
        path.to_string_lossy().into_owned(),
        "--cuda".to_owned(),
        "--context".to_owned(),
        "2048".to_owned(),
        "--batch".to_owned(),
        "256".to_owned(),
        "--ubatch".to_owned(),
        "256".to_owned(),
        "--threads".to_owned(),
        "4".to_owned(),
        "--flash-attention".to_owned(),
        "off".to_owned(),
        "--execution-mode".to_owned(),
        "fresh".to_owned(),
        "--request-batch-size".to_owned(),
        "1".to_owned(),
        "--warmup".to_owned(),
    ];
    if adapter {
        args.extend(["--lora".to_owned(), lora.to_string_lossy().into_owned()]);
    }
    println!("START {name}");
    let log = File::create_new(output.join(format!("{name}.log")))?;
    let mut command = Command::new(binary);
    command
        .args(&args)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log));
    command.env("LD_LIBRARY_PATH", runtime);
    for key in [
        "GGML_CUDA_CUBLAS_COMPUTE_TYPE",
        "GGML_CUDA_DISABLE_GRAPHS",
        "GGML_CUDA_DISABLE_FUSION",
        "GGML_CUDA_GRAPH_OPT",
    ] {
        command.env_remove(key);
    }
    let status = command.status()?;
    let mut recorded = vec![binary.to_string_lossy().into_owned()];
    recorded.extend(args);
    provenance["runs"].as_array_mut().unwrap().push(json!({"name":name,"command":recorded,"exit_code":status.code(),"request_sha256":sha256(request)?}));
    write_json(&output.join("adapter-provenance.json"), provenance)?;
    ensure!(status.success(), "evaluation failed: {name}");
    let rows = read_jsonl(&path)?;
    ensure!(
        rows.iter()
            .all(|r| r["response"]["results"][0]["truncated"] == false
                && r["response"]["backend"]["lora_path"]
                    .as_str()
                    .is_some_and(|s| !s.is_empty())
                    == adapter),
        "truncated result or adapter mismatch: {name}"
    );
    Ok(rows)
}
pub fn run(root: &Path, args: Args) -> Result<()> {
    fs::create_dir_all(&args.output)?;
    let model = args
        .model
        .unwrap_or_else(|| root.join("models/gemma-4-E2B-it-Q8_0.gguf"))
        .canonicalize()?;
    let binary = args
        .binary
        .unwrap_or_else(|| root.join("target/release/examples/evaluate_jsonl"))
        .canonicalize()?;
    let runtime = args
        .runtime
        .unwrap_or_else(|| {
            PathBuf::from("/home/lutica/personal/Openweight-Test/llama.cpp/build-cuda/bin")
        })
        .canonicalize()?;
    let adapter = args.adapter.canonicalize()?;
    let mut libraries = serde_json::Map::new();
    for entry in fs::read_dir(&runtime)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "so") {
            libraries.insert(
                path.file_name().unwrap().to_string_lossy().into_owned(),
                json!(sha256(&path)?),
            );
        }
    }
    let mut provenance = json!({"model_sha256":sha256(&model)?,"adapter_sha256":sha256(&adapter)?,
        "binary_sha256":sha256(&binary)?,"script_sha256":digest_bytes(include_bytes!("evaluate_decision_lora.rs")),
        "runtime":libraries,"runs":[]});
    for split in ["calibration", "test", "probe"] {
        run_one(
            &format!("lora-{split}"),
            &args.data.join(format!("{split}.jsonl")),
            true,
            &args.output,
            &model,
            &binary,
            &adapter,
            &runtime,
            &mut provenance,
        )?;
    }
    run_one(
        "lora-ag-news",
        &root.join("results/kaggle-ag-news/requests.jsonl"),
        true,
        &args.output,
        &model,
        &binary,
        &adapter,
        &runtime,
        &mut provenance,
    )?;
    let mut timings = json!({"base":[],"lora":[]});
    let mut reference = HashMap::<String, Value>::new();
    for rep in 1..=3 {
        let order = if rep % 2 == 1 {
            ["base", "lora"]
        } else {
            ["lora", "base"]
        };
        for kind in order {
            let rows = run_one(
                &format!("timing-{kind}-{rep}"),
                &args.data.join("test.jsonl"),
                kind == "lora",
                &args.output,
                &model,
                &binary,
                &adapter,
                &runtime,
                &mut provenance,
            )?;
            let mut scores = serde_json::Map::new();
            let mut elapsed = Vec::new();
            for row in rows {
                let id = row["id"].as_str().context("row ID")?;
                scores.insert(
                    id.to_owned(),
                    json!(
                        row["response"]["results"][0]["scores"]
                            .as_array()
                            .context("scores")?
                            .iter()
                            .map(|s| s["raw_logit"].clone())
                            .collect::<Vec<_>>()
                    ),
                );
                elapsed.push(row["batch_elapsed_ms"].as_f64().context("batch elapsed")?);
            }
            let scores = Value::Object(scores);
            if let Some(previous) = reference.insert(kind.to_owned(), scores.clone()) {
                ensure!(
                    previous == scores,
                    "Scores drifted between repeated measurements"
                );
            }
            timings[kind]
                .as_array_mut()
                .unwrap()
                .push(json!({"total_ms":elapsed.iter().sum::<f64>(),"p50_ms":median(&elapsed)?}));
        }
    }
    write_json(&args.output.join("timings.json"), &timings)?;
    println!("COMPLETE");
    Ok(())
}
