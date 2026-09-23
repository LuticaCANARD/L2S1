use crate::common::{as_str, git_revision, percentage, read_json, sha256, write_json};
use anyhow::{Context, Result, anyhow, bail};
use chrono::Utc;
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::HashSet,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long)]
    model: Vec<String>,
    #[arg(long, num_args = 1.., value_parser = ["cpu", "cuda"], default_value = "cpu")]
    device: Vec<String>,
    #[arg(long, default_value = "fresh", value_parser = ["fresh", "prefix-reuse"])]
    execution_mode: String,
    #[arg(long, default_value = "legacy", value_parser = ["legacy", "state-first"])]
    prompt_layout: String,
    #[arg(long, default_value_t = 3)]
    iterations: u32,
    #[arg(long, default_value_t = 1)]
    warmup: u32,
    #[arg(long, default_value_t = 2048)]
    context: u32,
    #[arg(long, default_value_t = 256)]
    batch: u32,
    #[arg(long, default_value_t = 4)]
    threads: u32,
    #[arg(long, default_value_t = 0.8)]
    min_top_probability: f64,
    #[arg(long, default_value_t = 0.05)]
    min_candidate_mass: f64,
    #[arg(long, default_value_t = 1800)]
    timeout: u64,
    #[arg(long)]
    output: Option<PathBuf>,
}

fn read_models(root: &Path, manifest: &Path, selected: &[String]) -> Result<Vec<Value>> {
    let source = read_json(manifest)?;
    let models = source["models"]
        .as_array()
        .context("manifest.models must be a list")?;
    if models.is_empty() {
        bail!("manifest.models must be a nonempty list");
    }
    let mut ids = HashSet::new();
    let mut resolved = Vec::new();
    for item in models {
        let id = as_str(item, "id")?;
        if id.is_empty()
            || !id.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_' || byte == b'-'
            })
            || !id.as_bytes()[0].is_ascii_lowercase() && !id.as_bytes()[0].is_ascii_digit()
            || !ids.insert(id.to_owned())
        {
            bail!("model IDs must be unique lowercase names without path separators");
        }
        let path = root.join(as_str(item, "path")?);
        let path = path.canonicalize().unwrap_or(path);
        let mut copy = item.clone();
        copy["path"] = json!(path);
        resolved.push(copy);
    }
    for name in selected {
        if !ids.contains(name) {
            bail!("unknown model ID: {name}");
        }
    }
    if selected.is_empty() {
        Ok(resolved)
    } else {
        Ok(resolved
            .into_iter()
            .filter(|item| selected.iter().any(|name| item["id"] == *name))
            .collect())
    }
}

fn build(root: &Path, output: &Path, cuda: bool) -> Result<PathBuf> {
    let feature = if cuda { "llama-cuda" } else { "llama" };
    let result = Command::new("cargo")
        .args([
            "test",
            "--release",
            "--locked",
            "--offline",
            "--features",
            feature,
            "--test",
            "benchmark",
            "--no-run",
            "--message-format=json",
        ])
        .current_dir(root)
        .output()
        .context("launch Cargo benchmark build")?;
    let mut log = File::create(output.join("build.log"))?;
    log.write_all(&result.stderr)?;
    log.write_all(&result.stdout)?;
    if !result.status.success() {
        bail!(
            "release build failed; see {}",
            output.join("build.log").display()
        );
    }
    for line in result.stdout.split(|byte| *byte == b'\n') {
        let Ok(message) = serde_json::from_slice::<Value>(line) else {
            continue;
        };
        if message["reason"] == "compiler-artifact"
            && message["target"]["name"] == "benchmark"
            && let Some(executable) = message["executable"].as_str()
        {
            return Ok(PathBuf::from(executable));
        }
    }
    bail!("Cargo did not return a benchmark test executable")
}

fn save_summary(output: &Path, summary: &Value) -> Result<()> {
    write_json(&output.join("summary.json"), summary)?;
    let mut lines = vec![
        "# Local decision benchmark".to_owned(),
        String::new(),
        format!("Suite: `{}`. Sequential runs; load and warmups excluded from request timings.", summary["suite"].as_str().unwrap_or("unknown")),
        String::new(),
        "| Model | Device | Status | Coverage | Accepted accuracy | Correct / all | Raw top-1 | p50 ms/request | p95 ms/request | Decisions/s | Correct accepted/s |".to_owned(),
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |".to_owned(),
    ];
    for run in summary["runs"].as_array().context("summary.runs")? {
        let model = run["model"].as_str().unwrap_or("?");
        let device = run["device"].as_str().unwrap_or("?");
        if run["status"] != "ok" {
            lines.push(format!(
                "| {model} | {device} | {} | — | — | — | — | — | — | — | — |",
                run["status"].as_str().unwrap_or("failed")
            ));
            continue;
        }
        let quality = &run["quality"];
        let latency = &run["latency_ms"];
        let percent = |key: &str| percentage(quality[key].as_f64(), 1);
        lines.push(format!(
            "| {model} | {device} | ok | {} | {} | {} | {} | {:.1} | {:.1} | {:.2} | {:.2} |",
            percent("coverage"),
            percent("accepted_accuracy"),
            percent("correct_fraction"),
            percent("top1_accuracy_before_abstention"),
            latency["p50"].as_f64().unwrap_or(0.0),
            latency["p95"].as_f64().unwrap_or(0.0),
            run["decisions_per_second"].as_f64().unwrap_or(0.0),
            run["accepted_correct_decisions_per_second"]
                .as_f64()
                .unwrap_or(0.0),
        ));
    }
    lines.extend([
        String::new(),
        "Each request contains three decisions. Accuracy denominators include repeated passes; repeats are not independent examples. Accepted accuracy is n/a when every answer abstains. Raw top-1 ignores policy abstention, with ties counted incorrect. Decisions/s includes abstentions. These synthetic rule tasks do not establish general model quality. Status 'ok' means execution completed, not that every answer was correct or the native consistency suite passed.".to_owned(),
        String::new(),
        "Per-run JSON includes all labels, predictions, scores, group metrics, timings, and repeat-consistency counts. Failures and timeouts remain in summary.json and per-run logs. No model weights are copied.".to_owned(),
        String::new(),
    ]);
    fs::write(output.join("summary.md"), lines.join("\n"))?;
    Ok(())
}

fn run_test(
    root: &Path,
    executable: &Path,
    log_path: &Path,
    report: &Path,
    model: &Path,
    device: &str,
    args: &Args,
) -> Result<()> {
    let log = File::create(log_path)?;
    let mut child = Command::new(executable)
        .args([
            "native::model_decision_benchmark",
            "--exact",
            "--ignored",
            "--nocapture",
        ])
        .current_dir(root)
        .env("SKID_MODEL", model)
        .env("SKID_CUDA", if device == "cuda" { "1" } else { "0" })
        .env("SKID_BENCH_OUTPUT", report)
        .env("SKID_BENCH_ITERATIONS", args.iterations.to_string())
        .env("SKID_BENCH_WARMUP", args.warmup.to_string())
        .env("SKID_CONTEXT", args.context.to_string())
        .env("SKID_BATCH", args.batch.to_string())
        .env("SKID_THREADS", args.threads.to_string())
        .env(
            "SKID_MIN_TOP_PROBABILITY",
            args.min_top_probability.to_string(),
        )
        .env(
            "SKID_MIN_CANDIDATE_MASS",
            args.min_candidate_mass.to_string(),
        )
        .env("SKID_EXECUTION_MODE", &args.execution_mode)
        .env("SKID_PROMPT_LAYOUT", &args.prompt_layout)
        .stdout(Stdio::from(log.try_clone()?))
        .stderr(Stdio::from(log))
        .spawn()
        .context("launch benchmark executable")?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            if !status.success() {
                bail!(
                    "test exited with status {status}; see {}",
                    log_path.display()
                );
            }
            return Ok(());
        }
        if start.elapsed() > Duration::from_secs(args.timeout) {
            child.kill()?;
            child.wait()?;
            bail!("timeout: exceeded {} seconds", args.timeout);
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub fn run(root: &Path, args: Args) -> Result<()> {
    if args.iterations == 0
        || args.context == 0
        || args.batch == 0
        || args.threads == 0
        || args.timeout == 0
    {
        bail!("iterations, context, batch, threads, and timeout must be positive");
    }
    for probability in [args.min_top_probability, args.min_candidate_mass] {
        if !(0.0..=1.0).contains(&probability) {
            bail!("probabilities must be in [0, 1]");
        }
    }
    let manifest = args
        .manifest
        .clone()
        .unwrap_or_else(|| root.join("tests/fixtures/benchmark_models.json"));
    let models = read_models(root, &manifest, &args.model)?;
    let timestamp = Utc::now().format("%Y%m%dT%H%M%S%.6fZ").to_string();
    let output = args
        .output
        .clone()
        .unwrap_or_else(|| root.join("results/benchmark").join(&timestamp));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(&output)
        .with_context(|| format!("output already exists: {}", output.display()))?;
    println!("Reports: {}", output.display());
    let llama_revision = std::env::var("L2S1_LLAMA_CPP_SOURCE")
        .or_else(|_| std::env::var("LLAMA_CPP_DIR"))
        .ok()
        .and_then(|path| git_revision(Path::new(&path)))
        .or_else(|| {
            fs::read_to_string(root.join("crates/l2s1-llama-sys/UPSTREAM_COMMIT"))
                .ok()
                .map(|text| text.trim().to_owned())
        });
    let mut summary = json!({
        "schema_version": 1, "created_at": timestamp, "suite": "decision-rules-v1",
        "suite_sha256": sha256(&root.join("tests/fixtures/decision_benchmark.json"))?,
        "host": {"platform": format!("{} {}", std::env::consts::OS, std::env::consts::ARCH), "cpu": cpu_model()},
        "repository_revision": git_revision(root), "llama_cpp_revision": llama_revision,
        "settings": {"model": args.model, "device": args.device, "execution_mode": args.execution_mode,
            "prompt_layout": args.prompt_layout, "iterations": args.iterations, "warmup": args.warmup,
            "context": args.context, "batch": args.batch, "threads": args.threads,
            "min_top_probability": args.min_top_probability, "min_candidate_mass": args.min_candidate_mass,
            "timeout": args.timeout}, "runs": []
    });
    save_summary(&output, &summary)?;
    let executable = match build(
        root,
        &output,
        args.device.iter().any(|device| device == "cuda"),
    ) {
        Ok(path) => path,
        Err(error) => {
            summary["build_error"] = json!(error.to_string());
            save_summary(&output, &summary)?;
            return Err(error);
        }
    };
    summary["benchmark_executable_sha256"] = json!(sha256(&executable)?);
    let mut failed = false;
    let mut seen_devices = HashSet::new();
    let devices = args
        .device
        .iter()
        .filter(|device| seen_devices.insert(device.as_str()))
        .collect::<Vec<_>>();
    for model in models {
        let name = as_str(&model, "id")?;
        let path = PathBuf::from(as_str(&model, "path")?);
        let mut preflight = None;
        let model_hash = match File::open(&path).and_then(|mut file| {
            let mut header = [0; 4];
            file.read_exact(&mut header)?;
            Ok(header)
        }) {
            Ok(header) if &header == b"GGUF" => match sha256(&path) {
                Ok(hash) => Some(hash),
                Err(error) => {
                    preflight = Some(error.to_string());
                    None
                }
            },
            Ok(_) => {
                preflight = Some("model does not have a GGUF header".to_owned());
                None
            }
            Err(error) => {
                preflight = Some(error.to_string());
                None
            }
        };
        for device in &devices {
            let stem = format!("{name}.{device}");
            let report = output.join(format!("{stem}.json"));
            let log = output.join(format!("{stem}.log"));
            let mut entry = json!({"model": name, "device": device, "sha256": model_hash,
                "model_file": path.file_name().map(|name| name.to_string_lossy().to_string()),
                "status": "failed", "log": log.file_name().unwrap().to_string_lossy()});
            if let Some(error) = &preflight {
                fs::write(&log, format!("{error}\n"))?;
                entry["error"] = json!(error);
            } else {
                match run_test(root, &executable, &log, &report, &path, device, &args)
                    .and_then(|_| read_json(&report))
                {
                    Ok(data)
                        if data["suite"] == summary["suite"]
                            && data["iterations"] == args.iterations =>
                    {
                        for key in [
                            "quality",
                            "latency_ms",
                            "decisions_per_second",
                            "accepted_correct_decisions_per_second",
                            "repeat_consistency",
                        ] {
                            entry[key] = data[key].clone();
                        }
                        entry["status"] = json!("ok");
                        entry["report"] = json!(report.file_name().unwrap().to_string_lossy());
                    }
                    Ok(_) => {
                        entry["error"] = json!("benchmark report does not match requested run")
                    }
                    Err(error) => {
                        if error.to_string().starts_with("timeout:") {
                            entry["status"] = json!("timeout");
                        }
                        entry["error"] = json!(error.to_string());
                    }
                }
            }
            failed |= entry["status"] != "ok";
            summary["runs"].as_array_mut().unwrap().push(entry);
            save_summary(&output, &summary)?;
        }
    }
    println!("Comparison: {}", output.join("summary.md").display());
    if failed {
        Err(anyhow!("one or more benchmark runs failed"))
    } else {
        Ok(())
    }
}

fn cpu_model() -> Option<String> {
    fs::read_to_string("/proc/cpuinfo").ok().and_then(|data| {
        data.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            (key.trim() == "model name").then(|| value.trim().to_owned())
        })
    })
}
