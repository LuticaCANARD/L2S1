use crate::common::{read_json, sha256, write_json};
use anyhow::{Context, Result, ensure};
use clap::{Args as ClapArgs, ValueEnum};
use serde_json::{Value, json};
use std::{
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
#[derive(Clone, ValueEnum)]
enum Mode {
    Download,
    Run,
}
#[derive(ClapArgs)]
pub struct Args {
    #[arg(value_enum)]
    mode: Mode,
    #[arg(long)]
    plan: PathBuf,
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    upstream: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn string<'a>(row: &'a Value, key: &str) -> Result<&'a str> {
    row[key].as_str().with_context(|| format!("missing {key}"))
}
fn sidecar(path: &Path, extension: &str) -> PathBuf {
    path.with_extension(extension)
}
fn percent_path(path: &str) -> String {
    let mut out = String::new();
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
            out.push(byte as char)
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
    }
    out
}
fn download(row: Value) -> Result<()> {
    let Some(repo) = row.get("repo").and_then(Value::as_str) else {
        return Ok(());
    };
    let path = PathBuf::from(string(&row, "path")?);
    let ready = sidecar(&path, "ready.json");
    let failed = sidecar(&path, "failed.json");
    if ready.exists() {
        return Ok(());
    }
    fs::create_dir_all(path.parent().context("model parent")?)?;
    let part = sidecar(&path, "part");
    let url = format!(
        "https://huggingface.co/{repo}/resolve/{}/{}?download=true",
        string(&row, "revision")?,
        percent_path(string(&row, "file")?)
    );
    println!("DOWNLOAD {}", path.display());
    let outcome = (|| -> Result<()> {
        let log = File::create(sidecar(&path, "download.log"))?;
        let status = Command::new("curl")
            .args([
                "-fL",
                "--http1.1",
                "--retry",
                "5",
                "--retry-all-errors",
                "--retry-delay",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "14400",
                "-C",
                "-",
                "-o",
            ])
            .arg(&part)
            .arg(&url)
            .stderr(Stdio::from(log))
            .status()?;
        ensure!(status.success(), "curl failed with {status}");
        ensure!(
            fs::metadata(&part)?.len() == row["size"].as_u64().context("pinned size")?
                && sha256(&part)? == string(&row, "sha256")?,
            "Downloaded checkpoint size or SHA256 differs from pinned LFS metadata"
        );
        fs::rename(&part, &path)?;
        write_json(&ready, &row)?;
        Ok(())
    })();
    match outcome {
        Ok(()) => println!("READY {}", path.display()),
        Err(error) => {
            let mut failure = row.clone();
            failure["error"] = json!(error.to_string());
            write_json(&failed, &failure)?;
            println!("DOWNLOAD_FAILED {} {error}", path.display());
        }
    }
    Ok(())
}
fn matrix_run(args: &Args, rows: &[Value]) -> Result<()> {
    let lock = File::create(args.output.join("matrix.lock"))?;
    lock.try_lock()
        .context("another matrix run holds matrix.lock")?;
    let status_path = args.output.join("matrix-status.json");
    let mut results = if status_path.exists() {
        read_json(&status_path)?
            .as_array()
            .context("status list")?
            .clone()
    } else {
        Vec::new()
    };
    let done = results
        .iter()
        .filter_map(|r| r["id"].as_str())
        .collect::<std::collections::HashSet<_>>();
    let mut pending = rows
        .iter()
        .filter(|r| !done.contains(r["id"].as_str().unwrap_or("")))
        .cloned()
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(21600);
    while !pending.is_empty() {
        let mut progress = false;
        let mut index = 0;
        while index < pending.len() {
            let row = &pending[index];
            let model = PathBuf::from(string(row, "path")?);
            if row.get("repo").is_some() && !sidecar(&model, "ready.json").exists() {
                if sidecar(&model, "failed.json").exists() {
                    let mut record = row.clone();
                    record["status"] = json!("download_failed");
                    results.push(record);
                    pending.remove(index);
                    write_json(&status_path, &json!(results))?;
                } else {
                    index += 1;
                }
                continue;
            }
            if !model.exists() {
                index += 1;
                continue;
            }
            let id = string(row, "id")?;
            let out = args.output.join(id);
            let command = vec![
                "jevbench-public".to_owned(),
                "run".to_owned(),
                "--jevbench".to_owned(),
                args.upstream.to_string_lossy().into_owned(),
                "--evaluator".to_owned(),
                args.source
                    .join("target/release/examples/evaluate_jsonl")
                    .to_string_lossy()
                    .into_owned(),
                "--model".to_owned(),
                model.to_string_lossy().into_owned(),
                "--output".to_owned(),
                out.to_string_lossy().into_owned(),
                "--context".to_owned(),
                "8192".to_owned(),
            ];
            let executable = std::env::current_exe()?;
            let mut recorded = vec![executable.to_string_lossy().into_owned()];
            recorded.extend(command.clone());
            println!("RUN {id}");
            let started = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
            let clock = Instant::now();
            let log = File::create(args.output.join(format!("{id}.log")))?;
            let mut process = Command::new(executable);
            process
                .args(&command)
                .stdout(Stdio::from(log.try_clone()?))
                .stderr(Stdio::from(log));
            if let Some(environment) = row.get("environment").and_then(Value::as_object) {
                for (name, value) in environment {
                    process.env(name, value.as_str().context("environment string")?);
                }
            }
            let status = process.status()?;
            if out.join("manifest.json").exists() {
                let mut manifest = read_json(&out.join("manifest.json"))?;
                manifest["runtime_environment_overrides"] =
                    row.get("environment").cloned().unwrap_or(json!({}));
                write_json(&out.join("manifest.json"), &manifest)?;
            }
            let mut record = row.clone();
            record["status"] = json!(if status.success() {
                "complete"
            } else {
                "failed"
            });
            record["exit_code"] = json!(status.code());
            record["elapsed_s"] = json!(clock.elapsed().as_secs_f64());
            record["started_unix"] = json!(started);
            record["command"] = json!(recorded);
            if out.join("summary.json").exists() {
                let summary = read_json(&out.join("summary.json"))?;
                for (key, value) in [
                    ("n_correct", &summary["n_correct"]),
                    ("n_scorable", &summary["n_scorable"]),
                    ("errors", &summary["selective_policy"]["error"]),
                ] {
                    record[key] = value.clone();
                }
            }
            println!("RESULT {}", serde_json::to_string(&record)?);
            results.push(record);
            write_json(&status_path, &json!(results))?;
            pending.remove(index);
            progress = true;
        }
        if !progress && !pending.is_empty() {
            ensure!(
                Instant::now() < deadline,
                "Model downloads did not finish within six hours"
            );
            thread::sleep(Duration::from_secs(5));
        }
    }
    Ok(())
}
pub fn run(args: Args) -> Result<()> {
    let plan = read_json(&args.plan)?;
    let rows = plan.as_array().context("model plan list")?;
    fs::create_dir_all(&args.output)?;
    match args.mode {
        Mode::Download => {
            for batch in rows.chunks(3) {
                let handles = batch
                    .iter()
                    .cloned()
                    .map(|row| thread::spawn(move || download(row)))
                    .collect::<Vec<_>>();
                for handle in handles {
                    handle
                        .join()
                        .map_err(|_| anyhow::anyhow!("download worker panicked"))??;
                }
            }
            Ok(())
        }
        Mode::Run => matrix_run(&args, rows),
    }
}
