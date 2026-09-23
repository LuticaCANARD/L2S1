use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    kaggle_ag_news::score,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    binary: PathBuf,
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    input: PathBuf,
    #[arg(long)]
    selection: PathBuf,
    #[arg(long)]
    matrix: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long)]
    runtime: PathBuf,
    #[arg(long)]
    reference: Option<PathBuf>,
    #[arg(long)]
    cpu: bool,
}

fn runtime_hashes(path: &Path) -> Result<BTreeMap<String, String>> {
    fs::read_dir(path)?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "so"))
        .map(|path| {
            Ok((
                path.file_name().unwrap().to_string_lossy().into_owned(),
                sha256(&path)?,
            ))
        })
        .collect()
}

fn compare(reference: &[Value], rows: &[Value]) -> Result<Value> {
    let baseline = reference
        .iter()
        .map(|r| {
            Ok((
                r["id"].as_str().context("reference ID")?.to_owned(),
                &r["response"]["results"][0],
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;
    let (mut changed, mut top_changed) = (0, 0);
    let (mut delta, mut mass_delta) = (0.0_f64, 0.0_f64);
    for row in rows {
        let id = row["id"].as_str().context("result ID")?;
        let a = baseline.get(id).context("unknown result ID")?;
        let b = &row["response"]["results"][0];
        ensure!(
            a["input_tokens"] == b["input_tokens"],
            "input token count changed"
        );
        let aa = a["scores"].as_array().context("reference scores")?;
        let bb = b["scores"].as_array().context("result scores")?;
        ensure!(
            aa.len() == bb.len()
                && aa
                    .iter()
                    .zip(bb)
                    .all(|(x, y)| x["token_id"] == y["token_id"]),
            "candidate tokens changed"
        );
        changed += usize::from(a["value"] != b["value"]);
        let top = |scores: &[Value]| {
            scores
                .iter()
                .max_by(|x, y| {
                    x["option_probability"]
                        .as_f64()
                        .unwrap_or(f64::NAN)
                        .total_cmp(&y["option_probability"].as_f64().unwrap_or(f64::NAN))
                })
                .map(|s| s["id"].clone())
        };
        top_changed += usize::from(top(aa) != top(bb));
        for (x, y) in aa.iter().zip(bb) {
            delta = delta.max(
                (x["option_probability"].as_f64().context("probability")?
                    - y["option_probability"].as_f64().context("probability")?)
                .abs(),
            );
        }
        mass_delta = mass_delta.max(
            (a["candidate_mass"].as_f64().context("mass")?
                - b["candidate_mass"].as_f64().context("mass")?)
            .abs(),
        );
    }
    Ok(
        json!({"changed_selection":changed,"changed_top1":top_changed,"max_probability_delta":delta,"max_candidate_mass_delta":mass_delta,
        "equivalent":changed==0&&top_changed==0&&delta<=0.02&&mass_delta<=0.02}),
    )
}

pub fn run(args: Args) -> Result<()> {
    fs::create_dir_all(&args.out)?;
    let manifest = args.out.join("summary.json");
    ensure!(!manifest.exists(), "use a new output directory");
    let selection = read_json(&args.selection)?;
    let cases = read_jsonl(&args.input)?;
    let mut ids = HashSet::new();
    let mut labels = serde_json::Map::new();
    for row in &cases {
        let id = row["id"].as_str().context("missing case ID")?;
        ensure!(ids.insert(id), "duplicate case ID");
        labels.insert(id.to_owned(), selection["labels"][id].clone());
    }
    let reference = match &args.reference {
        Some(path) => Some(read_jsonl(path)?),
        None => None,
    };
    let mut summary = json!({"binary_sha256":sha256(&args.binary)?,"model_sha256":sha256(&args.model)?,
        "input_sha256":sha256(&args.input)?,"selection_sha256":sha256(&args.selection)?,"runtime":runtime_hashes(&args.runtime)?,"runs":[]});
    let configs = read_json(&args.matrix)?;
    let mut binary_hashes = HashMap::new();
    binary_hashes.insert(args.binary.display().to_string(), sha256(&args.binary)?);
    let mut runtime_hash = HashMap::new();
    runtime_hash.insert(
        args.runtime.display().to_string(),
        runtime_hashes(&args.runtime)?,
    );
    for config in configs.as_array().context("matrix must be a list")? {
        let name = config["name"]
            .as_str()
            .context("missing configuration name")?;
        ensure!(
            name.bytes()
                .all(|v| v.is_ascii_alphanumeric() || b"_-".contains(&v)),
            "invalid configuration name"
        );
        let output = args.out.join(format!("{name}.jsonl"));
        ensure!(!output.exists(), "output exists");
        let binary = PathBuf::from(
            config["binary"]
                .as_str()
                .unwrap_or_else(|| args.binary.to_str().unwrap()),
        )
        .canonicalize()?;
        let runtime = PathBuf::from(
            config["runtime"]
                .as_str()
                .unwrap_or_else(|| args.runtime.to_str().unwrap()),
        )
        .canonicalize()?;
        let binary_key = binary.display().to_string();
        let runtime_key = runtime.display().to_string();
        if !binary_hashes.contains_key(&binary_key) {
            binary_hashes.insert(binary_key.clone(), sha256(&binary)?);
        }
        if !runtime_hash.contains_key(&runtime_key) {
            runtime_hash.insert(runtime_key.clone(), runtime_hashes(&runtime)?);
        }
        let mut cmd = vec![
            "--model".to_owned(),
            args.model.canonicalize()?.display().to_string(),
            "--input".to_owned(),
            args.input.canonicalize()?.display().to_string(),
            "--output".to_owned(),
            output.display().to_string(),
            "--warmup".to_owned(),
        ];
        if !args.cpu {
            cmd.push("--cuda".to_owned());
        }
        for arg in config["args"].as_array().context("missing config args")? {
            cmd.push(arg.as_str().context("invalid config arg")?.to_owned());
        }
        let mut env = std::env::vars().collect::<HashMap<_, _>>();
        env.insert("LD_LIBRARY_PATH".to_owned(), runtime_key.clone());
        for key in [
            "GGML_CUDA_CUBLAS_COMPUTE_TYPE",
            "GGML_CUDA_DISABLE_GRAPHS",
            "GGML_CUDA_DISABLE_FUSION",
            "GGML_CUDA_GRAPH_OPT",
        ] {
            env.remove(key);
        }
        if let Some(extra) = config["env"].as_object() {
            for (key, value) in extra {
                env.insert(
                    key.clone(),
                    value.as_str().context("invalid env value")?.to_owned(),
                );
            }
        }
        let log = File::create(args.out.join(format!("{name}.log")))?;
        let mut child = Command::new(&binary)
            .args(&cmd)
            .env_clear()
            .envs(&env)
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log))
            .spawn()?;
        let start = Instant::now();
        let timeout = config["timeout"].as_u64().unwrap_or(600);
        let code = loop {
            if let Some(status) = child.try_wait()? {
                break status.code().unwrap_or(1);
            }
            if start.elapsed() > Duration::from_secs(timeout) {
                child.kill()?;
                child.wait()?;
                break 124;
            }
            thread::sleep(Duration::from_millis(100));
        };
        let rows = if output.exists() {
            read_jsonl(&output)?
        } else {
            Vec::new()
        };
        let metrics = score(&labels, &rows)?;
        let mut batches = BTreeMap::new();
        for row in &rows {
            batches
                .entry(row["batch_index"].as_u64().unwrap_or(0))
                .or_insert(row);
        }
        let ms = batches
            .values()
            .map(|r| r["batch_elapsed_ms"].as_f64().unwrap_or(0.0))
            .sum::<f64>();
        let mut profile = serde_json::Map::new();
        for key in ["prepare_ms", "native_ms", "score_ms", "decisions"] {
            profile.insert(
                key.to_owned(),
                json!(
                    batches
                        .values()
                        .map(|r| r["batch_profile"][key].as_f64().unwrap_or(0.0))
                        .sum::<f64>()
                ),
            );
        }
        let mut result = json!({"config":config,"command":std::iter::once(binary_key.clone()).chain(cmd).collect::<Vec<_>>(),
            "binary_sha256":binary_hashes[&binary_key],"runtime":runtime_hash[&runtime_key],"exit_code":code,
            "wall_seconds":start.elapsed().as_secs_f64(),"metrics":metrics,"measured_ms":ms,
            "articles_per_second":(ms>0.0).then(||rows.len() as f64*1000.0/ms),"profile":profile});
        if code == 0
            && metrics["counts"]["errors"] == 0
            && metrics["counts"]["missing"] == 0
            && let Some(reference) = &reference
        {
            result["comparison"] = compare(reference, &rows)?;
        }
        summary["runs"].as_array_mut().unwrap().push(result);
        write_json(&manifest, &summary)?;
    }
    Ok(())
}
