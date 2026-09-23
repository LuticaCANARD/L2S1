#[cfg(not(feature = "llama"))]
fn main() {
    panic!("requires llama");
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        DecisionBackend, DecisionPolicy, DecisionRequest, ExecutionMode, PreparationCacheConfig,
        llama::LlamaBackend,
    };
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashSet,
        fs::{self, OpenOptions},
        io::{BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };
    #[derive(Clone, Copy, Debug, clap::ValueEnum, serde::Serialize)]
    enum Mode {
        FreshUncached,
        FreshCached,
        SessionCached,
    }
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_enum)]
        mode: Mode,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 3)]
        repeats: usize,
        #[arg(long, default_value_t = 8192)]
        context: u32,
        #[arg(long, default_value_t = 256)]
        batch: u32,
        #[arg(long, default_value_t = 4)]
        threads: i32,
        #[arg(long)]
        gpu_layers: Option<u32>,
        #[arg(long, default_value_t = 0)]
        cpu_moe_layers: u32,
        /// Read avoids whole-file mmap during model loading; auto preserves defaults.
        #[arg(long, value_enum, default_value_t = l2s1::ModelLoadMode::Auto)]
        model_load_mode: l2s1::ModelLoadMode,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        id: String,
        request: DecisionRequest,
    }
    let a = Args::parse();
    if a.repeats == 0 {
        return Err("repeats must be positive".into());
    }
    fs::create_dir(&a.output)?;
    fs::write(a.output.join("engine.pid"), std::process::id().to_string())?;
    let bytes = fs::read(&a.input)?;
    let cases: Vec<Case> = std::str::from_utf8(&bytes)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    if cases.len() != 400 {
        return Err("expected exactly 400 cases".into());
    }
    let mut ids = HashSet::new();
    for c in &cases {
        c.request.validate()?;
        if !ids.insert(&c.id) || c.request.decisions.len() != 1 {
            return Err("invalid logical cases".into());
        }
        let n = c.request.decisions[0].options().len();
        if !(c.id.starts_with("banking77-en:") && n == 77
            || c.id.starts_with("massive-ko:") && n == 60)
        {
            return Err("full official label set required".into());
        }
    }
    let started = Instant::now();
    let mut backend = LlamaBackend::load_with_options(
        &a.model,
        l2s1::ComputeOptions {
            context: a.context,
            batch: a.batch,
            ubatch: a.batch,
            threads: a.threads,
            flash_attention: l2s1::FlashAttention::Off,
            gpu_layers: a.gpu_layers,
            cpu_moe_layers: a.cpu_moe_layers,
            model_load_mode: a.model_load_mode,
        },
        true,
        DecisionPolicy::default(),
        l2s1::PromptProfile::Auto,
    )?;
    let enabled = !matches!(a.mode, Mode::FreshUncached);
    backend.set_preparation_cache(PreparationCacheConfig {
        max_entries: if enabled { 1024 } else { 0 },
        max_bytes: if enabled { 64 * 1024 * 1024 } else { 0 },
    });
    if matches!(a.mode, Mode::SessionCached) {
        backend.set_execution_mode(ExecutionMode::PrefixReuse);
    }
    let load_ms = started.elapsed().as_secs_f64() * 1000.;
    // Warm up both dataset prompt shapes without labels. No timed warmup output.
    backend.decide(&cases[0].request)?;
    let ko = cases
        .iter()
        .find(|c| c.id.starts_with("massive-ko:"))
        .ok_or("missing Korean")?;
    backend.decide(&ko.request)?;
    backend.clear_preparation_cache();
    backend.take_timings();
    fs::write(
        a.output.join("metadata.json"),
        serde_json::to_vec_pretty(&json!({
            "mode":a.mode,"pid":std::process::id(),"model_identity":backend.identity(),"input_sha256":format!("{:x}",Sha256::digest(&bytes)),
            "model_path":a.model,"load_ms":load_ms,"unique_cases":cases.len(),"repeats":a.repeats,
            "preparation_cache":{"enabled":enabled,"max_entries":if enabled {1024} else {0},"max_bytes":if enabled {64*1024*1024} else {0}},
            "protocol":"one resident model; two global warmups; per case one first call followed immediately by repeated identical calls; no labels in inference; session per case only in session-cached mode; timer covers decide only (decision cloning and session creation excluded)",
            "cache_hit_definition":"observed preparation-cache hit delta > 0; hot repeat is not assumed to be a cache hit",
            "latency_scope":"batch size one; excludes model loading, serialization, profiling reads, and external memory sampling work"
        }))?,
    )?;
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(a.output.join("predictions.jsonl"))?,
    );
    for (i, c) in cases.iter().enumerate() {
        // A closure keeps one session alive across this case's repeats, while
        // each mode records the same fields and call-only timing boundary.
        let mut record = |repeat,
                          response: l2s1::Result<l2s1::DecisionResponse>,
                          elapsed_ms,
                          before: l2s1::llama::PreparationCacheStats,
                          after: l2s1::llama::PreparationCacheStats,
                          timings|
         -> Result<(), Box<dyn std::error::Error>> {
            let mut row = json!({"id":c.id,"repeat":repeat,"phase":if repeat==0{"first"}else{"hot_repeat"},
                "elapsed_ms":elapsed_ms,"timings":timings,"cache_before":before,"cache_after":after,
                "prompt_hits":after.prompts.hits-before.prompts.hits,"candidate_hits":after.candidates.hits-before.candidates.hits});
            match response {
                Ok(response) => row["response"] = serde_json::to_value(response)?,
                Err(error) => row["error"] = format!("{error}").into(),
            }
            serde_json::to_writer(&mut out, &row)?;
            writeln!(out)?;
            Ok(())
        };
        if matches!(a.mode, Mode::SessionCached) {
            let mut session = backend.shared_state(c.request.state.clone())?;
            for repeat in 0..=a.repeats {
                let decisions = c.request.decisions.clone();
                let before = session.preparation_cache_stats();
                let start = Instant::now();
                let response = session.decide(decisions);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                let after = session.preparation_cache_stats();
                record(
                    repeat,
                    response,
                    elapsed_ms,
                    before,
                    after,
                    session.take_timings(),
                )?;
            }
        } else {
            for repeat in 0..=a.repeats {
                let before = backend.preparation_cache_stats();
                let start = Instant::now();
                let response = backend.decide(&c.request);
                let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
                let after = backend.preparation_cache_stats();
                record(
                    repeat,
                    response,
                    elapsed_ms,
                    before,
                    after,
                    backend.take_timings(),
                )?;
            }
        }
        out.flush()?;
        if (i + 1) % 25 == 0 {
            eprintln!("Completed {}/{}", i + 1, cases.len());
        }
    }
    fs::write(
        a.output.join("complete.json"),
        serde_json::to_vec_pretty(
            &json!({"complete":true,"logical_cases":cases.len(),"measured_calls":cases.len()*(a.repeats+1),"cache_stats":backend.preparation_cache_stats()}),
        )?,
    )?;
    Ok(())
}
