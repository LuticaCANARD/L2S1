//! Load one model and evaluate unlabeled JSONL requests, retaining every outcome.
#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!("Enable --features llama to evaluate a local model");
    std::process::exit(1);
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        ComputeOptions, DecisionPolicy, DecisionRequest, EvidenceTransfer, ExecutionMode,
        FlashAttention, ParallelPrefixAlignment, PreparationCacheConfig, PromptDetail,
        PromptLayout, PromptProfile, llama::LlamaBackend,
    };
    use serde::Deserialize;
    use std::{
        fs::{File, OpenOptions},
        io::{BufRead, BufReader, BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };

    #[derive(Clone, Copy, Default, clap::ValueEnum, PartialEq, Eq, serde::Serialize)]
    #[serde(rename_all = "snake_case")]
    enum Resident {
        #[default]
        None,
        FixedSchema,
        SharedPrefix,
    }

    enum Runner<'a> {
        Ordinary(&'a mut LlamaBackend),
        Fixed(l2s1::llama::SharedDecisionSession<'a>),
        Shared(l2s1::llama::ParallelPrefixSession<'a>),
    }
    impl Runner<'_> {
        fn decide_batch(
            &mut self,
            requests: &[DecisionRequest],
        ) -> l2s1::Result<Vec<l2s1::DecisionResponse>> {
            match self {
                Self::Ordinary(b) => b.decide_batch(requests),
                Self::Fixed(s) => requests.iter().map(|r| s.decide(r.state.clone())).collect(),
                Self::Shared(s) => s.decide_batch(requests),
            }
        }
        fn stats(&self) -> l2s1::llama::PreparationCacheStats {
            match self {
                Self::Ordinary(b) => b.preparation_cache_stats(),
                Self::Fixed(s) => s.preparation_cache_stats(),
                Self::Shared(s) => s.preparation_cache_stats(),
            }
        }
        fn timings(&mut self) -> l2s1::llama::InferenceTimings {
            match self {
                Self::Ordinary(b) => b.take_timings(),
                Self::Fixed(s) => s.take_timings(),
                Self::Shared(s) => s.take_timings(),
            }
        }
    }

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        /// Zero thresholds force selection; positive thresholds enable abstention.
        #[arg(long, default_value_t = 0.0)]
        min_top_probability: f64,
        #[arg(long, default_value_t = 0.0)]
        min_candidate_mass: f64,
        /// Keep a scoped native KV session across measured calls. First call is cold.
        #[arg(long, value_enum, default_value_t = Resident::None)]
        resident: Resident,
        /// Evaluate explicit per-row fact specs; labels are never read by this process.
        #[arg(long)]
        derive_facts: bool,
        #[arg(long, value_enum, default_value_t = EvidenceTransfer::Full)]
        evidence_transfer: EvidenceTransfer,
        #[arg(long, default_value_t = 0)]
        preparation_cache_bytes: usize,
        #[arg(long, default_value_t = 128)]
        preparation_cache_entries: usize,
        #[arg(long)]
        lora: Option<PathBuf>,
        #[arg(long, conflicts_with = "lora")]
        output_head: Option<PathBuf>,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        cuda: bool,
        #[arg(long, conflicts_with = "cuda")]
        metal: bool,
        #[arg(long, default_value_t = 2048)]
        context: u32,
        #[arg(long, default_value_t = 256)]
        batch: u32,
        #[arg(long)]
        ubatch: Option<u32>,
        #[arg(long, default_value_t = 4)]
        threads: i32,
        #[arg(long, value_enum, default_value_t = FlashAttention::Off)]
        flash_attention: FlashAttention,
        /// Maximum layers on CUDA; omit to preserve full offload.
        #[arg(long)]
        gpu_layers: Option<u32>,
        /// Keep expert weights of the first N MoE layers in CPU RAM.
        #[arg(long, default_value_t = 0)]
        cpu_moe_layers: u32,
        /// Read avoids whole-file mmap during model loading; auto preserves defaults.
        #[arg(long, value_enum, default_value_t = l2s1::ModelLoadMode::Auto)]
        model_load_mode: l2s1::ModelLoadMode,
        #[arg(long, value_enum, default_value_t = ExecutionMode::Fresh)]
        execution_mode: ExecutionMode,
        #[arg(long, value_enum, default_value_t = PromptLayout::Legacy)]
        prompt_layout: PromptLayout,
        #[arg(long, value_enum, default_value_t = PromptDetail::Minimal)]
        prompt_detail: PromptDetail,
        #[arg(long, value_enum, default_value_t = PromptProfile::Auto)]
        prompt_profile: PromptProfile,
        #[arg(long, default_value_t = 0)]
        code_rotation: u32,
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=32))]
        parallel_width: u32,
        /// Size parallel KV memory from actual input tokens plus one batch.
        #[arg(long)]
        parallel_context_dynamic: bool,
        /// `token` shares every common prefix token; scores can change slightly.
        #[arg(long, value_enum, default_value_t = ParallelPrefixAlignment::Batch)]
        parallel_prefix_alignment: ParallelPrefixAlignment,
        /// Independent article requests per call; prompts/states are never merged.
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=32))]
        request_batch_size: u32,
        /// Exclude one untimed batch from the measured run (no output labels used).
        #[arg(long)]
        warmup: bool,
    }

    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        id: String,
        request: DecisionRequest,
        #[serde(default)]
        facts: Vec<l2s1::FactSpec>,
    }

    let args = Args::parse();
    let cases: Vec<Case> = BufReader::new(File::open(&args.input)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let mut ids = std::collections::HashSet::new();
    for case in &cases {
        if !ids.insert(&case.id) || case.id.is_empty() {
            return Err("Expected unique, nonempty case IDs".into());
        }
        case.request.validate()?;
    }
    if cases.is_empty() {
        return Err("Dataset must not be empty".into());
    }
    if args.resident == Resident::FixedSchema {
        if args.execution_mode != ExecutionMode::PrefixReuse || args.request_batch_size != 1 {
            return Err(
                "fixed-schema requires --execution-mode prefix-reuse and --request-batch-size 1"
                    .into(),
            );
        }
        if args.output_head.is_some() {
            return Err("fixed-schema does not support output heads".into());
        }
        let schema = serde_json::to_value(&cases[0].request.decisions)?;
        for case in &cases {
            if case.request.shared.is_some()
                || case.request.decisions.len() != 1
                || serde_json::to_value(&case.request.decisions)? != schema
            {
                return Err("fixed-schema requires one identical decision and no shared evidence in every row".into());
            }
        }
    }
    if args.resident == Resident::SharedPrefix && args.execution_mode != ExecutionMode::Parallel {
        return Err("shared-prefix requires --execution-mode parallel".into());
    }
    let mut fact_hashes = Vec::with_capacity(cases.len());
    for case in &cases {
        use sha2::{Digest, Sha256};
        // Fail before loading a model or creating output. Recompute inside each
        // timed call so reported latency includes request-side preprocessing.
        if args.derive_facts {
            l2s1::derive_facts(&case.request.state, &case.facts)?;
        }
        fact_hashes.push(if args.derive_facts {
            Some(format!(
                "{:x}",
                Sha256::digest(serde_json::to_vec(&case.facts)?)
            ))
        } else {
            None
        });
    }
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    let started = Instant::now();
    let compute = ComputeOptions {
        context: args.context,
        batch: args.batch,
        ubatch: args.ubatch.unwrap_or(args.batch),
        threads: args.threads,
        flash_attention: args.flash_attention,
        gpu_layers: args.gpu_layers,
        cpu_moe_layers: args.cpu_moe_layers,
        model_load_mode: args.model_load_mode,
    };
    let mut backend = if args.metal {
        LlamaBackend::load_with_metal_options(
            &args.model,
            compute,
            DecisionPolicy {
                min_top_probability: args.min_top_probability,
                min_candidate_mass: args.min_candidate_mass,
            },
            args.prompt_profile,
        )?
    } else {
        LlamaBackend::load_with_options(
            &args.model,
            compute,
            args.cuda,
            DecisionPolicy {
                min_top_probability: args.min_top_probability,
                min_candidate_mass: args.min_candidate_mass,
            },
            args.prompt_profile,
        )?
    };
    if let Some(path) = &args.lora {
        backend.load_lora(path)?;
    }
    backend.set_execution_mode(args.execution_mode);
    backend.set_prompt_layout(args.prompt_layout);
    backend.set_prompt_detail(args.prompt_detail);
    backend.set_code_rotation(args.code_rotation as usize)?;
    backend.set_evidence_transfer(args.evidence_transfer)?;
    backend.set_preparation_cache(PreparationCacheConfig {
        max_entries: args.preparation_cache_entries,
        max_bytes: args.preparation_cache_bytes,
    });
    if let Some(path) = &args.output_head {
        backend.load_output_head(path)?;
    }
    backend.set_parallel_width(args.parallel_width as usize)?;
    backend.set_parallel_context_dynamic(args.parallel_context_dynamic);
    backend.set_parallel_prefix_alignment(args.parallel_prefix_alignment);
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let batch_size = args.request_batch_size as usize;
    if args.warmup {
        let requests: Vec<_> = cases
            .iter()
            .take(batch_size)
            .map(|c| {
                let mut request = c.request.clone();
                if args.derive_facts {
                    request.state = l2s1::derive_facts(&request.state, &c.facts)?;
                }
                Ok(request)
            })
            .collect::<l2s1::Result<_>>()?;
        backend.decide_batch(&requests)?;
    }
    // Warmup must not populate the measured preparation-cache hit counts.
    backend.clear_preparation_cache();
    backend.take_timings(); // Exclude warmup profiling.
    // Create after ordinary warmup: no warmup KV leaks into the first measured call.
    let mut runner = match args.resident {
        Resident::None => Runner::Ordinary(&mut backend),
        Resident::FixedSchema => {
            Runner::Fixed(backend.shared_decision(cases[0].request.decisions[0].clone())?)
        }
        Resident::SharedPrefix => Runner::Shared(backend.parallel_prefix_session()?),
    };
    for (batch_index, batch) in cases.chunks(batch_size).enumerate() {
        let cache_before = serde_json::to_value(runner.stats())?;
        let started = Instant::now();
        let requests: Vec<_> = batch
            .iter()
            .map(|case| {
                let mut request = case.request.clone();
                if args.derive_facts {
                    request.state = l2s1::derive_facts(&request.state, &case.facts)?;
                }
                Ok(request)
            })
            .collect::<l2s1::Result<_>>()?;
        let preprocess_ms = started.elapsed().as_secs_f64() * 1000.0;
        let inference_started = Instant::now();
        let response = runner.decide_batch(&requests);
        let inference_ms = inference_started.elapsed().as_secs_f64() * 1000.0;
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let offset = batch_index * batch_size;
        let timings = serde_json::to_value(runner.timings())?;
        let cache_after = serde_json::to_value(runner.stats())?;
        let records: Vec<_> = match response {
            Ok(responses) => batch
                .iter()
                .zip(responses)
                .map(|(case, response)| {
                    serde_json::json!({
                        "id":case.id,"response":response
                    })
                })
                .collect(),
            Err(error) => batch
                .iter()
                .map(|case| {
                    serde_json::json!({
                        "id":case.id,"error":error.to_string()
                    })
                })
                .collect(),
        };
        for (index, (mut record, case)) in records.into_iter().zip(batch).enumerate() {
            record["resident"] = serde_json::to_value(args.resident)?;
            record["facts_sha256"] = serde_json::to_value(&fact_hashes[offset + index])?;
            record["preprocess_ms"] = preprocess_ms.into();
            record["inference_ms"] = inference_ms.into();
            // elapsed_ms is the article's full batch completion latency, not
            // latency divided by batch size. Throughput uses unique batch times.
            record["batch_profile"] = timings.clone();
            // Batch-level snapshots repeat on each record in that batch;
            // consumers must count each batch only once.
            record["preparation_cache_before"] = cache_before.clone();
            record["preparation_cache_after"] = cache_after.clone();
            record["decision_count"] = case.request.decisions.len().into();
            record["elapsed_ms"] = elapsed_ms.into();
            record["batch_elapsed_ms"] = elapsed_ms.into();
            record["amortized_elapsed_ms"] = (elapsed_ms / batch.len() as f64).into();
            record["batch_index"] = batch_index.into();
            record["batch_size"] = batch.len().into();
            record["load_ms"] = load_ms.into();
            serde_json::to_writer(&mut output, &record)?;
            writeln!(output)?;
        }
        output.flush()?;
        let completed = (batch_index * batch_size + batch.len()).min(cases.len());
        if completed.is_multiple_of(25) || completed == cases.len() || batch_size > 1 {
            println!("Completed {}/{}", completed, cases.len());
            std::io::stdout().flush()?;
        }
    }
    Ok(())
}
