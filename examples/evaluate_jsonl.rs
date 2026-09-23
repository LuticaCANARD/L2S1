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
        ComputeOptions, DecisionPolicy, DecisionRequest, ExecutionMode, FlashAttention,
        PromptLayout, PromptProfile, llama::LlamaBackend,
    };
    use serde::Deserialize;
    use std::{
        fs::{File, OpenOptions},
        io::{BufRead, BufReader, BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
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
        #[arg(long, value_enum, default_value_t = ExecutionMode::Fresh)]
        execution_mode: ExecutionMode,
        #[arg(long, value_enum, default_value_t = PromptLayout::Legacy)]
        prompt_layout: PromptLayout,
        #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=32))]
        parallel_width: u32,
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
    }

    let args = Args::parse();
    let cases: Vec<Case> = BufReader::new(File::open(&args.input)?)
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<Result<_, Box<dyn std::error::Error>>>()?;
    let mut ids = std::collections::HashSet::new();
    for case in &cases {
        if !ids.insert(&case.id) || case.id.is_empty() || case.request.decisions.len() != 1 {
            return Err("Expected unique case IDs and one decision per case".into());
        }
        case.request.validate()?;
    }
    if cases.is_empty() {
        return Err("Dataset must not be empty".into());
    }
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    let started = Instant::now();
    let mut backend = LlamaBackend::load_with_options(
        &args.model,
        ComputeOptions {
            context: args.context,
            batch: args.batch,
            ubatch: args.ubatch.unwrap_or(args.batch),
            threads: args.threads,
            flash_attention: args.flash_attention,
        },
        args.cuda,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )?;
    if let Some(path) = &args.lora {
        backend.load_lora(path)?;
    }
    backend.set_execution_mode(args.execution_mode);
    backend.set_prompt_layout(args.prompt_layout);
    if let Some(path) = &args.output_head {
        backend.load_output_head(path)?;
    }
    backend.set_parallel_width(args.parallel_width as usize)?;
    let load_ms = started.elapsed().as_secs_f64() * 1000.0;
    let batch_size = args.request_batch_size as usize;
    if args.warmup {
        let requests: Vec<_> = cases
            .iter()
            .take(batch_size)
            .map(|c| c.request.clone())
            .collect();
        backend.decide_batch(&requests)?;
    }
    backend.take_timings(); // Exclude warmup profiling.
    for (batch_index, batch) in cases.chunks(batch_size).enumerate() {
        let requests: Vec<_> = batch.iter().map(|case| case.request.clone()).collect();
        let started = Instant::now();
        let response = backend.decide_batch(&requests);
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let timings = serde_json::to_value(backend.take_timings())?;
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
        for mut record in records {
            // elapsed_ms is the article's full batch completion latency, not
            // latency divided by batch size. Throughput uses unique batch times.
            record["batch_profile"] = timings.clone();
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
