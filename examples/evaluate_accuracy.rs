//! Label-free paired inference for prompt and answer-code experiments.
#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!("Enable --features llama");
    std::process::exit(1);
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{llama::LlamaBackend, *};
    use serde::Deserialize;
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{
        collections::HashSet,
        fs::OpenOptions,
        io::{BufRead, BufReader, BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        lora: Option<PathBuf>,
        #[arg(long)]
        cuda: bool,
        #[arg(long, value_enum, value_delimiter = ',', default_value = "minimal")]
        prompt_details: Vec<PromptDetail>,
        #[arg(long, value_enum, value_delimiter = ',', default_value = "legacy")]
        layouts: Vec<PromptLayout>,
        /// One pass for each distinct cyclic assignment; additionally emit their mixture.
        #[arg(long)]
        all_rotations: bool,
        #[arg(long, default_value_t = 2048)]
        context: u32,
        #[arg(long, default_value_t = 32)]
        batch: u32,
        #[arg(long, default_value_t = 4)]
        threads: i32,
    }

    #[derive(Deserialize)]
    struct Case {
        id: String,
        #[serde(default)]
        group: String,
        request: DecisionRequest,
    }

    let args = Args::parse();
    let input_bytes = std::fs::read(&args.input)?;
    let input_sha256 = format!("{:x}", Sha256::digest(&input_bytes));
    let cases: Vec<Case> = BufReader::new(input_bytes.as_slice())
        .lines()
        .map(|line| Ok(serde_json::from_str(&line?)?))
        .collect::<std::result::Result<_, Box<dyn std::error::Error>>>()?;
    if cases.is_empty() {
        return Err("empty dataset".into());
    }
    let mut ids = HashSet::new();
    for case in &cases {
        case.request.validate()?;
        if case.id.is_empty() || !ids.insert(&case.id) || case.request.decisions.len() != 1 {
            return Err("expected unique logical IDs and one decision per case".into());
        }
    }
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?,
    );
    let policy = DecisionPolicy::default();
    let mut backend = LlamaBackend::load(
        &args.model,
        args.context,
        args.batch,
        args.threads,
        args.cuda,
        policy.clone(),
    )?;
    if let Some(adapter) = &args.lora {
        backend.load_lora(adapter)?;
    }
    for layout in &args.layouts {
        for detail in &args.prompt_details {
            backend.set_prompt_layout(*layout);
            backend.set_prompt_detail(*detail);
            backend.set_code_rotation(0)?;
            backend.decide(&cases[0].request)?; // Excluded warmup, without labels.
            let setting = json!({"prompt_layout":layout,"prompt_detail":detail});
            for (index, case) in cases.iter().enumerate() {
                let decision = &case.request.decisions[0];
                let count = decision.options().len();
                let rotations = if args.all_rotations { count } else { 1 };
                let mut passes = Vec::new();
                let mut identities = Vec::new();
                let mut pass_ms = 0.0;
                for rotation in 0..rotations {
                    let pass_started = Instant::now();
                    backend.set_code_rotation(rotation)?;
                    let mut response = backend.decide(&case.request)?;
                    let elapsed_ms = pass_started.elapsed().as_secs_f64() * 1000.0;
                    pass_ms += elapsed_ms;
                    let identity = backend.identity();
                    // One owned backend fixes checkpoint, template, device, layout,
                    // detail and adapter for every pass. Only code rotation changes.
                    writeln!(
                        output,
                        "{}",
                        json!({"id":case.id,"group":case.group,"setting":setting,
                        "input_sha256":input_sha256,"latency_scope":"compute_path",
                        "kind":"single","rotation":rotation,"elapsed_ms":elapsed_ms,
                        "model_identity":identity,"response":response})
                    )?;
                    identities.push(identity);
                    passes.push(response.results.remove(0));
                }
                if rotations > 1 {
                    let started = Instant::now();
                    let result = score_semantic_mixture(decision, &passes, &policy)?;
                    let elapsed_ms = pass_ms + started.elapsed().as_secs_f64() * 1000.0;
                    writeln!(
                        output,
                        "{}",
                        json!({"id":case.id,"group":case.group,"setting":setting,
                        "input_sha256":input_sha256,"latency_scope":"compute_path",
                        "kind":"ensemble","rotations":(0..rotations).collect::<Vec<_>>(),
                        "elapsed_ms":elapsed_ms,"model_identities":identities,
                        "response":{"policy":policy,"results":[result]}})
                    )?;
                }
                output.flush()?;
                if (index + 1) % 30 == 0 {
                    eprintln!("{setting}: {}/{} logical cases", index + 1, cases.len());
                }
            }
        }
    }
    Ok(())
}
