use clap::Parser;
use skid_desion::{
    DecisionBackend, DecisionPolicy, DecisionRequest, PromptProfile, llama::LlamaBackend,
};
use std::{
    io::{self, Read},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    version,
    about = "Typed decisions from GGUF chat models. Read JSON from a file or standard input."
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Auto selects Qwen3 non-thinking or the model's embedded chat template.
    #[arg(long, value_enum, default_value_t = PromptProfile::Auto)]
    prompt_profile: PromptProfile,
    #[arg(long, default_value = "-")]
    input: String,
    /// CUDA device is required when selected; no silent CPU fallback.
    #[arg(long, value_enum, default_value_t = Device::Cpu)]
    device: Device,
    #[arg(long, default_value_t = 2048)]
    context: u32,
    #[arg(long, default_value_t = 256)]
    batch: u32,
    #[arg(long, default_value_t = 4)]
    threads: i32,
    #[arg(long, default_value_t = 0.8)]
    min_top_probability: f64,
    #[arg(long, default_value_t = 0.05)]
    min_candidate_mass: f64,
}

#[derive(Clone, Copy, clap::ValueEnum)]
enum Device {
    Cpu,
    Cuda,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let text = if args.input == "-" {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text)?;
        text
    } else {
        std::fs::read_to_string(&args.input)?
    };
    let request: DecisionRequest = serde_json::from_str(&text)?;
    request.validate()?;
    let policy = DecisionPolicy {
        min_top_probability: args.min_top_probability,
        min_candidate_mass: args.min_candidate_mass,
    };
    let mut backend = LlamaBackend::load_with_profile(
        &args.model,
        args.context,
        args.batch,
        args.threads,
        matches!(args.device, Device::Cuda),
        policy,
        args.prompt_profile,
    )?;
    let response = backend.decide(&request)?;
    serde_json::to_writer_pretty(io::stdout().lock(), &response)?;
    println!();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
