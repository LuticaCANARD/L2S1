use clap::Parser;
use l2s1::{DecisionRequest, openrouter::OpenRouterBackend};
use std::{
    io::{self, Read},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    version,
    about = "L2S1 selection-only decisions through OpenRouter chat completions"
)]
struct Args {
    /// OpenRouter model slug, for example prism-ml/ternary-bonsai-2-27b.
    #[arg(long)]
    model: String,
    /// JSON decision request file, or - for standard input.
    #[arg(long)]
    input: Option<String>,
    /// One PNG, JPEG, GIF or WebP image for CLI decisions.
    #[arg(long, conflicts_with = "listen")]
    image: Option<PathBuf>,
    /// Start POST /v1/decisions and GET /healthz at this address.
    #[arg(long, conflicts_with = "input")]
    listen: Option<String>,
    /// Maximum remote completion tokens per decision.
    #[arg(long, default_value_t = 1024, value_parser = clap::value_parser!(u32).range(1..=4096))]
    max_tokens: u32,
    /// Optional OpenRouter reasoning effort: none, minimal, low, medium, high or xhigh.
    #[arg(long)]
    reasoning_effort: Option<String>,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let key = std::env::var("OPENROUTER_API_KEY").map_err(|_| "OPENROUTER_API_KEY must be set")?;
    let mut backend = OpenRouterBackend::new(args.model, key)?;
    backend.set_max_tokens(args.max_tokens)?;
    if let Some(effort) = &args.reasoning_effort {
        backend.set_reasoning_effort(effort)?;
    }
    if let Some(address) = args.listen {
        return l2s1::http::serve(&address, &mut backend);
    }
    let text = if args.input.as_deref().unwrap_or("-") == "-" {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text)?;
        text
    } else {
        std::fs::read_to_string(args.input.expect("checked file input"))?
    };
    let request: DecisionRequest = serde_json::from_str(&text)?;
    let image = args.image.map(std::fs::read).transpose()?;
    let response = backend.decide(&request, image.as_deref())?;
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
