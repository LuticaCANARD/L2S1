use clap::Parser;
use l2s1::{
    DecisionBackend, DecisionPolicy, DecisionRequest, PromptDetail, PromptLayout,
    VisionDecisionBackend, wgpu::WgpuBackend,
};
use std::{
    io::{self, Read},
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "L2S1 typed decisions using native wgpu GPU inference")]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    mmproj: PathBuf,
    #[arg(long)]
    image: Option<PathBuf>,
    #[arg(long)]
    listen: Option<String>,
    #[arg(long, default_value = "-")]
    input: String,
    #[arg(long)]
    inspect: bool,
    #[arg(long, help = "Allow a CPU Vulkan adapter for local development only")]
    allow_software_adapter: bool,
    #[arg(long, value_enum, default_value_t = PromptLayout::Legacy)]
    prompt_layout: PromptLayout,
    #[arg(long, value_enum, default_value_t = PromptDetail::Minimal)]
    prompt_detail: PromptDetail,
    #[arg(long, default_value_t = 0.8)]
    min_top_probability: f64,
    #[arg(long, default_value_t = 0.05)]
    min_candidate_mass: f64,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let policy = DecisionPolicy {
        min_top_probability: args.min_top_probability,
        min_candidate_mass: args.min_candidate_mass,
    };
    let mut backend = WgpuBackend::load_with_software_adapter(
        &args.model,
        &args.mmproj,
        policy,
        args.allow_software_adapter,
    )?;
    backend.set_prompt_layout(args.prompt_layout);
    backend.set_prompt_detail(args.prompt_detail);
    if args.inspect {
        serde_json::to_writer_pretty(io::stdout().lock(), backend.inspect())?;
        println!();
        return Ok(());
    }
    if let Some(address) = &args.listen {
        return l2s1::http::serve(address, &mut backend);
    }
    let text = if args.input == "-" {
        let mut text = String::new();
        io::stdin().read_to_string(&mut text)?;
        text
    } else {
        std::fs::read_to_string(&args.input)?
    };
    let request: DecisionRequest = serde_json::from_str(&text)?;
    let response = if let Some(path) = &args.image {
        backend.decide_vision(&request, &std::fs::read(path)?)?
    } else {
        backend.decide(&request)?
    };
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
