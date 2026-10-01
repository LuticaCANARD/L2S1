use clap::Parser;
use l2s1::{
    DecisionPolicy, DecisionRequest, ExecutionMode, PromptDetail, PromptLayout, wgpu::WgpuBackend,
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
    mmproj: Option<PathBuf>,
    #[arg(long, requires = "mmproj")]
    image: Option<PathBuf>,
    #[arg(long)]
    listen: Option<String>,
    #[arg(long, default_value = "-")]
    input: String,
    #[arg(long)]
    inspect: bool,
    #[arg(long)]
    diagnostics: bool,
    #[arg(long, help = "Allow a CPU Vulkan adapter for local development only")]
    allow_software_adapter: bool,
    #[arg(long, help = "Fail unless wgpu selected the Metal backend")]
    require_metal: bool,
    #[arg(long, value_enum, default_value_t = PromptLayout::Legacy)]
    prompt_layout: PromptLayout,
    #[arg(long, value_enum, default_value_t = PromptDetail::Minimal)]
    prompt_detail: PromptDetail,
    #[arg(long, value_enum, default_value_t = ExecutionMode::Fresh)]
    execution_mode: ExecutionMode,
    #[arg(long, default_value_t = 268_435_456)]
    snapshot_limit_bytes: usize,
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
    let mut backend = if let Some(projector) = &args.mmproj {
        WgpuBackend::load_with_software_adapter(
            &args.model,
            projector,
            policy,
            args.allow_software_adapter,
        )?
    } else {
        WgpuBackend::load_text(&args.model, policy, args.allow_software_adapter)?
    };
    if args.require_metal && backend.adapter_backend() != wgpu::Backend::Metal {
        return Err(format!(
            "wgpu selected {:?}, but Metal was required",
            backend.adapter_backend()
        )
        .into());
    }
    backend.set_prompt_layout(args.prompt_layout);
    backend.set_prompt_detail(args.prompt_detail);
    backend.set_execution_mode(args.execution_mode)?;
    backend.set_snapshot_limit_bytes(args.snapshot_limit_bytes);
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
    let (response, execution) = if let Some(path) = &args.image {
        backend.decide_vision_detailed(&request, &std::fs::read(path)?)?
    } else {
        backend.decide_detailed(&request)?
    };
    if args.diagnostics {
        serde_json::to_writer_pretty(
            io::stdout().lock(),
            &serde_json::json!({"response":response,"execution":execution}),
        )?;
    } else {
        serde_json::to_writer_pretty(io::stdout().lock(), &response)?;
    }
    println!();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
