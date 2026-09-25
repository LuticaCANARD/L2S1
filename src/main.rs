use clap::Parser;
use l2s1::{
    ComputeOptions, DecisionBackend, DecisionPolicy, DecisionRequest, EvidenceTransfer,
    ExecutionMode, FlashAttention, PreparationCacheConfig, PromptDetail, PromptLayout,
    PromptProfile, llama::LlamaBackend,
};
use std::{
    io::{self, Read},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    version,
    about = "L2S1 (LLM to System 1). Typed decisions from GGUF chat models. Read JSON from a file or standard input."
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    /// Matching multimodal projector GGUF for direct image input.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Start a JSON HTTP API at this address, for example 127.0.0.1:8080.
    #[arg(long, conflicts_with_all = ["input", "image", "inspect", "preflight", "diagnostics"])]
    listen: Option<String>,
    /// One still image for CLI decisions; requires --mmproj.
    #[arg(long, requires = "mmproj", conflicts_with_all = ["inspect", "preflight", "diagnostics"])]
    image: Option<PathBuf>,
    /// Compact preserves full-vocabulary mass but only copies candidate scores to Rust.
    #[arg(long, value_enum, default_value_t = EvidenceTransfer::Full)]
    evidence_transfer: EvidenceTransfer,
    /// Retained preparation cache bytes; zero disables caching (default).
    #[arg(long, default_value_t = 0)]
    preparation_cache_bytes: usize,
    #[arg(long, default_value_t = 128)]
    preparation_cache_entries: usize,
    /// Print verified model identity and metadata capabilities, without reading input.
    #[arg(long, conflicts_with_all = ["preflight", "diagnostics"])]
    inspect: bool,
    /// Validate prompt, candidates and artifacts without inference.
    #[arg(long, conflicts_with = "diagnostics")]
    preflight: bool,
    /// Include request-local execution diagnostics and structured errors.
    #[arg(long)]
    diagnostics: bool,
    /// Task-scoped scalar calibration; repeat for multiple decision IDs.
    #[arg(long, conflicts_with = "output_head")]
    calibration: Vec<PathBuf>,
    /// Maximum bytes allocated for a request-local sequence snapshot.
    #[arg(long, default_value_t = 268435456)]
    snapshot_limit_bytes: usize,
    /// Optional compatible GGUF LoRA adapter, applied at scale 1.
    #[arg(long)]
    lora: Option<PathBuf>,
    /// Optional task-specific output head bound to this GGUF and compute profile.
    #[arg(long, conflicts_with = "lora")]
    output_head: Option<PathBuf>,
    /// Auto selects GPT-OSS final prefill, Qwen3 non-thinking, or the GGUF template.
    #[arg(long, value_enum, default_value_t = PromptProfile::Auto)]
    prompt_profile: PromptProfile,
    /// Fresh, prefix reuse, or experimental parallel questions; validate score drift first.
    #[arg(long, value_enum, default_value_t = ExecutionMode::Fresh)]
    execution_mode: ExecutionMode,
    /// Maximum questions per parallel wave (1..32); increases KV memory use.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=32))]
    parallel_width: u32,
    /// State-first improves shared-prefix reuse but can change model predictions.
    #[arg(long, value_enum, default_value_t = PromptLayout::Legacy)]
    prompt_layout: PromptLayout,
    /// Opt-in typed metadata and generic numeric comparison examples.
    #[arg(long, value_enum, default_value_t = PromptDetail::Minimal)]
    prompt_detail: PromptDetail,
    /// Cyclic answer-code assignment; semantic option/ordinal order is unchanged.
    #[arg(long, default_value_t = 0)]
    code_rotation: u32,
    #[arg(long, default_value = "-")]
    input: String,
    /// The selected GPU device is required; no silent CPU fallback.
    #[arg(long, value_enum, default_value_t = Device::Cpu)]
    device: Device,
    #[arg(long, default_value_t = 2048)]
    context: u32,
    #[arg(long, default_value_t = 256)]
    batch: u32,
    /// Physical token microbatch; defaults to --batch.
    #[arg(long)]
    ubatch: Option<u32>,
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
    Metal,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let policy = DecisionPolicy {
        min_top_probability: args.min_top_probability,
        min_candidate_mass: args.min_candidate_mass,
    };
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
    let mut backend = match args.device {
        Device::Metal => LlamaBackend::load_with_metal_options(
            &args.model,
            compute,
            policy,
            args.prompt_profile,
        )?,
        Device::Cpu | Device::Cuda => LlamaBackend::load_with_options(
            &args.model,
            compute,
            matches!(args.device, Device::Cuda),
            policy,
            args.prompt_profile,
        )?,
    };
    if let Some(path) = &args.lora {
        backend.load_lora(path)?;
    }
    if let Some(path) = &args.mmproj {
        backend.load_vision_projector(path)?;
    }
    backend.set_execution_mode(args.execution_mode);
    backend.set_parallel_width(args.parallel_width as usize)?;
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
    backend.set_snapshot_limit_bytes(args.snapshot_limit_bytes);
    for path in &args.calibration {
        backend.load_calibration(path)?;
    }
    if args.inspect {
        serde_json::to_writer_pretty(io::stdout().lock(), &backend.inspect())?;
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

    let output = if args.preflight {
        match backend.preflight(&request) {
            Ok(report) => serde_json::to_value(report)?,
            Err(error) => {
                serde_json::to_writer_pretty(
                    io::stdout().lock(),
                    &serde_json::json!({"error":error}),
                )?;
                println!();
                return Err(error.into());
            }
        }
    } else if args.diagnostics {
        match backend.decide_detailed(&request) {
            Ok(report) => serde_json::to_value(report)?,
            Err(error) => {
                serde_json::to_writer_pretty(
                    io::stdout().lock(),
                    &serde_json::json!({"error":error}),
                )?;
                println!();
                return Err(error.into());
            }
        }
    } else {
        serde_json::to_value(if let Some(path) = &args.image {
            backend.decide_vision(&request, &std::fs::read(path)?)?
        } else {
            backend.decide(&request)?
        })?
    };
    serde_json::to_writer_pretty(io::stdout().lock(), &output)?;
    println!();
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
