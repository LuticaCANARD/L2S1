use clap::Parser;
use l2s1::{
    ComputeOptions, DecisionBackend, DecisionPolicy, DecisionRequest, EvidenceTransfer,
    ExecutionMode, FlashAttention, ParallelPrefixAlignment, ParallelWaveOrder,
    PreparationCacheConfig, PromptDetail, PromptLayout, PromptProfile, ReasoningMode,
    ReasoningOptions, llama::LlamaBackend,
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
    /// Direct option scoring, or bounded Qwen3 text thinking followed by scoring.
    /// Thinking needs fresh execution, so it disables automatic prefix reuse.
    #[arg(long, value_enum, default_value_t = ReasoningMode::Direct)]
    reasoning_mode: ReasoningMode,
    /// Thinking token budget; incomplete thinking fails rather than forcing an answer.
    #[arg(long, default_value_t = 128, value_parser = clap::value_parser!(u32).range(1..=1024))]
    max_reasoning_tokens: u32,
    #[arg(long)]
    model: PathBuf,
    /// Matching multimodal projector GGUF for direct image input.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Reuse identical image chunks within a native wave, encoding unique chunks
    /// individually to preserve the single-chunk projector path.
    #[arg(long, requires = "mmproj", conflicts_with = "vision_optimized")]
    vision_projector_reuse: bool,
    /// Vision throughput profile: parallel4, dynamic KV, batch1024,
    /// FlashAttention, compact evidence, bounded preparation and image reuse.
    #[arg(long, requires = "mmproj", conflicts_with_all = ["execution_mode", "parallel_width", "parallel_context_dynamic", "batch", "ubatch", "flash_attention", "evidence_transfer", "preparation_cache_bytes", "preparation_cache_entries", "output_head", "calibration", "family_calibration"])]
    vision_optimized: bool,
    /// Start a JSON HTTP API at this address, for example 127.0.0.1:8080.
    #[arg(long, conflicts_with_all = ["input", "image", "inspect", "preflight", "diagnostics"])]
    listen: Option<String>,
    /// Require deterministic schema-prefix KV reuse. Compatible text servers use it by default.
    #[arg(long)]
    fixed_schema: bool,
    /// Optional Laya ONNX fast path. Without a validated cascade policy, use slow only.
    #[cfg(feature = "onnx")]
    #[arg(long, requires = "listen")]
    fast_model_dir: Option<PathBuf>,
    #[cfg(feature = "onnx")]
    #[arg(long, requires = "fast_model_dir")]
    fast_cuda: bool,
    #[cfg(feature = "onnx")]
    #[arg(long, requires = "fast_model_dir")]
    cascade_policy: Option<PathBuf>,
    /// Resident stdin/stdout RPC without opening a network listener.
    #[arg(long, conflicts_with_all = ["listen", "input", "image", "inspect", "preflight", "diagnostics"])]
    stdio: bool,
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
    /// Fallback calibration per decision kind and option count for tasks
    /// without a task-scoped calibration; repeat for multiple scopes.
    #[arg(long, conflicts_with = "output_head")]
    family_calibration: Vec<PathBuf>,
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
    /// Override automatic resident prefix reuse; use fresh for independent evaluation.
    #[arg(long, value_enum)]
    execution_mode: Option<ExecutionMode>,
    /// Maximum questions per parallel wave (1..32); increases KV memory use.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..=32))]
    parallel_width: u32,
    /// Size each parallel KV context from actual input tokens plus one batch of headroom.
    #[arg(long)]
    parallel_context_dynamic: bool,
    /// State-first improves shared-prefix reuse but can change model predictions.
    /// Defaults to state-first for text parallel execution and legacy otherwise.
    #[arg(long, value_enum)]
    prompt_layout: Option<PromptLayout>,
    /// Parallel shared-prefix rounding: batch keeps serial decode boundaries;
    /// token shares every common token and can change scores slightly.
    #[arg(long, value_enum, default_value_t = ParallelPrefixAlignment::Batch)]
    parallel_prefix_alignment: ParallelPrefixAlignment,
    /// Parallel wave membership: prefix groups questions with common token
    /// prefixes; request keeps consecutive request-order waves.
    #[arg(long, value_enum, default_value_t = ParallelWaveOrder::Prefix)]
    parallel_wave_order: ParallelWaveOrder,
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
    /// Per-question token limit; defaults to 2048, or 4096 with --vision-optimized.
    #[arg(long)]
    context: Option<u32>,
    #[arg(long, default_value_t = 256)]
    batch: u32,
    /// Physical token microbatch; defaults to --batch.
    #[arg(long)]
    ubatch: Option<u32>,
    #[arg(long, value_enum, default_value_t = FlashAttention::Off)]
    flash_attention: FlashAttention,
    /// Maximum layers on the selected GPU; omit to preserve full offload.
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

impl Args {
    fn automatic_fixed_schema(&self) -> bool {
        (self.listen.is_some() || self.stdio)
            && self.execution_mode.is_none()
            && !self.fixed_schema
            && self.mmproj.is_none()
            && self.output_head.is_none()
            && self.calibration.is_empty()
            && self.family_calibration.is_empty()
            && self.evidence_transfer == EvidenceTransfer::Full
            && self.reasoning_mode == ReasoningMode::Direct
    }

    fn selected_execution_mode(&self) -> ExecutionMode {
        self.execution_mode.unwrap_or_else(|| {
            if self.fixed_schema || self.automatic_fixed_schema() {
                ExecutionMode::PrefixReuse
            } else {
                ExecutionMode::Fresh
            }
        })
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.fixed_schema && args.listen.is_none() && !args.stdio {
        return Err("--fixed-schema requires --listen or --stdio".into());
    }
    let policy = DecisionPolicy {
        min_top_probability: args.min_top_probability,
        min_candidate_mass: args.min_candidate_mass,
    };
    if args.vision_optimized && matches!(args.device, Device::Cpu) {
        return Err("--vision-optimized requires --device cuda or --device metal".into());
    }
    let profile = if args.vision_optimized {
        ComputeOptions::vision_optimized()
    } else {
        ComputeOptions::default()
    };
    let compute = ComputeOptions {
        context: args.context.unwrap_or(profile.context),
        batch: if args.vision_optimized {
            profile.batch
        } else {
            args.batch
        },
        ubatch: if args.vision_optimized {
            profile.ubatch
        } else {
            args.ubatch.unwrap_or(args.batch)
        },
        threads: args.threads,
        flash_attention: if args.vision_optimized {
            profile.flash_attention
        } else {
            args.flash_attention
        },
        gpu_layers: args.gpu_layers,
        cpu_moe_layers: args.cpu_moe_layers,
        model_load_mode: if args.vision_optimized
            && args.model_load_mode == l2s1::ModelLoadMode::Auto
        {
            profile.model_load_mode
        } else {
            args.model_load_mode
        },
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
    backend.set_execution_mode(args.selected_execution_mode());
    backend.set_parallel_width(args.parallel_width as usize)?;
    backend.set_parallel_context_dynamic(args.parallel_context_dynamic);
    if let Some(layout) = args.prompt_layout {
        backend.set_prompt_layout(layout);
    }
    backend.set_parallel_prefix_alignment(args.parallel_prefix_alignment);
    backend.set_parallel_wave_order(args.parallel_wave_order);
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
    for path in &args.family_calibration {
        backend.load_family_calibration(path)?;
    }
    if args.vision_projector_reuse {
        backend.set_vision_projector_reuse(true);
    }
    if args.vision_optimized {
        backend.enable_vision_optimizations()?;
    }
    let mut fixed_schema = args.fixed_schema;
    if args.automatic_fixed_schema() {
        if l2s1::llama::FixedSchemaBackend::is_compatible(&backend) {
            fixed_schema = true;
        } else {
            backend.set_execution_mode(ExecutionMode::Fresh);
            eprintln!(
                "l2s1: automatic fixed-schema prefix reuse is unavailable for this model; using fresh execution"
            );
        }
    }
    backend.set_reasoning(ReasoningOptions {
        mode: args.reasoning_mode,
        max_tokens: args.max_reasoning_tokens as usize,
    })?;
    if args.inspect {
        serde_json::to_writer_pretty(io::stdout().lock(), &backend.inspect())?;
        println!();
        return Ok(());
    }
    if let Some(address) = &args.listen {
        if fixed_schema {
            let fixed = l2s1::llama::FixedSchemaBackend::new(backend)?;
            return serve_selected(&args, address, fixed);
        }
        return serve_selected(&args, address, backend);
    }
    if args.stdio {
        if fixed_schema {
            return l2s1::stdio::serve(&mut l2s1::llama::FixedSchemaBackend::new(backend)?);
        }
        return l2s1::stdio::serve(&mut backend);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_cli_is_explicit_and_bounded() {
        let direct = Args::try_parse_from(["l2s1", "--model", "m.gguf"]).unwrap();
        assert_eq!(direct.reasoning_mode, ReasoningMode::Direct);
        assert_eq!(direct.max_reasoning_tokens, 128);
        let thinking = Args::try_parse_from([
            "l2s1",
            "--model",
            "m.gguf",
            "--reasoning-mode",
            "thinking",
            "--max-reasoning-tokens",
            "512",
        ])
        .unwrap();
        assert_eq!(thinking.reasoning_mode, ReasoningMode::Thinking);
        assert_eq!(thinking.max_reasoning_tokens, 512);
        for budget in ["0", "1025"] {
            let argv = [
                "l2s1",
                "--model",
                "m.gguf",
                "--max-reasoning-tokens",
                budget,
            ];
            assert!(Args::try_parse_from(argv).is_err());
        }
    }

    #[test]
    fn resident_defaults_and_explicit_execution_contracts() {
        let parse = |flags: &[&str]| {
            let mut argv = vec!["l2s1", "--model", "m.gguf"];
            argv.extend_from_slice(flags);
            Args::try_parse_from(argv).unwrap()
        };
        for flags in [vec!["--stdio"], vec!["--listen", "127.0.0.1:0"]] {
            let args = parse(&flags);
            assert!(args.automatic_fixed_schema());
            assert_eq!(args.selected_execution_mode(), ExecutionMode::PrefixReuse);
        }
        let cli = parse(&[]);
        assert!(!cli.automatic_fixed_schema());
        assert_eq!(cli.selected_execution_mode(), ExecutionMode::Fresh);
        for (name, expected) in [
            ("fresh", ExecutionMode::Fresh),
            ("prefix-reuse", ExecutionMode::PrefixReuse),
            ("parallel", ExecutionMode::Parallel),
            ("state-restore", ExecutionMode::StateRestore),
        ] {
            let args = parse(&["--stdio", "--execution-mode", name]);
            assert!(!args.automatic_fixed_schema());
            assert_eq!(args.selected_execution_mode(), expected);
        }
        for flags in [
            vec!["--stdio", "--reasoning-mode", "thinking"],
            vec!["--stdio", "--mmproj", "projector.gguf"],
            vec!["--stdio", "--calibration", "calibration.json"],
            vec!["--stdio", "--family-calibration", "family.json"],
            vec!["--stdio", "--output-head", "head.json"],
            vec!["--stdio", "--evidence-transfer", "compact"],
        ] {
            let args = parse(&flags);
            assert!(!args.automatic_fixed_schema());
            assert_eq!(args.selected_execution_mode(), ExecutionMode::Fresh);
        }
        let required = parse(&["--stdio", "--fixed-schema"]);
        assert!(!required.automatic_fixed_schema());
        assert_eq!(
            required.selected_execution_mode(),
            ExecutionMode::PrefixReuse
        );
    }

    #[test]
    fn optimized_profile_requires_projector_and_rejects_partial_overrides() {
        assert!(Args::try_parse_from(["l2s1", "--model", "m.gguf", "--vision-optimized"]).is_err());
        for flag in [
            "--batch",
            "--ubatch",
            "--parallel-width",
            "--preparation-cache-bytes",
        ] {
            assert!(
                Args::try_parse_from([
                    "l2s1",
                    "--model",
                    "m.gguf",
                    "--mmproj",
                    "p.gguf",
                    "--vision-optimized",
                    flag,
                    "4",
                ])
                .is_err()
            );
        }
        assert!(
            Args::try_parse_from([
                "l2s1",
                "--model",
                "m.gguf",
                "--mmproj",
                "p.gguf",
                "--device",
                "cuda",
                "--vision-optimized",
                "--context",
                "8192",
            ])
            .is_ok()
        );
        assert!(
            Args::try_parse_from([
                "l2s1",
                "--model",
                "m.gguf",
                "--mmproj",
                "p.gguf",
                "--vision-optimized",
                "--flash-attention",
                "off",
            ])
            .is_err()
        );
        assert!(
            Args::try_parse_from(["l2s1", "--model", "m.gguf", "--vision-projector-reuse",])
                .is_err()
        );
        assert!(
            Args::try_parse_from([
                "l2s1",
                "--model",
                "m.gguf",
                "--mmproj",
                "p.gguf",
                "--vision-optimized",
                "--vision-projector-reuse",
            ])
            .is_err()
        );
        let reused = Args::try_parse_from([
            "l2s1",
            "--model",
            "m.gguf",
            "--mmproj",
            "p.gguf",
            "--execution-mode",
            "parallel",
            "--vision-projector-reuse",
        ])
        .unwrap();
        assert!(reused.vision_projector_reuse);
        assert!(!reused.vision_optimized);
        assert!(
            !Args::try_parse_from(["l2s1", "--model", "m.gguf", "--mmproj", "p.gguf",])
                .unwrap()
                .vision_projector_reuse
        );
        // The serial preparation/evidence optimizations remain constituents of
        // the batched optimized profile, not a separate public execution mode.
        assert!(
            Args::try_parse_from([
                "l2s1",
                "--model",
                "m.gguf",
                "--mmproj",
                "p.gguf",
                "--vision-preserving",
            ])
            .is_err()
        );
    }
}

fn serve_selected<B: l2s1::http::HttpDecisionBackend>(
    args: &Args,
    address: &str,
    mut backend: B,
) -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "onnx")]
    if let Some(directory) = &args.fast_model_dir {
        let fast =
            l2s1::onnx::OnnxBackend::load(directory, args.fast_cuda, args.threads as usize, 0.0)?;
        let policy = args
            .cascade_policy
            .as_ref()
            .map(|p| {
                std::fs::read(p)
                    .map_err(Box::<dyn std::error::Error>::from)
                    .and_then(|b| serde_json::from_slice(&b).map_err(Into::into))
            })
            .transpose()?;
        let mut cascade = l2s1::cascade::CascadeBackend::new(fast, backend, policy)?;
        return l2s1::http::serve(address, &mut cascade);
    }
    let _ = args;
    l2s1::http::serve(address, &mut backend)
}
