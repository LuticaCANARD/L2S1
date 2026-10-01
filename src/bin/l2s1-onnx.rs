use clap::Parser;
use l2s1::http::HttpDecisionBackend;
use std::{io::Read, path::PathBuf};
#[derive(Parser)]
#[command(about = "Laya ONNX typed decisions; set ORT_DYLIB_PATH to ONNX Runtime >=1.24")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    cuda: bool,
    #[arg(long, default_value_t = 4)]
    threads: usize,
    #[arg(long, default_value_t = 0.8)]
    min_top_probability: f64,
    #[arg(long, conflicts_with = "stdio")]
    listen: Option<String>,
    #[arg(long)]
    stdio: bool,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut backend = l2s1::onnx::OnnxBackend::load(
        &args.model_dir,
        args.cuda,
        args.threads,
        args.min_top_probability,
    )?;
    if let Some(address) = args.listen {
        return l2s1::http::serve(&address, &mut backend);
    }
    if args.stdio {
        return l2s1::stdio::serve(&mut backend);
    }
    let mut input = String::new();
    std::io::stdin().read_to_string(&mut input)?;
    let request = serde_json::from_str(&input)?;
    println!("{}", backend.decide_json(&request, &[])?);
    Ok(())
}
