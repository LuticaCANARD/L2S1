#[cfg(feature = "onnx")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{DecisionRequest, http::HttpDecisionBackend, onnx::OnnxBackend};
    use serde_json::{Value, json};
    use std::{path::PathBuf, time::Instant};
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        cuda: bool,
    }
    let args = Args::parse();
    let rows: Vec<Value> = std::fs::read_to_string(args.input)?
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    let mut backend = OnnxBackend::load(&args.model_dir, args.cuda, 4, 0.0)?;
    let requests: Vec<DecisionRequest> = rows
        .iter()
        .map(|r| serde_json::from_value(r["request"].clone()).unwrap())
        .collect();
    if requests.is_empty() {
        return Err("empty benchmark".into());
    }
    for r in requests.iter().cycle().take(20) {
        backend.decide_json(r, &[])?;
    }
    let mut ms = Vec::new();
    let mut outputs = Vec::new();
    for _ in 0..2 {
        for (row, r) in rows.iter().zip(&requests) {
            let start = Instant::now();
            let response = backend.decide_json(r, &[])?;
            ms.push(start.elapsed().as_secs_f64() * 1000.0);
            outputs.push(
                json!({"id":row["id"],"response":response,"inputs":backend.inspect_input(r)?}),
            );
        }
    }
    let mut sorted = ms.clone();
    sorted.sort_by(f64::total_cmp);
    std::fs::write(
        args.output,
        serde_json::to_vec_pretty(
            &json!({"capabilities":backend.capabilities(),"n":ms.len(),"p50_ms":sorted[sorted.len()/2],"p95_ms":sorted[sorted.len()*95/100],"latencies_ms":ms,"outputs":outputs,"scope":"in-process preparation + forward + probability scoring, excludes HTTP/model loading; 20 warmups, 2 passes"}),
        )?,
    )?;
    Ok(())
}
#[cfg(not(feature = "onnx"))]
fn main() {
    eprintln!("requires onnx feature");
}
