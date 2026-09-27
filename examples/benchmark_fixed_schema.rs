#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        llama::{FixedSchemaBackend, LlamaBackend},
        *,
    };
    use serde_json::{Value, json};
    use std::{path::PathBuf, time::Instant};
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 256)]
        batch: u32,
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        #[arg(long)]
        cuda: bool,
    }
    let args = Args::parse();
    if args.rounds == 0 {
        return Err("rounds must be positive".into());
    }
    let suite: Value =
        serde_json::from_str(include_str!("../tests/fixtures/decision_benchmark.json"))?;
    let requests: Vec<DecisionRequest> = suite["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| serde_json::from_value(c["request"].clone()).unwrap())
        .collect();
    let mut fresh = LlamaBackend::load(
        &args.model,
        2048,
        args.batch,
        4,
        args.cuda,
        DecisionPolicy::default(),
    )?;
    let identity = fresh.identity();
    let mut reference = Vec::new();
    let mut fresh_ms = Vec::new();
    for round in 0..=args.rounds {
        let start = Instant::now();
        let values: Vec<_> = requests
            .iter()
            .map(|r| fresh.decide(r))
            .collect::<l2s1::Result<_>>()?;
        if round > 0 {
            fresh_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        reference = values;
    }
    fresh.set_execution_mode(ExecutionMode::PrefixReuse);
    let mut fixed = FixedSchemaBackend::new(fresh)?;
    let mut cold_ms = Vec::new();
    let mut warm_ms = Vec::new();
    let mut passes = Vec::new();
    for round in 0..=args.rounds {
        let start = Instant::now();
        let cold: Vec<_> = requests
            .iter()
            .map(|r| fixed.decide_cold(r))
            .collect::<l2s1::Result<_>>()?;
        let cold_time = start.elapsed().as_secs_f64() * 1000.0;
        fixed.clear();
        let start = Instant::now();
        let warm: Vec<_> = requests
            .iter()
            .map(|r| fixed.decide(r))
            .collect::<l2s1::Result<_>>()?;
        let warm_time = start.elapsed().as_secs_f64() * 1000.0;
        let mut reused = 0;
        let mut tokens = 0;
        let mut split_delta: f64 = 0.0;
        let mut plan_delta: f64 = 0.0;
        let mut plan_selections = 0;
        for ((a, b), old) in cold.iter().zip(&warm).zip(&reference) {
            for ((x, y), z) in a.results.iter().zip(&b.results).zip(&old.results) {
                reused += y.reused_prefix_tokens;
                tokens += y.input_tokens;
                if json!(&x.value) != json!(&y.value)
                    || json!(&x.abstention_reasons) != json!(&y.abstention_reasons)
                {
                    return Err("cold/warm selection mismatch".into());
                }
                split_delta = split_delta.max((x.candidate_mass - y.candidate_mass).abs());
                for ((p, q), r) in x.scores.iter().zip(&y.scores).zip(&z.scores) {
                    split_delta =
                        split_delta.max((p.option_probability - q.option_probability).abs());
                    plan_delta =
                        plan_delta.max((p.option_probability - r.option_probability).abs());
                }
                plan_selections += usize::from(json!(&x.value) != json!(&z.value));
            }
        }
        if split_delta > 1e-6 {
            return Err(format!("cold/warm probability mismatch {split_delta}").into());
        }
        if round > 0 {
            cold_ms.push(cold_time);
            warm_ms.push(warm_time);
            passes.push(json!({"input_tokens":tokens,"reused_tokens":reused,"cold_warm_max_delta":split_delta,"flat_split_max_probability_delta":plan_delta,"flat_split_selection_changes":plan_selections}));
        }
    }
    std::fs::write(
        &args.output,
        serde_json::to_vec_pretty(
            &json!({"scope":"decision-rules-v1, original multi-decision requests, one full warmup then paired passes; whole-pass milliseconds","identity":identity,"batch":args.batch,"rounds":args.rounds,"fresh_ms":fresh_ms,"split_cold_ms":cold_ms,"split_reuse_ms":warm_ms,"passes":passes}),
        )?,
    )?;
    Ok(())
}
#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!("requires llama feature");
}
