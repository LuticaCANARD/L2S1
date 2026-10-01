//! Paired fixed-schema replay of decision-rules-v1 or unlabeled JSONL requests.
#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!("Enable --features llama");
    std::process::exit(1);
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        ComputeOptions, Decision, DecisionBackend, DecisionPolicy, DecisionRequest,
        DecisionResponse, ExecutionMode, PromptProfile, llama::LlamaBackend,
    };
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, fs::OpenOptions, path::PathBuf, time::Instant};
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        /// Unlabeled {id, request} JSONL; omit for the decision-rules-v1 fixture.
        #[arg(long)]
        input: Option<PathBuf>,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value="cpu", value_parser=["cpu","cuda","metal"])]
        device: String,
        #[arg(long, default_value_t = 3)]
        rounds: usize,
        #[arg(long, default_value_t = 256)]
        batch: u32,
        #[arg(long, default_value_t = 2048)]
        context: u32,
        #[arg(long, default_value_t = 4)]
        threads: i32,
        /// Number of layers offloaded to CUDA; omit for automatic placement.
        #[arg(long)]
        gpu_layers: Option<u32>,
        /// Keep the first N layers' MoE experts on CPU.
        #[arg(long, default_value_t = 0)]
        cpu_moe_layers: u32,
    }
    struct Group {
        decision: Decision,
        cases: Vec<(String, Value, Option<String>)>,
    }
    fn run(
        backend: &mut LlamaBackend,
        groups: &[Group],
        reuse: bool,
    ) -> l2s1::Result<Vec<DecisionResponse>> {
        backend.set_execution_mode(if reuse {
            ExecutionMode::PrefixReuse
        } else {
            ExecutionMode::Fresh
        });
        let mut out = Vec::new();
        for group in groups {
            if reuse {
                let mut session = backend.shared_decision(group.decision.clone())?;
                for (_, state, _) in &group.cases {
                    out.push(session.decide(state.clone())?);
                }
            } else {
                for (_, state, _) in &group.cases {
                    out.push(backend.decide(&DecisionRequest {
                        shared: None,
                        state: state.clone(),
                        decisions: vec![group.decision.clone()],
                    })?);
                }
            }
        }
        Ok(out)
    }
    let args = Args::parse();
    if args.rounds == 0 {
        return Err("rounds must be positive".into());
    }
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let suite: Value = if let Some(input) = &args.input {
        let cases: Vec<Value> = std::fs::read_to_string(input)?
            .lines()
            .map(serde_json::from_str)
            .collect::<Result<_, _>>()?;
        json!({"id":"jsonl-schema-reuse-v1", "cases":cases})
    } else {
        serde_json::from_str(include_str!("../tests/fixtures/decision_benchmark.json"))?
    };
    let mut grouped = BTreeMap::<String, Group>::new();
    let mut case_ids = std::collections::HashSet::new();
    for case in suite["cases"].as_array().unwrap() {
        let id = case["id"].as_str().ok_or("case ID must be a string")?;
        if id.is_empty() || !case_ids.insert(id.to_owned()) {
            return Err("case IDs must be unique and nonempty".into());
        }
        let request: DecisionRequest = serde_json::from_value(case["request"].clone())?;
        request.validate()?;
        for decision in request.decisions {
            let expected = case["expected"][&decision.id].as_str().map(str::to_owned);
            grouped
                .entry(serde_json::to_string(&decision)?)
                .or_insert_with(|| Group {
                    decision,
                    cases: Vec::new(),
                })
                .cases
                .push((id.to_owned(), request.state.clone(), expected));
        }
    }
    if grouped.is_empty() {
        return Err("input must contain at least one decision".into());
    }
    let groups = grouped.into_values().collect::<Vec<_>>();
    let compute = ComputeOptions {
        context: args.context,
        batch: args.batch,
        ubatch: args.batch,
        threads: args.threads,
        gpu_layers: args.gpu_layers,
        cpu_moe_layers: args.cpu_moe_layers,
        ..Default::default()
    };
    let load = Instant::now();
    let mut backend = if args.device == "metal" {
        LlamaBackend::load_with_metal_options(
            &args.model,
            compute,
            DecisionPolicy::default(),
            PromptProfile::Auto,
        )?
    } else {
        LlamaBackend::load_with_options(
            &args.model,
            compute,
            args.device == "cuda",
            DecisionPolicy::default(),
            PromptProfile::Auto,
        )?
    };
    let load_ms = load.elapsed().as_secs_f64() * 1000.;
    let identity = backend.identity();
    let mut records = Vec::new();
    for round in 0..args.rounds {
        let order = if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let mut pair = Vec::new();
        for reuse in order {
            let mode = if reuse { "shared_decision" } else { "fresh" };
            eprintln!("round {}: {mode} warmup", round + 1);
            run(&mut backend, &groups, reuse)?;
            backend.take_timings();
            eprintln!("round {}: {mode} measured pass", round + 1);
            let start = Instant::now();
            let responses = run(&mut backend, &groups, reuse)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
            let timings = backend.take_timings();
            pair.push((reuse, responses, elapsed_ms, timings));
        }
        let fresh = &pair.iter().find(|p| !p.0).unwrap().1;
        let reused = &pair.iter().find(|p| p.0).unwrap().1;
        let mut max_probability_delta = 0.0_f64;
        let mut max_mass_delta = 0.0_f64;
        let mut changed_selections = 0;
        let mut changed_top1 = 0;
        let mut changed_abstentions = 0;
        for (a, b) in fresh.iter().zip(reused) {
            let (a, b) = (&a.results[0], &b.results[0]);
            assert_eq!(a.input_tokens, b.input_tokens);
            let selected = |r: &l2s1::DecisionResult| match &r.value {
                l2s1::DecisionValue::Binary { value, .. } => json!(value),
                l2s1::DecisionValue::Choice { selected }
                | l2s1::DecisionValue::Ordinal { selected, .. } => json!(selected),
            };
            changed_selections += usize::from(selected(a) != selected(b));
            changed_abstentions +=
                usize::from(json!(&a.abstention_reasons) != json!(&b.abstention_reasons));
            let top = |r: &l2s1::DecisionResult| {
                r.scores
                    .iter()
                    .max_by(|x, y| x.option_probability.total_cmp(&y.option_probability))
                    .unwrap()
                    .id
                    .clone()
            };
            changed_top1 += usize::from(top(a) != top(b));
            max_mass_delta = max_mass_delta.max((a.candidate_mass - b.candidate_mass).abs());
            for (x, y) in a.scores.iter().zip(&b.scores) {
                assert_eq!((&x.id, &x.code, x.token_id), (&y.id, &y.code, y.token_id));
                max_probability_delta =
                    max_probability_delta.max((x.option_probability - y.option_probability).abs());
            }
        }
        let difference = json!({"max_probability_delta":max_probability_delta,"max_mass_delta":max_mass_delta,
            "changed_selections":changed_selections,"changed_top1":changed_top1,"changed_abstentions":changed_abstentions});
        for (reuse, responses, elapsed_ms, timings) in pair {
            let input: usize = responses.iter().map(|r| r.results[0].input_tokens).sum();
            let reused: usize = responses
                .iter()
                .map(|r| r.results[0].reused_prefix_tokens)
                .sum();
            records.push(
                json!({"round":round,"mode":if reuse{"shared_decision"}else{"fresh"},
                "elapsed_ms":elapsed_ms,"input_tokens":input,"reused_prefix_tokens":reused,
                "timings":timings,"difference":difference,"responses":responses}),
            );
        }
        eprintln!("round {}: {}", round + 1, difference);
    }
    let cases: Vec<_> = groups
        .iter()
        .flat_map(|g| {
            g.cases.iter().map(|(id, state, expected)| {
                json!({
        "case":id,"decision":g.decision,"state":state,"expected":expected})
            })
        })
        .collect();
    let passed = records.iter().all(|r| {
        r["difference"]["changed_selections"] == 0
            && r["difference"]["changed_top1"] == 0
            && r["difference"]["changed_abstentions"] == 0
            && r["difference"]["max_probability_delta"].as_f64().unwrap() <= 0.02
            && r["difference"]["max_mass_delta"].as_f64().unwrap() <= 0.02
    });
    serde_json::to_writer_pretty(
        file,
        &json!({"suite":suite["id"],"identity":identity,"load_ms_excluded":load_ms,
        "device":args.device,"batch":args.batch,"rounds":args.rounds,"unique_decisions":cases.len(),
        "schemas":groups.len(),"cases":cases,"records":records,"equivalence_passed":passed,
        "input":args.input,
        "scope":"All input decisions grouped by immutable schema with changing states. One decision per call in both paths; full group execution includes session creation/drop, preparation and scoring. Grouped pass timings are not original request p50. One resident model, untimed warmup before each path, alternating path order; preparation cache disabled. Repeats are not independent quality samples. Exact prefix matching and original batch alignment retained. External JSONL is unlabeled; equivalence is not accuracy."}),
    )?;
    if !passed {
        return Err(
            "schema reuse changed selections or exceeded score tolerance; inspect report".into(),
        );
    }
    Ok(())
}
