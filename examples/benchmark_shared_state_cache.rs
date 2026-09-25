//! Measures repeated-state, changing-question calls with and without decoder prefix reuse.
#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!("Enable --features llama to run this benchmark");
    std::process::exit(1);
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        Decision, DecisionBackend, DecisionPolicy, DecisionRequest, DecisionResult, ExecutionMode,
        PromptLayout, llama::LlamaBackend,
    };
    use serde_json::{Value, json};
    use std::{fs::OpenOptions, path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 5)]
        repeats: usize,
        #[arg(long, default_value_t = 2048)]
        context: u32,
        #[arg(long, default_value_t = 32)]
        batch: u32,
        #[arg(long, default_value_t = 4)]
        threads: i32,
        #[arg(long)]
        cuda: bool,
    }

    fn run(
        backend: &mut LlamaBackend,
        state: &Value,
        questions: &[Decision],
        shared: bool,
    ) -> l2s1::Result<Vec<DecisionResult>> {
        if shared {
            backend.set_execution_mode(ExecutionMode::PrefixReuse);
            let mut session = backend.shared_state(state.clone())?;
            questions
                .iter()
                .map(|question| Ok(session.decide(vec![question.clone()])?.results.remove(0)))
                .collect()
        } else {
            backend.set_execution_mode(ExecutionMode::Fresh);
            questions
                .iter()
                .map(|question| {
                    Ok(backend
                        .decide(&DecisionRequest {
                            state: state.clone(),
                            decisions: vec![question.clone()],
                        })?
                        .results
                        .remove(0))
                })
                .collect()
        }
    }

    fn difference(fresh: &[DecisionResult], shared: &[DecisionResult]) -> Value {
        assert_eq!(fresh.len(), shared.len());
        let (mut probability, mut mass, mut logit) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut changed_values = 0;
        let mut changed_abstentions = 0;
        for (a, b) in fresh.iter().zip(shared) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.input_tokens, b.input_tokens);
            assert_eq!(a.scores.len(), b.scores.len());
            mass = mass.max((a.candidate_mass - b.candidate_mass).abs());
            changed_values += usize::from(json!(&a.value) != json!(&b.value));
            changed_abstentions +=
                usize::from(json!(&a.abstention_reasons) != json!(&b.abstention_reasons));
            for (left, right) in a.scores.iter().zip(&b.scores) {
                assert_eq!(left.id, right.id);
                probability =
                    probability.max((left.option_probability - right.option_probability).abs());
                logit = logit.max((left.raw_logit - right.raw_logit).abs());
            }
        }
        json!({"max_probability_delta": probability, "max_candidate_mass_delta": mass,
            "max_raw_logit_delta": logit, "changed_values": changed_values,
            "changed_abstention_reasons": changed_abstentions})
    }

    let args = Args::parse();
    if args.repeats == 0 {
        return Err("repeats must be positive".into());
    }
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let fixture: DecisionRequest = serde_json::from_str(include_str!("warehouse.json"))?;
    let load = Instant::now();
    let mut backend = LlamaBackend::load(
        &args.model,
        args.context,
        args.batch,
        args.threads,
        args.cuda,
        DecisionPolicy::default(),
    )?;
    let load_ms = load.elapsed().as_secs_f64() * 1000.0;
    backend.set_prompt_layout(PromptLayout::StateFirst);
    let identity = backend.identity();
    let mut records = Vec::new();

    for filler_words in [0, 200] {
        let mut state = fixture.state.clone();
        if filler_words > 0 {
            state["irrelevant_packing_notes"] = json!("packing ".repeat(filler_words));
        }
        for count in [1, 4, 16] {
            let questions: Vec<Decision> = (0..count)
                .map(|index| {
                    let mut question = fixture.decisions[index % fixture.decisions.len()].clone();
                    question.id = format!("{}_{}", question.id, index);
                    question.instruction.push_str(&format!(
                        " Case reference {index}; apply the rule to this shipment."
                    ));
                    question
                })
                .collect();
            for round in 0..args.repeats {
                let order = if round % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                };
                let mut outcomes = Vec::new();
                for shared in order {
                    run(&mut backend, &state, &questions, shared)?;
                    backend.take_timings();
                    let start = Instant::now();
                    let results = run(&mut backend, &state, &questions, shared)?;
                    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let timings = backend.take_timings();
                    let input_tokens = results.iter().map(|r| r.input_tokens).sum::<usize>();
                    let reused_tokens = results
                        .iter()
                        .map(|r| r.reused_prefix_tokens)
                        .sum::<usize>();
                    outcomes.push((
                        shared,
                        results,
                        elapsed_ms,
                        timings,
                        input_tokens,
                        reused_tokens,
                    ));
                }
                let fresh = outcomes.iter().find(|entry| !entry.0).unwrap();
                let shared = outcomes.iter().find(|entry| entry.0).unwrap();
                let delta = difference(&fresh.1, &shared.1);
                for (is_shared, _, elapsed_ms, timings, input_tokens, reused_tokens) in outcomes {
                    records.push(
                        json!({"state_filler_words": filler_words, "questions": count,
                        "round": round, "mode": if is_shared { "shared_session" } else { "fresh" },
                        "elapsed_ms": elapsed_ms, "input_tokens": input_tokens,
                        "reused_prefix_tokens": reused_tokens, "timings": timings,
                        "difference": delta}),
                    );
                }
                eprintln!("filler={filler_words}, questions={count}, round={round} done");
            }
        }
    }
    serde_json::to_writer_pretty(
        output,
        &json!({"schema_version": 1, "model": args.model, "identity": identity,
            "load_ms_excluded": load_ms,
            "configuration": {"context": args.context, "batch": args.batch,
                "threads": args.threads, "cuda": args.cuda, "repeats": args.repeats},
            "methodology": {
                "workload": "Synthetic warehouse state; each question has a distinct instruction and ID",
                "comparison": "One question per call in both modes; same state-first prompt",
                "warmup": "One untimed run immediately before each timed mode; order alternates by round",
                "residency": "One resident model; loading and model hash excluded",
                "scope": "Explicit in-process session, not an automatic cross-request cache; no HTTP, parsing, cache lookup or eviction cost. No labeled quality claims."
            }, "records": records}),
    )?;
    Ok(())
}
