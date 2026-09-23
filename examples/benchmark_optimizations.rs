//! Reproducible local execution comparison; synthetic workload, not accuracy data.
#[cfg(not(feature = "llama"))]
fn main() {
    eprintln!("Enable --features llama to benchmark local models");
    std::process::exit(1);
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        DecisionBackend, DecisionPolicy, DecisionRequest, DecisionResponse, DecisionValue,
        EvidenceTransfer, ExecutionMode, PreparationCacheConfig, PromptLayout, llama::LlamaBackend,
    };
    use serde_json::{Value, json};
    use std::{fs::OpenOptions, io::Write, path::PathBuf, time::Instant};

    #[derive(Parser)]
    struct Args {
        #[arg(long, required = true)]
        model: Vec<PathBuf>,
        /// Refuse to overwrite an existing report.
        #[arg(long)]
        output: PathBuf,
        #[arg(long, default_value_t = 2048)]
        context: u32,
        #[arg(long, default_value_t = 32)]
        batch: u32,
        #[arg(long, default_value_t = 4)]
        threads: i32,
        #[arg(long)]
        cuda: bool,
        #[arg(long, default_value_t = 3, value_parser = clap::value_parser!(u32).range(1..))]
        repeats: u32,
    }

    const PATHS: [&str; 5] = [
        "legacy_fresh_full",
        "legacy_fresh_cached",
        "legacy_fresh_compact",
        "state_first_fresh",
        "state_first_shared_session",
    ];

    fn configure(backend: &mut LlamaBackend, path: usize) -> l2s1::Result<()> {
        backend.set_evidence_transfer(EvidenceTransfer::Full)?;
        backend.set_execution_mode(if path == 4 {
            ExecutionMode::PrefixReuse
        } else {
            ExecutionMode::Fresh
        });
        backend.set_prompt_layout(if path >= 3 {
            PromptLayout::StateFirst
        } else {
            PromptLayout::Legacy
        });
        backend.set_preparation_cache(PreparationCacheConfig {
            max_entries: if path == 1 { 128 } else { 0 },
            max_bytes: if path == 1 { 8 * 1024 * 1024 } else { 0 },
        });
        backend.set_evidence_transfer(if path == 2 {
            EvidenceTransfer::Compact
        } else {
            EvidenceTransfer::Full
        })
    }

    fn run(
        backend: &mut LlamaBackend,
        request: &DecisionRequest,
        path: usize,
    ) -> l2s1::Result<DecisionResponse> {
        if path != 4 {
            return backend.decide(request);
        }
        let mut session = backend.shared_state(request.state.clone())?;
        let mut response = session.decide(vec![request.decisions[0].clone()])?;
        for decision in &request.decisions[1..] {
            response
                .results
                .extend(session.decide(vec![decision.clone()])?.results);
        }
        Ok(response)
    }

    fn selected(value: &DecisionValue) -> Value {
        match value {
            DecisionValue::Binary { value, .. } => json!(value),
            DecisionValue::Choice { selected } | DecisionValue::Ordinal { selected, .. } => {
                json!(selected)
            }
        }
    }

    fn compare(actual: &DecisionResponse, reference: &DecisionResponse) -> Value {
        assert_eq!(actual.results.len(), reference.results.len());
        let (mut probability_delta, mut mass_delta, mut logit_delta) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut changed_selections = 0;
        let mut changed_raw_top1 = 0;
        for (a, b) in actual.results.iter().zip(&reference.results) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.scores.len(), b.scores.len());
            mass_delta = mass_delta.max((a.candidate_mass - b.candidate_mass).abs());
            changed_selections += usize::from(selected(&a.value) != selected(&b.value));
            let top1 = |scores: &[l2s1::OptionScore]| {
                scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.option_probability.total_cmp(&b.option_probability))
                    .map(|(index, _)| index)
            };
            changed_raw_top1 += usize::from(top1(&a.scores) != top1(&b.scores));
            for (a, b) in a.scores.iter().zip(&b.scores) {
                assert_eq!(a.id, b.id);
                probability_delta =
                    probability_delta.max((a.option_probability - b.option_probability).abs());
                logit_delta = logit_delta.max((a.raw_logit - b.raw_logit).abs());
            }
        }
        json!({"max_probability_delta": probability_delta, "max_candidate_mass_delta": mass_delta,
            "max_candidate_logit_delta": logit_delta, "changed_selections": changed_selections,
            "changed_raw_top1": changed_raw_top1})
    }

    let args = Args::parse();
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output)?;
    let fixture: DecisionRequest = serde_json::from_str(include_str!("warehouse.json"))?;
    let mut records = Vec::new();
    for model in &args.model {
        let load_start = Instant::now();
        let mut backend = LlamaBackend::load(
            model,
            args.context,
            args.batch,
            args.threads,
            args.cuda,
            DecisionPolicy::default(),
        )?;
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        for long_state in [false, true] {
            for question_count in [1, 4, 16] {
                let mut state = fixture.state.clone();
                if long_state {
                    state["irrelevant_packing_notes"] = json!("packing ".repeat(100));
                }
                let decisions = (0..question_count)
                    .map(|index| {
                        let mut decision =
                            fixture.decisions[index % fixture.decisions.len()].clone();
                        decision.id = format!("{}_{}", decision.id, index);
                        decision
                    })
                    .collect();
                let request = DecisionRequest { state, decisions };
                request.validate()?;
                for round in 0..args.repeats {
                    let mut measured = Vec::new();
                    for position in 0..PATHS.len() {
                        let path = (position + round as usize) % PATHS.len();
                        configure(&mut backend, path)?;
                        run(&mut backend, &request, path)?; // Untimed warmup after configuration.
                        backend.take_timings();
                        let cache_before = backend.preparation_cache_stats();
                        let identity = backend.identity();
                        let started = Instant::now();
                        let response = run(&mut backend, &request, path)?;
                        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                        let record = json!({
                            "model": model, "load_ms_excluded": load_ms, "round": round,
                            "execution_position": position, "path": PATHS[path], "identity": identity,
                            "state": if long_state { "long_100_filler_words" } else { "short" },
                            "questions": question_count, "elapsed_ms": elapsed_ms,
                            "amortized_ms_per_question": elapsed_ms / question_count as f64,
                            "timings": backend.take_timings(),
                            "cache_before": cache_before, "cache_after": backend.preparation_cache_stats(),
                            "total_input_tokens": response.results.iter().map(|r| r.input_tokens).sum::<usize>(),
                            "total_reused_tokens": response.results.iter().map(|r| r.reused_prefix_tokens).sum::<usize>()
                        });
                        measured.push((path, response, record));
                    }
                    for (path, response, record) in &measured {
                        let reference_path = if *path >= 3 { 3 } else { 0 };
                        let reference = &measured
                            .iter()
                            .find(|(p, _, _)| *p == reference_path)
                            .unwrap()
                            .1;
                        let mut record = record.clone();
                        record["comparison_reference"] = json!(PATHS[reference_path]);
                        record["difference"] = compare(response, reference);
                        record["response"] = serde_json::to_value(response)?;
                        records.push(record);
                    }
                    eprintln!(
                        "{}: long_state={long_state}, questions={question_count}, round={round} complete",
                        model.display()
                    );
                }
            }
        }
    }
    serde_json::to_writer_pretty(
        &mut output,
        &json!({
            "schema_version": 1,
            "methodology": {
                "workload": "Synthetic warehouse fixture, cycling choice/binary/ordinal questions with unique IDs; no quality labels",
                "residency": "One resident backend per model; model loading and identity hashing excluded from measurements",
                "warmup": "One untimed full run immediately before each path/scenario/round, after cache-invalidating configuration changes",
                "order": "Five execution paths rotated by round index; fresh baselines compared within the same round",
                "session": "One explicit fixed-state session, separate call for every question; creation and drop included in measured time",
                "latency": "elapsed_ms is entire run completion; amortized_ms_per_question is not independent request latency",
                "scope": "Preparation cache gets identical repeated inputs after warmup; changing-schema workload cycles three question kinds. State-first changes prompts and is compared only to state-first Fresh. No speed or accuracy conclusion implied."
            },
            "configuration": {"context": args.context, "batch": args.batch, "threads": args.threads, "cuda": args.cuda, "repeats": args.repeats},
            "records": records
        }),
    )?;
    writeln!(output)?;
    Ok(())
}
