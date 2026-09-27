//! Resident diagnostic harness for controlled fresh/state-cache comparisons.
#[cfg(not(feature = "llama"))]
fn main() {
    panic!("state_cache_probe requires --features llama");
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{DecisionPolicy, DecisionRequest, ExecutionMode, PromptLayout, llama::LlamaBackend};
    use serde::Deserialize;
    use serde_json::json;
    use std::io::{self, BufRead, Write};

    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: std::path::PathBuf,
        #[arg(long, default_value_t = 256)]
        batch: u32,
    }
    #[derive(Deserialize)]
    struct Call {
        request: DecisionRequest,
        mode: ExecutionMode,
        layout: PromptLayout,
        #[serde(default)]
        shared_session: bool,
        #[serde(default)]
        snapshot_limit: Option<usize>,
    }
    let args = Args::parse();
    let mut backend = LlamaBackend::load(
        &args.model,
        2048,
        args.batch,
        4,
        false,
        DecisionPolicy::default(),
    )?;
    let mut output = io::stdout().lock();
    serde_json::to_writer(&mut output, &json!({"inspection": backend.inspect()}))?;
    writeln!(output)?;
    output.flush()?;
    for line in io::stdin().lock().lines() {
        let call: Call = serde_json::from_str(&line?)?;
        backend.set_execution_mode(call.mode);
        backend.set_prompt_layout(call.layout);
        backend.set_snapshot_limit_bytes(call.snapshot_limit.unwrap_or(256 * 1024 * 1024));
        let result = if call.shared_session {
            // Separate one-question calls inside one immutable-state session.
            // Preflight matches the diagnostic work done by decide_detailed.
            let preflight = backend.preflight(&call.request)?;
            backend.take_timings();
            let mut session = backend.shared_state(call.request.state.clone())?;
            let mut results = Vec::new();
            for decision in call.request.decisions {
                results.extend(session.decide(vec![decision])?.results);
            }
            json!({"response": {"results": results}, "timings": session.take_timings(),
                   "model": preflight.model.identity, "state_restore": null})
        } else {
            serde_json::to_value(backend.decide_detailed(&call.request))?
        };
        serde_json::to_writer(&mut output, &result)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}
