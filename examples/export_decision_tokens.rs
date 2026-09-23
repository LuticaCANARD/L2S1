//! Export production-tokenized inputs; labels remain in a separate manifest.
#[cfg(not(feature = "llama"))]
fn main() {
    panic!("Enable --features llama");
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{DecisionPolicy, DecisionRequest, PromptProfile, llama::LlamaBackend};
    use serde::Deserialize;
    use std::{
        fs::{File, OpenOptions},
        io::{BufRead, BufReader, BufWriter, Write},
        path::PathBuf,
    };
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    }
    #[derive(Deserialize)]
    struct Case {
        id: String,
        request: DecisionRequest,
    }
    let args = Args::parse();
    let backend = LlamaBackend::load_with_profile(
        &args.model,
        2048,
        256,
        4,
        false,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )?;
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.output)?,
    );
    for line in BufReader::new(File::open(args.input)?).lines() {
        let case: Case = serde_json::from_str(&line?)?;
        case.request.validate()?;
        if case.request.decisions.len() != 1 {
            return Err("Expected one decision".into());
        }
        let decision = &case.request.decisions[0];
        let (input_ids, candidate_ids) = backend.encode_decision(&case.request.state, decision)?;
        let option_ids: Vec<_> = decision.options().into_iter().map(|o| o.id).collect();
        writeln!(
            output,
            "{}",
            serde_json::json!({"id":case.id,"input_ids":input_ids,"candidate_ids":candidate_ids,"option_ids":option_ids})
        )?;
    }
    Ok(())
}
