//! Export production-tokenized inputs; labels remain in a separate manifest.
#[cfg(not(feature = "llama"))]
fn main() {
    panic!("Enable --features llama");
}

#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{
        DecisionPolicy, DecisionRequest, PromptDetail, PromptLayout, PromptProfile,
        llama::LlamaBackend,
    };
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
        #[arg(long, value_enum, default_value_t = PromptDetail::Minimal)]
        prompt_detail: PromptDetail,
        #[arg(long, value_enum, default_value_t = PromptLayout::Legacy)]
        prompt_layout: PromptLayout,
        /// Export every cyclic code assignment, preserving the logical case ID.
        #[arg(long)]
        all_rotations: bool,
    }
    #[derive(Deserialize)]
    struct Case {
        id: String,
        request: DecisionRequest,
    }
    let args = Args::parse();
    let mut backend = LlamaBackend::load_with_profile(
        &args.model,
        2048,
        256,
        4,
        false,
        DecisionPolicy::default(),
        PromptProfile::Auto,
    )?;
    backend.set_prompt_layout(args.prompt_layout);
    backend.set_prompt_detail(args.prompt_detail);
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
        let option_ids: Vec<_> = decision.options().into_iter().map(|o| o.id).collect();
        let count = option_ids.len();
        for rotation in 0..if args.all_rotations { count } else { 1 } {
            backend.set_code_rotation(rotation)?;
            let (input_ids, candidate_ids) =
                backend.encode_decision(&case.request.state, decision)?;
            let candidate_codes: Vec<_> = (0..count)
                .map(|i| {
                    l2s1::option_code((i + count - rotation) % count, count)
                        .expect("validated options")
                })
                .collect();
            writeln!(
                output,
                "{}",
                serde_json::json!({"id":case.id,"decision_id":decision.id,"code_rotation":rotation,
                "input_ids":input_ids,"candidate_ids":candidate_ids,"candidate_codes":candidate_codes,
                "option_ids":option_ids,"model_identity":backend.identity()})
            )?;
        }
    }
    Ok(())
}
