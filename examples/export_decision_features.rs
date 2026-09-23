//! Export frozen deployment features and the corresponding base decisions.
#[cfg(not(feature = "llama"))]
fn main() {
    panic!("Enable --features llama");
}
#[cfg(feature = "llama")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use clap::Parser;
    use l2s1::{DecisionPolicy, DecisionRequest, llama::LlamaBackend};
    use std::{
        fs::{File, OpenOptions},
        io::{BufRead, BufReader, BufWriter, Write},
        path::PathBuf,
        time::Instant,
    };
    #[derive(Parser)]
    struct Args {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        cuda: bool,
    }
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    struct Case {
        id: String,
        request: DecisionRequest,
    }
    let args = Args::parse();
    let mut backend = LlamaBackend::load(
        &args.model,
        2048,
        256,
        4,
        args.cuda,
        DecisionPolicy::default(),
    )?;
    let mut output = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(args.output)?,
    );
    let mut ids = std::collections::HashSet::new();
    for (i, line) in BufReader::new(File::open(args.input)?).lines().enumerate() {
        let case: Case = serde_json::from_str(&line?)?;
        if case.id.is_empty() || !ids.insert(case.id.clone()) {
            return Err("duplicate/empty case ID".into());
        }
        let start = Instant::now();
        let (hidden, response) = backend.extract_features(&case.request)?;
        writeln!(
            output,
            "{}",
            serde_json::json!({"id":case.id,"hidden":hidden,"response":response,"elapsed_ms":start.elapsed().as_secs_f64()*1000.0})
        )?;
        output.flush()?;
        if (i + 1) % 100 == 0 {
            println!("Exported {}", i + 1);
        }
    }
    Ok(())
}
