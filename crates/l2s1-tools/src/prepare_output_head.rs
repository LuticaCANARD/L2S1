use crate::{
    common::{read_jsonl, sha256, write_json},
    kaggle_airline::{ARCHIVE_SHA256, LABELS, Row, available, case_json, identity, read_archive},
    python_random::PythonRandom,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::json;
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::Write,
    path::PathBuf,
};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    source: PathBuf,
    #[arg(long)]
    previous: PathBuf,
    #[arg(long)]
    output: PathBuf,
}

pub fn run(args: Args) -> Result<()> {
    let mut excluded = HashSet::new();
    for (folder, names) in [
        (&args.source, vec!["fit", "validation"]),
        (&args.previous, vec!["train", "calibration", "test"]),
    ] {
        for name in names {
            for row in read_jsonl(&folder.join(format!("{name}.jsonl")))? {
                excluded.insert(identity(
                    row["request"]["state"]["tweet"]
                        .as_str()
                        .context("missing prior tweet")?,
                ));
            }
        }
    }
    let rows = read_archive(&args.source.join("dataset.zip"))?;
    let (groups, _) = available(&rows, &excluded)?;
    let mut rng = PythonRandom::new(20260924);
    let mut splits: BTreeMap<&str, Vec<(usize, Row)>> = [
        ("train", Vec::new()),
        ("dev", Vec::new()),
        ("calibration", Vec::new()),
        ("test", Vec::new()),
    ]
    .into();
    for (group, n) in groups.iter().zip([134, 133, 133]) {
        ensure!(group.len() >= 400 + 2 * n, "too few eligible airline rows");
        let sampled = rng.sample_indices(group.len(), 400 + 2 * n);
        for (name, lo, hi) in [
            ("train", 0, 300),
            ("dev", 300, 400),
            ("calibration", 400, 400 + n),
            ("test", 400 + n, 400 + 2 * n),
        ] {
            splits
                .get_mut(name)
                .unwrap()
                .extend(sampled[lo..hi].iter().map(|i| group[*i].clone()));
        }
    }
    fs::create_dir(&args.output)?;
    let mut labels = serde_json::Map::new();
    let mut meta = serde_json::Map::new();
    for name in ["train", "dev", "calibration", "test"] {
        let selected = splits.get_mut(name).unwrap();
        rng.shuffle(selected);
        let path = args.output.join(format!("{name}.jsonl"));
        let mut file = File::create(&path)?;
        let mut ids = Vec::new();
        for rotation in 0..if name == "train" { 3 } else { 1 } {
            for (index, row) in selected.iter() {
                let id = format!("airline-row-{index:05}-r{rotation}");
                writeln!(file, "{}", case_json(&id, &row.text, rotation)?)?;
                labels.insert(id.clone(), json!(row.sentiment));
                ids.push(id);
            }
        }
        file.flush()?;
        meta.insert(name.to_owned(), json!({"ids":ids,"sha256":sha256(&path)?}));
    }
    let mut probe = File::create(args.output.join("probe.jsonl"))?;
    let mut probe_ids = Vec::new();
    for label in LABELS {
        for (index, row) in splits["test"]
            .iter()
            .filter(|(_, r)| r.sentiment == label)
            .take(20)
        {
            for rotation in 0..3 {
                let id = format!("airline-row-{index:05}-probe-r{rotation}");
                writeln!(probe, "{}", case_json(&id, &row.text, rotation)?)?;
                labels.insert(id.clone(), json!(label));
                probe_ids.push(id);
            }
        }
    }
    probe.flush()?;
    meta.insert(
        "probe".to_owned(),
        json!({"ids":probe_ids,"sha256":sha256(&args.output.join("probe.jsonl"))?}),
    );
    let unique = |name: &str| {
        splits[name]
            .iter()
            .map(|(_, r)| identity(&r.text))
            .collect::<HashSet<_>>()
    };
    for (i, a) in ["train", "dev", "calibration", "test"].iter().enumerate() {
        for b in ["train", "dev", "calibration", "test"].iter().skip(i + 1) {
            ensure!(unique(a).is_disjoint(&unique(b)), "split texts overlap");
        }
    }
    let manifest = json!({"seed":20260924,"source":"https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment",
        "archive_sha256":ARCHIVE_SHA256,"excluded_prior_unique":excluded.len(),"labels":labels,"splits":meta,
        "protocol":{"model":"gemma-4-E2B-it-Q8_0.gguf","model_sha256":"996d08777aadc6bfd3c7375ef70ba25a0f55240075860754fdb18d6d860aa63a",
            "methods":["base_temperature","logit_affine","hidden"],"l2_grid":[0.001,0.01,0.1,1.0],
            "objective":"mean cross entropy + L2/2 * squared standardized weights; CPU float64 LBFGS max_iter=200",
            "standardization":"train only; std clamp 1e-6; fold into exported weights",
            "hyperparameter_selection":"lowest dev NLL before temperature; never test",
            "calibration":"temperature fit only on independent 400 calibration cases",
            "deployment_selection":"lowest dev NLL across trained heads; explicit opt-in",
            "scope":"Balanced task-specific head. Normalized exact dedup, not semantic dedup. Not RLCD. Base candidate mass gate preserved."}});
    write_json(&args.output.join("selection.json"), &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest["splits"])?);
    Ok(())
}
