use crate::{
    common::{read_json, read_jsonl, sha256, write_json},
    kaggle_airline::{LABELS, Row, available, case_json, identity, read_archive},
    python_random::PythonRandom,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::json;
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    source: Option<PathBuf>,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    seal_tokens: bool,
}

fn make_splits(groups: &[Vec<(usize, Row)>]) -> Result<[Vec<(usize, Row)>; 3]> {
    let mut rng = PythonRandom::new(20260923);
    let mut splits: [Vec<(usize, Row)>; 3] = std::array::from_fn(|_| Vec::new());
    for (group, heldout) in groups.iter().zip([134, 133, 133]) {
        let length = 300 + heldout * 2;
        ensure!(group.len() >= length, "too few eligible airline rows");
        let chosen = rng.sample_indices(group.len(), length);
        splits[0].extend(chosen[..300].iter().map(|i| group[*i].clone()));
        splits[1].extend(chosen[300..300 + heldout].iter().map(|i| group[*i].clone()));
        splits[2].extend(chosen[300 + heldout..].iter().map(|i| group[*i].clone()));
    }
    for split in &mut splits {
        rng.shuffle(split);
    }
    Ok(splits)
}

fn prepare(source: &Path, output: &Path) -> Result<()> {
    fs::create_dir(output)?;
    let archive = source.join("dataset.zip");
    let prior = ["fit", "validation"]
        .iter()
        .map(|name| read_jsonl(&source.join(format!("{name}.jsonl"))))
        .collect::<Result<Vec<_>>>()?
        .concat();
    let excluded = prior
        .iter()
        .map(|row| {
            row["request"]["state"]["tweet"]
                .as_str()
                .context("prior tweet")
                .map(identity)
        })
        .collect::<Result<HashSet<_>>>()?;
    let rows = read_archive(&archive)?;
    let (groups, _) = available(&rows, &excluded)?;
    let [train, calibration, test] = make_splits(&groups)?;
    let mut labels = serde_json::Map::new();
    let mut split_meta = serde_json::Map::new();
    for (name, selected) in [
        ("train", &train),
        ("calibration", &calibration),
        ("test", &test),
    ] {
        let path = output.join(format!("{name}.jsonl"));
        let mut file = File::create(&path)?;
        let mut ids = Vec::new();
        let mut counts = BTreeMap::new();
        let rotations = if name == "train" { 3 } else { 1 };
        for rotation in 0..rotations {
            for (index, row) in selected {
                let id = format!("airline-row-{index:05}-r{rotation}");
                writeln!(file, "{}", case_json(&id, &row.text, rotation)?)?;
                labels.insert(id.clone(), json!(row.sentiment));
                ids.push(id);
            }
        }
        for (_, row) in selected {
            *counts.entry(row.sentiment.clone()).or_insert(0usize) += 1;
        }
        file.flush()?;
        split_meta.insert(name.to_owned(),json!({"ids":ids,"unique_ids":selected.iter().map(|(i,_)|format!("airline-row-{i:05}")).collect::<Vec<_>>(),
            "sha256":sha256(&path)?,"counts":counts}));
    }
    let mut probe_file = File::create(output.join("probe.jsonl"))?;
    let mut probe_ids = Vec::new();
    for label in LABELS {
        for (index, row) in test.iter().filter(|(_, r)| r.sentiment == label).take(20) {
            for rotation in 0..3 {
                let id = format!("airline-row-{index:05}-probe-r{rotation}");
                writeln!(probe_file, "{}", case_json(&id, &row.text, rotation)?)?;
                labels.insert(id.clone(), json!(label));
                probe_ids.push(id);
            }
        }
    }
    probe_file.flush()?;
    split_meta.insert(
        "probe".to_owned(),
        json!({"ids":probe_ids,"sha256":sha256(&output.join("probe.jsonl"))?,
        "scope":"60 test cases x 3 orders; not 180 independent cases"}),
    );
    let manifest = json!({"seed":20260923,"archive_sha256":crate::kaggle_airline::ARCHIVE_SHA256,
        "excluded_prior_ids":prior.iter().map(|r|r["id"].clone()).collect::<Vec<_>>(),"labels":labels,"splits":split_meta,
        "protocol":{"train_unique":900,"train_rotations":3,"epochs":1,"learning_rate":0.0001,"rank":8,"alpha":16,"dropout":0.0,
            "gradient_accumulation":12,"mass_loss_weight":0.1,"checkpoint_selection":"fixed final epoch; no test-based selection",
            "calibration":"fit temperature only on 400 new calibration cases",
            "scope":"Single-domain supervised pilot, not RLCD reproduction. Balanced own split; exact normalized deduplication only."}});
    write_json(&output.join("selection.json"), &manifest)?;
    println!("{}", serde_json::to_string_pretty(&manifest["splits"])?);
    Ok(())
}

fn seal_tokens(output: &Path) -> Result<()> {
    let path = output.join("selection.json");
    let mut manifest = read_json(&path)?;
    let splits = manifest["splits"]
        .as_object_mut()
        .context("missing splits")?;
    for (name, split) in splits {
        let tokens = output.join(format!("{name}-tokens.jsonl"));
        let rows = read_jsonl(&tokens)?;
        let requests = read_jsonl(&output.join(format!("{name}.jsonl")))?;
        ensure!(
            sha256(&output.join(format!("{name}.jsonl")))?
                == split["sha256"].as_str().context("missing request digest")?,
            "request changed"
        );
        let ids = split["ids"].as_array().context("missing IDs")?;
        ensure!(
            rows.len() == ids.len() && rows.len() == requests.len(),
            "token count changed"
        );
        for ((row, request), id) in rows.iter().zip(&requests).zip(ids) {
            ensure!(row["id"] == *id, "token ID order changed");
            let opts = request["request"]["decisions"][0]["kind"]["options"]
                .as_array()
                .context("missing options")?;
            let option_ids = opts.iter().map(|o| o["id"].clone()).collect::<Vec<_>>();
            ensure!(
                row["option_ids"] == json!(option_ids),
                "option order changed"
            );
            let input = row["input_ids"].as_array().context("missing input_ids")?;
            ensure!(
                !input.is_empty() && input.len() <= 2048,
                "invalid input length"
            );
            let candidates = row["candidate_ids"]
                .as_array()
                .context("missing candidate_ids")?;
            ensure!(
                candidates.len() == opts.len()
                    && candidates.iter().collect::<HashSet<_>>().len() == opts.len(),
                "invalid candidates"
            );
        }
        let digest = sha256(&tokens)?;
        if let Some(old) = split["token_sha256"].as_str() {
            ensure!(old == digest, "refusing to reseal changed tokens");
        }
        split["token_sha256"] = json!(digest);
        split["max_input_tokens"] = json!(
            rows.iter()
                .map(|r| r["input_ids"].as_array().map_or(0, Vec::len))
                .max()
                .unwrap_or(0)
        );
    }
    write_json(&path, &manifest)
}

pub fn run(args: Args) -> Result<()> {
    if args.seal_tokens {
        seal_tokens(&args.output)
    } else {
        prepare(
            args.source
                .as_deref()
                .context("--source required for prepare")?,
            &args.output,
        )
    }
}
