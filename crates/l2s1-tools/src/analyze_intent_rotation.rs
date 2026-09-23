use crate::{
    common::{read_jsonl, sha256, write_json},
    evaluate_intents_wide::code,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::PathBuf,
};

#[derive(ClapArgs)]
pub struct Args {
    root: PathBuf,
}

fn picked(result: &Value) -> Result<&Value> {
    result["scores"]
        .as_array()
        .context("missing scores")?
        .iter()
        .min_by(|a, b| {
            b["option_probability"]
                .as_f64()
                .unwrap_or(f64::NAN)
                .total_cmp(&a["option_probability"].as_f64().unwrap_or(f64::NAN))
                .then_with(|| {
                    a["id"]
                        .as_str()
                        .unwrap_or("")
                        .cmp(b["id"].as_str().unwrap_or(""))
                })
        })
        .context("empty scores")
}

pub fn run(args: Args) -> Result<()> {
    let root = args.root;
    let out = root.join("diagnostics");
    let gold_rows = read_jsonl(&root.join("prepared/gold.jsonl"))?
        .into_iter()
        .filter(|r| r["dataset"] == "banking77-en")
        .collect::<Vec<_>>();
    let gold = gold_rows
        .iter()
        .map(|r| Ok((r["id"].as_str().context("gold ID")?.to_owned(), r)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let before = read_jsonl(&root.join("runs/Qwen3-8B-Q8_0/predictions.jsonl"))?
        .into_iter()
        .filter(|r| gold.contains_key(r["id"].as_str().unwrap_or("")))
        .map(|r| Ok((r["id"].as_str().context("prediction ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    let after_rows = read_jsonl(&out.join("qwen-rotation38.jsonl"))?;
    let after = after_rows
        .iter()
        .map(|r| Ok((r["id"].as_str().context("rotation ID")?.to_owned(), r)))
        .collect::<Result<HashMap<_, _>>>()?;
    ensure!(
        gold.len() == 200 && before.len() == 200 && after.len() == 200 && after_rows.len() == 200,
        "paired coverage differs"
    );
    ensure!(
        gold.keys().collect::<HashSet<_>>() == before.keys().collect()
            && gold.keys().collect::<HashSet<_>>() == after.keys().collect(),
        "paired IDs differ"
    );
    let requests = read_jsonl(&root.join("prepared/requests.jsonl"))?;
    let expected = requests
        .into_iter()
        .filter(|r| gold.contains_key(r["id"].as_str().unwrap_or("")))
        .collect::<Vec<_>>();
    ensure!(
        read_jsonl(&out.join("banking77-requests.jsonl"))? == expected,
        "diagnostic requests differ"
    );
    ensure!(
        fs::read_to_string(out.join("rotation38.exit-code"))?.trim() == "0",
        "rotation run failed"
    );
    let mut counts = BTreeMap::<&str, usize>::new();
    let mut lengths = BTreeMap::<String, BTreeMap<&str, usize>>::new();
    let mut pairs = Vec::new();
    for gold in &gold_rows {
        let id = gold["id"].as_str().context("gold ID")?;
        let p = &before[id];
        let q = after[id];
        ensure!(
            p.get("error").is_none() && q.get("error").is_none(),
            "runtime error"
        );
        let a = &p["response"]["results"][0];
        let b = &q["response"]["results"][0];
        ensure!(
            a["truncated"] == false && b["truncated"] == false,
            "truncated result"
        );
        let old = &p["response"]["backend"];
        let new = &q["response"]["backend"];
        ensure!(
            old["code_rotation"].as_u64().unwrap_or(0) == 0 && new["code_rotation"] == 38,
            "rotation setting changed"
        );
        for field in [
            "model_path",
            "compute",
            "offload_device",
            "execution_mode",
            "prompt_layout",
            "prompt_profile",
        ] {
            ensure!(old[field] == new[field], "backend setting changed: {field}");
        }
        let aa = a["scores"].as_array().context("before scores")?;
        let bb = b["scores"].as_array().context("after scores")?;
        ensure!(
            aa.len() == 77
                && bb.len() == 77
                && aa.iter().map(|s| &s["id"]).eq(bb.iter().map(|s| &s["id"])),
            "candidate set changed"
        );
        for (i, score) in bb.iter().enumerate() {
            ensure!(
                score["code"] == code((i + 77 - 38) % 77, 77),
                "rotated code differs"
            );
        }
        let x = picked(a)?;
        let y = picked(b)?;
        let expected = gold["expected"].as_str().context("gold label")?;
        let correct_a = x["id"] == expected;
        let correct_b = y["id"] == expected;
        for (key, value) in [
            ("before_correct", correct_a),
            ("after_correct", correct_b),
            ("prediction_changed", x["id"] != y["id"]),
            ("correct_to_wrong", correct_a && !correct_b),
            ("wrong_to_correct", !correct_a && correct_b),
            ("wrong_both", !correct_a && !correct_b),
        ] {
            *counts.entry(key).or_default() += usize::from(value);
        }
        for (label, chosen, correct) in [("before", x, correct_a), ("after", y, correct_b)] {
            if chosen["option_probability"]
                .as_f64()
                .context("probability")?
                >= 0.9
            {
                *counts
                    .entry(if label == "before" {
                        "before_confidence90_count"
                    } else {
                        "after_confidence90_count"
                    })
                    .or_default() += 1;
                *counts
                    .entry(if label == "before" {
                        "before_confidence90_wrong"
                    } else {
                        "after_confidence90_wrong"
                    })
                    .or_default() += usize::from(!correct);
            }
        }
        let target_a = aa
            .iter()
            .find(|s| s["id"] == expected)
            .context("gold candidate absent")?;
        let target_b = bb
            .iter()
            .find(|s| s["id"] == expected)
            .context("gold candidate absent")?;
        let first = target_a["token_ids"].as_array().context("token IDs")?.len();
        let second = target_b["token_ids"].as_array().context("token IDs")?.len();
        let lengths_entry = lengths.entry(format!("{first}->{second}")).or_default();
        for (key, value) in [
            ("n", true),
            ("before_correct", correct_a),
            ("after_correct", correct_b),
        ] {
            *lengths_entry.entry(key).or_default() += usize::from(value);
        }
        pairs.push(json!({"id":id,"expected":expected,"before":x["id"],"after":y["id"],"before_correct":correct_a,"after_correct":correct_b,
            "before_code":x["code"],"after_code":y["code"],"gold_code_before":target_a["code"],"gold_code_after":target_b["code"],
            "gold_token_lengths":[first,second]}));
    }
    let summary = json!({"dataset":"BANKING77 English","n":200,"rotation":38,"counts":counts,"gold_token_length_transitions":lengths,
        "requests_sha256":sha256(&out.join("banking77-requests.jsonl"))?,"predictions_sha256":sha256(&out.join("qwen-rotation38.jsonl"))?,
        "scope":"Exploratory paired diagnostic; same text, candidate semantics and model. Display order and code assignment change together. Not a tuned replacement benchmark score."});
    write_json(&out.join("rotation-analysis.json"), &summary)?;
    write_json(&out.join("paired-predictions.json"), &json!(pairs))?;
    let c = &summary["counts"];
    let lines=vec!["# Intent classification diagnosis".to_owned(),String::new(),
        "Qwen3-8B Q8_0 on the same frozen 200 English BANKING77 requests; the diagnostic rotates displayed candidate order and answer-code assignment together.".to_owned(),String::new(),
        format!("Original correct: {}/200. Rotated correct: {}/200. Changed predictions: {}/200. Correct to wrong: {}; wrong to correct: {}.",
            c["before_correct"],c["after_correct"],c["prediction_changed"],c["correct_to_wrong"],c["wrong_to_correct"]),String::new(),
        format!("At rotated confidence >=90%: {} wrong among {} predictions.",c["after_confidence90_wrong"],c["after_confidence90_count"]),String::new(),
        "This measures combined order and code sensitivity on an existing test subset. It does not isolate tokenizer length, establish a better prompt, or replace the original benchmark score.".to_owned(),String::new(),
        "`rotation-analysis.json` and `paired-predictions.json` retain all paired outcomes and token-length transitions.".to_owned(),String::new()];
    fs::write(out.join("REPORT.md"), lines.join("\n"))?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
