use anyhow::{Context, Result, bail};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read, Write},
    path::Path,
    process::Command,
};

pub fn sha256(path: &Path) -> Result<String> {
    let mut source = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut block = [0_u8; 1024 * 1024];
    loop {
        let count = source.read(&mut block)?;
        if count == 0 {
            break;
        }
        hasher.update(&block[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn read_json(path: &Path) -> Result<Value> {
    serde_json::from_reader(File::open(path).with_context(|| format!("open {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

pub fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(File::open(path).with_context(|| format!("open {}", path.display()))?)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            other => Some((index, other)),
        })
        .map(|(index, line)| {
            let line = line?;
            serde_json::from_str(&line)
                .with_context(|| format!("parse {}:{}", path.display(), index + 1))
        })
        .collect()
}

pub fn write_json(path: &Path, value: &Value) -> Result<()> {
    let temp = path.with_extension("json.tmp");
    let mut file = File::create(&temp).with_context(|| format!("create {}", temp.display()))?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.flush()?;
    fs::rename(temp, path)?;
    Ok(())
}

pub fn write_jsonl(path: &Path, rows: &[Value]) -> Result<()> {
    if path.exists() {
        bail!("output already exists: {}", path.display());
    }
    let mut file = File::create(path)?;
    for row in rows {
        serde_json::to_writer(&mut file, row)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

pub fn git_revision(path: &Path) -> Option<String> {
    let output = Command::new("git")
        .args(["-C", path.to_str()?, "rev-parse", "HEAD"])
        .output()
        .ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
}

pub fn as_str<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {key}"))
}

pub fn percentage(value: Option<f64>, decimals: usize) -> String {
    value.map_or_else(
        || "n/a".to_owned(),
        |number| format!("{:.*}%", decimals, number * 100.0),
    )
}

pub fn nearest_rank(values: &[f64], fraction: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let index = ((ordered.len() as f64 * fraction).ceil() as usize).saturating_sub(1);
    ordered.get(index).copied()
}
