use anyhow::{Context, Result};
use parquet::file::reader::{FileReader, SerializedFileReader};
use serde_json::Value;
use std::{fs::File, path::Path};

pub fn rows(path: &Path) -> Result<Vec<Value>> {
    let reader = SerializedFileReader::new(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    )?;
    reader
        .get_row_iter(None)?
        .map(|row| Ok(row?.to_json_value()))
        .collect()
}

pub fn metadata(path: &Path, key: &str) -> Result<Value> {
    let reader = SerializedFileReader::new(
        File::open(path).with_context(|| format!("open {}", path.display()))?,
    )?;
    let values = reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .context("Parquet key/value metadata absent")?;
    let value = values
        .iter()
        .find(|item| item.key == key)
        .and_then(|item| item.value.as_deref())
        .with_context(|| format!("Parquet metadata {key} absent"))?;
    Ok(serde_json::from_str(value)?)
}
