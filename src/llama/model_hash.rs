//! Cache the expensive GGUF digest across process runs. Model identity still
//! contains a SHA-256 digest; the cache is only a way to avoid rereading an
//! unchanged multi-gigabyte file on every load.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::UNIX_EPOCH,
};

static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct FileIdentity {
    path: String,
    size: u64,
    modified_ns: u128,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    changed_sec: i64,
    #[cfg(unix)]
    changed_nsec: i64,
}

impl FileIdentity {
    fn read(path: &Path) -> Option<Self> {
        let metadata = fs::metadata(path).ok()?;
        let modified_ns = metadata
            .modified()
            .ok()?
            .duration_since(UNIX_EPOCH)
            .ok()?
            .as_nanos();
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            path: path.to_str()?.to_owned(),
            size: metadata.len(),
            modified_ns,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            changed_sec: metadata.ctime(),
            #[cfg(unix)]
            changed_nsec: metadata.ctime_nsec(),
        })
    }
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    version: u8,
    file: FileIdentity,
    sha256: String,
}

fn cache_directory() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("L2S1_MODEL_HASH_CACHE_DIR") {
        let path = PathBuf::from(path);
        return path.is_absolute().then_some(path);
    }
    if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
        let path = PathBuf::from(path);
        if path.is_absolute() {
            return Some(path.join("l2s1/model-hashes"));
        }
    }
    let home = PathBuf::from(std::env::var_os("HOME")?);
    if !home.is_absolute() {
        return None;
    }
    #[cfg(target_os = "macos")]
    return Some(home.join("Library/Caches/l2s1/model-hashes"));
    #[cfg(not(target_os = "macos"))]
    Some(home.join(".cache/l2s1/model-hashes"))
}

fn cache_path(directory: &Path, file: &FileIdentity) -> PathBuf {
    // A digest of the canonical path keeps cache filenames short and avoids
    // turning arbitrary model paths into nested cache paths.
    let path_key = Sha256::digest(file.path.as_bytes());
    directory.join(format!("{:x}.json", path_key))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn write_cache(path: &Path, entry: &CacheEntry) {
    let Some(directory) = path.parent() else {
        return;
    };
    if fs::create_dir_all(directory).is_err() {
        return;
    }
    let sequence = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let temporary = path.with_extension(format!("{}.{}.tmp", std::process::id(), sequence));
    let bytes = match serde_json::to_vec(entry) {
        Ok(bytes) => bytes,
        Err(_) => return,
    };
    // create_new avoids sharing a partial cache file with another process.
    let result = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .and_then(|mut file| {
            use std::io::Write;
            file.write_all(&bytes)?;
            file.sync_all()
        })
        .and_then(|_| fs::rename(&temporary, path));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
}

fn digest_with_cache(path: &Path, directory: Option<&Path>) -> Result<String> {
    let before = FileIdentity::read(path);
    let cache_path = before
        .as_ref()
        .zip(directory)
        .map(|(file, directory)| cache_path(directory, file));
    if let (Some(file), Some(cache_path)) = (&before, &cache_path) {
        if let Ok(bytes) = fs::read(cache_path) {
            if let Ok(entry) = serde_json::from_slice::<CacheEntry>(&bytes) {
                if entry.version == 1
                    && entry.file == *file
                    && valid_digest(&entry.sha256)
                    && FileIdentity::read(path).as_ref() == Some(file)
                {
                    return Ok(entry.sha256);
                }
            }
        }
    }

    let sha256 = crate::interoperability::file_digest(path)?;
    // A file changed while hashing cannot provide a reliable digest for its
    // final state. Reject the result instead of recording a misleading model
    // identity or poisoning future cache hits.
    if before != FileIdentity::read(path) {
        return Err(Error::Backend(
            "model file changed while computing SHA-256".into(),
        ));
    }
    if let (Some(file), Some(cache_path)) = (before, cache_path) {
        write_cache(
            &cache_path,
            &CacheEntry {
                version: 1,
                file,
                sha256: sha256.clone(),
            },
        );
    }
    Ok(sha256)
}

pub(super) fn model_digest(path: &Path) -> Result<String> {
    let directory = cache_directory();
    digest_with_cache(path, directory.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_and_invalidate_digest() {
        let dir = std::env::temp_dir().join(format!(
            "l2s1-model-hash-test-{}-{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("model.gguf");
        let cache = dir.join("cache");
        fs::write(&model, b"first GGUF").unwrap();
        let original = digest_with_cache(&model, Some(&cache)).unwrap();
        assert_eq!(
            original,
            crate::interoperability::file_digest(&model).unwrap()
        );

        let identity = FileIdentity::read(&model).unwrap();
        let cache_file = cache_path(&cache, &identity);
        let sentinel = "a".repeat(64);
        write_cache(
            &cache_file,
            &CacheEntry {
                version: 1,
                file: identity,
                sha256: sentinel.clone(),
            },
        );
        assert_eq!(digest_with_cache(&model, Some(&cache)).unwrap(), sentinel);

        fs::write(&model, b"second and longer GGUF").unwrap();
        let changed = digest_with_cache(&model, Some(&cache)).unwrap();
        assert_eq!(
            changed,
            crate::interoperability::file_digest(&model).unwrap()
        );
        assert_ne!(changed, sentinel);

        // A read-only or otherwise unusable cache must not prevent loading.
        let unusable_cache = dir.join("not-a-directory");
        fs::write(&unusable_cache, b"occupied").unwrap();
        assert_eq!(
            digest_with_cache(&model, Some(&unusable_cache)).unwrap(),
            changed
        );

        // Replacing a file at the same path and size changes its Unix inode.
        #[cfg(unix)]
        {
            let replacement = dir.join("replacement.gguf");
            fs::write(&replacement, b"third and longer  GGUF").unwrap();
            assert_eq!(
                fs::metadata(&replacement).unwrap().len(),
                fs::metadata(&model).unwrap().len()
            );
            fs::rename(&replacement, &model).unwrap();
            let replaced = digest_with_cache(&model, Some(&cache)).unwrap();
            assert_eq!(
                replaced,
                crate::interoperability::file_digest(&model).unwrap()
            );
            assert_ne!(replaced, changed);
        }
        let _ = fs::remove_dir_all(dir);
    }
}
