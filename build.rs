use sha2::{Digest, Sha256};
use std::{env, fs::File, io::Read, path::PathBuf};

fn main() {
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }
    let mut fingerprint = Sha256::new();
    fingerprint.update(
        env::var("DEP_L2S1_LLAMA_RUNTIME_SHA256")
            .expect("l2s1-llama-sys runtime identity unavailable")
            .as_bytes(),
    );
    for path in [
        "src/decision.rs",
        "src/codes.rs",
        "src/llama/code_sequences.rs",
        "src/evidence.rs",
        "src/prompt.rs",
        "src/llama.rs",
        "src/llama/interchange.rs",
        "src/llama/model_hash.rs",
        "src/llama/prepared_cache.rs",
        "src/llama/shared_state.rs",
        "src/llama/batching.rs",
        "src/optimization.rs",
        "src/worker.rs",
        "src/consensus.rs",
        "src/calibration.rs",
        "src/output_head.rs",
        "src/interoperability.rs",
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={path}");
        let mut file = File::open(path).expect("runtime identity input unavailable");
        // Keep build scripts below the Windows main thread's default stack limit.
        let mut buffer = [0; 64 * 1024];
        loop {
            let n = file.read(&mut buffer).expect("read runtime identity");
            if n == 0 {
                break;
            }
            fingerprint.update(&buffer[..n]);
        }
    }
    for key in ["TARGET", "PROFILE", "OPT_LEVEL", "CARGO_ENCODED_RUSTFLAGS"] {
        fingerprint.update(key.as_bytes());
        fingerprint.update(env::var(key).unwrap_or_default().as_bytes());
    }
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = std::process::Command::new(rustc)
        .arg("-vV")
        .output()
        .expect("read Rust compiler identity");
    assert!(
        version.status.success(),
        "Rust compiler identity unavailable"
    );
    fingerprint.update(version.stdout);
    println!(
        "cargo:rustc-env=L2S1_RUNTIME_BUILD_SHA256={:x}",
        fingerprint.finalize()
    );

    if matches!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("linux" | "macos")
    ) {
        let libdir = PathBuf::from(
            env::var_os("DEP_L2S1_LLAMA_LIBDIR")
                .expect("l2s1-llama-sys library directory unavailable"),
        );
        println!("cargo::metadata=libdir={}", libdir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", libdir.display());
    }
}
