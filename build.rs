use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_LIB_DIR");
    println!("cargo:rerun-if-changed=native/bridge.cpp");
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        return;
    }
    let source = PathBuf::from(env::var_os("LLAMA_CPP_DIR").unwrap_or("../llama.cpp".into()))
        .canonicalize()
        .expect("Set LLAMA_CPP_DIR to the matching llama.cpp source checkout");
    let lib = PathBuf::from(
        env::var_os("LLAMA_LIB_DIR")
            .unwrap_or_else(|| source.join("build-cuda/bin").into_os_string()),
    )
    .canonicalize()
    .expect("Set LLAMA_LIB_DIR to the built llama.cpp shared libraries");
    // Fingerprint the linked build and adapter sources, not a mutable model name.
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut fingerprint = Sha256::new();
    let mut inputs = vec![
        PathBuf::from("native/bridge.cpp"),
        PathBuf::from("native/chat.cpp"),
    ];
    inputs.extend(["libllama.so", "libggml.so", "libggml-base.so"].map(|name| lib.join(name)));
    inputs.push(source.join("include/llama.h"));
    inputs.extend(
        [
            "src/decision.rs",
            "src/evidence.rs",
            "src/prompt.rs",
            "src/llama.rs",
            "src/llama/interchange.rs",
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
        ]
        .map(PathBuf::from),
    );
    for file in [
        "jinja/lexer.cpp",
        "jinja/parser.cpp",
        "jinja/runtime.cpp",
        "jinja/string.cpp",
        "jinja/value.cpp",
        "json.cpp",
        "unicode.cpp",
    ] {
        inputs.push(source.join("common").join(file));
    }
    for path in inputs {
        println!("cargo:rerun-if-changed={}", path.display());
        let mut file = std::fs::File::open(&path).expect("runtime identity input unavailable");
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let n = file.read(&mut buffer).expect("read runtime identity");
            if n == 0 {
                break;
            }
            fingerprint.update(&buffer[..n]);
        }
    }
    let header = source.join("include/llama.h");
    println!("cargo:rerun-if-changed={}", header.display());
    println!(
        "cargo:rerun-if-changed={}",
        source.join("src/llama-ext.h").display()
    );
    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .std("c++17")
        .include(source.join("include"))
        .include(source.join("ggml/include"))
        .include(source.join("common"))
        .include(source.join("src"))
        .include(source.join("vendor"))
        .file("native/bridge.cpp")
        .file("native/chat.cpp");
    println!("cargo:rerun-if-changed=native/chat.cpp");
    // Compile the template engine from the same checkout as the C ABI headers.
    // libllama's convenience formatter recognizes patterns but does not execute
    // GGUF Jinja, which can lose model-specific EOS tokens and whitespace.
    for file in [
        "jinja/lexer.cpp",
        "jinja/parser.cpp",
        "jinja/runtime.cpp",
        "jinja/string.cpp",
        "jinja/value.cpp",
        "json.cpp",
        "unicode.cpp",
    ] {
        let path = source.join("common").join(file);
        assert!(
            path.is_file(),
            "LLAMA_CPP_DIR must include common/jinja and its JSON/Unicode helpers"
        );
        println!("cargo:rerun-if-changed={}", path.display());
        bridge.file(path);
    }
    for directory in ["common/jinja", "common", "vendor/nlohmann"] {
        println!(
            "cargo:rerun-if-changed={}",
            source.join(directory).display()
        );
    }
    bridge.compile("l2s1_bridge");
    // The compiled archive covers transitive Jinja/ggml headers and C++ flags.
    let archive = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR")).join("libl2s1_bridge.a");
    fingerprint.update(std::fs::read(archive).expect("read compiled bridge identity"));
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
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
