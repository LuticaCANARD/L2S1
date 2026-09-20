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
    let header = source.join("include/llama.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .std("c++17")
        .include(source.join("include"))
        .include(source.join("ggml/include"))
        .include(source.join("common"))
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
    bridge.compile("skid_desion_bridge");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
}
