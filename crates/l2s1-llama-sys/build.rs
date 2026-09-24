use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

const JINJA_FILES: &[&str] = &[
    "jinja/lexer.cpp",
    "jinja/parser.cpp",
    "jinja/runtime.cpp",
    "jinja/string.cpp",
    "jinja/value.cpp",
    "json.cpp",
    "unicode.cpp",
];

fn digest_file(hash: &mut Sha256, path: &Path) {
    hash.update(
        path.file_name()
            .expect("native identity file name")
            .as_encoded_bytes(),
    );
    let mut file = fs::File::open(path).expect("native runtime identity input unavailable");
    let mut buffer = [0; 1024 * 1024];
    loop {
        let n = file
            .read(&mut buffer)
            .expect("read native runtime identity");
        if n == 0 {
            break;
        }
        hash.update(&buffer[..n]);
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=L2S1_LLAMA_CPP_SOURCE");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=L2S1_CUDA_ARCHITECTURES");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let selected = env::var_os("L2S1_LLAMA_CPP_SOURCE")
        .or_else(|| env::var_os("LLAMA_CPP_DIR"))
        .map(PathBuf::from)
        .map(|path| {
            path.canonicalize()
                .expect("llama.cpp source directory unavailable")
        });
    if let Some(source) = &selected {
        for directory in ["include", "src", "ggml", "common", "vendor", "cmake"] {
            println!(
                "cargo:rerun-if-changed={}",
                source.join(directory).display()
            );
        }
        println!(
            "cargo:rerun-if-changed={}",
            source.join("CMakeLists.txt").display()
        );
    }
    for file in [
        "native/bridge.cpp",
        "native/chat.cpp",
        "cmake/CMakeLists.txt",
        "UPSTREAM_COMMIT",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }

    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let cuda_architectures = env::var("L2S1_CUDA_ARCHITECTURES").ok();
    let commit = if let Some(source) = &selected {
        fs::read_to_string(source.join("UPSTREAM_COMMIT")).unwrap_or_else(|_| {
            std::process::Command::new("git")
                .arg("-C")
                .arg(source)
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .unwrap_or_else(|| "custom-source".into())
        })
    } else {
        fs::read_to_string(manifest.join("UPSTREAM_COMMIT"))
            .expect("pinned llama.cpp commit unavailable")
    };
    let mut build_key = Sha256::new();
    build_key.update(
        selected
            .as_ref()
            .map_or("pinned", |source| {
                source.to_str().expect("source path is UTF-8")
            })
            .as_bytes(),
    );
    build_key.update(commit.trim().as_bytes());
    build_key.update(if cuda { "cuda" } else { "cpu" }.as_bytes());
    if let Some(architectures) = &cuda_architectures {
        build_key.update(architectures.as_bytes());
    }
    let build_dir = format!("{:x}", build_key.finalize());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let mut cmake = cmake::Config::new(manifest.join("cmake"));
    cmake
        .out_dir(out_dir.join("llama").join(&build_dir[..16]))
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_BUILD_MTMD", "ON")
        .define("MTMD_VIDEO", "OFF")
        .define("LLAMA_BUILD_COMMIT", commit.trim())
        .define("LLAMA_BUILD_NUMBER", "0")
        .define("GGML_CUDA", if cuda { "ON" } else { "OFF" });
    if let Some(source) = &selected {
        cmake.define("FETCHCONTENT_SOURCE_DIR_LLAMA_CPP", source);
    }
    if cuda {
        cmake.define("GGML_NATIVE", "OFF");
        if let Some(architectures) = &cuda_architectures {
            cmake.define("CMAKE_CUDA_ARCHITECTURES", architectures);
        }
    }
    let install = cmake.build();
    let lib = install.join("lib");
    assert!(
        lib.join("libllama.so").is_file(),
        "llama.cpp did not install libllama.so"
    );
    assert!(
        lib.join("libmtmd.so").is_file(),
        "llama.cpp did not install libmtmd.so"
    );
    let source = PathBuf::from(
        fs::read_to_string(install.join("build/l2s1-llama-source.txt"))
            .expect("CMake did not report its llama.cpp source directory"),
    )
    .canonicalize()
    .expect("llama.cpp source directory unavailable after CMake build");
    for required in [
        "CMakeLists.txt",
        "include/llama.h",
        "src/llama-ext.h",
        "tools/mtmd/mtmd.h",
        "tools/mtmd/mtmd-helper.h",
        "common/jinja/lexer.cpp",
    ] {
        assert!(
            source.join(required).is_file(),
            "incompatible llama.cpp source: missing {required}"
        );
    }

    let mut bridge = cc::Build::new();
    bridge
        .cpp(true)
        .std("c++17")
        .include(source.join("include"))
        .include(source.join("ggml/include"))
        .include(source.join("common"))
        .include(source.join("src"))
        .include(source.join("tools/mtmd"))
        .include(source.join("vendor"))
        .file(manifest.join("native/bridge.cpp"))
        .file(manifest.join("native/chat.cpp"));
    for file in JINJA_FILES {
        bridge.file(source.join("common").join(file));
    }
    bridge.compile("l2s1_bridge");

    let mut fingerprint = Sha256::new();
    fingerprint.update(commit.trim().as_bytes());
    fingerprint.update(if cuda { "cuda" } else { "cpu" }.as_bytes());
    for file in [
        "include/llama.h",
        "src/llama-ext.h",
        "tools/mtmd/mtmd.h",
        "tools/mtmd/mtmd-helper.h",
    ] {
        digest_file(&mut fingerprint, &source.join(file));
    }
    for file in JINJA_FILES {
        digest_file(&mut fingerprint, &source.join("common").join(file));
    }
    digest_file(&mut fingerprint, &manifest.join("native/bridge.cpp"));
    digest_file(&mut fingerprint, &manifest.join("native/chat.cpp"));
    let archive = out_dir.join("libl2s1_bridge.a");
    digest_file(&mut fingerprint, &archive);
    let mut libraries = fs::read_dir(&lib)
        .expect("read installed llama.cpp libraries")
        .map(|entry| entry.expect("library entry").path())
        .filter(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                (name.starts_with("libllama.so")
                    || name.starts_with("libmtmd.so")
                    || name.starts_with("libggml"))
                    && path.is_file()
            })
        })
        .collect::<Vec<_>>();
    libraries.sort();
    for path in libraries {
        digest_file(&mut fingerprint, &path);
    }
    println!(
        "cargo::metadata=runtime_sha256={:x}",
        fingerprint.finalize()
    );
    println!("cargo::metadata=libdir={}", lib.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
}
