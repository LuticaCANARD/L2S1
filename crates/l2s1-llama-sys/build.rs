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
    // Keep build scripts below the Windows main thread's default stack limit.
    let mut buffer = [0; 64 * 1024];
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
    println!("cargo:rerun-if-env-changed=L2S1_NATIVE_COMPILER_LAUNCHER");
    println!("cargo:rerun-if-env-changed=L2S1_PORTABLE_BUILD");
    println!("cargo:rerun-if-env-changed=L2S1_ARM64_DISPATCH");
    println!("cargo:rerun-if-env-changed=L2S1_OPENMP");
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
        "src",
        "native/chat.cpp",
        "native/exception.h",
        "native/exception.cpp",
        "cmake/CMakeLists.txt",
        "UPSTREAM_COMMIT",
        "build.rs",
    ] {
        println!("cargo:rerun-if-changed={file}");
    }

    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let metal = env::var_os("CARGO_FEATURE_METAL").is_some();
    let target_os = env::var("CARGO_CFG_TARGET_OS").expect("target OS unavailable");
    assert!(!(cuda && metal), "CUDA and Metal are mutually exclusive");
    assert!(!metal || target_os == "macos", "Metal requires macOS");
    let shared_extension = match target_os.as_str() {
        "macos" => ".dylib",
        "windows" => ".dll",
        _ => ".so",
    };
    let portable = env::var_os("L2S1_PORTABLE_BUILD").is_some_and(|v| v == "1");
    let arm64_linux =
        target_os == "linux" && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64");
    // Overrides support controlled build comparisons without editing source.
    let option = |name: &str, default: bool| match env::var(name).as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        Err(env::VarError::NotPresent) => default,
        _ => panic!("{name} must be 0 or 1"),
    };
    let arm64_dispatch = option("L2S1_ARM64_DISPATCH", portable && arm64_linux);
    assert!(
        !arm64_dispatch || arm64_linux,
        "ARM64 dispatch requires Linux aarch64"
    );
    let openmp = option("L2S1_OPENMP", !portable);
    let backend = if cuda {
        "cuda"
    } else if metal {
        "metal"
    } else {
        "cpu"
    };
    let cuda_architectures = env::var("L2S1_CUDA_ARCHITECTURES").ok();
    let native_launcher = env::var("L2S1_NATIVE_COMPILER_LAUNCHER")
        .ok()
        .filter(|launcher| !launcher.is_empty());
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
    build_key.update(backend.as_bytes());
    if portable {
        build_key.update(b"portable");
    }
    if let Some(architectures) = &cuda_architectures {
        build_key.update(architectures.as_bytes());
    }
    build_key.update([u8::from(arm64_dispatch), u8::from(openmp)]);
    let build_dir = format!("{:x}", build_key.finalize());
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    let mut cmake = cmake::Config::new(manifest.join("cmake"));
    cmake
        .out_dir(out_dir.join("llama").join(&build_dir[..16]))
        .profile("Release")
        .define("BUILD_SHARED_LIBS", "ON")
        .define("CMAKE_INSTALL_LIBDIR", "lib")
        // Shared llama/GGML libraries load one another. Give each installed
        // library a path relative to itself as well as the executable rpath.
        .define(
            "CMAKE_INSTALL_RPATH",
            if target_os == "macos" {
                "@loader_path"
            } else {
                "$ORIGIN"
            },
        )
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
        .define("GGML_CUDA", if cuda { "ON" } else { "OFF" })
        .define("GGML_METAL", if metal { "ON" } else { "OFF" });
    if let Some(source) = &selected {
        cmake.define("FETCHCONTENT_SOURCE_DIR_LLAMA_CPP", source);
    }
    if let Some(launcher) = &native_launcher {
        for language in ["C", "CXX", "CUDA"] {
            cmake.define(format!("CMAKE_{language}_COMPILER_LAUNCHER"), launcher);
        }
    }
    if cuda {
        cmake.define("GGML_NATIVE", "OFF");
        if let Some(architectures) = &cuda_architectures {
            cmake.define("CMAKE_CUDA_ARCHITECTURES", architectures);
        }
    }
    if portable {
        // Published CPU packages must not require the build host's CPU ISA or
        // a separately installed OpenMP runtime.
        for option in [
            "GGML_NATIVE",
            "GGML_SSE42",
            "GGML_BMI2",
            "GGML_AVX",
            "GGML_AVX2",
            "GGML_FMA",
            "GGML_F16C",
            "GGML_AVX512",
            "GGML_OPENMP",
        ] {
            cmake.define(option, "OFF");
        }
    }
    cmake.define("GGML_OPENMP", if openmp { "ON" } else { "OFF" });
    cmake.define("GGML_BACKEND_DL", if arm64_dispatch { "ON" } else { "OFF" });
    cmake.define(
        "GGML_CPU_ALL_VARIANTS",
        if arm64_dispatch { "ON" } else { "OFF" },
    );
    if arm64_dispatch {
        cmake.define("GGML_NATIVE", "OFF");
        // Upstream installs loadable CPU modules in bindir. Keep the complete
        // matching runtime together for hashing, packaging and library lookup.
        cmake.define("CMAKE_INSTALL_BINDIR", "lib");
    }
    let install = cmake.build();
    let lib = install.join("lib");
    let runtime_lib = if target_os == "windows" {
        install.join("bin")
    } else {
        lib.clone()
    };
    let prefix = if target_os == "windows" { "" } else { "lib" };
    assert!(
        runtime_lib
            .join(format!("{prefix}llama{shared_extension}"))
            .is_file(),
        "llama.cpp did not install the llama shared library"
    );
    assert!(
        runtime_lib
            .join(format!("{prefix}mtmd{shared_extension}"))
            .is_file(),
        "llama.cpp did not install the mtmd shared library"
    );
    let source = PathBuf::from(
        fs::read_to_string(install.join("build/l2s1-llama-source.txt"))
            .expect("CMake did not report its llama.cpp source directory"),
    );
    // CMake reports an absolute source path. Keep that spelling: on Windows,
    // canonicalize() adds a verbatim prefix that MSVC does not accept for /I.
    assert!(
        source.is_absolute() && source.is_dir(),
        "llama.cpp source directory unavailable after CMake build"
    );
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

    // Generate layouts and C/C++ symbol names from the exact linked revision.
    let mut bindings = bindgen::Builder::default()
        .header_contents("l2s1.h", "#include \"llama.h\"\n#include \"llama-ext.h\"\n#include \"ggml-backend.h\"\n#include \"mtmd.h\"\n#include \"mtmd-helper.h\"\n")
        .clang_args(["-x", "c++", "-std=c++17"])
        .header(manifest.join("native/exception.h").to_string_lossy())
        .allowlist_function("(llama|ggml_backend|mtmd|sd_native)_.*")
        .allowlist_var("(LLAMA|GGML|MTMD)_.*")
        .opaque_type("std::.*")
        .layout_tests(false)
        .generate_comments(false);
    for directory in ["include", "ggml/include", "src", "tools/mtmd"] {
        bindings = bindings.clang_arg(format!("-I{}", source.join(directory).display()));
    }
    bindings = bindings
        .blocklist_function("^llama_model_load_from_file$")
        .raw_line(
            "pub use self::sd_native_llama_model_load_from_file as llama_model_load_from_file;",
        );
    bindings = bindings
        .blocklist_function("^llama_init_from_model$")
        .raw_line("pub use self::sd_native_llama_init_from_model as llama_init_from_model;");
    bindings = bindings
        .blocklist_function("^llama_adapter_lora_init$")
        .raw_line("pub use self::sd_native_llama_adapter_lora_init as llama_adapter_lora_init;");
    bindings = bindings
        .blocklist_function("^llama_set_adapters_lora$")
        .raw_line("pub use self::sd_native_llama_set_adapters_lora as llama_set_adapters_lora;");
    bindings = bindings
        .blocklist_function("^llama_batch_init$")
        .raw_line("pub use self::sd_native_llama_batch_init as llama_batch_init;");
    bindings = bindings
        .blocklist_function("^llama_decode$")
        .raw_line("pub use self::sd_native_llama_decode as llama_decode;");
    bindings = bindings
        .blocklist_function("^llama_tokenize$")
        .raw_line("pub use self::sd_native_llama_tokenize as llama_tokenize;");
    bindings = bindings
        .blocklist_function("^llama_state_seq_get_size$")
        .raw_line("pub use self::sd_native_llama_state_seq_get_size as llama_state_seq_get_size;");
    bindings = bindings
        .blocklist_function("^llama_state_seq_get_data$")
        .raw_line("pub use self::sd_native_llama_state_seq_get_data as llama_state_seq_get_data;");
    bindings = bindings
        .blocklist_function("^llama_state_seq_set_data$")
        .raw_line("pub use self::sd_native_llama_state_seq_set_data as llama_state_seq_set_data;");
    bindings = bindings
        .blocklist_function("^mtmd_init_from_file$")
        .raw_line("pub use self::sd_native_mtmd_init_from_file as mtmd_init_from_file;");
    bindings = bindings
        .blocklist_function("^mtmd_input_chunks_init$")
        .raw_line("pub use self::sd_native_mtmd_input_chunks_init as mtmd_input_chunks_init;");
    bindings = bindings.blocklist_function("^mtmd_helper_bitmap_init_from_buf$").raw_line("pub use self::sd_native_mtmd_helper_bitmap_init_from_buf as mtmd_helper_bitmap_init_from_buf;");
    bindings = bindings
        .blocklist_function("^mtmd_tokenize_from_parts$")
        .raw_line("pub use self::sd_native_mtmd_tokenize_from_parts as mtmd_tokenize_from_parts;");
    bindings = bindings
        .blocklist_function("^mtmd_helper_eval_chunks$")
        .raw_line("pub use self::sd_native_mtmd_helper_eval_chunks as mtmd_helper_eval_chunks;");
    bindings = bindings
        .blocklist_function("^mtmd_batch_init$")
        .raw_line("pub use self::sd_native_mtmd_batch_init as mtmd_batch_init;");
    bindings = bindings
        .blocklist_function("^mtmd_batch_add_chunk$")
        .raw_line("pub use self::sd_native_mtmd_batch_add_chunk as mtmd_batch_add_chunk;");
    bindings = bindings
        .blocklist_function("^mtmd_batch_encode$")
        .raw_line("pub use self::sd_native_mtmd_batch_encode as mtmd_batch_encode;");
    bindings = bindings
        .blocklist_function("^ggml_backend_load_all$")
        .raw_line("pub use self::sd_native_ggml_backend_load_all as ggml_backend_load_all;");
    bindings = bindings
        .blocklist_function("^llama_backend_init$")
        .raw_line("pub use self::sd_native_llama_backend_init as llama_backend_init;");
    bindings = bindings
        .blocklist_function("^llama_memory_clear$")
        .raw_line("pub use self::sd_native_llama_memory_clear as llama_memory_clear;");
    bindings = bindings
        .blocklist_function("^llama_memory_seq_rm$")
        .raw_line("pub use self::sd_native_llama_memory_seq_rm as llama_memory_seq_rm;");
    bindings = bindings
        .blocklist_function("^llama_memory_seq_cp$")
        .raw_line("pub use self::sd_native_llama_memory_seq_cp as llama_memory_seq_cp;");
    bindings = bindings
        .blocklist_function("^llama_model_meta_val_str$")
        .raw_line("pub use self::sd_native_llama_model_meta_val_str as llama_model_meta_val_str;");
    bindings = bindings
        .blocklist_function("^llama_model_desc$")
        .raw_line("pub use self::sd_native_llama_model_desc as llama_model_desc;");
    bindings = bindings
        .blocklist_function("^llama_model_chat_template$")
        .raw_line(
            "pub use self::sd_native_llama_model_chat_template as llama_model_chat_template;",
        );
    bindings = bindings
        .blocklist_function("^llama_token_to_piece$")
        .raw_line("pub use self::sd_native_llama_token_to_piece as llama_token_to_piece;");
    bindings
        .generate()
        .expect("generate matching llama.cpp bindings")
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("write bindings");

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
        .file(manifest.join("native/chat.cpp"))
        .file(manifest.join("native/exception.cpp"));
    if arm64_dispatch {
        bridge.define("L2S1_ARM64_DISPATCH", None);
    }
    for file in JINJA_FILES {
        bridge.file(source.join("common").join(file));
    }
    bridge.compile("l2s1_bridge");

    let mut fingerprint = Sha256::new();
    fingerprint.update(commit.trim().as_bytes());
    fingerprint.update(backend.as_bytes());
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
    for file in [
        "src/lib.rs",
        "src/bridge.rs",
        "src/text.rs",
        "src/vision.rs",
    ] {
        digest_file(&mut fingerprint, &manifest.join(file));
    }
    digest_file(&mut fingerprint, &out_dir.join("bindings.rs"));
    digest_file(&mut fingerprint, &manifest.join("native/chat.cpp"));
    digest_file(&mut fingerprint, &manifest.join("native/exception.h"));
    digest_file(&mut fingerprint, &manifest.join("native/exception.cpp"));
    let archive = out_dir.join(
        if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
            "l2s1_bridge.lib"
        } else {
            "libl2s1_bridge.a"
        },
    );
    digest_file(&mut fingerprint, &archive);
    let mut libraries = fs::read_dir(&runtime_lib)
        .expect("read installed llama.cpp libraries")
        .map(|entry| entry.expect("library entry").path())
        .filter(|path| {
            path.file_name().is_some_and(|name| {
                let name = name.to_string_lossy();
                (name.starts_with(&format!("{prefix}llama"))
                    || name.starts_with(&format!("{prefix}mtmd"))
                    || name.starts_with(&format!("{prefix}ggml")))
                    && name.contains(shared_extension)
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
    println!("cargo::metadata=runtime_libdir={}", runtime_lib.display());
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=mtmd");
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");
    // Make this crate's own native tests runnable without a loader override.
    if target_os == "linux" || target_os == "macos" {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", runtime_lib.display());
    }
}
