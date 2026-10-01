//! Rust ownership and execution policy over the matching upstream ABI.
//!
//! Safety: exported entry points retain the original C ABI. Non-null pointers
//! must reference buffers of the stated length; engine handles must come from
//! sd_open_loading, be exclusively used, and be closed exactly once. Native
//! output views are consumed before any subsequent decode/context mutation.
#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::missing_safety_doc,
    clippy::too_many_arguments
)]
use crate::NativeVisionBatchMetrics;
use crate::raw::*;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    ptr,
    sync::{Mutex, OnceLock},
};

pub(crate) type Result<T> = std::result::Result<T, String>;
pub(crate) struct Engine {
    pub model: *mut llama_model,
    pub ctx: *mut llama_context,
    pub adapter: *mut llama_adapter_lora,
    pub vision: *mut mtmd_context,
    pub devices: [ggml_backend_dev_t; 2],
    pub features_enabled: bool,
    pub last_feature_row: i32,
    pub batch_size: u32,
    pub context_size: u32,
    sequence_capacity: u32,
    allocated_context_size: u32,
    parallel_context_dynamic: bool,
    parallel_shared_kv: bool,
    pub context_params: llama_context_params,
    description: CString,
    pub architecture: CString,
    runtime_libraries: CString,
    pub cached_tokens: Vec<i32>,
    pub cached_boundary: Option<usize>,
    pub split_cache: std::collections::VecDeque<(Vec<i32>, Vec<u8>)>,
    /// Tokens whose KV sequence 0 holds between retaining parallel calls.
    pub parallel_retained: Vec<i32>,
    pub vision_metrics: NativeVisionBatchMetrics,
    pub memory_dirty: bool,
    pub(crate) force_kv_clear: bool,
    pub vision_projector_reuse: bool,
    /// Tokenizer, template and metadata only; no weights or context.
    vocab_only: bool,
    cpu_moe_patterns: Vec<CString>,
    placement_overrides: Vec<llama_model_tensor_buft_override>,
}
impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            if !self.vision.is_null() {
                mtmd_free(self.vision);
            }
            if !self.ctx.is_null() {
                llama_free(self.ctx);
            }
            if !self.model.is_null() {
                llama_model_free(self.model);
            }
        }
    }
}
struct LogState {
    last_error: String,
    last_level: ggml_log_level,
    threshold: ggml_log_level,
}
static LOG: Mutex<LogState> = Mutex::new(LogState {
    last_error: String::new(),
    last_level: 0,
    threshold: 3,
});
static LOAD: Mutex<()> = Mutex::new(());
unsafe extern "C" fn log_callback(level: ggml_log_level, message: *const c_char, _: *mut c_void) {
    let _ = std::panic::catch_unwind(|| {
        if message.is_null() {
            return;
        }
        let text = CStr::from_ptr(message).to_string_lossy();
        let mut log = LOG.lock().unwrap_or_else(|p| p.into_inner());
        let print = if level == ggml_log_level_GGML_LOG_LEVEL_CONT {
            if log.last_level == ggml_log_level_GGML_LOG_LEVEL_ERROR {
                log.last_error.push_str(&text);
            }
            log.last_level >= log.threshold
        } else {
            log.last_level = level;
            if level == ggml_log_level_GGML_LOG_LEVEL_ERROR
                && (log.last_error.is_empty() || !text.contains("failed to load model"))
            {
                log.last_error = text.into_owned();
            }
            level >= log.threshold && level != ggml_log_level_GGML_LOG_LEVEL_NONE
        };
        if print {
            use std::io::Write;
            let _ = std::io::stderr().write_all(CStr::from_ptr(message).to_bytes());
        }
    });
}
fn clear_load_error() {
    let mut log = LOG.lock().unwrap_or_else(|p| p.into_inner());
    log.last_error.clear();
    log.last_level = 0;
}
fn load_error() -> String {
    let log = LOG.lock().unwrap_or_else(|p| p.into_inner());
    let msg = log.last_error.trim_end();
    if msg.is_empty() {
        "llama.cpp did not report a cause".into()
    } else {
        msg.into()
    }
}
fn log_info_enabled() -> bool {
    LOG.lock().unwrap_or_else(|p| p.into_inner()).threshold <= ggml_log_level_GGML_LOG_LEVEL_INFO
}
pub(crate) unsafe fn report(out: *mut c_char, cap: usize, message: &str) {
    if !out.is_null() && cap > 0 {
        let n = message.len().min(cap - 1);
        ptr::copy_nonoverlapping(message.as_ptr(), out.cast(), n);
        *out.add(n) = 0;
    }
}
pub(crate) unsafe fn guarded<T>(
    error: *mut c_char,
    cap: usize,
    f: impl FnOnce() -> Result<T>,
) -> Option<T> {
    sd_native_reset_error();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f));
    let native_error = CStr::from_ptr(sd_native_error());
    if !native_error.to_bytes().is_empty() {
        report(error, cap, &native_error.to_string_lossy());
        return None;
    }
    match result {
        Ok(Ok(value)) => Some(value),
        Ok(Err(message)) => {
            report(error, cap, &message);
            None
        }
        Err(_) => {
            report(error, cap, "Rust bridge panic");
            None
        }
    }
}
pub(crate) unsafe fn run(
    e: *mut Engine,
    error: *mut c_char,
    cap: usize,
    f: impl FnOnce(&mut Engine) -> Result<()>,
) -> bool {
    let ok = guarded(error, cap, || f(e.as_mut().ok_or("null engine")?)).is_some();
    if !ok {
        sd_clear(e);
    }
    ok
}
fn runtime_library_name(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    (name.starts_with("libllama") || name.starts_with("libmtmd") || name.starts_with("libggml"))
        && (name.contains(".so") || name.ends_with(".dylib"))
}
unsafe fn runtime_libraries() -> CString {
    let mut paths: Vec<String> = Vec::new();
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" fn visit(
            info: *mut libc::dl_phdr_info,
            _: usize,
            data: *mut c_void,
        ) -> i32 {
            let paths = &mut *data.cast::<Vec<String>>();
            let path = CStr::from_ptr((*info).dlpi_name).to_string_lossy();
            if runtime_library_name(&path) {
                paths.push(path.into_owned());
            }
            0
        }
        libc::dl_iterate_phdr(Some(visit), (&mut paths as *mut Vec<String>).cast());
    }
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn _dyld_image_count() -> u32;
            fn _dyld_get_image_name(index: u32) -> *const c_char;
        }
        for i in 0.._dyld_image_count() {
            let p = _dyld_get_image_name(i);
            if !p.is_null() {
                let path = CStr::from_ptr(p).to_string_lossy();
                if runtime_library_name(&path) {
                    paths.push(path.into_owned());
                }
            }
        }
    }
    paths.sort();
    CString::new(paths.into_iter().map(|p| p + "\n").collect::<String>()).unwrap()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_open_loading(
    path: *const c_char,
    context: u32,
    batch: u32,
    ubatch: u32,
    flash_attention: i32,
    threads: i32,
    device_kind: i32,
    gpu_layers: i32,
    cpu_moe_layers: i32,
    model_load_mode: i32,
    error: *mut c_char,
    cap: usize,
) -> *mut Engine {
    guarded(error, cap, || {
        if path.is_null() || context == 0 || batch > i32::MAX as u32 || ubatch == 0 || ubatch > batch || !(-1..=1).contains(&flash_attention) {
            return Err("invalid microbatch or FlashAttention option".into());
        }
        if !(0..=2).contains(&device_kind) || gpu_layers < -1 || cpu_moe_layers < 0 || cpu_moe_layers as usize >= llama_max_tensor_buft_overrides() || (device_kind == 0 && (gpu_layers != 0 || cpu_moe_layers != 0)) {
            return Err("invalid CPU/GPU placement options".into());
        }
        // 1 loads only vocabulary and metadata: tokenization without weights.
        let vocab_only = model_load_mode == 1;
        if !(-1..=1).contains(&model_load_mode) || (vocab_only && device_kind != 0) { return Err("invalid model loading mode".into()); }
        static INIT: OnceLock<Result<()>> = OnceLock::new();
        if let Err(cause) = INIT.get_or_init(|| {
            LOG.lock().unwrap_or_else(|p| p.into_inner()).threshold = match std::env::var("L2S1_LOG").as_deref() {
                Ok("off" | "none") => 6, Ok("error") => 4, Ok("info") => 2, Ok("debug") => 1, _ => 3,
            };
            llama_log_set(Some(log_callback), ptr::null_mut());
            mtmd_helper_log_set(Some(log_callback), ptr::null_mut());
            ggml_backend_load_all(); llama_backend_init();
            let error = CStr::from_ptr(sd_native_error());
            if error.to_bytes().is_empty() { Ok(()) } else { Err(error.to_string_lossy().into_owned()) }
        }) { return Err(cause.clone()); }
        // Allocate before giving native code pointers into the device array.
        let mut e = Box::new(Engine {
            model: ptr::null_mut(), ctx: ptr::null_mut(), adapter: ptr::null_mut(), vision: ptr::null_mut(), devices: [ptr::null_mut(); 2],
            features_enabled: false, last_feature_row: -1, batch_size: batch, context_size: context,
            sequence_capacity: 1, allocated_context_size: context, parallel_context_dynamic: false, parallel_shared_kv: true,
            context_params: llama_context_default_params(), description: CString::default(), architecture: CString::default(), runtime_libraries: CString::default(),
            cached_tokens: Vec::new(), cached_boundary: None, split_cache: Default::default(), parallel_retained: Vec::new(), vision_metrics: NativeVisionBatchMetrics::default(), memory_dirty: false,
            force_kv_clear: std::env::var("L2S1_FORCE_KV_CLEAR").as_deref() == Ok("1"), vision_projector_reuse: false, vocab_only,
            cpu_moe_patterns: Vec::new(), placement_overrides: Vec::new(),
        });
        let mut mp = llama_model_default_params();
        if device_kind != 0 {
            let requested = if device_kind == 1 { c"CUDA" } else { c"MTL" };
            for i in 0..ggml_backend_dev_count() {
                let dev = ggml_backend_dev_get(i);
                if CStr::from_ptr(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) == requested && ggml_backend_dev_type(dev) == ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_GPU { e.devices[0] = dev; break; }
            }
            if e.devices[0].is_null() { return Err(format!("{} device unavailable; select CPU explicitly to use CPU", requested.to_string_lossy())); }
        }
        mp.devices = e.devices.as_mut_ptr(); mp.n_gpu_layers = gpu_layers;
        if model_load_mode == 0 { mp.load_mode = llama_load_mode_LLAMA_LOAD_MODE_NONE; }
        mp.vocab_only = vocab_only;
        if log_info_enabled() { eprintln!("l2s1 model loading: {}", ["auto", "read", "vocab-only"][(model_load_mode + 1) as usize]); }
        if cpu_moe_layers > 0 {
            let cpu = ggml_backend_dev_by_type(ggml_backend_dev_type_GGML_BACKEND_DEVICE_TYPE_CPU);
            if cpu.is_null() { return Err("CPU backend unavailable for MoE split".into()); }
            let buft = ggml_backend_dev_buffer_type(cpu);
            e.cpu_moe_patterns = (0..cpu_moe_layers).map(|layer| CString::new(format!("^blk\\.{layer}\\.ffn_(up|down|gate|gate_up)_(ch|)exps\\.")).unwrap()).collect();
            e.placement_overrides = e.cpu_moe_patterns.iter().map(|p| llama_model_tensor_buft_override { pattern: p.as_ptr(), buft }).collect();
            e.placement_overrides.push(llama_model_tensor_buft_override { pattern: ptr::null(), buft: ptr::null_mut() });
            mp.tensor_buft_overrides = e.placement_overrides.as_ptr();
        }
        if log_info_enabled() { eprintln!("l2s1 placement: gpu_layers={gpu_layers} cpu_moe_layers={cpu_moe_layers}"); }
        {
            let _lock = LOAD.lock().unwrap_or_else(|p| p.into_inner()); clear_load_error();
            e.model = llama_model_load_from_file(path, mp);
            if e.model.is_null() {
                let mut cause = load_error();
                if cause.contains("wrong number of tensors") { cause.push_str("; if this is a bundled vision GGUF, model and projector must be separate GGUF files"); }
                return Err(format!("model load failed: {cause}"));
            }
        }
        let mut arch = [0; 64]; llama_model_meta_val_str(e.model, c"general.architecture".as_ptr(), arch.as_mut_ptr(), arch.len());
        e.architecture = CStr::from_ptr(arch.as_ptr()).to_owned();
        if cpu_moe_layers > 0 {
            let key = CString::new(format!("{}.expert_count", e.architecture.to_string_lossy())).unwrap(); let mut experts = [0; 32];
            llama_model_meta_val_str(e.model, key.as_ptr(), experts.as_mut_ptr(), experts.len());
            let count = CStr::from_ptr(experts.as_ptr()).to_string_lossy().parse::<i64>().unwrap_or(0);
            if count <= 0 || cpu_moe_layers > llama_model_n_layer(e.model) { return Err("CPU MoE split requires an MoE model and a layer count within the model".into()); }
        }
        if !llama_model_has_decoder(e.model) || llama_model_has_encoder(e.model) { return Err("decision scoring requires a decoder-only language model".into()); }
        if vocab_only {
            let mut desc = [0; 512]; llama_model_desc(e.model, desc.as_mut_ptr(), desc.len());
            e.description = CStr::from_ptr(desc.as_ptr()).to_owned(); e.runtime_libraries = runtime_libraries();
            return Ok(Box::into_raw(e));
        }
        let cp = &mut e.context_params;
        cp.n_ctx = context; cp.n_batch = batch; cp.n_ubatch = ubatch; cp.n_seq_max = 1; cp.n_threads = threads; cp.n_threads_batch = threads;
        cp.embeddings = false; cp.offload_kqv = device_kind != 0; cp.op_offload = device_kind != 0; cp.flash_attn_type = flash_attention as _;
        e.ctx = llama_init_from_model(e.model, *cp);
        if e.ctx.is_null() { return Err("context allocation failed".into()); }
        e.batch_size = llama_n_batch(e.ctx);
        let mut desc = [0; 512]; llama_model_desc(e.model, desc.as_mut_ptr(), desc.len());
        e.description = CStr::from_ptr(desc.as_ptr()).to_owned(); e.runtime_libraries = runtime_libraries();
        let native_error = CStr::from_ptr(sd_native_error());
        if !native_error.to_bytes().is_empty() { return Err(native_error.to_string_lossy().into_owned()); }
        Ok(Box::into_raw(e))
    }).unwrap_or(ptr::null_mut())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_open_placement(
    path: *const c_char,
    context: u32,
    batch: u32,
    ubatch: u32,
    flash: i32,
    threads: i32,
    cuda: bool,
    gpu: i32,
    moe: i32,
    error: *mut c_char,
    cap: usize,
) -> *mut Engine {
    sd_open_loading(
        path,
        context,
        batch,
        ubatch,
        flash,
        threads,
        i32::from(cuda),
        gpu,
        moe,
        -1,
        error,
        cap,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_open(
    path: *const c_char,
    context: u32,
    batch: u32,
    ubatch: u32,
    flash: i32,
    threads: i32,
    cuda: bool,
    error: *mut c_char,
    cap: usize,
) -> *mut Engine {
    sd_open_placement(
        path,
        context,
        batch,
        ubatch,
        flash,
        threads,
        cuda,
        if cuda { -1 } else { 0 },
        0,
        error,
        cap,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_close(e: *mut Engine) {
    if !e.is_null() {
        drop(Box::from_raw(e));
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_clear(e: *mut Engine) {
    if let Some(e) = e.as_mut() {
        e.clear();
    }
}
impl Engine {
    pub unsafe fn clear(&mut self) {
        self.cached_tokens.clear();
        self.cached_boundary = None;
        self.split_cache.clear();
        self.parallel_retained.clear();
        self.last_feature_row = -1;
        if !self.memory_dirty && !self.force_kv_clear {
            self.vision_metrics.kv_clear_skipped += 1;
            return;
        }
        if !self.ctx.is_null() {
            let memory = llama_get_memory(self.ctx);
            if !memory.is_null() {
                llama_memory_clear(memory, true);
                self.vision_metrics.kv_clear_calls += 1;
            }
        }
        self.memory_dirty = false;
    }
    pub unsafe fn vocab(&self) -> i32 {
        llama_vocab_n_tokens(llama_model_get_vocab(self.model))
    }
    pub unsafe fn recurrent(&self) -> bool {
        llama_model_is_recurrent(self.model) || llama_model_is_hybrid(self.model)
    }
    pub unsafe fn ensure_sequences(
        &mut self,
        capacity: u32,
        requested: u32,
        dynamic: bool,
        shared: bool,
    ) -> Result<()> {
        if self.vocab_only {
            return Err(
                "vocab-only backend has no weights; load the full model to compute logits".into(),
            );
        }
        if !(1..=32).contains(&capacity) || self.context_size > i32::MAX as u32 / capacity {
            return Err("parallel width requires 1..32 and context * width <= INT_MAX".into());
        }
        let maximum = self.context_size * capacity;
        let requested = if requested == 0 { maximum } else { requested };
        if requested > maximum {
            return Err("parallel context reservation is outside the configured limits".into());
        }
        if !self.ctx.is_null()
            && self.sequence_capacity == capacity
            && self.parallel_context_dynamic == dynamic
            && self.parallel_shared_kv == shared
            && (if dynamic {
                self.allocated_context_size >= requested
            } else {
                self.allocated_context_size == requested
            })
        {
            return Ok(());
        }
        self.clear();
        if !self.ctx.is_null() {
            llama_free(self.ctx);
        }
        self.ctx = ptr::null_mut();
        let mut cp = self.context_params;
        cp.n_seq_max = capacity;
        cp.n_ctx = requested;
        cp.kv_unified = if capacity > 1 { shared } else { cp.kv_unified };
        self.ctx = llama_init_from_model(self.model, cp);
        if self.ctx.is_null() {
            return Err("parallel context allocation failed; reduce parallel width".into());
        }
        if !self.adapter.is_null() {
            let mut scale = 1.0;
            if llama_set_adapters_lora(self.ctx, &mut self.adapter, 1, &mut scale) != 0 {
                llama_free(self.ctx);
                self.ctx = ptr::null_mut();
                return Err("LoRA activation failed after context resize".into());
            }
        }
        llama_set_embeddings_nextn(self.ctx, self.features_enabled, false);
        self.sequence_capacity = capacity;
        self.allocated_context_size = requested;
        self.parallel_context_dynamic = dynamic;
        self.parallel_shared_kv = shared;
        Ok(())
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_runtime_libraries(e: *const Engine) -> *const c_char {
    (*e).runtime_libraries.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_description(e: *const Engine) -> *const c_char {
    (*e).description.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_architecture(e: *const Engine) -> *const c_char {
    (*e).architecture.as_ptr()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_chat_template(e: *const Engine) -> *const c_char {
    llama_model_chat_template((*e).model, ptr::null())
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_required_bos(e: *const Engine) -> i32 {
    let v = llama_model_get_vocab((*e).model);
    if llama_vocab_get_add_bos(v) {
        llama_vocab_bos(v)
    } else {
        -1
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_bos_text(e: *const Engine) -> *const c_char {
    let v = llama_model_get_vocab((*e).model);
    let t = llama_vocab_bos(v);
    if t < 0 {
        c"".as_ptr()
    } else {
        llama_vocab_get_text(v, t)
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_eos_text(e: *const Engine) -> *const c_char {
    let v = llama_model_get_vocab((*e).model);
    let t = llama_vocab_eos(v);
    if t < 0 {
        c"".as_ptr()
    } else {
        llama_vocab_get_text(v, t)
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_device(e: *const Engine) -> *const c_char {
    if (*e).devices[0].is_null() {
        c"CPU".as_ptr()
    } else {
        ggml_backend_dev_description((*e).devices[0])
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_vocab_size(e: *const Engine) -> i32 {
    (*e).vocab()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_tokenize(
    e: *const Engine,
    text: *const c_char,
    length: i32,
    special: bool,
    out: *mut i32,
    capacity: i32,
) -> i32 {
    llama_tokenize(
        llama_model_get_vocab((*e).model),
        text,
        length,
        out,
        capacity,
        false,
        special,
    )
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_recurrent_or_hybrid(e: *const Engine) -> bool {
    (*e).recurrent()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_context_tokens(e: *const Engine) -> u32 {
    if (*e).ctx.is_null() {
        0
    } else {
        llama_n_ctx((*e).ctx)
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_training_context(e: *const Engine) -> u32 {
    llama_model_n_ctx_train((*e).model) as u32
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_set_features(e: *mut Engine, enabled: bool) -> bool {
    let e = &mut *e;
    if e.ctx.is_null()
        || (enabled
            && (e.architecture.as_c_str() != c"gemma4"
                || llama_pooling_type(e.ctx) != llama_pooling_type_LLAMA_POOLING_TYPE_NONE))
    {
        return false;
    }
    if e.features_enabled != enabled {
        e.clear();
        e.features_enabled = enabled;
        llama_set_embeddings_nextn(e.ctx, enabled, false);
    }
    true
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_feature_size(e: *mut Engine) -> i32 {
    llama_model_n_embd_out((*e).model)
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_copy_features(e: *mut Engine, output: *mut f32, size: usize) -> bool {
    let e = &*e;
    if e.ctx.is_null()
        || !e.features_enabled
        || e.last_feature_row < 0
        || output.is_null()
        || size != llama_model_n_embd_out(e.model) as usize
    {
        return false;
    }
    let features = llama_get_embeddings_nextn_ith(e.ctx, e.last_feature_row);
    if features.is_null() {
        return false;
    }
    ptr::copy_nonoverlapping(features, output, size);
    true
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_load_lora(
    e: *mut Engine,
    path: *const c_char,
    error: *mut c_char,
    cap: usize,
) -> bool {
    // A failed reload must preserve the existing adapter and its usable context.
    guarded(error, cap, || {
        let e = e.as_mut().ok_or("null engine")?;
        if !e.adapter.is_null() {
            return Err(
                "one LoRA adapter is supported per backend; load a new backend to change it".into(),
            );
        }
        if e.ctx.is_null() {
            return Err("context unavailable; create a new backend before loading LoRA".into());
        }
        if path.is_null() {
            return Err("invalid LoRA path".into());
        }
        let mut adapter = llama_adapter_lora_init(e.model, path);
        if adapter.is_null() {
            return Err("LoRA adapter load failed; see llama.cpp stderr".into());
        }
        e.clear();
        let mut scale = 1.0;
        if llama_set_adapters_lora(e.ctx, &mut adapter, 1, &mut scale) != 0 {
            llama_adapter_lora_free(adapter);
            return Err("LoRA adapter activation failed".into());
        }
        e.adapter = adapter;
        Ok(())
    })
    .is_some()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_vision_marker() -> *const c_char {
    mtmd_default_marker()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_load_vision_projector(
    e: *mut Engine,
    path: *const c_char,
    error: *mut c_char,
    cap: usize,
) -> bool {
    guarded(error, cap, || {
        let e = e.as_mut().ok_or("invalid vision projector path")?;
        if e.model.is_null() || path.is_null() || *path == 0 {
            return Err("invalid vision projector path".into());
        }
        if e.vocab_only {
            return Err("vocab-only backend cannot load a vision projector".into());
        }
        if !e.vision.is_null() {
            return Err("vision projector already loaded".into());
        }
        let mut params = mtmd_context_params_default();
        params.print_timings = log_info_enabled();
        params.use_gpu = !e.devices[0].is_null();
        params.device = e.devices[0];
        params.n_threads = e.context_params.n_threads;
        params.flash_attn_type = e.context_params.flash_attn_type;
        let _lock = LOAD.lock().unwrap_or_else(|p| p.into_inner());
        clear_load_error();
        let vision = mtmd_init_from_file(path, e.model, params);
        if vision.is_null() {
            return Err(format!("vision projector load failed: {}", load_error()));
        }
        if !mtmd_support_vision(vision) {
            mtmd_free(vision);
            return Err("projector does not support image input".into());
        }
        e.vision = vision;
        e.clear();
        Ok(())
    })
    .is_some()
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_set_vision_projector_reuse(e: *mut Engine, enabled: bool) {
    if let Some(e) = e.as_mut() {
        e.vision_projector_reuse = enabled;
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "C" fn sd_vision_batch_metrics(
    e: *const Engine,
    out: *mut NativeVisionBatchMetrics,
) -> bool {
    if e.is_null() || out.is_null() {
        return false;
    }
    *out = (*e).vision_metrics;
    true
}

// Scoped native allocations are released even on an error or caught Rust panic.
pub(crate) struct Batch(pub llama_batch);
impl Batch {
    pub unsafe fn new(size: u32) -> Result<Self> {
        let b = Self(llama_batch_init(size as i32, 0, 1));
        if b.0.token.is_null()
            || b.0.pos.is_null()
            || b.0.n_seq_id.is_null()
            || b.0.seq_id.is_null()
            || b.0.logits.is_null()
        {
            return Err("batch allocation failed".into());
        }
        Ok(b)
    }
    pub unsafe fn add(&mut self, token: i32, pos: i32, seq: i32, output: bool) -> i32 {
        let j = self.0.n_tokens as usize;
        *self.0.token.add(j) = token;
        *self.0.pos.add(j) = pos;
        *self.0.n_seq_id.add(j) = 1;
        **self.0.seq_id.add(j) = seq;
        *self.0.logits.add(j) = i8::from(output);
        self.0.n_tokens += 1;
        j as i32
    }
}
impl Drop for Batch {
    fn drop(&mut self) {
        unsafe {
            llama_batch_free(self.0);
        }
    }
}
pub(crate) fn validate_candidates(ids: &[i32], vocab: i32) -> Result<()> {
    if !(2..=26).contains(&ids.len()) || vocab <= 0 {
        return Err("invalid compact candidate bank".into());
    }
    for (i, id) in ids.iter().enumerate() {
        if *id < 0 || *id >= vocab {
            return Err("candidate token outside vocabulary".into());
        }
        if ids[..i].contains(id) {
            return Err("duplicate candidate token".into());
        }
    }
    Ok(())
}
pub(crate) fn compact(output: &[f32], ids: &[i32], logits: &mut [f32]) -> Result<f64> {
    validate_candidates(ids, output.len() as i32)?;
    if logits.len() != ids.len() {
        return Err("invalid compact evidence buffer size".into());
    }
    let mut maximum = f64::NEG_INFINITY;
    for &value in output {
        if value.is_nan() || value == f32::INFINITY {
            return Err("invalid vocabulary logit".into());
        }
        maximum = maximum.max(f64::from(value));
    }
    if !maximum.is_finite() {
        return Err("no finite vocabulary logit".into());
    }
    for &id in ids {
        if !output[id as usize].is_finite() {
            return Err("nonfinite candidate logit".into());
        }
    }
    let mut sum = 0.0;
    for &value in output {
        sum += (f64::from(value) - maximum).exp();
    }
    let normalizer = maximum + sum.ln();
    if !normalizer.is_finite() {
        return Err("invalid vocabulary normalizer".into());
    }
    for (out, &id) in logits.iter_mut().zip(ids) {
        *out = output[id as usize];
    }
    Ok(normalizer)
}
