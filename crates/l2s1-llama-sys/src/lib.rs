//! Raw FFI for the pinned native L2S1 llama.cpp bridge.
use std::ffi::{c_char, c_void};

#[repr(C)]
#[derive(Default)]
pub struct NativeRestoreMetrics {
    pub snapshot_bytes: usize,
    pub save_ms: f64,
    pub restore_ms: f64,
    pub prefill_ms: f64,
    pub suffix_ms: f64,
    pub restores: usize,
    pub fallback: i32,
}
unsafe extern "C" {
    pub fn sd_forward_compact(
        engine: *mut c_void,
        tokens: *const i32,
        count: i32,
        reuse: bool,
        reused: *mut i32,
        candidate_ids: *const i32,
        candidate_count: usize,
        candidate_logits: *mut f32,
        candidate_logits_count: usize,
        log_normalizer: *mut f64,
        vocabulary_size: *mut i32,
        error: *mut c_char,
        cap: usize,
    ) -> bool;
    pub fn sd_recurrent_or_hybrid(engine: *const c_void) -> bool;
    pub fn sd_training_context(engine: *const c_void) -> u32;
    pub fn sd_forward_restore(
        engine: *mut c_void,
        tokens: *const *const i32,
        counts: *const i32,
        sequences: i32,
        limit: usize,
        reused: *mut i32,
        logits: *mut f32,
        logits_count: usize,
        metrics: *mut NativeRestoreMetrics,
        error: *mut c_char,
        cap: usize,
    ) -> bool;
    pub fn sd_open(
        path: *const c_char,
        context: u32,
        batch: u32,
        ubatch: u32,
        flash_attention: i32,
        threads: i32,
        cuda: bool,
        error: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
    pub fn sd_open_loading(
        path: *const c_char,
        context: u32,
        batch: u32,
        ubatch: u32,
        flash_attention: i32,
        threads: i32,
        cuda: bool,
        gpu_layers: i32,
        cpu_moe_layers: i32,
        model_load_mode: i32,
        error: *mut c_char,
        cap: usize,
    ) -> *mut c_void;
    pub fn sd_close(engine: *mut c_void);
    pub fn sd_clear(engine: *mut c_void);
    pub fn sd_set_features(engine: *mut c_void, enabled: bool) -> bool;
    pub fn sd_feature_size(engine: *mut c_void) -> i32;
    pub fn sd_copy_features(engine: *mut c_void, output: *mut f32, size: usize) -> bool;
    pub fn sd_load_lora(
        engine: *mut c_void,
        path: *const c_char,
        error: *mut c_char,
        cap: usize,
    ) -> bool;
    pub fn sd_runtime_libraries(engine: *const c_void) -> *const c_char;
    pub fn sd_description(engine: *const c_void) -> *const c_char;
    pub fn sd_architecture(engine: *const c_void) -> *const c_char;
    pub fn sd_chat_template(engine: *const c_void) -> *const c_char;
    pub fn sd_required_bos(engine: *const c_void) -> i32;
    pub fn sd_bos_text(engine: *const c_void) -> *const c_char;
    pub fn sd_eos_text(engine: *const c_void) -> *const c_char;
    pub fn sd_render_chat(
        template: *const c_char,
        user: *const c_char,
        bos: *const c_char,
        eos: *const c_char,
        out: *mut c_char,
        capacity: i32,
    ) -> i32;
    pub fn sd_device(engine: *const c_void) -> *const c_char;
    pub fn sd_vocab_size(engine: *const c_void) -> i32;
    pub fn sd_tokenize(
        engine: *const c_void,
        text: *const c_char,
        length: i32,
        special: bool,
        out: *mut i32,
        capacity: i32,
    ) -> i32;
    pub fn sd_forward(
        engine: *mut c_void,
        tokens: *const i32,
        count: i32,
        reuse: bool,
        reused: *mut i32,
        logits: *mut f32,
        logits_count: usize,
        error: *mut c_char,
        cap: usize,
    ) -> bool;
    pub fn sd_forward_parallel(
        engine: *mut c_void,
        tokens: *const *const i32,
        counts: *const i32,
        sequences: i32,
        capacity: u32,
        reused: *mut i32,
        logits: *mut f32,
        logits_count: usize,
        error: *mut c_char,
        cap: usize,
    ) -> bool;
}
