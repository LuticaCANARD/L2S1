#![cfg(feature = "llama")]

use std::ffi::{CString, c_char, c_void};

use l2s1::{DecisionPolicy, DecisionRequest, llama::LlamaBackend};

unsafe extern "C" {
    fn sd_open(
        path: *const c_char,
        context: u32,
        batch: u32,
        ubatch: u32,
        flash_attention: i32,
        threads: i32,
        cuda: bool,
        error: *mut c_char,
        error_cap: usize,
    ) -> *mut c_void;
    fn sd_close(engine: *mut c_void);
    fn sd_clear(engine: *mut c_void);
    fn sd_vocab_size(engine: *const c_void) -> i32;
    fn sd_forward(
        engine: *mut c_void,
        tokens: *const i32,
        count: i32,
        reuse: bool,
        reused: *mut i32,
        logits: *mut f32,
        logits_count: usize,
        error: *mut c_char,
        error_cap: usize,
    ) -> bool;
    fn sd_forward_compact(
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
        error_cap: usize,
    ) -> bool;
}

struct NativeEngine(*mut c_void);
impl Drop for NativeEngine {
    fn drop(&mut self) {
        unsafe { sd_close(self.0) };
    }
}

/// Tests the native transfer boundary independently of Rust's evidence scorer.
/// Invalid candidate lists must clear the prefix so a later valid call recovers.
#[test]
#[ignore = "requires SKID_MODEL and optionally SKID_CUDA=1"]
fn compact_native_matches_full_logits_and_clears_failed_requests() {
    let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let request: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    let backend =
        LlamaBackend::load(model.as_ref(), 2048, 32, 4, cuda, DecisionPolicy::default()).unwrap();
    let (tokens, candidates) = backend
        .encode_decision(&request.state, &request.decisions[0])
        .unwrap();
    drop(backend);
    assert!(tokens.len() > 32);
    let path = CString::new(model).unwrap();
    let mut error = [0i8; 1024];
    let engine = NativeEngine(unsafe {
        sd_open(
            path.as_ptr(),
            2048,
            32,
            32,
            -1,
            4,
            cuda,
            error.as_mut_ptr(),
            error.len(),
        )
    });
    assert!(!engine.0.is_null(), "native open failed: {error:?}");
    let vocab = unsafe { sd_vocab_size(engine.0) };
    let mut full = vec![0.0f32; vocab as usize];
    let mut reused = -1;
    assert!(unsafe {
        sd_forward(
            engine.0,
            tokens.as_ptr(),
            tokens.len() as i32,
            false,
            &mut reused,
            full.as_mut_ptr(),
            full.len(),
            error.as_mut_ptr(),
            error.len(),
        )
    });
    let maximum = full
        .iter()
        .copied()
        .map(f64::from)
        .fold(f64::NEG_INFINITY, f64::max);
    let expected_normalizer = maximum
        + full
            .iter()
            .map(|&value| (f64::from(value) - maximum).exp())
            .sum::<f64>()
            .ln();
    let mut compact = vec![0.0f32; candidates.len()];
    let mut normalizer = f64::NAN;
    let mut actual_vocab = 0;
    let mut run = |ids: &[i32], count: i32, output_size: usize, reuse: bool| {
        let ok = unsafe {
            sd_forward_compact(
                engine.0,
                tokens.as_ptr(),
                count,
                reuse,
                &mut reused,
                ids.as_ptr(),
                ids.len(),
                compact.as_mut_ptr(),
                output_size,
                &mut normalizer,
                &mut actual_vocab,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if ok {
            assert_eq!(actual_vocab, vocab);
            for (&candidate, &value) in ids.iter().zip(&compact) {
                assert_eq!(value, full[candidate as usize]);
            }
            assert!((normalizer - expected_normalizer).abs() < 1e-12);
        }
        (ok, reused)
    };
    assert_eq!(
        run(&candidates, tokens.len() as i32, candidates.len(), false),
        (true, 0)
    );
    assert_eq!(
        run(&candidates, tokens.len() as i32, candidates.len(), true),
        (true, 0)
    );
    let (ok, reused_tokens) = run(&candidates, tokens.len() as i32, candidates.len(), true);
    assert!(ok);
    // Hybrid models correctly fall back to fresh; ordinary attention models
    // may reuse only complete prefill batches.
    assert_eq!(reused_tokens % 32, 0);
    let mut duplicate = candidates.clone();
    duplicate[1] = duplicate[0];
    let mut negative = candidates.clone();
    negative[0] = -1;
    let mut outside = candidates.clone();
    outside[0] = vocab;
    for invalid in [&duplicate, &negative, &outside] {
        assert_eq!(
            run(invalid, tokens.len() as i32, invalid.len(), true),
            (false, 0)
        );
        assert_eq!(
            run(&candidates, tokens.len() as i32, candidates.len(), true),
            (true, 0)
        );
    }
    assert_eq!(
        run(&candidates, tokens.len() as i32, candidates.len() - 1, true),
        (false, 0)
    );
    assert_eq!(
        run(&candidates, tokens.len() as i32, candidates.len(), true),
        (true, 0)
    );
    assert_eq!(run(&candidates, 0, candidates.len(), true), (false, 0));
    assert_eq!(
        run(&candidates, tokens.len() as i32, candidates.len(), true),
        (true, 0)
    );
    unsafe { sd_clear(engine.0) };
}
