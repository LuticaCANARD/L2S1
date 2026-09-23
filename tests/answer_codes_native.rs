#![cfg(feature = "llama")]
use l2s1::{llama::LlamaBackend, *};
use std::ffi::{CString, c_char, c_void};

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
        cap: usize,
    ) -> *mut c_void;
    fn sd_close(engine: *mut c_void);
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
        cap: usize,
    ) -> bool;
}

#[test]
#[ignore = "requires Gemma SKID_MODEL and optionally SKID_CUDA=1"]
fn three_letter_codes_score_real_multitoken_paths_against_fresh_reference() {
    let model = std::env::var("SKID_MODEL").expect("SKID_MODEL");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let mut backend = LlamaBackend::load(
        model.as_ref(),
        16384,
        256,
        4,
        cuda,
        DecisionPolicy::default(),
    )
    .unwrap();
    let req = DecisionRequest {
        state: serde_json::json!({"wanted":"intent_39"}),
        decisions: vec![Decision {
            id: "wide".into(),
            instruction: "Select the intent matching state.wanted.".into(),
            kind: DecisionKind::Choice {
                options: (0..677)
                    .map(|i| OptionSpec {
                        id: format!("intent_{i}"),
                        criterion: format!("intent_{i}"),
                    })
                    .collect(),
            },
        }],
    };
    let (input, paths) = backend
        .encode_decision_sequences(&req.state, &req.decisions[0])
        .unwrap();
    let multi = paths
        .iter()
        .position(|p| p.len() > 1)
        .expect("must exercise a real multi-token code");
    let response = backend.decide(&req).unwrap();
    let result = &response.results[0];
    assert_eq!(result.scores.len(), 677);
    assert_eq!(result.scores[676].code, "BAA");
    assert!(result.code_prefix_evaluations > 1);
    assert_eq!(result.scores[multi].token_id, -1);
    assert!(
        (result
            .scores
            .iter()
            .map(|s| s.option_probability)
            .sum::<f64>()
            - 1.0)
            .abs()
            < 1e-10
    );
    let path = &paths[multi];
    drop(backend); // Only one model/context allocation is live at a time.
    struct Engine(*mut c_void);
    impl Drop for Engine {
        fn drop(&mut self) {
            unsafe {
                sd_close(self.0);
            }
        }
    }
    let model = CString::new(model).unwrap();
    let mut error = [0 as c_char; 1024];
    let engine = Engine(unsafe {
        sd_open(
            model.as_ptr(),
            16384,
            256,
            256,
            0,
            4,
            cuda,
            error.as_mut_ptr(),
            error.len(),
        )
    });
    assert!(!engine.0.is_null());
    let vocab = unsafe { sd_vocab_size(engine.0) } as usize;
    let mut expected = 0.0;
    for depth in 0..path.len() {
        let mut tokens = input.clone();
        tokens.extend_from_slice(&path[..depth]);
        let mut logits = vec![0.0; vocab];
        let mut reused = 0;
        assert!(unsafe {
            sd_forward(
                engine.0,
                tokens.as_ptr(),
                tokens.len() as i32,
                false,
                &mut reused,
                logits.as_mut_ptr(),
                logits.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        });
        assert_eq!(reused, 0);
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let z = max
            + logits
                .iter()
                .map(|&v| (v as f64 - max).exp())
                .sum::<f64>()
                .ln();
        expected += logits[path[depth] as usize] as f64 - z;
    }
    let delta = (result.scores[multi].raw_logit - expected).abs();
    assert!(delta < 0.002, "multi-token joint likelihood delta {delta}");
    eprintln!(
        "677 candidates verified; code {}; token path {:?}; prefix evaluations {}; fresh-reference log-probability delta {}",
        result.scores[multi].code, path, result.code_prefix_evaluations, delta
    );
}
