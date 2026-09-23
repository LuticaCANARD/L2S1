#![cfg(feature = "llama")]

use l2s1::{llama::LlamaBackend, *};
use std::{sync::mpsc, thread, time::Duration};

fn load(model: &str, cuda: bool) -> LlamaBackend {
    let mut backend =
        LlamaBackend::load(model.as_ref(), 2048, 32, 4, cuda, DecisionPolicy::default()).unwrap();
    assert!(
        !backend.inspect().capabilities.recurrent_or_hybrid,
        "this ignored test requires a model supporting Parallel execution"
    );
    backend.set_prompt_layout(PromptLayout::StateFirst);
    backend.set_parallel_width(4).unwrap();
    backend.set_execution_mode(ExecutionMode::Parallel);
    backend
}

// Gate only the first estimate so all test submissions deterministically join
// the same batch. All token counting and inference use the real LlamaBackend.
struct GatedLlama {
    backend: LlamaBackend,
    owner: thread::ThreadId,
    entered: Option<mpsc::Sender<()>>,
    release: mpsc::Receiver<()>,
    batches: mpsc::Sender<usize>,
    dropped: mpsc::Sender<bool>,
}
impl DecisionBackend for GatedLlama {
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.backend.decide(request)
    }
}
impl BatchDecisionBackend for GatedLlama {
    fn estimate_tokens(&mut self, request: &DecisionRequest) -> Result<usize> {
        assert_eq!(self.owner, thread::current().id());
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
            self.release
                .recv_timeout(Duration::from_secs(10))
                .expect("release the first native estimate");
        }
        self.backend.estimate_tokens(request)
    }
    fn decide_many(&mut self, requests: &[DecisionRequest]) -> Vec<Result<DecisionResponse>> {
        assert_eq!(self.owner, thread::current().id());
        self.batches.send(requests.len()).unwrap();
        self.backend.decide_many(requests)
    }
}
impl Drop for GatedLlama {
    fn drop(&mut self) {
        let _ = self.dropped.send(self.owner == thread::current().id());
    }
}

fn selected(value: &DecisionValue) -> serde_json::Value {
    match value {
        DecisionValue::Binary { value, .. } => serde_json::json!(value),
        DecisionValue::Choice { selected } | DecisionValue::Ordinal { selected, .. } => {
            serde_json::json!(selected)
        }
    }
}
fn compare(reference: &DecisionResponse, actual: &DecisionResponse) {
    assert_eq!(actual.backend.execution_mode, ExecutionMode::Parallel);
    assert_eq!(actual.backend.parallel_width, 4);
    assert_eq!(actual.results.len(), reference.results.len());
    for (expected, observed) in reference.results.iter().zip(&actual.results) {
        assert_eq!(observed.id, expected.id);
        assert_eq!(observed.input_tokens, expected.input_tokens);
        assert_eq!(observed.reused_prefix_tokens, expected.reused_prefix_tokens);
        assert_eq!(observed.scoring_method, expected.scoring_method);
        assert_eq!(observed.calibration_id, expected.calibration_id);
        assert!(!observed.truncated);
        assert_eq!(observed.scores.len(), expected.scores.len());
        assert!((observed.candidate_mass - expected.candidate_mass).abs() <= 1e-4);
        assert!((observed.top_option_probability - expected.top_option_probability).abs() <= 1e-4);
        for (a, b) in observed.scores.iter().zip(&expected.scores) {
            assert_eq!((&a.id, &a.code, a.token_id), (&b.id, &b.code, b.token_id));
            assert!(
                (a.raw_logit - b.raw_logit).abs() <= 1e-3,
                "same native Parallel batch changed logits: {} versus {}",
                a.raw_logit,
                b.raw_logit
            );
            assert!((a.option_probability - b.option_probability).abs() <= 1e-4);
        }
        assert_eq!(selected(&observed.value), selected(&expected.value));
        assert_eq!(
            serde_json::to_value(&observed.abstention_reasons).unwrap(),
            serde_json::to_value(&expected.abstention_reasons).unwrap()
        );
        match (&observed.value, &expected.value) {
            (DecisionValue::Binary { p_true: a, .. }, DecisionValue::Binary { p_true: b, .. }) => {
                assert!((a - b).abs() <= 1e-4);
            }
            (
                DecisionValue::Ordinal {
                    expected_value: a, ..
                },
                DecisionValue::Ordinal {
                    expected_value: b, ..
                },
            ) => assert!((a - b).abs() <= 1e-3),
            _ => {}
        }
    }
}

/// Checks actual model execution and ticket mapping, not a throughput or quality
/// claim. Reference and worker use identical Parallel waves, so tolerances apply
/// to repeated execution only; this does not claim equivalence to Fresh mode.
#[test]
#[ignore = "requires SKID_MODEL supporting Parallel, optionally SKID_CUDA=1"]
fn native_batched_worker_preserves_mapping_isolates_invalid_input_and_drains() {
    let model = std::env::var("SKID_MODEL").expect("set SKID_MODEL");
    let cuda = std::env::var("SKID_CUDA").as_deref() == Ok("1");
    let template: DecisionRequest =
        serde_json::from_str(include_str!("../examples/warehouse.json")).unwrap();
    let mut valid = Vec::new();
    for (index, storage) in ["chilled", "ambient", "frozen"].into_iter().enumerate() {
        let mut request = template.clone();
        request.state["storage_requirement"] = storage.into();
        request.state["hours_until_dispatch"] = serde_json::json!(index * 12 + 1);
        request.decisions.truncate(index + 1);
        for decision in &mut request.decisions {
            decision.id = format!("request-{index}-{}", decision.id);
        }
        valid.push(request);
    }
    let mut oversized = valid[0].clone();
    oversized.decisions[0].instruction = "overlong ".repeat(8192);
    let mixed = [
        valid[0].clone(),
        oversized.clone(),
        valid[1].clone(),
        valid[2].clone(),
    ];
    let (reference, max_input_tokens) = {
        let mut backend = load(&model, cuda);
        let token_budget = valid
            .iter()
            .map(|r| backend.estimate_tokens(r).unwrap())
            .sum();
        let mut outcomes = backend.decide_many(&mixed).into_iter();
        let a = outcomes.next().unwrap().unwrap();
        assert!(outcomes.next().unwrap().is_err());
        let b = outcomes.next().unwrap().unwrap();
        let c = outcomes.next().unwrap().unwrap();
        assert!(outcomes.next().is_none());
        (vec![a, b, c], token_budget)
    }; // Drop the direct model before constructing the worker's owner-thread model.

    let (entered, entry) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let (batches, batch_rx) = mpsc::channel();
    let (dropped, drop_rx) = mpsc::channel();
    // Reservation bookkeeping is tested here; this is not an actual RSS limit.
    let estimated_bytes = 8usize * 1024 * 1024 * 1024;
    let budget = MemoryBudget::new(estimated_bytes);
    let mut worker = BackendWorker::spawn_batched(
        8,
        256 * 1024,
        &budget,
        estimated_bytes,
        BatchPolicy {
            max_requests: 3,
            max_input_tokens,
            max_wait: Duration::from_secs(30),
        },
        move || {
            Ok(GatedLlama {
                backend: load(&model, cuda),
                owner: thread::current().id(),
                entered: Some(entered),
                release: gate,
                batches,
                dropped,
            })
        },
    )
    .unwrap();
    let first = worker.submit(valid[0].clone()).unwrap();
    entry.recv_timeout(Duration::from_secs(10)).unwrap();
    // Schema-invalid requests fail synchronously; model-context-invalid requests
    // are admitted and fail only their own ticket during native token counting.
    let mut invalid_schema = valid[0].clone();
    invalid_schema.decisions.clear();
    assert!(worker.submit(invalid_schema).is_err());
    let invalid_ticket = worker.submit(oversized).unwrap();
    let second = worker.submit(valid[1].clone()).unwrap();
    let third = worker.submit(valid[2].clone()).unwrap();
    release.send(()).unwrap();
    worker.close().unwrap();
    assert!(drop_rx.recv().unwrap());
    assert_eq!(budget.reserved_bytes(), 0);
    assert_eq!(batch_rx.try_iter().collect::<Vec<_>>(), vec![3]);
    assert!(invalid_ticket.wait().is_err());
    for (ticket, expected) in [first, second, third].into_iter().zip(&reference) {
        compare(expected, &ticket.wait().unwrap());
    }
    assert!(worker.submit(valid[0].clone()).is_err());
    worker.close().unwrap();
}
