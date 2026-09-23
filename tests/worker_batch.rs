use l2s1::*;
use std::{rc::Rc, sync::mpsc, thread, time::Duration};

fn request(id: &str, tokens: usize) -> DecisionRequest {
    DecisionRequest {
        state: serde_json::json!({"id": id, "tokens": tokens}),
        decisions: vec![Decision {
            id: "decision".into(),
            instruction: "Decide".into(),
            kind: DecisionKind::Binary {
                false_label: "no".into(),
                true_label: "yes".into(),
            },
        }],
    }
}
fn response(id: &str) -> DecisionResponse {
    serde_json::from_value(serde_json::json!({
        "backend": {
            "model_path": id, "model_description": "fixture", "prompt_version": "fixture",
            "runtime": "mock", "offload_requested": false, "offload_device": null
        },
        "policy": {"min_top_probability": 0.8, "min_candidate_mass": 0.05},
        "results": []
    }))
    .unwrap()
}
struct Mock {
    _not_send: Rc<()>,
    owner: thread::ThreadId,
    entered: Option<mpsc::Sender<()>>,
    release: mpsc::Receiver<()>,
    batches: mpsc::Sender<Vec<String>>,
    dropped: mpsc::Sender<bool>,
}
impl DecisionBackend for Mock {
    fn decide(&mut self, _: &DecisionRequest) -> Result<DecisionResponse> {
        panic!("batched worker must use decide_many")
    }
}
impl BatchDecisionBackend for Mock {
    fn estimate_tokens(&mut self, request: &DecisionRequest) -> Result<usize> {
        assert_eq!(self.owner, thread::current().id());
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
            self.release.recv().unwrap();
        }
        if request.state["id"] == "invalid" {
            Err(Error::Invalid("tokenization failure".into()))
        } else {
            Ok(request.state["tokens"].as_u64().unwrap() as usize)
        }
    }
    fn decide_many(&mut self, requests: &[DecisionRequest]) -> Vec<Result<DecisionResponse>> {
        assert_eq!(self.owner, thread::current().id());
        let ids: Vec<String> = requests
            .iter()
            .map(|r| r.state["id"].as_str().unwrap().into())
            .collect();
        self.batches.send(ids.clone()).unwrap();
        if ids.iter().any(|id| id == "bad-length") {
            return vec![];
        }
        ids.iter()
            .map(|id| {
                if id == "execution-error" {
                    Err(Error::Backend("fixture execution failure".into()))
                } else {
                    Ok(response(id))
                }
            })
            .collect()
    }
}
impl Drop for Mock {
    fn drop(&mut self) {
        let _ = self.dropped.send(self.owner == thread::current().id());
    }
}
struct Rig {
    worker: BackendWorker,
    entered: mpsc::Receiver<()>,
    release: mpsc::Sender<()>,
    batches: mpsc::Receiver<Vec<String>>,
    dropped: mpsc::Receiver<bool>,
    budget: MemoryBudget,
}
fn rig(policy: BatchPolicy, queue_capacity: usize) -> Rig {
    let budget = MemoryBudget::new(100);
    let (entered, entry) = mpsc::channel();
    let (release, gate) = mpsc::channel();
    let (batches, batch_rx) = mpsc::channel();
    let (dropped, drop_rx) = mpsc::channel();
    let worker =
        BackendWorker::spawn_batched(queue_capacity, 4096, &budget, 80, policy, move || {
            Ok(Mock {
                _not_send: Rc::new(()),
                owner: thread::current().id(),
                entered: Some(entered),
                release: gate,
                batches,
                dropped,
            })
        })
        .unwrap();
    Rig {
        worker,
        entered: entry,
        release,
        batches: batch_rx,
        dropped: drop_rx,
        budget,
    }
}
fn policy(max_requests: usize, max_input_tokens: usize) -> BatchPolicy {
    BatchPolicy {
        max_requests,
        max_input_tokens,
        max_wait: Duration::from_secs(10),
    }
}
fn start(rig: &Rig, id: &str, tokens: usize) -> DecisionTicket {
    let first = rig.worker.submit(request(id, tokens)).unwrap();
    rig.entered.recv_timeout(Duration::from_secs(2)).unwrap();
    first
}
fn finish(rig: &mut Rig) -> Vec<Vec<String>> {
    rig.release.send(()).unwrap();
    rig.worker.close().unwrap();
    assert!(rig.dropped.recv().unwrap());
    assert_eq!(rig.budget.reserved_bytes(), 0);
    rig.batches.try_iter().collect()
}

#[test]
fn token_overflow_keeps_next_job_and_ticket_order_while_close_drains() {
    let mut rig = rig(policy(4, 7), 8);
    let first = start(&rig, "a", 3);
    let second = rig.worker.submit(request("b", 3)).unwrap();
    let third = rig.worker.submit(request("c", 2)).unwrap();
    let fourth = rig.worker.submit(request("d", 5)).unwrap();
    assert_eq!(finish(&mut rig), vec![vec!["a", "b"], vec!["c", "d"]]);
    for (ticket, id) in [first, second, third, fourth]
        .into_iter()
        .zip(["a", "b", "c", "d"])
    {
        assert_eq!(ticket.wait().unwrap().backend.model_path, id);
    }
    assert!(rig.worker.submit(request("late", 1)).is_err());
    rig.worker.close().unwrap();
}

#[test]
fn request_limit_bounds_batches_even_when_tokens_fit() {
    let mut rig = rig(policy(2, usize::MAX), 8);
    let mut tickets = vec![start(&rig, "a", 1)];
    for id in ["b", "c", "d", "e"] {
        tickets.push(rig.worker.submit(request(id, 1)).unwrap());
    }
    assert_eq!(
        finish(&mut rig),
        vec![vec!["a", "b"], vec!["c", "d"], vec!["e"]]
    );
    for ticket in tickets {
        ticket.wait().unwrap();
    }
}

#[test]
fn invalid_oversize_and_execution_errors_are_isolated_without_replay() {
    let mut rig = rig(policy(8, 10), 8);
    let first = start(&rig, "a", 1);
    let invalid = rig.worker.submit(request("invalid", 1)).unwrap();
    let oversize = rig.worker.submit(request("oversize", 11)).unwrap();
    let failure = rig.worker.submit(request("execution-error", 1)).unwrap();
    let last = rig.worker.submit(request("last", 1)).unwrap();
    assert_eq!(finish(&mut rig), vec![vec!["a", "execution-error", "last"]]);
    first.wait().unwrap();
    assert!(
        invalid
            .wait()
            .unwrap_err()
            .to_string()
            .contains("tokenization")
    );
    assert!(
        oversize
            .wait()
            .unwrap_err()
            .to_string()
            .contains("token limit")
    );
    assert!(
        failure
            .wait()
            .unwrap_err()
            .to_string()
            .contains("execution failure")
    );
    assert_eq!(last.wait().unwrap().backend.model_path, "last");
}

#[test]
fn incorrect_result_count_fails_whole_affected_batch_without_replay() {
    let mut rig = rig(policy(2, 10), 8);
    let first = start(&rig, "bad-length", 1);
    let second = rig.worker.submit(request("b", 1)).unwrap();
    let last = rig.worker.submit(request("c", 1)).unwrap();
    assert_eq!(finish(&mut rig), vec![vec!["bad-length", "b"], vec!["c"]]);
    for ticket in [first, second] {
        assert!(
            ticket
                .wait()
                .unwrap_err()
                .to_string()
                .contains("result count")
        );
    }
    last.wait().unwrap();
}

#[test]
fn zero_wait_does_not_collect_even_already_queued_jobs() {
    let mut policy = policy(8, 10);
    policy.max_wait = Duration::ZERO;
    let mut rig = rig(policy, 8);
    let first = start(&rig, "a", 1);
    let second = rig.worker.submit(request("b", 1)).unwrap();
    assert_eq!(finish(&mut rig), vec![vec!["a"], vec!["b"]]);
    first.wait().unwrap();
    second.wait().unwrap();
}

#[test]
fn collection_deadline_includes_time_since_admission() {
    let mut policy = policy(8, 10);
    policy.max_wait = Duration::from_millis(10);
    let mut rig = rig(policy, 8);
    let first = start(&rig, "a", 1);
    let second = rig.worker.submit(request("b", 1)).unwrap();
    // The first estimate is held while both admitted requests age past the deadline.
    thread::sleep(Duration::from_millis(20));
    assert_eq!(finish(&mut rig), vec![vec!["a"], vec!["b"]]);
    first.wait().unwrap();
    second.wait().unwrap();
}

#[test]
fn collection_timeout_executes_without_waiting_for_another_request_or_close() {
    let mut policy = policy(8, 10);
    policy.max_wait = Duration::from_millis(10);
    let mut rig = rig(policy, 8);
    let first = start(&rig, "a", 1);
    rig.release.send(()).unwrap();
    assert_eq!(
        rig.batches.recv_timeout(Duration::from_secs(2)).unwrap(),
        vec!["a"]
    );
    first.wait().unwrap();
    rig.worker.close().unwrap();
}

#[test]
fn existing_admission_and_memory_guarantees_apply_to_batch_workers() {
    let mut rig = rig(policy(2, 10), 1);
    assert_eq!(rig.budget.reserved_bytes(), 80);
    let first = start(&rig, "a", 1);
    let second = rig.worker.submit(request("b", 1)).unwrap();
    assert!(
        rig.worker
            .submit(request("full", 1))
            .unwrap_err_string()
            .contains("queue is full")
    );
    let mut huge = request("huge", 1);
    huge.state["padding"] = "x".repeat(4096).into();
    assert!(
        rig.worker
            .submit(huge)
            .unwrap_err_string()
            .contains("byte limit")
    );
    let mut invalid = request("invalid-schema", 1);
    invalid.decisions.clear();
    assert!(rig.worker.submit(invalid).is_err());
    assert!(
        BackendWorker::spawn_batched::<Mock, _>(
            1,
            4096,
            &rig.budget,
            30,
            policy(2, 10),
            || unreachable!()
        )
        .is_err()
    );
    finish(&mut rig);
    first.wait().unwrap();
    second.wait().unwrap();
}

// Ticket intentionally has no Debug implementation.
trait ErrorString {
    fn unwrap_err_string(self) -> String;
}
impl<T> ErrorString for Result<T> {
    fn unwrap_err_string(self) -> String {
        match self {
            Ok(_) => panic!("expected error"),
            Err(e) => e.to_string(),
        }
    }
}

#[test]
fn invalid_policy_and_factory_failure_release_reservation() {
    let budget = MemoryBudget::new(100);
    for policy in [policy(0, 1), policy(1, 0)] {
        assert!(
            BackendWorker::spawn_batched::<Mock, _>(
                1,
                4096,
                &budget,
                80,
                policy,
                || unreachable!()
            )
            .is_err()
        );
        assert_eq!(budget.reserved_bytes(), 0);
    }
    assert!(
        BackendWorker::spawn_batched::<Mock, _>(1, 4096, &budget, 80, policy(1, 1), || Err(
            Error::Backend("factory failure".into())
        ))
        .is_err()
    );
    assert_eq!(budget.reserved_bytes(), 0);
}

#[test]
fn full_token_budget_executes_without_waiting_for_collection_timeout() {
    let mut rig = rig(policy(8, 10), 8);
    let first = start(&rig, "a", 10);
    rig.release.send(()).unwrap();
    assert_eq!(
        rig.batches.recv_timeout(Duration::from_secs(2)).unwrap(),
        vec!["a"]
    );
    first.wait().unwrap();
    rig.worker.close().unwrap();
}
