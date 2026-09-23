//! A bounded queue with a backend constructed, used and dropped on its owner thread.
use crate::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc::{self, SyncSender},
};
use std::time::{Duration, Instant};

/// An opt-in backend capable of executing independent requests together.
/// Implementations must return exactly one result per request, in input order.
/// An execution error is delivered to its ticket; the worker never retries it.
pub trait BatchDecisionBackend: DecisionBackend {
    /// Count the actual model input tokens, including every decision prompt.
    fn estimate_tokens(&mut self, request: &DecisionRequest) -> Result<usize>;
    fn decide_many(&mut self, requests: &[DecisionRequest]) -> Vec<Result<DecisionResponse>>;
}

/// Limits for opt-in microbatching. Queue and serialized request limits still apply.
#[derive(Debug, Clone, Copy)]
pub struct BatchPolicy {
    pub max_requests: usize,
    pub max_input_tokens: usize,
    /// Maximum collection time since the first request was admitted. This is
    /// not a deadline for inference or time spent waiting behind an earlier batch.
    pub max_wait: Duration,
}

#[derive(Clone)]
pub struct MemoryBudget {
    limit: usize,
    reserved: Arc<AtomicUsize>,
}
impl MemoryBudget {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            reserved: Arc::new(AtomicUsize::new(0)),
        }
    }
    pub fn reserved_bytes(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }
    fn reserve(&self, bytes: usize) -> Result<Reservation> {
        if bytes == 0 {
            return Err(Error::Invalid(
                "worker memory reservation must be positive".into(),
            ));
        }
        self.reserved
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(bytes).filter(|n| *n <= self.limit)
            })
            .map_err(|_| Error::Backend("worker memory budget exhausted".into()))?;
        Ok(Reservation {
            budget: self.clone(),
            bytes,
        })
    }
}
struct Reservation {
    budget: MemoryBudget,
    bytes: usize,
}
impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
struct Job {
    admitted_at: Instant,
    request: DecisionRequest,
    reply: mpsc::Sender<Result<DecisionResponse>>,
}
pub struct DecisionTicket {
    reply: mpsc::Receiver<Result<DecisionResponse>>,
}
impl DecisionTicket {
    pub fn wait(self) -> Result<DecisionResponse> {
        self.reply
            .recv()
            .map_err(|_| Error::Backend("backend worker stopped before replying".into()))?
    }
}
pub struct BackendWorker {
    sender: Option<SyncSender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    max_request_bytes: usize,
}
impl BackendWorker {
    /// Reservation is an operator estimate (weights + context + native scratch), not RSS enforcement.
    /// The factory runs on the worker: B need not implement Send or Sync.
    pub fn spawn<B, F>(
        queue_capacity: usize,
        max_request_bytes: usize,
        budget: &MemoryBudget,
        estimated_bytes: usize,
        factory: F,
    ) -> Result<Self>
    where
        B: DecisionBackend + 'static,
        F: FnOnce() -> Result<B> + Send + 'static,
    {
        Self::spawn_inner(
            queue_capacity,
            max_request_bytes,
            budget,
            estimated_bytes,
            factory,
            |mut backend, receiver| {
                while let Ok(job) = receiver.recv() {
                    let result = backend.decide(&job.request);
                    let _ = job.reply.send(result);
                }
            },
        )
    }
    /// Collect bounded batches on the backend owner thread. A request exceeding
    /// the token limit fails through its ticket. Overflow waits for the next batch.
    /// The native backend never needs to implement Send or Sync.
    pub fn spawn_batched<B, F>(
        queue_capacity: usize,
        max_request_bytes: usize,
        budget: &MemoryBudget,
        estimated_bytes: usize,
        policy: BatchPolicy,
        factory: F,
    ) -> Result<Self>
    where
        B: BatchDecisionBackend + 'static,
        F: FnOnce() -> Result<B> + Send + 'static,
    {
        if policy.max_requests == 0 || policy.max_input_tokens == 0 {
            return Err(Error::Invalid(
                "batch request and token limits must be positive".into(),
            ));
        }
        Self::spawn_inner(
            queue_capacity,
            max_request_bytes,
            budget,
            estimated_bytes,
            factory,
            move |backend, receiver| run_batches(backend, receiver, policy),
        )
    }
    fn spawn_inner<B, F, R>(
        queue_capacity: usize,
        max_request_bytes: usize,
        budget: &MemoryBudget,
        estimated_bytes: usize,
        factory: F,
        run: R,
    ) -> Result<Self>
    where
        B: 'static,
        F: FnOnce() -> Result<B> + Send + 'static,
        R: FnOnce(B, mpsc::Receiver<Job>) + Send + 'static,
    {
        if queue_capacity == 0 || max_request_bytes == 0 {
            return Err(Error::Invalid(
                "worker queue and request limits must be positive".into(),
            ));
        }
        let reservation = budget.reserve(estimated_bytes)?;
        let (sender, receiver) = mpsc::sync_channel::<Job>(queue_capacity);
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let thread = std::thread::Builder::new()
            .name("l2s1-model".into())
            .spawn(move || {
                let _reservation = reservation;
                match factory() {
                    Err(error) => {
                        let _ = ready_tx.send(Err(error));
                    }
                    Ok(backend) => {
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        run(backend, receiver);
                    }
                }
            })
            .map_err(|e| Error::Backend(e.to_string()))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                sender: Some(sender),
                thread: Some(thread),
                max_request_bytes,
            }),
            other => {
                let _ = thread.join();
                Err(match other {
                    Ok(Err(e)) => e,
                    _ => Error::Backend("backend factory panicked".into()),
                })
            }
        }
    }
    /// Fail immediately when full; callers choose their own admission/retry policy.
    pub fn submit(&self, request: DecisionRequest) -> Result<DecisionTicket> {
        request.validate()?;
        if serde_json::to_vec(&request)
            .map_err(|e| Error::Invalid(e.to_string()))?
            .len()
            > self.max_request_bytes
        {
            return Err(Error::Invalid("worker request byte limit exceeded".into()));
        }
        let (reply, receiver) = mpsc::channel();
        self.sender
            .as_ref()
            .ok_or_else(|| Error::Backend("worker is closed".into()))?
            .try_send(Job {
                admitted_at: Instant::now(),
                request,
                reply,
            })
            .map_err(|e| match e {
                mpsc::TrySendError::Full(_) => Error::Backend("worker queue is full".into()),
                mpsc::TrySendError::Disconnected(_) => Error::Backend("worker is stopped".into()),
            })?;
        Ok(DecisionTicket { reply: receiver })
    }
    /// Drain admitted work, then destroy the backend on its owner thread.
    pub fn close(&mut self) -> Result<()> {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            thread
                .join()
                .map_err(|_| Error::Backend("backend worker panicked".into()))?;
        }
        Ok(())
    }
}
impl Drop for BackendWorker {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

// Only this thread calls the backend, including token estimation and destruction.
fn run_batches<B: BatchDecisionBackend>(
    mut backend: B,
    receiver: mpsc::Receiver<Job>,
    policy: BatchPolicy,
) {
    let mut pending: Option<(Job, usize)> = None;
    loop {
        let (first, mut tokens) = match pending.take() {
            Some(job) => job,
            None => {
                let Ok(job) = receiver.recv() else { break };
                let Some(tokens) = admissible_tokens(&mut backend, &job, policy) else {
                    continue;
                };
                (job, tokens)
            }
        };
        let admitted_at = first.admitted_at;
        let mut jobs = vec![first];
        while jobs.len() < policy.max_requests && tokens < policy.max_input_tokens {
            let remaining = policy.max_wait.saturating_sub(admitted_at.elapsed());
            if remaining.is_zero() {
                break;
            }
            let Ok(job) = receiver.recv_timeout(remaining) else {
                break;
            };
            let Some(next_tokens) = admissible_tokens(&mut backend, &job, policy) else {
                continue;
            };
            // Subtraction avoids integer overflow even for operator-supplied usize::MAX.
            if next_tokens > policy.max_input_tokens - tokens {
                pending = Some((job, next_tokens));
                break;
            }
            tokens += next_tokens;
            jobs.push(job);
        }
        let (requests, replies): (Vec<_>, Vec<_>) =
            jobs.into_iter().map(|job| (job.request, job.reply)).unzip();
        let results = backend.decide_many(&requests);
        if results.len() != replies.len() {
            for reply in replies {
                let _ = reply.send(Err(Error::Backend(
                    "batch backend returned an incorrect result count".into(),
                )));
            }
        } else {
            for (reply, result) in replies.into_iter().zip(results) {
                let _ = reply.send(result);
            }
        }
    }
}

fn admissible_tokens<B: BatchDecisionBackend>(
    backend: &mut B,
    job: &Job,
    policy: BatchPolicy,
) -> Option<usize> {
    match backend.estimate_tokens(&job.request) {
        Ok(tokens) if tokens <= policy.max_input_tokens => Some(tokens),
        result => {
            let error = match result {
                Err(error) => error,
                Ok(_) => Error::Invalid("worker batch token limit exceeded".into()),
            };
            let _ = job.reply.send(Err(error));
            None
        }
    }
}
