//! A bounded queue with a backend constructed, used and dropped on its owner thread.
use crate::*;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
    mpsc::{self, SyncSender},
};

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
                    Ok(mut backend) => {
                        if ready_tx.send(Ok(())).is_err() {
                            return;
                        }
                        while let Ok(job) = receiver.recv() {
                            let result = backend.decide(&job.request);
                            let _ = job.reply.send(result);
                        }
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
            .try_send(Job { request, reply })
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
