//! Owned, bounded fixed-schema session for resident applications and HTTP.
use super::*;
use std::collections::VecDeque;

/// Keeps one native KV context plus up to eight prefix snapshots (256 MiB total).
/// Snapshot eviction recomputes the prefix; schema tokens have a separate 4 MiB limit.
/// Cold and warm evaluations use the same explicit prefix/suffix split.
pub struct FixedSchemaBackend {
    backend: LlamaBackend,
    prefixes: VecDeque<(String, Vec<i32>)>,
    retained_bytes: usize,
}
impl FixedSchemaBackend {
    /// Whether this configured backend can preserve the fixed prefix/suffix plan.
    /// The execution mode must already be PrefixReuse. Automatic resident serving
    /// checks this before transferring ownership; explicit construction stays strict.
    pub fn is_compatible(backend: &LlamaBackend) -> bool {
        !unsafe { sd_recurrent_or_hybrid(backend.engine.as_ptr()) }
            && backend.output_head.is_none()
            && backend.calibrations.is_empty()
            && backend.execution_mode == ExecutionMode::PrefixReuse
            && backend.evidence_transfer == EvidenceTransfer::Full
    }

    pub fn new(backend: LlamaBackend) -> Result<Self> {
        if !Self::is_compatible(&backend) {
            return Err(Error::Invalid("fixed-schema requires PrefixReuse, full evidence, a non-recurrent model, and no preexisting calibration/output head; its split plan changes numerics".into()));
        }
        unsafe { sd_clear(backend.engine.as_ptr()) };
        Ok(Self {
            backend,
            prefixes: VecDeque::new(),
            retained_bytes: 0,
        })
    }

    /// Clear all retained tokens and KV, including on tenant/session boundaries.
    pub fn clear(&mut self) {
        unsafe { sd_clear(self.backend.engine.as_ptr()) };
        self.prefixes.clear();
        self.retained_bytes = 0;
    }

    fn prefix(&mut self, decision: &Decision) -> Result<Vec<i32>> {
        let key = serde_json::to_string(decision).map_err(|e| Error::Invalid(e.to_string()))?;
        if let Some(index) = self.prefixes.iter().position(|(k, _)| k == &key) {
            let entry = self.prefixes.remove(index).unwrap();
            let prefix = entry.1.clone();
            self.prefixes.push_back(entry);
            return Ok(prefix);
        }
        // Full prompt tokenization preserves BPE boundaries. Only the common
        // tokens of two distinct state encodings can be static schema tokens.
        // Each actual request is intersected with these tokens too: never assume
        // concatenating separately tokenized strings preserves the prompt.
        let (a, _) = self.backend.prepare(&serde_json::Value::Null, decision)?;
        let (b, _) = self.backend.prepare(&serde_json::json!({}), decision)?;
        let n = a.iter().zip(&b).take_while(|(a, b)| a == b).count();
        let prefix = a[..n].to_vec();
        let bytes = key.len() + prefix.len() * std::mem::size_of::<i32>();
        const LIMIT: usize = 4 * 1024 * 1024;
        while !self.prefixes.is_empty()
            && (self.prefixes.len() >= 64 || self.retained_bytes + bytes > LIMIT)
        {
            let (k, p) = self.prefixes.pop_front().unwrap();
            self.retained_bytes -= k.len() + p.len() * std::mem::size_of::<i32>();
        }
        if bytes <= LIMIT {
            self.retained_bytes += bytes;
            self.prefixes.push_back((key, prefix.clone()));
        }
        Ok(prefix)
    }

    /// Reference path: same split plan, recomputed prefix, zero retained KV hits.
    pub fn decide_cold(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.decide_inner(request, false)
    }

    fn decide_inner(&mut self, request: &DecisionRequest, reuse: bool) -> Result<DecisionResponse> {
        let result = (|| {
            request.validate()?;
            self.backend.check_artifacts(request)?;
            let mut results = Vec::with_capacity(request.decisions.len());
            for decision in &request.decisions {
                if decision.options().len() > 26 {
                    unsafe { sd_clear(self.backend.engine.as_ptr()) };
                    results.push(self.backend.evaluate(&request.state, decision)?);
                    unsafe { sd_clear(self.backend.engine.as_ptr()) };
                    continue;
                }
                let prefix = self.prefix(decision)?;
                let (tokens, _) = self.backend.prepare(&request.state, decision)?;
                let p = prefix
                    .iter()
                    .zip(&tokens)
                    .take(tokens.len().saturating_sub(1))
                    .take_while(|(a, b)| a == b)
                    .count();
                if p == 0 {
                    unsafe { sd_clear(self.backend.engine.as_ptr()) };
                }
                results.push(self.backend.evaluate_planned(
                    &request.state,
                    decision,
                    (p > 0).then_some(p),
                    reuse,
                )?);
            }
            Ok(DecisionResponse {
                backend: self.backend.info_for_request(request),
                policy: self.backend.policy.clone(),
                results,
            })
        })();
        if result.is_err() {
            self.clear();
        }
        result
    }
}
impl DecisionBackend for FixedSchemaBackend {
    fn decide(&mut self, request: &DecisionRequest) -> Result<DecisionResponse> {
        self.decide_inner(request, true)
    }
}
impl crate::http::HttpDecisionBackend for FixedSchemaBackend {
    fn capabilities(&self) -> serde_json::Value {
        let mut caps = self.backend.capabilities();
        caps["artifact_id"] = crate::interoperability::digest(
            format!(
                "{}:fixed-schema-split-v1",
                self.backend.serving_artifact_id()
            )
            .as_bytes(),
        )
        .into();
        caps["batch"]["supported"] = false.into();
        caps["batch"]["enabled"] = false.into();
        caps["prefix_reuse"] = serde_json::json!({"supported":true,"enabled":true,
            "plan":"fixed-schema-split-v1","kv_slots":1,"snapshot_entries":8,"snapshot_bytes":268435456,"schema_entries":64,"schema_bytes":4194304,
            "isolation":"dedicated server instance per trust domain","answers_cached":false});
        caps
    }
    fn decide_json(
        &mut self,
        request: &DecisionRequest,
        images: &[&[u8]],
    ) -> Result<serde_json::Value> {
        let mut response = if !images.is_empty() {
            self.clear();
            self.backend.decide_json(request, images)?
        } else {
            crate::http::contract::local_response_json(self.decide(request)?)?
        };
        response["backend"]["details"]["prefix_plan"] = "fixed-schema-split-v1".into();
        Ok(response)
    }
}
