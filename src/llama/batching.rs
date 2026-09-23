use super::*;

impl BatchDecisionBackend for LlamaBackend {
    fn estimate_tokens(&mut self, request: &DecisionRequest) -> Result<usize> {
        let report = self
            .preflight(request)
            .map_err(|e| Error::Invalid(e.to_string()))?;
        report.decisions.iter().try_fold(0usize, |sum, d| {
            sum.checked_add(d.input_tokens)
                .ok_or_else(|| Error::Invalid("batch token estimate overflow".into()))
        })
    }

    fn decide_many(&mut self, requests: &[DecisionRequest]) -> Vec<Result<DecisionResponse>> {
        // Serial modes preserve per-request failure isolation and never replay
        // completed inference. Only explicitly selected Parallel merges work.
        if self.execution_mode != ExecutionMode::Parallel {
            return requests
                .iter()
                .map(|request| self.decide(request))
                .collect();
        }
        let mut outcomes: Vec<Option<Result<DecisionResponse>>> =
            (0..requests.len()).map(|_| None).collect();
        let mut admitted = Vec::new();
        let mut positions = Vec::new();
        for (index, request) in requests.iter().enumerate() {
            match self.estimate_tokens(request) {
                Ok(_) => {
                    admitted.push(request.clone());
                    positions.push(index);
                }
                Err(error) => outcomes[index] = Some(Err(error)),
            }
        }
        if !admitted.is_empty() {
            match self.decide_batch(&admitted) {
                Ok(responses) => {
                    for (index, response) in positions.into_iter().zip(responses) {
                        outcomes[index] = Some(Ok(response));
                    }
                }
                Err(error) => {
                    let message = format!("native batch failed without replay: {error}");
                    for index in positions {
                        outcomes[index] = Some(Err(Error::Backend(message.clone())));
                    }
                }
            }
        }
        outcomes
            .into_iter()
            .map(|outcome| {
                outcome
                    .unwrap_or_else(|| Err(Error::Backend("batch response count mismatch".into())))
            })
            .collect()
    }
}
