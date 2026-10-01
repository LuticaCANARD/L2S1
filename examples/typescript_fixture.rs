//! Deterministic Rust HTTP boundary for the TypeScript integration tests.
//! Uses real Rust scoring/validation/HTTP, but no model or native inference.
use clap::Parser;
use l2s1::{DecisionPolicy, DecisionRequest, DecisionValue, Error, http::HttpDecisionBackend};
use serde_json::{Value, json};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: String,
    #[arg(long)]
    listen: Option<String>,
    #[arg(long)]
    stdio: bool,
}
struct Fixture;
impl HttpDecisionBackend for Fixture {
    fn capabilities(&self) -> Value {
        json!({"api_version":1,"backend":{"runtime":"rust-fixture","model":"synthetic"},
            "decision_types":["binary","choice","ordinal"],"evidence":"model_scored",
            "media":{"image":{"supported":false,"max_per_decision":0,"max_bytes_each":8388608}},
            "limits":{"max_decisions":128,"max_media":4},
            "batch":{"supported":true,"enabled":true,"execution":"native_parallel","max_requests":128,"max_decisions":128,"max_decisions_per_wave":4,"text":true,"image":false,"mixed_media":false,"reasoning_modes":["direct"]},
            "request_policy":{"supported":true,"target_error_rate":"model-score threshold; not a guaranteed correctness error rate"}})
    }
    fn decide_native_batch_json(
        &mut self,
        requests: &[DecisionRequest],
        images: &[Vec<&[u8]>],
    ) -> l2s1::Result<Vec<Value>> {
        requests
            .iter()
            .zip(images)
            .map(|(request, images)| self.decide_json(request, images))
            .collect()
    }
    fn decide_json(&mut self, request: &DecisionRequest, images: &[&[u8]]) -> l2s1::Result<Value> {
        if !images.is_empty() {
            return Err(Error::Invalid("fixture does not support images".into()));
        }
        let policy = DecisionPolicy::default();
        let results = request.decisions.iter().map(|decision| {
            let count = decision.options().len();
            let mut logits = vec![0.0f32; count + 1];
            logits[1] = 4.0;
            let tokens = (0..count as i32).collect::<Vec<_>>();
            let result = l2s1::score_logits(decision, &logits, &tokens, 17, &policy)?;
            let (value, estimate) = match result.value {
                DecisionValue::Binary { value, p_true } => (json!({"type":"binary","value":value}),json!({"p_true":p_true})),
                DecisionValue::Choice { selected } => (json!({"type":"choice","selected":selected}),json!({})),
                DecisionValue::Ordinal { selected, expected_value } => (json!({"type":"ordinal","selected":selected}),json!({"expected_value":expected_value})),
            };
            Ok(json!({"id":result.id,"value":value,
                "status":if result.abstention_reasons.is_empty() {"selected"} else {"abstained"},
                "abstention_reasons":result.abstention_reasons,
                "evidence":{"type":"model_scored","scores":result.scores,"candidate_mass":result.candidate_mass,
                    "top_option_probability":result.top_option_probability,"entropy_confidence":result.entropy_confidence,
                    "scoring_method":result.scoring_method,"calibration_id":null,"truncated":false,
                    "code_prefix_evaluations":0,"code_evaluated_tokens":0,"estimate":estimate},
                "usage":{"input_tokens":17,"reused_prefix_tokens":0}}))
        }).collect::<l2s1::Result<Vec<_>>>()?;
        Ok(
            json!({"backend":{"runtime":"rust-fixture","model":"synthetic","details":null},"policy":policy,"results":results}),
        )
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args.stdio {
        l2s1::stdio::serve(&mut Fixture)
    } else {
        l2s1::http::serve(
            args.listen.as_deref().expect("--listen or --stdio"),
            &mut Fixture,
        )
    }
}
