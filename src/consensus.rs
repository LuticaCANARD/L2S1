//! Semantic probability mixtures across explicitly aligned native passes.
use crate::{Decision, DecisionPolicy, DecisionRequest, DecisionResult, Error, Result};
use std::collections::{HashMap, HashSet};

/// Average full candidate probabilities across native option permutations.
///
/// For semantic option `y`, each pass contributes `q(y) = candidate_mass *
/// option_probability(y)`. The output candidate mass is the mean input mass;
/// conditional option probabilities normalize the averaged `q`. This gives more
/// weight to passes that allocate more vocabulary probability to the candidates.
/// It does not manufacture mass 1 or average only conditional probabilities.
///
/// This is not a single native forward pass: `raw_logit` contains the logarithm
/// of the averaged semantic probability. Code/token metadata comes from the
/// first pass for that semantic ID and is representative only. Work counters
/// sum all passes. No head, calibration, or previous mixture is accepted.
///
/// The caller must establish that all passes used the same model, tokenizer,
/// template, prompt detail/layout, device, compute and task state, differing only
/// in option rotation. Results alone cannot authenticate that provenance or
/// prove distinct rotations; this function validates their score/mapping shape.
pub fn score_semantic_mixture(
    decision: &Decision,
    passes: &[DecisionResult],
    policy: &DecisionPolicy,
) -> Result<DecisionResult> {
    DecisionRequest {
        state: serde_json::Value::Null,
        decisions: vec![decision.clone()],
    }
    .validate()?;
    policy.validate()?;
    if passes.len() < 2 {
        return Err(Error::Invalid(
            "semantic mixture requires at least two native passes".into(),
        ));
    }
    let options = decision.options();
    let option_ids: HashSet<_> = options.iter().map(|option| option.id.as_str()).collect();
    let mut log_probabilities = vec![Vec::with_capacity(passes.len()); options.len()];
    let mut first_code_tokens = vec![None; options.len()];
    let mut representative = Vec::new();
    let mut mass_sum = 0.0;
    let mut input_tokens = 0usize;
    let mut reused_tokens = 0usize;
    for (pass_index, pass) in passes.iter().enumerate() {
        if pass.id != decision.id
            || pass.scoring_method != "single_token_conditional_softmax_v1"
            || pass.calibration_id.is_some()
            || pass.truncated
            || pass.scores.len() != options.len()
            || !pass.candidate_mass.is_finite()
            || !(0.0 < pass.candidate_mass && pass.candidate_mass <= 1.0)
            || pass.reused_prefix_tokens > pass.input_tokens
        {
            return Err(Error::Invalid(
                "semantic mixture requires matching, uncalibrated, untruncated native evidence"
                    .into(),
            ));
        }
        let mut by_id = HashMap::new();
        let mut tokens = HashSet::new();
        let mut codes = HashSet::new();
        let mut probability_sum = 0.0;
        for score in &pass.scores {
            let code_index = crate::codes::option_code_index(&score.code, options.len());
            if !option_ids.contains(score.id.as_str())
                || by_id.insert(score.id.as_str(), score).is_some()
                || score.token_id < 0
                || !tokens.insert(score.token_id)
                || code_index.is_none()
                || !codes.insert(score.code.as_str())
                || !score.raw_logit.is_finite()
                || !score.option_probability.is_finite()
                || !(0.0 < score.option_probability && score.option_probability <= 1.0)
            {
                return Err(Error::Invalid(
                    "invalid semantic mixture option probabilities or token mapping".into(),
                ));
            }
            let code_index = code_index.expect("validated code");
            if pass_index == 0 {
                first_code_tokens[code_index] = Some(score.token_id);
            } else if first_code_tokens[code_index] != Some(score.token_id) {
                return Err(Error::Invalid(
                    "semantic mixture code-token mapping changed between passes".into(),
                ));
            }
            probability_sum += score.option_probability;
        }
        if (probability_sum - 1.0).abs() > 1e-8 {
            return Err(Error::Invalid(
                "semantic mixture probabilities must sum to one".into(),
            ));
        }
        for (index, option) in options.iter().enumerate() {
            let score = by_id
                .get(option.id.as_str())
                .ok_or_else(|| Error::Invalid("semantic mixture is missing an option ID".into()))?;
            log_probabilities[index].push(pass.candidate_mass.ln() + score.option_probability.ln());
            if pass_index == 0 {
                representative.push((score.code.clone(), score.token_id));
            }
        }
        mass_sum += pass.candidate_mass;
        input_tokens = input_tokens
            .checked_add(pass.input_tokens)
            .ok_or_else(|| Error::Invalid("semantic mixture input token count overflow".into()))?;
        reused_tokens = reused_tokens
            .checked_add(pass.reused_prefix_tokens)
            .ok_or_else(|| Error::Invalid("semantic mixture reused token count overflow".into()))?;
    }
    let count = passes.len() as f64;
    // Direct mass * probability can underflow for otherwise valid inputs. Pool
    // in log space, then let the existing scorer normalize the semantic options.
    let logits: Vec<_> = log_probabilities
        .iter()
        .map(|values| {
            let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            maximum
                + values
                    .iter()
                    .map(|value| (value - maximum).exp())
                    .sum::<f64>()
                    .ln()
                - count.ln()
        })
        .collect();
    let tokens: Vec<_> = representative.iter().map(|(_, token)| *token).collect();
    let mut result = crate::decision::score_candidate_logits(
        decision,
        &logits,
        &tokens,
        input_tokens,
        mass_sum / count,
        policy,
    )?;
    for (score, (code, _)) in result.scores.iter_mut().zip(representative) {
        score.code = code;
    }
    result.reused_prefix_tokens = reused_tokens;
    result.scoring_method = "semantic_probability_mixture_v1".into();
    Ok(result)
}
