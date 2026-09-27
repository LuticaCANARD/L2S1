//! Descriptive confidence ranking; never fits a deployment policy on test labels.
use serde_json::{Value, json};

/// Inputs are validated (maximum candidate probability, raw argmax correctness)
/// pairs. Missing/failed labeled decisions remain in `planned` for coverage.
pub fn risk_coverage(pairs: &[(f64, bool)], planned: usize) -> Value {
    let mut ranked = pairs.to_vec();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    let coverage = |n: usize| {
        if planned == 0 {
            Value::Null
        } else {
            json!(n as f64 / planned as f64)
        }
    };
    let empty = json!({"min_confidence":null,"accepted":0,"errors":0,
        "coverage":coverage(0),"risk":null});
    let mut curve = vec![empty.clone()];
    let budgets = [0.01, 0.05, 0.10];
    let mut best = vec![empty; budgets.len()];
    let mut end = 0;
    let mut errors = 0;
    while end < ranked.len() {
        let confidence = ranked[end].0;
        // A deployable threshold cannot pick just the correct members of a tie.
        while end < ranked.len() && ranked[end].0 == confidence {
            errors += usize::from(!ranked[end].1);
            end += 1;
        }
        let risk = errors as f64 / end as f64;
        let point = json!({"min_confidence":confidence,"accepted":end,"errors":errors,
            "coverage":coverage(end),"risk":risk});
        for (i, budget) in budgets.iter().enumerate() {
            // Empirical risk need not be monotone as the threshold decreases.
            if risk <= *budget {
                best[i] = point.clone();
            }
        }
        curve.push(point);
    }
    let at_budget: Vec<_> = budgets
        .iter()
        .zip(best)
        .map(|(budget, mut point)| {
            point["error_budget"] = json!(budget);
            point
        })
        .collect();
    json!({"planned":planned,"valid":pairs.len(),"curve":curve,
        "at_error_budget":at_budget,
        "scope":"Empirical raw-argmax ranking by maximum candidate probability, with equal-confidence groups kept whole. Coverage uses all planned labeled decisions. Ignores native mass/tie/acceptance gates. Test-label-selected thresholds are descriptive only, not deployment policies or error guarantees; select on independent development data and evaluate once on held-out data."})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_confidence_cannot_be_cherry_picked() {
        let a = risk_coverage(&[(0.9, true), (0.9, false)], 4);
        let b = risk_coverage(&[(0.9, false), (0.9, true)], 4);
        assert_eq!(a, b);
        assert_eq!(a["curve"].as_array().unwrap().len(), 2);
        assert_eq!(a["curve"][1]["coverage"], 0.5);
        assert_eq!(a["curve"][1]["risk"], 0.5);
        assert_eq!(a["at_error_budget"][1]["accepted"], 0);
        assert!(a["at_error_budget"][1]["risk"].is_null());
    }

    #[test]
    fn scans_past_bad_prefixes_and_keeps_failures_in_denominator() {
        let mut pairs = vec![(1.0, false)];
        pairs.extend((0..19).map(|_| (0.8, true)));
        let report = risk_coverage(&pairs, 25);
        assert_eq!(report["at_error_budget"][0]["accepted"], 0);
        assert_eq!(report["at_error_budget"][1]["accepted"], 20);
        assert_eq!(report["at_error_budget"][1]["coverage"], 0.8);
        assert_eq!(report["at_error_budget"][1]["risk"], 0.05);
    }

    #[test]
    fn empty_and_all_failed_are_distinct() {
        let empty = risk_coverage(&[], 0);
        let failed = risk_coverage(&[], 3);
        assert!(empty["curve"][0]["coverage"].is_null());
        assert_eq!(failed["curve"][0]["coverage"], 0.0);
        assert!(failed["curve"][0]["risk"].is_null());
    }
}
