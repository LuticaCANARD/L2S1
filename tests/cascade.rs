use l2s1::{cascade::*, http::HttpDecisionBackend, *};
use serde_json::{Value, json};
use std::{cell::RefCell, rc::Rc};
struct Mock {
    id: String,
    calls: Rc<RefCell<Vec<Vec<String>>>>,
    fast: bool,
}
impl HttpDecisionBackend for Mock {
    fn capabilities(&self) -> Value {
        json!({"artifact_id":self.id,"backend":{"runtime":"test","model":self.id},"evidence":"discriminative","media":{"image":{"supported":!self.fast,"max_per_decision":1}}})
    }
    fn decide_json(&mut self, r: &DecisionRequest, _: &[&[u8]]) -> Result<Value> {
        self.calls
            .borrow_mut()
            .push(r.decisions.iter().map(|d| d.id.clone()).collect());
        let results: Vec<_> = r
            .decisions
            .iter()
            .map(|d| {
                discriminative::score(
                    d,
                    if self.fast && d.id == "low" {
                        &[0.0, 0.1]
                    } else {
                        &[0.0, 5.0]
                    },
                    1.0,
                    0.0,
                    "test",
                    10,
                )
                .unwrap()
            })
            .collect();
        Ok(json!({"backend":{"runtime":"test","model":self.id},"policy":null,"results":results}))
    }
}
fn decision(id: &str) -> Decision {
    Decision {
        id: id.into(),
        instruction: "yes?".into(),
        kind: DecisionKind::Binary {
            false_label: "no".into(),
            true_label: "yes".into(),
        },
    }
}
fn rule(d: &Decision) -> CascadeRule {
    CascadeRule {
        schema_sha256: schema_id(d).unwrap(),
        min_probability: 0.9,
        calibration_dataset_sha256: "c".repeat(64),
        validation_dataset_sha256: "d".repeat(64),
        validation_items: 20,
        accepted: 10,
        errors: 0,
        max_empirical_error: 0.05,
    }
}
#[test]
fn routes_only_validated_confident_answers_preserving_order() {
    let f = Rc::new(RefCell::new(vec![]));
    let s = Rc::new(RefCell::new(vec![]));
    let ds = vec![decision("yes"), decision("low"), decision("unknown")];
    let policy = CascadePolicy {
        version: 1,
        fast_artifact_id: "a".repeat(64),
        slow_artifact_id: "b".repeat(64),
        rules: ds[..2].iter().map(rule).collect(),
    };
    let mut cascade = CascadeBackend::new(
        Mock {
            id: "a".repeat(64),
            calls: f.clone(),
            fast: true,
        },
        Mock {
            id: "b".repeat(64),
            calls: s.clone(),
            fast: false,
        },
        Some(policy),
    )
    .unwrap();
    let r = DecisionRequest {
        state: json!("test"),
        decisions: ds,
    };
    let out = cascade.decide_json(&r, &[]).unwrap();
    assert_eq!(*f.borrow(), vec![vec!["yes", "low"]]);
    assert_eq!(*s.borrow(), vec![vec!["low", "unknown"]]);
    assert_eq!(out["results"][0]["routing"]["stage"], "fast");
    assert_eq!(out["results"][1]["routing"]["stage"], "slow");
    assert_eq!(out["results"][2]["id"], "unknown");
    cascade.decide_json(&r, &[b"image"]).unwrap();
    assert_eq!(f.borrow().len(), 1);
    assert_eq!(s.borrow()[1].len(), 3);
}
#[test]
fn unconfigured_route_never_calls_fast_and_rejects_bad_artifacts() {
    let f = Rc::new(RefCell::new(vec![]));
    let s = Rc::new(RefCell::new(vec![]));
    let mut c = CascadeBackend::new(
        Mock {
            id: "a".repeat(64),
            calls: f.clone(),
            fast: true,
        },
        Mock {
            id: "b".repeat(64),
            calls: s,
            fast: false,
        },
        None,
    )
    .unwrap();
    c.decide_json(
        &DecisionRequest {
            state: json!(null),
            decisions: vec![decision("yes")],
        },
        &[],
    )
    .unwrap();
    assert!(f.borrow().is_empty());
    let mut p = CascadePolicy {
        version: 1,
        fast_artifact_id: "a".repeat(64),
        slow_artifact_id: "b".repeat(64),
        rules: vec![rule(&decision("yes"))],
    };
    let a = json!({"artifact_id":"a".repeat(64)});
    let b = json!({"artifact_id":"b".repeat(64)});
    assert!(p.validate(&a, &b).is_ok());
    assert!(p.validate(&b, &a).is_err());
    p.rules[0].validation_dataset_sha256 = p.rules[0].calibration_dataset_sha256.clone();
    assert!(p.validate(&a, &b).is_err());
    p.rules[0].validation_dataset_sha256 = "d".repeat(64);
    p.rules[0].errors = 1;
    assert!(p.validate(&a, &b).is_err());
}
