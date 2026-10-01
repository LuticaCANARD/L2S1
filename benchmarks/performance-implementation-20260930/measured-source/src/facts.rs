//! Opt-in, deterministic evidence for structured numeric and graph decisions.
//!
//! Specs reference the original state with JSON pointers, never another fact or
//! a label. Integers use checked i64 arithmetic; timestamp operands must already
//! be UTC Unix seconds. No prose/date parsing or unit inference is performed.
use crate::{Error, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactSpec {
    pub name: String,
    #[serde(flatten)]
    pub operation: FactOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum FactOperation {
    Add {
        left: String,
        right: String,
    },
    Subtract {
        left: String,
        right: String,
    },
    Compare {
        left: String,
        right: String,
    },
    /// `end - start`; negative intervals are valid and retain their sign.
    ElapsedSeconds {
        start: String,
        end: String,
    },
    /// Directed edges are arrays of exactly two nonempty string node IDs.
    ShortestPath {
        edges: String,
        from: String,
        to: String,
    },
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Invalid(message.into())
}

fn at<'a>(state: &'a Value, pointer: &str) -> Result<&'a Value> {
    if !pointer.starts_with('/') {
        return Err(invalid("fact operands must be nonempty JSON pointers"));
    }
    state
        .pointer(pointer)
        .ok_or_else(|| invalid(format!("missing fact operand {pointer}")))
}

fn integer(state: &Value, pointer: &str) -> Result<i64> {
    at(state, pointer)?
        .as_i64()
        .ok_or_else(|| invalid(format!("fact operand {pointer} must be an i64 integer")))
}

impl FactOperation {
    fn evaluate(&self, state: &Value) -> Result<Value> {
        match self {
            Self::Add { left, right } => integer(state, left)?
                .checked_add(integer(state, right)?)
                .map(Value::from)
                .ok_or_else(|| invalid("fact addition overflow")),
            Self::Subtract { left, right } => integer(state, left)?
                .checked_sub(integer(state, right)?)
                .map(Value::from)
                .ok_or_else(|| invalid("fact subtraction overflow")),
            Self::Compare { left, right } => Ok(json!(match integer(state, left)?
                .cmp(&integer(state, right)?)
            {
                std::cmp::Ordering::Less => "less",
                std::cmp::Ordering::Equal => "equal",
                std::cmp::Ordering::Greater => "greater",
            })),
            Self::ElapsedSeconds { start, end } => integer(state, end)?
                .checked_sub(integer(state, start)?)
                .map(|seconds| json!({"seconds":seconds}))
                .ok_or_else(|| invalid("fact elapsed time overflow")),
            Self::ShortestPath { edges, from, to } => {
                let edges = at(state, edges)?
                    .as_array()
                    .ok_or_else(|| invalid("fact edges must be an array"))?;
                if edges.len() > 4096 {
                    return Err(invalid("fact graph exceeds 4096 edges"));
                }
                let node = |value: &Value| -> Result<String> {
                    value
                        .as_str()
                        .filter(|v| !v.is_empty() && v.len() <= 256)
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("fact node IDs must contain 1..256 bytes"))
                };
                let start = node(at(state, from)?)?;
                let target = node(at(state, to)?)?;
                let mut graph: HashMap<String, Vec<String>> = HashMap::new();
                for edge in edges {
                    let pair = edge
                        .as_array()
                        .filter(|v| v.len() == 2)
                        .ok_or_else(|| invalid("fact edges must be pairs"))?;
                    graph
                        .entry(node(&pair[0])?)
                        .or_default()
                        .push(node(&pair[1])?);
                }
                let mut visited = HashSet::from([start.clone()]);
                let mut queue = VecDeque::from([(start, 0usize)]);
                while let Some((current, hops)) = queue.pop_front() {
                    if current == target {
                        return Ok(json!({"reachable":true,"hops":hops}));
                    }
                    if let Some(next) = graph.get(&current) {
                        for node in next {
                            if visited.insert(node.clone()) {
                                queue.push_back((node.clone(), hops + 1));
                            }
                        }
                    }
                }
                Ok(json!({"reachable":false,"hops":null}))
            }
        }
    }
}

/// Return a new state preserving every original field, with audited facts in
/// `_l2s1_facts`. Reject collisions, duplicate names, invalid operands and overflow
/// atomically. Empty specs preserve the state exactly (including non-objects).
/// Facts augment model input; they never select an option or bypass its policy.
pub fn derive_facts(state: &Value, specs: &[FactSpec]) -> Result<Value> {
    if specs.is_empty() {
        return Ok(state.clone());
    }
    if specs.len() > 64 {
        return Err(invalid("at most 64 facts are allowed"));
    }
    let object = state
        .as_object()
        .ok_or_else(|| invalid("derived facts require an object state"))?;
    if object.contains_key("_l2s1_facts") {
        return Err(invalid("state already contains _l2s1_facts"));
    }
    let mut facts = Map::new();
    for spec in specs {
        if spec.name.trim().is_empty() || spec.name.len() > 128 || facts.contains_key(&spec.name) {
            return Err(invalid(
                "fact names must be unique and contain 1..128 bytes",
            ));
        }
        facts.insert(spec.name.clone(), spec.operation.evaluate(state)?);
    }
    let mut result = object.clone();
    result.insert("_l2s1_facts".into(), json!({"version":1,"values":facts}));
    Ok(Value::Object(result))
}
