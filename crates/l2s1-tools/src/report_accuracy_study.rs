use crate::{
    common::{digest_bytes, read_json, read_jsonl, sha256},
    prepare_accuracy_study::{json_bytes, validate_dataset},
};
use anyhow::{Context, Result, bail, ensure};
use clap::Args as ClapArgs;
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::PathBuf,
};

type Key = (String, String, String, Option<usize>);
const TOL: f64 = 1e-12;

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    data: PathBuf,
    #[arg(long, required = true)]
    predictions: Vec<PathBuf>,
    #[arg(long)]
    split: String,
    #[arg(long)]
    variant: String,
    #[arg(long)]
    output: PathBuf,
    #[arg(long)]
    select: Option<PathBuf>,
}
fn number(v: &Value, name: &str, min: Option<f64>, max: Option<f64>) -> Result<f64> {
    let n = v
        .as_f64()
        .with_context(|| format!("{name} must be finite numeric"))?;
    ensure!(
        n.is_finite() && min.is_none_or(|m| n >= m) && max.is_none_or(|m| n <= m),
        "{name} outside valid range"
    );
    Ok(n)
}
fn text<'a>(v: &'a Value, key: &str) -> Result<&'a str> {
    v[key].as_str().with_context(|| format!("missing {key}"))
}
fn option_ids(decision: &Value) -> Result<Vec<String>> {
    let kind = &decision["kind"];
    if kind["type"] == "binary" {
        return Ok(vec!["false".to_owned(), "true".to_owned()]);
    }
    let options = if kind["type"] == "choice" {
        &kind["options"]
    } else {
        &kind["levels"]
    };
    options
        .as_array()
        .context("options/levels")?
        .iter()
        .map(|v| Ok(text(v, "id")?.to_owned()))
        .collect()
}
fn argmax(scores: &BTreeMap<String, f64>) -> Option<String> {
    let maximum = scores.values().copied().fold(f64::NEG_INFINITY, f64::max);
    let winners = scores
        .iter()
        .filter(|(_, value)| (*value - maximum).abs() <= TOL)
        .collect::<Vec<_>>();
    (winners.len() == 1).then(|| winners[0].0.clone())
}
fn config_key(row: &Value) -> Result<Key> {
    let detail = text(&row["setting"], "prompt_detail")?;
    let layout = text(&row["setting"], "prompt_layout")?;
    ensure!(
        ["minimal", "typed", "typed_examples"].contains(&detail)
            && ["legacy", "state_first"].contains(&layout),
        "unknown prompt detail/layout"
    );
    let kind = text(row, "kind")?;
    ensure!(
        ["single", "ensemble"].contains(&kind),
        "unknown prediction kind"
    );
    let rotation = if kind == "single" {
        Some(
            row["rotation"]
                .as_u64()
                .context("single predictions require a nonnegative integer rotation")?
                as usize,
        )
    } else {
        None
    };
    Ok((
        detail.to_owned(),
        layout.to_owned(),
        kind.to_owned(),
        rotation,
    ))
}
fn describe(key: &Key) -> Value {
    json!({"setting":{"prompt_detail":key.0,"prompt_layout":key.1},"kind":key.2,"rotation":key.3})
}
fn identity_digest(row: &Value, count: usize) -> Result<String> {
    let detail = match text(&row["setting"], "prompt_detail")? {
        "minimal" => "Minimal",
        "typed" => "Typed",
        "typed_examples" => "TypedExamples",
        _ => bail!("unknown detail"),
    };
    let (identities, rotations) = if row["kind"] == "single" {
        (
            vec![row["model_identity"].clone()],
            vec![row["rotation"].clone()],
        )
    } else {
        let identities = row["model_identities"]
            .as_array()
            .context("model identities")?
            .clone();
        let rotations = row["rotations"].as_array().context("rotations")?.clone();
        ensure!(
            identities.len() == count
                && rotations == (0..count).map(|i| json!(i)).collect::<Vec<_>>(),
            "ensemble must contain every distinct code rotation exactly once"
        );
        (identities, rotations)
    };
    let mut normalized = Vec::new();
    for (identity, rotation) in identities.iter().zip(rotations) {
        ensure!(
            identity.as_object().is_some_and(|o| !o.is_empty()),
            "missing model identity"
        );
        let rotation = rotation.as_u64().context("rotation")? as usize;
        let version = text(identity, "prompt_version")?;
        ensure!(!version.is_empty(), "missing model prompt version");
        let base = if let Some((base, suffix)) = version.rsplit_once("/detail-") {
            let (actual_detail, actual_rotation) = suffix
                .split_once("-v1/rotation-")
                .context("malformed prompt version")?;
            ensure!(
                actual_detail == detail
                    && actual_rotation.parse::<usize>()? == rotation
                    && !actual_rotation.starts_with('0')
                    || actual_detail == detail && actual_rotation == "0" && rotation == 0,
                "model identity prompt detail/rotation mismatch"
            );
            base
        } else {
            ensure!(
                detail == "Minimal" && rotation == 0,
                "only minimal rotation zero may omit model prompt suffix"
            );
            version
        };
        ensure!(
            !base.is_empty() && !base.contains("/detail-") && !base.contains("/rotation-"),
            "malformed model prompt identity suffix"
        );
        let mut item = identity.clone();
        item["prompt_version"] = json!(format!("{base}/detail-{detail}-v1"));
        normalized.push(item);
    }
    ensure!(
        normalized.windows(2).all(|w| w[0] == w[1]),
        "ensemble changes model/configuration between rotations"
    );
    Ok(digest_bytes(&json_bytes(&normalized[0])?))
}
fn validate_response(row: &Value, source: &Value, gold: &Value) -> Result<Value> {
    let decision = &source["request"]["decisions"][0];
    let semantic_ids = option_ids(decision)?;
    let kind = text(&decision["kind"], "type")?;
    let elapsed = number(&row["elapsed_ms"], "elapsed_ms", Some(0.), None)?;
    let mut result = json!({"id":row["id"],"expected":gold,"decision_kind":kind,"elapsed_ms":elapsed,
        "raw_top1":null,"selected":null,"probabilities":null,"tied":false,"abstention_reasons":[],"runtime_error":null});
    if !row["error"].is_null() {
        ensure!(
            row.get("response").is_none() && row["error"].as_str().is_some_and(|s| !s.is_empty()),
            "runtime error row must have a nonempty error and no response"
        );
        result["runtime_error"] = row["error"].clone();
        return Ok(result);
    }
    let response = &row["response"];
    let observations = response["results"].as_array().context("response results")?;
    ensure!(
        observations.len() == 1 && observations[0]["id"] == decision["id"],
        "response must match exactly one requested decision"
    );
    let observation = &observations[0];
    let scores = observation["scores"].as_array().context("scores")?;
    let ids = scores
        .iter()
        .map(|s| s["id"].as_str().unwrap_or("").to_owned())
        .collect::<HashSet<_>>();
    ensure!(
        scores.len() == semantic_ids.len() && ids == semantic_ids.iter().cloned().collect(),
        "missing, duplicate or unknown semantic option IDs"
    );
    let mut logits = BTreeMap::new();
    let mut probabilities = BTreeMap::new();
    for score in scores {
        let id = text(score, "id")?.to_owned();
        logits.insert(
            id.clone(),
            number(&score["raw_logit"], "raw_logit", None, None)?,
        );
        probabilities.insert(
            id,
            number(
                &score["option_probability"],
                "option_probability",
                Some(0.),
                Some(1.),
            )?,
        );
    }
    ensure!(
        (probabilities.values().sum::<f64>() - 1.).abs() <= 1e-6,
        "candidate probabilities do not sum to one"
    );
    let mass = number(
        &observation["candidate_mass"],
        "candidate_mass",
        Some(0.),
        Some(1.),
    )?;
    let top = probabilities.values().copied().fold(0., f64::max);
    ensure!(
        (number(
            &observation["top_option_probability"],
            "top probability",
            Some(0.),
            Some(1.)
        )? - top)
            .abs()
            <= 1e-8,
        "top probability does not match candidate scores"
    );
    let value = &observation["value"];
    ensure!(
        value["type"] == decision["kind"]["type"],
        "response decision kind mismatch"
    );
    let selected = if kind == "binary" {
        ensure!(
            value["value"].is_null() || value["value"].is_boolean(),
            "binary selected value must be bool or null"
        );
        ensure!(
            (number(&value["p_true"], "p_true", Some(0.), Some(1.))? - probabilities["true"]).abs()
                <= 1e-8,
            "binary probability mapping mismatch"
        );
        value["value"]
            .as_bool()
            .map(|v| if v { "true" } else { "false" }.to_owned())
    } else {
        if kind == "ordinal" {
            let expectation = decision["kind"]["levels"]
                .as_array()
                .context("levels")?
                .iter()
                .map(|level| {
                    probabilities[text(level, "id").unwrap()] * level["value"].as_f64().unwrap()
                })
                .sum::<f64>();
            let reported = number(
                &value["expected_value"],
                "ordinal expected value",
                None,
                None,
            )?;
            ensure!(
                (reported - expectation).abs() <= 1e-7 + 1e-8 * expectation.abs(),
                "ordinal expectation does not match semantic probabilities"
            );
        }
        value["selected"].as_str().map(str::to_owned)
    };
    ensure!(
        selected.as_ref().is_none_or(|s| semantic_ids.contains(s)),
        "selected unknown semantic option ID"
    );
    let reasons = observation["abstention_reasons"]
        .as_array()
        .context("invalid abstention reasons")?;
    ensure!(
        reasons.iter().all(Value::is_string),
        "invalid abstention reasons"
    );
    let policy = &response["policy"];
    let min_prob = number(
        &policy["min_top_probability"],
        "probability policy",
        Some(0.),
        Some(1.),
    )?;
    let min_mass = number(
        &policy["min_candidate_mass"],
        "mass policy",
        Some(0.),
        Some(1.),
    )?;
    if let Some(selection) = &selected {
        ensure!(
            reasons.is_empty()
                && Some(selection.clone()) == argmax(&probabilities)
                && mass >= min_mass
                && top >= min_prob,
            "accepted result violates its declared score/abstention policy"
        );
    } else {
        ensure!(!reasons.is_empty(), "abstained result lacks a reason");
    }
    let raw = argmax(&logits);
    result["raw_top1"] = json!(raw);
    result["selected"] = json!(selected);
    result["probabilities"] = json!(probabilities);
    result["tied"] = json!(raw.is_none());
    result["abstention_reasons"] = json!(reasons);
    Ok(result)
}
fn mean(values: &[f64]) -> Value {
    if values.is_empty() {
        Value::Null
    } else {
        json!(values.iter().sum::<f64>() / values.len() as f64)
    }
}
fn median(values: &[f64]) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    json!(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    })
}
fn percentile(values: &[f64], q: f64) -> Value {
    if values.is_empty() {
        return Value::Null;
    }
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    json!(values[((values.len() as f64 * q).ceil() as usize).saturating_sub(1)])
}
fn rate(n: usize, d: usize) -> Value {
    if d == 0 {
        Value::Null
    } else {
        json!(n as f64 / d as f64)
    }
}
fn metrics(rows: &[Value]) -> Result<Value> {
    let count = rows.len();
    let scored = rows
        .iter()
        .filter(|r| !r["probabilities"].is_null())
        .collect::<Vec<_>>();
    let accepted = rows
        .iter()
        .filter(|r| !r["selected"].is_null())
        .collect::<Vec<_>>();
    let raw_correct = rows
        .iter()
        .filter(|r| !r["raw_top1"].is_null() && r["raw_top1"] == r["expected"])
        .count();
    let accepted_correct = accepted
        .iter()
        .filter(|r| r["selected"] == r["expected"])
        .count();
    let mut nll = Vec::new();
    let mut brier = Vec::new();
    for row in &scored {
        let gold = text(row, "expected")?;
        let probabilities = row["probabilities"].as_object().context("probabilities")?;
        nll.push(-probabilities[gold].as_f64().unwrap().max(1e-15).ln());
        brier.push(
            probabilities
                .iter()
                .map(|(label, p)| {
                    let error = p.as_f64().unwrap() - f64::from(label == gold);
                    error * error
                })
                .sum::<f64>(),
        );
    }
    let latency = rows
        .iter()
        .map(|r| number(&r["elapsed_ms"], "elapsed_ms", Some(0.), None))
        .collect::<Result<Vec<_>>>()?;
    Ok(
        json!({"count":count,"raw_correct":raw_correct,"raw_top1":rate(raw_correct,count),
        "ties":rows.iter().filter(|r|r["tied"]==true).count(),"accepted":accepted.len(),
        "accepted_correct":accepted_correct,"accepted_wrong":accepted.len()-accepted_correct,
        "abstained":scored.len()-accepted.len(),"not_accepted":count-accepted.len(),
        "coverage":rate(accepted.len(),count),"accepted_accuracy":rate(accepted_correct,accepted.len()),
        "accepted_correct_over_all":rate(accepted_correct,count),
        "runtime_errors":rows.iter().filter(|r|!r["runtime_error"].is_null()).count(),
        "probability_evaluated":scored.len(),"nll_mean":mean(&nll),"brier_mean":mean(&brier),
        "latency_ms":{"total":latency.iter().sum::<f64>(),"p50":median(&latency),"p95":percentile(&latency,0.95)}}),
    )
}
fn rotation_metrics(
    grouped: &BTreeMap<Key, HashMap<String, Value>>,
    sources: &BTreeMap<String, Value>,
) -> Result<Vec<Value>> {
    let mut by_setting =
        BTreeMap::<(String, String), BTreeMap<usize, &HashMap<String, Value>>>::new();
    for (key, rows) in grouped {
        if key.2 == "single" {
            by_setting
                .entry((key.0.clone(), key.1.clone()))
                .or_default()
                .insert(key.3.unwrap(), rows);
        }
    }
    let mut out = Vec::new();
    for ((detail, layout), rotations) in by_setting {
        let setting = json!({"prompt_detail":detail,"prompt_layout":layout});
        if rotations.len() < 2 {
            out.push(json!({"setting":setting,"status":"not_measured; fewer than two rotation configurations"}));
            continue;
        }
        let mut changed = 0;
        let mut failures = 0;
        let mut deviations = Vec::new();
        for (id, source) in sources {
            let count = option_ids(&source["request"]["decisions"][0])?.len();
            let observations=(0..count).map(|rotation|rotations.get(&rotation).and_then(|rows|rows.get(id)).context("rotation probe is missing a required case/rotation; partial probes cannot be compared")).collect::<Result<Vec<_>>>()?;
            if observations
                .iter()
                .any(|row| !row["runtime_error"].is_null())
            {
                failures += 1;
                continue;
            }
            if observations
                .iter()
                .map(|r| r["raw_top1"].clone())
                .collect::<HashSet<_>>()
                .len()
                > 1
            {
                changed += 1;
            }
            let probabilities = observations[0]["probabilities"]
                .as_object()
                .context("probabilities")?;
            let mut maximum = 0_f64;
            for label in probabilities.keys() {
                let values = observations
                    .iter()
                    .map(|r| r["probabilities"][label].as_f64().unwrap())
                    .collect::<Vec<_>>();
                maximum = maximum.max(
                    values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                        - values.iter().copied().fold(f64::INFINITY, f64::min),
                );
            }
            deviations.push(maximum);
        }
        out.push(json!({"setting":setting,"status":"complete_distinct_rotations","logical_cases":sources.len(),
            "compared_cases":deviations.len(),"runtime_error_cases":failures,"changed_top1_cases":changed,
            "changed_top1_rate":rate(changed,deviations.len()),
            "probability_deviation":{"mean":mean(&deviations),"p95":percentile(&deviations,0.95),
                "maximum":deviations.iter().copied().reduce(f64::max)}}));
    }
    Ok(out)
}
fn build_report(args: &Args) -> Result<Value> {
    ensure!(
        ["dev", "calibration", "test"].contains(&args.split.as_str())
            && ["natural", "symbolic"].contains(&args.variant.as_str()),
        "invalid evaluation split/variant"
    );
    let manifest = validate_dataset(&args.data)?;
    let selected = &manifest["splits"][&args.split];
    let source_file = args
        .data
        .join(text(&selected["variants"][&args.variant], "path")?);
    let sources = read_jsonl(&source_file)?
        .into_iter()
        .map(|row| Ok((text(&row, "id")?.to_owned(), row)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let labels = read_json(&args.data.join(text(&manifest["labels"], "path")?))?;
    let mut grouped = BTreeMap::<Key, HashMap<String, Value>>::new();
    let mut identity_by_config = HashMap::<Key, String>::new();
    let mut policies = HashMap::<Key, Value>::new();
    let mut provenance = Vec::new();
    for path in &args.predictions {
        provenance.push(json!({"path":path.to_string_lossy(),"sha256":sha256(path)?}));
        let contents = fs::read_to_string(path)?;
        for line in contents.lines() {
            ensure!(!line.trim().is_empty(), "blank prediction row");
            let row: Value = serde_json::from_str(line)?;
            let key = config_key(&row)?;
            ensure!(
                row["input_sha256"] == selected["variants"][&args.variant]["requests_sha256"],
                "prediction input hash does not match frozen split/variant requests"
            );
            let id = text(&row, "id")?;
            let source = sources
                .get(id)
                .context("prediction ID is outside the declared split")?;
            ensure!(row["group"] == source["group"], "prediction group mismatch");
            let decision = &source["request"]["decisions"][0];
            let count = option_ids(decision)?.len();
            if let Some(rotation) = key.3 {
                ensure!(
                    rotation < count,
                    "rotation outside the distinct candidate code assignments"
                );
            }
            let identity = identity_digest(&row, count)?;
            if let Some(previous) = identity_by_config.insert(key.clone(), identity.clone()) {
                ensure!(
                    previous == identity,
                    "model identity changes within a prediction configuration"
                );
            }
            if row["error"].is_null() {
                let policy = row["response"]["policy"].clone();
                if let Some(previous) = policies.insert(key.clone(), policy.clone()) {
                    ensure!(
                        previous == policy,
                        "decision policy changes within a prediction configuration"
                    );
                }
            }
            let gold = &labels[id][decision["id"].as_str().context("decision ID")?];
            let audit = validate_response(&row, source, gold)?;
            ensure!(
                grouped
                    .entry(key)
                    .or_default()
                    .insert(id.to_owned(), audit)
                    .is_none(),
                "duplicate logical ID within prediction configuration"
            );
        }
    }
    ensure!(!grouped.is_empty(), "no predictions");
    let mut identity_by_setting = HashMap::<(String, String), HashSet<String>>::new();
    for (key, identity) in &identity_by_config {
        identity_by_setting
            .entry((key.0.clone(), key.1.clone()))
            .or_default()
            .insert(identity.clone());
    }
    ensure!(
        identity_by_setting.values().all(|ids| ids.len() == 1),
        "model identity changes across rotations or ensemble"
    );
    let mut configurations = Vec::new();
    for (key, rows) in &grouped {
        let required = sources
            .iter()
            .filter(|(_, source)| {
                key.2 == "ensemble"
                    || key.3.unwrap()
                        < option_ids(&source["request"]["decisions"][0])
                            .unwrap()
                            .len()
            })
            .map(|(id, _)| id.clone())
            .collect::<HashSet<_>>();
        ensure!(
            rows.keys().cloned().collect::<HashSet<_>>() == required,
            "incomplete logical-ID coverage for configuration {key:?}"
        );
        let ordered = selected["ids"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|id| rows.get(id.as_str().unwrap()).cloned())
            .collect::<Vec<_>>();
        let kinds = ordered
            .iter()
            .map(|r| text(r, "decision_kind").unwrap().to_owned())
            .collect::<HashSet<_>>();
        let mut by_kind = Map::new();
        for kind in kinds {
            by_kind.insert(
                kind.clone(),
                metrics(
                    &ordered
                        .iter()
                        .filter(|r| r["decision_kind"] == kind)
                        .cloned()
                        .collect::<Vec<_>>(),
                )?,
            );
        }
        let errors = ordered
            .iter()
            .filter(|r| r["raw_top1"] != r["expected"] || r["selected"] != r["expected"])
            .map(|r| {
                json!({
            "id":r["id"],"expected":r["expected"],"raw_top1":r["raw_top1"],"selected":r["selected"],
            "abstention_reasons":r["abstention_reasons"],"runtime_error":r["runtime_error"]})
            })
            .collect::<Vec<_>>();
        let full = rows.len() == sources.len();
        let mut config = describe(key);
        let extra = json!({"covers_full_split":full,"coverage_scope":if full{"all logical cases"}else{"only cases whose option_count exceeds rotation; ineligible for selection"},
            "identity_sha256":identity_by_config[key],"policy":policies.get(key),"metrics":metrics(&ordered)?,
            "by_decision_kind":by_kind,"errors":errors});
        config
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        configurations.push(config);
    }
    Ok(
        json!({"schema_version":1,"manifest_sha256":sha256(&args.data.join("manifest.json"))?,
        "labels_sha256":manifest["labels"]["sha256"],"split":args.split,"variant":args.variant,
        "logical_cases":sources.len(),"source_sha256":selected["variants"][&args.variant]["sha256"],
        "predictions":provenance,"configurations":configurations,"rotation_consistency":rotation_metrics(&grouped,&sources)?,
        "definitions":{
            "raw_top1":"unique raw_logit argmax before abstention; ties within 1e-12 count as incorrect",
            "probabilities":"candidate-normalized option_probability; NLL/Brier exclude runtime failures and report denominator",
            "nll_floor":1e-15,"brier":"sum of squared class-probability errors, no half factor",
            "coverage":"accepted/all logical cases; runtime failures remain in denominator",
            "accepted_accuracy":"correct/accepted, null when no predictions are accepted",
            "latency":"compute-path inference latency per logical request, not end-to-end request latency; ensemble elapsed_ms sums constituent passes; p95 nearest rank",
            "rotation_probe":"each logical case compared over all of its distinct rotations; incomplete probes rejected",
            "selection":"dev only and full-split configurations only; held-out test must never drive selection"
        }}),
    )
}
fn select_configuration(report: &Value, hash: &str) -> Result<Value> {
    ensure!(
        report["split"] == "dev",
        "configuration selection is allowed on dev only"
    );
    let mut eligible = report["configurations"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["covers_full_split"] == true)
        .collect::<Vec<_>>();
    ensure!(
        !eligible.is_empty(),
        "no full-split configurations eligible for selection"
    );
    eligible.sort_by(|a, b| {
        let key = |c: &Value| {
            let m = &c["metrics"];
            (
                -m["raw_top1"].as_f64().unwrap(),
                -m["accepted_correct_over_all"].as_f64().unwrap(),
                m["latency_ms"]["p50"].as_f64().unwrap(),
                serde_json::to_string(
                    &json!({"setting":c["setting"],"kind":c["kind"],"rotation":c["rotation"]}),
                )
                .unwrap(),
            )
        };
        let a = key(a);
        let b = key(b);
        a.0.total_cmp(&b.0)
            .then(a.1.total_cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
            .then(a.3.cmp(&b.3))
    });
    let winner = eligible[0];
    Ok(
        json!({"schema_version":1,"split":"dev","variant":report["variant"],
        "manifest_sha256":report["manifest_sha256"],"labels_sha256":report["labels_sha256"],
        "report_sha256":hash,"predictions":report["predictions"],
        "selected":{"setting":winner["setting"],"kind":winner["kind"],"rotation":winner["rotation"],
            "identity_sha256":winner["identity_sha256"],"policy":winner["policy"]},
        "metrics":winner["metrics"],
        "criterion":"max raw_top1, max accepted_correct_over_all, min p50 latency, lexical configuration key",
        "protocol":"dev selection only; freeze this file before accessing calibration/test predictions"}),
    )
}
pub fn run(args: Args) -> Result<()> {
    if args.select.is_some() {
        ensure!(args.split == "dev", "--select requires --split dev");
    }
    ensure!(!args.output.exists(), "refusing to overwrite a report file");
    if let Some(select) = &args.select {
        ensure!(
            !select.exists() && *select != args.output,
            "refusing to overwrite a selection file"
        );
    }
    let report = build_report(&args)?;
    let bytes = json_bytes(&report)?;
    let selected = args
        .select
        .as_ref()
        .map(|_| select_configuration(&report, &digest_bytes(&bytes)))
        .transpose()?;
    fs::write(&args.output, bytes)?;
    if let (Some(path), Some(selected)) = (args.select, selected) {
        fs::write(path, json_bytes(&selected)?)?;
    }
    println!(
        "{}",
        json!({"configurations":report["configurations"].as_array().unwrap().len(),"logical_cases":report["logical_cases"]})
    );
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failures_and_abstentions_use_distinct_denominators() {
        let good = json!({"id":"a","expected":"yes","decision_kind":"binary","elapsed_ms":2.,
            "raw_top1":"yes","selected":"yes","probabilities":{"yes":0.8,"no":0.2},"tied":false,"runtime_error":null});
        let failure = json!({"id":"b","expected":"yes","decision_kind":"binary","elapsed_ms":4.,
            "raw_top1":null,"selected":null,"probabilities":null,"tied":false,"runtime_error":"context error"});
        let abstained = json!({"id":"c","expected":"yes","decision_kind":"binary","elapsed_ms":6.,
            "raw_top1":"no","selected":null,"probabilities":{"yes":0.4,"no":0.6},"tied":false,"runtime_error":null});
        let measured = metrics(&[good, failure, abstained]).unwrap();
        assert_eq!(measured["count"], 3);
        assert_eq!(measured["probability_evaluated"], 2);
        assert_eq!(measured["runtime_errors"], 1);
        assert_eq!(measured["abstained"], 1);
        assert_eq!(measured["not_accepted"], 2);
        assert_eq!(measured["accepted_accuracy"], 1.0);
        assert_eq!(measured["coverage"], json!(1.0 / 3.0));
    }
}
