use crate::{
    common::{digest_bytes, read_json, read_jsonl, sha256},
    python_random::PythonRandom,
};
use anyhow::{Context, Result, ensure};
use clap::Args as ClapArgs;
use serde_json::{Map, Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
};

const SPLITS: [&str; 4] = ["train", "dev", "calibration", "test"];
const KINDS: [&str; 3] = ["choice", "binary", "ordinal"];
const VARIANTS: [&str; 2] = ["natural", "symbolic"];
const NOTES: [&str; 3] = [
    "Routine audit entry.",
    "Sensor label was refreshed.",
    "Historical annotation; ignore for this decision.",
];

#[derive(ClapArgs)]
pub struct Args {
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 20260926)]
    seed: u64,
    #[arg(long)]
    verify: bool,
    #[arg(long, default_value_t = 480)]
    train: usize,
    #[arg(long, default_value_t = 120)]
    dev: usize,
    #[arg(long, default_value_t = 120)]
    calibration: usize,
    #[arg(long, default_value_t = 180)]
    test: usize,
}

fn spec() -> Result<Value> {
    Ok(serde_json::from_str(include_str!("accuracy_spec.json"))?)
}

fn decimal(value: &str) -> Result<i64> {
    let negative = value.starts_with('-');
    let value = value.trim_start_matches('-');
    let (whole, fraction) = value.split_once('.').unwrap_or((value, ""));
    let whole = whole.parse::<i64>()?;
    let mut eighths = whole * 8;
    if !fraction.is_empty() {
        let digits = fraction.parse::<i64>()?;
        let scale = 10_i64.pow(fraction.len() as u32);
        ensure!(digits * 8 % scale == 0, "not an eighth: {value}");
        eighths += digits * 8 / scale;
    }
    Ok(if negative { -eighths } else { eighths })
}
fn format_decimal(value: i64) -> String {
    let sign = if value < 0 { "-" } else { "" };
    let absolute = value.abs();
    let whole = absolute / 8;
    let rest = absolute % 8;
    if rest == 0 {
        return format!("{sign}{whole}");
    }
    let fraction = match rest {
        1 => "125",
        2 => "25",
        3 => "375",
        4 => "5",
        5 => "625",
        6 => "75",
        7 => "875",
        _ => unreachable!(),
    };
    format!("{sign}{whole}.{fraction}")
}
fn numeric(value: &str) -> Result<Value> {
    let eighths = decimal(value)?;
    Ok(if eighths % 8 == 0 {
        json!(eighths / 8)
    } else {
        json!(eighths as f64 / 8.0)
    })
}

fn candidate(spec: &Value, split: &str, kind: &str) -> Result<Vec<Value>> {
    let thresholds = spec["thresholds"][split]
        .as_array()
        .context("missing thresholds")?;
    let templates = spec["template_allocation"][split]
        .as_array()
        .context("missing templates")?;
    let domains = spec["domains"].as_array().context("missing domains")?;
    let mut output = Vec::new();
    for (threshold_index, pair) in thresholds.iter().enumerate() {
        let pair = pair.as_array().context("invalid threshold")?;
        let lo_text = pair[0].as_str().context("lower threshold")?;
        let hi_text = pair[1].as_str().context("upper threshold")?;
        let lo = decimal(lo_text)?;
        let hi = decimal(hi_text)?;
        for template in templates {
            for domain_index in 0..domains.len() {
                for step_text in ["0.125", "1"] {
                    let step = decimal(step_text)?;
                    let common = json!({"threshold_id":format!("{split}-threshold-{threshold_index}"),"template_id":template,
                "domain_index":domain_index,"lower":lo_text,"upper":hi_text,"kind":kind,"step":step_text});
                    if kind == "binary" {
                        for (cutoff_name, cutoff) in [("lower", lo), ("upper", hi)] {
                            for true_if_ge in [false, true] {
                                for (relation, offset) in
                                    [("below", -step), ("exact", 0), ("above", step)]
                                {
                                    let value = cutoff + offset;
                                    let mut design = common.clone();
                                    design["value"] = json!(format_decimal(value));
                                    design["cutoff"] = json!(format_decimal(cutoff));
                                    design["true_if_ge"] = json!(true_if_ge);
                                    design["boundary"] = json!(format!("{cutoff_name}-{relation}"));
                                    design["label"] = json!(if (value >= cutoff) == true_if_ge {
                                        "true"
                                    } else {
                                        "false"
                                    });
                                    output.push(design);
                                }
                            }
                        }
                    } else {
                        for (boundary, value) in [
                            ("lower-below", lo - step),
                            ("lower-exact", lo),
                            ("lower-above", lo + step),
                            ("upper-below", hi - step),
                            ("upper-exact", hi),
                            ("upper-above", hi + step),
                        ] {
                            let mut design = common.clone();
                            design["value"] = json!(format_decimal(value));
                            design["boundary"] = json!(boundary);
                            design["label"] = json!(if value < lo {
                                0
                            } else if value < hi {
                                1
                            } else {
                                2
                            });
                            output.push(design);
                        }
                    }
                }
            }
        }
    }
    Ok(output)
}

fn choose_designs(
    spec: &Value,
    split: &str,
    count: usize,
    rng: &mut PythonRandom,
) -> Result<Vec<Value>> {
    ensure!(
        count > 0 && count.is_multiple_of(6),
        "each split count must be a positive multiple of six"
    );
    let mut selected = Vec::new();
    for kind in KINDS {
        let mut strata = BTreeMap::<String, Vec<Value>>::new();
        let mut seen = HashSet::new();
        for design in candidate(spec, split, kind)? {
            let mut identity = design.clone();
            identity.as_object_mut().unwrap().remove("step");
            if !seen.insert(serde_json::to_string(&identity)?) {
                continue;
            }
            let label = if kind == "binary" {
                design["label"].as_str().unwrap().to_owned()
            } else {
                design["label"].to_string()
            };
            strata.entry(label).or_default().push(design);
        }
        let labels = if kind == "binary" {
            vec!["false", "true"]
        } else {
            vec!["0", "1", "2"]
        };
        let per_kind = count / 3;
        for (i, label) in labels.iter().enumerate() {
            let required = per_kind / labels.len() + usize::from(i < per_kind % labels.len());
            let pool = strata.get_mut(*label).context("missing stratum")?;
            ensure!(
                required <= pool.len(),
                "{split}/{kind}: requested count exceeds unique pool"
            );
            rng.shuffle(pool);
            selected.extend(pool.iter().take(required).cloned());
        }
    }
    rng.shuffle(&mut selected);
    Ok(selected)
}

fn interpolate(template: &str, params: &[(&str, &str)]) -> String {
    let mut output = template.to_owned();
    for (name, value) in params {
        output = output.replace(&format!("{{{name}}}"), value);
    }
    output
}

fn render(
    spec: &Value,
    design: &Value,
    id: &str,
    rng: &mut PythonRandom,
    choice_position: usize,
) -> Result<(BTreeMap<String, Value>, Value)> {
    let domain_index = design["domain_index"].as_u64().context("domain index")? as usize;
    let domain_spec = spec["domains"][domain_index].as_array().context("domain")?;
    let domain = domain_spec[0].as_str().context("domain name")?;
    let field = domain_spec[1].as_str().context("field name")?;
    let base_ids = domain_spec[2].as_array().context("base IDs")?;
    let template = &spec["templates"][design["template_id"].as_str().context("template ID")?];
    let group = format!(
        "{}::{}",
        design["threshold_id"].as_str().unwrap(),
        design["template_id"].as_str().unwrap()
    );
    let kind = design["kind"].as_str().context("kind")?;
    let decision_id = format!("{domain}_{kind}_{}", &id[id.len() - 8..]);
    let mut state = Map::new();
    state.insert(
        field.to_owned(),
        numeric(design["value"].as_str().context("value")?)?,
    );
    state.insert(
        "record_tag".to_owned(),
        json!(&digest_bytes(format!("{id}-tag").as_bytes())[..10]),
    );
    state.insert(
        "unrelated_counter".to_owned(),
        json!(rng.randbelow(801) as i64 - 400),
    );
    state.insert("operator_note".to_owned(), json!(NOTES[rng.randbelow(3)]));
    state.insert(
        "auxiliary".to_owned(),
        json!({"revision":rng.randbelow(9)+1,"enabled":rng.randbelow(2)!=0}),
    );
    let semantic_ids = if kind == "binary" {
        vec!["false".to_owned(), "true".to_owned()]
    } else {
        base_ids
            .iter()
            .map(|base| {
                let base = base.as_str().unwrap();
                format!(
                    "{base}-{}",
                    &digest_bytes(format!("{id}{base}").as_bytes())[..6]
                )
            })
            .collect::<Vec<_>>()
    };
    let ordinal_values = [
        json!([-3, 0, 7]),
        json!([-1.5, 2.25, 9]),
        json!([10, 20, 40]),
    ];
    let ordinal_values = &ordinal_values[rng.randbelow(3)];
    let mut order = vec![0, 1, 2];
    rng.shuffle(&mut order);
    if kind == "choice" {
        let correct = design["label"].as_u64().context("choice label")? as usize;
        let current = order.iter().position(|v| *v == correct).unwrap();
        order.swap(current, choice_position);
    }
    let lo = design["lower"].as_str().unwrap();
    let hi = design["upper"].as_str().unwrap();
    let cutoff = design["cutoff"].as_str().unwrap_or("None");
    let params = [
        ("domain", domain),
        ("field", field),
        ("lo", lo),
        ("hi", hi),
        ("cutoff", cutoff),
    ];
    let expressions = [
        format!("{field} < {lo}"),
        format!("({field} >= {lo}) AND ({field} < {hi})"),
        format!("{field} >= {hi}"),
    ];
    let expected = if kind == "binary" {
        design["label"].clone()
    } else {
        json!(semantic_ids[design["label"].as_u64().unwrap() as usize])
    };
    let mut outputs = BTreeMap::new();
    for variant in VARIANTS {
        let criterion = |name: &str, expression: &str| -> Result<String> {
            Ok(if variant == "natural" {
                interpolate(
                    template[name].as_str().context("missing template")?,
                    &params,
                )
            } else {
                interpolate(
                    template["symbolic"]
                        .as_str()
                        .context("missing symbolic template")?,
                    &[("expression", expression)],
                )
            })
        };
        let decision_kind = if kind == "binary" {
            let positive = if design["true_if_ge"] == true {
                "ge"
            } else {
                "lt"
            };
            let negative = if design["true_if_ge"] == true {
                "lt"
            } else {
                "ge"
            };
            let expression = |side: &str| {
                if side == "ge" {
                    format!("{field} >= {cutoff}")
                } else {
                    format!("{field} < {cutoff}")
                }
            };
            json!({"type":"binary","false_label":criterion(negative,&expression(negative))?,"true_label":criterion(positive,&expression(positive))?})
        } else {
            let options = ["low", "mid", "high"]
                .iter()
                .enumerate()
                .map(|(i, name)| {
                    Ok(json!({"id":semantic_ids[i],"criterion":criterion(name,&expressions[i])?}))
                })
                .collect::<Result<Vec<_>>>()?;
            if kind == "choice" {
                json!({"type":"choice","options":order.iter().map(|i|options[*i].clone()).collect::<Vec<_>>()})
            } else {
                json!({"type":"ordinal","levels":options.iter().enumerate().map(|(i,option)|{
                let mut level=option.clone();level["value"]=ordinal_values[i].clone();level}).collect::<Vec<_>>()})
            }
        };
        let intro = if variant == "natural" {
            "intro"
        } else {
            "symbolic_intro"
        };
        let instruction = format!(
            "{} Use only {field}; other state fields are irrelevant.",
            interpolate(template[intro].as_str().context("missing intro")?, &params)
        );
        outputs.insert(variant.to_owned(),json!({"id":id,"group":group,"request":{"state":state,"decisions":[{"id":decision_id,
            "instruction":instruction,"kind":decision_kind}]},"expected":{decision_id.clone():expected}}));
    }
    let mut metadata = design.clone();
    let metadata = metadata.as_object_mut().context("design object")?;
    metadata.remove("label");
    metadata.remove("domain_index");
    metadata.insert("domain".to_owned(), json!(domain));
    metadata.insert("field".to_owned(), json!(field));
    metadata.insert("semantic_ids".to_owned(), json!(semantic_ids));
    Ok((outputs, Value::Object(metadata.clone())))
}

pub fn json_bytes(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    bytes.push(b'\n');
    Ok(bytes)
}

fn prepare(output: &Path, counts: [usize; 4], seed: u64) -> Result<Value> {
    let spec = spec()?;
    let mut rng = PythonRandom::new(seed);
    let mut study = BTreeMap::<String, BTreeMap<String, Vec<Value>>>::new();
    let mut cases = Map::new();
    let mut labels = Map::new();
    for (split, count) in SPLITS.into_iter().zip(counts) {
        let mut rows = BTreeMap::new();
        for variant in VARIANTS {
            rows.insert(variant.to_owned(), Vec::new());
        }
        let mut choice_count = 0;
        for (index, design) in choose_designs(&spec, split, count, &mut rng)?
            .into_iter()
            .enumerate()
        {
            let identity = json!({"seed":seed,"split":split,"design":design,"index":index});
            let id = format!("rule-{}", &digest_bytes(&json_bytes(&identity)?)[..20]);
            let (paired, metadata) = render(&spec, &design, &id, &mut rng, choice_count % 3)?;
            if design["kind"] == "choice" {
                choice_count += 1;
            }
            cases.insert(id.clone(), metadata);
            labels.insert(id.clone(), paired["natural"]["expected"].clone());
            for variant in VARIANTS {
                rows.get_mut(variant).unwrap().push(paired[variant].clone());
            }
        }
        study.insert(split.to_owned(), rows);
    }
    fs::create_dir(output)
        .with_context(|| format!("create fresh study directory {}", output.display()))?;
    let labels_path = output.join("labels.json");
    fs::write(&labels_path, json_bytes(&Value::Object(labels))?)?;
    let mut manifest = json!({
        "schema_version":1,"seed":seed,
        "labels":{"path":"labels.json","sha256":sha256(&labels_path)?},
        "protocol":{
            "study":"frozen_synthetic_threshold_rules_v1",
            "independent_unit":"logical case; natural/symbolic variants are paired, never independent",
            "split_policy":"disjoint threshold values and disjoint wording-template families, not random row splitting",
            "selection":"freeze all splits before model inference; no prompt, hyperparameter or checkpoint selection on test",
            "train":"training only; optional teacher labeling may access training requests only",
            "dev":"prompt, hyperparameter and checkpoint selection only",
            "calibration":"fit score calibration and abstention policy after model selection",
            "test":"final held-out evaluation only; do not fit or select using these rows",
            "reference":"exact Decimal comparisons; eighth/quarter/half fractions; below/exact/above boundaries",
            "scope":"synthetic arithmetic rule-following across six domains; not production accuracy or a real-world generalization claim",
            "prior_cases":"new generated rules; excludes warehouse fields and thresholds 6/24",
            "code_rotations":"none generated here; any training rotations retain source logical ID and group"
        },
        "threshold_allocations":spec["thresholds"],
        "template_allocations":spec["template_allocation"],
        "templates":spec["templates"],
        "splits":{},
        "cases":cases
    });
    for (split, rows) in &study {
        let base = &rows["natural"];
        let ids = base.iter().map(|r| r["id"].clone()).collect::<Vec<_>>();
        let mut groups = base
            .iter()
            .map(|r| r["group"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        groups.sort();
        groups.dedup();
        let mut kinds = Map::new();
        for row in base {
            let kind = manifest["cases"][row["id"].as_str().unwrap()]["kind"]
                .as_str()
                .unwrap();
            *kinds.entry(kind.to_owned()).or_insert(json!(0)) =
                json!(kinds.get(kind).and_then(Value::as_u64).unwrap_or(0) + 1);
        }
        let mut variants = Map::new();
        for (variant, records) in rows {
            let path_name = format!("{split}-{variant}.jsonl");
            let path = output.join(&path_name);
            let mut file = File::create(&path)?;
            for row in records {
                file.write_all(&json_bytes(row)?)?;
            }
            file.flush()?;
            let request_name = format!("{split}-{variant}-requests.jsonl");
            let request_path = output.join(&request_name);
            let mut file = File::create(&request_path)?;
            for row in records {
                file.write_all(&json_bytes(
                    &json!({"id":row["id"],"group":row["group"],"request":row["request"]}),
                )?)?;
            }
            file.flush()?;
            variants.insert(
                variant.clone(),
                json!({"path":path_name,"sha256":sha256(&path)?,
                "requests_path":request_name,"requests_sha256":sha256(&request_path)?}),
            );
        }
        manifest["splits"][split] =
            json!({"count":base.len(),"ids":ids,"groups":groups,"kinds":kinds,"variants":variants});
    }
    fs::write(output.join("manifest.json"), json_bytes(&manifest)?)?;
    Ok(manifest)
}

pub(crate) fn validate_dataset(output: &Path) -> Result<Value> {
    let manifest = read_json(&output.join("manifest.json"))?;
    ensure!(manifest["schema_version"] == 1, "unsupported study schema");
    let label_path = output.join(manifest["labels"]["path"].as_str().context("labels path")?);
    ensure!(
        sha256(&label_path)? == manifest["labels"]["sha256"],
        "labels hash mismatch"
    );
    let labels = read_json(&label_path)?;
    let splits = manifest["splits"].as_object().context("splits")?;
    ensure!(
        splits.len() == 4 && SPLITS.iter().all(|s| splits.contains_key(*s)),
        "missing or unknown study split"
    );
    let mut seen_ids = HashSet::new();
    let mut seen_groups = HashSet::new();
    for (split, entry) in splits {
        let ids = entry["ids"].as_array().context("split IDs")?;
        let groups = entry["groups"].as_array().context("split groups")?;
        ensure!(
            ids.len() == entry["count"].as_u64().unwrap_or(0) as usize,
            "count mismatch"
        );
        for id in ids {
            ensure!(
                seen_ids.insert(id.as_str().context("ID")?.to_owned()),
                "duplicate or cross-split ID"
            );
        }
        for group in groups {
            ensure!(
                seen_groups.insert(group.as_str().context("group")?.to_owned()),
                "cross-split group"
            );
        }
        let group_set = groups.iter().cloned().collect::<HashSet<_>>();
        let mut paired = Vec::new();
        for variant in VARIANTS {
            let meta = &entry["variants"][variant];
            for (path_key, hash_key) in [("path", "sha256"), ("requests_path", "requests_sha256")] {
                let path = output.join(meta[path_key].as_str().context("variant path")?);
                ensure!(
                    sha256(&path)? == meta[hash_key],
                    "{split}/{variant} hash mismatch"
                );
            }
            let rows = read_jsonl(&output.join(meta["path"].as_str().unwrap()))?;
            let requests = read_jsonl(&output.join(meta["requests_path"].as_str().unwrap()))?;
            ensure!(
                rows.iter().map(|r| &r["id"]).eq(ids.iter()),
                "manifest row identity mismatch"
            );
            ensure!(
                rows.iter()
                    .map(|r| r["group"].clone())
                    .collect::<HashSet<_>>()
                    == group_set,
                "manifest row group mismatch"
            );
            ensure!(
                requests.len() == rows.len(),
                "unlabeled request count mismatch"
            );
            for (row, request) in rows.iter().zip(&requests) {
                ensure!(
                    *request
                        == json!({"id":row["id"],"group":row["group"],"request":row["request"]}),
                    "unlabeled request mismatch"
                );
                let id = row["id"].as_str().context("row ID")?;
                ensure!(row["expected"] == labels[id], "label mapping mismatch");
                let design = &manifest["cases"][id];
                let allocation = &manifest["threshold_allocations"][split];
                let threshold = design["threshold_id"].as_str().context("threshold ID")?;
                let threshold_index = threshold
                    .rsplit('-')
                    .next()
                    .context("threshold index")?
                    .parse::<usize>()?;
                ensure!(
                    allocation[threshold_index] == json!([design["lower"], design["upper"]]),
                    "threshold allocation mismatch"
                );
                ensure!(
                    manifest["template_allocations"][split]
                        .as_array()
                        .unwrap()
                        .contains(&design["template_id"]),
                    "template allocation mismatch"
                );
                ensure!(
                    row["group"]
                        == json!(format!(
                            "{threshold}::{}",
                            design["template_id"].as_str().unwrap()
                        )),
                    "group mismatch"
                );
                let decisions = row["request"]["decisions"]
                    .as_array()
                    .context("decisions")?;
                ensure!(
                    decisions.len() == 1
                        && row["expected"]
                            .as_object()
                            .map(|v| v.len() == 1
                                && v.contains_key(decisions[0]["id"].as_str().unwrap_or("")))
                            .unwrap_or(false),
                    "expected exactly one labeled decision"
                );
            }
            paired.push(rows);
        }
        for (natural, symbolic) in paired[0].iter().zip(&paired[1]) {
            for field in ["id", "group", "expected"] {
                ensure!(
                    natural[field] == symbolic[field],
                    "variant pairing mismatch"
                );
            }
            ensure!(
                natural["request"]["state"] == symbolic["request"]["state"],
                "variant state mismatch"
            );
        }
    }
    ensure!(
        labels
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<HashSet<_>>()
            == seen_ids,
        "label coverage mismatch"
    );
    ensure!(
        manifest["cases"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<HashSet<_>>()
            == seen_ids,
        "case coverage mismatch"
    );
    for (i, first) in SPLITS.iter().enumerate() {
        for second in SPLITS.iter().skip(i + 1) {
            let a = manifest["threshold_allocations"][first]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| p.as_array().unwrap())
                .map(|v| decimal(v.as_str().unwrap()).unwrap())
                .collect::<HashSet<_>>();
            let b = manifest["threshold_allocations"][second]
                .as_array()
                .unwrap()
                .iter()
                .flat_map(|p| p.as_array().unwrap())
                .map(|v| decimal(v.as_str().unwrap()).unwrap())
                .collect::<HashSet<_>>();
            ensure!(a.is_disjoint(&b), "threshold leakage");
            let a = manifest["template_allocations"][first]
                .as_array()
                .unwrap()
                .iter()
                .collect::<HashSet<_>>();
            let b = manifest["template_allocations"][second]
                .as_array()
                .unwrap()
                .iter()
                .collect::<HashSet<_>>();
            ensure!(a.is_disjoint(&b), "template leakage");
        }
    }
    Ok(manifest)
}

pub fn run(args: Args) -> Result<()> {
    let manifest = if args.verify {
        validate_dataset(&args.output)?
    } else {
        prepare(
            &args.output,
            [args.train, args.dev, args.calibration, args.test],
            args.seed,
        )?;
        validate_dataset(&args.output)?
    };
    let counts = SPLITS
        .into_iter()
        .map(|split| (split.to_owned(), manifest["splits"][split]["count"].clone()))
        .collect::<Map<_, _>>();
    println!("{}", serde_json::to_string(&Value::Object(counts))?);
    Ok(())
}
