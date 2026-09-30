use l2s1::{FactSpec, derive_facts};
use serde_json::{Value, json};
fn specs(value: Value) -> Vec<FactSpec> {
    serde_json::from_value(value).unwrap()
}

#[test]
fn exact_arithmetic_and_utc_intervals_preserve_original_fields() {
    let state = json!({"a":9007199254740993i64,"b":-5,"start":86399,"end":86401});
    let s = specs(json!([
        {"name":"sum","op":"add","left":"/a","right":"/b"},
        {"name":"delta","op":"subtract","left":"/a","right":"/b"},
        {"name":"order","op":"compare","left":"/a","right":"/b"},
        {"name":"elapsed","op":"elapsed_seconds","start":"/start","end":"/end"}
    ]));
    let result = derive_facts(&state, &s).unwrap();
    assert_eq!(
        result["_l2s1_facts"]["values"],
        json!({"sum":9007199254740988i64,"delta":9007199254740998i64,"order":"greater","elapsed":{"seconds":2}})
    );
    for (key, value) in state.as_object().unwrap() {
        assert_eq!(&result[key], value);
    }
    assert!(state.get("_l2s1_facts").is_none());
}

#[test]
fn graphs_are_directed_cycle_safe_and_find_shortest_path() {
    let s = specs(
        json!([{ "name":"route", "op":"shortest_path", "edges":"/edges", "from":"/from", "to":"/to" }]),
    );
    let mut state =
        json!({"edges":[["a","b"],["b","a"],["a","c"],["c","d"],["b","c"]],"from":"a","to":"d"});
    assert_eq!(
        derive_facts(&state, &s).unwrap()["_l2s1_facts"]["values"]["route"],
        json!({"reachable":true,"hops":2})
    );
    state["from"] = json!("d");
    state["to"] = json!("a");
    assert_eq!(
        derive_facts(&state, &s).unwrap()["_l2s1_facts"]["values"]["route"],
        json!({"reachable":false,"hops":null})
    );
    state["to"] = json!("d");
    assert_eq!(
        derive_facts(&state, &s).unwrap()["_l2s1_facts"]["values"]["route"]["hops"],
        0
    );
    state["edges"] = json!([["a", "b", "c"]]);
    assert!(derive_facts(&state, &s).is_err());
}

#[test]
fn invalid_facts_fail_atomically_without_silent_coercion() {
    let s = specs(json!([{ "name":"sum", "op":"add", "left":"/a", "right":"/b" }]));
    for state in [
        json!({"a":i64::MAX,"b":1}),
        json!({"a":1.5,"b":2}),
        json!({"a":"1","b":2}),
        json!({"b":2}),
        json!({"a":1,"b":2,"_l2s1_facts":{}}),
        json!(null),
    ] {
        assert!(derive_facts(&state, &s).is_err());
    }
    assert!(derive_facts(&json!({"a":1,"b":2}), &[s[0].clone(), s[0].clone()]).is_err());
    assert_eq!(
        derive_facts(&json!("untouched"), &[]).unwrap(),
        json!("untouched")
    );
    assert!(
        serde_json::from_value::<FactSpec>(
            json!({"name":"x","op":"add","left":"/a","right":"/b","typo":true})
        )
        .is_err()
    );
}

#[test]
fn pointer_escaping_negative_elapsed_and_resource_limits() {
    let state = json!({"a/b":{"~x":7},"end":3});
    let spec =
        specs(json!([{"name":"elapsed","op":"elapsed_seconds","start":"/a~1b/~0x","end":"/end"}]));
    assert_eq!(
        derive_facts(&state, &spec).unwrap()["_l2s1_facts"]["values"]["elapsed"]["seconds"],
        -4
    );
    let invalid = specs(json!([{"name":"x","op":"compare","left":"end","right":"/end"}]));
    assert!(derive_facts(&state, &invalid).is_err());
    let graph = specs(
        json!([{"name":"route","op":"shortest_path","edges":"/edges","from":"/from","to":"/to"}]),
    );
    assert!(
        derive_facts(
            &json!({"from":"a","to":"b","edges":vec![json!(["a","b"]);4097]}),
            &graph
        )
        .is_err()
    );
    assert!(derive_facts(&state, &vec![spec[0].clone(); 65]).is_err());
    let overflow = specs(json!([{"name":"x","op":"subtract","left":"/a","right":"/b"}]));
    assert!(derive_facts(&json!({"a":i64::MIN,"b":1}), &overflow).is_err());
    let bad = specs(json!([{"name":"x","op":"elapsed_seconds","start":"/a","end":"/b"}]));
    assert!(derive_facts(&json!({"a":i64::MIN,"b":i64::MAX}), &bad).is_err());
}
