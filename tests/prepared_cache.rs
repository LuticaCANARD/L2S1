// The cache itself has no native dependency. Exercise its bounds and exact
// matching even in builds that do not enable llama.cpp.
#[path = "../src/llama/prepared_cache.rs"]
mod prepared_cache;

use prepared_cache::BoundedTokenCache;

#[test]
fn prompt_boundaries_and_model_identity_never_alias() {
    let mut cache = BoundedTokenCache::new(8, 4096);
    assert!(cache.insert("model-a/template-1".into(), "assistant:".into(), vec![11]));
    assert_eq!(
        cache.get("model-a/template-1", "assistant:"),
        Some(vec![11])
    );
    assert_eq!(cache.get("model-b/template-1", "assistant:"), None);
    assert_eq!(cache.get("model-a/template-2", "assistant:"), None);
    assert_eq!(cache.get("model-a/template-1", "assistant: "), None);
    assert_eq!(cache.get("model-a/template-1", "assistant:\n"), None);
    assert_eq!(cache.metrics().hits, 1);
    assert_eq!(cache.metrics().misses, 4);
}

#[test]
fn complete_preparation_clones_cannot_mutate_future_hits() {
    let mut cache = BoundedTokenCache::new(2, 4096);
    cache.insert(
        "model".into(),
        "state/question".into(),
        (vec![1, 2], vec![7, 9]),
    );
    let mut hit = cache.get("model", "state/question").unwrap();
    hit.0[0] = 99;
    hit.1.clear();
    assert_eq!(
        cache.get("model", "state/question"),
        Some((vec![1, 2], vec![7, 9]))
    );
    cache.clear();
    assert_eq!(cache.get("model", "state/question"), None);
    assert_eq!(cache.metrics().entries, 0);
    assert_eq!(cache.metrics().retained_bytes, 0);
    assert_eq!(cache.metrics().hits, 2);
}

#[test]
fn entry_limit_evicts_fifo_and_replacement_does_not_duplicate_keys() {
    let mut cache = BoundedTokenCache::new(2, 4096);
    cache.insert("m".into(), "a".into(), vec![1]);
    cache.insert("m".into(), "b".into(), vec![2]);
    cache.insert("m".into(), "a".into(), vec![3]);
    assert_eq!(cache.metrics().entries, 2);
    assert_eq!(cache.metrics().evictions, 0);
    cache.insert("m".into(), "c".into(), vec![4]);
    assert_eq!(cache.get("m", "b"), None);
    assert_eq!(cache.get("m", "a"), Some(vec![3]));
    assert_eq!(cache.get("m", "c"), Some(vec![4]));
    assert_eq!(cache.metrics().evictions, 1);
}

#[test]
fn byte_limit_counts_keys_and_allocated_token_capacity() {
    let mut probe = BoundedTokenCache::new(1, 4096);
    probe.insert("m".into(), "a".into(), vec![1]);
    let entry_bytes = probe.metrics().retained_bytes;
    let mut cache = BoundedTokenCache::new(100, entry_bytes * 2);
    cache.insert("m".into(), "a".into(), vec![1]);
    cache.insert("m".into(), "b".into(), vec![2]);
    cache.insert("m".into(), "c".into(), vec![3]);
    assert_eq!(cache.metrics().entries, 2);
    assert_eq!(cache.metrics().retained_bytes, entry_bytes * 2);
    assert_eq!(cache.get("m", "a"), None);

    let mut reserved = Vec::with_capacity(entry_bytes);
    reserved.push(4);
    assert!(!cache.insert("m".into(), "d".into(), reserved));
    assert!(!cache.insert("m".into(), "x".repeat(entry_bytes * 2), vec![4]));
    assert_eq!(cache.metrics().entries, 2);
    assert_eq!(cache.metrics().skipped, 2);
    assert_eq!(cache.get("m", "b"), Some(vec![2]));
    assert_eq!(cache.get("m", "c"), Some(vec![3]));
}

#[test]
fn disabled_storage_and_oversized_replacements_preserve_correctness() {
    for (entries, bytes) in [(0, 4096), (4, 0)] {
        let mut disabled = BoundedTokenCache::new(entries, bytes);
        assert!(!disabled.insert("m".into(), "a".into(), vec![1]));
        assert_eq!(disabled.get("m", "a"), None);
        assert_eq!(disabled.metrics().retained_bytes, 0);
    }
    let mut cache = BoundedTokenCache::new(4, 1024);
    cache.insert("m".into(), "a".into(), vec![1]);
    assert!(!cache.insert("m".into(), "a".into(), vec![2; 1024]));
    assert_eq!(cache.get("m", "a"), Some(vec![1]));
    assert_eq!(cache.metrics().evictions, 0);
}

#[test]
fn mixed_code_widths_share_limits_and_account_for_nested_capacity() {
    use prepared_cache::{CandidateTokens, TokenCacheValue};
    let mut paths = Vec::with_capacity(8);
    let mut path = Vec::with_capacity(32);
    path.push(5);
    paths.push(path);
    let expected = paths.capacity() * std::mem::size_of::<Vec<i32>>()
        + paths[0].capacity() * std::mem::size_of::<i32>();
    let sequences = CandidateTokens::Sequences(paths);
    assert_eq!(sequences.retained_bytes(), expected);
    let mut cache = BoundedTokenCache::new(1, 4096);
    cache.insert(
        "narrow".into(),
        "key".into(),
        CandidateTokens::Single(vec![1]),
    );
    cache.insert("wide".into(), "key".into(), sequences.clone());
    assert_eq!(cache.metrics().entries, 1);
    assert_eq!(cache.metrics().evictions, 1);
    assert_eq!(cache.get("narrow", "key"), None);
    let mut hit = cache.get("wide", "key").unwrap();
    if let CandidateTokens::Sequences(paths) = &mut hit {
        paths[0][0] = 99;
    }
    assert_eq!(cache.get("wide", "key"), Some(sequences));
    let before = cache.metrics().retained_bytes;
    assert!(!cache.insert(
        "wide".into(),
        "huge".into(),
        CandidateTokens::Sequences(vec![Vec::with_capacity(4096)])
    ));
    assert_eq!(cache.metrics().retained_bytes, before);
}
