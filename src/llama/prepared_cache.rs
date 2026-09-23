//! Model-local memoization of exact prompt preparation and candidate mappings.
//!
//! Keys include an explicit preparation identity and the complete input. Callers
//! must include every tokenizer, template, layout, and preparation setting in
//! that identity (or clear the cache when one changes). No normalized or hashed
//! input is used, so different token boundaries cannot alias.

use std::{collections::VecDeque, mem::size_of};

/// Heap bytes retained by a token value; the inline value is counted separately.
pub(crate) trait TokenCacheValue: Clone {
    fn retained_bytes(&self) -> usize;
}

impl TokenCacheValue for Vec<i32> {
    fn retained_bytes(&self) -> usize {
        self.capacity().saturating_mul(size_of::<i32>())
    }
}

impl TokenCacheValue for (Vec<i32>, Vec<i32>) {
    fn retained_bytes(&self) -> usize {
        self.0
            .retained_bytes()
            .saturating_add(self.1.retained_bytes())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub insertions: u64,
    pub evictions: u64,
    pub skipped: u64,
    pub entries: usize,
    /// Retained keys and values, including inline entry storage. This is a cache
    /// accounting bound, not an RSS bound: allocator metadata, queue spare
    /// capacity, temporary lookup keys, and returned clones are excluded.
    pub retained_bytes: usize,
}

struct Entry<T> {
    identity: String,
    input: String,
    value: T,
    bytes: usize,
}

/// Bounded FIFO cache. Reads clone token vectors to fit existing preparation
/// ownership; they avoid compilation and tokenization, not token-buffer copies.
pub(crate) struct BoundedTokenCache<T> {
    entries: VecDeque<Entry<T>>,
    max_entries: usize,
    max_bytes: usize,
    metrics: CacheMetrics,
}

impl<T: TokenCacheValue> BoundedTokenCache<T> {
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries,
            max_bytes,
            metrics: CacheMetrics::default(),
        }
    }

    pub fn get(&mut self, identity: &str, exact_input: &str) -> Option<T> {
        if let Some(entry) = self
            .entries
            .iter()
            .find(|entry| entry.identity == identity && entry.input == exact_input)
        {
            self.metrics.hits = self.metrics.hits.saturating_add(1);
            Some(entry.value.clone())
        } else {
            self.metrics.misses = self.metrics.misses.saturating_add(1);
            None
        }
    }

    /// Insert only complete, successfully validated preparation results. A zero
    /// entry/byte limit disables storage. Oversized values bypass the cache
    /// without evicting useful entries or replacing an existing value.
    pub fn insert(&mut self, identity: String, exact_input: String, value: T) -> bool {
        let bytes = size_of::<Entry<T>>()
            .checked_add(identity.capacity())
            .and_then(|bytes| bytes.checked_add(exact_input.capacity()))
            .and_then(|bytes| bytes.checked_add(value.retained_bytes()));
        let Some(bytes) = bytes.filter(|bytes| *bytes <= self.max_bytes) else {
            self.metrics.skipped = self.metrics.skipped.saturating_add(1);
            return false;
        };
        if self.max_entries == 0 {
            self.metrics.skipped = self.metrics.skipped.saturating_add(1);
            return false;
        }

        // Replacing an exact key renews its insertion position, without charging
        // the same key twice or treating the replacement as a capacity eviction.
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.identity == identity && entry.input == exact_input)
        {
            let replaced = self.entries.remove(index).expect("entry index is valid");
            self.metrics.retained_bytes -= replaced.bytes;
        }
        while self.entries.len() >= self.max_entries
            || self.metrics.retained_bytes > self.max_bytes - bytes
        {
            let evicted = self
                .entries
                .pop_front()
                .expect("capacity requires eviction");
            self.metrics.retained_bytes -= evicted.bytes;
            self.metrics.evictions = self.metrics.evictions.saturating_add(1);
        }
        self.entries.push_back(Entry {
            identity,
            input: exact_input,
            value,
            bytes,
        });
        self.metrics.retained_bytes += bytes;
        self.metrics.entries = self.entries.len();
        self.metrics.insertions = self.metrics.insertions.saturating_add(1);
        true
    }

    /// Invalidate stored preparation; lifetime counters remain available.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.metrics.entries = 0;
        self.metrics.retained_bytes = 0;
    }

    pub fn metrics(&self) -> CacheMetrics {
        self.metrics
    }
}
