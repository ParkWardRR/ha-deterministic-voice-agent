use crate::models::Candidate;
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// A cached entity entry with a timestamp.
#[derive(Debug, Clone)]
struct CachedEntry {
    candidate: Candidate,
    inserted_at: Instant,
}

/// In-memory entity cache backed by DashMap with TTL-based expiry.
pub struct EntityCache {
    entries: Arc<DashMap<String, CachedEntry>>,
    ttl: Duration,
}

impl EntityCache {
    pub fn new(ttl_secs: u64) -> Self {
        Self {
            entries: Arc::new(DashMap::with_capacity(512)),
            ttl: Duration::from_secs(ttl_secs),
        }
    }

    /// Insert or update a candidate in the cache.
    pub fn insert(&self, candidate: Candidate) {
        self.entries.insert(
            candidate.entity_id.clone(),
            CachedEntry {
                candidate,
                inserted_at: Instant::now(),
            },
        );
    }

    /// Bulk-load candidates (e.g., after a sync).
    pub fn load_all(&self, candidates: Vec<Candidate>) {
        self.entries.clear();
        let now = Instant::now();
        for c in candidates {
            self.entries.insert(
                c.entity_id.clone(),
                CachedEntry {
                    candidate: c,
                    inserted_at: now,
                },
            );
        }
    }

    /// Search cache by name substring (case-insensitive).
    pub fn lexical_search(&self, query: &str) -> Vec<Candidate> {
        let lower = query.to_lowercase();
        let now = Instant::now();
        let mut results = Vec::new();

        for entry in self.entries.iter() {
            let e = entry.value();
            if now.duration_since(e.inserted_at) > self.ttl {
                continue; // expired
            }
            let name_lower = e.candidate.name.to_lowercase();
            if name_lower.contains(&lower) || e.candidate.entity_id.contains(&lower) {
                let mut c = e.candidate.clone();
                // Score based on match quality
                if name_lower == lower {
                    c.score = 0.95;
                } else if name_lower.contains(&lower) {
                    c.score = 0.80;
                } else {
                    c.score = 0.60;
                }
                results.push(c);
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        if results.len() > 10 {
            results.truncate(10);
        }
        results
    }

    /// Get a candidate by exact entity_id.
    pub fn get(&self, entity_id: &str) -> Option<Candidate> {
        self.entries.get(entity_id).and_then(|e| {
            if Instant::now().duration_since(e.inserted_at) <= self.ttl {
                Some(e.candidate.clone())
            } else {
                None
            }
        })
    }

    /// Invalidate the entire cache.
    pub fn invalidate(&self) {
        self.entries.clear();
    }

    /// Remove expired entries.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        self.entries.retain(|_, e| now.duration_since(e.inserted_at) <= self.ttl);
    }

    /// Update a single entity (e.g., from a WebSocket state_changed event).
    pub fn update_entity(&self, entity_id: &str, name: &str, domain: &str, area: &str, enabled: bool) {
        if let Some(mut entry) = self.entries.get_mut(entity_id) {
            entry.candidate.name = name.to_string();
            entry.candidate.domain = domain.to_string();
            entry.candidate.area = area.to_string();
            entry.inserted_at = Instant::now();
        } else if enabled {
            self.insert(Candidate {
                item_id: 0,
                kind: "entity".to_string(),
                domain: domain.to_string(),
                entity_id: entity_id.to_string(),
                name: name.to_string(),
                area: area.to_string(),
                score: 0.0,
            });
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidate(id: &str, name: &str, domain: &str) -> Candidate {
        Candidate {
            item_id: 1,
            kind: "entity".to_string(),
            domain: domain.to_string(),
            entity_id: id.to_string(),
            name: name.to_string(),
            area: "".to_string(),
            score: 0.0,
        }
    }

    #[test]
    fn test_cache_insert_and_get() {
        let cache = EntityCache::new(300);
        let c = make_candidate("light.living_room", "Living Room", "light");
        cache.insert(c);
        assert!(cache.get("light.living_room").is_some());
        assert!(cache.get("light.nonexistent").is_none());
    }

    #[test]
    fn test_cache_lexical_search() {
        let cache = EntityCache::new(300);
        cache.insert(make_candidate("light.living_room", "Living Room Light", "light"));
        cache.insert(make_candidate("light.kitchen", "Kitchen Light", "light"));
        cache.insert(make_candidate("switch.tv", "TV Plug", "switch"));

        let results = cache.lexical_search("kitchen");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].entity_id, "light.kitchen");

        let results = cache.lexical_search("light");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_cache_invalidation() {
        let cache = EntityCache::new(300);
        cache.insert(make_candidate("light.test", "Test", "light"));
        assert_eq!(cache.len(), 1);
        cache.invalidate();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_bulk_load() {
        let cache = EntityCache::new(300);
        cache.load_all(vec![
            make_candidate("light.a", "A", "light"),
            make_candidate("light.b", "B", "light"),
        ]);
        assert_eq!(cache.len(), 2);
    }
}
