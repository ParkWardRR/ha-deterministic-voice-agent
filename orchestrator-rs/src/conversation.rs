use crate::models::{Action, Candidate};
use dashmap::DashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

const MAX_HISTORY: usize = 10;
const CONTEXT_TTL_SECS: u64 = 300; // 5 minutes

#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub text: String,
    pub entities: Vec<String>,     // entity_ids used
    pub actions: Vec<Action>,
    pub timestamp: Instant,
}

#[derive(Debug)]
struct ConversationContext {
    turns: Vec<ConversationTurn>,
    last_access: Instant,
}

/// Per-conversation_id context for follow-up command resolution.
pub struct ConversationMemory {
    contexts: Arc<DashMap<String, ConversationContext>>,
    ttl: Duration,
}

impl ConversationMemory {
    pub fn new() -> Self {
        Self {
            contexts: Arc::new(DashMap::with_capacity(64)),
            ttl: Duration::from_secs(CONTEXT_TTL_SECS),
        }
    }

    /// Record a turn in the conversation.
    pub fn record(&self, conversation_id: &str, text: &str, entities: &[String], actions: &[Action]) {
        let turn = ConversationTurn {
            text: text.to_string(),
            entities: entities.to_vec(),
            actions: actions.to_vec(),
            timestamp: Instant::now(),
        };

        let mut entry = self.contexts.entry(conversation_id.to_string()).or_insert_with(|| {
            ConversationContext {
                turns: Vec::new(),
                last_access: Instant::now(),
            }
        });

        entry.last_access = Instant::now();
        entry.turns.push(turn);
        if entry.turns.len() > MAX_HISTORY {
            entry.turns.remove(0);
        }
    }

    /// Get recent entity IDs from conversation context (for resolving "that one", "it", etc.).
    pub fn recent_entities(&self, conversation_id: &str) -> Vec<String> {
        if let Some(ctx) = self.contexts.get(conversation_id) {
            if Instant::now().duration_since(ctx.last_access) > self.ttl {
                return Vec::new();
            }
            // Return entities from the last few turns, most recent first
            let mut entities = Vec::new();
            let mut seen = std::collections::HashSet::new();
            for turn in ctx.turns.iter().rev().take(3) {
                for e in &turn.entities {
                    if seen.insert(e.clone()) {
                        entities.push(e.clone());
                    }
                }
            }
            entities
        } else {
            Vec::new()
        }
    }

    /// Check if the text is a follow-up command that references a previous context.
    pub fn is_followup(text: &str) -> bool {
        let lower = text.to_lowercase();
        let followup_markers = [
            "that one", "that", "it", "them", "those", "the same",
            "too", "also", "again", "as well",
        ];
        followup_markers.iter().any(|m| lower.contains(m))
    }

    /// Resolve follow-up references: if the user says "turn that off too",
    /// look up recent entities from conversation memory.
    pub fn resolve_followup(
        &self,
        conversation_id: &str,
        candidates: &[Candidate],
    ) -> Vec<Candidate> {
        if candidates.is_empty() {
            let recent = self.recent_entities(conversation_id);
            if recent.is_empty() {
                return Vec::new();
            }
            // Return recent entities as candidates
            return recent.iter().map(|eid| {
                Candidate {
                    item_id: 0,
                    kind: "entity".to_string(),
                    domain: eid.split('.').next().unwrap_or("unknown").to_string(),
                    entity_id: eid.clone(),
                    name: eid.clone(),
                    area: String::new(),
                    score: 0.85,
                }
            }).collect();
        }
        candidates.to_vec()
    }

    /// Evict expired conversations.
    pub fn evict_expired(&self) {
        let now = Instant::now();
        self.contexts.retain(|_, ctx| now.duration_since(ctx.last_access) <= self.ttl);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversation_record_and_recall() {
        let mem = ConversationMemory::new();
        mem.record("conv1", "turn on the kitchen light", &["light.kitchen".to_string()], &[]);
        let recent = mem.recent_entities("conv1");
        assert_eq!(recent, vec!["light.kitchen"]);
    }

    #[test]
    fn test_followup_detection() {
        assert!(ConversationMemory::is_followup("turn that off too"));
        assert!(ConversationMemory::is_followup("do it again"));
        assert!(!ConversationMemory::is_followup("turn on the kitchen light"));
    }

    #[test]
    fn test_conversation_eviction() {
        let mem = ConversationMemory::new();
        mem.record("conv1", "test", &[], &[]);
        assert!(mem.recent_entities("conv1").is_empty()); // no entities recorded
        mem.evict_expired();
    }

    #[test]
    fn test_followup_resolution() {
        let mem = ConversationMemory::new();
        mem.record("conv1", "turn on the kitchen light", &["light.kitchen".to_string()], &[]);

        let empty_candidates: Vec<Candidate> = vec![];
        let resolved = mem.resolve_followup("conv1", &empty_candidates);
        assert_eq!(resolved.len(), 1);
        assert_eq!(resolved[0].entity_id, "light.kitchen");
    }
}
