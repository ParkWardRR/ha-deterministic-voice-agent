use crate::config::SafetyConfig;
use std::collections::HashSet;

pub struct SafetyGate {
    allowed: HashSet<String>,
    blocked: HashSet<String>,
    confirm_only: HashSet<String>,
    confirm_entity_keywords: Vec<String>,
}

impl Default for SafetyGate {
    fn default() -> Self {
        Self::new()
    }
}

impl SafetyGate {
    pub fn with_config(cfg: &SafetyConfig) -> Self {
        Self {
            allowed: cfg.allowed_domains.iter().cloned().collect(),
            blocked: cfg.blocked_domains.iter().cloned().collect(),
            confirm_only: cfg.confirm_domains.iter().cloned().collect(),
            confirm_entity_keywords: cfg.confirm_entity_keywords.clone(),
        }
    }

    pub fn new() -> Self {
        let allowed = vec![
            "light", "switch", "cover", "climate", "media_player", "fan", "scene", "script",
            "automation", "input_boolean", "input_select", "input_number", "input_text",
            "vacuum", "humidifier", "water_heater", "siren", "remote", "button", "number", "select",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let blocked = vec![
            "shell_command",
            "rest_command",
            "notify",
            "persistent_notification",
            "homeassistant",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        let confirm_only = vec!["lock", "alarm_control_panel", "valve"]
            .into_iter()
            .map(String::from)
            .collect();

        let confirm_entity_keywords = vec!["garage".to_string()];

        Self {
            allowed,
            blocked,
            confirm_only,
            confirm_entity_keywords,
        }
    }

    pub fn is_allowed(&self, domain: &str) -> bool {
        if self.blocked.contains(domain) {
            return false;
        }
        self.allowed.contains(domain)
    }

    pub fn needs_confirmation(&self, domain: &str) -> bool {
        self.confirm_only.contains(domain)
    }

    pub fn needs_entity_confirmation(&self, entity_id: &str) -> bool {
        let lower = entity_id.to_lowercase();
        self.confirm_entity_keywords
            .iter()
            .any(|kw| lower.contains(kw))
    }
}

#[cfg(test)]
#[path = "safety_test.rs"]
mod safety_test;

