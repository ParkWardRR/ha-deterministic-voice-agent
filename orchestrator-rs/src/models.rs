use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ProcessRequest is the input from the HA custom integration.
#[derive(Debug, Deserialize, Serialize)]
pub struct ProcessRequest {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conversation_id: Option<String>,
}

// Action represents a single HA service call.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Action {
    pub entity_id: String,
    pub domain: String,
    pub service: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_data: Option<serde_json::Value>,
}

// ProcessResponse is returned to the HA custom integration.
#[derive(Debug, Deserialize, Serialize)]
pub struct ProcessResponse {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<Action>,
    pub speech: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub needs_confirmation: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub needs_clarification: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub non_ha_response: Option<String>,
}

// HealthResponse reports dependency status.
#[derive(Debug, Deserialize, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub uptime: String,
    pub deps: HashMap<String, String>,
}

// Candidate represents an entity match from the catalog.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    pub item_id: i64,
    pub kind: String,
    pub domain: String,
    pub entity_id: String,
    pub name: String,
    pub area: String,
    pub score: f64, // 1.0 = perfect match, 0.0 = no match
}

// IntentPlan is the parsed plan from the intent LLM.
#[derive(Debug, Deserialize, Serialize)]
pub struct IntentPlan {
    #[serde(default)]
    pub actions: Vec<IntentAction>,
    #[serde(default)]
    pub speech: String,
}

// IntentAction is a single action within a plan.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct IntentAction {
    pub entity_id: String,
    pub service: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_data: Option<serde_json::Value>,
}
