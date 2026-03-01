use crate::ml_clients::{EmbedderClient, GLMClient, IntentClient};
use crate::models::{Action, Candidate, HealthResponse, IntentAction, IntentPlan, ProcessRequest, ProcessResponse};
use crate::resolver::Resolver;
use crate::safety::SafetyGate;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;

mod ml_clients;
mod models;
mod resolver;
mod safety;

struct AppState {
    resolver: Arc<Resolver>,
    intent_llm: Arc<IntentClient>,
    glm_client: Arc<GLMClient>,
    safety: Arc<SafetyGate>,
    start_time: Instant,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let start_time = Instant::now();

    let pg_dsn = std::env::var("PG_DSN").unwrap_or_else(|_| "postgres://agent:change-me@localhost:5432/agent?sslmode=disable".to_string());
    let embedder_url = std::env::var("EMBEDDER_URL").unwrap_or_default();
    let intent_url = std::env::var("INTENT_URL").unwrap_or_else(|_| "http://localhost:8081".to_string());
    let glm_url = std::env::var("GLM_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());
    let listen_addr = std::env::var("LISTEN_ADDR").unwrap_or_else(|_| "0.0.0.0:5000".to_string());

    let embedder = if embedder_url.is_empty() {
        None
    } else {
        Some(Arc::new(EmbedderClient::new(embedder_url)))
    };

    let resolver = Arc::new(Resolver::new(&pg_dsn, embedder).await?);
    let intent_llm = Arc::new(IntentClient::new(intent_url));
    let glm_client = Arc::new(GLMClient::new(glm_url));
    let safety = Arc::new(SafetyGate::new());

    let state = Arc::new(AppState {
        resolver,
        intent_llm,
        glm_client,
        safety,
        start_time,
    });

    let app = Router::new()
        .route("/v1/ha/process", post(handle_process))
        .route("/healthz", get(handle_health))
        .with_state(state);

    tracing::info!("Deterministic agent orchestrator listening on {}", listen_addr);
    let listener = TcpListener::bind(listen_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_process(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProcessRequest>,
) -> (StatusCode, Json<ProcessResponse>) {
    if req.text.trim().is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(ProcessResponse {
                actions: vec![],
                speech: "No text provided.".to_string(),
                needs_confirmation: false,
                needs_clarification: false,
                non_ha_response: None,
            }),
        );
    }

    let start = Instant::now();
    tracing::info!("Processing: {:?}", req.text);

    // Step 1: Resolve
    let candidates = match state.resolver.resolve(&req.text).await {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Resolver error: {}", e);
            return ok_json(ProcessResponse {
                actions: vec![],
                speech: "Voice control is temporarily degraded. Please try again.".to_string(),
                needs_confirmation: false,
                needs_clarification: false,
                non_ha_response: None,
            });
        }
    };

    // Step 2: GLM fallback
    if candidates.is_empty() {
        if should_route_to_glm(&req.text) {
            tracing::info!("No entity match for question-like input, routing to GLM fallback");
            match state.glm_client.ask(&req.text).await {
                Ok(resp) => {
                    tracing::info!("GLM fallback responded in {:?}", start.elapsed());
                    return ok_json(ProcessResponse {
                        actions: vec![],
                        speech: resp.clone(),
                        needs_confirmation: false,
                        needs_clarification: false,
                        non_ha_response: Some(resp),
                    });
                }
                Err(e) => {
                    tracing::error!("GLM fallback error: {}", e);
                    return ok_json(ProcessResponse {
                        actions: vec![],
                        speech: "I couldn't find any matching devices or answer your question.".to_string(),
                        needs_confirmation: false,
                        needs_clarification: false,
                        non_ha_response: None,
                    });
                }
            }
        }
        return ok_json(ProcessResponse {
            actions: vec![],
            speech: "I couldn't find a matching device. Please include the room or exact device name.".to_string(),
            needs_confirmation: false,
            needs_clarification: true,
            non_ha_response: None,
        });
    }

    // Step 3: Confidence gate
    if candidates[0].score < 0.70 {
        let names = extract_top_names(&candidates, 3);
        return ok_json(ProcessResponse {
            actions: vec![],
            speech: format!("I'm not fully sure which device you meant. Did you mean: {}?", join_names(&names)),
            needs_confirmation: false,
            needs_clarification: true,
            non_ha_response: None,
        });
    }

    // Step 4: Tie detection
    if candidates.len() >= 2 && (candidates[0].score - candidates[1].score) < 0.05 {
        let max_n = std::cmp::min(candidates.len(), 5);
        let names = extract_top_names(&candidates[..max_n], 5);
        tracing::info!("Clarification needed: {:?}", names);
        return ok_json(ProcessResponse {
            actions: vec![],
            speech: format!("Which one did you mean? I found: {}", join_names(&names)),
            needs_confirmation: false,
            needs_clarification: true,
            non_ha_response: None,
        });
    }

    // Step 5: Intent parsing
    let plan = match state.intent_llm.parse(&req.text, &candidates).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Intent LLM error: {}, falling back to best-guess single action", e);
            IntentPlan {
                actions: vec![IntentAction {
                    entity_id: candidates[0].entity_id.clone(),
                    service: guess_service(&req.text, &candidates[0].domain),
                    service_data: None,
                }],
                speech: format!("OK, {} {}.", guess_service_verb(&req.text), candidates[0].name),
            }
        }
    };

    // Step 6: Safety gate
    let mut candidate_names = HashMap::new();
    for c in &candidates {
        candidate_names.insert(c.entity_id.clone(), c.name.clone());
    }

    let mut actions = Vec::new();
    for a in plan.actions {
        let domain = domain_of(&a.entity_id);
        if !state.safety.is_allowed(&domain) {
            tracing::info!("BLOCKED: domain {:?} for {}", domain, a.entity_id);
            continue;
        }

        if state.safety.needs_confirmation(&domain) || state.safety.needs_entity_confirmation(&a.entity_id) {
            let target = candidate_names.get(&a.entity_id).cloned().unwrap_or_else(|| a.entity_id.clone());
            let act = Action {
                entity_id: a.entity_id,
                domain,
                service: a.service.clone(),
                service_data: a.service_data,
            };
            return ok_json(ProcessResponse {
                actions: vec![act],
                speech: format!("Are you sure you want to {} {}?", a.service, target),
                needs_confirmation: true,
                needs_clarification: false,
                non_ha_response: None,
            });
        }

        actions.push(Action {
            entity_id: a.entity_id,
            domain,
            service: a.service,
            service_data: a.service_data,
        });
    }

    if actions.is_empty() {
        return ok_json(ProcessResponse {
            actions: vec![],
            speech: "I can't do that for safety reasons.".to_string(),
            needs_confirmation: false,
            needs_clarification: false,
            non_ha_response: None,
        });
    }

    if actions.len() > 5 {
        actions.truncate(5);
    }

    let speech = if plan.speech.is_empty() {
        "Done.".to_string()
    } else {
        plan.speech
    };

    tracing::info!("Planned {} actions in {:?}", actions.len(), start.elapsed());
    ok_json(ProcessResponse {
        actions,
        speech,
        needs_confirmation: false,
        needs_clarification: false,
        non_ha_response: None,
    })
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let mut deps = HashMap::new();
    deps.insert("pgvector".to_string(), "unknown".to_string());
    deps.insert("intent_llm".to_string(), "unknown".to_string());
    deps.insert("glm".to_string(), "unknown".to_string());

    if let Err(e) = state.resolver.ping().await {
        deps.insert("pgvector".to_string(), e.to_string());
    } else {
        deps.insert("pgvector".to_string(), "ok".to_string());
    }

    let intent_url = std::env::var("INTENT_URL").unwrap_or_else(|_| "http://localhost:8081".to_string());
    let glm_url = std::env::var("GLM_URL").unwrap_or_else(|_| "http://localhost:8080".to_string());

    let client = reqwest::Client::new();
    match client.get(format!("{}/health", intent_url)).send().await {
        Ok(resp) if resp.status().is_success() => {
            deps.insert("intent_llm".to_string(), "ok".to_string());
        }
        Ok(resp) => {
            deps.insert("intent_llm".to_string(), format!("status {}", resp.status()));
        }
        Err(e) => {
            deps.insert("intent_llm".to_string(), e.to_string());
        }
    }

    match client.get(format!("{}/health", glm_url)).send().await {
        Ok(resp) if resp.status().is_success() => {
            deps.insert("glm".to_string(), "ok".to_string());
        }
        Ok(resp) => {
            deps.insert("glm".to_string(), format!("status {}", resp.status()));
        }
        Err(e) => {
            deps.insert("glm".to_string(), e.to_string());
        }
    }

    let status = if deps.values().all(|v| v == "ok") {
        "ok".to_string()
    } else {
        "degraded".to_string()
    };

    Json(HealthResponse {
        status,
        uptime: format!("{:?}", state.start_time.elapsed()),
        deps,
    })
}

fn ok_json(resp: ProcessResponse) -> (StatusCode, Json<ProcessResponse>) {
    (StatusCode::OK, Json(resp))
}

fn domain_of(entity_id: &str) -> String {
    if let Some(idx) = entity_id.find('.') {
        entity_id[..idx].to_string()
    } else {
        entity_id.to_string()
    }
}

fn guess_service(text: &str, domain: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("turn off") || lower.contains("shut off") || lower.contains("disable") {
        "turn_off".to_string()
    } else if lower.contains("turn on") || lower.contains("enable") || lower.contains("start") {
        "turn_on".to_string()
    } else if lower.contains("toggle") {
        "toggle".to_string()
    } else if lower.contains("dim") || lower.contains("brightness") {
        "turn_on".to_string()
    } else if lower.contains("play") || lower.contains("resume") {
        "media_play".to_string()
    } else if lower.contains("pause") || lower.contains("stop") {
        "media_pause".to_string()
    } else if lower.contains("volume") {
        "volume_set".to_string()
    } else {
        match domain {
            "light" | "switch" | "fan" | "input_boolean" => "toggle".to_string(),
            "media_player" => "media_play".to_string(),
            "scene" => "turn_on".to_string(),
            "automation" => "trigger".to_string(),
            _ => "toggle".to_string(),
        }
    }
}

fn guess_service_verb(text: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("turn off") || lower.contains("shut off") {
        "turning off".to_string()
    } else if lower.contains("turn on") {
        "turning on".to_string()
    } else if lower.contains("toggle") {
        "toggling".to_string()
    } else {
        "controlling".to_string()
    }
}

fn should_route_to_glm(text: &str) -> bool {
    let lower = text.to_lowercase();
    if lower.contains('?') {
        return true;
    }
    let starts = [
        "what ", "who ", "when ", "where ", "why ", "how ", "tell me ", "explain ", "define ",
    ];
    for q in starts {
        if lower.starts_with(q) {
            return true;
        }
    }
    false
}

fn extract_top_names(candidates: &[Candidate], max: usize) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for c in candidates {
        if c.name.is_empty() || seen.contains(&c.name) {
            continue;
        }
        seen.insert(c.name.clone());
        names.push(c.name.clone());
        if names.len() == max {
            break;
        }
    }
    if names.is_empty() {
        names.push("that device".to_string());
    }
    names
}

fn join_names(names: &[String]) -> String {
    if names.is_empty() {
        return "".to_string();
    }
    if names.len() == 1 {
        return names[0].clone();
    }
    let mut result = String::new();
    for (i, name) in names.iter().enumerate() {
        if i > 0 {
            if i == names.len() - 1 {
                result.push_str(", or ");
            } else {
                result.push_str(", ");
            }
        }
        result.push_str(name);
    }
    result
}

#[cfg(test)]
#[path = "main_test.rs"]
mod main_test;

