use crate::cache::EntityCache;
use crate::config::Config;
use crate::conversation::ConversationMemory;
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
use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;

mod cache;
mod config;
mod conversation;
mod embedder_local;
mod ha_ws;
mod ml_clients;
mod models;
mod resolver;
mod safety;
mod simd_ops;
mod sync;

#[derive(Parser)]
#[command(name = "orchestrator", version = "0.2.0")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    
    #[arg(short, long)]
    config: Option<String>,
}

#[derive(Subcommand)]
enum Commands {
    /// Perform a one-off sync of all Home Assistant entities to pgvector
    Sync,
    /// Run the API orchestrator daemon (default)
    Serve,
}

struct AppState {
    resolver: Arc<Resolver>,
    intent_llm: Arc<IntentClient>,
    glm_client: Arc<GLMClient>,
    safety: Arc<SafetyGate>,
    memory: Arc<ConversationMemory>,
    start_time: Instant,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let cfg = Arc::new(Config::load(cli.config.as_deref()));

    match cli.command.unwrap_or(Commands::Serve) {
        Commands::Sync => {
            tracing::info!("Running manual entity sync to pgvector...");
            sync::run_sync(&cfg).await?;
        }
        Commands::Serve => {
            run_server(cfg).await?;
        }
    }

    Ok(())
}

async fn run_server(cfg: Arc<Config>) -> Result<(), Box<dyn std::error::Error>> {
    let start_time = Instant::now();

    let embedder = if cfg.embedder_url.is_empty() {
        None
    } else {
        Some(Arc::new(EmbedderClient::new(cfg.embedder_url.clone())))
    };

    let cache = Arc::new(EntityCache::new(3600));
    
    // Connect to DB and initialize the resolver
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(10)
        .min_connections(2)
        .connect(&cfg.pg_dsn)
        .await?;

    let resolver = Arc::new(Resolver::new(pool, cache.clone(), cfg.clone(), embedder).await?);
    
    let intent_llm = Arc::new(IntentClient::new(cfg.intent_url.clone()));
    let glm_client = Arc::new(GLMClient::new(cfg.glm_url.clone()));
    let safety = Arc::new(SafetyGate::with_config(&cfg.safety));
    let memory = Arc::new(ConversationMemory::new());

    // Spawn HA WebSocket sync thread
    let ws_cfg = cfg.clone();
    let ws_cache = cache.clone();
    tokio::spawn(async move {
        ha_ws::run_ha_websocket(ws_cfg, ws_cache).await;
    });
    
    // Spawn memory cleaner
    let gc_memory = memory.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
            gc_memory.evict_expired();
            cache.evict_expired();
        }
    });

    let state = Arc::new(AppState {
        resolver,
        intent_llm,
        glm_client,
        safety,
        memory,
        start_time,
    });

    let app = Router::new()
        .route("/v1/ha/process", post(handle_process))
        .route("/healthz", get(handle_health))
        .with_state(state);

    tracing::info!("Deterministic agent orchestrator listening on {}", cfg.listen_addr);
    let listener = TcpListener::bind(&cfg.listen_addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn handle_process(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProcessRequest>,
) -> (StatusCode, Json<ProcessResponse>) {
    // Generate an ephemeral conversation ID if one is not provided.
    // In HA, conversation_id is passed; we will assume text here but real integration uses HA payload.
    // For now we use "default" as HA integration sets it via HTTP header ideally, 
    // but we can extend ProcessRequest later.
    let conv_id = "default";
    
    if req.text.trim().is_empty() {
        return ok_resp(ProcessResponse {
            actions: vec![],
            speech: "No text provided.".to_string(),
            needs_confirmation: false,
            needs_clarification: false,
            non_ha_response: None,
        });
    }

    let start = Instant::now();
    tracing::info!("[{}] Processing: {:?}", conv_id, req.text);

    // Step 1: Resolve Entities
    let candidates: Vec<Candidate> = match state.resolver.resolve(&req.text).await {
        Ok(c) => c,
        Err(e) => {
            tracing::error!("Resolver error: {}", e);
            return ok_resp(ProcessResponse {
                actions: vec![],
                speech: "Voice control is temporarily degraded.".to_string(),
                needs_confirmation: false,
                needs_clarification: false,
                non_ha_response: None,
            });
        }
    };

    // Step 2: Follow-up handling (Conversation Memory)
    let candidates = state.memory.resolve_followup(conv_id, &candidates);

    // Step 3: GLM fallback (No entities found and it's a question)
    if candidates.is_empty() {
        if ConversationMemory::is_followup(&req.text) {
             return ok_resp(ProcessResponse {
                 actions: vec![],
                 speech: "I don't know which device you are referring to.".to_string(),
                 needs_confirmation: false,
                 needs_clarification: true,
                 non_ha_response: None,
             });
        }
        
        if should_route_to_glm(&req.text) {
            tracing::info!("Routing to GLM fallback");
            match state.glm_client.ask(&req.text).await {
                Ok(resp) => return ok_resp(ProcessResponse {
                    actions: vec![],
                    speech: resp.clone(),
                    needs_confirmation: false,
                    needs_clarification: false,
                    non_ha_response: Some(resp),
                }),
                Err(e) => {
                    tracing::error!("GLM fallback error: {}", e);
                    return ok_resp(ProcessResponse {
                        actions: vec![],
                        speech: "I couldn't find a matching device or answer your question.".to_string(),
                        needs_confirmation: false,
                        needs_clarification: false,
                        non_ha_response: None,
                    });
                }
            }
        }
        
        return ok_resp(ProcessResponse {
            actions: vec![],
            speech: "I couldn't find a matching device. Please include the room or exact device name.".to_string(),
            needs_confirmation: false,
            needs_clarification: true,
            non_ha_response: None,
        });
    }

    // Step 4: Confidence & Tie detection
    if candidates[0].score < 0.70 {
        let names = extract_top_names(&candidates, 3);
        return ok_resp(ProcessResponse {
            actions: vec![],
            speech: format!("I'm not fully sure which device you meant. Did you mean: {}?", join_names(&names)),
            needs_confirmation: false,
            needs_clarification: true,
            non_ha_response: None,
        });
    }

    if candidates.len() >= 2 && (candidates[0].score - candidates[1].score) < 0.05 {
        let names = extract_top_names(&candidates[..std::cmp::min(candidates.len(), 5)], 5);
        return ok_resp(ProcessResponse {
            actions: vec![],
            speech: format!("Which one did you mean? I found: {}", join_names(&names)),
            needs_confirmation: false,
            needs_clarification: true,
            non_ha_response: None,
        });
    }

    // Step 5: Intent parsing via Local LLM
    let plan = match state.intent_llm.parse(&req.text, &candidates).await {
        Ok(p) => p,
        Err(e) => {
            tracing::error!("Intent LLM error: {}, falling back to single best action", e);
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

    // Step 6: Safety policy execution & History Recording
    let mut candidate_names = HashMap::new();
    for c in &candidates {
        candidate_names.insert(c.entity_id.clone(), c.name.clone());
    }

    let mut final_actions = Vec::new();
    let mut recorded_entities = Vec::new();
    
    for a in plan.actions {
        let domain = domain_of(&a.entity_id);
        if !state.safety.is_allowed(&domain) {
            tracing::info!("BLOCKED: domain {:?} for {}", domain, a.entity_id);
            continue;
        }

        if state.safety.needs_confirmation(&domain) || state.safety.needs_entity_confirmation(&a.entity_id) {
            let target = candidate_names.get(&a.entity_id).cloned().unwrap_or_else(|| a.entity_id.clone());
            return ok_resp(ProcessResponse {
                actions: vec![Action {
                    entity_id: a.entity_id,
                    domain,
                    service: a.service.clone(),
                    service_data: a.service_data,
                }],
                speech: format!("Are you sure you want to {} {}?", a.service, target),
                needs_confirmation: true,
                needs_clarification: false,
                non_ha_response: None,
            });
        }

        recorded_entities.push(a.entity_id.clone());
        final_actions.push(Action {
            entity_id: a.entity_id,
            domain,
            service: a.service,
            service_data: a.service_data,
        });
    }

    if final_actions.is_empty() {
        return ok_resp(ProcessResponse {
            actions: vec![],
            speech: "I can't do that for safety reasons.".to_string(),
            needs_confirmation: false,
            needs_clarification: false,
            non_ha_response: None,
        });
    }

    if final_actions.len() > 5 {
        final_actions.truncate(5);
        recorded_entities.truncate(5);
    }

    // Record turn in context memory
    state.memory.record(conv_id, &req.text, &recorded_entities, &final_actions);

    let speech = if plan.speech.is_empty() {
        "Done.".to_string()
    } else {
        plan.speech
    };

    tracing::info!("Planned {} actions in {:?}", final_actions.len(), start.elapsed());
    ok_resp(ProcessResponse {
        actions: final_actions,
        speech,
        needs_confirmation: false,
        needs_clarification: false,
        non_ha_response: None,
    })
}

async fn handle_health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let mut deps = HashMap::new();
    deps.insert("database".to_string(), "unknown".to_string());
    
    if let Err(e) = state.resolver.ping().await {
        deps.insert("database".to_string(), e.to_string());
    } else {
        deps.insert("database".to_string(), "ok".to_string());
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

fn ok_resp(resp: ProcessResponse) -> (StatusCode, Json<ProcessResponse>) {
    (StatusCode::OK, Json(resp))
}

fn domain_of(entity_id: &str) -> String {
    if let Some(idx) = entity_id.find('.') {
        entity_id[..idx].to_string()
    } else {
        entity_id.to_string()
    }
}

// Same helper functions as before...
fn guess_service(text: &str, domain: &str) -> String {
    let lower = text.to_lowercase();
    if lower.contains("turn off") || lower.contains("shut off") || lower.contains("disable") {
        "turn_off".to_string()
    } else if lower.contains("turn on") || lower.contains("enable") || lower.contains("start") {
        "turn_on".to_string()
    } else if lower.contains("toggle") {
        "toggle".to_string()
    } else if lower.contains("play") || lower.contains("resume") {
        "media_play".to_string()
    } else if lower.contains("pause") || lower.contains("stop") {
        "media_pause".to_string()
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
    let starts = ["what ", "who ", "when ", "where ", "why ", "how ", "tell me ", "explain "];
    starts.iter().any(|q| lower.starts_with(q))
}

fn extract_top_names(candidates: &[Candidate], max: usize) -> Vec<String> {
    let mut names = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for c in candidates {
        if !c.name.is_empty() && seen.insert(c.name.clone()) {
            names.push(c.name.clone());
            if names.len() == max { break; }
        }
    }
    if names.is_empty() { names.push("that device".to_string()); }
    names
}

fn join_names(names: &[String]) -> String {
    if names.is_empty() { return String::new(); }
    if names.len() == 1 { return names[0].clone(); }
    let mut result = names[0..names.len()-1].join(", ");
    result.push_str(", or ");
    result.push_str(&names[names.len()-1]);
    result
}

#[cfg(test)]
#[path = "main_test.rs"]
mod main_test;
