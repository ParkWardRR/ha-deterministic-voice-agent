use crate::config::Config;
use reqwest::Client;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use std::time::Duration;
use url::Url;

#[derive(Debug, Deserialize)]
struct HAState {
    entity_id: String,
    state: Option<String>,
    #[serde(default)]
    attributes: serde_json::Value,
}

const ACTIONABLE_DOMAINS: &[&str] = &[
    "light", "switch", "cover", "climate", "media_player", "fan", "scene", "script",
    "automation", "input_boolean", "input_select", "input_number", "input_text",
    "vacuum", "humidifier", "lock", "alarm_control_panel", "valve", "siren", "remote",
    "water_heater",
];

const SKIP_KEYWORDS: &[&str] = &["identify", "ping", "reset accumulated", "over-load"];

#[derive(Debug)]
struct CatalogItem {
    kind: String,
    domain: String,
    entity_id: String,
    name: String,
    area: String,
    enabled: bool,
    embed_text: String,
}

pub async fn run_sync(cfg: &Config) -> Result<(), Box<dyn std::error::Error>> {
    let start = std::time::Instant::now();

    if cfg.ha_url.is_empty() || cfg.ha_token.is_empty() {
        return Err("HA_URL and HA_TOKEN must be set for sync".into());
    }

    let parsed_url = match Url::parse(&cfg.ha_url) {
        Ok(u) => u,
        Err(e) => return Err(format!("Invalid HA_URL ({}): {}", cfg.ha_url, e).into()),
    };
    let api_url = parsed_url.join("api/states")?;

    tracing::info!("Fetching HA states from {}...", api_url);
    let client = Client::builder().timeout(Duration::from_secs(30)).build()?;
    let resp = client
        .get(api_url.as_str())
        .header("Authorization", format!("Bearer {}", cfg.ha_token))
        .send()
        .await?;

    if !resp.status().is_success() {
        return Err(format!("HA returned status {}", resp.status()).into());
    }

    let states: Vec<HAState> = resp.json().await?;
    tracing::info!("Got {} total entities from HA", states.len());

    let items = filter_actionable(&states);
    tracing::info!("Filtered to {} actionable entities", items.len());

    if items.is_empty() {
        tracing::warn!("No actionable entities found, skipping sync");
        return Ok(());
    }

    // Connect to DB
    let pool = sqlx::postgres::PgPoolOptions::new()
        .max_connections(3)
        .connect(&cfg.pg_dsn)
        .await?;

    // Check hash
    let new_hash = compute_hash(&items);
    let existing: Option<(String,)> = sqlx::query_as("SELECT source_hash FROM catalog_meta WHERE id = 1")
        .fetch_optional(&pool)
        .await?;

    if let Some((old_hash,)) = &existing {
        if old_hash == &new_hash {
            tracing::info!("Entity list unchanged, skipping embedding generation");
            return Ok(());
        }
    }

    // Generate embeddings
    tracing::info!("Generating embeddings...");
    let texts: Vec<&str> = items.iter().map(|i| i.embed_text.as_str()).collect();
    let embeddings = generate_embeddings(cfg, &texts).await?;
    tracing::info!("Generated {} embeddings", embeddings.len());

    // Upsert
    tracing::info!("Upserting into pgvector...");
    upsert_items(&pool, &items, &embeddings).await?;

    // Remove stale
    let current_ids: Vec<&str> = items.iter().map(|i| i.entity_id.as_str()).collect();
    let deleted = remove_stale(&pool, &current_ids).await?;
    if deleted > 0 {
        tracing::info!("Removed {} stale entities", deleted);
    }

    // Update meta
    sqlx::query(
        "INSERT INTO catalog_meta (id, source_hash, last_full_sync) VALUES (1, $1, now()) \
         ON CONFLICT (id) DO UPDATE SET source_hash = EXCLUDED.source_hash, last_full_sync = now()"
    )
    .bind(&new_hash)
    .execute(&pool)
    .await?;

    let elapsed = start.elapsed();
    tracing::info!("Sync complete: {} entities in {:.1}s", items.len(), elapsed.as_secs_f64());
    Ok(())
}

fn filter_actionable(states: &[HAState]) -> Vec<CatalogItem> {
    let domains: std::collections::HashSet<&str> = ACTIONABLE_DOMAINS.iter().copied().collect();
    let mut items = Vec::new();

    for s in states {
        let domain = s.entity_id.split('.').next().unwrap_or("");
        if !domains.contains(domain) {
            continue;
        }

        let name = s.attributes.get("friendly_name")
            .and_then(|v| v.as_str())
            .unwrap_or(&s.entity_id)
            .to_string();

        let name_lower = name.to_lowercase();
        if SKIP_KEYWORDS.iter().any(|kw| name_lower.contains(kw)) {
            continue;
        }

        let kind = match domain {
            "script" => "script",
            "automation" => "automation",
            "scene" => "scene",
            _ => "entity",
        }.to_string();

        let area = s.attributes.get("area")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let enabled = s.state.as_deref() != Some("unavailable");

        let mut embed_parts = vec![name.clone(), domain.to_string()];
        if !area.is_empty() {
            embed_parts.push(area.clone());
        }
        let embed_text = embed_parts.join(" ");

        items.push(CatalogItem {
            kind,
            domain: domain.to_string(),
            entity_id: s.entity_id.clone(),
            name,
            area,
            enabled,
            embed_text,
        });
    }

    items
}

fn compute_hash(items: &[CatalogItem]) -> String {
    let mut ids: Vec<&str> = items.iter().map(|i| i.entity_id.as_str()).collect();
    ids.sort();
    let data = serde_json::to_string(&ids).unwrap_or_default();
    let hash = Sha256::digest(data.as_bytes());
    format!("{:x}", hash)[..16].to_string()
}

async fn generate_embeddings(
    cfg: &Config,
    texts: &[&str],
) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    // Try local ONNX model first
    let onnx_path = format!("{}/all-MiniLM-L6-v2/model.onnx", cfg.model_dir);
    if std::path::Path::new(&onnx_path).exists() {
        tracing::info!("Using local ONNX model for embeddings");
        return crate::embedder_local::batch_embed_onnx(&cfg.model_dir, texts);
    }

    // Fallback to external embedder
    if cfg.embedder_url.is_empty() {
        return Err("No local model and no EMBEDDER_URL configured".into());
    }

    let parsed_url = match Url::parse(&cfg.embedder_url) {
        Ok(u) => u,
        Err(e) => return Err(format!("Invalid EMBEDDER_URL ({}): {}", cfg.embedder_url, e).into()),
    };
    let api_url = parsed_url.join("v1/embeddings")?;

    tracing::info!("Using external embedder at {}", api_url);
    let client = Client::builder().timeout(Duration::from_secs(60)).build()?;
    let mut all_embeddings = Vec::with_capacity(texts.len());

    for text in texts {
        let resp = client
            .post(api_url.as_str())
            .json(&serde_json::json!({"input": text}))
            .send()
            .await?;

        let raw: serde_json::Value = resp.json().await?;

        let embedding = if let Some(data) = raw.get("data").and_then(|d| d.as_array()) {
            data.first()
                .and_then(|f| f.get("embedding"))
                .and_then(|e| e.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        } else {
            raw.get("embedding")
                .and_then(|e| e.as_array())
                .map(|arr| arr.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
        };

        all_embeddings.push(embedding.ok_or("Missing embedding in response")?);
    }

    Ok(all_embeddings)
}

async fn upsert_items(
    pool: &PgPool,
    items: &[CatalogItem],
    embeddings: &[Vec<f32>],
) -> Result<(), sqlx::Error> {
    for (item, emb) in items.iter().zip(embeddings.iter()) {
        let vector = pgvector::Vector::from(emb.clone());
        sqlx::query(
            "INSERT INTO catalog_items (kind, domain, entity_id, name, area, aliases, capabilities, enabled, embedding, updated_at) \
             VALUES ($1, $2, $3, $4, $5, '[]', '{}', $6, $7, now()) \
             ON CONFLICT (entity_id) DO UPDATE SET \
                kind = EXCLUDED.kind, domain = EXCLUDED.domain, name = EXCLUDED.name, \
                area = EXCLUDED.area, enabled = EXCLUDED.enabled, embedding = EXCLUDED.embedding, updated_at = now()"
        )
        .bind(&item.kind)
        .bind(&item.domain)
        .bind(&item.entity_id)
        .bind(&item.name)
        .bind(&item.area)
        .bind(item.enabled)
        .bind(vector)
        .execute(pool)
        .await?;
    }

    Ok(())
}

async fn remove_stale(pool: &PgPool, current_ids: &[&str]) -> Result<u64, sqlx::Error> {
    if current_ids.is_empty() {
        return Ok(0);
    }

    // Build parameterized IN clause
    let placeholders: Vec<String> = (1..=current_ids.len()).map(|i| format!("${}", i)).collect();
    let query_str = format!(
        "DELETE FROM catalog_items WHERE entity_id NOT IN ({})",
        placeholders.join(", ")
    );

    let mut query = sqlx::query(&query_str);
    for id in current_ids {
        query = query.bind(id);
    }

    let result = query.execute(pool).await?;
    Ok(result.rows_affected())
}
