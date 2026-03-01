use crate::cache::EntityCache;
use crate::config::Config;
use futures_util::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tokio_tungstenite::{connect_async, tungstenite::protocol::Message};
use url::Url;

#[derive(Serialize)]
struct AuthMessage {
    #[serde(rename = "type")]
    msg_type: String,
    access_token: String,
}

#[derive(Serialize)]
struct SubscribeMessage {
    id: u64,
    #[serde(rename = "type")]
    msg_type: String,
    event_type: String,
}

#[derive(Deserialize, Debug)]
struct HAResponse {
    #[serde(rename = "type")]
    msg_type: String,
    #[serde(default)]
    event: Option<HAEvent>,
}

#[derive(Deserialize, Debug)]
struct HAEvent {
    data: Option<HAEventData>,
}

#[derive(Deserialize, Debug)]
struct HAEventData {
    entity_id: String,
    new_state: Option<HAState>,
}

#[derive(Deserialize, Debug)]
struct HAState {
    state: String,
    attributes: serde_json::Value,
}

pub async fn run_ha_websocket(cfg: Arc<Config>, cache: Arc<EntityCache>) {
    loop {
        if cfg.ha_url.is_empty() || cfg.ha_token.is_empty() {
            tracing::warn!("HA_URL or HA_TOKEN not set, skipping WebSocket real-time sync");
            return;
        }

        let ws_url = cfg.ha_url.replace("http://", "ws://").replace("https://", "wss://");
        let url = format!("{}/api/websocket", ws_url);
        
        tracing::info!("Connecting to HA WebSocket at {}", url);
        match connect_async(Url::parse(&url).unwrap()).await {
            Ok((mut ws_stream, _)) => {
                tracing::info!("WebSocket connected. Authenticating...");
                
                // Wait for auth_required message
                // Simply sending auth first
                let auth_msg = serde_json::to_string(&AuthMessage {
                    msg_type: "auth".to_string(),
                    access_token: cfg.ha_token.clone(),
                }).unwrap();
                
                if let Err(e) = ws_stream.send(Message::Text(auth_msg)).await {
                    tracing::error!("Failed to send auth: {}", e);
                    continue; // Reconnect
                }

                // Subscribe to state_changed events
                let sub_msg = serde_json::to_string(&SubscribeMessage {
                    id: 1,
                    msg_type: "subscribe_events".to_string(),
                    event_type: "state_changed".to_string(),
                }).unwrap();

                if let Err(e) = ws_stream.send(Message::Text(sub_msg)).await {
                    tracing::error!("Failed to subscribe to state_changed: {}", e);
                    continue;
                }

                tracing::info!("Subscribed to state_changed events. Listening for updates...");
                
                while let Some(msg) = ws_stream.next().await {
                    match msg {
                        Ok(Message::Text(text)) => {
                            if let Ok(resp) = serde_json::from_str::<HAResponse>(&text) {
                                if resp.msg_type == "event" {
                                    if let Some(event) = resp.event {
                                        if let Some(data) = event.data {
                                            if let Some(new_state) = data.new_state {
                                                handle_state_update(&cache, &data.entity_id, &new_state);
                                            } else {
                                                // Entity deleted/unavailable
                                                handle_state_update(&cache, &data.entity_id, &HAState {
                                                    state: "unavailable".to_string(),
                                                    attributes: serde_json::json!({}),
                                                });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        Ok(_) => {} // Ignore ping, pong, binary
                        Err(e) => {
                            tracing::error!("WebSocket error: {}", e);
                            break; // Reconnect
                        }
                    }
                }
            }
            Err(e) => {
                tracing::error!("WebSocket connection failed: {}", e);
            }
        }

        tracing::info!("WebSocket disconnected. Reconnecting in 10s...");
        sleep(Duration::from_secs(10)).await;
    }
}

fn handle_state_update(cache: &Arc<EntityCache>, entity_id: &str, state: &HAState) {
    let domain = entity_id.split('.').next().unwrap_or("");
    let name = state.attributes.get("friendly_name")
        .and_then(|v| v.as_str())
        .unwrap_or(entity_id);
    let area = state.attributes.get("area")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    
    let enabled = state.state != "unavailable";
    cache.update_entity(entity_id, name, domain, area, enabled);
    tracing::debug!("WebSocket: Updated cache for {}", entity_id);
}
