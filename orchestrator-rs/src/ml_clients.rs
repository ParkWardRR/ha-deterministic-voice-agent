use crate::models::{Candidate, IntentPlan};
use reqwest::Client;
use serde_json::json;
use std::time::Duration;

pub struct EmbedderClient {
    base_url: Option<String>,
    client: Client,
}

impl EmbedderClient {
    pub fn new(base_url: String) -> Self {
        let trimmed = base_url.trim().trim_end_matches('/').to_string();
        let base_url_opt = if trimmed.is_empty() {
            None
        } else {
            Some(trimmed)
        };
        Self {
            base_url: base_url_opt,
            client: Client::builder()
                .timeout(Duration::from_secs(20))
                .build()
                .unwrap(),
        }
    }

    pub async fn embed(&self, text: &str) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let base_url = match &self.base_url {
            Some(url) => url,
            None => return Err("Embedder URL not configured".into()),
        };

        let req_body = json!({ "input": text });
        let url = format!("{}/v1/embeddings", base_url);

        let resp = self.client.post(&url).json(&req_body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("embedder status {}: {}", status, body.trim()).into());
        }

        let bytes = resp.bytes().await?;
        // We'll use serde_json for compatibility.
        let raw: serde_json::Value = serde_json::from_slice(&bytes)?;

        // Try OpenAI style
        if let Some(data) = raw.get("data").and_then(|d| d.as_array()) {
            if let Some(first) = data.first() {
                if let Some(embedding) = first.get("embedding").and_then(|e| e.as_array()) {
                    let mut vec = Vec::with_capacity(embedding.len());
                    for v in embedding {
                        if let Some(f) = v.as_f64() {
                            vec.push(f);
                        }
                    }
                    if !vec.is_empty() {
                        return Ok(vec);
                    }
                }
            }
        }

        // Try Simple style
        if let Some(embedding) = raw.get("embedding").and_then(|e| e.as_array()) {
            let mut vec = Vec::with_capacity(embedding.len());
            for v in embedding {
                if let Some(f) = v.as_f64() {
                    vec.push(f);
                }
            }
            if !vec.is_empty() {
                return Ok(vec);
            }
        }

        Err("embedder response missing embedding".into())
    }
}

pub struct GLMClient {
    base_url: String,
    client: Client,
}

impl GLMClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .unwrap(),
        }
    }

    pub async fn ask(&self, text: &str) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let req_body = json!({
            "model": "glm-4.7-flash",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant in a smart home. Reply with only a short final answer (1-2 sentences). Do not include reasoning steps. Do not attempt to control any devices."
                },
                {"role": "user", "content": text}
            ],
            "max_tokens": 200,
            "temperature": 0.3
        });

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self.client.post(&url).json(&req_body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("GLM status {}: {}", status, body.trim()).into());
        }

        let mut raw: serde_json::Value = resp.json().await?;
        
        let choices = raw.get_mut("choices").and_then(|c| c.as_array_mut());
        if let Some(choices) = choices {
            if let Some(first) = choices.first_mut() {
                if let Some(message) = first.get_mut("message") {
                    let mut content = message.get("content").and_then(|c| c.as_str()).unwrap_or("").trim().to_string();
                    if content.is_empty() {
                        content = message.get("reasoning_content").and_then(|c| c.as_str()).unwrap_or("").trim().to_string();
                    }
                    if content.is_empty() {
                        return Ok("I'm not sure how to answer that.".to_string());
                    }
                    return Ok(Self::sanitize_glm_content(text, &content));
                }
            }
        }

        Err("GLM returned no choices".into())
    }

    fn sanitize_glm_content(user_text: &str, content: &str) -> String {
        let lower = content.to_lowercase();
        if lower.contains("the user is asking") || lower.contains("analyze the user's request") {
            let user_lower = user_text.to_lowercase();
            if user_lower.contains("weather") {
                return "I can help with weather. Tell me your city or zip code.".to_string();
            }
            return "I can help with that. Could you rephrase it briefly?".to_string();
        }

        for line in content.lines() {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                return trimmed.to_string();
            }
        }
        content.to_string()
    }
}


pub struct IntentClient {
    base_url: String,
    client: Client,
}

impl IntentClient {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap(),
        }
    }

    pub async fn parse(&self, user_text: &str, candidates: &[Candidate]) -> Result<IntentPlan, Box<dyn std::error::Error + Send + Sync>> {
        let mut candidate_lines = Vec::new();
        for c in candidates {
            let mut line = format!("- entity_id: {}, name: \"{}\", domain: {}", c.entity_id, c.name, c.domain);
            if !c.area.is_empty() {
                line.push_str(&format!(", area: {}", c.area));
            }
            candidate_lines.push(line);
        }

        let system_prompt = r#"You are an intent parser for Home Assistant voice commands. Given user text and candidate entities, output ONLY valid JSON matching this schema:
{"actions": [{"entity_id": "...", "service": "...", "service_data": {}}], "speech": "..."}

Rules:
- entity_id MUST be one of the provided candidates. Never invent entity IDs.
- service must be a valid HA service for the domain (e.g., turn_on, turn_off, toggle for lights/switches; media_play, media_pause for media_player).
- service_data is optional (e.g., {"brightness": 128} for lights).
- speech is a short confirmation message (e.g., "Turning on the basement stairs light").
- For compound commands, include multiple actions.
- Output ONLY the JSON object. No markdown, no explanation."#;

        let user_prompt = format!("User said: \"{}\"\n\nCandidate entities:\n{}\n\nOutput the JSON plan:",
            user_text, candidate_lines.join("\n"));

        let req_body = json!({
            "model": "qwen2.5-1.5b-instruct",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 256,
            "temperature": 0.1,
            "top_p": 1.0
        });

        let url = format!("{}/v1/chat/completions", self.base_url);
        let resp = self.client.post(&url).json(&req_body).send().await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("intent LLM status {}: {}", status, body.trim()).into());
        }

        let raw: serde_json::Value = resp.json().await?;
        
        let content = raw.get("choices")
            .and_then(|c| c.as_array())
            .and_then(|c| c.first())
            .and_then(|f| f.get("message"))
            .and_then(|m| m.get("content"))
            .and_then(|c| c.as_str())
            .unwrap_or("");
        
        if content.is_empty() {
            return Err("intent LLM returned no choices".into());
        }

        let mut content = content.trim();
        if content.starts_with("```json") {
            content = content.trim_start_matches("```json");
        } else if content.starts_with("```") {
            content = content.trim_start_matches("```");
        }
        if content.ends_with("```") {
            content = content.trim_end_matches("```");
        }
        content = content.trim();

        // Here we can use simd-json for performance parsing user intents if we want.
        // For now, serde_json is safe.
        let mut plan: IntentPlan = serde_json::from_str(content).map_err(|e| format!("intent LLM JSON parse: {} (raw: {})", e, content))?;

        // Validate IDs
        let valid_ids: std::collections::HashSet<_> = candidates.iter().map(|c| c.entity_id.clone()).collect();
        plan.actions.retain(|a| valid_ids.contains(&a.entity_id));

        Ok(plan)
    }
}
