use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    #[serde(default = "default_listen")]
    pub listen_addr: String,

    #[serde(default = "default_pg_dsn")]
    pub pg_dsn: String,

    #[serde(default = "default_db_min_conns")]
    pub db_min_conns: u32,

    #[serde(default = "default_db_max_conns")]
    pub db_max_conns: u32,

    #[serde(default = "default_db_idle_timeout")]
    pub db_idle_timeout: u64,

    #[serde(default)]
    pub embedder_url: String,

    #[serde(default = "default_intent_url")]
    pub intent_url: String,

    #[serde(default = "default_glm_url")]
    pub glm_url: String,

    #[serde(default)]
    pub ha_url: String,

    #[serde(default)]
    pub ha_token: String,

    #[serde(default = "default_model_dir")]
    pub model_dir: String,

    #[serde(default)]
    pub safety: SafetyConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SafetyConfig {
    #[serde(default = "default_allowed")]
    pub allowed_domains: Vec<String>,

    #[serde(default = "default_blocked")]
    pub blocked_domains: Vec<String>,

    #[serde(default = "default_confirm")]
    pub confirm_domains: Vec<String>,

    #[serde(default = "default_confirm_keywords")]
    pub confirm_entity_keywords: Vec<String>,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            allowed_domains: default_allowed(),
            blocked_domains: default_blocked(),
            confirm_domains: default_confirm(),
            confirm_entity_keywords: default_confirm_keywords(),
        }
    }
}

impl Config {
    /// Load from file, then overlay environment variables.
    pub fn load(path: Option<&str>) -> Self {
        let mut cfg: Config = if let Some(p) = path {
            if let Ok(contents) = std::fs::read_to_string(p) {
                toml::from_str(&contents).unwrap_or_else(|e| {
                    tracing::warn!("Failed to parse config {}: {}, using defaults", p, e);
                    Config::default()
                })
            } else {
                Config::default()
            }
        } else {
            Config::default()
        };

        // Environment variable overrides
        if let Ok(v) = std::env::var("LISTEN_ADDR") { cfg.listen_addr = v; }
        if let Ok(v) = std::env::var("PG_DSN") { cfg.pg_dsn = v; }
        if let Ok(v) = std::env::var("DB_MIN_CONNS") { if let Ok(n) = v.parse() { cfg.db_min_conns = n; } }
        if let Ok(v) = std::env::var("DB_MAX_CONNS") { if let Ok(n) = v.parse() { cfg.db_max_conns = n; } }
        if let Ok(v) = std::env::var("DB_IDLE_TIMEOUT") { if let Ok(n) = v.parse() { cfg.db_idle_timeout = n; } }
        if let Ok(v) = std::env::var("EMBEDDER_URL") { cfg.embedder_url = v; }
        if let Ok(v) = std::env::var("INTENT_URL") { cfg.intent_url = v; }
        if let Ok(v) = std::env::var("GLM_URL") { cfg.glm_url = v; }
        if let Ok(v) = std::env::var("HA_URL") { cfg.ha_url = v; }
        if let Ok(v) = std::env::var("HA_TOKEN") { cfg.ha_token = v; }
        if let Ok(v) = std::env::var("MODEL_DIR") { cfg.model_dir = v; }

        cfg
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            listen_addr: default_listen(),
            pg_dsn: default_pg_dsn(),
            db_min_conns: default_db_min_conns(),
            db_max_conns: default_db_max_conns(),
            db_idle_timeout: default_db_idle_timeout(),
            embedder_url: String::new(),
            intent_url: default_intent_url(),
            glm_url: default_glm_url(),
            ha_url: String::new(),
            ha_token: String::new(),
            model_dir: default_model_dir(),
            safety: SafetyConfig::default(),
        }
    }
}

fn default_listen() -> String { "0.0.0.0:5000".into() }
fn default_pg_dsn() -> String { "postgres://agent:change-me@localhost:5432/agent?sslmode=disable".into() }
fn default_db_min_conns() -> u32 { 2 }
fn default_db_max_conns() -> u32 { 20 }
fn default_db_idle_timeout() -> u64 { 60 }
fn default_intent_url() -> String { "http://localhost:8081".into() }
fn default_glm_url() -> String { "http://localhost:8080".into() }
fn default_model_dir() -> String { "/opt/zagato-agent/models".into() }

fn default_allowed() -> Vec<String> {
    vec![
        "light", "switch", "cover", "climate", "media_player", "fan", "scene", "script",
        "automation", "input_boolean", "input_select", "input_number", "input_text",
        "vacuum", "humidifier", "water_heater", "siren", "remote", "button", "number", "select",
    ].into_iter().map(String::from).collect()
}

fn default_blocked() -> Vec<String> {
    vec!["shell_command", "rest_command", "notify", "persistent_notification", "homeassistant"]
        .into_iter().map(String::from).collect()
}

fn default_confirm() -> Vec<String> {
    vec!["lock", "alarm_control_panel", "valve"].into_iter().map(String::from).collect()
}

fn default_confirm_keywords() -> Vec<String> {
    vec!["garage".into()]
}
