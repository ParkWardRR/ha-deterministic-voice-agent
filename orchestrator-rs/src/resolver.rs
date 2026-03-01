use crate::ml_clients::EmbedderClient;
use crate::models::Candidate;
use sqlx::postgres::PgPoolOptions;
use sqlx::{PgPool, Row};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

pub struct Resolver {
    db: PgPool,
    embedder: Option<Arc<EmbedderClient>>,
}

impl Resolver {
    pub async fn new(dsn: &str, embedder: Option<Arc<EmbedderClient>>) -> Result<Self, sqlx::Error> {
        let pool = PgPoolOptions::new()
            .max_connections(5)
            .min_connections(2)
            .connect(dsn)
            .await?;
            
        Ok(Self { db: pool, embedder })
    }

    pub async fn ping(&self) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT 1").execute(&self.db).await?;
        Ok(())
    }

    pub async fn resolve(&self, text: &str) -> Result<Vec<Candidate>, sqlx::Error> {
        let lower = text.trim().to_lowercase();

        // Pass A: Lexical matching
        let mut lexical = self.lexical_match(&lower).await.unwrap_or_default();
        lexical = Self::dedupe_candidates(lexical);

        // Strong match short-circuit
        for c in &lexical {
            if c.score >= 0.95 {
                tracing::info!("Strong lexical match: {} ({:.2})", c.entity_id, c.score);
                return Ok(vec![c.clone()]);
            }
        }

        // Pass B: Vector Search
        let mut vector = Vec::new();
        if let Some(embedder) = &self.embedder {
            if let Ok(emb) = embedder.embed(&lower).await {
                vector = self.vector_search(&emb).await.unwrap_or_default();
            } else {
                tracing::warn!("Embedder error, falling back to text search");
            }
        }

        if vector.is_empty() {
            vector = self.text_search(&lower).await.unwrap_or_default();
        }
        vector = Self::dedupe_candidates(vector);

        // Merge
        let mut by_id = HashMap::with_capacity(lexical.len() + vector.len());
        let mut order = Vec::with_capacity(lexical.len() + vector.len());

        for c in lexical {
            if !by_id.contains_key(&c.entity_id) {
                order.push(c.entity_id.clone());
            }
            by_id.insert(c.entity_id.clone(), c);
        }

        for c in vector {
            if let Some(existing) = by_id.get_mut(&c.entity_id) {
                if c.score > existing.score {
                    *existing = c;
                }
            } else {
                order.push(c.entity_id.clone());
                by_id.insert(c.entity_id.clone(), c);
            }
        }

        let mut merged: Vec<Candidate> = order.into_iter().filter_map(|id| by_id.remove(&id)).collect();

        // Sort descending
        merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        if merged.len() > 8 {
            merged.truncate(8);
        }

        Ok(merged)
    }

    async fn vector_search(&self, embedding: &[f64]) -> Result<Vec<Candidate>, sqlx::Error> {
        if embedding.is_empty() {
            return Ok(Vec::new());
        }

        // pgvector requires f32
        let f32_embedding: Vec<f32> = embedding.iter().map(|&x| x as f32).collect();
        let vector = pgvector::Vector::from(f32_embedding);

        let query = r#"
            SELECT item_id, kind, domain, entity_id, name, COALESCE(area, ''),
                   1 - (embedding <=> $1) AS score
            FROM catalog_items
            WHERE enabled = true
            ORDER BY embedding <=> $1
            LIMIT 8
        "#;

        let rows = sqlx::query(query)
            .bind(vector)
            .fetch_all(&self.db)
            .await?;

        let mut candidates = Vec::with_capacity(rows.len());
        for row in rows {
            candidates.push(Candidate {
                item_id: row.try_get::<i64, _>(0)?,
                kind: row.try_get::<String, _>(1)?,
                domain: row.try_get::<String, _>(2)?,
                entity_id: row.try_get::<String, _>(3)?,
                name: row.try_get::<String, _>(4)?,
                area: row.try_get::<String, _>(5)?,
                score: row.try_get::<f64, _>(6)?,
            });
        }

        Ok(candidates)
    }

    async fn lexical_match(&self, text: &str) -> Result<Vec<Candidate>, sqlx::Error> {
        let mut candidates = Vec::new();

        // Exact entity_id match
        let query_exact = r#"
            SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '')
            FROM catalog_items
            WHERE enabled = true AND entity_id = $1
        "#;
        
        let exact_rows = sqlx::query(query_exact)
            .bind(text)
            .fetch_all(&self.db)
            .await?;

        for row in exact_rows {
            candidates.push(Candidate {
                item_id: row.try_get(0)?,
                kind: row.try_get(1)?,
                domain: row.try_get(2)?,
                entity_id: row.try_get(3)?,
                name: row.try_get(4)?,
                area: row.try_get(5)?,
                score: 1.0,
            });
        }

        // Name match
        let words = Self::extract_keywords(text);
        if !words.is_empty() {
            for word in words {
                if word.len() < 3 {
                    continue;
                }
                
                let query_name = r#"
                    SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '')
                    FROM catalog_items
                    WHERE enabled = true AND lower(name) LIKE '%' || $1 || '%'
                    LIMIT 10
                "#;
                
                let name_rows = sqlx::query(query_name)
                    .bind(&word)
                    .fetch_all(&self.db)
                    .await;
                
                if let Ok(rows) = name_rows {
                    for row in rows {
                        let name: String = row.try_get(4)?;
                        let name_lower = name.to_lowercase();
                        
                        let score = if name_lower == text {
                            0.95
                        } else if name_lower.contains(text) {
                            0.85
                        } else {
                            0.6 + 0.1 * (word.len() as f64) / (name_lower.len() as f64)
                        };

                        candidates.push(Candidate {
                            item_id: row.try_get(0)?,
                            kind: row.try_get(1)?,
                            domain: row.try_get(2)?,
                            entity_id: row.try_get(3)?,
                            name,
                            area: row.try_get(5)?,
                            score,
                        });
                    }
                }
            }
        }

        Ok(candidates)
    }

    async fn text_search(&self, text: &str) -> Result<Vec<Candidate>, sqlx::Error> {
        let words = Self::extract_keywords(text);
        if words.is_empty() {
            return Ok(Vec::new());
        }

        let mut conditions = Vec::with_capacity(words.len());
        for i in 0..words.len() {
            let idx = i + 1;
            conditions.push(format!(
                "(lower(name) LIKE '%' || ${} || '%' OR lower(domain) = ${} OR lower(COALESCE(area,'')) LIKE '%' || ${} || '%')",
                idx, idx, idx
            ));
        }

        let query_str = format!(
            r#"
            SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '')
            FROM catalog_items
            WHERE enabled = true AND ({})
            LIMIT 20
            "#, 
            conditions.join(" OR ")
        );

        let mut query = sqlx::query(&query_str);
        for w in &words {
            query = query.bind(w);
        }

        let rows = query.fetch_all(&self.db).await?;
        let mut candidates = Vec::with_capacity(rows.len());

        for row in rows {
            let name: String = row.try_get(4)?;
            let domain: String = row.try_get(2)?;
            let name_lower = name.to_lowercase();
            let domain_lower = domain.to_lowercase();
            
            let mut match_count = 0;
            for w in &words {
                if name_lower.contains(w) || domain_lower == *w {
                    match_count += 1;
                }
            }

            candidates.push(Candidate {
                item_id: row.try_get(0)?,
                kind: row.try_get(1)?,
                domain,
                entity_id: row.try_get(3)?,
                name,
                area: row.try_get(5)?,
                score: 0.4 + 0.15 * (match_count as f64),
            });
        }

        Ok(candidates)
    }

    fn extract_keywords(text: &str) -> Vec<String> {
        let set: HashSet<&'static str> = vec![
            "the", "a", "an", "to", "in", "on", "off", "turn", "set", "make",
            "please", "can", "you", "my", "me", "and", "or", "but", "is", "it",
            "of", "for", "with", "at", "from", "up", "down", "do", "this", "that",
            "what", "how", "all", "i", "hey",
        ].into_iter().collect();

        text.split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_ascii_lowercase())
                    .collect::<String>()
            })
            .filter(|w| !w.is_empty() && !set.contains(w.as_str()))
            .collect()
    }

    fn dedupe_candidates(candidates: Vec<Candidate>) -> Vec<Candidate> {
        if candidates.len() < 2 {
            return candidates;
        }

        let mut seen: HashMap<String, Candidate> = HashMap::with_capacity(candidates.len());
        let mut order: Vec<String> = Vec::with_capacity(candidates.len());

        for c in candidates {
            if let Some(existing) = seen.get_mut(&c.entity_id) {
                if c.score > existing.score {
                    *existing = c;
                }
            } else {
                order.push(c.entity_id.clone());
                seen.insert(c.entity_id.clone(), c);
            }
        }

        order.into_iter().filter_map(|id| seen.remove(&id)).collect()
    }
}
