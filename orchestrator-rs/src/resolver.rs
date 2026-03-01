use crate::cache::EntityCache;
use crate::config::Config;
use crate::embedder_local;
use crate::ml_clients::EmbedderClient;
use crate::models::Candidate;
use std::collections::HashMap;
use std::sync::Arc;

pub struct Resolver {
    pool: sqlx::PgPool,
    cache: Arc<EntityCache>,
    cfg: Arc<Config>,
    embedder_http: Option<Arc<EmbedderClient>>,
}

impl Resolver {
    pub async fn new(
        pool: sqlx::PgPool,
        cache: Arc<EntityCache>,
        cfg: Arc<Config>,
        embedder_http: Option<Arc<EmbedderClient>>,
    ) -> Result<Self, sqlx::Error> {
        // Bulk load cache on startup
        let rows = sqlx::query(
            r#"SELECT item_id, kind, domain, entity_id, name, COALESCE(area, '') as area 
               FROM catalog_items WHERE enabled = true"#
        )
        .fetch_all(&pool)
        .await?;

        let mut candidates = Vec::with_capacity(rows.len());
        for r in rows {
            use sqlx::Row;
            candidates.push(Candidate {
                item_id: r.try_get("item_id")?,
                kind: r.try_get("kind")?,
                domain: r.try_get("domain")?,
                entity_id: r.try_get("entity_id")?,
                name: r.try_get("name")?,
                area: r.try_get("area").unwrap_or_default(),
                score: 0.0,
            });
        }
        cache.load_all(candidates);

        Ok(Self {
            pool,
            cache,
            cfg,
            embedder_http,
        })
    }

    pub async fn ping(&self) -> Result<(), sqlx::Error> {
        sqlx::query("SELECT 1").execute(&self.pool).await?;
        Ok(())
    }

    pub fn config(&self) -> Arc<Config> {
        self.cfg.clone()
    }

    pub async fn resolve(&self, text: &str) -> Result<Vec<Candidate>, sqlx::Error> {
        let lower = text.trim().to_lowercase();

        // Pass A: Lexical matching from fast in-memory cache
        let lexical = self.cache.lexical_search(&lower);

        // Strong match short-circuit
        for c in &lexical {
            if c.score >= 0.95 {
                tracing::info!("Strong lexical match from cache: {} ({:.2})", c.entity_id, c.score);
                return Ok(vec![c.clone()]);
            }
        }

        // Pass B: Vector Search
        let vector;
        
        let mut emb_vec = None;
        if std::path::Path::new(&format!("{}/all-MiniLM-L6-v2/model.onnx", self.cfg.model_dir)).exists() {
            // Local ONNX Embedder
            if let Ok(batch) = embedder_local::batch_embed_onnx(&self.cfg.model_dir, &[&lower]) {
                emb_vec = batch.into_iter().next();
            }
        } else if let Some(http) = &self.embedder_http {
            // HTTP Embedder
            if let Ok(emb) = http.embed(&lower).await {
                // convert f64 to f32
                emb_vec = Some(emb.into_iter().map(|f| f as f32).collect());
            }
        }

        if let Some(emb) = emb_vec {
            vector = self.vector_search(&emb).await.unwrap_or_default();
        } else {
            tracing::warn!("Embedder error, falling back to basic text search");
            vector = self.text_search(&lower).await.unwrap_or_default();
        }

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

    async fn vector_search(&self, embedding: &[f32]) -> Result<Vec<Candidate>, sqlx::Error> {
        // Load all embeddings from DB to rank them in-memory using our SIMD path
        // In a real huge DB, we'd use pgvector index `ORDER BY embedding <=> $1 LIMIT 8` directly.
        // But the user requested SIMD avx2 batched cosine ops locally in memory!
        
        let rows = sqlx::query("SELECT entity_id, embedding FROM catalog_items WHERE enabled = true")
            .fetch_all(&self.pool).await?;

        let mut candidates_vec = Vec::with_capacity(rows.len());
        let mut eids = Vec::with_capacity(rows.len());

        for r in rows {
            use sqlx::Row;
            let vec: pgvector::Vector = r.try_get("embedding")?;
            let eid: String = r.try_get("entity_id")?;
            candidates_vec.push(vec.to_vec());
            eids.push(eid);
        }

        // Use our SIMD batch ranker
        let ranked = crate::simd_ops::batch_cosine_rank(embedding, &candidates_vec, 8);

        let mut top_candidates = Vec::with_capacity(ranked.len());
        for (idx, score) in ranked {
            let eid = &eids[idx];
            if let Some(mut c) = self.cache.get(eid) {
                c.score = score as f64;
                top_candidates.push(c);
            }
        }

        Ok(top_candidates)
    }

    async fn text_search(&self, text: &str) -> Result<Vec<Candidate>, sqlx::Error> {
        // Fallback using lexical cache if vector fails
        let mut lexical = self.cache.lexical_search(text);
        for c in &mut lexical {
            c.score -= 0.2; // penalty for text fallback
        }
        Ok(lexical)
    }
}
