use crate::models::{Candidate, IntentPlan};
use ndarray::Array2;
use ort::{
    execution_providers::{CUDAExecutionProvider, TensorRTExecutionProvider},
    session::{builder::GraphOptimizationLevel, builder::SessionBuilder, Session},
    value::Tensor,
};
use tokio::sync::Mutex;
use tokenizers::Tokenizer;

pub struct LocalIntentClient {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_tokens: usize,
}

impl LocalIntentClient {
    pub fn new(model_dir: &str) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let model_path = format!("{}/qwen2.5-1.5b-instruct/model.onnx", model_dir);
        let tokenizer_path = format!("{}/qwen2.5-1.5b-instruct/tokenizer.json", model_dir);

        tracing::info!("Loading local intent model from {}", model_path);
        
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("Failed to load tokenizer from {}: {}", tokenizer_path, e))?;

        let session = SessionBuilder::new()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_intra_threads(4)
            .unwrap()
            .with_execution_providers([
                TensorRTExecutionProvider::default().build(),
                CUDAExecutionProvider::default().build(),
            ])
            .unwrap()
            .commit_from_file(&model_path)
            .unwrap();

        tracing::info!("Successfully loaded Qwen ONNX intent model.");

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_tokens: 128,
        })
    }

    pub async fn parse(
        &self,
        user_text: &str,
        candidates: &[Candidate],
    ) -> Result<IntentPlan, Box<dyn std::error::Error + Send + Sync>> {
        let mut candidate_lines = Vec::new();
        for c in candidates {
            candidate_lines.push(format!("- entity_id: {}, name: \"{}\", domain: {}", c.entity_id, c.name, c.domain));
        }

        let prompt = format!(
            "<|im_start|>system\nYou are an intent parser. Output ONLY valid JSON matching this schema:\n{{\"plan\": [ {{\"type\": \"ha_call\", \"entity_id\": \"...\", \"service\": \"...\", \"service_data\": {{}} }}, {{\"type\": \"ask_clarifying\", \"speech\": \"...\"}}, {{\"type\": \"non_ha\", \"speech\": \"...\"}} ]}}\nRules: entity_id MUST be one of the candidates.<|im_end|>\n<|im_start|>user\nUser said: \"{}\"\nCandidates:\n{}\nOutput JSON:<|im_end|>\n<|im_start|>assistant\n",
            user_text, candidate_lines.join("\n")
        );

        let mut input_ids = self.tokenizer.encode(prompt, true)
            .map_err(|_| "Tokenizer encode error")?
            .get_ids()
            .to_vec();

        let mut generated_tokens = Vec::new();

        // Autoregressive decoding loop (Greedy, no KV cache for simplest basic ONNX evaluation)
        for _ in 0..self.max_tokens {
            let seq_len = input_ids.len();
            let input_ids_arr = Array2::from_shape_vec((1, seq_len), input_ids.iter().map(|&x| x as i64).collect()).unwrap();
            let attention_mask_arr = Array2::from_elem((1, seq_len), 1i64);

            let input_ids_tensor = Tensor::from_array(input_ids_arr).unwrap();
            let attention_mask_tensor = Tensor::from_array(attention_mask_arr).unwrap();

            // Run session
            let inputs = ort::inputs![
                "input_ids" => input_ids_tensor,
                "attention_mask" => attention_mask_tensor,
            ];
            
            let mut session_guard = self.session.lock().await;
            let outputs = session_guard.run(inputs).unwrap();

            let extracted = outputs["logits"].try_extract_tensor::<f32>().unwrap();
            let shape = extracted.0;
            let slice = extracted.1;
            
            let vocab_size = shape[2] as usize;
            let last_token_start = (seq_len - 1) * vocab_size;
            let last_token_logits = &slice[last_token_start .. last_token_start + vocab_size];

            // Greedy argmax
            let mut best_score = f32::NEG_INFINITY;
            let mut best_token = 0u32;
            for (token_id, &score) in last_token_logits.iter().enumerate() {
                if score > best_score {
                    best_score = score;
                    best_token = token_id as u32;
                }
            }

            // Stop tokens (e.g., <|im_end|> is typically 151645 for Qwen)
            if best_token == 151645 {
                break;
            }

            generated_tokens.push(best_token);
            input_ids.push(best_token);
        }

        let mut content = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|_| "Decode error")?;

        content = content.trim().to_string();
        if content.starts_with("```json") { content = content.trim_start_matches("```json").to_string(); }
        else if content.starts_with("```") { content = content.trim_start_matches("```").to_string(); }
        if content.ends_with("```") { content = content.trim_end_matches("```").to_string(); }
        content = content.trim().to_string();

        let mut plan_bytes = content.as_bytes().to_vec();
        let mut plan: IntentPlan = simd_json::from_slice(&mut plan_bytes)
            .map_err(|e| format!("Local ONNX Intent JSON parse error: {} (raw: {})", e, content))?;

        let valid_ids: std::collections::HashSet<_> = candidates.iter().map(|c| c.entity_id.clone()).collect();
        plan.plan.retain(|step| {
            if let crate::models::PlanStep::HaCall { entity_id, .. } = step {
                valid_ids.contains(entity_id)
            } else {
                true // Keep clarification and non-HA responses
            }
        });

        Ok(plan)
    }
}
