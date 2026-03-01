use ndarray::Array2;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::Path;
use tokenizers::Tokenizer;

pub fn batch_embed_onnx(model_dir: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error>> {
    let model_path = Path::new(model_dir).join("all-MiniLM-L6-v2").join("model.onnx");
    let tokenizer_path = Path::new(model_dir).join("all-MiniLM-L6-v2").join("tokenizer.json");

    if !model_path.exists() || !tokenizer_path.exists() {
        return Err(format!("ONNX model or tokenizer not found in {}", model_dir).into());
    }

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Tokenizer load error: {}", e))?;

    // Encode texts
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| format!("Tokenizer encode error: {}", e))?;

    let batch_size = texts.len();
    // Assuming padding to max length in the batch
    let seq_len = encodings.first().map(|e| e.get_ids().len()).unwrap_or(0);

    let mut input_ids = Array2::<i64>::zeros((batch_size, seq_len));
    let mut attention_mask = Array2::<i64>::zeros((batch_size, seq_len));
    let mut token_type_ids = Array2::<i64>::zeros((batch_size, seq_len));

    for (i, encoding) in encodings.iter().enumerate() {
        let ids = encoding.get_ids();
        let mask = encoding.get_attention_mask();
        let types = encoding.get_type_ids();

        for j in 0..ids.len() {
            input_ids[[i, j]] = ids[j] as i64;
            attention_mask[[i, j]] = mask[j] as i64;
            token_type_ids[[i, j]] = types[j] as i64;
        }
    }

    // Initialize ORT with CUDA fallback to CPU Execution Provider
    let mut session = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        // Try CUDA first, if it fails ORT will fallback to CPU automatically if configured correctly
        // Or we can explicitly set execution providers:
        // .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default().build()])
        .commit_from_file(model_path)?;

    let input_ids_tensor = Tensor::from_array(input_ids)?;
    let attention_mask_tensor = Tensor::from_array(attention_mask.clone())?;
    let token_type_ids_tensor = Tensor::from_array(token_type_ids)?;

    // Run inference
    let inputs = ort::inputs![
        "input_ids" => input_ids_tensor,
        "attention_mask" => attention_mask_tensor,
        "token_type_ids" => token_type_ids_tensor,
    ];
    let outputs = session.run(inputs)?;

    // the output name depends on the exported model, usually "last_hidden_state" or similar
    // For sentence-transformers, we usually mean pool the last hidden state
    let extracted = outputs[0].try_extract_tensor::<f32>()?;
    let (shape, flat_data) = extracted;
    // shape: (batch_size, seq_len, hidden_size)

    let dims = shape;
    let hidden_size = dims[2] as usize;
    let seq_len = dims[1] as usize;
    let batch_size = dims[0] as usize;

    let mut embeddings = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let mut pooled = vec![0.0f32; hidden_size];
        let mut mask_sum = 0.0f32;

        for j in 0..seq_len {
            let mask_val = attention_mask[[i, j]] as f32;
            for k in 0..hidden_size {
                let flat_idx = i * seq_len * hidden_size + j * hidden_size + k;
                let val = flat_data[flat_idx];
                pooled[k] += val * mask_val;
            }
            mask_sum += mask_val;
        }

        // Mean pooling
        if mask_sum > 0.0 {
            for k in 0..hidden_size {
                pooled[k] /= mask_sum;
            }
        }

        // L2 Normalize
        crate::simd_ops::normalize(&mut pooled);
        embeddings.push(pooled);
    }

    Ok(embeddings)
}
