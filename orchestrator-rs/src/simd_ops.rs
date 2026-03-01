/// SIMD-accelerated vector operations with runtime dispatch.
/// Uses multiversion to compile AVX2 and scalar versions, selecting the best at runtime.

use multiversion::multiversion;

/// Cosine similarity between two f32 slices.
/// Dispatches to AVX2 or scalar at runtime.
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f"))]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "vectors must be same length");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt()) * (norm_b.sqrt());
    if denom < 1e-10 {
        0.0
    } else {
        dot / denom
    }
}

/// Batch cosine similarity: compute similarity of a query vector against
/// a list of candidates, returning (index, similarity) sorted descending.
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f"))]
pub fn batch_cosine_rank(query: &[f32], candidates: &[Vec<f32>], top_k: usize) -> Vec<(usize, f32)> {
    let mut scores: Vec<(usize, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(query, c)))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(top_k);
    scores
}

/// L2 (Euclidean) distance between two f32 slices.
#[multiversion(targets("x86_64+avx2", "x86_64+avx512f"))]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum.sqrt()
}

/// Normalize a vector in-place.
pub fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_batch_rank() {
        let query = vec![1.0, 0.0, 0.0];
        let candidates = vec![
            vec![0.0, 1.0, 0.0],  // orthogonal
            vec![1.0, 0.0, 0.0],  // identical
            vec![0.5, 0.5, 0.0],  // partial
        ];
        let ranked = batch_cosine_rank(&query, &candidates, 3);
        assert_eq!(ranked[0].0, 1); // identical first
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_distance() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5);
    }
}
