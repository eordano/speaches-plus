use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Tensor};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_weights::WeightLoader;

pub(crate) fn load_tensor(
    weights: &WeightLoader,
    name: &str,
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    if !weights.has(name) {
        bail!("missing tensor {name}");
    }
    let actual = weights
        .shape_of(name)
        .ok_or_else(|| anyhow!("no shape for {name}"))?;
    if actual != shape {
        bail!("tensor {name}: expected shape {shape:?}, got {actual:?}");
    }
    weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))
}

pub(crate) fn load_rmsnorm(
    weights: &WeightLoader,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let w = load_tensor(weights, name, &[dim], dtype)?;
    Ok(RmsNorm::new(w, eps))
}

pub(crate) fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = load_tensor(weights, name, &[out_features, in_features], dtype)?;
    Linear::new(w, None)
}

pub(crate) fn argmax_f32(xs: &[f32]) -> (usize, f32) {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in xs.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    (best, best_v)
}

pub(crate) fn top_k_indices(xs: &[f32], k: usize) -> Vec<usize> {
    let k = k.min(xs.len());
    let mut idx: Vec<usize> = (0..xs.len()).collect();
    idx.sort_by(|&a, &b| {
        xs[b]
            .partial_cmp(&xs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(k);
    idx
}
