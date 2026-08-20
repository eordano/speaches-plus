use anyhow::{Context, Result};
use candle_core::DType;
use nv_layers::RmsNorm;
use nv_weights::WeightLoader;

use crate::dense::DenseLinear;

pub(crate) fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<DenseLinear> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!("linear {name}: expected [{out_features}, {in_features}], got {d:?}");
    }
    DenseLinear::new(w, None)
}

pub(crate) fn load_rmsnorm(
    weights: &WeightLoader,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("rmsnorm {name}: expected [{dim}], got {d:?}");
    }
    Ok(RmsNorm::new(w, eps))
}
