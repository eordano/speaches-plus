use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use nv_layers::linear::Linear;

pub(crate) fn load_1d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    dim: usize,
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("{name}: expected [{}], got {:?}", dim, d);
    }
    Ok(w)
}

pub(crate) fn load_2d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    shape: (usize, usize),
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != shape.0 || d[1] != shape.1 {
        anyhow::bail!("{name}: expected [{}, {}], got {:?}", shape.0, shape.1, d);
    }
    Ok(w)
}

pub(crate) fn load_lin_bias(
    weights: &nv_weights::WeightLoader,
    prefix: &str,
    out_f: usize,
    in_f: usize,
    dtype: DType,
) -> Result<Linear> {
    let w = load_2d(weights, &format!("{prefix}.weight"), (out_f, in_f), dtype)?;
    let b = load_1d(weights, &format!("{prefix}.bias"), out_f, dtype)?;
    Linear::new(w, Some(b))
}
