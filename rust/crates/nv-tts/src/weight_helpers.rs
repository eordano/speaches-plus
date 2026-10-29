use anyhow::{Context, Result};
use candle_core::DType;
use nv_weights::WeightLoader;

use crate::dense::{DenseLinear, LinearQuant};

pub(crate) fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    quant: &LinearQuant,
) -> Result<DenseLinear> {
    DenseLinear::quantized(
        nv_layers::weight_load::load_tensor(weights, name, &[out_features, in_features], dtype)?,
        None,
        quant,
    )
    .with_context(|| format!("build linear {name}"))
}

pub(crate) fn load_linear_with_bias(
    weights: &WeightLoader,
    prefix: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<DenseLinear> {
    let wname = format!("{prefix}.weight");
    let w = weights
        .get(&wname, dtype)
        .with_context(|| format!("load {wname}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != out_features || d[1] != in_features {
        anyhow::bail!("linear {wname}: expected [{out_features}, {in_features}], got {d:?}");
    }
    let bname = format!("{prefix}.bias");
    let b = weights
        .get(&bname, dtype)
        .with_context(|| format!("load {bname}"))?;
    let bd = b.dims();
    if bd.len() != 1 || bd[0] != out_features {
        anyhow::bail!("linear bias {bname}: expected [{out_features}], got {bd:?}");
    }
    DenseLinear::new(w, Some(b))
}
