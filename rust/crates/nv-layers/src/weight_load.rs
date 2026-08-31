use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use nv_weights::TensorSource;

use crate::linear::Linear;
use crate::norm::RmsNorm;

pub type NameResolver<S> = fn(&S, &str) -> Result<String>;

pub fn resolve_stripping_model_prefix<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
) -> Result<String> {
    if weights.has(name) {
        return Ok(name.to_string());
    }
    if let Some(stripped) = name.strip_prefix("model.") {
        if weights.has(stripped) {
            return Ok(stripped.to_string());
        }
    }
    anyhow::bail!("tensor not found (tried {name} and the model.-stripped variant)")
}

fn verbatim_name<S: TensorSource + ?Sized>(_weights: &S, name: &str) -> Result<String> {
    Ok(name.to_string())
}

pub fn load_tensor<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    expected: &[usize],
    dtype: DType,
) -> Result<Tensor> {
    load_tensor_resolving(weights, name, expected, dtype, verbatim_name)
}

pub fn load_tensor_resolving<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    expected: &[usize],
    dtype: DType,
    resolver: NameResolver<S>,
) -> Result<Tensor> {
    let resolved = resolver(weights, name)?;
    let t = weights
        .get(&resolved, dtype)
        .with_context(|| format!("load {resolved}"))?;
    anyhow::ensure!(
        t.dims() == expected,
        "tensor {resolved}: expected {expected:?}, got {:?}",
        t.dims()
    );
    Ok(t)
}

pub fn load_linear<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    load_linear_resolving(
        weights,
        name,
        out_features,
        in_features,
        dtype,
        verbatim_name,
    )
}

pub fn load_linear_resolving<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
    resolver: NameResolver<S>,
) -> Result<Linear> {
    Linear::new(
        load_tensor_resolving(weights, name, &[out_features, in_features], dtype, resolver)?,
        None,
    )
}

pub fn load_linear_with_bias_names<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    out_features: usize,
    in_features: usize,
    bias: Option<&str>,
    dtype: DType,
) -> Result<Linear> {
    let w = load_tensor(weights, name, &[out_features, in_features], dtype)?;
    let b = match bias {
        Some(bn) => Some(load_tensor(weights, bn, &[out_features], dtype)?),
        None => None,
    };
    Linear::new(w.contiguous()?, b)
}

pub fn load_rmsnorm<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
) -> Result<RmsNorm> {
    load_rmsnorm_resolving(weights, name, dim, eps, dtype, verbatim_name)
}

pub fn load_rmsnorm_resolving<S: TensorSource + ?Sized>(
    weights: &S,
    name: &str,
    dim: usize,
    eps: f64,
    dtype: DType,
    resolver: NameResolver<S>,
) -> Result<RmsNorm> {
    Ok(RmsNorm::new(
        load_tensor_resolving(weights, name, &[dim], dtype, resolver)?,
        eps,
    ))
}
