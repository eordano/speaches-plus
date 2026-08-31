use anyhow::Result;
use candle_core::DType;
use nv_weights::WeightLoader;

use crate::dense::DenseLinear;

pub(crate) fn load_linear(
    weights: &WeightLoader,
    name: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<DenseLinear> {
    DenseLinear::new(
        nv_layers::weight_load::load_tensor(weights, name, &[out_features, in_features], dtype)?,
        None,
    )
}
