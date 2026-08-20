use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use nv_weights::WeightLoader;

#[derive(Clone, Debug)]
pub struct CompressorConfig {
    pub in_dim: usize,
    pub neck_dim: usize,
    pub mid_dim: usize,
    pub out_dim: usize,
    pub ln_eps: f64,
}

impl CompressorConfig {
    pub fn deepseek_ocr2() -> Self {
        Self {
            in_dim: 768,
            neck_dim: 256,
            mid_dim: 512,
            out_dim: 896,
            ln_eps: 1e-6,
        }
    }
}

pub fn layer_norm_2d(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let c = x.dim(1)?;
    let x32 = x.to_dtype(DType::F32)?;
    let mu = x32.mean_keepdim(1)?;
    let xc = x32.broadcast_sub(&mu)?;
    let var = xc.sqr()?.mean_keepdim(1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    let w = w.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?;
    let b = b.to_dtype(DType::F32)?.reshape((1, c, 1, 1))?;
    Ok(normed
        .broadcast_mul(&w)?
        .broadcast_add(&b)?
        .to_dtype(dtype)?)
}

pub struct Compressor {
    cfg: CompressorConfig,
    neck0_w: Tensor,
    neck1_w: Tensor,
    neck1_b: Tensor,
    neck2_w: Tensor,
    neck3_w: Tensor,
    neck3_b: Tensor,
    net2_w: Tensor,
    net3_w: Tensor,
}

impl Compressor {
    pub fn from_loader(
        weights: &WeightLoader,
        prefix: &str,
        cfg: CompressorConfig,
        dtype: DType,
    ) -> Result<Self> {
        let g = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}{name}"), dtype)
                .with_context(|| format!("load {prefix}{name}"))
        };
        Ok(Self {
            neck0_w: g("neck.0.weight")?,
            neck1_w: g("neck.1.weight")?,
            neck1_b: g("neck.1.bias")?,
            neck2_w: g("neck.2.weight")?,
            neck3_w: g("neck.3.weight")?,
            neck3_b: g("neck.3.bias")?,
            net2_w: g("net_2.weight")?,
            net3_w: g("net_3.weight")?,
            cfg,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.permute((0, 3, 1, 2))?.contiguous()?;
        let x = x.conv2d(&self.neck0_w, 0, 1, 1, 1)?;
        let x = layer_norm_2d(&x, &self.neck1_w, &self.neck1_b, self.cfg.ln_eps)?;
        let x = x.conv2d(&self.neck2_w, 1, 1, 1, 1)?;
        let x = layer_norm_2d(&x, &self.neck3_w, &self.neck3_b, self.cfg.ln_eps)?;
        let x = x.conv2d(&self.net2_w, 1, 2, 1, 1)?;
        let x = x.conv2d(&self.net3_w, 1, 2, 1, 1)?;
        Ok(x)
    }

    pub fn config(&self) -> &CompressorConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn layer_norm_2d_normalizes_channels() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 3.0, 5.0, 7.0], (1, 4, 1, 1), &dev).unwrap();
        let w = Tensor::ones(4, DType::F32, &dev).unwrap();
        let b = Tensor::zeros(4, DType::F32, &dev).unwrap();
        let y = layer_norm_2d(&x, &w, &b, 1e-6).unwrap();
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        let mean: f32 = v.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((v[0] + v[3]).abs() < 1e-5);
    }
}
