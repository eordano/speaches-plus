use anyhow::Result;
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use candle_core::{DType, Device};
#[cfg(feature = "cuda")]
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub enum LinearQuant {
    Dense,
    #[cfg(feature = "cuda")]
    Fp8E4m3(Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>),
}

impl std::fmt::Debug for LinearQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dense => f.write_str("Dense"),
            #[cfg(feature = "cuda")]
            Self::Fp8E4m3(_) => f.write_str("Fp8E4m3"),
        }
    }
}

impl Default for LinearQuant {
    fn default() -> Self {
        Self::Dense
    }
}

impl LinearQuant {
    #[cfg(feature = "cuda")]
    pub fn fp8_for(device: &Device) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("fp8 talker weights need a CUDA device"),
        };
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let runner = nv_quant::fp8::Fp8GemmRunner::new(stream)?;
        Ok(Self::Fp8E4m3(Arc::new(Mutex::new(runner))))
    }

    #[cfg(not(feature = "cuda"))]
    pub fn fp8_for(_device: &candle_core::Device) -> Result<Self> {
        anyhow::bail!("fp8 talker weights need the cuda feature")
    }

    pub fn is_dense(&self) -> bool {
        matches!(self, Self::Dense)
    }
}

enum Kernel {
    Dense { wt: Tensor },
    #[cfg(feature = "cuda")]
    Fp8(nv_layers::linear::Linear),
}

pub struct DenseLinear {
    kernel: Kernel,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl DenseLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let (out_features, in_features) = Self::check_shapes(&weight, bias.as_ref())?;
        let wt = weight.t()?.contiguous()?;
        Ok(Self {
            kernel: Kernel::Dense { wt },
            bias,
            in_features,
            out_features,
        })
    }

    pub fn quantized(weight: Tensor, bias: Option<Tensor>, quant: &LinearQuant) -> Result<Self> {
        match quant {
            LinearQuant::Dense => Self::new(weight, bias),
            #[cfg(feature = "cuda")]
            LinearQuant::Fp8E4m3(runner) => Self::new_fp8(weight, bias, runner.clone()),
        }
    }

    #[cfg(feature = "cuda")]
    fn new_fp8(
        weight: Tensor,
        bias: Option<Tensor>,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        let (out_features, in_features) = Self::check_shapes(&weight, bias.as_ref())?;
        if !in_features.is_multiple_of(16) || out_features < 2 {
            return Self::new(weight, bias);
        }
        let dev = match weight.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("fp8 DenseLinear: weight must live on a CUDA device"),
        };
        let host: Vec<half::bf16> = weight
            .to_dtype(DType::BF16)?
            .contiguous()?
            .flatten_all()?
            .to_vec1()?;
        let (bytes, rows) = nv_layers::linear::fp8_weight_payload(
            &host,
            out_features,
            in_features,
            None,
            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
        )?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let weight_u8 = stream.clone_htod(&bytes)?;
        let lin = nv_layers::linear::Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
            weight_u8,
            rows,
            in_features,
            out_features,
            None,
            weight.device(),
            runner,
            nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
        )?;
        Ok(Self {
            kernel: Kernel::Fp8(lin),
            bias,
            in_features,
            out_features,
        })
    }

    fn check_shapes(weight: &Tensor, bias: Option<&Tensor>) -> Result<(usize, usize)> {
        let d = weight.dims().to_vec();
        if d.len() != 2 {
            anyhow::bail!("DenseLinear: weight must be [out, in], got {d:?}");
        }
        let (out_features, in_features) = (d[0], d[1]);
        if let Some(b) = bias {
            let bd = b.dims();
            if bd.len() != 1 || bd[0] != out_features {
                anyhow::bail!("DenseLinear: bias must be [{out_features}], got {bd:?}");
            }
        }
        Ok((out_features, in_features))
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn is_fp8(&self) -> bool {
        match &self.kernel {
            Kernel::Dense { .. } => false,
            #[cfg(feature = "cuda")]
            Kernel::Fp8(_) => true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if *dims.last().unwrap_or(&0) != self.in_features {
            anyhow::bail!(
                "DenseLinear: input last dim {:?} != in_features {}",
                dims.last(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;
        let mut out = match &self.kernel {
            Kernel::Dense { wt } => {
                let x2 = if x2.dtype() == wt.dtype() {
                    x2
                } else {
                    x2.to_dtype(wt.dtype())?
                };
                x2.matmul(wt)?
            }
            #[cfg(feature = "cuda")]
            Kernel::Fp8(lin) => {
                let x_bf = if x2.dtype() == DType::BF16 {
                    x2.clone()
                } else {
                    x2.to_dtype(DType::BF16)?
                };
                let y = lin.forward(&x_bf.contiguous()?)?;
                if y.dtype() == x2.dtype() {
                    y
                } else {
                    y.to_dtype(x2.dtype())?
                }
            }
        };
        if let Some(b) = &self.bias {
            let b = if b.dtype() == out.dtype() {
                b.clone()
            } else {
                b.to_dtype(out.dtype())?
            };
            out = out.broadcast_add(&b)?;
        }
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        Ok(out.reshape(out_dims)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn matches_manual_matmul_with_bias_and_rejects_bad_shapes() {
        let dev = Device::Cpu;
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let weight = Tensor::from_vec(w, (3usize, 4usize), &dev).unwrap();
        let bias = Tensor::from_vec(vec![1.0f32, -1.0, 0.5], 3usize, &dev).unwrap();
        let lin = DenseLinear::new(weight.clone(), Some(bias.clone())).unwrap();
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 0.0625],
            (2usize, 4usize),
            &dev,
        )
        .unwrap();
        let y = lin.forward(&x).unwrap();
        let expected = x
            .matmul(&weight.t().unwrap())
            .unwrap()
            .broadcast_add(&bias)
            .unwrap();
        let diff = (y - expected).unwrap().abs().unwrap().flatten_all().unwrap();
        let max: f32 = diff.max(0).unwrap().to_scalar().unwrap();
        assert!(max < 1e-6, "max abs diff {max}");

        let bad_w = Tensor::zeros((3usize,), DType::F32, &dev).unwrap();
        assert!(DenseLinear::new(bad_w, None).is_err());
        let bad_b = Tensor::zeros((2usize,), DType::F32, &dev).unwrap();
        assert!(DenseLinear::new(weight.clone(), Some(bad_b)).is_err());
        let bad_x = Tensor::zeros((2usize, 5usize), DType::F32, &dev).unwrap();
        assert!(lin.forward(&bad_x).is_err());
        assert!(!lin.is_fp8());
        assert!(LinearQuant::default().is_dense());
    }

    #[test]
    fn a_dense_quant_request_builds_the_plain_kernel() {
        let dev = Device::Cpu;
        let weight = Tensor::zeros((4usize, 16usize), DType::F32, &dev).unwrap();
        let lin = DenseLinear::quantized(weight, None, &LinearQuant::Dense).unwrap();
        assert_eq!((lin.in_features(), lin.out_features()), (16, 4));
        assert!(!lin.is_fp8());
    }
}
