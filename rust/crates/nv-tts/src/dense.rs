use anyhow::Result;
use candle_core::Tensor;

pub struct DenseLinear {
    wt: Tensor,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
}

impl DenseLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let d = weight.dims().to_vec();
        if d.len() != 2 {
            anyhow::bail!("DenseLinear: weight must be [out, in], got {d:?}");
        }
        let (out_features, in_features) = (d[0], d[1]);
        if let Some(b) = &bias {
            let bd = b.dims();
            if bd.len() != 1 || bd[0] != out_features {
                anyhow::bail!("DenseLinear: bias must be [{out_features}], got {bd:?}");
            }
        }
        let wt = weight.t()?.contiguous()?;
        Ok(Self {
            wt,
            bias,
            in_features,
            out_features,
        })
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
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
        let x2 = if x2.dtype() == self.wt.dtype() {
            x2
        } else {
            x2.to_dtype(self.wt.dtype())?
        };
        let mut out = x2.matmul(&self.wt)?;
        if let Some(b) = &self.bias {
            out = out.broadcast_add(b)?;
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
    fn matches_manual_matmul_with_bias() {
        let dev = Device::Cpu;
        let w: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let weight = Tensor::from_vec(w, (3usize, 4usize), &dev).unwrap();
        let bias = Tensor::from_vec(vec![1.0f32, -1.0, 0.5], 3usize, &dev).unwrap();
        let lin = DenseLinear::new(weight.clone(), Some(bias.clone())).unwrap();
        let x = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 0.25, 0.125, 0.0625],
            (1usize, 2usize, 4usize),
            &dev,
        )
        .unwrap();
        let got = lin.forward(&x).unwrap();
        let want = x
            .reshape((2usize, 4usize))
            .unwrap()
            .matmul(&weight.t().unwrap().contiguous().unwrap())
            .unwrap()
            .broadcast_add(&bias)
            .unwrap();
        let g = got.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let e = want.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(g, e);
        assert_eq!(got.dims(), &[1, 2, 3]);
        let _ = DType::F32;
    }

    #[test]
    fn rejects_bad_shapes() {
        let dev = Device::Cpu;
        let w = Tensor::zeros((3usize, 4usize), DType::F32, &dev).unwrap();
        let lin = DenseLinear::new(w, None).unwrap();
        let x = Tensor::zeros((1usize, 5usize), DType::F32, &dev).unwrap();
        assert!(lin.forward(&x).is_err());
    }
}
