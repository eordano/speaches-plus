use anyhow::{Context, Result};
use candle_core::Tensor;

use crate::VarBuilder;

pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv2d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let d = weight.dims();
        if d.len() != 4 {
            anyhow::bail!(
                "Conv2d weight: expected rank-4 (oc, ic/groups, kH, kW), got {:?}",
                d
            );
        }
        if let Some(b) = &bias {
            let bd = b.dims();
            if bd.len() != 1 || bd[0] != d[0] {
                anyhow::bail!("Conv2d bias: expected [{}], got {:?}", d[0], bd);
            }
        }
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation: 1,
            groups: 1,
        })
    }

    pub fn with_dilation_groups(mut self, dilation: usize, groups: usize) -> Self {
        self.dilation = dilation;
        self.groups = groups;
        self
    }

    pub fn from_candle_vb(
        vb: VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Result<Self> {
        let weight = vb
            .get((out_channels, in_channels, kernel, kernel), "weight")
            .context("Conv2d weight")?;
        let bias = if bias {
            Some(vb.get(out_channels, "bias").context("Conv2d bias")?)
        } else {
            None
        };
        Self::new(weight, bias, stride, padding)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x
            .conv2d(
                &self.weight,
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        if let Some(b) = &self.bias {
            let b = b
                .reshape((1, b.dims()[0], 1, 1))
                .map_err(|e| anyhow::anyhow!(e))?;
            Ok(y.broadcast_add(&b).map_err(|e| anyhow::anyhow!(e))?)
        } else {
            Ok(y)
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

pub struct Conv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
}

impl Conv1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let d = weight.dims();
        if d.len() != 3 {
            anyhow::bail!(
                "Conv1d weight: expected rank-3 (oc, ic/groups, k), got {:?}",
                d
            );
        }
        if let Some(b) = &bias {
            let bd = b.dims();
            if bd.len() != 1 || bd[0] != d[0] {
                anyhow::bail!("Conv1d bias: expected [{}], got {:?}", d[0], bd);
            }
        }
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation: 1,
            groups: 1,
        })
    }

    pub fn with_dilation_groups(mut self, dilation: usize, groups: usize) -> Self {
        self.dilation = dilation;
        self.groups = groups;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x
            .conv1d(
                &self.weight,
                self.padding,
                self.stride,
                self.dilation,
                self.groups,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        if let Some(b) = &self.bias {
            let b = b
                .reshape((1, b.dims()[0], 1))
                .map_err(|e| anyhow::anyhow!(e))?;
            Ok(y.broadcast_add(&b).map_err(|e| anyhow::anyhow!(e))?)
        } else {
            Ok(y)
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

pub struct ConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
}

impl ConvTranspose1d {
    pub fn new(
        weight: Tensor,
        bias: Option<Tensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Result<Self> {
        let d = weight.dims();
        if d.len() != 3 {
            anyhow::bail!(
                "ConvTranspose1d weight: expected rank-3 (in_channels, out_channels, k), got {:?}",
                d
            );
        }
        if let Some(b) = &bias {
            let bd = b.dims();
            if bd.len() != 1 || bd[0] != d[1] {
                anyhow::bail!("ConvTranspose1d bias: expected [{}], got {:?}", d[1], bd);
            }
        }
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation: 1,
            groups: 1,
        })
    }

    pub fn with_dilation_groups(mut self, dilation: usize, groups: usize) -> Self {
        self.dilation = dilation;
        self.groups = groups;
        self
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x
            .conv_transpose1d(
                &self.weight,
                self.padding,
                self.output_padding,
                self.stride,
                self.dilation,
                self.groups,
            )
            .map_err(|e| anyhow::anyhow!(e))?;
        if let Some(b) = &self.bias {
            let b = b
                .reshape((1, b.dims()[0], 1))
                .map_err(|e| anyhow::anyhow!(e))?;
            Ok(y.broadcast_add(&b).map_err(|e| anyhow::anyhow!(e))?)
        } else {
            Ok(y)
        }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn padding(&self) -> usize {
        self.padding
    }

    pub fn output_padding(&self) -> usize {
        self.output_padding
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn conv_layers_produce_the_subsampled_padded_and_upsampled_output_shapes() {
        let dev = Device::Cpu;
        let z = |shape: &[usize]| Tensor::zeros(shape, DType::F32, &dev).unwrap();
        let conv2d = Conv2d::new(z(&[8, 4, 3, 3]), Some(z(&[8])), 2, 1).unwrap();
        assert_eq!(conv2d.forward(&z(&[1, 4, 16, 16])).unwrap().dims(), &[1, 8, 8, 8]);
        let conv1d = Conv1d::new(z(&[4, 2, 5]), None, 1, 2).unwrap();
        assert_eq!(conv1d.forward(&z(&[1, 2, 32])).unwrap().dims(), &[1, 4, 32]);
        let up = ConvTranspose1d::new(z(&[2, 4, 4]), Some(z(&[4])), 2, 0, 0).unwrap();
        assert_eq!(up.forward(&z(&[1, 2, 8])).unwrap().dims(), &[1, 4, 18]);
        let padded = ConvTranspose1d::new(z(&[3, 2, 10]), None, 5, 2, 1).unwrap();
        assert_eq!(padded.forward(&z(&[1, 3, 4])).unwrap().dims(), &[1, 2, 22]);
    }
}
