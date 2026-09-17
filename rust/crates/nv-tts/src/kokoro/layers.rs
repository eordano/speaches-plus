use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Module, Tensor, Var, D};
use candle_nn::rnn::{Direction, LSTMConfig};
use candle_nn::{lstm, Conv1d, Conv1dConfig, Linear, VarBuilder, LSTM, RNN};

pub const STYLE_DIM: usize = 128;

static DIFFERENTIABLE: AtomicBool = AtomicBool::new(false);

pub fn set_differentiable(on: bool) {
    DIFFERENTIABLE.store(on, Ordering::SeqCst);
}
pub fn differentiable() -> bool {
    DIFFERENTIABLE.load(Ordering::SeqCst)
}

pub struct ConvT1d {
    weight: Tensor,
    bias: Option<Tensor>,
    conv_weight: Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
    k: usize,
}

impl ConvT1d {
    pub fn load(
        vb: VarBuilder,
        cin: usize,
        cout: usize,
        k: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        groups: usize,
    ) -> Result<Self> {
        let weight = vb.get((cin, cout / groups, k), "weight")?;
        let bias = vb.get(cout, "bias").ok();
        let conv_weight = transpose_kernel_to_conv(&weight, groups)?;
        Ok(Self {
            weight,
            bias,
            conv_weight,
            stride,
            padding,
            output_padding,
            groups,
            k,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if differentiable() {
            self.forward_via_conv1d(x)
        } else {
            let y = x.conv_transpose1d(
                &self.weight,
                self.padding,
                self.output_padding,
                self.stride,
                1,
                self.groups,
            )?;
            self.add_bias(y)
        }
    }

    fn add_bias(&self, y: Tensor) -> Result<Tensor> {
        Ok(match &self.bias {
            Some(b) => y.broadcast_add(&b.reshape((1, b.dim(0)?, 1))?)?,
            None => y,
        })
    }

    pub fn forward_via_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let y = conv_transpose1d_diff(
            x,
            &self.conv_weight,
            self.k,
            self.stride,
            self.padding,
            self.output_padding,
            self.groups,
        )?;
        self.add_bias(y)
    }
}

pub fn transpose_kernel_to_conv(weight: &Tensor, groups: usize) -> Result<Tensor> {
    let (cin, cout_g, k) = weight.dims3()?;
    let w = weight
        .reshape((groups, cin / groups, cout_g, k))?
        .transpose(1, 2)?
        .contiguous()?
        .reshape((cout_g * groups, cin / groups, k))?;
    Ok(flip(&w, 2)?.contiguous()?)
}

pub fn conv_transpose1d_diff(
    x: &Tensor,
    conv_weight: &Tensor,
    k: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    groups: usize,
) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    let stuffed = if stride == 1 {
        x.clone()
    } else {
        let z = Tensor::zeros((b, c, t, stride - 1), x.dtype(), x.device())?;
        Tensor::cat(&[x.unsqueeze(3)?, z], 3)?
            .reshape((b, c, t * stride))?
            .narrow(2, 0, (t - 1) * stride + 1)?
    };
    let pad = k - 1 - padding;
    let padded = stuffed.pad_with_zeros(2, pad, pad + output_padding)?;
    Ok(padded.conv1d(conv_weight, 0, 1, 1, groups)?)
}

pub fn flip(x: &Tensor, dim: usize) -> Result<Tensor> {
    let n = x.dim(dim)?;
    let idx: Vec<u32> = (0..n as u32).rev().collect();
    Ok(x.index_select(&Tensor::new(idx, x.device())?, dim)?)
}

pub struct BiLstm {
    fwd: LSTM,
    bwd: LSTM,
}

impl BiLstm {
    pub fn load(vb: VarBuilder, in_dim: usize, hidden: usize) -> Result<Self> {
        let fwd = lstm(in_dim, hidden, LSTMConfig::default(), vb.clone())?;
        let bwd = lstm(
            in_dim,
            hidden,
            LSTMConfig {
                direction: Direction::Backward,
                ..LSTMConfig::default()
            },
            vb,
        )?;
        Ok(Self { fwd, bwd })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let f = self.fwd.states_to_tensor(&self.fwd.seq(x)?)?;
        let xr = flip(x, 1)?;
        let b = flip(&self.bwd.states_to_tensor(&self.bwd.seq(&xr)?)?, 1)?;
        Ok(Tensor::cat(&[f, b], D::Minus1)?)
    }
}

pub struct ChannelLayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl ChannelLayerNorm {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            gamma: vb.get(channels, "gamma")?,
            beta: vb.get(channels, "beta")?,
            eps: 1e-5,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let xt = x.transpose(1, 2)?.contiguous()?;
        let y = candle_nn::ops::layer_norm(&xt, &self.gamma, &self.beta, self.eps as f32)?;
        Ok(y.transpose(1, 2)?.contiguous()?)
    }
}

pub fn layer_norm_plain(x: &Tensor, eps: f64) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    Ok(xc.broadcast_div(&(var + eps)?.sqrt()?)?)
}

pub struct AdaLayerNorm {
    fc: Linear,
    channels: usize,
}

impl AdaLayerNorm {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            fc: candle_nn::linear(STYLE_DIM, channels * 2, vb.pp("fc"))?,
            channels,
        })
    }

    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let h = self.fc.forward(style)?;
        let gamma = h.narrow(1, 0, self.channels)?.unsqueeze(1)?;
        let beta = h.narrow(1, self.channels, self.channels)?.unsqueeze(1)?;
        let y = layer_norm_plain(x, 1e-5)?;
        Ok(y.broadcast_mul(&(gamma + 1.0)?)?.broadcast_add(&beta)?)
    }
}

pub struct InstanceNorm1d {
    weight: Tensor,
    bias: Tensor,
}

impl InstanceNorm1d {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        Ok(Self {
            weight: vb.get(channels, "weight")?,
            bias: vb.get(channels, "bias")?,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc = x.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let y = xc.broadcast_div(&(var + 1e-5)?.sqrt()?)?;
        let c = self.weight.dim(0)?;
        Ok(y.broadcast_mul(&self.weight.reshape((1, c, 1))?)?
            .broadcast_add(&self.bias.reshape((1, c, 1))?)?)
    }
}

pub struct AdaIn1d {
    norm: InstanceNorm1d,
    pub fc_weight: Tensor,
    pub fc_bias: Tensor,
    pub fc_name: String,
    channels: usize,
}

impl AdaIn1d {
    pub fn load(vb: VarBuilder, channels: usize) -> Result<Self> {
        let fc = vb.pp("fc");
        Ok(Self {
            norm: InstanceNorm1d::load(vb.pp("norm"), channels)?,
            fc_weight: fc.get((channels * 2, STYLE_DIM), "weight")?,
            fc_bias: fc.get(channels * 2, "bias")?,
            fc_name: fc.prefix(),
            channels,
        })
    }

    pub fn make_trainable(&mut self) -> Result<Vec<Var>> {
        let w = Var::from_tensor(&self.fc_weight)?;
        let b = Var::from_tensor(&self.fc_bias)?;
        self.fc_weight = w.as_tensor().clone();
        self.fc_bias = b.as_tensor().clone();
        Ok(vec![w, b])
    }

    pub fn set_fc(&mut self, weight: Tensor, bias: Tensor) {
        self.fc_weight = weight;
        self.fc_bias = bias;
    }
    fn fc(&self, style: &Tensor) -> Result<Tensor> {
        Ok(style
            .matmul(&self.fc_weight.t()?)?
            .broadcast_add(&self.fc_bias)?)
    }

    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let h = self.fc(style)?;
        let gamma = h.narrow(1, 0, self.channels)?.unsqueeze(2)?;
        let beta = h.narrow(1, self.channels, self.channels)?.unsqueeze(2)?;
        let y = self.norm.forward(x)?;
        Ok(y.broadcast_mul(&(gamma + 1.0)?)?.broadcast_add(&beta)?)
    }
}

pub fn leaky_relu(x: &Tensor, slope: f64) -> Result<Tensor> {
    Ok(candle_nn::ops::leaky_relu(x, slope)?)
}

pub fn snake(x: &Tensor, alpha: &Tensor) -> Result<Tensor> {
    let ax = x.broadcast_mul(alpha)?;
    let s = ax.sin()?.sqr()?;
    Ok((x + s.broadcast_div(alpha)?)?)
}

pub fn conv1d_load(
    vb: VarBuilder,
    cin: usize,
    cout: usize,
    k: usize,
    cfg: Conv1dConfig,
    bias: bool,
) -> Result<Conv1d> {
    Ok(if bias {
        candle_nn::conv1d(cin, cout, k, cfg, vb)?
    } else {
        candle_nn::conv1d_no_bias(cin, cout, k, cfg, vb)?
    })
}

pub fn same_conv(k: usize, dilation: usize) -> Conv1dConfig {
    Conv1dConfig {
        padding: (k * dilation - dilation) / 2,
        stride: 1,
        dilation,
        groups: 1,
        cudnn_fwd_algo: None,
    }
}

pub fn upsample2(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    Ok(x.unsqueeze(3)?
        .repeat((1, 1, 1, 2))?
        .reshape((b, c, t * 2))?)
}

pub struct AdainResBlk1d {
    pub norm1: AdaIn1d,
    pub norm2: AdaIn1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv1x1: Option<Conv1d>,
    pool: Option<ConvT1d>,
}

impl AdainResBlk1d {
    pub fn load(vb: VarBuilder, dim_in: usize, dim_out: usize, upsample: bool) -> Result<Self> {
        let conv1x1 = if dim_in != dim_out {
            Some(conv1d_load(
                vb.pp("conv1x1"),
                dim_in,
                dim_out,
                1,
                Conv1dConfig::default(),
                false,
            )?)
        } else {
            None
        };
        let pool = if upsample {
            Some(ConvT1d::load(
                vb.pp("pool"),
                dim_in,
                dim_in,
                3,
                2,
                1,
                1,
                dim_in,
            )?)
        } else {
            None
        };
        Ok(Self {
            norm1: AdaIn1d::load(vb.pp("norm1"), dim_in)?,
            norm2: AdaIn1d::load(vb.pp("norm2"), dim_out)?,
            conv1: conv1d_load(vb.pp("conv1"), dim_in, dim_out, 3, same_conv(3, 1), true)?,
            conv2: conv1d_load(vb.pp("conv2"), dim_out, dim_out, 3, same_conv(3, 1), true)?,
            conv1x1,
            pool,
        })
    }

    pub fn forward(&self, x: &Tensor, style: &Tensor) -> Result<Tensor> {
        let mut h = leaky_relu(&self.norm1.forward(x, style)?, 0.2)?;
        if let Some(pool) = &self.pool {
            h = pool.forward(&h)?;
        }
        h = self.conv1.forward(&h)?;
        h = leaky_relu(&self.norm2.forward(&h, style)?, 0.2)?;
        h = self.conv2.forward(&h)?;

        let mut s = if self.pool.is_some() {
            upsample2(x)?
        } else {
            x.clone()
        };
        if let Some(c) = &self.conv1x1 {
            s = c.forward(&s)?;
        }
        Ok(((h + s)? * (0.5f64).sqrt())?)
    }

    pub fn adains_mut(&mut self) -> Vec<&mut AdaIn1d> {
        vec![&mut self.norm1, &mut self.norm2]
    }

    pub fn upsamples(&self) -> bool {
        self.pool.is_some()
    }
}

pub fn assert_close(name: &str, got: &Tensor, want: &Tensor, tol: f32) -> Result<f32> {
    if got.shape() != want.shape() {
        return Err(anyhow!(
            "{name}: shape {:?} != reference {:?}",
            got.shape(),
            want.shape()
        ));
    }
    let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?.abs()?;
    let max = diff.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    let scale = want
        .to_dtype(DType::F32)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?
        .max(1e-6);
    if max > tol * scale.max(1.0) {
        return Err(anyhow!(
            "{name}: max abs diff {max:.5} (ref scale {scale:.3}) exceeds {tol}"
        ));
    }
    Ok(max)
}

pub fn cpu() -> Device {
    Device::Cpu
}
