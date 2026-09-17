use anyhow::Result;
use candle_core::{Module, Tensor, D};
use candle_nn::{embedding, linear, Conv1d, Embedding, Linear, VarBuilder};

use super::layers::{
    conv1d_load, leaky_relu, same_conv, AdaLayerNorm, BiLstm, ChannelLayerNorm, STYLE_DIM,
};

pub const HIDDEN: usize = 512;
pub const N_TOKEN: usize = 178;
pub const MAX_DUR: usize = 50;

pub struct TextEncoder {
    embedding: Embedding,
    cnn: Vec<(Conv1d, ChannelLayerNorm)>,
    lstm: BiLstm,
}

impl TextEncoder {
    pub fn load(vb: VarBuilder, depth: usize, kernel: usize) -> Result<Self> {
        let mut cnn = Vec::with_capacity(depth);
        for i in 0..depth {
            let c = vb.pp(format!("cnn.{i}"));
            cnn.push((
                conv1d_load(
                    c.pp("0"),
                    HIDDEN,
                    HIDDEN,
                    kernel,
                    same_conv(kernel, 1),
                    true,
                )?,
                ChannelLayerNorm::load(c.pp("1"), HIDDEN)?,
            ));
        }
        Ok(Self {
            embedding: embedding(N_TOKEN, HIDDEN, vb.pp("embedding"))?,
            cnn,
            lstm: BiLstm::load(vb.pp("lstm"), HIDDEN, HIDDEN / 2)?,
        })
    }
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let mut x = self
            .embedding
            .forward(input_ids)?
            .transpose(1, 2)?
            .contiguous()?;
        for (conv, ln) in &self.cnn {
            x = leaky_relu(&ln.forward(&conv.forward(&x)?)?, 0.2)?;
        }
        let y = self.lstm.forward(&x.transpose(1, 2)?.contiguous()?)?;
        Ok(y.transpose(1, 2)?.contiguous()?)
    }
}

enum DurBlock {
    Lstm(BiLstm),
    Norm(AdaLayerNorm),
}

pub struct DurationEncoder {
    blocks: Vec<DurBlock>,
}

impl DurationEncoder {
    pub fn load(vb: VarBuilder, nlayers: usize) -> Result<Self> {
        let mut blocks = Vec::with_capacity(nlayers * 2);
        for i in 0..nlayers {
            blocks.push(DurBlock::Lstm(BiLstm::load(
                vb.pp(format!("lstms.{}", 2 * i)),
                HIDDEN + STYLE_DIM,
                HIDDEN / 2,
            )?));
            blocks.push(DurBlock::Norm(AdaLayerNorm::load(
                vb.pp(format!("lstms.{}", 2 * i + 1)),
                HIDDEN,
            )?));
        }
        Ok(Self { blocks })
    }
    pub fn forward(&self, d_en: &Tensor, style: &Tensor) -> Result<Tensor> {
        let (b, _c, t) = d_en.dims3()?;
        let s_t = style.unsqueeze(2)?.repeat((1, 1, t))?;
        let mut x = Tensor::cat(&[d_en, &s_t], 1)?;
        for blk in &self.blocks {
            x = match blk {
                DurBlock::Lstm(l) => l
                    .forward(&x.transpose(1, 2)?.contiguous()?)?
                    .transpose(1, 2)?
                    .contiguous()?,
                DurBlock::Norm(n) => {
                    let y = n.forward(&x.transpose(1, 2)?.contiguous()?, style)?;
                    Tensor::cat(&[y.transpose(1, 2)?.contiguous()?, s_t.clone()], 1)?
                }
            };
        }
        let _ = b;
        Ok(x.transpose(1, 2)?.contiguous()?)
    }
}

pub struct DurationPath {
    pub bert_encoder: Linear,
    pub text_encoder: DurationEncoder,
    lstm: BiLstm,
    duration_proj: Linear,
}

impl DurationPath {
    pub fn load(vb_root: VarBuilder, nlayers: usize) -> Result<Self> {
        let p = vb_root.pp("predictor");
        Ok(Self {
            bert_encoder: linear(768, HIDDEN, vb_root.pp("bert_encoder"))?,
            text_encoder: DurationEncoder::load(p.pp("text_encoder"), nlayers)?,
            lstm: BiLstm::load(p.pp("lstm"), HIDDEN + STYLE_DIM, HIDDEN / 2)?,
            duration_proj: linear(HIDDEN, MAX_DUR, p.pp("duration_proj.linear_layer"))?,
        })
    }

    pub fn forward(
        &self,
        bert_hidden: &Tensor,
        style: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let d_en = self
            .bert_encoder
            .forward(bert_hidden)?
            .transpose(1, 2)?
            .contiguous()?;
        let d = self.text_encoder.forward(&d_en, style)?;
        let x = self.lstm.forward(&d)?;
        let logits = self.duration_proj.forward(&x)?;
        Ok((d_en, d, logits))
    }
}

pub fn durations(logits: &Tensor, speed: f64) -> Result<Vec<usize>> {
    let s = candle_nn::ops::sigmoid(logits)?.sum(D::Minus1)?;
    let v: Vec<f32> = s.flatten_all()?.to_vec1()?;
    Ok(v.iter()
        .map(|d| ((*d as f64 / speed).round() as i64).max(1) as usize)
        .collect())
}

pub fn alignment(dur: &[usize], device: &candle_core::Device) -> Result<Tensor> {
    let total: usize = dur.iter().sum();
    let mut m = vec![0f32; dur.len() * total];
    let mut j = 0;
    for (i, &d) in dur.iter().enumerate() {
        for _ in 0..d {
            m[i * total + j] = 1.0;
            j += 1;
        }
    }
    Ok(Tensor::from_vec(m, (dur.len(), total), device)?)
}
