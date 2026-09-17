use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

use super::frontend::HIDDEN;
use super::layers::{conv1d_load, AdainResBlk1d};

pub struct DecoderBody {
    encode: AdainResBlk1d,
    decode: Vec<AdainResBlk1d>,
    f0_conv: Conv1d,
    n_conv: Conv1d,
    asr_res: Conv1d,
}

impl DecoderBody {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let stride2 = Conv1dConfig {
            padding: 1,
            stride: 2,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        Ok(Self {
            encode: AdainResBlk1d::load(vb.pp("encode"), HIDDEN + 2, 1024, false)?,
            decode: vec![
                AdainResBlk1d::load(vb.pp("decode.0"), 1024 + 2 + 64, 1024, false)?,
                AdainResBlk1d::load(vb.pp("decode.1"), 1024 + 2 + 64, 1024, false)?,
                AdainResBlk1d::load(vb.pp("decode.2"), 1024 + 2 + 64, 1024, false)?,
                AdainResBlk1d::load(vb.pp("decode.3"), 1024 + 2 + 64, 512, true)?,
            ],
            f0_conv: conv1d_load(vb.pp("F0_conv"), 1, 1, 3, stride2, true)?,
            n_conv: conv1d_load(vb.pp("N_conv"), 1, 1, 3, stride2, true)?,
            asr_res: conv1d_load(
                vb.pp("asr_res.0"),
                HIDDEN,
                64,
                1,
                Conv1dConfig::default(),
                true,
            )?,
        })
    }

    pub fn forward(
        &self,
        asr: &Tensor,
        f0_curve: &Tensor,
        n_curve: &Tensor,
        style: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let f0 = self.f0_conv.forward(&f0_curve.unsqueeze(1)?)?;
        let n = self.n_conv.forward(&n_curve.unsqueeze(1)?)?;
        let mut x = self
            .encode
            .forward(&Tensor::cat(&[asr, &f0, &n], 1)?, style)?;
        let enc = x.clone();
        let asr_res = self.asr_res.forward(asr)?;
        let mut res = true;
        for blk in &self.decode {
            if res {
                x = Tensor::cat(&[&x, &asr_res, &f0, &n], 1)?;
            }
            x = blk.forward(&x, style)?;
            if blk.upsamples() {
                res = false;
            }
        }
        Ok((enc, x))
    }

    pub fn adains_mut(&mut self) -> Vec<&mut super::layers::AdaIn1d> {
        std::iter::once(&mut self.encode)
            .chain(self.decode.iter_mut())
            .flat_map(|b| b.adains_mut())
            .collect()
    }
}
