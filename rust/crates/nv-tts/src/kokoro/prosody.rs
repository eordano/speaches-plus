use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

use super::frontend::HIDDEN;
use super::layers::{conv1d_load, AdainResBlk1d, BiLstm, STYLE_DIM};

pub struct Prosody {
    shared: BiLstm,
    f0: Vec<AdainResBlk1d>,
    n: Vec<AdainResBlk1d>,
    f0_proj: Conv1d,
    n_proj: Conv1d,
}

impl Prosody {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let blocks = |name: &str| -> Result<Vec<AdainResBlk1d>> {
            Ok(vec![
                AdainResBlk1d::load(vb.pp(format!("{name}.0")), HIDDEN, HIDDEN, false)?,
                AdainResBlk1d::load(vb.pp(format!("{name}.1")), HIDDEN, HIDDEN / 2, true)?,
                AdainResBlk1d::load(vb.pp(format!("{name}.2")), HIDDEN / 2, HIDDEN / 2, false)?,
            ])
        };
        Ok(Self {
            shared: BiLstm::load(vb.pp("shared"), HIDDEN + STYLE_DIM, HIDDEN / 2)?,
            f0: blocks("F0")?,
            n: blocks("N")?,
            f0_proj: conv1d_load(
                vb.pp("F0_proj"),
                HIDDEN / 2,
                1,
                1,
                Conv1dConfig::default(),
                true,
            )?,
            n_proj: conv1d_load(
                vb.pp("N_proj"),
                HIDDEN / 2,
                1,
                1,
                Conv1dConfig::default(),
                true,
            )?,
        })
    }

    pub fn forward(&self, en: &Tensor, style: &Tensor) -> Result<(Tensor, Tensor)> {
        let x = self.shared.forward(&en.transpose(1, 2)?.contiguous()?)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let mut f0 = x.clone();
        for b in &self.f0 {
            f0 = b.forward(&f0, style)?;
        }
        let f0 = self.f0_proj.forward(&f0)?.squeeze(1)?;
        let mut n = x;
        for b in &self.n {
            n = b.forward(&n, style)?;
        }
        let n = self.n_proj.forward(&n)?.squeeze(1)?;
        Ok((f0, n))
    }

    pub fn adains_mut(&mut self) -> Vec<&mut super::layers::AdaIn1d> {
        self.f0
            .iter_mut()
            .chain(self.n.iter_mut())
            .flat_map(|b| b.adains_mut())
            .collect()
    }
}
