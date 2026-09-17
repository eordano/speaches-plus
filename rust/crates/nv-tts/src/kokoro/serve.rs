use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};

use super::generator::Rng;
use super::train::{adapt, apply_adain, Adaptation, Encoders, Targets, TrainConfig};
use super::KokoroFrontEnd;

pub struct KokoroCandle {
    model: Mutex<KokoroFrontEnd>,
    pub base_adain: HashMap<String, Tensor>,
    pub device: Device,
    encoders: Mutex<Option<Encoders>>,
    encoder_dirs: Option<(std::path::PathBuf, std::path::PathBuf)>,
}

impl KokoroCandle {
    pub fn load(weights: &Path, encoders: Option<(&Path, &Path)>, device: &Device) -> Result<Self> {
        let mut c = Self::from_parts(KokoroFrontEnd::load(weights, device)?, None);
        c.encoder_dirs = encoders.map(|(q, w)| (q.to_path_buf(), w.to_path_buf()));
        Ok(c)
    }

    pub fn from_parts(mut model: KokoroFrontEnd, encoders: Option<Encoders>) -> Self {
        let device = model.device.clone();
        let names = model.adain_names();
        let mut base = HashMap::with_capacity(names.len() * 2);
        for (a, name) in model
            .decoder
            .adains_mut()
            .into_iter()
            .chain(model.generator.adains_mut())
            .zip(&names)
        {
            base.insert(format!("decoder.{name}.weight"), a.fc_weight.clone());
            base.insert(format!("decoder.{name}.bias"), a.fc_bias.clone());
        }
        Self {
            model: Mutex::new(model),
            base_adain: base,
            device,
            encoders: Mutex::new(encoders),
            encoder_dirs: None,
        }
    }

    pub fn has_encoders(&self) -> bool {
        self.encoder_dirs.is_some()
            || self
                .encoders
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .is_some()
    }

    pub fn n_adain(&self) -> usize {
        self.base_adain.len() / 2
    }

    pub fn synthesize(
        &self,
        ids: &[u32],
        style: &[f32],
        speed: f64,
        adapter: Option<&HashMap<String, Tensor>>,
        seed: u64,
    ) -> Result<Vec<f32>> {
        let mut m = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let style_t = Tensor::from_vec(style.to_vec(), (1, style.len()), &self.device)?;
        let mut rng = Rng(seed.max(1));
        let result = (|| -> Result<Vec<f32>> {
            if let Some(a) = adapter {
                let n = apply_adain(&mut m, a)?;
                if n == 0 {
                    return Err(anyhow!(
                        "adapter matched none of the {} AdaIN layers",
                        self.n_adain()
                    ));
                }
            }
            let (audio, _dur) = m.synthesize_tokens(ids, &style_t, speed, false, &mut rng)?;
            Ok(audio.to_device(&Device::Cpu)?.flatten_all()?.to_vec1()?)
        })();
        if adapter.is_some() {
            apply_adain(&mut m, &self.base_adain)?;
        }
        result
    }

    fn encoders(&self) -> Result<std::sync::MutexGuard<'_, Option<Encoders>>> {
        let mut g = self
            .encoders
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if g.is_none() {
            let (q, w) = self
                .encoder_dirs
                .as_ref()
                .ok_or_else(|| anyhow!("fitting needs the speaker encoder directories"))?;
            *g = Some(Encoders::load(q, w, &self.device)?);
        }
        Ok(g)
    }

    pub fn targets(&self, refs: &[Vec<f32>]) -> Result<Targets> {
        let g = self.encoders()?;
        let enc = g.as_ref().expect("loaded");
        let ts: Vec<Tensor> = refs
            .iter()
            .map(|r| Tensor::from_vec(r.clone(), r.len(), &self.device))
            .collect::<candle_core::Result<_>>()?;
        enc.targets(&ts)
    }

    pub fn score(&self, wav24: &[f32], targets: &Targets) -> Result<f32> {
        let g = self.encoders()?;
        let enc = g.as_ref().expect("loaded");
        let w = Tensor::from_vec(wav24.to_vec(), wav24.len(), &self.device)?;
        Ok(super::cosine(&enc.wespeaker_embed(&w)?, &targets.wespeaker)?.to_scalar::<f32>()?)
    }

    pub fn fit_adapter(
        &self,
        targets: &Targets,
        style: &[f32],
        train: &[Vec<u32>],
        heldout: &[Vec<u32>],
        cfg: &TrainConfig,
        log: impl FnMut(&super::train::EpochReport),
    ) -> Result<Adaptation> {
        let g = self.encoders()?;
        let enc = g.as_ref().expect("loaded");
        let mut m = self
            .model
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let style_t = Tensor::from_vec(style.to_vec(), (1, style.len()), &self.device)?;
        let out = adapt(&mut m, enc, targets, &style_t, train, heldout, cfg, log);

        apply_adain(&mut m, &self.base_adain)?;
        super::set_differentiable(false);
        out
    }
}
