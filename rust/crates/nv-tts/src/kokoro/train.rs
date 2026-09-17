use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};
use candle_core::{Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW};

use super::generator::Rng;
use super::melspec::LogMel24k;
use super::wespeaker::{resample_24k_to_16k, WeSpeaker};
use super::{cosine, KokoroFrontEnd};
use crate::SpeakerEncoder;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub epochs: usize,
    pub patience: usize,
    pub min_delta: f32,
    pub lr: f64,
    pub anchor: f64,

    pub wmix: f64,
    pub seed: u64,
    pub window_frames: Option<usize>,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 12,
            patience: 3,
            min_delta: 0.005,
            lr: 2e-4,
            anchor: 1.0,
            wmix: 0.5,
            seed: 0,
            window_frames: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct EpochReport {
    pub epoch: usize,
    pub train_loss: f32,
    pub heldout_wespeaker: f32,
    pub heldout_qwen: f32,
}

pub struct Targets {
    pub qwen: Tensor,
    pub wespeaker: Tensor,
}

pub struct Adaptation {
    pub reports: Vec<EpochReport>,
    pub best_heldout: f32,
    pub best_epoch: usize,

    pub adain: HashMap<String, Tensor>,
    pub stop: &'static str,
}

pub const A_ZERO_MIX_SKIPS_THE_WESPEAKER_TAPE_AND_A_UNIT_MIX_SKIPS_THE_QWEN_TAPE: &str =
    "an encoder whose weight in the training loss is zero is not run under autograd at all, \
     because ResNet293 alone retains about 0.6 GiB of activations per second of audio; the \
     held-out score still reports both cosines, so a WeSpeaker-free training step is still \
     selected by WeSpeaker";

pub const VALIDATION_SWAPS_IN_DETACHED_ADAIN_WEIGHTS_SO_A_HELD_OUT_SENTENCE_RECORDS_NO_TAPE: &str =
    "every op on a Var-backed AdaIN weight is recorded whether or not a backward follows, so a \
     full-sentence validation synthesis used to hold the largest tape of the whole epoch; the \
     held-out pass runs on detached copies of the same weights and puts the tracked ones back \
     afterwards, which leaves the training window's tape as the epoch's peak";

pub struct Encoders {
    pub qwen: SpeakerEncoder,
    pub mel24: LogMel24k,
    pub wespeaker: WeSpeaker,
}

impl Encoders {
    pub fn load(qwen_dir: &Path, wespeaker_dir: &Path, device: &Device) -> Result<Self> {
        Self::from_parts(
            super::load_speaker_encoder(qwen_dir, device)?,
            WeSpeaker::load(wespeaker_dir, device)?,
            device,
        )
    }

    pub fn from_parts(qwen: SpeakerEncoder, wespeaker: WeSpeaker, device: &Device) -> Result<Self> {
        Ok(Self {
            qwen,
            mel24: LogMel24k::new(device)?,
            wespeaker,
        })
    }
    pub fn qwen_embed(&self, wav24: &Tensor) -> Result<Tensor> {
        Ok(self
            .qwen
            .forward(&self.mel24.forward(wav24)?)?
            .flatten_all()?)
    }
    pub fn wespeaker_embed(&self, wav24: &Tensor) -> Result<Tensor> {
        self.wespeaker.embed_wav16k(&resample_24k_to_16k(wav24)?)
    }

    pub fn targets(&self, refs: &[Tensor]) -> Result<Targets> {
        let mut q: Option<Tensor> = None;
        let mut w: Option<Tensor> = None;
        for r in refs {
            let eq = unit(&self.qwen_embed(r)?)?;
            let ew = unit(&self.wespeaker_embed(r)?)?;
            q = Some(match q {
                None => eq,
                Some(a) => (a + eq)?,
            });
            w = Some(match w {
                None => ew,
                Some(a) => (a + ew)?,
            });
        }
        Ok(Targets {
            qwen: unit(&q.ok_or_else(|| anyhow!("no reference clips"))?)?.detach(),
            wespeaker: unit(&w.ok_or_else(|| anyhow!("no reference clips"))?)?.detach(),
        })
    }
}

fn unit(v: &Tensor) -> Result<Tensor> {
    let n = v.sqr()?.sum_all()?.sqrt()?;
    Ok(v.broadcast_div(&(n + 1e-9)?)?)
}

pub fn adapt(
    model: &mut KokoroFrontEnd,
    enc: &Encoders,
    targets: &Targets,
    style: &Tensor,
    train: &[Vec<u32>],
    heldout: &[Vec<u32>],
    cfg: &TrainConfig,
    mut log: impl FnMut(&EpochReport),
) -> Result<Adaptation> {
    let names = model.adain_names();
    let vars = model.make_adain_trainable()?;
    if vars.len() != names.len() * 2 {
        return Err(anyhow!(
            "adain vars {} != 2 x names {}",
            vars.len(),
            names.len()
        ));
    }
    let init: Vec<Tensor> = vars
        .iter()
        .map(|v| v.as_tensor().detach().copy())
        .collect::<candle_core::Result<Vec<_>>>()?;
    let mut opt = AdamW::new(
        vars.clone(),
        ParamsAdamW {
            lr: cfg.lr,
            ..Default::default()
        },
    )?;
    let mut es = EarlyStopping::new(cfg.patience, cfg.min_delta, cfg.epochs);
    let mut best: Option<(f32, usize, Vec<Tensor>)> = None;
    let mut reports = Vec::new();
    let mut rng = Rng(cfg.seed.max(1));
    let heldout_score = |model: &mut KokoroFrontEnd, rng: &mut Rng| -> Result<(f32, f32)> {
        let _ = VALIDATION_SWAPS_IN_DETACHED_ADAIN_WEIGHTS_SO_A_HELD_OUT_SENTENCE_RECORDS_NO_TAPE;
        let mut frozen = HashMap::with_capacity(vars.len());
        let mut live = HashMap::with_capacity(vars.len());
        for (k, name) in names.iter().enumerate() {
            for (leaf, var) in [("weight", &vars[2 * k]), ("bias", &vars[2 * k + 1])] {
                let key = format!("decoder.{name}.{leaf}");
                frozen.insert(key.clone(), var.as_tensor().detach());
                live.insert(key, var.as_tensor().clone());
            }
        }
        apply_adain(model, &frozen)?;
        let scored = (|| -> Result<(f32, f32)> {
            let (mut sw, mut sq) = (0f32, 0f32);
            for ids in heldout {
                let (audio, _) = model.synthesize_tokens(ids, style, 1.0, true, rng)?;
                sw += cosine(&enc.wespeaker_embed(&audio)?, &targets.wespeaker)?
                    .to_scalar::<f32>()?;
                sq += cosine(&enc.qwen_embed(&audio)?, &targets.qwen)?.to_scalar::<f32>()?;
            }
            let n = heldout.len().max(1) as f32;
            Ok((sw / n, sq / n))
        })();
        apply_adain(model, &live)?;
        scored
    };
    let mut stop = "budget";
    for epoch in 0..cfg.epochs {
        let mut order: Vec<usize> = (0..train.len()).collect();

        for i in (1..order.len()).rev() {
            let j = (rng.uniform() * (i + 1) as f32) as usize;
            order.swap(i, j.min(i));
        }
        let mut loss_sum = 0f32;
        for &j in &order {
            let (audio, _) = model.synthesize_tokens_window(
                &train[j],
                style,
                1.0,
                true,
                &mut rng,
                cfg.window_frames,
            )?;
            let _ = A_ZERO_MIX_SKIPS_THE_WESPEAKER_TAPE_AND_A_UNIT_MIX_SKIPS_THE_QWEN_TAPE;
            let loss = match cfg.wmix {
                w if w <= 0.0 => (1.0 - cosine(&enc.qwen_embed(&audio)?, &targets.qwen)?)?,
                w if w >= 1.0 => {
                    (1.0 - cosine(&enc.wespeaker_embed(&audio)?, &targets.wespeaker)?)?
                }
                w => {
                    let cq = cosine(&enc.qwen_embed(&audio)?, &targets.qwen)?;
                    let cw = cosine(&enc.wespeaker_embed(&audio)?, &targets.wespeaker)?;
                    (((1.0 - cq)? * (1.0 - w))? + ((1.0 - cw)? * w)?)?
                }
            };
            let mut anchor = Tensor::zeros((), audio.dtype(), audio.device())?;
            for (v, i) in vars.iter().zip(&init) {
                anchor = (anchor + (v.as_tensor() - i)?.sqr()?.sum_all()?)?;
            }
            let total = (loss.clone() + (anchor * cfg.anchor)?)?;
            let grads = total.backward()?;
            opt.step(&grads)?;
            loss_sum += loss.to_scalar::<f32>()?;
        }
        let (hw, hq) = heldout_score(model, &mut rng)?;
        let rep = EpochReport {
            epoch,
            train_loss: loss_sum / train.len().max(1) as f32,
            heldout_wespeaker: hw,
            heldout_qwen: hq,
        };
        log(&rep);
        reports.push(rep);
        let should_stop = es.update(hw);
        if es.improved() {
            best = Some((
                hw,
                epoch,
                vars.iter()
                    .map(|v| v.as_tensor().detach().copy())
                    .collect::<candle_core::Result<Vec<_>>>()?,
            ));
        }
        if should_stop {
            stop = if es.steps < cfg.epochs {
                "patience"
            } else {
                "budget"
            };
            break;
        }
    }
    let (best_heldout, best_epoch, best_w) = best.ok_or_else(|| anyhow!("no epoch ran"))?;

    let mut adain = HashMap::with_capacity(best_w.len());
    for (k, name) in names.iter().enumerate() {
        let w = &best_w[2 * k];
        let b = &best_w[2 * k + 1];
        vars[2 * k].set(w)?;
        vars[2 * k + 1].set(b)?;
        adain.insert(format!("decoder.{name}.weight"), w.clone());
        adain.insert(format!("decoder.{name}.bias"), b.clone());
    }
    super::set_differentiable(false);
    Ok(Adaptation {
        reports,
        best_heldout,
        best_epoch,
        adain,
        stop,
    })
}

pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f32,
    pub budget: usize,
    pub best: f32,
    pub best_step: usize,
    pub steps: usize,
    since: usize,
}
impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f32, budget: usize) -> Self {
        Self {
            patience,
            min_delta,
            budget,
            best: f32::NEG_INFINITY,
            best_step: 0,
            steps: 0,
            since: 0,
        }
    }
    pub fn update(&mut self, score: f32) -> bool {
        self.steps += 1;
        if score > self.best + self.min_delta {
            self.best = score;
            self.best_step = self.steps - 1;
            self.since = 0;
        } else {
            self.since += 1;
        }
        self.since >= self.patience || self.steps >= self.budget
    }
    pub fn improved(&self) -> bool {
        self.since == 0
    }
}

pub fn apply_adain(model: &mut KokoroFrontEnd, adain: &HashMap<String, Tensor>) -> Result<usize> {
    let names = model.adain_names();
    let device = model.device.clone();
    let mut n = 0;
    let mut adains: Vec<_> = model
        .decoder
        .adains_mut()
        .into_iter()
        .chain(model.generator.adains_mut())
        .collect();
    for (a, name) in adains.iter_mut().zip(&names) {
        let (w, b) = (
            adain.get(&format!("decoder.{name}.weight")),
            adain.get(&format!("decoder.{name}.bias")),
        );
        if let (Some(w), Some(b)) = (w, b) {
            if w.dims() != a.fc_weight.dims() || b.dims() != a.fc_bias.dims() {
                return Err(anyhow!(
                    "adapter decoder.{name}: shape {:?}/{:?}, model {:?}/{:?}",
                    w.dims(),
                    b.dims(),
                    a.fc_weight.dims(),
                    a.fc_bias.dims()
                ));
            }
            let w = w.to_device(&device)?.to_dtype(a.fc_weight.dtype())?;
            let b = b.to_device(&device)?.to_dtype(a.fc_bias.dtype())?;
            a.set_fc(w, b);
            n += 1;
        }
    }
    Ok(n)
}
