use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

pub const ADAM_BETA1: f64 = 0.9;
pub const ADAM_BETA2: f64 = 0.999;
pub const ADAM_EPS: f64 = 1e-8;
pub const NORMALIZATION_EPS: f32 = 1e-6;
pub const DECODER_RENORM_EPS: f64 = 1e-8;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainConfig {
    pub width: usize,
    pub top_k: usize,
    pub steps: usize,
    pub batch: usize,
    pub lr: f64,
    pub l1: f64,
    pub seed: u64,
}

pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(0x2545_F491_4F6C_DD1D)
            | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
    pub fn normal(&mut self) -> f32 {
        let u1 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

pub fn device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(d) = Device::new_cuda(0) {
            return d;
        }
    }
    Device::Cpu
}

pub struct Sae {
    pub encoder: Var,
    pub encoder_bias: Var,
    pub decoder: Var,
    pub decoder_bias: Var,
    pub mu: Tensor,
    pub sd: Tensor,
    pub top_k: usize,
}

impl Sae {
    pub fn width(&self) -> usize {
        self.encoder.dims()[1]
    }

    pub fn normalize(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.broadcast_sub(&self.mu)?.broadcast_div(&self.sd)?)
    }

    pub fn encode(&self, z: &Tensor) -> Result<Tensor> {
        let pre = z
            .matmul(self.encoder.as_tensor())?
            .broadcast_add(self.encoder_bias.as_tensor())?;
        topk_relu(&pre, self.top_k)
    }

    pub fn decode(&self, h: &Tensor) -> Result<Tensor> {
        Ok(h.matmul(self.decoder.as_tensor())?
            .broadcast_add(self.decoder_bias.as_tensor())?)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let mut m = HashMap::new();
        m.insert("encoder".to_string(), self.encoder.as_tensor().clone());
        m.insert(
            "encoder_bias".to_string(),
            self.encoder_bias.as_tensor().clone(),
        );
        m.insert("decoder".to_string(), self.decoder.as_tensor().clone());
        m.insert(
            "decoder_bias".to_string(),
            self.decoder_bias.as_tensor().clone(),
        );
        m.insert("mu".to_string(), self.mu.clone());
        m.insert("sd".to_string(), self.sd.clone());
        m.insert(
            "top_k".to_string(),
            Tensor::new(&[self.top_k as f32], self.mu.device())?,
        );
        candle_core::safetensors::save(&m, path)?;
        Ok(())
    }

    pub fn load(path: &Path, dev: &Device) -> Result<Self> {
        let m = candle_core::safetensors::load(path, dev)?;
        let get = |k: &str| {
            m.get(k)
                .cloned()
                .with_context(|| format!("missing {k} in {}", path.display()))
        };
        let top_k = get("top_k")?.to_vec1::<f32>()?[0] as usize;
        Ok(Sae {
            encoder: Var::from_tensor(&get("encoder")?)?,
            encoder_bias: Var::from_tensor(&get("encoder_bias")?)?,
            decoder: Var::from_tensor(&get("decoder")?)?,
            decoder_bias: Var::from_tensor(&get("decoder_bias")?)?,
            mu: get("mu")?,
            sd: get("sd")?,
            top_k,
        })
    }
}

pub fn topk_relu(pre: &Tensor, k: usize) -> Result<Tensor> {
    let width = pre.dims()[1];
    let relu = pre.relu()?;
    if k >= width {
        return Ok(relu);
    }
    let (sorted, _) = pre.sort_last_dim(false)?;
    let thr = sorted.narrow(1, k - 1, 1)?;
    let mask = pre.broadcast_ge(&thr)?.to_dtype(pre.dtype())?;
    Ok((relu * mask)?)
}

struct Adam {
    m: Vec<Tensor>,
    v: Vec<Tensor>,
    t: usize,
}

impl Adam {
    fn new(vars: &[&Var]) -> Result<Self> {
        let z = |v: &Var| v.as_tensor().zeros_like();
        Ok(Adam {
            m: vars
                .iter()
                .map(|v| z(v))
                .collect::<candle_core::Result<_>>()?,
            v: vars
                .iter()
                .map(|v| z(v))
                .collect::<candle_core::Result<_>>()?,
            t: 0,
        })
    }

    fn step(
        &mut self,
        vars: &[&Var],
        grads: &candle_core::backprop::GradStore,
        lr: f64,
    ) -> Result<()> {
        self.t += 1;
        let bc1 = 1.0 - ADAM_BETA1.powi(self.t as i32);
        let bc2 = 1.0 - ADAM_BETA2.powi(self.t as i32);
        for (i, var) in vars.iter().enumerate() {
            let Some(g) = grads.get(var.as_tensor()) else {
                continue;
            };
            let g = g.detach();
            self.m[i] = ((&self.m[i] * ADAM_BETA1)? + (&g * (1.0 - ADAM_BETA1))?)?.detach();
            self.v[i] = ((&self.v[i] * ADAM_BETA2)? + (g.sqr()? * (1.0 - ADAM_BETA2))?)?.detach();
            let mhat = (&self.m[i] / bc1)?;
            let vhat = (&self.v[i] / bc2)?;
            let upd = (mhat / (vhat.sqrt()? + ADAM_EPS)?)?;
            var.set(&(var.as_tensor() - (upd * lr)?)?)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct TrainReport {
    pub config: TrainConfig,
    pub samples: usize,
    pub train: usize,
    pub validation: usize,
    pub dim: usize,
    pub mse_train: f64,
    pub mse_validation: f64,
    pub validation_mean_baseline_mse: f64,
    pub dead_features: usize,
    pub mean_active: f64,
    pub seconds: f64,
    pub progress: Vec<(usize, f64, f64)>,
}

pub const EVAL_CHUNK_ROWS: usize = 8192;
pub const SYNC_EVERY_STEPS: usize = 4;

pub fn rows_to_device(host: &Tensor, start: usize, len: usize, dev: &Device) -> Result<Tensor> {
    let idx = Tensor::arange(start as u32, (start + len) as u32, host.device())?;
    Ok(host.index_select(&idx, 0)?.to_device(dev)?)
}

struct Eval {
    mse: f64,
    active_per_feature: Vec<f32>,
    active_total: f64,
}

fn eval_chunks(sae: &Sae, z_host: &Tensor, dev: &Device) -> Result<Eval> {
    let (n, d) = (z_host.dims()[0], z_host.dims()[1]);
    let mut sse = 0f64;
    let mut per_feature = vec![0f32; sae.width()];
    let mut active_total = 0f64;
    let mut start = 0;
    while start < n {
        let len = EVAL_CHUNK_ROWS.min(n - start);
        let z = rows_to_device(z_host, start, len, dev)?;
        let h = sae.encode(&z)?;
        let r = sae.decode(&h)?;
        sse += (r - &z)?
            .sqr()?
            .sum_all()?
            .to_dtype(DType::F64)?
            .to_scalar::<f64>()?;
        let active = h.gt(0f32)?.to_dtype(DType::F32)?;
        for (acc, c) in per_feature.iter_mut().zip(active.sum(0)?.to_vec1::<f32>()?) {
            *acc += c;
        }
        active_total += active.sum_all()?.to_dtype(DType::F64)?.to_scalar::<f64>()?;
        start += len;
    }
    Ok(Eval {
        mse: sse / (n * d) as f64,
        active_per_feature: per_feature,
        active_total,
    })
}

fn mse(a: &Tensor, b: &Tensor) -> Result<f64> {
    Ok((a - b)?
        .sqr()?
        .mean_all()?
        .to_dtype(DType::F64)?
        .to_scalar::<f64>()?)
}

pub fn train(
    x: &Tensor,
    is_train: &[bool],
    cfg: &TrainConfig,
    dev: &Device,
) -> Result<(Sae, TrainReport)> {
    let start = std::time::Instant::now();
    let (n, d) = (x.dims()[0], x.dims()[1]);
    ensure!(n == is_train.len(), "partition mask length");
    ensure!(
        cfg.top_k >= 1 && cfg.top_k <= cfg.width,
        "top_k must be within 1..=width"
    );
    let train_idx: Vec<u32> = (0..n as u32).filter(|&i| is_train[i as usize]).collect();
    let val_idx: Vec<u32> = (0..n as u32).filter(|&i| !is_train[i as usize]).collect();
    ensure!(
        train_idx.len() >= cfg.batch && !val_idx.is_empty(),
        "need a training batch and some validation rows"
    );
    let host = Device::Cpu;
    let x = x.to_device(&host)?;
    let xt = x.index_select(&Tensor::new(train_idx.as_slice(), &host)?, 0)?;
    let xv = x.index_select(&Tensor::new(val_idx.as_slice(), &host)?, 0)?;
    drop(x);
    let mu = xt.mean_keepdim(0)?;
    let sd = (xt.broadcast_sub(&mu)?.sqr()?.mean_keepdim(0)?.sqrt()? + NORMALIZATION_EPS as f64)?;
    let zt = xt.broadcast_sub(&mu)?.broadcast_div(&sd)?;
    let zv = xv.broadcast_sub(&mu)?.broadcast_div(&sd)?;
    drop((xt, xv));
    let mut rng = Rng::new(cfg.seed);
    let w = cfg.width;
    let init: Vec<f32> = (0..d * w)
        .map(|_| rng.normal() / (d as f32).sqrt())
        .collect();
    let enc = Tensor::from_vec(init, (d, w), dev)?;
    let dec0 = enc.t()?.contiguous()?;
    let dec0 = dec0.broadcast_div(&(dec0.sqr()?.sum_keepdim(1)?.sqrt()? + DECODER_RENORM_EPS)?)?;
    let sae = Sae {
        encoder: Var::from_tensor(&enc)?,
        encoder_bias: Var::zeros(w, DType::F32, dev)?,
        decoder: Var::from_tensor(&dec0)?,
        decoder_bias: Var::zeros(d, DType::F32, dev)?,
        mu: mu.squeeze(0)?.to_device(dev)?,
        sd: sd.squeeze(0)?.to_device(dev)?,
        top_k: cfg.top_k,
    };
    let vars = [
        &sae.encoder,
        &sae.encoder_bias,
        &sae.decoder,
        &sae.decoder_bias,
    ];
    let mut adam = Adam::new(&vars)?;
    let mut progress = Vec::new();
    let nt = zt.dims()[0];
    for step in 1..=cfg.steps {
        let idx: Vec<u32> = (0..cfg.batch).map(|_| rng.below(nt) as u32).collect();
        let z = zt
            .index_select(&Tensor::new(idx.as_slice(), &host)?, 0)?
            .to_device(dev)?;
        let h = sae.encode(&z)?;
        let r = sae.decode(&h)?;
        let mut loss = (&r - &z)?.sqr()?.mean_all()?;
        if cfg.l1 > 0.0 {
            loss = (loss + (h.abs()?.mean_all()? * cfg.l1)?)?;
        }
        let grads = loss.backward()?;
        adam.step(&vars, &grads, cfg.lr)?;
        let dec = sae.decoder.as_tensor();
        let renorm =
            dec.broadcast_div(&(dec.sqr()?.sum_keepdim(1)?.sqrt()? + DECODER_RENORM_EPS)?)?;
        sae.decoder.set(&renorm)?;
        if step % SYNC_EVERY_STEPS == 0 {
            dev.synchronize()?;
        }
        if step % 100 == 0 || step == cfg.steps {
            let val = eval_chunks(&sae, &zv, dev)?.mse;
            progress.push((step, loss.to_dtype(DType::F64)?.to_scalar::<f64>()?, val));
        }
    }
    let mean_base = mse(
        &zt.mean_keepdim(0)?.broadcast_as(zv.shape())?.contiguous()?,
        &zv,
    )?;
    let ev_t = eval_chunks(&sae, &zt, dev)?;
    let ev_v = eval_chunks(&sae, &zv, dev)?;
    let dead = ev_v
        .active_per_feature
        .iter()
        .filter(|&&c| c <= 0.0)
        .count();
    let mean_active = ev_v.active_total / val_idx.len() as f64;
    let report = TrainReport {
        config: cfg.clone(),
        samples: n,
        train: train_idx.len(),
        validation: val_idx.len(),
        dim: d,
        mse_train: ev_t.mse,
        mse_validation: ev_v.mse,
        validation_mean_baseline_mse: mean_base,
        dead_features: dead,
        mean_active,
        seconds: start.elapsed().as_secs_f64(),
        progress,
    };
    Ok((sae, report))
}

pub const STREAM_BUFFER_TOKENS: usize = 250_000;
pub const STREAM_STEPS_PER_REFILL: usize = 200;

pub struct StreamTrainer {
    pub sae: Sae,
    cfg: TrainConfig,
    dev: Device,
    buffer: Vec<f32>,
    dim: usize,
    fill: usize,
    write_pos: usize,
    rng: Rng,
    adam: Adam,
    pub steps_done: usize,
    pub losses: Vec<(usize, f64)>,
}

impl StreamTrainer {
    pub fn new(cfg: &TrainConfig, mu: &[f32], sd: &[f32], dev: &Device) -> Result<Self> {
        let d = mu.len();
        let mut rng = Rng::new(cfg.seed);
        let init: Vec<f32> = (0..d * cfg.width)
            .map(|_| rng.normal() / (d as f32).sqrt())
            .collect();
        let enc = Tensor::from_vec(init, (d, cfg.width), dev)?;
        let dec0 = enc.t()?.contiguous()?;
        let dec0 =
            dec0.broadcast_div(&(dec0.sqr()?.sum_keepdim(1)?.sqrt()? + DECODER_RENORM_EPS)?)?;
        let sae = Sae {
            encoder: Var::from_tensor(&enc)?,
            encoder_bias: Var::zeros(cfg.width, DType::F32, dev)?,
            decoder: Var::from_tensor(&dec0)?,
            decoder_bias: Var::zeros(d, DType::F32, dev)?,
            mu: Tensor::new(mu, dev)?,
            sd: Tensor::new(sd, dev)?,
            top_k: cfg.top_k,
        };
        let adam = Adam::new(&[
            &sae.encoder,
            &sae.encoder_bias,
            &sae.decoder,
            &sae.decoder_bias,
        ])?;
        Ok(StreamTrainer {
            sae,
            cfg: cfg.clone(),
            dev: dev.clone(),
            buffer: vec![0f32; STREAM_BUFFER_TOKENS * d],
            dim: d,
            fill: 0,
            write_pos: 0,
            rng,
            adam,
            steps_done: 0,
            losses: Vec::new(),
        })
    }

    pub fn push(&mut self, rows: &[Vec<f32>]) {
        for r in rows {
            let at = self.write_pos * self.dim;
            self.buffer[at..at + self.dim].copy_from_slice(r);
            self.write_pos = (self.write_pos + 1) % STREAM_BUFFER_TOKENS;
            self.fill = (self.fill + 1).min(STREAM_BUFFER_TOKENS);
        }
    }

    pub fn train_steps(&mut self, steps: usize) -> Result<()> {
        if self.fill < self.cfg.batch {
            return Ok(());
        }
        let vars = [
            &self.sae.encoder,
            &self.sae.encoder_bias,
            &self.sae.decoder,
            &self.sae.decoder_bias,
        ];
        for _ in 0..steps {
            let mut raw = Vec::with_capacity(self.cfg.batch * self.dim);
            for _ in 0..self.cfg.batch {
                let i = self.rng.below(self.fill) * self.dim;
                raw.extend_from_slice(&self.buffer[i..i + self.dim]);
            }
            let x = Tensor::from_vec(raw, (self.cfg.batch, self.dim), &self.dev)?;
            let z = self.sae.normalize(&x)?;
            let h = self.sae.encode(&z)?;
            let r = self.sae.decode(&h)?;
            let loss = (&r - &z)?.sqr()?.mean_all()?;
            let grads = loss.backward()?;
            self.adam.step(&vars, &grads, self.cfg.lr)?;
            let dec = self.sae.decoder.as_tensor();
            let renorm =
                dec.broadcast_div(&(dec.sqr()?.sum_keepdim(1)?.sqrt()? + DECODER_RENORM_EPS)?)?;
            self.sae.decoder.set(&renorm)?;
            self.steps_done += 1;
            if self.steps_done % 100 == 0 {
                self.losses.push((
                    self.steps_done,
                    loss.to_dtype(DType::F64)?.to_scalar::<f64>()?,
                ));
            }
        }
        Ok(())
    }
}

pub fn running_moments(rows: &[Vec<f32>], mean: &mut [f64], m2: &mut [f64], n: &mut f64) {
    for r in rows {
        *n += 1.0;
        for (j, v) in r.iter().enumerate() {
            let d = *v as f64 - mean[j];
            mean[j] += d / *n;
            m2[j] += d * (*v as f64 - mean[j]);
        }
    }
}
