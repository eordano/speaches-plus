use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use nv_layers::conv::Conv1d;
use nv_layers::linear::Linear;
use nv_weights::WeightLoader;

#[derive(Clone, Debug)]
pub struct LearnedVelFieldConfig {
    pub latent_dim: usize,
    pub d_model: usize,
    pub time_embed_dim: usize,
    pub n_blocks: usize,
    pub kernel: usize,
    pub dtype: DType,
}

impl Default for LearnedVelFieldConfig {
    fn default() -> Self {
        Self {
            latent_dim: 80,
            d_model: 256,
            time_embed_dim: 128,
            n_blocks: 4,
            kernel: 3,
            dtype: DType::F32,
        }
    }
}

struct ResBlock {
    conv1: Conv1d,
    conv2: Conv1d,
    film_t: Linear,
    film_c: Linear,
    d_model: usize,
}

pub struct LearnedVelField {
    cfg: LearnedVelFieldConfig,
    device: Device,
    time_lin1: Linear,
    time_lin2: Linear,
    in_proj: Conv1d,
    out_proj: Conv1d,
    blocks: Vec<ResBlock>,
}

impl LearnedVelField {
    pub fn new(cfg: LearnedVelFieldConfig, device: &Device) -> Result<Self> {
        if cfg.latent_dim == 0 {
            anyhow::bail!("LearnedVelFieldConfig: latent_dim must be > 0");
        }
        if cfg.d_model == 0 {
            anyhow::bail!("LearnedVelFieldConfig: d_model must be > 0");
        }
        if cfg.time_embed_dim == 0 {
            anyhow::bail!("LearnedVelFieldConfig: time_embed_dim must be > 0");
        }
        if !cfg.time_embed_dim.is_multiple_of(2) {
            anyhow::bail!(
                "LearnedVelFieldConfig: time_embed_dim must be even (got {})",
                cfg.time_embed_dim
            );
        }
        if cfg.kernel == 0 || cfg.kernel.is_multiple_of(2) {
            anyhow::bail!(
                "LearnedVelFieldConfig: kernel must be odd and >0 (got {})",
                cfg.kernel
            );
        }
        if cfg.n_blocks == 0 {
            anyhow::bail!("LearnedVelFieldConfig: n_blocks must be > 0");
        }

        let dt = cfg.dtype;
        let time_lin1 = Linear::new(
            Tensor::zeros((cfg.d_model, cfg.time_embed_dim), dt, device)?,
            Some(Tensor::zeros(cfg.d_model, dt, device)?),
        )?;
        let time_lin2 = Linear::new(
            Tensor::zeros((cfg.d_model, cfg.d_model), dt, device)?,
            Some(Tensor::zeros(cfg.d_model, dt, device)?),
        )?;

        let in_proj = Conv1d::new(
            Tensor::zeros((cfg.d_model, cfg.latent_dim, 1usize), dt, device)?,
            Some(Tensor::zeros(cfg.d_model, dt, device)?),
            1,
            0,
        )?;
        let out_proj = Conv1d::new(
            Tensor::zeros((cfg.latent_dim, cfg.d_model, 1usize), dt, device)?,
            Some(Tensor::zeros(cfg.latent_dim, dt, device)?),
            1,
            0,
        )?;

        let pad = (cfg.kernel - 1) / 2;
        let mut blocks = Vec::with_capacity(cfg.n_blocks);
        for _ in 0..cfg.n_blocks {
            let conv1 = Conv1d::new(
                Tensor::zeros((cfg.d_model, cfg.d_model, cfg.kernel), dt, device)?,
                Some(Tensor::zeros(cfg.d_model, dt, device)?),
                1,
                pad,
            )?;
            let conv2 = Conv1d::new(
                Tensor::zeros((cfg.d_model, cfg.d_model, cfg.kernel), dt, device)?,
                Some(Tensor::zeros(cfg.d_model, dt, device)?),
                1,
                pad,
            )?;
            let film_t = Linear::new(
                Tensor::zeros((2 * cfg.d_model, cfg.d_model), dt, device)?,
                Some(Tensor::zeros(2 * cfg.d_model, dt, device)?),
            )?;
            let film_c = Linear::new(
                Tensor::zeros((2 * cfg.d_model, cfg.d_model), dt, device)?,
                Some(Tensor::zeros(2 * cfg.d_model, dt, device)?),
            )?;
            blocks.push(ResBlock {
                conv1,
                conv2,
                film_t,
                film_c,
                d_model: cfg.d_model,
            });
        }

        Ok(Self {
            cfg,
            device: device.clone(),
            time_lin1,
            time_lin2,
            in_proj,
            out_proj,
            blocks,
        })
    }

    pub fn config(&self) -> &LearnedVelFieldConfig {
        &self.cfg
    }

    pub fn load_weights(&mut self, weights: &WeightLoader) -> Result<()> {
        self.load_weights_with_prefix(weights, "codec.vel")
    }

    pub fn load_weights_with_prefix(&mut self, weights: &WeightLoader, prefix: &str) -> Result<()> {
        let dt = self.cfg.dtype;
        let p = prefix.trim_end_matches('.');

        self.time_lin1 = load_linear(
            weights,
            &format!("{p}.time_embed.linear1"),
            self.cfg.d_model,
            self.cfg.time_embed_dim,
            dt,
        )?;
        self.time_lin2 = load_linear(
            weights,
            &format!("{p}.time_embed.linear2"),
            self.cfg.d_model,
            self.cfg.d_model,
            dt,
        )?;
        self.in_proj = load_conv1d(
            weights,
            &format!("{p}.in_proj"),
            self.cfg.d_model,
            self.cfg.latent_dim,
            1,
            0,
            dt,
        )?;
        self.out_proj = load_conv1d(
            weights,
            &format!("{p}.out_proj"),
            self.cfg.latent_dim,
            self.cfg.d_model,
            1,
            0,
            dt,
        )?;

        let pad = (self.cfg.kernel - 1) / 2;
        for i in 0..self.cfg.n_blocks {
            let blk_p = format!("{p}.blocks.{i}");
            let conv1 = load_conv1d(
                weights,
                &format!("{blk_p}.conv1"),
                self.cfg.d_model,
                self.cfg.d_model,
                self.cfg.kernel,
                pad,
                dt,
            )?;
            let conv2 = load_conv1d(
                weights,
                &format!("{blk_p}.conv2"),
                self.cfg.d_model,
                self.cfg.d_model,
                self.cfg.kernel,
                pad,
                dt,
            )?;
            let film_t = load_linear(
                weights,
                &format!("{blk_p}.film_t"),
                2 * self.cfg.d_model,
                self.cfg.d_model,
                dt,
            )?;
            let film_c = load_linear(
                weights,
                &format!("{blk_p}.film_c"),
                2 * self.cfg.d_model,
                self.cfg.d_model,
                dt,
            )?;
            self.blocks[i] = ResBlock {
                conv1,
                conv2,
                film_t,
                film_c,
                d_model: self.cfg.d_model,
            };
        }
        Ok(())
    }

    fn time_sinusoidal(&self, t: f32) -> Result<Tensor> {
        let half = self.cfg.time_embed_dim / 2;
        let mut vals = Vec::with_capacity(self.cfg.time_embed_dim);

        let log_max = 10_000f32.ln();
        for k in 0..half {
            let omega = (-log_max * (k as f32) / (half as f32)).exp();
            vals.push((t * omega).sin());
        }
        for k in 0..half {
            let omega = (-log_max * (k as f32) / (half as f32)).exp();
            vals.push((t * omega).cos());
        }
        let t_emb = Tensor::from_vec(vals, (1usize, self.cfg.time_embed_dim), &self.device)?
            .to_dtype(self.cfg.dtype)?;
        Ok(t_emb)
    }

    fn time_embed(&self, t: f32) -> Result<Tensor> {
        let raw = self.time_sinusoidal(t)?;
        let h = self.time_lin1.forward(&raw)?;
        let h = silu(&h)?;
        let h = self.time_lin2.forward(&h)?;
        Ok(h)
    }

    fn cond_pool(&self, cond_dm: &Tensor) -> Result<Tensor> {
        let pooled = cond_dm.mean(1)?;
        Ok(pooled)
    }
}

impl LearnedVelField {
    pub fn vel(&self, x: &Tensor, t: f32, cond: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 {
            anyhow::bail!(
                "LearnedVelField::vel: expected x [B, T, latent_dim], got {:?}",
                dims
            );
        }
        if dims[2] != self.cfg.latent_dim {
            anyhow::bail!(
                "LearnedVelField::vel: x last dim {} != latent_dim {}",
                dims[2],
                self.cfg.latent_dim
            );
        }
        let cdims = cond.dims();
        if cdims != dims {
            anyhow::bail!(
                "LearnedVelField::vel: cond dims {:?} != x dims {:?}",
                cdims,
                dims
            );
        }
        let b = dims[0];
        let tlen = dims[1];

        let t_emb = self.time_embed(t)?;
        let t_emb_btc = t_emb
            .reshape((1usize, 1usize, self.cfg.d_model))?
            .broadcast_as((b, 1usize, self.cfg.d_model))?;

        let cond_bct = cond.transpose(1, 2)?.contiguous()?;
        let cond_dm = self.in_proj.forward(&cond_bct)?;
        let cond_btc = cond_dm.transpose(1, 2)?.contiguous()?;
        let cond_pooled = self.cond_pool(&cond_btc)?;
        let cond_pooled_btc = cond_pooled.reshape((b, 1usize, self.cfg.d_model))?;

        let x_bct = x.transpose(1, 2)?.contiguous()?;
        let mut h = self.in_proj.forward(&x_bct)?;
        let _ = tlen;

        for block in &self.blocks {
            h = block.forward(&h, &t_emb_btc, &cond_pooled_btc)?;
        }

        let out_bct = self.out_proj.forward(&h)?;
        let out = out_bct.transpose(1, 2)?.contiguous()?;
        Ok(out)
    }
}

impl ResBlock {
    fn forward(&self, h: &Tensor, t_emb_btc: &Tensor, cond_btc: &Tensor) -> Result<Tensor> {
        let residual = h.clone();

        let mut y = self.conv1.forward(h)?;

        let m_t = self.film_t.forward(t_emb_btc)?;
        let m_c = self.film_c.forward(cond_btc)?;
        let (scale_t, shift_t) = split_2x(&m_t, self.d_model)?;
        let (scale_c, shift_c) = split_2x(&m_c, self.d_model)?;

        let scale_t = scale_t.transpose(1, 2)?.contiguous()?;
        let shift_t = shift_t.transpose(1, 2)?.contiguous()?;
        let scale_c = scale_c.transpose(1, 2)?.contiguous()?;
        let shift_c = shift_c.transpose(1, 2)?.contiguous()?;

        let one_plus = ((scale_t.broadcast_add(&scale_c)?) + 1.0f64)?;
        let shift = shift_t.broadcast_add(&shift_c)?;

        y = y.broadcast_mul(&one_plus)?;
        y = y.broadcast_add(&shift)?;
        y = silu(&y)?;

        y = self.conv2.forward(&y)?;
        Ok((y + residual)?)
    }
}

fn split_2x(x: &Tensor, d_model: usize) -> Result<(Tensor, Tensor)> {
    let scale = x.narrow(2, 0, d_model)?;
    let shift = x.narrow(2, d_model, d_model)?;
    Ok((scale, shift))
}

fn silu(x: &Tensor) -> Result<Tensor> {
    let sig = candle_nn::ops::sigmoid(x).map_err(|e| anyhow::anyhow!("sigmoid: {e}"))?;
    Ok((x * sig)?)
}

fn load_linear(
    weights: &WeightLoader,
    base: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let w_name = format!("{base}.weight");
    let b_name = format!("{base}.bias");
    let w = weights
        .get(&w_name, dtype)
        .with_context(|| format!("load weight {w_name}"))?;
    let wd = w.dims();
    if wd != [out_features, in_features] {
        anyhow::bail!(
            "load_linear({base}): expected [{out_features}, {in_features}], got {:?}",
            wd
        );
    }
    let bias = if weights.has(&b_name) {
        let b = weights
            .get(&b_name, dtype)
            .with_context(|| format!("load bias {b_name}"))?;
        let bd = b.dims();
        if bd != [out_features] {
            anyhow::bail!(
                "load_linear({base}.bias): expected [{out_features}], got {:?}",
                bd
            );
        }
        Some(b)
    } else {
        None
    };
    Linear::new(w, bias)
}

fn load_conv1d(
    weights: &WeightLoader,
    base: &str,
    out_channels: usize,
    in_channels: usize,
    kernel: usize,
    padding: usize,
    dtype: DType,
) -> Result<Conv1d> {
    let w_name = format!("{base}.weight");
    let b_name = format!("{base}.bias");
    let w = weights
        .get(&w_name, dtype)
        .with_context(|| format!("load weight {w_name}"))?;
    let wd = w.dims();
    if wd != [out_channels, in_channels, kernel] {
        anyhow::bail!(
            "load_conv1d({base}): expected [{out_channels}, {in_channels}, {kernel}], got {:?}",
            wd
        );
    }
    let bias = if weights.has(&b_name) {
        let b = weights
            .get(&b_name, dtype)
            .with_context(|| format!("load bias {b_name}"))?;
        let bd = b.dims();
        if bd != [out_channels] {
            anyhow::bail!(
                "load_conv1d({base}.bias): expected [{out_channels}], got {:?}",
                bd
            );
        }
        Some(b)
    } else {
        None
    };
    Conv1d::new(w, bias, 1, padding)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn constructs_and_forwards_with_zero_weights() {
        let cfg = LearnedVelFieldConfig {
            latent_dim: 8,
            d_model: 16,
            time_embed_dim: 8,
            n_blocks: 2,
            kernel: 3,
            dtype: DType::F32,
        };
        let dev = Device::Cpu;
        let net = LearnedVelField::new(cfg.clone(), &dev).expect("build net");

        let b = 1usize;
        let t = 5usize;
        let x = Tensor::ones((b, t, cfg.latent_dim), DType::F32, &dev).unwrap();
        let cond = Tensor::ones((b, t, cfg.latent_dim), DType::F32, &dev).unwrap();

        let v = net.vel(&x, 0.25, &cond).expect("vel");
        assert_eq!(v.dims(), &[b, t, cfg.latent_dim]);

        let total = v
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert_eq!(total, 0.0, "zero-init vel-net must produce a zero velocity");
    }

    #[test]
    fn vel_field_rejects_rank_or_dim_mismatch() {
        let cfg = LearnedVelFieldConfig {
            latent_dim: 4,
            d_model: 8,
            time_embed_dim: 4,
            n_blocks: 1,
            kernel: 3,
            dtype: DType::F32,
        };
        let dev = Device::Cpu;
        let net = LearnedVelField::new(cfg.clone(), &dev).unwrap();

        let bad_rank = Tensor::zeros((1usize, cfg.latent_dim), DType::F32, &dev).unwrap();
        let cond = Tensor::zeros((1usize, 1usize, cfg.latent_dim), DType::F32, &dev).unwrap();
        assert!(net.vel(&bad_rank, 0.0, &cond).is_err());

        let bad_last =
            Tensor::zeros((1usize, 2usize, cfg.latent_dim + 1), DType::F32, &dev).unwrap();
        let cond_good = Tensor::zeros((1usize, 2usize, cfg.latent_dim), DType::F32, &dev).unwrap();
        assert!(net.vel(&bad_last, 0.0, &cond_good).is_err());

        let x = Tensor::zeros((1usize, 2usize, cfg.latent_dim), DType::F32, &dev).unwrap();
        let cond_bad = Tensor::zeros((1usize, 3usize, cfg.latent_dim), DType::F32, &dev).unwrap();
        assert!(net.vel(&x, 0.0, &cond_bad).is_err());
    }

    #[test]
    fn config_rejects_invalid_values() {
        let dev = Device::Cpu;
        let mut cfg = LearnedVelFieldConfig {
            kernel: 4,
            ..Default::default()
        };
        assert!(LearnedVelField::new(cfg.clone(), &dev).is_err());

        cfg.kernel = 3;
        cfg.time_embed_dim = 7;
        assert!(LearnedVelField::new(cfg.clone(), &dev).is_err());

        cfg.time_embed_dim = 8;
        cfg.n_blocks = 0;
        assert!(LearnedVelField::new(cfg.clone(), &dev).is_err());
    }
}
