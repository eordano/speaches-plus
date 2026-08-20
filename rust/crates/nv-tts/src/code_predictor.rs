use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::{
    attn::{sdpa, AttnConfig},
    RmsNorm, Rope, RopeConfig, RopeKind,
};
use nv_weights::WeightLoader;

use crate::dense::DenseLinear;
use crate::weight_helpers::{load_linear, load_rmsnorm};

pub const NUM_EXTRA_CODEBOOKS: usize = 15;

#[derive(Clone, Debug)]
pub struct CodecDecoderConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,

    pub codebook_vocab_size: usize,

    pub num_extra_codebooks: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,

    pub mrope_section: Vec<usize>,
    pub dtype: DType,
}

impl Default for CodecDecoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 5,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
            codebook_vocab_size: 2048,
            num_extra_codebooks: NUM_EXTRA_CODEBOOKS,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 65536,
            rms_norm_eps: 1e-6,
            mrope_section: vec![24, 20, 20],
            dtype: DType::BF16,
        }
    }
}

impl CodecDecoderConfig {
    pub fn from_hf_config_file(p: &Path) -> Result<Self> {
        let raw = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
        Self::from_hf_config_str(std::str::from_utf8(&raw)?)
    }

    pub fn from_hf_config_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse config json")?;

        let cp = v
            .get("talker_config")
            .and_then(|t| t.get("code_predictor_config"))
            .or_else(|| v.get("code_predictor_config"))
            .ok_or_else(|| anyhow!("config.json: missing code_predictor_config"))?;
        let mut cfg = Self::default();
        macro_rules! grab {
            ($field:ident, $key:expr) => {
                if let Some(v) = cp.get($key).and_then(|x| x.as_u64()) {
                    cfg.$field = v as usize;
                }
            };
        }
        grab!(hidden_size, "hidden_size");
        grab!(num_hidden_layers, "num_hidden_layers");
        grab!(num_attention_heads, "num_attention_heads");
        grab!(num_key_value_heads, "num_key_value_heads");
        grab!(head_dim, "head_dim");
        grab!(intermediate_size, "intermediate_size");
        grab!(codebook_vocab_size, "vocab_size");
        grab!(max_position_embeddings, "max_position_embeddings");
        if let Some(v) = cp.get("rope_theta").and_then(|x| x.as_f64()) {
            cfg.rope_theta = v as f32;
        }
        if let Some(v) = cp.get("rms_norm_eps").and_then(|x| x.as_f64()) {
            cfg.rms_norm_eps = v;
        }
        if let Some(n) = cp.get("num_code_groups").and_then(|x| x.as_u64()) {
            cfg.num_extra_codebooks = (n as usize).saturating_sub(1);
        }
        let half = cfg.head_dim / 2;
        if cfg.mrope_section.iter().sum::<usize>() != half {
            cfg.mrope_section = vec![half];
        }
        Ok(cfg)
    }
}

struct CodecLayer {
    input_norm: RmsNorm,
    q_proj: DenseLinear,
    k_proj: DenseLinear,
    v_proj: DenseLinear,
    o_proj: DenseLinear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    gate_proj: DenseLinear,
    up_proj: DenseLinear,
    down_proj: DenseLinear,
}

pub struct CpKvCache {
    layers: Vec<(Tensor, Tensor)>,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
}

impl CpKvCache {
    fn new(cfg: &CodecDecoderConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        let shape = (1usize, max_seq_len, cfg.num_key_value_heads, cfg.head_dim);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push((
                Tensor::zeros(shape, cfg.dtype, device)?,
                Tensor::zeros(shape, cfg.dtype, device)?,
            ));
        }
        Ok(Self {
            layers,
            current_len: 0,
            max_seq_len,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn write_at(
        &mut self,
        layer: usize,
        start: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        let t = k_new.dims()[1];
        let end = start + t;
        if end > self.max_seq_len {
            anyhow::bail!("CpKvCache.write_at: end {} > max {}", end, self.max_seq_len);
        }
        let (k_buf, v_buf) = &self.layers[layer];
        let ranges = [0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim];
        let k_updated = k_buf.slice_assign(&ranges, k_new)?;
        let v_updated = v_buf.slice_assign(&ranges, v_new)?;
        self.layers[layer] = (k_updated, v_updated);
        Ok(())
    }

    fn view(&self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let (k, v) = &self.layers[layer];
        Ok((k.narrow(1, 0, len)?, v.narrow(1, 0, len)?))
    }
}

impl CodecLayer {
    fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        rope: &Rope,
        cfg: &CodecDecoderConfig,
        cache: Option<(&mut CpKvCache, usize)>,
    ) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            anyhow::bail!("CodecLayer.forward: expected [B,T,H], got {:?}", dims);
        }
        let (b, t, _h) = (dims[0], dims[1], dims[2]);
        let h_q = cfg.num_attention_heads * cfg.head_dim;

        let normed = self.input_norm.forward(x)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;
        let q = q.reshape((b, t, cfg.num_attention_heads, cfg.head_dim))?;
        let k = k.reshape((b, t, cfg.num_key_value_heads, cfg.head_dim))?;
        let v = v.reshape((b, t, cfg.num_key_value_heads, cfg.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let sections = if cfg.mrope_section.iter().sum::<usize>() == cfg.head_dim / 2 {
            &cfg.mrope_section[..]
        } else {
            anyhow::bail!(
                "CodecLayer.forward: mrope_section sum != head_dim/2 (cfg={:?})",
                cfg.mrope_section
            )
        };

        let pos_axes: Vec<&Tensor> = vec![positions; sections.len()];
        let (q_rot, k_rot) = rope.apply_mrope(&q, &k, &pos_axes, sections)?;

        let (k_full, v_full) = if let Some((cache_ref, layer_idx)) = cache {
            let write_start = cache_ref.current_len;
            let new_total = write_start + t;
            cache_ref.write_at(
                layer_idx,
                write_start,
                &k_rot.contiguous()?,
                &v.contiguous()?,
            )?;
            cache_ref.view(layer_idx, new_total)?
        } else {
            (k_rot.contiguous()?, v.contiguous()?)
        };

        let attn_cfg = AttnConfig {
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            softmax_scale: 1.0 / (cfg.head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = sdpa(&q_rot, &k_full, &v_full, &attn_cfg)?;
        let attn_out = attn_out.reshape((b, t, h_q))?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        let x = x.add(&attn_out)?;

        let normed = self.post_attn_norm.forward(&x)?;
        let gate = self.gate_proj.forward(&normed)?;
        let up = self.up_proj.forward(&normed)?;
        let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = self.down_proj.forward(&act)?;
        Ok(x.add(&mlp_out)?)
    }
}

pub struct Qwen3TtsCodecDecoder {
    cfg: CodecDecoderConfig,
    device: Device,

    codec_embeddings: Vec<Tensor>,
    layers: Vec<CodecLayer>,
    final_norm: RmsNorm,

    lm_heads: Vec<DenseLinear>,
    rope: Rope,
}

impl Qwen3TtsCodecDecoder {
    pub fn config(&self) -> &CodecDecoderConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new(cfg: CodecDecoderConfig, device: &Device) -> Result<Self> {
        let dtype = cfg.dtype;
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let h_q = cfg.num_attention_heads * cfg.head_dim;
        let h_kv = cfg.num_key_value_heads * cfg.head_dim;

        let zeros_2d = |out: usize, inp: usize| -> Result<Tensor> {
            Ok(Tensor::zeros((out, inp), dtype, device)?)
        };
        let ones_1d = |n: usize| -> Result<Tensor> { Ok(Tensor::ones(n, dtype, device)?) };

        let mut codec_embeddings = Vec::with_capacity(cfg.num_extra_codebooks);
        for _ in 0..cfg.num_extra_codebooks {
            codec_embeddings.push(Tensor::zeros((cfg.codebook_vocab_size, h), dtype, device)?);
        }

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(CodecLayer {
                input_norm: RmsNorm::new(ones_1d(h)?, cfg.rms_norm_eps),
                q_proj: DenseLinear::new(zeros_2d(h_q, h)?, None)?,
                k_proj: DenseLinear::new(zeros_2d(h_kv, h)?, None)?,
                v_proj: DenseLinear::new(zeros_2d(h_kv, h)?, None)?,
                o_proj: DenseLinear::new(zeros_2d(h, h_q)?, None)?,
                q_norm: RmsNorm::new(ones_1d(cfg.head_dim)?, cfg.rms_norm_eps),
                k_norm: RmsNorm::new(ones_1d(cfg.head_dim)?, cfg.rms_norm_eps),
                post_attn_norm: RmsNorm::new(ones_1d(h)?, cfg.rms_norm_eps),
                gate_proj: DenseLinear::new(zeros_2d(inter, h)?, None)?,
                up_proj: DenseLinear::new(zeros_2d(inter, h)?, None)?,
                down_proj: DenseLinear::new(zeros_2d(h, inter)?, None)?,
            });
        }

        let final_norm = RmsNorm::new(ones_1d(h)?, cfg.rms_norm_eps);
        let mut lm_heads = Vec::with_capacity(cfg.num_extra_codebooks);
        for _ in 0..cfg.num_extra_codebooks {
            lm_heads.push(DenseLinear::new(
                zeros_2d(cfg.codebook_vocab_size, h)?,
                None,
            )?);
        }

        let rope = Rope::new(
            RopeConfig {
                head_dim: cfg.head_dim,
                max_seq_len: cfg.max_position_embeddings,
                base: cfg.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;

        let _ = h_kv;
        Ok(Self {
            cfg,
            device: device.clone(),
            codec_embeddings,
            layers,
            final_norm,
            lm_heads,
            rope,
        })
    }

    pub fn load_weights(&mut self, weights: &WeightLoader) -> Result<()> {
        let dtype = self.cfg.dtype;
        let prefix = "talker.code_predictor";

        for k in 0..self.cfg.num_extra_codebooks {
            let name = format!("{prefix}.model.codec_embedding.{k}.weight");
            let w = weights
                .get(&name, dtype)
                .with_context(|| format!("load {name}"))?;
            let d = w.dims();
            if d != [self.cfg.codebook_vocab_size, self.cfg.hidden_size] {
                anyhow::bail!(
                    "{name}: expected [{}, {}], got {:?}",
                    self.cfg.codebook_vocab_size,
                    self.cfg.hidden_size,
                    d
                );
            }
            self.codec_embeddings[k] = w;
        }

        let h = self.cfg.hidden_size;
        let h_q = self.cfg.num_attention_heads * self.cfg.head_dim;
        let h_kv = self.cfg.num_key_value_heads * self.cfg.head_dim;
        let inter = self.cfg.intermediate_size;
        let eps = self.cfg.rms_norm_eps;
        for i in 0..self.cfg.num_hidden_layers {
            let p = format!("{prefix}.model.layers.{i}");
            let l = &mut self.layers[i];
            l.input_norm = load_rmsnorm(
                weights,
                &format!("{p}.input_layernorm.weight"),
                h,
                eps,
                dtype,
            )?;
            l.post_attn_norm = load_rmsnorm(
                weights,
                &format!("{p}.post_attention_layernorm.weight"),
                h,
                eps,
                dtype,
            )?;
            l.q_proj = load_linear(
                weights,
                &format!("{p}.self_attn.q_proj.weight"),
                h_q,
                h,
                dtype,
            )?;
            l.k_proj = load_linear(
                weights,
                &format!("{p}.self_attn.k_proj.weight"),
                h_kv,
                h,
                dtype,
            )?;
            l.v_proj = load_linear(
                weights,
                &format!("{p}.self_attn.v_proj.weight"),
                h_kv,
                h,
                dtype,
            )?;
            l.o_proj = load_linear(
                weights,
                &format!("{p}.self_attn.o_proj.weight"),
                h,
                h_q,
                dtype,
            )?;
            l.q_norm = load_rmsnorm(
                weights,
                &format!("{p}.self_attn.q_norm.weight"),
                self.cfg.head_dim,
                eps,
                dtype,
            )?;
            l.k_norm = load_rmsnorm(
                weights,
                &format!("{p}.self_attn.k_norm.weight"),
                self.cfg.head_dim,
                eps,
                dtype,
            )?;
            l.gate_proj = load_linear(
                weights,
                &format!("{p}.mlp.gate_proj.weight"),
                inter,
                h,
                dtype,
            )?;
            l.up_proj = load_linear(weights, &format!("{p}.mlp.up_proj.weight"), inter, h, dtype)?;
            l.down_proj = load_linear(
                weights,
                &format!("{p}.mlp.down_proj.weight"),
                h,
                inter,
                dtype,
            )?;
        }

        self.final_norm = load_rmsnorm(
            weights,
            &format!("{prefix}.model.norm.weight"),
            h,
            eps,
            dtype,
        )?;

        for k in 0..self.cfg.num_extra_codebooks {
            let name = format!("{prefix}.lm_head.{k}.weight");
            self.lm_heads[k] = load_linear(weights, &name, self.cfg.codebook_vocab_size, h, dtype)?;
        }
        Ok(())
    }

    fn forward_last_hidden(&self, residual: &Tensor) -> Result<Tensor> {
        let dims = residual.dims().to_vec();
        if dims.len() != 3 || dims[2] != self.cfg.hidden_size {
            anyhow::bail!(
                "forward_last_hidden: expected [B,T,{}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let (b, t) = (dims[0], dims[1]);
        if t == 0 {
            anyhow::bail!("forward_last_hidden: T must be >= 1");
        }

        let pos_row: Vec<u32> = (0..t as u32).collect();
        let mut pos_tiled: Vec<u32> = Vec::with_capacity(b * t);
        for _ in 0..b {
            pos_tiled.extend_from_slice(&pos_row);
        }
        let positions = Tensor::from_vec(pos_tiled, (b, t), &self.device)?;
        let mut x = residual.clone();
        for layer in &self.layers {
            x = layer.forward(&x, &positions, &self.rope, &self.cfg, None)?;
        }
        let x = self.final_norm.forward(&x)?;
        Ok(x.narrow(1, t - 1, 1)?.squeeze(1)?)
    }

    fn forward_step_cached(&self, new_embeds: &Tensor, cache: &mut CpKvCache) -> Result<Tensor> {
        let dims = new_embeds.dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != self.cfg.hidden_size {
            anyhow::bail!(
                "forward_step_cached: expected [1,T,{}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let t = dims[1];
        let start = cache.current_len;
        let pos_row: Vec<u32> = (start as u32..(start + t) as u32).collect();
        let positions = Tensor::from_vec(pos_row, (1usize, t), &self.device)?;
        let mut x = new_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &positions, &self.rope, &self.cfg, Some((cache, i)))?;
        }
        cache.current_len += t;
        let x = self.final_norm.forward(&x)?;
        Ok(x.narrow(1, t - 1, 1)?.squeeze(1)?)
    }

    fn embed_extra(&self, k: usize, tok: u32) -> Result<Tensor> {
        if k >= self.codec_embeddings.len() {
            anyhow::bail!("embed_extra: codebook idx {k} out of range");
        }
        if (tok as usize) >= self.cfg.codebook_vocab_size {
            anyhow::bail!(
                "embed_extra: token {tok} >= codebook_vocab_size {}",
                self.cfg.codebook_vocab_size
            );
        }
        let id = Tensor::from_vec(vec![tok], 1usize, &self.device)?;
        let emb = self.codec_embeddings[k].index_select(&id, 0)?;
        Ok(emb.reshape((1usize, 1usize, self.cfg.hidden_size))?)
    }

    pub fn predict(
        &self,
        talker_hidden: &Tensor,
        _base_token: u32,
        _prev_codec_tokens: &[[u32; NUM_EXTRA_CODEBOOKS]],
    ) -> Result<[u32; NUM_EXTRA_CODEBOOKS]> {
        let dims = talker_hidden.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 || dims[2] != self.cfg.hidden_size {
            anyhow::bail!(
                "predict: talker_hidden must be [1, 1, {}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let dtype = self.cfg.dtype;
        let mut residual = if talker_hidden.dtype() == dtype {
            talker_hidden.clone()
        } else {
            talker_hidden.to_dtype(dtype)?
        };

        let mut out = [0u32; NUM_EXTRA_CODEBOOKS];
        for (k, o) in out
            .iter_mut()
            .enumerate()
            .take(self.cfg.num_extra_codebooks)
        {
            let h = self.forward_last_hidden(&residual)?;
            let logits = self.lm_heads[k].forward(&h)?;
            let tok = argmax_u32(&logits)?;
            *o = tok;
            if k + 1 < self.cfg.num_extra_codebooks {
                let emb = self.embed_extra(k, tok)?;
                residual = Tensor::cat(&[&residual, &emb], 1)?;
            }
        }
        Ok(out)
    }

    pub fn predict_sampled(
        &self,
        talker_hidden: &Tensor,
        base_embed: &Tensor,
        sampler: &mut crate::sampling::Sampler,
    ) -> Result<[u32; NUM_EXTRA_CODEBOOKS]> {
        let dims = talker_hidden.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 || dims[2] != self.cfg.hidden_size {
            anyhow::bail!(
                "predict_sampled: talker_hidden must be [1, 1, {}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let bd = base_embed.dims();
        if bd != [1, 1, self.cfg.hidden_size] {
            anyhow::bail!(
                "predict_sampled: base_embed must be [1, 1, {}], got {:?}",
                self.cfg.hidden_size,
                bd
            );
        }
        let dtype = self.cfg.dtype;
        let hid = talker_hidden.to_dtype(dtype)?;
        let base = base_embed.to_dtype(dtype)?;
        let prefill = Tensor::cat(&[&hid, &base], 1)?;
        let mut cache = CpKvCache::new(&self.cfg, 2 + self.cfg.num_extra_codebooks, &self.device)?;

        let mut out = [0u32; NUM_EXTRA_CODEBOOKS];
        let mut h = self.forward_step_cached(&prefill, &mut cache)?;
        for (k, o) in out
            .iter_mut()
            .enumerate()
            .take(self.cfg.num_extra_codebooks)
        {
            let logits = self.lm_heads[k].forward(&h)?;
            let row = logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let tok = sampler.sample(&row, &[], |_| true)?;
            *o = tok;
            if k + 1 < self.cfg.num_extra_codebooks {
                let emb = self.embed_extra(k, tok)?;
                h = self.forward_step_cached(&emb, &mut cache)?;
            }
        }
        Ok(out)
    }

    pub fn sum_extra_embeds(&self, extras: &[u32; NUM_EXTRA_CODEBOOKS]) -> Result<Tensor> {
        if self.cfg.num_extra_codebooks != NUM_EXTRA_CODEBOOKS {
            anyhow::bail!(
                "sum_extra_embeds: num_extra_codebooks {} != {}",
                self.cfg.num_extra_codebooks,
                NUM_EXTRA_CODEBOOKS
            );
        }
        let mut acc = self.embed_extra(0, extras[0])?;
        for (k, &tok) in extras.iter().enumerate().skip(1) {
            acc = acc.add(&self.embed_extra(k, tok)?)?;
        }
        Ok(acc)
    }
}

fn argmax_u32(logits: &Tensor) -> Result<u32> {
    let d = logits.dims();
    if d.len() != 2 {
        anyhow::bail!("argmax_u32: expected [B, V], got {d:?}");
    }
    if d[0] != 1 {
        anyhow::bail!("argmax_u32: B must be 1, got {}", d[0]);
    }
    let row = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let mut best = 0usize;
    let mut best_val = row[0];
    for (i, &v) in row.iter().enumerate().skip(1) {
        if v > best_val {
            best_val = v;
            best = i;
        }
    }
    Ok(best as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> CodecDecoderConfig {
        CodecDecoderConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            codebook_vocab_size: 16,
            num_extra_codebooks: 15,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            mrope_section: vec![2, 1, 1],
            dtype: DType::F32,
        }
    }

    #[test]
    fn builds_on_cpu_with_zero_weights() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let dec = Qwen3TtsCodecDecoder::new(cfg.clone(), &dev).expect("build");
        assert_eq!(dec.codec_embeddings.len(), cfg.num_extra_codebooks);
        assert_eq!(dec.lm_heads.len(), cfg.num_extra_codebooks);
        assert_eq!(dec.layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn predict_emits_15_tokens_in_range() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let dec = Qwen3TtsCodecDecoder::new(cfg.clone(), &dev).expect("build");
        let h = Tensor::ones((1usize, 1usize, cfg.hidden_size), DType::F32, &dev).unwrap();
        let toks = dec.predict(&h, 0, &[]).expect("predict");
        for (i, &t) in toks.iter().enumerate() {
            assert!(
                (t as usize) < cfg.codebook_vocab_size,
                "tok[{i}] = {t} out of vocab {}",
                cfg.codebook_vocab_size
            );
        }

        assert_eq!(toks, [0u32; 15]);
    }

    #[test]
    fn parses_real_code_predictor_config_when_cached() {
        let Some(dir) = crate::model_gate::require(
            "code_predictor::parses_real_code_predictor_config_when_cached",
        ) else {
            return;
        };
        let cfg_path = dir.join("config.json");
        let cfg = CodecDecoderConfig::from_hf_config_file(&cfg_path).expect("parse");
        assert_eq!(cfg.hidden_size, 1024);
        assert_eq!(cfg.num_hidden_layers, 5);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.intermediate_size, 3072);
        assert_eq!(cfg.codebook_vocab_size, 2048);
        assert_eq!(cfg.num_extra_codebooks, 15);
    }
}
