use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::{RmsNorm, Rope, RopeConfig, RopeKind};
use nv_weights::WeightLoader;

use crate::dense::{DenseLinear, LinearQuant};
use crate::talker::{DecoderLayer, LayerShape, Qwen3TtsKvCache};
use crate::weight_helpers::{load_linear, load_linear_with_bias};
use nv_layers::weight_load::load_rmsnorm;

pub const NUM_EXTRA_CODEBOOKS: usize = 15;

#[derive(Clone, Debug)]
pub struct CodecDecoderConfig {
    pub hidden_size: usize,

    pub talker_hidden_size: usize,
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
    pub quant: LinearQuant,
}

impl Default for CodecDecoderConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            talker_hidden_size: 1024,
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
            quant: LinearQuant::Dense,
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
        cfg.talker_hidden_size = v
            .get("talker_config")
            .and_then(|t| t.get("hidden_size"))
            .and_then(|x| x.as_u64())
            .map_or(cfg.hidden_size, |x| x as usize);
        let half = cfg.head_dim / 2;
        if cfg.mrope_section.iter().sum::<usize>() != half {
            cfg.mrope_section = vec![half];
        }
        Ok(cfg)
    }
}

pub struct Qwen3TtsCodecDecoder {
    cfg: CodecDecoderConfig,
    device: Device,

    codec_embeddings: Vec<Tensor>,
    small_to_mtp_projection: Option<DenseLinear>,
    layers: Vec<DecoderLayer>,
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
        let th = cfg.talker_hidden_size;

        let zeros_2d = |out: usize, inp: usize| -> Result<Tensor> {
            Ok(Tensor::zeros((out, inp), dtype, device)?)
        };
        let ones_1d = |n: usize| -> Result<Tensor> { Ok(Tensor::ones(n, dtype, device)?) };

        let mut codec_embeddings = Vec::with_capacity(cfg.num_extra_codebooks);
        for _ in 0..cfg.num_extra_codebooks {
            codec_embeddings.push(Tensor::zeros((cfg.codebook_vocab_size, th), dtype, device)?);
        }
        let small_to_mtp_projection = if th == h {
            None
        } else {
            Some(DenseLinear::new(
                zeros_2d(h, th)?,
                Some(Tensor::zeros(h, dtype, device)?),
            )?)
        };

        let shape = LayerShape {
            hidden: h,
            heads: cfg.num_attention_heads,
            kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            inter: cfg.intermediate_size,
            eps: cfg.rms_norm_eps,
        };
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| DecoderLayer::zeros(shape, dtype, device))
            .collect::<Result<Vec<_>>>()?;

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

        Ok(Self {
            cfg,
            device: device.clone(),
            codec_embeddings,
            small_to_mtp_projection,
            layers,
            final_norm,
            lm_heads,
            rope,
        })
    }

    pub fn load_weights(&mut self, weights: &WeightLoader) -> Result<()> {
        let dtype = self.cfg.dtype;
        let quant = self.cfg.quant.clone();
        let prefix = "talker.code_predictor";

        for k in 0..self.cfg.num_extra_codebooks {
            let name = format!("{prefix}.model.codec_embedding.{k}.weight");
            let w = weights
                .get(&name, dtype)
                .with_context(|| format!("load {name}"))?;
            let d = w.dims();
            if d != [self.cfg.codebook_vocab_size, self.cfg.talker_hidden_size] {
                anyhow::bail!(
                    "{name}: expected [{}, {}], got {:?}",
                    self.cfg.codebook_vocab_size,
                    self.cfg.talker_hidden_size,
                    d
                );
            }
            self.codec_embeddings[k] = w;
        }
        if self.small_to_mtp_projection.is_some() {
            self.small_to_mtp_projection = Some(load_linear_with_bias(
                weights,
                &format!("{prefix}.small_to_mtp_projection"),
                self.cfg.hidden_size,
                self.cfg.talker_hidden_size,
                dtype,
            )?);
        }

        let h = self.cfg.hidden_size;
        let eps = self.cfg.rms_norm_eps;
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.load(weights, &format!("{prefix}.model.layers.{i}"), dtype, &quant)?;
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
            self.lm_heads[k] = load_linear(
                weights,
                &name,
                self.cfg.codebook_vocab_size,
                h,
                dtype,
                &LinearQuant::Dense,
            )?;
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
            x = layer.forward(&x, &positions, &self.rope, &self.cfg.mrope_section, None)?;
        }
        let x = self.final_norm.forward(&x)?;
        Ok(x.narrow(1, t - 1, 1)?.squeeze(1)?)
    }

    fn forward_step_cached(
        &self,
        new_embeds: &Tensor,
        cache: &mut Qwen3TtsKvCache,
    ) -> Result<Tensor> {
        let dims = new_embeds.dims().to_vec();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != self.cfg.hidden_size {
            anyhow::bail!(
                "forward_step_cached: expected [1,T,{}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let t = dims[1];
        let start = cache.current_len();
        let pos_row: Vec<u32> = (start as u32..(start + t) as u32).collect();
        let positions = Tensor::from_vec(pos_row, (1usize, t), &self.device)?;
        let mut x = new_embeds.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(
                &x,
                &positions,
                &self.rope,
                &self.cfg.mrope_section,
                Some((cache, i)),
            )?;
        }
        cache.advance(t);
        let x = self.final_norm.forward(&x)?;
        Ok(x.narrow(1, t - 1, 1)?.squeeze(1)?)
    }

    fn to_mtp_width(&self, x: &Tensor) -> Result<Tensor> {
        match &self.small_to_mtp_projection {
            Some(p) => p.forward(x),
            None => Ok(x.clone()),
        }
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
        Ok(emb.reshape((1usize, 1usize, self.cfg.talker_hidden_size))?)
    }

    pub fn predict(
        &self,
        talker_hidden: &Tensor,
        _base_token: u32,
        _prev_codec_tokens: &[[u32; NUM_EXTRA_CODEBOOKS]],
    ) -> Result<[u32; NUM_EXTRA_CODEBOOKS]> {
        let dims = talker_hidden.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 || dims[2] != self.cfg.talker_hidden_size
        {
            anyhow::bail!(
                "predict: talker_hidden must be [1, 1, {}], got {:?}",
                self.cfg.talker_hidden_size,
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
            let h = self.forward_last_hidden(&self.to_mtp_width(&residual)?)?;
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
        let th = self.cfg.talker_hidden_size;
        if dims.len() != 3 || dims[0] != 1 || dims[1] != 1 || dims[2] != th {
            anyhow::bail!(
                "predict_sampled: talker_hidden must be [1, 1, {th}], got {:?}",
                dims
            );
        }
        let bd = base_embed.dims();
        if bd != [1, 1, th] {
            anyhow::bail!(
                "predict_sampled: base_embed must be [1, 1, {th}], got {:?}",
                bd
            );
        }
        let dtype = self.cfg.dtype;
        let hid = talker_hidden.to_dtype(dtype)?;
        let base = base_embed.to_dtype(dtype)?;
        let prefill = self.to_mtp_width(&Tensor::cat(&[&hid, &base], 1)?)?;
        let mut cache = Qwen3TtsKvCache::for_layers(
            self.cfg.num_hidden_layers,
            self.cfg.num_key_value_heads,
            self.cfg.head_dim,
            self.cfg.dtype,
            2 + self.cfg.num_extra_codebooks,
            &self.device,
        )?;

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
                let emb = self.to_mtp_width(&self.embed_extra(k, tok)?)?;
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
            talker_hidden_size: 32,
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
    fn a_zero_weight_decoder_builds_and_predicts_15_tokens_in_range() {
        let cfg = tiny_cfg();
        let dev = Device::Cpu;
        let dec = Qwen3TtsCodecDecoder::new(cfg.clone(), &dev).expect("build");
        assert_eq!(dec.codec_embeddings.len(), cfg.num_extra_codebooks);
        assert_eq!(dec.lm_heads.len(), cfg.num_extra_codebooks);
        assert_eq!(dec.layers.len(), cfg.num_hidden_layers);
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
    fn a_talker_wider_than_the_predictor_projects_into_it_and_sums_extras_at_talker_width() {
        let cfg = CodecDecoderConfig {
            talker_hidden_size: 48,
            ..tiny_cfg()
        };
        let dev = Device::Cpu;
        let dec = Qwen3TtsCodecDecoder::new(cfg.clone(), &dev).expect("build");
        assert!(dec.small_to_mtp_projection.is_some());
        let h = Tensor::ones((1usize, 1usize, 48), DType::F32, &dev).unwrap();
        assert_eq!(dec.predict(&h, 0, &[]).expect("predict"), [0u32; 15]);
        let mut sampler = crate::sampling::Sampler::new(crate::sampling::SamplerConfig {
            do_sample: false,
            ..Default::default()
        });
        let extras = dec.predict_sampled(&h, &h, &mut sampler).expect("sampled");
        assert!(extras
            .iter()
            .all(|&t| (t as usize) < cfg.codebook_vocab_size));
        assert_eq!(dec.sum_extra_embeds(&extras).unwrap().dims(), &[1, 1, 48]);
        let narrow = Tensor::ones((1usize, 1usize, 32), DType::F32, &dev).unwrap();
        assert!(
            dec.predict(&narrow, 0, &[]).is_err(),
            "the talker hidden state arrives at talker width, not predictor width"
        );
        let parsed = CodecDecoderConfig::from_hf_config_str(
            r#"{"talker_config":{"hidden_size":2048,"code_predictor_config":{"hidden_size":1024}}}"#,
        )
        .unwrap();
        assert_eq!(
            (parsed.hidden_size, parsed.talker_hidden_size),
            (1024, 2048)
        );
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
