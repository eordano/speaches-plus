use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_layers::{
    attn::{sdpa, AttnConfig},
    RmsNorm, Rope, RopeConfig, RopeKind,
};
use nv_weights::WeightLoader;

use crate::dense::{DenseLinear, LinearQuant};
use crate::weight_helpers::{load_linear, load_linear_with_bias};
use nv_layers::weight_load::load_rmsnorm;

pub const CODEC_PAD_ID: u32 = 2148;

pub const CODEC_BOS_ID: u32 = 2149;

pub const CODEC_EOS_ID: u32 = 2150;

pub const CODEC_THINK_ID: u32 = 2154;

pub const CODEC_NOTHINK_ID: u32 = 2155;
pub const CODEC_THINK_BOS_ID: u32 = 2156;
pub const CODEC_THINK_EOS_ID: u32 = 2157;

#[derive(Clone, Debug)]
pub struct Qwen3TtsTalkerConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,

    pub speech_vocab_size: usize,

    pub text_vocab_size: usize,

    pub text_hidden_size: usize,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,

    pub mrope_section: Vec<usize>,

    pub dtype: DType,

    pub spk_id: Vec<(String, u32)>,

    pub language_id: Vec<(String, u32)>,

    pub quant: LinearQuant,
}

impl Default for Qwen3TtsTalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            intermediate_size: 3072,
            speech_vocab_size: 3072,
            text_vocab_size: 151936,
            text_hidden_size: 2048,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            mrope_section: vec![24, 20, 20],
            dtype: DType::BF16,
            spk_id: Vec::new(),
            language_id: Vec::new(),
            quant: LinearQuant::Dense,
        }
    }
}

impl Qwen3TtsTalkerConfig {
    pub fn from_hf_config_file(p: &Path) -> Result<Self> {
        let raw = std::fs::read(p).with_context(|| format!("read {}", p.display()))?;
        Self::from_hf_config_str(std::str::from_utf8(&raw)?)
    }

    pub fn from_hf_config_str(s: &str) -> Result<Self> {
        let v: serde_json::Value = serde_json::from_str(s).context("parse talker config json")?;
        let tc = v
            .get("talker_config")
            .ok_or_else(|| anyhow!("config.json: missing talker_config"))?;
        let mut cfg = Self::default();
        macro_rules! grab {
            ($field:ident, $key:expr) => {
                if let Some(v) = tc.get($key).and_then(|x| x.as_u64()) {
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
        grab!(speech_vocab_size, "vocab_size");
        grab!(text_vocab_size, "text_vocab_size");
        grab!(text_hidden_size, "text_hidden_size");
        grab!(max_position_embeddings, "max_position_embeddings");
        if let Some(v) = tc.get("rope_theta").and_then(|x| x.as_f64()) {
            cfg.rope_theta = v as f32;
        }
        if let Some(v) = tc.get("rms_norm_eps").and_then(|x| x.as_f64()) {
            cfg.rms_norm_eps = v;
        }
        if let Some(arr) = tc
            .get("rope_scaling")
            .and_then(|s| s.get("mrope_section"))
            .and_then(|x| x.as_array())
        {
            let parsed: Vec<usize> = arr
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as usize))
                .collect();
            if !parsed.is_empty() {
                cfg.mrope_section = parsed;
            }
        }

        for (field, key) in [("spk_id", "spk_id"), ("language_id", "codec_language_id")] {
            if let Some(map) = tc.get(key).and_then(|x| x.as_object()) {
                let mut entries: Vec<(String, u32)> = map
                    .iter()
                    .filter_map(|(k, v)| v.as_u64().map(|id| (k.to_lowercase(), id as u32)))
                    .collect();
                entries.sort();
                match field {
                    "spk_id" => cfg.spk_id = entries,
                    _ => cfg.language_id = entries,
                }
            }
        }

        let sum: usize = cfg.mrope_section.iter().sum();
        if sum != cfg.head_dim / 2 {
            anyhow::bail!(
                "talker_config: mrope_section sum {sum} != head_dim/2 = {}",
                cfg.head_dim / 2
            );
        }
        Ok(cfg)
    }
}

struct TextProjection {
    fc1: DenseLinear,
    fc2: DenseLinear,
}

impl TextProjection {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.fc1.forward(x)?;
        let h = candle_nn::ops::silu(&h)?;
        self.fc2.forward(&h)
    }
}

#[derive(Clone, Copy)]
pub(crate) struct LayerShape {
    pub(crate) hidden: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) inter: usize,
    pub(crate) eps: f64,
}

pub(crate) struct DecoderLayer {
    shape: LayerShape,
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

impl DecoderLayer {
    pub(crate) fn zeros(shape: LayerShape, dtype: DType, device: &Device) -> Result<Self> {
        let s = shape;
        let (h, h_q, h_kv) = (s.hidden, s.heads * s.head_dim, s.kv_heads * s.head_dim);
        let lin = |out: usize, inp: usize| {
            DenseLinear::new(Tensor::zeros((out, inp), dtype, device)?, None)
        };
        let norm = |n: usize| -> Result<RmsNorm> {
            Ok(RmsNorm::new(Tensor::ones(n, dtype, device)?, s.eps))
        };
        Ok(Self {
            shape,
            input_norm: norm(h)?,
            q_proj: lin(h_q, h)?,
            k_proj: lin(h_kv, h)?,
            v_proj: lin(h_kv, h)?,
            o_proj: lin(h, h_q)?,
            q_norm: norm(s.head_dim)?,
            k_norm: norm(s.head_dim)?,
            post_attn_norm: norm(h)?,
            gate_proj: lin(s.inter, h)?,
            up_proj: lin(s.inter, h)?,
            down_proj: lin(h, s.inter)?,
        })
    }

    pub(crate) fn load(
        &mut self,
        weights: &WeightLoader,
        p: &str,
        dtype: DType,
        quant: &LinearQuant,
    ) -> Result<()> {
        let s = self.shape;
        let (h, h_q, h_kv) = (s.hidden, s.heads * s.head_dim, s.kv_heads * s.head_dim);
        let lin = |name: &str, out: usize, inp: usize| {
            load_linear(weights, &format!("{p}.{name}.weight"), out, inp, dtype, quant)
        };
        let norm = |name: &str, n: usize| {
            load_rmsnorm(weights, &format!("{p}.{name}.weight"), n, s.eps, dtype)
        };
        self.input_norm = norm("input_layernorm", h)?;
        self.post_attn_norm = norm("post_attention_layernorm", h)?;
        self.q_proj = lin("self_attn.q_proj", h_q, h)?;
        self.k_proj = lin("self_attn.k_proj", h_kv, h)?;
        self.v_proj = lin("self_attn.v_proj", h_kv, h)?;
        self.o_proj = lin("self_attn.o_proj", h, h_q)?;
        self.q_norm = norm("self_attn.q_norm", s.head_dim)?;
        self.k_norm = norm("self_attn.k_norm", s.head_dim)?;
        self.gate_proj = lin("mlp.gate_proj", s.inter, h)?;
        self.up_proj = lin("mlp.up_proj", s.inter, h)?;
        self.down_proj = lin("mlp.down_proj", h, s.inter)?;
        Ok(())
    }

    pub(crate) fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        rope: &Rope,
        mrope_section: &[usize],
        cache: Option<(&mut Qwen3TtsKvCache, usize)>,
    ) -> Result<Tensor> {
        let s = self.shape;
        let dims = x.dims().to_vec();
        if dims.len() != 3 {
            anyhow::bail!("DecoderLayer.forward: expected [B,T,H], got {:?}", dims);
        }
        let (b, t) = (dims[0], dims[1]);

        let normed = self.input_norm.forward(x)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;
        let q = q.reshape((b, t, s.heads, s.head_dim))?;
        let k = k.reshape((b, t, s.kv_heads, s.head_dim))?;
        let v = v.reshape((b, t, s.kv_heads, s.head_dim))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        if mrope_section.iter().sum::<usize>() != s.head_dim / 2 {
            anyhow::bail!(
                "DecoderLayer.forward: mrope_section sum != head_dim/2 ({:?})",
                mrope_section
            );
        }
        let pos_axes: Vec<&Tensor> = vec![positions; mrope_section.len()];
        let (q_rot, k_rot) = rope.apply_mrope(&q, &k, &pos_axes, mrope_section)?;

        let (k_full, v_full) = if let Some((cache_ref, layer_idx)) = cache {
            let write_start = cache_ref.current_len();
            cache_ref.write_at(
                layer_idx,
                write_start,
                &k_rot.contiguous()?,
                &v.contiguous()?,
            )?;
            cache_ref.view(layer_idx, write_start + t)?
        } else {
            (k_rot.contiguous()?, v.contiguous()?)
        };

        let attn_cfg = AttnConfig {
            num_heads: s.heads,
            num_kv_heads: s.kv_heads,
            head_dim: s.head_dim,
            softmax_scale: 1.0 / (s.head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = sdpa(&q_rot, &k_full, &v_full, &attn_cfg)?;
        let attn_out = attn_out.reshape((b, t, s.heads * s.head_dim))?;
        let x = x.add(&self.o_proj.forward(&attn_out)?)?;

        let normed = self.post_attn_norm.forward(&x)?;
        let gate = self.gate_proj.forward(&normed)?;
        let up = self.up_proj.forward(&normed)?;
        let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
        Ok(x.add(&self.down_proj.forward(&act)?)?)
    }
}

pub struct Qwen3TtsKvCache {
    layers: Vec<(Tensor, Tensor)>,
    current_len: usize,
    max_seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: Device,
    dtype: DType,
}

impl Qwen3TtsKvCache {
    pub fn new(cfg: &Qwen3TtsTalkerConfig, max_seq_len: usize, device: &Device) -> Result<Self> {
        Self::for_layers(
            cfg.num_hidden_layers,
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.dtype,
            max_seq_len,
            device,
        )
    }

    pub(crate) fn for_layers(
        num_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        dtype: DType,
        max_seq_len: usize,
        device: &Device,
    ) -> Result<Self> {
        let shape = (1usize, max_seq_len, n_kv_heads, head_dim);
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k = Tensor::zeros(shape, dtype, device)?;
            let v = Tensor::zeros(shape, dtype, device)?;
            layers.push((k, v));
        }
        Ok(Self {
            layers,
            current_len: 0,
            max_seq_len,
            n_kv_heads,
            head_dim,
            device: device.clone(),
            dtype,
        })
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn reset(&mut self) {
        self.current_len = 0;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    fn write_at(
        &mut self,
        layer: usize,
        start: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<()> {
        let dims = k_new.dims();
        if dims.len() != 4 || dims[0] != 1 || dims[2] != self.n_kv_heads || dims[3] != self.head_dim
        {
            anyhow::bail!(
                "Qwen3TtsKvCache.write_at: expected [1, t, {}, {}], got {:?}",
                self.n_kv_heads,
                self.head_dim,
                dims
            );
        }
        if v_new.dims() != dims {
            anyhow::bail!(
                "Qwen3TtsKvCache.write_at: k/v shape mismatch k={:?} v={:?}",
                dims,
                v_new.dims()
            );
        }
        let t = dims[1];
        let end = start + t;
        if end > self.max_seq_len {
            anyhow::bail!(
                "Qwen3TtsKvCache.write_at: end {} exceeds max_seq_len {}",
                end,
                self.max_seq_len
            );
        }
        if layer >= self.layers.len() {
            anyhow::bail!("Qwen3TtsKvCache.write_at: layer {} out of range", layer);
        }
        let (k_buf, v_buf) = &self.layers[layer];
        let k_updated = k_buf.slice_assign(
            &[0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim],
            k_new,
        )?;
        let v_updated = v_buf.slice_assign(
            &[0..1, start..end, 0..self.n_kv_heads, 0..self.head_dim],
            v_new,
        )?;
        self.layers[layer] = (k_updated, v_updated);
        Ok(())
    }

    fn view(&self, layer: usize, len: usize) -> Result<(Tensor, Tensor)> {
        if layer >= self.layers.len() {
            anyhow::bail!("Qwen3TtsKvCache.view: layer {} out of range", layer);
        }
        if len > self.max_seq_len {
            anyhow::bail!(
                "Qwen3TtsKvCache.view: len {} > max {}",
                len,
                self.max_seq_len
            );
        }
        let (k, v) = &self.layers[layer];
        let k = k.narrow(1, 0, len)?;
        let v = v.narrow(1, 0, len)?;
        Ok((k, v))
    }

    pub(crate) fn advance(&mut self, n: usize) {
        self.current_len += n;
    }
}

pub struct Qwen3TtsTalker {
    cfg: Qwen3TtsTalkerConfig,
    device: Device,
    text_embedding: Option<Tensor>,
    text_projection: TextProjection,
    codec_embedding: Tensor,
    layers: Vec<DecoderLayer>,
    final_norm: RmsNorm,
    codec_head: DenseLinear,
    rope: Rope,
}

impl Qwen3TtsTalker {
    pub fn config(&self) -> &Qwen3TtsTalkerConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn new(cfg: Qwen3TtsTalkerConfig, device: &Device) -> Result<Self> {
        let dtype = cfg.dtype;
        let h = cfg.hidden_size;
        let th = cfg.text_hidden_size;

        let zeros_2d = |out: usize, inp: usize| -> Result<Tensor> {
            Ok(Tensor::zeros((out, inp), dtype, device)?)
        };
        let zeros_1d = |n: usize| -> Result<Tensor> { Ok(Tensor::zeros(n, dtype, device)?) };

        let ones_1d = |n: usize| -> Result<Tensor> { Ok(Tensor::ones(n, dtype, device)?) };

        let text_projection = TextProjection {
            fc1: DenseLinear::new(zeros_2d(th, th)?, Some(zeros_1d(th)?))?,
            fc2: DenseLinear::new(zeros_2d(h, th)?, Some(zeros_1d(h)?))?,
        };

        let codec_embedding = Tensor::zeros((cfg.speech_vocab_size, h), dtype, device)?;

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
        let codec_head = DenseLinear::new(zeros_2d(cfg.speech_vocab_size, h)?, None)?;

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
            text_embedding: None,
            text_projection,
            codec_embedding,
            layers,
            final_norm,
            codec_head,
            rope,
        })
    }

    pub fn load_weights(&mut self, weights: &WeightLoader) -> Result<()> {
        let dtype = self.cfg.dtype;
        let quant = self.cfg.quant.clone();
        let prefix = "talker";

        self.text_projection.fc1 = load_linear_with_bias(
            weights,
            &format!("{prefix}.text_projection.linear_fc1"),
            self.cfg.text_hidden_size,
            self.cfg.text_hidden_size,
            dtype,
        )?;
        self.text_projection.fc2 = load_linear_with_bias(
            weights,
            &format!("{prefix}.text_projection.linear_fc2"),
            self.cfg.hidden_size,
            self.cfg.text_hidden_size,
            dtype,
        )?;

        let ce_name = format!("{prefix}.model.codec_embedding.weight");
        self.codec_embedding = weights
            .get(&ce_name, dtype)
            .with_context(|| format!("load {ce_name}"))?;
        let cd = self.codec_embedding.dims();
        if cd != [self.cfg.speech_vocab_size, self.cfg.hidden_size] {
            anyhow::bail!(
                "codec_embedding: expected [{}, {}], got {:?}",
                self.cfg.speech_vocab_size,
                self.cfg.hidden_size,
                cd
            );
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
        self.codec_head = load_linear(
            weights,
            &format!("{prefix}.codec_head.weight"),
            self.cfg.speech_vocab_size,
            h,
            dtype,
            &LinearQuant::Dense,
        )?;

        let te_name = format!("{prefix}.model.text_embedding.weight");
        if let Ok(te) = weights.get(&te_name, dtype) {
            let td = te.dims();
            if td.len() == 2
                && td[0] == self.cfg.text_vocab_size
                && td[1] == self.cfg.text_hidden_size
            {
                self.text_embedding = Some(te);
            }
        }
        Ok(())
    }

    fn embed_codec(&self, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            return Ok(Tensor::zeros(
                (1, 0, self.cfg.hidden_size),
                self.cfg.dtype,
                &self.device,
            )?);
        }
        let id_tensor = Tensor::from_vec(ids.to_vec(), ids.len(), &self.device)?;
        let emb = self.codec_embedding.index_select(&id_tensor, 0)?;
        Ok(emb.reshape((1usize, ids.len(), self.cfg.hidden_size))?)
    }

    pub fn project_text(&self, text_hidden: &Tensor) -> Result<Tensor> {
        let dims = text_hidden.dims();
        let last = *dims.last().unwrap_or(&0);
        if last != self.cfg.text_hidden_size {
            anyhow::bail!(
                "project_text: expected last dim {}, got {:?}",
                self.cfg.text_hidden_size,
                dims
            );
        }
        let dtype = self.cfg.dtype;
        let x = if text_hidden.dtype() != dtype {
            text_hidden.to_dtype(dtype)?
        } else {
            text_hidden.clone()
        };
        self.text_projection.forward(&x)
    }

    fn forward_last_logits(&self, residual: &Tensor) -> Result<Tensor> {
        let (logits, _hidden) = self.forward_last_logits_and_hidden(residual)?;
        Ok(logits)
    }

    fn forward_last_logits_and_hidden(&self, residual: &Tensor) -> Result<(Tensor, Tensor)> {
        let dims = residual.dims().to_vec();
        if dims.len() != 3 || dims[2] != self.cfg.hidden_size {
            anyhow::bail!(
                "forward_last_logits: expected [B,T,{}], got {:?}",
                self.cfg.hidden_size,
                dims
            );
        }
        let (b, t) = (dims[0], dims[1]);
        if t == 0 {
            anyhow::bail!("forward_last_logits: T must be >= 1");
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

        let last = x.narrow(1, t - 1, 1)?.squeeze(1)?;
        let logits = self.codec_head.forward(&last)?;
        Ok((logits, last))
    }

    fn build_residual(&self, text_hidden: &Tensor, prev_speech: &[u32]) -> Result<Tensor> {
        self.build_residual_with_speaker(text_hidden, None, prev_speech)
    }

    pub fn build_residual_with_speaker(
        &self,
        text_hidden: &Tensor,
        speaker_prefix: Option<&Tensor>,
        prev_speech: &[u32],
    ) -> Result<Tensor> {
        let proj = self.project_text(text_hidden)?;
        let proj_dims = proj.dims().to_vec();
        let t_text = match proj_dims.len() {
            2 => proj_dims[0],
            3 => proj_dims[1],
            _ => anyhow::bail!("text_hidden must be rank 2 or 3, got {:?}", proj_dims),
        };
        let proj = proj.reshape((1usize, t_text, self.cfg.hidden_size))?;

        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        if let Some(sp) = speaker_prefix {
            let sp_dims = sp.dims().to_vec();
            if sp_dims.len() != 3 || sp_dims[0] != 1 || sp_dims[2] != self.cfg.hidden_size {
                anyhow::bail!(
                    "build_residual_with_speaker: speaker_prefix must be [1, T, {}], got {:?}",
                    self.cfg.hidden_size,
                    sp_dims
                );
            }
            let sp = if sp.dtype() != self.cfg.dtype {
                sp.to_dtype(self.cfg.dtype)?
            } else {
                sp.clone()
            };
            parts.push(sp);
        }
        parts.push(proj);
        if !prev_speech.is_empty() {
            parts.push(self.embed_codec(prev_speech)?);
        }
        if parts.len() == 1 {
            return Ok(parts.into_iter().next().unwrap());
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    }

    pub fn step_with_speaker(
        &self,
        text_hidden: &Tensor,
        speaker_prefix: Option<&Tensor>,
        prev_speech: &[u32],
    ) -> Result<u32> {
        let residual =
            self.build_residual_with_speaker(text_hidden, speaker_prefix, prev_speech)?;
        let logits = self.forward_last_logits(&residual)?;
        argmax_u32(&logits)
    }

    pub fn step(&self, text_hidden: &Tensor, prev_speech: &[u32]) -> Result<u32> {
        let residual = self.build_residual(text_hidden, prev_speech)?;
        let logits = self.forward_last_logits(&residual)?;
        argmax_u32(&logits)
    }

    pub fn step_full_frame_with_speaker(
        &self,
        text_hidden: &Tensor,
        speaker_prefix: Option<&Tensor>,
        prev_speech: &[u32],
        code_predictor: &crate::code_predictor::Qwen3TtsCodecDecoder,
        prev_extras: &[[u32; crate::code_predictor::NUM_EXTRA_CODEBOOKS]],
    ) -> Result<(u32, [u32; crate::code_predictor::NUM_EXTRA_CODEBOOKS])> {
        let residual =
            self.build_residual_with_speaker(text_hidden, speaker_prefix, prev_speech)?;
        let (logits, hidden) = self.forward_last_logits_and_hidden(&residual)?;
        let base = argmax_u32(&logits)?;
        let h = hidden.reshape((1usize, 1usize, self.cfg.hidden_size))?;
        let extras = code_predictor.predict(&h, base, prev_extras)?;
        Ok((base, extras))
    }

    pub fn new_kv_cache(&self, max_seq_len: usize) -> Result<Qwen3TtsKvCache> {
        Qwen3TtsKvCache::new(&self.cfg, max_seq_len, &self.device)
    }

    pub fn step_cached(
        &self,
        text_hidden: &Tensor,
        new_codec_token: Option<u32>,
        cache: &mut Qwen3TtsKvCache,
    ) -> Result<u32> {
        let (tok, _hidden) = self.step_cached_with_hidden(text_hidden, new_codec_token, cache)?;
        Ok(tok)
    }

    pub fn step_cached_with_hidden(
        &self,
        text_hidden: &Tensor,
        new_codec_token: Option<u32>,
        cache: &mut Qwen3TtsKvCache,
    ) -> Result<(u32, Tensor)> {
        let residual = if cache.current_len() == 0 {
            let proj = self.project_text(text_hidden)?;
            let proj_dims = proj.dims().to_vec();
            let t_text = match proj_dims.len() {
                2 => proj_dims[0],
                3 => proj_dims[1],
                _ => anyhow::bail!("text_hidden must be rank 2 or 3, got {:?}", proj_dims),
            };
            let proj = proj.reshape((1usize, t_text, self.cfg.hidden_size))?;
            match new_codec_token {
                None => proj,
                Some(tok) => {
                    let codec = self.embed_codec(&[tok])?;
                    Tensor::cat(&[&proj, &codec], 1)?
                }
            }
        } else {
            let tok = new_codec_token.ok_or_else(|| {
                anyhow!("step_cached: incremental call requires a new codec token id")
            })?;
            self.embed_codec(&[tok])?
        };

        let (logits, last) = self.forward_cached_embeds(&residual, cache)?;
        let tok = argmax_u32(&logits)?;
        Ok((tok, last))
    }

    pub fn forward_cached_embeds(
        &self,
        residual: &Tensor,
        cache: &mut Qwen3TtsKvCache,
    ) -> Result<(Tensor, Tensor)> {
        let t = residual.dims()[1];
        if t == 0 {
            anyhow::bail!("forward_cached_embeds: empty residual");
        }
        let write_start = cache.current_len();
        let new_total = write_start + t;
        if new_total > cache.max_seq_len() {
            anyhow::bail!(
                "forward_cached_embeds: new_total {} > cache.max_seq_len {}",
                new_total,
                cache.max_seq_len()
            );
        }

        let pos_row: Vec<u32> = (write_start as u32..new_total as u32).collect();
        let positions = Tensor::from_vec(pos_row, (1usize, t), &self.device)?;

        let mut x = residual.clone();
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
        let last = x.narrow(1, t - 1, 1)?.squeeze(1)?;
        let logits = self.codec_head.forward(&last)?;
        Ok((logits, last))
    }

    pub fn generate(
        &self,
        text_hidden: &Tensor,
        max_steps: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        let mut prev: Vec<u32> = Vec::with_capacity(max_steps);
        for _ in 0..max_steps {
            let tok = self.step(text_hidden, &prev)?;
            prev.push(tok);
            if tok == eos_id {
                break;
            }
        }
        Ok(prev)
    }

    pub fn has_text_embedding(&self) -> bool {
        self.text_embedding.is_some()
    }

    pub fn embed_text_ids(&self, token_ids: &[u32]) -> Result<Tensor> {
        let te = self
            .text_embedding
            .as_ref()
            .ok_or_else(|| anyhow!("embed_text_ids: text_embedding not loaded"))?;
        if token_ids.is_empty() {
            return Ok(Tensor::zeros(
                (1usize, 0, self.cfg.text_hidden_size),
                self.cfg.dtype,
                &self.device,
            )?);
        }
        let ids = Tensor::from_vec(token_ids.to_vec(), token_ids.len(), &self.device)?;
        let emb = te.index_select(&ids, 0)?;
        Ok(emb.reshape((1usize, token_ids.len(), self.cfg.text_hidden_size))?)
    }

    pub fn generate_full_frames(
        &self,
        text_hidden: &Tensor,
        max_steps: usize,
        eos_id: u32,
        code_predictor: &crate::code_predictor::Qwen3TtsCodecDecoder,
    ) -> Result<Vec<[u32; nv_omni::vocoder::NUM_CODEBOOKS]>> {
        use nv_omni::vocoder::NUM_CODEBOOKS;
        let text_len = match text_hidden.dims() {
            [_, t, _] => *t,
            [t, _] => *t,
            d => anyhow::bail!("generate_full_frames: unexpected text_hidden shape {:?}", d),
        };
        let cache_len = text_len + max_steps + 1;
        let mut cache = self.new_kv_cache(cache_len)?;
        let mut prev_extras: Vec<[u32; crate::code_predictor::NUM_EXTRA_CODEBOOKS]> =
            Vec::with_capacity(max_steps);
        let mut frames: Vec<[u32; NUM_CODEBOOKS]> = Vec::with_capacity(max_steps);

        let (base, hidden) = self.step_cached_with_hidden(text_hidden, None, &mut cache)?;
        let h = hidden.reshape((1usize, 1usize, self.cfg.hidden_size))?;
        let extras = code_predictor.predict(&h, base, &prev_extras)?;
        prev_extras.push(extras);
        if base != eos_id {
            let mut row = [0u32; NUM_CODEBOOKS];
            row[0] = base;
            for (i, &e) in extras.iter().enumerate() {
                row[i + 1] = e;
            }
            frames.push(row);
        }
        let mut prev_tok = base;

        for _ in 1..max_steps {
            if prev_tok == eos_id {
                break;
            }
            let (base, hidden) =
                self.step_cached_with_hidden(text_hidden, Some(prev_tok), &mut cache)?;
            let h = hidden.reshape((1usize, 1usize, self.cfg.hidden_size))?;
            let extras = code_predictor.predict(&h, base, &prev_extras)?;
            prev_extras.push(extras);
            if base == eos_id {
                break;
            }
            let mut row = [0u32; NUM_CODEBOOKS];
            row[0] = base;
            for (i, &e) in extras.iter().enumerate() {
                row[i + 1] = e;
            }
            frames.push(row);
            prev_tok = base;
        }
        Ok(frames)
    }

    pub fn project_text_ids(&self, ids: &[u32]) -> Result<Tensor> {
        let emb = self.embed_text_ids(ids)?;
        let proj = self.project_text(&emb)?;
        Ok(proj.reshape((1usize, ids.len(), self.cfg.hidden_size))?)
    }

    pub fn codec_embed_rows(&self, ids: &[u32]) -> Result<Tensor> {
        self.embed_codec(ids)
    }

    pub fn build_nonstreaming_prefill(
        &self,
        instruct_ids: &[u32],
        role_ids: &[u32],
        body_ids: &[u32],
        speaker_embed: Option<&Tensor>,
        language_id: Option<u32>,
        tts_bos_id: u32,
        tts_eos_id: u32,
        tts_pad_id: u32,
    ) -> Result<(Tensor, Tensor)> {
        if role_ids.is_empty() || body_ids.is_empty() {
            anyhow::bail!(
                "build_nonstreaming_prefill: role_ids ({}) and body_ids ({}) must be non-empty",
                role_ids.len(),
                body_ids.len()
            );
        }
        let h = self.cfg.hidden_size;
        let tts_bos = self.project_text_ids(&[tts_bos_id])?;
        let tts_eos = self.project_text_ids(&[tts_eos_id])?;
        let tts_pad = self.project_text_ids(&[tts_pad_id])?;

        let role = self.project_text_ids(role_ids)?;

        let think_ids: Vec<u32> = match language_id {
            None => vec![CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID],
            Some(lang) => vec![CODEC_THINK_ID, CODEC_THINK_BOS_ID, lang, CODEC_THINK_EOS_ID],
        };
        let mut prefix_parts: Vec<Tensor> = vec![self.embed_codec(&think_ids)?];
        if let Some(sp) = speaker_embed {
            let sp = sp.to_dtype(self.cfg.dtype)?.reshape((1usize, 1usize, h))?;
            prefix_parts.push(sp);
        }
        prefix_parts.push(self.embed_codec(&[CODEC_PAD_ID, CODEC_BOS_ID])?);
        let refs: Vec<&Tensor> = prefix_parts.iter().collect();
        let codec_prefix = Tensor::cat(&refs, 1)?;
        let l = codec_prefix.dims()[1];

        let mut text_side: Vec<&Tensor> = Vec::with_capacity(l - 1);
        for _ in 0..(l - 2) {
            text_side.push(&tts_pad);
        }
        text_side.push(&tts_bos);
        let text_over_prefix = Tensor::cat(&text_side, 1)?;
        let aligned = text_over_prefix.add(&codec_prefix.narrow(1, 0, l - 1)?)?;

        let body = self.project_text_ids(body_ids)?;
        let full_text = Tensor::cat(&[&body, &tts_eos], 1)?;
        let pad_run = self.embed_codec(&vec![CODEC_PAD_ID; body_ids.len() + 1])?;
        let text_block = full_text.add(&pad_run)?;

        let last = tts_pad.add(&self.embed_codec(&[CODEC_BOS_ID])?)?;

        let mut parts = vec![role, aligned, text_block, last];
        if !instruct_ids.is_empty() {
            parts.insert(0, self.project_text_ids(instruct_ids)?);
        }
        Ok((Tensor::cat(&parts, 1)?, tts_pad))
    }

    pub fn generate_frames_streaming(
        &self,
        prefill: &Tensor,
        tts_pad_embed: &Tensor,
        code_predictor: &crate::code_predictor::Qwen3TtsCodecDecoder,
        max_steps: usize,
        min_steps: usize,
        talker_sampler: &mut crate::sampling::Sampler,
        sub_sampler: &mut crate::sampling::Sampler,
        on_frame: &mut dyn FnMut([u32; nv_omni::vocoder::NUM_CODEBOOKS]) -> Result<bool>,
    ) -> Result<usize> {
        use nv_omni::vocoder::NUM_CODEBOOKS;
        let codebook = code_predictor.config().codebook_vocab_size;
        let eos = CODEC_EOS_ID;
        let prefill_len = prefill.dims()[1];
        let mut cache = self.new_kv_cache(prefill_len + max_steps + 2)?;

        let (mut logits, mut hidden) = self.forward_cached_embeds(prefill, &mut cache)?;
        let mut bases: Vec<u32> = Vec::with_capacity(max_steps);
        let mut produced = 0usize;

        for step in 0..max_steps {
            let row = logits
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let eos_ok = produced >= min_steps;
            let base = talker_sampler.sample(&row, &bases, |i| {
                i < codebook || (i == eos as usize && eos_ok)
            })?;
            bases.push(base);
            if base == eos {
                break;
            }
            let base_emb = self.embed_codec(&[base])?;
            let h = hidden.reshape((1usize, 1usize, self.cfg.hidden_size))?;
            let extras = code_predictor.predict_sampled(&h, &base_emb, sub_sampler)?;
            let mut frame = [0u32; NUM_CODEBOOKS];
            frame[0] = base;
            for (i, &e) in extras.iter().enumerate() {
                frame[i + 1] = e;
            }
            produced += 1;
            if !on_frame(frame)? {
                break;
            }
            if step + 1 == max_steps {
                break;
            }
            let extras_sum = code_predictor.sum_extra_embeds(&extras)?;
            let next = base_emb
                .add(&extras_sum.to_dtype(self.cfg.dtype)?)?
                .add(&tts_pad_embed.to_dtype(self.cfg.dtype)?)?;
            let (l, hd) = self.forward_cached_embeds(&next, &mut cache)?;
            logits = l;
            hidden = hd;
        }
        Ok(produced)
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
    if row.is_empty() {
        anyhow::bail!("argmax_u32: empty logits");
    }
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

    fn cpu() -> Device {
        Device::Cpu
    }

    fn tiny_cfg() -> Qwen3TtsTalkerConfig {
        Qwen3TtsTalkerConfig {
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 64,
            speech_vocab_size: 16,
            text_vocab_size: 64,
            text_hidden_size: 16,
            rope_theta: 10000.0,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-6,
            mrope_section: vec![2, 1, 1],
            dtype: DType::F32,
            spk_id: Vec::new(),
            language_id: Vec::new(),
            quant: LinearQuant::Dense,
        }
    }

    #[test]
    fn a_zero_weight_talker_builds_and_steps_to_token_zero_with_or_without_a_prefix() {
        let dev = cpu();
        let cfg = tiny_cfg();
        let talker = Qwen3TtsTalker::new(cfg.clone(), &dev).expect("build");
        assert_eq!(talker.cfg.num_hidden_layers, cfg.num_hidden_layers);
        assert_eq!(
            talker.codec_embedding.dims(),
            &[cfg.speech_vocab_size, cfg.hidden_size]
        );
        assert_eq!(talker.layers.len(), cfg.num_hidden_layers);

        let text_hidden =
            Tensor::ones((1usize, 4, cfg.text_hidden_size), DType::F32, &dev).unwrap();
        let tok = talker.step(&text_hidden, &[]).expect("step");
        assert!(
            (tok as usize) < cfg.speech_vocab_size,
            "token id {tok} out of vocab {}",
            cfg.speech_vocab_size
        );
        assert_eq!(
            tok, 0,
            "zero-weight talker should return argmax-of-zeros = 0"
        );
        let tok = talker.step(&text_hidden, &[1u32, 2, 3]).expect("step w/prefix");
        assert!((tok as usize) < cfg.speech_vocab_size);
    }

    #[test]
    fn a_voice_design_instruction_is_a_projected_text_prefix_ahead_of_the_unchanged_prefill() {
        let dev = cpu();
        let cfg = Qwen3TtsTalkerConfig {
            speech_vocab_size: 2160,
            ..tiny_cfg()
        };
        let mut talker = Qwen3TtsTalker::new(cfg.clone(), &dev).expect("build");
        talker.text_embedding = Some(
            Tensor::randn(0f32, 1.0, (cfg.text_vocab_size, cfg.text_hidden_size), &dev).unwrap(),
        );
        talker.codec_embedding =
            Tensor::randn(0f32, 1.0, (cfg.speech_vocab_size, cfg.hidden_size), &dev).unwrap();
        let (th, h) = (cfg.text_hidden_size, cfg.hidden_size);
        let randn = |r: usize, c: usize| Tensor::randn(0f32, 1.0, (r, c), &dev).unwrap();
        talker.text_projection = TextProjection {
            fc1: DenseLinear::new(randn(th, th), None).unwrap(),
            fc2: DenseLinear::new(randn(h, th), None).unwrap(),
        };
        let prefill = |instruct: &[u32]| {
            talker
                .build_nonstreaming_prefill(instruct, &[1, 2, 3], &[4, 5], None, None, 6, 7, 8)
                .expect("prefill")
                .0
        };
        let plain = prefill(&[]);
        let designed = prefill(&[9, 10, 11, 12]);
        let n = plain.dims()[1];
        assert_eq!(designed.dims()[1], n + 4);
        let flat = |t: &Tensor| t.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(flat(&designed.narrow(1, 4, n).unwrap()), flat(&plain));
        assert_eq!(
            flat(&designed.narrow(1, 0, 4).unwrap()),
            flat(&talker.project_text_ids(&[9, 10, 11, 12]).unwrap())
        );
    }

    #[test]
    fn generate_stops_at_eos_or_max_steps() {
        let dev = cpu();
        let cfg = tiny_cfg();
        let talker = Qwen3TtsTalker::new(cfg.clone(), &dev).expect("build");
        let text_hidden =
            Tensor::ones((1usize, 1, cfg.text_hidden_size), DType::F32, &dev).unwrap();

        let tokens = talker.generate(&text_hidden, 8, 0).unwrap();
        assert_eq!(tokens, vec![0u32], "should stop after first EOS");

        let tokens = talker.generate(&text_hidden, 5, 999).unwrap();
        assert_eq!(tokens.len(), 5);
    }
}
