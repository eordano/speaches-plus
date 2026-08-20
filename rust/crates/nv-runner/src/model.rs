use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use nv_layers::{Linear, RmsNorm, Rope, RopeConfig, RopeKind};

use crate::config::Qwen3Config;

pub struct KvCache {
    keys: Vec<Option<Tensor>>,
    values: Vec<Option<Tensor>>,
    current_len: usize,
    max_seq_len: usize,
    dtype: DType,
    #[allow(dead_code)]
    device: Device,
}

impl KvCache {
    pub fn new(
        config: &Qwen3Config,
        max_seq_len: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let n = config.num_hidden_layers;
        Ok(Self {
            keys: (0..n).map(|_| None).collect(),
            values: (0..n).map(|_| None).collect(),
            current_len: 0,
            max_seq_len,
            dtype,
            device: device.clone(),
        })
    }

    pub fn reset(&mut self) {
        for k in self.keys.iter_mut() {
            *k = None;
        }
        for v in self.values.iter_mut() {
            *v = None;
        }
        self.current_len = 0;
    }

    pub fn current_len(&self) -> usize {
        self.current_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn num_layers(&self) -> usize {
        self.keys.len()
    }

    pub fn advance(&mut self, by: usize) {
        self.current_len += by;
    }

    pub fn keys(&self, layer: usize) -> Option<&Tensor> {
        self.keys[layer].as_ref()
    }

    pub fn values(&self, layer: usize) -> Option<&Tensor> {
        self.values[layer].as_ref()
    }

    pub fn set_layer(&mut self, layer: usize, k: Tensor, v: Tensor) {
        self.keys[layer] = Some(k);
        self.values[layer] = Some(v);
    }
}

struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    softmax_scale: f64,
}

impl Qwen3Attention {
    fn from_vb(vb: VarBuilder, cfg: &Qwen3Config) -> Result<Self> {
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let q_proj = Linear::from_candle_vb(vb.pp("q_proj"), cfg.hidden_size, q_dim, false)?;
        let k_proj = Linear::from_candle_vb(vb.pp("k_proj"), cfg.hidden_size, kv_dim, false)?;
        let v_proj = Linear::from_candle_vb(vb.pp("v_proj"), cfg.hidden_size, kv_dim, false)?;
        let o_proj = Linear::from_candle_vb(vb.pp("o_proj"), q_dim, cfg.hidden_size, false)?;
        let q_norm = RmsNorm::from_candle_vb(vb.pp("q_norm"), cfg.head_dim, cfg.rms_norm_eps)?;
        let k_norm = RmsNorm::from_candle_vb(vb.pp("k_norm"), cfg.head_dim, cfg.rms_norm_eps)?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        rope: &Rope,
        cache: &mut KvCache,
        layer_idx: usize,
        past_len: usize,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let (b, t, _h) = (dims[0], dims[1], dims[2]);

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape((b, t, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, t, self.num_kv_heads, self.head_dim))?;

        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k.to_dtype(DType::F32)?;

        let positions_2d = positions_to_i32_2d(positions, b, t, x.device())?;

        let (q_rot, k_rot) = rope.apply(&q_f32, &k_f32, &positions_2d)?;
        let q = q_rot.to_dtype(x.dtype())?;
        let k_new = k_rot.to_dtype(x.dtype())?;

        let v_new = v.contiguous()?;

        let (k_full, v_full) = if past_len > 0 {
            let k_past = cache
                .keys(layer_idx)
                .ok_or_else(|| anyhow::anyhow!("kv cache empty at layer {layer_idx}"))?
                .clone();
            let v_past = cache
                .values(layer_idx)
                .ok_or_else(|| anyhow::anyhow!("kv cache empty at layer {layer_idx}"))?
                .clone();
            let k = Tensor::cat(&[&k_past, &k_new], 1)?.contiguous()?;
            let v = Tensor::cat(&[&v_past, &v_new], 1)?.contiguous()?;
            (k, v)
        } else {
            (k_new.contiguous()?, v_new)
        };

        cache.set_layer(layer_idx, k_full.clone(), v_full.clone());

        let total_kv = past_len + t;
        let groups = self.num_heads / self.num_kv_heads;

        let q_btd = q.transpose(1, 2)?.contiguous()?;
        let k_btd = k_full.transpose(1, 2)?.contiguous()?;
        let v_btd = v_full.transpose(1, 2)?.contiguous()?;

        let k_expanded = k_btd
            .unsqueeze(2)?
            .expand((b, self.num_kv_heads, groups, total_kv, self.head_dim))?
            .reshape((b, self.num_heads, total_kv, self.head_dim))?
            .contiguous()?;
        let v_expanded = v_btd
            .unsqueeze(2)?
            .expand((b, self.num_kv_heads, groups, total_kv, self.head_dim))?
            .reshape((b, self.num_heads, total_kv, self.head_dim))?
            .contiguous()?;

        let q_f = q_btd.to_dtype(DType::F32)?;
        let k_f = k_expanded.to_dtype(DType::F32)?;
        let v_f = v_expanded.to_dtype(DType::F32)?;

        let scale = Tensor::new(self.softmax_scale as f32, x.device())?;
        let scores = q_f
            .matmul(&k_f.transpose(D::Minus2, D::Minus1)?.contiguous()?)?
            .broadcast_mul(&scale)?;

        let scores = if t > 1 {
            let mask = causal_mask(t, total_kv, past_len, x.device())?;
            scores.broadcast_add(&mask)?
        } else {
            scores
        };

        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let context = probs.matmul(&v_f)?;

        let context = context
            .to_dtype(x.dtype())?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.num_heads * self.head_dim))?;

        let out = self.o_proj.forward(&context)?;
        Ok(out)
    }
}

fn positions_to_i32_2d(positions: &Tensor, b: usize, t: usize, device: &Device) -> Result<Tensor> {
    let pos_cpu = positions.to_device(&Device::Cpu)?;
    let flat_i32: Vec<i32> = match pos_cpu.dtype() {
        DType::I32 => pos_cpu.flatten_all()?.to_vec1::<i32>()?,
        DType::I64 => pos_cpu
            .flatten_all()?
            .to_vec1::<i64>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        DType::U32 => pos_cpu
            .flatten_all()?
            .to_vec1::<u32>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        DType::U8 => pos_cpu
            .flatten_all()?
            .to_vec1::<u8>()?
            .into_iter()
            .map(|v| v as i32)
            .collect(),
        other => anyhow::bail!("unsupported positions dtype {other:?}"),
    };
    let dims = positions.dims();
    if dims.len() == 1 {
        let n = dims[0];
        if n == t {
            let mut tiled = Vec::with_capacity(b * t);
            for _ in 0..b {
                tiled.extend_from_slice(&flat_i32);
            }
            Ok(Tensor::from_vec(tiled, (b, t), device)?)
        } else if n == b * t {
            Ok(Tensor::from_vec(flat_i32, (b, t), device)?)
        } else {
            anyhow::bail!("positions length {n} != b*t {}", b * t);
        }
    } else if dims.len() == 2 && dims[0] == b && dims[1] == t {
        Ok(Tensor::from_vec(flat_i32, (b, t), device)?)
    } else {
        anyhow::bail!("positions has unexpected shape {:?}", dims);
    }
}

fn causal_mask(q_len: usize, kv_len: usize, past_len: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; q_len * kv_len];
    for i in 0..q_len {
        let allowed = past_len + i + 1;
        for j in 0..kv_len {
            if j >= allowed {
                data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let m = Tensor::from_vec(data, (q_len, kv_len), device)?;
    Ok(m.unsqueeze(0)?.unsqueeze(0)?)
}

struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn from_vb(vb: VarBuilder, cfg: &Qwen3Config) -> Result<Self> {
        let gate_proj = Linear::from_candle_vb(
            vb.pp("gate_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let up_proj = Linear::from_candle_vb(
            vb.pp("up_proj"),
            cfg.hidden_size,
            cfg.intermediate_size,
            false,
        )?;
        let down_proj = Linear::from_candle_vb(
            vb.pp("down_proj"),
            cfg.intermediate_size,
            cfg.hidden_size,
            false,
        )?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let activated = candle_nn::ops::silu(&gate)?.mul(&up)?;
        self.down_proj.forward(&activated)
    }
}

struct Qwen3Layer {
    pre_attn_norm: RmsNorm,
    attn: Qwen3Attention,
    pre_mlp_norm: RmsNorm,
    mlp: Qwen3Mlp,
}

impl Qwen3Layer {
    fn from_vb(vb: VarBuilder, cfg: &Qwen3Config) -> Result<Self> {
        let pre_attn_norm =
            RmsNorm::from_candle_vb(vb.pp("input_layernorm"), cfg.hidden_size, cfg.rms_norm_eps)?;
        let attn = Qwen3Attention::from_vb(vb.pp("self_attn"), cfg)?;
        let pre_mlp_norm = RmsNorm::from_candle_vb(
            vb.pp("post_attention_layernorm"),
            cfg.hidden_size,
            cfg.rms_norm_eps,
        )?;
        let mlp = Qwen3Mlp::from_vb(vb.pp("mlp"), cfg)?;
        Ok(Self {
            pre_attn_norm,
            attn,
            pre_mlp_norm,
            mlp,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        positions: &Tensor,
        rope: &Rope,
        cache: &mut KvCache,
        layer_idx: usize,
        past_len: usize,
    ) -> Result<Tensor> {
        let normed = self.pre_attn_norm.forward(x)?;
        let attn_out = self
            .attn
            .forward(&normed, positions, rope, cache, layer_idx, past_len)?;
        let h = x.add(&attn_out)?;
        let normed2 = self.pre_mlp_norm.forward(&h)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        Ok(h.add(&mlp_out)?)
    }
}

pub struct Qwen3 {
    config: Qwen3Config,
    embed_tokens: Embedding,
    layers: Vec<Qwen3Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    #[allow(dead_code)]
    device: Device,
}

impl Qwen3 {
    pub fn from_var_builder(config: Qwen3Config, vb: VarBuilder, device: &Device) -> Result<Self> {
        let embed_tokens =
            candle_nn::embedding(config.vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = Qwen3Layer::from_vb(vb.pp(format!("layers.{i}")), &config)?;
            layers.push(layer);
        }

        let norm = RmsNorm::from_candle_vb(vb.pp("norm"), config.hidden_size, config.rms_norm_eps)?;

        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)?
        } else {
            Linear::from_candle_vb(
                vb.pp("lm_head"),
                config.hidden_size,
                config.vocab_size,
                false,
            )?
        };

        let rope = Rope::new(
            RopeConfig {
                head_dim: config.head_dim,
                max_seq_len: config.max_position_embeddings.min(8192),
                base: config.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope,
            device: device.clone(),
        })
    }

    pub fn from_pretrained(
        config: Qwen3Config,
        safetensor_files: &[std::path::PathBuf],
        device: &Device,
    ) -> Result<Self> {
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(safetensor_files, DType::BF16, device)? };
        Self::from_var_builder(config, vb, device)
    }

    pub fn config(&self) -> &Qwen3Config {
        &self.config
    }

    pub fn forward(
        &self,
        tokens: &Tensor,
        positions: &Tensor,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        let past_len = cache.current_len();
        let token_dims = tokens.dims();
        let (b, t) = (token_dims[0], token_dims[1]);

        let tokens_flat = tokens.reshape(b * t)?;
        let emb = self
            .embed_tokens
            .embeddings()
            .index_select(&tokens_flat, 0)?;
        let mut h = emb.reshape((b, t, self.config.hidden_size))?;

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, positions, &self.rope, cache, i, past_len)?;
        }

        let h = self.norm.forward(&h)?;
        let logits = self.lm_head.forward(&h)?;
        cache.advance(t);
        Ok(logits)
    }
}
