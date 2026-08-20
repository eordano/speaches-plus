use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};

use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};

#[derive(Clone, Debug)]
pub struct OmniThinkerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub moe_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub mrope_section: [usize; 3],
    pub dtype: DType,
}

impl OmniThinkerConfig {
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn from_hf_config_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = path.as_ref();
        let raw = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?;
        let tc = v
            .get("thinker_config")
            .and_then(|t| t.get("text_config"))
            .ok_or_else(|| anyhow::anyhow!("config.json: missing thinker_config.text_config"))?;
        let obj = tc
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("text_config must be an object"))?;
        let geti = |k: &str| -> Result<usize> {
            obj.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow::anyhow!("text_config: missing or non-int {k}"))
        };

        let head_dim = geti("head_dim")?;
        let num_experts = geti("num_experts")?;
        if num_experts == 0 {
            anyhow::bail!("OmniThinkerConfig: num_experts must be > 0, got {num_experts}");
        }
        let norm_topk = obj
            .get("norm_topk_prob")
            .and_then(|x| x.as_bool())
            .unwrap_or(false);
        if !norm_topk {
            anyhow::bail!("OmniThinkerConfig: norm_topk_prob must be true for this decoder");
        }
        let mrope_section: Vec<usize> = tc
            .get("rope_scaling")
            .and_then(|r| r.get("mrope_section"))
            .and_then(|s| s.as_array())
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|v| v as usize)).collect())
            .unwrap_or_default();
        if mrope_section.len() != 3 {
            anyhow::bail!(
                "OmniThinkerConfig: rope_scaling.mrope_section must have 3 entries, got {:?}",
                mrope_section
            );
        }
        let sum: usize = mrope_section.iter().sum();
        if sum != head_dim / 2 {
            anyhow::bail!(
                "OmniThinkerConfig: mrope_section sum {} != head_dim/2 {}",
                sum,
                head_dim / 2
            );
        }
        let rope_theta = obj
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .unwrap_or(1_000_000.0) as f32;

        Ok(Self {
            vocab_size: geti("vocab_size")?,
            hidden_size: geti("hidden_size")?,
            num_hidden_layers: geti("num_hidden_layers")?,
            num_attention_heads: geti("num_attention_heads")?,
            num_key_value_heads: geti("num_key_value_heads")?,
            head_dim,
            moe_intermediate_size: geti("moe_intermediate_size")?,
            num_experts,
            num_experts_per_tok: geti("num_experts_per_tok")?,
            rms_norm_eps: obj
                .get("rms_norm_eps")
                .and_then(|x| x.as_f64())
                .unwrap_or(1e-6),
            rope_theta,
            max_position_embeddings: geti("max_position_embeddings")?,
            mrope_section: [mrope_section[0], mrope_section[1], mrope_section[2]],
            dtype: DType::BF16,
        })
    }
}

#[derive(Clone, Debug)]
pub struct OmniSpecialIds {
    pub im_start: u32,
    pub im_end: u32,
    pub vision_start: u32,
    pub vision_end: u32,
    pub image_pad: u32,
    pub audio_start: u32,
    pub audio_end: u32,
    pub audio_pad: u32,
    pub endoftext: u32,
    pub position_id_per_seconds: u32,
}

impl Default for OmniSpecialIds {
    fn default() -> Self {
        Self {
            im_start: 151644,
            im_end: 151645,
            vision_start: 151652,
            vision_end: 151653,
            image_pad: 151655,
            audio_start: 151669,
            audio_end: 151670,
            audio_pad: 151675,
            endoftext: 151643,
            position_id_per_seconds: 13,
        }
    }
}

impl OmniSpecialIds {
    pub fn from_hf_config_json(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let p = path.as_ref();
        let raw = std::fs::read_to_string(p).with_context(|| format!("read {}", p.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", p.display()))?;
        let tc = v
            .get("thinker_config")
            .ok_or_else(|| anyhow::anyhow!("config.json: missing thinker_config"))?;
        let d = Self::default();
        let getu = |root: &serde_json::Value, k: &str, fallback: u32| -> u32 {
            root.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as u32)
                .unwrap_or(fallback)
        };
        let ids = Self {
            im_start: getu(&v, "im_start_token_id", d.im_start),
            im_end: getu(&v, "im_end_token_id", d.im_end),
            vision_start: getu(tc, "vision_start_token_id", d.vision_start),
            vision_end: getu(tc, "vision_end_token_id", d.vision_end),
            image_pad: getu(tc, "image_token_id", d.image_pad),
            audio_start: getu(tc, "audio_start_token_id", d.audio_start),
            audio_end: getu(tc, "audio_end_token_id", d.audio_end),
            audio_pad: getu(tc, "audio_token_id", d.audio_pad),
            endoftext: d.endoftext,
            position_id_per_seconds: getu(tc, "position_id_per_seconds", d.position_id_per_seconds),
        };
        Ok(ids)
    }
}

#[derive(Debug)]
pub struct ModalitySplice {
    pub position: usize,
    pub embedding: Tensor,
}

#[derive(Clone, Debug, Default)]
pub struct OmniPositions {
    pub t: Vec<u32>,
    pub h: Vec<u32>,
    pub w: Vec<u32>,
}

impl OmniPositions {
    pub fn uniform(positions: &[u32]) -> Self {
        Self {
            t: positions.to_vec(),
            h: positions.to_vec(),
            w: positions.to_vec(),
        }
    }
    pub fn len(&self) -> usize {
        self.t.len()
    }
    pub fn is_empty(&self) -> bool {
        self.t.is_empty()
    }
}

pub struct OmniKvCache {
    layers: Vec<Option<(Tensor, Tensor)>>,
    len: usize,
}

impl OmniKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            len: 0,
        }
    }
    pub fn reset(&mut self) {
        for l in self.layers.iter_mut() {
            *l = None;
        }
        self.len = 0;
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

pub struct OmniDeepstack {
    pub rows: Tensor,
    pub embeds: Vec<Tensor>,
}

struct OmniMoe {
    gate: Linear,
    experts: Vec<Mlp>,
    num_experts: usize,
    num_experts_per_tok: usize,
    hidden_size: usize,
}

impl OmniMoe {
    fn forward(&self, x_flat: &Tensor) -> Result<Tensor> {
        let (n_tokens, hidden) = x_flat.dims2().map_err(|e| anyhow::anyhow!(e))?;
        if hidden != self.hidden_size {
            anyhow::bail!("OmniMoe: hidden {} != {}", hidden, self.hidden_size);
        }
        let device = x_flat.device().clone();
        let in_dtype = x_flat.dtype();
        let logits = self.gate.forward(x_flat)?.to_dtype(DType::F32)?.contiguous()?;
        let (sorted_logits, sorted_idx) = logits.sort_last_dim(false)?;
        let k = self.num_experts_per_tok;
        let top_logits = sorted_logits.narrow(1, 0, k)?.contiguous()?;
        let top_idx = sorted_idx.narrow(1, 0, k)?.contiguous()?;
        let top_weights = candle_nn::ops::softmax_last_dim(&top_logits)?.contiguous()?;

        let top_idx_host: Vec<u32> = top_idx.flatten_all()?.to_vec1::<u32>()?;
        let top_weights_host: Vec<f32> = top_weights.flatten_all()?.to_vec1::<f32>()?;
        if let Some(&bad) = top_idx_host.iter().find(|&&e| e as usize >= self.num_experts) {
            anyhow::bail!(
                "OmniMoe: routed expert id {} out of range ({} experts)",
                bad,
                self.num_experts
            );
        }

        let mut expert_rows: Vec<Vec<u32>> = vec![Vec::new(); self.num_experts];
        let mut expert_w: Vec<Vec<f32>> = vec![Vec::new(); self.num_experts];
        for n in 0..n_tokens {
            for j in 0..k {
                let e = top_idx_host[n * k + j] as usize;
                expert_rows[e].push(n as u32);
                expert_w[e].push(top_weights_host[n * k + j]);
            }
        }

        let mut acc = Tensor::zeros((n_tokens, hidden), DType::F32, &device)?;
        for e in 0..self.num_experts {
            let rows = &expert_rows[e];
            if rows.is_empty() {
                continue;
            }
            let m = rows.len();
            let idx_t = Tensor::from_vec(rows.clone(), m, &device)?;
            let gathered = x_flat.index_select(&idx_t, 0)?.contiguous()?;
            let y_e = self.experts[e].forward(&gathered)?.to_dtype(DType::F32)?;
            let w_t = Tensor::from_vec(expert_w[e].clone(), (m, 1), &device)?;
            let weighted = y_e.broadcast_mul(&w_t)?;
            acc = acc.index_add(&idx_t, &weighted, 0)?;
        }
        Ok(acc.to_dtype(in_dtype)?)
    }
}

struct OmniLayer {
    input_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    moe: OmniMoe,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl OmniLayer {
    fn new(cfg: &OmniThinkerConfig, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let q = cfg.q_dim();
        let kv = cfg.kv_dim();
        let hd = cfg.head_dim;
        let dtype = cfg.dtype;
        let eps = cfg.rms_norm_eps;

        let lin = |o: usize, i: usize| -> Result<Linear> {
            Linear::new(Tensor::zeros((o, i), dtype, device)?, None)
        };
        let expert_lin = |o: usize, i: usize| -> Result<Linear> {
            Linear::new_no_pretranspose(Tensor::zeros((o, i), dtype, device)?, None)
        };
        let mut experts = Vec::with_capacity(cfg.num_experts);
        for _ in 0..cfg.num_experts {
            experts.push(Mlp::new(
                expert_lin(cfg.moe_intermediate_size, h)?,
                expert_lin(cfg.moe_intermediate_size, h)?,
                expert_lin(h, cfg.moe_intermediate_size)?,
            )?);
        }
        let moe = OmniMoe {
            gate: lin(cfg.num_experts, h)?,
            experts,
            num_experts: cfg.num_experts,
            num_experts_per_tok: cfg.num_experts_per_tok,
            hidden_size: h,
        };

        Ok(Self {
            input_norm: RmsNorm::new(Tensor::ones(h, dtype, device)?, eps),
            q_proj: lin(q, h)?,
            k_proj: lin(kv, h)?,
            v_proj: lin(kv, h)?,
            o_proj: lin(h, q)?,
            q_norm: RmsNorm::new(Tensor::ones(hd, dtype, device)?, eps),
            k_norm: RmsNorm::new(Tensor::ones(hd, dtype, device)?, eps),
            post_attn_norm: RmsNorm::new(Tensor::ones(h, dtype, device)?, eps),
            moe,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: hd,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        pos: &OmniPositions,
        rope: &Rope,
        sections: [usize; 3],
        cache: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, t, _h) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;

        let normed = self.input_norm.forward(x)?;
        let q = self.q_proj.forward(&normed)?.reshape((b, t, nh, hd))?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_proj.forward(&normed)?.reshape((b, t, nkv, hd))?;
        let k = self.k_norm.forward(&k)?;
        let v = self.v_proj.forward(&normed)?.reshape((b, t, nkv, hd))?;

        let (q, k) = apply_interleaved_mrope(rope, &q, &k, pos, sections)?;

        let (k_all, v_all) = match cache.take() {
            Some((k_prev, v_prev)) => {
                let k_all = Tensor::cat(&[&k_prev, &k], 1)?.contiguous()?;
                let v_all = Tensor::cat(&[&v_prev, &v], 1)?.contiguous()?;
                (k_all, v_all)
            }
            None => (k.contiguous()?, v.contiguous()?),
        };
        *cache = Some((k_all.clone(), v_all.clone()));

        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: true,
        };
        let attn_out = sdpa(&q, &k_all, &v_all, &attn_cfg)?;
        let attn_out = attn_out.reshape((b, t, nh * hd)).map_err(|e| anyhow::anyhow!(e))?;
        let x = (x + self.o_proj.forward(&attn_out)?).map_err(|e| anyhow::anyhow!(e))?;

        let normed = self.post_attn_norm.forward(&x)?;
        let normed_flat = normed.reshape((b * t, self.moe.hidden_size))?;
        let moe_out = self.moe.forward(&normed_flat)?.reshape((b, t, self.moe.hidden_size))?;
        (x + moe_out).map_err(|e| anyhow::anyhow!(e))
    }
}

pub struct OmniThinker {
    cfg: OmniThinkerConfig,
    embed_tokens: Tensor,
    layers: Vec<OmniLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rope: Rope,
    device: Device,
}

impl OmniThinker {
    pub fn new(cfg: OmniThinkerConfig, device: &Device) -> Result<Self> {
        if cfg.q_dim() != cfg.num_attention_heads * cfg.head_dim {
            anyhow::bail!("OmniThinkerConfig: inconsistent q_dim");
        }
        if !cfg.num_attention_heads.is_multiple_of(cfg.num_key_value_heads) {
            anyhow::bail!(
                "OmniThinkerConfig: num_attention_heads {} not divisible by kv_heads {}",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        if !cfg.head_dim.is_multiple_of(2) {
            anyhow::bail!("OmniThinkerConfig: head_dim {} must be even", cfg.head_dim);
        }

        let embed_tokens = Tensor::zeros((cfg.vocab_size, cfg.hidden_size), cfg.dtype, device)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(OmniLayer::new(&cfg, device)?);
        }
        let final_norm = RmsNorm::new(Tensor::ones(cfg.hidden_size, cfg.dtype, device)?, cfg.rms_norm_eps);
        let lm_head = Linear::new(
            Tensor::zeros((cfg.vocab_size, cfg.hidden_size), cfg.dtype, device)?,
            None,
        )?;
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
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            rope,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &OmniThinkerConfig {
        &self.cfg
    }
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn embed_with_splices(&self, token_ids: &[u32], splices: &[ModalitySplice]) -> Result<Tensor> {
        let seq = token_ids.len();
        if seq == 0 {
            anyhow::bail!("OmniThinker::embed_with_splices: empty token sequence");
        }
        let ids = Tensor::from_vec(token_ids.to_vec(), seq, &self.device)?.to_dtype(DType::U32)?;
        let mut x = self
            .embed_tokens
            .index_select(&ids, 0)?
            .reshape((seq, self.cfg.hidden_size))?
            .to_dtype(self.cfg.dtype)?;

        for splice in splices {
            let pos = splice.position;
            let dims = splice.embedding.dims();
            if dims.len() != 2 {
                anyhow::bail!(
                    "ModalitySplice at pos {}: embedding must be rank-2, got {:?}",
                    pos,
                    dims
                );
            }
            let slots = dims[0];
            if dims[1] != self.cfg.hidden_size {
                anyhow::bail!(
                    "ModalitySplice at pos {}: hidden {} != {}",
                    pos,
                    dims[1],
                    self.cfg.hidden_size
                );
            }
            if pos + slots > seq {
                anyhow::bail!(
                    "ModalitySplice at pos {} with {} slots exceeds seq len {}",
                    pos,
                    slots,
                    seq
                );
            }
            let splice_emb = splice.embedding.to_dtype(self.cfg.dtype)?;
            x = splice_into(&x, pos, &splice_emb)?;
        }
        Ok(x)
    }

    pub fn forward_step(
        &self,
        x: &Tensor,
        pos: &OmniPositions,
        cache: &mut OmniKvCache,
        deepstack: Option<&OmniDeepstack>,
    ) -> Result<Tensor> {
        let (b, t, h) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
        if b != 1 {
            anyhow::bail!("OmniThinker::forward_step: batch must be 1, got {b}");
        }
        if h != self.cfg.hidden_size {
            anyhow::bail!("OmniThinker::forward_step: hidden {} != {}", h, self.cfg.hidden_size);
        }
        if pos.len() != t {
            anyhow::bail!("OmniThinker::forward_step: positions {} != tokens {}", pos.len(), t);
        }

        let mut x = x.contiguous()?;
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, pos, &self.rope, self.cfg.mrope_section, &mut cache.layers[i])?;
            if let Some(ds) = deepstack {
                if i < ds.embeds.len() {
                    let flat = x.reshape((t, self.cfg.hidden_size))?;
                    let add = ds.embeds[i].to_dtype(self.cfg.dtype)?;
                    let flat = flat.index_add(&ds.rows, &add, 0)?;
                    x = flat.reshape((1, t, self.cfg.hidden_size))?;
                }
            }
        }
        cache.len += t;

        let x = self.final_norm.forward(&x)?;
        let last = x.narrow(1, t - 1, 1)?.contiguous()?;
        let logits = self.lm_head.forward(&last)?;
        let logits = logits.reshape(self.cfg.vocab_size)?.to_dtype(DType::F32)?;
        Ok(logits)
    }

    pub fn load_weights(&mut self, weights: &nv_weights::WeightLoader) -> Result<usize> {
        let dtype = self.cfg.dtype;
        let h = self.cfg.hidden_size;
        let q = self.cfg.q_dim();
        let kv = self.cfg.kv_dim();
        let hd = self.cfg.head_dim;
        let inter = self.cfg.moe_intermediate_size;
        let vocab = self.cfg.vocab_size;
        let eps = self.cfg.rms_norm_eps;
        let mut count = 0usize;

        self.embed_tokens = load_2d(weights, "thinker.model.embed_tokens.weight", (vocab, h), dtype)?;
        count += 1;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("thinker.model.layers.{i}");
            layer.input_norm =
                RmsNorm::new(load_1d(weights, &format!("{p}.input_layernorm.weight"), h, dtype)?, eps);
            layer.post_attn_norm = RmsNorm::new(
                load_1d(weights, &format!("{p}.post_attention_layernorm.weight"), h, dtype)?,
                eps,
            );
            layer.q_proj =
                Linear::new(load_2d(weights, &format!("{p}.self_attn.q_proj.weight"), (q, h), dtype)?, None)?;
            layer.k_proj =
                Linear::new(load_2d(weights, &format!("{p}.self_attn.k_proj.weight"), (kv, h), dtype)?, None)?;
            layer.v_proj =
                Linear::new(load_2d(weights, &format!("{p}.self_attn.v_proj.weight"), (kv, h), dtype)?, None)?;
            layer.o_proj =
                Linear::new(load_2d(weights, &format!("{p}.self_attn.o_proj.weight"), (h, q), dtype)?, None)?;
            layer.q_norm =
                RmsNorm::new(load_1d(weights, &format!("{p}.self_attn.q_norm.weight"), hd, dtype)?, eps);
            layer.k_norm =
                RmsNorm::new(load_1d(weights, &format!("{p}.self_attn.k_norm.weight"), hd, dtype)?, eps);
            layer.moe.gate =
                Linear::new(load_2d(weights, &format!("{p}.mlp.gate.weight"), (self.cfg.num_experts, h), dtype)?, None)?;
            count += 9;
            for (j, expert) in layer.moe.experts.iter_mut().enumerate() {
                let ep = format!("{p}.mlp.experts.{j}");
                let gate = Linear::new_no_pretranspose(
                    load_2d(weights, &format!("{ep}.gate_proj.weight"), (inter, h), dtype)?,
                    None,
                )?;
                let up = Linear::new_no_pretranspose(
                    load_2d(weights, &format!("{ep}.up_proj.weight"), (inter, h), dtype)?,
                    None,
                )?;
                let down = Linear::new_no_pretranspose(
                    load_2d(weights, &format!("{ep}.down_proj.weight"), (h, inter), dtype)?,
                    None,
                )?;
                *expert = Mlp::new(gate, up, down)?;
                count += 3;
            }
            if i % 8 == 0 {
                eprintln!("[omni-thinker] loaded layer {i}/{}", self.cfg.num_hidden_layers);
            }
        }

        self.final_norm = RmsNorm::new(load_1d(weights, "thinker.model.norm.weight", h, dtype)?, eps);
        self.lm_head = Linear::new(load_2d(weights, "thinker.lm_head.weight", (vocab, h), dtype)?, None)?;
        count += 2;
        Ok(count)
    }
}

fn apply_interleaved_mrope(
    rope: &Rope,
    q: &Tensor,
    k: &Tensor,
    pos: &OmniPositions,
    sections: [usize; 3],
) -> Result<(Tensor, Tensor)> {
    if pos.t == pos.h && pos.h == pos.w {
        let positions = Tensor::from_vec(pos.t.clone(), pos.t.len(), q.device())?;
        return rope.apply(q, k, &positions);
    }
    let device = q.device();
    let half = rope.config().head_dim / 2;
    let (cos, sin) = mrope_cos_sin(rope, pos, sections, half, device)?;
    let q_out = apply_cos_sin(q, &cos, &sin, half)?;
    let k_out = apply_cos_sin(k, &cos, &sin, half)?;
    Ok((q_out, k_out))
}

fn axis_for_freq(i: usize, sections: [usize; 3]) -> usize {
    let boundary = 3 * sections[1].min(sections[2]);
    if i >= boundary {
        0
    } else {
        i % 3
    }
}

fn mrope_cos_sin(
    rope: &Rope,
    pos: &OmniPositions,
    sections: [usize; 3],
    half: usize,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let tokens = pos.t.len();
    let pos_t = Tensor::from_vec(pos.t.clone(), tokens, device)?;
    let pos_h = Tensor::from_vec(pos.h.clone(), tokens, device)?;
    let pos_w = Tensor::from_vec(pos.w.clone(), tokens, device)?;
    let cos_full = rope.cos();
    let sin_full = rope.sin();
    let cos_t = cos_full.index_select(&pos_t, 0)?.to_dtype(DType::F32)?;
    let cos_h = cos_full.index_select(&pos_h, 0)?.to_dtype(DType::F32)?;
    let cos_w = cos_full.index_select(&pos_w, 0)?.to_dtype(DType::F32)?;
    let sin_t = sin_full.index_select(&pos_t, 0)?.to_dtype(DType::F32)?;
    let sin_h = sin_full.index_select(&pos_h, 0)?.to_dtype(DType::F32)?;
    let sin_w = sin_full.index_select(&pos_w, 0)?.to_dtype(DType::F32)?;

    let mut mask_t = vec![0f32; half];
    let mut mask_h = vec![0f32; half];
    let mut mask_w = vec![0f32; half];
    for i in 0..half {
        match axis_for_freq(i, sections) {
            0 => mask_t[i] = 1.0,
            1 => mask_h[i] = 1.0,
            _ => mask_w[i] = 1.0,
        }
    }
    let mt = Tensor::from_vec(mask_t, (1, half), device)?;
    let mh = Tensor::from_vec(mask_h, (1, half), device)?;
    let mw = Tensor::from_vec(mask_w, (1, half), device)?;

    let cos = cos_t
        .broadcast_mul(&mt)?
        .add(&cos_h.broadcast_mul(&mh)?)?
        .add(&cos_w.broadcast_mul(&mw)?)?;
    let sin = sin_t
        .broadcast_mul(&mt)?
        .add(&sin_h.broadcast_mul(&mh)?)?
        .add(&sin_w.broadcast_mul(&mw)?)?;
    Ok((cos, sin))
}

fn apply_cos_sin(x: &Tensor, cos: &Tensor, sin: &Tensor, half: usize) -> Result<Tensor> {
    let dtype = x.dtype();
    let dims = x.dims().to_vec();
    let head_dim = *dims.last().unwrap();
    let n_heads = dims[dims.len() - 2];
    let tokens: usize = dims[..dims.len() - 2].iter().product();
    let xf = x.to_dtype(DType::F32)?.reshape((tokens, n_heads, head_dim))?;
    let cos = cos.unsqueeze(1)?;
    let sin = sin.unsqueeze(1)?;
    let lo = xf.narrow(2, 0, half)?;
    let hi = xf.narrow(2, half, half)?;
    let out_lo = lo.broadcast_mul(&cos)?.sub(&hi.broadcast_mul(&sin)?)?;
    let out_hi = lo.broadcast_mul(&sin)?.add(&hi.broadcast_mul(&cos)?)?;
    let out = Tensor::cat(&[&out_lo, &out_hi], 2)?;
    Ok(out.reshape(dims)?.to_dtype(dtype)?)
}

pub fn build_mrope_positions(
    tokens: &[u32],
    ids: &OmniSpecialIds,
    image_grids: &[(usize, usize, usize)],
    audio_token_lens: &[usize],
) -> Result<(OmniPositions, u32)> {
    let n = tokens.len();
    let mut tp: Vec<u32> = Vec::with_capacity(n);
    let mut hp: Vec<u32> = Vec::with_capacity(n);
    let mut wp: Vec<u32> = Vec::with_capacity(n);
    let pps = ids.position_id_per_seconds;

    let cur_max = |tp: &[u32], hp: &[u32], wp: &[u32]| -> i64 {
        let mx = tp.iter().chain(hp.iter()).chain(wp.iter()).copied().max();
        match mx {
            Some(v) => v as i64,
            None => -1,
        }
    };
    let find_from = |id: u32, from: usize| -> usize {
        tokens[from..]
            .iter()
            .position(|&x| x == id)
            .map(|p| p + from)
            .unwrap_or(n + 1)
    };

    let mut st = 0usize;
    let mut img_i = 0usize;
    let mut aud_i = 0usize;
    let n_media = image_grids.len() + audio_token_lens.len();

    for _ in 0..n_media {
        let mut st_idx = (cur_max(&tp, &hp, &wp) + 1) as u32;
        let ed_vision = if img_i < image_grids.len() {
            find_from(ids.vision_start, st)
        } else {
            n + 1
        };
        let ed_audio = if aud_i < audio_token_lens.len() {
            find_from(ids.audio_start, st)
        } else {
            n + 1
        };
        let min_ed = ed_vision.min(ed_audio);
        if min_ed > n {
            break;
        }
        let text_len = min_ed - st;
        for p in 0..text_len {
            let v = st_idx + p as u32;
            tp.push(v);
            hp.push(v);
            wp.push(v);
        }
        st_idx += text_len as u32;
        tp.push(st_idx);
        hp.push(st_idx);
        wp.push(st_idx);
        st_idx += 1;

        if ed_audio <= ed_vision {
            let al = audio_token_lens[aud_i];
            for p in 0..al {
                let v = st_idx + p as u32;
                tp.push(v);
                hp.push(v);
                wp.push(v);
            }
            st += text_len + 1 + al;
            aud_i += 1;
        } else {
            let (gt, lh, lw) = image_grids[img_i];
            for ti in 0..gt {
                let tval = (ti as u32) * pps;
                for hh in 0..lh {
                    for ww in 0..lw {
                        tp.push(st_idx + tval);
                        hp.push(st_idx + hh as u32);
                        wp.push(st_idx + ww as u32);
                    }
                }
            }
            st += text_len + 1 + gt * lh * lw;
            img_i += 1;
        }
    }

    if st < n {
        let st_idx = (cur_max(&tp, &hp, &wp) + 1) as u32;
        let text_len = n - st;
        for p in 0..text_len {
            let v = st_idx + p as u32;
            tp.push(v);
            hp.push(v);
            wp.push(v);
        }
    }

    if tp.len() != n {
        anyhow::bail!(
            "build_mrope_positions: produced {} positions for {} tokens",
            tp.len(),
            n
        );
    }
    let next = (cur_max(&tp, &hp, &wp) + 1) as u32;
    Ok((OmniPositions { t: tp, h: hp, w: wp }, next))
}

fn splice_into(x: &Tensor, position: usize, splice: &Tensor) -> Result<Tensor> {
    let (seq, _h) = x.dims2().map_err(|e| anyhow::anyhow!(e))?;
    let slots = splice.dim(0).map_err(|e| anyhow::anyhow!(e))?;
    let end = position + slots;
    let mut parts: Vec<Tensor> = Vec::with_capacity(3);
    if position > 0 {
        parts.push(x.i(0..position)?);
    }
    parts.push(splice.clone());
    if end < seq {
        parts.push(x.i(end..seq)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 0)?)
}

fn load_1d(weights: &nv_weights::WeightLoader, name: &str, dim: usize, dtype: DType) -> Result<Tensor> {
    let w = weights.get(name, dtype).with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 1 || d[0] != dim {
        anyhow::bail!("{name}: expected [{}], got {:?}", dim, d);
    }
    Ok(w)
}

fn load_2d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    shape: (usize, usize),
    dtype: DType,
) -> Result<Tensor> {
    let w = weights.get(name, dtype).with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != shape.0 || d[1] != shape.1 {
        anyhow::bail!("{name}: expected [{}, {}], got {:?}", shape.0, shape.1, d);
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> OmniThinkerConfig {
        OmniThinkerConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            moe_intermediate_size: 12,
            num_experts: 4,
            num_experts_per_tok: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
            max_position_embeddings: 128,
            mrope_section: [2, 1, 1],
            dtype: DType::F32,
        }
    }

    #[test]
    fn builds_and_forward_step_runs() {
        let cfg = tiny_cfg();
        let t = OmniThinker::new(cfg.clone(), &Device::Cpu).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let x = t.embed_with_splices(&tokens, &[]).unwrap();
        let x = x.unsqueeze(0).unwrap();
        let mut cache = OmniKvCache::new(cfg.num_hidden_layers);
        let pos = OmniPositions::uniform(&[0, 1, 2, 3, 4]);
        let logits = t.forward_step(&x, &pos, &mut cache, None).unwrap();
        assert_eq!(logits.dims(), &[cfg.vocab_size]);
        assert_eq!(cache.len(), 5);
        let v: Vec<f32> = logits.to_vec1().unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn kv_cache_decode_matches_full_prefill() {
        let cfg = tiny_cfg();
        let mut t = OmniThinker::new(cfg.clone(), &Device::Cpu).unwrap();
        let vocab = cfg.vocab_size;
        let h = cfg.hidden_size;
        let embed: Vec<f32> = (0..vocab * h).map(|i| ((i % 13) as f32) * 0.03 - 0.2).collect();
        t.embed_tokens = Tensor::from_vec(embed, (vocab, h), &Device::Cpu).unwrap();

        let tokens: Vec<u32> = vec![3, 5, 7, 11, 9];
        let x = t.embed_with_splices(&tokens, &[]).unwrap().unsqueeze(0).unwrap();
        let mut cache = OmniKvCache::new(cfg.num_hidden_layers);
        let pos = OmniPositions::uniform(&[0, 1, 2, 3, 4]);
        let full = t.forward_step(&x, &pos, &mut cache, None).unwrap();

        let mut cache2 = OmniKvCache::new(cfg.num_hidden_layers);
        let mut last = None;
        for (step, tok) in tokens.iter().enumerate() {
            let xi = t.embed_with_splices(&[*tok], &[]).unwrap().unsqueeze(0).unwrap();
            let pi = OmniPositions::uniform(&[step as u32]);
            last = Some(t.forward_step(&xi, &pi, &mut cache2, None).unwrap());
        }
        let a: Vec<f32> = full.to_vec1().unwrap();
        let b: Vec<f32> = last.unwrap().to_vec1().unwrap();
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(diff < 1e-3, "cached decode last-token logits diverge from prefill: {diff}");
    }

    #[test]
    fn moe_topk_softmax_equals_softmax_all_then_renorm() {
        let device = Device::Cpu;
        let n_exp = 6usize;
        let k = 3usize;
        let logits_host: Vec<f32> = (0..2 * n_exp).map(|i| ((i as f32) * 0.531).sin()).collect();
        let logits = Tensor::from_vec(logits_host.clone(), (2, n_exp), &device).unwrap();
        let (sorted, idx) = logits.sort_last_dim(false).unwrap();
        let top = sorted.narrow(1, 0, k).unwrap().contiguous().unwrap();
        let top_idx: Vec<u32> = idx.narrow(1, 0, k).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let w = candle_nn::ops::softmax_last_dim(&top).unwrap();
        let w_host: Vec<f32> = w.flatten_all().unwrap().to_vec1().unwrap();
        for row in 0..2 {
            let r = &logits_host[row * n_exp..(row + 1) * n_exp];
            let max = r.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = r.iter().map(|l| (l - max).exp()).collect();
            let z: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / z).collect();
            let sel: Vec<usize> = (0..k).map(|j| top_idx[row * k + j] as usize).collect();
            let sel_sum: f32 = sel.iter().map(|&e| probs[e]).sum();
            for j in 0..k {
                let want = probs[sel[j]] / sel_sum;
                assert!(
                    (w_host[row * k + j] - want).abs() < 1e-5,
                    "row {row} slot {j}: {} vs renorm {want}",
                    w_host[row * k + j]
                );
            }
        }
    }

    #[test]
    fn interleaved_mrope_equal_positions_is_standard_rope() {
        let cfg = tiny_cfg();
        let rope = Rope::new(
            RopeConfig {
                head_dim: cfg.head_dim,
                max_seq_len: 64,
                base: cfg.rope_theta,
                kind: RopeKind::Standard,
            },
            &Device::Cpu,
        )
        .unwrap();
        let hd = cfg.head_dim;
        let vals: Vec<f32> = (0..3 * 4 * hd).map(|i| ((i as f32) * 0.17).cos()).collect();
        let q = Tensor::from_vec(vals.clone(), (1, 3, 4, hd), &Device::Cpu).unwrap();
        let k = Tensor::from_vec(vals[..3 * 2 * hd].to_vec(), (1, 3, 2, hd), &Device::Cpu).unwrap();
        let pos = OmniPositions::uniform(&[0, 1, 2]);
        let (q1, _) = apply_interleaved_mrope(&rope, &q, &k, &pos, cfg.mrope_section).unwrap();
        let positions = Tensor::from_vec(vec![0u32, 1, 2], 3usize, &Device::Cpu).unwrap();
        let (q2, _) = rope.apply(&q, &k, &positions).unwrap();
        let a: Vec<f32> = q1.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = q2.flatten_all().unwrap().to_vec1().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5, "{x} vs {y}");
        }
    }

    #[test]
    fn interleaved_mrope_axis_map_matches_hf_overwrite_semantics() {
        let sections = [24usize, 20, 20];
        for i in 0..64usize {
            let axis = axis_for_freq(i, sections);
            let want = if i >= 60 {
                0
            } else {
                match i % 3 {
                    0 => 0,
                    1 => 1,
                    _ => 2,
                }
            };
            assert_eq!(axis, want, "freq {i}");
        }
    }

    #[test]
    fn interleaved_mrope_rotation_matches_scalar_reference() {
        let head_dim = 8usize;
        let sections = [2usize, 1, 1];
        let base = 10_000f32;
        let rope = Rope::new(
            RopeConfig {
                head_dim,
                max_seq_len: 64,
                base,
                kind: RopeKind::Standard,
            },
            &Device::Cpu,
        )
        .unwrap();
        let vals: Vec<f32> = (0..head_dim).map(|i| ((i as f32) * 0.3).sin() + 0.1).collect();
        let q = Tensor::from_vec(vals.clone(), (1, 1, 1, head_dim), &Device::Cpu).unwrap();
        let pos = OmniPositions {
            t: vec![5],
            h: vec![9],
            w: vec![2],
        };
        let (q1, _) = apply_interleaved_mrope(&rope, &q, &q, &pos, sections).unwrap();
        let got: Vec<f32> = q1.flatten_all().unwrap().to_vec1().unwrap();
        let half = head_dim / 2;
        let posv = [5f32, 9.0, 2.0];
        for i in 0..half {
            let axis = axis_for_freq(i, sections);
            let inv = 1.0f32 / base.powf((i as f32 * 2.0) / head_dim as f32);
            let theta = posv[axis] * inv;
            let (s, c) = theta.sin_cos();
            let lo = vals[i];
            let hi = vals[half + i];
            let want_lo = lo * c - hi * s;
            let want_hi = lo * s + hi * c;
            assert!((got[i] - want_lo).abs() < 1e-4, "lo {i}: {} vs {want_lo}", got[i]);
            assert!((got[half + i] - want_hi).abs() < 1e-4, "hi {i}: {} vs {want_hi}", got[half + i]);
        }
    }

    #[test]
    fn build_mrope_positions_golden_image_example() {
        let ids = OmniSpecialIds::default();
        let vs = ids.vision_start;
        let ve = ids.vision_end;
        let ip = ids.image_pad;
        let tokens: Vec<u32> = vec![10, 11, vs, ip, ip, ip, ip, ve, 20, 21, ids.im_end];
        let (pos, next) = build_mrope_positions(&tokens, &ids, &[(1, 2, 2)], &[]).unwrap();
        assert_eq!(pos.t, vec![0, 1, 2, 3, 3, 3, 3, 5, 6, 7, 8]);
        assert_eq!(pos.h, vec![0, 1, 2, 3, 3, 4, 4, 5, 6, 7, 8]);
        assert_eq!(pos.w, vec![0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 8]);
        assert_eq!(next, 9);
    }

    #[test]
    fn build_mrope_positions_text_only_is_sequential() {
        let ids = OmniSpecialIds::default();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let (pos, next) = build_mrope_positions(&tokens, &ids, &[], &[]).unwrap();
        assert_eq!(pos.t, vec![0, 1, 2, 3]);
        assert_eq!(pos.h, pos.t);
        assert_eq!(pos.w, pos.t);
        assert_eq!(next, 4);
    }

    #[test]
    fn build_mrope_positions_golden_image_audio() {
        let ids = OmniSpecialIds::default();
        let (vs, ve, ip) = (ids.vision_start, ids.vision_end, ids.image_pad);
        let (as_, ae, ap) = (ids.audio_start, ids.audio_end, ids.audio_pad);
        let mut tokens = vec![10, vs];
        tokens.extend([ip; 4]);
        tokens.push(ve);
        tokens.push(30);
        tokens.push(as_);
        tokens.extend([ap; 13]);
        tokens.push(ae);
        tokens.extend([40, 41]);
        let (pos, _next) = build_mrope_positions(&tokens, &ids, &[(1, 2, 2)], &[13]).unwrap();
        assert_eq!(
            pos.t,
            vec![0, 1, 2, 2, 2, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        );
        assert_eq!(
            pos.h,
            vec![0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
        );
    }

    #[test]
    fn splice_overwrites_position_range() {
        let cfg = tiny_cfg();
        let t = OmniThinker::new(cfg.clone(), &Device::Cpu).unwrap();
        let tokens: Vec<u32> = vec![0, 0, 0, 0, 0];
        let ones = Tensor::ones((2, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let out = t
            .embed_with_splices(&tokens, &[ModalitySplice { position: 1, embedding: ones }])
            .unwrap();
        let row1: Vec<f32> = out.i(1).unwrap().to_vec1().unwrap();
        let row2: Vec<f32> = out.i(2).unwrap().to_vec1().unwrap();
        let row3: Vec<f32> = out.i(3).unwrap().to_vec1().unwrap();
        assert!(row1.iter().all(|x| (*x - 1.0).abs() < 1e-6));
        assert!(row2.iter().all(|x| (*x - 1.0).abs() < 1e-6));
        assert!(row3.iter().all(|x| *x == 0.0));
        assert_eq!(out.dims(), &[5, cfg.hidden_size]);
    }

    #[test]
    fn rejects_oversized_splice() {
        let cfg = tiny_cfg();
        let t = OmniThinker::new(cfg.clone(), &Device::Cpu).unwrap();
        let tokens: Vec<u32> = vec![0, 0, 0];
        let big = Tensor::zeros((5, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let err = t
            .embed_with_splices(&tokens, &[ModalitySplice { position: 1, embedding: big }])
            .unwrap_err();
        assert!(err.to_string().contains("exceeds seq len"));
    }

    #[test]
    fn deepstack_injection_changes_hidden_state() {
        let cfg = tiny_cfg();
        let mut t = OmniThinker::new(cfg.clone(), &Device::Cpu).unwrap();
        let vocab = cfg.vocab_size;
        let h = cfg.hidden_size;
        let embed: Vec<f32> = (0..vocab * h).map(|i| ((i % 7) as f32) * 0.1).collect();
        t.embed_tokens = Tensor::from_vec(embed, (vocab, h), &Device::Cpu).unwrap();
        let head: Vec<f32> = (0..vocab * h).map(|i| ((i % 5) as f32) * 0.2 - 0.4).collect();
        t.lm_head = Linear::new(Tensor::from_vec(head, (vocab, h), &Device::Cpu).unwrap(), None).unwrap();
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let x = t.embed_with_splices(&tokens, &[]).unwrap().unsqueeze(0).unwrap();
        let pos = OmniPositions::uniform(&[0, 1, 2, 3]);

        let mut c1 = OmniKvCache::new(cfg.num_hidden_layers);
        let base = t.forward_step(&x, &pos, &mut c1, None).unwrap();

        let rows = Tensor::from_vec(vec![3u32], 1usize, &Device::Cpu).unwrap();
        let embeds: Vec<Tensor> = (0..cfg.num_hidden_layers)
            .map(|_| Tensor::ones((1, h), DType::F32, &Device::Cpu).unwrap())
            .collect();
        let ds = OmniDeepstack { rows, embeds };
        let mut c2 = OmniKvCache::new(cfg.num_hidden_layers);
        let with = t.forward_step(&x, &pos, &mut c2, Some(&ds)).unwrap();
        let a: Vec<f32> = base.to_vec1().unwrap();
        let b: Vec<f32> = with.to_vec1().unwrap();
        let diff: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-4, "deepstack injection must change logits, diff={diff}");
    }
}
