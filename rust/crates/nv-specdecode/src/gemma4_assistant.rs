use anyhow::{anyhow, bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_weights::WeightLoader;
use crate::util::{load_linear, load_rmsnorm, load_tensor};
use serde_json::Value;
use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AssistantLayerType {
    Sliding,
    Full,
}

#[derive(Clone, Debug)]
pub struct Gemma4AssistantConfig {
    pub backbone_hidden_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub vocab_size: usize,
    pub num_centroids: usize,
    pub centroid_top_k: usize,
    pub use_ordered_embeddings: bool,
    pub rms_norm_eps: f64,
    pub sliding_window: usize,
    pub sliding_rope_theta: f32,
    pub full_rope_theta: f32,
    pub full_partial_rotary_factor: f64,
    pub layer_types: Vec<AssistantLayerType>,
    pub eos_token_ids: Vec<u32>,
}

impl Default for Gemma4AssistantConfig {
    fn default() -> Self {
        Self {
            backbone_hidden_size: 2560,
            hidden_size: 256,
            intermediate_size: 2048,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            head_dim: 256,
            global_head_dim: 512,
            vocab_size: 262144,
            num_centroids: 2048,
            centroid_top_k: 32,
            use_ordered_embeddings: true,
            rms_norm_eps: 1e-6,
            sliding_window: 512,
            sliding_rope_theta: 10000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            layer_types: vec![
                AssistantLayerType::Sliding,
                AssistantLayerType::Sliding,
                AssistantLayerType::Sliding,
                AssistantLayerType::Full,
            ],
            eos_token_ids: vec![1, 106],
        }
    }
}

impl Gemma4AssistantConfig {
    pub fn from_hf_json_str(s: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(s).context("parse gemma4_assistant config json")?;
        let model_type = v.get("model_type").and_then(|x| x.as_str()).unwrap_or("");
        if model_type != "gemma4_assistant" {
            bail!("expected model_type gemma4_assistant, got {model_type:?}");
        }
        let tc = v
            .get("text_config")
            .ok_or_else(|| anyhow!("missing text_config"))?;
        let usize_field = |obj: &Value, key: &str| -> Result<usize> {
            obj.get(key)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| anyhow!("missing {key}"))
        };

        let mut cfg = Self::default();
        cfg.backbone_hidden_size = usize_field(&v, "backbone_hidden_size")?;
        cfg.num_centroids = usize_field(&v, "num_centroids")?;
        cfg.centroid_top_k = usize_field(&v, "centroid_intermediate_top_k")?;
        cfg.use_ordered_embeddings = v
            .get("use_ordered_embeddings")
            .and_then(|x| x.as_bool())
            .unwrap_or(true);

        cfg.hidden_size = usize_field(tc, "hidden_size")?;
        cfg.intermediate_size = usize_field(tc, "intermediate_size")?;
        cfg.num_hidden_layers = usize_field(tc, "num_hidden_layers")?;
        cfg.num_attention_heads = usize_field(tc, "num_attention_heads")?;
        cfg.head_dim = usize_field(tc, "head_dim")?;
        cfg.global_head_dim = tc
            .get("global_head_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(cfg.head_dim);
        cfg.vocab_size = usize_field(tc, "vocab_size")?;
        cfg.sliding_window = usize_field(tc, "sliding_window")?;
        if let Some(eps) = tc.get("rms_norm_eps").and_then(|x| x.as_f64()) {
            cfg.rms_norm_eps = eps;
        }

        let lt = tc
            .get("layer_types")
            .and_then(|x| x.as_array())
            .ok_or_else(|| anyhow!("missing layer_types"))?;
        cfg.layer_types = lt
            .iter()
            .map(|x| match x.as_str() {
                Some("sliding_attention") => Ok(AssistantLayerType::Sliding),
                Some("full_attention") => Ok(AssistantLayerType::Full),
                other => Err(anyhow!("unknown layer_type {other:?}")),
            })
            .collect::<Result<Vec<_>>>()?;
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            bail!(
                "layer_types len {} != num_hidden_layers {}",
                cfg.layer_types.len(),
                cfg.num_hidden_layers
            );
        }

        if let Some(rp) = tc.get("rope_parameters") {
            if let Some(theta) = rp
                .get("sliding_attention")
                .and_then(|x| x.get("rope_theta"))
                .and_then(|x| x.as_f64())
            {
                cfg.sliding_rope_theta = theta as f32;
            }
            if let Some(fa) = rp.get("full_attention") {
                if let Some(theta) = fa.get("rope_theta").and_then(|x| x.as_f64()) {
                    cfg.full_rope_theta = theta as f32;
                }
                if let Some(f) = fa.get("partial_rotary_factor").and_then(|x| x.as_f64()) {
                    cfg.full_partial_rotary_factor = f;
                }
            }
        }

        if let Some(eos) = v.get("eos_token_id") {
            cfg.eos_token_ids = match eos {
                Value::Number(n) => n.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
                Value::Array(a) => a
                    .iter()
                    .filter_map(|x| x.as_u64().map(|v| v as u32))
                    .collect(),
                _ => cfg.eos_token_ids.clone(),
            };
        }

        if cfg.vocab_size % cfg.num_centroids != 0 {
            bail!(
                "vocab_size {} not divisible by num_centroids {}",
                cfg.vocab_size,
                cfg.num_centroids
            );
        }
        Ok(cfg)
    }

    nv_weights::hf_json_from_file!(from_hf_json_file, from_hf_json_str);

    pub fn vocab_per_centroid(&self) -> usize {
        self.vocab_size / self.num_centroids
    }

    pub fn head_dim_for(&self, lt: AssistantLayerType) -> usize {
        match lt {
            AssistantLayerType::Sliding => self.head_dim,
            AssistantLayerType::Full => self.global_head_dim,
        }
    }
}

pub trait SharedKvSource {
    fn shared_kv(&self, layer_type: AssistantLayerType) -> Result<(Tensor, Tensor)>;
}

pub trait BackboneEmbedder {
    fn embed_scaled(&self, token: u32) -> Result<Tensor>;
}

pub struct FixedSharedKv {
    pub sliding: (Tensor, Tensor),
    pub full: (Tensor, Tensor),
}

impl SharedKvSource for FixedSharedKv {
    fn shared_kv(&self, layer_type: AssistantLayerType) -> Result<(Tensor, Tensor)> {
        let (k, v) = match layer_type {
            AssistantLayerType::Sliding => &self.sliding,
            AssistantLayerType::Full => &self.full,
        };
        Ok((k.clone(), v.clone()))
    }
}

impl<F> BackboneEmbedder for F
where
    F: Fn(u32) -> Result<Tensor>,
{
    fn embed_scaled(&self, token: u32) -> Result<Tensor> {
        (self)(token)
    }
}

struct AssistantLayer {
    layer_type: AssistantLayerType,
    head_dim: usize,
    q_proj: Linear,
    q_norm: RmsNorm,
    o_proj: Linear,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    input_ln: RmsNorm,
    post_attn_ln: RmsNorm,
    pre_ff_ln: RmsNorm,
    post_ff_ln: RmsNorm,
    layer_scalar: f64,
}

pub struct Gemma4AssistantDrafter {
    cfg: Gemma4AssistantConfig,
    device: Device,
    dtype: DType,
    pre_projection: Linear,
    post_projection: Linear,
    layers: Vec<AssistantLayer>,
    norm: RmsNorm,
    lm_head: Tensor,
    centroids: Linear,
    token_ordering: Vec<u32>,
    sliding_inv_freq: Vec<f32>,
    full_inv_freq: Vec<f32>,
}

pub struct DraftStep {
    pub token: u32,
    pub next_hidden: Tensor,
    pub candidate_logit: f32,
}

impl Gemma4AssistantDrafter {
    pub fn config(&self) -> &Gemma4AssistantConfig {
        &self.cfg
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn try_load(model_dir: &Path, device: &Device) -> Result<Self> {
        let cfg = Gemma4AssistantConfig::from_hf_json_file(&model_dir.join("config.json"))?;
        let st = model_dir.join("model.safetensors");
        if !st.is_file() {
            bail!("missing model.safetensors at {}", st.display());
        }
        Self::load_from_safetensors(&cfg, &st, device)
    }

    pub fn load_from_safetensors(
        cfg: &Gemma4AssistantConfig,
        safetensors_path: &Path,
        device: &Device,
    ) -> Result<Self> {
        let weights = WeightLoader::open_file(safetensors_path, device)
            .with_context(|| format!("open {}", safetensors_path.display()))?;
        let dtype = model_dtype(device);
        let h = cfg.hidden_size;

        let pre_projection = load_linear(
            &weights,
            "pre_projection.weight",
            h,
            2 * cfg.backbone_hidden_size,
            dtype,
        )?;
        let post_projection = load_linear(
            &weights,
            "post_projection.weight",
            cfg.backbone_hidden_size,
            h,
            dtype,
        )?;
        let lm_head = load_tensor(
            &weights,
            "model.embed_tokens.weight",
            &[cfg.vocab_size, h],
            dtype,
        )?;
        let centroids = load_linear(
            &weights,
            "masked_embedding.centroids.weight",
            cfg.num_centroids,
            h,
            dtype,
        )?;
        let token_ordering =
            load_token_ordering(&weights, "masked_embedding.token_ordering", cfg.vocab_size)?;
        let norm = load_rmsnorm(&weights, "model.norm.weight", h, cfg.rms_norm_eps, dtype)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, &lt) in cfg.layer_types.iter().enumerate() {
            let p = format!("model.layers.{i}");
            let hd = cfg.head_dim_for(lt);
            let q_dim = cfg.num_attention_heads * hd;
            let layer = AssistantLayer {
                layer_type: lt,
                head_dim: hd,
                q_proj: load_linear(
                    &weights,
                    &format!("{p}.self_attn.q_proj.weight"),
                    q_dim,
                    h,
                    dtype,
                )?,
                q_norm: load_rmsnorm(
                    &weights,
                    &format!("{p}.self_attn.q_norm.weight"),
                    hd,
                    cfg.rms_norm_eps,
                    dtype,
                )?,
                o_proj: load_linear(
                    &weights,
                    &format!("{p}.self_attn.o_proj.weight"),
                    h,
                    q_dim,
                    dtype,
                )?,
                gate_proj: load_linear(
                    &weights,
                    &format!("{p}.mlp.gate_proj.weight"),
                    cfg.intermediate_size,
                    h,
                    dtype,
                )?,
                up_proj: load_linear(
                    &weights,
                    &format!("{p}.mlp.up_proj.weight"),
                    cfg.intermediate_size,
                    h,
                    dtype,
                )?,
                down_proj: load_linear(
                    &weights,
                    &format!("{p}.mlp.down_proj.weight"),
                    h,
                    cfg.intermediate_size,
                    dtype,
                )?,
                input_ln: load_rmsnorm(
                    &weights,
                    &format!("{p}.input_layernorm.weight"),
                    h,
                    cfg.rms_norm_eps,
                    dtype,
                )?,
                post_attn_ln: load_rmsnorm(
                    &weights,
                    &format!("{p}.post_attention_layernorm.weight"),
                    h,
                    cfg.rms_norm_eps,
                    dtype,
                )?,
                pre_ff_ln: load_rmsnorm(
                    &weights,
                    &format!("{p}.pre_feedforward_layernorm.weight"),
                    h,
                    cfg.rms_norm_eps,
                    dtype,
                )?,
                post_ff_ln: load_rmsnorm(
                    &weights,
                    &format!("{p}.post_feedforward_layernorm.weight"),
                    h,
                    cfg.rms_norm_eps,
                    dtype,
                )?,
                layer_scalar: load_scalar(&weights, &format!("{p}.layer_scalar"))?,
            };
            layers.push(layer);
        }

        Self::from_parts(
            cfg.clone(),
            device.clone(),
            dtype,
            pre_projection,
            post_projection,
            layers,
            norm,
            lm_head,
            centroids,
            token_ordering,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        cfg: Gemma4AssistantConfig,
        device: Device,
        dtype: DType,
        pre_projection: Linear,
        post_projection: Linear,
        layers: Vec<AssistantLayer>,
        norm: RmsNorm,
        lm_head: Tensor,
        centroids: Linear,
        token_ordering: Vec<u32>,
    ) -> Result<Self> {
        if token_ordering.len() != cfg.vocab_size {
            bail!(
                "token_ordering len {} != vocab_size {}",
                token_ordering.len(),
                cfg.vocab_size
            );
        }
        let sliding_inv_freq = default_inv_freq(cfg.head_dim, cfg.sliding_rope_theta);
        let full_inv_freq = proportional_inv_freq(
            cfg.global_head_dim,
            cfg.full_rope_theta,
            cfg.full_partial_rotary_factor,
        );
        Ok(Self {
            cfg,
            device,
            dtype,
            pre_projection,
            post_projection,
            layers,
            norm,
            lm_head,
            centroids,
            token_ordering,
            sliding_inv_freq,
            full_inv_freq,
        })
    }

    pub fn draft_step(
        &self,
        token_embed: &Tensor,
        last_hidden: &Tensor,
        position: usize,
        kv: &dyn SharedKvSource,
    ) -> Result<DraftStep> {
        let bh = self.cfg.backbone_hidden_size;
        let te = token_embed.reshape(bh)?.to_dtype(self.dtype)?;
        let lh = last_hidden.reshape(bh)?.to_dtype(self.dtype)?;
        let x = Tensor::cat(&[&te, &lh], 0)?.reshape((1, 2 * bh))?;
        let mut x = self.pre_projection.forward(&x)?;

        for layer in &self.layers {
            x = self.layer_forward(layer, &x, position, kv)?;
        }

        let h = self.norm.forward(&x)?;
        let next_hidden = self.post_projection.forward(&h)?.reshape(bh)?;
        let (token, candidate_logit) = self.head_argmax(&h)?;
        Ok(DraftStep {
            token,
            next_hidden,
            candidate_logit,
        })
    }

    pub fn propose(
        &self,
        last_token: u32,
        prefix_last_hidden: &Tensor,
        position: usize,
        k: usize,
        embedder: &dyn BackboneEmbedder,
        kv: &dyn SharedKvSource,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(k);
        let mut token = last_token;
        let mut hidden = prefix_last_hidden.clone();
        for _ in 0..k {
            let embed = embedder.embed_scaled(token)?;
            let step = self.draft_step(&embed, &hidden, position, kv)?;
            out.push(step.token);
            token = step.token;
            hidden = step.next_hidden;
            if self.cfg.eos_token_ids.contains(&step.token) {
                break;
            }
        }
        Ok(out)
    }

    fn layer_forward(
        &self,
        layer: &AssistantLayer,
        x: &Tensor,
        position: usize,
        kv: &dyn SharedKvSource,
    ) -> Result<Tensor> {
        let residual = x.clone();
        let h = layer.input_ln.forward(x)?;
        let attn = self.attention(layer, &h, position, kv)?;
        let h = layer.post_attn_ln.forward(&attn)?;
        let x = residual.add(&h)?;

        let residual = x.clone();
        let h = layer.pre_ff_ln.forward(&x)?;
        let gate = layer.gate_proj.forward(&h)?.gelu()?;
        let up = layer.up_proj.forward(&h)?;
        let h = layer.down_proj.forward(&gate.mul(&up)?)?;
        let h = layer.post_ff_ln.forward(&h)?;
        let x = residual.add(&h)?;

        Ok(x.affine(layer.layer_scalar, 0.0)?)
    }

    fn attention(
        &self,
        layer: &AssistantLayer,
        h: &Tensor,
        position: usize,
        kv: &dyn SharedKvSource,
    ) -> Result<Tensor> {
        let n_heads = self.cfg.num_attention_heads;
        let hd = layer.head_dim;

        let q = layer.q_proj.forward(h)?.reshape((n_heads, hd))?;
        let q = layer.q_norm.forward(&q)?.to_dtype(DType::F32)?;
        let q = self.apply_rope(&q, layer.layer_type, position)?;

        let (k, v) = kv.shared_kv(layer.layer_type)?;
        let k = normalize_kv(&k, hd)?.to_dtype(DType::F32)?;
        let v = normalize_kv(&v, hd)?.to_dtype(DType::F32)?;
        let (n_kv, kv_len, k_hd) = k.dims3()?;
        let (v_n_kv, v_kv_len, v_hd) = v.dims3()?;
        if (n_kv, kv_len, k_hd) != (v_n_kv, v_kv_len, v_hd) {
            bail!("shared K {:?} and V {:?} disagree", k.dims(), v.dims());
        }
        if k_hd != hd {
            bail!("shared KV head_dim {k_hd} != layer head_dim {hd}");
        }
        if !n_heads.is_multiple_of(n_kv) {
            bail!("num_heads {n_heads} not divisible by kv heads {n_kv}");
        }
        if kv_len == 0 {
            bail!("shared KV has zero positions");
        }

        let (k, v) = if layer.layer_type == AssistantLayerType::Sliding
            && kv_len > self.cfg.sliding_window
        {
            let start = kv_len - self.cfg.sliding_window;
            (
                k.narrow(1, start, self.cfg.sliding_window)?,
                v.narrow(1, start, self.cfg.sliding_window)?,
            )
        } else {
            (k, v)
        };

        let groups = n_heads / n_kv;
        let q = q.reshape((n_kv, groups, hd))?;
        let scores = q.matmul(&k.transpose(1, 2)?.contiguous()?)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v.contiguous()?)?;
        let ctx = ctx.reshape((1, n_heads * hd))?.to_dtype(self.dtype)?;
        layer.o_proj.forward(&ctx)
    }

    fn apply_rope(
        &self,
        q: &Tensor,
        layer_type: AssistantLayerType,
        position: usize,
    ) -> Result<Tensor> {
        let inv_freq = match layer_type {
            AssistantLayerType::Sliding => &self.sliding_inv_freq,
            AssistantLayerType::Full => &self.full_inv_freq,
        };
        let hd = inv_freq.len() * 2;
        let mut cos = Vec::with_capacity(hd);
        let mut sin = Vec::with_capacity(hd);
        for _ in 0..2 {
            for &f in inv_freq {
                let ang = position as f32 * f;
                cos.push(ang.cos());
                sin.push(ang.sin());
            }
        }
        let cos = Tensor::from_vec(cos, hd, &self.device)?;
        let sin = Tensor::from_vec(sin, hd, &self.device)?;
        let half = hd / 2;
        let x1 = q.narrow(D::Minus1, 0, half)?;
        let x2 = q.narrow(D::Minus1, half, half)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        Ok(q.broadcast_mul(&cos)?.add(&rotated.broadcast_mul(&sin)?)?)
    }

    fn head_argmax(&self, h: &Tensor) -> Result<(u32, f32)> {
        if !self.cfg.use_ordered_embeddings {
            let logits = h.matmul(&self.lm_head.t()?)?;
            let row: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let (idx, val) = crate::util::argmax_f32(&row);
            return Ok((idx as u32, val));
        }

        let cl = self.centroids.forward(h)?;
        let cl: Vec<f32> = cl.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let top = crate::util::top_k_indices(&cl, self.cfg.centroid_top_k);

        let vpc = self.cfg.vocab_per_centroid();
        let mut cand_ids: Vec<u32> = Vec::with_capacity(top.len() * vpc);
        for &c in &top {
            cand_ids.extend_from_slice(&self.token_ordering[c * vpc..(c + 1) * vpc]);
        }
        let ids = Tensor::from_vec(cand_ids.clone(), cand_ids.len(), &self.device)?;
        let rows = self.lm_head.index_select(&ids, 0)?;
        let logits = rows.matmul(&h.t()?)?;
        let vals: Vec<f32> = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let (idx, val) = crate::util::argmax_f32(&vals);
        Ok((cand_ids[idx], val))
    }
}

fn model_dtype(device: &Device) -> DType {
    if device.is_cuda() {
        DType::BF16
    } else {
        DType::F32
    }
}

fn normalize_kv(t: &Tensor, _hd: usize) -> Result<Tensor> {
    let t = match t.dims().len() {
        4 => {
            if t.dims()[0] != 1 {
                bail!("expected batch-1 shared KV, got {:?}", t.dims());
            }
            t.squeeze(0)?
        }
        3 => t.clone(),
        other => bail!("shared KV must be rank 3 or 4, got rank {other}"),
    };
    Ok(t)
}

fn default_inv_freq(head_dim: usize, theta: f32) -> Vec<f32> {
    (0..head_dim / 2)
        .map(|i| 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect()
}

fn proportional_inv_freq(head_dim: usize, theta: f32, partial_rotary_factor: f64) -> Vec<f32> {
    let rope_angles = ((partial_rotary_factor * head_dim as f64) as usize) / 2;
    let mut out = Vec::with_capacity(head_dim / 2);
    for i in 0..rope_angles {
        out.push(1.0 / theta.powf(2.0 * i as f32 / head_dim as f32));
    }
    out.resize(head_dim / 2, 0.0);
    out
}

fn load_scalar(weights: &WeightLoader, name: &str) -> Result<f64> {
    let t = load_tensor(weights, name, &[1], DType::F32)?;
    let v: Vec<f32> = t.to_vec1()?;
    Ok(v[0] as f64)
}

fn load_token_ordering(weights: &WeightLoader, name: &str, vocab: usize) -> Result<Vec<u32>> {
    if !weights.has(name) {
        bail!("missing tensor {name}");
    }
    let shape = weights
        .shape_of(name)
        .ok_or_else(|| anyhow!("no shape for {name}"))?;
    if shape != [vocab] {
        bail!("{name}: expected [{vocab}], got {shape:?}");
    }
    let t = weights.get(name, DType::I64)?;
    let vals: Vec<i64> = t.to_vec1()?;
    let mut out = Vec::with_capacity(vocab);
    for v in vals {
        if v < 0 || v as usize >= vocab {
            bail!("{name}: value {v} out of range [0, {vocab})");
        }
        out.push(v as u32);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_cfg() -> Gemma4AssistantConfig {
        Gemma4AssistantConfig {
            backbone_hidden_size: 32,
            hidden_size: 16,
            intermediate_size: 24,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            head_dim: 8,
            global_head_dim: 16,
            vocab_size: 64,
            num_centroids: 8,
            centroid_top_k: 2,
            use_ordered_embeddings: true,
            rms_norm_eps: 1e-6,
            sliding_window: 4,
            sliding_rope_theta: 10000.0,
            full_rope_theta: 1_000_000.0,
            full_partial_rotary_factor: 0.25,
            layer_types: vec![AssistantLayerType::Sliding, AssistantLayerType::Full],
            eos_token_ids: vec![1],
        }
    }

    fn mk_linear(out: usize, inp: usize, dev: &Device, seed: f32) -> Result<Linear> {
        let data: Vec<f32> = (0..out * inp)
            .map(|i| ((i as f32) * 0.017 + seed).sin() * 0.05)
            .collect();
        Linear::new(Tensor::from_vec(data, (out, inp), dev)?, None)
    }

    fn mk_rms(dim: usize, dev: &Device, eps: f64) -> Result<RmsNorm> {
        let data: Vec<f32> = (0..dim).map(|_| 1.0f32).collect();
        Ok(RmsNorm::new(Tensor::from_vec(data, dim, dev)?, eps))
    }

    fn synth_drafter(dev: &Device) -> Result<Gemma4AssistantDrafter> {
        let cfg = synth_cfg();
        let h = cfg.hidden_size;
        let dtype = model_dtype(dev);

        let mut layers = Vec::new();
        for (i, &lt) in cfg.layer_types.iter().enumerate() {
            let hd = cfg.head_dim_for(lt);
            let q_dim = cfg.num_attention_heads * hd;
            layers.push(AssistantLayer {
                layer_type: lt,
                head_dim: hd,
                q_proj: mk_linear(q_dim, h, dev, i as f32)?,
                q_norm: mk_rms(hd, dev, cfg.rms_norm_eps)?,
                o_proj: mk_linear(h, q_dim, dev, i as f32 + 0.1)?,
                gate_proj: mk_linear(cfg.intermediate_size, h, dev, i as f32 + 0.2)?,
                up_proj: mk_linear(cfg.intermediate_size, h, dev, i as f32 + 0.3)?,
                down_proj: mk_linear(h, cfg.intermediate_size, dev, i as f32 + 0.4)?,
                input_ln: mk_rms(h, dev, cfg.rms_norm_eps)?,
                post_attn_ln: mk_rms(h, dev, cfg.rms_norm_eps)?,
                pre_ff_ln: mk_rms(h, dev, cfg.rms_norm_eps)?,
                post_ff_ln: mk_rms(h, dev, cfg.rms_norm_eps)?,
                layer_scalar: 1.0,
            });
        }

        let lm_data: Vec<f32> = (0..cfg.vocab_size * h)
            .map(|i| ((i as f32) * 0.013).cos() * 0.05)
            .collect();
        let lm_head = Tensor::from_vec(lm_data, (cfg.vocab_size, h), dev)?;
        let token_ordering: Vec<u32> = (0..cfg.vocab_size as u32).collect();

        Gemma4AssistantDrafter::from_parts(
            cfg.clone(),
            dev.clone(),
            dtype,
            mk_linear(h, 2 * cfg.backbone_hidden_size, dev, 7.0)?,
            mk_linear(cfg.backbone_hidden_size, h, dev, 8.0)?,
            layers,
            mk_rms(h, dev, cfg.rms_norm_eps)?,
            lm_head,
            mk_linear(cfg.num_centroids, h, dev, 9.0)?,
            token_ordering,
        )
    }

    fn synth_kv(cfg: &Gemma4AssistantConfig, kv_len: usize, dev: &Device) -> Result<FixedSharedKv> {
        let mk = |n_kv: usize, hd: usize, seed: f32| -> Result<Tensor> {
            let data: Vec<f32> = (0..n_kv * kv_len * hd)
                .map(|i| ((i as f32) * 0.011 + seed).sin() * 0.1)
                .collect();
            Ok(Tensor::from_vec(data, (n_kv, kv_len, hd), dev)?)
        };
        Ok(FixedSharedKv {
            sliding: (mk(1, cfg.head_dim, 0.0)?, mk(1, cfg.head_dim, 1.0)?),
            full: (
                mk(2, cfg.global_head_dim, 2.0)?,
                mk(2, cfg.global_head_dim, 3.0)?,
            ),
        })
    }

    #[test]
    fn synthetic_draft_step_shapes() {
        let dev = Device::Cpu;
        let drafter = synth_drafter(&dev).expect("build");
        let cfg = drafter.config().clone();
        let kv = synth_kv(&cfg, 6, &dev).expect("kv");

        let embed = Tensor::zeros(cfg.backbone_hidden_size, DType::F32, &dev).unwrap();
        let hidden_data: Vec<f32> = (0..cfg.backbone_hidden_size)
            .map(|i| (i as f32 * 0.3).sin())
            .collect();
        let hidden = Tensor::from_vec(hidden_data, cfg.backbone_hidden_size, &dev).unwrap();

        let step = drafter
            .draft_step(&embed, &hidden, 5, &kv)
            .expect("draft_step");
        assert!((step.token as usize) < cfg.vocab_size);
        assert_eq!(step.next_hidden.dims(), &[cfg.backbone_hidden_size]);
        assert!(step.candidate_logit.is_finite());
        let nh: Vec<f32> = step
            .next_hidden
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(nh.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn synthetic_propose_chains() {
        let dev = Device::Cpu;
        let drafter = synth_drafter(&dev).expect("build");
        let cfg = drafter.config().clone();
        let kv = synth_kv(&cfg, 10, &dev).expect("kv");
        let bh = cfg.backbone_hidden_size;

        let embedder = move |tok: u32| -> Result<Tensor> {
            let data: Vec<f32> = (0..bh)
                .map(|i| ((i + tok as usize) as f32 * 0.07).sin())
                .collect();
            Ok(Tensor::from_vec(data, bh, &Device::Cpu)?)
        };
        let hidden = Tensor::ones(bh, DType::F32, &dev).unwrap();
        let out = drafter
            .propose(3, &hidden, 9, 4, &embedder, &kv)
            .expect("propose");
        assert!(!out.is_empty());
        assert!(out.len() <= 4);
        for &t in &out {
            assert!((t as usize) < cfg.vocab_size);
        }
    }

    #[test]
    fn sliding_window_narrows_long_kv() {
        let dev = Device::Cpu;
        let drafter = synth_drafter(&dev).expect("build");
        let cfg = drafter.config().clone();
        let kv = synth_kv(&cfg, 9, &dev).expect("kv");
        let embed = Tensor::zeros(cfg.backbone_hidden_size, DType::F32, &dev).unwrap();
        let hidden = Tensor::ones(cfg.backbone_hidden_size, DType::F32, &dev).unwrap();
        let step = drafter
            .draft_step(&embed, &hidden, 8, &kv)
            .expect("draft_step");
        assert!(step.candidate_logit.is_finite());
    }

    #[test]
    fn proportional_inv_freq_pads_with_zeros() {
        let f = proportional_inv_freq(512, 1_000_000.0, 0.25);
        assert_eq!(f.len(), 256);
        assert!(f[0] == 1.0);
        assert!(f[63] > 0.0);
        assert!(f[64] == 0.0);
        assert!(f[255] == 0.0);
    }

    #[test]
    fn config_parses_real_json() {
        let json = r#"{
            "architectures": ["Gemma4AssistantForCausalLM"],
            "model_type": "gemma4_assistant",
            "backbone_hidden_size": 2560,
            "num_centroids": 2048,
            "centroid_intermediate_top_k": 32,
            "use_ordered_embeddings": true,
            "eos_token_id": [1, 106],
            "text_config": {
                "hidden_size": 256,
                "intermediate_size": 2048,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "global_head_dim": 512,
                "num_kv_shared_layers": 4,
                "vocab_size": 262144,
                "sliding_window": 512,
                "rms_norm_eps": 1e-06,
                "layer_types": [
                    "sliding_attention",
                    "sliding_attention",
                    "sliding_attention",
                    "full_attention"
                ],
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0,
                        "rope_type": "proportional"
                    },
                    "sliding_attention": {
                        "rope_theta": 10000.0,
                        "rope_type": "default"
                    }
                }
            }
        }"#;
        let cfg = Gemma4AssistantConfig::from_hf_json_str(json).expect("parse");
        assert_eq!(cfg.backbone_hidden_size, 2560);
        assert_eq!(cfg.hidden_size, 256);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(cfg.global_head_dim, 512);
        assert_eq!(cfg.num_centroids, 2048);
        assert_eq!(cfg.centroid_top_k, 32);
        assert_eq!(cfg.vocab_per_centroid(), 128);
        assert_eq!(
            cfg.layer_types,
            vec![
                AssistantLayerType::Sliding,
                AssistantLayerType::Sliding,
                AssistantLayerType::Sliding,
                AssistantLayerType::Full
            ]
        );
        assert_eq!(cfg.eos_token_ids, vec![1, 106]);
    }

    const ALLOW_SKIP: &str = "NV_SPECDECODE_ALLOW_SKIP";

    fn require<T>(test: &str, what: &str, found: Option<T>) -> Option<T> {
        if found.is_none() {
            if std::env::var(ALLOW_SKIP).as_deref() != Ok("1") {
                panic!(
                    "{test}: no {what}. This test is #[ignore]d, so it runs only when asked for \
                     by name; reporting a pass without the artifact answers a question that was \
                     never put. Provide it or set {ALLOW_SKIP}=1."
                );
            }
            eprintln!("SKIP ({ALLOW_SKIP}=1): {test}: no {what}; nothing was exercised");
        }
        found
    }

    fn cached_snapshot_dir() -> Option<std::path::PathBuf> {
        let home = std::env::var("HOME").ok()?;
        let root = std::path::PathBuf::from(home).join(
            ".cache/huggingface/hub/\
             models--google--gemma-4-E4B-it-qat-q4_0-unquantized-assistant/snapshots",
        );
        for entry in std::fs::read_dir(&root).ok()? {
            let p = entry.ok()?.path();
            if p.is_dir() && p.join("model.safetensors").is_file() {
                return Some(p);
            }
        }
        None
    }

    #[test]
    #[ignore]
    fn loads_real_checkpoint_cpu() {
        let Some(dir) = require(
            "loads_real_checkpoint_cpu",
            "gemma-4 assistant snapshot in the HF cache",
            cached_snapshot_dir(),
        ) else {
            return;
        };
        let drafter =
            Gemma4AssistantDrafter::try_load(&dir, &Device::Cpu).expect("load real checkpoint");
        let cfg = drafter.config();
        assert_eq!(cfg.backbone_hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 4);
        assert_eq!(drafter.token_ordering.len(), 262144);

        let kv = FixedSharedKv {
            sliding: (
                Tensor::zeros((2, 8, 256), DType::F32, &Device::Cpu).unwrap(),
                Tensor::zeros((2, 8, 256), DType::F32, &Device::Cpu).unwrap(),
            ),
            full: (
                Tensor::zeros((2, 8, 512), DType::F32, &Device::Cpu).unwrap(),
                Tensor::zeros((2, 8, 512), DType::F32, &Device::Cpu).unwrap(),
            ),
        };
        let embed = Tensor::zeros(2560, DType::F32, &Device::Cpu).unwrap();
        let hidden_data: Vec<f32> = (0..2560).map(|i| (i as f32 * 0.01).sin()).collect();
        let hidden = Tensor::from_vec(hidden_data, 2560, &Device::Cpu).unwrap();
        let step = drafter
            .draft_step(&embed, &hidden, 7, &kv)
            .expect("draft_step");
        assert!((step.token as usize) < 262144);
        assert!(step.candidate_logit.is_finite());
    }
}
