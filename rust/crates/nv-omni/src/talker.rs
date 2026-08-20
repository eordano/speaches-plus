use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use nv_layers::attn::{sdpa, AttnConfig};
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};

#[derive(Clone, Debug)]
pub struct TalkerConfig {
    pub hidden_size: usize,

    pub thinker_hidden_size: usize,

    pub speech_vocab_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,

    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub dtype: DType,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            thinker_hidden_size: 4096,
            speech_vocab_size: 4096 + 16,
            num_layers: 2,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 64,
            intermediate_size: 4096,
            max_position_embeddings: 2048,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            dtype: DType::BF16,
        }
    }
}

impl TalkerConfig {
    pub fn q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }
    pub fn kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }
    pub fn input_proj_is_identity(&self) -> bool {
        self.hidden_size == self.thinker_hidden_size
    }
}

struct TalkerLayer {
    pre_attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    pre_mlp_norm: RmsNorm,
    mlp: Mlp,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl TalkerLayer {
    fn new(cfg: &TalkerConfig, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let q = cfg.q_dim();
        let kv = cfg.kv_dim();
        let ffn = cfg.intermediate_size;
        let dtype = cfg.dtype;

        let pre_attn_norm = RmsNorm::new(Tensor::ones(h, dtype, device)?, cfg.rms_norm_eps);
        let pre_mlp_norm = RmsNorm::new(Tensor::ones(h, dtype, device)?, cfg.rms_norm_eps);

        let q_proj = Linear::new(Tensor::zeros((q, h), dtype, device)?, None)?;
        let k_proj = Linear::new(Tensor::zeros((kv, h), dtype, device)?, None)?;
        let v_proj = Linear::new(Tensor::zeros((kv, h), dtype, device)?, None)?;
        let o_proj = Linear::new(Tensor::zeros((h, q), dtype, device)?, None)?;

        let gate_proj = Linear::new(Tensor::zeros((ffn, h), dtype, device)?, None)?;
        let up_proj = Linear::new(Tensor::zeros((ffn, h), dtype, device)?, None)?;
        let down_proj = Linear::new(Tensor::zeros((h, ffn), dtype, device)?, None)?;
        let mlp = Mlp::new(gate_proj, up_proj, down_proj)?;

        Ok(Self {
            pre_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            pre_mlp_norm,
            mlp,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
        })
    }

    fn forward(&self, x: &Tensor, positions: &Tensor, rope: &Rope) -> Result<Tensor> {
        let (b, t, _h) = x.dims3().map_err(|e| anyhow::anyhow!(e))?;
        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;

        let normed = self.pre_attn_norm.forward(x)?;
        let q = self.q_proj.forward(&normed)?.reshape((b, t, nh, hd))?;
        let k = self.k_proj.forward(&normed)?.reshape((b, t, nkv, hd))?;
        let v = self.v_proj.forward(&normed)?.reshape((b, t, nkv, hd))?;
        let (q, k) = rope.apply(&q, &k, positions)?;

        let attn_cfg = AttnConfig {
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            causal: true,
        };
        let attn_out = sdpa(&q, &k, &v, &attn_cfg)?;
        let attn_out = attn_out
            .reshape((b, t, nh * hd))
            .map_err(|e| anyhow::anyhow!(e))?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        let x = (x + attn_out).map_err(|e| anyhow::anyhow!(e))?;

        let normed = self.pre_mlp_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        (x + mlp_out).map_err(|e| anyhow::anyhow!(e))
    }
}

pub struct Talker {
    cfg: TalkerConfig,

    input_proj: Option<Linear>,
    embed_tokens: Tensor,
    layers: Vec<TalkerLayer>,
    norm: RmsNorm,
    head: Linear,
    rope: Rope,
    device: Device,
}

impl Talker {
    pub fn new(cfg: TalkerConfig, device: &Device) -> Result<Self> {
        if cfg.hidden_size != cfg.num_attention_heads * cfg.head_dim {
            anyhow::bail!(
                "TalkerConfig: hidden_size {} != num_attention_heads {} * head_dim {}",
                cfg.hidden_size,
                cfg.num_attention_heads,
                cfg.head_dim
            );
        }
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            anyhow::bail!(
                "TalkerConfig: num_attention_heads {} not divisible by num_key_value_heads {}",
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        if !cfg.head_dim.is_multiple_of(2) {
            anyhow::bail!(
                "TalkerConfig: head_dim {} must be even (RoPE)",
                cfg.head_dim
            );
        }

        let input_proj = if cfg.input_proj_is_identity() {
            None
        } else {
            Some(Linear::new(
                Tensor::zeros(
                    (cfg.hidden_size, cfg.thinker_hidden_size),
                    cfg.dtype,
                    device,
                )?,
                None,
            )?)
        };

        let embed_tokens =
            Tensor::zeros((cfg.speech_vocab_size, cfg.hidden_size), cfg.dtype, device)?;

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for _ in 0..cfg.num_layers {
            layers.push(TalkerLayer::new(&cfg, device)?);
        }
        let norm = RmsNorm::new(
            Tensor::ones(cfg.hidden_size, cfg.dtype, device)?,
            cfg.rms_norm_eps,
        );
        let head = Linear::new(
            Tensor::zeros((cfg.speech_vocab_size, cfg.hidden_size), cfg.dtype, device)?,
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
            input_proj,
            embed_tokens,
            layers,
            norm,
            head,
            rope,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &TalkerConfig {
        &self.cfg
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn set_head(&mut self, head: Linear) {
        self.head = head;
    }

    fn project_thinker(&self, thinker_hidden: &Tensor) -> Result<Tensor> {
        let dims = thinker_hidden.dims();
        if dims.len() != 2 {
            anyhow::bail!(
                "Talker::step: thinker_hidden must be rank-2 (T, H_thinker), got {:?}",
                dims
            );
        }
        if dims[1] != self.cfg.thinker_hidden_size {
            anyhow::bail!(
                "Talker::step: thinker_hidden last dim {} != cfg.thinker_hidden_size {}",
                dims[1],
                self.cfg.thinker_hidden_size
            );
        }
        let x = thinker_hidden.to_dtype(self.cfg.dtype)?;
        match &self.input_proj {
            Some(p) => p.forward(&x),
            None => Ok(x),
        }
    }

    pub fn step(&self, thinker_hidden: &Tensor, prev_speech_tokens: &[u32]) -> Result<u32> {
        let context = self.project_thinker(thinker_hidden)?;
        let t_ctx = context.dim(0).map_err(|e| anyhow::anyhow!(e))?;

        let mut parts: Vec<Tensor> = vec![context];
        let t_prev = prev_speech_tokens.len();
        if t_prev > 0 {
            let ids = Tensor::from_vec(prev_speech_tokens.to_vec(), t_prev, &self.device)?
                .to_dtype(DType::U32)?;
            let prev_emb = self
                .embed_tokens
                .index_select(&ids, 0)?
                .reshape((t_prev, self.cfg.hidden_size))?
                .to_dtype(self.cfg.dtype)?;
            parts.push(prev_emb);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let x = Tensor::cat(&refs, 0)?;
        let seq = t_ctx + t_prev;
        if seq > self.cfg.max_position_embeddings {
            anyhow::bail!(
                "Talker::step: seq {} exceeds max_position_embeddings {}",
                seq,
                self.cfg.max_position_embeddings
            );
        }

        let mut h = x.unsqueeze(0)?;
        let positions =
            Tensor::from_vec((0u32..seq as u32).collect::<Vec<_>>(), seq, &self.device)?;
        for layer in &self.layers {
            h = layer.forward(&h, &positions, &self.rope)?;
        }
        let h = self.norm.forward(&h)?;
        let h = h.squeeze(0)?;
        let last = h.narrow(0, seq - 1, 1)?;
        let logits = self.head.forward(&last)?;
        let logits = logits.to_dtype(DType::F32)?.squeeze(0)?;
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let (best_idx, _best_val) =
            logits_v
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv {
                        (i, v)
                    } else {
                        (bi, bv)
                    }
                });
        Ok(best_idx as u32)
    }

    pub fn generate(
        &self,
        thinker_hidden: &Tensor,
        max_speech_tokens: usize,
        eos_id: u32,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(max_speech_tokens);
        for _ in 0..max_speech_tokens {
            let tok = self.step(thinker_hidden, &out)?;
            if tok == eos_id {
                break;
            }
            out.push(tok);
        }
        Ok(out)
    }

    pub fn load_weights(&mut self, weights: &nv_weights::WeightLoader) -> Result<()> {
        let dtype = self.cfg.dtype;
        let h = self.cfg.hidden_size;
        let h_in = self.cfg.thinker_hidden_size;
        let q = self.cfg.q_dim();
        let kv = self.cfg.kv_dim();
        let ffn = self.cfg.intermediate_size;
        let vocab = self.cfg.speech_vocab_size;

        if !self.cfg.input_proj_is_identity() {
            let key = "talker.input_proj.weight";
            if weights.has(key) {
                self.input_proj =
                    Some(Linear::new(load_2d(weights, key, (h, h_in), dtype)?, None)?);
            }
        }

        self.embed_tokens = load_2d(weights, "talker.embed_tokens.weight", (vocab, h), dtype)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("talker.layers.{i}");
            layer.pre_attn_norm = RmsNorm::new(
                load_1d(
                    weights,
                    &format!("{prefix}.input_layernorm.weight"),
                    h,
                    dtype,
                )?,
                self.cfg.rms_norm_eps,
            );
            layer.pre_mlp_norm = RmsNorm::new(
                load_1d(
                    weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                    h,
                    dtype,
                )?,
                self.cfg.rms_norm_eps,
            );
            layer.q_proj = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.self_attn.q_proj.weight"),
                    (q, h),
                    dtype,
                )?,
                None,
            )?;
            layer.k_proj = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.self_attn.k_proj.weight"),
                    (kv, h),
                    dtype,
                )?,
                None,
            )?;
            layer.v_proj = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.self_attn.v_proj.weight"),
                    (kv, h),
                    dtype,
                )?,
                None,
            )?;
            layer.o_proj = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.self_attn.o_proj.weight"),
                    (h, q),
                    dtype,
                )?,
                None,
            )?;
            let gate = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.mlp.gate_proj.weight"),
                    (ffn, h),
                    dtype,
                )?,
                None,
            )?;
            let up = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.mlp.up_proj.weight"),
                    (ffn, h),
                    dtype,
                )?,
                None,
            )?;
            let down = Linear::new(
                load_2d(
                    weights,
                    &format!("{prefix}.mlp.down_proj.weight"),
                    (h, ffn),
                    dtype,
                )?,
                None,
            )?;
            layer.mlp = Mlp::new(gate, up, down)?;
        }

        self.norm = RmsNorm::new(
            load_1d(weights, "talker.norm.weight", h, dtype)?,
            self.cfg.rms_norm_eps,
        );
        let head_w = load_2d(weights, "talker.head.weight", (vocab, h), dtype)?;
        let head_b = if weights.has("talker.head.bias") {
            Some(load_1d(weights, "talker.head.bias", vocab, dtype)?)
        } else {
            None
        };
        self.head = Linear::new(head_w, head_b)?;
        Ok(())
    }
}

fn load_1d(
    weights: &nv_weights::WeightLoader,
    name: &str,
    dim: usize,
    dtype: DType,
) -> Result<Tensor> {
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
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
    let w = weights
        .get(name, dtype)
        .with_context(|| format!("load {name}"))?;
    let d = w.dims();
    if d.len() != 2 || d[0] != shape.0 || d[1] != shape.1 {
        anyhow::bail!("{name}: expected [{}, {}], got {:?}", shape.0, shape.1, d);
    }
    Ok(w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> TalkerConfig {
        TalkerConfig {
            hidden_size: 32,
            thinker_hidden_size: 32,
            speech_vocab_size: 16,
            num_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            intermediate_size: 48,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            dtype: DType::F32,
        }
    }

    fn tiny_cfg_with_proj() -> TalkerConfig {
        TalkerConfig {
            thinker_hidden_size: 48,
            ..tiny_cfg()
        }
    }

    #[test]
    fn builds_on_cpu_with_defaults() {
        let cfg = tiny_cfg();
        let t = Talker::new(cfg.clone(), &Device::Cpu).expect("Talker::new");
        assert_eq!(t.num_layers(), cfg.num_layers);
        assert_eq!(t.config().hidden_size, 32);
        assert!(t.input_proj.is_none(), "matching dims -> no input_proj");
    }

    #[test]
    fn builds_with_input_proj_when_dims_differ() {
        let cfg = tiny_cfg_with_proj();
        let t = Talker::new(cfg, &Device::Cpu).unwrap();
        assert!(t.input_proj.is_some());
    }

    #[test]
    fn step_returns_valid_token_id() {
        let cfg = tiny_cfg();
        let t = Talker::new(cfg.clone(), &Device::Cpu).unwrap();
        let h = Tensor::zeros((4usize, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let tok = t.step(&h, &[]).expect("step");
        assert!(
            (tok as usize) < cfg.speech_vocab_size,
            "token {} not in vocab {}",
            tok,
            cfg.speech_vocab_size
        );
        assert_eq!(tok, 0);
    }

    #[test]
    fn step_accepts_prev_tokens() {
        let cfg = tiny_cfg();
        let t = Talker::new(cfg.clone(), &Device::Cpu).unwrap();
        let h = Tensor::zeros((2usize, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap();
        let tok = t.step(&h, &[1, 2, 3]).expect("step with prev");
        assert!((tok as usize) < cfg.speech_vocab_size);
    }

    #[test]
    fn generate_stops_at_eos() {
        let cfg = tiny_cfg();
        let mut t = Talker::new(cfg.clone(), &Device::Cpu).unwrap();
        let mut bias = vec![0f32; cfg.speech_vocab_size];
        bias[5] = 100.0;
        let head_w = Tensor::zeros(
            (cfg.speech_vocab_size, cfg.hidden_size),
            DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        let head_b = Tensor::from_vec(bias, cfg.speech_vocab_size, &Device::Cpu).unwrap();
        t.set_head(Linear::new(head_w, Some(head_b)).unwrap());

        let hidden = Tensor::zeros((2usize, cfg.hidden_size), DType::F32, &Device::Cpu).unwrap();

        let out = t.generate(&hidden, 128, 5).expect("generate");
        assert!(out.is_empty(), "expected immediate EOS, got {:?}", out);

        let out2 = t.generate(&hidden, 4, 7).expect("generate no-eos");
        assert_eq!(out2, vec![5, 5, 5, 5]);
    }

    #[test]
    fn rejects_wrong_thinker_hidden() {
        let cfg = tiny_cfg();
        let t = Talker::new(cfg.clone(), &Device::Cpu).unwrap();
        let bad = Tensor::zeros((2usize, 64usize), DType::F32, &Device::Cpu).unwrap();
        let err = t.step(&bad, &[]).unwrap_err();
        assert!(err.to_string().contains("thinker_hidden"));
    }
}
