use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use crate::attn::{flash_attn, AttnConfig};
use crate::linear::Linear;
use crate::mlp::Mlp;
use crate::norm::RmsNorm;
use crate::rope::{Rope, RopeConfig, RopeKind};

#[derive(Clone, Copy, Debug)]
pub struct BlockConfig {
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_dim: usize,
    pub rope_base: f32,
    pub rms_eps: f64,
    pub max_seq_len: usize,
    pub qk_norm: bool,
}

pub struct TransformerBlock {
    cfg: BlockConfig,
    pre_attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    pre_mlp_norm: RmsNorm,
    mlp: Mlp,
    rope: Rope,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: BlockConfig,
        pre_attn_norm: RmsNorm,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        pre_mlp_norm: RmsNorm,
        mlp: Mlp,
        rope: Rope,
    ) -> Self {
        Self {
            cfg,
            pre_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: None,
            k_norm: None,
            pre_mlp_norm,
            mlp,
            rope,
        }
    }

    pub fn with_qk_norm(mut self, q_norm: RmsNorm, k_norm: RmsNorm) -> Self {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
    }

    pub fn q_norm(&self) -> Option<&RmsNorm> {
        self.q_norm.as_ref()
    }

    pub fn k_norm(&self) -> Option<&RmsNorm> {
        self.k_norm.as_ref()
    }

    pub fn pre_attn_norm(&self) -> &RmsNorm {
        &self.pre_attn_norm
    }

    pub fn pre_mlp_norm(&self) -> &RmsNorm {
        &self.pre_mlp_norm
    }

    pub fn q_proj(&self) -> &Linear {
        &self.q_proj
    }

    pub fn k_proj(&self) -> &Linear {
        &self.k_proj
    }

    pub fn v_proj(&self) -> &Linear {
        &self.v_proj
    }

    pub fn o_proj(&self) -> &Linear {
        &self.o_proj
    }

    pub fn mlp(&self) -> &Mlp {
        &self.mlp
    }

    pub fn rope(&self) -> &Rope {
        &self.rope
    }

    pub fn from_candle_vb(vb: VarBuilder, cfg: BlockConfig) -> Result<Self> {
        let pre_attn_norm =
            RmsNorm::from_candle_vb(vb.pp("input_layernorm"), cfg.hidden_dim, cfg.rms_eps)?;
        let pre_mlp_norm = RmsNorm::from_candle_vb(
            vb.pp("post_attention_layernorm"),
            cfg.hidden_dim,
            cfg.rms_eps,
        )?;
        let q_proj = Linear::from_candle_vb(
            vb.pp("self_attn.q_proj"),
            cfg.hidden_dim,
            cfg.num_heads * cfg.head_dim,
            false,
        )?;
        let k_proj = Linear::from_candle_vb(
            vb.pp("self_attn.k_proj"),
            cfg.hidden_dim,
            cfg.num_kv_heads * cfg.head_dim,
            false,
        )?;
        let v_proj = Linear::from_candle_vb(
            vb.pp("self_attn.v_proj"),
            cfg.hidden_dim,
            cfg.num_kv_heads * cfg.head_dim,
            false,
        )?;
        let o_proj = Linear::from_candle_vb(
            vb.pp("self_attn.o_proj"),
            cfg.num_heads * cfg.head_dim,
            cfg.hidden_dim,
            false,
        )?;
        let mlp = Mlp::from_candle_vb(vb.pp("mlp"), cfg.hidden_dim, cfg.intermediate_dim)?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: cfg.head_dim,
                max_seq_len: cfg.max_seq_len,
                base: cfg.rope_base,
                kind: RopeKind::Standard,
            },
            vb.device(),
        )?;
        let mut block = Self::new(
            cfg,
            pre_attn_norm,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            pre_mlp_norm,
            mlp,
            rope,
        );
        if cfg.qk_norm {
            let q_norm =
                RmsNorm::from_candle_vb(vb.pp("self_attn.q_norm"), cfg.head_dim, cfg.rms_eps)?;
            let k_norm =
                RmsNorm::from_candle_vb(vb.pp("self_attn.k_norm"), cfg.head_dim, cfg.rms_eps)?;
            block = block.with_qk_norm(q_norm, k_norm);
        }
        Ok(block)
    }

    pub fn config(&self) -> &BlockConfig {
        &self.cfg
    }

    pub fn forward(&self, x: &Tensor, positions: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() != 3 || dims[2] != self.cfg.hidden_dim {
            anyhow::bail!(
                "TransformerBlock: expected (B, T, hidden={}) got {:?}",
                self.cfg.hidden_dim,
                dims
            );
        }
        let (b, t, _) = (dims[0], dims[1], dims[2]);

        let normed = self.pre_attn_norm.forward(x)?;
        let q = self.q_proj.forward(&normed)?;
        let k = self.k_proj.forward(&normed)?;
        let v = self.v_proj.forward(&normed)?;

        let q = q.reshape((b, t, self.cfg.num_heads, self.cfg.head_dim))?;
        let k = k.reshape((b, t, self.cfg.num_kv_heads, self.cfg.head_dim))?;
        let v = v.reshape((b, t, self.cfg.num_kv_heads, self.cfg.head_dim))?;

        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(qn), Some(kn)) => (qn.forward(&q)?, kn.forward(&k)?),
            _ => (q, k),
        };

        let q_f32 = q.to_dtype(DType::F32)?;
        let k_f32 = k.to_dtype(DType::F32)?;
        let positions_2d = if positions.dims().len() == 1 {
            let pos_cpu = positions.to_device(&candle_core::Device::Cpu)?;
            let pos_vec: Vec<i32> = match pos_cpu.dtype() {
                DType::I32 => pos_cpu.to_vec1::<i32>()?,
                DType::I64 => pos_cpu
                    .to_vec1::<i64>()?
                    .into_iter()
                    .map(|v| v as i32)
                    .collect(),
                DType::U32 => pos_cpu
                    .to_vec1::<u32>()?
                    .into_iter()
                    .map(|v| v as i32)
                    .collect(),
                other => anyhow::bail!("unsupported positions dtype {other:?}"),
            };
            let mut tiled = Vec::with_capacity(b * t);
            for _ in 0..b {
                tiled.extend_from_slice(&pos_vec);
            }
            candle_core::Tensor::from_vec(tiled, (b, t), x.device())?
        } else {
            positions.clone()
        };
        let (q_rot, k_rot) = self.rope.apply(&q_f32, &k_f32, &positions_2d)?;
        let q = q_rot.to_dtype(x.dtype())?;
        let k = k_rot.to_dtype(x.dtype())?;

        let attn_cfg = AttnConfig {
            num_heads: self.cfg.num_heads,
            num_kv_heads: self.cfg.num_kv_heads,
            head_dim: self.cfg.head_dim,
            softmax_scale: 1.0 / (self.cfg.head_dim as f32).sqrt(),
            causal: true,
        };
        let attn_out = flash_attn(
            &q.contiguous()?,
            &k.contiguous()?,
            &v.contiguous()?,
            &attn_cfg,
        )?;
        let attn_out = attn_out.reshape((b, t, self.cfg.num_heads * self.cfg.head_dim))?;
        let attn_out = self.o_proj.forward(&attn_out)?;

        let x_after_attn = x.add(&attn_out)?;

        let normed2 = self.pre_mlp_norm.forward(&x_after_attn)?;
        let mlp_out = self.mlp.forward(&normed2)?;
        let out = x_after_attn.add(&mlp_out)?;
        Ok(out)
    }
}
