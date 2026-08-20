use anyhow::Result;

pub use crate::gemma4::LayerType;
pub use crate::laguna::{
    yarn_inv_freq, LagunaConfig, LagunaGating, LagunaRopeParams, MlpLayerType,
};

pub const NVFP4_BLOCK: usize = 16;
pub const MAX_HEAD_DIM: usize = 256;
pub const MAX_TOPK: usize = 16;
pub const ARGMAX_GROUPS: usize = 256;
pub const STAGING_FLUSH_BYTES: u64 = 256 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FfnKind {
    Dense,
    Moe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GateKind {
    None,
    PerHead,
    PerElement,
}

impl GateKind {
    pub fn from_config(gating: LagunaGating) -> Self {
        match gating {
            LagunaGating::None => GateKind::None,
            LagunaGating::PerHead => GateKind::PerHead,
            LagunaGating::PerElement => GateKind::PerElement,
        }
    }

    pub fn rows_for(self, num_q_heads: usize, head_dim: usize) -> usize {
        match self {
            GateKind::None => 0,
            GateKind::PerHead => num_q_heads,
            GateKind::PerElement => num_q_heads * head_dim,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct LayerShape {
    pub idx: usize,
    pub attn_kind: LayerType,
    pub ffn_kind: FfnKind,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub gqa_group: usize,
    pub q_rows: usize,
    pub kv_rows: usize,
    pub gate_rows: usize,
    pub rotary_dim: usize,
    pub rope_out_scale: f32,
    pub attn_softmax_scale: f32,
    pub window_tokens: Option<usize>,
    pub kv_capacity_tokens: usize,
    pub ffn_intermediate: usize,
}

impl LayerShape {
    pub fn is_moe(&self) -> bool {
        self.ffn_kind == FfnKind::Moe
    }

    pub fn is_sliding(&self) -> bool {
        matches!(self.attn_kind, LayerType::SlidingAttention)
    }

    pub fn attn_out_elems(&self) -> usize {
        self.num_q_heads * self.head_dim
    }
}

#[derive(Clone, Debug)]
pub struct LagunaShapes {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_layers: usize,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    pub max_q_heads: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: usize,
    pub gate_kind: GateKind,
    pub num_experts: usize,
    pub top_k: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub dense_intermediate_size: usize,
    pub norm_topk_prob: bool,
    pub routed_scaling: f32,
    pub router_softcap: f32,
    pub tie_word_embeddings: bool,
    pub max_seq_tokens: usize,
    pub rotary_dim_full: usize,
    pub rotary_dim_sliding: usize,
    pub rope_theta_full: f32,
    pub rope_theta_sliding: f32,
    pub rope_inv_freq_full: Vec<f32>,
    pub rope_inv_freq_sliding: Vec<f32>,
    pub layers: Vec<LayerShape>,
}

impl LagunaShapes {
    pub fn derive(config: &LagunaConfig, max_seq_tokens: usize) -> Result<Self> {
        anyhow::ensure!(max_seq_tokens > 0, "max_seq_tokens must be positive");
        anyhow::ensure!(
            config.layer_types.len() == config.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            config.layer_types.len(),
            config.num_hidden_layers
        );
        anyhow::ensure!(
            config.head_dim <= MAX_HEAD_DIM && config.head_dim.is_multiple_of(2),
            "head_dim {} must be even and <= {MAX_HEAD_DIM}",
            config.head_dim
        );
        anyhow::ensure!(
            config.hidden_size.is_multiple_of(2),
            "hidden_size {} must be even (bf16 word packing)",
            config.hidden_size
        );
        anyhow::ensure!(
            config.num_experts_per_tok <= MAX_TOPK,
            "num_experts_per_tok {} exceeds {MAX_TOPK}",
            config.num_experts_per_tok
        );

        let gate_kind = GateKind::from_config(config.gating);
        let rotary_dim_full = config.rotary_dim_full();
        let rotary_dim_sliding = config.rotary_dim_sliding();
        let full_attn_factor = config.full_attention_factor();
        let softmax_scale = (config.head_dim as f32).powf(-0.5);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            let attn_kind = config.layer_kind(idx);
            let num_q_heads = config.num_heads_for_layer(idx);
            anyhow::ensure!(
                num_q_heads > 0 && num_q_heads.is_multiple_of(config.num_key_value_heads),
                "layer {idx} q heads {num_q_heads} not a multiple of kv heads {}",
                config.num_key_value_heads
            );
            let ffn_kind = if config.is_moe_layer(idx) {
                FfnKind::Moe
            } else {
                FfnKind::Dense
            };
            let (rotary_dim, rope_out_scale) = match attn_kind {
                LayerType::SlidingAttention => (rotary_dim_sliding, 1.0f32),
                LayerType::FullAttention => (rotary_dim_full, full_attn_factor),
            };
            anyhow::ensure!(
                rotary_dim % 2 == 0 && rotary_dim <= config.head_dim,
                "layer {idx} rotary_dim {rotary_dim} invalid for head_dim {}",
                config.head_dim
            );
            let window_tokens = match attn_kind {
                LayerType::SlidingAttention => Some(config.sliding_window.max(1)),
                LayerType::FullAttention => None,
            };
            layers.push(LayerShape {
                idx,
                attn_kind,
                ffn_kind,
                num_q_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                gqa_group: num_q_heads / config.num_key_value_heads,
                q_rows: num_q_heads * config.head_dim,
                kv_rows: config.num_key_value_heads * config.head_dim,
                gate_rows: gate_kind.rows_for(num_q_heads, config.head_dim),
                rotary_dim,
                rope_out_scale,
                attn_softmax_scale: softmax_scale,
                window_tokens,
                kv_capacity_tokens: max_seq_tokens,
                ffn_intermediate: match ffn_kind {
                    FfnKind::Dense => config.intermediate_size,
                    FfnKind::Moe => config.moe_intermediate_size,
                },
            });
        }

        let max_q_heads = layers.iter().map(|l| l.num_q_heads).max().unwrap_or(0);

        Ok(Self {
            hidden_size: config.hidden_size,
            vocab_size: config.vocab_size,
            num_layers: config.num_hidden_layers,
            head_dim: config.head_dim,
            num_kv_heads: config.num_key_value_heads,
            max_q_heads,
            rms_norm_eps: config.rms_norm_eps as f32,
            sliding_window: config.sliding_window.max(1),
            gate_kind,
            num_experts: config.num_experts,
            top_k: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            shared_expert_intermediate_size: config.shared_expert_intermediate_size,
            dense_intermediate_size: config.intermediate_size,
            norm_topk_prob: config.norm_topk_prob,
            routed_scaling: config.moe_routed_scaling_factor,
            router_softcap: config.moe_router_logit_softcapping,
            tie_word_embeddings: config.tie_word_embeddings,
            max_seq_tokens,
            rotary_dim_full,
            rotary_dim_sliding,
            rope_theta_full: config.full_rope_params().rope_theta,
            rope_theta_sliding: config.sliding_rope_params().rope_theta,
            rope_inv_freq_full: rope_inv_freq(rotary_dim_full, config.full_rope_params()),
            rope_inv_freq_sliding: rope_inv_freq(rotary_dim_sliding, config.sliding_rope_params()),
            layers,
        })
    }

    pub fn layer(&self, idx: usize) -> &LayerShape {
        &self.layers[idx]
    }

    pub fn hidden_words(&self) -> usize {
        self.hidden_size / 2
    }

    pub fn moe_layer_indices(&self) -> Vec<usize> {
        self.layers
            .iter()
            .filter(|l| l.is_moe())
            .map(|l| l.idx)
            .collect()
    }

    pub fn dense_layer_indices(&self) -> Vec<usize> {
        self.layers
            .iter()
            .filter(|l| !l.is_moe())
            .map(|l| l.idx)
            .collect()
    }

    pub fn kv_cache_bytes_bf16(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| 2 * (l.kv_capacity_tokens * l.kv_rows * 2) as u64)
            .sum()
    }

    pub fn validate_nvfp4_alignment(&self) -> Result<()> {
        let block = NVFP4_BLOCK;
        anyhow::ensure!(
            self.hidden_size.is_multiple_of(4 * block),
            "hidden_size {} must be a multiple of {}",
            self.hidden_size,
            4 * block
        );
        anyhow::ensure!(
            self.moe_intermediate_size.is_multiple_of(4 * block),
            "moe_intermediate_size {} must be a multiple of {}",
            self.moe_intermediate_size,
            4 * block
        );
        anyhow::ensure!(
            self.shared_expert_intermediate_size
                .is_multiple_of(4 * block),
            "shared_expert_intermediate_size {} must be a multiple of {}",
            self.shared_expert_intermediate_size,
            4 * block
        );
        anyhow::ensure!(
            self.dense_intermediate_size.is_multiple_of(4 * block),
            "intermediate_size {} must be a multiple of {}",
            self.dense_intermediate_size,
            4 * block
        );
        Ok(())
    }
}

pub fn rope_inv_freq(rotary_dim: usize, params: &LagunaRopeParams) -> Vec<f32> {
    yarn_inv_freq(rotary_dim, params)
}

pub use crate::gemma4_wgpu_shared::rope_tables_from_inv_freq;

pub fn window_start(total_tokens: usize, window: Option<usize>) -> usize {
    match window {
        Some(w) if w > 0 && total_tokens > w => total_tokens - w,
        _ => 0,
    }
}
