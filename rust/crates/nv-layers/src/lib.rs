pub use candle_core::{DType, Device, Tensor};
pub use candle_nn::{Module, VarBuilder};

#[cfg(feature = "cuda")]
pub use cudarc;

pub mod layer_mixer;
pub mod attn;
pub mod backend;
pub mod block;
pub mod conv;
pub mod cuda_stream;
pub mod linear;
pub mod linear_attn;
pub mod lora_slots;
pub mod mlp;
pub mod moe;
pub mod moe_bf16_grouped;
#[cfg(feature = "cuda")]
pub mod moe_grouped;
#[cfg(feature = "wgpu")]
pub mod moe_wgpu;
pub mod norm;
pub mod rope;
pub mod sampler;

pub use block::{BlockConfig, TransformerBlock};
pub use conv::{Conv1d, Conv2d, ConvTranspose1d};
pub use linear::Linear;
pub use linear_attn::{LinearAttention, LinearAttentionConfig};
pub use mlp::Mlp;
pub use moe::{MoeBlock, MoeConfig};
pub use norm::RmsNorm;
pub use rope::{Rope, RopeConfig, RopeKind};
