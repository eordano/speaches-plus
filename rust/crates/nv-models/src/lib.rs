pub mod decode_probe;
pub mod deepseek_ocr;
pub mod dots_ocr;
pub mod embed_row_splice;
pub mod gemma4;
#[cfg(feature = "wgpu")]
pub mod gemma4_assistant_wgpu;
pub mod gemma4_audio;
pub mod gemma4_batch_graph;
pub mod gemma4_e4b;
#[cfg(feature = "wgpu")]
pub mod gemma4_e4b_wgpu;
pub mod gemma4_gguf;
#[cfg(feature = "cuda")]
pub mod gemma4_graph;
pub mod gemma4_mm_splice;
pub mod gemma4_moe;
#[cfg(feature = "wgpu")]
pub mod gemma4_moe_wgpu;
pub mod gemma4_vision;
#[cfg(feature = "cuda")]
pub mod gemma4_vision_graph;
#[cfg(feature = "wgpu")]
pub mod gemma4_wgpu;
#[cfg(feature = "wgpu")]
pub mod gemma4_wgpu_shared;
pub mod got_ocr;
pub mod gpt_oss;
#[cfg(feature = "cuda")]
pub mod gpt_oss_cuda;
#[cfg(feature = "wgpu")]
pub mod gpt_oss_wgpu;
#[cfg(feature = "cuda")]
pub mod graph_engine;
pub mod hadamard_kv;
pub mod host_serving_util;
pub mod laguna;
#[cfg(feature = "cuda")]
pub mod laguna_batch_graph;
pub mod laguna_dflash;
#[cfg(feature = "cuda")]
pub mod laguna_fa2;
#[cfg(feature = "cuda")]
pub mod laguna_fp8;
#[cfg(feature = "cuda")]
pub mod laguna_graph;
#[cfg(feature = "cuda")]
pub mod laguna_prefill_attn;
#[cfg(feature = "cuda")]
pub mod laguna_serve;
#[cfg(feature = "cuda")]
pub mod laguna_step_graph;
pub mod laguna_train;
#[cfg(feature = "laguna-wip")]
pub mod laguna_wgpu;
#[cfg(all(test, feature = "cuda"))]
mod marlin_zero_drop_lifecycle;
pub use nv_config::{
    env_arm_opt_in_eq_some_1_cached_first_read_wins,
    env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins,
    env_arm_present_at_all_cached_first_read_wins,
};
pub mod laguna_prefix;
pub mod laguna_prefix_diagnostics;
#[cfg(feature = "wgpu")]
pub mod nvfp4_host;
pub mod paged_fp8;
pub mod prefix_reuse;
pub mod qwen3;
#[cfg(feature = "wgpu")]
pub mod qwen3_5_dense_wgpu;
pub mod qwen3_5_moe;
#[cfg(feature = "wgpu")]
pub mod qwen3_5_moe_wgpu;
#[cfg(feature = "cuda")]
pub mod qwen3_5_mtp;
pub mod qwen3_mm_splice;
#[cfg(feature = "wgpu")]
pub mod wgpu_ledger;
#[cfg(feature = "wgpu")]
pub mod wgpu_lora;
#[cfg(feature = "wgpu")]
pub(crate) mod wgpu_state_snapshot;

pub mod dense_train;
pub mod eagle3_train;
pub mod train_runner;

pub trait CausalLm {
    fn forward(&mut self, tokens: &[u32], positions: &[u32]) -> anyhow::Result<Vec<f32>>;
    fn vocab_size(&self) -> usize;
}
