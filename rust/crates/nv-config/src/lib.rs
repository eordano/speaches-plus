pub mod env_arm;
pub mod measure;

pub use env_arm::{
    env_arm_dependency_assert_enabled_arm_requires_prerequisite,
    env_arm_opt_in_eq_some_1_reread_every_call,
    env_arm_opt_in_set_to_anything_but_0_reread_every_call,
    env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call,
    env_arm_present_at_all_reread_every_call,
};

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Dtype {
    F32,
    F16,
    Bf16,
    Fp8E4m3,
    Fp8E5m2,
    Int8,
    Int4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EngineConfig {
    pub backend: Backend,
    pub compute_dtype: Dtype,
    pub kv_dtype: Dtype,
    pub max_batch_size: usize,
    pub max_seq_len: usize,
    pub block_size: usize,
    pub tensor_parallel: usize,
    pub seed: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            backend: Backend::Cuda,
            compute_dtype: Dtype::Bf16,
            kv_dtype: Dtype::Bf16,
            max_batch_size: 32,
            max_seq_len: 8192,
            block_size: 16,
            tensor_parallel: 1,
            seed: 0,
        }
    }
}
