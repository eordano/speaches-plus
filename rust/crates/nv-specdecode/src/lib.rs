pub mod chain;
pub mod dflash;
pub mod eagle3;
pub mod eagle3_loader;
pub mod gemma4_assistant;
pub use nv_lookup::suffix_automaton;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub mod gemma4_verifier;
pub mod lora_spec;
pub mod qwen38_mtp;
mod util;
#[cfg(feature = "wgpu")]
pub mod wgpu_spec;

#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub use gemma4_verifier::{Gemma4Verifier, VerifyStep};

pub use chain::{
    accept_prefix_argmax, aux_row_extract, build_chain_batch, chain_positions, lower_tri_mask,
    ChainAccept, ChainJudgment, ChainState, ChainVerifier, ChainVerifyOut,
};
pub use dflash::{DFlashContextKv, DFlashSpeculatorConfig, LoadedDFlashDrafter};
pub use eagle3::{DraftScorer, DraftTree, Eagle3Config, Eagle3Proposer};
pub use eagle3_loader::{Eagle3SpeculatorConfig, LoadedEagle3Scorer};
pub use nv_lookup::{suffix_arm_wins, AcceptEma, SuffixAutomaton};
pub use qwen38_mtp::{
    assert_mtp_chain_depth_fits_verify_lm_head_rows_ceiling, mtp_chain_depth,
    mtp_chain_depth_from_env, mtp_draft_dir_override_from_env, mtp_drafter_selected,
    mtp_drafter_selected_from_env, mtp_round_hidden_reanchor_index, mtp_verify_replay_selected,
    mtp_verify_replay_selected_from_env, mtp_verify_rows_per_round, resolve_mtp_weight_files,
    run_mtp_verify_round, validate_mtp_named_shapes, verify_lm_head_rows_per_call_ceiling,
    MtpBatchedVerifyRound, MtpBatchedVerifyTarget, Qwen38MtpGeometry, MTP_CHAIN_DEPTH_DEFAULT,
    NV_Q38_MTP_VERIFY_REPLAY_ENV, QWEN38_27B_MTP_GEOMETRY,
};
#[cfg(feature = "cuda")]
pub use qwen38_mtp::{MtpKvCache, Qwen38DenseMtpHead, Qwen38MtpDecodeSession, Qwen38MtpSelfSpecEngine};
#[cfg(feature = "wgpu")]
pub use wgpu_spec::{
    rope_tables, ChainDrafter, LockstepChainSpec, ModelDrafter, PromptLookupDrafter, RealSpecStats,
    SpecDims, SpecStats, SpecWeights, StepDecoder, WgpuChainSpec, WgpuSpecModel,
};
