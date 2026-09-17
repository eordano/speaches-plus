pub mod e4b;
pub mod lora;
pub mod peft;
pub mod schedule;
pub mod weights;

pub use lora::{AdamW, LoraVar};
pub use schedule::{
    refuse_rows_beyond_the_native_window_bound, run_qlora, run_qlora_pumped, TrainOptions,
    TrainReport, TrainSample, BUCKET_PADDED_TOKEN_BUDGET_MATCHES_NV_TRAIN_DOTS,
    EXAMPLE_TOKEN_CEILING_WHERE_THE_NATIVE_LIVE_GRAPH_OOMS_INTO_THE_CHECKPOINT_FALLBACK,
};
pub use weights::{TrainDims, TrainLayer, W4Mod};

pub const TRAINER_IS_AN_EXPLICIT_SCHEDULE_OVER_THE_QLORA_WGSL_KERNELS_NOT_AN_AUTOGRAD: &str =
    "this trainer replays nv-train/src/dots.rs as a fixed forward and backward pass order over \
     the nv-kernels qlora WGSL set: frozen W4 prefix forward, length-sorted padded buckets, \
     top-N-layer LoRA window, chunked W4-head CE, one backward sweep per live bucket, and a \
     candle-semantics AdamW; every gradient the backward emits is named in the schedule, so a \
     missing term is a compile error here instead of a silent autograd hole";

pub const WGSL_ARM_TRACKS_THE_CANONICAL_TRAJECTORY_AT_TOLERANCE_NEVER_BITWISE: &str =
    "wgpu-hal's Metal path compiles WGSL under fast math with no MTLMathMode::Safe override, so \
     the candle-Metal canonical 20-step trajectory (0.212043 -> 0.001704 on dump-n25, seed 7, \
     rank 16, grad_layers 8) is reproduced at the reassociation class, not bitwise; the native \
     gate prints both trajectories side by side and bounds the final loss, and the trainer \
     stays an opt-in test-gated arm per the default-flip law until a corpus instrument blesses \
     it";
