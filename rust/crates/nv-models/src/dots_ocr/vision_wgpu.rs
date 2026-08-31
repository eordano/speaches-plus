use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use half::f16;
use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as coop;
use nv_kernels::wgpu_backend::{compose_enabled, WgpuContext};
use nv_weights::WeightLoader;

use super::preprocess::{PreparedImage, MERGE_SIZE, PATCH_DIM};
use super::vision::{vision_rope_tables, DotsVisionConfig};

pub const DOTS_VISION_WGPU_STAGES_EVERY_WEIGHT_AS_F16_BECAUSE_THE_ONLY_GENERAL_LARGE_M_GEMM_IN_THIS_TREE_IS_COOP_MAT_F16: &str =
    "nv-kernels ships no general large-M bf16 GEMM: gemm_bf16_small_m::MAX_M is 9 and \
     qwen3_5_dense_wgpu's m-row bf16 arm stops at 16, so a 14k-row vision prefill has exactly one \
     wide route in this tree, gemm_coop_f16, whose operands are f16 or f32. This tower therefore \
     stages the bf16 checkpoint as f16 once at load: f16 carries three more mantissa bits than \
     bf16 and every dots.ocr vision weight is inside f16 range, so the staging is not the accuracy \
     floor.";

pub const DOTS_VISION_WGPU_GEMM_ENV: &str = "NV_DOTS_VISION_WGPU_GEMM";

pub const DOTS_VISION_WGPU_WQ_ENV: &str = "NV_DOTS_VISION_WGPU_WQ";

pub const DOTS_VISION_NVFP4_WEIGHT_ARM_IS_ENV_SELECTED_AND_ANY_VALUE_BUT_NVFP4_KEEPS_THE_F16_STAGING_BYTE_FOR_BYTE: &str =
    "NV_DOTS_VISION_WGPU_WQ=nvfp4 stages every projection weight as NVFP4 block-16 -- the bf16 \
     checkpoint is quantized host-side at load and unpacked in-kernel by \
     gemm_coop_f16::entry_w4a16 -- and every other value, unset and 0 included, keeps the f16 \
     staging and the f16 coop kernels untouched. Only the projection weight operand and the \
     kernel that unpacks it change: activations, biases, norms and the attention GEMMs, whose \
     weight operands are activations, are identical on both arms.";

pub const DOTS_VISION_NVFP4_ARM_NEEDS_THE_16X16_COOP_ROUTE_BECAUSE_SOURCE_W4A16_STAGES_ONE_16_CODE_BLOCK_PER_K_TILE_ROW: &str =
    "gemm_coop_f16::source_w4a16 asserts tile 16: the in-kernel unpack stages exactly one \
     16-code NVFP4 block per k-tile row, so an adapter serving only 8x8x8 fragments has no \
     w4a16 route and the nvfp4 arm refuses to load rather than silently serving f16.";

pub const DOTS_VISION_MLX_W4_ARM_SERVES_PACKED_NIBBLES_THROUGH_A_PER_USE_DEQUANT_BECAUSE_THE_COOP_GEMM_IS_COMPUTE_BOUND_AT_THESE_M: &str =
    "NV_DOTS_VISION_WGPU_WQ=w4g32|w4g64 quantizes every projection whose k divides the group size \
     to mlx-affine u4 at load (bf16-rounded scale and bias per group, lsb-first nibbles, the \
     nv_quant::mlx_affine layout) and keeps only the packed words plus widened scales and biases \
     resident, dequantizing each weight into one shared f16 scratch immediately before the coop \
     GEMM that consumes it. At vision M sizes the GEMMs are compute-bound, so the arm buys VRAM \
     (about 0.625 against 2 bytes per weight) and NOT throughput; the dequant traffic per forward \
     is the staged weight bytes once per projection use. The -twin suffix instead dequantizes \
     host-side and stages a resident f16 twin: byte-for-byte the f16 serving path over \
     w4-rounded weights, buying none of VRAM, load time or bandwidth -- it exists to price the \
     w4 rounding alone. Linears whose k is not a multiple of the group size stay f16 on both arms.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionWeightFmt {
    F16,
    Nvfp4Block16,
    MlxW4 { gs: usize, twin: bool },
}

impl VisionWeightFmt {
    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Nvfp4Block16 => "nvfp4",
            Self::MlxW4 { gs: 32, twin: false } => "w4g32",
            Self::MlxW4 { gs: 32, twin: true } => "w4g32-twin",
            Self::MlxW4 { gs: 64, twin: false } => "w4g64",
            Self::MlxW4 { gs: 64, twin: true } => "w4g64-twin",
            Self::MlxW4 { .. } => "w4",
        }
    }

    pub fn from_env() -> Self {
        let _ = DOTS_VISION_NVFP4_WEIGHT_ARM_IS_ENV_SELECTED_AND_ANY_VALUE_BUT_NVFP4_KEEPS_THE_F16_STAGING_BYTE_FOR_BYTE;
        let _ = DOTS_VISION_MLX_W4_ARM_SERVES_PACKED_NIBBLES_THROUGH_A_PER_USE_DEQUANT_BECAUSE_THE_COOP_GEMM_IS_COMPUTE_BOUND_AT_THESE_M;
        match std::env::var(DOTS_VISION_WGPU_WQ_ENV).ok().as_deref() {
            Some("nvfp4") => Self::Nvfp4Block16,
            Some("w4g32") => Self::MlxW4 { gs: 32, twin: false },
            Some("w4g64") => Self::MlxW4 { gs: 64, twin: false },
            Some("w4g32-twin") => Self::MlxW4 { gs: 32, twin: true },
            Some("w4g64-twin") => Self::MlxW4 { gs: 64, twin: true },
            _ => Self::F16,
        }
    }
}

pub const DOTS_VISION_WGPU_MERGER_ENV: &str = "NV_DOTS_VISION_WGPU_MERGER";

pub const DOTS_VISION_GPU_MERGER_IS_THE_DEFAULT_BECAUSE_THE_HOST_ROUND_TRIPS_WERE_THE_WHOLE_RESIDUAL_AND_THE_TOKEN_GATE_HELD: &str =
    "The host merger reads the whole trunk back, layer-norms and gelu_erf's it on the CPU between \
     the two merger GEMMs, and by deleted work those round trips are 0.288s of a 6.435s invoice \
     forward on Apple M2 Ultra -- the residual bracket's entire finding, every other residual \
     arm reading noise. The gpu merger keeps the trunk on the device: a layer-norm kernel writes \
     the merge-grouped f16 GEMM operand straight from the post-trunk-norm f16 buffer, a gelu \
     kernel bridges fc0 to fc2, and the only readback left is the merged feature map -- host \
     6.435s +-0.006 to gpu 6.168s +-0.004, rel-rms 1.9733e-5 against the host arm. Its \
     layer-norm reduction order differs from candle's and its erf is the Abramowitz-Stegun \
     7.1.26 polynomial, the same function to every consumer but not the same bits, so the flip \
     gate was corpus token equality: all twelve stable pages hold CER and WER at exactly 0.0000, \
     every bistable keeps its standing trajectory and token count, mean CER 0.0736 \
     byte-identical. NV_DOTS_VISION_WGPU_MERGER=host restores the host merger bit-for-bit. The \
     gpu route requires post_norm because it reads the f16 post-trunk-norm buffer; a tower \
     without one keeps the host merger.";

pub fn merger_gpu_arm() -> bool {
    let _ = DOTS_VISION_GPU_MERGER_IS_THE_DEFAULT_BECAUSE_THE_HOST_ROUND_TRIPS_WERE_THE_WHOLE_RESIDUAL_AND_THE_TOKEN_GATE_HELD;
    !matches!(
        std::env::var(DOTS_VISION_WGPU_MERGER_ENV).ok().as_deref(),
        Some("host") | Some("0")
    )
}

pub const DOTS_VISION_WGPU_FITTED_PLANS_ENV: &str = "NV_DOTS_VISION_WGPU_FITTED_PLANS";

pub const DOTS_VISION_M5_MAX_FITTED_PLANS_ADD_ONLY_THE_NARROW_N_ATTN_CTX_WINNER_THE_WIDTH_GATED_DEFAULT_CANNOT_EXPRESS: &str =
    "resolve_plan's width-gated default already serves tm8_tn4_sg4 to every shape wider than one \
     column band and the square tile to attn_ctx, the one narrow shape. On Apple M5 Max [Metal] \
     at 7400 rows the attn_ctx ladder prefers tm8_tn2_sg2 over that square, 7.89 -> 8.76 TFLOP/s \
     (ladder tests/dots_ocr_vision_phase_profile.rs; arm spreads are within +-8 percent run to \
     run, and this is the only shape whose winner repeats across ladders). \
     NV_DOTS_VISION_WGPU_FITTED_PLANS=1 installs exactly that one shape row at forward time; the \
     install is bit-identity preserving because tm/tn/sg/ku never reorder the k loop.";

pub const DOTS_VISION_WGPU_ATTN_ENV: &str = "NV_DOTS_VISION_WGPU_ATTN";

pub const DOTS_VISION_PLAN_ENV: &str = "NV_DOTS_VISION_PLAN";

pub const DOTS_VISION_WGPU_FLASH_PLAN_ENV: &str = "NV_DOTS_VISION_WGPU_FLASH_PLAN";

pub const DOTS_VISION_ATTENTION_DEFAULTS_TO_ZMERGE_AND_SLAB_IS_THE_ESCAPE_THAT_REPRODUCES_THE_PRE_FLIP_RECORD_BIT_FOR_BIT: &str =
    "zmerge is the default. slab is the arm every dots.ocr wgpu number published before this flip \
     was measured on, and NV_DOTS_VISION_WGPU_ATTN=slab restores it bit for bit: a q x n_pad f32 \
     score slab, a three-read row softmax that normalises by 1/z before storing f16 \
     probabilities, and a merge pass. zmerge is the same slab and the same reduction order with \
     the 1/z moved into the merge, which deletes one of the three reads and stops storing \
     probabilities as f16 subnormals. flash never writes the slab: it is an online-softmax kernel \
     whose rescaling is different arithmetic, not a reassociation, so it cannot be gated bitwise \
     and is adjudicated on the per-layer parity table against the CPU F32 twin. On Apple M2 Ultra \
     at 13600 patches zmerge is worth 0.259 s of an 8.272 s forward on top of the fitted plan, \
     all of it out of the softmax phase, and it holds every fixture of the 070/071 set to the \
     same CER, WER and generated-token count as slab -- mean CER 0.0736, the twelve stable pages \
     still 0.0000 and the four bistable ones unmoved at 0.4544, 0.4204, 0.2414 and 0.0622. Since \
     it is a reassociation and not different arithmetic, slab and zmerge are gated against each \
     other on token equality rather than on bits.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttnArm {
    Slab,
    ZMerge,
    Flash,
    Steel,
}

impl AttnArm {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Slab => "slab",
            Self::ZMerge => "zmerge",
            Self::Flash => "flash",
            Self::Steel => "steel",
        }
    }

    pub fn from_env() -> Self {
        let _ = DOTS_VISION_ATTENTION_DEFAULTS_TO_ZMERGE_AND_SLAB_IS_THE_ESCAPE_THAT_REPRODUCES_THE_PRE_FLIP_RECORD_BIT_FOR_BIT;
        match std::env::var(DOTS_VISION_WGPU_ATTN_ENV).ok().as_deref() {
            Some("slab") => Self::Slab,
            Some("flash") => Self::Flash,
            Some("steel") => Self::Steel,
            _ => Self::ZMerge,
        }
    }
}


pub const DOTS_VISION_WGPU_SCALAR_ROUTE_IS_A_CORRECTNESS_FALLBACK_NOT_A_SERVING_PATH: &str =
    "An adapter without EXPERIMENTAL_COOPERATIVE_MATRIX falls back to dv_gemm_f16_scalar, one \
     invocation per output element. It is correct and it is orders off a tiled GEMM; it exists so \
     the vision tower has a defined answer on that hardware, not so it serves a page there. METAL \
     IS NOT THAT HARDWARE: an Apple M2 Ultra reports the feature, advertises 8x8x8 in f32/f32->f32, \
     f16/f16->f32 and f16/f16->f16, and its runtime-probed subgroup width is 32, so \
     gemm_coop_f16::select skips the absent 16x16x16 and compiles the 8x8 route. The tower serves \
     there on coop_mat8x8_f16 at 15 to 18 TFLOP/s. Only the 16x16 arms are unavailable, which is \
     why the nvfp4 weight route refuses to load and the fitted-on-16x16 plan is not the plan this \
     adapter wants.";

pub const ATTENTION_QUERY_BLOCK_ROWS_CEILING_IS_WHERE_THE_DISPATCH_LADDER_STOPPED_PAYING: usize =
    8192;

pub const ATTENTION_QUERY_BLOCK_IS_SIZED_FROM_THE_REPORTED_STORAGE_BINDING_LIMIT_NOT_A_FIXED_CONSTANT: &str =
    "The f32 score slab is query_block x n_pad x 4 bytes and n_pad is the patch count, which on a \
     21 MP scan is 56656 -- so a fixed block is a portability bug: 16384 rows is 3.7 GB and this \
     adapter rejects the binding outright. Widening the block merges attention dispatches, and on \
     this tree that is worth roughly linear in how many it removes: 512/2048/4096/8192 rows ran \
     the same page in 374/120/87.5/56.9 s, bit-identical output. So take the ceiling from the \
     ladder and the floor from the device.";

pub const WORKGROUP_SIZE: u32 = 256;

const MERGER_LAYER_NORM_EPS_MATCHES_THE_REFERENCE_PATCH_MERGER: f64 = 1e-6;

pub const DOTS_VISION_UNIFORM_BLOCK_IS_EIGHT_U32_IN_EVERY_PARAM_STRUCT_SO_ITS_CONTENT_IS_THE_CACHE_KEY: usize = 32;

pub const DOTS_VISION_PASSES_PER_SUBMIT_BECAUSE_MERGING_D_DISPATCHES_INTO_ONE_SUBMIT_PAYS_ROUGHLY_LINEARLY_IN_D: usize =
    2048;

pub const DOTS_VISION_CTX_MERGE_IS_A_STRIDE_AND_AN_OFFSET_BUT_AN_F16_ACCUMULATOR_COSTS_NINE_PERCENT_SO_THE_MERGE_PASS_STAYS: &str =
    "attn_ctx writes ctxb as f32 [head][row][dim] and dv_attn_merge_f16 copies it to attn_x as f16 \
     [row][head*head_dim+dim]. That copy IS expressible as a stride and an offset -- y_stride \
     heads*head_dim, y_off row_block*heads*head_dim + head*head_dim -- so an f16-output context \
     GEMM writes attn_x directly and both the merge pass and the ctxb round trip disappear. \
     Measured: it is worth 0.066 s of an 8.247 s full-resolution forward, 0.8 percent, and it costs \
     9.4234e-2 rms-relative against the f32-accumulator arm on a 3300-patch page, with 1267198 of \
     1267200 merged features moved. That is larger than the 1.0162e-1 the CUDA bf16 path reads \
     against the CPU F32 twin and fourteen times the slab arm's own 6.5276e-3, so it spends the \
     entire accuracy lead this tower holds for under one percent. The tempting argument -- that the \
     context GEMM is a convex combination over normalised probabilities and therefore bounded by \
     max|V| -- is true and IRRELEVANT: convexity bounds the magnitude, not the rounding accumulated \
     over n_pad/tile sequential fragment adds, which is 1700 of them at full resolution. An f32 \
     accumulator with an f16 store would keep the pass-kill without the error, but that needs a new \
     epilogue, and the pass is worth 0.033 s.";

pub const DOTS_VISION_WGPU_PHASES_ENV: &str = "NV_DOTS_VISION_WGPU_PHASES";

pub const DOTS_VISION_PHASE_TIMING_IS_DRAIN_DISTORTED_BECAUSE_ATTRIBUTING_A_QUEUED_PASS_TO_A_PHASE_COSTS_A_DEVICE_WAIT_AT_EVERY_BOUNDARY: &str =
    "The tower queues up to DOTS_VISION_PASSES_PER_SUBMIT dispatches before it submits and waits on \
     the device only at the readbacks, so a host instant taken inside a layer attributes nothing. A \
     phase boundary therefore submits and waits for device idle, which drains the pipeline: every \
     row is an upper bound on its phase, their sum is an upper bound on the untimed forward, and \
     the forward prints the sum beside its own wall so the distortion is the difference. Unset, \
     mark() is one branch on an Option and the forward is the serving path unchanged.";

#[derive(Default)]
pub struct PhaseLog {
    at: Option<std::time::Instant>,
    rows: Vec<(&'static str, f64, u64)>,
}

impl PhaseLog {
    pub fn from_env() -> Self {
        let _ = DOTS_VISION_PHASE_TIMING_IS_DRAIN_DISTORTED_BECAUSE_ATTRIBUTING_A_QUEUED_PASS_TO_A_PHASE_COSTS_A_DEVICE_WAIT_AT_EVERY_BOUNDARY;
        let on = std::env::var(DOTS_VISION_WGPU_PHASES_ENV)
            .is_ok_and(|v| !v.is_empty() && v != "0");
        Self {
            at: on.then(std::time::Instant::now),
            rows: Vec::new(),
        }
    }

    pub fn rows(&self) -> &[(&'static str, f64, u64)] {
        &self.rows
    }

    pub fn total(&self) -> f64 {
        self.rows.iter().map(|r| r.1).sum()
    }

    fn mark(&mut self, g: &VisionGemm, name: &'static str) -> Result<()> {
        let Some(at) = self.at else {
            return Ok(());
        };
        g.drain()?;
        let dt = at.elapsed().as_secs_f64();
        self.at = Some(std::time::Instant::now());
        match self.rows.iter_mut().find(|r| r.0 == name) {
            Some(r) => {
                r.1 += dt;
                r.2 += 1;
            }
            None => self.rows.push((name, dt, 1)),
        }
        Ok(())
    }

    fn eprint(&self, wall: f64) {
        if self.at.is_none() {
            return;
        }
        let sum = self.total().max(1e-12);
        for (name, secs, calls) in &self.rows {
            eprintln!(
                "[dots-vision-phase] {name} {secs:.3}s over {calls} marks, {:.1}%",
                secs / sum * 100.0
            );
        }
        eprintln!(
            "[dots-vision-phase] drained sum {sum:.3}s inside a {wall:.3}s forward, drain cost {:.3}s",
            wall - sum
        );
    }
}

pub const DOTS_VISION_SOFTMAX_TRAFFIC_IS_UNBUYABLE_ON_THIS_ADAPTER_THREE_MECHANISMS_DIED_PAYING_FOR_IT: &str =
    "The deferred-norm softmax reads the f32 score slab twice and writes f16 probabilities once, \
     0.961s of the invoice tower by deleted work against a ~0.72s pure-traffic floor, and every \
     mechanism that halves the traffic pays more elsewhere on Apple M2 Ultra. (a) Exponentiating \
     inside the context GEMM's load is structurally closed: coopLoadT feeds the matrix unit with \
     no per-element hook. (b) An f16 score slab halves all three passes but coopStoreT emits \
     only f32, so the GEMM store staged every fragment through smem with scalar conversion \
     lanes: +38 percent on the scores dispatch (2.26 to 3.12 ms) and a 0.787s end-to-end loss \
     on a 6.42s wall. (c) Caching the scaled row in workgroup memory as f16 during the max pass \
     (27.2KB of the 32KB budget at 13600 cols) deletes the second read but leaves one resident \
     workgroup per core: the forward read 101.2s against 6.43s, a 15.7x occupancy collapse. \
     The two-read kernel stays; the ~0.24s between it and its floor is priced as unreachable.";

pub const DOTS_VISION_SPLITK_ENV: &str = "NV_DOTS_VISION_SPLITK";

pub const DOTS_VISION_SPLITK_SERVES_ONLY_ATTN_CTX_BECAUSE_IT_IS_THE_ONE_OCCUPANCY_STARVED_SHAPE_THE_BRACKET_NAMES: &str =
    "attn_ctx (m up to 8192, n 128, k 13600) is 30.1% of the invoice tower by deleted work and \
     runs 11.8 TFLOP/s as 256 workgroups of 1700 serial k steps -- the only shape whose column \
     width admits a single band. The split-k route is the default at S=16: S partial slabs in a \
     reused scratch, one elementwise sum into the y target, worth 0.603s of the 7.055s invoice \
     tower (fitted plan tm8_tn4_sg4, 15.59 TFLOP/s against the plain 10.8-11.8). The sum is a \
     reassociation, so the gate was token equality across the fixture corpus, never bits: all \
     twelve stable pages hold CER and WER at exactly 0.0000 and every bistable keeps its \
     standing trajectory and token count (0.2414/0.4544@414/0.4204@284/0.0622@174). \
     NV_DOTS_VISION_SPLITK=0 restores the plain single-dispatch kernel bit-for-bit; =<S> pins \
     another split count. The scratch is one buffer reused by every ctx dispatch in the queue, \
     safe because the reduce that reads slot s is queued before the next ctx write and the \
     backend orders dispatches on the same buffer.";

pub const DOTS_VISION_SPLITK_DEFAULT: u32 = 16;

fn splitk_arm() -> Option<u32> {
    match std::env::var(DOTS_VISION_SPLITK_ENV).ok().as_deref() {
        Some("0") | Some("off") => None,
        Some(v) => v
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|s| (2..=32).contains(s))
            .or(Some(DOTS_VISION_SPLITK_DEFAULT)),
        None => Some(DOTS_VISION_SPLITK_DEFAULT),
    }
}

pub const DOTS_VISION_SPLITK_FITTED_PLAN_IS_TM8_BECAUSE_THE_S16_LADDER_READS_15_59_TFLOPS_THERE_AGAINST_THE_SQUARE_PLANS_11_9: CoopPlan =
    CoopPlan { tm: 8, tn: 4, sg: 4, ku: 1 };

pub const DOTS_VISION_STEEL_IS_THE_DEFAULT_WHERE_PASSTHROUGH_GRANTS_AND_ZMERGE_SERVES_EVERY_OTHER_ADAPTER: &str =
    "With NV_DOTS_VISION_WGPU_ATTN unset the tower probes for the steel fused arm at forward      time -- head_dim 128 and a built passthrough pipeline, capability-checked never assumed --      and serves zmerge wherever the probe fails, which is every adapter without      PASSTHROUGH_SHADERS. Flip basis, invoice page under the flock: zmerge 6.151s +-0.006 to      steel 4.519s +-0.001 (minus 26.5 percent, against the rival tower's own measured 5.31s);      parity closest of every arm to the cpu f32 twin (4.4427e-2 vs the zmerge default's own      4.5656e-2); corpus token gate green with twelve stable zeros exact, bistables on their      standing trajectories and token counts, mean CER 0.0736 byte-identical, page mean 3717.0      ms. NV_DOTS_VISION_WGPU_ATTN=zmerge and =slab are the unchanged escapes, and the      slab-pipeline instruments (ablation bracket, split-k arms) pin zmerge explicitly so their      rows keep their meaning.";

pub const DOTS_VISION_STEEL_GEMM_ENV: &str = "NV_DOTS_VISION_STEEL_GEMM";

pub const DOTS_VISION_STEEL_GEMM_DEFAULTS_ON_WHERE_THE_BUILT_PASSTHROUGH_PIPELINE_PROBE_PASSES_BECAUSE_ZERO_OF_5222400_MERGED_FEATURES_MOVED: &str =
    "Unset routes every servable F16 projection through the vendored steel GEMM (nt pattern, f32      output, real-row M -- see STEEL_GEMM_RIDES_THE_SAME_DRIVER_PATTERNS_THE_ATTENTION_PORT_PROVED)      at the ladder champion bm64_bn64_bk32_wm2_wn2: 1.25-1.40x over the coop incumbent on every      shape it can serve (~24 TFLOP/s vs coop's 17.4), invoice tower 4.586 -> 4.059s, corpus page      mean 2988.1 ms with the twelve stable fixtures exactly 0.0000. The gate class is bit equality,      not tokens: steel-nt and coop are the same ascending-k chain of 8x8 simdgroup MMAs with f32      accumulators, and 0 of 5,222,400 merged features differ in a single f32 bit. The probe is      gemm_steel itself, same class as the attention flip: it asks steel_gemm_pipelines for a BUILT      msl passthrough pipeline at forward time -- capability-checked, never assumed -- and answers      false so the coop kernel serves on any adapter that cannot build one, and per shape wherever      k_pad is not a multiple of the k-block (patch_proj k=784 keeps coop).      NV_DOTS_VISION_STEEL_GEMM=0 restores coop everywhere, trivially byte-identical; =1 pins the      fitted champion explicitly; a bm,bn,bk,wm,wn 5-tuple pins another ladder config.";

pub const DOTS_VISION_STEEL_GEMM_FITTED: (u32, u32, u32, u32, u32) = (64, 64, 32, 2, 2);

fn steel_gemm_cfg() -> Option<(u32, u32, u32, u32, u32)> {
    let _ = DOTS_VISION_STEEL_GEMM_DEFAULTS_ON_WHERE_THE_BUILT_PASSTHROUGH_PIPELINE_PROBE_PASSES_BECAUSE_ZERO_OF_5222400_MERGED_FEATURES_MOVED;
    let Ok(v) = std::env::var(DOTS_VISION_STEEL_GEMM_ENV) else {
        return Some(DOTS_VISION_STEEL_GEMM_FITTED);
    };
    let t = v.trim();
    if t.is_empty() || t == "0" || t == "off" {
        return None;
    }
    if t == "1" {
        return Some(DOTS_VISION_STEEL_GEMM_FITTED);
    }
    let parts: Vec<u32> = t.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    match parts.as_slice() {
        [bm, bn, bk, wm, wn] => Some((*bm, *bn, *bk, *wm, *wn)),
        _ => Some(DOTS_VISION_STEEL_GEMM_FITTED),
    }
}

pub const DOTS_VISION_ABLATE_ENV: &str = "NV_DOTS_VISION_ABLATE";

pub const VISION_ABLATION_DELETES_WORK_BECAUSE_THE_DRAINED_PHASE_TABLE_CARRIES_A_1_36X_DISTORTION_AND_A_DISTORTED_INSTRUMENT_HAS_INVERTED_A_PHASE_MAP: &str =
    "The drained phase table waits for device idle at every boundary, so its rows are upper \
     bounds whose sum overshoots the untimed forward, and the prefill bracket showed such an \
     instrument can invert a phase map outright. The honest share of a phase is what the wall \
     loses when its work is deleted: each arm here skips whole dispatches, leaving zero-filled \
     buffers whose downstream cost is value-independent -- wgpu zero-initialises storage, softmax \
     over zeros is finite, and every elementwise consumer is data-oblivious. Arms: scores and ctx \
     delete one attention GEMM each, softmax deletes the row-softmax dispatches, proj deletes the \
     five per-block projections; the wall is read at the merged-features readback and the \
     residual is what no arm accounts for. On the invoice page (13600 patches, 3 interleaved \
     reps, spreads under +-0.023s) the 7.077s tower prices as: attn scores 1.419s (20.1%), attn \
     ctx 2.130s (30.1%), softmax 0.994s (14.1%), block projections 1.850s (26.1%), residual \
     0.683s (9.6%) -- the drained table had projections largest, and the bracket inverts that.";

fn vis_ablated(part: &str) -> bool {
    let _ = VISION_ABLATION_DELETES_WORK_BECAUSE_THE_DRAINED_PHASE_TABLE_CARRIES_A_1_36X_DISTORTION_AND_A_DISTORTED_INSTRUMENT_HAS_INVERTED_A_PHASE_MAP;
    std::env::var(DOTS_VISION_ABLATE_ENV).ok().as_deref() == Some(part)
}

pub const DOTS_VISION_COOP_PLAN_IS_BIT_IDENTITY_PRESERVING_BECAUSE_TM_TN_SG_KU_MOVE_WHICH_BLOCK_OWNS_A_TILE_NEVER_THE_K_ORDER: &str =
    "gemm_coop_f16 accumulates over kt in one ascending loop whatever tm/tn/sg/ku are: tm and tn \
     choose how many fragments one subgroup owns, sg how many subgroups share a workgroup, ku how \
     many k-tiles are loaded before the multiply-adds run -- and the multiply-adds still execute in \
     ascending k. So a plan change moves work between blocks and never reorders a reduction, and \
     the tower's output must stay bit-identical across every plan in the sweep.";

pub const DOTS_VISION_COOP_ACCUMULATOR_DTYPE_IS_A_PROBE_SURFACE_BECAUSE_GEMM_COOP_F16_EMITS_ONLY_F32_C_FRAGMENTS: &str =
    "gemm_coop_f16::CoopGemm hardcodes CoopScalar::F32 for the C operand and writes coop_mat<f32, C> \
     into every entry it generates, so the f16/f16->f16 config this adapter advertises alongside \
     f16/f16->f32 has no route in the tree. An 8x8 fragment holds 64 values across a 32-lane \
     subgroup, so an f32 accumulator costs 2 lane dwords per fragment and an f16 one costs 1: \
     tm8_tn8 is 128 of the 128-dword budget accumulating in f32 and 64 of it accumulating in f16, \
     which is why the arm that doubles fragment-load intensity to 32 flop per byte spills on one \
     accumulator dtype and not the other. This surface exists to price that arm, not to serve it -- \
     an f16 C fragment can only be stored to an f16 y operand, and every consumer the tower has \
     downstream of a projection reads f32.";

pub const A_COOP_ACC_ARM_IS_TIMED_TO_A_READBACK_WHOSE_BYTE_COUNT_IS_A_MULTIPLE_OF_FOUR_BECAUSE_A_TWO_BYTE_COPY_IS_REFUSED_NOT_SLOW: &str =
    "A rate that does not end in a readback times submission rather than completion, so every arm \
     settles on one. wgpu's buffer-to-buffer copy demands a 4-byte multiple: settling an f16 output \
     on a single element asks for a 2-byte range and is refused outright, which drops the arm and \
     leaves a table whose missing rows read as an absent speedup rather than as a harness fault. \
     Two f16 elements is the smallest legal settle, and the probe asserts an f16 row exists so the \
     failure can never again be silent.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoopAcc {
    F32,
    F16,
}

impl CoopAcc {
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "accf32",
            Self::F16 => "accf16",
        }
    }

    fn wgsl(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
        }
    }

    pub fn lane_dwords(&self, tile: u32, tm: u32, tn: u32) -> u32 {
        let cells = tm * tn * tile * tile;
        match self {
            Self::F32 => cells / coop::SUBGROUP,
            Self::F16 => cells / (2 * coop::SUBGROUP),
        }
    }
}

pub fn coop_source_acc(tile: u32, tm: u32, tn: u32, sg: u32, ku: u32, acc: CoopAcc) -> String {
    use std::fmt::Write as _;
    let _ = DOTS_VISION_COOP_ACCUMULATOR_DTYPE_IS_A_PROBE_SURFACE_BECAUSE_GEMM_COOP_F16_EMITS_ONLY_F32_C_FRAGMENTS;
    let (t, wg, cw) = (tile, sg * coop::SUBGROUP, acc.wgsl());
    let cols = tile * tn * sg;
    let entry = coop_entry_acc(tile, tm, tn, sg, ku, acc);
    let mut b = String::new();
    writeln!(b, "alias CGA = coop_mat{t}x{t}<f16, A>;").unwrap();
    writeln!(b, "alias CGB = coop_mat{t}x{t}<f16, B>;").unwrap();
    writeln!(b, "alias CGC = coop_mat{t}x{t}<{cw}, C>;\n").unwrap();
    b.push_str("struct CoopGemmParams {\n    n_rows: u32,\n    k_elems: u32,\n    m_rows: u32,\n    blocks_n: u32,\n    y_stride: u32,\n    groups_x: u32,\n    pad0: u32,\n    pad1: u32,\n};\n\n");
    b.push_str("@group(0) @binding(0) var<storage, read> cg_w: array<f16>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> cg_x: array<f16>;\n");
    writeln!(
        b,
        "@group(0) @binding(2) var<storage, read_write> cg_y: array<{cw}>;"
    )
    .unwrap();
    b.push_str("@group(0) @binding(3) var<uniform> cg_p: CoopGemmParams;\n");
    writeln!(b, "@group(0) @binding(4) var<storage, read> cg_zero: array<{cw}>;\n").unwrap();
    writeln!(b, "@compute @workgroup_size({wg})").unwrap();
    writeln!(b, "fn {entry}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_index) lidx: u32\n) {\n");
    writeln!(b, "    let sgid = lidx / {}u;", coop::SUBGROUP).unwrap();
    b.push_str("    let block = wid.x + wid.y * cg_p.groups_x;\n");
    b.push_str("    let bm = block / cg_p.blocks_n;\n");
    b.push_str("    let bn = block - bm * cg_p.blocks_n;\n");
    writeln!(b, "    let m0 = bm * {}u;", t * tm).unwrap();
    writeln!(b, "    let n0 = bn * {cols}u + sgid * {}u;", t * tn).unwrap();
    b.push_str("    let kk = cg_p.k_elems;\n    let ys = cg_p.y_stride;\n");
    b.push_str("    let mrows = cg_p.m_rows;\n    let nrows = cg_p.n_rows;\n");
    for r in 0..tm {
        writeln!(b, "    let arow{r} = min(m0 + {}u, mrows - {t}u) * kk;", r * t).unwrap();
    }
    for j in 0..tn {
        writeln!(b, "    let bcol{j} = min(n0 + {}u, nrows - {t}u) * kk;", j * t).unwrap();
    }
    for r in 0..tm {
        for j in 0..tn {
            writeln!(b, "    var c{r}_{j} = coopLoadT<CGC>(&cg_zero[0], {t}u);").unwrap();
        }
    }
    writeln!(b, "    for (var kt = 0u; kt < kk; kt = kt + {}u) {{", t * ku).unwrap();
    for u in 0..ku {
        for r in 0..tm {
            writeln!(
                b,
                "        let a{u}_{r} = coopLoadT<CGA>(&cg_x[arow{r} + kt + {}u], kk);",
                u * t
            )
            .unwrap();
        }
        for j in 0..tn {
            writeln!(
                b,
                "        let b{u}_{j} = coopLoad<CGB>(&cg_w[bcol{j} + kt + {}u], kk);",
                u * t
            )
            .unwrap();
        }
    }
    for u in 0..ku {
        for r in 0..tm {
            for j in 0..tn {
                writeln!(
                    b,
                    "        c{r}_{j} = coopMultiplyAdd(a{u}_{r}, b{u}_{j}, c{r}_{j});"
                )
                .unwrap();
            }
        }
    }
    b.push_str("    }\n");
    for r in 0..tm {
        for j in 0..tn {
            writeln!(
                b,
                "    if (m0 + {}u < mrows && n0 + {}u < nrows) {{",
                r * t,
                j * t
            )
            .unwrap();
            writeln!(
                b,
                "        coopStoreT(c{r}_{j}, &cg_y[(m0 + {}u) * ys + n0 + {}u], ys);",
                r * t,
                j * t
            )
            .unwrap();
            b.push_str("    }\n");
        }
    }
    b.push_str("}\n");
    compose_enabled(&coop::ENABLES, &b)
}

pub fn coop_entry_acc(tile: u32, tm: u32, tn: u32, sg: u32, ku: u32, acc: CoopAcc) -> String {
    format!(
        "dv_gemm_coop{tile}_{}_tm{tm}_tn{tn}_sg{sg}_ku{ku}",
        acc.name()
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopPlan {
    pub tm: u32,
    pub tn: u32,
    pub sg: u32,
    pub ku: u32,
}

impl CoopPlan {
    pub fn name(&self) -> String {
        format!("tm{}_tn{}_sg{}_ku{}", self.tm, self.tn, self.sg, self.ku)
    }
}

pub const DOTS_VISION_COOP_PLAN_IS_A_SQUARE_FRAGMENT_TILE_BECAUSE_TM_PLUS_TN_OPERAND_LOADS_FEED_TM_TIMES_TN_MULTIPLY_ADDS: CoopPlan =
    CoopPlan {
        tm: 4,
        tn: 4,
        sg: 4,
        ku: 1,
    };

pub const DOTS_VISION_COOP_PLAN_FITTED_TM8_TN4_SG4_IS_THE_DEFAULT_WHEREVER_THE_OUTPUT_IS_WIDER_THAN_ONE_COLUMN_BAND: CoopPlan =
    CoopPlan {
        tm: 8,
        tn: 4,
        sg: 4,
        ku: 1,
    };

pub const DOTS_VISION_A_TALLER_ROW_BLOCK_ONLY_PAYS_WHERE_THERE_IS_COLUMN_PARALLELISM_TO_TRADE_AGAINST_IT: &str =
    "Doubling tm halves the row blocks and so halves the workgroup count. Every tower shape but \
     one buys that back in column parallelism, because its output is many column bands wide and \
     the grid keeps its width. attn_ctx is the exception and the reason a blanket flip loses: n \
     is head_dim, 128 against embed_dim's 1536, so tile*tn*sg already covers the ENTIRE output \
     width in one band and there is no column dimension left to trade against the halved rows -- \
     the taller block just runs half as many workgroups over twice the work. So the predicate is \
     not a shape name or a fitted table, which would go stale the moment an image resizes n_pad: \
     it is whether n_pad still exceeds one column band, which is the property that made the \
     difference and holds at every resolution.";

pub const DOTS_VISION_COOP_PLAN_REPLACES_THE_TILES_DERIVED_TM16_TN1_ON_EVERY_SHAPE_THIS_TOWER_ISSUES: &str =
    "CoopGemm::tiles(m, ACC_FRAGS) hands any m of at least 256 rows tm=16, tn=1 -- 16 accumulator \
     fragments in one column strip, which costs 17 fragment loads per 16 multiply-adds and pins the \
     whole 128-dword lane accumulator budget. The square tm=tn=4 owns the same 16 accumulators for \
     8 loads. Measured on this tower's own shapes it is 1.5x on the attention score GEMM, 2.0x on \
     the ctx GEMM after the column shrink, and 1.9-2.1x on every projection; nothing in the ladder \
     preferred the tall tile. tiles() is fitted for qwen3.8 prefill and carries no row for any \
     dots.ocr vision shape.";

pub const DOTS_VISION_COOP_PLANS_ARE_FITTED_ON_THIS_TOWERS_OWN_SHAPES_BECAUSE_Q38_PF_ROUTE_CARRIES_NO_ROW_FOR_ANY_OF_THEM:
    &[(usize, usize, CoopPlan)] = &[];

pub const DOTS_VISION_COOP_PLAN_TM8_TN4_IS_FASTER_ON_EVERY_SHAPE_BUT_ATTN_CTX_WHERE_IT_IS_A_LOSS_SO_A_BLANKET_FLIP_TAKES_THAT_LOSS_SILENTLY: &str =
    "Whoever moves the default plan off tm4_tn4_sg4 must do it per shape and must leave attn_ctx \
     alone. Measured on an 8x8x8 Metal adapter at the full-resolution operating point (13600 \
     patches, m=13600 except the two attention shapes at m=8192), tm8_tn4_sg4 against the shipped \
     tm4_tn4_sg4: qkv 12.523 -> 11.252 ms 1.11x, mlp fc1/fc3 11.488 -> 10.343 1.11x, mlp fc2 \
     11.605 -> 10.617 1.09x, attn_proj 4.266 -> 3.905 1.09x, merger_fc0 17.065 -> 15.611 1.09x, \
     attn_scores 2.095 -> 1.825 1.15x, and attn_ctx 2.405 -> 2.635, which is 0.91x AND THE ONE \
     SHAPE THAT MUST KEEP THE SHIPPED PLAN -- n=head_dim against k=n_pad is the tower's only \
     narrow-n GEMM and a taller row block costs it more than the column band saves. Installed on \
     the seven winners the tower's full-resolution forward goes 8.688 -> 8.284 s, 1.049x, and the \
     page is bit-equal at 0 of 1267200 merged features, because tm/tn/sg/ku move which block owns \
     a tile and never the k order. The harness arm is NV_DOTS_VISION_FITTED_PLAN in \
     tests/dots_ocr_vision_phase_profile.rs and the bit-equality suite is \
     the_fitted_coop_plan_moves_which_block_owns_a_tile_and_never_the_k_order_so_the_page_is_bit_equal. \
     A blanket flip is not the win: it is the win minus a 0.91x regression on 29 percent of the \
     tower.";

pub const DOTS_VISION_W4A16_KU_CEILING_IS_FOUR_BECAUSE_SOURCE_WQ16_STAGES_AT_MOST_FOUR_K_TILES: u32 = 4;

pub const DOTS_VISION_COOP_COLUMN_BAND_SHRINKS_TO_FIT_N_BECAUSE_SUBGROUPS_PAST_COLUMN_N_STILL_WALK_THE_WHOLE_K_LOOP: &str =
    "A workgroup covers tile*tn*sg columns whatever n is, and the subgroups whose band starts past \
     n do not exit -- they run the same k loop, and gemm_coop_f16 guards its stores but not its \
     loads, so each of their coopLoad calls strides 16 rows from a pointer naga's index Restrict \
     has clamped to the end of the weight buffer. So tn, then sg, halve until the band fits n.";

pub const DOTS_VISION_COOP_OUT_OF_RANGE_WEIGHT_READ_BUDGET_BYTES: usize = 1 << 20;

pub const DOTS_VISION_COOP_ROW_BAND_OVERSHOOTS_M_THE_SAME_WAY_THE_COLUMN_BAND_OVERSHOOTS_N: &str =
    "The column budget above covers only the weight operand. gemm_coop_f16 issues \
     ceil(m_pad / (tile*tm)) row blocks and loads an A fragment for every row of every one of \
     them, guarding only the epilogue store, so a m_pad that is not a multiple of tile*tm makes \
     the last block read up to tile*tm-1 rows past the x operand. That over-read is harmless \
     while it lands in mapped memory and it wedges the adapter when it does not: an x operand \
     larger than gpu-allocator's 256 MiB device memblock gets a VkDeviceMemory sized to the \
     buffer, so the overshoot leaves the allocation outright. The first forward of a process \
     survives it only because the weight-upload staging blocks still map the pages behind it \
     until the first device poll retires them, which is why the failure is always the second \
     page and never the first. Every x operand therefore carries a whole row block of slack.";

pub const DOTS_VISION_COOP_ROW_BLOCK_CEILING_ROWS: usize = 256;

pub const DOTS_VISION_COOP_ROW_BLOCK_CEILING_IS_THE_TALLEST_TILE_THE_PLAN_LADDER_ASKS_FOR: &str =
    "tile*tm for the tallest plan the ladder sweeps, tm=16 on a 16x16x16 adapter. It is the unit \
     every x operand's slack is rounded to, so resolve_plan refuses any plan whose row block \
     exceeds it rather than letting an unbudgeted overshoot reach the driver.";

pub const DOTS_VISION_COOP_OUT_OF_RANGE_WEIGHT_READ_IS_BUDGETED_IN_BYTES_NOT_IN_WASTED_COLUMNS: &str =
    "The cost of reading past the weight buffer is not proportional to the wasted columns, it is a \
     step: 384 KB of over-read on the 4224-wide MLP GEMM is the fastest arm in the ladder, while \
     6 MB of over-read on the 1536-wide merger GEMM ran orders slower -- minutes where its \
     neighbours took milliseconds -- which is what a suballocated buffer's overshoot leaving mapped \
     memory looks like. So the bar is the over-read in bytes, (covered_columns - n) * k * 2, and a \
     32-column overshoot on a 96-wide probe operand is 8 KB and fine.";

pub const DOTS_VISION_GEMM_LADDER_OPERANDS_ARE_DEVICE_ZEROS_BECAUSE_A_COOPERATIVE_MATRIX_RATE_IS_NOT_DATA_DEPENDENT: &str =
    "time_gemm_plans allocates its operands with storage_zeroed and never fills them from the host: \
     a coop_mat multiply-add costs the same on zeros as on weights, the byte traffic is identical, \
     and filling the 928 MB probability operand of the attention context shape from the host would \
     cost more than every arm of the ladder put together.";

pub const DOTS_VISION_FLASH_ACCUMULATOR_IS_SPLIT_ACROSS_SUBGROUPS_BY_HEAD_DIM_BECAUSE_ONE_SUBGROUP_CANNOT_HOLD_A_BM_BY_128_F32_TILE: &str =
    "A 16x16 f32 accumulator fragment costs 8 dwords of every lane's 128-dword budget, so a 64x128 \
     output block is 32 fragments and 256 dwords -- twice what the generator will honour. The head \
     dimension is therefore the axis the subgroups divide: each subgroup owns hd/sg columns of O \
     for every row of the block, which is why the O split is columns-only and the S split is the \
     only one that carries a row factor.";

pub const DOTS_VISION_FLASH_RESCALE_IS_SKIPPED_WHEN_ALPHA_IS_EXACTLY_ONE_BECAUSE_MULTIPLYING_BY_ONE_IS_A_NO_OP_IN_F32: &str =
    "The online rescale exp(m_old - m_new) is exactly 1.0 on every key block that does not raise a \
     row's running maximum, and f32 multiplication by 1.0 returns the operand unchanged, so \
     skipping the whole store-scale-reload round trip on those blocks computes the identical \
     function. The running maximum of a row stops moving after a handful of blocks, so the branch \
     is taken on the order of ln(blocks) times out of blocks_n. The flag is published through \
     workgroupUniformLoad so the barriers inside the branch stay in uniform control flow.";

pub const DOTS_VISION_FLASH_STAGES_THE_RESCALE_THROUGH_THE_SCORE_TILE_BECAUSE_IT_IS_DEAD_BY_THEN: &str =
    "The bm x bn f32 score tile has already been consumed into the f16 probability tile by the time \
     a rescale runs, so it is the staging area for the accumulator round trip and no second \
     workgroup array is declared. When bn < head_dim the block's rows are rescaled in \
     ceil(head_dim / bn) chunks so the staging never exceeds bm*bn floats.";

pub const DOTS_VISION_FLASH_READS_PAST_A_HEADS_SLAB_ON_THE_RAGGED_LAST_BLOCK_SO_THE_QKV_BUFFERS_CARRY_SLACK: usize =
    1 << 15;

pub const EVERY_WORKGROUP_TILE_THIS_KERNEL_COOP_LOADS_MUST_START_ON_A_16_BYTE_BOUNDARY_WHICH_MAKING_IT_AN_ARRAY_DOES_NOT_BY_ITSELF_ACHIEVE: &str =
    "On naga 30.0.0 / wgpu 30.0.0 / NVIDIA 590.48.01, coopLoadT/coopLoad/coopStoreT against a \
     `var<workgroup>` pointer truncate the byte address DOWN to a 16-byte boundary. A workgroup \
     variable sits at the total size of everything declared after it, so a coop-loaded tile is \
     safe only when that total is a multiple of 16. A trailing `u32` or `atomic<u32>` costs 4 \
     bytes and shifts the tile two f16 early: measured against an identity value matrix, one \
     `atomic<u32>` plus one `u32` dropped key columns 0..3 of every block for query row 0 and the \
     tower read 11 percent off an f64 oracle while still producing finite, right-magnitude \
     features. `array<u32, 1>` DOES NOT FIX IT -- it is still 4 bytes and shifts identically. \
     This kernel is correct because `fa_need` is array<u32, WORKGROUP_SIZE> and `fa_needv` is \
     array<u32, 4>, which is exactly 16 bytes; shrinking fa_needv to 1..3 elements silently \
     reintroduces the corruption. f16 is not special, a coop store into a workgroup f32 array \
     shifts the same way, and a uniform-derived stride argument can make the same misaligned load \
     read correctly, which is a false-negative trap when comparing two similar kernels. Plain \
     indexed access is unaffected. Probe: nv-kernels/tests/wgpu_workgroup_scalar_alias.rs.";

pub const FA_NEEDV_ELEMS_IS_FOUR_BECAUSE_FOUR_U32_IS_EXACTLY_THE_16_BYTE_COOP_LOAD_ALIGNMENT_QUANTUM: usize = 4;

const _: () = assert!(
    (FA_NEEDV_ELEMS_IS_FOUR_BECAUSE_FOUR_U32_IS_EXACTLY_THE_16_BYTE_COOP_LOAD_ALIGNMENT_QUANTUM * 4)
        % 16
        == 0,
    "fa_needv must be a whole number of 16-byte units or the probability tile it precedes is coop-loaded from a truncated address"
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashPlan {
    pub bm: usize,
    pub bn: usize,
    pub sg: usize,
    pub scs: usize,
}

pub const DOTS_VISION_FLASH_PLAN_DEFAULT_IS_THE_WIDEST_KEY_BAND_A_49152_BYTE_WORKGROUP_ADMITS_AT_64_QUERY_ROWS: FlashPlan =
    FlashPlan {
        bm: 64,
        bn: 96,
        sg: 4,
        scs: 2,
    };

pub const DOTS_VISION_FLASH_IS_MEASURED_SLOWER_THAN_THE_SLAB_ON_AN_8X8_FRAGMENT_ADAPTER_AND_THE_WORKGROUP_BUDGET_IS_NOT_WHY: &str =
    "On Apple M2 Ultra [Metal], route coop_mat8x8_f16, the fused arm loses to the slab at every \
     plan that fits: attention 0.882 s against 0.762 s at 3300 patches and 10.466 s against \
     5.823 s at 13600. Two independent reasons, so it does not reopen from either direction. \
     First the budget is not the constraint: the default bm64_bn96_sg4_scs2 needs 38672 bytes \
     against this adapter's 32768, and that total is 63.6 percent fa_s and 31.8 percent fa_pr \
     which is ALREADY f16, leaving 1808 bytes in everything else -- so no dtype change outside \
     fa_s can close a 5904 byte gap, and fa_s cannot become f16 because it is also the online \
     accumulator's rescale stage and the epilogue stage, which would round O every rescale and \
     the output before the softmax division, 42 layers deep, besides being a type error against \
     coop_mat<f32, C>. Splitting those roles costs 4096 bytes MORE at the default nchunk. Second \
     the payoff is not there either: bm is the sensitive axis and bn is flat -- at bm32, bn 64 to \
     96 moved 0.3 percent -- and bm64_sg4 already fits, so the unreachable default differs from \
     the best fitting plan only along the axis that does not matter. UNRESOLVED, and worth \
     answering before anyone builds a fused kernel here: why the gap WIDENS with n (1.16x at 3300 \
     patches, 1.80x at 13600) while the slab's own GEMMs get more efficient at scale. Barrier \
     count is the natural suspect at roughly 30k rounds per head per layer, but widening bn cuts \
     blocks_n by 1.5x and bought 0.3 percent, which cuts against it.";

impl FlashPlan {
    pub fn name(&self) -> String {
        format!("bm{}_bn{}_sg{}_scs{}", self.bm, self.bn, self.sg, self.scs)
    }

    pub fn from_env() -> Self {
        let base =
            DOTS_VISION_FLASH_PLAN_DEFAULT_IS_THE_WIDEST_KEY_BAND_A_49152_BYTE_WORKGROUP_ADMITS_AT_64_QUERY_ROWS;
        let Ok(v) = std::env::var(DOTS_VISION_WGPU_FLASH_PLAN_ENV) else {
            return base;
        };
        let f: Vec<usize> = v
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect();
        if f.len() != 4 {
            return base;
        }
        Self {
            bm: f[0],
            bn: f[1],
            sg: f[2],
            scs: f[3],
        }
    }

    pub fn geometry(&self, tile: usize, hd: usize) -> Result<FlashGeometry> {
        let _ = DOTS_VISION_FLASH_IS_MEASURED_SLOWER_THAN_THE_SLAB_ON_AN_8X8_FRAGMENT_ADAPTER_AND_THE_WORKGROUP_BUDGET_IS_NOT_WHY;
        let t = tile;
        anyhow::ensure!(
            t > 0 && self.bm % t == 0 && self.bn % t == 0 && hd % t == 0,
            "flash plan {}: bm, bn and head_dim {hd} must all be multiples of the {t}-wide fragment",
            self.name()
        );
        anyhow::ensure!(
            (1..=8).contains(&self.sg) && self.scs >= 1 && self.sg % self.scs == 0,
            "flash plan {}: sg must be 1..8 and the score column split must divide it",
            self.name()
        );
        let wg = self.sg * (coop::SUBGROUP as usize);
        anyhow::ensure!(
            wg % self.bm == 0,
            "flash plan {}: a {wg}-invocation workgroup must cover {} rows an equal number of \
             lanes each",
            self.name(),
            self.bm
        );
        let rm = self.bm / t;
        let cn = self.bn / t;
        let ht = hd / t;
        let srs = self.sg / self.scs;
        anyhow::ensure!(
            rm % srs == 0 && cn % self.scs == 0 && ht % self.sg == 0,
            "flash plan {}: the {rm}x{cn} score tile grid and the {ht} head-dim tiles must divide \
             across {} subgroups",
            self.name(),
            self.sg
        );
        let nchunk = hd.div_ceil(self.bn).max(1);
        anyhow::ensure!(
            rm % nchunk == 0,
            "flash plan {}: {}",
            self.name(),
            DOTS_VISION_FLASH_STAGES_THE_RESCALE_THROUGH_THE_SCORE_TILE_BECAUSE_IT_IS_DEAD_BY_THEN
        );
        let g = FlashGeometry {
            tile: t,
            hd,
            wg,
            rm,
            cn,
            ht,
            srm: rm / srs,
            scn: cn / self.scs,
            ocn: ht / self.sg,
            lr: wg / self.bm,
            nchunk,
            rpc: self.bm / nchunk,
            tpc: rm / nchunk,
            plan: *self,
        };
        let _ = DOTS_VISION_FLASH_ACCUMULATOR_IS_SPLIT_ACROSS_SUBGROUPS_BY_HEAD_DIM_BECAUSE_ONE_SUBGROUP_CANNOT_HOLD_A_BM_BY_128_F32_TILE;
        anyhow::ensure!(
            g.acc_lane_dwords() <= coop::ACC_LANE_DWORD_BUDGET as usize,
            "flash plan {}: {} accumulator dwords per lane exceeds the {} the generator honours",
            self.name(),
            g.acc_lane_dwords(),
            coop::ACC_LANE_DWORD_BUDGET
        );
        Ok(g)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FlashGeometry {
    pub tile: usize,
    pub hd: usize,
    pub wg: usize,
    pub rm: usize,
    pub cn: usize,
    pub ht: usize,
    pub srm: usize,
    pub scn: usize,
    pub ocn: usize,
    pub lr: usize,
    pub nchunk: usize,
    pub rpc: usize,
    pub tpc: usize,
    pub plan: FlashPlan,
}

impl FlashGeometry {
    pub fn acc_lane_dwords(&self) -> usize {
        (self.srm * self.scn + self.rm * self.ocn) * self.tile * self.tile
            / (coop::SUBGROUP as usize)
    }

    pub fn workgroup_bytes(&self) -> usize {
        let cells = self.plan.bm * self.plan.bn;
        cells * 4 + cells * 2 + self.plan.bm * 3 * 4 + self.wg * 8 + 16
    }

    pub fn entry(&self) -> String {
        format!(
            "dv_flash_t{}_hd{}_{}",
            self.tile,
            self.hd,
            self.plan.name()
        )
    }

    pub fn source(&self) -> String {
        use std::fmt::Write as _;
        let (t, hd, wg) = (self.tile, self.hd, self.wg);
        let (bm, bn, scs) = (self.plan.bm, self.plan.bn, self.plan.scs);
        let (rm, srm, scn, ocn, lr) = (self.rm, self.srm, self.scn, self.ocn, self.lr);
        let cells = bm * bn;
        let entry = self.entry();
        let mut b = String::new();
        writeln!(b, "alias FQA = coop_mat{t}x{t}<f16, A>;").unwrap();
        writeln!(b, "alias FQB = coop_mat{t}x{t}<f16, B>;").unwrap();
        writeln!(b, "alias FQC = coop_mat{t}x{t}<f32, C>;\n").unwrap();
        b.push_str("struct FlashP { n: u32, n_pad: u32, heads: u32, hd: u32, blocks_m: u32, blocks_n: u32, groups_x: u32, scale: f32 };\n");
        b.push_str("@group(0) @binding(0) var<storage, read> fa_q: array<f16>;\n");
        b.push_str("@group(0) @binding(1) var<storage, read> fa_k: array<f16>;\n");
        b.push_str("@group(0) @binding(2) var<storage, read> fa_v: array<f16>;\n");
        b.push_str("@group(0) @binding(3) var<storage, read_write> fa_o: array<f16>;\n");
        b.push_str("@group(0) @binding(4) var<uniform> fa_p: FlashP;\n");
        b.push_str("@group(0) @binding(5) var<storage, read> fa_zero: array<f32>;\n\n");
        writeln!(b, "var<workgroup> fa_s: array<f32, {cells}>;").unwrap();
        writeln!(b, "var<workgroup> fa_pr: array<f16, {cells}>;").unwrap();
        writeln!(b, "var<workgroup> fa_mx: array<f32, {bm}>;").unwrap();
        writeln!(b, "var<workgroup> fa_ln: array<f32, {bm}>;").unwrap();
        writeln!(b, "var<workgroup> fa_al: array<f32, {bm}>;").unwrap();
        writeln!(b, "var<workgroup> fa_red: array<f32, {wg}>;").unwrap();
        let _ = EVERY_WORKGROUP_TILE_THIS_KERNEL_COOP_LOADS_MUST_START_ON_A_16_BYTE_BOUNDARY_WHICH_MAKING_IT_AN_ARRAY_DOES_NOT_BY_ITSELF_ACHIEVE;
        writeln!(b, "var<workgroup> fa_need: array<u32, {wg}>;").unwrap();
        b.push_str(&format!("var<workgroup> fa_needv: array<u32, {}>;\n\n", FA_NEEDV_ELEMS_IS_FOUR_BECAUSE_FOUR_U32_IS_EXACTLY_THE_16_BYTE_COOP_LOAD_ALIGNMENT_QUANTUM));
        writeln!(b, "@compute @workgroup_size({wg})").unwrap();
        writeln!(b, "fn {entry}(").unwrap();
        b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_index) lidx: u32\n) {\n");
        b.push_str("    let block = wid.x + wid.y * fa_p.groups_x;\n");
        b.push_str("    let h = block / fa_p.blocks_m;\n");
        b.push_str("    if (h >= fa_p.heads) { return; }\n");
        b.push_str("    let bmi = block - h * fa_p.blocks_m;\n");
        writeln!(b, "    let m0 = bmi * {bm}u;").unwrap();
        b.push_str("    let np = fa_p.n_pad;\n");
        writeln!(b, "    let hdw = {hd}u;").unwrap();
        writeln!(b, "    let bnw = {bn}u;").unwrap();
        b.push_str("    let dim = fa_p.heads * hdw;\n");
        b.push_str("    let qbase = h * np * hdw;\n");
        b.push_str("    let vbase = h * hdw * np;\n");
        b.push_str("    let sgid = lidx / 32u;\n");
        writeln!(b, "    let sgr = sgid / {scs}u;").unwrap();
        writeln!(b, "    let sgc = sgid - sgr * {scs}u;").unwrap();
        for i in 0..srm {
            writeln!(b, "    let srow{i} = (sgr * {srm}u + {i}u) * {t}u;").unwrap();
        }
        for j in 0..scn {
            writeln!(b, "    let scol{j} = (sgc * {scn}u + {j}u) * {t}u;").unwrap();
        }
        for j in 0..ocn {
            writeln!(b, "    let ocol{j} = (sgid * {ocn}u + {j}u) * {t}u;").unwrap();
        }
        for i in 0..rm {
            for j in 0..ocn {
                writeln!(b, "    var o{i}_{j} = coopLoadT<FQC>(&fa_zero[0], {t}u);").unwrap();
            }
        }
        writeln!(
            b,
            "    for (var r = lidx; r < {bm}u; r = r + {wg}u) {{ fa_mx[r] = -1.0e30; fa_ln[r] = 0.0; }}"
        )
        .unwrap();
        writeln!(b, "    let row = lidx / {lr}u;").unwrap();
        writeln!(b, "    let sub = lidx - row * {lr}u;").unwrap();
        b.push_str("    let sbase = row * bnw;\n");
        b.push_str("    for (var kb = 0u; kb < fa_p.blocks_n; kb = kb + 1u) {\n");
        b.push_str("        let j0 = kb * bnw;\n");
        for i in 0..srm {
            for j in 0..scn {
                writeln!(b, "        var c{i}_{j} = coopLoadT<FQC>(&fa_zero[0], {t}u);").unwrap();
            }
        }
        writeln!(b, "        for (var kt = 0u; kt < hdw; kt = kt + {t}u) {{").unwrap();
        for i in 0..srm {
            writeln!(
                b,
                "            let qa{i} = coopLoadT<FQA>(&fa_q[qbase + (m0 + srow{i}) * hdw + kt], hdw);"
            )
            .unwrap();
        }
        for j in 0..scn {
            writeln!(
                b,
                "            let kf{j} = coopLoad<FQB>(&fa_k[qbase + (j0 + scol{j}) * hdw + kt], hdw);"
            )
            .unwrap();
        }
        for i in 0..srm {
            for j in 0..scn {
                writeln!(
                    b,
                    "            c{i}_{j} = coopMultiplyAdd(qa{i}, kf{j}, c{i}_{j});"
                )
                .unwrap();
            }
        }
        b.push_str("        }\n");
        b.push_str("        workgroupBarrier();\n");
        b.push_str("        fa_need[lidx] = 0u;\n");
        for i in 0..srm {
            for j in 0..scn {
                writeln!(b, "        let sv{i}_{j} = c{i}_{j};").unwrap();
                writeln!(b, "        let so{i}_{j} = srow{i} * bnw + scol{j};").unwrap();
                writeln!(b, "        coopStoreT(sv{i}_{j}, &fa_s[so{i}_{j}], bnw);").unwrap();
            }
        }
        b.push_str("        workgroupBarrier();\n");
        b.push_str("        var mxv = -1.0e30;\n");
        writeln!(
            b,
            "        for (var c = sub; c < bnw; c = c + {lr}u) {{ if (j0 + c < fa_p.n) {{ mxv = max(mxv, fa_s[sbase + c] * fa_p.scale); }} }}"
        )
        .unwrap();
        b.push_str("        fa_red[lidx] = mxv;\n        workgroupBarrier();\n");
        if lr > 1 {
            writeln!(b, "        if (sub == 0u) {{").unwrap();
            writeln!(b, "            var racc = fa_red[row * {lr}u];").unwrap();
            writeln!(
                b,
                "            for (var q = 1u; q < {lr}u; q = q + 1u) {{ racc = max(racc, fa_red[row * {lr}u + q]); }}"
            )
            .unwrap();
            writeln!(b, "            fa_red[row * {lr}u] = racc;").unwrap();
            b.push_str("        }\n        workgroupBarrier();\n");
        }
        writeln!(b, "        let bmax = fa_red[row * {lr}u];").unwrap();
        b.push_str("        let mold = fa_mx[row];\n");
        b.push_str("        let mnew = max(mold, bmax);\n");
        b.push_str("        let alpha = exp(mold - mnew);\n");
        b.push_str("        var zsum = 0.0;\n");
        writeln!(b, "        for (var c = sub; c < bnw; c = c + {lr}u) {{").unwrap();
        b.push_str("            var pv = 0.0;\n");
        b.push_str("            if (j0 + c < fa_p.n) { pv = exp(fa_s[sbase + c] * fa_p.scale - mnew); }\n");
        b.push_str("            fa_pr[sbase + c] = f16(pv);\n");
        b.push_str("            zsum = zsum + pv;\n        }\n");
        b.push_str("        workgroupBarrier();\n");
        b.push_str("        fa_red[lidx] = zsum;\n        workgroupBarrier();\n");
        b.push_str("        if (sub == 0u) {\n            var zacc = 0.0;\n");
        writeln!(
            b,
            "            for (var q = 0u; q < {lr}u; q = q + 1u) {{ zacc = zacc + fa_red[row * {lr}u + q]; }}"
        )
        .unwrap();
        b.push_str("            fa_mx[row] = mnew;\n");
        b.push_str("            fa_ln[row] = fa_ln[row] * alpha + zacc;\n");
        b.push_str("            fa_al[row] = alpha;\n");
        b.push_str("            if (alpha != 1.0) { fa_need[lidx] = 1u; }\n        }\n");
        b.push_str("        workgroupBarrier();\n");
        writeln!(
            b,
            "        if (lidx == 0u) {{ var nf = 0u; for (var q = 0u; q < {wg}u; q = q + 1u) {{ nf = nf | fa_need[q]; }} fa_needv[0] = nf; }}"
        )
        .unwrap();
        b.push_str("        let need = workgroupUniformLoad(&fa_needv[0]);\n");
        let _ = DOTS_VISION_FLASH_RESCALE_IS_SKIPPED_WHEN_ALPHA_IS_EXACTLY_ONE_BECAUSE_MULTIPLYING_BY_ONE_IS_A_NO_OP_IN_F32;
        b.push_str("        if (need != 0u) {\n");
        for ch in 0..self.nchunk {
            for i in (ch * self.tpc)..((ch + 1) * self.tpc) {
                for j in 0..ocn {
                    let off = (i - ch * self.tpc) * t;
                    writeln!(b, "            let rv{i}_{j} = o{i}_{j};").unwrap();
                    writeln!(b, "            let ro{i}_{j} = {off}u * hdw + ocol{j};").unwrap();
                    writeln!(b, "            coopStoreT(rv{i}_{j}, &fa_s[ro{i}_{j}], hdw);")
                        .unwrap();
                }
            }
            b.push_str("            workgroupBarrier();\n");
            writeln!(
                b,
                "            for (var e = lidx; e < {}u; e = e + {wg}u) {{ fa_s[e] = fa_s[e] * fa_al[{}u + e / hdw]; }}",
                self.rpc * hd,
                ch * self.rpc
            )
            .unwrap();
            b.push_str("            workgroupBarrier();\n");
            for i in (ch * self.tpc)..((ch + 1) * self.tpc) {
                for j in 0..ocn {
                    writeln!(
                        b,
                        "            o{i}_{j} = coopLoadT<FQC>(&fa_s[ro{i}_{j}], hdw);"
                    )
                    .unwrap();
                }
            }
            b.push_str("            workgroupBarrier();\n");
        }
        b.push_str("        }\n");
        writeln!(b, "        for (var jt = 0u; jt < bnw; jt = jt + {t}u) {{").unwrap();
        for i in 0..rm {
            writeln!(
                b,
                "            let pa{i} = coopLoadT<FQA>(&fa_pr[{}u * bnw + jt], bnw);",
                i * t
            )
            .unwrap();
        }
        for j in 0..ocn {
            writeln!(
                b,
                "            let vf{j} = coopLoad<FQB>(&fa_v[vbase + ocol{j} * np + j0 + jt], np);"
            )
            .unwrap();
        }
        for i in 0..rm {
            for j in 0..ocn {
                writeln!(
                    b,
                    "            o{i}_{j} = coopMultiplyAdd(pa{i}, vf{j}, o{i}_{j});"
                )
                .unwrap();
            }
        }
        b.push_str("        }\n    }\n");
        b.push_str("    workgroupBarrier();\n");
        for ch in 0..self.nchunk {
            for i in (ch * self.tpc)..((ch + 1) * self.tpc) {
                for j in 0..ocn {
                    let off = (i - ch * self.tpc) * t;
                    writeln!(b, "    let ev{i}_{j} = o{i}_{j};").unwrap();
                    writeln!(b, "    let eo{i}_{j} = {off}u * hdw + ocol{j};").unwrap();
                    writeln!(b, "    coopStoreT(ev{i}_{j}, &fa_s[eo{i}_{j}], hdw);").unwrap();
                }
            }
            b.push_str("    workgroupBarrier();\n");
            writeln!(
                b,
                "    for (var e = lidx; e < {}u; e = e + {wg}u) {{",
                self.rpc * hd
            )
            .unwrap();
            b.push_str("        let rr = e / hdw;\n");
            b.push_str("        let cc = e - rr * hdw;\n");
            writeln!(b, "        let lrow = {}u + rr;", ch * self.rpc).unwrap();
            b.push_str("        let grow = m0 + lrow;\n");
            b.push_str("        if (grow < fa_p.n) { fa_o[grow * dim + h * hdw + cc] = f16(fa_s[e] / fa_ln[lrow]); }\n");
            b.push_str("    }\n");
            b.push_str("    workgroupBarrier();\n");
        }
        b.push_str("}\n");
        compose_enabled(&coop::ENABLES, &b)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashP {
    n: u32,
    n_pad: u32,
    heads: u32,
    hd: u32,
    blocks_m: u32,
    blocks_n: u32,
    groups_x: u32,
    scale: f32,
}

pub const DOTS_VISION_M_ROW_FLASH_TRADES_QUERY_REUSE_FOR_NOT_MATERIALISING_SCORES_WHICH_IS_FREE_AT_SMALL_M_AND_COSTLY_AT_M_EQUALS_N: &str =
    "An M-row flash stages M query rows per workgroup and streams every key past them, so each \
     position costs one K row plus one V row of loads -- 2*hd*2 bytes -- to feed 2*M*hd*2 flops of \
     QK plus PV. Its fragment-load intensity is therefore exactly M flop per byte, against the \
     8*tm*tn/(tm+tn) the coop route gets: 16 at tm4_tn4 and 21.3 at tm8_tn4. That is why the \
     structure wins a decoder prefill, where m is 8 to 32 and a coop gemm's tm clamps to m/tile so \
     there is no query reuse to give up, and why it is not free for this tower, where m = n = the \
     patch count and the coop route collects that reuse. Break-even is M=16, and per-lane state is \
     about 7*M dwords -- M rows of accumulator at 4, plus running max, sum and score at one each -- \
     so M=16 sits at 112 of the 128-dword budget, which is where tm8_tn8 collapsed. This surface \
     prices that cliff rather than assuming which side of it M=16 lands on.";

pub fn flash_m_entry(m_rows: usize, hd: usize) -> String {
    format!("dv_flash_m{m_rows}_hd{hd}")
}

pub fn flash_m_source(m_rows: usize, hd: usize) -> String {
    use std::fmt::Write as _;
    let _ = DOTS_VISION_M_ROW_FLASH_TRADES_QUERY_REUSE_FOR_NOT_MATERIALISING_SCORES_WHICH_IS_FREE_AT_SMALL_M_AND_COSTLY_AT_M_EQUALS_N;
    let (m, g) = (m_rows, m_rows / 4);
    let cells = m * hd;
    let entry = flash_m_entry(m_rows, hd);
    let mut b = String::new();
    b.push_str("struct FmP { n: u32, n_pad: u32, heads: u32, tiles: u32, groups_x: u32, pad0: u32, pad1: u32, scale: f32 };\n");
    b.push_str("@group(0) @binding(0) var<storage, read> fm_q: array<f16>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> fm_k: array<f16>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read> fm_v: array<f16>;\n");
    b.push_str("@group(0) @binding(3) var<storage, read_write> fm_o: array<f16>;\n");
    b.push_str("@group(0) @binding(4) var<uniform> fm_p: FmP;\n\n");
    writeln!(b, "var<workgroup> fm_qs: array<f32, {cells}>;").unwrap();
    writeln!(b, "var<workgroup> fm_ac: array<f32, {cells}>;").unwrap();
    writeln!(b, "var<workgroup> fm_am: array<f32, {}>;", 9 * m).unwrap();
    writeln!(b, "var<workgroup> fm_az: array<f32, {}>;\n", 9 * m).unwrap();
    b.push_str("@compute @workgroup_size(256)\n");
    writeln!(b, "fn {entry}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_index) tid: u32\n) {\n");
    b.push_str("    let block = wid.x + wid.y * fm_p.groups_x;\n");
    b.push_str("    let h = block / fm_p.tiles;\n");
    b.push_str("    if (h >= fm_p.heads) { return; }\n");
    writeln!(b, "    let row0 = (block - h * fm_p.tiles) * {m}u;").unwrap();
    b.push_str("    if (row0 >= fm_p.n) { return; }\n");
    b.push_str("    let lane = tid & 31u;\n    let warp = tid >> 5u;\n");
    writeln!(b, "    let qbase = h * fm_p.n_pad * {hd}u;").unwrap();
    writeln!(
        b,
        "    for (var e = tid; e < {cells}u; e = e + 256u) {{\n        let r = e / {hd}u;\n        let qr = min(row0 + r, fm_p.n - 1u);\n        fm_qs[e] = f32(fm_q[qbase + qr * {hd}u + e - r * {hd}u]) * fm_p.scale;\n    }}"
    )
    .unwrap();
    b.push_str("    workgroupBarrier();\n    let d0 = lane * 4u;\n");
    for i in 0..g {
        writeln!(b, "    var mv{i} = vec4<f32>(-3.4028235e38);").unwrap();
        writeln!(b, "    var zv{i} = vec4<f32>(0.0);").unwrap();
        for c in 0..4 {
            writeln!(b, "    var av{i}_{c} = vec4<f32>(0.0);").unwrap();
        }
    }
    b.push_str("    for (var t = warp; t < fm_p.n; t = t + 8u) {\n");
    writeln!(b, "        let kb = qbase + t * {hd}u + d0;").unwrap();
    for c in 0..4 {
        writeln!(b, "        let kv{c} = f32(fm_k[kb + {c}u]);").unwrap();
    }
    for i in 0..g {
        writeln!(b, "        var sv{i} = vec4<f32>(0.0);").unwrap();
        for r in 0..4 {
            let qb = (i * 4 + r) * hd;
            writeln!(
                b,
                "        sv{i}[{r}] = fma(kv0, fm_qs[{qb}u + d0], fma(kv1, fm_qs[{qb}u + d0 + 1u], fma(kv2, fm_qs[{qb}u + d0 + 2u], kv3 * fm_qs[{qb}u + d0 + 3u])));"
            )
            .unwrap();
        }
        for s in [16u32, 8, 4, 2, 1] {
            writeln!(b, "        sv{i} = sv{i} + subgroupShuffleXor(sv{i}, {s}u);").unwrap();
        }
    }
    for c in 0..4 {
        writeln!(b, "        let vv{c} = f32(fm_v[kb + {c}u]);").unwrap();
    }
    for i in 0..g {
        writeln!(b, "        let nm{i} = max(mv{i}, sv{i});").unwrap();
        writeln!(b, "        let rs{i} = exp(mv{i} - nm{i});").unwrap();
        writeln!(b, "        let ex{i} = exp(sv{i} - nm{i});").unwrap();
        writeln!(b, "        zv{i} = fma(zv{i}, rs{i}, ex{i});").unwrap();
        for c in 0..4 {
            writeln!(b, "        av{i}_{c} = fma(av{i}_{c}, rs{i}, ex{i} * vv{c});").unwrap();
        }
        writeln!(b, "        mv{i} = nm{i};").unwrap();
    }
    b.push_str("    }\n");
    b.push_str("    if (lane == 0u) {\n");
    for i in 0..g {
        for r in 0..4 {
            writeln!(
                b,
                "        fm_am[warp * {m}u + {}u] = mv{i}[{r}];",
                i * 4 + r
            )
            .unwrap();
            writeln!(
                b,
                "        fm_az[warp * {m}u + {}u] = zv{i}[{r}];",
                i * 4 + r
            )
            .unwrap();
        }
    }
    b.push_str("    }\n    workgroupBarrier();\n");
    writeln!(
        b,
        "    if (tid < {m}u) {{\n        var gm = -3.4028235e38;\n        for (var w = 0u; w < 8u; w = w + 1u) {{ gm = max(gm, fm_am[w * {m}u + tid]); }}\n        var gz = 0.0;\n        for (var w = 0u; w < 8u; w = w + 1u) {{ gz = gz + fm_az[w * {m}u + tid] * exp(fm_am[w * {m}u + tid] - gm); }}\n        fm_am[{}u + tid] = gm;\n        fm_az[{}u + tid] = gz;\n    }}",
        8 * m,
        8 * m
    )
    .unwrap();
    b.push_str("    workgroupBarrier();\n");
    writeln!(
        b,
        "    for (var e = tid; e < {cells}u; e = e + 256u) {{ fm_ac[e] = 0.0; }}"
    )
    .unwrap();
    b.push_str("    workgroupBarrier();\n");
    b.push_str("    for (var w = 0u; w < 8u; w = w + 1u) {\n        if (warp == w) {\n");
    for i in 0..g {
        writeln!(
            b,
            "            let sc{i} = exp(mv{i} - vec4<f32>(fm_am[{}u + {}u], fm_am[{}u + {}u], fm_am[{}u + {}u], fm_am[{}u + {}u]));",
            8 * m, i * 4, 8 * m, i * 4 + 1, 8 * m, i * 4 + 2, 8 * m, i * 4 + 3
        )
        .unwrap();
        for r in 0..4 {
            for c in 0..4 {
                writeln!(
                    b,
                    "            fm_ac[{}u * {hd}u + d0 + {c}u] = fm_ac[{}u * {hd}u + d0 + {c}u] + av{i}_{c}[{r}] * sc{i}[{r}];",
                    i * 4 + r,
                    i * 4 + r
                )
                .unwrap();
            }
        }
    }
    b.push_str("        }\n        workgroupBarrier();\n    }\n");
    writeln!(
        b,
        "    for (var e = tid; e < {cells}u; e = e + 256u) {{\n        let r = e / {hd}u;\n        if (row0 + r < fm_p.n) {{\n            fm_o[(row0 + r) * fm_p.heads * {hd}u + h * {hd}u + e - r * {hd}u] = f16(fm_ac[e] / fm_az[{}u + r]);\n        }}\n    }}",
        8 * m
    )
    .unwrap();
    b.push_str("}\n");
    compose_enabled(&["f16"], &b)
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FmP {
    n: u32,
    n_pad: u32,
    heads: u32,
    tiles: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VisionGemmRoute {
    Coop(coop::CoopGemm),
    ScalarF16,
}

impl VisionGemmRoute {
    pub fn tile(&self) -> usize {
        match self {
            Self::Coop(g) => g.tile as usize,
            Self::ScalarF16 => 1,
        }
    }

    pub fn name(&self) -> String {
        match self {
            Self::Coop(g) => format!("coop_mat{}x{}_f16", g.tile, g.tile),
            Self::ScalarF16 => "scalar_f16".to_string(),
        }
    }

    pub fn is_coop(&self) -> bool {
        matches!(self, Self::Coop(_))
    }

    pub fn from_env_and_adapter(ctx: &WgpuContext) -> Self {
        if matches!(
            std::env::var(DOTS_VISION_WGPU_GEMM_ENV).ok().as_deref(),
            Some("scalar")
        ) {
            return Self::ScalarF16;
        }
        match coop::select(ctx, coop::Operand::F16) {
            Ok(g) => Self::Coop(g),
            Err(why) => {
                eprintln!(
                    "[dots-vision-wgpu] no cooperative-matrix GEMM ({why}); \
                     falling back to dv_gemm_f16_scalar"
                );
                Self::ScalarF16
            }
        }
    }
}

pub fn pad_to(v: usize, tile: usize) -> usize {
    let t = tile.max(1);
    v.div_ceil(t) * t
}

fn f16_bits_padded(v: &[f32], rows: usize, k: usize, k_pad: usize) -> Vec<u16> {
    if k == k_pad {
        return v.iter().map(|&x| f16::from_f32(x).to_bits()).collect();
    }
    let mut out = vec![0u16; rows * k_pad];
    for r in 0..rows {
        for c in 0..k {
            out[r * k_pad + c] = f16::from_f32(v[r * k + c]).to_bits();
        }
    }
    out
}

fn le_words(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

pub fn nvfp4_block16_rows_padded(
    v: &[f32],
    rows: usize,
    k: usize,
    k_pad: usize,
) -> (Vec<u32>, Vec<u32>, nv_quant::nvfp4::Nvfp4Tensor) {
    let padded: Vec<Vec<f32>> = (0..rows)
        .map(|r| {
            let mut row = vec![0f32; k_pad];
            row[..k].copy_from_slice(&v[r * k..(r + 1) * k]);
            row
        })
        .collect();
    let _ = DOTS_VISION_NVFP4_STAGING_USES_THE_CLOSED_FORM_SCALE_SEARCH_BECAUSE_THE_KERNEL_APPLIES_NO_GLOBAL_SCALE;
    let q = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global_closed_form(&padded, 1.0);
    (le_words(&q.data), le_words(&q.scales), q)
}

pub struct MlxW4Rows {
    pub words: Vec<u32>,
    pub scales_bf16: Vec<u16>,
    pub biases_bf16: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub gs: usize,
}

fn bf16_round(v: f32) -> f32 {
    f32::from_bits((half::bf16::from_f32(v).to_bits() as u32) << 16)
}

pub const MLX_W4_QUANTIZE_REQUANTIZES_AGAINST_THE_BF16_ROUNDED_SCALE_AND_BIAS_BECAUSE_THOSE_ARE_THE_VALUES_THE_DEQUANT_WILL_USE: &str =
    "mlx stores affine scales and biases as bf16, so the nibble that minimizes reconstruction \
     error is the one chosen against the rounded pair, not the f32 pair the group statistics \
     produce. A group whose values are all equal gets scale 0 and rides entirely on the bias.";

pub fn mlx_w4_quantize_rows(w: &[f32], n: usize, k: usize, gs: usize) -> MlxW4Rows {
    assert!(k.is_multiple_of(gs) && gs.is_multiple_of(8), "mlx w4 wants k {k} divisible by gs {gs} divisible by 8");
    assert_eq!(w.len(), n * k, "mlx w4 quantize wants an n*k row-major weight");
    let _ = MLX_W4_QUANTIZE_REQUANTIZES_AGAINST_THE_BF16_ROUNDED_SCALE_AND_BIAS_BECAUSE_THOSE_ARE_THE_VALUES_THE_DEQUANT_WILL_USE;
    let groups = k / gs;
    let words_per_row = k / 8;
    let mut words = vec![0u32; n * words_per_row];
    let mut scales = vec![0u16; n * groups];
    let mut biases = vec![0u16; n * groups];
    let lanes = std::thread::available_parallelism().map_or(4, |p| p.get()).min(12);
    let rows_per_chunk = n.div_ceil(lanes).max(1);
    let mut chunk_starts = Vec::new();
    let mut r0 = 0usize;
    while r0 < n {
        chunk_starts.push(r0);
        r0 += rows_per_chunk;
    }
    std::thread::scope(|sc| {
        for ((wc, sc_c), (bc, &row0)) in words
            .chunks_mut(rows_per_chunk * words_per_row)
            .zip(scales.chunks_mut(rows_per_chunk * groups))
            .zip(biases.chunks_mut(rows_per_chunk * groups).zip(chunk_starts.iter()))
        {
            sc.spawn(move || {
                for (ri, wrow) in wc.chunks_mut(words_per_row).enumerate() {
                    let row = row0 + ri;
                    let src = &w[row * k..(row + 1) * k];
                    for g in 0..groups {
                        let grp = &src[g * gs..(g + 1) * gs];
                        let amin = grp.iter().copied().fold(f32::INFINITY, f32::min);
                        let amax = grp.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let s = bf16_round((amax - amin) / 15.0);
                        let b = bf16_round(amin);
                        sc_c[ri * groups + g] = half::bf16::from_f32(s).to_bits();
                        bc[ri * groups + g] = half::bf16::from_f32(b).to_bits();
                        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                        for (e, &v) in grp.iter().enumerate() {
                            let q = (((v - b) * inv).round().clamp(0.0, 15.0)) as u32;
                            let idx = g * gs + e;
                            wrow[idx / 8] |= q << (4 * (idx % 8));
                        }
                    }
                }
            });
        }
    });
    MlxW4Rows {
        words,
        scales_bf16: scales,
        biases_bf16: biases,
        n,
        k,
        gs,
    }
}

pub fn mlx_w4_dequant_f16_bits_padded(q: &MlxW4Rows, k_pad: usize) -> Vec<u16> {
    let scales: Vec<f32> = q.scales_bf16.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let biases: Vec<f32> = q.biases_bf16.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
    let deq = nv_quant::mlx_affine::dequantize_mlx_affine_linear(&q.words, &scales, &biases, q.n, q.k, q.gs);
    f16_bits_padded(&deq, q.n, q.k, k_pad)
}

pub const DOTS_VISION_NVFP4_STAGING_USES_THE_CLOSED_FORM_SCALE_SEARCH_BECAUSE_THE_KERNEL_APPLIES_NO_GLOBAL_SCALE: &str =
    "gemm_coop_f16's w4a16 unpack multiplies each nibble by ue4m3_decode(scale_byte) and nothing \
     else, so the stored global must be 1.0 and the only per-block freedom is the scale byte \
     itself. quantize_rows_with_global_closed_form searches the ue4m3 grid around amax/6 for the \
     byte minimizing block reconstruction error, at load time and at no serving cost, where \
     quantize_rows takes the nearest byte unconditionally.";

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmP {
    n_rows: u32,
    k_pad: u32,
    m_rows: u32,
    y_stride: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RowsP {
    rows: u32,
    cols: u32,
    cols_pad: u32,
    groups_x: u32,
    scalar: f32,
    zbase: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitP {
    rows: u32,
    rows_pad: u32,
    heads: u32,
    head_dim: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const PRELUDE: &str = "
fn dv_linear_index(wid: vec3<u32>, lid: vec3<u32>, groups_x: u32) -> u32 {
    return (wid.x + wid.y * groups_x) * 256u + lid.x;
}
";

const CAST_PAD_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> cp_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cp_dst: array<f16>;
@group(0) @binding(2) var<uniform> cp_p: RowsP;

@compute @workgroup_size(256)
fn dv_cast_pad_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, cp_p.groups_x);
    if (i >= cp_p.rows * cp_p.cols) { return; }
    let r = i / cp_p.cols;
    let c = i - r * cp_p.cols;
    cp_dst[r * cp_p.cols_pad + c] = f16(cp_src[i]);
}
";

const RMSNORM_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> rn_x: array<f32>;
@group(0) @binding(1) var<storage, read> rn_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> rn_y: array<f16>;
@group(0) @binding(3) var<uniform> rn_p: RowsP;

var<workgroup> rn_red: array<f32, 256>;

@compute @workgroup_size(256)
fn dv_rmsnorm_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * rn_p.groups_x;
    if (row >= rn_p.rows) { return; }
    let base = row * rn_p.cols;
    var acc = 0.0;
    for (var c = lid.x; c < rn_p.cols; c = c + 256u) {
        let v = rn_x[base + c];
        acc = acc + v * v;
    }
    rn_red[lid.x] = acc;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) { rn_red[lid.x] = rn_red[lid.x] + rn_red[lid.x + stride]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(rn_red[0] / f32(rn_p.cols) + rn_p.scalar);
    let obase = row * rn_p.cols_pad;
    for (var c = lid.x; c < rn_p.cols; c = c + 256u) {
        rn_y[obase + c] = f16(rn_x[base + c] * inv * rn_w[c]);
    }
}
";

const RMSNORM_F32_INPLACE_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> ri_x: array<f32>;
@group(0) @binding(1) var<storage, read> ri_w: array<f32>;
@group(0) @binding(2) var<uniform> ri_p: RowsP;

var<workgroup> ri_red: array<f32, 256>;

@compute @workgroup_size(256)
fn dv_rmsnorm_f32_inplace(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * ri_p.groups_x;
    if (row >= ri_p.rows) { return; }
    let base = row * ri_p.cols;
    var acc = 0.0;
    for (var c = lid.x; c < ri_p.cols; c = c + 256u) {
        let v = ri_x[base + c];
        acc = acc + v * v;
    }
    ri_red[lid.x] = acc;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) { ri_red[lid.x] = ri_red[lid.x] + ri_red[lid.x + stride]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(ri_red[0] / f32(ri_p.cols) + ri_p.scalar);
    for (var c = lid.x; c < ri_p.cols; c = c + 256u) {
        ri_x[base + c] = ri_x[base + c] * inv * ri_w[c];
    }
}
";

const GEMM_SCALAR_SRC: &str = "
struct GemmP { n_rows: u32, k_pad: u32, m_rows: u32, y_stride: u32, groups_x: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> gs_w: array<f16>;
@group(0) @binding(1) var<storage, read> gs_x: array<f16>;
@group(0) @binding(2) var<storage, read_write> gs_y: array<f32>;
@group(0) @binding(3) var<uniform> gs_p: GemmP;

@compute @workgroup_size(256)
fn dv_gemm_f16_scalar(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = dv_linear_index(wid, lid, gs_p.groups_x);
    if (idx >= gs_p.m_rows * gs_p.n_rows) { return; }
    let m = idx / gs_p.n_rows;
    let n = idx - m * gs_p.n_rows;
    let xo = m * gs_p.k_pad;
    let wo = n * gs_p.k_pad;
    var acc = 0.0;
    for (var k = 0u; k < gs_p.k_pad; k = k + 1u) {
        acc = acc + f32(gs_x[xo + k]) * f32(gs_w[wo + k]);
    }
    gs_y[m * gs_p.y_stride + n] = acc;
}
";

const QKV_ROPE_SPLIT_SRC: &str = "
struct SplitP { rows: u32, rows_pad: u32, heads: u32, head_dim: u32, groups_x: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> qs_qkv: array<f32>;
@group(0) @binding(1) var<storage, read> qs_cos: array<f32>;
@group(0) @binding(2) var<storage, read> qs_sin: array<f32>;
@group(0) @binding(3) var<storage, read_write> qs_q: array<f16>;
@group(0) @binding(4) var<storage, read_write> qs_k: array<f16>;
@group(0) @binding(5) var<storage, read_write> qs_vt: array<f16>;
@group(0) @binding(6) var<uniform> qs_p: SplitP;

fn qs_rope(base: u32, lane: u32, hd: u32, c: f32, s: f32) -> f32 {
    let half = hd >> 1u;
    let x = qs_qkv[base + lane];
    var rot = 0.0;
    if (lane < half) { rot = -qs_qkv[base + lane + half]; } else { rot = qs_qkv[base + lane - half]; }
    return x * c + rot * s;
}

@compute @workgroup_size(256)
fn dv_qkv_rope_split(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, qs_p.groups_x);
    let hd = qs_p.head_dim;
    let per_row = qs_p.heads * hd;
    if (i >= qs_p.rows * per_row) { return; }
    let row = i / per_row;
    let rest = i - row * per_row;
    let h = rest / hd;
    let lane = rest - h * hd;
    let qkv_row = row * 3u * per_row;
    let c = qs_cos[row * hd + lane];
    let s = qs_sin[row * hd + lane];
    let qb = qkv_row + h * hd;
    let kb = qkv_row + per_row + h * hd;
    let vb = qkv_row + 2u * per_row + h * hd;
    let slab = h * qs_p.rows_pad * hd;
    qs_q[slab + row * hd + lane] = f16(qs_rope(qb, lane, hd, c, s));
    qs_k[slab + row * hd + lane] = f16(qs_rope(kb, lane, hd, c, s));
    if (qs_p.pad0 == 1u) {
        qs_vt[slab + row * hd + lane] = f16(qs_qkv[vb + lane]);
    } else {
        qs_vt[h * hd * qs_p.rows_pad + lane * qs_p.rows_pad + row] = f16(qs_qkv[vb + lane]);
    }
}
";

pub const DOTS_VISION_SOFTMAX_IS_BANDWIDTH_BOUND_AND_ZMERGE_IS_ALREADY_THE_FEWEST_SLAB_PASSES_A_COOP_FRAGMENT_CONSUMER_ADMITS: &str =
    "This kernel moves bytes rather than doing arithmetic, which is why it never answered a plan or \
     a dtype change: on Apple M2 Ultra at 13600 patches the slab arm reads the f32 score slab three \
     times and writes f16 probabilities once, 2.589 GB per head-layer and 1305 GB per forward, in \
     1.471 s -- 887 GB/s -- and zmerge deletes one read for 932 GB in 1.207 s, 772 GB/s. Both sit \
     at or above this adapter's nominal 800 GB/s, so the phase is at its traffic floor and zmerge \
     is worth 0.259 s of an 8.272 s forward. Deleting the pass outright, by having attn_ctx consume \
     raw scores and exponentiate as it loads them, is blocked and not by arithmetic: the context \
     GEMM's A operand is taken by coopLoadT straight from storage into the matrix unit, so there is \
     no point between the load and coopMultiplyAdd where a per-element exp can be applied, and \
     staging the values through workgroup memory to transform them IS the fused kernel this tower \
     has measured dead twice. Three smaller obstacles sit behind that one: a score row spans \
     n_pad/(tile*tn*sg) = 107 workgroups at full resolution so the scores epilogue can only produce \
     partial maxima; the row sum cannot be produced there at all because it needs the final max, so \
     it would have to come from a ones-column appended to V at head_dim 128 -> 136; and raw scores \
     are f32 where probabilities are f16, doubling the context GEMM's operand read from 0.370 to \
     0.740 GB per head-layer. The floor for a one-read one-write softmax is 559 GB, about 0.72 s, \
     so roughly 0.49 s sits behind that blocked structure and nowhere else.";

const SOFTMAX_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> sm_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_o: array<f16>;
@group(0) @binding(2) var<uniform> sm_p: RowsP;

var<workgroup> sm_red: array<f32, 256>;

@compute @workgroup_size(256)
fn dv_softmax_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * sm_p.groups_x;
    if (row >= sm_p.rows) { return; }
    let base = row * sm_p.cols_pad;
    var m = -3.4028235e38;
    for (var c = lid.x; c < sm_p.cols; c = c + 256u) {
        m = max(m, sm_s[base + c] * sm_p.scalar);
    }
    sm_red[lid.x] = m;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) { sm_red[lid.x] = max(sm_red[lid.x], sm_red[lid.x + stride]); }
        workgroupBarrier();
    }
    let rowmax = sm_red[0];
    workgroupBarrier();
    var z = 0.0;
    for (var c = lid.x; c < sm_p.cols; c = c + 256u) {
        z = z + exp(sm_s[base + c] * sm_p.scalar - rowmax);
    }
    workgroupBarrier();
    sm_red[lid.x] = z;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) { sm_red[lid.x] = sm_red[lid.x] + sm_red[lid.x + stride]; }
        workgroupBarrier();
    }
    let inv = 1.0 / sm_red[0];
    for (var c = lid.x; c < sm_p.cols_pad; c = c + 256u) {
        if (c < sm_p.cols) {
            sm_o[base + c] = f16(exp(sm_s[base + c] * sm_p.scalar - rowmax) * inv);
        } else {
            sm_o[base + c] = f16(0.0);
        }
    }
}
";

const SOFTMAX_Z_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, zbase: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> sz_s: array<f32>;
@group(0) @binding(1) var<storage, read_write> sz_o: array<f16>;
@group(0) @binding(2) var<uniform> sz_p: RowsP;
@group(0) @binding(3) var<storage, read_write> sz_z: array<f32>;

var<workgroup> sz_red: array<f32, 256>;

@compute @workgroup_size(256)
fn dv_softmax_rows_deferred_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * sz_p.groups_x;
    if (row >= sz_p.rows) { return; }
    let base = row * sz_p.cols_pad;
    var m = -3.4028235e38;
    for (var c = lid.x; c < sz_p.cols; c = c + 256u) {
        m = max(m, sz_s[base + c] * sz_p.scalar);
    }
    sz_red[lid.x] = m;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) { sz_red[lid.x] = max(sz_red[lid.x], sz_red[lid.x + stride]); }
        workgroupBarrier();
    }
    let rowmax = sz_red[0];
    workgroupBarrier();
    var z = 0.0;
    for (var c = lid.x; c < sz_p.cols_pad; c = c + 256u) {
        var p = 0.0;
        if (c < sz_p.cols) { p = exp(sz_s[base + c] * sz_p.scalar - rowmax); }
        sz_o[base + c] = f16(p);
        z = z + p;
    }
    workgroupBarrier();
    sz_red[lid.x] = z;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) { sz_red[lid.x] = sz_red[lid.x] + sz_red[lid.x + stride]; }
        workgroupBarrier();
    }
    if (lid.x == 0u) { sz_z[sz_p.zbase + row] = 1.0 / sz_red[0]; }
}
";

const ATTN_MERGE_Z_SRC: &str = "
struct SplitP { rows: u32, rows_pad: u32, heads: u32, head_dim: u32, groups_x: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> mz_ctx: array<f32>;
@group(0) @binding(1) var<storage, read_write> mz_out: array<f16>;
@group(0) @binding(2) var<uniform> mz_p: SplitP;
@group(0) @binding(3) var<storage, read> mz_z: array<f32>;

@compute @workgroup_size(256)
fn dv_attn_merge_deferred_norm_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, mz_p.groups_x);
    let hd = mz_p.head_dim;
    let per_row = mz_p.heads * hd;
    if (i >= mz_p.rows * per_row) { return; }
    let row = i / per_row;
    let rest = i - row * per_row;
    let h = rest / hd;
    let lane = rest - h * hd;
    let inv = mz_z[h * mz_p.rows_pad + row];
    mz_out[row * per_row + h * hd + lane] = f16(mz_ctx[h * mz_p.rows_pad * hd + row * hd + lane] * inv);
}
";

const ATTN_MERGE_SRC: &str = "
struct SplitP { rows: u32, rows_pad: u32, heads: u32, head_dim: u32, groups_x: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> am_ctx: array<f32>;
@group(0) @binding(1) var<storage, read_write> am_out: array<f16>;
@group(0) @binding(2) var<uniform> am_p: SplitP;

@compute @workgroup_size(256)
fn dv_attn_merge_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, am_p.groups_x);
    let hd = am_p.head_dim;
    let per_row = am_p.heads * hd;
    if (i >= am_p.rows * per_row) { return; }
    let row = i / per_row;
    let rest = i - row * per_row;
    let h = rest / hd;
    let lane = rest - h * hd;
    am_out[row * per_row + h * hd + lane] = f16(am_ctx[h * am_p.rows_pad * hd + row * hd + lane]);
}
";

const ADD_BIAS_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> ab_y: array<f32>;
@group(0) @binding(1) var<storage, read> ab_b: array<f32>;
@group(0) @binding(2) var<uniform> ab_p: RowsP;

@compute @workgroup_size(256)
fn dv_add_bias_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, ab_p.groups_x);
    if (i >= ab_p.rows * ab_p.cols) { return; }
    let r = i / ab_p.cols;
    let c = i - r * ab_p.cols;
    ab_y[r * ab_p.cols_pad + c] = ab_y[r * ab_p.cols_pad + c] + ab_b[c];
}
";

const ADD_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read_write> ax_a: array<f32>;
@group(0) @binding(1) var<storage, read> ax_b: array<f32>;
@group(0) @binding(2) var<uniform> ax_p: RowsP;

@compute @workgroup_size(256)
fn dv_add_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, ax_p.groups_x);
    if (i >= ax_p.rows * ax_p.cols) { return; }
    ax_a[i] = ax_a[i] + ax_b[i];
}
";

const SPLITK_REDUCE_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, zbase: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> kr_p: array<f32>;
@group(0) @binding(1) var<storage, read_write> kr_y: array<f32>;
@group(0) @binding(2) var<uniform> kr_pp: RowsP;

@compute @workgroup_size(256)
fn dv_splitk_reduce_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, kr_pp.groups_x);
    if (i >= kr_pp.rows) { return; }
    var acc = 0.0;
    for (var s = 0u; s < kr_pp.cols; s = s + 1u) {
        acc = acc + kr_p[s * kr_pp.cols_pad + i];
    }
    kr_y[i] = acc;
}
";

const SILU_MUL_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> sw_g: array<f32>;
@group(0) @binding(1) var<storage, read> sw_u: array<f32>;
@group(0) @binding(2) var<storage, read_write> sw_y: array<f16>;
@group(0) @binding(3) var<uniform> sw_p: RowsP;

@compute @workgroup_size(256)
fn dv_silu_mul_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, sw_p.groups_x);
    if (i >= sw_p.rows * sw_p.cols) { return; }
    let g = sw_g[i];
    sw_y[i] = f16(g / (1.0 + exp(-g)) * sw_u[i]);
}
";

const LAYERNORM_F16_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> lnq_x: array<f16>;
@group(0) @binding(1) var<storage, read> lnq_w: array<f32>;
@group(0) @binding(2) var<storage, read> lnq_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> lnq_y: array<f16>;
@group(0) @binding(4) var<uniform> lnq_p: RowsP;

var<workgroup> lnq_sum: array<f32, 256>;
var<workgroup> lnq_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn dv_layernorm_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let row = wid.x + wid.y * lnq_p.groups_x;
    if (row >= lnq_p.rows) { return; }
    let base = row * lnq_p.cols_pad;
    var acc = 0.0;
    var sq = 0.0;
    for (var c = lid.x; c < lnq_p.cols; c = c + 256u) {
        let v = f32(lnq_x[base + c]);
        acc = acc + v;
        sq = sq + v * v;
    }
    lnq_sum[lid.x] = acc;
    lnq_sq[lid.x] = sq;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            lnq_sum[lid.x] = lnq_sum[lid.x] + lnq_sum[lid.x + stride];
            lnq_sq[lid.x] = lnq_sq[lid.x] + lnq_sq[lid.x + stride];
        }
        workgroupBarrier();
    }
    let mean = lnq_sum[0] / f32(lnq_p.cols);
    let variance = max(lnq_sq[0] / f32(lnq_p.cols) - mean * mean, 0.0);
    let inv = inverseSqrt(variance + lnq_p.scalar);
    let obase = row * lnq_p.cols;
    for (var c = lid.x; c < lnq_p.cols; c = c + 256u) {
        lnq_y[obase + c] = f16((f32(lnq_x[base + c]) - mean) * inv * lnq_w[c] + lnq_b[c]);
    }
}
";

const GELU_ERF_F16_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> ge_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> ge_y: array<f16>;
@group(0) @binding(2) var<uniform> ge_p: RowsP;

fn ge_erf_as7126(x: f32) -> f32 {
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = 1.0 - poly * exp(-ax * ax);
    return select(e, -e, x < 0.0);
}

@compute @workgroup_size(256)
fn dv_gelu_erf_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, ge_p.groups_x);
    if (i >= ge_p.rows * ge_p.cols) { return; }
    let v = ge_x[i];
    ge_y[i] = f16(0.5 * v * (1.0 + ge_erf_as7126(v * 0.7071067811865476)));
}
";

const DEQUANT_W4_F16_SRC: &str = "
struct RowsP { rows: u32, cols: u32, cols_pad: u32, groups_x: u32, scalar: f32, zbase: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> dq_w: array<u32>;
@group(0) @binding(1) var<storage, read> dq_s: array<f32>;
@group(0) @binding(2) var<storage, read> dq_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> dq_y: array<f16>;
@group(0) @binding(4) var<uniform> dq_p: RowsP;

@compute @workgroup_size(256)
fn dv_dequant_w4_f16(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = dv_linear_index(wid, lid, dq_p.groups_x);
    if (i >= dq_p.rows * dq_p.cols_pad) { return; }
    let r = i / dq_p.cols_pad;
    let c = i - r * dq_p.cols_pad;
    if (c >= dq_p.cols) { dq_y[i] = f16(0.0); return; }
    let word = dq_w[r * (dq_p.cols >> 3u) + (c >> 3u)];
    let q = f32((word >> (4u * (c & 7u))) & 0xFu);
    let gi = r * (dq_p.cols / dq_p.zbase) + c / dq_p.zbase;
    dq_y[i] = f16(q * dq_s[gi] + dq_b[gi]);
}
";

fn source_of(body: &str) -> String {
    compose_enabled(&["f16"], &format!("{PRELUDE}{body}"))
}

type QueuedPass = (Arc<wgpu::ComputePipeline>, wgpu::BindGroup, (u32, u32, u32));

pub struct VisionGemm {
    ctx: &'static WgpuContext,
    route: VisionGemmRoute,
    coop_zero: wgpu::Buffer,
    coop_zero16: wgpu::Buffer,
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    uniforms: Mutex<
        HashMap<
            [u8; DOTS_VISION_UNIFORM_BLOCK_IS_EIGHT_U32_IN_EVERY_PARAM_STRUCT_SO_ITS_CONTENT_IS_THE_CACHE_KEY],
            Arc<wgpu::Buffer>,
        >,
    >,
    queued: Mutex<Vec<QueuedPass>>,
    plans: Mutex<HashMap<(usize, usize), CoopPlan>>,
    uniforms_wide: Mutex<HashMap<Vec<u8>, Arc<wgpu::Buffer>>>,
    splitk_scratch: Mutex<Option<(usize, Arc<wgpu::Buffer>)>>,
    submits: std::sync::atomic::AtomicU64,
    dispatched: std::sync::atomic::AtomicU64,
}

impl VisionGemm {
    pub fn new(ctx: &'static WgpuContext, route: VisionGemmRoute) -> Result<Self> {
        anyhow::ensure!(
            ctx.caps.shader_f16,
            "{}",
            DOTS_VISION_WGPU_STAGES_EVERY_WEIGHT_AS_F16_BECAUSE_THE_ONLY_GENERAL_LARGE_M_GEMM_IN_THIS_TREE_IS_COOP_MAT_F16
        );
        let zero_elems = match route {
            VisionGemmRoute::Coop(g) => g.zero_elems(),
            VisionGemmRoute::ScalarF16 => 1,
        };
        let coop_zero =
            dispatch::storage_from_slice(ctx, "dv-coop-zero", &vec![0f32; zero_elems.max(1)]);
        let coop_zero16 =
            dispatch::storage_zeroed(ctx, "dv-coop-zero16", (zero_elems.max(1) * 2) as u64);
        Ok(Self {
            ctx,
            route,
            coop_zero,
            coop_zero16,
            pipelines: Mutex::new(HashMap::new()),
            uniforms: Mutex::new(HashMap::new()),
            queued: Mutex::new(Vec::new()),
            plans: Mutex::new(HashMap::new()),
            uniforms_wide: Mutex::new(HashMap::new()),
            splitk_scratch: Mutex::new(None),
            submits: std::sync::atomic::AtomicU64::new(0),
            dispatched: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn submits_and_dispatches(&self) -> (u64, u64) {
        (
            self.submits.load(std::sync::atomic::Ordering::Relaxed),
            self.dispatched.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub fn route(&self) -> VisionGemmRoute {
        self.route
    }

    pub fn tile(&self) -> usize {
        self.route.tile()
    }

    pub fn set_plan(&self, n: usize, k_pad: usize, plan: Option<CoopPlan>) {
        let mut m = self.plans.lock().unwrap();
        match plan {
            Some(p) => {
                m.insert((n, k_pad), p);
            }
            None => {
                m.remove(&(n, k_pad));
            }
        }
    }

    pub fn plan_for(&self, n: usize, k_pad: usize, m_pad: usize) -> Result<CoopPlan> {
        let VisionGemmRoute::Coop(g) = self.route else {
            anyhow::bail!("plan_for is a cooperative-matrix concept; this route is scalar_f16");
        };
        self.resolve_plan(&g, n, k_pad, m_pad)
    }

    fn resolve_plan(
        &self,
        g: &coop::CoopGemm,
        n: usize,
        k_pad: usize,
        m_pad: usize,
    ) -> Result<CoopPlan> {
        let listed = self
            .plans
            .lock()
            .unwrap()
            .get(&(n, k_pad))
            .copied()
            .or_else(|| {
                DOTS_VISION_COOP_PLANS_ARE_FITTED_ON_THIS_TOWERS_OWN_SHAPES_BECAUSE_Q38_PF_ROUTE_CARRIES_NO_ROW_FOR_ANY_OF_THEM
                    .iter()
                    .find(|(rn, rk, _)| *rn == n && *rk == k_pad)
                    .map(|(_, _, p)| *p)
            });
        let _ = DOTS_VISION_COOP_PLAN_TM8_TN4_IS_FASTER_ON_EVERY_SHAPE_BUT_ATTN_CTX_WHERE_IT_IS_A_LOSS_SO_A_BLANKET_FLIP_TAKES_THAT_LOSS_SILENTLY;
        let tile = g.tile as usize;
        let n_pad = pad_to(n, tile);
        let _ = DOTS_VISION_A_TALLER_ROW_BLOCK_ONLY_PAYS_WHERE_THERE_IS_COLUMN_PARALLELISM_TO_TRADE_AGAINST_IT;
        let square = DOTS_VISION_COOP_PLAN_IS_A_SQUARE_FRAGMENT_TILE_BECAUSE_TM_PLUS_TN_OPERAND_LOADS_FEED_TM_TIMES_TN_MULTIPLY_ADDS;
        let fitted = DOTS_VISION_COOP_PLAN_FITTED_TM8_TN4_SG4_IS_THE_DEFAULT_WHEREVER_THE_OUTPUT_IS_WIDER_THAN_ONE_COLUMN_BAND;
        let wider_than_one_band = n_pad > g.cols_per_workgroup(fitted.tn, fitted.sg) as usize;
        let mut p = listed.unwrap_or(
            match std::env::var(DOTS_VISION_PLAN_ENV).ok().as_deref() {
                Some("square") => square,
                _ if wider_than_one_band => fitted,
                _ => square,
            },
        );
        p.tm = p.tm.clamp(1, m_pad.div_ceil(tile).max(1) as u32);
        p.tn = p.tn.max(1);
        p.sg = p.sg.max(1);
        let _ = DOTS_VISION_COOP_COLUMN_BAND_SHRINKS_TO_FIT_N_BECAUSE_SUBGROUPS_PAST_COLUMN_N_STILL_WALK_THE_WHOLE_K_LOOP;
        while p.tn > 1 && (g.cols_per_workgroup(p.tn, p.sg) as usize) > n_pad {
            p.tn /= 2;
        }
        while p.sg > 1 && (g.cols_per_workgroup(p.tn, p.sg) as usize) > n_pad {
            p.sg /= 2;
        }
        let k_tiles = (k_pad / tile.max(1)) as u32;
        if p.ku == 0 || k_tiles == 0 || !k_tiles.is_multiple_of(p.ku) {
            p.ku = 1;
        }
        anyhow::ensure!(
            g.acc_fits_a_register_file(p.tm, p.tn) && p.tm * p.tn <= 64,
            "coop plan {} at n={n} k={k_pad}: {} accumulator dwords per lane exceeds the register \
             budget the generator will honour",
            p.name(),
            g.acc_lane_dwords(p.tm, p.tn)
        );
        anyhow::ensure!(
            (1..=8).contains(&p.sg)
                && self.ctx.caps.max_compute_invocations_per_workgroup >= p.sg * coop::SUBGROUP,
            "coop plan {} needs a {}-invocation workgroup; this adapter allows {}",
            p.name(),
            p.sg * coop::SUBGROUP,
            self.ctx.caps.max_compute_invocations_per_workgroup
        );
        let _ = DOTS_VISION_COOP_ROW_BLOCK_CEILING_IS_THE_TALLEST_TILE_THE_PLAN_LADDER_ASKS_FOR;
        anyhow::ensure!(
            g.rows_per_block(p.tm) as usize <= DOTS_VISION_COOP_ROW_BLOCK_CEILING_ROWS,
            "coop plan {} at n={n} k={k_pad} asks for a {}-row block, past the {} rows every x \
             operand's slack is sized for: {}",
            p.name(),
            g.rows_per_block(p.tm),
            DOTS_VISION_COOP_ROW_BLOCK_CEILING_ROWS,
            DOTS_VISION_COOP_ROW_BAND_OVERSHOOTS_M_THE_SAME_WAY_THE_COLUMN_BAND_OVERSHOOTS_N
        );
        let cols = g.cols_per_workgroup(p.tn, p.sg) as usize;
        let covered = n_pad.div_ceil(cols) * cols;
        let over_read = (covered - n_pad) * k_pad * g.ab.bytes() as usize;
        anyhow::ensure!(
            cols <= n_pad && over_read <= DOTS_VISION_COOP_OUT_OF_RANGE_WEIGHT_READ_BUDGET_BYTES,
            "coop plan {} covers {covered} columns of an n={n} k={k_pad} operand in {cols}-column \
             bands, reading {over_read} bytes past its end: {}",
            p.name(),
            DOTS_VISION_COOP_OUT_OF_RANGE_WEIGHT_READ_IS_BUDGETED_IN_BYTES_NOT_IN_WASTED_COLUMNS
        );
        Ok(p)
    }

    fn pipeline(
        &self,
        label: &str,
        entry: &str,
        source: impl FnOnce() -> String,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        if let Some(hit) = self.pipelines.lock().unwrap().get(entry) {
            return Ok(hit.clone());
        }
        let built = dispatch::cached_compute_pipeline(self.ctx, label, &source(), entry)
            .map_err(|e| anyhow::anyhow!("{entry}: {e}"))?;
        self.pipelines
            .lock()
            .unwrap()
            .insert(entry.to_string(), built.clone());
        Ok(built)
    }

    fn uniform<T: bytemuck::Pod>(&self, label: &str, value: &T) -> Arc<wgpu::Buffer> {
        let bytes = bytemuck::bytes_of(value);
        let mut key =
            [0u8; DOTS_VISION_UNIFORM_BLOCK_IS_EIGHT_U32_IN_EVERY_PARAM_STRUCT_SO_ITS_CONTENT_IS_THE_CACHE_KEY];
        key[..bytes.len()].copy_from_slice(bytes);
        if let Some(hit) = self.uniforms.lock().unwrap().get(&key) {
            return hit.clone();
        }
        let made = Arc::new(dispatch::uniform_from(self.ctx, label, value));
        self.uniforms.lock().unwrap().insert(key, made.clone());
        made
    }

    fn record(
        &self,
        label: &str,
        entry: &str,
        source: impl FnOnce() -> String,
        bindings: &[(u32, &wgpu::Buffer, u64)],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = self.pipeline(label, entry, source)?;
        let group = dispatch::bind_group_offsets(self.ctx, &pipeline, bindings);
        let full = {
            let mut q = self.queued.lock().unwrap();
            q.push((pipeline, group, groups));
            q.len()
                >= DOTS_VISION_PASSES_PER_SUBMIT_BECAUSE_MERGING_D_DISPATCHES_INTO_ONE_SUBMIT_PAYS_ROUGHLY_LINEARLY_IN_D
        };
        if full {
            self.flush()?;
        }
        Ok(())
    }

    fn record_src(
        &self,
        label: &str,
        body: &'static str,
        entry: &str,
        bindings: &[(u32, &wgpu::Buffer, u64)],
        groups: (u32, u32, u32),
    ) -> Result<()> {
        self.record(label, entry, || source_of(body), bindings, groups)
    }

    fn uniform_wide<T: bytemuck::Pod>(&self, label: &str, value: &T) -> Arc<wgpu::Buffer> {
        let bytes = bytemuck::bytes_of(value);
        if let Some(hit) = self.uniforms_wide.lock().unwrap().get(bytes) {
            return hit.clone();
        }
        let made = Arc::new(dispatch::uniform_from(self.ctx, label, value));
        self.uniforms_wide
            .lock()
            .unwrap()
            .insert(bytes.to_vec(), made.clone());
        made
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_steel(
        &self,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        n: usize,
        k_pad: usize,
        y_stride: usize,
        cfg: (u32, u32, u32, u32, u32),
    ) -> Result<bool> {
        if !k_pad.is_multiple_of(cfg.2 as usize) {
            return Ok(false);
        }
        let Ok(pls) = super::vision_attn_msl::steel_gemm_pipelines(self.ctx) else {
            return Ok(false);
        };
        let Some(pipeline) = pls.pick(cfg) else {
            return Ok(false);
        };
        let p = super::vision_attn_msl::steel_gemm_params(rows, n, k_pad, y_stride, cfg);
        let pb = self.uniform_wide("dv-steel-gemm-p", &p);
        let bind = self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("dv-steel-gemm"),
            layout: &pls.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: pb.as_entire_binding() },
            ],
        });
        self.record_prebuilt(
            pipeline,
            bind,
            super::vision_attn_msl::steel_gemm_grid(rows, n, cfg),
        )?;
        Ok(true)
    }

    pub fn record_prebuilt(
        &self,
        pipeline: Arc<wgpu::ComputePipeline>,
        group: wgpu::BindGroup,
        groups: (u32, u32, u32),
    ) -> Result<()> {
        let full = {
            let mut q = self.queued.lock().unwrap();
            q.push((pipeline, group, groups));
            q.len()
                >= DOTS_VISION_PASSES_PER_SUBMIT_BECAUSE_MERGING_D_DISPATCHES_INTO_ONE_SUBMIT_PAYS_ROUGHLY_LINEARLY_IN_D
        };
        if full {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&self) -> Result<()> {
        let batch: Vec<QueuedPass> = std::mem::take(&mut *self.queued.lock().unwrap());
        if batch.is_empty() {
            return Ok(());
        }
        let scope = self
            .ctx
            .device
            .push_error_scope(wgpu::ErrorFilter::Validation);
        let refs: Vec<dispatch::PassRef<'_>> = batch
            .iter()
            .map(|(p, g, wg)| (p.as_ref(), g, *wg))
            .collect();
        dispatch::submit_pass_list(self.ctx, refs);
        self.submits
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.dispatched
            .fetch_add(batch.len() as u64, std::sync::atomic::Ordering::Relaxed);
        if let Some(err) = pollster::block_on(scope.pop()) {
            anyhow::bail!("dots vision submit of {} passes: {err}", batch.len());
        }
        Ok(())
    }

    fn elementwise_groups(&self, invocations: usize) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations as u64, WORKGROUP_SIZE)
    }

    fn row_groups(&self, rows: usize) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, rows as u64, 1)
    }

    fn rows_params(&self, rows: usize, cols: usize, cols_pad: usize, scalar: f32, groups_x: u32) -> RowsP {
        self.rows_params_z(rows, cols, cols_pad, scalar, groups_x, 0)
    }

    fn rows_params_z(
        &self,
        rows: usize,
        cols: usize,
        cols_pad: usize,
        scalar: f32,
        groups_x: u32,
        zbase: usize,
    ) -> RowsP {
        RowsP {
            rows: rows as u32,
            cols: cols as u32,
            cols_pad: cols_pad as u32,
            groups_x,
            scalar,
            zbase: zbase as u32,
            pad1: 0,
            pad2: 0,
        }
    }

    pub fn attention_query_block_rows(&self, n_pad: usize) -> usize {
        let _ = ATTENTION_QUERY_BLOCK_IS_SIZED_FROM_THE_REPORTED_STORAGE_BINDING_LIMIT_NOT_A_FIXED_CONSTANT;
        let tile = self.tile();
        let binding_rows =
            (self.ctx.caps.max_storage_buffer_binding_size as usize) / (n_pad * 4).max(1);
        pad_to(
            ATTENTION_QUERY_BLOCK_ROWS_CEILING_IS_WHERE_THE_DISPATCH_LADDER_STOPPED_PAYING
                .min(n_pad)
                .min(binding_rows.max(tile))
                .max(tile),
            tile,
        )
    }

    pub fn time_gemm_plans(
        &self,
        label: &str,
        n: usize,
        k_pad: usize,
        m_pad: usize,
        repeats: usize,
        plans: &[CoopPlan],
    ) -> Result<Vec<(CoopPlan, f64)>> {
        let _ = DOTS_VISION_GEMM_LADDER_OPERANDS_ARE_DEVICE_ZEROS_BECAUSE_A_COOPERATIVE_MATRIX_RATE_IS_NOT_DATA_DEPENDENT;
        let reps = repeats.max(1);
        let w = self.f16_buf("dv-ladder-w", n * k_pad);
        let x = self.f16_buf("dv-ladder-x", self.coop_x_rows(m_pad) * k_pad);
        let y = self.f32_buf("dv-ladder-y", m_pad * n);
        let mut out = Vec::new();
        for want in plans {
            self.set_plan(n, k_pad, Some(*want));
            let Ok(resolved) = self.plan_for(n, k_pad, m_pad) else {
                self.queued.lock().unwrap().clear();
                continue;
            };
            let one = self
                .gemm(label, &w, 0, &x, 0, &y, 0, m_pad, n, k_pad, n)
                .and_then(|()| self.read_f32(&y, 1).map(|_| ()));
            if one.is_err() {
                self.queued.lock().unwrap().clear();
                continue;
            }
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                self.gemm(label, &w, 0, &x, 0, &y, 0, m_pad, n, k_pad, n)?;
            }
            self.read_f32(&y, 1)?;
            out.push((resolved, t0.elapsed().as_secs_f64() / reps as f64));
        }
        self.set_plan(n, k_pad, None);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    fn coop_acc_dispatch(
        &self,
        g: &coop::CoopGemm,
        plan: &CoopPlan,
        acc: CoopAcc,
        w: &wgpu::Buffer,
        w_off: u64,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        zero: &wgpu::Buffer,
        m_pad: usize,
        n: usize,
        k_pad: usize,
        y_off: u64,
        y_stride: usize,
    ) -> Result<()> {
        g.check_shape(m_pad as u32, n as u32, k_pad as u32)
            .map_err(|e| anyhow::anyhow!("coop acc probe: {e}"))?;
        let (bm, bn) = g.grid(m_pad as u32, n as u32, plan.tm, plan.tn, plan.sg);
        let groups = dispatch::workgroup_count_1d(self.ctx, (bm as u64) * (bn as u64), 1);
        let p = coop::CoopGemmParams {
            n_rows: n as u32,
            k_elems: k_pad as u32,
            m_rows: m_pad as u32,
            blocks_n: bn,
            y_stride: y_stride as u32,
            groups_x: groups.0,
            pad0: 0,
            pad1: 0,
        };
        let pb = self.uniform("dv-acc-p", &p);
        let (t, tm, tn, sg, ku) = (g.tile, plan.tm, plan.tn, plan.sg, plan.ku);
        self.record(
            "nv_models_dots_vision_coop_acc",
            &coop_entry_acc(t, tm, tn, sg, ku, acc),
            || coop_source_acc(t, tm, tn, sg, ku, acc),
            &[(0, w, w_off), (1, x, 0), (2, y, y_off), (3, &pb, 0), (4, zero, 0)],
            groups,
        )
    }

    pub fn time_coop_acc_arms(
        &self,
        label: &str,
        n: usize,
        k_pad: usize,
        m_pad: usize,
        repeats: usize,
        arms: &[(CoopPlan, CoopAcc)],
    ) -> Result<Vec<(CoopPlan, CoopAcc, f64)>> {
        let VisionGemmRoute::Coop(g) = self.route else {
            anyhow::bail!("{label}: an accumulator-dtype ladder is a cooperative-matrix concept");
        };
        let _ = DOTS_VISION_GEMM_LADDER_OPERANDS_ARE_DEVICE_ZEROS_BECAUSE_A_COOPERATIVE_MATRIX_RATE_IS_NOT_DATA_DEPENDENT;
        let reps = repeats.max(1);
        let w = self.f16_buf("dv-acc-w", n * k_pad);
        let x = self.f16_buf("dv-acc-x", self.coop_x_rows(m_pad) * k_pad);
        let y32 = self.f32_buf("dv-acc-y32", m_pad * n);
        let y16 = self.f16_buf("dv-acc-y16", m_pad * n);
        let z16 = self.f16_buf("dv-acc-zero16", g.zero_elems());
        let mut out = Vec::new();
        for (want, acc) in arms {
            self.set_plan(n, k_pad, Some(*want));
            let plan = match self.plan_for(n, k_pad, m_pad) {
                Ok(p) => p,
                Err(why) => {
                    eprintln!(
                        "[dv-acc] {label} {} {} has no legal plan: {why}",
                        want.name(),
                        acc.name()
                    );
                    self.queued.lock().unwrap().clear();
                    continue;
                }
            };
            let (y, zero) = match acc {
                CoopAcc::F32 => (&y32, &self.coop_zero),
                CoopAcc::F16 => (&y16, &z16),
            };
            let _ = A_COOP_ACC_ARM_IS_TIMED_TO_A_READBACK_WHOSE_BYTE_COUNT_IS_A_MULTIPLE_OF_FOUR_BECAUSE_A_TWO_BYTE_COPY_IS_REFUSED_NOT_SLOW;
            let settle = |g2: &Self| match acc {
                CoopAcc::F32 => g2.read_f32(&y32, 1).map(|_| ()),
                CoopAcc::F16 => g2.read_f16(&y16, 2).map(|_| ()),
            };
            let warm = self
                .coop_acc_dispatch(&g, &plan, *acc, &w, 0, &x, y, zero, m_pad, n, k_pad, 0, n)
                .and_then(|()| settle(self));
            if let Err(why) = warm {
                eprintln!(
                    "[dv-acc] {label} {} {} did not run: {why}",
                    plan.name(),
                    acc.name()
                );
                self.queued.lock().unwrap().clear();
                continue;
            }
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                self.coop_acc_dispatch(&g, &plan, *acc, &w, 0, &x, y, zero, m_pad, n, k_pad, 0, n)?;
            }
            settle(self)?;
            out.push((plan, *acc, t0.elapsed().as_secs_f64() / reps as f64));
        }
        self.set_plan(n, k_pad, None);
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn time_flash_m(
        &self,
        n: usize,
        n_pad: usize,
        heads: usize,
        hd: usize,
        m_rows: usize,
        repeats: usize,
    ) -> Result<f64> {
        anyhow::ensure!(
            m_rows % 4 == 0 && hd % 4 == 0,
            "an M-row flash groups four query rows per vec4 and four head dims per lane, so M={m_rows} and head_dim={hd} must both be multiples of four"
        );
        let _ = DOTS_VISION_GEMM_LADDER_OPERANDS_ARE_DEVICE_ZEROS_BECAUSE_A_COOPERATIVE_MATRIX_RATE_IS_NOT_DATA_DEPENDENT;
        let reps = repeats.max(1);
        let q = self.f16_buf("dv-fm-q", heads * n_pad * hd);
        let k = self.f16_buf("dv-fm-k", heads * n_pad * hd);
        let v = self.f16_buf("dv-fm-v", heads * n_pad * hd);
        let o = self.f16_buf("dv-fm-o", n_pad * heads * hd);
        let tiles = n.div_ceil(m_rows);
        let groups = dispatch::workgroup_count_1d(self.ctx, (heads * tiles) as u64, 1);
        let p = FmP {
            n: n as u32,
            n_pad: n_pad as u32,
            heads: heads as u32,
            tiles: tiles as u32,
            groups_x: groups.0,
            pad0: 0,
            pad1: 0,
            scale: 1.0 / (hd as f32).sqrt(),
        };
        let pb = self.uniform("dv-fm-p", &p);
        let one = |g: &Self| {
            g.record(
                "nv_models_dots_vision_flash_m",
                &flash_m_entry(m_rows, hd),
                || flash_m_source(m_rows, hd),
                &[(0, &q, 0), (1, &k, 0), (2, &v, 0), (3, &o, 0), (4, &pb, 0)],
                groups,
            )
        };
        let _ = A_COOP_ACC_ARM_IS_TIMED_TO_A_READBACK_WHOSE_BYTE_COUNT_IS_A_MULTIPLE_OF_FOUR_BECAUSE_A_TWO_BYTE_COPY_IS_REFUSED_NOT_SLOW;
        one(self)?;
        self.read_f16(&o, 2)?;
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            one(self)?;
        }
        self.read_f16(&o, 2)?;
        Ok(t0.elapsed().as_secs_f64() / reps as f64)
    }

    pub fn reclaim_dropped_device_buffers_because_wgpu_frees_only_on_poll_and_page_two_otherwise_runs_against_page_ones_residency(
        &self,
    ) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn drain(&self) -> Result<()> {
        self.flush()?;
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
        Ok(())
    }

    pub fn f32_buf(&self, label: &str, elems: usize) -> wgpu::Buffer {
        dispatch::storage_zeroed(self.ctx, label, (elems.max(1) * 4) as u64)
    }

    pub fn f16_buf(&self, label: &str, elems: usize) -> wgpu::Buffer {
        dispatch::storage_zeroed(self.ctx, label, (elems.max(1) * 2) as u64)
    }

    pub fn coop_x_rows(&self, m_pad: usize) -> usize {
        let _ = DOTS_VISION_COOP_ROW_BAND_OVERSHOOTS_M_THE_SAME_WAY_THE_COLUMN_BAND_OVERSHOOTS_N;
        match self.route {
            VisionGemmRoute::Coop(_) => {
                m_pad.next_multiple_of(DOTS_VISION_COOP_ROW_BLOCK_CEILING_ROWS)
            }
            VisionGemmRoute::ScalarF16 => m_pad,
        }
    }

    pub fn upload_f32(&self, label: &str, v: &[f32]) -> wgpu::Buffer {
        dispatch::storage_from_slice(self.ctx, label, v)
    }

    pub fn read_f32(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        self.flush()?;
        dispatch::read_back(self.ctx, buf, len).map_err(|e| anyhow::anyhow!("read_back f32: {e}"))
    }

    pub fn read_f16(&self, buf: &wgpu::Buffer, len: usize) -> Result<Vec<f32>> {
        self.flush()?;
        let bits: Vec<u16> = dispatch::read_back(self.ctx, buf, len)
            .map_err(|e| anyhow::anyhow!("read_back f16: {e}"))?;
        Ok(bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_splitk(
        &self,
        g: &coop::CoopGemm,
        splits: u32,
        w: &wgpu::Buffer,
        w_off: u64,
        x: &wgpu::Buffer,
        x_off: u64,
        y: &wgpu::Buffer,
        y_off: u64,
        m_pad: usize,
        n: usize,
        k_pad: usize,
        y_stride: usize,
    ) -> Result<()> {
        let _ = DOTS_VISION_SPLITK_SERVES_ONLY_ATTN_CTX_BECAUSE_IT_IS_THE_ONE_OCCUPANCY_STARVED_SHAPE_THE_BRACKET_NAMES;
        let listed = self.plans.lock().unwrap().contains_key(&(n, k_pad));
        let mut plan = self.resolve_plan(g, n, k_pad, m_pad)?;
        if !listed {
            plan = DOTS_VISION_SPLITK_FITTED_PLAN_IS_TM8_BECAUSE_THE_S16_LADDER_READS_15_59_TFLOPS_THERE_AGAINST_THE_SQUARE_PLANS_11_9;
            plan.tm = plan.tm.clamp(1, (m_pad.div_ceil(g.tile as usize)).max(1) as u32);
        }
        let (bm, bn) = g.grid(m_pad as u32, n as u32, plan.tm, plan.tn, plan.sg);
        let blocks_mn = bm * bn;
        let groups =
            dispatch::workgroup_count_1d(self.ctx, (blocks_mn as u64) * (splits as u64), 1);
        let tile = g.tile as usize;
        let kseg = ((k_pad / tile) as u32).div_ceil(splits) * g.tile;
        let slab = m_pad * y_stride;
        let need = slab * splits as usize;
        let scratch = {
            let mut lock = self.splitk_scratch.lock().unwrap();
            match lock.as_ref() {
                Some((have, buf)) if *have >= need => buf.clone(),
                _ => {
                    let made = Arc::new(dispatch::storage_zeroed(
                        self.ctx,
                        "dv-splitk-scratch",
                        (need * 4) as u64,
                    ));
                    *lock = Some((need, made.clone()));
                    made
                }
            }
        };
        let p = coop::CoopGemmParams {
            n_rows: n as u32,
            k_elems: k_pad as u32,
            m_rows: m_pad as u32,
            blocks_n: bn,
            y_stride: y_stride as u32,
            groups_x: groups.0,
            pad0: blocks_mn,
            pad1: kseg,
        };
        let pb = self.uniform("dv-splitk-p", &p);
        self.record(
            "nv_models_dots_vision_coop_splitk",
            &coop::entry_splitk(plan.tm, plan.tn, plan.sg, splits),
            || coop::source_splitk(plan.tm, plan.tn, plan.sg, splits),
            &[
                (0, w, w_off),
                (1, x, x_off),
                (2, &scratch, 0),
                (3, &pb, 0),
                (4, &self.coop_zero, 0),
            ],
            groups,
        )?;
        let rgroups = self.elementwise_groups(slab);
        let rp = self.rows_params(slab, splits as usize, slab, 0.0, rgroups.0);
        let rpb = self.uniform("dv-splitk-rp", &rp);
        self.record_src(
            "nv_models_dots_vision_splitk_reduce",
            SPLITK_REDUCE_SRC,
            "dv_splitk_reduce_f32",
            &[(0, &scratch, 0), (1, y, y_off), (2, &rpb, 0)],
            rgroups,
        )
    }

    pub fn gemm(
        &self,
        label: &str,
        w: &wgpu::Buffer,
        w_off: u64,
        x: &wgpu::Buffer,
        x_off: u64,
        y: &wgpu::Buffer,
        y_off: u64,
        m_pad: usize,
        n: usize,
        k_pad: usize,
        y_stride: usize,
    ) -> Result<()> {
        match self.route {
            VisionGemmRoute::Coop(g) => {
                g.check_shape(m_pad as u32, n as u32, k_pad as u32)
                    .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
                if label == "attn_ctx" {
                    if let Some(s) = splitk_arm() {
                        return self.gemm_splitk(
                            &g, s, w, w_off, x, x_off, y, y_off, m_pad, n, k_pad, y_stride,
                        );
                    }
                }
                let plan = self
                    .resolve_plan(&g, n, k_pad, m_pad)
                    .with_context(|| format!("{label}"))?;
                let (bm, bn) = g.grid(m_pad as u32, n as u32, plan.tm, plan.tn, plan.sg);
                let groups = dispatch::workgroup_count_1d(self.ctx, (bm as u64) * (bn as u64), 1);
                let p = coop::CoopGemmParams {
                    n_rows: n as u32,
                    k_elems: k_pad as u32,
                    m_rows: m_pad as u32,
                    blocks_n: bn,
                    y_stride: y_stride as u32,
                    groups_x: groups.0,
                    pad0: 0,
                    pad1: 0,
                };
                let pb = self.uniform("dv-coop-p", &p);
                self.record(
                    "nv_models_dots_vision_coop",
                    &g.entry(plan.tm, plan.tn, plan.sg, plan.ku),
                    || g.source(plan.tm, plan.tn, plan.sg, plan.ku),
                    &[
                        (0, w, w_off),
                        (1, x, x_off),
                        (2, y, y_off),
                        (3, &pb, 0),
                        (4, &self.coop_zero, 0),
                    ],
                    groups,
                )
            }
            VisionGemmRoute::ScalarF16 => {
                let groups = self.elementwise_groups(m_pad * n);
                let p = GemmP {
                    n_rows: n as u32,
                    k_pad: k_pad as u32,
                    m_rows: m_pad as u32,
                    y_stride: y_stride as u32,
                    groups_x: groups.0,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                };
                let pb = self.uniform("dv-scalar-p", &p);
                self.record_src(
                    "nv_models_dots_vision_scalar",
                    GEMM_SCALAR_SRC,
                    "dv_gemm_f16_scalar",
                    &[(0, w, w_off), (1, x, x_off), (2, y, y_off), (3, &pb, 0)],
                    groups,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gemm_w4a16(
        &self,
        label: &str,
        w4: &wgpu::Buffer,
        sf: &wgpu::Buffer,
        x: &wgpu::Buffer,
        x_off: u64,
        y: &wgpu::Buffer,
        y_off: u64,
        m_pad: usize,
        n: usize,
        k_pad: usize,
        y_stride: usize,
    ) -> Result<()> {
        let VisionGemmRoute::Coop(g) = self.route else {
            anyhow::bail!(
                "{label}: {}",
                DOTS_VISION_NVFP4_ARM_NEEDS_THE_16X16_COOP_ROUTE_BECAUSE_SOURCE_W4A16_STAGES_ONE_16_CODE_BLOCK_PER_K_TILE_ROW
            );
        };
        anyhow::ensure!(
            g.tile == 16,
            "{label}: {}",
            DOTS_VISION_NVFP4_ARM_NEEDS_THE_16X16_COOP_ROUTE_BECAUSE_SOURCE_W4A16_STAGES_ONE_16_CODE_BLOCK_PER_K_TILE_ROW
        );
        g.check_shape(m_pad as u32, n as u32, k_pad as u32)
            .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
        let mut plan = self
            .resolve_plan(&g, n, k_pad, m_pad)
            .with_context(|| format!("{label}"))?;
        if plan.ku > DOTS_VISION_W4A16_KU_CEILING_IS_FOUR_BECAUSE_SOURCE_WQ16_STAGES_AT_MOST_FOUR_K_TILES {
            plan.ku = DOTS_VISION_W4A16_KU_CEILING_IS_FOUR_BECAUSE_SOURCE_WQ16_STAGES_AT_MOST_FOUR_K_TILES;
        }
        anyhow::ensure!(
            coop::wq16_act_stage_elems(coop::WqAct::F16, plan.tm, plan.tn, plan.sg, plan.ku)
                <= coop::W4A16_STAGE_BUDGET_F16_IS_HALF_THE_48K_WORKGROUP_LIMIT,
            "{label}: w4a16 plan {} stages more f16 than the workgroup budget admits",
            plan.name()
        );
        let (bm, bn) = g.grid(m_pad as u32, n as u32, plan.tm, plan.tn, plan.sg);
        let groups = dispatch::workgroup_count_1d(self.ctx, (bm as u64) * (bn as u64), 1);
        let p = coop::CoopGemmParams {
            n_rows: n as u32,
            k_elems: k_pad as u32,
            m_rows: m_pad as u32,
            blocks_n: bn,
            y_stride: y_stride as u32,
            groups_x: groups.0,
            pad0: 0,
            pad1: 0,
        };
        let pb = self.uniform("dv-w4a16-p", &p);
        self.record(
            "nv_models_dots_vision_w4a16",
            &coop::entry_w4a16(plan.tm, plan.tn, plan.sg, plan.ku),
            || coop::source_w4a16(plan.tm, plan.tn, plan.sg, plan.ku),
            &[
                (0, w4, 0),
                (1, x, x_off),
                (2, y, y_off),
                (3, &pb, 0),
                (4, &self.coop_zero, 0),
                (5, sf, 0),
            ],
            groups,
        )
    }

    pub fn flash_geometry(&self, plan: &FlashPlan, hd: usize) -> Result<FlashGeometry> {
        anyhow::ensure!(
            self.route.is_coop(),
            "the fused attention is a cooperative-matrix kernel; this route is scalar_f16 and \
             keeps the slab arm"
        );
        let g = plan.geometry(self.tile(), hd)?;
        anyhow::ensure!(
            g.workgroup_bytes() <= self.ctx.caps.max_compute_workgroup_storage_size as usize,
            "flash plan {} needs {} bytes of workgroup storage; this adapter allows {}",
            plan.name(),
            g.workgroup_bytes(),
            self.ctx.caps.max_compute_workgroup_storage_size
        );
        anyhow::ensure!(
            g.wg <= self.ctx.caps.max_compute_invocations_per_workgroup as usize,
            "flash plan {} needs a {}-invocation workgroup; this adapter allows {}",
            plan.name(),
            g.wg,
            self.ctx.caps.max_compute_invocations_per_workgroup
        );
        Ok(g)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention(
        &self,
        geom: &FlashGeometry,
        q: &wgpu::Buffer,
        k: &wgpu::Buffer,
        v: &wgpu::Buffer,
        out: &wgpu::Buffer,
        n: usize,
        n_pad: usize,
        heads: usize,
        scale: f32,
    ) -> Result<()> {
        let blocks_m = n.div_ceil(geom.plan.bm);
        let blocks_n = n.div_ceil(geom.plan.bn);
        let groups = dispatch::workgroup_count_1d(self.ctx, (heads * blocks_m) as u64, 1);
        let p = FlashP {
            n: n as u32,
            n_pad: n_pad as u32,
            heads: heads as u32,
            hd: geom.hd as u32,
            blocks_m: blocks_m as u32,
            blocks_n: blocks_n as u32,
            groups_x: groups.0,
            scale,
        };
        let pb = self.uniform("dv-flash-p", &p);
        let entry = geom.entry();
        self.record(
            "nv_models_dots_vision_flash",
            &entry,
            || geom.source(),
            &[
                (0, q, 0),
                (1, k, 0),
                (2, v, 0),
                (3, out, 0),
                (4, &pb, 0),
                (5, &self.coop_zero, 0),
            ],
            groups,
        )
    }

    fn project(
        &self,
        lin: &WgpuLin,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        m_pad: usize,
        w4_scratch: Option<&wgpu::Buffer>,
    ) -> Result<()> {
        match &lin.w {
            WgpuLinW::F16(w) => {
                let steel = match steel_gemm_cfg() {
                    Some(cfg) => self.gemm_steel(w, x, y, rows, lin.n, lin.k_pad, lin.n, cfg)?,
                    None => false,
                };
                if !steel {
                    self.gemm(
                        "projection",
                        w,
                        0,
                        x,
                        0,
                        y,
                        0,
                        m_pad,
                        lin.n,
                        lin.k_pad,
                        lin.n,
                    )?;
                }
            }
            WgpuLinW::Nvfp4Block16 { words, scales } => self.gemm_w4a16(
                "projection",
                words,
                scales,
                x,
                0,
                y,
                0,
                m_pad,
                lin.n,
                lin.k_pad,
                lin.n,
            )?,
            WgpuLinW::MlxW4 {
                words,
                scales,
                biases,
                gs,
            } => {
                let scratch = w4_scratch.context(
                    "a MlxW4 projection needs the shared dequant scratch the tower sizes at load",
                )?;
                self.dequant_w4_into(words, scales, biases, scratch, lin.n, lin.k, lin.k_pad, *gs)?;
                self.gemm(
                    "projection",
                    scratch,
                    0,
                    x,
                    0,
                    y,
                    0,
                    m_pad,
                    lin.n,
                    lin.k_pad,
                    lin.n,
                )?
            }
        }
        if let Some(b) = &lin.bias {
            let groups = self.elementwise_groups(rows * lin.n);
            let p = self.rows_params(rows, lin.n, lin.n, 0.0, groups.0);
            let pb = self.uniform("dv-bias-p", &p);
            self.record_src(
                "nv_models_dots_vision_bias",
                ADD_BIAS_SRC,
                "dv_add_bias_f32",
                &[(0, y, 0), (1, b, 0), (2, &pb, 0)],
                groups,
            )?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn dequant_w4_into(
        &self,
        words: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        biases: &wgpu::Buffer,
        scratch: &wgpu::Buffer,
        n: usize,
        k: usize,
        k_pad: usize,
        gs: usize,
    ) -> Result<()> {
        let groups = self.elementwise_groups(n * k_pad);
        let p = self.rows_params_z(n, k, k_pad, 0.0, groups.0, gs);
        let pb = self.uniform("dv-dq4-p", &p);
        self.record_src(
            "nv_models_dots_vision_dequant_w4",
            DEQUANT_W4_F16_SRC,
            "dv_dequant_w4_f16",
            &[
                (0, words, 0),
                (1, scales, 0),
                (2, biases, 0),
                (3, scratch, 0),
                (4, &pb, 0),
            ],
            groups,
        )
    }

    fn layernorm_f16(
        &self,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let groups = self.row_groups(rows);
        let p = self.rows_params(rows, cols, cols, eps, groups.0);
        let pb = self.uniform("dv-ln-p", &p);
        self.record_src(
            "nv_models_dots_vision_layernorm",
            LAYERNORM_F16_SRC,
            "dv_layernorm_f16",
            &[(0, x, 0), (1, w, 0), (2, b, 0), (3, y, 0), (4, &pb, 0)],
            groups,
        )
    }

    fn gelu_erf_f16(
        &self,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        count: usize,
    ) -> Result<()> {
        let groups = self.elementwise_groups(count);
        let p = self.rows_params(count, 1, 1, 0.0, groups.0);
        let pb = self.uniform("dv-gelu-p", &p);
        self.record_src(
            "nv_models_dots_vision_gelu",
            GELU_ERF_F16_SRC,
            "dv_gelu_erf_f16",
            &[(0, x, 0), (1, y, 0), (2, &pb, 0)],
            groups,
        )
    }

    fn rmsnorm_to_f16(
        &self,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let groups = self.row_groups(rows);
        let p = self.rows_params(rows, cols, cols, eps, groups.0);
        let pb = self.uniform("dv-rms-p", &p);
        self.record_src(
            "nv_models_dots_vision_rmsnorm",
            RMSNORM_SRC,
            "dv_rmsnorm_f16",
            &[(0, x, 0), (1, w, 0), (2, y, 0), (3, &pb, 0)],
            groups,
        )
    }

    fn rmsnorm_f32_inplace(
        &self,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> Result<()> {
        let groups = self.row_groups(rows);
        let p = self.rows_params(rows, cols, cols, eps, groups.0);
        let pb = self.uniform("dv-rms32-p", &p);
        self.record_src(
            "nv_models_dots_vision_rmsnorm32",
            RMSNORM_F32_INPLACE_SRC,
            "dv_rmsnorm_f32_inplace",
            &[(0, x, 0), (1, w, 0), (2, &pb, 0)],
            groups,
        )
    }

    fn cast_pad(
        &self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        rows: usize,
        cols: usize,
        cols_pad: usize,
    ) -> Result<()> {
        let groups = self.elementwise_groups(rows * cols);
        let p = self.rows_params(rows, cols, cols_pad, 0.0, groups.0);
        let pb = self.uniform("dv-cast-p", &p);
        self.record_src(
            "nv_models_dots_vision_cast",
            CAST_PAD_SRC,
            "dv_cast_pad_f16",
            &[(0, src, 0), (1, dst, 0), (2, &pb, 0)],
            groups,
        )
    }

    fn add_into(&self, acc: &wgpu::Buffer, add: &wgpu::Buffer, count: usize) -> Result<()> {
        let groups = self.elementwise_groups(count);
        let p = self.rows_params(count, 1, 1, 0.0, groups.0);
        let pb = self.uniform("dv-add-p", &p);
        self.record_src(
            "nv_models_dots_vision_add",
            ADD_SRC,
            "dv_add_f32",
            &[(0, acc, 0), (1, add, 0), (2, &pb, 0)],
            groups,
        )
    }
}

pub fn probe_gemm_f16(
    g: &VisionGemm,
    w: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<Vec<f32>> {
    let tile = g.tile();
    let m_pad = pad_to(m, tile);
    let k_pad = pad_to(k, tile);
    let n_pad = pad_to(n, tile);
    anyhow::ensure!(w.len() == n * k, "probe w is {} not {n}x{k}", w.len());
    anyhow::ensure!(x.len() == m * k, "probe x is {} not {m}x{k}", x.len());
    let mut wp = vec![0f32; n_pad * k];
    wp[..n * k].copy_from_slice(w);
    let wb = dispatch::storage_from_slice(
        g.ctx,
        "dv-probe-w",
        &f16_bits_padded(&wp, n_pad, k, k_pad),
    );
    let xb = g.f16_buf("dv-probe-x", g.coop_x_rows(m_pad) * k_pad);
    let xsrc = g.upload_f32("dv-probe-xsrc", x);
    g.cast_pad(&xsrc, &xb, m, k, k_pad)?;
    let yb = g.f32_buf("dv-probe-y", m_pad * n_pad);
    g.gemm(
        "probe", &wb, 0, &xb, 0, &yb, 0, m_pad, n_pad, k_pad, n_pad,
    )?;
    let all = g.read_f32(&yb, m_pad * n_pad)?;
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        out[r * n..(r + 1) * n].copy_from_slice(&all[r * n_pad..r * n_pad + n]);
    }
    Ok(out)
}

pub fn probe_gemm_w4a16(
    g: &VisionGemm,
    w: &[f32],
    x: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let tile = g.tile();
    anyhow::ensure!(
        tile == 16,
        "{}",
        DOTS_VISION_NVFP4_ARM_NEEDS_THE_16X16_COOP_ROUTE_BECAUSE_SOURCE_W4A16_STAGES_ONE_16_CODE_BLOCK_PER_K_TILE_ROW
    );
    let m_pad = pad_to(m, tile);
    let k_pad = pad_to(k, tile);
    let n_pad = pad_to(n, tile);
    anyhow::ensure!(w.len() == n * k, "probe w is {} not {n}x{k}", w.len());
    anyhow::ensure!(x.len() == m * k, "probe x is {} not {m}x{k}", x.len());
    let mut wp = vec![0f32; n_pad * k];
    wp[..n * k].copy_from_slice(w);
    let (words, scales, q) = nvfp4_block16_rows_padded(&wp, n_pad, k, k_pad);
    let wb = dispatch::storage_from_slice(g.ctx, "dv-probe-w4", &words);
    let sb = dispatch::storage_from_slice(g.ctx, "dv-probe-w4sf", &scales);
    let xb = g.f16_buf("dv-probe-x4", g.coop_x_rows(m_pad) * k_pad);
    let xsrc = g.upload_f32("dv-probe-x4src", x);
    g.cast_pad(&xsrc, &xb, m, k, k_pad)?;
    let yb = g.f32_buf("dv-probe-y4", m_pad * n_pad);
    g.gemm_w4a16(
        "probe_w4a16", &wb, &sb, &xb, 0, &yb, 0, m_pad, n_pad, k_pad, n_pad,
    )?;
    let all = g.read_f32(&yb, m_pad * n_pad)?;
    let mut out = vec![0f32; m * n];
    for r in 0..m {
        out[r * n..(r + 1) * n].copy_from_slice(&all[r * n_pad..r * n_pad + n]);
    }
    let deq = q.dequantize();
    let mut wdeq = vec![0f32; n * k];
    for r in 0..n {
        wdeq[r * k..(r + 1) * k].copy_from_slice(&deq[r][..k]);
    }
    Ok((out, wdeq))
}

pub fn probe_flash_attention(
    g: &VisionGemm,
    plan: &FlashPlan,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    heads: usize,
    hd: usize,
) -> Result<Vec<f32>> {
    let dim = heads * hd;
    anyhow::ensure!(
        q.len() == n * dim && k.len() == n * dim && v.len() == n * dim,
        "probe_flash_attention wants three {n}x{dim} slabs"
    );
    let geom = g.flash_geometry(plan, hd)?;
    let tile = g.tile();
    let n_pad = pad_to(n, tile);
    let slack = DOTS_VISION_FLASH_READS_PAST_A_HEADS_SLAB_ON_THE_RAGGED_LAST_BLOCK_SO_THE_QKV_BUFFERS_CARRY_SLACK;
    let mut qb = vec![0u16; heads * n_pad * hd + slack];
    let mut kb = vec![0u16; heads * n_pad * hd + slack];
    let mut vb = vec![0u16; heads * hd * n_pad + slack];
    for r in 0..n {
        for h in 0..heads {
            for d in 0..hd {
                let src = r * dim + h * hd + d;
                qb[h * n_pad * hd + r * hd + d] = f16::from_f32(q[src]).to_bits();
                kb[h * n_pad * hd + r * hd + d] = f16::from_f32(k[src]).to_bits();
                vb[h * hd * n_pad + d * n_pad + r] = f16::from_f32(v[src]).to_bits();
            }
        }
    }
    let qbuf = dispatch::storage_from_slice(g.ctx, "dv-probe-q", &qb);
    let kbuf = dispatch::storage_from_slice(g.ctx, "dv-probe-k", &kb);
    let vbuf = dispatch::storage_from_slice(g.ctx, "dv-probe-v", &vb);
    let obuf = g.f16_buf("dv-probe-attn", n_pad * dim);
    let scale = 1.0f32 / (hd as f32).sqrt();
    g.flash_attention(&geom, &qbuf, &kbuf, &vbuf, &obuf, n, n_pad, heads, scale)?;
    let all = g.read_f16(&obuf, n * dim)?;
    Ok(all)
}

enum WgpuLinW {
    F16(wgpu::Buffer),
    Nvfp4Block16 {
        words: wgpu::Buffer,
        scales: wgpu::Buffer,
    },
    MlxW4 {
        words: wgpu::Buffer,
        scales: wgpu::Buffer,
        biases: wgpu::Buffer,
        gs: usize,
    },
}

struct WgpuLin {
    w: WgpuLinW,
    bias: Option<wgpu::Buffer>,
    n: usize,
    k: usize,
    k_pad: usize,
}

struct WgpuVisionBlock {
    norm1: wgpu::Buffer,
    norm2: wgpu::Buffer,
    qkv: WgpuLin,
    proj: WgpuLin,
    fc1: WgpuLin,
    fc3: WgpuLin,
    fc2: WgpuLin,
}

struct WgpuMerger {
    ln_w: Vec<f32>,
    ln_b: Vec<f32>,
    ln_w_dev: wgpu::Buffer,
    ln_b_dev: wgpu::Buffer,
    fc0: WgpuLin,
    fc2: WgpuLin,
    context_dim: usize,
    merge_units: usize,
}

pub struct DotsVisionTowerWgpu {
    cfg: DotsVisionConfig,
    g: VisionGemm,
    wq: VisionWeightFmt,
    patch_proj: WgpuLin,
    patch_norm: wgpu::Buffer,
    blocks: Vec<WgpuVisionBlock>,
    post_trunk_norm: Option<wgpu::Buffer>,
    merger: WgpuMerger,
    w4_scratch: Option<wgpu::Buffer>,
}

fn host_rows(weights: &WeightLoader, name: &str, expect: &[usize]) -> Result<Vec<f32>> {
    let t = weights
        .get(name, DType::F32)
        .with_context(|| format!("load {name}"))?;
    let t = if t.dims().len() > 2 && expect.len() == 2 {
        let rows = t.dim(0)?;
        let cols: usize = t.dims()[1..].iter().product();
        t.reshape((rows, cols))?
    } else {
        t
    };
    anyhow::ensure!(
        t.dims() == expect,
        "{name}: expected {expect:?}, got {:?}",
        t.dims()
    );
    Ok(t.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?)
}

impl DotsVisionTowerWgpu {
    pub fn from_loader(weights: &WeightLoader, prefix: &str, cfg: DotsVisionConfig) -> Result<Self> {
        let ctx =
            WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu adapter unavailable: {e}"))?;
        let route = VisionGemmRoute::from_env_and_adapter(ctx);
        Self::from_loader_route(weights, prefix, cfg, route)
    }

    pub fn from_loader_route(
        weights: &WeightLoader,
        prefix: &str,
        cfg: DotsVisionConfig,
        route: VisionGemmRoute,
    ) -> Result<Self> {
        Self::from_loader_route_wq(weights, prefix, cfg, route, VisionWeightFmt::from_env())
    }

    pub fn from_loader_route_wq(
        weights: &WeightLoader,
        prefix: &str,
        cfg: DotsVisionConfig,
        route: VisionGemmRoute,
        wq: VisionWeightFmt,
    ) -> Result<Self> {
        let ctx =
            WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu adapter unavailable: {e}"))?;
        if wq == VisionWeightFmt::Nvfp4Block16 {
            anyhow::ensure!(
                matches!(route, VisionGemmRoute::Coop(g) if g.tile == 16),
                "{DOTS_VISION_WGPU_WQ_ENV}=nvfp4 on route {}: {}",
                route.name(),
                DOTS_VISION_NVFP4_ARM_NEEDS_THE_16X16_COOP_ROUTE_BECAUSE_SOURCE_W4A16_STAGES_ONE_16_CODE_BLOCK_PER_K_TILE_ROW
            );
        }
        let g = VisionGemm::new(ctx, route)?;
        let tile = route.tile();
        let dim = cfg.embed_dim;
        anyhow::ensure!(
            dim.is_multiple_of(cfg.num_attention_heads),
            "embed_dim {dim} is not divisible by {} heads",
            cfg.num_attention_heads
        );
        let lin = |name: &str, bias: Option<&str>, n: usize, k: usize| -> Result<WgpuLin> {
            let w = host_rows(weights, name, &[n, k])?;
            let k_pad = pad_to(k, tile);
            let staged = match wq {
                VisionWeightFmt::F16 => WgpuLinW::F16(dispatch::storage_from_slice(
                    ctx,
                    name,
                    &f16_bits_padded(&w, n, k, k_pad),
                )),
                VisionWeightFmt::Nvfp4Block16 => {
                    let (words, scales, _) = nvfp4_block16_rows_padded(&w, n, k, k_pad);
                    WgpuLinW::Nvfp4Block16 {
                        words: dispatch::storage_from_slice(ctx, name, &words),
                        scales: dispatch::storage_from_slice(ctx, name, &scales),
                    }
                }
                VisionWeightFmt::MlxW4 { gs, .. } if !k.is_multiple_of(gs) => {
                    WgpuLinW::F16(dispatch::storage_from_slice(
                        ctx,
                        name,
                        &f16_bits_padded(&w, n, k, k_pad),
                    ))
                }
                VisionWeightFmt::MlxW4 { gs, twin } => {
                    let q = mlx_w4_quantize_rows(&w, n, k, gs);
                    if twin {
                        WgpuLinW::F16(dispatch::storage_from_slice(
                            ctx,
                            name,
                            &mlx_w4_dequant_f16_bits_padded(&q, k_pad),
                        ))
                    } else {
                        let widen = |v: &[u16]| -> Vec<f32> {
                            v.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect()
                        };
                        WgpuLinW::MlxW4 {
                            words: dispatch::storage_from_slice(ctx, name, &q.words),
                            scales: dispatch::storage_from_slice(ctx, name, &widen(&q.scales_bf16)),
                            biases: dispatch::storage_from_slice(ctx, name, &widen(&q.biases_bf16)),
                            gs,
                        }
                    }
                }
            };
            let bias = match bias {
                Some(bn) => Some(dispatch::storage_from_slice(
                    ctx,
                    bn,
                    &host_rows(weights, bn, &[n])?,
                )),
                None => None,
            };
            Ok(WgpuLin {
                w: staged,
                bias,
                n,
                k,
                k_pad,
            })
        };
        let norm = |name: &str, n: usize| -> Result<wgpu::Buffer> {
            Ok(dispatch::storage_from_slice(
                ctx,
                name,
                &host_rows(weights, name, &[n])?,
            ))
        };

        let patch_proj = lin(
            &format!("{prefix}patch_embed.patchifier.proj.weight"),
            Some(&format!("{prefix}patch_embed.patchifier.proj.bias")),
            dim,
            PATCH_DIM,
        )?;
        let patch_norm = norm(&format!("{prefix}patch_embed.patchifier.norm.weight"), dim)?;

        let mut blocks = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let bp = format!("{prefix}blocks.{i}.");
            blocks.push(WgpuVisionBlock {
                norm1: norm(&format!("{bp}norm1.weight"), dim)?,
                norm2: norm(&format!("{bp}norm2.weight"), dim)?,
                qkv: lin(&format!("{bp}attn.qkv.weight"), None, dim * 3, dim)?,
                proj: lin(&format!("{bp}attn.proj.weight"), None, dim, dim)?,
                fc1: lin(
                    &format!("{bp}mlp.fc1.weight"),
                    None,
                    cfg.intermediate_size,
                    dim,
                )?,
                fc3: lin(
                    &format!("{bp}mlp.fc3.weight"),
                    None,
                    cfg.intermediate_size,
                    dim,
                )?,
                fc2: lin(
                    &format!("{bp}mlp.fc2.weight"),
                    None,
                    dim,
                    cfg.intermediate_size,
                )?,
            });
        }

        let post_trunk_norm = if cfg.post_norm {
            Some(norm(&format!("{prefix}post_trunk_norm.weight"), dim)?)
        } else {
            None
        };

        let mp = format!("{prefix}merger.");
        let merge_units = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let hidden = dim * merge_units;
        let ln_w = host_rows(weights, &format!("{mp}ln_q.weight"), &[dim])?;
        let ln_b = host_rows(weights, &format!("{mp}ln_q.bias"), &[dim])?;
        let merger = WgpuMerger {
            ln_w_dev: dispatch::storage_from_slice(ctx, "dv-merger-lnw", &ln_w),
            ln_b_dev: dispatch::storage_from_slice(ctx, "dv-merger-lnb", &ln_b),
            ln_w,
            ln_b,
            fc0: lin(
                &format!("{mp}mlp.0.weight"),
                Some(&format!("{mp}mlp.0.bias")),
                hidden,
                hidden,
            )?,
            fc2: lin(
                &format!("{mp}mlp.2.weight"),
                Some(&format!("{mp}mlp.2.bias")),
                cfg.hidden_size,
                hidden,
            )?,
            context_dim: dim,
            merge_units,
        };

        let mut scratch_elems = 0usize;
        {
            let mut note = |l: &WgpuLin| {
                if matches!(l.w, WgpuLinW::MlxW4 { .. }) {
                    scratch_elems = scratch_elems.max(l.n * l.k_pad);
                }
            };
            note(&patch_proj);
            for b in &blocks {
                for l in [&b.qkv, &b.proj, &b.fc1, &b.fc3, &b.fc2] {
                    note(l);
                }
            }
            note(&merger.fc0);
            note(&merger.fc2);
        }
        let w4_scratch = (scratch_elems > 0).then(|| g.f16_buf("dv-w4-scratch", scratch_elems));

        Ok(Self {
            cfg,
            g,
            wq,
            patch_proj,
            patch_norm,
            blocks,
            post_trunk_norm,
            merger,
            w4_scratch,
        })
    }

    pub fn config(&self) -> &DotsVisionConfig {
        &self.cfg
    }

    pub fn weight_fmt(&self) -> VisionWeightFmt {
        self.wq
    }

    pub fn route(&self) -> VisionGemmRoute {
        self.g.route()
    }

    pub fn gemm(&self) -> &VisionGemm {
        &self.g
    }

    pub fn set_plan(&self, n: usize, k_pad: usize, plan: Option<CoopPlan>) {
        self.g.set_plan(n, k_pad, plan);
    }

    pub fn gemm_shapes(&self, n_patches: usize) -> Vec<(usize, usize)> {
        let tile = self.g.tile();
        let n_pad = pad_to(n_patches, tile);
        let mut v = vec![
            (self.patch_proj.n, self.patch_proj.k_pad),
            (n_pad, self.cfg.head_dim()),
            (self.cfg.head_dim(), n_pad),
            (self.merger.fc0.n, self.merger.fc0.k_pad),
            (self.merger.fc2.n, self.merger.fc2.k_pad),
        ];
        if let Some(b) = self.blocks.first() {
            for l in [&b.qkv, &b.proj, &b.fc1, &b.fc3, &b.fc2] {
                v.push((l.n, l.k_pad));
            }
        }
        v.sort_unstable();
        v.dedup();
        v
    }

    pub fn encode(&self, prep: &PreparedImage) -> Result<Vec<f32>> {
        self.forward_patches(&prep.patches, prep.grid_h, prep.grid_w)
    }

    fn proj(
        &self,
        lin: &WgpuLin,
        x: &wgpu::Buffer,
        y: &wgpu::Buffer,
        rows: usize,
        m_pad: usize,
    ) -> Result<()> {
        self.g
            .project(lin, x, y, rows, m_pad, self.w4_scratch.as_ref())
    }

    fn install_fitted_plans_when_asked(&self, n_pad: usize) {
        if std::env::var(DOTS_VISION_WGPU_FITTED_PLANS_ENV)
            .is_ok_and(|v| !v.is_empty() && v != "0")
        {
            self.apply_m5_fitted_plans(n_pad, true);
        }
    }

    pub fn apply_m5_fitted_plans(&self, n_pad: usize, on: bool) {
        let _ = DOTS_VISION_M5_MAX_FITTED_PLANS_ADD_ONLY_THE_NARROW_N_ATTN_CTX_WINNER_THE_WIDTH_GATED_DEFAULT_CANNOT_EXPRESS;
        self.g.set_plan(
            self.cfg.head_dim(),
            n_pad,
            on.then_some(CoopPlan { tm: 8, tn: 2, sg: 2, ku: 1 }),
        );
    }

    pub fn forward_patches(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<Vec<f32>> {
        Ok(self
            .forward_patches_traced(patches, grid_h, grid_w, false)?
            .0)
    }

    pub fn forward_patches_with_layer_trace(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        self.forward_patches_traced(patches, grid_h, grid_w, true)
    }

    fn forward_patches_traced(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        trace: bool,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let mut phases = PhaseLog::from_env();
        let started = std::time::Instant::now();
        let out = self.forward_patches_traced_holding_every_transient(
            patches,
            grid_h,
            grid_w,
            trace,
            &mut phases,
        );
        self.g
            .reclaim_dropped_device_buffers_because_wgpu_frees_only_on_poll_and_page_two_otherwise_runs_against_page_ones_residency();
        phases.eprint(started.elapsed().as_secs_f64());
        out
    }

    fn forward_patches_traced_holding_every_transient(
        &self,
        patches: &[f32],
        grid_h: usize,
        grid_w: usize,
        trace: bool,
        ph: &mut PhaseLog,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let n = grid_h * grid_w;
        anyhow::ensure!(
            patches.len() == n * PATCH_DIM,
            "forward_patches: {} values is not {n} x {PATCH_DIM}",
            patches.len()
        );
        let g = &self.g;
        let tile = g.tile();
        let dim = self.cfg.embed_dim;
        let heads = self.cfg.num_attention_heads;
        let hd = self.cfg.head_dim();
        let inter = self.cfg.intermediate_size;
        let eps = self.cfg.rms_norm_eps as f32;
        let n_pad = pad_to(n, tile);
        anyhow::ensure!(
            hd.is_multiple_of(tile),
            "head_dim {hd} is not a multiple of the {tile}-wide GEMM tile"
        );

        self.install_fitted_plans_when_asked(n_pad);
        let patches_pad = pad_to(PATCH_DIM, tile);
        let x_rows = g.coop_x_rows(n_pad);
        let patch_x = g.f16_buf("dv-patch-x", x_rows * patches_pad);
        if !vis_ablated("patch_embed") {
            let src = g.upload_f32("dv-patch-src", patches);
            g.cast_pad(&src, &patch_x, n, PATCH_DIM, patches_pad)?;
        }

        ph.mark(g, "patch_upload_cast")?;

        let x = g.f32_buf("dv-x", n_pad * dim);
        if !vis_ablated("patch_embed") {
            self.proj(&self.patch_proj, &patch_x, &x, n, n_pad)?;
            g.rmsnorm_f32_inplace(&x, &self.patch_norm, n, dim, eps)?;
        }
        let xn = g.f16_buf("dv-xn", x_rows * dim);
        ph.mark(g, "patch_embed")?;

        let (cos, sin) = vision_rope_tables(grid_h, grid_w, hd, &Device::Cpu)?;
        let cos_b = g.upload_f32("dv-cos", &cos.flatten_all()?.to_vec1::<f32>()?);
        let sin_b = g.upload_f32("dv-sin", &sin.flatten_all()?.to_vec1::<f32>()?);
        ph.mark(g, "rope_tables")?;

        let mut arm = AttnArm::from_env();
        let _ = DOTS_VISION_STEEL_IS_THE_DEFAULT_WHERE_PASSTHROUGH_GRANTS_AND_ZMERGE_SERVES_EVERY_OTHER_ADAPTER;
        if std::env::var(DOTS_VISION_WGPU_ATTN_ENV).is_err()
            && hd == super::vision_attn_msl::HEAD_DIM as usize
            && super::vision_attn_msl::steel_pipelines(
                g.ctx,
                n.is_multiple_of(super::vision_attn_msl::STEEL_BQ as usize),
                n.is_multiple_of(super::vision_attn_msl::STEEL_BK as usize),
            )
            .is_ok()
        {
            arm = AttnArm::Steel;
        }
        let _ = DOTS_VISION_CTX_MERGE_IS_A_STRIDE_AND_AN_OFFSET_BUT_AN_F16_ACCUMULATOR_COSTS_NINE_PERCENT_SO_THE_MERGE_PASS_STAYS;
        let flash = match arm {
            AttnArm::Flash => match g.flash_geometry(&FlashPlan::from_env(), hd) {
                Ok(geom) => Some(geom),
                Err(why) => anyhow::bail!("{DOTS_VISION_WGPU_ATTN_ENV}=flash: {why}"),
            },
            _ => None,
        };
        let slack = DOTS_VISION_FLASH_READS_PAST_A_HEADS_SLAB_ON_THE_RAGGED_LAST_BLOCK_SO_THE_QKV_BUFFERS_CARRY_SLACK;

        let fused = flash.is_some() || arm == AttnArm::Steel;
        let qkv_y = g.f32_buf("dv-qkv", n_pad * dim * 3);
        let qh = g.f16_buf("dv-qh", heads * n_pad * hd + slack);
        let kh = g.f16_buf("dv-kh", heads * n_pad * hd + slack);
        let vt = g.f16_buf("dv-vt", heads * hd * n_pad + slack);
        let ctxb = g.f32_buf("dv-ctx", if fused { 1 } else { heads * n_pad * hd });
        let attn_x = g.f16_buf("dv-attn-x", x_rows * dim);
        let proj_y = g.f32_buf("dv-proj", n_pad * dim);
        let gate_y = g.f32_buf("dv-gate", n_pad * inter);
        let up_y = g.f32_buf("dv-up", n_pad * inter);
        let act = g.f16_buf("dv-act", x_rows * inter);
        let ff_y = g.f32_buf("dv-ff", n_pad * dim);

        let qb = g.attention_query_block_rows(n_pad);
        let slabbed = !fused;
        anyhow::ensure!(
            !slabbed || (qb * n_pad * 4) as u64 <= g.ctx.caps.max_storage_buffer_binding_size,
            "attention score slab {qb}x{n_pad} f32 exceeds the {} byte storage binding limit",
            g.ctx.caps.max_storage_buffer_binding_size
        );
        let scores = g.f32_buf("dv-scores", if slabbed { qb * n_pad } else { 1 });
        let probs = g.f16_buf(
            "dv-probs",
            if slabbed {
                g.coop_x_rows(qb) * n_pad
            } else {
                1
            },
        );
        let zinv = g.f32_buf(
            "dv-zinv",
            if arm == AttnArm::ZMerge {
                heads * n_pad
            } else {
                1
            },
        );
        let scale = 1.0f32 / (hd as f32).sqrt();
        let split_p = |groups_x: u32| SplitP {
            rows: n as u32,
            rows_pad: n_pad as u32,
            heads: heads as u32,
            head_dim: hd as u32,
            groups_x,
            pad0: u32::from(arm == AttnArm::Steel),
            pad1: 0,
            pad2: 0,
        };

        let msl = if arm == AttnArm::Steel {
            anyhow::ensure!(
                hd == super::vision_attn_msl::HEAD_DIM as usize,
                "the steel attention kernel is instantiated at head_dim {}, not {hd}",
                super::vision_attn_msl::HEAD_DIM
            );
            let pls = super::vision_attn_msl::steel_pipelines(
                g.ctx,
                n.is_multiple_of(super::vision_attn_msl::STEEL_BQ as usize),
                n.is_multiple_of(super::vision_attn_msl::STEEL_BK as usize),
            )?;
            let bq = super::vision_attn_msl::STEEL_BQ as usize;
            let bk = super::vision_attn_msl::STEEL_BK as usize;
            let p = super::vision_attn_msl::SteelAttnParams {
                b: 1,
                h: heads as i32,
                d: hd as i32,
                q_l: n as i32,
                k_l: n as i32,
                gqa_factor: 1,
                scale,
                nq: n.div_ceil(bq) as i32,
                nk: n.div_ceil(bk) as i32,
                nq_aligned: (n / bq) as i32,
                nk_aligned: (n / bk) as i32,
                q_l_rem: (n - (n / bq) * bq) as i32,
                k_l_rem: (n - (n / bk) * bk) as i32,
                q_l_off: 0,
                q_strides: [(heads * n_pad * hd) as i64, (n_pad * hd) as i64, hd as i64],
                k_strides: [(heads * n_pad * hd) as i64, (n_pad * hd) as i64, hd as i64],
                v_strides: [(heads * n_pad * hd) as i64, (n_pad * hd) as i64, hd as i64],
                o_strides: [(n_pad * dim) as i64, hd as i64, dim as i64],
            };
            let pb = dispatch::uniform_from(g.ctx, "dv-steel-p", &p);
            let mp = dispatch::uniform_from(g.ctx, "dv-steel-mp", &[0i64; 3]);
            let dummy = g.f32_buf("dv-steel-dummy", 1);
            let bind = g.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("dv-steel-attn"),
                layout: &pls.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: qh.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: kh.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: vt.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: attn_x.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: pb.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: mp.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: dummy.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: dummy.as_entire_binding() },
                ],
            });
            Some((
                pls.pipeline.clone(),
                bind,
                super::vision_attn_msl::steel_grid(heads as u32, n as u32),
            ))
        } else {
            None
        };

        let mut layer_trace = Vec::new();
        ph.mark(g, "transient_alloc")?;

        for blk in self.blocks.iter() {
            if !vis_ablated("rmsnorm") {
                g.rmsnorm_to_f16(&x, &blk.norm1, &xn, n, dim, eps)?;
            }
            ph.mark(g, "rmsnorm")?;
            if !vis_ablated("proj") {
                self.proj(&blk.qkv, &xn, &qkv_y, n, n_pad)?;
            }
            ph.mark(g, "qkv_project")?;

            if !vis_ablated("rope_split") {
                let groups = g.elementwise_groups(n * heads * hd);
                let pb = g.uniform("dv-split-p", &split_p(groups.0));
                g.record_src(
                    "nv_models_dots_vision_split",
                    QKV_ROPE_SPLIT_SRC,
                    "dv_qkv_rope_split",
                    &[
                        (0, &qkv_y, 0),
                        (1, &cos_b, 0),
                        (2, &sin_b, 0),
                        (3, &qh, 0),
                        (4, &kh, 0),
                        (5, &vt, 0),
                        (6, &pb, 0),
                    ],
                    groups,
                )?;
            }
            ph.mark(g, "rope_split")?;

            if let Some((pl, bind, grid)) = &msl {
                g.record_prebuilt(pl.clone(), bind.clone(), *grid)?;
                ph.mark(g, "msl_attention")?;
            } else if let Some(geom) = &flash {
                g.flash_attention(geom, &qh, &kh, &vt, &attn_x, n, n_pad, heads, scale)?;
                ph.mark(g, "flash_attention")?;
            } else {
                for h in 0..heads {
                    let head_f16 = (h * n_pad * hd * 2) as u64;
                    let vt_off = (h * hd * n_pad * 2) as u64;
                    let ctx_off = (h * n_pad * hd * 4) as u64;
                    let mut m0 = 0usize;
                    while m0 < n_pad {
                        let rows = qb.min(n_pad - m0);
                        let real = n.saturating_sub(m0).min(rows);
                        if real == 0 {
                            break;
                        }
                        let live = pad_to(real, tile);
                        if !vis_ablated("scores") {
                            g.gemm(
                                "attn_scores",
                                &kh,
                                head_f16,
                                &qh,
                                head_f16 + (m0 * hd * 2) as u64,
                                &scores,
                                0,
                                live,
                                n_pad,
                                hd,
                                n_pad,
                            )?;
                        }
                        ph.mark(g, "attn_scores_gemm")?;
                        let groups = g.row_groups(real);
                        if vis_ablated("softmax") {
                        } else if arm == AttnArm::ZMerge {
                            let p = g.rows_params_z(
                                real,
                                n,
                                n_pad,
                                scale,
                                groups.0,
                                h * n_pad + m0,
                            );
                            let pb = g.uniform("dv-softmaxz-p", &p);
                            g.record_src(
                                "nv_models_dots_vision_softmax_z",
                                SOFTMAX_Z_SRC,
                                "dv_softmax_rows_deferred_norm",
                                &[(0, &scores, 0), (1, &probs, 0), (2, &pb, 0), (3, &zinv, 0)],
                            groups,
                            )?;
                        } else {
                            let p = g.rows_params(real, n, n_pad, scale, groups.0);
                            let pb = g.uniform("dv-softmax-p", &p);
                            g.record_src(
                                "nv_models_dots_vision_softmax",
                                SOFTMAX_SRC,
                                "dv_softmax_rows",
                                &[(0, &scores, 0), (1, &probs, 0), (2, &pb, 0)],
                                groups,
                            )?;
                        }
                        ph.mark(g, "attn_softmax")?;
                        if !vis_ablated("ctx") {
                            g.gemm(
                                "attn_ctx",
                                &vt,
                                vt_off,
                                &probs,
                                0,
                                &ctxb,
                                ctx_off + (m0 * hd * 4) as u64,
                                live,
                                hd,
                                n_pad,
                                hd,
                            )?;
                        }
                        ph.mark(g, "attn_ctx_gemm")?;
                        m0 += rows;
                    }
                }

                let groups = g.elementwise_groups(n * heads * hd);
                let pb = g.uniform("dv-merge-p", &split_p(groups.0));
                if vis_ablated("attn_merge") {
                } else if arm == AttnArm::ZMerge {
                    g.record_src(
                        "nv_models_dots_vision_merge_z",
                        ATTN_MERGE_Z_SRC,
                        "dv_attn_merge_deferred_norm_f16",
                        &[(0, &ctxb, 0), (1, &attn_x, 0), (2, &pb, 0), (3, &zinv, 0)],
                        groups,
                    )?;
                } else {
                    g.record_src(
                        "nv_models_dots_vision_merge",
                        ATTN_MERGE_SRC,
                        "dv_attn_merge_f16",
                        &[(0, &ctxb, 0), (1, &attn_x, 0), (2, &pb, 0)],
                        groups,
                    )?;
                }
                ph.mark(g, "attn_merge")?;
            }

            if !vis_ablated("proj") {
                self.proj(&blk.proj, &attn_x, &proj_y, n, n_pad)?;
            }
            ph.mark(g, "attn_out_project")?;
            if !vis_ablated("residual_add") {
                g.add_into(&x, &proj_y, n * dim)?;
            }
            ph.mark(g, "residual_add")?;

            if !vis_ablated("rmsnorm") {
                g.rmsnorm_to_f16(&x, &blk.norm2, &xn, n, dim, eps)?;
            }
            ph.mark(g, "rmsnorm")?;
            if !vis_ablated("proj") {
                self.proj(&blk.fc1, &xn, &gate_y, n, n_pad)?;
                self.proj(&blk.fc3, &xn, &up_y, n, n_pad)?;
            }
            ph.mark(g, "mlp_gate_up_project")?;
            if !vis_ablated("silu") {
                let groups = g.elementwise_groups(n * inter);
                let p = g.rows_params(n * inter, 1, 1, 0.0, groups.0);
                let pb = g.uniform("dv-silu-p", &p);
                g.record_src(
                    "nv_models_dots_vision_silu",
                    SILU_MUL_SRC,
                    "dv_silu_mul_f16",
                    &[(0, &gate_y, 0), (1, &up_y, 0), (2, &act, 0), (3, &pb, 0)],
                    groups,
                )?;
            }
            ph.mark(g, "silu_mul")?;
            if !vis_ablated("proj") {
                self.proj(&blk.fc2, &act, &ff_y, n, n_pad)?;
            }
            ph.mark(g, "mlp_down_project")?;
            if !vis_ablated("residual_add") {
                g.add_into(&x, &ff_y, n * dim)?;
            }
            ph.mark(g, "residual_add")?;

            if trace {
                layer_trace.push(g.read_f32(&x, n * dim)?);
            }
        }

        let merged_zeros =
            || vec![0f32; (n / (MERGE_SIZE * MERGE_SIZE)) * self.merger.fc2.n];
        if vis_ablated("trunk_readback") {
            g.flush()?;
            g.drain()?;
            return Ok((merged_zeros(), layer_trace));
        }
        if let (Some(norm), true) = (&self.post_trunk_norm, merger_gpu_arm()) {
            g.rmsnorm_to_f16(&x, norm, &xn, n, dim, eps)?;
            if vis_ablated("merger") {
                g.flush()?;
                g.drain()?;
                return Ok((merged_zeros(), layer_trace));
            }
            let merged = self.merger_forward_gpu(&xn, n)?;
            ph.mark(g, "merger")?;
            return Ok((merged, layer_trace));
        }
        let trunk = match &self.post_trunk_norm {
            Some(norm) => {
                g.rmsnorm_to_f16(&x, norm, &xn, n, dim, eps)?;
                g.read_f16(&xn, n * dim)?
            }
            None => g.read_f32(&x, n * dim)?,
        };
        ph.mark(g, "trunk_readback")?;
        if vis_ablated("merger") {
            return Ok((merged_zeros(), layer_trace));
        }
        let merged = self.merger_forward(&trunk, n)?;
        ph.mark(g, "merger")?;
        Ok((merged, layer_trace))
    }

    fn merger_forward_gpu(&self, xn: &wgpu::Buffer, n: usize) -> Result<Vec<f32>> {
        let _ = DOTS_VISION_GPU_MERGER_IS_THE_DEFAULT_BECAUSE_THE_HOST_ROUND_TRIPS_WERE_THE_WHOLE_RESIDUAL_AND_THE_TOKEN_GATE_HELD;
        let m = &self.merger;
        let g = &self.g;
        anyhow::ensure!(
            n.is_multiple_of(m.merge_units),
            "PatchMerger: {n} patches not divisible by {}",
            m.merge_units
        );
        let hidden = m.context_dim * m.merge_units;
        anyhow::ensure!(
            m.fc0.k_pad == hidden,
            "the gpu merger writes merge-grouped rows contiguously, which is only the fc0 operand \
             layout when k_pad {} equals the {hidden}-wide merged row",
            m.fc0.k_pad
        );
        let rows = n / m.merge_units;
        let tile = g.tile();
        let rows_pad = pad_to(rows, tile);
        let eps = MERGER_LAYER_NORM_EPS_MATCHES_THE_REFERENCE_PATCH_MERGER as f32;
        let xm = g.f16_buf("dv-merger-x", g.coop_x_rows(rows_pad) * m.fc0.k_pad);
        g.layernorm_f16(xn, &m.ln_w_dev, &m.ln_b_dev, &xm, n, m.context_dim, eps)?;
        let h = g.f32_buf("dv-merger-h", rows_pad * m.fc0.n);
        self.proj(&m.fc0, &xm, &h, rows, rows_pad)?;
        let x2 = g.f16_buf("dv-merger-x2", g.coop_x_rows(rows_pad) * m.fc2.k_pad);
        g.gelu_erf_f16(&h, &x2, rows * m.fc0.n)?;
        let y = g.f32_buf("dv-merger-y", rows_pad * m.fc2.n);
        self.proj(&m.fc2, &x2, &y, rows, rows_pad)?;
        g.read_f32(&y, rows * m.fc2.n)
    }

    fn merger_forward(&self, trunk: &[f32], n: usize) -> Result<Vec<f32>> {
        let m = &self.merger;
        let g = &self.g;
        anyhow::ensure!(
            n.is_multiple_of(m.merge_units),
            "PatchMerger: {n} patches not divisible by {}",
            m.merge_units
        );
        let dev = Device::Cpu;
        let x = Tensor::from_slice(trunk, (n, m.context_dim), &dev)?;
        let w = Tensor::from_slice(&m.ln_w, (m.context_dim,), &dev)?;
        let b = Tensor::from_slice(&m.ln_b, (m.context_dim,), &dev)?;
        let normed = crate::deepseek_ocr::sam::layer_norm(
            &x,
            &w,
            &b,
            MERGER_LAYER_NORM_EPS_MATCHES_THE_REFERENCE_PATCH_MERGER,
        )?;
        let rows = n / m.merge_units;
        let hidden = m.context_dim * m.merge_units;
        let grouped: Vec<f32> = normed.flatten_all()?.to_vec1::<f32>()?;

        let tile = g.tile();
        let rows_pad = pad_to(rows, tile);
        let src = g.upload_f32("dv-merger-src", &grouped);
        let xf16 = g.f16_buf("dv-merger-x", g.coop_x_rows(rows_pad) * m.fc0.k_pad);
        g.cast_pad(&src, &xf16, rows, hidden, m.fc0.k_pad)?;
        let h = g.f32_buf("dv-merger-h", rows_pad * m.fc0.n);
        self.proj(&m.fc0, &xf16, &h, rows, rows_pad)?;
        let hv = g.read_f32(&h, rows * m.fc0.n)?;
        let activated = Tensor::from_slice(&hv, (rows, m.fc0.n), &dev)?
            .gelu_erf()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let src2 = g.upload_f32("dv-merger-act", &activated);
        let x2 = g.f16_buf("dv-merger-x2", g.coop_x_rows(rows_pad) * m.fc2.k_pad);
        g.cast_pad(&src2, &x2, rows, m.fc2.k, m.fc2.k_pad)?;
        let y = g.f32_buf("dv-merger-y", rows_pad * m.fc2.n);
        self.proj(&m.fc2, &x2, &y, rows, rows_pad)?;
        g.read_f32(&y, rows * m.fc2.n)
    }

    pub fn merged_rows(&self, num_patches: usize) -> usize {
        num_patches / (MERGE_SIZE * MERGE_SIZE)
    }
}
