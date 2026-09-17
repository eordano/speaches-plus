use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use nv_quant::LinearKind;

#[cfg(feature = "cuda")]
use candle_core::Device;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

pub trait LoraDeltaHook: Send + Sync {
    fn in_features(&self) -> usize;
    fn out_features(&self) -> usize;
    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>>;
}

#[cfg(feature = "cuda")]
impl LoraDeltaHook for crate::lora_slots::LoraHook {
    fn in_features(&self) -> usize {
        crate::lora_slots::LoraHook::in_features(self)
    }

    fn out_features(&self) -> usize {
        crate::lora_slots::LoraHook::out_features(self)
    }

    fn apply(
        &self,
        x2: &Tensor,
        y2: &Tensor,
        win: Option<(usize, usize)>,
    ) -> Result<Option<Tensor>> {
        crate::lora_slots::LoraHook::apply(self, x2, y2, win)?;
        Ok(None)
    }
}

pub struct Linear {
    kind: LinearKind,
    storage: LinearStorage,
    bias: Option<Tensor>,
    in_features: usize,
    out_features: usize,
    lora: std::sync::RwLock<Option<Arc<dyn LoraDeltaHook>>>,
}

#[cfg(feature = "cuda")]
pub struct FusedPreNorm<'a> {
    pub weight_bf16: &'a Tensor,
    pub eps: f32,
}

thread_local! {

    static FORCE_DENSE_BF16: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn with_dense_bf16<R>(f: impl FnOnce() -> R) -> R {
    struct Guard(bool);
    impl Drop for Guard {
        fn drop(&mut self) {
            FORCE_DENSE_BF16.with(|c| c.set(self.0));
        }
    }
    let prev = FORCE_DENSE_BF16.with(|c| {
        let p = c.get();
        c.set(true);
        p
    });
    let _g = Guard(prev);
    f()
}

#[cfg(feature = "cuda")]
fn force_dense_bf16_enabled() -> bool {
    FORCE_DENSE_BF16.with(|c| c.get())
}

pub fn fp8_scale_mode_from(raw: Option<&str>) -> nv_quant::fp8::Fp8ScaleMode {
    use nv_quant::fp8::Fp8ScaleMode;
    match raw {
        Some(v)
            if v.eq_ignore_ascii_case("tensor")
                || v.eq_ignore_ascii_case("per_tensor")
                || v.eq_ignore_ascii_case("per-tensor")
                || v == "0" =>
        {
            Fp8ScaleMode::PerTensor
        }
        _ => Fp8ScaleMode::PerOuterRow,
    }
}

pub fn fp8_scale_mode() -> nv_quant::fp8::Fp8ScaleMode {
    fp8_scale_mode_from(std::env::var("NV_FP8_SCALE_MODE").ok().as_deref())
}

#[cfg(feature = "cuda")]
fn fp8_per_row_probe(
    runner: &Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ordinal: usize,
) -> std::result::Result<(), String> {
    static PROBE: std::sync::Mutex<
        std::collections::BTreeMap<usize, std::result::Result<(), String>>,
    > = std::sync::Mutex::new(std::collections::BTreeMap::new());
    if let Some(v) = PROBE.lock().unwrap().get(&ordinal) {
        return v.clone();
    }
    let verdict = match runner.lock() {
        Err(e) => return Err(format!("fp8 runner mutex poisoned: {e}")),
        Ok(mut r) => r
            .probe_per_row_scale_support()
            .map_err(|e| format!("{e:#}")),
    };
    PROBE
        .lock()
        .unwrap()
        .entry(ordinal)
        .or_insert(verdict)
        .clone()
}

pub fn fp8_weight_payload(
    weight_host: &[half::bf16],
    out_features: usize,
    in_features: usize,
    checkpoint_rows: Option<&[f32]>,
    mode: nv_quant::fp8::Fp8ScaleMode,
) -> Result<(Vec<u8>, Vec<f32>)> {
    use nv_quant::fp8::{
        quantize_e4m3_per_row, quantize_e4m3_per_tensor, quantize_e4m3_with_row_scales,
        Fp8ScaleMode,
    };
    if weight_host.len() != out_features * in_features {
        anyhow::bail!(
            "fp8 weight payload: {} values for a [{out_features}, {in_features}] weight",
            weight_host.len()
        );
    }
    match (mode, checkpoint_rows) {
        (Fp8ScaleMode::PerOuterRow, Some(rows)) => {
            if rows.len() != out_features {
                anyhow::bail!(
                    "fp8 weight payload: {} checkpoint scales for {out_features} output rows",
                    rows.len()
                );
            }
            let bytes =
                quantize_e4m3_with_row_scales(weight_host, out_features, in_features, rows)?;
            Ok((bytes, rows.to_vec()))
        }
        (Fp8ScaleMode::PerOuterRow, None) => {
            quantize_e4m3_per_row(weight_host, out_features, in_features)
        }
        (Fp8ScaleMode::PerTensor, Some(rows)) => {
            let first = *rows
                .first()
                .ok_or_else(|| anyhow::anyhow!("empty checkpoint weight_scale"))?;
            if rows.iter().any(|s| *s != first) {
                anyhow::bail!(
                    "NV_FP8_SCALE_MODE=tensor cannot represent this checkpoint: its \
                     weight_scale varies across output rows ({} distinct values), and \
                     collapsing it to one scale would silently coarsen the checkpoint's own \
                     granularity. Unset NV_FP8_SCALE_MODE to use per-row scaling.",
                    {
                        let mut v = rows.to_vec();
                        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                        v.dedup();
                        v.len()
                    }
                );
            }
            let bytes =
                quantize_e4m3_with_row_scales(weight_host, out_features, in_features, rows)?;
            Ok((bytes, vec![first; out_features]))
        }
        (Fp8ScaleMode::PerTensor, None) => {
            let (bytes, scale) = quantize_e4m3_per_tensor(weight_host);
            Ok((bytes, vec![scale; out_features]))
        }
    }
}

pub fn small_m_det_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

#[cfg(feature = "cuda")]
pub const NVFP4_M1_GEMV_IS_A_SERVING_DEFAULT_WHERE_A_SERVING_AB_LANDED_AND_OPT_IN_EVERYWHERE_ELSE: &str =
    "NV_NVFP4_M1_GEMV=1 routes short-row NVFP4 linears through nvfp4_gemv_bf16act (bf16 \
     activations, no activation quantize, no m=16 padding) -- it outruns the padded-16 LT route \
     at 31B decode shapes and is numerically TIGHTER than the default because it skips the fp4 \
     activation round-trip. The serving A/B this key once waited on has landed, and the argmax \
     question it named is answered rather than deferred: on Gemma-4-31B-IT-NVFP4 the key alone \
     scores 16.308 -> 15.566 ppl with the shuffled control intact, the three-key record it \
     belongs to scores 16.344 on the graphed serving shape against a 16.276 basis, and the \
     128-step real-chat-template greedy chains fork where the zero-env eager and zero-env \
     graphed chains already fork; on ig1/Qwen3.5-9B-NVFP4 it scores 4.862 -> 4.773 alone and \
     rides an 11-arm record at 4.833, whose eager and graphed chains are token-identical 128/128. \
     Verify blocks take this same kernel row by row up to \
     NVFP4_VERIFY_BLOCK_ROWS_LOOPED_THROUGH_THE_M1_GEMV_SO_A_DRAFT_BLOCK_REPRODUCES_DECODE, so a \
     draft block reproduces decode bit-for-bit instead of disagreeing with it. This crate cannot \
     see a model family, so the key stays default-off HERE and is turned on per family by the \
     serving engine: oapi::chat_engine::build names which checkpoints arm it and which env \
     restores the LT route -- today gemma4 NVFP4 dense and the qwen3.5-9B dense cuda opt-in. \
     Every other caller still opts in explicitly.";

#[cfg(feature = "cuda")]
pub const A_VERIFY_BLOCK_MUST_TAKE_THE_SAME_KERNEL_DECODE_TAKES_OR_IT_CANNOT_REPRODUCE_DECODES_ROWS: &str =
    "speculative verify exists to recompute, in one m-row block, exactly what m=1 decode would \
     have produced row by row. Routing m=1 to nvfp4_gemv_bf16act while routing the m-row block to \
     the padded cutlass block-scaled tile makes the two arithmetically different, and \
     gemma4_verify_forward::gemma4_verify_batch_vs_sequential reads that difference as a \
     row-argmax mismatch. Looping the m=1 gemv once per row keeps every verify row \
     bit-identical to the decode row it stands in for: same kernel, same n, same k, same alpha, \
     same input bytes";

#[cfg(feature = "cuda")]
pub const LOOPING_THE_M1_GEMV_COSTS_ONE_FULL_WEIGHT_READ_PER_ROW_SO_IT_ONLY_OUTRUNS_THE_PADDED_CUTLASS_TILE_AT_ONE_ROW: &str =
    "measured on the gemma4-31B NVFP4 projection shapes (qkv 20736x5376, o 5376x8192 and \
     5376x16384, gate_up 43008x5376, down 5376x21504), release, quiet card, 200 iterations, \
     4-way weight rotation: summed over one sliding layer the padded cutlass block-scaled tile \
     is flat at 256-286 us across m=1..16, while the looped gemv runs 223 us at m=1, 378 at \
     m=2 and 2974 at m=16. So rows above one buy verify/decode row agreement with bandwidth, \
     not with speed. The route that would buy the same agreement without paying it is \
     gemm_nvfp4_w4a16_mk_dual (kMMax=16, bf16 activations, one weight read for all rows), which \
     no serving path wires today";

#[cfg(feature = "cuda")]
pub const NVFP4_VERIFY_BLOCK_ROWS_LOOPED_THROUGH_THE_M1_GEMV_SO_A_DRAFT_BLOCK_REPRODUCES_DECODE:
    usize = 8;

#[cfg(feature = "cuda")]
pub const NVFP4_GEMV_MROWS_LADDER_MATCHES_THE_CUDA_SWITCH_AND_ROWS_ABOVE_IT_STILL_LOOP: usize = 8;

#[cfg(feature = "cuda")]
pub fn nvfp4_gemv_mrows_case_instantiated_mirroring_gemv_act_mrows_case_instantiated(
    rows: usize,
) -> bool {
    rows <= NVFP4_GEMV_MROWS_LADDER_MATCHES_THE_CUDA_SWITCH_AND_ROWS_ABOVE_IT_STILL_LOOP
        || rows == 12
        || rows == 16
        || rows == 24
}

#[cfg(feature = "cuda")]
pub const ONE_MROWS_LAUNCH_READS_THE_WEIGHT_BLOCK_ONCE_FOR_EVERY_VERIFY_ROW_INSTEAD_OF_ONCE_PER_ROW: &str =
    "nvfp4_gemv_bf16act_kernel<MROWS> keeps one warp on output row n and accumulates MROWS \
     independent acc[] against MROWS activation rows from a single uint2 weight load, so an \
     m-row verify block costs ONE full weight read and ONE launch where the row loop cost m of \
     each -- the cost LOOPING_THE_M1_GEMV_COSTS_ONE_FULL_WEIGHT_READ_PER_ROW_SO_IT_ONLY_OUTRUNS_\
     THE_PADDED_CUTLASS_TILE_AT_ONE_ROW measured. It stays bit-identical to the row loop rather \
     than merely close to it: acc[r] runs the same fmaf chain, in the same order, over the same \
     scale and the same input bytes that the MROWS=1 instantiation runs, and MROWS=1 IS the \
     instantiation nv_kernels_nvfp4_gemv_bf16act now calls, so decode and verify share one \
     kernel body by construction. That keeps \
     A_VERIFY_BLOCK_MUST_TAKE_THE_SAME_KERNEL_DECODE_TAKES_OR_IT_CANNOT_REPRODUCE_DECODES_ROWS \
     true while removing the per-row cost that held NV_NVFP4_M1_GEMV off every spec-decode \
     deployment";

#[cfg(feature = "cuda")]
pub const THE_VERIFY_BLOCK_BATCHES_THE_GEMV_BUT_NEVER_THE_NORM_BECAUSE_A_MULTI_ROW_RMSNORM_TAKES_A_DIFFERENT_LADDER_THAN_DECODES_BATCH_ONE: &str =
    "nv_kernels_rmsnorm_bf16 tries rmsnorm_m1_split_launch and rmsnorm_m1_wide_launch before the \
     stock one-block-per-row kernel, and those rungs are batch-one arms -- rmsnorm.cu names it \
     g_norm_m1_split_partials_one_row_per_rung_because_the_arm_serves_batch_one_only. Folding an \
     m-row block in one rmsnorm call would therefore reduce on a DIFFERENT ladder than the \
     batch-one call decode makes, and the folded row would stop being bit-identical to decode's. \
     So the prenorm route folds each row with its own batch-one rmsnorm and batches only the \
     gemv that follows: m norm launches plus one gemv, against the m norm plus m gemv launches \
     the row loop paid, with the weight read dropping from m to 1";

#[cfg(feature = "cuda")]
pub const THE_STAGED_FP8_GEMV_KEEPS_BF16_ACTIVATIONS_AND_PAYS_M_SCALAR_MACS_PER_WEIGHT_BYTE_WHILE_THE_CUBLASLT_E4M3_TILE_QUANTISES_THE_ACTIVATIONS_AND_PAYS_NONE: &str =
    "measured on the qwen3.8-27B fp8 projection shapes, release, quiet card, 200 iterations over \
     a 768 MB rotating weight set so nothing is served from L2. The cublasLt e4m3 tile route \
     (rowquant_e4m3 + cublasLtMatmul + scale_rowcol_bf16) is FLAT in m: gdn.in_proj_qkv \
     10240x5120 reads 1104 GB/s at m=1 and 1095 at m=16, gdn.out_proj 5120x6144 reads 855 and \
     851, attn.o_proj 855 and 852 -- the x1.03 round 23's kernel census measured on vLLM's own \
     SM120 FP8 pingpong tile. The staged gemv is NOT flat: the same shape reads 1421 GB/s at \
     m=1, 942 at m=4, 416 at m=8 and REFUSES from m=9 because staging the whole x row for m \
     rows needs m*(K/16)*9*4 bytes against a 101376-byte dynamic-shared opt-in cap. A scalar \
     FFMA route owes m multiply-adds per weight byte and a tile owes none, so the two routes \
     cross and the crossing is the whole of the m-row fp8 residual";

#[cfg(feature = "cuda")]
pub const E4M3_STAGED_GEMV_OUTRUNS_THE_TILE_TO_THIS_MANY_ROWS_AND_LOSES_WIDER_SO_THE_TILE_MINIMUM_IS_A_CROSSOVER_NOT_A_CAP: usize = 3;

#[cfg(feature = "cuda")]
pub const VERIFY_ROWS_TEMPLATE_CAP_8_THE_RANGE_THAT_MUST_STAY_ON_ONE_ARITHMETIC: usize = 8;

#[cfg(feature = "cuda")]
fn verify_one_arithmetic_rows_ceiling_env_nv_verify_one_arith_rows_default_8_max_24_serving_never_exceeds_8_only_the_mrow_ladder_raises_it(
) -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("NV_VERIFY_ONE_ARITH_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(VERIFY_ROWS_TEMPLATE_CAP_8_THE_RANGE_THAT_MUST_STAY_ON_ONE_ARITHMETIC)
            .clamp(
                VERIFY_ROWS_TEMPLATE_CAP_8_THE_RANGE_THAT_MUST_STAY_ON_ONE_ARITHMETIC,
                24,
            )
    })
}

#[cfg(feature = "cuda")]
pub const VERIFY_INTERIOR_REACHES_3_TO_7_ARE_THE_SHAPES_WHOSE_TILE_FALLBACK_SPLIT_THE_VERIFY_RANGE:
    std::ops::RangeInclusive<usize> = 3..=7;

#[cfg(feature = "cuda")]
pub const LT_TILE_ROWS_PER_CHUNK_6_THE_WIDEST_ROW_COUNT_WHOSE_HEURISTIC_PLAN_FAMILY_COVERS_EVERY_SHIPPED_VERIFY_CELL: usize =
    6;

#[cfg(feature = "cuda")]
pub const LT_TILE_SPAN_4_THE_PRODUCTION_MTP_VERIFY_LAUNCH_WHOSE_PLAN_FAMILY_ALSO_COVERS_M5_AND_M6_AT_THE_CHEAPEST_ROW_COUNT: usize =
    4;

#[cfg(feature = "cuda")]
pub const A_ROW_CHUNKED_LT_TILE_RIDES_THE_M6_PLAN_BECAUSE_THE_HEURISTIC_MOVES_PLANS_BETWEEN_M6_AND_M7: &str =
    "cublasLtMatmulAlgoGetHeuristic is re-queried per call with the true m in the activation \
     layout, and at the fp8-resident down_proj shape (n=5120, k=17408: layers 56..=63 of the \
     Qwen3.8 checkpoint keep their MLPs e4m3, and that shape's staged-gemv reach is 2, so every \
     verify-sized call lands on this tile) the chosen plan's shared-row outputs form bitwise \
     families {m2,m3} {m4,m5,m6} {m7,m8} -- the suite covering this is \
     nv-quant/tests/fp8_lt_row0_m_invariance (down_proj family map, and under NV_Q38_VERIFY_TC=1 \
     the same tile serves every e4m3 projection, where (10240,5120) moves plans at m=7 and \
     (12288,5120) at m=8: the all-shapes span gate). A 7- or 8-row verify therefore computed a \
     different row 0 than the 4..6-row verifies for the same anchor and kv: the one-time 6->7 \
     trunk transition that capped suffix drafts at 5. A 7- or 8-row verify runs as ONE m-row \
     launch riding the algo the heuristic returns for the 4-row layout -- the production mtp \
     verify plan, whose family also covers m5 and m6 -- so every row, deep rows included, \
     rides the {m4,m5,m6} plan family every emission-bearing mtp and suffix verify row rides \
     instead of an m=7/m=8 plan (the row-0 flip) or an m=1/m=2 tail plan (the deep-row \
     near-tie flip), at single-launch weight-read cost; the launch is bitwise the \
     (0,4)+(m-4,4) span composition, which stays as the fallback should the m4 algo ever \
     refuse the wider problem (the planned-vs-composition and position-invariance gates in \
     the same suite), so row 0 of every verify in 2..=8 is the m<=6 arithmetic and every \
     shipped cell (m<=6 single launch) is an untouched call";

#[cfg(feature = "cuda")]
pub const A_ROW_CHUNKED_GEMV_IS_BITWISE_THE_SINGLE_LAUNCH_BECAUSE_EVERY_ROW_ACCUMULATES_ALONE_IN_ONE_K_ORDER: &str =
    "each gemv_e4m3_mk template accumulates output row j over the identical lane-strided K walk \
     with acc[j] private to the row, so launching rows [s, s+c) with offset activation and \
     output pointers reproduces those rows of one wide launch bit for bit; the suite covering \
     this is nv-kernels/tests/gemv_e4m3_mk_row0_m_invariance. The k=6144 o_proj and gdn \
     out_proj fit the staging smem for seven rows and not eight, so before the chunk arm an \
     eight-row verify silently fell to the activation-quantizing cublasLt tile in 64 layers and \
     returned a different row 0 for the same anchor and kv than the four-row verify -- a chain \
     length must never change what an earlier row computes. Rows past \
     VERIFY_ROWS_TEMPLATE_CAP_8 (the m=16 drafter fc) and shapes whose reach is outside 3..=7 \
     (k=17408 reaches 2: the production m=4 verify already serves it from the tile) keep their \
     routes, so the shipped m<=7 cells and the m=16 drafter cell are untouched launches";

#[cfg(feature = "cuda")]
pub const E4M3_GROUP_LEAVES_THE_M1_ROUTE_ALONE_BECAUSE_ITS_TWO_ROWS_PER_WARP_KERNEL_IS_ALREADY_THE_TUNED_ONE_ROW_SHAPE: usize = 2;

#[cfg(feature = "cuda")]
pub const THE_GROUP_GEMV_NEEDS_THIS_MANY_OUTPUT_ROWS_BEFORE_ITS_K_TILE_PAYS_FOR_ITS_EXTRA_SYNCS_BECAUSE_A_NARROW_N_GIVES_EACH_BLOCK_TOO_FEW_WEIGHT_BYTES_PER_STAGED_TILE: usize =
    2048;

#[cfg(feature = "cuda")]
pub const THE_TILE_NEEDS_THIS_MANY_OUTPUT_ROWS_BEFORE_IT_BEATS_THE_STAGED_GEMV_BECAUSE_A_NARROW_N_STARVES_ITS_CTAS: usize =
    2048;

#[cfg(feature = "cuda")]
fn e4m3_route_line_once_per_slot_and_row_count_because_a_single_once_only_ever_reports_whichever_shape_warmed_first(
    slot: usize,
    branch: &'static str,
    leading: usize,
    n: usize,
    k: usize,
    extra: &str,
) {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEEN: [AtomicU32; 3] = [AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)];
    let bit = 1u32 << (leading.min(31) as u32);
    if SEEN[slot].fetch_or(bit, Ordering::Relaxed) & bit == 0 {
        eprintln!("[linear] {branch} m={leading} n={n} k={k} {extra}");
    }
}

#[cfg(feature = "cuda")]
thread_local! {
    static QWEN38_BATCH_E4M3_GROUP_SCOPE:
        std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "cuda")]
pub struct Qwen38BatchE4m3GroupScopeGuard {
    previous: usize,
    same_thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(feature = "cuda")]
impl Qwen38BatchE4m3GroupScopeGuard {
    pub fn enter(batch_size: usize) -> Self {
        let previous = QWEN38_BATCH_E4M3_GROUP_SCOPE.with(|scope| {
            let previous = scope.get();
            scope.set(batch_size);
            previous
        });
        Self {
            previous,
            same_thread: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for Qwen38BatchE4m3GroupScopeGuard {
    fn drop(&mut self) {
        let _ = QWEN38_BATCH_E4M3_GROUP_SCOPE.try_with(|scope| scope.set(self.previous));
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy)]
enum E4m3GroupPolicy {
    SixOrEightLaneBatchDefault,
    Disabled,
    Rows(i32),
}

#[cfg(feature = "cuda")]
impl E4m3GroupPolicy {
    fn from_env_value(raw: Option<&str>) -> Self {
        match raw {
            None => Self::SixOrEightLaneBatchDefault,
            Some("1") => Self::Rows(2),
            Some(v) => v
                .parse::<i32>()
                .ok()
                .filter(|r| [1, 2, 4].contains(r))
                .map(Self::Rows)
                .unwrap_or(Self::Disabled),
        }
    }

    fn rows_per_warp(self) -> Option<i32> {
        match self {
            Self::SixOrEightLaneBatchDefault => QWEN38_BATCH_E4M3_GROUP_SCOPE
                .with(|scope| matches!(scope.get(), 6 | 8).then_some(2)),
            Self::Disabled => None,
            Self::Rows(rows) => Some(rows),
        }
    }
}

#[cfg(feature = "cuda")]
fn e4m3_group_rows_per_warp() -> Option<i32> {
    static POLICY: std::sync::OnceLock<E4m3GroupPolicy> = std::sync::OnceLock::new();
    POLICY
        .get_or_init(|| {
            E4m3GroupPolicy::from_env_value(std::env::var("NV_Q38_E4M3_GROUP").ok().as_deref())
        })
        .rows_per_warp()
}

#[cfg(feature = "cuda")]
fn e4m3_tile_min_m() -> Option<usize> {
    static M: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *M.get_or_init(|| match std::env::var("NV_Q38_E4M3_TILE_MIN_M").ok().as_deref() {
        None | Some("") => None,
        Some("1") => Some(
            E4M3_STAGED_GEMV_OUTRUNS_THE_TILE_TO_THIS_MANY_ROWS_AND_LOSES_WIDER_SO_THE_TILE_MINIMUM_IS_A_CROSSOVER_NOT_A_CAP
                + 1,
        ),
        Some(v) => v.parse::<usize>().ok().filter(|m| *m >= 2),
    })
}

#[cfg(feature = "cuda")]
fn nvfp4_m1_gemv_opted_in() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_NVFP4_M1_GEMV").ok().as_deref() == Some("1"))
}

#[cfg(feature = "cuda")]
fn nvfp4_gemv_row_block_ceiling() -> usize {
    static ROWS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("NV_NVFP4_GEMV_ROWS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(NVFP4_VERIFY_BLOCK_ROWS_LOOPED_THROUGH_THE_M1_GEMV_SO_A_DRAFT_BLOCK_REPRODUCES_DECODE)
            .max(1)
    })
}

#[cfg(feature = "cuda")]
fn small_m_det_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| small_m_det_from(std::env::var("NV_BF16_SMALLM_DET").ok().as_deref()))
}

#[cfg(feature = "cuda")]
fn nvfp4_quant_fullpad_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_NVFP4_QUANT_FULLPAD").is_ok_and(|v| v != "0"))
}

#[cfg(feature = "cuda")]
static CFX_RUNTIME_OVERRIDE_MINUS1_UNSET_0_OFF_1_ON_FOR_INTERLEAVED_AB_ONLY:
    std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

#[cfg(feature = "cuda")]
pub fn nvfp4_cfx_set_runtime_override_for_interleaved_ab(on: Option<bool>) {
    let v = match on {
        None => -1i8,
        Some(false) => 0,
        Some(true) => 1,
    };
    CFX_RUNTIME_OVERRIDE_MINUS1_UNSET_0_OFF_1_ON_FOR_INTERLEAVED_AB_ONLY
        .store(v, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
static CFX_PREFILL_TOTAL_TOKENS_HINT_0_UNSET_SET_PER_PREFILL_BY_THE_SERVING_LOOP:
    std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

#[cfg(feature = "cuda")]
pub fn nvfp4_cfx_set_prefill_total_tokens_hint(total_tokens: usize) {
    CFX_PREFILL_TOTAL_TOKENS_HINT_0_UNSET_SET_PER_PREFILL_BY_THE_SERVING_LOOP
        .store(total_tokens, std::sync::atomic::Ordering::Relaxed);
}

#[cfg(feature = "cuda")]
fn nvfp4_cfx_min_context_tokens_the_measured_e2e_crossover_is_between_8k_loss_and_16k_win() -> usize
{
    static T: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("NV_Q38_CFX_MIN_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14336)
    })
}

#[cfg(feature = "cuda")]
fn nvfp4_cfx_env_opt_in_nv_q38_mlp_fp4_cfx_1_captured_colfax_cubins_beat_dp_k256_by_4_to_5pct_at_the_serving_chunk(
) -> bool {
    match CFX_RUNTIME_OVERRIDE_MINUS1_UNSET_0_OFF_1_ON_FOR_INTERLEAVED_AB_ONLY
        .load(std::sync::atomic::Ordering::Relaxed)
    {
        0 => return false,
        1 => return true,
        _ => {}
    }
    static FORCE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    if let Some(v) = FORCE.get_or_init(|| match std::env::var("NV_Q38_MLP_FP4_CFX").as_deref() {
        Ok("1") => Some(true),
        Ok("0") => Some(false),
        _ => None,
    }) {
        return *v;
    }
    let total = CFX_PREFILL_TOTAL_TOKENS_HINT_0_UNSET_SET_PER_PREFILL_BY_THE_SERVING_LOOP
        .load(std::sync::atomic::Ordering::Relaxed);
    total
        >= nvfp4_cfx_min_context_tokens_the_measured_e2e_crossover_is_between_8k_loss_and_16k_win()
}

#[cfg(feature = "cuda")]
pub fn nvfp4_cfx_effective_route_selected_for_current_prefill_hint() -> bool {
    nvfp4_cfx_env_opt_in_nv_q38_mlp_fp4_cfx_1_captured_colfax_cubins_beat_dp_k256_by_4_to_5pct_at_the_serving_chunk()
}

#[cfg(feature = "cuda")]
const NVFP4_CFX_CUBIN_EMBEDDED_SO_THE_DEFAULT_ON_16K_PATH_NEEDS_NO_CACHE_DIR: [(
    usize,
    usize,
    usize,
    &str,
    &[u8],
); 3] = [
    (
        4096,
        17408,
        5120,
        "gateup.cubin",
        include_bytes!("cfx_cubins/gateup.cubin"),
    ),
    (
        4096,
        34816,
        5120,
        "gateupfused.cubin",
        include_bytes!("cfx_cubins/gateupfused.cubin"),
    ),
    (
        4096,
        5120,
        17408,
        "down.cubin",
        include_bytes!("cfx_cubins/down.cubin"),
    ),
];

#[cfg(feature = "cuda")]
fn nvfp4_cfx_load_all_once_from_cubin_dir() {
    static LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    LOADED.get_or_init(|| {
        let dir_override = std::env::var("NV_Q38_CFX_CUBIN_DIR").ok();
        for (m, n, k, fname, embedded) in
            NVFP4_CFX_CUBIN_EMBEDDED_SO_THE_DEFAULT_ON_16K_PATH_NEEDS_NO_CACHE_DIR
        {
            let bytes: std::borrow::Cow<[u8]> = match &dir_override {
                Some(dir) => match std::fs::read(format!("{dir}/{fname}")) {
                    Ok(b) => std::borrow::Cow::Owned(b),
                    Err(e) => {
                        eprintln!(
                            "[linear] NV_Q38_CFX_CUBIN_DIR set but {dir}/{fname} unreadable ({e}); \
                             shape {m}x{n}x{k} stays on the cutlass arm"
                        );
                        continue;
                    }
                },
                None => std::borrow::Cow::Borrowed(embedded),
            };
            let rc = unsafe {
                nv_kernels::cuda::nvfp4_cfx_load(
                    bytes.as_ptr() as *const std::ffi::c_void,
                    bytes.len(),
                    m as i32,
                    n as i32,
                    k as i32,
                )
            };
            if rc != 0 {
                eprintln!(
                    "[linear] nvfp4_cfx_load({fname}) rc={rc}; shape {m}x{n}x{k} stays on the \
                     cutlass arm"
                );
            }
        }
    });
}

#[cfg(feature = "cuda")]
fn nvfp4_cfx_route_ready(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    if !nvfp4_cfx_env_opt_in_nv_q38_mlp_fp4_cfx_1_captured_colfax_cubins_beat_dp_k256_by_4_to_5pct_at_the_serving_chunk()
    {
        return false;
    }
    if unsafe { nv_kernels::cuda::nvfp4_cfx_shape_index(m as i32, n as i32, k as i32) } < 0 {
        return false;
    }
    if !stream_is_capturing(stream) {
        nvfp4_cfx_load_all_once_from_cubin_dir();
    }
    unsafe { nv_kernels::cuda::nvfp4_cfx_ready(m as i32, n as i32, k as i32) == 1 }
}

#[cfg(feature = "cuda")]
thread_local! {
    static VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD:
        std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(feature = "cuda")]
pub fn verify_tc_fp8_lt_gemm_scope_active() -> bool {
    VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD.with(|c| c.get())
}

#[cfg(feature = "cuda")]
pub struct VerifyTcFp8LtGemmScopeGuard {
    was_off_so_drop_restores_off: bool,
}

#[cfg(feature = "cuda")]
impl VerifyTcFp8LtGemmScopeGuard {
    pub fn enter_if(on: bool) -> Self {
        let was_off = !verify_tc_fp8_lt_gemm_scope_active();
        if on && was_off {
            VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD
                .with(|c| c.set(true));
        }
        Self {
            was_off_so_drop_restores_off: on && was_off,
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for VerifyTcFp8LtGemmScopeGuard {
    fn drop(&mut self) {
        if self.was_off_so_drop_restores_off {
            let _ = VERIFY_TC_FP8_LT_GEMM_SCOPE_SET_ONLY_AROUND_A_VERIFY_CHAIN_FORWARD
                .try_with(|c| c.set(false));
        }
    }
}

#[cfg(feature = "cuda")]
fn stream_is_capturing(stream: &std::sync::Arc<cudarc::driver::CudaStream>) -> bool {
    use cudarc::driver::sys as drv;
    let mut st = drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
    let rc = unsafe { drv::cuStreamIsCapturing(stream.cu_stream(), &mut st) };
    rc == drv::CUresult::CUDA_SUCCESS
        && st != drv::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
}

enum LinearStorage {
    Bf16 {
        weight: Tensor,

        weight_t: Option<Tensor>,
    },
    GgmlQuant(Arc<GgmlQuantStorage>),
    #[cfg(feature = "cuda")]
    Fp8E4m3(Arc<Fp8Storage>),
    #[cfg(feature = "cuda")]
    Nvfp4(Arc<Nvfp4Storage>),
}

struct GgmlQuantStorage {
    qm: candle_core::quantized::QMatMul,
}

#[cfg(feature = "cuda")]
struct Fp8Storage {
    weight_u8: cudarc::driver::CudaSlice<u8>,
    #[allow(dead_code)]
    a_scale_dev: cudarc::driver::CudaSlice<f32>,

    b_scale_dev: cudarc::driver::CudaSlice<f32>,

    b_scale_rows_dev: cudarc::driver::CudaSlice<f32>,

    weight_scale_rows: Vec<f32>,
    scale_mode: nv_quant::fp8::Fp8ScaleMode,
    cuda_device: candle_core::CudaDevice,
    runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    marlin_fp8_twin_built_once_from_the_resident_e4m3_rows:
        std::sync::OnceLock<Option<Arc<MarlinFp8Prepared>>>,
    dx_fp8_twin_of_the_transposed_weight_bytes_with_the_channel_scale_folded_into_dy:
        std::sync::OnceLock<Option<Arc<Fp8DxTwin>>>,
}

#[cfg(feature = "cuda")]
pub struct Fp8DxTwin {
    pub wq_t: cudarc::driver::CudaSlice<u8>,
    pub unit_scales_for_the_k_in_rows: cudarc::driver::CudaSlice<f32>,
    pub channel_scale_bf16: Tensor,
}

#[cfg(feature = "cuda")]
pub struct MarlinFp8Prepared {
    pub b_q_marlin: cudarc::driver::CudaSlice<u32>,
    pub scales_bf16: cudarc::driver::CudaSlice<u16>,
    pub n: usize,
    pub k: usize,
}

#[cfg(feature = "cuda")]
const MARLIN_FP8_ROWS_MAX_16_IS_THE_INSTANTIATED_THREAD_M_BLOCKS_1: usize = 16;

#[cfg(feature = "cuda")]
fn marlin_fp8_rows_opt_in() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_MARLIN_FP8_ROWS")
}

#[cfg(feature = "cuda")]
fn build_marlin_fp8_twin(s: &Fp8Storage, n: usize, k: usize) -> Result<MarlinFp8Prepared> {
    anyhow::ensure!(
        s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow && s.weight_scale_rows.len() == n,
        "marlin fp8 twin needs per-outer-row scales (n={n} k={k} mode={:?})",
        s.scale_mode
    );
    let stream = s.cuda_device.cuda_stream();
    let bytes = stream
        .clone_dtoh(&s.weight_u8)
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin dtoh: {e:?}"))?;
    marlin_fp8_twin_from_e4m3_rows_host(&s.cuda_device, &bytes, &s.weight_scale_rows, n, k)
}

#[cfg(feature = "cuda")]
pub fn marlin_fp8_twin_from_e4m3_rows_dev(
    dev: &candle_core::CudaDevice,
    weight_bytes: &cudarc::driver::CudaSlice<u8>,
    row_scale: &cudarc::driver::CudaSlice<f32>,
    n: usize,
    k: usize,
) -> Result<MarlinFp8Prepared> {
    let stream = dev.cuda_stream();
    let bytes = stream
        .clone_dtoh(weight_bytes)
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin dtoh: {e:?}"))?;
    let scales = stream
        .clone_dtoh(row_scale)
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin dtoh scales: {e:?}"))?;
    marlin_fp8_twin_from_e4m3_rows_host(dev, &bytes, &scales, n, k)
}

#[cfg(feature = "cuda")]
pub fn marlin_fp8_twin_from_e4m3_rows_host(
    dev: &candle_core::CudaDevice,
    bytes: &[u8],
    scale_rows: &[f32],
    n: usize,
    k: usize,
) -> Result<MarlinFp8Prepared> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    anyhow::ensure!(
        n % 64 == 0 && k % 16 == 0,
        "marlin fp8 twin needs n%64==0 and k%16==0 (n={n} k={k})"
    );
    anyhow::ensure!(
        scale_rows.len() == n,
        "marlin fp8 twin: {} scales for n={n}",
        scale_rows.len()
    );
    let stream = dev.cuda_stream();
    anyhow::ensure!(
        bytes.len() == n * k,
        "marlin fp8 twin: {} weight bytes != n*k",
        bytes.len()
    );
    let kmajor = nv_quant::marlin_fp8::kmajor_int32_from_e4m3_rows(bytes, n, k);
    let scales = nv_quant::marlin_fp8::marlin_per_channel_scales_bf16(scale_rows, n);
    #[allow(deprecated)]
    let d_kmajor: cudarc::driver::CudaSlice<u32> = stream
        .clone_htod(&kmajor)
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin htod: {e:?}"))?;
    let mut b_q_marlin: cudarc::driver::CudaSlice<u32> = stream
        .alloc_zeros::<u32>(k * n / 4)
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin alloc: {e:?}"))?;
    let rc = unsafe {
        let (pi, _g1) = d_kmajor.device_ptr(&stream);
        let (po, _g2) = b_q_marlin.device_ptr_mut(&stream);
        nv_kernels::cuda::marlin_repack_w4a16(
            stream.cu_stream() as *mut std::ffi::c_void,
            pi as *const std::ffi::c_void,
            po as *mut std::ffi::c_void,
            k as i32,
            n as i32,
            8,
        )
    };
    anyhow::ensure!(rc == 0, "marlin fp8 twin repack rc={rc}");
    #[allow(deprecated)]
    let scales_bf16: cudarc::driver::CudaSlice<u16> = stream
        .clone_htod(&scales)
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin htod scales: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("marlin fp8 twin sync: {e:?}"))?;
    Ok(MarlinFp8Prepared {
        b_q_marlin,
        scales_bf16,
        n,
        k,
    })
}

#[cfg(feature = "cuda")]
pub fn marlin_fp8_rows_gemm_into(
    stream: &Arc<cudarc::driver::CudaStream>,
    tw: &MarlinFp8Prepared,
    x_bf16_ptr: u64,
    m: usize,
    out: &mut cudarc::driver::CudaSlice<half::bf16>,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    anyhow::ensure!(
        (1..=MARLIN_FP8_ROWS_MAX_16_IS_THE_INSTANTIATED_THREAD_M_BLOCKS_1).contains(&m)
            && out.len() >= m * tw.n,
        "marlin fp8 rows: m={m} out={} n={}",
        out.len(),
        tw.n
    );
    let mut elems = 0i32;
    let rc = unsafe { nv_kernels::cuda::marlin_workspace_elems(&mut elems as *mut i32) };
    anyhow::ensure!(rc == 0 && elems > 0, "marlin workspace elems rc={rc}");
    let workspace: cudarc::driver::CudaSlice<i32> = stream
        .alloc_zeros::<i32>(elems as usize)
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut c_tmp_floats = 0i64;
    let rc = unsafe { nv_kernels::cuda::marlin_nvfp4_c_tmp_floats(&mut c_tmp_floats as *mut i64) };
    anyhow::ensure!(rc == 0 && c_tmp_floats > 0, "marlin c_tmp floats rc={rc}");
    let mut c_tmp: cudarc::driver::CudaSlice<f32> = unsafe {
        stream
            .alloc::<f32>(c_tmp_floats as usize)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let rc = {
        let (bp, _gb) = tw.b_q_marlin.device_ptr(stream);
        let (sp, _gs) = tw.scales_bf16.device_ptr(stream);
        let (cp, _gc) = out.device_ptr_mut(stream);
        let (tp, _gt) = c_tmp.device_ptr_mut(stream);
        let (wp, _gw) = workspace.device_ptr(stream);
        unsafe {
            nv_kernels::cuda::marlin_gemm_fp8_w8a16(
                stream.cu_stream() as *mut std::ffi::c_void,
                x_bf16_ptr as *const std::ffi::c_void,
                bp as *const std::ffi::c_void,
                sp as *const std::ffi::c_void,
                cp as *mut std::ffi::c_void,
                tp as *mut std::ffi::c_void,
                c_tmp_floats,
                wp as *mut std::ffi::c_void,
                m as i32,
                tw.n as i32,
                tw.k as i32,
            )
        }
    };
    anyhow::ensure!(
        rc == 0,
        "marlin fp8 w8a16 rows rc={rc} m={m} n={} k={}",
        tw.n,
        tw.k
    );
    Ok(())
}

#[cfg(feature = "cuda")]
pub fn marlin_fp8_rows_env_on() -> bool {
    marlin_fp8_rows_opt_in()
}

#[cfg(feature = "cuda")]
struct Nvfp4Storage {
    weight_u8: cudarc::driver::CudaSlice<u8>,
    weight_scales_cm: cudarc::driver::CudaSlice<u8>,
    cuda_device: candle_core::CudaDevice,
    runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    weight_alpha: f32,

    weight_alpha_dev: cudarc::driver::CudaSlice<f32>,
    input_stored_global: f32,
    bf16act_row_limit: usize,
    a_staging: Mutex<std::collections::HashMap<(usize, usize), StagedActivations>>,
    marlin_w4a16_twin_built_once_from_the_resident_nvfp4_buffers:
        std::sync::OnceLock<Option<Arc<MarlinNvfp4Prepared>>>,
    dx_marlin_twin_of_the_transposed_weight_requantized_once_to_nvfp4_so_the_input_gradient_never_touches_a_bf16_copy:
        std::sync::OnceLock<Option<Arc<MarlinNvfp4Prepared>>>,
}

#[cfg(feature = "cuda")]
pub struct MarlinNvfp4Prepared {
    pub b_q_marlin: cudarc::driver::CudaSlice<u32>,
    pub scales_e5m3: cudarc::driver::CudaSlice<u8>,
    pub global_scale_dev: cudarc::driver::CudaSlice<f32>,
    pub n: usize,
    pub k: usize,
    pub folded_pow2: i32,
}

pub const NV_TRAIN_DX_QUANT_TWIN_ENV_DEFAULT_ON_USER_SIGNOFF_2026_09_13_QLORA_QUANTIZED_ALWAYS_0_RESTORES_THE_TRANSIENT_BF16_DEQUANT_THE_INPUT_GRADIENT_OF_A_FROZEN_NVFP4_LINEAR_RUNS_A_MARLIN_W4A16_GEMM_ON_A_TRANSPOSED_NVFP4_TWIN:
    &str = "NV_TRAIN_DX_QUANT_TWIN";

pub fn train_dx_quant_twin_enabled() -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
        NV_TRAIN_DX_QUANT_TWIN_ENV_DEFAULT_ON_USER_SIGNOFF_2026_09_13_QLORA_QUANTIZED_ALWAYS_0_RESTORES_THE_TRANSIENT_BF16_DEQUANT_THE_INPUT_GRADIENT_OF_A_FROZEN_NVFP4_LINEAR_RUNS_A_MARLIN_W4A16_GEMM_ON_A_TRANSPOSED_NVFP4_TWIN,
    )
}

struct RunOnDrop<F: FnMut()>(F);
impl<F: FnMut()> Drop for RunOnDrop<F> {
    fn drop(&mut self) {
        (self.0)()
    }
}
fn scopeguard_like<F: FnMut()>(f: F) -> RunOnDrop<F> {
    RunOnDrop(f)
}

pub const DX_TWIN_FOLD_IS_DEVICE_RESIDENT_SO_THE_HOST_FIELD_CARRIES_MINUS_ONE: i32 = -1;

#[cfg(feature = "cuda")]
fn build_dx_marlin_nvfp4_twin_on_the_device(
    lin: &Linear,
    s: &Nvfp4Storage,
) -> Result<MarlinNvfp4Prepared> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let stream = crate::cuda_stream::current_stream(&s.cuda_device);
    let (n_out, k_in) = (lin.out_features, lin.in_features);
    let t_build = std::time::Instant::now();
    static BUILDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let nth = BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    let _report_the_build_because_a_twin_built_inside_a_measured_step_is_step_time =
        scopeguard_like(move || {
            if nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_TRAIN_DX_TWIN_BUILD_LOG") {
                eprintln!(
                    "[dx_nvfp4_twin] device build #{nth} [{n_out}x{k_in}] {:.3} ms enqueued",
                    t_build.elapsed().as_secs_f64() * 1e3
                );
            }
        });
    anyhow::ensure!(
        n_out % 16 == 0 && k_in % 16 == 0 && (k_in * (n_out / 16)) % 64 == 0,
        "dx twin: [{n_out}x{k_in}] does not tile the NVFP4 twin layout"
    );
    let words = n_out * k_in / 8;
    let kb = n_out / 16;
    let alpha = s.weight_alpha * s.input_stored_global;
    let mut kmajor: cudarc::driver::CudaSlice<u32> = unsafe { stream.alloc::<u32>(words) }
        .map_err(|e| anyhow::anyhow!("dx twin kmajor: {e:?}"))?;
    let mut scales_linear: cudarc::driver::CudaSlice<u8> = unsafe { stream.alloc::<u8>(k_in * kb) }
        .map_err(|e| anyhow::anyhow!("dx twin scales: {e:?}"))?;
    let rc = unsafe {
        let (wp, _g0) = s.weight_u8.device_ptr(&stream);
        let (sp, _g1) = s.weight_scales_cm.device_ptr(&stream);
        let (kp, _g2) = kmajor.device_ptr_mut(&stream);
        let (lp, _g3) = scales_linear.device_ptr_mut(&stream);
        nv_kernels::cuda::dx_twin_nvfp4_requant_transposed_kmajor(
            stream.cu_stream() as *mut std::ffi::c_void,
            wp as *const u8,
            sp as *const u8,
            alpha,
            n_out as i32,
            k_in as i32,
            kp as *mut u32,
            lp as *mut u8,
        )
    };
    anyhow::ensure!(rc == 0, "dx twin requant rc={rc}");
    let mut b_q_marlin: cudarc::driver::CudaSlice<u32> = stream
        .alloc_zeros::<u32>(words)
        .map_err(|e| anyhow::anyhow!("dx twin alloc: {e:?}"))?;
    let rc = unsafe {
        let (pi, _g1) = kmajor.device_ptr(&stream);
        let (po, _g2) = b_q_marlin.device_ptr_mut(&stream);
        nv_kernels::cuda::marlin_repack_w4a16(
            stream.cu_stream() as *mut std::ffi::c_void,
            pi as *const std::ffi::c_void,
            po as *mut std::ffi::c_void,
            n_out as i32,
            k_in as i32,
            4,
        )
    };
    anyhow::ensure!(rc == 0, "dx twin repack rc={rc}");
    let mut scales_e5m3: cudarc::driver::CudaSlice<u8> =
        unsafe { stream.alloc::<u8>(k_in * kb) }
            .map_err(|e| anyhow::anyhow!("dx twin e5m3: {e:?}"))?;
    let mut global_scale_dev: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(1) }.map_err(|e| anyhow::anyhow!("dx twin global: {e:?}"))?;
    let mut vmin: cudarc::driver::CudaSlice<u32> =
        unsafe { stream.alloc::<u32>(1) }.map_err(|e| anyhow::anyhow!("dx twin vmin: {e:?}"))?;
    let mut flags: cudarc::driver::CudaSlice<i32> =
        unsafe { stream.alloc::<i32>(1) }.map_err(|e| anyhow::anyhow!("dx twin flags: {e:?}"))?;
    let rc = unsafe {
        let (lp, _g0) = scales_linear.device_ptr(&stream);
        let (ep, _g1) = scales_e5m3.device_ptr_mut(&stream);
        let (gp, _g2) = global_scale_dev.device_ptr_mut(&stream);
        let (vp, _g3) = vmin.device_ptr_mut(&stream);
        let (fp, _g4) = flags.device_ptr_mut(&stream);
        nv_kernels::cuda::dx_twin_nvfp4_scales_to_marlin(
            stream.cu_stream() as *mut std::ffi::c_void,
            lp as *const u8,
            k_in as i32,
            kb as i32,
            1.0,
            ep as *mut u8,
            gp as *mut f32,
            vp as *mut u32,
            fp as *mut i32,
        )
    };
    anyhow::ensure!(rc == 0, "dx twin scales rc={rc}");
    Ok(MarlinNvfp4Prepared {
        b_q_marlin,
        scales_e5m3,
        global_scale_dev,
        n: k_in,
        k: n_out,
        folded_pow2: DX_TWIN_FOLD_IS_DEVICE_RESIDENT_SO_THE_HOST_FIELD_CARRIES_MINUS_ONE,
    })
}

#[cfg(feature = "cuda")]
fn build_dx_marlin_nvfp4_twin_on_the_host_the_bitwise_oracle_of_the_device_build(
    lin: &Linear,
    s: &Nvfp4Storage,
) -> Result<MarlinNvfp4Prepared> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let stream = s.cuda_device.cuda_stream();
    let (n_out, k_in) = (lin.out_features, lin.in_features);
    let dequant =
        lin.device_dequant_transient_so_the_dx_gemm_reads_bf16_without_a_host_round_trip(&stream)?;
    let host: Vec<half::bf16> = stream
        .clone_dtoh(&dequant)
        .map_err(|e| anyhow::anyhow!("dx twin dtoh: {e:?}"))?;
    drop(dequant);
    let mut rows_t: Vec<Vec<f32>> = vec![vec![0f32; n_out]; k_in];
    for r in 0..n_out {
        for c in 0..k_in {
            rows_t[c][r] = host[r * k_in + c].to_f32();
        }
    }
    drop(host);
    let packed = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&rows_t, 1.0);
    drop(rows_t);
    let kmajor = nv_quant::marlin_nvfp4::kmajor_int32_from_packed_rows(&packed.data, k_in, n_out);
    let sc = nv_quant::marlin_nvfp4::marlin_scales_from_linear(&packed.scales, k_in, n_out, 1.0);
    #[allow(deprecated)]
    let d_kmajor: cudarc::driver::CudaSlice<u32> = stream
        .clone_htod(&kmajor)
        .map_err(|e| anyhow::anyhow!("dx twin htod kmajor: {e:?}"))?;
    let mut b_q_marlin: cudarc::driver::CudaSlice<u32> = stream
        .alloc_zeros::<u32>(n_out * k_in / 8)
        .map_err(|e| anyhow::anyhow!("dx twin alloc: {e:?}"))?;
    let rc = unsafe {
        let (pi, _g1) = d_kmajor.device_ptr(&stream);
        let (po, _g2) = b_q_marlin.device_ptr_mut(&stream);
        nv_kernels::cuda::marlin_repack_w4a16(
            stream.cu_stream() as *mut std::ffi::c_void,
            pi as *const std::ffi::c_void,
            po as *mut std::ffi::c_void,
            n_out as i32,
            k_in as i32,
            4,
        )
    };
    anyhow::ensure!(rc == 0, "dx twin repack rc={rc}");
    #[allow(deprecated)]
    let scales_e5m3: cudarc::driver::CudaSlice<u8> = stream
        .clone_htod(&sc.bytes_e5m3)
        .map_err(|e| anyhow::anyhow!("dx twin htod scales: {e:?}"))?;
    #[allow(deprecated)]
    let global_scale_dev: cudarc::driver::CudaSlice<f32> = stream
        .clone_htod(&[sc.global_scale_processed])
        .map_err(|e| anyhow::anyhow!("dx twin htod global: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("dx twin sync: {e:?}"))?;
    Ok(MarlinNvfp4Prepared {
        b_q_marlin,
        scales_e5m3,
        global_scale_dev,
        n: k_in,
        k: n_out,
        folded_pow2: sc.folded_pow2,
    })
}

pub const DX_TWIN_FROM_64_ROWS_BECAUSE_A_PER_EXPERT_CALL_OF_A_FEW_ROWS_PAYS_MORE_FOR_THE_MARLIN_LAUNCH_THAN_FOR_THE_DEQUANT: usize = 64;

#[cfg(feature = "cuda")]
struct DxGemmScratch {
    workspace: cudarc::driver::CudaSlice<i32>,
    c_tmp: cudarc::driver::CudaSlice<f32>,
    c_tmp_floats: i64,
}

#[cfg(feature = "cuda")]
unsafe impl Send for DxGemmScratch {}

#[cfg(feature = "cuda")]
static DX_GEMM_SCRATCH_PER_STREAM_BECAUSE_A_WORKSPACE_MEMSET_AND_TWO_ALLOCS_PER_BACKWARD_ARE_THE_LAUNCH_FLOOR: std::sync::OnceLock<
    Mutex<std::collections::HashMap<usize, DxGemmScratch>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn build_dx_fp8_twin_of_the_transposed_weight(lin: &Linear, s: &Fp8Storage) -> Result<Fp8DxTwin> {
    let dev = Device::Cuda(s.cuda_device.clone());
    let (n_out, k_in) = (lin.out_features, lin.in_features);
    let w = Tensor::from_storage(
        candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
            s.weight_u8
                .try_clone()
                .map_err(|e| anyhow::anyhow!("fp8 dx twin clone bytes: {e:?}"))?,
            s.cuda_device.clone(),
        )),
        (n_out, k_in),
        candle_core::op::BackpropOp::none(),
        false,
    );
    let wt = w.t()?.contiguous()?;
    let wq_t = {
        let (st, _l) = wt.storage_and_layout();
        match &*st {
            candle_core::Storage::Cuda(c) => c
                .as_cuda_slice::<u8>()?
                .try_clone()
                .map_err(|e| anyhow::anyhow!("fp8 dx twin clone transposed: {e:?}"))?,
            _ => anyhow::bail!("fp8 dx twin: not cuda"),
        }
    };
    let stream = s.cuda_device.cuda_stream();
    #[allow(deprecated)]
    let unit_scales_for_the_k_in_rows: cudarc::driver::CudaSlice<f32> = stream
        .clone_htod(&vec![1f32; k_in])
        .map_err(|e| anyhow::anyhow!("fp8 dx twin htod unit scales: {e:?}"))?;
    let channel_scale_bf16 =
        Tensor::from_vec(s.weight_scale_rows.clone(), n_out, &dev)?.to_dtype(DType::BF16)?;
    Ok(Fp8DxTwin {
        wq_t,
        unit_scales_for_the_k_in_rows,
        channel_scale_bf16,
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn fp8_dx_gemm_dy_rowquantized_e4m3_times_the_transposed_twin(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    s: &Fp8Storage,
    tw: &Fp8DxTwin,
    dy_bf16: &Tensor,
    m: usize,
    n_out: usize,
    k_in: usize,
    out: &mut cudarc::driver::CudaSlice<half::bf16>,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let dy_scaled = dy_bf16
        .broadcast_mul(&tw.channel_scale_bf16.reshape((1, n_out))?)?
        .contiguous()?;
    let mut dy_q: cudarc::driver::CudaSlice<u8> =
        unsafe { stream.alloc::<u8>(m * n_out) }.map_err(|e| anyhow::anyhow!(e))?;
    let mut a_scales: cudarc::driver::CudaSlice<f32> =
        unsafe { stream.alloc::<f32>(m) }.map_err(|e| anyhow::anyhow!(e))?;
    let rc = {
        let (st, l) = dy_scaled.storage_and_layout();
        let c = match &*st {
            candle_core::Storage::Cuda(c) => c,
            _ => anyhow::bail!("fp8 dx: dy not cuda"),
        };
        let dsl = c.as_cuda_slice::<half::bf16>()?;
        let dview = dsl.slice(l.start_offset()..);
        let (dp, _g) = dview.device_ptr(stream);
        let (qp, _gq) = dy_q.device_ptr_mut(stream);
        let (sp, _gs) = a_scales.device_ptr_mut(stream);
        unsafe {
            nv_kernels::cuda::rowquant_e4m3(
                stream.cu_stream() as *mut _,
                dp as *const u16,
                qp as *mut u8,
                sp as *mut f32,
                m as i32,
                n_out as i32,
            )
        }
    };
    anyhow::ensure!(rc == 0, "fp8 dx rowquant rc={rc}");
    {
        let mut runner = s
            .runner
            .lock()
            .map_err(|e| anyhow::anyhow!("fp8 runner poisoned: {e}"))?;
        runner.with_stream(stream, |runner| {
            runner.matmul_e4m3_weight_row(
                &dy_q,
                &tw.wq_t,
                out,
                m as u64,
                k_in as u64,
                n_out as u64,
                &s.a_scale_dev,
                &s.a_scale_dev,
            )
        })?;
    }
    let (dp, _gd) = out.device_ptr_mut(stream);
    let (rp, _gr) = a_scales.device_ptr(stream);
    let (cp, _gc) = tw.unit_scales_for_the_k_in_rows.device_ptr(stream);
    let rc = unsafe {
        nv_kernels::cuda::scale_rowcol_bf16(
            stream.cu_stream() as *mut _,
            dp as *mut u16,
            rp as *const f32,
            cp as *const f32,
            m as i32,
            k_in as i32,
        )
    };
    anyhow::ensure!(rc == 0, "fp8 dx scale_rowcol rc={rc}");
    Ok(())
}

#[cfg(feature = "cuda")]
fn marlin_nvfp4_dx_gemm(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    tw: &MarlinNvfp4Prepared,
    dy_bf16_ptr: u64,
    m: usize,
    out: &mut cudarc::driver::CudaSlice<half::bf16>,
) -> Result<()> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    anyhow::ensure!(
        out.len() >= m * tw.n,
        "dx gemm: out {} < m*n {}",
        out.len(),
        m * tw.n
    );
    let scratch = DX_GEMM_SCRATCH_PER_STREAM_BECAUSE_A_WORKSPACE_MEMSET_AND_TWO_ALLOCS_PER_BACKWARD_ARE_THE_LAUNCH_FLOOR
        .get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut scratch = scratch
        .lock()
        .map_err(|e| anyhow::anyhow!("dx gemm scratch poisoned: {e}"))?;
    let key = nv_quant::stream_cache_key(stream);
    if !scratch.contains_key(&key) {
        let mut elems = 0i32;
        let rc = unsafe { nv_kernels::cuda::marlin_workspace_elems(&mut elems as *mut i32) };
        anyhow::ensure!(rc == 0 && elems > 0, "marlin workspace elems rc={rc}");
        let workspace: cudarc::driver::CudaSlice<i32> = stream
            .alloc_zeros::<i32>(elems as usize)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut c_tmp_floats = 0i64;
        let rc =
            unsafe { nv_kernels::cuda::marlin_nvfp4_c_tmp_floats(&mut c_tmp_floats as *mut i64) };
        anyhow::ensure!(rc == 0 && c_tmp_floats > 0, "marlin c_tmp floats rc={rc}");
        let c_tmp: cudarc::driver::CudaSlice<f32> = unsafe {
            stream
                .alloc::<f32>(c_tmp_floats as usize)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        scratch.insert(
            key,
            DxGemmScratch {
                workspace,
                c_tmp,
                c_tmp_floats,
            },
        );
    }
    let sc = scratch.get_mut(&key).expect("inserted above");
    let c_tmp_floats = sc.c_tmp_floats;
    let (workspace, c_tmp) = (&sc.workspace, &mut sc.c_tmp);
    let rc = {
        let (bp, _gb) = tw.b_q_marlin.device_ptr(stream);
        let (sp, _gs) = tw.scales_e5m3.device_ptr(stream);
        let (gp, _gg) = tw.global_scale_dev.device_ptr(stream);
        let (cp, _gc) = out.device_ptr_mut(stream);
        let (tp, _gt) = c_tmp.device_ptr_mut(stream);
        let (wp, _gw) = workspace.device_ptr(stream);
        unsafe {
            nv_kernels::cuda::marlin_gemm_nvfp4_w4a16(
                stream.cu_stream() as *mut std::ffi::c_void,
                dy_bf16_ptr as *const std::ffi::c_void,
                bp as *const std::ffi::c_void,
                sp as *const std::ffi::c_void,
                gp as *const f32,
                cp as *mut std::ffi::c_void,
                tp as *mut std::ffi::c_void,
                c_tmp_floats,
                wp as *mut std::ffi::c_void,
                m as i32,
                tw.n as i32,
                tw.k as i32,
            )
        }
    };
    anyhow::ensure!(
        rc == 0,
        "marlin nvfp4 dx gemm rc={rc} m={m} n={} k={}",
        tw.n,
        tw.k
    );
    Ok(())
}

#[cfg(feature = "cuda")]
fn build_marlin_nvfp4_twin(s: &Nvfp4Storage, n: usize, k: usize) -> Result<MarlinNvfp4Prepared> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let stream = s.cuda_device.cuda_stream();
    let packed = stream
        .clone_dtoh(&s.weight_u8)
        .map_err(|e| anyhow::anyhow!("marlin twin dtoh weights: {e:?}"))?;
    let swizzled = stream
        .clone_dtoh(&s.weight_scales_cm)
        .map_err(|e| anyhow::anyhow!("marlin twin dtoh scales: {e:?}"))?;
    anyhow::ensure!(
        packed.len() == n * k / 2,
        "marlin twin: packed bytes {} != n*k/2 for n={n} k={k}",
        packed.len()
    );
    let kmajor = nv_quant::marlin_nvfp4::kmajor_int32_from_packed_rows(&packed, n, k);
    let global = s.weight_alpha * s.input_stored_global;
    let sc = nv_quant::marlin_nvfp4::marlin_scales_from_swizzled(&swizzled, n, k, global);
    #[allow(deprecated)]
    let d_kmajor: cudarc::driver::CudaSlice<u32> = stream
        .clone_htod(&kmajor)
        .map_err(|e| anyhow::anyhow!("marlin twin htod kmajor: {e:?}"))?;
    let mut b_q_marlin: cudarc::driver::CudaSlice<u32> = stream
        .alloc_zeros::<u32>(k * n / 8)
        .map_err(|e| anyhow::anyhow!("marlin twin alloc: {e:?}"))?;
    let rc = unsafe {
        let (pi, _g1) = d_kmajor.device_ptr(&stream);
        let (po, _g2) = b_q_marlin.device_ptr_mut(&stream);
        nv_kernels::cuda::marlin_repack_w4a16(
            stream.cu_stream() as *mut std::ffi::c_void,
            pi as *const std::ffi::c_void,
            po as *mut std::ffi::c_void,
            k as i32,
            n as i32,
            4,
        )
    };
    anyhow::ensure!(rc == 0, "marlin twin repack rc={rc}");
    #[allow(deprecated)]
    let scales_e5m3: cudarc::driver::CudaSlice<u8> = stream
        .clone_htod(&sc.bytes_e5m3)
        .map_err(|e| anyhow::anyhow!("marlin twin htod scales: {e:?}"))?;
    #[allow(deprecated)]
    let global_scale_dev: cudarc::driver::CudaSlice<f32> = stream
        .clone_htod(&[sc.global_scale_processed])
        .map_err(|e| anyhow::anyhow!("marlin twin htod global: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("marlin twin sync: {e:?}"))?;
    Ok(MarlinNvfp4Prepared {
        b_q_marlin,
        scales_e5m3,
        global_scale_dev,
        n,
        k,
        folded_pow2: sc.folded_pow2,
    })
}

#[cfg(feature = "cuda")]
struct StagedActivations {
    epoch: u64,
    hwm_rows: usize,
    packed: cudarc::driver::CudaSlice<u8>,
    scales: cudarc::driver::CudaSlice<u8>,
}

pub(crate) fn this_call_is_building_an_autograd_graph(x: &Tensor) -> bool {
    if !x.track_op() {
        return false;
    }
    !leaf_gradient_loss_was_explicitly_allowed()
}

pub(crate) fn leaf_gradient_loss_was_explicitly_allowed() -> bool {
    std::env::var("NV_ALLOW_LEAF_GRADIENT_LOSS").ok().as_deref() == Some("1")
}

#[cfg(feature = "cuda")]
fn a_quantized_fast_path_cannot_be_differentiated(path: &str, x: &Tensor) -> Result<()> {
    if this_call_is_building_an_autograd_graph(x) {
        anyhow::bail!(
            "{path}: this quantized CUDA fast path has no backward and no dense weight to \
             fall back on, so its output would be a graph leaf and a training step through \
             it would contribute no gradient at all, silently -- including to every adapter \
             in an earlier layer. Train against bf16 weights, or give this path a backward."
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) struct Bf16LinearOp {
    frozen_weight_kept_inside_the_op_so_candle_sees_one_input_and_never_builds_a_weight_side_graph:
        Tensor,
}

#[cfg(feature = "cuda")]
impl Bf16LinearOp {
    pub(crate) fn holding_the_frozen_weight(weight: &Tensor) -> Result<Self> {
        anyhow::ensure!(
            weight.dims().len() == 2
                && weight.dtype() == DType::BF16
                && matches!(weight.device(), Device::Cuda(_)),
            "Bf16LinearOp needs a 2-D bf16 CUDA weight, got {:?} {:?}",
            weight.dims(),
            weight.dtype()
        );
        anyhow::ensure!(
            !this_call_is_building_an_autograd_graph(weight),
            "Bf16LinearOp detaches its weight, so a Var-backed or graph-carrying weight \
             would receive zero gradient silently while dx and the loss keep moving -- the \
             weight would train as a no-op. Use matmul_fallback for a trainable weight, or \
             set NV_ALLOW_LEAF_GRADIENT_LOSS=1 to accept the loss."
        );
        Ok(Self {
            frozen_weight_kept_inside_the_op_so_candle_sees_one_input_and_never_builds_a_weight_side_graph:
                weight.contiguous()?,
        })
    }

    fn weight(&self) -> &Tensor {
        &self.frozen_weight_kept_inside_the_op_so_candle_sees_one_input_and_never_builds_a_weight_side_graph
    }

    fn out_features(&self) -> usize {
        self.weight().dims()[0]
    }

    fn in_features(&self) -> usize {
        self.weight().dims()[1]
    }

    fn weight_cuda_view(
        &self,
    ) -> candle_core::Result<(std::sync::RwLockReadGuard<'_, candle_core::Storage>, usize)> {
        let (w_storage, wl) = self.weight().storage_and_layout();
        Ok((w_storage, wl.start_offset()))
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for Bf16LinearOp {
    fn name(&self) -> &'static str {
        "nv-bf16-linear-forward-with-dx-only-backward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Err(candle_core::Error::Msg(
            "Bf16LinearOp is CUDA-only; the CPU training path stays on matmul_fallback".into(),
        ))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        let (m, k) = match layout.shape().dims() {
            [m, k] => (*m, *k),
            d => {
                return Err(candle_core::Error::Msg(format!(
                    "Bf16LinearOp forward needs a rank-2 input, got {d:?}"
                )))
            }
        };
        if k != self.in_features() {
            return Err(candle_core::Error::Msg(format!(
                "Bf16LinearOp forward: input last dim {k} != in_features {}",
                self.in_features()
            )));
        }
        let Some((x_start, _x_end)) = layout.contiguous_offsets() else {
            return Err(candle_core::Error::Msg(
                "Bf16LinearOp forward needs a contiguous input".into(),
            ));
        };
        let dev = storage.device.clone();
        let stream = crate::cuda_stream::current_stream(&dev);
        let n = self.out_features();
        let mut y_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * n)
                .map_err(|e| candle_core::Error::Msg(format!("Bf16LinearOp alloc y: {e}")))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())
            .map_err(|e| candle_core::Error::Msg(format!("Bf16LinearOp gemm handle: {e:#}")))?;
        {
            let x_slice = storage.as_cuda_slice::<bf16>()?;
            let (w_storage, w_off) = self.weight_cuda_view()?;
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "Bf16LinearOp weight must be CUDA-resident".into(),
                    ))
                }
            };
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            gemm.bf16_matmul_row_major_bt_offs(
                &stream, x_slice, x_start, w_slice, w_off, &mut y_dev, m as u64, n as u64,
                k as u64, 1.0, 0.0,
            )
            .map_err(|e| candle_core::Error::Msg(format!("Bf16LinearOp forward gemm: {e:#}")))?;
        }
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(y_dev, dev),
            (m, n).into(),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        let dy = grad_res.to_dtype(DType::BF16)?.contiguous()?;
        let (m, n) = dy.dims2()?;
        if n != self.out_features() {
            return Err(candle_core::Error::Msg(format!(
                "Bf16LinearOp backward: dy last dim {n} != out_features {}",
                self.out_features()
            )));
        }
        let dev = match dy.device() {
            Device::Cuda(d) => d.clone(),
            _ => {
                return Err(candle_core::Error::Msg(
                    "Bf16LinearOp backward needs a CUDA gradient".into(),
                ))
            }
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let k = self.in_features();
        let mut dx_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * k)
                .map_err(|e| candle_core::Error::Msg(format!("Bf16LinearOp alloc dx: {e}")))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())
            .map_err(|e| candle_core::Error::Msg(format!("Bf16LinearOp gemm handle: {e:#}")))?;
        {
            let (dy_storage, dyl) = dy.storage_and_layout();
            let dy_cuda = match &*dy_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "Bf16LinearOp backward: expected cuda storage for dy".into(),
                    ))
                }
            };
            let dy_slice = dy_cuda.as_cuda_slice::<bf16>()?;
            let (w_storage, w_off) = self.weight_cuda_view()?;
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "Bf16LinearOp weight must be CUDA-resident".into(),
                    ))
                }
            };
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let dx_cols = k as u64;
            let contraction_over_out_features = n as u64;
            gemm.bf16_matmul_row_major_offs(
                &stream,
                dy_slice,
                dyl.start_offset(),
                w_slice,
                w_off,
                &mut dx_dev,
                m as u64,
                dx_cols,
                contraction_over_out_features,
                1.0,
                0.0,
            )
            .map_err(|e| candle_core::Error::Msg(format!("Bf16LinearOp backward gemm: {e:#}")))?;
        }
        let dx = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(dx_dev, dev)),
            (m, k),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(dx.to_dtype(arg.dtype())?))
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct QuantLinearOp {
    serving_twin_sharing_the_resident_storage_arc_so_a_recompute_reruns_the_quantized_serving_gemm:
        Linear,
}

#[cfg(feature = "cuda")]
impl QuantLinearOp {
    pub(crate) fn sharing_the_resident_quantized_storage(lin: &Linear) -> Result<Self> {
        let storage = match &lin.storage {
            LinearStorage::Fp8E4m3(s) => LinearStorage::Fp8E4m3(Arc::clone(s)),
            LinearStorage::Nvfp4(s) => LinearStorage::Nvfp4(Arc::clone(s)),
            LinearStorage::GgmlQuant(_) => anyhow::bail!(
                "QuantLinearOp has no ggml-quant sibling: ggml storage serves the ocr \
                 decode path and carries no trainable identity"
            ),
            LinearStorage::Bf16 { .. } => anyhow::bail!(
                "QuantLinearOp is the quantized-storage sibling of Bf16LinearOp and its \
                 weights are frozen raw device bytes with no autograd identity to detach; \
                 bf16 storage trains through Bf16LinearOp, which is where a trainable \
                 weight is refused"
            ),
        };
        Ok(Self {
            serving_twin_sharing_the_resident_storage_arc_so_a_recompute_reruns_the_quantized_serving_gemm:
                Linear {
                    kind: lin.kind,
                    storage,
                    bias: None,
                    in_features: lin.in_features,
                    out_features: lin.out_features,
                    lora: std::sync::RwLock::new(None),
                },
        })
    }

    fn twin(&self) -> &Linear {
        &self.serving_twin_sharing_the_resident_storage_arc_so_a_recompute_reruns_the_quantized_serving_gemm
    }
}

#[cfg(feature = "cuda")]
impl candle_core::CustomOp1 for QuantLinearOp {
    fn name(&self) -> &'static str {
        "nv-quant-resident-linear-forward-with-straight-through-dx-only-backward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        Err(candle_core::Error::Msg(
            "QuantLinearOp is CUDA-only; quantized-resident storages do not exist on CPU".into(),
        ))
    }

    fn cuda_fwd(
        &self,
        storage: &candle_core::CudaStorage,
        layout: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CudaStorage, candle_core::Shape)> {
        use half::bf16;

        let twin = self.twin();
        let (m, k) = match layout.shape().dims() {
            [m, k] => (*m, *k),
            d => {
                return Err(candle_core::Error::Msg(format!(
                    "QuantLinearOp forward needs a rank-2 input, got {d:?}"
                )))
            }
        };
        if k != twin.in_features {
            return Err(candle_core::Error::Msg(format!(
                "QuantLinearOp forward: input last dim {k} != in_features {}",
                twin.in_features
            )));
        }
        let Some((x_start, x_end)) = layout.contiguous_offsets() else {
            return Err(candle_core::Error::Msg(
                "QuantLinearOp forward needs a contiguous input".into(),
            ));
        };
        let dev = storage.device.clone();
        let stream = crate::cuda_stream::current_stream(&dev);
        let _span = crate::prof_events::span(
            match (&twin.storage, m >= DX_TWIN_FROM_64_ROWS_BECAUSE_A_PER_EXPERT_CALL_OF_A_FEW_ROWS_PAYS_MORE_FOR_THE_MARLIN_LAUNCH_THAN_FOR_THE_DEQUANT) {
                (LinearStorage::Nvfp4(_), true) => "fwd_quant_linear_nvfp4_m64plus",
                (LinearStorage::Nvfp4(_), false) => "fwd_quant_linear_nvfp4_small_m",
                (LinearStorage::Fp8E4m3(_), true) => "fwd_quant_linear_fp8_m64plus",
                (LinearStorage::Fp8E4m3(_), false) => "fwd_quant_linear_fp8_small_m",
                _ => "fwd_quant_linear_other",
            },
            &stream,
        );
        let mut x_owned: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * k)
                .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp alloc x: {e}")))?
        };
        {
            let x_slice = storage.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(x_start..x_end);
            stream
                .memcpy_dtod(&x_view, &mut x_owned)
                .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp dtod x: {e}")))?;
        }
        let x_t = Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                x_owned,
                dev.clone(),
            )),
            (m, k),
            candle_core::op::BackpropOp::none(),
            false,
        );
        let y_t = twin.matmul_impl(&x_t, m, false).map_err(|e| {
            candle_core::Error::Msg(format!(
                "QuantLinearOp forward via the serving quantized gemm: {e:#}"
            ))
        })?;
        if y_t.dims() != [m, twin.out_features] || y_t.dtype() != DType::BF16 {
            return Err(candle_core::Error::Msg(format!(
                "QuantLinearOp forward: the serving gemm returned {:?} {:?}, expected \
                 [{m}, {}] BF16",
                y_t.dims(),
                y_t.dtype(),
                twin.out_features
            )));
        }
        let y_c = y_t.contiguous()?;
        let mut y_owned: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * twin.out_features)
                .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp alloc y: {e}")))?
        };
        {
            let (y_storage, yl) = y_c.storage_and_layout();
            let y_cuda = match &*y_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "QuantLinearOp forward: expected cuda storage for the gemm output".into(),
                    ))
                }
            };
            let y_slice = y_cuda.as_cuda_slice::<bf16>()?;
            let (y_start, y_end) = yl.contiguous_offsets().ok_or_else(|| {
                candle_core::Error::Msg("QuantLinearOp forward: non-contiguous gemm output".into())
            })?;
            let y_view = y_slice.slice(y_start..y_end);
            stream
                .memcpy_dtod(&y_view, &mut y_owned)
                .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp dtod y: {e}")))?;
        }
        Ok((
            candle_core::CudaStorage::wrap_cuda_slice(y_owned, dev),
            (m, twin.out_features).into(),
        ))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        let twin = self.twin();
        let dy = grad_res.to_dtype(DType::BF16)?.contiguous()?;
        let (m, n) = dy.dims2()?;
        if n != twin.out_features {
            return Err(candle_core::Error::Msg(format!(
                "QuantLinearOp backward: dy last dim {n} != out_features {}",
                twin.out_features
            )));
        }
        let dev = match dy.device() {
            Device::Cuda(d) => d.clone(),
            _ => {
                return Err(candle_core::Error::Msg(
                    "QuantLinearOp backward needs a CUDA gradient".into(),
                ))
            }
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let k = twin.in_features;
        let _span = crate::prof_events::span(
            match (&twin.storage, m >= DX_TWIN_FROM_64_ROWS_BECAUSE_A_PER_EXPERT_CALL_OF_A_FEW_ROWS_PAYS_MORE_FOR_THE_MARLIN_LAUNCH_THAN_FOR_THE_DEQUANT) {
                (LinearStorage::Nvfp4(_), true) => "bwd_quant_linear_nvfp4_twin_m64plus",
                (LinearStorage::Nvfp4(_), false) => "bwd_quant_linear_nvfp4_transient_small_m",
                (LinearStorage::Fp8E4m3(_), true) => "bwd_quant_linear_fp8_twin_m64plus",
                (LinearStorage::Fp8E4m3(_), false) => "bwd_quant_linear_fp8_transient_small_m",
                _ => "bwd_quant_linear_other",
            },
            &stream,
        );
        if train_dx_quant_twin_enabled()
            && m >= DX_TWIN_FROM_64_ROWS_BECAUSE_A_PER_EXPERT_CALL_OF_A_FEW_ROWS_PAYS_MORE_FOR_THE_MARLIN_LAUNCH_THAN_FOR_THE_DEQUANT
        {
            if let (LinearStorage::Fp8E4m3(s), Some(tw)) =
                (&twin.storage, twin.dx_fp8_twin_of_the_transposed_weight())
            {
                let mut dx_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
                    stream.alloc::<bf16>(m * k).map_err(|e| {
                        candle_core::Error::Msg(format!("QuantLinearOp alloc dx: {e}"))
                    })?
                };
                fp8_dx_gemm_dy_rowquantized_e4m3_times_the_transposed_twin(
                    &stream, s, &tw, &dy, m, n, k, &mut dx_dev,
                )
                .map_err(|e| {
                    candle_core::Error::Msg(format!("QuantLinearOp backward fp8 twin gemm: {e:#}"))
                })?;
                let dx = candle_core::Tensor::from_storage(
                    candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                        dx_dev, dev,
                    )),
                    (m, k),
                    candle_core::op::BackpropOp::none(),
                    false,
                );
                return Ok(Some(dx.to_dtype(arg.dtype())?));
            }
            if let Some(tw) = twin.dx_marlin_nvfp4_twin_of_the_transposed_weight() {
                let mut dx_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
                    stream.alloc::<bf16>(m * k).map_err(|e| {
                        candle_core::Error::Msg(format!("QuantLinearOp alloc dx: {e}"))
                    })?
                };
                let dy_ptr = {
                    let (dy_storage, dyl) = dy.storage_and_layout();
                    let dy_cuda = match &*dy_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => {
                            return Err(candle_core::Error::Msg(
                                "QuantLinearOp backward: expected cuda storage for dy".into(),
                            ))
                        }
                    };
                    let dy_slice = dy_cuda.as_cuda_slice::<bf16>()?;
                    let (p, _g) = dy_slice.device_ptr(&stream);
                    p as u64 + (dyl.start_offset() * 2) as u64
                };
                marlin_nvfp4_dx_gemm(&stream, &tw, dy_ptr, m, &mut dx_dev).map_err(|e| {
                    candle_core::Error::Msg(format!(
                        "QuantLinearOp backward nvfp4 twin gemm: {e:#}"
                    ))
                })?;
                let dx = candle_core::Tensor::from_storage(
                    candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(
                        dx_dev, dev,
                    )),
                    (m, k),
                    candle_core::op::BackpropOp::none(),
                    false,
                );
                return Ok(Some(dx.to_dtype(arg.dtype())?));
            }
        }
        let w_dequant = twin
            .device_dequant_transient_so_the_dx_gemm_reads_bf16_without_a_host_round_trip(&stream)
            .map_err(|e| {
                candle_core::Error::Msg(format!("QuantLinearOp backward dequant: {e:#}"))
            })?;
        let mut dx_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * k)
                .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp alloc dx: {e}")))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())
            .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp gemm handle: {e:#}")))?;
        {
            let (dy_storage, dyl) = dy.storage_and_layout();
            let dy_cuda = match &*dy_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => {
                    return Err(candle_core::Error::Msg(
                        "QuantLinearOp backward: expected cuda storage for dy".into(),
                    ))
                }
            };
            let dy_slice = dy_cuda.as_cuda_slice::<bf16>()?;
            let dx_cols = k as u64;
            let contraction_over_out_features = n as u64;
            gemm.bf16_matmul_row_major_offs(
                &stream,
                dy_slice,
                dyl.start_offset(),
                &w_dequant,
                0,
                &mut dx_dev,
                m as u64,
                dx_cols,
                contraction_over_out_features,
                1.0,
                0.0,
            )
            .map_err(|e| candle_core::Error::Msg(format!("QuantLinearOp backward gemm: {e:#}")))?;
        }
        drop(w_dequant);
        let dx = candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(candle_core::CudaStorage::wrap_cuda_slice(dx_dev, dev)),
            (m, k),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(dx.to_dtype(arg.dtype())?))
    }
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weight_t = Some(weight.t()?.contiguous()?);
        Ok(Self {
            kind: LinearKind::Bf16,
            storage: LinearStorage::Bf16 { weight, weight_t },
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn new_no_pretranspose(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        Ok(Self {
            kind: LinearKind::Bf16,
            storage: LinearStorage::Bf16 {
                weight,
                weight_t: None,
            },
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn from_candle_vb(
        vb: VarBuilder,
        in_features: usize,
        out_features: usize,
        bias: bool,
    ) -> Result<Self> {
        let vb_bf = vb.clone().to_dtype(DType::BF16);
        let weight = vb_bf.get((out_features, in_features), "weight")?;
        let bias = if bias {
            Some(vb_bf.get(out_features, "bias")?)
        } else {
            None
        };
        Self::new(weight, bias)
    }

    pub fn from_bf16_quantized_ggml(
        weight: &Tensor,
        bias: Option<Tensor>,
        gdt: candle_core::quantized::GgmlDType,
    ) -> Result<Self> {
        let (out_features, in_features) = weight.dims2()?;
        anyhow::ensure!(
            in_features.is_multiple_of(gdt.block_size()),
            "ggml quant {gdt:?}: in_features {in_features} must be a multiple of the {} block",
            gdt.block_size()
        );
        let kind = match gdt {
            candle_core::quantized::GgmlDType::Q4K => LinearKind::GgmlQ4k,
            candle_core::quantized::GgmlDType::Q8_0 => LinearKind::GgmlQ8_0,
            other => anyhow::bail!(
                "from_bf16_quantized_ggml carries the two OCR serving grades only (Q4K, Q8_0);                  got {other:?}"
            ),
        };
        let qt = candle_core::quantized::QTensor::quantize(weight, gdt)?;
        let qm = candle_core::quantized::QMatMul::from_qtensor(qt)?;
        Ok(Self {
            kind,
            storage: LinearStorage::GgmlQuant(Arc::new(GgmlQuantStorage { qm })),
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn kind(&self) -> &LinearKind {
        &self.kind
    }

    #[cfg(feature = "cuda")]
    pub fn nvfp4_parts(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<u8>,
        f32,
        f32,
    )> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => Some((
                &s.weight_u8,
                &s.weight_scales_cm,
                s.weight_alpha,
                s.input_stored_global,
            )),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::type_complexity)]
    pub fn nvfp4_parts_full(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
        f32,
        f32,
    )> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => Some((
                &s.weight_u8,
                &s.weight_scales_cm,
                &s.weight_alpha_dev,
                s.weight_alpha,
                s.input_stored_global,
            )),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dx_fp8_twin_of_the_transposed_weight(&self) -> Option<Arc<Fp8DxTwin>> {
        match &self.storage {
            LinearStorage::Fp8E4m3(s) => s
                .dx_fp8_twin_of_the_transposed_weight_bytes_with_the_channel_scale_folded_into_dy
                .get_or_init(|| match build_dx_fp8_twin_of_the_transposed_weight(self, s) {
                    Ok(t) => Some(Arc::new(t)),
                    Err(e) => {
                        eprintln!(
                            "[dx_fp8_twin] build failed for [{}x{}]: {e:#}; the input gradient falls back to the transient dequant",
                            self.out_features, self.in_features
                        );
                        None
                    }
                })
                .clone(),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dx_marlin_nvfp4_twin_of_the_transposed_weight(
        &self,
    ) -> Option<Arc<MarlinNvfp4Prepared>> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => s
                .dx_marlin_twin_of_the_transposed_weight_requantized_once_to_nvfp4_so_the_input_gradient_never_touches_a_bf16_copy
                .get_or_init(|| match build_dx_marlin_nvfp4_twin_on_the_device(self, s) {
                    Ok(t) => Some(Arc::new(t)),
                    Err(e) => {
                        eprintln!(
                            "[dx_nvfp4_twin] build failed for [{}x{}]: {e:#}; the input gradient falls back to the transient dequant",
                            self.out_features, self.in_features
                        );
                        None
                    }
                })
                .clone(),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dx_marlin_nvfp4_twin_built_on_the_host_oracle(&self) -> Result<MarlinNvfp4Prepared> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => {
                build_dx_marlin_nvfp4_twin_on_the_host_the_bitwise_oracle_of_the_device_build(
                    self, s,
                )
            }
            _ => anyhow::bail!("the dx twin oracle needs an NVFP4 storage"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dx_marlin_nvfp4_twin_built_on_the_device(&self) -> Result<MarlinNvfp4Prepared> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => build_dx_marlin_nvfp4_twin_on_the_device(self, s),
            _ => anyhow::bail!("the device dx twin needs an NVFP4 storage"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn marlin_nvfp4_w4a16_twin(&self) -> Option<Arc<MarlinNvfp4Prepared>> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => s
                .marlin_w4a16_twin_built_once_from_the_resident_nvfp4_buffers
                .get_or_init(|| {
                    match build_marlin_nvfp4_twin(s, self.out_features, self.in_features) {
                        Ok(t) => {
                            eprintln!(
                                "[marlin_nvfp4_twin] built n={} k={} fold_pow2={}",
                                t.n, t.k, t.folded_pow2
                            );
                            Some(Arc::new(t))
                        }
                        Err(e) => {
                            eprintln!(
                                "[marlin_nvfp4_twin] unavailable n={} k={}: {e:#}",
                                self.out_features, self.in_features
                            );
                            None
                        }
                    }
                })
                .clone(),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_scale_parts(&self) -> Option<(&[f32], nv_quant::fp8::Fp8ScaleMode)> {
        match &self.storage {
            LinearStorage::Fp8E4m3(s) => Some((&s.weight_scale_rows, s.scale_mode)),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn fp8_e4m3_row_weight_and_scales_so_gdn_prenorm_folds_into_gemv_e4m3_mk_h(
        &self,
    ) -> Option<(
        &cudarc::driver::CudaSlice<u8>,
        &cudarc::driver::CudaSlice<f32>,
    )> {
        match &self.storage {
            LinearStorage::Fp8E4m3(s)
                if s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow
                    && self.bias.is_none()
                    && self.in_features.is_multiple_of(16) =>
            {
                Some((&s.weight_u8, &s.b_scale_rows_dev))
            }
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn nvfp4_runner(
        &self,
    ) -> Option<std::sync::Arc<std::sync::Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => Some(s.runner.clone()),
            _ => None,
        }
    }

    pub fn weight(&self) -> Option<&Tensor> {
        #[cfg(feature = "cuda")]
        {
            match &self.storage {
                LinearStorage::Bf16 { weight, .. } => Some(weight),
                LinearStorage::Fp8E4m3(_) => None,
                LinearStorage::Nvfp4(_) => None,
                LinearStorage::GgmlQuant(_) => None,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            match &self.storage {
                LinearStorage::Bf16 { weight, .. } => Some(weight),
                LinearStorage::GgmlQuant(_) => None,
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn dequant_weight(&self) -> Result<Option<Tensor>> {
        use half::bf16;
        match &self.storage {
            LinearStorage::Bf16 { weight, .. } => Ok(Some(weight.clone())),
            LinearStorage::Fp8E4m3(_) => Ok(None),
            LinearStorage::GgmlQuant(_) => Ok(None),
            LinearStorage::Nvfp4(s) => {
                let stream = crate::cuda_stream::current_stream(&s.cuda_device);
                #[allow(deprecated)]
                let packed = stream
                    .memcpy_dtov(&s.weight_u8)
                    .map_err(|e| anyhow::anyhow!("dtoh nvfp4 packed: {e}"))?;
                #[allow(deprecated)]
                let scales_sw = stream
                    .memcpy_dtov(&s.weight_scales_cm)
                    .map_err(|e| anyhow::anyhow!("dtoh nvfp4 scales: {e}"))?;
                let vals = nv_quant::nvfp4::dequantize_packed_swizzled(
                    &packed,
                    &scales_sw,
                    self.out_features,
                    self.in_features,
                    s.weight_alpha * s.input_stored_global,
                );
                let bf: Vec<bf16> = vals.iter().map(|&v| bf16::from_f32(v)).collect();
                let t = Tensor::from_vec(
                    bf,
                    (self.out_features, self.in_features),
                    &Device::Cuda(s.cuda_device.clone()),
                )?;
                Ok(Some(t))
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn dequant_weight(&self) -> Result<Option<Tensor>> {
        Ok(self.weight().cloned())
    }

    #[cfg(feature = "cuda")]
    fn device_dequant_transient_so_the_dx_gemm_reads_bf16_without_a_host_round_trip(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<cudarc::driver::CudaSlice<half::bf16>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        let mut out: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(self.out_features * self.in_features)
                .map_err(|e| anyhow::anyhow!("alloc transient dequant weight: {e}"))?
        };
        let rc = match &self.storage {
            LinearStorage::Bf16 { .. } => anyhow::bail!(
                "bf16 storage never reaches the straight-through backward: Bf16LinearOp \
                 holds its weight directly"
            ),
            LinearStorage::GgmlQuant(_) => anyhow::bail!(
                "ggml-quant storage has no straight-through backward: the ocr decode \
                 path never trains"
            ),
            LinearStorage::Fp8E4m3(s) => {
                let (wp, _) = s.weight_u8.device_ptr(stream);
                let (sp, _) = s.b_scale_rows_dev.device_ptr(stream);
                let (op, _) = out.device_ptr_mut(stream);
                unsafe {
                    nv_kernels::cuda::dequant_weight_e4m3_rowscale_to_bf16_rowmajor(
                        stream.cu_stream() as *mut _,
                        wp as *const u8,
                        sp as *const f32,
                        op as *mut u16,
                        self.out_features as i32,
                        self.in_features as i32,
                    )
                }
            }
            LinearStorage::Nvfp4(s) => {
                let (wp, _) = s.weight_u8.device_ptr(stream);
                let (sp, _) = s.weight_scales_cm.device_ptr(stream);
                let (op, _) = out.device_ptr_mut(stream);
                unsafe {
                    nv_kernels::cuda::dequant_weight_nvfp4_swizzled_to_bf16_rowmajor(
                        stream.cu_stream() as *mut _,
                        wp as *const u8,
                        sp as *const u8,
                        op as *mut u16,
                        s.weight_alpha * s.input_stored_global,
                        self.out_features as i32,
                        self.in_features as i32,
                    )
                }
            }
        };
        anyhow::ensure!(
            rc == 0,
            "device weight dequant kernel returned {rc} for [{}x{}] {} storage",
            self.out_features,
            self.in_features,
            match &self.storage {
                LinearStorage::Fp8E4m3(_) => "fp8-rowscale",
                LinearStorage::Nvfp4(_) => "nvfp4-swizzled",
                LinearStorage::Bf16 { .. } => "bf16",
                LinearStorage::GgmlQuant(_) => "ggml-quant",
            }
        );
        Ok(out)
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn attach_lora(&self, hook: Arc<dyn LoraDeltaHook>) -> Result<()> {
        if hook.in_features() != self.in_features {
            anyhow::bail!(
                "attach_lora: hook in_features {} != linear in_features {}",
                hook.in_features(),
                self.in_features
            );
        }
        if hook.out_features() != self.out_features {
            anyhow::bail!(
                "attach_lora: hook out_features {} != linear out_features {}",
                hook.out_features(),
                self.out_features
            );
        }
        let mut slot = self
            .lora
            .write()
            .map_err(|e| anyhow::anyhow!("lora hook lock poisoned: {e}"))?;
        *slot = Some(hook);
        Ok(())
    }

    pub fn detach_lora(&self) {
        if let Ok(mut slot) = self.lora.write() {
            *slot = None;
        }
    }

    pub fn has_lora(&self) -> bool {
        self.lora.read().map(|s| s.is_some()).unwrap_or(false)
    }

    fn lora_delta(&self, x2: &Tensor, y2: Tensor, win: Option<(usize, usize)>) -> Result<Tensor> {
        let hook = {
            let slot = self
                .lora
                .read()
                .map_err(|e| anyhow::anyhow!("lora hook lock poisoned: {e}"))?;
            match &*slot {
                Some(h) => h.clone(),
                None => return Ok(y2),
            }
        };
        match hook.apply(x2, &y2, win)? {
            Some(replaced) => Ok(replaced),
            None => Ok(y2),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if *dims.last().unwrap() != self.in_features {
            anyhow::bail!(
                "Linear: input last dim {} != in_features {}",
                dims.last().unwrap(),
                self.in_features
            );
        }
        self.forward_inner(x, false)
    }

    pub fn forward_dense(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_inner(x, true)
    }

    fn forward_inner(&self, x: &Tensor, force_dense: bool) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        if *dims.last().unwrap() != self.in_features {
            anyhow::bail!(
                "Linear: input last dim {} != in_features {}",
                dims.last().unwrap(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;

        let out2 = self.matmul_impl(&x2, leading, force_dense)?;
        let out2 = self.lora_delta(&x2, out2, None)?;

        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        let mut out = out2.reshape(out_dims)?;
        if let Some(b) = &self.bias {
            out = out.broadcast_add(&b.to_dtype(out.dtype())?)?;
        }
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    pub fn prenorm_nvfp4_eligible(&self) -> bool {
        matches!(self.storage, LinearStorage::Nvfp4(_)) && self.bias.is_none() && !self.has_lora()
    }

    #[cfg(feature = "cuda")]
    pub fn try_forward_silu_nvfp4(&self, gate: &Tensor, up: &Tensor) -> Result<Option<Tensor>> {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var("NV_Q38_SILU_NVFP4").as_deref() == Ok("1"))
            || self.in_features != 17408
            || self.out_features != 5120
            || self.has_lora()
            || gate.dtype() != DType::BF16
            || up.dtype() != DType::BF16
            || !matches!(gate.device(), Device::Cuda(_))
        {
            return Ok(None);
        }
        let LinearStorage::Nvfp4(s) = &self.storage else {
            return Ok(None);
        };
        let dims = gate.dims();
        anyhow::ensure!(
            dims == up.dims()
                && dims.last() == Some(&self.in_features)
                && gate.device().same_device(up.device()),
            "NVFP4 SiLU gate/up shape or device mismatch"
        );
        let leading = gate.elem_count() / self.in_features;
        if !matches!(leading, 2048 | 4096)
            || leading <= s.bf16act_row_limit
            || (nvfp4_m1_gemv_opted_in() && leading <= nvfp4_gemv_row_block_ceiling())
        {
            return Ok(None);
        }
        a_quantized_fast_path_cannot_be_differentiated("NVFP4 SiLU up", up)?;
        let gate2 = gate.reshape((leading, self.in_features))?;
        let up2 = up.reshape((leading, self.in_features))?.contiguous()?;
        let out2 = self.matmul_nvfp4_impl(s, &gate2, leading, None, Some(&up2))?;
        let mut shape = dims[..dims.len() - 1].to_vec();
        shape.push(self.out_features);
        let mut out = out2.reshape(shape)?;
        if let Some(bias) = &self.bias {
            out = out.broadcast_add(&bias.to_dtype(out.dtype())?)?;
        }
        static TRACED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !TRACED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[linear] NV_Q38_SILU_NVFP4 engaged m={leading} k=17408 n=5120 arithmetic=candle_bf16 learned_weights=nvfp4");
        }
        Ok(Some(out))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_prenorm_nvfp4(
        &self,
        x_pre: &Tensor,
        norm_weight: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        if !self.prenorm_nvfp4_eligible() {
            anyhow::bail!("forward_prenorm_nvfp4: linear is not an eligible NVFP4 projection");
        }
        let dims = x_pre.dims().to_vec();
        if *dims.last().unwrap() != self.in_features {
            anyhow::bail!(
                "forward_prenorm_nvfp4: input last dim {} != in_features {}",
                dims.last().unwrap(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x_pre.reshape((leading, self.in_features))?;
        let s = match &self.storage {
            LinearStorage::Nvfp4(s) => s,
            _ => unreachable!("checked eligible above"),
        };
        let out2 = self.matmul_nvfp4_impl(
            s,
            &x2,
            leading,
            Some(FusedPreNorm {
                weight_bf16: norm_weight,
                eps,
            }),
            None,
        )?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        Ok(out2.reshape(out_dims)?)
    }

    fn matmul_impl(&self, x2: &Tensor, leading: usize, force_dense: bool) -> Result<Tensor> {
        match &self.storage {
            LinearStorage::Bf16 { weight, weight_t } => {
                self.matmul_bf16(weight, weight_t.as_ref(), x2, leading, force_dense)
            }
            LinearStorage::GgmlQuant(s) => {
                let xdt = x2.dtype();
                let xq = if xdt == candle_core::DType::F32 {
                    x2.clone()
                } else {
                    x2.to_dtype(candle_core::DType::F32)?
                };
                let y = candle_core::Module::forward(&s.qm, &xq)?;
                Ok(if y.dtype() == xdt {
                    y
                } else {
                    y.to_dtype(xdt)?
                })
            }
            #[cfg(feature = "cuda")]
            LinearStorage::Fp8E4m3(s) => self.matmul_fp8(s, x2, leading),
            #[cfg(feature = "cuda")]
            LinearStorage::Nvfp4(s) => self.matmul_nvfp4(s, x2, leading),
        }
    }

    pub fn ensure_pretransposed(&mut self) -> Result<bool> {
        match &mut self.storage {
            LinearStorage::Bf16 { weight, weight_t } => {
                if weight_t.is_none() {
                    *weight_t = Some(weight.t()?.contiguous()?);
                }
                Ok(true)
            }
            #[allow(unreachable_patterns)]
            _ => Ok(false),
        }
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16(
        &self,
        weight: &Tensor,
        weight_t: Option<&Tensor>,
        x2: &Tensor,
        leading: usize,
        force_dense: bool,
    ) -> Result<Tensor> {
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        if matches!(x2.device(), Device::Cuda(_)) && this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_bf16_dx_only(weight, x2);
        }
        if !matches!(x2.device(), Device::Cuda(_)) || x2.dtype() != DType::BF16 {
            return self.matmul_fallback(weight, x2);
        }
        let x_bf = x2.contiguous()?;
        let dev = match x_bf.device() {
            Device::Cuda(d) => d.clone(),
            _ => return self.matmul_fallback(weight, &x_bf),
        };
        let stream = crate::cuda_stream::current_stream(&dev);

        if leading == 1
            && !force_dense
            && !force_dense_bf16_enabled()
            && self.in_features.is_multiple_of(2)
            && std::env::var("NV_LINEAR_BF16_FORCE_CUBLAS").is_err()
        {
            return self.matmul_bf16_gemv(weight, &x_bf, &dev);
        }

        let w_bf = match weight_t {
            Some(wt) => wt.contiguous()?,
            None => weight.contiguous()?,
        };
        let m = leading as u64;
        let n = self.out_features as u64;
        let k = self.in_features as u64;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let gemm = TensorCoreGemm::new(stream.clone())?;
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let x_off = xl.start_offset();
            let w_off = wl.start_offset();
            if weight_t.is_some() {
                gemm.bf16_matmul_row_major_offs(
                    &stream, x_slice, x_off, w_slice, w_off, &mut c_dev, m, n, k, 1.0, 0.0,
                )?;
            } else if (2..=16).contains(&leading) && small_m_det_enabled() {
                gemm.bf16_matmul_row_major_bt_det_offs(
                    &stream, x_slice, x_off, w_slice, w_off, &mut c_dev, m, n, k, 1.0, 0.0,
                )?;
            } else {
                gemm.bf16_matmul_row_major_bt_offs(
                    &stream, x_slice, x_off, w_slice, w_off, &mut c_dev, m, n, k, 1.0, 0.0,
                )?;
            }
        }

        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
        let storage = candle_core::Storage::Cuda(out_storage);
        let out = candle_core::Tensor::from_storage(
            storage,
            (leading, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_gemv(
        &self,
        weight: &Tensor,
        x_bf: &Tensor,
        dev: &candle_core::CudaDevice,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        if this_call_is_building_an_autograd_graph(x_bf) {
            return self.matmul_bf16_dx_only(weight, x_bf);
        }
        let w_bf = weight.contiguous()?;
        let stream = crate::cuda_stream::current_stream(dev);
        let n = self.out_features;
        let k = self.in_features;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> =
            unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };

        let rc = {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let w_view = w_slice.slice(wl.start_offset()..);
            let (xp, _) = x_view.device_ptr(&stream);
            let (wp, _) = w_view.device_ptr(&stream);
            let (yp, _) = c_dev.device_ptr_mut(&stream);
            unsafe {
                nvk::gemv_bf16(
                    stream.cu_stream() as *mut _,
                    wp as *const u16,
                    xp as *const u16,
                    yp as *mut u16,
                    n as i32,
                    k as i32,
                )
            }
        };
        if rc != 0 {
            anyhow::bail!("gemv_bf16 kernel returned {rc}");
        }

        let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        Ok(candle_core::Tensor::from_storage(
            storage,
            (1usize, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_bf16(
        &self,
        weight: &Tensor,
        _weight_t: Option<&Tensor>,
        x2: &Tensor,
        _leading: usize,
        _force_dense: bool,
    ) -> Result<Tensor> {
        self.matmul_fallback(weight, x2)
    }

    pub fn forward_rows(&self, x: &Tensor, row_off: usize, rows: usize) -> Result<Tensor> {
        let weight = match &self.storage {
            LinearStorage::Bf16 { weight, .. } => weight,
            _ => anyhow::bail!("forward_rows requires bf16 storage"),
        };
        if self.bias.is_some() {
            anyhow::bail!("forward_rows does not support bias");
        }
        if row_off + rows > self.out_features {
            anyhow::bail!(
                "forward_rows: rows {row_off}..{} out of range (out_features {})",
                row_off + rows,
                self.out_features
            );
        }
        let dims = x.dims().to_vec();
        if *dims.last().unwrap_or(&0) != self.in_features {
            anyhow::bail!(
                "forward_rows: input last dim {:?} != in_features {}",
                dims.last(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;
        let out2 = self.matmul_bf16_rows(weight, &x2, leading, row_off, rows)?;
        let out2 = self.lora_delta(&x2, out2, Some((row_off, rows)))?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(rows);
        Ok(out2.reshape(out_dims)?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_rows(
        &self,
        weight: &Tensor,
        x2: &Tensor,
        leading: usize,
        row_off: usize,
        rows: usize,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;
        use nv_quant::matmul::TensorCoreGemm;

        if matches!(x2.device(), Device::Cuda(_)) && this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_bf16_dx_only(&weight.narrow(0, row_off, rows)?, x2);
        }
        if !matches!(x2.device(), Device::Cuda(_)) || x2.dtype() != DType::BF16 {
            return self.matmul_fallback(&weight.narrow(0, row_off, rows)?, x2);
        }
        let x_bf = x2.contiguous()?;
        let dev = match x_bf.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let k = self.in_features;

        if leading == 1
            && !force_dense_bf16_enabled()
            && k.is_multiple_of(2)
            && std::env::var("NV_LINEAR_BF16_FORCE_CUBLAS").is_err()
        {
            let w_bf = weight.contiguous()?;
            let mut c_dev: cudarc::driver::CudaSlice<bf16> =
                unsafe { stream.alloc::<bf16>(rows).map_err(|e| anyhow::anyhow!(e))? };
            let rc = {
                let (x_storage, xl) = x_bf.storage_and_layout();
                let (w_storage, wl) = w_bf.storage_and_layout();
                let x_cuda = match &*x_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage for input"),
                };
                let w_cuda = match &*w_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage for weight"),
                };
                let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
                let x_view = x_slice.slice(xl.start_offset()..);
                let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                let w_view = w_slice.slice(wl.start_offset() + row_off * k..);
                let (xp, _) = x_view.device_ptr(&stream);
                let (wp, _) = w_view.device_ptr(&stream);
                let (yp, _) = c_dev.device_ptr_mut(&stream);
                unsafe {
                    nvk::gemv_bf16(
                        stream.cu_stream() as *mut _,
                        wp as *const u16,
                        xp as *const u16,
                        yp as *mut u16,
                        rows as i32,
                        k as i32,
                    )
                }
            };
            if rc != 0 {
                anyhow::bail!("gemv_bf16 kernel returned {rc}");
            }
            let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
            return Ok(candle_core::Tensor::from_storage(
                candle_core::Storage::Cuda(storage),
                (1usize, rows),
                candle_core::op::BackpropOp::none(),
                false,
            ));
        }

        let w_bf = weight.contiguous()?;
        let m = leading as u64;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * rows)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())?;
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            gemm.bf16_matmul_row_major_bt_offs(
                &stream,
                x_slice,
                xl.start_offset(),
                w_slice,
                wl.start_offset() + row_off * k,
                &mut c_dev,
                m,
                rows as u64,
                k as u64,
                1.0,
                0.0,
            )?;
        }
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
        Ok(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            (leading, rows),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_bf16_rows(
        &self,
        weight: &Tensor,
        x2: &Tensor,
        _leading: usize,
        row_off: usize,
        rows: usize,
    ) -> Result<Tensor> {
        self.matmul_fallback(&weight.narrow(0, row_off, rows)?, x2)
    }

    pub fn forward_dense_det(&self, x: &Tensor) -> Result<Tensor> {
        let weight = match &self.storage {
            LinearStorage::Bf16 { weight, .. } => weight,
            _ => anyhow::bail!("forward_dense_det requires bf16 storage"),
        };
        if self.bias.is_some() {
            anyhow::bail!("forward_dense_det does not support bias");
        }
        let dims = x.dims().to_vec();
        if *dims.last().unwrap_or(&0) != self.in_features {
            anyhow::bail!(
                "forward_dense_det: input last dim {:?} != in_features {}",
                dims.last(),
                self.in_features
            );
        }
        let leading: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((leading, self.in_features))?;
        let out2 = self.matmul_bf16_det(weight, &x2, leading)?;
        let out2 = self.lora_delta(&x2, out2, None)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(self.out_features);
        Ok(out2.reshape(out_dims)?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_det(&self, weight: &Tensor, x2: &Tensor, leading: usize) -> Result<Tensor> {
        use half::bf16;
        use nv_quant::matmul::TensorCoreGemm;

        if matches!(x2.device(), Device::Cuda(_)) && this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_bf16_dx_only(weight, x2);
        }
        if !matches!(x2.device(), Device::Cuda(_)) || x2.dtype() != DType::BF16 {
            return self.matmul_fallback(weight, x2);
        }
        let x_bf = x2.contiguous()?;
        let dev = match x_bf.device() {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        let w_bf = weight.contiguous()?;
        let m = leading as u64;
        let n = self.out_features as u64;
        let k = self.in_features as u64;

        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let gemm = TensorCoreGemm::new(stream.clone())?;
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let (w_storage, wl) = w_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            gemm.bf16_matmul_row_major_bt_det_offs(
                &stream,
                x_slice,
                xl.start_offset(),
                w_slice,
                wl.start_offset(),
                &mut c_dev,
                m,
                n,
                k,
                1.0,
                0.0,
            )?;
        }
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev);
        Ok(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            (leading, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(not(feature = "cuda"))]
    fn matmul_bf16_det(&self, weight: &Tensor, x2: &Tensor, _leading: usize) -> Result<Tensor> {
        self.matmul_fallback(weight, x2)
    }

    const FALLBACK_TAKES_THE_PRETRANSPOSED_RESIDENT_INSTEAD_OF_REMATERIALIZING_IT: &'static str =
        "weight_t is built as weight.t().contiguous() at construction, byte-identical to what \
         this fallback used to materialize on every forward, so reusing it is bitwise by \
         construction and stops a full weight-sized transpose per call (the qlora window paid \
         ~9 GB of rematerialized transposes per training step); the shape guard keeps narrowed \
         and quant-dequant callers on the materializing path, and the needs_bf16_upcast guard \
         keeps cpu bf16 on the f32-upcast path because the candle cpu backend has no bf16 matmul \
         and every cpu oracle depends on that upcast";

    fn matmul_fallback(&self, weight: &Tensor, x2: &Tensor) -> Result<Tensor> {
        let on_cpu = matches!(x2.device(), candle_core::Device::Cpu);
        let needs_bf16_upcast =
            on_cpu && (weight.dtype() == DType::BF16 || x2.dtype() == DType::BF16);
        if !needs_bf16_upcast {
            if let LinearStorage::Bf16 {
                weight_t: Some(wt), ..
            } = &self.storage
            {
                let same_shape = wt.dims() == [self.in_features, self.out_features]
                    && weight.dims() == [self.out_features, self.in_features];
                if same_shape && wt.dtype() == x2.dtype() && wt.device().same_device(x2.device()) {
                    let _ =
                        Self::FALLBACK_TAKES_THE_PRETRANSPOSED_RESIDENT_INSTEAD_OF_REMATERIALIZING_IT;
                    return Ok(x2.matmul(wt)?);
                }
            }
        }
        if needs_bf16_upcast {
            let orig_dtype = x2.dtype();
            let x_f32 = x2.to_dtype(DType::F32)?;
            let w_f32 = weight.to_dtype(DType::F32)?;
            let out = x_f32.matmul(&w_f32.t()?.contiguous()?)?;
            return Ok(out.to_dtype(orig_dtype)?);
        }
        let w = weight.to_dtype(x2.dtype())?;
        Ok(x2.matmul(&w.t()?.contiguous()?)?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_bf16_dx_only(&self, weight: &Tensor, x2: &Tensor) -> Result<Tensor> {
        if !matches!(x2.device(), Device::Cuda(_))
            || !matches!(weight.device(), Device::Cuda(_))
            || weight.dtype() != DType::BF16
        {
            return self.matmul_fallback(weight, x2);
        }
        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
        let y_bf = x_bf.apply_op1(Bf16LinearOp::holding_the_frozen_weight(weight)?)?;
        Ok(y_bf.to_dtype(x2.dtype())?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_quant_dx_only(&self, x2: &Tensor) -> Result<Tensor> {
        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
        let y_bf = x_bf.apply_op1(QuantLinearOp::sharing_the_resident_quantized_storage(self)?)?;
        Ok(y_bf.to_dtype(x2.dtype())?)
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8(&self, s: &Fp8Storage, x2: &Tensor, leading: usize) -> Result<Tensor> {
        use half::bf16;

        if matches!(x2.device(), Device::Cuda(_)) && this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_quant_dx_only(x2);
        }
        let dev = match x2.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires CUDA input"),
        };
        e4m3_route_line_once_per_slot_and_row_count_because_a_single_once_only_ever_reports_whichever_shape_warmed_first(
            0,
            "fp8 e4m3 linear reached at this row count",
            leading,
            self.out_features,
            self.in_features,
            &format!("scale_mode={:?} xdtype={:?}", s.scale_mode, x2.dtype()),
        );
        if let Some(out) = self.matmul_fp8_marlin_rows(s, x2, leading, &dev)? {
            return Ok(out);
        }
        if let Some(out) = self.matmul_fp8_small_m_gemv(s, x2, leading, &dev)? {
            return Ok(out);
        }
        if s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow {
            return self.matmul_fp8_device_rowquant_pertensor_raw_gemm_rowcol_epilogue(
                s, x2, leading, &dev,
            );
        }
        let _ = &s.cuda_device;
        let stream = crate::cuda_stream::current_stream(&dev);
        let m = leading as u64;
        let n = self.out_features as u64;
        let k = self.in_features as u64;

        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
        let x_host: Vec<bf16> = x_bf.flatten_all()?.to_vec1::<bf16>()?;

        let (x_fp8_bytes, x_row_scales) = {
            let (bytes, scale) = nv_quant::fp8::quantize_e4m3_per_tensor(&x_host);
            (bytes, vec![scale])
        };
        let scale_factor = match self.kind {
            LinearKind::Fp8E4m3 { a_scale, .. } => a_scale,
            _ => 1.0,
        };
        let effective_a_scales: Vec<f32> = x_row_scales.iter().map(|v| v * scale_factor).collect();
        #[allow(deprecated)]
        let x_fp8_dev = stream
            .clone_htod(&x_fp8_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let a_scale_dev_call = stream
            .clone_htod(&effective_a_scales)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut d_dev: cudarc::driver::CudaSlice<bf16> = stream
            .alloc_zeros::<bf16>(leading * self.out_features)
            .map_err(|e| anyhow::anyhow!(e))?;

        {
            let mut runner = s
                .runner
                .lock()
                .map_err(|e| anyhow::anyhow!("Fp8 runner mutex poisoned: {e}"))?;
            runner.with_stream(&stream, |runner| {
                runner.matmul_e4m3_weight_row(
                    &x_fp8_dev,
                    &s.weight_u8,
                    &mut d_dev,
                    m,
                    n,
                    k,
                    &a_scale_dev_call,
                    &s.b_scale_dev,
                )
            })?
        }

        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(d_dev, dev);
        let storage = candle_core::Storage::Cuda(out_storage);
        let out = candle_core::Tensor::from_storage(
            storage,
            (leading, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(out)
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8_marlin_rows(
        &self,
        s: &Fp8Storage,
        x2: &Tensor,
        leading: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::DevicePtr;
        use half::bf16;
        if !marlin_fp8_rows_opt_in()
            || s.scale_mode != nv_quant::fp8::Fp8ScaleMode::PerOuterRow
            || !(1..=MARLIN_FP8_ROWS_MAX_16_IS_THE_INSTANTIATED_THREAD_M_BLOCKS_1)
                .contains(&leading)
            || self.out_features % 64 != 0
            || self.in_features % 16 != 0
            || x2.dtype() != DType::BF16
        {
            return Ok(None);
        }
        let Some(tw) = s
            .marlin_fp8_twin_built_once_from_the_resident_e4m3_rows
            .get()
            .cloned()
            .flatten()
        else {
            return Ok(None);
        };
        let n = self.out_features;
        let stream = crate::cuda_stream::current_stream(dev);
        let x_bf = x2.contiguous()?;
        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * n)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("expected cuda storage for fp8 marlin input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (xp, _gx) = x_view.device_ptr(&stream);
            marlin_fp8_rows_gemm_into(&stream, &tw, xp, leading, &mut c_dev)?;
        }
        static MARLIN_FP8_ROWS_FIRED_ONCE: std::sync::Once = std::sync::Once::new();
        MARLIN_FP8_ROWS_FIRED_ONCE.call_once(|| eprintln!("[marlin_fp8_rows] fired m={leading} n={n} k={} (bf16 rows on the fp8 Marlin twin, fp32 reduce)", tw.k));
        let out_storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev.clone());
        Ok(Some(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(out_storage),
            (leading, n),
            candle_core::op::BackpropOp::none(),
            false,
        )))
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8_small_m_gemv(
        &self,
        s: &Fp8Storage,
        x2: &Tensor,
        leading: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Option<Tensor>> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        if s.scale_mode != nv_quant::fp8::Fp8ScaleMode::PerOuterRow
            || !(1..=16).contains(&leading)
            || !self.in_features.is_multiple_of(16)
            || x2.dtype() != DType::BF16
        {
            return Ok(None);
        }
        if leading >= 2 && verify_tc_fp8_lt_gemm_scope_active() {
            return Ok(None);
        }
        if let Some(min_m) = e4m3_tile_min_m() {
            if leading >= min_m
                && self.out_features
                    >= THE_TILE_NEEDS_THIS_MANY_OUTPUT_ROWS_BEFORE_IT_BEATS_THE_STAGED_GEMV_BECAUSE_A_NARROW_N_STARVES_ITS_CTAS
            {
                e4m3_route_line_once_per_slot_and_row_count_because_a_single_once_only_ever_reports_whichever_shape_warmed_first(
                    1,
                    "NV_Q38_E4M3_TILE_MIN_M active: the staged bf16-activation gemv yields to the \
                     cublasLt e4m3 tile, which quantises the activations to e4m3 per row",
                    leading,
                    self.out_features,
                    self.in_features,
                    &format!("tile_min_m={min_m}"),
                );
                return Ok(None);
            }
        }
        let x_bf = x2.contiguous()?;
        let stream = crate::cuda_stream::current_stream(dev);
        let n = self.out_features;
        let k = self.in_features;
        let mut c_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(leading * n)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let rc = {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("expected cuda storage for fp8 gemv input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (xp, _gx) = x_view.device_ptr(&stream);
            let (wp, _gw) = s.weight_u8.device_ptr(&stream);
            let (sp, _gs) = s.b_scale_rows_dev.device_ptr(&stream);
            let (yp, _gy) = c_dev.device_ptr_mut(&stream);
            let group = e4m3_group_rows_per_warp().filter(|_| {
                leading >= E4M3_GROUP_LEAVES_THE_M1_ROUTE_ALONE_BECAUSE_ITS_TWO_ROWS_PER_WARP_KERNEL_IS_ALREADY_THE_TUNED_ONE_ROW_SHAPE
                    && n >= THE_GROUP_GEMV_NEEDS_THIS_MANY_OUTPUT_ROWS_BEFORE_ITS_K_TILE_PAYS_FOR_ITS_EXTRA_SYNCS_BECAUSE_A_NARROW_N_GIVES_EACH_BLOCK_TOO_FEW_WEIGHT_BYTES_PER_STAGED_TILE
                    && leading as i32
                        <= unsafe { nvk::gemv_e4m3_untiled_staging_reaches_this_m(k as i32) }
            });
            e4m3_route_line_once_per_slot_and_row_count_because_a_single_once_only_ever_reports_whichever_shape_warmed_first(
                2,
                match group {
                    Some(_) => "NV_Q38_E4M3_GROUP active: the k-tiled group gemv replaces the \
                                whole-row-staged gemv_e4m3_mk row for row",
                    None => "staged gemv route: whole-row-staged gemv_e4m3_mk (group arm not \
                             engaged at this row count)",
                },
                leading,
                n,
                k,
                &format!(
                    "rows_per_warp={} tile_v={} untiled_reaches_m={}",
                    group.unwrap_or(0),
                    unsafe { nvk::gemv_e4m3_group_tile_v(leading as i32, k as i32) },
                    unsafe { nvk::gemv_e4m3_untiled_staging_reaches_this_m(k as i32) }
                ),
            );
            unsafe {
                match group {
                    Some(rows_per_warp) => nvk::gemv_e4m3_group(
                        stream.cu_stream() as *mut _,
                        wp as *const u8,
                        sp as *const f32,
                        xp as *const u16,
                        yp as *mut u16,
                        n as i32,
                        k as i32,
                        leading as i32,
                        rows_per_warp,
                    ),
                    None => {
                        let reach =
                            nvk::gemv_e4m3_untiled_staging_reaches_this_m(k as i32) as usize;
                        let chunk_rows_past_the_staging_cap = leading > reach
                            && leading
                                <= verify_one_arithmetic_rows_ceiling_env_nv_verify_one_arith_rows_default_8_max_24_serving_never_exceeds_8_only_the_mrow_ladder_raises_it()
                            && VERIFY_INTERIOR_REACHES_3_TO_7_ARE_THE_SHAPES_WHOSE_TILE_FALLBACK_SPLIT_THE_VERIFY_RANGE
                                .contains(&reach);
                        if chunk_rows_past_the_staging_cap {
                            let _ = A_ROW_CHUNKED_GEMV_IS_BITWISE_THE_SINGLE_LAUNCH_BECAUSE_EVERY_ROW_ACCUMULATES_ALONE_IN_ONE_K_ORDER;
                            let mut rc = 0i32;
                            let mut start = 0usize;
                            while start < leading && rc == 0 {
                                let rows = reach.min(leading - start);
                                rc = nvk::gemv_e4m3_mk(
                                    stream.cu_stream() as *mut _,
                                    wp as *const u8,
                                    sp as *const f32,
                                    (xp + (start * k * 2) as u64) as *const u16,
                                    (yp + (start * n * 2) as u64) as *mut u16,
                                    n as i32,
                                    k as i32,
                                    rows as i32,
                                );
                                start += rows;
                            }
                            rc
                        } else {
                            nvk::gemv_e4m3_mk(
                                stream.cu_stream() as *mut _,
                                wp as *const u8,
                                sp as *const f32,
                                xp as *const u16,
                                yp as *mut u16,
                                n as i32,
                                k as i32,
                                leading as i32,
                            )
                        }
                    }
                }
            }
        };
        if rc == -1 {
            return Ok(None);
        }
        if rc != 0 {
            anyhow::bail!("gemv_e4m3_mk kernel returned {rc}");
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(c_dev, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        Ok(Some(candle_core::Tensor::from_storage(
            storage,
            (leading, n),
            candle_core::op::BackpropOp::none(),
            false,
        )))
    }

    #[cfg(feature = "cuda")]
    fn matmul_fp8_device_rowquant_pertensor_raw_gemm_rowcol_epilogue(
        &self,
        s: &Fp8Storage,
        x2: &Tensor,
        leading: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_kernels::cuda as nvk;

        let m = leading;
        let n = self.out_features;
        let k = self.in_features;
        anyhow::ensure!(
            k.is_multiple_of(16),
            "fp8 row-scaled linear: in_features {k} % 16 != 0; cuBLASLt e4m3 needs 16-element \
             aligned K and every q38 fp8 module satisfies it, so a trip here is a new module"
        );
        let stream = crate::cuda_stream::current_stream(dev);
        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;

        let mut x_q: cudarc::driver::CudaSlice<u8> =
            unsafe { stream.alloc::<u8>(m * k).map_err(|e| anyhow::anyhow!(e))? };
        let mut a_scale_rows: cudarc::driver::CudaSlice<f32> =
            unsafe { stream.alloc::<f32>(m).map_err(|e| anyhow::anyhow!(e))? };
        {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(st) => st,
                _ => anyhow::bail!("expected cuda storage for fp8 rowquant input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (xp, _gx) = x_view.device_ptr(&stream);
            let (qp, _gq) = x_q.device_ptr_mut(&stream);
            let (sp, _gs) = a_scale_rows.device_ptr_mut(&stream);
            let rc = unsafe {
                nvk::rowquant_e4m3(
                    stream.cu_stream() as *mut _,
                    xp as *const u16,
                    qp as *mut u8,
                    sp as *mut f32,
                    m as i32,
                    k as i32,
                )
            };
            anyhow::ensure!(rc == 0, "rowquant_e4m3 rc={rc}");
        }

        let mut d_dev: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m * n)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        {
            let mut runner = s
                .runner
                .lock()
                .map_err(|e| anyhow::anyhow!("Fp8 runner mutex poisoned: {e}"))?;
            runner.with_stream(&stream, |runner| {
            let chunk_rows_past_the_m6_plan = m
                > LT_TILE_ROWS_PER_CHUNK_6_THE_WIDEST_ROW_COUNT_WHOSE_HEURISTIC_PLAN_FAMILY_COVERS_EVERY_SHIPPED_VERIFY_CELL
                && m <= verify_one_arithmetic_rows_ceiling_env_nv_verify_one_arith_rows_default_8_max_24_serving_never_exceeds_8_only_the_mrow_ladder_raises_it();
            if chunk_rows_past_the_m6_plan {
                let _ = A_ROW_CHUNKED_LT_TILE_RIDES_THE_M6_PLAN_BECAUSE_THE_HEURISTIC_MOVES_PLANS_BETWEEN_M6_AND_M7;
                let span = LT_TILE_SPAN_4_THE_PRODUCTION_MTP_VERIFY_LAUNCH_WHOSE_PLAN_FAMILY_ALSO_COVERS_M5_AND_M6_AT_THE_CHEAPEST_ROW_COUNT;
                let planned = runner.matmul_e4m3_weight_row_span_with_plan_rows(
                    &x_q,
                    &s.weight_u8,
                    &mut d_dev,
                    0,
                    m as u64,
                    span as u64,
                    n as u64,
                    k as u64,
                    &s.a_scale_dev,
                    &s.a_scale_dev,
                );
                if let Err(e) = planned {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| {
                        eprintln!(
                            "[linear] m4-plan single launch refused at m={m} n={n} k={k} \
                             ({e:#}); riding the bitwise-equal (0,{span})+({},{span}) span \
                             composition at its second full weight read",
                            m - span
                        );
                    });
                    for start in [0usize, m - span] {
                        runner.matmul_e4m3_weight_row_span(
                            &x_q,
                            &s.weight_u8,
                            &mut d_dev,
                            start as u64,
                            span as u64,
                            n as u64,
                            k as u64,
                            &s.a_scale_dev,
                            &s.a_scale_dev,
                        )?;
                    }
                }
            } else {
                runner.matmul_e4m3_weight_row(
                    &x_q,
                    &s.weight_u8,
                    &mut d_dev,
                    m as u64,
                    n as u64,
                    k as u64,
                    &s.a_scale_dev,
                    &s.a_scale_dev,
                )?
            }
            Ok(())
            })?;
        }
        {
            let (dp, _gd) = d_dev.device_ptr_mut(&stream);
            let (rp, _gr) = a_scale_rows.device_ptr(&stream);
            let (cp, _gc) = s.b_scale_rows_dev.device_ptr(&stream);
            let rc = unsafe {
                nvk::scale_rowcol_bf16(
                    stream.cu_stream() as *mut _,
                    dp as *mut u16,
                    rp as *const f32,
                    cp as *const f32,
                    m as i32,
                    n as i32,
                )
            };
            anyhow::ensure!(rc == 0, "scale_rowcol_bf16 rc={rc}");
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(d_dev, dev.clone());
        Ok(candle_core::Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (m, n),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn matmul_nvfp4(&self, s: &Nvfp4Storage, x2: &Tensor, leading: usize) -> Result<Tensor> {
        if matches!(x2.device(), Device::Cuda(_)) && this_call_is_building_an_autograd_graph(x2) {
            return self.matmul_quant_dx_only(x2);
        }
        self.matmul_nvfp4_impl(s, x2, leading, None, None)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_nvfp4_gemv_m1_unconditionally_which_a_draft_only_lm_head_requires(
        &self,
        x: &Tensor,
    ) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let leading: usize = dims[..dims.len().saturating_sub(1)].iter().product();
        anyhow::ensure!(
            leading == 1,
            "forward_nvfp4_gemv_m1: draft lm_head serves one row per step, got leading {leading}"
        );
        anyhow::ensure!(
            dims.last() == Some(&self.in_features),
            "forward_nvfp4_gemv_m1: input last dim {:?} != in_features {}",
            dims.last(),
            self.in_features
        );
        let x2 = x.reshape((1usize, self.in_features))?;
        match &self.storage {
            LinearStorage::Nvfp4(s) => self.matmul_nvfp4_m1_gemv(s, &x2, 1, None),
            _ => anyhow::bail!("forward_nvfp4_gemv_m1: linear is not NVFP4-resident"),
        }
    }

    #[cfg(feature = "cuda")]
    pub fn set_nvfp4_bf16act_rows_before_sharing(&mut self, max_rows: usize) -> Result<()> {
        anyhow::ensure!(
            max_rows > 0,
            "NVFP4 BF16-activation row limit must be positive"
        );
        let storage = match &mut self.storage {
            LinearStorage::Nvfp4(storage) => storage,
            _ => anyhow::bail!("NVFP4 BF16-activation policy requires NVFP4 storage"),
        };
        let storage = Arc::get_mut(storage).ok_or_else(|| {
            anyhow::anyhow!("NVFP4 BF16-activation policy must be set before storage is shared")
        })?;
        storage.bf16act_row_limit = max_rows;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub fn nvfp4_bf16act_row_limit(&self) -> Option<usize> {
        match &self.storage {
            LinearStorage::Nvfp4(storage) => Some(storage.bf16act_row_limit),
            _ => None,
        }
    }

    #[cfg(feature = "cuda")]
    fn matmul_nvfp4_m1_gemv(
        &self,
        s: &Nvfp4Storage,
        x2: &Tensor,
        rows_each_launched_exactly_as_decode_launches_its_one_row: usize,
        prenorm: Option<&FusedPreNorm<'_>>,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use std::ffi::c_void;

        let rows = rows_each_launched_exactly_as_decode_launches_its_one_row;
        anyhow::ensure!(
            rows >= 1,
            "matmul_nvfp4_m1_gemv: empty row block ({})",
            A_VERIFY_BLOCK_MUST_TAKE_THE_SAME_KERNEL_DECODE_TAKES_OR_IT_CANNOT_REPRODUCE_DECODES_ROWS
        );
        let dev = match x2.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires CUDA input"),
        };
        a_quantized_fast_path_cannot_be_differentiated("matmul_nvfp4_m1_gemv", x2)?;
        let stream = crate::cuda_stream::current_stream(&dev);
        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;
        let mut d: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(rows * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };
        let mut xn: Option<cudarc::driver::CudaSlice<bf16>> = None;
        let norm_w_kept_alive_past_the_pointer_guard = prenorm.map(|pn| pn.weight_bf16.clone());
        let rc = {
            let (x_storage, xl) = x_bf.storage_and_layout();
            let x_cuda = match &*x_storage {
                candle_core::Storage::Cuda(cs) => cs,
                _ => anyhow::bail!("expected cuda storage for input"),
            };
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
            let x_view = x_slice.slice(xl.start_offset()..);
            let (x_ptr, _gx) = x_view.device_ptr(&stream);
            let (w_ptr, _gw) = s.weight_u8.device_ptr(&stream);
            let (s_ptr, _gs) = s.weight_scales_cm.device_ptr(&stream);
            let (d_ptr, _gd) = d.device_ptr_mut(&stream);
            let bf16_act_alpha_refolds_the_input_global_the_mma_route_spends_quantizing_x =
                s.weight_alpha * s.input_stored_global;
            let x_row_bytes = self.in_features * std::mem::size_of::<bf16>();
            let d_row_bytes = self.out_features * std::mem::size_of::<bf16>();
            let x_base = x_ptr as usize;
            let d_base = d_ptr as usize;
            match (prenorm, norm_w_kept_alive_past_the_pointer_guard.as_ref()) {
                (Some(pn), Some(nw)) => {
                    let (w_storage, wl) = nw.storage_and_layout();
                    let w_cuda = match &*w_storage {
                        candle_core::Storage::Cuda(cs) => cs,
                        _ => anyhow::bail!("expected cuda storage for norm weight"),
                    };
                    let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                    let w_view = w_slice.slice(wl.start_offset()..);
                    let (nw_ptr, _gn) = w_view.device_ptr(&stream);
                    let folds_in_the_gemv = unsafe {
                        nv_kernels::cuda::nvfp4_gemv_prenorm_fuses_this_row(self.in_features as i32)
                    } != 0;
                    let batched = rows > 1
                        && !folds_in_the_gemv
                        && nvfp4_gemv_mrows_case_instantiated_mirroring_gemv_act_mrows_case_instantiated(
                            rows,
                        );
                    let mut buf: cudarc::driver::CudaSlice<bf16> = unsafe {
                        stream
                            .alloc::<bf16>(if batched {
                                rows * self.in_features
                            } else {
                                self.in_features
                            })
                            .map_err(|e| anyhow::anyhow!(e))?
                    };
                    let rc_fused = {
                        let (xn_ptr, _gxn) = buf.device_ptr_mut(&stream);
                        let xn_base = xn_ptr as usize;
                        let mut rc_rows = 0i32;
                        if batched {
                            for row in 0..rows {
                                rc_rows = unsafe {
                                    nv_kernels::cuda::rmsnorm_bf16(
                                        stream.cu_stream() as *mut c_void,
                                        (x_base + row * x_row_bytes) as *const u16,
                                        nw_ptr as *const u16,
                                        (xn_base + row * x_row_bytes) as *mut u16,
                                        1,
                                        self.in_features,
                                        pn.eps,
                                    )
                                };
                                if rc_rows != 0 {
                                    break;
                                }
                            }
                            if rc_rows == 0 {
                                rc_rows = unsafe {
                                    nv_kernels::cuda::nvfp4_gemv_bf16act_mrows(
                                        stream.cu_stream() as *mut c_void,
                                        w_ptr as *const u8,
                                        s_ptr as *const u8,
                                        xn_base as *const u16,
                                        d_base as *mut u16,
                                        bf16_act_alpha_refolds_the_input_global_the_mma_route_spends_quantizing_x,
                                        self.out_features as i32,
                                        self.in_features as i32,
                                        rows as i32,
                                    )
                                };
                            }
                        } else {
                            for row in 0..rows {
                                rc_rows = unsafe {
                                    nv_kernels::cuda::nvfp4_gemv_bf16act_prenorm(
                                        stream.cu_stream() as *mut c_void,
                                        w_ptr as *const u8,
                                        s_ptr as *const u8,
                                        (x_base + row * x_row_bytes) as *const u16,
                                        nw_ptr as *const u16,
                                        xn_base as *mut u16,
                                        (d_base + row * d_row_bytes) as *mut u16,
                                        bf16_act_alpha_refolds_the_input_global_the_mma_route_spends_quantizing_x,
                                        self.out_features as i32,
                                        self.in_features as i32,
                                        pn.eps,
                                    )
                                };
                                if rc_rows != 0 {
                                    break;
                                }
                            }
                        }
                        rc_rows
                    };
                    xn = Some(buf);
                    rc_fused
                }
                _ => {
                    if nvfp4_gemv_mrows_case_instantiated_mirroring_gemv_act_mrows_case_instantiated(
                        rows,
                    ) {
                        unsafe {
                            nv_kernels::cuda::nvfp4_gemv_bf16act_mrows(
                                stream.cu_stream() as *mut c_void,
                                w_ptr as *const u8,
                                s_ptr as *const u8,
                                x_base as *const u16,
                                d_base as *mut u16,
                                bf16_act_alpha_refolds_the_input_global_the_mma_route_spends_quantizing_x,
                                self.out_features as i32,
                                self.in_features as i32,
                                rows as i32,
                            )
                        }
                    } else {
                        let mut rc_rows = 0i32;
                        for row in 0..rows {
                            rc_rows = unsafe {
                                nv_kernels::cuda::nvfp4_gemv_bf16act(
                                    stream.cu_stream() as *mut c_void,
                                    w_ptr as *const u8,
                                    s_ptr as *const u8,
                                    (x_base + row * x_row_bytes) as *const u16,
                                    (d_base + row * d_row_bytes) as *mut u16,
                                    bf16_act_alpha_refolds_the_input_global_the_mma_route_spends_quantizing_x,
                                    self.out_features as i32,
                                    self.in_features as i32,
                                )
                            };
                            if rc_rows != 0 {
                                break;
                            }
                        }
                        rc_rows
                    }
                }
            }
        };
        drop(xn);
        if rc != 0 {
            anyhow::bail!("nvfp4_gemv_bf16act rc={rc}");
        }
        let d_storage = candle_core::CudaStorage::wrap_cuda_slice(d, dev);
        let storage = candle_core::Storage::Cuda(d_storage);
        Ok(candle_core::Tensor::from_storage(
            storage,
            (rows, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    #[cfg(feature = "cuda")]
    fn matmul_nvfp4_impl(
        &self,
        s: &Nvfp4Storage,
        x2: &Tensor,
        leading: usize,
        prenorm: Option<FusedPreNorm<'_>>,
        silu_up: Option<&Tensor>,
    ) -> Result<Tensor> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
        use std::ffi::c_void;

        let dev = match x2.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires CUDA input"),
        };
        a_quantized_fast_path_cannot_be_differentiated("matmul_nvfp4_impl", x2)?;
        let _ = &s.cuda_device;
        let stream = crate::cuda_stream::current_stream(&dev);
        let n = self.out_features as u64;
        let k = self.in_features as u64;
        if !self.in_features.is_multiple_of(BLOCK_SIZE) {
            anyhow::bail!(
                "NVFP4 Linear: in_features {} must be a multiple of {}",
                self.in_features,
                BLOCK_SIZE
            );
        }
        if self.out_features < MIN_TILE || self.in_features < MIN_TILE {
            anyhow::bail!(
                "NVFP4 Linear: out_features and in_features must each be >= {} (got {} and {})",
                MIN_TILE,
                self.out_features,
                self.in_features
            );
        }

        let m_logical = leading;
        if m_logical == 0 {
            anyhow::bail!("NVFP4 Linear: empty input batch");
        }

        if s.bf16act_row_limit > 0 && m_logical <= s.bf16act_row_limit {
            return self.matmul_nvfp4_m1_gemv(s, x2, m_logical, prenorm.as_ref());
        }

        if nvfp4_m1_gemv_opted_in() && m_logical <= nvfp4_gemv_row_block_ceiling() {
            return self.matmul_nvfp4_m1_gemv(s, x2, m_logical, prenorm.as_ref());
        }

        let m_padded = m_logical.max(MIN_TILE);

        let x_bf = x2.to_dtype(DType::BF16)?.contiguous()?;

        let cfx_route = prenorm.is_none()
            && m_padded == m_logical
            && nvfp4_cfx_route_ready(&stream, m_padded, self.out_features, self.in_features);

        let blocks_per_row = self.in_features / BLOCK_SIZE;
        let (packed_bytes, scales_bytes) = if cfx_route {
            let m96 = m_padded.div_ceil(96) * 96;
            (m96 * self.in_features / 2, m96 * blocks_per_row)
        } else {
            (
                m_padded * self.in_features / 2,
                m_padded.div_ceil(128) * 128 * blocks_per_row.div_ceil(4) * 4,
            )
        };

        let mut d_padded: cudarc::driver::CudaSlice<bf16> = unsafe {
            stream
                .alloc::<bf16>(m_padded * self.out_features)
                .map_err(|e| anyhow::anyhow!(e))?
        };

        let skip_pad = m_padded > m_logical && !nvfp4_quant_fullpad_enabled();
        let key = (nv_quant::stream_cache_key(&stream), m_padded);
        let mut staged_map = if skip_pad {
            let mut map = s
                .a_staging
                .lock()
                .map_err(|e| anyhow::anyhow!("NVFP4 staging mutex poisoned: {e}"))?;
            if !stream_is_capturing(&stream) {
                map.retain(|k, v| v.epoch == nv_quant::stream_epoch(k.0));
                if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(key) {
                    let a = stream
                        .alloc_zeros::<u8>(packed_bytes)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    let sc = stream
                        .alloc_zeros::<u8>(scales_bytes)
                        .map_err(|e| anyhow::anyhow!(e))?;
                    entry.insert(StagedActivations {
                        epoch: nv_quant::stream_epoch(key.0),
                        hwm_rows: 0,
                        packed: a,
                        scales: sc,
                    });
                }
            }
            if map.contains_key(&key) {
                if let Some(entry) = map.get_mut(&key) {
                    if m_logical < entry.hwm_rows {
                        stream
                            .memset_zeros(&mut entry.packed)
                            .map_err(|e| anyhow::anyhow!(e))?;
                        stream
                            .memset_zeros(&mut entry.scales)
                            .map_err(|e| anyhow::anyhow!(e))?;
                    }
                    entry.hwm_rows = m_logical;
                }
                Some(map)
            } else {
                None
            }
        } else {
            None
        };

        let mut a_local: Option<(cudarc::driver::CudaSlice<u8>, cudarc::driver::CudaSlice<u8>)> =
            None;
        if staged_map.is_none() {
            let a = unsafe {
                stream
                    .alloc::<u8>(packed_bytes)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            let sc = unsafe {
                stream
                    .alloc::<u8>(scales_bytes)
                    .map_err(|e| anyhow::anyhow!(e))?
            };
            a_local = Some((a, sc));
        }

        {
            let rows_only = staged_map.is_some();
            let (a_data, a_scales) = match staged_map.as_mut() {
                Some(map) => {
                    let entry = map.get_mut(&key).expect("inserted above");
                    (&mut entry.packed, &mut entry.scales)
                }
                None => {
                    let (a, sc) = a_local.as_mut().expect("allocated above");
                    (a, sc)
                }
            };
            {
                let (x_storage, xl) = x_bf.storage_and_layout();
                let x_cuda = match &*x_storage {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("expected cuda storage for input"),
                };
                let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
                let x_view = x_slice.slice(xl.start_offset()..);
                let (x_ptr, _gx) = x_view.device_ptr(&stream);
                let (a_ptr, _ga) = a_data.device_ptr_mut(&stream);
                let (s_ptr, _gs) = a_scales.device_ptr_mut(&stream);
                let rc = if let Some(up) = silu_up {
                    let (up_storage, ul) = up.storage_and_layout();
                    let candle_core::Storage::Cuda(up_cuda) = &*up_storage else {
                        anyhow::bail!("NVFP4 SiLU up requires CUDA storage");
                    };
                    let up_slice = up_cuda.as_cuda_slice::<bf16>()?;
                    let up_view =
                        up_slice.slice(ul.start_offset()..ul.start_offset() + up.elem_count());
                    let (up_ptr, _gu) = up_view.device_ptr(&stream);
                    unsafe {
                        nv_kernels::cuda::quantize_nvfp4_silu_candle_bf16(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            up_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            (if cfx_route {
                                m_padded.div_ceil(96) * 96
                            } else {
                                m_padded
                            }) as i32,
                            m_logical as i32,
                            self.in_features as i32,
                            if cfx_route { 96 } else { 128 },
                        )
                    }
                } else if cfx_route {
                    unsafe {
                        nv_kernels::cuda::quantize_nvfp4_bf16_sf96(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                } else if let Some(pn) = prenorm.as_ref() {
                    let w_c = pn.weight_bf16.clone();
                    let (w_storage, wl) = w_c.storage_and_layout();
                    let w_cuda = match &*w_storage {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("expected cuda storage for norm weight"),
                    };
                    let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
                    let w_view = w_slice.slice(wl.start_offset()..);
                    let (w_ptr, _gw) = w_view.device_ptr(&stream);
                    let m_padded_arg = if rows_only { m_logical } else { m_padded };
                    unsafe {
                        nv_kernels::cuda::rmsnorm_quantize_nvfp4_bf16(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            w_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            pn.eps,
                            m_padded_arg as i32,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                } else if rows_only {
                    unsafe {
                        nv_kernels::cuda::quantize_nvfp4_bf16_rows(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                } else {
                    unsafe {
                        nv_kernels::cuda::quantize_nvfp4_bf16(
                            stream.cu_stream() as *mut c_void,
                            x_ptr as *const u16,
                            a_ptr as *mut u8,
                            s_ptr as *mut u8,
                            s.input_stored_global,
                            m_padded as i32,
                            m_logical as i32,
                            self.in_features as i32,
                        )
                    }
                };
                if rc != 0 {
                    anyhow::bail!("quantize_nvfp4_bf16 rc={rc}");
                }
            }

            if cfx_route {
                static CFX_FIRINGS: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let prev = CFX_FIRINGS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if prev == 0 && std::env::var("NV_Q38_CFX_TRACE").is_ok() {
                    eprintln!(
                        "[linear] cfx route ENGAGED first fire m={m_padded} n={} k={}",
                        self.out_features, self.in_features
                    );
                }
                let (ap, _ga) = a_data.device_ptr(&stream);
                let (asp, _gas) = a_scales.device_ptr(&stream);
                let (wp, _gw) = s.weight_u8.device_ptr(&stream);
                let (wsp, _gws) = s.weight_scales_cm.device_ptr(&stream);
                let (alp, _gal) = s.weight_alpha_dev.device_ptr(&stream);
                let (dp, _gd) = d_padded.device_ptr_mut(&stream);
                let rc = unsafe {
                    nv_kernels::cuda::nvfp4_cfx_gemm_bf16(
                        stream.cu_stream() as *mut c_void,
                        ap as *const c_void,
                        asp as *const c_void,
                        wp as *const c_void,
                        wsp as *const c_void,
                        alp as *const f32,
                        dp as *mut c_void,
                        m_padded as i32,
                        self.out_features as i32,
                        self.in_features as i32,
                    )
                };
                if rc != 0 {
                    anyhow::bail!("nvfp4_cfx_gemm_bf16 rc={rc}");
                }
            } else {
                let mut runner = s
                    .runner
                    .lock()
                    .map_err(|e| anyhow::anyhow!("NVFP4 runner mutex poisoned: {e}"))?;

                runner.set_stream(stream.clone());
                runner.matmul_scaled_alpha_dev(
                    a_data,
                    a_scales,
                    &s.weight_u8,
                    &s.weight_scales_cm,
                    &mut d_padded,
                    m_padded as u64,
                    n,
                    k,
                    &s.weight_alpha_dev,
                    s.weight_alpha,
                )?;
            }
        }

        let d_storage = candle_core::CudaStorage::wrap_cuda_slice(d_padded, dev);
        let storage = candle_core::Storage::Cuda(d_storage);
        let full = candle_core::Tensor::from_storage(
            storage,
            (m_padded, self.out_features),
            candle_core::op::BackpropOp::none(),
            false,
        );
        if m_padded == m_logical {
            Ok(full)
        } else {
            Ok(full.narrow(0, 0, m_logical)?.contiguous()?)
        }
    }
}

#[cfg(feature = "cuda")]
impl Linear {
    #[doc(hidden)]
    #[cfg(feature = "cuda")]
    pub fn nvfp4_staging_diag(&self, stream_key: usize, m_padded: usize) -> Option<(u64, usize)> {
        match &self.storage {
            LinearStorage::Nvfp4(s) => s.a_staging.lock().ok().and_then(|m| {
                m.get(&(stream_key, m_padded))
                    .map(|e| (e.epoch, e.hwm_rows))
            }),
            _ => None,
        }
    }

    pub fn new_fp8_e4m3(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale: f32,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::new_fp8_e4m3_row_scales(
            weight_u8,
            vec![weight_scale; out_features],
            in_features,
            out_features,
            bias,
            device,
            runner,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fp8_e4m3_row_scales(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale_rows: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::new_fp8_e4m3_row_scales_in_mode(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            bias,
            device,
            runner,
            fp8_scale_mode(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fp8_e4m3_row_scales_in_mode(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale_rows: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
        scale_mode: nv_quant::fp8::Fp8ScaleMode,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires a CUDA device"),
        };
        if scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow {
            let ordinal = dev.cuda_stream().context().ordinal();
            if let Err(detail) = fp8_per_row_probe(&runner, ordinal) {
                anyhow::bail!(
                    "fp8 per-row (OUTER_VEC_32F) scaling is not served by cuBLASLt on this \
                     device, so the default fp8 scale mode cannot run a single GEMM -- \
                     refusing at load instead of failing on the first request. Probe: \
                     {detail}. Set NV_FP8_SCALE_MODE=tensor to fall back to per-tensor fp8 \
                     scales, or drop the fp8 opt-in (e.g. unset NV_ATTN_PROJ_QUANT)."
                );
            }
        }
        Self::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            bias,
            device,
            runner,
            scale_mode,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scale_rows: Vec<f32>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
        scale_mode: nv_quant::fp8::Fp8ScaleMode,
    ) -> Result<Self> {
        if weight_scale_rows.len() != out_features {
            anyhow::bail!(
                "Fp8 Linear: {} weight scales for {out_features} output rows",
                weight_scale_rows.len()
            );
        }
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let a_scale_dev = stream
            .clone_htod(&[1.0f32])
            .map_err(|e| anyhow::anyhow!(e))?;

        let b_scale_scalar = weight_scale_rows.iter().copied().fold(0f32, f32::max);
        let b_scale_scalar = if b_scale_scalar > 0.0 {
            b_scale_scalar
        } else {
            1.0
        };
        #[allow(deprecated)]
        let b_scale_dev = stream
            .clone_htod(&[b_scale_scalar])
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let b_scale_rows_dev = stream
            .clone_htod(&weight_scale_rows)
            .map_err(|e| anyhow::anyhow!(e))?;
        let storage = LinearStorage::Fp8E4m3(Arc::new(Fp8Storage {
            weight_u8,
            a_scale_dev,
            b_scale_dev,
            b_scale_rows_dev,
            weight_scale_rows,
            scale_mode,
            cuda_device: dev,
            runner,
            marlin_fp8_twin_built_once_from_the_resident_e4m3_rows: std::sync::OnceLock::new(),
            dx_fp8_twin_of_the_transposed_weight_bytes_with_the_channel_scale_folded_into_dy:
                std::sync::OnceLock::new(),
        }));
        if marlin_fp8_rows_opt_in() && out_features % 64 == 0 && in_features % 16 == 0 {
            if let LinearStorage::Fp8E4m3(st) = &storage {
                let _ = st
                    .marlin_fp8_twin_built_once_from_the_resident_e4m3_rows
                    .get_or_init(|| match build_marlin_fp8_twin(st, out_features, in_features) {
                        Ok(t) => Some(Arc::new(t)),
                        Err(e) => {
                            eprintln!("[marlin_fp8_twin] unavailable n={out_features} k={in_features}: {e:#}");
                            None
                        }
                    });
            }
        }
        Ok(Self {
            kind: LinearKind::Fp8E4m3 {
                a_scale: 1.0,
                b_scale: b_scale_scalar,
            },
            storage,
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn from_bf16_quantized_fp8(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::from_bf16_quantized_fp8_with_scales(weight, None, bias, device, runner)
    }

    pub fn from_bf16_quantized_fp8_with_scales(
        weight: &Tensor,
        checkpoint_scale_rows: Option<&[f32]>,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        Self::from_bf16_quantized_fp8_in_mode(
            weight,
            checkpoint_scale_rows,
            bias,
            device,
            runner,
            fp8_scale_mode(),
        )
    }

    pub fn from_bf16_quantized_fp8_in_mode(
        weight: &Tensor,
        checkpoint_scale_rows: Option<&[f32]>,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
        scale_mode: nv_quant::fp8::Fp8ScaleMode,
    ) -> Result<Self> {
        use half::bf16;

        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weight_bf = weight.to_dtype(DType::BF16)?.contiguous()?;
        let weight_host: Vec<bf16> = weight_bf.flatten_all()?.to_vec1::<bf16>()?;
        let (weight_bytes, weight_scale_rows) = fp8_weight_payload(
            &weight_host,
            out_features,
            in_features,
            checkpoint_scale_rows,
            scale_mode,
        )?;
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Fp8 Linear requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let weight_u8 = stream
            .clone_htod(&weight_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        Self::new_fp8_e4m3_row_scales_in_mode(
            weight_u8,
            weight_scale_rows,
            in_features,
            out_features,
            bias,
            device,
            runner,
            scale_mode,
        )
    }

    pub fn from_quantized_weight_fp8(
        qw: &nv_weights::QuantizedWeight,
        dequantized: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
    ) -> Result<Self> {
        if qw.scheme != nv_weights::QuantScheme::Fp8E4m3 {
            anyhow::bail!(
                "from_quantized_weight_fp8: checkpoint scheme is {:?}, not Fp8E4m3",
                qw.scheme
            );
        }
        if qw.shape != dequantized.dims() {
            anyhow::bail!(
                "from_quantized_weight_fp8: checkpoint shape {:?} != dequantized {:?}",
                qw.shape,
                dequantized.dims()
            );
        }
        let rows = qw.fp8_weight_scale_rows()?;
        Self::from_bf16_quantized_fp8_with_scales(
            dequantized,
            rows.as_deref(),
            bias,
            device,
            runner,
        )
    }

    pub fn new_nvfp4(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scales_cm: cudarc::driver::CudaSlice<u8>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Result<Self> {
        Self::new_nvfp4_scaled(
            weight_u8,
            weight_scales_cm,
            in_features,
            out_features,
            bias,
            device,
            runner,
            1.0,
            1.0,
        )
    }

    pub fn new_nvfp4_scaled(
        weight_u8: cudarc::driver::CudaSlice<u8>,
        weight_scales_cm: cudarc::driver::CudaSlice<u8>,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        weight_alpha: f32,
        input_stored_global: f32,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires a CUDA device"),
        };

        let stream = dev.cuda_stream();
        let mut weight_alpha_dev = stream
            .alloc_zeros::<f32>(1)
            .map_err(|e| anyhow::anyhow!("alloc weight_alpha_dev: {e:?}"))?;
        stream
            .memcpy_htod(&[weight_alpha], &mut weight_alpha_dev)
            .map_err(|e| anyhow::anyhow!("htod weight_alpha_dev: {e:?}"))?;
        let storage = LinearStorage::Nvfp4(Arc::new(Nvfp4Storage {
            weight_u8,
            weight_scales_cm,
            cuda_device: dev,
            runner,
            weight_alpha,
            weight_alpha_dev,
            input_stored_global,
            bf16act_row_limit: 0,
            a_staging: Mutex::new(std::collections::HashMap::new()),
            marlin_w4a16_twin_built_once_from_the_resident_nvfp4_buffers: std::sync::OnceLock::new(
            ),
            dx_marlin_twin_of_the_transposed_weight_requantized_once_to_nvfp4_so_the_input_gradient_never_touches_a_bf16_copy:
                std::sync::OnceLock::new(),
        }));
        Ok(Self {
            kind: LinearKind::Nvfp4,
            storage,
            bias,
            in_features,
            out_features,
            lora: std::sync::RwLock::new(None),
        })
    }

    pub fn from_bf16_quantized_nvfp4_dev(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Result<Self> {
        Self::from_bf16_quantized_nvfp4_dev_with_stored_input_global(
            weight, bias, device, runner, 1.0,
        )
    }

    pub fn from_bf16_quantized_nvfp4_dev_with_stored_input_global(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        stored_input_global: f32,
    ) -> Result<Self> {
        use cudarc::driver::{DevicePtr, DevicePtrMut};
        use half::bf16;
        use nv_quant::nvfp4::{BLOCK_SIZE, MIN_TILE};
        use std::ffi::c_void;

        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        if !in_features.is_multiple_of(BLOCK_SIZE) {
            anyhow::bail!(
                "nvfp4 dev quant: in_features {} must be a multiple of {}",
                in_features,
                BLOCK_SIZE
            );
        }
        if out_features < MIN_TILE || in_features < MIN_TILE {
            anyhow::bail!(
                "nvfp4 dev quant: dims [{out_features}, {in_features}] below min tile {MIN_TILE}"
            );
        }
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires a CUDA device"),
        };
        let weight_bf = weight
            .to_device(device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let amax = weight_bf
            .abs()?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        let stored_weight_global = if amax.is_finite() && amax > 0.0 {
            (448.0f32 * 6.0) / amax
        } else {
            1.0
        };
        let safe_ig = if stored_input_global.is_finite() && stored_input_global != 0.0 {
            stored_input_global
        } else {
            1.0
        };
        let weight_alpha = if stored_weight_global.is_finite() && stored_weight_global != 0.0 {
            1.0 / stored_weight_global / safe_ig
        } else {
            1.0 / safe_ig
        };

        let stream = crate::cuda_stream::current_stream(&dev);
        let blocks_per_row = in_features / BLOCK_SIZE;
        let sf_rows = out_features.div_ceil(128) * 128;
        let scales_bytes = sf_rows * blocks_per_row.div_ceil(4) * 4;
        let mut w_packed: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(out_features * in_features / 2)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut w_scales: cudarc::driver::CudaSlice<u8> = stream
            .alloc_zeros::<u8>(scales_bytes)
            .map_err(|e| anyhow::anyhow!(e))?;
        {
            let (w_storage, wl) = weight_bf.storage_and_layout();
            let w_cuda = match &*w_storage {
                candle_core::Storage::Cuda(s) => s,
                _ => anyhow::bail!("expected cuda storage for weight"),
            };
            let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
            let w_view = w_slice.slice(wl.start_offset()..);
            let (wp, _gw) = w_view.device_ptr(&stream);
            let (pp, _gp) = w_packed.device_ptr_mut(&stream);
            let (sp, _gs) = w_scales.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::quantize_nvfp4_bf16(
                    stream.cu_stream() as *mut c_void,
                    wp as *const u16,
                    pp as *mut u8,
                    sp as *mut u8,
                    stored_weight_global,
                    out_features as i32,
                    out_features as i32,
                    in_features as i32,
                )
            };
            if rc != 0 {
                anyhow::bail!("quantize_nvfp4_bf16 (weight) rc={rc}");
            }
        }
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        Self::new_nvfp4_scaled(
            w_packed,
            w_scales,
            in_features,
            out_features,
            bias,
            device,
            runner,
            weight_alpha,
            safe_ig,
        )
    }

    pub fn from_bf16_quantized_nvfp4(
        weight: &Tensor,
        bias: Option<Tensor>,
        device: &Device,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
    ) -> Result<Self> {
        use half::bf16;
        use nv_quant::nvfp4::Nvfp4Tensor;

        let dims = weight.dims();
        if dims.len() != 2 {
            anyhow::bail!("Linear weight must be 2-D, got rank {}", dims.len());
        }
        let out_features = dims[0];
        let in_features = dims[1];
        let weight_bf = weight.to_dtype(DType::BF16)?.contiguous()?;
        let host_bf: Vec<bf16> = weight_bf.flatten_all()?.to_vec1::<bf16>()?;
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(out_features);
        for r in 0..out_features {
            let mut row = Vec::with_capacity(in_features);
            for j in 0..in_features {
                row.push(host_bf[r * in_features + j].to_f32());
            }
            rows.push(row);
        }
        let q = Nvfp4Tensor::quantize_rows(&rows);
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("NVFP4 Linear requires a CUDA device"),
        };
        let stream = crate::cuda_stream::current_stream(&dev);
        #[allow(deprecated)]
        let weight_u8 = stream.clone_htod(&q.data).map_err(|e| anyhow::anyhow!(e))?;
        let weight_scales_sw = q.scales_swizzled();
        #[allow(deprecated)]
        let weight_scales = stream
            .clone_htod(&weight_scales_sw)
            .map_err(|e| anyhow::anyhow!(e))?;
        Self::new_nvfp4(
            weight_u8,
            weight_scales,
            in_features,
            out_features,
            bias,
            device,
            runner,
        )
    }

    pub fn nvfp4_draft_twin_rows_quantized_from_this_resident_copy_which_stays_untouched(
        &self,
        runner: Arc<Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>,
        row0: usize,
        rows: usize,
    ) -> Result<Linear> {
        anyhow::ensure!(
            rows >= 1 && row0 + rows <= self.out_features,
            "nvfp4 draft twin: rows [{row0}, {row0}+{rows}) out of range for out_features {}",
            self.out_features
        );
        match &self.storage {
            LinearStorage::Bf16 { weight, .. } => {
                let device = weight.device().clone();
                let w = weight.narrow(0, row0, rows)?.contiguous()?;
                Self::from_bf16_quantized_nvfp4_dev(&w, None, &device, runner)
            }
            LinearStorage::Fp8E4m3(s) => {
                anyhow::ensure!(
                    s.scale_mode == nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
                    "nvfp4 draft twin: fp8 source must carry per-row scales, got {:?}",
                    s.scale_mode
                );
                let k = self.in_features;
                let stream = crate::cuda_stream::current_stream(&s.cuda_device);
                let bytes_view = s.weight_u8.slice(row0 * k..(row0 + rows) * k);
                #[allow(deprecated)]
                let bytes = stream
                    .memcpy_dtov(&bytes_view)
                    .map_err(|e| anyhow::anyhow!("dtoh fp8 rows for nvfp4 draft twin: {e}"))?;
                let host = fp8_rowscale_dequant_bf16_host(
                    &bytes,
                    &s.weight_scale_rows[row0..row0 + rows],
                    rows,
                    k,
                );
                drop(bytes);
                let device = Device::Cuda(s.cuda_device.clone());
                let w = Tensor::from_vec(host, (rows, k), &device)?;
                Self::from_bf16_quantized_nvfp4_dev(&w, None, &device, runner)
            }
            LinearStorage::Nvfp4(_) => anyhow::bail!(
                "nvfp4 draft twin: source linear is already NVFP4-resident; use it directly"
            ),
            LinearStorage::GgmlQuant(_) => anyhow::bail!(
                "nvfp4 draft twin: ggml-quant storage carries no bf16/fp8 source rows to \
                 requantize"
            ),
        }
    }
}

pub fn quantize_fp8_per_tensor(values: &[half::bf16]) -> (Vec<u8>, f32) {
    nv_quant::fp8::quantize_e4m3_per_tensor(values)
}

pub fn fp8_rowscale_dequant_bf16_host(
    bytes: &[u8],
    row_scales: &[f32],
    out_features: usize,
    in_features: usize,
) -> Vec<half::bf16> {
    let mut host = vec![half::bf16::ZERO; out_features * in_features];
    let jobs = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(1, out_features.max(1));
    let rows_per_job = out_features.div_ceil(jobs);
    std::thread::scope(|s| {
        for (ci, dst_chunk) in host.chunks_mut(rows_per_job * in_features).enumerate() {
            let r0 = ci * rows_per_job;
            let src_chunk = &bytes[r0 * in_features..r0 * in_features + dst_chunk.len()];
            let chunk_scales = &row_scales[r0..r0 + dst_chunk.len() / in_features];
            s.spawn(move || {
                for (r, dst_row) in dst_chunk.chunks_mut(in_features).enumerate() {
                    let scale = chunk_scales[r];
                    let src_row = &src_chunk[r * in_features..(r + 1) * in_features];
                    for (dst, b) in dst_row.iter_mut().zip(src_row) {
                        *dst =
                            half::bf16::from_f32(f32::from(float8::F8E4M3::from_bits(*b)) * scale);
                    }
                }
            });
        }
    });
    host
}

pub fn checkpoint_module_is_fp8_e4m3_weight_with_scale(
    weights: &nv_weights::WeightLoader,
    module: &str,
) -> bool {
    weights.has(&format!("{module}.weight_scale"))
        && weights.st_dtype_of(&format!("{module}.weight")) == Some(nv_weights::StDtype::F8_E4M3)
}

pub fn fp8_e4m3_rowscale_checkpoint_dequant_linear(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    dtype: DType,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let scale_name = format!("{module}.weight_scale");
    let st = weights.st_dtype_of(&weight_name);
    anyhow::ensure!(
        st == Some(nv_weights::StDtype::F8_E4M3),
        "{weight_name}: fp8 dequant arm expects F8_E4M3 storage next to {scale_name}, got {st:?}"
    );
    let shape = weights
        .shape_of(&weight_name)
        .ok_or_else(|| anyhow::anyhow!("missing shape for {weight_name}"))?;
    anyhow::ensure!(
        shape == [out_features, in_features],
        "{weight_name}: shape {shape:?} != [{out_features}, {in_features}]"
    );
    let bytes = weights
        .raw_bytes(&weight_name)
        .map_err(|e| anyhow::anyhow!("read {weight_name}: {e}"))?;
    anyhow::ensure!(
        bytes.len() == out_features * in_features,
        "{weight_name}: {} bytes != {}",
        bytes.len(),
        out_features * in_features
    );
    let scale_t = weights
        .get(&scale_name, DType::F32)
        .map_err(|e| anyhow::anyhow!("load {scale_name}: {e}"))?;
    let scale_dims = scale_t.dims().to_vec();
    let scale_vals: Vec<f32> = scale_t.flatten_all()?.to_vec1()?;
    let row_scales = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, out_features)
        .map_err(|e| anyhow::anyhow!("{scale_name} shape {scale_dims:?}: {e}"))?;
    let host = fp8_rowscale_dequant_bf16_host(bytes, &row_scales, out_features, in_features);
    let w = Tensor::from_vec(host, (out_features, in_features), weights.device())?;
    let w = if dtype == DType::BF16 {
        w
    } else {
        w.to_dtype(dtype)?
    };
    Linear::new(w, None)
}

#[cfg(feature = "cuda")]
pub const FP8_RESIDENT_WEIGHT_BYTES_PER_PARAM_REPLACING_4_FROM_BF16_DEQUANT_PLUS_PRETRANSPOSE:
    usize = 1;

#[cfg(feature = "cuda")]
#[cfg(feature = "cuda")]
pub fn fp8_e4m3_rowscale_checkpoint_resident_linear(
    weights: &nv_weights::WeightLoader,
    module: &str,
    out_features: usize,
    in_features: usize,
    device: &Device,
    runner: Arc<Mutex<nv_quant::fp8::Fp8GemmRunner>>,
) -> Result<Linear> {
    let weight_name = format!("{module}.weight");
    let scale_name = format!("{module}.weight_scale");
    let st = weights.st_dtype_of(&weight_name);
    anyhow::ensure!(
        st == Some(nv_weights::StDtype::F8_E4M3),
        "{weight_name}: fp8 resident arm expects F8_E4M3 storage next to {scale_name}, got {st:?}"
    );
    let shape = weights
        .shape_of(&weight_name)
        .ok_or_else(|| anyhow::anyhow!("missing shape for {weight_name}"))?;
    anyhow::ensure!(
        shape == [out_features, in_features],
        "{weight_name}: shape {shape:?} != [{out_features}, {in_features}]"
    );
    let bytes = weights
        .raw_bytes(&weight_name)
        .map_err(|e| anyhow::anyhow!("read {weight_name}: {e}"))?;
    anyhow::ensure!(
        bytes.len()
            == out_features
                * in_features
                * FP8_RESIDENT_WEIGHT_BYTES_PER_PARAM_REPLACING_4_FROM_BF16_DEQUANT_PLUS_PRETRANSPOSE,
        "{weight_name}: {} bytes != {}",
        bytes.len(),
        out_features * in_features
    );
    let scale_t = weights
        .get(&scale_name, DType::F32)
        .map_err(|e| anyhow::anyhow!("load {scale_name}: {e}"))?;
    let scale_dims = scale_t.dims().to_vec();
    let scale_vals: Vec<f32> = scale_t.flatten_all()?.to_vec1()?;
    let row_scales = nv_weights::fp8_row_scales_from(&scale_dims, &scale_vals, out_features)
        .map_err(|e| anyhow::anyhow!("{scale_name} shape {scale_dims:?}: {e}"))?;
    let dev = match device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("fp8 resident linear requires a CUDA device"),
    };
    let stream = crate::cuda_stream::current_stream(&dev);
    #[allow(deprecated)]
    let weight_u8 = stream.clone_htod(bytes).map_err(|e| anyhow::anyhow!(e))?;
    Linear::new_fp8_e4m3_row_scales_without_the_cublaslt_probe(
        weight_u8,
        row_scales,
        in_features,
        out_features,
        None,
        device,
        runner,
        nv_quant::fp8::Fp8ScaleMode::PerOuterRow,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    mod qwen38_e4m3_batch_scope {
        use super::*;

        #[test]
        fn unset_policy_is_batch_scoped_and_nested_guards_restore_the_previous_route() {
            let policy = E4m3GroupPolicy::from_env_value(None);
            assert_eq!(policy.rows_per_warp(), None);
            for batch_size in 0..=16 {
                let _batch = Qwen38BatchE4m3GroupScopeGuard::enter(batch_size);
                let expected = matches!(batch_size, 6 | 8).then_some(2);
                assert_eq!(policy.rows_per_warp(), expected, "batch_size={batch_size}");
                for nested_size in 0..=16 {
                    let _nested = Qwen38BatchE4m3GroupScopeGuard::enter(nested_size);
                    assert_eq!(
                        policy.rows_per_warp(),
                        matches!(nested_size, 6 | 8).then_some(2),
                        "batch_size={batch_size} nested_size={nested_size}"
                    );
                }
                assert_eq!(
                    policy.rows_per_warp(),
                    expected,
                    "outer batch scope restored"
                );
            }
            assert_eq!(policy.rows_per_warp(), None);
        }

        #[test]
        fn explicit_values_keep_their_routes_inside_and_outside_the_batch_scope() {
            let cases = [
                ("0", None),
                ("", None),
                ("invalid", None),
                ("3", None),
                ("1", Some(2)),
                ("01", Some(1)),
                ("2", Some(2)),
                ("4", Some(4)),
            ];
            for batch_size in [0, 1, 4, 6, 8, 10] {
                let _scope = Qwen38BatchE4m3GroupScopeGuard::enter(batch_size);
                for (raw, expected) in cases {
                    assert_eq!(
                        E4m3GroupPolicy::from_env_value(Some(raw)).rows_per_warp(),
                        expected,
                        "explicit {raw:?} must retain its route when batch_size={batch_size}"
                    );
                }
            }
        }

        #[test]
        fn batch_scope_does_not_escape_to_another_thread_or_survive_unwinding() {
            let policy = E4m3GroupPolicy::from_env_value(None);
            let unwound = std::panic::catch_unwind(|| {
                let _batch = Qwen38BatchE4m3GroupScopeGuard::enter(6);
                assert_eq!(policy.rows_per_warp(), Some(2));
                assert_eq!(
                    std::thread::spawn(move || policy.rows_per_warp())
                        .join()
                        .unwrap(),
                    None
                );
                panic!("exercise scope cleanup after a failed batch forward");
            });
            assert!(unwound.is_err());
            assert_eq!(policy.rows_per_warp(), None);
        }
    }
    use candle_core::Device;
    use half::bf16;

    #[test]
    fn small_m_det_ships_default_on_and_zero_disables() {
        assert!(small_m_det_from(None));
        assert!(!small_m_det_from(Some("0")));
        assert!(small_m_det_from(Some("1")));
        assert!(small_m_det_from(Some("")));
    }

    #[test]
    fn fp8_ships_per_row_scales_and_only_an_explicit_env_coarsens_them() {
        use nv_quant::fp8::Fp8ScaleMode;
        assert_eq!(fp8_scale_mode_from(None), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("")), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("row")), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("1")), Fp8ScaleMode::PerOuterRow);
        assert_eq!(fp8_scale_mode_from(Some("tensor")), Fp8ScaleMode::PerTensor);
        assert_eq!(fp8_scale_mode_from(Some("TENSOR")), Fp8ScaleMode::PerTensor);
        assert_eq!(
            fp8_scale_mode_from(Some("per_tensor")),
            Fp8ScaleMode::PerTensor
        );
        assert_eq!(
            fp8_scale_mode_from(Some("per-tensor")),
            Fp8ScaleMode::PerTensor
        );
        assert_eq!(fp8_scale_mode_from(Some("0")), Fp8ScaleMode::PerTensor);
    }

    #[test]
    fn fp8_weight_payload_defaults_to_one_scale_per_output_row() {
        use nv_quant::fp8::{dequantize_e4m3_per_row, Fp8ScaleMode};
        let (out_f, in_f) = (6usize, 32usize);
        let host: Vec<bf16> = (0..out_f * in_f)
            .map(|i| {
                let r = i / in_f;
                let c = i % in_f;
                bf16::from_f32(2f32.powi(-(4 * r as i32)) * ((c as f32) * 0.31).cos())
            })
            .collect();
        let want: Vec<f32> = host.iter().map(|v| v.to_f32()).collect();

        let (row_bytes, row_scales) =
            fp8_weight_payload(&host, out_f, in_f, None, Fp8ScaleMode::PerOuterRow).unwrap();
        assert_eq!(row_scales.len(), out_f);
        assert!(
            row_scales.windows(2).all(|w| w[0] > w[1]),
            "each quieter row must get its own smaller scale: {row_scales:?}"
        );

        let (tensor_bytes, tensor_scales) =
            fp8_weight_payload(&host, out_f, in_f, None, Fp8ScaleMode::PerTensor).unwrap();
        assert!(tensor_scales.windows(2).all(|w| w[0] == w[1]));

        let row_back = dequantize_e4m3_per_row(&row_bytes, out_f, in_f, &row_scales).unwrap();
        let tensor_back =
            dequantize_e4m3_per_row(&tensor_bytes, out_f, in_f, &tensor_scales).unwrap();
        let quiet = (out_f - 1) * in_f;
        let err = |got: &[f32]| -> f32 {
            got[quiet..]
                .iter()
                .zip(&want[quiet..])
                .map(|(g, w)| (g - w).abs() / w.abs().max(1e-12))
                .fold(0f32, f32::max)
        };
        let row_err = err(&row_back);
        let tensor_err = err(&tensor_back);
        assert!(row_err < 0.10, "per-row err {row_err} on the quiet row");
        assert!(
            tensor_err > 0.5,
            "per-tensor err {tensor_err} should destroy the quiet row"
        );
    }

    #[test]
    fn fp8_weight_payload_uses_checkpoint_scales_instead_of_recomputing() {
        use nv_quant::fp8::Fp8ScaleMode;
        let (out_f, in_f) = (3usize, 16usize);
        let host: Vec<bf16> = (0..out_f * in_f)
            .map(|i| bf16::from_f32(((i as f32) * 0.17 + 0.3).sin()))
            .collect();
        let (_auto_bytes, auto_scales) =
            fp8_weight_payload(&host, out_f, in_f, None, Fp8ScaleMode::PerOuterRow).unwrap();
        let ckpt: Vec<f32> = auto_scales.iter().map(|s| s * 1.5).collect();
        let (_bytes, used) =
            fp8_weight_payload(&host, out_f, in_f, Some(&ckpt), Fp8ScaleMode::PerOuterRow).unwrap();
        assert_eq!(used, ckpt, "checkpoint weight_scale must survive verbatim");

        assert!(fp8_weight_payload(
            &host,
            out_f,
            in_f,
            Some(&ckpt[..2]),
            Fp8ScaleMode::PerOuterRow
        )
        .is_err());

        assert!(
            fp8_weight_payload(&host, out_f, in_f, Some(&ckpt), Fp8ScaleMode::PerTensor).is_err()
        );
        let uniform = vec![auto_scales[0]; out_f];
        assert!(
            fp8_weight_payload(&host, out_f, in_f, Some(&uniform), Fp8ScaleMode::PerTensor).is_ok()
        );
    }

    #[test]
    fn fp8_weight_payload_rejects_a_length_mismatch() {
        use nv_quant::fp8::Fp8ScaleMode;
        let host: Vec<bf16> = vec![bf16::from_f32(1.0); 12];
        assert!(fp8_weight_payload(&host, 3, 5, None, Fp8ScaleMode::PerOuterRow).is_err());
        assert!(fp8_weight_payload(&host, 3, 4, None, Fp8ScaleMode::PerOuterRow).is_ok());
    }

    #[test]
    fn cpu_bf16_matmul_via_linear_forward() {
        let dev = Device::Cpu;

        let in_features = 4usize;
        let out_features = 3usize;
        let w_f32: Vec<f32> = vec![
            0.5, -1.0, 0.25, 0.75, -0.5, 1.5, -0.25, 0.0, 1.0, 0.0, -1.0, 0.5,
        ];
        let w_bf: Vec<bf16> = w_f32.iter().copied().map(bf16::from_f32).collect();
        let weight = Tensor::from_vec(w_bf, (out_features, in_features), &dev).unwrap();
        let linear = Linear::new(weight, None).expect("build bf16 Linear");

        let x_f32: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.0, 0.5, 0.5];
        let x_bf: Vec<bf16> = x_f32.iter().copied().map(bf16::from_f32).collect();
        let x = Tensor::from_vec(x_bf, (2, in_features), &dev).unwrap();

        let out = linear.forward(&x).expect("forward");

        assert_eq!(out.dims(), &[2, out_features]);

        assert_eq!(out.dtype(), DType::BF16);

        let mut expected = vec![0f32; 2 * out_features];
        for r in 0..2 {
            for o in 0..out_features {
                let mut s = 0f32;
                for k in 0..in_features {
                    s += x_f32[r * in_features + k] * w_f32[o * in_features + k];
                }
                expected[r * out_features + o] = s;
            }
        }
        let got_bf: Vec<bf16> = out
            .to_vec2::<bf16>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let got_f32: Vec<f32> = got_bf.iter().map(|x| x.to_f32()).collect();
        for (i, (g, e)) in got_f32.iter().zip(expected.iter()).enumerate() {
            let denom = e.abs().max(1e-3);
            let rel = (g - e).abs() / denom;
            assert!(
                rel < 1e-2,
                "element {i}: got {g}, expected {e}, rel err {rel}"
            );
        }
    }

    #[test]
    fn cpu_forward_honors_input_start_offset() {
        let dev = Device::Cpu;
        let in_features = 4usize;
        let out_features = 3usize;
        let w_bf: Vec<bf16> = (0..out_features * in_features)
            .map(|i| bf16::from_f32(((i as f32) * 0.13).sin()))
            .collect();
        let weight = Tensor::from_vec(w_bf, (out_features, in_features), &dev).unwrap();
        let linear = Linear::new(weight, None).unwrap();

        let rows: Vec<bf16> = (0..2 * in_features)
            .map(|i| bf16::from_f32(((i as f32) * 0.11).cos()))
            .collect();
        let mut full: Vec<bf16> = (0..in_features)
            .map(|i| bf16::from_f32(900.0 + i as f32))
            .collect();
        full.extend_from_slice(&rows);
        let full_t = Tensor::from_vec(full, (3, in_features), &dev).unwrap();
        let view = full_t.narrow(0, 1, 2).unwrap();
        let ref_t = Tensor::from_vec(rows, (2, in_features), &dev).unwrap();

        let y_view = linear.forward(&view).unwrap();
        let y_ref = linear.forward(&ref_t).unwrap();
        let a: Vec<bf16> = y_view
            .to_vec2::<bf16>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        let b: Vec<bf16> = y_ref
            .to_vec2::<bf16>()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        );
    }
}
