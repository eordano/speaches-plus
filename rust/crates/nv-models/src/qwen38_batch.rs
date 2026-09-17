use super::*;
use crate::gemma4_batch_graph::BucketPlan;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;
use nv_kernels::graph::CudaGraphRunner;

pub fn nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_BATCH")
}

pub const Q38_BATCH_GRAPH_SHAPE_TOKEN_BASE_KEEPS_SOLO_KEY_1_AND_VERIFY_BASE_0X51000_UNTOUCHED: u64 =
    0x38b000;

pub const Q38_BATCH_DEFAULT_BUCKETS: &str = "1,2,4";

fn batch_prefill_chunk_tokens_explicit_setting_preserves_unset_one_shot(
    raw: Option<&str>,
    prompt_tokens: usize,
) -> Option<usize> {
    raw.and_then(|v| q38_prefill_chunk_tokens(Some(v)))
        .filter(|&chunk| prompt_tokens > chunk)
}

pub const A_PADDED_LANE_IS_CLOBBERED_AND_MUST_REPREFILL_BEFORE_REUSE_BECAUSE_THE_FLAT_FP8_CACHE_AND_GDN_STATE_ARE_BOUND_BY_POINTER_NOT_BY_TABLE:
    () = ();

pub fn rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin(
    g: &str,
) -> bool {
    std::env::var("NV_Q38_BATCH_ROWWISE")
        .map(|v| v.split(',').any(|t| t.trim() == g || t.trim() == "all"))
        .unwrap_or(false)
}

pub fn batch_prof_env_nv_q38_batch_prof_1_plus_nv_prof_gdn_1_plus_nv_graph_off_1_because_every_lap_syncs(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_BATCH_PROF")
        && nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_GRAPH_OFF")
}

pub fn batch_gemm_env_opt_in_nv_q38_batch_gemm_1_lane_concurrent_attn_gdn_mrow_ab_and_fp8_lt_arms(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_BATCH_GEMM")
}

pub const THE_FUSED_FP8_DECODE_SPLIT_PICK_TARGETS_TWO_WORKGROUPS_PER_SM_FOR_ONE_LANE_AND_THE_BATCH_STEP_FORKS_M_OF_THEM_AT_ONCE_SO_AT_M_LANES_THE_CARD_CARRIES_M_TIMES_THE_TARGET_AND_EVERY_EXTRA_BLOCK_WRITES_A_SPLIT_K_PARTIAL_NOBODY_NEEDED:
    &str = "24 heads x 16 splits x 16 lanes = 6144 workgroups against a 376 target, and each of \
            them writes a (head_dim+2)-float partial into the lane's split-k scratch that the \
            fan-in block then reads back";

pub fn attn_lane_split_env_opt_in_nv_q38_attn_lane_split_1_the_split_pick_divides_the_one_lane_rung_by_the_lane_group_width(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_ATTN_LANE_SPLIT")
}

pub const A_NULLED_STAGE_SERVES_MEANINGLESS_TOKENS_BY_CONSTRUCTION_BECAUSE_IT_WRITES_ZEROS_INSTEAD_OF_ITS_FUNCTION_AND_ONLY_THE_GRAPHED_STEP_DELTA_AGAINST_THE_SAME_SESSIONS_UNNULLED_CELL_IS_A_MEASUREMENT:
    &str = "an eager sync-lap total pro-rated onto a graphed step is a launch census wearing a \
            timer's clothes; the only thing that prices a stage with graphs on is replacing its \
            launches with a node that does no work and reading the replay delta";

pub const THE_NULL_NODE_IS_A_ZEROING_ALLOCATION_ISSUED_AT_THE_STAGES_OWN_POINT_IN_THE_CAPTURED_STREAM_SO_STREAM_ORDER_AND_EVERY_DOWNSTREAM_SHAPE_SURVIVE_WHILE_THE_STAGES_KERNELS_AND_ITS_GRAPH_NODES_BOTH_LEAVE:
    &str = "capture derives a graph edge from stream order, not from data, so a memset issued \
            where the stage was keeps the same predecessor and the same successor; what it does \
            not keep is the stage's node COUNT, so the delta is kernel time plus node replay cost \
            and is therefore the cost of the stage TO THE STEP rather than of its kernels alone";

pub fn null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream(
    stage: &str,
) -> bool {
    static NULLED: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    let set = NULLED.get_or_init(|| {
        let raw = std::env::var("NV_Q38_NULL_STAGE").unwrap_or_default();
        let v: Vec<String> = raw
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        if !v.is_empty() {
            eprintln!(
                "[qwen38_batch] NV_Q38_NULL_STAGE active: {} -- {A_NULLED_STAGE_SERVES_MEANINGLESS_TOKENS_BY_CONSTRUCTION_BECAUSE_IT_WRITES_ZEROS_INSTEAD_OF_ITS_FUNCTION_AND_ONLY_THE_GRAPHED_STEP_DELTA_AGAINST_THE_SAME_SESSIONS_UNNULLED_CELL_IS_A_MEASUREMENT}",
                v.join("+")
            );
        }
        v
    });
    set.iter().any(|s| s == stage)
}

fn one_zeroing_node_where_a_stages_launches_were(
    like: &Tensor,
    dims: (usize, usize, usize),
) -> Result<Tensor> {
    let _ = THE_NULL_NODE_IS_A_ZEROING_ALLOCATION_ISSUED_AT_THE_STAGES_OWN_POINT_IN_THE_CAPTURED_STREAM_SO_STREAM_ORDER_AND_EVERY_DOWNSTREAM_SHAPE_SURVIVE_WHILE_THE_STAGES_KERNELS_AND_ITS_GRAPH_NODES_BOTH_LEAVE;
    let dev = match like.device() {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("NV_Q38_NULL_STAGE requires CUDA"),
    };
    anyhow::ensure!(
        like.dtype() == DType::BF16,
        "NV_Q38_NULL_STAGE emits bf16 zeros and this stage's neighbour is {:?}; a null whose dtype \
         differs would add or remove a cast the real stage never ran and price it as the stage",
        like.dtype()
    );
    let s = nv_layers::cuda_stream::current_stream(&dev);
    let n = dims.0 * dims.1 * dims.2;
    let sl = s
        .alloc_zeros::<bf16>(n)
        .map_err(|e| anyhow::anyhow!("NV_Q38_NULL_STAGE alloc_zeros {n}: {e:?}"))?;
    let st = candle_core::CudaStorage::wrap_cuda_slice(sl, dev);
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(st),
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

pub const PACKED_FORMATION_PREFILL_IS_ONE_WIDE_CALL_BECAUSE_CHAT_PREFILL_IS_PER_CALL_WEIGHT_TRAFFIC_BOUND_29_9MS_AT_20_TOKENS_73MS_AT_320_SO_N_SERIAL_CALLS_PAY_THE_WEIGHTS_N_TIMES:
    () = ();

pub const PACKED_PREFILL_FFN_NORMS_AND_HEAD_RIDE_THE_PACKED_ROWS_SO_THEIR_BYTES_ARE_M_ROW_CLASS_WHILE_EACH_MIXER_SEGMENT_RUNS_THE_SOLO_PREFILL_ROUTE_ON_ITS_OWN_LANE_CACHE:
    () = ();

pub fn form_pack_env_opt_in_nv_q38_form_pack_1_packed_formation_prefill_rides_the_m_row_gemm_routes_so_it_requires_nv_q38_batch_gemm_1(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_FORM_PACK")
        && batch_gemm_env_opt_in_nv_q38_batch_gemm_1_lane_concurrent_attn_gdn_mrow_ab_and_fp8_lt_arms()
}

pub const FP8_LT_MROW_MIN_LANES_4_BECAUSE_THE_MK_GEMV_HOLDS_ITS_ONE_WEIGHTS_READ_SHAPE_TO_M3_AND_COLLAPSES_TO_3X_BY_M8:
    usize = 4;

pub const THE_FP8_LT_CROSSOVER_IS_A_PROPERTY_OF_WHICHEVER_BF16_ACTIVATION_GEMV_IS_WIRED_SO_A_FASTER_ONE_MOVES_IT_AND_NV_Q38_FP8LT_MIN_LANES_IS_HOW_A_ROUND_PRICES_THE_MOVE: &str =
    "measured on the qwen3.8-27B fp8 projection shapes over a 768 MB rotating weight set, so \
     nothing is served from L2. The whole-row-staged gemv_e4m3_mk beats the cublasLt e4m3 tile \
     to m=3 and loses from m=4, which is what the shipped 4 encodes. The k-tiled group gemv \
     (NV_Q38_E4M3_GROUP) stages a k-tile instead of the whole row and gives each warp more than \
     one output row, and on the gdn shapes it still beats the tile at m=4 and m=5 \
     (gdn.in_proj_qkv 47.03 us against 47.54 at m=4, gdn.out_proj 28.91 against 36.85), so a \
     round that arms it may price a crossover of 6. Above the crossover the tile is flat -- \
     1104 GB/s at m=1 and 1095 at m=16 -- and the gemv is not, so the number is a crossover and \
     never a cap";

pub fn fp8_lt_mrow_min_lanes_env_override_nv_q38_fp8lt_min_lanes() -> usize {
    std::env::var("NV_Q38_FP8LT_MIN_LANES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|m| *m >= 2)
        .unwrap_or(
            FP8_LT_MROW_MIN_LANES_4_BECAUSE_THE_MK_GEMV_HOLDS_ITS_ONE_WEIGHTS_READ_SHAPE_TO_M3_AND_COLLAPSES_TO_3X_BY_M8,
        )
}

pub const Q38_BATCH_LANES_MAX_16_MIRRORING_THE_MK_KERNEL_FAMILY_TEMPLATE_ARM_CAP: usize = 16;

pub const Q38_BATCH_LANES_ROW_TWIN_CEILING_8_THE_SOLE_M9_TO_16_PROJECTION_ROUTE_IS_THE_FP8_LT_ROWQUANT_GEMM_THE_BATCH_GEMM_ARM_OWNS:
    usize = 8;

pub const Q38_BATCH_BOOT_HEADROOM_MIN_FREE_BYTES_512MB_BECAUSE_16_LANES_OF_8K_KV_PLUS_GDN_STATE_TAKE_3_4GB_AND_GRAPH_CAPTURE_STILL_NEEDS_POOL_PAGES:
    usize = 512 << 20;

pub fn batch_gemm_fp8lt_subarm_default_on_kill_nv_q38_batch_gemm_fp8lt_0_it_rowquants_activations_to_e4m3_unlike_the_bf16_exact_mk_gemv(
) -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
        "NV_Q38_BATCH_GEMM_FP8LT",
    )
}

pub const LANE_CONCURRENT_ATTN_AND_GDN_ARE_BIT_IDENTICAL_TO_THE_SERIAL_LOOP_ONLY_THE_MROW_AB_GEMM_AND_FP8_LT_ROUTES_CHANGE_NUMERIC_ROUTE:
    () = ();

pub const THE_M1_FUSED_DECODE_STACK_IS_SHAPE_GATED_TO_ONE_BY_ONE_BY_HIDDEN_SO_A_B_ROW_GROUP_REACHES_IT_ONLY_BY_LOOPING_B_TIMES_AND_ROUND_22_PRICED_THAT_LOOP_AT_223US_AT_M1_378_AT_M2_AND_2974_AT_M16_AGAINST_A_FLAT_256_286US_TILE:
    () = ();

pub const B1_SOLO_TRUNK_TAKES_A_STABLE_ZERO_POSITIONS_TENSOR_AND_A_CAPTURE_TIME_WRITE_START_BECAUSE_THE_M1_FUSED_QKV_ROPE_STORE_ARM_READS_ITS_ROPE_POSITION_FROM_THE_DEVICE_WRITE_POS_BUFFER_THAT_PREPARE_FOR_STEP_REFRESHES_BEFORE_EVERY_REPLAY:
    () = ();

pub const A_LAYER_WHOSE_ROTARY_GEOMETRY_MISSES_THE_M1_FUSED_GATE_WOULD_FALL_BACK_TO_THE_POSITIONS_TENSOR_AND_A_CAPTURED_GRAPH_WOULD_THEN_REPLAY_A_FROZEN_POSITION_SO_THE_ARM_REFUSES_INSTEAD_OF_ENGAGING_PARTIALLY:
    () = ();

pub const THE_B1_SOLO_TRUNK_REFUSES_UNDER_ANY_NV_Q38_BATCH_ROWWISE_PIN_FOR_A_REASON_AND_A_DEFECT:
    &str =
    "the reason: a rowwise pin exists to make the group a per-row m=1 TWIN of the eager solo \
     body, and routing b=1 through a second, different m=1 route defeats exactly what the pin \
     is asserting. the defect: with NV_Q38_BATCH_ROWWISE=all and this arm on, a process that \
     builds a SECOND Qwen38BatchLanes after dropping the first SIGSEGVs inside that second \
     engine's bucket-4 boot-warm capture, right after bucket 1 (the trunk, 2821 nodes) and \
     bucket 2 (6074 nodes) capture cleanly. Reproducer: the eight-wide suite with only \
     NV_Q38_BATCH_SOLO_ARMS=1 added to the published arms; the same suite is green with \
     NV_Q38_SOLO_VIA_LANES=1 alone and with NV_Q38_BATCH_WINDOW_MS=0 alone, and the m-row \
     ladder builds two batch engines per pass with this arm on and never faults, so the \
     trigger is the pin, not the arm alone";

pub fn b1_solo_arm_stack_env_opt_in_nv_q38_batch_solo_arms_1_a_one_row_step_runs_the_m1_fused_decode_trunk_the_batch_body_cannot_reach_above_m1(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q38_BATCH_SOLO_ARMS")
}

fn b1_solo_arm_stack_engages_for_this_model(model: &Qwen3Moe) -> bool {
    let () = A_LAYER_WHOSE_ROTARY_GEOMETRY_MISSES_THE_M1_FUSED_GATE_WOULD_FALL_BACK_TO_THE_POSITIONS_TENSOR_AND_A_CAPTURED_GRAPH_WOULD_THEN_REPLAY_A_FROZEN_POSITION_SO_THE_ARM_REFUSES_INSTEAD_OF_ENGAGING_PARTIALLY;
    if !b1_solo_arm_stack_env_opt_in_nv_q38_batch_solo_arms_1_a_one_row_step_runs_the_m1_fused_decode_trunk_the_batch_body_cannot_reach_above_m1(
    ) {
        return false;
    }
    let _ = THE_B1_SOLO_TRUNK_REFUSES_UNDER_ANY_NV_Q38_BATCH_ROWWISE_PIN_FOR_A_REASON_AND_A_DEFECT;
    if std::env::var("NV_Q38_BATCH_ROWWISE").is_ok()
        || !nv_q38_fused_qkv_prep_env_kill_switch_nv_q38_fused_qkv_0()
        || model.dtype() != DType::BF16
        || model.rope.cos().dtype() != DType::F32
    {
        return false;
    }
    model.layers.iter().all(|l| match &l.mixer {
        LayerMixer::Full(a) => {
            a.head_dim % 32 == 0
                && a.head_dim <= 1024
                && a.rotary_dim >= 2
                && a.rotary_dim % 2 == 0
                && a.rotary_dim <= a.head_dim
        }
        LayerMixer::Linear(_) => true,
    })
}

pub const DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED:
    () = ();

pub const DEVICE_ARGMAX_RETURNS_TOKEN_0_ON_AN_ALL_NONFINITE_ROW_WHERE_HOST_SAMPLING_REPORTS_EXHAUSTED:
    () = ();

type BatchProf = Option<nv_layers::linear_attn::gdn_step_prof::SyncLaps>;

fn lap(prof: &mut BatchProf, label: &'static str) {
    if let Some(p) = prof.as_mut() {
        p.lap(label);
    }
}

pub fn q38_batch_bucket_plan_env_nv_q38_batch_sizes() -> BucketPlan {
    BucketPlan::parse(Some(
        std::env::var("NV_Q38_BATCH_SIZES")
            .ok()
            .as_deref()
            .unwrap_or(Q38_BATCH_DEFAULT_BUCKETS),
    ))
}

pub struct Qwen38BatchLanes {
    model: Qwen3Moe,
    caches: Vec<Qwen3MoeKvCache>,
    lane_pos: Vec<usize>,
    plan: BucketPlan,
    device: Device,
    stream: Arc<CudaStream>,
    lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i: Vec<Arc<CudaStream>>,
    runner: CudaGraphRunner,
    token_buf: CudaSlice<u32>,
    host_tok: Box<[u32]>,
    stable_zero_positions_row_for_the_b1_solo_trunk_whose_m1_fused_arm_ignores_it: Tensor,
    logits_buf: CudaSlice<f32>,
    amax_val: CudaSlice<f32>,
    amax_idx: CudaSlice<i32>,
    amax_out: CudaSlice<u32>,
    vocab: usize,
    capture_supported: bool,
    capture_failed: bool,
    captures: u64,
    replays: u64,
    buckets_whose_lazy_inits_are_materialized_so_recapture_skips_the_eager_warm_and_lin_state_snapshot:
        std::collections::HashSet<usize>,

    _err_drain: CtxErrDrain,
}

struct CtxErrDrain(Arc<CudaContext>);

impl Drop for CtxErrDrain {
    fn drop(&mut self) {
        let _ = self.0.check_err();
    }
}

impl Qwen38BatchLanes {
    pub fn new(
        model: Qwen3Moe,
        device: &Device,
        max_seq_len: usize,
        plan: BucketPlan,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Qwen38BatchLanes requires a CUDA device"),
        };
        anyhow::ensure!(
            model.is_dense(),
            "Qwen38BatchLanes: only the dense trunk batches; MoE routing stays on the solo engine"
        );
        anyhow::ensure!(
            model.dtype() == DType::BF16,
            "Qwen38BatchLanes: trunk dtype must be BF16, got {:?}",
            model.dtype()
        );
        anyhow::ensure!(
            nv_q38_fused_qkv_prep_env_kill_switch_nv_q38_fused_qkv_0(),
            "Qwen38BatchLanes: NV_Q38_FUSED_QKV=0 disables the fused qkv+rope+store decode arm \
             the per-lane batch attention is built from"
        );
        let cfg = model.config();
        let hd = cfg.head_dim;
        anyhow::ensure!(
            hd.is_multiple_of(32) && hd <= 1024 && model.rope.cos().dtype() == DType::F32,
            "Qwen38BatchLanes: fused qkv decode geometry contract broken (head_dim {hd})"
        );
        for (li, layer) in model.layers.iter().enumerate() {
            anyhow::ensure!(
                layer.ffn.as_dense().is_some(),
                "Qwen38BatchLanes: layer {li} ffn is {}, dense required",
                layer.ffn.label()
            );
            if let LayerMixer::Linear(la) = &layer.mixer {
                anyhow::ensure!(
                    la.fused_decode_supported(),
                    "Qwen38BatchLanes: layer {li} linear-attn fused decode unsupported; the \
                     non-fused recurrent path is neither capture-safe nor lane-splittable"
                );
            }
        }
        let b_max = plan.max_bucket();
        anyhow::ensure!(
            b_max <= Q38_BATCH_LANES_MAX_16_MIRRORING_THE_MK_KERNEL_FAMILY_TEMPLATE_ARM_CAP,
            "Qwen38BatchLanes: bucket {b_max} exceeds the m<=16 mk kernel-family template-arm \
             ceiling (gemv_e4m3_mk / gemm_bf16_mk / flash_decode all stop at 16 rows)"
        );
        anyhow::ensure!(
            b_max <= Q38_BATCH_LANES_ROW_TWIN_CEILING_8_THE_SOLE_M9_TO_16_PROJECTION_ROUTE_IS_THE_FP8_LT_ROWQUANT_GEMM_THE_BATCH_GEMM_ARM_OWNS
                || batch_gemm_env_opt_in_nv_q38_batch_gemm_1_lane_concurrent_attn_gdn_mrow_ab_and_fp8_lt_arms(),
            "Qwen38BatchLanes: bucket {b_max} > 8 requires NV_Q38_BATCH_GEMM=1"
        );

        let dev_stream = dev.cuda_stream();
        let capture_supported = !dev_stream.cu_stream().is_null();
        let stream = dev_stream;
        disable_event_tracking_at_first_capture(&dev);

        nv_quant::nvfp4::ensure_workspace_for_stream(&stream)?;
        let _ = nv_quant::matmul::TensorCoreGemm::new(stream.clone())?;

        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let mut lane_streams = vec![stream.clone()];
        for _ in 1..b_max {
            lane_streams.push(
                ctor_guard
                    .fork(stream.context())
                    .map_err(|e| anyhow::anyhow!("fork lane stream: {e:?}"))?,
            );
        }

        let mut caches = Vec::with_capacity(b_max);
        for _ in 0..b_max {
            let mut c = model.new_kv_cache(max_seq_len)?;
            c.set_fused_lin_attn(true);
            caches.push(c);
        }
        let vocab = model.vocab_size();
        let token_buf = stream
            .alloc_zeros::<u32>(b_max)
            .map_err(|e| anyhow::anyhow!("alloc token_buf: {e:?}"))?;
        let logits_buf = stream
            .alloc_zeros::<f32>(b_max * vocab)
            .map_err(|e| anyhow::anyhow!("alloc logits_buf: {e:?}"))?;
        let amax_parts = nv_kernels::cuda::argmax_parts();
        let amax_val = stream
            .alloc_zeros::<f32>(b_max * amax_parts)
            .map_err(|e| anyhow::anyhow!("alloc amax_val: {e:?}"))?;
        let amax_idx = stream
            .alloc_zeros::<i32>(b_max * amax_parts)
            .map_err(|e| anyhow::anyhow!("alloc amax_idx: {e:?}"))?;
        let amax_out = stream
            .alloc_zeros::<u32>(b_max)
            .map_err(|e| anyhow::anyhow!("alloc amax_out: {e:?}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after alloc: {e:?}"))?;
        if let Ok((free, total)) = cudarc::driver::result::mem_get_info() {
            anyhow::ensure!(
                free >= Q38_BATCH_BOOT_HEADROOM_MIN_FREE_BYTES_512MB_BECAUSE_16_LANES_OF_8K_KV_PLUS_GDN_STATE_TAKE_3_4GB_AND_GRAPH_CAPTURE_STILL_NEEDS_POOL_PAGES,
                "Qwen38BatchLanes: {b_max} lanes leave {:.2} GiB of {:.2} GiB free, under the \
                 boot headroom floor",
                free as f64 / (1u64 << 30) as f64,
                total as f64 / (1u64 << 30) as f64,
            );
        }
        let runner = CudaGraphRunner::new(stream.clone());
        ctor_guard.the_built_engine_owns_teardown_now();
        let mut built = Self {
            model,
            caches,
            lane_pos: vec![0usize; b_max],
            plan,
            device: device.clone(),
            stream,
            lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i: lane_streams,
            runner,
            token_buf,
            host_tok: vec![0u32; b_max].into_boxed_slice(),
            stable_zero_positions_row_for_the_b1_solo_trunk_whose_m1_fused_arm_ignores_it:
                Tensor::from_vec(vec![0i32], 1usize, device)?,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            capture_supported,
            capture_failed: false,
            captures: 0,
            replays: 0,
            buckets_whose_lazy_inits_are_materialized_so_recapture_skips_the_eager_warm_and_lin_state_snapshot:
                std::collections::HashSet::new(),
            _err_drain: CtxErrDrain(dev.cuda_stream().context().clone()),
        };
        built.boot_warm_every_bucket_on_garbage_states_then_reset_so_serving_groups_capture_without_the_eager_warm_pass()?;
        Ok(built)
    }

    fn boot_warm_every_bucket_on_garbage_states_then_reset_so_serving_groups_capture_without_the_eager_warm_pass(
        &mut self,
    ) -> Result<()> {
        if !self.capture_supported
            || nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_GRAPH_OFF")
            || !nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(
                "NV_Q38_BATCH_BOOT_WARM",
            )
        {
            return Ok(());
        }
        let sizes: Vec<usize> = self.plan.sizes().to_vec();
        for b in sizes {
            let tokens: Vec<Option<u32>> = vec![Some(0u32); b];
            let _ = self.step_batch(&tokens)?;
        }
        for lane in 0..self.caches.len() {
            self.reset_lane(lane)?;
        }
        Ok(())
    }

    pub fn model(&self) -> &Qwen3Moe {
        &self.model
    }

    pub fn lanes(&self) -> usize {
        self.caches.len()
    }

    pub fn plan(&self) -> &BucketPlan {
        &self.plan
    }

    pub fn lane_pos(&self, lane: usize) -> usize {
        self.lane_pos[lane]
    }

    pub fn lane_max_seq_len(&self) -> usize {
        self.caches[0].max_seq_len()
    }

    pub fn quiesce_graphs_before_external_eager_work_because_solo_requests_share_the_capture_stream(
        &mut self,
    ) -> Result<()> {
        self.invalidate_graphs_synced_because_eager_work_after_capture_faults_like_the_solo_engines_prefill()
    }

    pub fn captures(&self) -> u64 {
        self.captures
    }

    pub fn replays(&self) -> u64 {
        self.replays
    }

    pub fn captured_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("batch stream sync: {e:?}"))
    }

    pub fn reset_lane(&mut self, lane: usize) -> Result<()> {
        anyhow::ensure!(
            lane < self.caches.len(),
            "reset_lane: lane {lane} out of range"
        );
        self.invalidate_graphs_synced_because_eager_work_after_capture_faults_like_the_solo_engines_prefill()?;
        self.caches[lane].reset();
        self.lane_pos[lane] = 0;
        Ok(())
    }

    fn return_graph_mempool_reserved_pages(&self) {
        let ordinal = self.stream.context().ordinal() as i32;
        if let Ok(devh) = cudarc::driver::result::device::get(ordinal) {
            let _ = unsafe { cudarc::driver::sys::cuDeviceGraphMemTrim(devh) };
        }
    }

    fn invalidate_graphs_synced_because_eager_work_after_capture_faults_like_the_solo_engines_prefill(
        &mut self,
    ) -> Result<()> {
        let had_graphs = self.runner.has_cached();
        if had_graphs {
            self.synchronize()?;
        }
        self.runner.invalidate();
        if had_graphs {
            self.synchronize()?;
            self.return_graph_mempool_reserved_pages();
            self.synchronize()?;
        }
        Ok(())
    }

    pub fn prefill_lane(&mut self, lane: usize, prompt: &[u32]) -> Result<Vec<f32>> {
        let last = self.prefill_lane_forward_keep_last_row_on_device(lane, prompt)?;
        let row: Vec<f32> = last.flatten_all()?.to_vec1()?;
        Ok(row)
    }

    pub fn prefill_lane_greedy_token_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
        &mut self,
        lane: usize,
        prompt: &[u32],
    ) -> Result<u32> {
        let () = DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED;
        let () = DEVICE_ARGMAX_RETURNS_TOKEN_0_ON_AN_ALL_NONFINITE_ROW_WHERE_HOST_SAMPLING_REPORTS_EXHAUSTED;
        let last = self.prefill_lane_forward_keep_last_row_on_device(lane, prompt)?;
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (ls, ll) = last.storage_and_layout();
            let lcuda = match &*ls {
                candle_core::Storage::Cuda(c) => c,
                _ => anyhow::bail!("prefill logits must be on CUDA"),
            };
            let lsl = lcuda.as_cuda_slice::<f32>()?;
            let view = lsl.slice(ll.start_offset()..);
            let s = &self.stream;
            let (lp, _gl) = view.device_ptr(s);
            let (vp, _gv) = self.amax_val.device_ptr_mut(s);
            let (ip, _gi) = self.amax_idx.device_ptr_mut(s);
            let (op, _go) = self.amax_out.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::argmax_f32_rows(
                    s.cu_stream() as *mut std::ffi::c_void,
                    lp as *const f32,
                    1i32,
                    self.vocab as i32,
                    vp as *mut f32,
                    ip as *mut i32,
                    op as *mut u32,
                )
            };
            anyhow::ensure!(rc == 0, "prefill argmax_f32_rows rc={rc}");
        }
        let host: Vec<u32> = self
            .stream
            .clone_dtoh(&self.amax_out)
            .map_err(|e| anyhow::anyhow!("dtoh prefill argmax token: {e:?}"))?;
        Ok(host[0])
    }

    fn prefill_lane_forward_keep_last_row_on_device(
        &mut self,
        lane: usize,
        prompt: &[u32],
    ) -> Result<Tensor> {
        anyhow::ensure!(!prompt.is_empty(), "prefill_lane: empty prompt");
        anyhow::ensure!(
            lane < self.caches.len(),
            "prefill_lane: lane {lane} out of range"
        );
        self.invalidate_graphs_synced_because_eager_work_after_capture_faults_like_the_solo_engines_prefill()?;
        self.reset_lane(lane)?;
        let seq = prompt.len();
        anyhow::ensure!(
            seq < self.caches[lane].max_seq_len(),
            "prefill_lane: {seq} tokens leave no decode slot in max_seq_len {}",
            self.caches[lane].max_seq_len()
        );
        let chunk_raw = std::env::var(NV_Q38_PREFILL_CHUNK_ENV_DEFAULT_4096_SET_0_FOR_THE_ONE_SHOT_FORWARD_WHOSE_TRANSIENTS_MEASURED_0_21_MIB_PER_TOKEN).ok();
        nv_layers::linear::nvfp4_cfx_set_prefill_total_tokens_hint(seq);
        let logits = match batch_prefill_chunk_tokens_explicit_setting_preserves_unset_one_shot(
            chunk_raw.as_deref(),
            seq,
        ) {
            Some(chunk) => self
                .model
                .prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
                    prompt,
                    chunk,
                    &mut self.caches[lane],
                    None,
                )?,
            None => {
                let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &self.device)?;
                let positions_v: Vec<i32> = (0..seq as i32).collect();
                let positions = Tensor::from_vec(positions_v, seq, &self.device)?;
                self.model.forward_with_cache_dispatched_rows(
                    &tokens,
                    &positions,
                    &mut self.caches[lane],
                    None,
                    Some(1),
                )?
            }
        };
        self.lane_pos[lane] = seq;
        let (_b, s, _v) = logits.dims3()?;
        Ok(logits
            .narrow(1, s - 1, 1)?
            .to_dtype(DType::F32)?
            .contiguous()?)
    }

    pub fn prime_lane_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
        &mut self,
        lane: usize,
        k_chunk: &Tensor,
        v_chunk: &Tensor,
    ) -> Result<()> {
        anyhow::ensure!(
            lane < self.caches.len(),
            "prime_lane: lane {lane} out of range"
        );
        let t = k_chunk.dims().get(1).copied().unwrap_or(0);
        let pos = self.lane_pos[lane];
        anyhow::ensure!(
            pos + t <= self.caches[lane].max_seq_len(),
            "prime_lane: {pos} + {t} exceeds max_seq_len {}",
            self.caches[lane].max_seq_len()
        );
        self.caches[lane]
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                pos, k_chunk, v_chunk,
            )?;
        self.lane_pos[lane] = pos + t;
        Ok(())
    }

    pub fn prefill_lanes_packed_greedy_tokens_on_device(
        &mut self,
        joins: &[(usize, &[u32])],
    ) -> Result<Vec<u32>> {
        let () = PACKED_FORMATION_PREFILL_IS_ONE_WIDE_CALL_BECAUSE_CHAT_PREFILL_IS_PER_CALL_WEIGHT_TRAFFIC_BOUND_29_9MS_AT_20_TOKENS_73MS_AT_320_SO_N_SERIAL_CALLS_PAY_THE_WEIGHTS_N_TIMES;
        let () = PACKED_PREFILL_FFN_NORMS_AND_HEAD_RIDE_THE_PACKED_ROWS_SO_THEIR_BYTES_ARE_M_ROW_CLASS_WHILE_EACH_MIXER_SEGMENT_RUNS_THE_SOLO_PREFILL_ROUTE_ON_ITS_OWN_LANE_CACHE;
        let () = DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED;
        anyhow::ensure!(
            form_pack_env_opt_in_nv_q38_form_pack_1_packed_formation_prefill_rides_the_m_row_gemm_routes_so_it_requires_nv_q38_batch_gemm_1(),
            "packed prefill: NV_Q38_FORM_PACK=1 plus NV_Q38_BATCH_GEMM=1 are required; without \
             the GEMM arm the batch route is byte-gated and packed rows are m-row class"
        );
        anyhow::ensure!(!joins.is_empty(), "packed prefill: no lanes");
        let b = joins.len();
        anyhow::ensure!(
            b <= self.caches.len(),
            "packed prefill: {b} joins but only {} lanes",
            self.caches.len()
        );
        let mut seen = std::collections::HashSet::new();
        for &(lane, prompt) in joins {
            anyhow::ensure!(
                lane < self.caches.len(),
                "packed prefill: lane {lane} out of range"
            );
            anyhow::ensure!(
                seen.insert(lane),
                "packed prefill: lane {lane} joined twice"
            );
            anyhow::ensure!(
                !prompt.is_empty(),
                "packed prefill: empty prompt on lane {lane}"
            );
            anyhow::ensure!(
                prompt.len() < self.caches[lane].max_seq_len(),
                "packed prefill: {} tokens leave no decode slot in max_seq_len {}",
                prompt.len(),
                self.caches[lane].max_seq_len()
            );
        }
        self.invalidate_graphs_synced_because_eager_work_after_capture_faults_like_the_solo_engines_prefill()?;
        let device = self.device.clone();
        let stream = self.stream.clone();
        let Qwen38BatchLanes {
            model,
            caches,
            lane_pos,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            ..
        } = self;
        let mut packed: Vec<u32> = Vec::new();
        let mut segs: Vec<(usize, usize)> = Vec::with_capacity(b);
        for &(lane, prompt) in joins {
            caches[lane].reset();
            lane_pos[lane] = 0;
            caches[lane].prepare_for_step(0, prompt.len())?;
            segs.push((packed.len(), prompt.len()));
            packed.extend_from_slice(prompt);
        }
        let total = packed.len();
        let tokens_flat = Tensor::from_vec(packed, total, &device)?.to_dtype(DType::U32)?;
        let x = match embed_lookup_bf16(&model.embed_weight, &tokens_flat)? {
            Some(rows) => rows,
            None => model.embed_weight.index_select(&tokens_flat, 0)?,
        }
        .reshape((1usize, total, model.config().hidden_size))?
        .to_dtype(model.dtype())?;
        let mut residual = x.clone();
        let mut normed = model.layers[0].pre_norm.forward(&x)?;
        for (li, layer) in model.layers.iter().enumerate() {
            let mixed = {
                let mut rows: Vec<Tensor> = Vec::with_capacity(b);
                for (j, &(lane, _)) in joins.iter().enumerate() {
                    let (start, len) = segs[j];
                    let seg = normed.narrow(1, start, len)?.contiguous()?;
                    let out = match &layer.mixer {
                        LayerMixer::Full(attn) => {
                            let slot = caches[lane].full_slot_for_layer(li).ok_or_else(|| {
                                anyhow::anyhow!(
                                    "packed prefill: layer {li} has no full-attention cache slot"
                                )
                            })?;
                            let positions = Tensor::from_vec(
                                (0..len as i32).collect::<Vec<i32>>(),
                                len,
                                &device,
                            )?;
                            attn.forward_with_cache(
                                &seg,
                                &model.rope,
                                &positions,
                                &mut caches[lane],
                                slot,
                                0,
                                len,
                            )?
                        }
                        LayerMixer::Linear(la) => {
                            let slot =
                                caches[lane].lin_attn_slot_for_layer(li).ok_or_else(|| {
                                    anyhow::anyhow!(
                                        "packed prefill: layer {li} has no linear-attention slot"
                                    )
                                })?;
                            caches[lane].lin_attn_step(slot, la, &seg)?
                        }
                    };
                    rows.push(out);
                }
                let refs: Vec<&Tensor> = rows.iter().collect();
                Tensor::cat(&refs, 1)?
            };
            let (normed_post, residual_after_attn) =
                layer.post_norm.forward_residual(&mixed, &residual)?;
            let ffn_out = dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
                &layer.ffn,
                &normed_post,
                total,
            )?;
            if li + 1 < model.layers.len() {
                let (normed_next, residual_next) = model.layers[li + 1]
                    .pre_norm
                    .forward_residual(&ffn_out, &residual_after_attn)?;
                normed = normed_next;
                residual = residual_next;
            } else {
                residual = residual_after_attn.add(&ffn_out)?;
            }
        }
        let mut last_rows: Vec<Tensor> = Vec::with_capacity(b);
        for &(start, len) in segs.iter() {
            last_rows.push(residual.narrow(1, start + len - 1, 1)?);
        }
        let refs: Vec<&Tensor> = last_rows.iter().collect();
        let last = Tensor::cat(&refs, 1)?;
        let xh = model.final_norm.forward(&last)?;
        let logits = model.lm_head.forward(&xh)?;
        let lc = logits.to_dtype(DType::F32)?.contiguous()?;
        {
            use cudarc::driver::{DevicePtr, DevicePtrMut};
            let (ls, ll) = lc.storage_and_layout();
            let lcuda = match &*ls {
                candle_core::Storage::Cuda(c) => c,
                _ => anyhow::bail!("packed prefill logits must be on CUDA"),
            };
            let lsl = lcuda.as_cuda_slice::<f32>()?;
            let view = lsl.slice(ll.start_offset()..);
            let (lp, _gl) = view.device_ptr(&stream);
            let (vp, _gv) = amax_val.device_ptr_mut(&stream);
            let (ip, _gi) = amax_idx.device_ptr_mut(&stream);
            let (op, _go) = amax_out.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::argmax_f32_rows(
                    stream.cu_stream() as *mut std::ffi::c_void,
                    lp as *const f32,
                    b as i32,
                    *vocab as i32,
                    vp as *mut f32,
                    ip as *mut i32,
                    op as *mut u32,
                )
            };
            anyhow::ensure!(rc == 0, "packed prefill argmax_f32_rows rc={rc}");
        }
        for (j, &(lane, _)) in joins.iter().enumerate() {
            let (_, len) = segs[j];
            caches[lane].advance(len);
            lane_pos[lane] = len;
        }
        let host: Vec<u32> = stream
            .clone_dtoh(amax_out)
            .map_err(|e| anyhow::anyhow!("dtoh packed prefill argmax tokens: {e:?}"))?;
        Ok(host[..b].to_vec())
    }

    fn ensure_lane_lin_states_fused_rerun_each_step_because_reset_drops_them_and_early_returns_make_it_cheap(
        &mut self,
        b: usize,
    ) -> Result<()> {
        let Qwen38BatchLanes { model, caches, .. } = self;
        for cache in caches[..b].iter_mut() {
            model.ensure_fused_lin_states(cache)?;
        }
        Ok(())
    }

    pub fn step_batch(&mut self, tokens: &[Option<u32>]) -> Result<Vec<Option<Vec<f32>>>> {
        let _ = self.step_batch_run_common(tokens)?;
        let host: Vec<f32> = self
            .stream
            .clone_dtoh(&self.logits_buf)
            .map_err(|e| anyhow::anyhow!("dtoh batch logits: {e:?}"))?;
        let mut out = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            out.push(t.map(|_| host[i * self.vocab..(i + 1) * self.vocab].to_vec()));
        }
        Ok(out)
    }

    pub fn step_batch_greedy_tokens_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
        &mut self,
        tokens: &[Option<u32>],
    ) -> Result<Vec<Option<u32>>> {
        let () = DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED;
        let () = DEVICE_ARGMAX_RETURNS_TOKEN_0_ON_AN_ALL_NONFINITE_ROW_WHERE_HOST_SAMPLING_REPORTS_EXHAUSTED;
        let _ = self.step_batch_run_common(tokens)?;
        let host: Vec<u32> = self
            .stream
            .clone_dtoh(&self.amax_out)
            .map_err(|e| anyhow::anyhow!("dtoh batch argmax tokens: {e:?}"))?;
        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, t)| t.map(|_| host[i]))
            .collect())
    }

    pub fn step_batch_greedy_tokens_launch_without_final_sync_so_the_callers_detok_and_sse_of_the_prior_step_overlap_this_gpu_step(
        &mut self,
        tokens: &[Option<u32>],
    ) -> Result<()> {
        let () = DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED;
        let _ = self.step_batch_enqueue_common(tokens)?;
        Ok(())
    }

    pub fn step_batch_greedy_tokens_collect_after_launch(
        &mut self,
        tokens: &[Option<u32>],
    ) -> Result<Vec<Option<u32>>> {
        self.synchronize()?;
        let host: Vec<u32> = self
            .stream
            .clone_dtoh(&self.amax_out)
            .map_err(|e| anyhow::anyhow!("dtoh batch argmax tokens: {e:?}"))?;
        Ok(tokens
            .iter()
            .enumerate()
            .map(|(i, t)| t.map(|_| host[i]))
            .collect())
    }

    fn step_batch_run_common(&mut self, tokens: &[Option<u32>]) -> Result<usize> {
        let b_bucket = self.step_batch_enqueue_common(tokens)?;
        self.synchronize()?;
        if batch_prof_env_nv_q38_batch_prof_1_plus_nv_prof_gdn_1_plus_nv_graph_off_1_because_every_lap_syncs() {
            nv_layers::linear_attn::gdn_step_prof::report_and_reset(self.lane_pos[0]);
        }
        Ok(b_bucket)
    }

    fn step_batch_enqueue_common(&mut self, tokens: &[Option<u32>]) -> Result<usize> {
        anyhow::ensure!(!tokens.is_empty(), "step_batch: empty step");
        anyhow::ensure!(
            tokens.len() <= self.caches.len(),
            "step_batch: {} rows but only {} lanes",
            tokens.len(),
            self.caches.len()
        );
        anyhow::ensure!(
            tokens.iter().any(|t| t.is_some()),
            "step_batch: all rows padded; nothing to decode"
        );
        let need = tokens
            .iter()
            .rposition(|t| t.is_some())
            .map(|i| i + 1)
            .unwrap();
        let b_bucket = self.plan.bucket_for(need).ok_or_else(|| {
            anyhow::anyhow!(
                "step_batch: {need} lanes exceed max bucket {}",
                self.plan.max_bucket()
            )
        })?;
        for (i, t) in tokens.iter().enumerate() {
            if t.is_some() {
                anyhow::ensure!(
                    self.lane_pos[i] < self.caches[i].max_seq_len(),
                    "step_batch: lane {i} cache full at {}",
                    self.lane_pos[i]
                );
            }
        }
        self.ensure_lane_lin_states_fused_rerun_each_step_because_reset_drops_them_and_early_returns_make_it_cheap(
            b_bucket,
        )?;

        for i in 0..b_bucket {
            let (tok, pos) = match tokens.get(i).copied().flatten() {
                Some(t) => (t, self.lane_pos[i]),
                None => (0u32, 0usize),
            };
            self.host_tok[i] = tok;
            self.caches[i].set_pending_pos_host_only(pos, pos + 1);
        }

        if batch_prof_env_nv_q38_batch_prof_1_plus_nv_prof_gdn_1_plus_nv_graph_off_1_because_every_lap_syncs() {
            nv_layers::linear_attn::gdn_step_prof::arm_only_while_an_eager_decode_profiler_is_active_because_every_lap_syncs(true);
        }
        let captured = if self.capture_supported
            && !self.capture_failed
            && !nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_GRAPH_OFF")
        {
            self.step_batch_captured(b_bucket)?
        } else {
            self.step_batch_eager(b_bucket)?;
            false
        };
        let _ = captured;

        for (i, t) in tokens.iter().enumerate() {
            if t.is_some() {
                let new_len = self.lane_pos[i] + 1;
                self.caches[i].set_current_len(new_len);
                self.lane_pos[i] = new_len;
            }
        }
        Ok(b_bucket)
    }

    fn step_batch_eager(&mut self, b_bucket: usize) -> Result<()> {
        let stream = self.stream.clone();
        let Qwen38BatchLanes {
            model,
            caches,
            token_buf,
            host_tok,
            stable_zero_positions_row_for_the_b1_solo_trunk_whose_m1_fused_arm_ignores_it: zpos,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i: lane_streams,
            ..
        } = self;
        run_batch_step_body(
            &stream,
            model,
            &mut caches[..b_bucket],
            token_buf,
            &host_tok[..],
            zpos,
            logits_buf,
            ArgmaxScratch {
                val: amax_val,
                idx: amax_idx,
                out: amax_out,
            },
            *vocab,
            b_bucket,
            lane_streams,
        )
    }

    fn eager_warm_pass_with_lin_state_snapshot_restore_to_materialize_lazy_inits_before_capture(
        &mut self,
        b_bucket: usize,
    ) -> Result<()> {
        use anyhow::Context as _;
        let stream = self.stream.clone();
        let Qwen38BatchLanes {
            model,
            caches,
            token_buf,
            host_tok,
            stable_zero_positions_row_for_the_b1_solo_trunk_whose_m1_fused_arm_ignores_it: zpos,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i: lane_streams,
            ..
        } = self;
        let mut snaps = Vec::with_capacity(b_bucket);
        for cache in caches[..b_bucket].iter() {
            snaps.push(cache.snapshot_lin_states()?);
        }
        run_batch_step_body(
            &stream,
            model,
            &mut caches[..b_bucket],
            token_buf,
            &host_tok[..],
            zpos,
            logits_buf,
            ArgmaxScratch {
                val: &mut *amax_val,
                idx: &mut *amax_idx,
                out: &mut *amax_out,
            },
            *vocab,
            b_bucket,
            lane_streams,
        )
        .context("warm pass before batch capture")?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("batch warm sync: {e:?}"))?;
        for (cache, snap) in caches[..b_bucket].iter_mut().zip(snaps.iter()) {
            cache.restore_lin_states(snap)?;
        }
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("batch post-restore sync: {e:?}"))?;
        Ok(())
    }

    fn capture_or_replay_bucket_graph(&mut self, key: u64, b_bucket: usize) -> Result<()> {
        let Qwen38BatchLanes {
            model,
            caches,
            runner,
            token_buf,
            host_tok,
            stable_zero_positions_row_for_the_b1_solo_trunk_whose_m1_fused_arm_ignores_it: zpos,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i: lane_streams,
            ..
        } = self;
        let vocab = *vocab;
        runner.run(key, |s| {
            run_batch_step_body(
                s,
                model,
                &mut caches[..b_bucket],
                token_buf,
                &host_tok[..],
                zpos,
                logits_buf,
                ArgmaxScratch {
                    val: amax_val,
                    idx: amax_idx,
                    out: amax_out,
                },
                vocab,
                b_bucket,
                lane_streams,
            )
        })
    }

    fn step_batch_captured(&mut self, b_bucket: usize) -> Result<bool> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let key =
            Q38_BATCH_GRAPH_SHAPE_TOKEN_BASE_KEEPS_SOLO_KEY_1_AND_VERIFY_BASE_0X51000_UNTOUCHED
                + b_bucket as u64;
        disable_event_tracking_at_first_capture(&dev);
        let need_capture = !self.runner.has_cached_token(key);
        if need_capture {
            if let Err(e) = self.runner.probe_capture() {
                self.capture_failed = true;
                eprintln!("[qwen38_batch] graph capture unavailable; batching uncaptured: {e:#}");
                self.step_batch_eager(b_bucket)?;
                return Ok(false);
            }
        }
        let warm_skipped_because_this_bucket_already_materialized_its_lazy_inits_this_process =
            need_capture
                && self
                    .buckets_whose_lazy_inits_are_materialized_so_recapture_skips_the_eager_warm_and_lin_state_snapshot
                    .contains(&b_bucket);
        if need_capture
            && !warm_skipped_because_this_bucket_already_materialized_its_lazy_inits_this_process
        {
            self.eager_warm_pass_with_lin_state_snapshot_restore_to_materialize_lazy_inits_before_capture(
                b_bucket,
            )?;
        }
        let mut result = self.capture_or_replay_bucket_graph(key, b_bucket);
        if result.is_err()
            && warm_skipped_because_this_bucket_already_materialized_its_lazy_inits_this_process
        {
            self.runner.invalidate();
            let _ = self.synchronize();
            if self
                .eager_warm_pass_with_lin_state_snapshot_restore_to_materialize_lazy_inits_before_capture(
                    b_bucket,
                )
                .is_ok()
            {
                result = self.capture_or_replay_bucket_graph(key, b_bucket);
            }
        }
        match result {
            Ok(()) => {
                if need_capture {
                    self.captures += 1;
                } else {
                    self.replays += 1;
                }
                self.buckets_whose_lazy_inits_are_materialized_so_recapture_skips_the_eager_warm_and_lin_state_snapshot
                    .insert(b_bucket);
                Ok(true)
            }
            Err(e) => {
                self.capture_failed = true;
                self.runner.invalidate();
                let _ = self.synchronize();
                self.return_graph_mempool_reserved_pages();
                for cache in self.caches.iter_mut() {
                    cache.reset();
                }
                for p in self.lane_pos.iter_mut() {
                    *p = 0;
                }
                Err(e).context(
                    "batch graph capture failed; every lane cache was reset -- re-prefill and \
                     rerun with NV_GRAPH_OFF=1",
                )
            }
        }
    }
}

impl Drop for Qwen38BatchLanes {
    fn drop(&mut self) {
        let mut td = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(
            &self.stream,
        );
        for ls in self
            .lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i
            .iter()
            .skip(1)
        {
            td = td.with_stream(ls);
        }
        let runner = &mut self.runner;
        td.run(|| runner.invalidate());
    }
}

fn disable_event_tracking_at_first_capture(dev: &candle_core::CudaDevice) {
    if nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_GRAPH_KEEP_EVENT_TRACKING") {
        return;
    }
    let stream = dev.cuda_stream();
    if !stream.context().is_event_tracking() {
        return;
    }
    let _ = stream.synchronize();
    crate::gemma4_batch_graph::graph_teardown::disable_event_tracking_before_capture(
        stream.context(),
    );
}

struct ArgmaxScratch<'a> {
    val: &'a mut CudaSlice<f32>,
    idx: &'a mut CudaSlice<i32>,
    out: &'a mut CudaSlice<u32>,
}

#[allow(clippy::too_many_arguments)]
fn run_batch_step_body(
    s: &Arc<CudaStream>,
    model: &Qwen3Moe,
    caches: &mut [Qwen3MoeKvCache],
    token_buf: &mut CudaSlice<u32>,
    host_tok: &[u32],
    zero_positions: &Tensor,
    logits_buf: &mut CudaSlice<f32>,
    amax: ArgmaxScratch<'_>,
    vocab: usize,
    b: usize,
    lane_streams: &[Arc<CudaStream>],
) -> Result<()> {
    nv_layers::cuda_stream::with_stream(s.clone(), || {
        batch_step_body_inner(
            s,
            model,
            caches,
            token_buf,
            host_tok,
            zero_positions,
            logits_buf,
            amax,
            vocab,
            b,
            lane_streams,
        )
    })
}

fn copy_logits_then_device_argmax_rows(
    s: &Arc<CudaStream>,
    logits: &Tensor,
    logits_buf: &mut CudaSlice<f32>,
    amax: ArgmaxScratch<'_>,
    vocab: usize,
    b: usize,
) -> Result<()> {
    let lc = logits.to_dtype(DType::F32)?.contiguous()?;
    let (ls, ll) = lc.storage_and_layout();
    let lcuda = match &*ls {
        candle_core::Storage::Cuda(c) => c,
        _ => anyhow::bail!("batch logits must be on CUDA"),
    };
    let lsl = lcuda.as_cuda_slice::<f32>()?;
    let n = b * vocab;
    let off = ll.start_offset();
    let src = lsl.slice(off..off + n);
    let mut dst = logits_buf.slice_mut(0..n);
    s.memcpy_dtod(&src, &mut dst)
        .map_err(|e| anyhow::anyhow!("dtod batch logits: {e:?}"))?;
    use cudarc::driver::{DevicePtr, DevicePtrMut};
    let (lp, _gl) = logits_buf.device_ptr(s);
    let (vp, _gv) = amax.val.device_ptr_mut(s);
    let (ip, _gi) = amax.idx.device_ptr_mut(s);
    let (op, _go) = amax.out.device_ptr_mut(s);
    let rc = unsafe {
        nv_kernels::cuda::argmax_f32_rows(
            s.cu_stream() as *mut std::ffi::c_void,
            lp as *const f32,
            b as i32,
            vocab as i32,
            vp as *mut f32,
            ip as *mut i32,
            op as *mut u32,
        )
    };
    anyhow::ensure!(rc == 0, "argmax_f32_rows rc={rc}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_step_body_inner(
    s: &Arc<CudaStream>,
    model: &Qwen3Moe,
    caches: &mut [Qwen3MoeKvCache],
    token_buf: &mut CudaSlice<u32>,
    host_tok: &[u32],
    zero_positions: &Tensor,
    logits_buf: &mut CudaSlice<f32>,
    amax: ArgmaxScratch<'_>,
    vocab: usize,
    b: usize,
    lane_streams: &[Arc<CudaStream>],
) -> Result<()> {
    let _batch_e4m3_group_scope = nv_layers::linear::Qwen38BatchE4m3GroupScopeGuard::enter(b);
    let dev = match &model.device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("cuda device required"),
    };
    {
        let mut dst = token_buf.slice_mut(0..b);
        s.memcpy_htod(&host_tok[..b], &mut dst)
            .map_err(|e| anyhow::anyhow!("htod batch tokens: {e:?}"))?;
    }
    for cache in caches.iter_mut() {
        let (wp, nt) = (cache.host_write_pos[0], cache.host_n_total[0]);
        cache.prepare_for_step(wp as usize, nt as usize)?;
    }

    let mut tok_b = unsafe { s.alloc::<u32>(b).map_err(|e| anyhow::anyhow!(e))? };
    s.memcpy_dtod(&token_buf.slice(0..b), &mut tok_b)
        .map_err(|e| anyhow::anyhow!(e))?;
    let tokens_t = {
        let st = candle_core::CudaStorage::wrap_cuda_slice(tok_b, dev.clone());
        Tensor::from_storage(
            candle_core::Storage::Cuda(st),
            (b,),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };

    let mut prof: BatchProf = nv_layers::linear_attn::gdn_step_prof::SyncLaps::begin_if_armed(&dev);

    let gemm_arm =
        batch_gemm_env_opt_in_nv_q38_batch_gemm_1_lane_concurrent_attn_gdn_mrow_ab_and_fp8_lt_arms(
        );
    let lanes_conc: Option<&[Arc<CudaStream>]> = if gemm_arm && b > 1 && lane_streams.len() >= b {
        Some(&lane_streams[..b])
    } else {
        None
    };
    let _fp8_lt_mrow_scope = nv_layers::linear::VerifyTcFp8LtGemmScopeGuard::enter_if(
        gemm_arm
            && batch_gemm_fp8lt_subarm_default_on_kill_nv_q38_batch_gemm_fp8lt_0_it_rowquants_activations_to_e4m3_unlike_the_bf16_exact_mk_gemv()
            && b >= fp8_lt_mrow_min_lanes_env_override_nv_q38_fp8lt_min_lanes(),
    );

    let hidden_size = model.config().hidden_size;
    let x = if null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("embed") {
        one_zeroing_node_where_a_stages_launches_were(&model.embed_weight, (1usize, b, hidden_size))?
    } else {
        match embed_lookup_bf16(&model.embed_weight, &tokens_t)? {
            Some(rows) => rows,
            None => model.embed_weight.index_select(&tokens_t, 0)?,
        }
        .reshape((1usize, b, hidden_size))?
        .to_dtype(model.dtype())?
    };
    lap(&mut prof, "b.embed");

    if b == 1 && b1_solo_arm_stack_engages_for_this_model(model) {
        let () = B1_SOLO_TRUNK_TAKES_A_STABLE_ZERO_POSITIONS_TENSOR_AND_A_CAPTURE_TIME_WRITE_START_BECAUSE_THE_M1_FUSED_QKV_ROPE_STORE_ARM_READS_ITS_ROPE_POSITION_FROM_THE_DEVICE_WRITE_POS_BUFFER_THAT_PREPARE_FOR_STEP_REFRESHES_BEFORE_EVERY_REPLAY;
        let () = THE_M1_FUSED_DECODE_STACK_IS_SHAPE_GATED_TO_ONE_BY_ONE_BY_HIDDEN_SO_A_B_ROW_GROUP_REACHES_IT_ONLY_BY_LOOPING_B_TIMES_AND_ROUND_22_PRICED_THAT_LOOP_AT_223US_AT_M1_378_AT_M2_AND_2974_AT_M16_AGAINST_A_FLAT_256_286US_TILE;
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qwen38_batch] NV_Q38_BATCH_SOLO_ARMS b=1 step routed through the m1 fused \
                 decode trunk (norm writeout / norm-quant fold / w4a8 mlp / gdn prenorm fold / \
                 attn qkv-one all reachable, each still under its own env arm)"
            );
        });
        let write_start = caches[0].host_write_pos[0] as usize;
        let logits = model.forward_embedded_with_cache_dispatched_rows(
            x,
            &model.rope,
            zero_positions,
            &mut caches[0],
            None,
            Some(1),
            write_start,
        )?;
        lap(&mut prof, "b.head");
        return copy_logits_then_device_argmax_rows(s, &logits, logits_buf, amax, vocab, 1);
    }

    let layers = &model.layers;
    let rowwise_norms = rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("norms");
    let rowwise_ffn = rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("ffn");
    let rowwise_head = rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("head");
    let null_norm = null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("norm");
    let null_ffn = null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("ffn");
    let norm_rows = |n: &RmsNorm, x: &Tensor| -> Result<Tensor> {
        if null_norm {
            return one_zeroing_node_where_a_stages_launches_were(x, x.dims3()?);
        }
        if !rowwise_norms || b == 1 {
            return n.forward(x);
        }
        let mut rows = Vec::with_capacity(b);
        for i in 0..b {
            rows.push(n.forward(&x.narrow(1, i, 1)?)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    };
    let norm_residual_rows = |n: &RmsNorm, x: &Tensor, r: &Tensor| -> Result<(Tensor, Tensor)> {
        if null_norm {
            let d = x.dims3()?;
            return Ok((
                one_zeroing_node_where_a_stages_launches_were(x, d)?,
                one_zeroing_node_where_a_stages_launches_were(x, d)?,
            ));
        }
        if !rowwise_norms || b == 1 {
            return n.forward_residual(x, r);
        }
        let mut ns = Vec::with_capacity(b);
        let mut rs = Vec::with_capacity(b);
        for i in 0..b {
            let (a, s2) = n.forward_residual(&x.narrow(1, i, 1)?, &r.narrow(1, i, 1)?)?;
            ns.push(a);
            rs.push(s2);
        }
        let nrefs: Vec<&Tensor> = ns.iter().collect();
        let rrefs: Vec<&Tensor> = rs.iter().collect();
        Ok((Tensor::cat(&nrefs, 1)?, Tensor::cat(&rrefs, 1)?))
    };
    let ffn_rows = |ffn: &LayerFfn, x: &Tensor| -> Result<Tensor> {
        if null_ffn {
            return one_zeroing_node_where_a_stages_launches_were(x, x.dims3()?);
        }
        if !rowwise_ffn || b == 1 {
            return dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(ffn, x, b);
        }
        let mut rows = Vec::with_capacity(b);
        for i in 0..b {
            rows.push(dense_ffn_decode_forward_m1_nvfp4_gemv_else_layer_default(
                ffn,
                &x.narrow(1, i, 1)?.contiguous()?,
                1,
            )?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    };
    let mut residual = x.clone();
    let mut normed = norm_rows(&layers[0].pre_norm, &x)?;
    lap(&mut prof, "b.norm");
    for (li, layer) in layers.iter().enumerate() {
        let mixed = match &layer.mixer {
            LayerMixer::Full(attn) => {
                batch_full_attention(model, attn, li, &normed, caches, b, &mut prof, lanes_conc)?
            }
            LayerMixer::Linear(la) => batch_gdn(la, li, &normed, caches, b, &mut prof, lanes_conc)?,
        };
        let (normed_post, residual_after_attn) =
            norm_residual_rows(&layer.post_norm, &mixed, &residual)?;
        lap(&mut prof, "b.norm");
        let ffn_out = ffn_rows(&layer.ffn, &normed_post)?;
        lap(&mut prof, "b.ffn");
        if li + 1 < layers.len() {
            let (normed_next, residual_next) =
                norm_residual_rows(&layers[li + 1].pre_norm, &ffn_out, &residual_after_attn)?;
            normed = normed_next;
            residual = residual_next;
        } else {
            residual = residual_after_attn.add(&ffn_out)?;
        }
        lap(&mut prof, "b.norm");
    }
    let (x, logits) = if null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("head") {
        let lg = one_zeroing_node_where_a_stages_launches_were(&residual, (1usize, b, vocab))?;
        (residual.clone(), lg)
    } else if rowwise_head && b > 1 {
        let mut rows = Vec::with_capacity(b);
        for i in 0..b {
            let xr = model.final_norm.forward(&residual.narrow(1, i, 1)?)?;
            rows.push(model.lm_head.forward(&xr)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let lg = Tensor::cat(&refs, 1)?;
        (residual, lg)
    } else {
        let x = model.final_norm.forward(&residual)?;
        let lg = model.lm_head.forward(&x)?;
        (x, lg)
    };
    let _ = x;
    lap(&mut prof, "b.head");
    copy_logits_then_device_argmax_rows(s, &logits, logits_buf, amax, vocab, b)
}

fn proj_rows(lin: &Linear, x: &Tensor, b: usize) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(b);
    for i in 0..b {
        rows.push(lin.forward(&x.narrow(1, i, 1)?.contiguous()?)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

fn fork_lanes(main: &Arc<CudaStream>, lanes: &[Arc<CudaStream>]) -> Result<()> {
    let ev = main
        .record_event(None)
        .map_err(|e| anyhow::anyhow!("lane fork event: {e:?}"))?;
    for ls in lanes.iter().skip(1) {
        ls.wait(&ev)
            .map_err(|e| anyhow::anyhow!("lane fork wait: {e:?}"))?;
    }
    Ok(())
}

fn join_lanes(main: &Arc<CudaStream>, lanes: &[Arc<CudaStream>]) -> Result<()> {
    for ls in lanes.iter().skip(1) {
        let ev = ls
            .record_event(None)
            .map_err(|e| anyhow::anyhow!("lane join event: {e:?}"))?;
        main.wait(&ev)
            .map_err(|e| anyhow::anyhow!("lane join wait: {e:?}"))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn batch_full_attention(
    model: &Qwen3Moe,
    attn: &AttentionLayer,
    li: usize,
    normed: &Tensor,
    caches: &mut [Qwen3MoeKvCache],
    b: usize,
    prof: &mut BatchProf,
    lanes_conc: Option<&[Arc<CudaStream>]>,
) -> Result<Tensor> {
    let rowwise =
        rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("attnproj")
            && b > 1;
    let (q_raw, k_raw, v_raw, o_rowwise) = if null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("attn.qkv_proj") {
        (
            one_zeroing_node_where_a_stages_launches_were(normed, (1, b, attn.q_proj.out_features()))?,
            one_zeroing_node_where_a_stages_launches_were(normed, (1, b, attn.k_proj.out_features()))?,
            one_zeroing_node_where_a_stages_launches_were(normed, (1, b, attn.v_proj.out_features()))?,
            false,
        )
    } else if rowwise {
        (
            proj_rows(&attn.q_proj, normed, b)?,
            proj_rows(&attn.k_proj, normed, b)?,
            proj_rows(&attn.v_proj, normed, b)?,
            true,
        )
    } else {
        (
            attn.q_proj.forward(normed)?,
            attn.k_proj.forward(normed)?,
            attn.v_proj.forward(normed)?,
            false,
        )
    };
    lap(prof, "b.attn.qkv_proj");
    anyhow::ensure!(
        q_raw.dtype() == DType::BF16
            && k_raw.dtype() == DType::BF16
            && v_raw.dtype() == DType::BF16,
        "batch_full_attention layer {li}: projections not bf16; the fused per-lane store has no \
         fallback in the batch body"
    );
    let n_heads = attn.n_heads;
    let hd = attn.head_dim;
    let scaling = 1.0 / (hd as f32).sqrt();
    let dev = match &model.device {
        Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("batch_full_attention requires CUDA"),
    };
    let main = nv_layers::cuda_stream::current_stream(&dev);
    let conc = lanes_conc.filter(|ls| b > 1 && ls.len() >= b);
    let gated = if null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("attn.lanes") {
        one_zeroing_node_where_a_stages_launches_were(normed, (1, b, n_heads * hd))?
    } else if let Some(ls) = conc {
        batch_full_attention_lanes_concurrent(
            model, attn, li, &q_raw, &k_raw, &v_raw, caches, b, &main, &ls[..b],
        )?
    } else {
        let mut rows: Vec<Tensor> = Vec::with_capacity(b);
        for (i, cache) in caches.iter_mut().enumerate().take(b) {
            let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
                anyhow::anyhow!(
                    "batch_full_attention: layer {li} has no full-attention cache slot"
                )
            })?;
            let q_i = q_raw.narrow(1, i, 1)?;
            let k_i = k_raw.narrow(1, i, 1)?;
            let v_i = v_raw.narrow(1, i, 1)?;
            let (q_final, q_sig) = cache
                .fused_qkv_norm_rope_store_decode_rope_pos_reads_write_pos_dev_because_decode_positions_equal_write_start(
                    slot,
                    &q_i,
                    &k_i,
                    &v_i,
                    attn.q_norm.weight_bf16(),
                    attn.k_norm.weight_bf16(),
                    &model.rope,
                    n_heads,
                    attn.rotary_dim,
                    attn.q_norm.eps() as f32,
                    attn.attn_output_gate,
                )?;
            let out = cache.decode_attention_fp8(slot, &q_final, n_heads, scaling)?;
            let flat = out.reshape((1usize, 1usize, n_heads * hd))?;
            rows.push(match q_sig {
                Some(sig) => flat.mul(&sig)?,
                None => flat,
            });
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 1)?
    };
    lap(prof, "b.attn.lanes");
    let out = if null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("attn.o_proj") {
        one_zeroing_node_where_a_stages_launches_were(&gated, (1, b, attn.o_proj.out_features()))?
    } else if o_rowwise {
        proj_rows(&attn.o_proj, &gated, b)?
    } else {
        attn.o_proj.forward(&gated)?
    };
    lap(prof, "b.attn.o_proj");
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn batch_full_attention_lanes_concurrent(
    model: &Qwen3Moe,
    attn: &AttentionLayer,
    li: usize,
    q_raw: &Tensor,
    k_raw: &Tensor,
    v_raw: &Tensor,
    caches: &mut [Qwen3MoeKvCache],
    b: usize,
    main: &Arc<CudaStream>,
    ls: &[Arc<CudaStream>],
) -> Result<Tensor> {
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    let n_heads = attn.n_heads;
    let hd = attn.head_dim;
    let hh = n_heads * hd;
    let scaling = 1.0 / (hd as f32).sqrt();
    let dev = match &model.device {
        Device::Cuda(d) => d.clone(),
        _ => unreachable!(),
    };
    let mut out_all = unsafe {
        main.alloc::<bf16>(b * hh)
            .map_err(|e| anyhow::anyhow!("alloc lane out scratch: {e:?}"))?
    };
    let mut sig_all = if attn.attn_output_gate {
        Some(unsafe {
            main.alloc::<bf16>(b * hh)
                .map_err(|e| anyhow::anyhow!("alloc lane sig scratch: {e:?}"))?
        })
    } else {
        None
    };
    let lanes_for_the_attention_split_pick = if attn_lane_split_env_opt_in_nv_q38_attn_lane_split_1_the_split_pick_divides_the_one_lane_rung_by_the_lane_group_width() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            eprintln!(
                "[qwen38_batch] NV_Q38_ATTN_LANE_SPLIT active: {} lanes in flight take split rung \
                 {} instead of {} (nh={n_heads}) -- {THE_FUSED_FP8_DECODE_SPLIT_PICK_TARGETS_TWO_WORKGROUPS_PER_SM_FOR_ONE_LANE_AND_THE_BATCH_STEP_FORKS_M_OF_THEM_AT_ONCE_SO_AT_M_LANES_THE_CARD_CARRIES_M_TIMES_THE_TARGET_AND_EVERY_EXTRA_BLOCK_WRITES_A_SPLIT_K_PARTIAL_NOBODY_NEEDED}",
                b,
                nv_kernels::cuda::flash_decode_fp8_splits_pick_for_a_lane_group(
                    n_heads as i32,
                    0,
                    b as i32
                ),
                nv_kernels::cuda::flash_decode_fp8_splits_pick(n_heads as i32, 0),
            );
        });
        b
    } else {
        crate::qwen3_5_moe::SOLO_AND_EVERY_FAMILY_OUTSIDE_THE_QWEN38_LANE_GROUP_ENTERS_AT_ONE_LANE_IN_FLIGHT_SO_THE_SPLIT_PICK_IS_THE_ONE_THE_RECORD_WAS_MEASURED_ON
    };
    fork_lanes(main, ls)?;
    for (i, cache) in caches.iter_mut().enumerate().take(b) {
        let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
            anyhow::anyhow!("batch_full_attention: layer {li} has no full-attention cache slot")
        })?;
        let q_i = q_raw.narrow(1, i, 1)?;
        let k_i = k_raw.narrow(1, i, 1)?;
        let v_i = v_raw.narrow(1, i, 1)?;
        let lane_stream = if i > 0 { ls[i].clone() } else { main.clone() };
        nv_layers::cuda_stream::with_stream(lane_stream.clone(), || -> Result<()> {
            let (q_final, q_sig) = cache
                .fused_qkv_norm_rope_store_decode_rope_pos_reads_write_pos_dev_because_decode_positions_equal_write_start(
                    slot,
                    &q_i,
                    &k_i,
                    &v_i,
                    attn.q_norm.weight_bf16(),
                    attn.k_norm.weight_bf16(),
                    &model.rope,
                    n_heads,
                    attn.rotary_dim,
                    attn.q_norm.eps() as f32,
                    attn.attn_output_gate,
                )?;
            let every_lane_tensor_dies_before_the_join_because_its_async_free_lands_on_the_lane_stream =
                cache.decode_attention_fp8_whose_split_pick_sees_the_lane_group_width(
                    slot,
                    &q_final,
                    n_heads,
                    scaling,
                    lanes_for_the_attention_split_pick,
                )?;
            let copy_into = |src: &Tensor, dst: &mut CudaSlice<bf16>| -> Result<()> {
                let c = src.contiguous()?;
                let (st, l) = c.storage_and_layout();
                let cu = match &*st {
                    candle_core::Storage::Cuda(x) => x,
                    _ => anyhow::bail!("lane output must be CUDA"),
                };
                let sl = cu.as_cuda_slice::<bf16>()?;
                let view = sl.slice(l.start_offset()..);
                let (sp, _g) = view.device_ptr(&lane_stream);
                let (dp, _g2) = dst.device_ptr_mut(&lane_stream);
                let rc = unsafe {
                    nv_kernels::cuda::copy_cols_bf16(
                        lane_stream.cu_stream() as *mut std::ffi::c_void,
                        sp as *const u16,
                        dp as *mut u16,
                        1,
                        hh as i32,
                        hh as i64,
                        hh as i64,
                        0,
                        (i * hh) as i64,
                    )
                };
                anyhow::ensure!(rc == 0, "copy_cols_bf16 rc={rc}");
                Ok(())
            };
            copy_into(
                &every_lane_tensor_dies_before_the_join_because_its_async_free_lands_on_the_lane_stream,
                &mut out_all,
            )?;
            match (q_sig.as_ref(), sig_all.as_mut()) {
                (Some(sig), Some(sa)) => copy_into(sig, sa)?,
                (None, None) => {}
                _ => anyhow::bail!(
                    "batch_full_attention_lanes_concurrent: q_sig presence disagrees with \
                     attn_output_gate"
                ),
            }
            Ok(())
        })?;
    }
    join_lanes(main, ls)?;
    let out_t = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out_all, dev.clone());
        Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, b, hh),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    match sig_all {
        Some(sa) => {
            let sig_t = {
                let storage = candle_core::CudaStorage::wrap_cuda_slice(sa, dev);
                Tensor::from_storage(
                    candle_core::Storage::Cuda(storage),
                    (1usize, b, hh),
                    candle_core::op::BackpropOp::none(),
                    false,
                )
            };
            Ok(out_t.mul(&sig_t)?)
        }
        None => Ok(out_t),
    }
}

#[allow(clippy::too_many_arguments)]
fn batch_gdn(
    la: &LinearAttention,
    li: usize,
    normed: &Tensor,
    caches: &mut [Qwen3MoeKvCache],
    b: usize,
    prof: &mut BatchProf,
    lanes_conc: Option<&[Arc<CudaStream>]>,
) -> Result<Tensor> {
    if null_stage_probe_env_nv_q38_null_stage_replaces_every_launch_a_named_stage_issues_with_one_zeroing_allocation_node_at_the_same_point_in_the_captured_stream("gdn") {
        return one_zeroing_node_where_a_stages_launches_were(normed, normed.dims3()?);
    }
    let slot_of = |cache: &Qwen3MoeKvCache| -> Result<usize> {
        cache
            .lin_attn_slot_for_layer(li)
            .ok_or_else(|| anyhow::anyhow!("batch_gdn: layer {li} has no linear-attention slot"))
    };
    let rowwise =
        rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("gdnproj")
            && b > 1;
    if !rowwise {
        let mut states: Vec<&LinAttnState> = Vec::with_capacity(b);
        for cache in caches.iter().take(b) {
            let slot = slot_of(cache)?;
            match cache.lin_attn_states[slot].as_ref() {
                Some(st) if st.is_fused() => states.push(st),
                _ => anyhow::bail!(
                    "batch_gdn: lane state for layer {li} is not fused; \
                     ensure_all_lane_lin_states_fused must run before any batch step"
                ),
            }
        }
        if let Some(out) = la
            .forward_decode_fused_batch_lanes_projections_once_then_per_lane_step_kernels_prof(
                normed,
                &states,
                prof.as_mut(),
                lanes_conc,
            )?
        {
            return Ok(out);
        }
    }
    let mut rows: Vec<Tensor> = Vec::with_capacity(b);
    for (i, cache) in caches.iter().enumerate().take(b) {
        let slot = slot_of(cache)?;
        let st = cache.lin_attn_states[slot].as_ref().unwrap();
        let x_row = normed.narrow(1, i, 1)?.contiguous()?;
        let out = la.forward_decode_fused(&x_row, st)?.ok_or_else(|| {
            anyhow::anyhow!(
                "batch_gdn: fused decode refused a lane it was preconditioned for (layer {li})"
            )
        })?;
        rows.push(out);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    lap(prof, "b.gdn.rowwise_lanes");
    Ok(Tensor::cat(&refs, 1)?)
}

#[cfg(test)]
mod batch_prefill_chunk_policy_tests {
    use super::batch_prefill_chunk_tokens_explicit_setting_preserves_unset_one_shot as chunk;

    #[test]
    fn unset_disabled_or_invalid_setting_keeps_the_existing_one_shot_route() {
        for raw in [
            None,
            Some("0"),
            Some(""),
            Some(" "),
            Some("bad"),
            Some("-1"),
        ] {
            assert_eq!(chunk(raw, 168000), None, "raw={raw:?}");
        }
    }

    #[test]
    fn explicit_positive_chunk_only_changes_prompts_crossing_its_boundary() {
        for tokens in [0, 1, 4095, 4096] {
            assert_eq!(chunk(Some("4096"), tokens), None);
        }
        for tokens in [4097, 8192, 168000] {
            assert_eq!(chunk(Some("4096"), tokens), Some(4096));
        }
        assert_eq!(chunk(Some(" 1 "), 2), Some(1));
        assert_eq!(chunk(Some("168000"), 168000), None);
    }
}
