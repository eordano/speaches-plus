use super::qwen38_batch::{
    DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED,
    DEVICE_ARGMAX_RETURNS_TOKEN_0_ON_AN_ALL_NONFINITE_ROW_WHERE_HOST_SAMPLING_REPORTS_EXHAUSTED,
};
use super::*;
use crate::gemma4_batch_graph::BucketPlan;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use nv_kernels::graph::CudaGraphRunner;

pub const Q36_BATCH_GRAPH_SHAPE_TOKEN_BASE_KEEPS_SOLO_KEY_1_VERIFY_BASE_0X51000_AND_Q38_BATCH_BASE_0X38B000_UNTOUCHED:
    u64 = 0x36b000;

pub const Q36_BATCH_DEFAULT_BUCKETS: &str = "1,2,4";

pub fn nv_q36_batch_env_opt_in_nv_q36_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group(
) -> bool {
    nv_config::env_arm_opt_in_eq_some_1_reread_every_call("NV_Q36_BATCH")
}

pub const Q36_BATCH_LANES_MAX_8_THE_ROUTE_FUSE_ONE_CTA_PER_TOKEN_CAP_AND_THE_MK_PROJECTION_ROW_TWIN_FAMILY_BOTH_STOP_THERE:
    usize = 8;

pub const Q36_BATCH_BOOT_HEADROOM_MIN_FREE_BYTES_512MB_BECAUSE_GRAPH_CAPTURE_STILL_NEEDS_POOL_PAGES_ON_TOP_OF_LANE_KV_AND_GDN_STATE:
    usize = 512 << 20;

pub const A_PADDED_LANE_IS_CLOBBERED_AND_MUST_REPREFILL_BEFORE_REUSE_BECAUSE_THE_FLAT_FP8_CACHE_AND_GDN_STATE_ARE_BOUND_BY_POINTER_NOT_BY_TABLE:
    () = ();

pub const PER_LANE_KV_IS_ONE_QWEN3MOEKVCACHE_PER_LANE_THE_QWEN38_LANE_POOL_PATTERN_EACH_CACHE_OWNS_ITS_FLAT_FP8_SLABS_AND_GDN_STATES_SO_NO_PAGED_POOL_IS_NEEDED_AT_THESE_BUCKETS:
    () = ();

pub const ROWS_B_GROUPED_MOE_IS_THE_SOLE_NON_TWIN_ROUTE_AT_B_LE_4_ROWWISE_MOE_RESTORES_BYTE_CLASS_THERE_WHILE_AT_B8_NON_FFN_M_ROW_ROUTES_ALSO_DRIFT_PER_THE_CAD219D1F_RESCOPE:
    () = ();

pub const THE_MOE_HOOK_IS_BUILT_ONCE_AND_REUSED_ACROSS_EVERY_RESET_AND_GRAPH_TEARDOWN_BECAUSE_DROPPING_A_DISPATCH_WHOSE_FOLDED_WEIGHTS_DESTROYED_GRAPHS_BAKED_POISONS_THE_CONTEXT:
    () = ();

pub fn rowwise_group_env_nv_q36_batch_rowwise_per_row_m1_twins_the_moe_group_is_the_bit_exactness_lever_because_the_rows_b_grouped_route_is_the_sole_non_twin_at_b_le_4(
    g: &str,
) -> bool {
    std::env::var("NV_Q36_BATCH_ROWWISE")
        .map(|v| v.split(',').any(|t| t.trim() == g || t.trim() == "all"))
        .unwrap_or(false)
}

pub fn q36_batch_bucket_plan_env_nv_q36_batch_sizes() -> BucketPlan {
    BucketPlan::parse(Some(
        std::env::var("NV_Q36_BATCH_SIZES")
            .ok()
            .as_deref()
            .unwrap_or(Q36_BATCH_DEFAULT_BUCKETS),
    ))
}

pub struct Qwen36BatchLanes {
    model: Qwen3Moe,
    moe_hook_built_once_in_new_and_never_rebuilt: Box<GroupedMoeDispatch>,
    caches: Vec<Qwen3MoeKvCache>,
    lane_pos: Vec<usize>,
    plan: BucketPlan,
    device: Device,
    stream: Arc<CudaStream>,
    runner: CudaGraphRunner,
    token_buf: CudaSlice<u32>,
    host_tok: Box<[u32]>,
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

impl Qwen36BatchLanes {
    pub fn new(
        model: Qwen3Moe,
        device: &Device,
        max_seq_len: usize,
        plan: BucketPlan,
    ) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("Qwen36BatchLanes requires a CUDA device"),
        };
        anyhow::ensure!(
            !model.is_dense(),
            "Qwen36BatchLanes batches the routed-MoE trunk; the dense trunk batches on \
             Qwen38BatchLanes"
        );
        anyhow::ensure!(
            model.dtype() == DType::BF16,
            "Qwen36BatchLanes: trunk dtype must be BF16, got {:?}",
            model.dtype()
        );
        anyhow::ensure!(
            nv_q38_fused_qkv_prep_env_kill_switch_nv_q38_fused_qkv_0(),
            "Qwen36BatchLanes: NV_Q38_FUSED_QKV=0 disables the fused qkv+rope+store decode arm \
             the per-lane batch attention is built from"
        );
        let cfg = model.config();
        let hd = cfg.head_dim;
        anyhow::ensure!(
            hd.is_multiple_of(32) && hd <= 1024 && model.rope.cos().dtype() == DType::F32,
            "Qwen36BatchLanes: fused qkv decode geometry contract broken (head_dim {hd})"
        );
        for (li, layer) in model.layers.iter().enumerate() {
            anyhow::ensure!(
                layer.ffn.as_moe().is_some(),
                "Qwen36BatchLanes: layer {li} ffn is {}, routed MoE required",
                layer.ffn.label()
            );
            if let LayerMixer::Linear(la) = &layer.mixer {
                anyhow::ensure!(
                    la.fused_decode_supported(),
                    "Qwen36BatchLanes: layer {li} linear-attn fused decode unsupported; the \
                     non-fused recurrent path is neither capture-safe nor lane-splittable"
                );
            }
        }
        let b_max = plan.max_bucket();
        anyhow::ensure!(
            b_max <= Q36_BATCH_LANES_MAX_8_THE_ROUTE_FUSE_ONE_CTA_PER_TOKEN_CAP_AND_THE_MK_PROJECTION_ROW_TWIN_FAMILY_BOTH_STOP_THERE,
            "Qwen36BatchLanes: bucket {b_max} exceeds the 8-row cap (route-fuse one-CTA-per-token \
             and the mk projection row-twin family both stop at 8 rows)"
        );

        let moe_hook = Box::new(GroupedMoeDispatch::from_model(&model)?);
        let () = THE_MOE_HOOK_IS_BUILT_ONCE_AND_REUSED_ACROSS_EVERY_RESET_AND_GRAPH_TEARDOWN_BECAUSE_DROPPING_A_DISPATCH_WHOSE_FOLDED_WEIGHTS_DESTROYED_GRAPHS_BAKED_POISONS_THE_CONTEXT;

        let dev_stream = dev.cuda_stream();
        let capture_supported = !dev_stream.cu_stream().is_null();
        let stream = dev_stream;
        disable_event_tracking_at_first_capture(&dev);

        nv_quant::nvfp4::ensure_workspace_for_stream(&stream)?;
        let _ = nv_quant::matmul::TensorCoreGemm::new(stream.clone())?;

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
                free >= Q36_BATCH_BOOT_HEADROOM_MIN_FREE_BYTES_512MB_BECAUSE_GRAPH_CAPTURE_STILL_NEEDS_POOL_PAGES_ON_TOP_OF_LANE_KV_AND_GDN_STATE,
                "Qwen36BatchLanes: {b_max} lanes leave {:.2} GiB of {:.2} GiB free, under the \
                 boot headroom floor",
                free as f64 / (1u64 << 30) as f64,
                total as f64 / (1u64 << 30) as f64,
            );
        }
        let runner = CudaGraphRunner::new(stream.clone());
        let mut built = Self {
            model,
            moe_hook_built_once_in_new_and_never_rebuilt: moe_hook,
            caches,
            lane_pos: vec![0usize; b_max],
            plan,
            device: device.clone(),
            stream,
            runner,
            token_buf,
            host_tok: vec![0u32; b_max].into_boxed_slice(),
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
                "NV_Q36_BATCH_BOOT_WARM",
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

    pub fn moe_dispatch_ref(&self) -> &dyn MoeDispatch {
        &*self.moe_hook_built_once_in_new_and_never_rebuilt
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
        let last = self.prefill_lane_forward_keep_last_row_on_device(lane, prompt, false)?;
        let row: Vec<f32> = last.flatten_all()?.to_vec1()?;
        Ok(row)
    }

    pub fn prefill_lane_hooked_matching_the_eager_serving_prefill_route_so_a_group_lane_and_its_solo_reference_share_prefill_bytes(
        &mut self,
        lane: usize,
        prompt: &[u32],
    ) -> Result<Vec<f32>> {
        let last = self.prefill_lane_forward_keep_last_row_on_device(lane, prompt, true)?;
        let row: Vec<f32> = last.flatten_all()?.to_vec1()?;
        Ok(row)
    }

    pub fn prefill_lane_hooked_greedy_token_on_device_byte_exact_vs_host_sampling_by_the_widening_and_tie_break_constants(
        &mut self,
        lane: usize,
        prompt: &[u32],
    ) -> Result<u32> {
        let () = DEVICE_ARGMAX_IS_BYTE_EXACT_VS_HOST_SAMPLING_BECAUSE_BF16_TO_F32_WIDENS_EXACTLY_AND_ARGMAX_F32_ROWS_SKIPS_NONFINITE_AND_BREAKS_TIES_LOWEST_INDEX_LIKE_ARGMAX_CHECKED;
        let () = DEVICE_ARGMAX_RETURNS_TOKEN_0_ON_AN_ALL_NONFINITE_ROW_WHERE_HOST_SAMPLING_REPORTS_EXHAUSTED;
        let last = self.prefill_lane_forward_keep_last_row_on_device(lane, prompt, true)?;
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
        route_moe_through_the_installed_hook_like_the_eager_serving_prefill: bool,
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
        let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &self.device)?;
        let positions_v: Vec<i32> = (0..seq as i32).collect();
        let positions = Tensor::from_vec(positions_v, seq, &self.device)?;
        let Qwen36BatchLanes {
            model,
            moe_hook_built_once_in_new_and_never_rebuilt: moe_hook,
            caches,
            lane_pos,
            ..
        } = self;
        let either_way_the_installed_hook_stays_untouched_no_take_restore_needed_because_nothing_here_drops_it:
            Option<&dyn MoeDispatch> =
            if route_moe_through_the_installed_hook_like_the_eager_serving_prefill {
                Some(&**moe_hook)
            } else {
                None
            };
        let logits = model.forward_with_cache_dispatched_rows(
            &tokens,
            &positions,
            &mut caches[lane],
            either_way_the_installed_hook_stays_untouched_no_take_restore_needed_because_nothing_here_drops_it,
            Some(1),
        )?;
        lane_pos[lane] = seq;
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
        if self.runner.has_cached() {
            self.invalidate_graphs_synced_because_eager_work_after_capture_faults_like_the_solo_engines_prefill()?;
        }
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

    fn ensure_lane_lin_states_fused_rerun_each_step_because_reset_drops_them_and_early_returns_make_it_cheap(
        &mut self,
        b: usize,
    ) -> Result<()> {
        let Qwen36BatchLanes { model, caches, .. } = self;
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

    fn step_batch_run_common(&mut self, tokens: &[Option<u32>]) -> Result<usize> {
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
        self.synchronize()?;
        Ok(b_bucket)
    }

    fn step_batch_eager(&mut self, b_bucket: usize) -> Result<()> {
        let stream = self.stream.clone();
        let Qwen36BatchLanes {
            model,
            moe_hook_built_once_in_new_and_never_rebuilt: moe_hook,
            caches,
            token_buf,
            host_tok,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            ..
        } = self;
        run_batch_step_body(
            &stream,
            model,
            &**moe_hook,
            &mut caches[..b_bucket],
            token_buf,
            &host_tok[..],
            logits_buf,
            ArgmaxScratch {
                val: amax_val,
                idx: amax_idx,
                out: amax_out,
            },
            *vocab,
            b_bucket,
        )
    }

    fn eager_warm_pass_with_lin_state_snapshot_restore_to_materialize_lazy_inits_before_capture(
        &mut self,
        b_bucket: usize,
    ) -> Result<()> {
        use anyhow::Context as _;
        let stream = self.stream.clone();
        let Qwen36BatchLanes {
            model,
            moe_hook_built_once_in_new_and_never_rebuilt: moe_hook,
            caches,
            token_buf,
            host_tok,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            ..
        } = self;
        let mut snaps = Vec::with_capacity(b_bucket);
        for cache in caches[..b_bucket].iter() {
            snaps.push(cache.snapshot_lin_states()?);
        }
        run_batch_step_body(
            &stream,
            model,
            &**moe_hook,
            &mut caches[..b_bucket],
            token_buf,
            &host_tok[..],
            logits_buf,
            ArgmaxScratch {
                val: &mut *amax_val,
                idx: &mut *amax_idx,
                out: &mut *amax_out,
            },
            *vocab,
            b_bucket,
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
        let Qwen36BatchLanes {
            model,
            moe_hook_built_once_in_new_and_never_rebuilt: moe_hook,
            caches,
            runner,
            token_buf,
            host_tok,
            logits_buf,
            amax_val,
            amax_idx,
            amax_out,
            vocab,
            ..
        } = self;
        let vocab = *vocab;
        runner.run(key, |s| {
            run_batch_step_body(
                s,
                model,
                &**moe_hook,
                &mut caches[..b_bucket],
                token_buf,
                &host_tok[..],
                logits_buf,
                ArgmaxScratch {
                    val: amax_val,
                    idx: amax_idx,
                    out: amax_out,
                },
                vocab,
                b_bucket,
            )
        })
    }

    fn step_batch_captured(&mut self, b_bucket: usize) -> Result<bool> {
        use anyhow::Context as _;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let key =
            Q36_BATCH_GRAPH_SHAPE_TOKEN_BASE_KEEPS_SOLO_KEY_1_VERIFY_BASE_0X51000_AND_Q38_BATCH_BASE_0X38B000_UNTOUCHED
                + b_bucket as u64;
        disable_event_tracking_at_first_capture(&dev);
        let need_capture = !self.runner.has_cached_token(key);
        if need_capture {
            if let Err(e) = self.runner.probe_capture() {
                self.capture_failed = true;
                eprintln!("[qwen36_batch] graph capture unavailable; batching uncaptured: {e:#}");
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
                     rerun with NV_GRAPH_OFF=1 (the MoE hook stays installed and reused)",
                )
            }
        }
    }
}

impl Drop for Qwen36BatchLanes {
    fn drop(&mut self) {
        let td = crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(
            &self.stream,
        );
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
    moe_hook: &dyn MoeDispatch,
    caches: &mut [Qwen3MoeKvCache],
    token_buf: &mut CudaSlice<u32>,
    host_tok: &[u32],
    logits_buf: &mut CudaSlice<f32>,
    amax: ArgmaxScratch<'_>,
    vocab: usize,
    b: usize,
) -> Result<()> {
    nv_layers::cuda_stream::with_stream(s.clone(), || {
        batch_step_body_inner(
            s, model, moe_hook, caches, token_buf, host_tok, logits_buf, amax, vocab, b,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn batch_step_body_inner(
    s: &Arc<CudaStream>,
    model: &Qwen3Moe,
    moe_hook: &dyn MoeDispatch,
    caches: &mut [Qwen3MoeKvCache],
    token_buf: &mut CudaSlice<u32>,
    host_tok: &[u32],
    logits_buf: &mut CudaSlice<f32>,
    amax: ArgmaxScratch<'_>,
    vocab: usize,
    b: usize,
) -> Result<()> {
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

    let hidden_size = model.config().hidden_size;
    let x = match embed_lookup_bf16(&model.embed_weight, &tokens_t)? {
        Some(rows) => rows,
        None => model.embed_weight.index_select(&tokens_t, 0)?,
    }
    .reshape((1usize, b, hidden_size))?
    .to_dtype(model.dtype())?;

    let layers = &model.layers;
    let rowwise = |g: &str| {
        rowwise_group_env_nv_q36_batch_rowwise_per_row_m1_twins_the_moe_group_is_the_bit_exactness_lever_because_the_rows_b_grouped_route_is_the_sole_non_twin_at_b_le_4(g)
    };
    let rowwise_norms = rowwise("norms");
    let rowwise_moe = rowwise("moe");
    let rowwise_head = rowwise("head");
    let rowwise_attnproj = rowwise("attnproj");
    let norm_rows = |n: &RmsNorm, x: &Tensor| -> Result<Tensor> {
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
    let moe_rows = |li: usize, ffn: &LayerFfn, x: &Tensor| -> Result<Tensor> {
        let moe = ffn.as_moe().ok_or_else(|| {
            anyhow::anyhow!(
                "qwen36 batch: layer {li} ffn is {}, routed MoE required",
                ffn.label()
            )
        })?;
        if !rowwise_moe || b == 1 {
            let () = ROWS_B_GROUPED_MOE_IS_THE_SOLE_NON_TWIN_ROUTE_AT_B_LE_4_ROWWISE_MOE_RESTORES_BYTE_CLASS_THERE_WHILE_AT_B8_NON_FFN_M_ROW_ROUTES_ALSO_DRIFT_PER_THE_CAD219D1F_RESCOPE;
            return moe_hook.forward(li, moe, x);
        }
        let mut rows = Vec::with_capacity(b);
        for i in 0..b {
            rows.push(moe_hook.forward(li, moe, &x.narrow(1, i, 1)?.contiguous()?)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Ok(Tensor::cat(&refs, 1)?)
    };
    let mut residual = x.clone();
    let mut normed = norm_rows(&layers[0].pre_norm, &x)?;
    for (li, layer) in layers.iter().enumerate() {
        let mixed = match &layer.mixer {
            LayerMixer::Full(attn) => {
                batch_full_attention(model, attn, li, &normed, caches, b, rowwise_attnproj)?
            }
            LayerMixer::Linear(la) => batch_gdn(la, li, &normed, caches, b)?,
        };
        let (normed_post, residual_after_attn) =
            norm_residual_rows(&layer.post_norm, &mixed, &residual)?;
        let ffn_out = moe_rows(li, &layer.ffn, &normed_post)?;
        if li + 1 < layers.len() {
            let (normed_next, residual_next) =
                norm_residual_rows(&layers[li + 1].pre_norm, &ffn_out, &residual_after_attn)?;
            normed = normed_next;
            residual = residual_next;
        } else {
            residual = residual_after_attn.add(&ffn_out)?;
        }
    }
    let logits = if rowwise_head && b > 1 {
        let mut rows = Vec::with_capacity(b);
        for i in 0..b {
            let xr = model.final_norm.forward(&residual.narrow(1, i, 1)?)?;
            rows.push(model.lm_head.forward(&xr)?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::cat(&refs, 1)?
    } else {
        let x = model.final_norm.forward(&residual)?;
        model.lm_head.forward(&x)?
    };
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
    {
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
    }
    Ok(())
}

fn proj_rows(lin: &Linear, x: &Tensor, b: usize) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(b);
    for i in 0..b {
        rows.push(lin.forward(&x.narrow(1, i, 1)?.contiguous()?)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Tensor::cat(&refs, 1)?)
}

#[allow(clippy::too_many_arguments)]
fn batch_full_attention(
    model: &Qwen3Moe,
    attn: &AttentionLayer,
    li: usize,
    normed: &Tensor,
    caches: &mut [Qwen3MoeKvCache],
    b: usize,
    rowwise_attnproj: bool,
) -> Result<Tensor> {
    let rowwise = rowwise_attnproj && b > 1;
    let (q_raw, k_raw, v_raw) = if rowwise {
        (
            proj_rows(&attn.q_proj, normed, b)?,
            proj_rows(&attn.k_proj, normed, b)?,
            proj_rows(&attn.v_proj, normed, b)?,
        )
    } else {
        (
            attn.q_proj.forward(normed)?,
            attn.k_proj.forward(normed)?,
            attn.v_proj.forward(normed)?,
        )
    };
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
    let mut rows: Vec<Tensor> = Vec::with_capacity(b);
    for (i, cache) in caches.iter_mut().enumerate().take(b) {
        let slot = cache.full_slot_for_layer(li).ok_or_else(|| {
            anyhow::anyhow!("batch_full_attention: layer {li} has no full-attention cache slot")
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
    let gated = Tensor::cat(&refs, 1)?;
    let out = if rowwise {
        proj_rows(&attn.o_proj, &gated, b)?
    } else {
        attn.o_proj.forward(&gated)?
    };
    Ok(out)
}

fn batch_gdn(
    la: &LinearAttention,
    li: usize,
    normed: &Tensor,
    caches: &mut [Qwen3MoeKvCache],
    b: usize,
) -> Result<Tensor> {
    let slot_of = |cache: &Qwen3MoeKvCache| -> Result<usize> {
        cache
            .lin_attn_slot_for_layer(li)
            .ok_or_else(|| anyhow::anyhow!("batch_gdn: layer {li} has no linear-attention slot"))
    };
    let rowwise =
        rowwise_group_env_nv_q36_batch_rowwise_per_row_m1_twins_the_moe_group_is_the_bit_exactness_lever_because_the_rows_b_grouped_route_is_the_sole_non_twin_at_b_le_4("gdnproj")
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
                normed, &states, None, None,
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
    Ok(Tensor::cat(&refs, 1)?)
}
