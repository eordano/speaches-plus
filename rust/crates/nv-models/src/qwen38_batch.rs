use super::*;
use crate::gemma4_batch_graph::BucketPlan;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::bf16;
use nv_kernels::graph::CudaGraphRunner;

pub fn nv_q38_batch_env_opt_in_nv_q38_batch_1_the_serving_loop_routes_batch_xor_spec_per_request_group(
) -> bool {
    std::env::var("NV_Q38_BATCH").ok().as_deref() == Some("1")
}

pub const Q38_BATCH_GRAPH_SHAPE_TOKEN_BASE_KEEPS_SOLO_KEY_1_AND_VERIFY_BASE_0X51000_UNTOUCHED: u64 =
    0x38b000;

pub const Q38_BATCH_DEFAULT_BUCKETS: &str = "1,2,4";

pub const A_PADDED_LANE_IS_CLOBBERED_AND_MUST_REPREFILL_BEFORE_REUSE_BECAUSE_THE_FLAT_FP8_CACHE_AND_GDN_STATE_ARE_BOUND_BY_POINTER_NOT_BY_TABLE:
    () = ();

pub fn rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin(g: &str) -> bool {
    std::env::var("NV_Q38_BATCH_ROWWISE")
        .map(|v| v.split(',').any(|t| t.trim() == g || t.trim() == "all"))
        .unwrap_or(false)
}

pub fn batch_prof_env_nv_q38_batch_prof_1_plus_nv_prof_gdn_1_plus_nv_graph_off_1_because_every_lap_syncs(
) -> bool {
    std::env::var("NV_Q38_BATCH_PROF").ok().as_deref() == Some("1")
        && std::env::var("NV_GRAPH_OFF").ok().as_deref() == Some("1")
}

pub fn batch_gemm_env_opt_in_nv_q38_batch_gemm_1_lane_concurrent_attn_gdn_mrow_ab_and_fp8_lt_arms(
) -> bool {
    std::env::var("NV_Q38_BATCH_GEMM").ok().as_deref() == Some("1")
}

pub const FP8_LT_MROW_MIN_LANES_4_BECAUSE_THE_MK_GEMV_HOLDS_ITS_ONE_WEIGHTS_READ_SHAPE_TO_M3_AND_COLLAPSES_TO_3X_BY_M8:
    usize = 4;

pub const Q38_BATCH_LANES_MAX_16_MIRRORING_THE_MK_KERNEL_FAMILY_TEMPLATE_ARM_CAP: usize = 16;

pub const Q38_BATCH_LANES_ROW_TWIN_CEILING_8_THE_SOLE_M9_TO_16_PROJECTION_ROUTE_IS_THE_FP8_LT_ROWQUANT_GEMM_THE_BATCH_GEMM_ARM_OWNS:
    usize = 8;

pub const Q38_BATCH_BOOT_HEADROOM_MIN_FREE_BYTES_512MB_BECAUSE_16_LANES_OF_8K_KV_PLUS_GDN_STATE_TAKE_3_4GB_AND_GRAPH_CAPTURE_STILL_NEEDS_POOL_PAGES:
    usize = 512 << 20;

pub fn batch_gemm_fp8lt_subarm_default_on_kill_nv_q38_batch_gemm_fp8lt_0_it_rowquants_activations_to_e4m3_unlike_the_bf16_exact_mk_gemv(
) -> bool {
    std::env::var("NV_Q38_BATCH_GEMM_FP8LT").ok().as_deref() != Some("0")
}

pub const LANE_CONCURRENT_ATTN_AND_GDN_ARE_BIT_IDENTICAL_TO_THE_SERIAL_LOOP_ONLY_THE_MROW_AB_GEMM_AND_FP8_LT_ROUTES_CHANGE_NUMERIC_ROUTE:
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
    logits_buf: CudaSlice<bf16>,
    vocab: usize,
    capture_supported: bool,
    capture_failed: bool,
    captures: u64,
    replays: u64,

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
            hd % 32 == 0 && hd <= 1024 && model.rope.cos().dtype() == DType::F32,
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

        let mut ctor_guard =
            crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
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
            .alloc_zeros::<bf16>(b_max * vocab)
            .map_err(|e| anyhow::anyhow!("alloc logits_buf: {e:?}"))?;
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
        Ok(Self {
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
            logits_buf,
            vocab,
            capture_supported,
            capture_failed: false,
            captures: 0,
            replays: 0,
            _err_drain: CtxErrDrain(dev.cuda_stream().context().clone()),
        })
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
        anyhow::ensure!(lane < self.caches.len(), "reset_lane: lane {lane} out of range");
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
        anyhow::ensure!(!prompt.is_empty(), "prefill_lane: empty prompt");
        anyhow::ensure!(lane < self.caches.len(), "prefill_lane: lane {lane} out of range");
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
        let logits = self.model.forward_with_cache_dispatched_rows(
            &tokens,
            &positions,
            &mut self.caches[lane],
            None,
            Some(1),
        )?;
        self.lane_pos[lane] = seq;
        let (_b, s, _v) = logits.dims3()?;
        let row: Vec<f32> = logits
            .narrow(1, s - 1, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok(row)
    }

    pub fn prime_lane_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
        &mut self,
        lane: usize,
        k_chunk: &Tensor,
        v_chunk: &Tensor,
    ) -> Result<()> {
        anyhow::ensure!(lane < self.caches.len(), "prime_lane: lane {lane} out of range");
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
        let Qwen38BatchLanes { model, caches, .. } = self;
        for cache in caches[..b].iter_mut() {
            model.ensure_fused_lin_states(cache)?;
        }
        Ok(())
    }

    pub fn step_batch(&mut self, tokens: &[Option<u32>]) -> Result<Vec<Option<Vec<f32>>>> {
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
            anyhow::anyhow!("step_batch: {need} lanes exceed max bucket {}", self.plan.max_bucket())
        })?;
        for (i, t) in tokens.iter().enumerate() {
            if t.is_some() {
                anyhow::ensure!(
                    self.lane_pos[i] + 1 <= self.caches[i].max_seq_len(),
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
            && std::env::var("NV_GRAPH_OFF").ok().as_deref() != Some("1")
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
        if batch_prof_env_nv_q38_batch_prof_1_plus_nv_prof_gdn_1_plus_nv_graph_off_1_because_every_lap_syncs() {
            nv_layers::linear_attn::gdn_step_prof::report_and_reset(self.lane_pos[0]);
        }
        let host: Vec<bf16> = self
            .stream
            .clone_dtoh(&self.logits_buf)
            .map_err(|e| anyhow::anyhow!("dtoh batch logits: {e:?}"))?;
        let mut out = Vec::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            out.push(t.map(|_| {
                host[i * self.vocab..(i + 1) * self.vocab]
                    .iter()
                    .map(|x| x.to_f32())
                    .collect()
            }));
        }
        Ok(out)
    }

    fn step_batch_eager(&mut self, b_bucket: usize) -> Result<()> {
        let stream = self.stream.clone();
        let Qwen38BatchLanes {
            model,
            caches,
            token_buf,
            host_tok,
            logits_buf,
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
            logits_buf,
            *vocab,
            b_bucket,
            lane_streams,
        )
    }

    fn step_batch_captured(&mut self, b_bucket: usize) -> Result<bool> {
        use anyhow::Context as _;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let key =
            Q38_BATCH_GRAPH_SHAPE_TOKEN_BASE_KEEPS_SOLO_KEY_1_AND_VERIFY_BASE_0X51000_UNTOUCHED
                + b_bucket as u64;
        disable_event_tracking_at_first_capture(&dev);
        let need_warm = !self.runner.has_cached_token(key);
        if need_warm {
            if let Err(e) = self.runner.probe_capture() {
                self.capture_failed = true;
                eprintln!(
                    "[qwen38_batch] graph capture unavailable; batching uncaptured: {e:#}"
                );
                self.step_batch_eager(b_bucket)?;
                return Ok(false);
            }
        }
        let stream = self.stream.clone();
        let Qwen38BatchLanes {
            model,
            caches,
            runner,
            token_buf,
            host_tok,
            logits_buf,
            vocab,
            lane_streams_index_0_is_the_capture_stream_so_lane_i_maps_to_slot_i: lane_streams,
            ..
        } = self;
        let vocab = *vocab;
        if need_warm {
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
                logits_buf,
                vocab,
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
        }
        let result = runner.run(key, |s| {
            run_batch_step_body(
                s,
                model,
                &mut caches[..b_bucket],
                token_buf,
                &host_tok[..],
                logits_buf,
                vocab,
                b_bucket,
                lane_streams,
            )
        });
        match result {
            Ok(()) => {
                if need_warm {
                    self.captures += 1;
                } else {
                    self.replays += 1;
                }
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
    if std::env::var("NV_GRAPH_KEEP_EVENT_TRACKING").ok().as_deref() == Some("1") {
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

#[allow(clippy::too_many_arguments)]
fn run_batch_step_body(
    s: &Arc<CudaStream>,
    model: &Qwen3Moe,
    caches: &mut [Qwen3MoeKvCache],
    token_buf: &mut CudaSlice<u32>,
    host_tok: &[u32],
    logits_buf: &mut CudaSlice<bf16>,
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
            logits_buf,
            vocab,
            b,
            lane_streams,
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn batch_step_body_inner(
    s: &Arc<CudaStream>,
    model: &Qwen3Moe,
    caches: &mut [Qwen3MoeKvCache],
    token_buf: &mut CudaSlice<u32>,
    host_tok: &[u32],
    logits_buf: &mut CudaSlice<bf16>,
    vocab: usize,
    b: usize,
    lane_streams: &[Arc<CudaStream>],
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

    let mut prof: BatchProf =
        nv_layers::linear_attn::gdn_step_prof::SyncLaps::begin_if_armed(&dev);

    let gemm_arm = batch_gemm_env_opt_in_nv_q38_batch_gemm_1_lane_concurrent_attn_gdn_mrow_ab_and_fp8_lt_arms();
    let lanes_conc: Option<&[Arc<CudaStream>]> = if gemm_arm && b > 1 && lane_streams.len() >= b {
        Some(&lane_streams[..b])
    } else {
        None
    };
    let _fp8_lt_mrow_scope = nv_layers::linear::VerifyTcFp8LtGemmScopeGuard::enter_if(
        gemm_arm
            && batch_gemm_fp8lt_subarm_default_on_kill_nv_q38_batch_gemm_fp8lt_0_it_rowquants_activations_to_e4m3_unlike_the_bf16_exact_mk_gemv()
            && b >= FP8_LT_MROW_MIN_LANES_4_BECAUSE_THE_MK_GEMV_HOLDS_ITS_ONE_WEIGHTS_READ_SHAPE_TO_M3_AND_COLLAPSES_TO_3X_BY_M8,
    );

    let hidden_size = model.config().hidden_size;
    let x = match embed_lookup_bf16(&model.embed_weight, &tokens_t)? {
        Some(rows) => rows,
        None => model.embed_weight.index_select(&tokens_t, 0)?,
    }
    .reshape((1usize, b, hidden_size))?
    .to_dtype(model.dtype())?;
    lap(&mut prof, "b.embed");

    let layers = &model.layers;
    let rowwise_norms = rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("norms");
    let rowwise_ffn = rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("ffn");
    let rowwise_head = rowwise_group_env_nv_q38_batch_rowwise_per_row_m1_twins_set_ffn_for_bit_exactness_because_the_nvfp4_mlp_m_row_route_is_the_sole_non_twin("head");
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
    let ffn_rows = |ffn: &LayerFfn, x: &Tensor| -> Result<Tensor> {
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
            LayerMixer::Linear(la) => {
                batch_gdn(la, li, &normed, caches, b, &mut prof, lanes_conc)?
            }
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
    let (x, logits) = if rowwise_head && b > 1 {
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
    let lc = logits.contiguous()?;
    let (ls, ll) = lc.storage_and_layout();
    let lcuda = match &*ls {
        candle_core::Storage::Cuda(c) => c,
        _ => anyhow::bail!("batch logits must be on CUDA"),
    };
    let lsl = lcuda.as_cuda_slice::<bf16>()?;
    let n = b * vocab;
    let off = ll.start_offset();
    let src = lsl.slice(off..off + n);
    let mut dst = logits_buf.slice_mut(0..n);
    s.memcpy_dtod(&src, &mut dst)
        .map_err(|e| anyhow::anyhow!("dtod batch logits: {e:?}"))?;
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
    let (q_raw, k_raw, v_raw, o_rowwise) = if rowwise {
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
    let gated = if let Some(ls) = conc {
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
    let out = if o_rowwise {
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
                cache.decode_attention_fp8(slot, &q_final, n_heads, scaling)?;
            let mut copy_into = |src: &Tensor,
                                 dst: &mut CudaSlice<bf16>|
             -> Result<()> {
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
    let slot_of = |cache: &Qwen3MoeKvCache| -> Result<usize> {
        cache.lin_attn_slot_for_layer(li).ok_or_else(|| {
            anyhow::anyhow!("batch_gdn: layer {li} has no linear-attention slot")
        })
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
