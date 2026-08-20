use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::gemma4_batch_graph::graph_teardown::GraphTeardown;
use crate::qwen3_5_moe::{GroupedMoeDispatch, MoeDispatch, Qwen3Moe, Qwen3MoeKvCache};

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use nv_kernels::graph::CudaGraphRunner;
use std::sync::Arc;

pub const THE_EAGER_ALL_NAN_IS_INSTALL_GROUPED_MOE_NOT_THIS_DISABLE_AND_NEEDS_A_HOST_STALL: &str =
    "This engine allocates token_buf/pos_buf/logits_buf and nv_quant's 64 MiB nvfp4 workspace with \
     cudarc event tracking still on, and disables it only when a decode is about to be captured. \
     That is late by the letter of disable_event_tracking_before_capture, and it stays late \
     because WHERE IT BELONGS IS CURRENTLY UNMEASURABLE, not because moving it is known to be \
     unsafe. The 248320-of-248320 all-NaN eager prefill once attributed to hoisting this call is \
     produced by install_grouped_moe alone, with event tracking ON for the whole process, no \
     capture taken and this disable never reached: on RedHatAI/Qwen3.6-35B-A3B-NVFP4 with \
     candle_core::Device::new_cuda_with_stream, qwen36_chat_validate arm `plain` answers `The \
     capital of France is Paris.` with 0 of 248320 prefill logits NaN, while `grouped+unrouted` \
     (capture_active=false) and `grouped+routed` both return 248320 of 248320 NaN -- three runs of \
     qwen36_chat_validate and four of qwen_graph_reset_bisect, one per event-tracking policy, all \
     NaN including the policy that never touches the flag. NV_MOE_DECODE_PROF=1, whose only effect \
     is a stream synchronize between the stages of nv_layers::moe_grouped::forward_grouped_decode, \
     makes `grouped+unrouted` emit the same seven token ids as `plain`. So the eager path depends \
     on a host stall inside the grouped MoE, which is the grouped-MoE defect and not this call, \
     and this engine's capture arm cannot discriminate any placement of this disable until \
     forward_grouped_decode returns finite logits without that stall. When it can: the placement \
     to try is a SCOPED disable rather than a hoist, because cudarc's flag lives on the \
     CudaContext and only decides whether slices allocated WHILE IT IS OFF carry read/write \
     CudaEvents -- disabling it around this engine's own allocations and re-enabling immediately \
     after yields event-free capture buffers while every candle temporary the eager path allocates \
     afterwards still gets its events";

pub struct GraphedQwen3Moe {
    model: Qwen3Moe,
    cache: Qwen3MoeKvCache,
    device: Device,
    stream: Arc<CudaStream>,
    runner: CudaGraphRunner,
    token_buf: CudaSlice<u32>,
    pos_buf: CudaSlice<i32>,
    host_tok: Box<[u32; 1]>,
    host_pos: Box<[i32; 1]>,
    logits_buf: CudaSlice<half::bf16>,
    moe_dispatch: Option<Box<dyn MoeDispatch>>,
    current_pos: usize,
    call_count: u64,
    capture_active: bool,
    capture_failed: bool,
    capture_supported: bool,
    owns_stream: bool,
    device_routing: bool,
    force_capture_env: bool,
    verify: Option<VerifyLane>,
    pending_verify_pos_and_rows: Option<(usize, usize)>,

    _err_drain: CtxErrDrain,
}

struct CtxErrDrain(Arc<CudaContext>);

impl Drop for CtxErrDrain {
    fn drop(&mut self) {
        let _ = self.0.check_err();
    }
}

const VERIFY_GRAPH_SHAPE_TOKEN_BASE_KEEPS_THE_M1_DECODE_KEY_1_UNTOUCHED: u64 = 0x51000;

struct VerifyLane {
    rows: usize,
    token_buf: CudaSlice<u32>,
    pos_buf: CudaSlice<i32>,
    host_tok: Box<[u32]>,
    host_pos: Box<[i32]>,
    logits_buf: CudaSlice<half::bf16>,
    hidden_buf: CudaSlice<half::bf16>,
}

impl GraphedQwen3Moe {
    pub fn new(model: Qwen3Moe, device: &Device, max_seq_len: usize) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("GraphedQwen3Moe requires a CUDA device"),
        };
        let mut cache = model.new_kv_cache(max_seq_len)?;
        cache.set_fused_lin_attn(true);

        let dev_stream = dev.cuda_stream();
        let capture_supported = !dev_stream.cu_stream().is_null();
        let mut ctor_guard = crate::gemma4_batch_graph::graph_teardown::CtorForkGuard::new();
        let (stream, owns_stream) = if capture_supported {
            (dev_stream, false)
        } else {
            let forked = ctor_guard
                .fork(dev_stream.context())
                .map_err(|e| anyhow::anyhow!("forked stream: {e:?}"))?;
            (forked, true)
        };

        nv_quant::nvfp4::ensure_workspace_for_stream(&stream)?;
        let _ = nv_quant::matmul::TensorCoreGemm::new(stream.clone())?;

        let token_buf = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!("alloc token_buf: {e:?}"))?;
        let pos_buf = stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!("alloc pos_buf: {e:?}"))?;
        let logits_buf = stream
            .alloc_zeros::<half::bf16>(model.vocab_size())
            .map_err(|e| anyhow::anyhow!("alloc logits_buf: {e:?}"))?;
        stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after alloc: {e:?}"))?;

        let runner = CudaGraphRunner::new(stream.clone());
        let force_capture_env = std::env::var("NV_GRAPH_CAPTURE_FORWARD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        ctor_guard.the_built_engine_owns_teardown_now();
        Ok(Self {
            model,
            cache,
            device: device.clone(),
            stream,
            runner,
            token_buf,
            pos_buf,
            host_tok: Box::new([0u32; 1]),
            host_pos: Box::new([0i32; 1]),
            logits_buf,
            moe_dispatch: None,
            current_pos: 0,
            call_count: 0,
            capture_active: false,
            capture_failed: false,
            capture_supported,
            owns_stream,
            device_routing: false,
            force_capture_env,
            verify: None,
            pending_verify_pos_and_rows: None,
            _err_drain: CtxErrDrain(dev.cuda_stream().context().clone()),
        })
    }

    pub fn underlying(&self) -> &Qwen3Moe {
        &self.model
    }

    pub fn moe_dispatch_ref(&self) -> Option<&dyn MoeDispatch> {
        self.moe_dispatch.as_deref()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn vocab_size(&self) -> usize {
        self.model.vocab_size()
    }

    pub fn cache(&self) -> &Qwen3MoeKvCache {
        &self.cache
    }

    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    pub fn call_count(&self) -> u64 {
        self.call_count
    }

    pub fn capture_active(&self) -> bool {
        self.capture_active
    }

    pub fn captured_graph_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn device_routing(&self) -> bool {
        self.device_routing
    }

    pub fn set_device_routing(&mut self, on: bool) {
        self.device_routing = on;
        if !on {
            self.runner.invalidate();
            self.capture_active = false;
        }
    }

    pub fn set_moe_dispatch(&mut self, hook: Option<Box<dyn MoeDispatch>>) {
        self.moe_dispatch = hook;
        self.runner.invalidate();
        self.capture_active = false;
    }

    pub fn dense_trunk_needs_no_moe_dispatch_for_capture_because_every_ffn_is_a_plain_mlp(
        &self,
    ) -> bool {
        self.model.is_dense()
    }

    pub fn install_grouped_moe(&mut self) -> Result<()> {
        if self.dense_trunk_needs_no_moe_dispatch_for_capture_because_every_ffn_is_a_plain_mlp() {
            self.set_moe_dispatch(None);
            self.set_device_routing(true);
            self.capture_failed = false;
            return Ok(());
        }
        let hook = GroupedMoeDispatch::from_model(&self.model)?;
        self.set_moe_dispatch(Some(Box::new(hook)));
        self.set_device_routing(true);
        self.capture_failed = false;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<()> {
        self.invalidate_graphs_synced()?;
        self.cache.reset();
        self.current_pos = 0;
        self.synchronize()
    }

    fn return_graph_mempool_reserved_pages(&self) {
        let ordinal = self.stream.context().ordinal() as i32;
        if let Ok(devh) = cudarc::driver::result::device::get(ordinal) {
            let _ = unsafe { cudarc::driver::sys::cuDeviceGraphMemTrim(devh) };
        }
    }

    fn invalidate_graphs_synced(&mut self) -> Result<()> {
        let had_graphs = self.runner.has_cached();
        if had_graphs {
            self.synchronize()?;
        }
        self.runner.invalidate();
        self.capture_active = false;
        if had_graphs {
            self.synchronize()?;
            self.return_graph_mempool_reserved_pages();
            self.synchronize()?;
        }
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("capture stream sync: {e:?}"))?;

        if let Device::Cuda(d) = &self.device {
            d.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("default stream sync: {e:?}"))?;
        }
        Ok(())
    }

    pub fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(!prompt.is_empty(), "prefill: empty prompt");
        self.invalidate_graphs_synced()?;
        anyhow::ensure!(
            self.current_pos + prompt.len() <= self.cache.max_seq_len(),
            "prefill: {} + {} tokens exceed cache max_seq_len {}",
            self.current_pos,
            prompt.len(),
            self.cache.max_seq_len()
        );
        let seq = prompt.len();
        let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &self.device)?;
        let positions_v: Vec<i32> =
            (self.current_pos as i32..(self.current_pos + seq) as i32).collect();
        let positions = Tensor::from_vec(positions_v, seq, &self.device)?;
        let logits = self.model.forward_with_cache_dispatched_rows(
            &tokens,
            &positions,
            &mut self.cache,
            self.moe_dispatch.as_deref(),
            Some(1),
        )?;
        self.current_pos += seq;
        let (_b, s, _v) = logits.dims3()?;
        let row: Vec<f32> = logits
            .narrow(1, s - 1, 1)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok(row)
    }

    pub fn prefill_hidden_serving_last_row_logits(
        &mut self,
        prompt: &[u32],
    ) -> Result<(Vec<f32>, Tensor)> {
        anyhow::ensure!(!prompt.is_empty(), "prefill_hidden: empty prompt");
        self.invalidate_graphs_synced()?;
        anyhow::ensure!(
            self.current_pos + prompt.len() <= self.cache.max_seq_len(),
            "prefill_hidden: {} + {} tokens exceed cache max_seq_len {}",
            self.current_pos,
            prompt.len(),
            self.cache.max_seq_len()
        );
        let seq = prompt.len();
        let tokens = Tensor::from_vec(prompt.to_vec(), (1usize, seq), &self.device)?;
        let positions_v: Vec<i32> =
            (self.current_pos as i32..(self.current_pos + seq) as i32).collect();
        let positions = Tensor::from_vec(positions_v, seq, &self.device)?;
        let (logits, hidden) = self.model.forward_with_cache_dispatched_hidden_rows(
            &tokens,
            &positions,
            &mut self.cache,
            self.moe_dispatch.as_deref(),
            Some(1),
        )?;
        self.current_pos += seq;
        let row: Vec<f32> = logits
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1()?;
        Ok((row, hidden))
    }

    pub fn ensure_verify_lane(&mut self, rows: usize) -> Result<()> {
        anyhow::ensure!(
            rows >= 2,
            "ensure_verify_lane: a {rows}-row verify is a plain decode; use forward_decode"
        );
        anyhow::ensure!(
            self.cache.mk_verify_route_available(rows),
            "ensure_verify_lane: the mk verify attention route is unavailable for rows={rows} \
             (NV_Q38_MK_VERIFY=0, rows > 8, or head_dim > 512); a captured verify would bake the \
             kv length of the capture-time dequant view and read stale kv on every deeper replay"
        );
        if let Some(lane) = &self.verify {
            anyhow::ensure!(
                lane.rows == rows,
                "ensure_verify_lane: lane holds rows={} but {rows} requested; one verify shape \
                 per engine keeps the graph cache single-keyed",
                lane.rows
            );
            return Ok(());
        }
        let capture_stream = self.runner.stream().clone();
        let GraphedQwen3Moe { model, cache, .. } = &mut *self;
        nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
            model.ensure_fused_lin_verify_ckpts(cache, rows)
        })?;
        capture_stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("post-ckpt-prealloc sync: {e:?}"))?;
        let vocab = self.model.vocab_size();
        let hidden = self.model.config().hidden_size;
        let token_buf = self
            .stream
            .alloc_zeros::<u32>(rows)
            .map_err(|e| anyhow::anyhow!("alloc verify token_buf: {e:?}"))?;
        let pos_buf = self
            .stream
            .alloc_zeros::<i32>(rows)
            .map_err(|e| anyhow::anyhow!("alloc verify pos_buf: {e:?}"))?;
        let logits_buf = self
            .stream
            .alloc_zeros::<half::bf16>(rows * vocab)
            .map_err(|e| anyhow::anyhow!("alloc verify logits_buf: {e:?}"))?;
        let hidden_buf = self
            .stream
            .alloc_zeros::<half::bf16>(rows * hidden)
            .map_err(|e| anyhow::anyhow!("alloc verify hidden_buf: {e:?}"))?;
        self.stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after verify lane alloc: {e:?}"))?;
        self.verify = Some(VerifyLane {
            rows,
            token_buf,
            pos_buf,
            host_tok: vec![0u32; rows].into_boxed_slice(),
            host_pos: vec![0i32; rows].into_boxed_slice(),
            logits_buf,
            hidden_buf,
        });
        Ok(())
    }

    pub fn forward_verify_chain(&mut self, chain: &[u32]) -> Result<()> {
        let rows = match &self.verify {
            Some(lane) => lane.rows,
            None => anyhow::bail!("forward_verify_chain: call ensure_verify_lane first"),
        };
        anyhow::ensure!(
            chain.len() == rows,
            "forward_verify_chain: chain of {} rows but the lane holds {rows}",
            chain.len()
        );
        anyhow::ensure!(
            self.pending_verify_pos_and_rows.is_none(),
            "forward_verify_chain called twice without commit_verify_consumed; the lin-attn \
             rollback checkpoints of the first call would be lost"
        );
        anyhow::ensure!(
            self.current_pos + rows <= self.cache.max_seq_len(),
            "forward_verify_chain: {} + {rows} exceeds cache max_seq_len {}",
            self.current_pos,
            self.cache.max_seq_len()
        );
        let pos = self.current_pos;
        {
            let lane = self.verify.as_mut().unwrap();
            for (j, &t) in chain.iter().enumerate() {
                lane.host_tok[j] = t;
                lane.host_pos[j] = (pos + j) as i32;
            }
        }
        let want_capture = (self.device_routing || self.force_capture_env)
            && !self.capture_failed
            && self.capture_supported
            && std::env::var("NV_GRAPH_OFF").ok().as_deref() != Some("1")
            && (self.moe_dispatch.is_some()
                || self.dense_trunk_needs_no_moe_dispatch_for_capture_because_every_ffn_is_a_plain_mlp());
        if !want_capture {
            return self.forward_verify_uncaptured(pos, rows);
        }
        self.forward_verify_captured(pos, rows)
    }

    fn forward_verify_uncaptured(&mut self, pos: usize, rows: usize) -> Result<()> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = dev.cuda_stream();
        let vocab = self.model.vocab_size();
        let hidden = self.model.config().hidden_size;
        let GraphedQwen3Moe {
            model,
            cache,
            moe_dispatch,
            verify,
            ..
        } = self;
        model.ensure_fused_lin_states(cache)?;
        let lane = verify.as_mut().unwrap();
        stream
            .memcpy_htod(&lane.host_tok[..], &mut lane.token_buf)
            .map_err(|e| anyhow::anyhow!("htod verify tok: {e:?}"))?;
        stream
            .memcpy_htod(&lane.host_pos[..], &mut lane.pos_buf)
            .map_err(|e| anyhow::anyhow!("htod verify pos: {e:?}"))?;
        let tokens = wrap_slice_u32(&lane.token_buf, &dev, (1usize, rows))?;
        let positions = wrap_slice_i32(&lane.pos_buf, &dev, rows)?;
        let _tc = nv_layers::linear::VerifyTcFp8LtGemmScopeGuard::enter_if(
            nv_layers::linear_attn::verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms(),
        );
        let (logits, hid) = model.forward_with_cache_dispatched_hidden_rows(
            &tokens,
            &positions,
            cache,
            moe_dispatch.as_deref(),
            None,
        )?;
        copy_last_row_bf16(&logits, rows * vocab, &mut lane.logits_buf, &dev)?;
        copy_last_row_bf16(&hid, rows * hidden, &mut lane.hidden_buf, &dev)?;
        cache.set_current_len(pos + rows);
        cache.set_fused_lin_verify_rows_pending(rows);
        self.pending_verify_pos_and_rows = Some((pos, rows));
        self.call_count += 1;
        Ok(())
    }

    fn forward_verify_captured(&mut self, pos: usize, rows: usize) -> Result<()> {
        use anyhow::Context as _;

        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        disable_event_tracking_no_earlier_than_the_first_capture(&dev);
        let key = VERIFY_GRAPH_SHAPE_TOKEN_BASE_KEEPS_THE_M1_DECODE_KEY_1_UNTOUCHED + rows as u64;
        let need_warm = !self.runner.has_cached_token(key);
        if need_warm {
            if let Err(e) = self.runner.probe_capture() {
                self.capture_failed = true;
                self.capture_active = false;
                eprintln!(
                    "[graph_engine] verify graph capture unavailable; continuing uncaptured: {e:#}"
                );
                return self.forward_verify_uncaptured(pos, rows);
            }
        }
        let vocab = self.model.vocab_size();
        let hidden = self.model.config().hidden_size;
        self.cache.set_pending_pos_host_only(pos, pos + rows);
        let capture_stream = self.runner.stream().clone();
        let GraphedQwen3Moe {
            model,
            cache,
            runner,
            moe_dispatch,
            verify,
            ..
        } = self;
        let lane = verify.as_mut().unwrap();
        let hook = moe_dispatch.as_deref();
        let host_tok_slice: &[u32] = &lane.host_tok[..];
        let host_pos_slice: &[i32] = &lane.host_pos[..];
        let token_buf = &mut lane.token_buf;
        let pos_buf = &mut lane.pos_buf;
        let logits_buf = &mut lane.logits_buf;
        let hidden_buf = &mut lane.hidden_buf;

        let step = |s: &Arc<CudaStream>,
                    cache: &mut Qwen3MoeKvCache,
                    logits_buf: &mut CudaSlice<half::bf16>,
                    hidden_buf: &mut CudaSlice<half::bf16>,
                    token_buf: &mut CudaSlice<u32>,
                    pos_buf: &mut CudaSlice<i32>|
         -> Result<()> {
            s.memcpy_htod(host_tok_slice, token_buf)
                .map_err(|e| anyhow::anyhow!("htod verify tok: {e:?}"))?;
            s.memcpy_htod(host_pos_slice, pos_buf)
                .map_err(|e| anyhow::anyhow!("htod verify pos: {e:?}"))?;
            let tokens = wrap_slice_u32(token_buf, &dev, (1usize, rows))?;
            let positions = wrap_slice_i32(pos_buf, &dev, rows)?;
            let _tc = nv_layers::linear::VerifyTcFp8LtGemmScopeGuard::enter_if(
                nv_layers::linear_attn::verify_tc_env_read_per_call_nv_q38_verify_tc_1_selects_projections_once_plus_lt_gemm_verify_arms(),
            );
            let (logits, hid) = model.forward_with_cache_dispatched_hidden_rows(
                &tokens, &positions, cache, hook, None,
            )?;
            copy_last_row_bf16(&logits, rows * vocab, logits_buf, &dev)?;
            copy_last_row_bf16(&hid, rows * hidden, hidden_buf, &dev)
        };

        if need_warm {
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                model.ensure_fused_lin_states(cache)
            })
            .context("pre-fuse lin states before verify warm")?;
            let lin_snap = nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                cache.snapshot_lin_states()
            })
            .context("snapshot lin states before verify warm")?;
            cache.set_current_len(pos);
            let warm = nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                step(
                    &capture_stream,
                    cache,
                    logits_buf,
                    hidden_buf,
                    token_buf,
                    pos_buf,
                )
            });
            warm.context("warm pass before verify capture")?;
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("verify warm sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                cache.restore_lin_states(&lin_snap)
            })
            .context("restore lin states after verify warm")?;
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("verify post-restore sync: {e:?}"))?;
        }

        cache.set_current_len(pos);
        let result = runner.run(key, |s| {
            nv_layers::cuda_stream::with_stream(s.clone(), || {
                step(s, cache, logits_buf, hidden_buf, token_buf, pos_buf)
            })
        });

        match result {
            Ok(()) => {
                self.cache.set_current_len(pos + rows);
                self.cache.set_fused_lin_verify_rows_pending(rows);
                self.pending_verify_pos_and_rows = Some((pos, rows));
                self.capture_active = true;
                self.call_count += 1;
                Ok(())
            }
            Err(e) => {
                self.capture_failed = true;
                self.capture_active = false;
                self.runner.invalidate();
                let _ = self.synchronize();
                self.return_graph_mempool_reserved_pages();
                self.cache.reset();
                self.current_pos = 0;
                self.pending_verify_pos_and_rows = None;
                Err(e).context(
                    "verify graph capture failed; cache was reset -- re-prefill and rerun with \
                     device_routing off",
                )
            }
        }
    }

    pub fn verify_logits_host(&self) -> Result<Vec<f32>> {
        let lane = self
            .verify
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify_logits_host: no verify lane"))?;
        self.synchronize()?;
        let bf: Vec<half::bf16> = self
            .stream
            .clone_dtoh(&lane.logits_buf)
            .map_err(|e| anyhow::anyhow!("dtoh verify logits: {e:?}"))?;
        Ok(bf.iter().map(|x| x.to_f32()).collect())
    }

    pub fn verify_hidden_rows_tensor_valid_until_next_forward(&self) -> Result<Tensor> {
        let lane = self
            .verify
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify_hidden_rows: no verify lane"))?;
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let hidden = self.model.config().hidden_size;
        let clone = lane
            .hidden_buf
            .try_clone()
            .map_err(|e| anyhow::anyhow!("clone hidden buf: {e:?}"))?;
        let storage = candle_core::CudaStorage::wrap_cuda_slice(clone, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (1usize, lane.rows, hidden),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }

    pub fn commit_verify_consumed(&mut self, consumed: usize) -> Result<()> {
        let (pos, rows) = self
            .pending_verify_pos_and_rows
            .take()
            .ok_or_else(|| anyhow::anyhow!("commit_verify_consumed without a pending verify"))?;
        anyhow::ensure!(
            consumed >= 1 && consumed <= rows,
            "commit_verify_consumed: consumed {consumed} out of 1..={rows}"
        );
        self.cache.set_current_len(pos + consumed);
        self.cache.rollback_lin_to(consumed)?;
        self.current_pos = pos + consumed;
        Ok(())
    }

    pub fn prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
        &mut self,
        k_chunk: &Tensor,
        v_chunk: &Tensor,
    ) -> Result<()> {
        let t = k_chunk.dims().get(1).copied().unwrap_or(0);
        anyhow::ensure!(
            self.current_pos + t <= self.cache.max_seq_len(),
            "synthetic prime: {} + {} tokens exceed cache max_seq_len {}",
            self.current_pos,
            t,
            self.cache.max_seq_len()
        );
        if self.runner.has_cached() {
            self.invalidate_graphs_synced()?;
        }
        self.cache
            .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                self.current_pos,
                k_chunk,
                v_chunk,
            )?;
        self.current_pos += t;
        Ok(())
    }

    pub fn forward_decode(&mut self, token_id: u32) -> Result<()> {
        anyhow::ensure!(
            self.current_pos + 1 <= self.cache.max_seq_len(),
            "forward_decode: cache full at {}",
            self.current_pos
        );
        let want_capture = (self.device_routing || self.force_capture_env) && !self.capture_failed;
        if want_capture {
            self.forward_decode_captured(token_id)
        } else {
            self.forward_decode_uncaptured(token_id)
        }
    }

    pub fn forward_decode_logits(&mut self, token_id: u32) -> Result<u32> {
        self.forward_decode(token_id)?;
        let host = self.logits_host()?;
        nv_layers::sampler::argmax_host_row(&host)
    }

    pub fn forward_decode_logits_vec(&mut self, token_id: u32) -> Result<Vec<f32>> {
        self.forward_decode(token_id)?;
        self.logits_host()
    }

    pub fn logits_host(&self) -> Result<Vec<f32>> {
        self.synchronize()?;
        let bf: Vec<half::bf16> = self
            .stream
            .clone_dtoh(&self.logits_buf)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))?;
        Ok(bf.iter().map(|x| x.to_f32()).collect())
    }

    fn forward_decode_uncaptured(&mut self, token_id: u32) -> Result<()> {
        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = dev.cuda_stream();
        self.host_tok[0] = token_id;
        self.host_pos[0] = self.current_pos as i32;
        stream
            .memcpy_htod(&self.host_tok[..], &mut self.token_buf)
            .map_err(|e| anyhow::anyhow!("htod tok: {e:?}"))?;
        stream
            .memcpy_htod(&self.host_pos[..], &mut self.pos_buf)
            .map_err(|e| anyhow::anyhow!("htod pos: {e:?}"))?;

        let tokens = wrap_slice_u32(&self.token_buf, &dev, (1usize, 1usize))?;
        let positions = wrap_slice_i32(&self.pos_buf, &dev, 1usize)?;

        let logits = self.model.forward_with_cache_dispatched(
            &tokens,
            &positions,
            &mut self.cache,
            self.moe_dispatch.as_deref(),
        )?;
        copy_last_row_bf16(&logits, self.model.vocab_size(), &mut self.logits_buf, &dev)?;

        self.current_pos += 1;
        self.call_count += 1;
        Ok(())
    }

    fn forward_decode_captured(&mut self, token_id: u32) -> Result<()> {
        use anyhow::Context as _;

        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        if !self.capture_supported {
            self.capture_failed = true;
            self.capture_active = false;
            eprintln!(
                "[graph_engine] graph capture blocked: the CUDA device uses the legacy NULL \
                 stream, so candle ops in the decode step cannot be captured (build the device \
                 with Device::new_cuda_with_stream). Continuing with uncaptured decode."
            );
            return self.forward_decode_uncaptured(token_id);
        }
        if !self.runner.has_cached()
            && self.moe_dispatch.is_none()
            && !self.dense_trunk_needs_no_moe_dispatch_for_capture_because_every_ffn_is_a_plain_mlp()
        {
            self.capture_failed = true;
            self.capture_active = false;
            eprintln!(
                "[graph_engine] graph capture blocked: MoE routing is host-side; install a \
                 device-routed MoeDispatch (set_moe_dispatch + set_device_routing) to capture. \
                 Continuing with uncaptured decode."
            );
            return self.forward_decode_uncaptured(token_id);
        }
        if !self.runner.has_cached()
            && self.cache.has_lin_attn_layers()
            && !self.cache.fused_lin_attn()
        {
            self.capture_failed = true;
            self.capture_active = false;
            eprintln!(
                "[graph_engine] graph capture blocked: fused lin-attn decode is disabled \
                 (NV_GDN_FUSED_DECODE=0); the non-fused recurrent path is not capture-safe. \
                 Continuing with uncaptured decode."
            );
            return self.forward_decode_uncaptured(token_id);
        }

        if std::env::var("NV_GRAPH_OFF").ok().as_deref() == Some("1") {
            self.capture_failed = true;
            self.capture_active = false;
            if std::env::var("NV_GRAPH_FORCE_DISABLE_ET").ok().as_deref() == Some("1") {
                disable_event_tracking_no_earlier_than_the_first_capture(&dev);
            }
            return self.forward_decode_uncaptured(token_id);
        }
        disable_event_tracking_no_earlier_than_the_first_capture(&dev);
        if !self.runner.has_cached() {
            if let Err(e) = self.runner.probe_capture() {
                self.capture_failed = true;
                self.capture_active = false;
                eprintln!(
                    "[graph_engine] graph capture unavailable; continuing with uncaptured decode: {e:#}"
                );
                return self.forward_decode_uncaptured(token_id);
            }
        }
        if !self.runner.has_cached() && self.cache.has_lin_attn_layers() {
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-fuse legacy sync: {e:?}"))?;
            let capture_stream = self.runner.stream().clone();
            let GraphedQwen3Moe { model, cache, .. } = &mut *self;
            let fused = nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                model.ensure_fused_lin_states(cache)
            });
            if let Err(e) = fused {
                self.capture_failed = true;
                self.capture_active = false;
                eprintln!(
                    "[graph_engine] graph capture blocked: lin-attn slots could not be \
                     pre-fused; continuing with uncaptured decode: {e:#}"
                );
                return self.forward_decode_uncaptured(token_id);
            }
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-fuse sync: {e:?}"))?;
        }
        let pos = self.current_pos;
        self.host_tok[0] = token_id;
        self.host_pos[0] = pos as i32;
        self.cache.set_pending_pos_host_only(pos, pos + 1);

        let vocab = self.model.vocab_size();
        let need_warm = !self.runner.has_cached();
        let capture_stream = self.runner.stream().clone();
        let GraphedQwen3Moe {
            model,
            cache,
            runner,
            token_buf,
            pos_buf,
            host_tok,
            host_pos,
            logits_buf,
            moe_dispatch,
            ..
        } = self;
        let hook = moe_dispatch.as_deref();
        let host_tok_slice: &[u32] = &host_tok[..];
        let host_pos_slice: &[i32] = &host_pos[..];

        let step = |s: &Arc<CudaStream>,
                    cache: &mut Qwen3MoeKvCache,
                    logits_buf: &mut CudaSlice<half::bf16>,
                    token_buf: &mut CudaSlice<u32>,
                    pos_buf: &mut CudaSlice<i32>|
         -> Result<()> {
            s.memcpy_htod(host_tok_slice, token_buf)
                .map_err(|e| anyhow::anyhow!("htod tok: {e:?}"))?;
            s.memcpy_htod(host_pos_slice, pos_buf)
                .map_err(|e| anyhow::anyhow!("htod pos: {e:?}"))?;
            let tokens = wrap_slice_u32(token_buf, &dev, (1usize, 1usize))?;
            let positions = wrap_slice_i32(pos_buf, &dev, 1usize)?;
            let logits = model.forward_with_cache_dispatched(&tokens, &positions, cache, hook)?;
            copy_last_row_bf16(&logits, vocab, logits_buf, &dev)
        };

        if need_warm {
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            let lin_snap = nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                cache.snapshot_lin_states()
            })
            .context("snapshot lin states before warm")?;
            cache.set_current_len(pos);
            let warm = nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                step(&capture_stream, cache, logits_buf, token_buf, pos_buf)
            });
            warm.context("warm pass before capture")?;
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                cache.restore_lin_states(&lin_snap)
            })
            .context("restore lin states after warm")?;
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-restore sync: {e:?}"))?;
        }

        cache.set_current_len(pos);
        let result = runner.run(1u64, |s| {
            nv_layers::cuda_stream::with_stream(s.clone(), || {
                step(s, cache, logits_buf, token_buf, pos_buf)
            })
        });

        match result {
            Ok(()) => {
                self.cache.set_current_len(pos + 1);
                self.capture_active = true;
                self.current_pos += 1;
                self.call_count += 1;
                Ok(())
            }
            Err(e) => {
                self.capture_failed = true;
                self.capture_active = false;
                self.runner.invalidate();
                let _ = self.synchronize();
                self.return_graph_mempool_reserved_pages();
                self.cache.reset();
                self.current_pos = 0;
                Err(e).context(
                    "graph capture failed (see graph_engine module docs for the blocker list); \
                     cache was reset -- re-prefill and rerun with device_routing off",
                )
            }
        }
    }
}

fn disable_event_tracking_no_earlier_than_the_first_capture(dev: &candle_core::CudaDevice) {
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

impl Drop for GraphedQwen3Moe {
    fn drop(&mut self) {
        let td = if self.owns_stream {
            GraphTeardown::new(&self.stream)
        } else {
            GraphTeardown::for_a_stream_this_engine_did_not_fork(&self.stream)
        };
        let runner = &mut self.runner;
        td.run(|| runner.invalidate());
    }
}

fn wrap_slice_u32<S: Into<candle_core::Shape>>(
    buf: &CudaSlice<u32>,
    dev: &candle_core::CudaDevice,
    shape: S,
) -> Result<Tensor> {
    let clone = buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone u32 buf: {e:?}"))?;
    let storage = candle_core::CudaStorage::wrap_cuda_slice(clone, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn wrap_slice_i32<S: Into<candle_core::Shape>>(
    buf: &CudaSlice<i32>,
    dev: &candle_core::CudaDevice,
    shape: S,
) -> Result<Tensor> {
    let clone = buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone i32 buf: {e:?}"))?;
    let storage = candle_core::CudaStorage::wrap_cuda_slice(clone, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn copy_last_row_bf16(
    logits: &Tensor,
    vocab: usize,
    dst: &mut CudaSlice<half::bf16>,
    dev: &candle_core::CudaDevice,
) -> Result<()> {
    let contig = logits.contiguous()?;
    let (storage, _layout) = contig.storage_and_layout();
    let cuda = match &*storage {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("logits must be on CUDA"),
    };
    let slice = cuda.as_cuda_slice::<half::bf16>()?;
    let n = slice.len();
    anyhow::ensure!(n >= vocab, "logits len {} < vocab {}", n, vocab);
    let src = slice.slice(n - vocab..);
    let stream = nv_layers::cuda_stream::current_stream(dev);
    stream
        .memcpy_dtod(&src, dst)
        .map_err(|e| anyhow::anyhow!("dtod logits: {e:?}"))?;
    Ok(())
}
