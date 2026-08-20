use anyhow::Result;
use candle_core::{Device, Tensor};

use crate::gemma4::{Gemma4, Gemma4KvCacheFp8};

#[cfg(feature = "cuda")]
use crate::gemma4_batch_graph::capture_stream::CaptureStream;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, PinnedHostSlice};
#[cfg(feature = "cuda")]
use nv_kernels::graph::CudaGraphRunner;
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
pub const GEMMA4_GRAPH_COINCIDENCE_GATE: &str =
    "nv-models tests/gemma4_capture_stream_policy.rs";

#[cfg(feature = "cuda")]
pub const WARM_PASS_MATERIALIZES_WHAT_CAPTURE_MUST_NOT_ALLOCATE: &str =
    "warm pass before the first capture, run ONLY when the capture stream is the device stream. \
     There, the candle forward is inside the graph and every first-touch lazy allocation it makes \
     would be captured too -- nv-layers' NVFP4 a_staging map (linear.rs) refuses to populate \
     itself while stream_is_capturing, so an uncaptured first call is the only way the captured \
     body equals the body every replay executes; measured on nvidia/Gemma-4-31B-IT-NVFP4 as \
     4133 captured nodes without it and 3653 with it. On the forked capture the pass is skipped: \
     the fork's captured body was measured identical (4133 nodes, logits bit-equal to the eager \
     path) with and without it, so paying an extra 28 ms on the first decode of every request \
     would buy nothing. The pass is NOT free of host side effects: the forward body derives \
     write_pos from cache.current_len() and advances it, so warm plus capture advance the len \
     twice for one token. With one captured arm the off-by-one only makes the first launch \
     write an exact duplicate of the pos-0 kv at slot 1, which softmax cancels bitwise and the \
     first replay overwrites; with a SECOND captured arm the drift reaches +2, the second arm's \
     kv lands at slots 2 and 3, and every later replay of the first arm attends a stale \
     duplicate at slot 1 forever (max_abs_diff 0.265625 then compounding, identical in both \
     capture orders, bit-matched by the double-advance sim probe in \
     nv-models tests/gemma4_graph_dtod_arm_keys.rs). The len restore right after the warm sync \
     is what keeps the capture pass computing the same write_pos the warm used";

#[cfg(feature = "cuda")]
pub const BOTH_DTOD_ARMS_ARE_SEQ_LEN_1_SO_THE_GRAPH_KEY_MUST_CARRY_THE_ARM: &str =
    "decode_step_body captures two different bodies at the same [1,1] shape: forward_with_cache, \
     which leaves logits_buf untouched, and forward_with_cache_into, which softcaps the lm_head \
     row into it. CudaGraphRunner keys its cache on the shape token alone, so keying both arms on \
     1 replays whichever was captured first for the other -- a forward_decode_logits that returns \
     a stale logits_buf because the no-logits graph is what actually ran. The gate is \
     nv-models tests/gemma4_graph_dtod_arm_keys.rs";

#[cfg(feature = "cuda")]
pub const DECODE_GRAPH_KEY_NO_LOGITS_DTOD: u64 = 1;
#[cfg(feature = "cuda")]
pub const DECODE_GRAPH_KEY_LOGITS_DTOD: u64 = 2;

#[cfg(feature = "cuda")]
pub fn decode_graph_key(dtod_logits: bool) -> u64 {
    if dtod_logits {
        DECODE_GRAPH_KEY_LOGITS_DTOD
    } else {
        DECODE_GRAPH_KEY_NO_LOGITS_DTOD
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn decode_step_body(
    s: &Arc<CudaStream>,
    dev: &candle_core::CudaDevice,
    model: &Gemma4,
    cache: &mut Gemma4KvCacheFp8,
    token_buf: &mut CudaSlice<u32>,
    pos_buf: &mut CudaSlice<i32>,
    host_tok: &[u32],
    host_pos: &[i32],
    logits_buf: &mut CudaSlice<f32>,
    dtod_logits: bool,
) -> Result<()> {
    s.memcpy_htod(host_tok, token_buf)
        .map_err(|e| anyhow::anyhow!("htod tok: {e:?}"))?;
    s.memcpy_htod(host_pos, pos_buf)
        .map_err(|e| anyhow::anyhow!("htod pos: {e:?}"))?;

    let tok_clone = token_buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone token_buf: {e:?}"))?;
    let pos_clone = pos_buf
        .try_clone()
        .map_err(|e| anyhow::anyhow!("clone pos_buf: {e:?}"))?;
    let tokens = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        Tensor::from_storage(
            storage,
            (1usize, 1usize),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };
    let positions = {
        let storage = candle_core::CudaStorage::wrap_cuda_slice(pos_clone, dev.clone());
        let storage = candle_core::Storage::Cuda(storage);
        Tensor::from_storage(
            storage,
            (1usize,),
            candle_core::op::BackpropOp::none(),
            false,
        )
    };

    if dtod_logits {
        model.forward_with_cache_into(&tokens, &positions, cache, logits_buf)?;
    } else {
        let _ = model.forward_with_cache(&tokens, &positions, cache)?;
    }
    Ok(())
}

pub struct GraphedGemma4Decoder<'m> {
    model: &'m Gemma4,
    cache: Gemma4KvCacheFp8,
    device: Device,

    #[cfg(feature = "cuda")]
    capture: CaptureStream,
    #[cfg(feature = "cuda")]
    runner: CudaGraphRunner,

    #[cfg(feature = "cuda")]
    token_buf: CudaSlice<u32>,
    #[cfg(feature = "cuda")]
    pos_buf: CudaSlice<i32>,

    #[cfg(feature = "cuda")]
    host_tok: Box<[u32; 1]>,
    #[cfg(feature = "cuda")]
    host_pos: Box<[i32; 1]>,

    #[cfg(feature = "cuda")]
    logits_buf: CudaSlice<f32>,

    #[cfg(feature = "cuda")]
    logits_host: PinnedHostSlice<f32>,

    #[cfg(feature = "cuda")]
    probe_ev: Option<(CudaEvent, CudaEvent)>,

    current_pos: usize,
    call_count: u64,
    capture_active: bool,
}

impl<'m> GraphedGemma4Decoder<'m> {
    #[cfg(feature = "cuda")]
    pub fn new(model: &'m Gemma4, cache: Gemma4KvCacheFp8, device: &Device) -> Result<Self> {
        let dev = match device {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("GraphedGemma4Decoder requires a CUDA device"),
        };

        let capture = CaptureStream::for_device(device)?;
        capture.forked_candle_capture_is_an_asserted_coincidence(
            "GraphedGemma4Decoder",
            GEMMA4_GRAPH_COINCIDENCE_GATE,
        );
        let capture_stream = capture.stream().clone();
        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();

        let token_buf = capture_stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!("alloc token_buf: {e:?}"))?;
        let pos_buf = capture_stream
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!("alloc pos_buf: {e:?}"))?;

        let vocab = model.config().vocab_size;
        let logits_buf = capture_stream
            .alloc_zeros::<f32>(vocab)
            .map_err(|e| anyhow::anyhow!("alloc logits_buf: {e:?}"))?;
        let logits_host = unsafe { raw_ctx.alloc_pinned::<f32>(vocab) }
            .map_err(|e| anyhow::anyhow!("alloc pinned logits_host: {e:?}"))?;

        capture_stream
            .synchronize()
            .map_err(|e| anyhow::anyhow!("sync after alloc: {e:?}"))?;

        let runner = CudaGraphRunner::new(capture_stream.clone());

        let probe_ev = if crate::decode_probe::enabled() {
            let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
            let a = raw_ctx
                .new_event(Some(flags))
                .map_err(|e| anyhow::anyhow!("probe event start: {e:?}"))?;
            let b = raw_ctx
                .new_event(Some(flags))
                .map_err(|e| anyhow::anyhow!("probe event end: {e:?}"))?;
            Some((a, b))
        } else {
            None
        };

        let current_pos = cache.current_len();

        Ok(Self {
            model,
            cache,
            device: device.clone(),
            capture,
            runner,
            token_buf,
            pos_buf,
            host_tok: Box::new([0u32; 1]),
            host_pos: Box::new([0i32; 1]),
            logits_buf,
            logits_host,
            probe_ev,
            current_pos,
            call_count: 0,
            capture_active: false,
        })
    }

    pub fn cache(&self) -> &Gemma4KvCacheFp8 {
        &self.cache
    }
    pub fn call_count(&self) -> u64 {
        self.call_count
    }
    pub fn capture_active(&self) -> bool {
        self.capture_active
    }
    pub fn current_pos(&self) -> usize {
        self.current_pos
    }

    #[cfg(feature = "cuda")]
    pub fn has_captured_arm(&self, dtod_logits: bool) -> bool {
        self.runner.has_cached_token(decode_graph_key(dtod_logits))
    }

    #[cfg(feature = "cuda")]
    pub fn captured_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    #[cfg(feature = "cuda")]
    pub fn logits_buf_snapshot(&self) -> Result<Vec<f32>> {
        self.synchronize()?;
        self.capture
            .stream()
            .clone_dtoh(&self.logits_buf)
            .map_err(|e| anyhow::anyhow!("dtoh logits snapshot: {e:?}"))
    }

    #[cfg(feature = "cuda")]
    pub fn synchronize(&self) -> Result<()> {
        self.capture
            .stream()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("capture stream sync: {e:?}"))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode(&mut self, token_id: u32) -> Result<()> {
        self.forward_decode_inner(token_id, false)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_logits(&mut self, token_id: u32) -> Result<u32> {
        if !crate::decode_probe::enabled() {
            self.forward_decode_inner(token_id, true)?;
            self.synchronize()?;
            let host: Vec<f32> = self
                .capture
                .stream()
                .clone_dtoh(&self.logits_buf)
                .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))?;
            return nv_layers::sampler::argmax_host_row(&host);
        }

        let in_flight = crate::decode_probe::enter();
        let out = self.forward_decode_logits_probed(token_id, in_flight);
        crate::decode_probe::leave();
        out
    }

    #[cfg(feature = "cuda")]
    fn forward_decode_logits_probed(&mut self, token_id: u32, in_flight: usize) -> Result<u32> {
        use std::time::Instant;
        let t_enter = Instant::now();
        self.forward_decode_inner(token_id, true)?;
        let t_launched = Instant::now();
        self.synchronize()?;
        let t_synced = Instant::now();
        let host: Vec<f32> = self
            .capture
            .stream()
            .clone_dtoh(&self.logits_buf)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))?;
        let t_dtoh = Instant::now();
        let top_id = nv_layers::sampler::argmax_host_row(&host)?;
        let t_argmax = Instant::now();

        let gpu_ms = match self.probe_ev.as_ref() {
            None => None,
            Some((a, b)) => match a.elapsed_ms(b) {
                Ok(v) => Some(v as f64),
                Err(e) => {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    WARNED.call_once(|| {
                        eprintln!("[decode-probe] in-graph event elapsed failed: {e:?}");
                    });
                    None
                }
            },
        };

        let ms = |a: Instant, b: Instant| b.duration_since(a).as_secs_f64() * 1000.0;
        crate::decode_probe::record(crate::decode_probe::Sample {
            in_flight,
            wall_ms: ms(t_enter, t_argmax),
            launch_ms: ms(t_enter, t_launched),
            sync_ms: ms(t_launched, t_synced),
            dtoh_ms: ms(t_synced, t_dtoh),
            argmax_ms: ms(t_dtoh, t_argmax),
            gpu_ms,
        });
        Ok(top_id)
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_logits_into(&mut self, token_id: u32) -> Result<&[f32]> {
        self.forward_decode_inner(token_id, true)?;
        self.synchronize()?;
        let stream = self.capture.stream().clone();
        stream
            .memcpy_dtoh(&self.logits_buf, &mut self.logits_host)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))?;
        self.logits_host
            .as_slice()
            .map_err(|e| anyhow::anyhow!("pinned logits slice: {e:?}"))
    }

    #[cfg(feature = "cuda")]
    pub fn forward_decode_logits_vec(&mut self, token_id: u32) -> Result<Vec<f32>> {
        Ok(self.forward_decode_logits_into(token_id)?.to_vec())
    }

    #[cfg(feature = "cuda")]
    fn forward_decode_inner(&mut self, token_id: u32, dtod_logits: bool) -> Result<()> {
        use anyhow::Context as _;
        let dbg = std::env::var_os("NV_DEBUG_GRAPH").is_some();
        let dbg_call = self.call_count;
        let dbg_pos = self.current_pos;
        if dbg {
            eprintln!(
                "[graph] call={dbg_call} pos={dbg_pos} token={token_id} dtod={dtod_logits} captured={}",
                self.capture_active
            );
        }

        self.host_tok[0] = token_id;
        self.host_pos[0] = self.current_pos as i32;
        self.cache
            .set_pending_pos_host_only(self.current_pos, self.current_pos + 1)?;

        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };

        let ctx = dev.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let warm_before_capture = self.capture.candle_launches_reach_this_stream();

        let GraphedGemma4Decoder {
            model,
            cache,
            runner,
            token_buf,
            pos_buf,
            host_tok,
            host_pos,
            logits_buf,
            probe_ev,
            ..
        } = self;

        let host_tok_slice: &[u32] = &host_tok[..];
        let host_pos_slice: &[i32] = &host_pos[..];
        let model: &Gemma4 = model;

        let graph_key = decode_graph_key(dtod_logits);
        let capture_stream = runner.stream().clone();
        if warm_before_capture && !runner.has_cached_token(graph_key) {
            let len_before_warm = cache.current_len();
            nv_layers::cuda_stream::with_stream(capture_stream.clone(), || {
                decode_step_body(
                    &capture_stream,
                    &dev,
                    model,
                    cache,
                    token_buf,
                    pos_buf,
                    host_tok_slice,
                    host_pos_slice,
                    logits_buf,
                    dtod_logits,
                )
            })
            .context(WARM_PASS_MATERIALIZES_WHAT_CAPTURE_MUST_NOT_ALLOCATE)?;
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
            cache.reset();
            cache.advance(len_before_warm);
        }

        if let Some((ev_start, _)) = probe_ev.as_ref() {
            ev_start
                .record(&capture_stream)
                .map_err(|e| anyhow::anyhow!("probe ev_start record: {e:?}"))?;
        }
        let result: Result<()> = runner.run(graph_key, |s| -> Result<()> {
            let body = nv_layers::cuda_stream::with_stream(s.clone(), || -> Result<()> {
                decode_step_body(
                    s,
                    &dev,
                    model,
                    cache,
                    token_buf,
                    pos_buf,
                    host_tok_slice,
                    host_pos_slice,
                    logits_buf,
                    dtod_logits,
                )
            });
            body?;
            Ok(())
        });
        if let Some((_, ev_end)) = probe_ev.as_ref() {
            ev_end
                .record(&capture_stream)
                .map_err(|e| anyhow::anyhow!("probe ev_end record: {e:?}"))?;
        }

        result.with_context(|| {
            format!(
                "graph forward_decode_inner call={dbg_call} pos={dbg_pos} token={token_id} dtod={dtod_logits}"
            )
        })?;

        self.capture_active = true;
        self.call_count += 1;
        self.current_pos += 1;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl Drop for GraphedGemma4Decoder<'_> {
    fn drop(&mut self) {
        let td =
            crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_capture(&self.capture);
        let runner = &mut self.runner;
        td.run(|| runner.invalidate());
    }
}

#[cfg(feature = "cuda")]
use crate::gemma4::Gemma4VerifyCache;

#[cfg(feature = "cuda")]
pub struct GraphedGemma4Verify<M> {
    model: M,
    cache: Gemma4VerifyCache,
    device: Device,
    capture: CaptureStream,
    runner: CudaGraphRunner,
    k: usize,
    aux_layers: Vec<usize>,
    tokens_buf: CudaSlice<u32>,
    pos_buf: CudaSlice<i32>,
    mask_buf: CudaSlice<u8>,
    logits_buf: CudaSlice<f32>,
    aux_buf: CudaSlice<half::bf16>,

    amax_val: CudaSlice<f32>,
    amax_idx: CudaSlice<i32>,
    amax_out: CudaSlice<u32>,
    host_toks: Vec<u32>,
    host_pos: Vec<i32>,
    host_mask: Vec<u8>,
    host_committed: Box<[i32; 1]>,
    captured: bool,

    pub last_replay_ms: Option<f32>,
}

#[cfg(feature = "cuda")]
impl<M: std::ops::Deref<Target = Gemma4>> GraphedGemma4Verify<M> {
    pub fn new(
        model: M,
        cache: Gemma4VerifyCache,
        device: &Device,
        k: usize,
        aux_layers: Vec<usize>,
    ) -> Result<Self> {
        if !matches!(device, Device::Cuda(_)) {
            anyhow::bail!("GraphedGemma4Verify requires cuda");
        }
        let capture = CaptureStream::for_device(device)?;
        capture.forked_candle_capture_is_an_asserted_coincidence(
            "GraphedGemma4Verify",
            GEMMA4_GRAPH_COINCIDENCE_GATE,
        );
        let capture_stream = capture.stream().clone();
        let tokens_buf = capture_stream
            .alloc_zeros::<u32>(k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pos_buf = capture_stream
            .alloc_zeros::<i32>(k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mask_buf = capture_stream
            .alloc_zeros::<u8>(k * k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let vocab = model.config().vocab_size;
        let logits_buf = capture_stream
            .alloc_zeros::<f32>(k * vocab)
            .map_err(|e| anyhow::anyhow!(e))?;
        let hidden = model.config().hidden_size;
        let aux_buf = capture_stream
            .alloc_zeros::<half::bf16>(aux_layers.len() * k * hidden)
            .map_err(|e| anyhow::anyhow!(e))?;
        let parts = nv_kernels::cuda::argmax_parts();
        let amax_val = capture_stream
            .alloc_zeros::<f32>(k * parts)
            .map_err(|e| anyhow::anyhow!(e))?;
        let amax_idx = capture_stream
            .alloc_zeros::<i32>(k * parts)
            .map_err(|e| anyhow::anyhow!(e))?;
        let amax_out = capture_stream
            .alloc_zeros::<u32>(k)
            .map_err(|e| anyhow::anyhow!(e))?;
        capture_stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        let runner = CudaGraphRunner::new(capture_stream.clone());
        Ok(Self {
            model,
            cache,
            device: device.clone(),
            capture,
            runner,
            k,
            aux_layers,
            tokens_buf,
            pos_buf,
            mask_buf,
            logits_buf,
            aux_buf,
            amax_val,
            amax_idx,
            amax_out,
            host_toks: vec![0u32; k],
            host_pos: vec![0i32; k],
            host_mask: vec![0u8; k * k],
            host_committed: Box::new([0i32; 1]),
            captured: false,
            last_replay_ms: None,
        })
    }

    pub fn cache_mut(&mut self) -> &mut Gemma4VerifyCache {
        &mut self.cache
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn cache_capacity(&self) -> usize {
        self.cache.max_seq()
    }

    pub fn graph_node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn run(
        &mut self,
        toks: &[u32],
        positions: &[i32],
        mask: &[u8],
        committed: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        self.run_inner(toks, positions, mask, committed)?;
        let logits: Vec<f32> = self
            .capture
            .stream()
            .clone_dtoh(&self.logits_buf)
            .map_err(|e| anyhow::anyhow!(e))?;
        let aux = self.fetch_aux()?;
        Ok((logits, aux))
    }

    pub fn run_argmax(
        &mut self,
        toks: &[u32],
        positions: &[i32],
        mask: &[u8],
        committed: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        self.run_inner(toks, positions, mask, committed)?;
        let amax: Vec<u32> = self
            .capture
            .stream()
            .clone_dtoh(&self.amax_out)
            .map_err(|e| anyhow::anyhow!(e))?;
        let aux = self.fetch_aux()?;
        Ok((amax, aux))
    }

    fn fetch_aux(&self) -> Result<Vec<f32>> {
        let aux_bf: Vec<half::bf16> = self
            .capture
            .stream()
            .clone_dtoh(&self.aux_buf)
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(aux_bf.iter().map(|x| x.to_f32()).collect())
    }

    fn run_inner(
        &mut self,
        toks: &[u32],
        positions: &[i32],
        mask: &[u8],
        committed: usize,
    ) -> Result<()> {
        let k = self.k;
        anyhow::ensure!(
            toks.len() == k && positions.len() == k && mask.len() == k * k,
            "shape"
        );

        anyhow::ensure!(
            committed + k <= self.cache.max_seq(),
            "verify cache overflow: committed={committed} + k={k} > capacity={}",
            self.cache.max_seq()
        );
        self.host_toks.copy_from_slice(toks);
        self.host_pos.copy_from_slice(positions);
        self.host_mask.copy_from_slice(mask);
        self.host_committed[0] = committed as i32;

        let dev = match &self.device {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let ctx = dev.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        let hidden = self.model.config().hidden_size;
        let vocab = self.model.config().vocab_size;
        let was_captured = self.captured;
        let capture_stream = self.capture.stream().clone();
        let GraphedGemma4Verify {
            model,
            cache,
            runner,
            k,
            aux_layers,
            tokens_buf,
            pos_buf,
            mask_buf,
            logits_buf,
            aux_buf,
            amax_val,
            amax_idx,
            amax_out,
            host_toks,
            host_pos,
            host_mask,
            host_committed,
            ..
        } = self;
        let k = *k;
        let dev2 = dev.clone();

        if !was_captured {
            nv_layers::cuda_stream::with_stream(capture_stream.clone(), || -> Result<()> {
                capture_stream
                    .memcpy_htod(&host_toks[..], tokens_buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                capture_stream
                    .memcpy_htod(&host_pos[..], pos_buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                capture_stream
                    .memcpy_htod(&host_mask[..], mask_buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                capture_stream
                    .memcpy_htod(&host_committed[..], cache.n_committed_mut())
                    .map_err(|e| anyhow::anyhow!(e))?;
                let tc = tokens_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let pc = pos_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let ids_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(tc, dev2.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (1usize, k),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                let pos_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(pc, dev2.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (1usize, k),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                let _ = model.forward_verify_dev(&ids_t, &pos_t, mask_buf, k, aux_layers, cache)?;
                Ok(())
            })?;
            capture_stream
                .synchronize()
                .map_err(|e| anyhow::anyhow!(e))?;
        }

        let sol_evt = std::env::var_os("NV_SOL_EVT").is_some();
        let evt_pair = if sol_evt {
            let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
            let ctx2 = capture_stream.context().clone();
            let e0 = ctx2.new_event(flags).map_err(|e| anyhow::anyhow!(e))?;
            let e1 = ctx2.new_event(flags).map_err(|e| anyhow::anyhow!(e))?;
            e0.record(&capture_stream)
                .map_err(|e| anyhow::anyhow!(e))?;
            Some((e0, e1))
        } else {
            None
        };
        runner.run(1u64, |s| -> Result<()> {
            nv_layers::cuda_stream::with_stream(s.clone(), || -> Result<()> {
                s.memcpy_htod(&host_toks[..], tokens_buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                s.memcpy_htod(&host_pos[..], pos_buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                s.memcpy_htod(&host_mask[..], mask_buf)
                    .map_err(|e| anyhow::anyhow!(e))?;
                s.memcpy_htod(&host_committed[..], cache.n_committed_mut())
                    .map_err(|e| anyhow::anyhow!(e))?;

                let tok_clone = tokens_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let pos_clone = pos_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let ids_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev2.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (1usize, k),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                let pos_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(pos_clone, dev2.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (1usize, k),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                let (logits, aux) =
                    model.forward_verify_dev(&ids_t, &pos_t, mask_buf, k, aux_layers, cache)?;

                let stream = nv_layers::cuda_stream::current_stream(&dev2);
                {
                    let lc = logits.contiguous()?;
                    let (ls, _l) = lc.storage_and_layout();
                    let lcuda = match &*ls {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("logits"),
                    };
                    let lsl = lcuda.as_cuda_slice::<f32>()?;
                    stream
                        .memcpy_dtod(lsl, logits_buf)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }
                for (i, a) in aux.iter().enumerate() {
                    let ac = a.reshape((k * hidden,))?.contiguous()?;
                    let (as_, _l) = ac.storage_and_layout();
                    let acuda = match &*as_ {
                        candle_core::Storage::Cuda(s) => s,
                        _ => anyhow::bail!("aux"),
                    };
                    let asl = acuda.as_cuda_slice::<half::bf16>()?;
                    let mut dst = aux_buf.slice_mut(i * k * hidden..(i + 1) * k * hidden);
                    stream
                        .memcpy_dtod(asl, &mut dst)
                        .map_err(|e| anyhow::anyhow!(e))?;
                }

                {
                    use cudarc::driver::{DevicePtr, DevicePtrMut};
                    let (lp, _gl) = logits_buf.device_ptr(s);
                    let (vp, _gv) = amax_val.device_ptr_mut(s);
                    let (ip, _gi) = amax_idx.device_ptr_mut(s);
                    let (op, _go) = amax_out.device_ptr_mut(s);
                    let rc = unsafe {
                        nv_kernels::cuda::argmax_f32_rows(
                            s.cu_stream() as *mut _,
                            lp as *const f32,
                            k as i32,
                            vocab as i32,
                            vp as *mut f32,
                            ip as *mut i32,
                            op as *mut u32,
                        )
                    };
                    anyhow::ensure!(rc == 0, "argmax_f32_rows returned {rc}");
                }
                Ok(())
            })
        })?;
        if let Some((e0, e1)) = evt_pair {
            e1.record(self.capture.stream())
                .map_err(|e| anyhow::anyhow!(e))?;
            self.capture
                .stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!(e))?;
            self.last_replay_ms = Some(e0.elapsed_ms(&e1).map_err(|e| anyhow::anyhow!(e))?);
        }
        self.captured = true;
        self.capture
            .stream()
            .synchronize()
            .map_err(|e| anyhow::anyhow!(e))?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
impl<M> Drop for GraphedGemma4Verify<M> {
    fn drop(&mut self) {
        let td =
            crate::gemma4_batch_graph::graph_teardown::GraphTeardown::for_capture(&self.capture);
        let runner = &mut self.runner;
        td.run(|| runner.invalidate());
    }
}
