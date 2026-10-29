#![cfg(feature = "cuda")]

use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use candle_core::{CudaDevice, DType, Device, Storage, Tensor};
use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use nv_layers::step_arena;
use nv_kernels::voxtral_spec as spec;

use crate::qwen3::KvCache;
use crate::voxtral::{VoxtralQuant, VoxtralRealtime, STREAMING_PAD_TOKEN};

pub const TOKEN_RING_SLOTS: usize = 64;
pub const REPLAYS_PER_BATCH_TRADE_EOS_LATENCY_AGAINST_HOST_ROUND_TRIPS: usize = 16;
pub const VOXTRAL_SPEC_ROWS_ENV: &str = "NV_VOXTRAL_SPEC_ROWS";
pub const PAD_DRAFT_ROWS_DEFAULT_FOR_NVFP4: usize = 4;

pub const THE_STEP_IS_CAPTURED_ON_THE_DEVICE_STREAM_SO_CANDLE_OPS_AND_NV_KERNELS_LAND_IN_ONE_GRAPH:
    &str = "candle ops run on the device's own stream while nv-layers kernels run on \
     cuda_stream::current_stream; capturing on the device stream keeps both inside the capture \
     without a stream override, and the only host-side state the step reads (the previous token \
     and the position) lives in device buffers that the accept and begin kernels advance";

pub const PAD_DRAFT_SPECULATION_COMMITS_EVERY_LEADING_PAD_OF_AN_M_ROW_STEP: &str =
    "row 0 carries the last committed token and rows 1..M carry [STREAMING_PAD] as the draft; \
     the step writes K/V at M consecutive positions, attends causally within the block, and the \
     accept kernel commits rows while the previous row's argmax equals the pad, so the first \
     non-pad token ends the chain and the positions past it are recomputed next step over the \
     same K/V slots; greedy acceptance is exact and the weights are read once per block";

pub const ROPE_AND_THE_KV_WRITE_SHARE_ONE_KERNEL_THAT_SPLITS_THE_BF16_QKV_ROWS: &str =
    "spec_qkv_prep reads the GEMV's bf16 qkv rows, rotates q in f32 into a contiguous \
     [M, nh, hd] buffer for the M-row decode attention, rotates k and stores k and v into the \
     cache at each row's position; the bf16-to-f32 widening is exact, so the arithmetic is the \
     one rope_f32 and write_kv_bf16 apply after a cast, and spec_silu_mul_halves applies \
     candle's bf16 silu and product to the gate_up halves without the two narrow copies";

pub const DECODE_ATTENTION_EXPECTS_A_PRESCALED_QUERY: &str =
    "flash_decode_fused_bf16kv_mk softmaxes the raw q.k dot products, so the q rows of the \
     fused qkv weight carry 1/sqrt(head_dim) folded in at load and the uncaptured path passes \
     softmax_scale = 1";

pub const M_ROWS_STAY_BITWISE_WITH_ONE_ROW_BECAUSE_EVERY_KERNEL_KEEPS_ONE_LANE_PER_ROW: &str =
    "nvfp4_gemv_bf16act_mrows runs the MROWS=1 fmaf chain per row, the rmsnorm wide rung keeps \
     one CTA per row for batches up to eight, and the mk decode attention runs each row's \
     online softmax over its own prefix; the single-row graph is NV_VOXTRAL_SPEC_ROWS=1 of the \
     same body and the parity suite holds it bitwise against the eager loop";

pub const THE_DRAFT_IS_A_PER_POSITION_TABLE_AND_THE_ACCEPT_COMPARES_EACH_ROW_WITH_ITS_OWN_DRAFT: &str =
    "rows 1..M of a step carry draft[pos + i], a table the host fills per clip before decoding \
     (every entry [STREAMING_PAD] when no drafter is loaded), so the accept kernel commits row \
     i while the previous row's argmax equals toks[i] rather than a constant pad, then refills \
     toks[1..M] from the table at the new position; the nv-kernels pad accept still runs for \
     its row argmaxes, whose partials this kernel reduces, and its own commit lands in dummy \
     buffers";

pub const DRAFT_ACCEPT_SRC: &str = r#"
__device__ inline void merge(float& v, int& i, float ov, int oi) {
    if (ov > v || (ov == v && oi < i)) { v = ov; i = oi; }
}
extern "C" __global__ void voxtral_draft_accept(
    const float* part_val, const int* part_idx, int m, int* pos, int* toks,
    unsigned int* ring, int ring_mask, const int* draft
) {
    __shared__ float sval[256];
    __shared__ int sidx[256];
    __shared__ int out[8];
    int tid = threadIdx.x;
    for (int r = 0; r < m; ++r) {
        sval[tid] = part_val[r * 256 + tid];
        sidx[tid] = part_idx[r * 256 + tid];
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (tid < s) merge(sval[tid], sidx[tid], sval[tid + s], sidx[tid + s]);
            __syncthreads();
        }
        if (tid == 0) out[r] = (sidx[0] == 0x7fffffff) ? 0 : sidx[0];
        __syncthreads();
    }
    if (tid == 0) {
        int p = pos[0];
        int n = 1;
        while (n < m && out[n - 1] == toks[n]) ++n;
        for (int i = 0; i < n; ++i) ring[(p + i) & ring_mask] = (unsigned int)out[i];
        int np = p + n;
        toks[0] = out[n - 1];
        for (int i = 1; i < m; ++i) toks[i] = draft[np + i];
        pos[0] = np;
    }
}
"#;

fn draft_accept_kernel(dev: &CudaDevice) -> Result<CudaFunction> {
    static PTX: std::sync::OnceLock<std::sync::Mutex<Option<cudarc::nvrtc::Ptx>>> =
        std::sync::OnceLock::new();
    let mut slot = PTX
        .get_or_init(|| std::sync::Mutex::new(None))
        .lock()
        .map_err(|_| anyhow::anyhow!("draft accept ptx cache poisoned"))?;
    if slot.is_none() {
        ensure!(spec::argmax_parts() == 256, "the draft accept reduces 256 argmax partials");
        *slot = Some(cudarc::nvrtc::compile_ptx(DRAFT_ACCEPT_SRC).map_err(|e| anyhow::anyhow!("{e:?}"))?);
    }
    let module = dev
        .cuda_stream()
        .context()
        .load_module(slot.as_ref().expect("ptx").clone())?;
    Ok(module.load_function("voxtral_draft_accept")?)
}

pub const BF16_KEEPS_ONE_ROW_BECAUSE_ITS_M_ROW_LINEAR_IS_A_CUBLAS_TILE: &str =
    "the bf16 Linear routes rows 2..16 through a cuBLAS tile whose reduction differs from \
     gemv_bf16, so the dense grade decodes one row per step unless NV_VOXTRAL_SPEC_ROWS opts \
     in; the NVFP4 grade drafts pads by default";

fn cuda_ptr<T: candle_core::cuda_backend::CudaDType + cudarc::driver::DeviceRepr>(
    t: &Tensor,
    stream: &Arc<CudaStream>,
) -> Result<u64> {
    let (s, l) = t.storage_and_layout();
    let c = match &*s {
        Storage::Cuda(c) => c,
        _ => bail!("tensor not on cuda"),
    };
    let sl = c.as_cuda_slice::<T>()?;
    let (p, _g) = sl.device_ptr(stream);
    Ok(p + (l.start_offset() * std::mem::size_of::<T>()) as u64)
}

fn gather_rows(table: &Tensor, idx_i32_ptr: u64, m: usize, dev: &CudaDevice) -> Result<Tensor> {
    let (rows, hidden) = table.dims2()?;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let out = step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
        &Device::Cuda(dev.clone()),
        DType::BF16,
        &[m, hidden],
    )?;
    let tp = cuda_ptr::<half::bf16>(table, &stream)?;
    let op = step_arena::device_ptr_for_a_kernel_to_write(&out)?;
    let rc = unsafe {
        nv_kernels::cuda::gather_rows_bf16(
            stream.cu_stream() as *mut _,
            tp as *const u16,
            idx_i32_ptr as *const i32,
            op as *mut u16,
            m as i32,
            hidden as i32,
            rows as i32,
        )
    };
    ensure!(rc == 0, "gather_rows_bf16 returned {rc}");
    out.reshape((1, m, hidden)).map_err(Into::into)
}

pub fn pad_draft_rows_for(quant: VoxtralQuant) -> Result<usize> {
    let _ = BF16_KEEPS_ONE_ROW_BECAUSE_ITS_M_ROW_LINEAR_IS_A_CUBLAS_TILE;
    let rows = match std::env::var(VOXTRAL_SPEC_ROWS_ENV).ok().as_deref() {
        None | Some("") => match quant {
            VoxtralQuant::Nvfp4 => PAD_DRAFT_ROWS_DEFAULT_FOR_NVFP4,
            _ => 1,
        },
        Some(v) => v.parse().map_err(|_| {
            anyhow::anyhow!("{VOXTRAL_SPEC_ROWS_ENV}={v:?}: expected 1..={}", spec::PAD_DRAFT_ROWS_MAX)
        })?,
    };
    ensure!(
        (1..=spec::PAD_DRAFT_ROWS_MAX).contains(&rows),
        "{VOXTRAL_SPEC_ROWS_ENV}={rows}: expected 1..={}",
        spec::PAD_DRAFT_ROWS_MAX
    );
    Ok(rows)
}

pub struct GraphState {
    cache: KvCache,
    audio: Tensor,
    capacity: usize,
    rows: usize,
    n_audio: usize,
    dev: CudaDevice,
    stream: Arc<CudaStream>,
    runner: nv_kernels::graph::CudaGraphRunner,
    pos: CudaSlice<i32>,
    rows_pos: CudaSlice<i32>,
    pos_end: CudaSlice<i32>,
    toks: CudaSlice<i32>,
    ring: CudaSlice<u32>,
    amax_val: CudaSlice<f32>,
    amax_idx: CudaSlice<i32>,
    fan_in: CudaSlice<u32>,
    draft: CudaSlice<i32>,
    draft_host: Vec<i32>,
    dummy_commit: (CudaSlice<i32>, CudaSlice<i32>, CudaSlice<u32>),
    accept: CudaFunction,
    arena: Option<step_arena::StepArena>,
    next_input_pos: usize,
    steps: usize,
    committed: usize,
}

pub const FAN_IN_COUNTERS_ARE_SELF_RESETTING_SO_ONE_ZEROED_BUFFER_SERVES_EVERY_REPLAY: &str =
    "flash_decode_fused_bf16kv_mk's split blocks take an atomic ticket per head in fan_in and \
     the last block to finish writes the head's output and zeroes its counter, so the buffer is \
     allocated zeroed once and never touched by the host";

pub const ONE_CAPTURE_SERVES_EVERY_CLIP_BECAUSE_THE_GRAPH_BAKES_ITS_BUFFER_ADDRESSES: &str =
    "a CUDA graph replays the exact device pointers it was captured with, so the KV buffers, the \
     audio-embedding table and the token/position scalars are allocated once at the decoder's \
     capacity and reused: a clip copies its embeddings into the table, the prefill rewrites the \
     cache head, and the step graph captured on the first clip replays unchanged";

pub const THE_CACHE_CARRIES_M_SPARE_ROWS_FOR_THE_LAST_BLOCKS_UNCOMMITTED_TAIL: &str =
    "a block may start within M positions of the audio's end and still writes M K/V rows, so \
     the cache is allocated capacity + M rows and the host truncates the tokens it reads back";

impl GraphState {
    pub fn new(model: &VoxtralRealtime, capacity: usize, rows: usize) -> Result<Self> {
        let _ = THE_STEP_IS_CAPTURED_ON_THE_DEVICE_STREAM_SO_CANDLE_OPS_AND_NV_KERNELS_LAND_IN_ONE_GRAPH;
        let _ = ONE_CAPTURE_SERVES_EVERY_CLIP_BECAUSE_THE_GRAPH_BAKES_ITS_BUFFER_ADDRESSES;
        let _ = THE_CACHE_CARRIES_M_SPARE_ROWS_FOR_THE_LAST_BLOCKS_UNCOMMITTED_TAIL;
        ensure!((1..=spec::PAD_DRAFT_ROWS_MAX).contains(&rows), "bad row count {rows}");
        let Device::Cuda(dev) = model.device().clone() else {
            bail!("GraphState requires a CUDA device")
        };
        let ctx = dev.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
        }
        let stream = dev.cuda_stream();
        let t = &model.config().text_config;
        let cache = KvCache::with_dims(
            t.num_hidden_layers,
            capacity + rows,
            t.num_key_value_heads,
            t.head_dim,
            &model.device().clone(),
            DType::BF16,
        )?;
        let audio = Tensor::zeros((capacity + rows, t.hidden_size), DType::BF16, model.device())?;
        let pos = stream.clone_htod(&[0i32])?;
        let rows_pos = stream.alloc_zeros::<i32>(rows)?;
        let pos_end = stream.clone_htod(&[0i32])?;
        let toks = stream.clone_htod(&vec![STREAMING_PAD_TOKEN as i32; rows])?;
        let ring = stream.alloc_zeros::<u32>(TOKEN_RING_SLOTS)?;
        let parts = spec::argmax_parts() * rows;
        let amax_val = stream.alloc_zeros::<f32>(parts)?;
        let amax_idx = stream.alloc_zeros::<i32>(parts)?;
        let _ = FAN_IN_COUNTERS_ARE_SELF_RESETTING_SO_ONE_ZEROED_BUFFER_SERVES_EVERY_REPLAY;
        let fan_in = stream.alloc_zeros::<u32>(t.num_attention_heads.max(64))?;
        let draft_host = vec![STREAMING_PAD_TOKEN as i32; capacity + 2 * spec::PAD_DRAFT_ROWS_MAX + 2];
        let draft = stream.clone_htod(&draft_host)?;
        let dummy_commit = (
            stream.alloc_zeros::<i32>(1)?,
            stream.alloc_zeros::<i32>(rows)?,
            stream.alloc_zeros::<u32>(TOKEN_RING_SLOTS)?,
        );
        let accept = draft_accept_kernel(&dev)?;
        stream.synchronize()?;
        Ok(Self {
            cache,
            audio,
            capacity,
            rows,
            n_audio: 0,
            dev: dev.clone(),
            stream: stream.clone(),
            runner: nv_kernels::graph::CudaGraphRunner::new(stream),
            pos,
            rows_pos,
            pos_end,
            toks,
            ring,
            amax_val,
            amax_idx,
            fan_in,
            draft,
            draft_host,
            dummy_commit,
            accept,
            arena: Some(step_arena::StepArena::new()),
            next_input_pos: 0,
            steps: 0,
            committed: 0,
        })
    }

    pub fn cache_mut(&mut self) -> &mut KvCache {
        &mut self.cache
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn steps_and_committed_positions(&self) -> (usize, usize) {
        (self.steps, self.committed)
    }

    pub fn load_audio(&mut self, audio_embeds: &Tensor) -> Result<()> {
        let n_audio = audio_embeds.dims()[1];
        let hidden = audio_embeds.dims()[2];
        ensure!(
            n_audio <= self.capacity,
            "clip of {n_audio} tokens exceeds capacity {}",
            self.capacity
        );
        let rows = audio_embeds
            .reshape((n_audio, hidden))?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        self.audio.slice_set(&rows, 0, 0)?;
        self.n_audio = n_audio;
        Ok(())
    }

    pub fn load_draft(&mut self, draft: Option<&[u32]>) -> Result<()> {
        let _ = THE_DRAFT_IS_A_PER_POSITION_TABLE_AND_THE_ACCEPT_COMPARES_EACH_ROW_WITH_ITS_OWN_DRAFT;
        self.draft_host.fill(STREAMING_PAD_TOKEN as i32);
        if let Some(d) = draft {
            ensure!(d.len() <= self.n_audio, "draft table longer than the clip");
            for (slot, &t) in self.draft_host.iter_mut().zip(d) {
                *slot = t as i32;
            }
        }
        self.stream.memcpy_htod(&self.draft_host, &mut self.draft)?;
        Ok(())
    }

    pub fn begin_decode(&mut self, first_token: u32) -> Result<()> {
        let prefix_len = self.cache.current_len();
        ensure!(
            prefix_len > 0 && prefix_len < self.n_audio,
            "cache must hold the prefill"
        );
        self.stream
            .memcpy_htod(&[prefix_len as i32], &mut self.pos)?;
        let mut toks: Vec<i32> = (0..self.rows).map(|i| self.draft_host[prefix_len + i]).collect();
        toks[0] = first_token as i32;
        self.stream.memcpy_htod(&toks, &mut self.toks)?;
        self.next_input_pos = prefix_len;
        Ok(())
    }

    fn replay(&mut self, model: &VoxtralRealtime) -> Result<()> {
        if let Some(a) = self.arena.take() {
            step_arena::install(a);
        }
        let GraphState {
            cache,
            audio,
            dev,
            runner,
            rows,
            pos,
            rows_pos,
            pos_end,
            toks,
            ring,
            amax_val,
            amax_idx,
            fan_in,
            draft,
            dummy_commit,
            accept,
            ..
        } = &mut *self;
        let dev = dev.clone();
        let m = *rows;
        let result = runner.run(m as u64, |_s| {
            step_arena::begin_step();
            step_body(
                model, cache, audio, &dev, m, pos, rows_pos, pos_end, toks, ring, amax_val,
                amax_idx, fan_in, draft, dummy_commit, accept,
            )
        });
        self.arena = step_arena::uninstall();
        result
    }

    pub fn remaining_steps(&self) -> usize {
        (self.n_audio - 1).saturating_sub(self.next_input_pos)
    }

    pub fn batch_replays(&self) -> usize {
        (self.remaining_steps() / self.rows)
            .clamp(1, REPLAYS_PER_BATCH_TRADE_EOS_LATENCY_AGAINST_HOST_ROUND_TRIPS)
            .min(TOKEN_RING_SLOTS / self.rows)
    }

    pub fn run_batch(&mut self, model: &VoxtralRealtime, k: usize) -> Result<Vec<u32>> {
        ensure!(k >= 1 && k * self.rows <= TOKEN_RING_SLOTS, "bad batch size {k}");
        let remaining = self.remaining_steps();
        ensure!(remaining > 0, "batch past the audio");
        let start = self.next_input_pos;
        for _ in 0..k {
            self.replay(model)?;
        }
        self.stream.synchronize()?;
        let host: Vec<u32> = self.stream.clone_dtoh(&self.ring)?;
        let pos: Vec<i32> = self.stream.clone_dtoh(&self.pos)?;
        let advanced = (pos[0] as usize - start).min(remaining);
        self.next_input_pos += advanced;
        self.steps += k;
        self.committed += advanced;
        Ok((0..advanced)
            .map(|i| host[(start + i) & (TOKEN_RING_SLOTS - 1)])
            .collect())
    }
}

impl Drop for GraphState {
    fn drop(&mut self) {
        let teardown =
            nv_quant::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(
                &self.stream,
            );
        let runner = &mut self.runner;
        teardown.run_owned_streams_only(|| runner.invalidate());
    }
}

#[allow(clippy::too_many_arguments)]
fn step_body(
    model: &VoxtralRealtime,
    cache: &mut KvCache,
    audio: &Tensor,
    dev: &CudaDevice,
    m: usize,
    pos: &mut CudaSlice<i32>,
    rows_pos: &mut CudaSlice<i32>,
    pos_end: &mut CudaSlice<i32>,
    toks: &mut CudaSlice<i32>,
    ring: &mut CudaSlice<u32>,
    amax_val: &mut CudaSlice<f32>,
    amax_idx: &mut CudaSlice<i32>,
    fan_in: &mut CudaSlice<u32>,
    draft: &CudaSlice<i32>,
    dummy_commit: &mut (CudaSlice<i32>, CudaSlice<i32>, CudaSlice<u32>),
    accept: &CudaFunction,
) -> Result<()> {
    let _ = DECODE_ATTENTION_EXPECTS_A_PRESCALED_QUERY;
    let _ = PAD_DRAFT_SPECULATION_COMMITS_EVERY_LEADING_PAD_OF_AN_M_ROW_STEP;
    let _ = M_ROWS_STAY_BITWISE_WITH_ONE_ROW_BECAUSE_EVERY_KERNEL_KEEPS_ONE_LANE_PER_ROW;
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let cu = stream.cu_stream() as *mut _;
    let t = &model.config().text_config;
    let (nh, nkv, hd) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim);
    let rc = {
        let (pp, _g1) = pos.device_ptr(&stream);
        let (rp, _g2) = rows_pos.device_ptr_mut(&stream);
        let (ep, _g3) = pos_end.device_ptr_mut(&stream);
        unsafe {
            spec::nv_kernels_voxtral_spec_begin(
                cu,
                pp as *const i32,
                rp as *mut i32,
                ep as *mut i32,
                m as i32,
            )
        }
    };
    ensure!(rc == 0, "voxtral_spec_begin returned {rc}");

    let tok_emb = {
        let (tp, _g) = toks.device_ptr(&stream);
        gather_rows(model.embed_table(), tp, m, dev)?
    };
    let aud_emb = {
        let (rp, _g) = rows_pos.device_ptr(&stream);
        gather_rows(audio, rp, m, dev)?
    };
    let mut x = (tok_emb + aud_emb)?;

    let scratch_elems =
        nv_kernels::cuda::flash_splitk_scratch_elems_mk(nh as i32, hd as i32, m as i32);
    let cos = model.decoder_rope().cos().contiguous()?;
    let sin = model.decoder_rope().sin().contiguous()?;
    let cuda_dev = Device::Cuda(dev.clone());
    for (i, layer) in model.decoder_layers().iter().enumerate() {
        let h = layer.attn_norm.forward(&x)?;
        let _ = ROPE_AND_THE_KV_WRITE_SHARE_ONE_KERNEL_THAT_SPLITS_THE_BF16_QKV_ROWS;
        let qkv = layer.qkv_proj.forward(&h)?.contiguous()?;
        ensure!(qkv.dtype() == DType::BF16, "qkv rows must be bf16");
        let q = step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &cuda_dev,
            DType::F32,
            &[m * nh * hd],
        )?;
        let (kc, vc) = cache.layer_tensors(i)?;
        let kcp = step_arena::device_ptr_for_a_kernel_to_write(kc)?;
        let vcp = step_arena::device_ptr_for_a_kernel_to_write(vc)?;
        let q_ptr = step_arena::device_ptr_for_a_kernel_to_write(&q)?;
        let rc = {
            let qkv_ptr = cuda_ptr::<half::bf16>(&qkv, &stream)?;
            let cp = cuda_ptr::<f32>(&cos, &stream)?;
            let sp = cuda_ptr::<f32>(&sin, &stream)?;
            let (rp, _g) = rows_pos.device_ptr(&stream);
            unsafe {
                spec::nv_kernels_voxtral_spec_qkv_prep(
                    cu,
                    qkv_ptr as *const u16,
                    q_ptr as *mut f32,
                    kcp as *mut u16,
                    vcp as *mut u16,
                    cp as *const f32,
                    sp as *const f32,
                    rp as *const i32,
                    m as i32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                )
            }
        };
        ensure!(rc == 0, "voxtral_spec_qkv_prep returned {rc}");
        drop(qkv);
        let attn = step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &cuda_dev,
            DType::BF16,
            &[m * nh * hd],
        )?;
        let scratch = step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &cuda_dev,
            DType::F32,
            &[scratch_elems],
        )?;
        let rc = {
            let op = step_arena::device_ptr_for_a_kernel_to_write(&attn)?;
            let sp = step_arena::device_ptr_for_a_kernel_to_write(&scratch)?;
            let (ep, _g) = pos_end.device_ptr(&stream);
            let (fp, _gf) = fan_in.device_ptr_mut(&stream);
            unsafe {
                nv_kernels::cuda::flash_decode_fused_bf16kv_mk(
                    cu,
                    q_ptr as *const f32,
                    kcp as *const u16,
                    vcp as *const u16,
                    op as *mut u16,
                    ep as *const i32,
                    0,
                    m as i32,
                    sp as *mut f32,
                    fp as *mut u32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    0,
                )
            }
        };
        ensure!(rc == 0, "flash_decode_fused_bf16kv_mk returned {rc}");
        let attn = layer.o_proj.forward(&attn.reshape((1, m, nh * hd))?)?;
        let (h, x_mid) = layer.ffn_norm.forward_residual(&attn, &x)?;
        let gate_up = layer.gate_up_proj.forward(&h)?.contiguous()?;
        let inter = t.intermediate_size;
        let act = step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(
            &cuda_dev,
            DType::BF16,
            &[1, m, inter],
        )?;
        let rc = {
            let gp = cuda_ptr::<half::bf16>(&gate_up, &stream)?;
            let ap = step_arena::device_ptr_for_a_kernel_to_write(&act)?;
            unsafe {
                spec::nv_kernels_voxtral_spec_silu_mul_halves(
                    cu,
                    gp as *const u16,
                    ap as *mut u16,
                    m as i32,
                    inter as i32,
                )
            }
        };
        ensure!(rc == 0, "voxtral_spec_silu_mul_halves returned {rc}");
        drop(gate_up);
        x = (&x_mid + layer.down_proj.forward(&act)?)?;
    }
    let logits = model.logits_from_hidden(&x)?.contiguous()?;
    ensure!(logits.dtype() == DType::BF16, "logits must be bf16");
    let n = logits.elem_count() / m;
    let rc = {
        let lp = cuda_ptr::<half::bf16>(&logits, &stream)?;
        let (vp, _gv) = amax_val.device_ptr_mut(&stream);
        let (ip, _gi) = amax_idx.device_ptr_mut(&stream);
        let (pp, _gp) = dummy_commit.0.device_ptr_mut(&stream);
        let (tp, _gt) = dummy_commit.1.device_ptr_mut(&stream);
        let (rp, _gr) = dummy_commit.2.device_ptr_mut(&stream);
        unsafe {
            spec::nv_kernels_voxtral_spec_accept(
                cu,
                lp as *const u16,
                m as i32,
                n as i32,
                vp as *mut f32,
                ip as *mut i32,
                pp as *mut i32,
                tp as *mut i32,
                rp as *mut u32,
                (TOKEN_RING_SLOTS - 1) as i32,
                STREAMING_PAD_TOKEN as i32,
            )
        }
    };
    ensure!(rc == 0, "voxtral_spec_accept returned {rc}");
    let _ = THE_DRAFT_IS_A_PER_POSITION_TABLE_AND_THE_ACCEPT_COMPARES_EACH_ROW_WITH_ITS_OWN_DRAFT;
    let (vp, _gv) = amax_val.device_ptr(&stream);
    let (ip, _gi) = amax_idx.device_ptr(&stream);
    let (pp, _gp) = pos.device_ptr_mut(&stream);
    let (tp, _gt) = toks.device_ptr_mut(&stream);
    let (rp, _gr) = ring.device_ptr_mut(&stream);
    let (dp, _gd) = draft.device_ptr(&stream);
    let (mi, mask) = (m as i32, (TOKEN_RING_SLOTS - 1) as i32);
    let mut lb = stream.launch_builder(accept);
    lb.arg(&vp).arg(&ip).arg(&mi).arg(&pp).arg(&tp).arg(&rp).arg(&mask).arg(&dp);
    unsafe {
        lb.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }?;
    Ok(())
}

pub fn decode_stream_graphed(
    model: &VoxtralRealtime,
    state: &mut GraphState,
    audio_embeds: &Tensor,
    draft: Option<&[u32]>,
) -> Result<Vec<u32>> {
    let eos = model.config().text_config.eos_token_id;
    model.audio_tokens_in_range(audio_embeds.dims()[1])?;
    state.load_audio(audio_embeds)?;
    state.load_draft(draft)?;
    state.cache_mut().reset();
    let first_token = model.prefill_into(audio_embeds, state.cache_mut())?;
    state.begin_decode(first_token)?;
    if first_token == eos {
        return Ok(Vec::new());
    }
    let mut out = vec![first_token];
    while state.remaining_steps() > 0 {
        let k = state.batch_replays();
        for t in state.run_batch(model, k)? {
            if t == eos {
                return Ok(out);
            }
            out.push(t);
        }
    }
    Ok(out)
}
