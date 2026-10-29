#![cfg(feature = "cuda")]

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{bail, ensure, Result};
use candle_core::{CudaDevice, DType, Device, Tensor};
use cudarc::driver::sys::CUevent_flags;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use half::bf16;
use nv_kernels::voxtral_spec as spec;
use nv_kernels::voxtral_tts_backbone as backbone;
use nv_kernels::voxtral_tts_frame as frame;
use nv_layers::linear::Linear;
use nv_layers::step_arena;

use crate::qwen3::KvCache;
use crate::voxtral_tts::{Synthesis, VoxtralTts, END_AUDIO, N_AUDIO_SPECIAL};

pub const FRAMES_PER_READBACK: usize = 8;
pub const FIRST_READBACK_IS_TWO_FRAMES_SO_THE_FIRST_AUDIO_WAITS_FOR_TWO_REPLAYS_NOT_EIGHT: usize = 2;

pub const ONE_GRAPH_PER_FRAME_RUNS_THE_SEMANTIC_HEAD_THE_FLOW_ODE_THE_EMBEDDING_AND_THE_NEXT_BACKBONE_STEP: &str =
    "the body reads the frame hidden state h from a persistent buffer, writes the semantic argmax \
     and the FSQ codes of the nfe Euler steps into a ring slot, sums the 37 codebook rows of the \
     frame (or of the teacher's frame) into the persistent input buffer, runs the backbone step at \
     the device position and copies the new h back; the host uploads FRAMES_PER_READBACK noise rows, \
     replays that many times, synchronizes once and reads the ring, so a clip costs one host round \
     trip per eight frames and the frames after [END_AUDIO] in the last batch are discarded";

pub const THE_FSQ_KERNEL_ROUNDS_AFTER_EVERY_OPERATION_LIKE_THE_HOST_BF16_PATH: &str =
    "tts_fsq, like the Euler step of voxtral_tts_flow_step, computes each product and sum with \
     __fmul_rn/__fadd_rn in f32 and rounds to bf16 after every operation, the arithmetic \
     half::bf16 performs on the host, so the codes of a trajectory depend on the backend only \
     through the velocity GEMMs";

struct FrameBufs {
    dev: CudaDevice,
    h: Tensor,
    input: Tensor,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    pos: CudaSlice<i32>,
    rows_pos: CudaSlice<i32>,
    pos_end: CudaSlice<i32>,
    fi: CudaSlice<i32>,
    noise: CudaSlice<u16>,
    ring: CudaSlice<u32>,
    offsets: CudaSlice<i32>,
    teacher: CudaSlice<i32>,
    teacher_len: CudaSlice<i32>,
    fan_in: CudaSlice<u32>,
}

pub struct TtsGraphState {
    cache: KvCache,
    capacity: usize,
    stream: Arc<CudaStream>,
    runner: nv_kernels::graph::CudaGraphRunner,
    arena: Option<step_arena::StepArena>,
    arena_filled_eagerly: bool,
    bufs: FrameBufs,
    e4m3: Option<Vec<[E4m3Rows; 4]>>,
}

pub const CANDLE_LAYOUT_PARAMS_GO_THROUGH_THE_CAPTURE_CACHE_OR_A_REPLAY_READS_A_FREED_HOST_VEC: &str =
    "a strided candle op uploads its dims and strides from a host Vec; captured as a memcpy node \
     with a pageable source, every replay re-reads that address after the Vec is gone, so the \
     body keeps every candle operand contiguous and runs under enable_cuda_graph_htod_cache, \
     which serves such uploads from a content-keyed device cache filled by the eager first frame \
     and refuses an uncached upload during capture";

pub const THE_FIRST_FRAME_RUNS_EAGERLY_SO_THE_ARENA_SLOTS_ARE_ORDINARY_ALLOCATIONS_NOT_GRAPH_MEMORY: &str =
    "a slot allocated inside the capture is a graph allocation that is never freed in the graph; \
     the graph allocator then overlaps a later temporary with it and the replay writes through \
     the overlap, so the arena is filled by one eager frame before the capture and every graph \
     allocation is a temporary that is freed inside the graph";

fn sp<T>(s: &CudaSlice<T>, stream: &Arc<CudaStream>) -> u64 {
    s.device_ptr(stream).0
}

fn tp(t: &Tensor) -> Result<u64> {
    step_arena::device_ptr_for_a_kernel_to_write(t)
}

fn arena(dev: &Device, dtype: DType, shape: &[usize]) -> Result<Tensor> {
    step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(dev, dtype, shape)
}

fn rc(code: i32, what: &str) -> Result<()> {
    ensure!(code == 0, "{what} returned {code}");
    Ok(())
}

fn backbone_step(m: &VoxtralTts, cache: &KvCache, b: &FrameBufs, dev: &Device, cu: *mut c_void) -> Result<Tensor> {
    let stream = nv_layers::cuda_stream::current_stream(&b.dev);
    let t = &m.backbone_cfg;
    let (nh, nkv, hd, inter) =
        (t.num_attention_heads, t.num_key_value_heads, t.head_dim, t.intermediate_size);
    let scratch_elems = nv_kernels::cuda::flash_splitk_scratch_elems_mk(nh as i32, hd as i32, 1);
    let mut x = b.input.clone();
    for (i, layer) in m.layers.iter().enumerate() {
        let h = layer.attn_norm.forward(&x)?;
        let qkv = layer.qkv_proj.forward(&h)?.contiguous()?;
        ensure!(qkv.dtype() == DType::BF16, "qkv rows must be bf16");
        let q = arena(dev, DType::F32, &[nh * hd])?;
        let (kc, vc) = cache.layer_tensors(i)?;
        let (kcp, vcp) = (tp(kc)?, tp(vc)?);
        let code = unsafe {
            spec::nv_kernels_voxtral_spec_qkv_prep(
                cu,
                tp(&qkv)? as *const u16,
                tp(&q)? as *mut f32,
                kcp as *mut u16,
                vcp as *mut u16,
                tp(&b.cos)? as *const f32,
                tp(&b.sin)? as *const f32,
                sp(&b.rows_pos, &stream) as *const i32,
                1,
                nh as i32,
                nkv as i32,
                hd as i32,
            )
        };
        rc(code, "voxtral_spec_qkv_prep")?;
        let attn = arena(dev, DType::BF16, &[nh * hd])?;
        let scratch = arena(dev, DType::F32, &[scratch_elems])?;
        let code = unsafe {
            nv_kernels::cuda::flash_decode_fused_bf16kv_mk(
                cu,
                tp(&q)? as *const f32,
                kcp as *const u16,
                vcp as *const u16,
                tp(&attn)? as *mut u16,
                sp(&b.pos_end, &stream) as *const i32,
                0,
                1,
                tp(&scratch)? as *mut f32,
                sp(&b.fan_in, &stream) as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
            )
        };
        rc(code, "flash_decode_fused_bf16kv_mk")?;
        let attn = layer.o_proj.forward(&attn.reshape((1, 1, nh * hd))?)?;
        let (h, x_mid) = layer.ffn_norm.forward_residual(&attn, &x)?;
        let gate_up = layer.gate_up_proj.forward(&h)?.contiguous()?;
        let act = arena(dev, DType::BF16, &[1, 1, inter])?;
        let code = unsafe {
            spec::nv_kernels_voxtral_spec_silu_mul_halves(
                cu,
                tp(&gate_up)? as *const u16,
                tp(&act)? as *mut u16,
                1,
                inter as i32,
            )
        };
        rc(code, "voxtral_spec_silu_mul_halves")?;
        x = (&x_mid + layer.down_proj.forward(&act)?)?;
    }
    m.norm.forward(&x)
}

pub const THE_BF16_BACKBONE_LAYER_IS_SIX_KERNELS_BECAUSE_THE_NORMS_THE_SILU_AND_THE_RESIDUALS_RIDE_IN_THE_GEMV_PROLOGUES_AND_EPILOGUES: &str =
    "voxtral_tts_backbone_gemv stages the activation row once per block in shared memory, where \
     a prologue applies the attention norm, the residual sum and the ffn norm, or the silu \
     product of the gate_up halves, with the arithmetic and reduction order of the rmsnorm, \
     rmsnorm_residual and spec_silu_mul_halves kernels, and the down epilogue adds the residual \
     as candle's bf16 add does; the dot product keeps gemv_bf16's lane order with four weight \
     loads in flight, so the layer's output is bitwise the ten-kernel chain's and reads the \
     weights once at a higher fraction of the memory bandwidth";

fn backbone_fusable(m: &VoxtralTts) -> bool {
    m.layers.iter().all(|l| {
        [&l.qkv_proj, &l.o_proj, &l.gate_up_proj, &l.down_proj].iter().all(|lin| lin.weight().is_some())
    })
}

#[allow(clippy::too_many_arguments)]
pub const BACKBONE_WEIGHTS_ENV: &str = "NV_VOXTRAL_TTS_BACKBONE";

pub const AN_E4M3_BACKBONE_IS_AN_OPT_IN_ARM_UNTIL_THE_CORPUS_INSTRUMENT_BLESSES_IT: &str =
    "NV_VOXTRAL_TTS_BACKBONE=e4m3 makes the graph state quantize the 26 layers' bf16 weights to \
     per-row absmax e4m3 on the device and run the fused step over them, halving the backbone's \
     bytes per frame; the bf16 weights stay resident for the eager and traced paths, so this is \
     a measurement arm for corpus_word_errors_per_configuration, and a default needs the loader \
     to build the rows in place of the bf16 linears";

pub struct E4m3Rows {
    wq: CudaSlice<u8>,
    scale: CudaSlice<f32>,
    n: usize,
    k: usize,
}

fn e4m3_backbone_opted_in_by_env() -> Result<bool> {
    let _ = AN_E4M3_BACKBONE_IS_AN_OPT_IN_ARM_UNTIL_THE_CORPUS_INSTRUMENT_BLESSES_IT;
    match std::env::var(BACKBONE_WEIGHTS_ENV).unwrap_or_default().as_str() {
        "" | "bf16" => Ok(false),
        "e4m3" => Ok(true),
        other => bail!("{BACKBONE_WEIGHTS_ENV}={other:?}: expected bf16 or e4m3"),
    }
}

fn e4m3_layers(m: &VoxtralTts, e4m3: bool) -> Result<Option<Vec<[E4m3Rows; 4]>>> {
    if !e4m3 {
        return Ok(None);
    }
    ensure!(backbone_fusable(m), "{BACKBONE_WEIGHTS_ENV}=e4m3 quantizes bf16 backbone weights");
    let rows = |lin: &Linear| -> Result<E4m3Rows> {
        let w = lin.weight().expect("fusable");
        let (n, k) = w.dims2()?;
        let (wq, scale) = nv_layers::lm_head_resident_quant::rowquant_e4m3_dev(w)?;
        Ok(E4m3Rows { wq, scale, n, k })
    };
    m.layers
        .iter()
        .map(|l| Ok([rows(&l.qkv_proj)?, rows(&l.o_proj)?, rows(&l.gate_up_proj)?, rows(&l.down_proj)?]))
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

enum Weight<'a> {
    Bf16(&'a Tensor),
    E4m3(&'a E4m3Rows),
}

#[allow(clippy::too_many_arguments)]
fn backbone_gemv(
    cu: *mut c_void,
    stream: &Arc<CudaStream>,
    mode: i32,
    w: Weight,
    x: &Tensor,
    aux: Option<&Tensor>,
    wn: Option<&Tensor>,
    eps: f32,
    side: Option<&Tensor>,
    y: &Tensor,
) -> Result<()> {
    let (wp, scale, n, k) = match w {
        Weight::Bf16(t) => {
            let (n, k) = t.dims2()?;
            (tp(t)?, 0u64, n, k)
        }
        Weight::E4m3(r) => (sp(&r.wq, stream), sp(&r.scale, stream), r.n, r.k),
    };
    let opt = |t: Option<&Tensor>| t.map_or(Ok(0u64), tp);
    let code = unsafe {
        backbone::nv_kernels_voxtral_tts_backbone_gemv(
            cu,
            mode,
            wp as *const c_void,
            scale as *const f32,
            tp(x)? as *const u16,
            opt(aux)? as *const u16,
            opt(wn)? as *const u16,
            eps,
            opt(side)? as *mut u16,
            tp(y)? as *mut u16,
            n as i32,
            k as i32,
        )
    };
    rc(code, "voxtral_tts_backbone_gemv")
}

fn backbone_step_fused(
    m: &VoxtralTts,
    cache: &KvCache,
    b: &FrameBufs,
    e4m3: Option<&[[E4m3Rows; 4]]>,
    dev: &Device,
    cu: *mut c_void,
) -> Result<Tensor> {
    let _ = THE_BF16_BACKBONE_LAYER_IS_SIX_KERNELS_BECAUSE_THE_NORMS_THE_SILU_AND_THE_RESIDUALS_RIDE_IN_THE_GEMV_PROLOGUES_AND_EPILOGUES;
    let stream = nv_layers::cuda_stream::current_stream(&b.dev);
    let t = &m.backbone_cfg;
    let (nh, nkv, hd, d, inter) =
        (t.num_attention_heads, t.num_key_value_heads, t.head_dim, t.hidden_size, t.intermediate_size);
    let eps = t.rms_norm_eps as f32;
    let scratch_elems = nv_kernels::cuda::flash_splitk_scratch_elems_mk(nh as i32, hd as i32, 1);
    fn weight<'a>(e4m3: Option<&'a [[E4m3Rows; 4]]>, i: usize, j: usize, lin: &'a Linear) -> Result<Weight<'a>> {
        Ok(match e4m3 {
            Some(rows) => Weight::E4m3(&rows[i][j]),
            None => Weight::Bf16(lin.weight().ok_or_else(|| anyhow::anyhow!("the fused backbone needs bf16 weights"))?),
        })
    }
    let mut x = b.input.reshape(d)?;
    for (i, layer) in m.layers.iter().enumerate() {
        let qkv = arena(dev, DType::BF16, &[(nh + 2 * nkv) * hd])?;
        backbone_gemv(cu, &stream, backbone::GEMV_RMSNORM_PROLOGUE, weight(e4m3, i, 0, &layer.qkv_proj)?, &x, None, Some(layer.attn_norm.weight_bf16()), eps, None, &qkv)?;
        let q = arena(dev, DType::F32, &[nh * hd])?;
        let (kc, vc) = cache.layer_tensors(i)?;
        let (kcp, vcp) = (tp(kc)?, tp(vc)?);
        let code = unsafe {
            spec::nv_kernels_voxtral_spec_qkv_prep(
                cu,
                tp(&qkv)? as *const u16,
                tp(&q)? as *mut f32,
                kcp as *mut u16,
                vcp as *mut u16,
                tp(&b.cos)? as *const f32,
                tp(&b.sin)? as *const f32,
                sp(&b.rows_pos, &stream) as *const i32,
                1,
                nh as i32,
                nkv as i32,
                hd as i32,
            )
        };
        rc(code, "voxtral_spec_qkv_prep")?;
        let attn = arena(dev, DType::BF16, &[nh * hd])?;
        let scratch = arena(dev, DType::F32, &[scratch_elems])?;
        let code = unsafe {
            nv_kernels::cuda::flash_decode_fused_bf16kv_mk(
                cu,
                tp(&q)? as *const f32,
                kcp as *const u16,
                vcp as *const u16,
                tp(&attn)? as *mut u16,
                sp(&b.pos_end, &stream) as *const i32,
                0,
                1,
                tp(&scratch)? as *mut f32,
                sp(&b.fan_in, &stream) as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
            )
        };
        rc(code, "flash_decode_fused_bf16kv_mk")?;
        let o = arena(dev, DType::BF16, &[d])?;
        backbone_gemv(cu, &stream, backbone::GEMV_PLAIN, weight(e4m3, i, 1, &layer.o_proj)?, &attn, None, None, 0.0, None, &o)?;
        let x_mid = arena(dev, DType::BF16, &[d])?;
        let gate_up = arena(dev, DType::BF16, &[2 * inter])?;
        backbone_gemv(
            cu,
            &stream,
            backbone::GEMV_RESIDUAL_RMSNORM_PROLOGUE_WRITES_THE_SUM,
            weight(e4m3, i, 2, &layer.gate_up_proj)?,
            &o,
            Some(&x),
            Some(layer.ffn_norm.weight_bf16()),
            eps,
            Some(&x_mid),
            &gate_up,
        )?;
        let x_new = arena(dev, DType::BF16, &[d])?;
        backbone_gemv(cu, &stream, backbone::GEMV_SILU_PROLOGUE_RESIDUAL_EPILOGUE, weight(e4m3, i, 3, &layer.down_proj)?, &gate_up, Some(&x_mid), None, 0.0, None, &x_new)?;
        x = x_new;
    }
    m.norm.forward(&x.reshape((1, 1, d))?)
}

fn frame_body(m: &VoxtralTts, cache: &KvCache, b: &FrameBufs, e4m3: Option<&[[E4m3Rows; 4]]>) -> Result<()> {
    let _ = ONE_GRAPH_PER_FRAME_RUNS_THE_SEMANTIC_HEAD_THE_FLOW_ODE_THE_EMBEDDING_AND_THE_NEXT_BACKBONE_STEP;
    let _ = THE_FSQ_KERNEL_ROUNDS_AFTER_EVERY_OPERATION_LIKE_THE_HOST_BF16_PATH;
    let stream = nv_layers::cuda_stream::current_stream(&b.dev);
    let cu = stream.cu_stream() as *mut c_void;
    let dev = Device::Cuda(b.dev.clone());
    let cb = m.codebooks();
    let (n, nc) = (cb.n_acoustic, cb.n_codebooks());
    let d = m.backbone_cfg.hidden_size;
    let slots = FRAMES_PER_READBACK as i32;
    let fi = sp(&b.fi, &stream) as *const i32;
    let ring = sp(&b.ring, &stream) as *mut u32;
    let code = unsafe {
        frame::nv_kernels_voxtral_tts_begin(
            cu,
            sp(&b.pos, &stream) as *const i32,
            sp(&b.rows_pos, &stream) as *mut i32,
            sp(&b.pos_end, &stream) as *mut i32,
            fi,
            sp(&b.noise, &stream) as *const u16,
            tp(&b.x)? as *mut u16,
            slots,
            n as i32,
        )
    };
    rc(code, "tts_begin")?;
    let logits = m.semantic_head.forward(&b.h)?.contiguous()?;
    ensure!(logits.dtype() == DType::BF16, "semantic logits must be bf16");
    let code = unsafe {
        frame::nv_kernels_voxtral_tts_semantic_argmax(
            cu,
            tp(&logits)? as *const u16,
            cb.semantic_rows() as i32,
            fi,
            slots,
            nc as i32,
            ring,
            END_AUDIO as i32,
        )
    };
    rc(code, "tts_semantic_argmax")?;
    m.flow_ode_device(&b.h, &b.x)?;
    let code = unsafe {
        frame::nv_kernels_voxtral_tts_fsq(
            cu,
            tp(&b.x)? as *const u16,
            n as i32,
            (cb.acoustic_levels - 1) as f32,
            fi,
            slots,
            nc as i32,
            ring,
            END_AUDIO as i32,
            N_AUDIO_SPECIAL as i32,
        )
    };
    rc(code, "tts_fsq")?;
    let code = unsafe {
        frame::nv_kernels_voxtral_tts_embed_sum(
            cu,
            tp(&m.codebook_embed)? as *const u16,
            sp(&b.offsets, &stream) as *const i32,
            ring,
            sp(&b.teacher, &stream) as *const i32,
            sp(&b.teacher_len, &stream) as *const i32,
            fi,
            slots,
            nc as i32,
            tp(&b.input)? as *mut u16,
            d as i32,
        )
    };
    rc(code, "tts_embed_sum")?;
    let h = if backbone_fusable(m) {
        backbone_step_fused(m, cache, b, e4m3, &dev, cu)?
    } else {
        backbone_step(m, cache, b, &dev, cu)?
    }
    .contiguous()?;
    unsafe {
        cudarc::driver::result::memcpy_dtod_async(tp(&b.h)?, tp(&h)?, d * 2, stream.cu_stream())?;
    }
    let code = unsafe {
        frame::nv_kernels_voxtral_tts_advance(
            cu,
            sp(&b.pos, &stream) as *mut i32,
            fi as *mut i32,
        )
    };
    rc(code, "tts_advance")
}

impl TtsGraphState {
    pub fn new(model: &VoxtralTts, capacity: usize) -> Result<Self> {
        Self::new_with(model, capacity, e4m3_backbone_opted_in_by_env()?)
    }

    pub fn new_with(model: &VoxtralTts, capacity: usize, e4m3_backbone: bool) -> Result<Self> {
        let Device::Cuda(dev) = model.device().clone() else {
            bail!("TtsGraphState requires a CUDA device")
        };
        let ctx = dev.cuda_stream().context().clone();
        if ctx.is_event_tracking() {
            unsafe { ctx.disable_event_tracking() };
        }
        let stream = dev.cuda_stream();
        let t = &model.backbone_cfg;
        let d = t.hidden_size;
        ensure!(model.flow_cfg.hidden_size == d, "the flow head and the backbone share one hidden size");
        let cb = model.codebooks();
        let (n, nc) = (cb.n_acoustic, cb.n_codebooks());
        let device = model.device();
        let cache = KvCache::with_dims(
            t.num_hidden_layers,
            capacity,
            t.num_key_value_heads,
            t.head_dim,
            device,
            DType::BF16,
        )?;
        let offsets: Vec<i32> = (0..nc).map(|k| cb.offset(k) as i32).collect();
        let bufs = FrameBufs {
            dev: dev.clone(),
            h: Tensor::zeros((1, 1, d), DType::BF16, device)?,
            input: Tensor::zeros((1, 1, d), DType::BF16, device)?,
            x: Tensor::zeros((1, n), DType::BF16, device)?,
            cos: model.rope.cos().contiguous()?,
            sin: model.rope.sin().contiguous()?,
            pos: stream.alloc_zeros::<i32>(1)?,
            rows_pos: stream.alloc_zeros::<i32>(1)?,
            pos_end: stream.alloc_zeros::<i32>(1)?,
            fi: stream.alloc_zeros::<i32>(1)?,
            noise: stream.alloc_zeros::<u16>(FRAMES_PER_READBACK * n)?,
            ring: stream.alloc_zeros::<u32>(FRAMES_PER_READBACK * nc)?,
            offsets: stream.clone_htod(&offsets)?,
            teacher: stream.alloc_zeros::<i32>(capacity * nc)?,
            teacher_len: stream.alloc_zeros::<i32>(1)?,
            fan_in: stream.alloc_zeros::<u32>(t.num_attention_heads.max(64))?,
        };
        stream.synchronize()?;
        Ok(Self {
            cache,
            capacity,
            stream: stream.clone(),
            runner: nv_kernels::graph::CudaGraphRunner::new(stream),
            arena: Some(step_arena::StepArena::new()),
            arena_filled_eagerly: false,
            bufs,
            e4m3: e4m3_layers(model, e4m3_backbone)?,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn graph_nodes(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn graph_census(&self) -> (std::collections::BTreeMap<u32, usize>, std::collections::BTreeMap<String, usize>) {
        (
            self.runner.cached_node_type_census_type_ids_are_cugraphnodetype(),
            self.runner.cached_kernel_name_census(),
        )
    }

    fn begin(&mut self, prompt_len: usize, h: &Tensor, teacher: Option<&[Vec<u32>]>, nc: usize) -> Result<()> {
        let stream = &self.stream;
        stream.memcpy_htod(&[prompt_len as i32], &mut self.bufs.pos)?;
        stream.memcpy_htod(&[0i32], &mut self.bufs.fi)?;
        let flat: Vec<i32> = teacher.into_iter().flatten().flatten().map(|&c| c as i32).collect();
        let len = teacher.map_or(0, <[Vec<u32>]>::len);
        ensure!(flat.len() == len * nc, "teacher frames must hold {nc} codes each");
        stream.memcpy_htod(&[len as i32], &mut self.bufs.teacher_len)?;
        if len > 0 {
            stream.memcpy_htod(&flat, &mut self.bufs.teacher.slice_mut(0..flat.len()))?;
        }
        let h = h.contiguous()?;
        unsafe {
            cudarc::driver::result::memcpy_dtod_async(tp(&self.bufs.h)?, tp(&h)?, h.elem_count() * 2, stream.cu_stream())?;
        }
        stream.synchronize()?;
        Ok(())
    }

    fn replay(&mut self, model: &VoxtralTts) -> Result<()> {
        if let Some(a) = self.arena.take() {
            step_arena::install(a);
        }
        let Self { cache, runner, bufs, arena_filled_eagerly, e4m3, .. } = self;
        let e4m3 = e4m3.as_deref();
        let _ = THE_FIRST_FRAME_RUNS_EAGERLY_SO_THE_ARENA_SLOTS_ARE_ORDINARY_ALLOCATIONS_NOT_GRAPH_MEMORY;
        let _ = CANDLE_LAYOUT_PARAMS_GO_THROUGH_THE_CAPTURE_CACHE_OR_A_REPLAY_READS_A_FREED_HOST_VEC;
        let _htod_cache = bufs.dev.enable_cuda_graph_htod_cache();
        let result = if !*arena_filled_eagerly {
            *arena_filled_eagerly = true;
            step_arena::begin_step();
            frame_body(model, cache, bufs, e4m3)
        } else {
            runner.run(1, |_s| {
                step_arena::begin_step();
                frame_body(model, cache, bufs, e4m3)
            })
        };
        self.arena = step_arena::uninstall();
        result
    }
}

impl Drop for TtsGraphState {
    fn drop(&mut self) {
        let teardown =
            nv_quant::graph_teardown::GraphTeardown::for_a_stream_this_engine_did_not_fork(
                &self.stream,
            );
        let runner = &mut self.runner;
        teardown.run_owned_streams_only(|| runner.invalidate());
    }
}

impl VoxtralTts {
    pub fn synthesize_graphed(
        &self,
        state: &mut TtsGraphState,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        teacher: Option<&[Vec<u32>]>,
    ) -> Result<Synthesis> {
        self.synthesize_graphed_with(state, text, voice, noise, max_frames, teacher, &mut |_| Ok(true))
    }

    pub fn synthesize_graphed_streaming(
        &self,
        state: &mut TtsGraphState,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Synthesis> {
        self.synthesize_graphed_with(state, text, voice, noise, max_frames, None, on_frame)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn synthesize_graphed_with(
        &self,
        state: &mut TtsGraphState,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        teacher: Option<&[Vec<u32>]>,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Synthesis> {
        let ids = self.prompt_ids(text, voice)?;
        let max_frames = teacher.map_or(max_frames, <[Vec<u32>]>::len);
        ensure!(
            ids.len() + max_frames + 1 <= state.capacity,
            "prompt of {} tokens plus {max_frames} frames exceeds the graph's capacity {}",
            ids.len(),
            state.capacity
        );
        let cb = self.codebooks();
        let (n, nc) = (cb.n_acoustic, cb.n_codebooks());
        let t0 = std::time::Instant::now();
        state.cache.reset();
        let h = self.step(&self.prompt_embeds(&ids, voice)?, &mut state.cache)?;
        state.begin(ids.len(), &h, teacher, nc)?;
        let mut out = Synthesis { prompt_ids: ids, ..Default::default() };
        out.stage_ms.prefill = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let mut frame_idx = 0;
        'frames: while frame_idx < max_frames {
            let slot = frame_idx % FRAMES_PER_READBACK;
            let k = (FRAMES_PER_READBACK - slot)
                .min(max_frames - frame_idx)
                .min(if frame_idx == 0 { FIRST_READBACK_IS_TWO_FRAMES_SO_THE_FIRST_AUDIO_WAITS_FOR_TWO_REPLAYS_NOT_EIGHT } else { usize::MAX });
            let mut rows = Vec::with_capacity(k * n);
            for j in 0..k {
                let r = noise(frame_idx + j);
                ensure!(r.len() == n, "noise row has {} values, the flow head needs {n}", r.len());
                rows.extend(r.iter().map(|v| bf16::from_f32(*v).to_bits()));
            }
            state.stream.memcpy_htod(&rows, &mut state.bufs.noise.slice_mut(slot * n..(slot + k) * n))?;
            let e0 = state.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            for _ in 0..k {
                state.replay(self)?;
            }
            let e1 = state.stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
            state.stream.synchronize()?;
            out.stage_ms.gpu += e0.elapsed_ms(&e1)? as f64;
            let ring: Vec<u32> = state.stream.clone_dtoh(&state.bufs.ring)?;
            for j in slot..slot + k {
                let frame = ring[j * nc..(j + 1) * nc].to_vec();
                let end = frame[0] == END_AUDIO;
                let go_on = on_frame(&frame)?;
                out.frames.push(frame);
                frame_idx += 1;
                out.stage_ms.steps += 1;
                if (end && teacher.is_none()) || !go_on {
                    break 'frames;
                }
            }
        }
        out.ended = out.frames.last().is_some_and(|f| f[0] == END_AUDIO);
        out.stage_ms.graph = t1.elapsed().as_secs_f64() * 1000.0;
        Ok(out)
    }
}

#[cfg(test)]
mod bench {
    use super::*;
    use crate::voxtral_tts::SeededNoise;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    fn timed_graph(
        stream: &Arc<CudaStream>,
        dev: &CudaDevice,
        reps: usize,
        body: &mut dyn FnMut() -> Result<()>,
    ) -> Result<(f64, usize)> {
        let mut runner = nv_kernels::graph::CudaGraphRunner::new(stream.clone());
        let mut arena = Some(step_arena::StepArena::new());
        let mut once = |runner: &mut nv_kernels::graph::CudaGraphRunner, eager: bool, body: &mut dyn FnMut() -> Result<()>| -> Result<()> {
            if let Some(a) = arena.take() {
                step_arena::install(a);
            }
            let _h = dev.enable_cuda_graph_htod_cache();
            let r = if eager {
                step_arena::begin_step();
                body()
            } else {
                runner.run(1, |_| {
                    step_arena::begin_step();
                    body()
                })
            };
            arena = step_arena::uninstall();
            r
        };
        once(&mut runner, true, body)?;
        once(&mut runner, false, body)?;
        stream.synchronize()?;
        let e0 = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        for _ in 0..reps {
            once(&mut runner, false, body)?;
        }
        let e1 = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
        stream.synchronize()?;
        Ok((e0.elapsed_ms(&e1)? as f64 / reps as f64, runner.cached_node_count()))
    }

    #[test]
    #[ignore]
    fn frame_sections_bench() {
        if std::env::var("NV_VOXTRAL_TTS_TEST").ok().as_deref() != Some("1") {
            eprintln!("skip: NV_VOXTRAL_TTS_TEST=1 opts in");
            return;
        }
        let dir = VoxtralTts::default_dir().unwrap();
        let device = Device::new_cuda_with_stream(0).unwrap();
        let m = VoxtralTts::load_dir(&dir, &device).unwrap();
        let Device::Cuda(dev) = device.clone() else { unreachable!() };
        let stream = dev.cuda_stream();
        let text = "The quick brown fox jumps over the lazy dog.";
        let mut state = TtsGraphState::new(&m, 1024).unwrap();
        let rounds: usize = std::env::var("NV_BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let mut frame_ms = Vec::new();
        for _ in 0..rounds {
            let mut noise = SeededNoise::new(42);
            let n = m.codebooks().n_acoustic;
            let s = m.synthesize_graphed(&mut state, text, "neutral_female", &mut |_| noise.frame(n), 64, None).unwrap();
            frame_ms.push(s.stage_ms.gpu / s.stage_ms.steps as f64);
        }
        eprintln!("frame graph: {} nodes; per frame gpu ms {:?} median {:.3}", state.graph_nodes(), frame_ms, median(frame_ms.clone()));
        let (types, kernels) = state.graph_census();
        eprintln!("node types {types:?}");
        for (k, n) in &kernels {
            eprintln!("  {n:4} {k}");
        }
        for (k, n) in state.runner.cached_memcpy_consumer_census_key_is_bytes_and_kernel_name() {
            eprintln!("  memcpy {n:4} {k:?}");
        }
        let cu = stream.cu_stream() as *mut c_void;
        let TtsGraphState { cache, bufs, e4m3, .. } = &state;
        let e4m3 = e4m3.as_deref();
        let bits = |t: &Tensor| -> Vec<u16> { t.flatten_all().unwrap().to_vec1::<bf16>().unwrap().iter().map(|v| v.to_bits()).collect() };
        let h_ref = bits(&backbone_step(&m, cache, bufs, &device, cu).unwrap());
        let h_fused = bits(&backbone_step_fused(&m, cache, bufs, e4m3, &device, cu).unwrap());
        let diff = h_ref.iter().zip(&h_fused).filter(|(a, b)| a != b).count();
        eprintln!("fused backbone vs ten-kernel chain: {diff} of {} h bits differ", h_ref.len());
        let mut bb = Vec::new();
        let mut fb = Vec::new();
        let (mut nodes, mut fnodes) = (0, 0);
        for _ in 0..rounds {
            let (ms, n) = timed_graph(&stream, &dev, 20, &mut || backbone_step(&m, cache, bufs, &device, cu).map(|_| ())).unwrap();
            bb.push(ms);
            nodes = n;
            let (ms, n) = timed_graph(&stream, &dev, 20, &mut || backbone_step_fused(&m, cache, bufs, e4m3, &device, cu).map(|_| ())).unwrap();
            fb.push(ms);
            fnodes = n;
        }
        eprintln!("backbone-only graph: {nodes} nodes; per step ms {bb:?} median {:.3}", median(bb.clone()));
        eprintln!("fused backbone graph: {fnodes} nodes; per step ms {fb:?} median {:.3}", median(fb.clone()));
        let x3072 = Tensor::zeros((1, 1, 3072), DType::BF16, &device).unwrap();
        let x4096 = Tensor::zeros((1, 1, 4096), DType::BF16, &device).unwrap();
        let x9216 = Tensor::zeros((1, 1, 9216), DType::BF16, &device).unwrap();
        let names = ["qkv", "o", "gate_up", "down"];
        let mut per: [Vec<f64>; 4] = Default::default();
        for _ in 0..rounds {
            let mut sums = [0f64; 4];
            for layer in &m.layers {
                let lins = [
                    (&layer.qkv_proj, &x3072),
                    (&layer.o_proj, &x4096),
                    (&layer.gate_up_proj, &x3072),
                    (&layer.down_proj, &x9216),
                ];
                for (i, (lin, x)) in lins.iter().enumerate() {
                    let e0 = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    let _y = lin.forward(x).unwrap();
                    let e1 = stream.record_event(Some(CUevent_flags::CU_EVENT_DEFAULT)).unwrap();
                    stream.synchronize().unwrap();
                    sums[i] += e0.elapsed_ms(&e1).unwrap() as f64;
                }
            }
            for i in 0..4 {
                per[i].push(sums[i] / m.layers.len() as f64);
            }
        }
        let bytes = [6144.0 * 3072.0 * 2.0, 3072.0 * 4096.0 * 2.0, 18432.0 * 3072.0 * 2.0, 3072.0 * 9216.0 * 2.0];
        for i in 0..4 {
            let ms = median(per[i].clone());
            eprintln!("gemv {:8}: {ms:.4} ms/layer, {:.0} GB/s", names[i], bytes[i] / ms / 1e6);
        }
    }
}
