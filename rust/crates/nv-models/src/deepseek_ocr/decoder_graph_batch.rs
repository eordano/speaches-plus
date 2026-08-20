#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{CudaDevice, DType, Device, Tensor};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;

use super::decoder::{
    select_next_token, DeepseekMoe, DeepseekOcrDecoder, DeepseekOcrKvCache, FeedForward,
    GenerateOptions, SplitMix64,
};
use super::decoder_graph::{
    cast_bf16_to_f32_into, cast_f32_to_bf16_into, cast_tensor_bf16_to_f32, cast_tensor_f32_to_bf16,
    cuda_dev, dense_decode_ffn, graph_debug, graph_prealloc_enabled, graph_supported, lock_init,
    model_uses_nvfp4, residual_add_scale_bf16_into, rope_apply_inplace_f32, tensor_ptr_bf16,
    tensor_ptr_u32, wrap_bf16, wrap_f32, MoePrealloc,
};
use crate::laguna_fa2::{batch_decode_fwd_bf16, varlen_fwd_bf16, BatchDecodeArgs, VarlenArgs};

pub fn attn_batched() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_ATTN_BATCH")
            .map(|v| v != "0")
            .unwrap_or(true)
    })
}
use nv_kernels::graph::CudaGraphRunner;

pub const DEFAULT_BUCKETS: &[usize] = &[1, 2, 4, 8];

pub fn parse_buckets(spec: Option<&str>) -> Vec<usize> {
    let mut v: Vec<usize> = match spec {
        Some(s) => s
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .collect(),
        None => Vec::new(),
    };
    if v.is_empty() {
        v = DEFAULT_BUCKETS.to_vec();
    }
    v.sort_unstable();
    v.dedup();
    v
}

pub fn buckets_from_env() -> Vec<usize> {
    parse_buckets(std::env::var("NV_DSOCR_BSN_BUCKETS").ok().as_deref())
}

pub fn bucket_for(buckets: &[usize], b: usize) -> Option<usize> {
    if b == 0 {
        return None;
    }
    buckets.iter().copied().find(|&s| s >= b)
}

pub struct BatchSampler {
    rng: SplitMix64,
}

impl BatchSampler {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: SplitMix64::new(seed),
        }
    }

    pub fn next_token(
        &mut self,
        logits: &mut [f32],
        all_tokens: &[u32],
        opts: &GenerateOptions,
    ) -> Result<u32> {
        select_next_token(logits, all_tokens, opts, &mut self.rng)
    }
}

fn off(base: u64, elems: usize, bytes_per: usize) -> u64 {
    base + (elems * bytes_per) as u64
}

struct BatchScratch {
    k: usize,
    e: usize,
    inter: usize,
    hidden: usize,
    max_b: usize,
    seeds: CudaSlice<u64>,
    scores_f32: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    trash_token: CudaSlice<u32>,
    ids: CudaSlice<i32>,
    ids_log: Option<CudaSlice<i32>>,
    weights: CudaSlice<f32>,
    h_rows: CudaSlice<bf16>,
    shared_f32: CudaSlice<f32>,
    out: Tensor,
    attn_o: Tensor,
    attn_lse: CudaSlice<f32>,
}

pub fn moe_slot_batched() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        !matches!(
            std::env::var("NV_DSOCR_MOE_SLOTBATCH").as_deref(),
            Ok("0") | Ok("off") | Ok("false")
        )
    })
}

pub fn prefill_per_thread_stream() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("NV_DSOCR_PREFILL_STREAM").as_deref(),
            Ok("fork") | Ok("perthread") | Ok("1")
        )
    })
}

pub fn idlog_path() -> Option<String> {
    match std::env::var("NV_DSOCR_MOE_IDLOG") {
        Ok(v) if !v.is_empty() && v != "0" => Some(v),
        _ => None,
    }
}

unsafe impl Send for BatchScratch {}

impl BatchScratch {
    #[allow(clippy::too_many_arguments)]
    fn new(
        k: usize,
        e: usize,
        inter: usize,
        hidden: usize,
        n_heads: usize,
        head_dim: usize,
        max_b: usize,
        layers: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(
            k > 0 && e > 0 && k <= e,
            "dsocr batch scratch: bad k={k} e={e}"
        );
        anyhow::ensure!(max_b > 0, "dsocr batch scratch: max_b must be > 0");
        let dev = cuda_dev(device)?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let seeds = stream
            .alloc_zeros::<u64>(max_b)
            .map_err(|e| anyhow::anyhow!(e))?;
        let scores_f32 = stream
            .alloc_zeros::<f32>(max_b * e)
            .map_err(|e| anyhow::anyhow!(e))?;
        let probs = stream
            .alloc_zeros::<f32>(max_b * e)
            .map_err(|e| anyhow::anyhow!(e))?;
        let trash_token = stream
            .alloc_zeros::<u32>(max_b)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ids = stream
            .alloc_zeros::<i32>(max_b * k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ids_log = match idlog_path() {
            Some(_) => Some(
                stream
                    .alloc_zeros::<i32>(layers * max_b * k)
                    .map_err(|e| anyhow::anyhow!(e))?,
            ),
            None => None,
        };
        let weights = stream
            .alloc_zeros::<f32>(max_b * k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let h_rows = stream
            .alloc_zeros::<bf16>(max_b * k * inter)
            .map_err(|e| anyhow::anyhow!(e))?;
        let shared_f32 = stream
            .alloc_zeros::<f32>(max_b * hidden)
            .map_err(|e| anyhow::anyhow!(e))?;
        let attn_lse = stream
            .alloc_zeros::<f32>(max_b * n_heads)
            .map_err(|e| anyhow::anyhow!(e))?;
        let out = Tensor::zeros((max_b, hidden), DType::BF16, device)?;
        let attn_o = Tensor::zeros((max_b, n_heads, head_dim), DType::BF16, device)?;
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            k,
            e,
            inter,
            hidden,
            max_b,
            seeds,
            scores_f32,
            probs,
            trash_token,
            ids,
            ids_log,
            weights,
            h_rows,
            shared_f32,
            out,
            attn_o,
            attn_lse,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn moe_decode_ffn_batch(
    moe: &DeepseekMoe,
    x_normed: &Tensor,
    resid: &Tensor,
    scratch: &mut BatchScratch,
    mut moe_pre: Option<&mut MoePrealloc>,
    b: usize,
    li: usize,
    device: &Device,
) -> Result<Tensor> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let stacked = moe
        .stacked()
        .ok_or_else(|| anyhow::anyhow!("moe_decode_ffn_batch: experts not stacked"))?;
    let k = moe.top_k();
    let e = moe.num_experts();
    let hidden = scratch.hidden;
    let inter = scratch.inter;
    anyhow::ensure!(
        k == scratch.k && e == scratch.e,
        "moe_decode_ffn_batch: scratch shape mismatch"
    );
    anyhow::ensure!(
        b > 0 && b <= scratch.max_b,
        "moe_decode_ffn_batch: bad b={b}"
    );
    anyhow::ensure!(
        x_normed.elem_count() == b * hidden && resid.elem_count() == b * hidden,
        "moe_decode_ffn_batch: expected [{b}, {hidden}] input"
    );

    match moe_pre.as_deref_mut() {
        Some(mp) => mp.gate_scores_cast(li, x_normed, b, &mut scratch.scores_f32, &stream)?,
        None => {
            let scores = moe.gate_bf16().forward(x_normed)?.contiguous()?;
            let sp = tensor_ptr_bf16(&scores, &stream, b * e)?;
            let (dp, _g) = scratch.scores_f32.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    stream.cu_stream() as *mut c_void,
                    sp as *const u16,
                    dp as *mut f32,
                    (b * e) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe batch gate cast rc={rc}");
        }
    }
    {
        let (lp, _g1) = scratch.scores_f32.device_ptr(&stream);
        let (sp, _g2) = scratch.seeds.device_ptr(&stream);
        let (pp, _g3) = scratch.probs.device_ptr_mut(&stream);
        let (tp, _g4) = scratch.trash_token.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::sampler_topk_topp(
                stream.cu_stream() as *mut c_void,
                lp as *const f32,
                sp as *const u64,
                pp as *mut f32,
                tp as *mut u32,
                b,
                e,
                1.0,
                0,
                1.0,
            )
        };
        anyhow::ensure!(rc == 0, "moe batch softmax rc={rc}");
    }
    {
        let (pp, _g1) = scratch.probs.device_ptr(&stream);
        let (ip, _g2) = scratch.ids.device_ptr_mut(&stream);
        let (wp, _g3) = scratch.weights.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_route_topk(
                stream.cu_stream() as *mut c_void,
                pp as *const f32,
                std::ptr::null(),
                ip as *mut i32,
                wp as *mut f32,
                b as i32,
                e as i32,
                k as i32,
                0,
                0.0,
                0,
                1.0,
            )
        };
        anyhow::ensure!(rc == 0, "moe batch route rc={rc}");
    }
    if scratch.ids_log.is_some() {
        let base = li * scratch.max_b * k;
        let src = scratch.ids.slice(0..b * k);
        let log = scratch.ids_log.as_mut().unwrap();
        let mut dst = log.slice_mut(base..base + b * k);
        stream
            .memcpy_dtod(&src, &mut dst)
            .map_err(|e| anyhow::anyhow!("moe idlog dtod: {e:?}"))?;
    }
    {
        let (pp, _g1) = scratch.probs.device_ptr(&stream);
        let (ip, _g2) = scratch.ids.device_ptr(&stream);
        let (wp, _g3) = scratch.weights.device_ptr_mut(&stream);
        for j in 0..b {
            let rc = unsafe {
                nv_kernels::cuda::gather_f32_by_ids(
                    stream.cu_stream() as *mut c_void,
                    off(pp, j * e, 4) as *const f32,
                    off(ip, j * k, 4) as *const i32,
                    off(wp, j * k, 4) as *mut f32,
                    k as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe batch weight gather rc={rc}");
        }
    }
    match moe_pre.as_deref_mut() {
        Some(mp) => mp.shared_cast(li, x_normed, b, &mut scratch.shared_f32, &stream)?,
        None => {
            let shared_t = moe.shared_expert().forward_fused_cuda(x_normed)?;
            let sp = tensor_ptr_bf16(&shared_t, &stream, b * hidden)?;
            let (dp, _g) = scratch.shared_f32.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    stream.cu_stream() as *mut c_void,
                    sp as *const u16,
                    dp as *mut f32,
                    (b * hidden) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe batch shared cast rc={rc}");
        }
    }

    let gate_src = tensor_ptr_bf16(&stacked.gate, &stream, e * inter * hidden)?;
    let up_src = tensor_ptr_bf16(&stacked.up, &stream, e * inter * hidden)?;
    let down_src = tensor_ptr_bf16(&stacked.down, &stream, e * hidden * inter)?;
    let x_ptr = tensor_ptr_bf16(x_normed, &stream, b * hidden)?;
    let resid_ptr = tensor_ptr_bf16(resid, &stream, b * hidden)?;
    let out_ptr = tensor_ptr_bf16(&scratch.out, &stream, scratch.max_b * hidden)?;
    let slot_batched = moe_slot_batched();
    {
        let (ids_ptr, _g1) = scratch.ids.device_ptr(&stream);
        let (hp, _g2) = scratch.h_rows.device_ptr_mut(&stream);
        if slot_batched {
            let rc = unsafe {
                nv_kernels::cuda::moe_gemv_swiglu_bf16_mb(
                    stream.cu_stream() as *mut c_void,
                    gate_src as *const u16,
                    up_src as *const u16,
                    ids_ptr as *const i32,
                    x_ptr as *const u16,
                    hp as *mut u16,
                    b as i32,
                    k as i32,
                    e as i32,
                    inter as i32,
                    hidden as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe batch swiglu gemv mb rc={rc}");
        } else {
            for j in 0..b {
                let rc = unsafe {
                    nv_kernels::cuda::moe_gemv_swiglu_bf16_m1(
                        stream.cu_stream() as *mut c_void,
                        gate_src as *const u16,
                        up_src as *const u16,
                        off(ids_ptr, j * k, 4) as *const i32,
                        off(x_ptr, j * hidden, 2) as *const u16,
                        off(hp, j * k * inter, 2) as *mut u16,
                        k as i32,
                        e as i32,
                        inter as i32,
                        hidden as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "moe batch swiglu gemv rc={rc}");
            }
        }
    }
    {
        let (ids_ptr, _g1) = scratch.ids.device_ptr(&stream);
        let (wp, _g2) = scratch.weights.device_ptr(&stream);
        let (hp, _g3) = scratch.h_rows.device_ptr(&stream);
        let (sp, _g4) = scratch.shared_f32.device_ptr(&stream);
        if slot_batched {
            let rc = unsafe {
                nv_kernels::cuda::moe_gemv_down_tail_bf16_mb(
                    stream.cu_stream() as *mut c_void,
                    down_src as *const u16,
                    ids_ptr as *const i32,
                    wp as *const f32,
                    hp as *const u16,
                    sp as *const f32,
                    resid_ptr as *const u16,
                    out_ptr as *mut u16,
                    b as i32,
                    k as i32,
                    e as i32,
                    hidden as i32,
                    inter as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe batch down gemv tail mb rc={rc}");
        } else {
            for j in 0..b {
                let rc = unsafe {
                    nv_kernels::cuda::moe_gemv_down_tail_bf16_m1(
                        stream.cu_stream() as *mut c_void,
                        down_src as *const u16,
                        off(ids_ptr, j * k, 4) as *const i32,
                        off(wp, j * k, 4) as *const f32,
                        off(hp, j * k * inter, 2) as *const u16,
                        off(sp, j * hidden, 4) as *const f32,
                        off(resid_ptr, j * hidden, 2) as *const u16,
                        off(out_ptr, j * hidden, 2) as *mut u16,
                        k as i32,
                        e as i32,
                        hidden as i32,
                        inter as i32,
                    )
                };
                anyhow::ensure!(rc == 0, "moe batch down gemv tail rc={rc}");
            }
        }
    }
    Ok(scratch.out.narrow(0, 0, b)?)
}

struct CtxErrDrain(Arc<CudaContext>);

impl Drop for CtxErrDrain {
    fn drop(&mut self) {
        if let Err(e) = self.0.check_err() {
            if graph_debug() {
                eprintln!("[dsocr-bsn-drop] drained deferred ctx error from teardown: {e:?}");
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    Free,
    Active,
}

struct BatchPrealloc {
    tok_t: Tensor,
    q_f32: Tensor,
    k_f32: Tensor,
    q_bf: Tensor,
    k_bf: Tensor,
    resid_a: Tensor,
    resid_b: Tensor,
    cos_c: Tensor,
    sin_c: Tensor,
    moe: Option<MoePrealloc>,
}

impl BatchPrealloc {
    fn new(
        model: &DeepseekOcrDecoder,
        forked: &Arc<CudaStream>,
        dev: &CudaDevice,
        max_b: usize,
    ) -> Result<Self> {
        let cfg = model.config();
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        let hidden = cfg.hidden_size;
        anyhow::ensure!(
            model.rope().config().head_dim == hd,
            "dsocr batch prealloc: rope head_dim {} != cfg {}",
            model.rope().config().head_dim,
            hd
        );
        let tok_t = {
            let slice = forked
                .alloc_zeros::<u32>(max_b)
                .map_err(|e| anyhow::anyhow!(e))?;
            let st = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(st),
                (max_b,),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let alloc_f32 = |n: usize, dims: &[usize]| -> Result<Tensor> {
            wrap_f32(
                forked
                    .alloc_zeros::<f32>(n)
                    .map_err(|e| anyhow::anyhow!(e))?,
                dims,
                dev,
            )
        };
        let alloc_bf = |n: usize, dims: &[usize]| -> Result<Tensor> {
            wrap_bf16(
                forked
                    .alloc_zeros::<bf16>(n)
                    .map_err(|e| anyhow::anyhow!(e))?,
                dims,
                dev,
            )
        };
        let q_f32 = alloc_f32(max_b * n_heads * hd, &[max_b, n_heads, hd])?;
        let k_f32 = alloc_f32(max_b * n_kv * hd, &[max_b, n_kv, hd])?;
        let q_bf = alloc_bf(max_b * n_heads * hd, &[max_b, n_heads, hd])?;
        let k_bf = alloc_bf(max_b * n_kv * hd, &[max_b, n_kv, hd])?;
        let resid_a = alloc_bf(max_b * hidden, &[max_b, hidden])?;
        let resid_b = alloc_bf(max_b * hidden, &[max_b, hidden])?;
        let cos_c = model.rope().cos().contiguous()?;
        let sin_c = model.rope().sin().contiguous()?;
        anyhow::ensure!(
            cos_c.dtype() == candle_core::DType::F32 && sin_c.dtype() == candle_core::DType::F32,
            "dsocr batch prealloc: rope tables must be f32"
        );
        let moe = MoePrealloc::new(model, forked, max_b)?;
        if moe.is_none() {
            eprintln!(
                "[dsocr] bs=N graph prealloc: MoE gate/shared linears not plain bf16; \
                 MoE internals keep in-capture allocations"
            );
        }
        Ok(Self {
            tok_t,
            q_f32,
            k_f32,
            q_bf,
            k_bf,
            resid_a,
            resid_b,
            cos_c,
            sin_c,
            moe,
        })
    }
}

pub struct DsocrBatchDecodeGraph {
    model: Arc<DeepseekOcrDecoder>,
    dev: CudaDevice,
    device: Device,
    forked: Arc<CudaStream>,
    runner: CudaGraphRunner,
    buckets: Vec<usize>,
    max_b: usize,
    cap: usize,
    tok_buf: CudaSlice<u32>,
    host_tok: Box<[u32]>,
    pos_buf: CudaSlice<i32>,
    host_pos: Box<[i32]>,
    cu_q: CudaSlice<i32>,
    cu_k: CudaSlice<i32>,
    host_cu_k: Box<[i32]>,
    seq_k: CudaSlice<i32>,
    host_seq_k: Box<[i32]>,
    kv: Vec<(CudaSlice<bf16>, CudaSlice<bf16>)>,
    logits_buf: CudaSlice<f32>,
    scratch: BatchScratch,
    prealloc: Option<BatchPrealloc>,
    lens: Vec<usize>,
    state: Vec<SlotState>,
    captured: Vec<usize>,
    pending: bool,
    idlog: Option<IdLogSink>,
    err_drain: CtxErrDrain,
}

struct IdLogSink {
    out: std::io::BufWriter<std::fs::File>,
    step: usize,
    host: Vec<i32>,
}

unsafe impl Send for DsocrBatchDecodeGraph {}

impl DsocrBatchDecodeGraph {
    pub fn new(model: Arc<DeepseekOcrDecoder>, cap: usize, buckets: Vec<usize>) -> Result<Self> {
        graph_supported(&model)
            .map_err(|e| anyhow::anyhow!("dsocr bs=N graph unsupported: {e}"))?;
        anyhow::ensure!(cap > 0, "dsocr bs=N graph: cap must be > 0");
        let buckets = parse_buckets(Some(
            &buckets
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join(","),
        ));
        let max_b = *buckets.last().unwrap();
        let device = model.device().clone();
        let dev = cuda_dev(&device)?;
        let cfg = model.config();
        let n_kv = cfg.num_key_value_heads;
        let n_heads = cfg.num_attention_heads;
        let hd = cfg.head_dim();

        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        let _init = lock_init();
        if raw_ctx.is_event_tracking() {
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-untrack legacy sync: {e:?}"))?;
            unsafe { raw_ctx.disable_event_tracking() };
        }
        let forked = raw_ctx
            .new_stream()
            .map_err(|e| anyhow::anyhow!("dsocr bs=N graph stream: {e:?}"))?;
        nv_quant::nvfp4::ensure_workspace_for_stream(&forked)?;
        let _ = nv_quant::matmul::TensorCoreGemm::new(forked.clone())?;

        let tok_buf = forked
            .alloc_zeros::<u32>(max_b)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pos_buf = forked
            .alloc_zeros::<i32>(max_b)
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let cu_q = forked
            .memcpy_stod(&[0i32, 1i32])
            .map_err(|e| anyhow::anyhow!(e))?;
        let cu_k = forked
            .alloc_zeros::<i32>(2 * max_b)
            .map_err(|e| anyhow::anyhow!(e))?;
        let seq_k = forked
            .alloc_zeros::<i32>(max_b)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut kv = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let k = forked
                .alloc_zeros::<bf16>(max_b * cap * n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?;
            let v = forked
                .alloc_zeros::<bf16>(max_b * cap * n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?;
            kv.push((k, v));
        }
        let logits_buf = forked
            .alloc_zeros::<f32>(max_b * cfg.vocab_size)
            .map_err(|e| anyhow::anyhow!(e))?;
        let scratch = nv_layers::cuda_stream::with_stream(forked.clone(), || {
            BatchScratch::new(
                cfg.num_experts_per_tok,
                cfg.n_routed_experts,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                n_heads,
                hd,
                max_b,
                cfg.num_hidden_layers,
                &device,
            )
        })?;
        forked.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        let runner = CudaGraphRunner::new(forked.clone());
        let idlog_host = cfg.num_hidden_layers * max_b * cfg.num_experts_per_tok;
        let idlog = match idlog_path() {
            Some(p) => {
                let f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&p)
                    .with_context(|| format!("open NV_DSOCR_MOE_IDLOG {p}"))?;
                eprintln!("[dsocr-idlog] logging routed expert ids to {p}");
                Some(IdLogSink {
                    out: std::io::BufWriter::new(f),
                    step: 0,
                    host: vec![0i32; idlog_host],
                })
            }
            None => None,
        };
        let prealloc = if graph_prealloc_enabled() {
            Some(BatchPrealloc::new(&model, &forked, &dev, max_b)?)
        } else {
            None
        };

        Ok(Self {
            model,
            dev,
            device,
            forked,
            runner,
            buckets,
            max_b,
            cap,
            tok_buf,
            host_tok: vec![0u32; max_b].into_boxed_slice(),
            pos_buf,
            host_pos: vec![0i32; max_b].into_boxed_slice(),
            cu_q,
            cu_k,
            host_cu_k: vec![0i32; 2 * max_b].into_boxed_slice(),
            seq_k,
            host_seq_k: vec![0i32; max_b].into_boxed_slice(),
            kv,
            logits_buf,
            scratch,
            prealloc,
            lens: vec![0; max_b],
            state: vec![SlotState::Free; max_b],
            captured: Vec::new(),
            pending: false,
            idlog,
            err_drain: CtxErrDrain(raw_ctx),
        })
    }

    pub fn from_env(model: Arc<DeepseekOcrDecoder>, cap: usize) -> Result<Self> {
        Self::new(model, cap, buckets_from_env())
    }

    pub fn max_batch(&self) -> usize {
        self.max_b
    }

    pub fn node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn buckets(&self) -> &[usize] {
        &self.buckets
    }

    pub fn bucket_for(&self, b: usize) -> Option<usize> {
        bucket_for(&self.buckets, b)
    }

    pub fn slot_len(&self, slot: usize) -> usize {
        self.lens[slot]
    }

    pub fn is_active(&self, slot: usize) -> bool {
        self.state[slot] == SlotState::Active
    }

    pub fn free_slot(&self) -> Option<usize> {
        self.state.iter().position(|s| *s == SlotState::Free)
    }

    pub fn release_slot(&mut self, slot: usize) {
        self.state[slot] = SlotState::Free;
        self.lens[slot] = 0;
    }

    pub fn active_extent(&self) -> usize {
        self.state
            .iter()
            .rposition(|s| *s == SlotState::Active)
            .map(|i| i + 1)
            .unwrap_or(0)
    }

    fn sync_pending(&mut self) -> Result<()> {
        if self.pending {
            self.forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pending replay sync: {e:?}"))?;
            self.pending = false;
        }
        Ok(())
    }

    pub fn prefill_detached(
        model: &DeepseekOcrDecoder,
        prompt_tokens: &[u32],
        vision_features: Option<&Tensor>,
        max_len: usize,
    ) -> Result<(DeepseekOcrKvCache, Vec<f32>)> {
        anyhow::ensure!(
            prompt_tokens.len() < max_len,
            "prefill_detached: prompt {} leaves no room (max {max_len})",
            prompt_tokens.len()
        );
        if prefill_per_thread_stream() && !model_uses_nvfp4(model) {
            return Self::prefill_detached_per_thread(
                model,
                prompt_tokens,
                vision_features,
                max_len,
            );
        }
        let mut cache = model.new_kv_cache(max_len)?;
        let x = model.embed_tokens_with_vision(prompt_tokens, vision_features)?;
        let hidden = model.forward_embeds_hidden(&x, &mut cache)?;
        let logits = model.last_logits(&hidden)?;
        Ok((cache, logits))
    }

    fn prefill_detached_per_thread(
        model: &DeepseekOcrDecoder,
        prompt_tokens: &[u32],
        vision_features: Option<&Tensor>,
        max_len: usize,
    ) -> Result<(DeepseekOcrKvCache, Vec<f32>)> {
        let dev = cuda_dev(model.device())?;
        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        if raw_ctx.is_event_tracking() {
            let _init = lock_init();
            if raw_ctx.is_event_tracking() {
                dev.cuda_stream()
                    .synchronize()
                    .map_err(|e| anyhow::anyhow!("pre-untrack legacy sync: {e:?}"))?;
                unsafe { raw_ctx.disable_event_tracking() };
            }
        }
        let pts = raw_ctx.per_thread_stream();
        let out = nv_layers::cuda_stream::with_stream(pts.clone(), || -> Result<_> {
            let mut cache = model.new_kv_cache(max_len)?;
            let x = model.embed_tokens_with_vision(prompt_tokens, vision_features)?;
            let hidden = model.forward_embeds_hidden(&x, &mut cache)?;
            let logits = model.last_logits(&hidden)?;
            Ok((cache, logits))
        })?;
        pts.synchronize()
            .map_err(|e| anyhow::anyhow!("per-thread prefill sync: {e:?}"))?;
        Ok(out)
    }

    pub fn install_prefilled(&mut self, slot: usize, cache: &DeepseekOcrKvCache) -> Result<()> {
        anyhow::ensure!(slot < self.max_b, "install_prefilled: slot out of range");
        self.load_kv_into_slot_inner(slot, cache, false)?;
        self.state[slot] = SlotState::Active;
        Ok(())
    }

    pub fn prefill_slot(
        &mut self,
        slot: usize,
        prompt_tokens: &[u32],
        vision_features: Option<&Tensor>,
        max_len: usize,
    ) -> Result<Vec<f32>> {
        anyhow::ensure!(slot < self.max_b, "prefill_slot: slot {slot} out of range");
        anyhow::ensure!(
            max_len <= self.cap,
            "prefill_slot: max_len {max_len} exceeds cap {}",
            self.cap
        );
        anyhow::ensure!(
            prompt_tokens.len() < max_len,
            "prefill_slot: prompt {} leaves no room (max {max_len})",
            prompt_tokens.len()
        );
        self.sync_pending()?;
        let model = self.model.clone();
        let mut cache = model.new_kv_cache(max_len)?;
        let x = model.embed_tokens_with_vision(prompt_tokens, vision_features)?;
        let hidden = model.forward_embeds_hidden(&x, &mut cache)?;
        let logits = model.last_logits(&hidden)?;
        self.load_kv_into_slot_inner(slot, &cache, true)?;
        drop(cache);
        self.state[slot] = SlotState::Active;
        Ok(logits)
    }

    pub fn load_kv_into_slot(&mut self, slot: usize, cache: &DeepseekOcrKvCache) -> Result<()> {
        self.load_kv_into_slot_inner(slot, cache, true)
    }

    fn load_kv_into_slot_inner(
        &mut self,
        slot: usize,
        cache: &DeepseekOcrKvCache,
        sync_legacy: bool,
    ) -> Result<()> {
        anyhow::ensure!(slot < self.max_b, "load_kv_into_slot: slot out of range");
        let (n_layers, n_kv, hd) = {
            let cfg = self.model.config();
            (
                cfg.num_hidden_layers,
                cfg.num_key_value_heads,
                cfg.head_dim(),
            )
        };
        let len = cache.current_len();
        anyhow::ensure!(
            len <= self.cap,
            "prefill length {len} exceeds graph capacity {}",
            self.cap
        );
        let per_slot = self.cap * n_kv * hd;
        let n = len * n_kv * hd;
        self.sync_pending()?;
        if sync_legacy {
            self.dev
                .cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-copy legacy sync: {e:?}"))?;
        }
        for li in 0..n_layers {
            let (kt, vt) = cache.layer_bufs(li);
            let base = slot * per_slot;
            let (kd, vd) = &mut self.kv[li];
            for (src_t, dst) in [(kt, kd), (vt, vd)] {
                if n == 0 {
                    continue;
                }
                let (st, l) = src_t.storage_and_layout();
                anyhow::ensure!(l.is_contiguous(), "kv cache tensor not contiguous");
                let cuda = match &*st {
                    candle_core::Storage::Cuda(s) => s,
                    _ => anyhow::bail!("kv cache tensor not CUDA"),
                };
                let slice = cuda.as_cuda_slice::<bf16>()?;
                let view = slice.slice(l.start_offset()..l.start_offset() + n);
                let mut dst_view = dst.slice_mut(base..base + n);
                self.forked
                    .memcpy_dtod(&view, &mut dst_view)
                    .map_err(|e| anyhow::anyhow!("kv load dtod: {e:?}"))?;
            }
        }
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("kv load sync: {e:?}"))?;
        self.lens[slot] = len;
        Ok(())
    }

    pub fn step_batch(&mut self, tokens: &[Option<u32>]) -> Result<usize> {
        let want = tokens.len();
        anyhow::ensure!(want > 0, "step_batch: empty token slice");
        let b = bucket_for(&self.buckets, want)
            .ok_or_else(|| anyhow::anyhow!("step_batch: no bucket covers {want} slots"))?;
        anyhow::ensure!(
            b <= self.max_b,
            "step_batch: bucket {b} > max_b {}",
            self.max_b
        );
        for (j, t) in tokens.iter().enumerate() {
            if t.is_some() {
                anyhow::ensure!(
                    self.lens[j] + 1 <= self.cap,
                    "dsocr bs=N step overflows capacity {} on slot {j}",
                    self.cap
                );
            }
        }
        self.sync_pending()?;

        for j in 0..b {
            match tokens.get(j).copied().flatten() {
                Some(t) => {
                    self.host_tok[j] = t;
                    self.host_pos[j] = self.lens[j] as i32;
                    self.host_cu_k[2 * j] = 0;
                    self.host_cu_k[2 * j + 1] = (self.lens[j] + 1) as i32;
                    self.host_seq_k[j] = (self.lens[j] + 1) as i32;
                }
                None => {
                    self.host_tok[j] = 0;
                    self.host_pos[j] = 0;
                    self.host_cu_k[2 * j] = 0;
                    self.host_cu_k[2 * j + 1] = 1;
                    self.host_seq_k[j] = 1;
                }
            }
        }

        let legacy = self.dev.cuda_stream();
        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured.contains(&b);
        let forked = self.forked.clone();
        let DsocrBatchDecodeGraph {
            model,
            dev,
            device,
            runner,
            tok_buf,
            host_tok,
            pos_buf,
            host_pos,
            cu_q,
            cu_k,
            host_cu_k,
            seq_k,
            host_seq_k,
            kv,
            logits_buf,
            scratch,
            prealloc,
            cap,
            max_b,
            ..
        } = self;
        let cap = *cap;
        let max_b = *max_b;
        let cfg = model.config();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        let vocab = cfg.vocab_size;
        let scale = 1.0 / (hd as f32).sqrt();
        let per_slot_kv = cap * n_kv * hd;

        let mut body = |s: &Arc<CudaStream>, scratch: &mut BatchScratch| -> Result<()> {
            s.memcpy_htod(&host_tok[..b], tok_buf)
                .map_err(|e| anyhow::anyhow!("htod tok: {e:?}"))?;
            s.memcpy_htod(&host_pos[..b], pos_buf)
                .map_err(|e| anyhow::anyhow!("htod pos: {e:?}"))?;
            s.memcpy_htod(&host_cu_k[..2 * b], cu_k)
                .map_err(|e| anyhow::anyhow!("htod cu_k: {e:?}"))?;
            s.memcpy_htod(&host_seq_k[..b], seq_k)
                .map_err(|e| anyhow::anyhow!("htod seq_k: {e:?}"))?;
            let (tokens_t, pos_t) = if let Some(p) = prealloc.as_mut() {
                let (tok_src, _gt) = tok_buf.device_ptr(s);
                let tok_dst = tensor_ptr_u32(&p.tok_t, s, max_b)?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        tok_dst,
                        tok_src,
                        b * std::mem::size_of::<u32>(),
                        s.cu_stream(),
                    )
                }
                .map_err(|e| anyhow::anyhow!("tok dtod: {e:?}"))?;
                (p.tok_t.narrow(0, 0, b)?, None)
            } else {
                let tok_clone = tok_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let pos_clone = pos_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let tokens_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (max_b,),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                    .narrow(0, 0, b)?
                };
                let pos_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(pos_clone, dev.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (max_b,),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                    .narrow(0, 0, b)?
                };
                (tokens_t, Some(pos_t))
            };
            let (cu_q_ptr, _gq) = cu_q.device_ptr(s);
            let (cu_k_ptr, _gk) = cu_k.device_ptr(s);
            let (pos_ptr, _gp) = pos_buf.device_ptr(s);

            let mut x =
                crate::gemma4::embed_lookup_bf16_op(model.embed_weight_t(), &tokens_t, device)?
                    .reshape((b, hidden))?;
            for (li, layer) in model.layers().iter().enumerate() {
                let normed = layer.input_layernorm.forward(&x)?;
                let q = layer.q_proj.forward(&normed)?.reshape((b, n_heads, hd))?;
                let k = layer.k_proj.forward(&normed)?.reshape((b, n_kv, hd))?;
                let v = layer.v_proj.forward(&normed)?.reshape((b, n_kv, hd))?;
                let (q_bf, k_bf) = if let Some(p) = prealloc.as_mut() {
                    let qv = p.q_f32.narrow(0, 0, b)?;
                    let kv_ = p.k_f32.narrow(0, 0, b)?;
                    cast_bf16_to_f32_into(&q, &qv, device)?;
                    cast_bf16_to_f32_into(&k, &kv_, device)?;
                    rope_apply_inplace_f32(
                        &qv, &kv_, &p.cos_c, &p.sin_c, pos_ptr, b, n_heads, n_kv, hd, device,
                    )?;
                    let qb = p.q_bf.narrow(0, 0, b)?;
                    let kb = p.k_bf.narrow(0, 0, b)?;
                    cast_f32_to_bf16_into(&qv, &qb, device)?;
                    cast_f32_to_bf16_into(&kv_, &kb, device)?;
                    (qb, kb)
                } else {
                    let q_f32 = cast_tensor_bf16_to_f32(&q, device)?;
                    let k_f32 = cast_tensor_bf16_to_f32(&k, device)?;
                    let pos_t = pos_t.as_ref().expect("pos_t present without prealloc");
                    let (q_rot, k_rot) = model.rope().apply(&q_f32, &k_f32, pos_t)?;
                    (
                        cast_tensor_f32_to_bf16(&q_rot, device)?,
                        cast_tensor_f32_to_bf16(&k_rot, device)?,
                    )
                };
                let k_src = tensor_ptr_bf16(&k_bf, s, b * n_kv * hd)?;
                let v_src = tensor_ptr_bf16(&v, s, b * n_kv * hd)?;
                let q_src = tensor_ptr_bf16(&q_bf, s, b * n_heads * hd)?;
                let (k_base, _g1) = kv[li].0.device_ptr(s);
                let (v_base, _g2) = kv[li].1.device_ptr(s);
                for j in 0..b {
                    for (src, base) in [(k_src, k_base), (v_src, v_base)] {
                        let rc = unsafe {
                            nv_kernels::cuda::kv_ring_append_bf16(
                                s.cu_stream() as *mut c_void,
                                off(src, j * n_kv * hd, 2) as *const u16,
                                off(base, j * per_slot_kv, 2) as *mut u16,
                                off(pos_ptr, j, 4) as *const i32,
                                1,
                                cap as i32,
                                n_kv as i32,
                                hd as i32,
                            )
                        };
                        anyhow::ensure!(rc == 0, "kv append rc={rc} layer {li} slot {j}");
                    }
                }
                {
                    let o_base = tensor_ptr_bf16(&scratch.attn_o, s, max_b * n_heads * hd)?;
                    let (lse_base, _g3) = scratch.attn_lse.device_ptr(s);
                    if attn_batched() {
                        let (seq_ptr, _g4) = seq_k.device_ptr(s);
                        unsafe {
                            batch_decode_fwd_bf16(
                                s.cu_stream() as *mut c_void,
                                &BatchDecodeArgs {
                                    q_ptr: q_src,
                                    k_ptr: k_base,
                                    v_ptr: v_base,
                                    o_ptr: o_base,
                                    lse_ptr: lse_base,
                                    seqused_k: seq_ptr,
                                    b,
                                    q_batch_stride: n_heads * hd,
                                    kv_batch_stride: per_slot_kv,
                                    max_seqlen_k: cap,
                                    h: n_heads,
                                    h_k: n_kv,
                                    d: hd,
                                    softmax_scale: scale,
                                },
                            )?;
                        }
                    } else {
                        for j in 0..b {
                            unsafe {
                                varlen_fwd_bf16(
                                    s.cu_stream() as *mut c_void,
                                    &VarlenArgs {
                                        q_ptr: off(q_src, j * n_heads * hd, 2),
                                        k_ptr: off(k_base, j * per_slot_kv, 2),
                                        v_ptr: off(v_base, j * per_slot_kv, 2),
                                        o_ptr: off(o_base, j * n_heads * hd, 2),
                                        lse_ptr: off(lse_base, j * n_heads, 4),
                                        cu_seqlens_q: cu_q_ptr,
                                        cu_seqlens_k: off(cu_k_ptr, 2 * j, 4),
                                        max_seqlen_q: 1,
                                        max_seqlen_k: cap,
                                        h: n_heads,
                                        h_k: n_kv,
                                        d: hd,
                                        softmax_scale: scale,
                                        window_size_left: None,
                                        window_size_right: Some(0),
                                    },
                                )?;
                            }
                        }
                    }
                }
                let o2 = scratch.attn_o.narrow(0, 0, b)?.reshape((b, n_heads * hd))?;
                let attn_out = layer.o_proj.forward(&o2)?;
                let x_after = if let Some(p) = prealloc.as_mut() {
                    let ra = p.resid_a.narrow(0, 0, b)?;
                    residual_add_scale_bf16_into(&x, &attn_out, 1.0, &ra, device)?;
                    ra
                } else {
                    crate::gemma4::residual_add_scale_bf16_op(&x, &attn_out, 1.0, device)?
                };
                let normed2 = layer.post_attention_layernorm.forward(&x_after)?;
                x = match &layer.ff {
                    FeedForward::Dense(mlp) => {
                        if let Some(p) = prealloc.as_mut() {
                            let y = mlp.forward_fused_cuda(&normed2)?;
                            let rb = p.resid_b.narrow(0, 0, b)?;
                            residual_add_scale_bf16_into(&x_after, &y, 1.0, &rb, device)?;
                            rb
                        } else {
                            dense_decode_ffn(mlp, &normed2, &x_after, device)?
                        }
                    }
                    FeedForward::Moe(m) => moe_decode_ffn_batch(
                        m,
                        &normed2,
                        &x_after,
                        scratch,
                        prealloc.as_mut().and_then(|p| p.moe.as_mut()),
                        b,
                        li,
                        device,
                    )?,
                };
            }
            let h = model.final_norm().forward(&x)?;
            let logits_bf = model.lm_head().forward(&h)?;
            let lp_src = tensor_ptr_bf16(&logits_bf, s, b * vocab)?;
            let (lp_dst, _g) = logits_buf.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    s.cu_stream() as *mut c_void,
                    lp_src as *const u16,
                    lp_dst as *mut f32,
                    (b * vocab) as i32,
                )
            };
            anyhow::ensure!(rc == 0, "logits cast rc={rc}");
            Ok(())
        };

        let _init = if was_captured {
            None
        } else {
            Some(lock_init())
        };
        if !was_captured {
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-warm legacy sync: {e:?}"))?;
            nv_layers::cuda_stream::with_stream(forked.clone(), || body(&forked, scratch))
                .context("dsocr bs=N graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }
        runner
            .run_on(b as u64, None, |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || body(s, scratch))
            })
            .context("dsocr bs=N graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
            self.captured.push(b);
        }
        self.pending = true;
        if self.idlog.is_some() {
            let active: Vec<usize> = (0..b)
                .filter(|&j| tokens.get(j).copied().flatten().is_some())
                .collect();
            self.flush_idlog(b, &active)?;
        }
        for (j, t) in tokens.iter().enumerate() {
            if t.is_some() {
                self.lens[j] += 1;
            }
        }
        Ok(b)
    }

    fn flush_idlog(&mut self, b: usize, active: &[usize]) -> Result<()> {
        use std::io::Write;
        let k = self.scratch.k;
        let max_b = self.scratch.max_b;
        let layers = self.model.config().num_hidden_layers;
        let n = layers * max_b * k;
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("idlog sync: {e:?}"))?;
        {
            let src = match self.scratch.ids_log.as_ref() {
                Some(s) => s,
                None => return Ok(()),
            };
            let sink = self.idlog.as_mut().unwrap();
            let view = src.slice(0..n);
            self.forked
                .memcpy_dtoh(&view, &mut sink.host[..n])
                .map_err(|e| anyhow::anyhow!("idlog dtoh: {e:?}"))?;
        }
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("idlog dtoh sync: {e:?}"))?;
        let lens: Vec<usize> = active.iter().map(|&j| self.lens[j]).collect();
        let sink = self.idlog.as_mut().unwrap();
        let mut line = String::with_capacity(1024);
        line.push_str(&format!(
            "{{\"step\":{},\"b\":{},\"k\":{},\"active\":{:?},\"lens\":{:?},\"layers\":[",
            sink.step, b, k, active, lens
        ));
        for li in 0..layers {
            if li > 0 {
                line.push(',');
            }
            line.push('[');
            for (si, &j) in active.iter().enumerate() {
                if si > 0 {
                    line.push(',');
                }
                let base = li * max_b * k + j * k;
                line.push('[');
                for t in 0..k {
                    if t > 0 {
                        line.push(',');
                    }
                    line.push_str(&sink.host[base + t].to_string());
                }
                line.push(']');
            }
            line.push(']');
        }
        line.push_str("]}\n");
        sink.out
            .write_all(line.as_bytes())
            .map_err(|e| anyhow::anyhow!("idlog write: {e:?}"))?;
        sink.step += 1;
        Ok(())
    }

    pub fn logits_batch(&mut self, b: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(b > 0 && b <= self.max_b, "logits_batch: bad b={b}");
        let vocab = self.model.config().vocab_size;
        let mut out = vec![0f32; b * vocab];
        let view = self.logits_buf.slice(0..b * vocab);
        self.forked
            .memcpy_dtoh(&view, &mut out)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))?;
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("logits forked sync: {e:?}"))?;
        self.pending = false;
        Ok(out)
    }
}

impl Drop for DsocrBatchDecodeGraph {
    fn drop(&mut self) {
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .run(|| self.runner.invalidate());
        let _ = &self.err_drain;
    }
}
