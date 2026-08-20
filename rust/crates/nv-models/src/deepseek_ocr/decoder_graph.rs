#![cfg(feature = "cuda")]

use anyhow::{Context, Result};
use candle_core::{CudaDevice, DType, Device, Tensor};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use std::ffi::c_void;
use std::sync::Arc;

use super::decoder::{
    detect_loop, select_next_token, DeepseekMoe, DeepseekOcrDecoder, DeepseekOcrDecoderConfig,
    DeepseekOcrKvCache, FeedForward, GenerateOptions, GenerateOutcome, SplitMix64,
    LOOP_CHECK_STRIDE,
};
use crate::laguna_fa2::{varlen_fwd_bf16, VarlenArgs};
use nv_kernels::graph::CudaGraphRunner;
use nv_layers::attn::AttnConfig;
use nv_layers::mlp::Mlp;

pub(crate) fn kernel_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_DECODE")
            .map(|v| v == "kernel")
            .unwrap_or(false)
    })
}

pub fn graph_enabled() -> bool {
    std::env::var("NV_DSOCR_GRAPH")
        .map(|v| v != "0")
        .unwrap_or(true)
}

fn init_lock() -> &'static std::sync::Mutex<()> {
    static L: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    L.get_or_init(|| std::sync::Mutex::new(()))
}

pub(crate) fn lock_init() -> std::sync::MutexGuard<'static, ()> {
    init_lock()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn nullstream_arm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_GRAPH_NULLSTREAM")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

pub(crate) fn graph_debug() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_GRAPH_DEBUG")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

pub fn graph_prealloc_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_GRAPH_PREALLOC")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

fn graph_mem_report(tag: &str, ordinal: usize) {
    use cudarc::driver::sys as drv;
    let Ok(devh) = cudarc::driver::result::device::get(ordinal as i32) else {
        return;
    };
    let mut vals = [0u64; 4];
    let attrs = [
        drv::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT,
        drv::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_USED_MEM_HIGH,
        drv::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT,
        drv::CUgraphMem_attribute::CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH,
    ];
    for (i, a) in attrs.into_iter().enumerate() {
        unsafe {
            let _ =
                drv::cuDeviceGetGraphMemAttribute(devh, a, &mut vals[i] as *mut u64 as *mut c_void);
        }
    }
    eprintln!(
        "[dsocr-graph-mem] {tag}: used_cur={:.1}MB used_high={:.1}MB reserved_cur={:.1}MB reserved_high={:.1}MB",
        vals[0] as f64 / 1048576.0,
        vals[1] as f64 / 1048576.0,
        vals[2] as f64 / 1048576.0,
        vals[3] as f64 / 1048576.0
    );
}

pub(crate) fn cuda_dev(device: &Device) -> Result<CudaDevice> {
    match device {
        Device::Cuda(d) => Ok(d.clone()),
        _ => anyhow::bail!("dsocr decode kernels require a CUDA device"),
    }
}

pub(crate) fn tensor_ptr_bf16(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    anyhow::ensure!(l.is_contiguous(), "dsocr decode: tensor not contiguous");
    let cuda = match &*st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("dsocr decode: tensor not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<bf16>()?;
    let view = slice.slice(l.start_offset()..l.start_offset() + len);
    let (p, _g) = view.device_ptr(stream);
    Ok(p)
}

pub(crate) fn wrap_bf16(
    slice: CudaSlice<bf16>,
    dims: &[usize],
    dev: &CudaDevice,
) -> Result<Tensor> {
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

pub(crate) fn wrap_f32(slice: CudaSlice<f32>, dims: &[usize], dev: &CudaDevice) -> Result<Tensor> {
    let storage = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
    Ok(Tensor::from_storage(
        candle_core::Storage::Cuda(storage),
        dims,
        candle_core::op::BackpropOp::none(),
        false,
    ))
}

fn tensor_ptr_f32(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    anyhow::ensure!(l.is_contiguous(), "dsocr decode: tensor not contiguous");
    let cuda = match &*st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("dsocr decode: tensor not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<f32>()?;
    let view = slice.slice(l.start_offset()..l.start_offset() + len);
    let (p, _g) = view.device_ptr(stream);
    Ok(p)
}

pub(crate) fn tensor_ptr_u32(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
    let (st, l) = t.storage_and_layout();
    anyhow::ensure!(l.is_contiguous(), "dsocr decode: tensor not contiguous");
    let cuda = match &*st {
        candle_core::Storage::Cuda(s) => s,
        _ => anyhow::bail!("dsocr decode: tensor not CUDA"),
    };
    let slice = cuda.as_cuda_slice::<u32>()?;
    let view = slice.slice(l.start_offset()..l.start_offset() + len);
    let (p, _g) = view.device_ptr(stream);
    Ok(p)
}

pub(crate) trait CastElem: cudarc::driver::DeviceRepr + Sized {
    type Raw;
    const NAME: &'static str;
    fn tensor_ptr(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64>;
    fn wrap(slice: CudaSlice<Self>, dims: &[usize], dev: &CudaDevice) -> Result<Tensor>;
}

impl CastElem for bf16 {
    type Raw = u16;
    const NAME: &'static str = "bf16";
    fn tensor_ptr(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
        tensor_ptr_bf16(t, stream, len)
    }
    fn wrap(slice: CudaSlice<Self>, dims: &[usize], dev: &CudaDevice) -> Result<Tensor> {
        wrap_bf16(slice, dims, dev)
    }
}

impl CastElem for f32 {
    type Raw = f32;
    const NAME: &'static str = "f32";
    fn tensor_ptr(t: &Tensor, stream: &Arc<CudaStream>, len: usize) -> Result<u64> {
        tensor_ptr_f32(t, stream, len)
    }
    fn wrap(slice: CudaSlice<Self>, dims: &[usize], dev: &CudaDevice) -> Result<Tensor> {
        wrap_f32(slice, dims, dev)
    }
}

type CastKernel<S, D> = unsafe fn(*mut c_void, *const S, *mut D, i32) -> i32;

fn cast_into<S: CastElem, D: CastElem>(
    t: &Tensor,
    out: &Tensor,
    device: &Device,
    kernel: CastKernel<S::Raw, D::Raw>,
) -> Result<()> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let n = t.elem_count();
    anyhow::ensure!(
        out.elem_count() == n,
        "cast_{}_{} into: len mismatch {} vs {}",
        S::NAME,
        D::NAME,
        n,
        out.elem_count()
    );
    let src = S::tensor_ptr(t, &stream, n)?;
    let dst = D::tensor_ptr(out, &stream, n)?;
    let rc = unsafe {
        kernel(
            stream.cu_stream() as *mut c_void,
            src as *const S::Raw,
            dst as *mut D::Raw,
            n as i32,
        )
    };
    anyhow::ensure!(rc == 0, "cast_{}_{} rc={rc}", S::NAME, D::NAME);
    Ok(())
}

fn cast_alloc<S: CastElem, D: CastElem>(
    t: &Tensor,
    device: &Device,
    kernel: CastKernel<S::Raw, D::Raw>,
) -> Result<Tensor> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let n = t.elem_count();
    let raw = unsafe { stream.alloc::<D>(n).map_err(|e| anyhow::anyhow!(e))? };
    let out = D::wrap(raw, t.dims(), &dev)?;
    cast_into::<S, D>(t, &out, device, kernel)?;
    Ok(out)
}

pub(crate) fn cast_tensor_bf16_to_f32(t: &Tensor, device: &Device) -> Result<Tensor> {
    cast_alloc::<bf16, f32>(t, device, nv_kernels::cuda::cast_bf16_f32)
}

pub(crate) fn cast_tensor_f32_to_bf16(t: &Tensor, device: &Device) -> Result<Tensor> {
    cast_alloc::<f32, bf16>(t, device, nv_kernels::cuda::cast_f32_bf16)
}

pub(crate) fn cast_bf16_to_f32_into(t: &Tensor, out: &Tensor, device: &Device) -> Result<()> {
    cast_into::<bf16, f32>(t, out, device, nv_kernels::cuda::cast_bf16_f32)
}

pub(crate) fn cast_f32_to_bf16_into(t: &Tensor, out: &Tensor, device: &Device) -> Result<()> {
    cast_into::<f32, bf16>(t, out, device, nv_kernels::cuda::cast_f32_bf16)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_apply_inplace_f32(
    q: &Tensor,
    k: &Tensor,
    cos_c: &Tensor,
    sin_c: &Tensor,
    pos_ptr: u64,
    n_tokens: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    device: &Device,
) -> Result<()> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let qp = tensor_ptr_f32(q, &stream, n_tokens * n_heads * head_dim)?;
    let kp = tensor_ptr_f32(k, &stream, n_tokens * n_kv_heads * head_dim)?;
    let cp = tensor_ptr_f32(cos_c, &stream, cos_c.elem_count())?;
    let sp = tensor_ptr_f32(sin_c, &stream, sin_c.elem_count())?;
    let rc = unsafe {
        nv_kernels::cuda::rope_f32(
            stream.cu_stream() as *mut c_void,
            qp as *mut f32,
            kp as *mut f32,
            cp as *const f32,
            sp as *const f32,
            pos_ptr as *const i32,
            n_tokens,
            n_heads,
            n_kv_heads,
            head_dim,
        )
    };
    anyhow::ensure!(rc == 0, "rope_f32 rc={rc}");
    Ok(())
}

pub(crate) fn residual_add_scale_bf16_into(
    a: &Tensor,
    b: &Tensor,
    scale: f32,
    out: &Tensor,
    device: &Device,
) -> Result<()> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let n = a.elem_count();
    anyhow::ensure!(
        b.elem_count() == n && out.elem_count() == n,
        "residual into: shape mismatch a={:?} b={:?} out={:?}",
        a.dims(),
        b.dims(),
        out.dims()
    );
    let pa = tensor_ptr_bf16(a, &stream, n)?;
    let pb = tensor_ptr_bf16(b, &stream, n)?;
    let py = tensor_ptr_bf16(out, &stream, n)?;
    let rc = unsafe {
        nv_kernels::cuda::residual_add_scale_bf16(
            stream.cu_stream() as *mut c_void,
            pa as *const u16,
            pb as *const u16,
            py as *mut u16,
            scale,
            n,
        )
    };
    anyhow::ensure!(rc == 0, "residual_add_scale_bf16 rc={rc}");
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn varlen_decode_attn(
    q: &Tensor,
    k_ptr: u64,
    v_ptr: u64,
    cu_q_ptr: u64,
    cu_k_ptr: u64,
    max_k: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let q_ptr = tensor_ptr_bf16(q, &stream, num_heads * head_dim)?;
    let mut o: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(num_heads * head_dim)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    let lse: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(num_heads)
        .map_err(|e| anyhow::anyhow!(e))?;
    {
        let (op, _g1) = o.device_ptr_mut(&stream);
        let (lp, _g2) = lse.device_ptr(&stream);
        unsafe {
            varlen_fwd_bf16(
                stream.cu_stream() as *mut c_void,
                &VarlenArgs {
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    o_ptr: op,
                    lse_ptr: lp,
                    cu_seqlens_q: cu_q_ptr,
                    cu_seqlens_k: cu_k_ptr,
                    max_seqlen_q: 1,
                    max_seqlen_k: max_k,
                    h: num_heads,
                    h_k: num_kv_heads,
                    d: head_dim,
                    softmax_scale: scale,
                    window_size_left: None,
                    window_size_right: Some(0),
                },
            )?;
        }
    }
    wrap_bf16(o, &[1, 1, num_heads, head_dim], &dev)
}

#[allow(clippy::too_many_arguments)]
fn varlen_decode_attn_into(
    q: &Tensor,
    k_ptr: u64,
    v_ptr: u64,
    cu_q_ptr: u64,
    cu_k_ptr: u64,
    max_k: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    o_t: &Tensor,
    lse: &mut CudaSlice<f32>,
    device: &Device,
) -> Result<()> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    anyhow::ensure!(
        o_t.elem_count() == num_heads * head_dim && lse.len() == num_heads,
        "varlen into: buffer shape mismatch"
    );
    let q_ptr = tensor_ptr_bf16(q, &stream, num_heads * head_dim)?;
    stream.memset_zeros(lse).map_err(|e| anyhow::anyhow!(e))?;
    let op = tensor_ptr_bf16(o_t, &stream, num_heads * head_dim)?;
    let (lp, _g) = lse.device_ptr(&stream);
    unsafe {
        varlen_fwd_bf16(
            stream.cu_stream() as *mut c_void,
            &VarlenArgs {
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr: op,
                lse_ptr: lp,
                cu_seqlens_q: cu_q_ptr,
                cu_seqlens_k: cu_k_ptr,
                max_seqlen_q: 1,
                max_seqlen_k: max_k,
                h: num_heads,
                h_k: num_kv_heads,
                d: head_dim,
                softmax_scale: scale,
                window_size_left: None,
                window_size_right: Some(0),
            },
        )?;
    }
    Ok(())
}

pub(crate) fn splitk_decode_supported(cfg: &DeepseekOcrDecoderConfig) -> bool {
    cfg.head_dim() == 128 && cfg.num_attention_heads == cfg.num_key_value_heads
}

pub(crate) fn splitk_decode_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_DSOCR_ATTN_SPLITK")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(true)
    })
}

pub(crate) fn splitk_scratch_elems(n_kv: usize) -> usize {
    nv_kernels::cuda::laguna_flash_decode_gqa_scratch_elems(n_kv as i32)
}

#[allow(clippy::too_many_arguments)]
fn splitk_decode_attn(
    q: &Tensor,
    k_ptr: u64,
    v_ptr: u64,
    total_ptr: u64,
    scratch_ptr: u64,
    fan_in_ptr: u64,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    device: &Device,
) -> Result<Tensor> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let q_ptr = tensor_ptr_bf16(q, &stream, num_heads * head_dim)?;
    let mut o: CudaSlice<bf16> = unsafe {
        stream
            .alloc::<bf16>(num_heads * head_dim)
            .map_err(|e| anyhow::anyhow!(e))?
    };
    {
        let (op, _g1) = o.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::laguna_flash_decode_gqa(
                stream.cu_stream() as *mut c_void,
                q_ptr as *const u16,
                k_ptr as *const u16,
                v_ptr as *const u16,
                op as *mut u16,
                total_ptr as *const i32,
                0,
                scratch_ptr as *mut f32,
                fan_in_ptr as *mut u32,
                num_heads as i32,
                num_heads as i32,
                head_dim as i32,
                0,
                scale,
            )
        };
        anyhow::ensure!(rc == 0, "dsocr splitk decode attn rc={rc}");
    }
    wrap_bf16(o, &[1, 1, num_heads, head_dim], &dev)
}

#[allow(clippy::too_many_arguments)]
fn splitk_decode_attn_into(
    q: &Tensor,
    k_ptr: u64,
    v_ptr: u64,
    total_ptr: u64,
    scratch_ptr: u64,
    fan_in_ptr: u64,
    num_heads: usize,
    head_dim: usize,
    scale: f32,
    o_t: &Tensor,
    device: &Device,
) -> Result<()> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    anyhow::ensure!(
        o_t.elem_count() == num_heads * head_dim,
        "splitk into: buffer shape mismatch"
    );
    let q_ptr = tensor_ptr_bf16(q, &stream, num_heads * head_dim)?;
    let op = tensor_ptr_bf16(o_t, &stream, num_heads * head_dim)?;
    let rc = unsafe {
        nv_kernels::cuda::laguna_flash_decode_gqa(
            stream.cu_stream() as *mut c_void,
            q_ptr as *const u16,
            k_ptr as *const u16,
            v_ptr as *const u16,
            op as *mut u16,
            total_ptr as *const i32,
            0,
            scratch_ptr as *mut f32,
            fan_in_ptr as *mut u32,
            num_heads as i32,
            num_heads as i32,
            head_dim as i32,
            0,
            scale,
        )
    };
    anyhow::ensure!(rc == 0, "dsocr splitk decode attn rc={rc}");
    Ok(())
}

pub(crate) fn decode_attention_eager(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
) -> Result<Tensor> {
    let dev = cuda_dev(q.device())?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let len = k.dims()[1];
    #[allow(deprecated)]
    let cu_q = stream
        .memcpy_stod(&[0i32, 1i32])
        .map_err(|e| anyhow::anyhow!(e))?;
    #[allow(deprecated)]
    let cu_k = stream
        .memcpy_stod(&[0i32, len as i32])
        .map_err(|e| anyhow::anyhow!(e))?;
    let (cu_q_ptr, _g1) = cu_q.device_ptr(&stream);
    let (cu_k_ptr, _g2) = cu_k.device_ptr(&stream);
    let k_ptr = tensor_ptr_bf16(k, &stream, len * cfg.num_kv_heads * cfg.head_dim)?;
    let v_ptr = tensor_ptr_bf16(v, &stream, len * cfg.num_kv_heads * cfg.head_dim)?;
    varlen_decode_attn(
        q,
        k_ptr,
        v_ptr,
        cu_q_ptr,
        cu_k_ptr,
        len,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.softmax_scale,
        q.device(),
    )
}

pub(crate) fn dense_decode_ffn(
    mlp: &Mlp,
    x_normed: &Tensor,
    resid: &Tensor,
    device: &Device,
) -> Result<Tensor> {
    let y = mlp.forward_fused_cuda(x_normed)?;
    crate::gemma4::residual_add_scale_bf16_op(resid, &y, 1.0, device)
}

fn gemv_env_ok() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_LINEAR_BF16_FORCE_CUBLAS").is_err())
}

#[allow(clippy::too_many_arguments)]
fn linear_bf16_into_raw(
    w: &Tensor,
    wt: Option<&Tensor>,
    x_slice: &CudaSlice<bf16>,
    x_off: usize,
    out: &mut CudaSlice<bf16>,
    m: usize,
    n: usize,
    k: usize,
    s: &Arc<CudaStream>,
) -> Result<()> {
    anyhow::ensure!(
        out.len() >= m * n,
        "moe prealloc linear: out {} < {m}x{n}",
        out.len()
    );
    if m == 1 && gemv_env_ok() && k % 2 == 0 {
        let (ws, wl) = w.storage_and_layout();
        anyhow::ensure!(
            wl.is_contiguous(),
            "moe prealloc linear: weight not contiguous"
        );
        let w_cuda = match &*ws {
            candle_core::Storage::Cuda(c) => c,
            _ => anyhow::bail!("moe prealloc linear: weight not CUDA"),
        };
        let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
        let w_view = w_slice.slice(wl.start_offset()..);
        let x_view = x_slice.slice(x_off..);
        let (xp, _g1) = x_view.device_ptr(s);
        let (wp, _g2) = w_view.device_ptr(s);
        let (yp, _g3) = out.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::gemv_bf16(
                s.cu_stream() as *mut c_void,
                wp as *const u16,
                xp as *const u16,
                yp as *mut u16,
                n as i32,
                k as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe prealloc gemv rc={rc}");
        return Ok(());
    }
    let wt =
        wt.ok_or_else(|| anyhow::anyhow!("moe prealloc linear: gemm path without weight_t"))?;
    let (ws, wl) = wt.storage_and_layout();
    anyhow::ensure!(
        wl.is_contiguous(),
        "moe prealloc linear: weight_t not contiguous"
    );
    let w_cuda = match &*ws {
        candle_core::Storage::Cuda(c) => c,
        _ => anyhow::bail!("moe prealloc linear: weight_t not CUDA"),
    };
    let w_slice = w_cuda.as_cuda_slice::<bf16>()?;
    let gemm = nv_quant::matmul::TensorCoreGemm::new(s.clone())?;
    gemm.bf16_matmul_row_major_offs(
        s,
        x_slice,
        x_off,
        w_slice,
        wl.start_offset(),
        out,
        m as u64,
        n as u64,
        k as u64,
        1.0,
        0.0,
    )
}

#[allow(clippy::too_many_arguments)]
fn linear_bf16_into(
    w: &Tensor,
    wt: Option<&Tensor>,
    x: &Tensor,
    out: &mut CudaSlice<bf16>,
    m: usize,
    n: usize,
    k: usize,
    s: &Arc<CudaStream>,
) -> Result<()> {
    anyhow::ensure!(
        x.elem_count() == m * k,
        "moe prealloc linear: x {:?} != {m}x{k}",
        x.dims()
    );
    let (xs, xl) = x.storage_and_layout();
    anyhow::ensure!(xl.is_contiguous(), "moe prealloc linear: x not contiguous");
    let x_cuda = match &*xs {
        candle_core::Storage::Cuda(c) => c,
        _ => anyhow::bail!("moe prealloc linear: x not CUDA"),
    };
    let x_slice = x_cuda.as_cuda_slice::<bf16>()?;
    linear_bf16_into_raw(w, wt, x_slice, xl.start_offset(), out, m, n, k, s)
}

pub(crate) struct MoeLayerPrealloc {
    gate_w: Tensor,
    gate_wt: Option<Tensor>,
    sh_gate_w: Tensor,
    sh_gate_wt: Option<Tensor>,
    sh_up_w: Tensor,
    sh_up_wt: Option<Tensor>,
    sh_down_w: Tensor,
    sh_down_wt: Option<Tensor>,
}

pub(crate) struct MoePrealloc {
    layers: Vec<Option<MoeLayerPrealloc>>,
    scores: CudaSlice<bf16>,
    sh_gate: CudaSlice<bf16>,
    sh_up: CudaSlice<bf16>,
    sh_act: CudaSlice<bf16>,
    sh_down: CudaSlice<bf16>,
    max_b: usize,
    e: usize,
    sh_inter: usize,
    hidden: usize,
}

unsafe impl Send for MoePrealloc {}

impl MoePrealloc {
    pub(crate) fn new(
        model: &DeepseekOcrDecoder,
        forked: &Arc<CudaStream>,
        max_b: usize,
    ) -> Result<Option<Self>> {
        anyhow::ensure!(max_b > 0, "moe prealloc: max_b must be > 0");
        let cfg = model.config();
        let hidden = cfg.hidden_size;
        let e = cfg.n_routed_experts;
        let need_wt = |k: usize| max_b > 1 || !gemv_env_ok() || k % 2 != 0;
        let eligible = |l: &nv_layers::Linear| -> bool {
            l.weight().is_some() && l.bias().is_none() && !l.has_lora()
        };
        let mk_wt = |l: &nv_layers::Linear, k: usize| -> Result<Option<Tensor>> {
            if need_wt(k) {
                let w = l.weight().expect("eligible bf16 weight");
                Ok(Some(w.t()?.contiguous()?))
            } else {
                Ok(None)
            }
        };
        let mut sh_inter = 0usize;
        let mut layers: Vec<Option<MoeLayerPrealloc>> = Vec::with_capacity(model.layers().len());
        for layer in model.layers() {
            let m = match &layer.ff {
                FeedForward::Moe(m) => m,
                FeedForward::Dense(_) => {
                    layers.push(None);
                    continue;
                }
            };
            let g = m.gate_bf16();
            let sh = m.shared_expert();
            if !(eligible(g)
                && eligible(sh.gate_proj())
                && eligible(sh.up_proj())
                && eligible(sh.down_proj()))
            {
                return Ok(None);
            }
            let inter = sh.gate_proj().out_features();
            if sh_inter == 0 {
                sh_inter = inter;
            }
            let shapes_ok = inter == sh_inter
                && sh.up_proj().out_features() == sh_inter
                && sh.down_proj().in_features() == sh_inter
                && sh.down_proj().out_features() == hidden
                && sh.gate_proj().in_features() == hidden
                && sh.up_proj().in_features() == hidden
                && g.in_features() == hidden
                && g.out_features() == e;
            if !shapes_ok {
                return Ok(None);
            }
            layers.push(Some(MoeLayerPrealloc {
                gate_w: g.weight().expect("gate bf16").clone(),
                gate_wt: mk_wt(g, hidden)?,
                sh_gate_w: sh.gate_proj().weight().expect("shared gate bf16").clone(),
                sh_gate_wt: mk_wt(sh.gate_proj(), hidden)?,
                sh_up_w: sh.up_proj().weight().expect("shared up bf16").clone(),
                sh_up_wt: mk_wt(sh.up_proj(), hidden)?,
                sh_down_w: sh.down_proj().weight().expect("shared down bf16").clone(),
                sh_down_wt: mk_wt(sh.down_proj(), sh_inter)?,
            }));
        }
        if sh_inter == 0 {
            return Ok(None);
        }
        let alloc = |n: usize| -> Result<CudaSlice<bf16>> {
            forked
                .alloc_zeros::<bf16>(n)
                .map_err(|err| anyhow::anyhow!(err))
        };
        Ok(Some(Self {
            layers,
            scores: alloc(max_b * e)?,
            sh_gate: alloc(max_b * sh_inter)?,
            sh_up: alloc(max_b * sh_inter)?,
            sh_act: alloc(max_b * sh_inter)?,
            sh_down: alloc(max_b * hidden)?,
            max_b,
            e,
            sh_inter,
            hidden,
        }))
    }

    pub(crate) fn gate_scores_cast(
        &mut self,
        li: usize,
        x_normed: &Tensor,
        b: usize,
        scores_f32: &mut CudaSlice<f32>,
        s: &Arc<CudaStream>,
    ) -> Result<()> {
        anyhow::ensure!(b > 0 && b <= self.max_b, "moe prealloc: bad b={b}");
        let Self {
            layers,
            scores,
            e,
            hidden,
            ..
        } = self;
        let lp = layers
            .get(li)
            .and_then(|l| l.as_ref())
            .ok_or_else(|| anyhow::anyhow!("moe prealloc: no entry for layer {li}"))?;
        linear_bf16_into(
            &lp.gate_w,
            lp.gate_wt.as_ref(),
            x_normed,
            scores,
            b,
            *e,
            *hidden,
            s,
        )?;
        let n = b * *e;
        anyhow::ensure!(scores_f32.len() >= n, "moe prealloc: scores_f32 too small");
        let (sp, _g1) = scores.device_ptr(s);
        let (dp, _g2) = scores_f32.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::cast_bf16_f32(
                s.cu_stream() as *mut c_void,
                sp as *const u16,
                dp as *mut f32,
                n as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe gate cast rc={rc}");
        Ok(())
    }

    pub(crate) fn shared_cast(
        &mut self,
        li: usize,
        x_normed: &Tensor,
        b: usize,
        shared_f32: &mut CudaSlice<f32>,
        s: &Arc<CudaStream>,
    ) -> Result<()> {
        anyhow::ensure!(b > 0 && b <= self.max_b, "moe prealloc: bad b={b}");
        let Self {
            layers,
            sh_gate,
            sh_up,
            sh_act,
            sh_down,
            sh_inter,
            hidden,
            ..
        } = self;
        let lp = layers
            .get(li)
            .and_then(|l| l.as_ref())
            .ok_or_else(|| anyhow::anyhow!("moe prealloc: no entry for layer {li}"))?;
        linear_bf16_into(
            &lp.sh_gate_w,
            lp.sh_gate_wt.as_ref(),
            x_normed,
            sh_gate,
            b,
            *sh_inter,
            *hidden,
            s,
        )?;
        linear_bf16_into(
            &lp.sh_up_w,
            lp.sh_up_wt.as_ref(),
            x_normed,
            sh_up,
            b,
            *sh_inter,
            *hidden,
            s,
        )?;
        {
            let (gp, _g1) = sh_gate.device_ptr(s);
            let (up, _g2) = sh_up.device_ptr(s);
            let (ap, _g3) = sh_act.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::silu_mul_bf16_candle(
                    s.cu_stream() as *mut c_void,
                    gp as *const u16,
                    up as *const u16,
                    ap as *mut u16,
                    b * *sh_inter,
                )
            };
            anyhow::ensure!(rc == 0, "moe shared silu rc={rc}");
        }
        linear_bf16_into_raw(
            &lp.sh_down_w,
            lp.sh_down_wt.as_ref(),
            sh_act,
            0,
            sh_down,
            b,
            *hidden,
            *sh_inter,
            s,
        )?;
        let n = b * *hidden;
        anyhow::ensure!(shared_f32.len() >= n, "moe prealloc: shared_f32 too small");
        let (sp, _g1) = sh_down.device_ptr(s);
        let (dp, _g2) = shared_f32.device_ptr_mut(s);
        let rc = unsafe {
            nv_kernels::cuda::cast_bf16_f32(
                s.cu_stream() as *mut c_void,
                sp as *const u16,
                dp as *mut f32,
                n as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe shared cast rc={rc}");
        Ok(())
    }
}

pub(crate) struct DecodeScratch {
    k: usize,
    e: usize,
    inter: usize,
    hidden: usize,
    seeds: CudaSlice<u64>,
    scores_f32: CudaSlice<f32>,
    probs: CudaSlice<f32>,
    trash_token: CudaSlice<u32>,
    ids: CudaSlice<i32>,
    weights: CudaSlice<f32>,
    h_rows: CudaSlice<bf16>,
    shared_f32: CudaSlice<f32>,
    out: Tensor,
}

unsafe impl Send for DecodeScratch {}

impl DecodeScratch {
    pub(crate) fn new(
        k: usize,
        e: usize,
        inter: usize,
        hidden: usize,
        device: &Device,
    ) -> Result<Self> {
        anyhow::ensure!(k > 0 && e > 0 && k <= e, "dsocr scratch: bad k={k} e={e}");
        let dev = cuda_dev(device)?;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let seeds = stream
            .alloc_zeros::<u64>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let scores_f32 = stream
            .alloc_zeros::<f32>(e)
            .map_err(|e| anyhow::anyhow!(e))?;
        let probs = stream
            .alloc_zeros::<f32>(e)
            .map_err(|e| anyhow::anyhow!(e))?;
        let trash_token = stream
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let ids = stream
            .alloc_zeros::<i32>(k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let weights = stream
            .alloc_zeros::<f32>(k)
            .map_err(|e| anyhow::anyhow!(e))?;
        let h_rows = stream
            .alloc_zeros::<bf16>(k * inter)
            .map_err(|e| anyhow::anyhow!(e))?;
        let shared_f32 = stream
            .alloc_zeros::<f32>(hidden)
            .map_err(|e| anyhow::anyhow!(e))?;
        let out = Tensor::zeros((1usize, 1usize, hidden), DType::BF16, device)?;
        stream.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        Ok(Self {
            k,
            e,
            inter,
            hidden,
            seeds,
            scores_f32,
            probs,
            trash_token,
            ids,
            weights,
            h_rows,
            shared_f32,
            out,
        })
    }
}

pub(crate) fn moe_decode_ffn(
    moe: &DeepseekMoe,
    x_normed: &Tensor,
    resid: &Tensor,
    scratch: &mut DecodeScratch,
    mut moe_pre: Option<&mut MoePrealloc>,
    li: usize,
    device: &Device,
) -> Result<Tensor> {
    let dev = cuda_dev(device)?;
    let stream = nv_layers::cuda_stream::current_stream(&dev);
    let stacked = moe
        .stacked()
        .ok_or_else(|| anyhow::anyhow!("moe_decode_ffn: experts not stacked"))?;
    let k = moe.top_k();
    let e = moe.num_experts();
    let hidden = scratch.hidden;
    let inter = scratch.inter;
    anyhow::ensure!(
        k == scratch.k && e == scratch.e,
        "moe_decode_ffn: scratch shape mismatch"
    );
    anyhow::ensure!(
        x_normed.elem_count() == hidden && resid.elem_count() == hidden,
        "moe_decode_ffn: expected single-token input"
    );

    match moe_pre.as_deref_mut() {
        Some(mp) => mp.gate_scores_cast(li, x_normed, 1, &mut scratch.scores_f32, &stream)?,
        None => {
            let scores = moe.gate_bf16().forward(x_normed)?.contiguous()?;
            let sp = tensor_ptr_bf16(&scores, &stream, e)?;
            let (dp, _g) = scratch.scores_f32.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    stream.cu_stream() as *mut c_void,
                    sp as *const u16,
                    dp as *mut f32,
                    e as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe gate cast rc={rc}");
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
                1,
                e,
                1.0,
                0,
                1.0,
            )
        };
        anyhow::ensure!(rc == 0, "moe softmax rc={rc}");
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
                1,
                e as i32,
                k as i32,
                0,
                0.0,
                0,
                1.0,
            )
        };
        anyhow::ensure!(rc == 0, "moe route rc={rc}");
    }
    {
        let (pp, _g1) = scratch.probs.device_ptr(&stream);
        let (ip, _g2) = scratch.ids.device_ptr(&stream);
        let (wp, _g3) = scratch.weights.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::gather_f32_by_ids(
                stream.cu_stream() as *mut c_void,
                pp as *const f32,
                ip as *const i32,
                wp as *mut f32,
                k as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe weight gather rc={rc}");
    }

    match moe_pre.as_deref_mut() {
        Some(mp) => mp.shared_cast(li, x_normed, 1, &mut scratch.shared_f32, &stream)?,
        None => {
            let shared_t = moe.shared_expert().forward_fused_cuda(x_normed)?;
            let sp = tensor_ptr_bf16(&shared_t, &stream, hidden)?;
            let (dp, _g) = scratch.shared_f32.device_ptr_mut(&stream);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    stream.cu_stream() as *mut c_void,
                    sp as *const u16,
                    dp as *mut f32,
                    hidden as i32,
                )
            };
            anyhow::ensure!(rc == 0, "moe shared cast rc={rc}");
        }
    }

    let gate_src = tensor_ptr_bf16(&stacked.gate, &stream, e * inter * hidden)?;
    let up_src = tensor_ptr_bf16(&stacked.up, &stream, e * inter * hidden)?;
    let down_src = tensor_ptr_bf16(&stacked.down, &stream, e * hidden * inter)?;
    let x_ptr = tensor_ptr_bf16(x_normed, &stream, hidden)?;
    {
        let (ids_ptr, _g1) = scratch.ids.device_ptr(&stream);
        let (hp, _g2) = scratch.h_rows.device_ptr_mut(&stream);
        let rc = unsafe {
            nv_kernels::cuda::moe_gemv_swiglu_bf16_m1(
                stream.cu_stream() as *mut c_void,
                gate_src as *const u16,
                up_src as *const u16,
                ids_ptr as *const i32,
                x_ptr as *const u16,
                hp as *mut u16,
                k as i32,
                e as i32,
                inter as i32,
                hidden as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe swiglu gemv rc={rc}");
    }
    {
        let (ids_ptr, _g1) = scratch.ids.device_ptr(&stream);
        let (wp, _g2) = scratch.weights.device_ptr(&stream);
        let (hp, _g3) = scratch.h_rows.device_ptr(&stream);
        let (sp, _g4) = scratch.shared_f32.device_ptr(&stream);
        let rp = tensor_ptr_bf16(resid, &stream, hidden)?;
        let op = tensor_ptr_bf16(&scratch.out, &stream, hidden)?;
        let rc = unsafe {
            nv_kernels::cuda::moe_gemv_down_tail_bf16_m1(
                stream.cu_stream() as *mut c_void,
                down_src as *const u16,
                ids_ptr as *const i32,
                wp as *const f32,
                hp as *const u16,
                sp as *const f32,
                rp as *const u16,
                op as *mut u16,
                k as i32,
                e as i32,
                hidden as i32,
                inter as i32,
            )
        };
        anyhow::ensure!(rc == 0, "moe down gemv tail rc={rc}");
    }
    Ok(scratch.out.clone())
}

pub fn graph_supported(model: &DeepseekOcrDecoder) -> std::result::Result<(), String> {
    if !matches!(model.device(), Device::Cuda(_)) {
        return Err("device is not CUDA".into());
    }
    if model.dtype() != DType::BF16 {
        return Err(format!("decoder dtype {:?} != BF16", model.dtype()));
    }
    let cfg = model.config();
    if cfg.head_dim() % 8 != 0 || cfg.head_dim() > 256 {
        return Err(format!(
            "head_dim {} unsupported by fa2 shim",
            cfg.head_dim()
        ));
    }
    if cfg.num_experts_per_tok > 32 {
        return Err(format!("top_k {} > 32", cfg.num_experts_per_tok));
    }
    for (li, layer) in model.layers().iter().enumerate() {
        match &layer.ff {
            FeedForward::Dense(mlp) => {
                if mlp.gate_proj().weight().is_none() {
                    return Err(format!("layer {li}: dense mlp has no bf16 weights"));
                }
            }
            FeedForward::Moe(m) => {
                if !m.decode_ready() {
                    return Err(format!(
                        "layer {li}: moe not decode-ready (stacked bf16 experts required, \
                         norm_topk_prob=false, routed_scaling=1.0)"
                    ));
                }
            }
        }
        if layer.q_proj.weight().is_none() {
            return Err(format!("layer {li}: attention projections are not bf16"));
        }
    }
    Ok(())
}

pub fn model_uses_nvfp4(model: &DeepseekOcrDecoder) -> bool {
    model.layers().iter().any(|layer| {
        [&layer.q_proj, &layer.k_proj, &layer.v_proj, &layer.o_proj]
            .iter()
            .any(|l| l.nvfp4_runner().is_some())
            || match &layer.ff {
                FeedForward::Dense(mlp) => mlp.gate_proj().nvfp4_runner().is_some(),
                FeedForward::Moe(_) => false,
            }
    })
}

struct CtxErrDrain(Arc<CudaContext>);

impl Drop for CtxErrDrain {
    fn drop(&mut self) {
        if let Err(e) = self.0.check_err() {
            if graph_debug() {
                eprintln!("[dsocr-graph-drop] drained deferred ctx error from teardown: {e:?}");
            }
        }
    }
}

struct PreallocBufs {
    tok_t: Tensor,
    q_f32: Tensor,
    k_f32: Tensor,
    q_bf: Tensor,
    k_bf: Tensor,
    attn_o: Tensor,
    attn_lse: CudaSlice<f32>,
    resid_a: Tensor,
    resid_b: Tensor,
    cos_c: Tensor,
    sin_c: Tensor,
    moe: Option<MoePrealloc>,
}

impl PreallocBufs {
    fn new(model: &DeepseekOcrDecoder, forked: &Arc<CudaStream>, dev: &CudaDevice) -> Result<Self> {
        let cfg = model.config();
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        let hidden = cfg.hidden_size;
        anyhow::ensure!(
            model.rope().config().head_dim == hd,
            "dsocr prealloc: rope head_dim {} != cfg {}",
            model.rope().config().head_dim,
            hd
        );
        let tok_t = {
            let slice = forked
                .alloc_zeros::<u32>(1)
                .map_err(|e| anyhow::anyhow!(e))?;
            let st = candle_core::CudaStorage::wrap_cuda_slice(slice, dev.clone());
            Tensor::from_storage(
                candle_core::Storage::Cuda(st),
                (1usize,),
                candle_core::op::BackpropOp::none(),
                false,
            )
        };
        let q_f32 = wrap_f32(
            forked
                .alloc_zeros::<f32>(n_heads * hd)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, n_heads, hd],
            dev,
        )?;
        let k_f32 = wrap_f32(
            forked
                .alloc_zeros::<f32>(n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, n_kv, hd],
            dev,
        )?;
        let q_bf = wrap_bf16(
            forked
                .alloc_zeros::<bf16>(n_heads * hd)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, n_heads, hd],
            dev,
        )?;
        let k_bf = wrap_bf16(
            forked
                .alloc_zeros::<bf16>(n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, n_kv, hd],
            dev,
        )?;
        let attn_o = wrap_bf16(
            forked
                .alloc_zeros::<bf16>(n_heads * hd)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, n_heads, hd],
            dev,
        )?;
        let attn_lse = forked
            .alloc_zeros::<f32>(n_heads)
            .map_err(|e| anyhow::anyhow!(e))?;
        let resid_a = wrap_bf16(
            forked
                .alloc_zeros::<bf16>(hidden)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, hidden],
            dev,
        )?;
        let resid_b = wrap_bf16(
            forked
                .alloc_zeros::<bf16>(hidden)
                .map_err(|e| anyhow::anyhow!(e))?,
            &[1, 1, hidden],
            dev,
        )?;
        let cos_c = model.rope().cos().contiguous()?;
        let sin_c = model.rope().sin().contiguous()?;
        anyhow::ensure!(
            cos_c.dtype() == DType::F32 && sin_c.dtype() == DType::F32,
            "dsocr prealloc: rope tables must be f32"
        );
        let moe = MoePrealloc::new(model, forked, 1)?;
        if moe.is_none() {
            eprintln!(
                "[dsocr] graph prealloc: MoE gate/shared linears not plain bf16; \
                 MoE internals keep in-capture allocations"
            );
        }
        Ok(Self {
            tok_t,
            q_f32,
            k_f32,
            q_bf,
            k_bf,
            attn_o,
            attn_lse,
            resid_a,
            resid_b,
            cos_c,
            sin_c,
            moe,
        })
    }
}

pub struct DsocrDecodeGraph {
    model: Arc<DeepseekOcrDecoder>,
    dev: CudaDevice,
    device: Device,
    forked: Arc<CudaStream>,
    runner: CudaGraphRunner,
    tok_buf: CudaSlice<u32>,
    host_tok: Box<[u32; 1]>,
    pos_buf: CudaSlice<i32>,
    host_pos: Box<[i32; 1]>,
    cu_q: CudaSlice<i32>,
    cu_k: CudaSlice<i32>,
    host_cu_k: Box<[i32; 2]>,
    kv: Vec<(CudaSlice<bf16>, CudaSlice<bf16>)>,
    logits_buf: CudaSlice<f32>,
    scratch: DecodeScratch,
    attn_scratch: CudaSlice<f32>,
    attn_fan_in: CudaSlice<u32>,
    prealloc: Option<PreallocBufs>,
    splitk: bool,
    cap: usize,
    cur_len: usize,
    captured: bool,
    pending: std::sync::atomic::AtomicBool,
    #[allow(dead_code)]
    err_drain: CtxErrDrain,
}

unsafe impl Send for DsocrDecodeGraph {}

impl DsocrDecodeGraph {
    pub fn new(model: Arc<DeepseekOcrDecoder>, cap: usize) -> Result<Self> {
        graph_supported(&model).map_err(|e| anyhow::anyhow!("dsocr graph unsupported: {e}"))?;
        anyhow::ensure!(cap > 0, "dsocr graph: cap must be > 0");
        let device = model.device().clone();
        let dev = cuda_dev(&device)?;
        let cfg = model.config();
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();

        let raw_ctx: Arc<CudaContext> = dev.cuda_stream().context().clone();
        if graph_debug() {
            if let Err(e) = raw_ctx.check_err() {
                eprintln!("[dsocr-graph-new] pre-existing deferred ctx error: {e:?}");
            }
        }
        let _init = lock_init();
        if raw_ctx.is_event_tracking() {
            dev.cuda_stream()
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-untrack legacy sync: {e:?}"))?;
            unsafe { raw_ctx.disable_event_tracking() };
        }
        let forked = raw_ctx
            .new_stream()
            .map_err(|e| anyhow::anyhow!("dsocr graph stream: {e:?}"))?;
        if model_uses_nvfp4(&model) {
            nv_quant::nvfp4::ensure_workspace_for_stream(&forked)?;
        }
        let _ = nv_quant::matmul::TensorCoreGemm::new(forked.clone())?;

        let tok_buf = forked
            .alloc_zeros::<u32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pos_buf = forked
            .alloc_zeros::<i32>(1)
            .map_err(|e| anyhow::anyhow!(e))?;
        #[allow(deprecated)]
        let cu_q = forked
            .memcpy_stod(&[0i32, 1i32])
            .map_err(|e| anyhow::anyhow!(e))?;
        let cu_k = forked
            .alloc_zeros::<i32>(2)
            .map_err(|e| anyhow::anyhow!(e))?;
        let mut kv = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            let k = forked
                .alloc_zeros::<bf16>(cap * n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?;
            let v = forked
                .alloc_zeros::<bf16>(cap * n_kv * hd)
                .map_err(|e| anyhow::anyhow!(e))?;
            kv.push((k, v));
        }
        let logits_buf = forked
            .alloc_zeros::<f32>(cfg.vocab_size)
            .map_err(|e| anyhow::anyhow!(e))?;
        let scratch = nv_layers::cuda_stream::with_stream(forked.clone(), || {
            DecodeScratch::new(
                cfg.num_experts_per_tok,
                cfg.n_routed_experts,
                cfg.moe_intermediate_size,
                cfg.hidden_size,
                &device,
            )
        })?;
        let splitk = splitk_decode_enabled() && splitk_decode_supported(cfg);
        let attn_scratch = forked
            .alloc_zeros::<f32>(if splitk {
                splitk_scratch_elems(n_kv)
            } else {
                1
            })
            .map_err(|e| anyhow::anyhow!(e))?;
        let attn_fan_in = forked
            .alloc_zeros::<u32>(if splitk { n_kv } else { 1 })
            .map_err(|e| anyhow::anyhow!(e))?;
        let prealloc = if graph_prealloc_enabled() {
            Some(PreallocBufs::new(&model, &forked, &dev)?)
        } else {
            None
        };
        if splitk {
            eprintln!(
                "[dsocr] decode attention: split-K (grid {}x{} blocks), heads {} hd {}",
                n_kv,
                std::env::var("NV_LAGUNA_M1_SPLITS")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .filter(|v| *v == 8 || *v == 16 || *v == 32)
                    .unwrap_or(16),
                cfg.num_attention_heads,
                hd
            );
        }
        if splitk_decode_enabled() && !splitk {
            eprintln!(
                "[dsocr] NV_DSOCR_ATTN_SPLITK set but unsupported for this checkpoint \
                 (head_dim {} must be 128, heads {} must equal kv heads {}); using varlen",
                cfg.head_dim(),
                cfg.num_attention_heads,
                cfg.num_key_value_heads
            );
        }
        forked.synchronize().map_err(|e| anyhow::anyhow!(e))?;
        let runner = CudaGraphRunner::new(forked.clone());
        Ok(Self {
            model,
            dev,
            device,
            forked,
            runner,
            tok_buf,
            host_tok: Box::new([0u32; 1]),
            pos_buf,
            host_pos: Box::new([0i32; 1]),
            cu_q,
            cu_k,
            host_cu_k: Box::new([0i32; 2]),
            kv,
            logits_buf,
            scratch,
            attn_scratch,
            attn_fan_in,
            prealloc,
            splitk,
            cap,
            cur_len: 0,
            captured: false,
            pending: std::sync::atomic::AtomicBool::new(false),
            err_drain: CtxErrDrain(raw_ctx),
        })
    }

    pub fn node_count(&self) -> usize {
        self.runner.cached_node_count()
    }

    pub fn cap(&self) -> usize {
        self.cap
    }

    pub fn current_len(&self) -> usize {
        self.cur_len
    }

    pub fn captured(&self) -> bool {
        self.captured
    }

    pub fn reset(&mut self) {
        self.cur_len = 0;
    }

    pub fn synchronize(&self) -> Result<()> {
        nv_layers::cuda_stream::sync_legacy_then_forked(&self.dev, &self.forked)
    }

    pub fn load_kv_from_cache(&mut self, cache: &DeepseekOcrKvCache) -> Result<()> {
        let cfg = self.model.config();
        let len = cache.current_len();
        anyhow::ensure!(
            len <= self.cap,
            "prefill length {len} exceeds graph capacity {}",
            self.cap
        );
        let n = len * cfg.num_key_value_heads * cfg.head_dim();
        self.dev
            .cuda_stream()
            .synchronize()
            .map_err(|e| anyhow::anyhow!("pre-copy legacy sync: {e:?}"))?;
        for li in 0..cfg.num_hidden_layers {
            let (kt, vt) = cache.layer_bufs(li);
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
                let mut dst_view = dst.slice_mut(0..n);
                self.forked
                    .memcpy_dtod(&view, &mut dst_view)
                    .map_err(|e| anyhow::anyhow!("kv load dtod: {e:?}"))?;
            }
        }
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("kv load sync: {e:?}"))?;
        self.cur_len = len;
        Ok(())
    }

    pub fn step(&mut self, token: u32) -> Result<()> {
        anyhow::ensure!(
            self.cur_len + 1 <= self.cap,
            "dsocr graph step overflows capacity {}",
            self.cap
        );
        if self.pending.load(std::sync::atomic::Ordering::Relaxed) {
            self.forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pending replay sync: {e:?}"))?;
            self.pending
                .store(false, std::sync::atomic::Ordering::Relaxed);
        }
        self.host_tok[0] = token;
        self.host_pos[0] = self.cur_len as i32;
        self.host_cu_k[0] = 0;
        self.host_cu_k[1] = (self.cur_len + 1) as i32;

        let legacy = self.dev.cuda_stream();
        let raw_ctx = legacy.context().clone();
        if raw_ctx.is_event_tracking() {
            unsafe { raw_ctx.disable_event_tracking() };
            legacy
                .synchronize()
                .map_err(|e| anyhow::anyhow!("pre-capture legacy sync: {e:?}"))?;
        }

        let was_captured = self.captured;
        let forked = self.forked.clone();
        let DsocrDecodeGraph {
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
            kv,
            logits_buf,
            scratch,
            attn_scratch,
            attn_fan_in,
            prealloc,
            splitk,
            cap,
            ..
        } = self;
        let cap = *cap;
        let splitk = *splitk;
        let cfg = model.config();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let hd = cfg.head_dim();
        let vocab = cfg.vocab_size;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut body = |s: &Arc<CudaStream>,
                        scratch: &mut DecodeScratch,
                        mut prealloc: Option<&mut PreallocBufs>|
         -> Result<()> {
            s.memcpy_htod(&host_tok[..], tok_buf)
                .map_err(|e| anyhow::anyhow!("htod tok: {e:?}"))?;
            s.memcpy_htod(&host_pos[..], pos_buf)
                .map_err(|e| anyhow::anyhow!("htod pos: {e:?}"))?;
            s.memcpy_htod(&host_cu_k[..], cu_k)
                .map_err(|e| anyhow::anyhow!("htod cu_k: {e:?}"))?;
            let (tokens_t, pos_t) = if let Some(p) = prealloc.as_mut() {
                let (tok_src, _gt) = tok_buf.device_ptr(s);
                let tok_dst = tensor_ptr_u32(&p.tok_t, s, 1)?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        tok_dst,
                        tok_src,
                        std::mem::size_of::<u32>(),
                        s.cu_stream(),
                    )
                }
                .map_err(|e| anyhow::anyhow!("tok dtod: {e:?}"))?;
                (p.tok_t.clone(), None)
            } else {
                let tok_clone = tok_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let pos_clone = pos_buf.try_clone().map_err(|e| anyhow::anyhow!(e))?;
                let tokens_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(tok_clone, dev.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (1usize,),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                let pos_t = {
                    let st = candle_core::CudaStorage::wrap_cuda_slice(pos_clone, dev.clone());
                    Tensor::from_storage(
                        candle_core::Storage::Cuda(st),
                        (1usize,),
                        candle_core::op::BackpropOp::none(),
                        false,
                    )
                };
                (tokens_t, Some(pos_t))
            };
            let (cu_q_ptr, _gq) = cu_q.device_ptr(s);
            let (cu_k_ptr, _gk) = cu_k.device_ptr(s);
            let (pos_ptr, _gp) = pos_buf.device_ptr(s);
            let (attn_scratch_ptr, _gs) = attn_scratch.device_ptr(s);
            let (attn_fan_in_ptr, _gf) = attn_fan_in.device_ptr(s);

            let mut x =
                crate::gemma4::embed_lookup_bf16_op(model.embed_weight_t(), &tokens_t, device)?
                    .reshape((1usize, 1usize, hidden))?;
            for (li, layer) in model.layers().iter().enumerate() {
                let normed = layer.input_layernorm.forward(&x)?;
                let q = layer
                    .q_proj
                    .forward(&normed)?
                    .reshape((1usize, 1usize, n_heads, hd))?;
                let k = layer
                    .k_proj
                    .forward(&normed)?
                    .reshape((1usize, 1usize, n_kv, hd))?;
                let v = layer
                    .v_proj
                    .forward(&normed)?
                    .reshape((1usize, 1usize, n_kv, hd))?;
                let (q_bf, k_bf) = if let Some(p) = prealloc.as_mut() {
                    cast_bf16_to_f32_into(&q, &p.q_f32, device)?;
                    cast_bf16_to_f32_into(&k, &p.k_f32, device)?;
                    rope_apply_inplace_f32(
                        &p.q_f32, &p.k_f32, &p.cos_c, &p.sin_c, pos_ptr, 1, n_heads, n_kv, hd,
                        device,
                    )?;
                    cast_f32_to_bf16_into(&p.q_f32, &p.q_bf, device)?;
                    cast_f32_to_bf16_into(&p.k_f32, &p.k_bf, device)?;
                    (p.q_bf.clone(), p.k_bf.clone())
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
                let k_src = tensor_ptr_bf16(&k_bf, s, n_kv * hd)?;
                let v_src = tensor_ptr_bf16(&v, s, n_kv * hd)?;
                let (k_dst, _g1) = kv[li].0.device_ptr(s);
                let (v_dst, _g2) = kv[li].1.device_ptr(s);
                for (src, dst) in [(k_src, k_dst), (v_src, v_dst)] {
                    let rc = unsafe {
                        nv_kernels::cuda::kv_ring_append_bf16(
                            s.cu_stream() as *mut c_void,
                            src as *const u16,
                            dst as *mut u16,
                            pos_ptr as *const i32,
                            1,
                            cap as i32,
                            n_kv as i32,
                            hd as i32,
                        )
                    };
                    anyhow::ensure!(rc == 0, "kv append rc={rc} layer {li}");
                }
                let o = if splitk {
                    if let Some(p) = prealloc.as_mut() {
                        splitk_decode_attn_into(
                            &q_bf,
                            k_dst,
                            v_dst,
                            cu_k_ptr + 4,
                            attn_scratch_ptr,
                            attn_fan_in_ptr,
                            n_heads,
                            hd,
                            scale,
                            &p.attn_o,
                            device,
                        )?;
                        p.attn_o.clone()
                    } else {
                        splitk_decode_attn(
                            &q_bf,
                            k_dst,
                            v_dst,
                            cu_k_ptr + 4,
                            attn_scratch_ptr,
                            attn_fan_in_ptr,
                            n_heads,
                            hd,
                            scale,
                            device,
                        )?
                    }
                } else if let Some(p) = prealloc.as_mut() {
                    let PreallocBufs {
                        attn_o, attn_lse, ..
                    } = &mut **p;
                    varlen_decode_attn_into(
                        &q_bf, k_dst, v_dst, cu_q_ptr, cu_k_ptr, cap, n_heads, n_kv, hd, scale,
                        attn_o, attn_lse, device,
                    )?;
                    attn_o.clone()
                } else {
                    varlen_decode_attn(
                        &q_bf, k_dst, v_dst, cu_q_ptr, cu_k_ptr, cap, n_heads, n_kv, hd, scale,
                        device,
                    )?
                };
                let o2 = o.reshape((1usize, 1usize, n_heads * hd))?;
                let attn_out = layer.o_proj.forward(&o2)?;
                let x_after = if let Some(p) = prealloc.as_mut() {
                    residual_add_scale_bf16_into(&x, &attn_out, 1.0, &p.resid_a, device)?;
                    p.resid_a.clone()
                } else {
                    crate::gemma4::residual_add_scale_bf16_op(&x, &attn_out, 1.0, device)?
                };
                let normed2 = layer.post_attention_layernorm.forward(&x_after)?;
                x = match &layer.ff {
                    FeedForward::Dense(mlp) => {
                        if let Some(p) = prealloc.as_mut() {
                            let y = mlp.forward_fused_cuda(&normed2)?;
                            residual_add_scale_bf16_into(&x_after, &y, 1.0, &p.resid_b, device)?;
                            p.resid_b.clone()
                        } else {
                            dense_decode_ffn(mlp, &normed2, &x_after, device)?
                        }
                    }
                    FeedForward::Moe(m) => moe_decode_ffn(
                        m,
                        &normed2,
                        &x_after,
                        scratch,
                        prealloc.as_mut().and_then(|p| p.moe.as_mut()),
                        li,
                        device,
                    )?,
                };
            }
            let h = model.final_norm().forward(&x)?;
            let logits_bf = model.lm_head().forward(&h)?;
            let lp_src = tensor_ptr_bf16(&logits_bf, s, vocab)?;
            let (lp_dst, _g) = logits_buf.device_ptr_mut(s);
            let rc = unsafe {
                nv_kernels::cuda::cast_bf16_f32(
                    s.cu_stream() as *mut c_void,
                    lp_src as *const u16,
                    lp_dst as *mut f32,
                    vocab as i32,
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
            nv_layers::cuda_stream::with_stream(forked.clone(), || {
                body(&forked, scratch, prealloc.as_mut())
            })
            .context("dsocr graph warm pass")?;
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("warm sync: {e:?}"))?;
        }
        let launch_on = if nullstream_arm() {
            Some(&legacy)
        } else {
            None
        };
        runner
            .run_on(1u64, launch_on, |s| {
                nv_layers::cuda_stream::with_stream(s.clone(), || {
                    body(s, scratch, prealloc.as_mut())
                })
            })
            .context("dsocr graph capture/replay")?;
        if !was_captured {
            forked
                .synchronize()
                .map_err(|e| anyhow::anyhow!("post-capture sync: {e:?}"))?;
            self.captured = true;
            if graph_debug() {
                graph_mem_report("post-capture", forked.context().ordinal());
            }
        }
        self.pending
            .store(true, std::sync::atomic::Ordering::Relaxed);
        self.cur_len += 1;
        Ok(())
    }

    pub fn logits_host(&self) -> Result<Vec<f32>> {
        if nullstream_arm() {
            self.synchronize()?;
        }
        let mut out = vec![0f32; self.logits_buf.len()];
        self.forked
            .memcpy_dtoh(&self.logits_buf, &mut out)
            .map_err(|e| anyhow::anyhow!("dtoh logits: {e:?}"))?;
        self.forked
            .synchronize()
            .map_err(|e| anyhow::anyhow!("logits forked sync: {e:?}"))?;
        self.pending
            .store(false, std::sync::atomic::Ordering::Relaxed);
        Ok(out)
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        vision_features: Option<&Tensor>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let model = self.model.clone();
        let cfg_max = model.config().max_position_embeddings;
        let max_len = (prompt_tokens.len() + opts.max_new_tokens)
            .min(cfg_max)
            .min(self.cap);
        if prompt_tokens.len() >= max_len {
            anyhow::bail!(
                "prompt length {} leaves no room to generate (max {})",
                prompt_tokens.len(),
                max_len
            );
        }
        let mut cache = model.new_kv_cache(max_len)?;
        let x = model.embed_tokens_with_vision(prompt_tokens, vision_features)?;
        let hidden = model.forward_embeds_hidden(&x, &mut cache)?;
        let mut logits = model.last_logits(&hidden)?;
        self.reset();
        self.load_kv_from_cache(&cache)?;
        drop(cache);

        let eos = model.config().eos_token_id;
        let mut all_tokens: Vec<u32> = prompt_tokens.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut rng = SplitMix64::new(opts.seed);
        let mut hit_eos = false;
        loop {
            let next = select_next_token(&mut logits, &all_tokens, opts, &mut rng)?;
            generated.push(next);
            all_tokens.push(next);
            if next == eos {
                hit_eos = true;
                break;
            }
            if generated.len() >= opts.max_new_tokens {
                break;
            }
            if generated.len().is_multiple_of(LOOP_CHECK_STRIDE) {
                if let Some(d) = detect_loop(&generated) {
                    return Ok(GenerateOutcome {
                        tokens: generated,
                        loop_detection: Some(d),
                        hit_eos: false,
                    });
                }
            }
            if self.cur_len + 1 > max_len {
                break;
            }
            self.step(next)?;
            logits = self.logits_host()?;
        }
        let loop_detection = detect_loop(&generated);
        Ok(GenerateOutcome {
            tokens: generated,
            loop_detection,
            hit_eos,
        })
    }
}

impl Drop for DsocrDecodeGraph {
    fn drop(&mut self) {
        let dbg = graph_debug();
        let ctx = self.forked.context().clone();
        let probe = |tag: &str| {
            if dbg {
                if let Err(e) = ctx.check_err() {
                    eprintln!("[dsocr-graph-drop] deferred ctx error after {tag}: {e:?}");
                }
            }
        };
        probe("entry");
        let ordinal = self.forked.context().ordinal();
        crate::gemma4_batch_graph::graph_teardown::GraphTeardown::new(&self.forked)
            .run(|| self.runner.invalidate());
        probe("graph-teardown");
        if dbg {
            graph_mem_report("post-drop", ordinal);
        }
    }
}
