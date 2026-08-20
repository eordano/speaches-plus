use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use super::linear;
use super::preprocess::resize_plane_f32;

fn prof_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("NV_SAM_PROF")
            .map(|v| v != "0")
            .unwrap_or(false)
    })
}

fn prof_table() -> &'static Mutex<Vec<(&'static str, f64)>> {
    static T: OnceLock<Mutex<Vec<(&'static str, f64)>>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(Vec::new()))
}

struct Span {
    label: &'static str,
    dev: Option<Device>,
    t0: Instant,
}

impl Span {
    fn new(label: &'static str, dev: &Device) -> Option<Self> {
        if !prof_on() {
            return None;
        }
        let _ = dev.synchronize();
        Some(Self {
            label,
            dev: Some(dev.clone()),
            t0: Instant::now(),
        })
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(d) = &self.dev {
            let _ = d.synchronize();
        }
        if let Ok(mut t) = prof_table().lock() {
            t.push((self.label, self.t0.elapsed().as_secs_f64()));
        }
    }
}

pub fn sam_prof_report(pages: f64) -> String {
    let Ok(mut t) = prof_table().lock() else {
        return String::new();
    };
    let mut agg: Vec<(&'static str, f64, usize)> = Vec::new();
    for (label, dt) in t.iter() {
        match agg.iter_mut().find(|(l, _, _)| l == label) {
            Some(e) => {
                e.1 += dt;
                e.2 += 1;
            }
            None => agg.push((label, *dt, 1)),
        }
    }
    agg.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let total: f64 = agg.iter().map(|(_, v, _)| v).sum();
    let mut out =
        String::from("| sam sub-stage | ms/page | calls/page | share |\n|---|---|---|---|\n");
    for (l, v, n) in &agg {
        out.push_str(&format!(
            "| {l} | {:.1} | {:.0} | {:.1}% |\n",
            v / pages * 1e3,
            *n as f64 / pages,
            v / total * 100.0
        ));
    }
    out.push_str(&format!("| SUM | {:.1} | | 100% |\n", total / pages * 1e3));
    t.clear();
    out
}

pub fn sam_prof_reset() {
    if let Ok(mut t) = prof_table().lock() {
        t.clear();
    }
}

#[cfg(feature = "cuda")]
mod fused {
    use std::sync::{Arc, Mutex, OnceLock};

    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::{CudaFunction, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use half::bf16;

    const SRC: &str = r#"
__device__ __forceinline__ float bf2f(unsigned short x) {
    return __uint_as_float(((unsigned int)x) << 16);
}
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int r = ((u >> 16) & 1u) + 0x7fffu;
    return (unsigned short)((u + r) >> 16);
}
extern "C" __global__ void sam_relpos_softmax_bf16(
    const unsigned short* __restrict__ scores,
    const unsigned short* __restrict__ rel_h,
    const unsigned short* __restrict__ rel_w,
    unsigned short* __restrict__ out,
    const int h, const int w, const int hw)
{
    extern __shared__ float sh[];
    __shared__ float redmax[32];
    __shared__ float redsum[32];
    __shared__ float bcast;

    const int tid = threadIdx.x;
    const int bs = blockDim.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const int nw = (bs + 31) >> 5;

    const long long row = (long long)blockIdx.x;
    const long long base = row * (long long)hw;
    const unsigned short* rh = rel_h + row * (long long)h;
    const unsigned short* rw = rel_w + row * (long long)w;

    float m = -3.4e38f;
    for (int c = tid; c < hw; c += bs) {
        int kh = c / w;
        int kw = c - kh * w;
        float v = bf2f(scores[base + c]) + bf2f(rh[kh]) + bf2f(rw[kw]);
        sh[c] = v;
        m = fmaxf(m, v);
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off, 32));
    if (lane == 0) redmax[wid] = m;
    __syncthreads();
    if (wid == 0) {
        m = (lane < nw) ? redmax[lane] : -3.4e38f;
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off, 32));
        if (lane == 0) bcast = m;
    }
    __syncthreads();
    m = bcast;
    __syncthreads();

    float s = 0.0f;
    for (int c = tid; c < hw; c += bs) {
        float e = expf(sh[c] - m);
        sh[c] = e;
        s += e;
    }
    for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, 32);
    if (lane == 0) redsum[wid] = s;
    __syncthreads();
    if (wid == 0) {
        s = (lane < nw) ? redsum[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off, 32);
        if (lane == 0) bcast = s;
    }
    __syncthreads();
    const float inv = 1.0f / bcast;

    for (int c = tid; c < hw; c += bs) {
        out[base + c] = f2bf(sh[c] * inv);
    }
}
"#;

    fn kernel(dev: &candle_core::CudaDevice) -> Result<CudaFunction> {
        static F: OnceLock<Mutex<Option<CudaFunction>>> = OnceLock::new();
        let cell = F.get_or_init(|| Mutex::new(None));
        let mut slot = cell
            .lock()
            .map_err(|e| anyhow::anyhow!("sam fused kernel cache poisoned: {e}"))?;
        if let Some(f) = slot.as_ref() {
            return Ok(f.clone());
        }
        let ptx = cudarc::nvrtc::compile_ptx(SRC).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let ctx: Arc<cudarc::driver::CudaContext> = dev.cuda_stream().context().clone();
        let module = ctx.load_module(ptx).map_err(|e| anyhow::anyhow!(e))?;
        let f = module
            .load_function("sam_relpos_softmax_bf16")
            .map_err(|e| anyhow::anyhow!(e))?;
        *slot = Some(f.clone());
        Ok(f)
    }

    fn ptr(t: &Tensor, stream: &Arc<cudarc::driver::CudaStream>) -> Result<u64> {
        let (st, l) = t.storage_and_layout();
        anyhow::ensure!(l.is_contiguous(), "sam fused: tensor not contiguous");
        let cuda = match &*st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("sam fused: tensor not on cuda"),
        };
        let slice = cuda.as_cuda_slice::<bf16>()?;
        let view = slice.slice(l.start_offset()..l.start_offset() + t.elem_count());
        let (p, _g) = view.device_ptr(stream);
        Ok(p)
    }

    fn enabled() -> bool {
        std::env::var("NV_SAM_FUSED")
            .map(|v| v != "0")
            .unwrap_or(true)
    }

    pub fn supported(scores: &Tensor, rel_h: &Tensor, rel_w: &Tensor) -> bool {
        enabled()
            && matches!(scores.device(), Device::Cuda(_))
            && scores.dtype() == DType::BF16
            && rel_h.dtype() == DType::BF16
            && rel_w.dtype() == DType::BF16
            && scores.is_contiguous()
            && rel_h.is_contiguous()
            && rel_w.is_contiguous()
    }

    pub fn relpos_softmax(
        scores: &Tensor,
        rel_h: &Tensor,
        rel_w: &Tensor,
        h: usize,
        w: usize,
    ) -> Result<Tensor> {
        let dev = match scores.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("sam fused: cuda device required"),
        };
        let hw = h * w;
        let n = scores.elem_count();
        anyhow::ensure!(
            n % hw == 0,
            "sam fused: scores {n} not divisible by hw {hw}"
        );
        let rows = n / hw;
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let f = kernel(&dev)?;
        let s_ptr = ptr(scores, &stream)?;
        let h_ptr = ptr(rel_h, &stream)?;
        let w_ptr = ptr(rel_w, &stream)?;
        let mut out = unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
        let block: u32 = if hw >= 512 { 256 } else { 128 };
        let cfg = LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: (hw * 4) as u32,
        };
        {
            let (o_ptr, _g) = out.device_ptr_mut(&stream);
            let hi = h as i32;
            let wi = w as i32;
            let hwi = hw as i32;
            let mut lb = stream.launch_builder(&f);
            lb.arg(&s_ptr)
                .arg(&h_ptr)
                .arg(&w_ptr)
                .arg(&o_ptr)
                .arg(&hi)
                .arg(&wi)
                .arg(&hwi);
            unsafe { lb.launch(cfg) }.map_err(|e| anyhow::anyhow!(e))?;
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            scores.shape().clone(),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[cfg(feature = "cuda")]
mod flash {
    use std::sync::{Arc, Mutex, OnceLock};

    use anyhow::Result;
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::{CudaFunction, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
    use half::bf16;

    const SRC: &str = r#"
__device__ __forceinline__ float bf2f(unsigned short x) {
    return __uint_as_float(((unsigned int)x) << 16);
}
__device__ __forceinline__ unsigned short f2bf(float f) {
    unsigned int u = __float_as_uint(f);
    unsigned int r = ((u >> 16) & 1u) + 0x7fffu;
    return (unsigned short)((u + r) >> 16);
}
extern "C" __global__ void sam_flash_relpos_bf16(
    const unsigned short* __restrict__ q,
    const unsigned short* __restrict__ k,
    const unsigned short* __restrict__ v,
    const unsigned short* __restrict__ rel_h,
    const unsigned short* __restrict__ rel_w,
    unsigned short* __restrict__ out,
    const int h, const int w, const int hd, const float scale)
{
    const int hw = h * w;
    const int qi = blockIdx.x;
    const long long bn = (long long)blockIdx.y;
    const long long row = bn * (long long)hw + (long long)qi;
    const unsigned short* qp = q + row * (long long)hd;
    const unsigned short* kb = k + bn * (long long)hw * (long long)hd;
    const unsigned short* vb = v + bn * (long long)hw * (long long)hd;
    const unsigned short* rh = rel_h + row * (long long)h;
    const unsigned short* rw = rel_w + row * (long long)w;

    __shared__ float q_sh[128];
    __shared__ float acc[128];
    __shared__ float p_sh[64];
    __shared__ unsigned short kv_sh[64 * 128];
    __shared__ float red[2];
    __shared__ float m_sh, s_sh;

    const int tid = threadIdx.x;
    for (int d = tid; d < hd; d += 64) { q_sh[d] = bf2f(qp[d]); acc[d] = 0.0f; }
    if (tid == 0) { m_sh = -3.4e38f; s_sh = 0.0f; }
    __syncthreads();

    for (int t0 = 0; t0 < hw; t0 += 64) {
        const int tlen = min(64, hw - t0);
        for (int i = tid; i < tlen * hd; i += 64) kv_sh[i] = kb[(long long)t0 * hd + i];
        __syncthreads();
        float sc = -3.4e38f;
        if (tid < tlen) {
            const int c = t0 + tid;
            float dot = 0.0f;
            for (int d = 0; d < hd; ++d) dot += q_sh[d] * bf2f(kv_sh[tid * hd + d]);
            const int kh = c / w;
            const int kw = c - kh * w;
            sc = dot * scale + bf2f(rh[kh]) + bf2f(rw[kw]);
        }
        float mv = sc;
        for (int off = 16; off > 0; off >>= 1) mv = fmaxf(mv, __shfl_xor_sync(0xffffffffu, mv, off, 32));
        if ((tid & 31) == 0) red[tid >> 5] = mv;
        __syncthreads();
        const float m_old = m_sh;
        const float m_new = fmaxf(m_old, fmaxf(red[0], red[1]));
        const float alpha = expf(m_old - m_new);
        const float p = (tid < tlen) ? expf(sc - m_new) : 0.0f;
        p_sh[tid] = p;
        __syncthreads();
        float ps = p;
        for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off, 32);
        if ((tid & 31) == 0) red[tid >> 5] = ps;
        __syncthreads();
        if (tid == 0) { s_sh = s_sh * alpha + red[0] + red[1]; m_sh = m_new; }
        __syncthreads();
        for (int i = tid; i < tlen * hd; i += 64) kv_sh[i] = vb[(long long)t0 * hd + i];
        __syncthreads();
        for (int d = tid; d < hd; d += 64) {
            float a = acc[d] * alpha;
            for (int c = 0; c < tlen; ++c) a += p_sh[c] * bf2f(kv_sh[c * hd + d]);
            acc[d] = a;
        }
        __syncthreads();
    }
    const float inv = 1.0f / s_sh;
    for (int d = tid; d < hd; d += 64) out[row * (long long)hd + d] = f2bf(acc[d] * inv);
}
"#;

    fn kernel(dev: &candle_core::CudaDevice) -> Result<CudaFunction> {
        static F: OnceLock<Mutex<Option<CudaFunction>>> = OnceLock::new();
        let cell = F.get_or_init(|| Mutex::new(None));
        let mut slot = cell
            .lock()
            .map_err(|e| anyhow::anyhow!("sam flash kernel cache poisoned: {e}"))?;
        if let Some(f) = slot.as_ref() {
            return Ok(f.clone());
        }
        let ptx = cudarc::nvrtc::compile_ptx(SRC).map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let ctx: Arc<cudarc::driver::CudaContext> = dev.cuda_stream().context().clone();
        let module = ctx.load_module(ptx).map_err(|e| anyhow::anyhow!(e))?;
        let f = module
            .load_function("sam_flash_relpos_bf16")
            .map_err(|e| anyhow::anyhow!(e))?;
        *slot = Some(f.clone());
        Ok(f)
    }

    fn ptr(t: &Tensor, stream: &Arc<cudarc::driver::CudaStream>) -> Result<u64> {
        let (st, l) = t.storage_and_layout();
        anyhow::ensure!(l.is_contiguous(), "sam flash: tensor not contiguous");
        let cuda = match &*st {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("sam flash: tensor not on cuda"),
        };
        let slice = cuda.as_cuda_slice::<bf16>()?;
        let view = slice.slice(l.start_offset()..l.start_offset() + t.elem_count());
        let (p, _g) = view.device_ptr(stream);
        Ok(p)
    }

    pub fn enabled() -> bool {
        std::env::var("NV_SAM_FLASH")
            .map(|v| v != "0" && !v.is_empty())
            .unwrap_or(false)
    }

    pub fn supported(q: &Tensor, k: &Tensor, v: &Tensor, rel_h: &Tensor, rel_w: &Tensor) -> bool {
        let bf = |t: &Tensor| t.dtype() == DType::BF16 && t.is_contiguous();
        matches!(q.device(), Device::Cuda(_))
            && bf(q)
            && bf(k)
            && bf(v)
            && bf(rel_h)
            && bf(rel_w)
            && q.dims3().map(|(_, _, hd)| hd <= 128).unwrap_or(false)
    }

    pub fn attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        rel_h: &Tensor,
        rel_w: &Tensor,
        h: usize,
        w: usize,
        scale: f64,
    ) -> Result<Tensor> {
        let dev = match q.device() {
            Device::Cuda(d) => d.clone(),
            _ => anyhow::bail!("sam flash: cuda device required"),
        };
        let (b_nh, hw, hd) = q.dims3()?;
        anyhow::ensure!(hw == h * w, "sam flash: hw {hw} != h*w {}", h * w);
        anyhow::ensure!(hd <= 128, "sam flash: head_dim {hd} > 128");
        anyhow::ensure!(b_nh <= 65535, "sam flash: b_nh {b_nh} exceeds grid.y");
        anyhow::ensure!(k.dims3()? == (b_nh, hw, hd), "sam flash: k shape mismatch");
        anyhow::ensure!(v.dims3()? == (b_nh, hw, hd), "sam flash: v shape mismatch");
        anyhow::ensure!(
            rel_h.elem_count() == b_nh * hw * h,
            "sam flash: rel_h size mismatch"
        );
        anyhow::ensure!(
            rel_w.elem_count() == b_nh * hw * w,
            "sam flash: rel_w size mismatch"
        );
        let stream = nv_layers::cuda_stream::current_stream(&dev);
        let f = kernel(&dev)?;
        let q_ptr = ptr(q, &stream)?;
        let k_ptr = ptr(k, &stream)?;
        let v_ptr = ptr(v, &stream)?;
        let rh_ptr = ptr(rel_h, &stream)?;
        let rw_ptr = ptr(rel_w, &stream)?;
        let n = b_nh * hw * hd;
        let mut out = unsafe { stream.alloc::<bf16>(n).map_err(|e| anyhow::anyhow!(e))? };
        let cfg = LaunchConfig {
            grid_dim: (hw as u32, b_nh as u32, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let (o_ptr, _g) = out.device_ptr_mut(&stream);
            let hi = h as i32;
            let wi = w as i32;
            let hdi = hd as i32;
            let sc = scale as f32;
            let mut lb = stream.launch_builder(&f);
            lb.arg(&q_ptr)
                .arg(&k_ptr)
                .arg(&v_ptr)
                .arg(&rh_ptr)
                .arg(&rw_ptr)
                .arg(&o_ptr)
                .arg(&hi)
                .arg(&wi)
                .arg(&hdi)
                .arg(&sc);
            unsafe { lb.launch(cfg) }.map_err(|e| anyhow::anyhow!(e))?;
        }
        let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev);
        Ok(Tensor::from_storage(
            candle_core::Storage::Cuda(storage),
            (b_nh, hw, hd),
            candle_core::op::BackpropOp::none(),
            false,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct SamConfig {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub mlp_ratio: usize,
    pub patch_size: usize,
    pub window_size: usize,
    pub global_attn_indexes: Vec<usize>,
    pub pos_grid: usize,
    pub ln_eps: f64,
}

impl SamConfig {
    pub fn vit_b() -> Self {
        Self {
            embed_dim: 768,
            depth: 12,
            num_heads: 12,
            mlp_ratio: 4,
            patch_size: 16,
            window_size: 14,
            global_attn_indexes: vec![2, 5, 8, 11],
            pos_grid: 64,
            ln_eps: 1e-6,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
}

fn fused_norm_ok(x: &Tensor) -> bool {
    !x.device().is_cpu()
        && x.is_contiguous()
        && matches!(x.dtype(), DType::F32 | DType::F16 | DType::BF16)
}

fn softmax_rows(scores: &Tensor, out_dtype: DType) -> Result<Tensor> {
    if scores.device().is_cpu() {
        return Ok(
            candle_nn::ops::softmax(&scores.to_dtype(DType::F32)?, D::Minus1)?
                .to_dtype(out_dtype)?,
        );
    }
    Ok(candle_nn::ops::softmax_last_dim(&scores.contiguous()?)?.to_dtype(out_dtype)?)
}

pub fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor> {
    if fused_norm_ok(x) && w.dtype() == x.dtype() && b.dtype() == x.dtype() {
        return Ok(candle_nn::ops::layer_norm(x, w, b, eps as f32)?);
    }
    let dtype = x.dtype();
    let x32 = x.to_dtype(DType::F32)?;
    let mu = x32.mean_keepdim(D::Minus1)?;
    let xc = x32.broadcast_sub(&mu)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let normed = xc.broadcast_div(&(var + eps)?.sqrt()?)?;
    let out = normed
        .broadcast_mul(&w.to_dtype(DType::F32)?)?
        .broadcast_add(&b.to_dtype(DType::F32)?)?;
    Ok(out.to_dtype(dtype)?)
}

pub fn window_partition(x: &Tensor, ws: usize) -> Result<(Tensor, (usize, usize))> {
    let (b, h, w, c) = x.dims4()?;
    let pad_h = (ws - h % ws) % ws;
    let pad_w = (ws - w % ws) % ws;
    let mut x = x.clone();
    if pad_h > 0 {
        x = x.pad_with_zeros(1, 0, pad_h)?;
    }
    if pad_w > 0 {
        x = x.pad_with_zeros(2, 0, pad_w)?;
    }
    let (hp, wp) = (h + pad_h, w + pad_w);
    let x = x
        .reshape((b, hp / ws, ws, wp / ws, ws, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * (hp / ws) * (wp / ws), ws, ws, c))?;
    Ok((x, (hp, wp)))
}

pub fn window_unpartition(
    windows: &Tensor,
    ws: usize,
    pad_hw: (usize, usize),
    hw: (usize, usize),
) -> Result<Tensor> {
    let (hp, wp) = pad_hw;
    let (h, w) = hw;
    let nb = windows.dim(0)?;
    let c = windows.dim(3)?;
    let b = nb / ((hp / ws) * (wp / ws));
    let x = windows
        .reshape((b, hp / ws, wp / ws, ws, ws, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b, hp, wp, c))?;
    let x = if hp > h || wp > w {
        x.i((.., ..h, ..w, ..))?.contiguous()?
    } else {
        x
    };
    Ok(x)
}

fn interp_rows_linear(rows: &[Vec<f32>], out_len: usize) -> Vec<Vec<f32>> {
    let in_len = rows.len();
    let dim = rows[0].len();
    let scale = in_len as f64 / out_len as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let x = ((i as f64 + 0.5) * scale - 0.5).max(0.0);
        let x0 = (x.floor() as usize).min(in_len - 1);
        let x1 = (x0 + 1).min(in_len - 1);
        let f = (x - x0 as f64) as f32;
        let mut row = vec![0f32; dim];
        for d in 0..dim {
            row[d] = rows[x0][d] * (1.0 - f) + rows[x1][d] * f;
        }
        out.push(row);
    }
    out
}

fn resolve_rel_pos(table: &Tensor, q_size: usize, k_size: usize) -> Result<Tensor> {
    let max_rel = 2 * q_size.max(k_size) - 1;
    let (rows, dim) = table.dims2()?;
    let device = table.device().clone();
    let dtype = table.dtype();
    let resized = if rows != max_rel {
        let host: Vec<f32> = table
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        let row_vecs: Vec<Vec<f32>> = host.chunks(dim).map(|c| c.to_vec()).collect();
        let out = interp_rows_linear(&row_vecs, max_rel);
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        Tensor::from_vec(flat, (max_rel, dim), &Device::Cpu)?
            .to_device(&device)?
            .to_dtype(dtype)?
    } else {
        table.clone()
    };
    let qr = (k_size as f64 / q_size as f64).max(1.0);
    let kr = (q_size as f64 / k_size as f64).max(1.0);
    let mut idx = Vec::with_capacity(q_size * k_size);
    for qi in 0..q_size {
        for ki in 0..k_size {
            let rel = qi as f64 * qr - ki as f64 * kr + (k_size as f64 - 1.0) * kr;
            idx.push(rel as u32);
        }
    }
    let idx_t = Tensor::from_vec(idx, q_size * k_size, &device)?;
    Ok(resized
        .index_select(&idx_t, 0)?
        .reshape((q_size, k_size, dim))?)
}

pub struct SamAttention {
    qkv_w: Tensor,
    qkv_b: Tensor,
    proj_w: Tensor,
    proj_b: Tensor,
    rel_pos_h: Option<Tensor>,
    rel_pos_w: Option<Tensor>,
    num_heads: usize,
    rel_cache: Mutex<HashMap<(usize, usize), (Tensor, Tensor)>>,
}

const SAM_REL_CACHE_MAX_DISTINCT_GRIDS_PER_LAYER_8_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_SAM_RUNS_EAGER_AND_INFLIGHT_CLONES_KEEP_TENSORS_ALIVE:
    usize = 8;

impl SamAttention {
    fn rel_tables(&self, h: usize, w: usize) -> Result<Option<(Tensor, Tensor)>> {
        let (rh, rw) = match (&self.rel_pos_h, &self.rel_pos_w) {
            (Some(a), Some(b)) => (a, b),
            _ => return Ok(None),
        };
        let mut cache = self
            .rel_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("sam rel-pos cache poisoned: {e}"))?;
        if let Some(v) = cache.get(&(h, w)) {
            return Ok(Some(v.clone()));
        }
        let rh_t = resolve_rel_pos(rh, h, h)?.transpose(1, 2)?.contiguous()?;
        let rw_t = resolve_rel_pos(rw, w, w)?.transpose(1, 2)?.contiguous()?;
        if cache.len()
            >= SAM_REL_CACHE_MAX_DISTINCT_GRIDS_PER_LAYER_8_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_SAM_RUNS_EAGER_AND_INFLIGHT_CLONES_KEEP_TENSORS_ALIVE
        {
            cache.clear();
        }
        cache.insert((h, w), (rh_t.clone(), rw_t.clone()));
        Ok(Some((rh_t, rw_t)))
    }

    fn rel_terms(&self, q: &Tensor, h: usize, w: usize) -> Result<Option<(Tensor, Tensor)>> {
        let (b_nh, _hw, hd) = q.dims3()?;
        let dev = q.device().clone();
        let sp = Span::new("attn.relpos_resolve", &dev);
        let tables = self.rel_tables(h, w)?;
        drop(sp);
        let Some((rh_t, rw_t)) = tables else {
            return Ok(None);
        };
        let _sp = Span::new("attn.relbias_mm", &dev);
        let r_q = q.reshape((b_nh, h, w, hd))?;
        let rel_h = r_q
            .permute((1, 0, 2, 3))?
            .contiguous()?
            .reshape((h, b_nh * w, hd))?
            .matmul(&rh_t)?
            .reshape((h, b_nh, w, h))?
            .permute((1, 0, 2, 3))?
            .contiguous()?;
        let rel_w = r_q
            .permute((2, 0, 1, 3))?
            .contiguous()?
            .reshape((w, b_nh * h, hd))?
            .matmul(&rw_t)?
            .reshape((w, b_nh, h, w))?
            .permute((1, 2, 0, 3))?
            .contiguous()?;
        Ok(Some((rel_h, rel_w)))
    }

    fn bias_softmax(
        &self,
        scores: Tensor,
        rel: Option<(Tensor, Tensor)>,
        h: usize,
        w: usize,
        out_dtype: DType,
    ) -> Result<Tensor> {
        let dev = scores.device().clone();
        let Some((rel_h, rel_w)) = rel else {
            let _sp = Span::new("attn.softmax", &dev);
            return softmax_rows(&scores, out_dtype);
        };
        #[cfg(feature = "cuda")]
        if fused::supported(&scores, &rel_h, &rel_w) {
            let _sp = Span::new("attn.biassoftmax", &dev);
            return fused::relpos_softmax(&scores, &rel_h, &rel_w, h, w);
        }
        let b_nh = scores.dim(0)?;
        let sp = Span::new("attn.biasadd", &dev);
        let scores = scores
            .reshape((b_nh, h, w, h, w))?
            .broadcast_add(&rel_h.unsqueeze(4)?)?
            .broadcast_add(&rel_w.unsqueeze(3)?)?
            .reshape((b_nh, h * w, h * w))?;
        drop(sp);
        let _sp = Span::new("attn.softmax", &dev);
        softmax_rows(&scores, out_dtype)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, w, c) = x.dims4()?;
        let hw = h * w;
        let nh = self.num_heads;
        let hd = c / nh;
        let dev = x.device().clone();
        let sp = Span::new("attn.qkv", &dev);
        let qkv = linear(&x.reshape((b, hw, c))?, &self.qkv_w, Some(&self.qkv_b))?;
        let qkv = qkv
            .reshape((b, hw, 3, nh, hd))?
            .permute((2, 0, 3, 1, 4))?
            .contiguous()?
            .reshape((3, b * nh, hw, hd))?;
        let q = qkv.i(0)?.contiguous()?;
        let k = qkv.i(1)?.contiguous()?;
        let v = qkv.i(2)?.contiguous()?;
        drop(sp);
        let scale = (hd as f64).powf(-0.5);
        let sp = Span::new("attn.relbias", &dev);
        let rel = self.rel_terms(&q, h, w)?;
        drop(sp);
        #[cfg(feature = "cuda")]
        if let Some((rel_h, rel_w)) = rel.as_ref() {
            if flash::enabled() && flash::supported(&q, &k, &v, rel_h, rel_w) {
                let sp = Span::new("attn.flash", &dev);
                let out = flash::attention(&q, &k, &v, rel_h, rel_w, h, w, scale)?;
                let out = out
                    .reshape((b, nh, h, w, hd))?
                    .permute((0, 2, 3, 1, 4))?
                    .contiguous()?
                    .reshape((b, h, w, c))?;
                drop(sp);
                let _sp = Span::new("attn.proj", &dev);
                return linear(&out, &self.proj_w, Some(&self.proj_b));
            }
        }
        let sp = Span::new("attn.qk", &dev);
        let scores = (q * scale)?.matmul(&k.transpose(1, 2)?)?;
        drop(sp);
        let probs = self.bias_softmax(scores, rel, h, w, x.dtype())?;
        let sp = Span::new("attn.av", &dev);
        let out = probs.matmul(&v)?;
        let out = out
            .reshape((b, nh, h, w, hd))?
            .permute((0, 2, 3, 1, 4))?
            .contiguous()?
            .reshape((b, h, w, c))?;
        drop(sp);
        let _sp = Span::new("attn.proj", &dev);
        linear(&out, &self.proj_w, Some(&self.proj_b))
    }
}

pub struct SamBlock {
    norm1_w: Tensor,
    norm1_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    attn: SamAttention,
    lin1_w: Tensor,
    lin1_b: Tensor,
    lin2_w: Tensor,
    lin2_b: Tensor,
    window_size: usize,
    ln_eps: f64,
}

impl SamBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dev = x.device().clone();
        let shortcut = x.clone();
        let sp = Span::new("block.ln", &dev);
        let mut y = layer_norm(x, &self.norm1_w, &self.norm1_b, self.ln_eps)?;
        drop(sp);
        let (h, w) = (y.dim(1)?, y.dim(2)?);
        let attn_out = if self.window_size > 0 {
            let sp = Span::new("block.window", &dev);
            let (windows, pad_hw) = window_partition(&y, self.window_size)?;
            drop(sp);
            let attended = self.attn.forward(&windows)?;
            let _sp = Span::new("block.window", &dev);
            window_unpartition(&attended, self.window_size, pad_hw, (h, w))?
        } else {
            self.attn.forward(&y)?
        };
        y = (shortcut + attn_out)?;
        let sp = Span::new("block.ln", &dev);
        let normed = layer_norm(&y, &self.norm2_w, &self.norm2_b, self.ln_eps)?;
        drop(sp);
        let _sp = Span::new("block.mlp", &dev);
        let mlp = linear(&normed, &self.lin1_w, Some(&self.lin1_b))?.gelu_erf()?;
        let mlp = linear(&mlp, &self.lin2_w, Some(&self.lin2_b))?;
        Ok((y + mlp)?)
    }
}

pub struct SamEncoder {
    cfg: SamConfig,
    patch_w: Tensor,
    patch_b: Tensor,
    pos_embed: Tensor,
    pos_host: Vec<f32>,
    blocks: Vec<SamBlock>,
    pos_cache: Mutex<HashMap<usize, Tensor>>,
}

const SAM_POS_CACHE_MAX_DISTINCT_GRIDS_4_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_SAM_RUNS_EAGER_AND_INFLIGHT_CLONES_KEEP_TENSORS_ALIVE:
    usize = 4;

impl SamEncoder {
    pub fn from_loader(
        weights: &dyn nv_weights::TensorSource,
        prefix: &str,
        cfg: SamConfig,
        dtype: DType,
    ) -> Result<Self> {
        let g = |name: &str| -> Result<Tensor> {
            weights
                .get(&format!("{prefix}{name}"), dtype)
                .with_context(|| format!("load {prefix}{name}"))
        };
        let patch_w = g("patch_embed.proj.weight")?;
        let patch_b = g("patch_embed.proj.bias")?;
        let pos_embed = g("pos_embed")?;
        let pos_host: Vec<f32> = pos_embed
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .flatten_all()?
            .to_vec1()?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let is_global = cfg.global_attn_indexes.contains(&i);
            let p = format!("blocks.{i}.");
            let attn = SamAttention {
                qkv_w: g(&format!("{p}attn.qkv.weight"))?,
                qkv_b: g(&format!("{p}attn.qkv.bias"))?,
                proj_w: g(&format!("{p}attn.proj.weight"))?,
                proj_b: g(&format!("{p}attn.proj.bias"))?,
                rel_pos_h: Some(g(&format!("{p}attn.rel_pos_h"))?),
                rel_pos_w: Some(g(&format!("{p}attn.rel_pos_w"))?),
                num_heads: cfg.num_heads,
                rel_cache: Mutex::new(HashMap::new()),
            };
            blocks.push(SamBlock {
                norm1_w: g(&format!("{p}norm1.weight"))?,
                norm1_b: g(&format!("{p}norm1.bias"))?,
                norm2_w: g(&format!("{p}norm2.weight"))?,
                norm2_b: g(&format!("{p}norm2.bias"))?,
                attn,
                lin1_w: g(&format!("{p}mlp.lin1.weight"))?,
                lin1_b: g(&format!("{p}mlp.lin1.bias"))?,
                lin2_w: g(&format!("{p}mlp.lin2.weight"))?,
                lin2_b: g(&format!("{p}mlp.lin2.bias"))?,
                window_size: if is_global { 0 } else { cfg.window_size },
                ln_eps: cfg.ln_eps,
            });
        }
        Ok(Self {
            cfg,
            patch_w,
            patch_b,
            pos_embed,
            pos_host,
            blocks,
            pos_cache: Mutex::new(HashMap::new()),
        })
    }

    fn abs_pos(&self, grid: usize) -> Result<Tensor> {
        let src = self.cfg.pos_grid;
        if grid == src {
            return Ok(self.pos_embed.clone());
        }
        let mut cache = self
            .pos_cache
            .lock()
            .map_err(|e| anyhow::anyhow!("sam pos-embed cache poisoned: {e}"))?;
        if let Some(t) = cache.get(&grid) {
            return Ok(t.clone());
        }
        let dim = self.cfg.embed_dim;
        let mut out = vec![0f32; grid * grid * dim];
        let mut plane = vec![0f32; src * src];
        for d in 0..dim {
            for i in 0..src * src {
                plane[i] = self.pos_host[i * dim + d];
            }
            let resized = resize_plane_f32(&plane, src, src, grid, grid);
            for i in 0..grid * grid {
                out[i * dim + d] = resized[i];
            }
        }
        let t = Tensor::from_vec(out, (1, grid, grid, dim), &Device::Cpu)?
            .to_device(self.pos_embed.device())?
            .to_dtype(self.pos_embed.dtype())?;
        if cache.len()
            >= SAM_POS_CACHE_MAX_DISTINCT_GRIDS_4_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_SAM_RUNS_EAGER_AND_INFLIGHT_CLONES_KEEP_TENSORS_ALIVE
        {
            cache.clear();
        }
        cache.insert(grid, t.clone());
        Ok(t)
    }

    pub fn forward(&self, pixels: &Tensor) -> Result<Tensor> {
        let pixels = if pixels.dtype() != self.patch_w.dtype() {
            pixels.to_dtype(self.patch_w.dtype())?
        } else {
            pixels.clone()
        };
        let dev = pixels.device().clone();
        let sp = Span::new("patch_embed", &dev);
        let x = pixels.conv2d(&self.patch_w, 0, self.cfg.patch_size, 1, 1)?;
        let x = x.broadcast_add(&self.patch_b.reshape((1, self.cfg.embed_dim, 1, 1))?)?;
        let x = x.permute((0, 2, 3, 1))?.contiguous()?;
        drop(sp);
        let grid = x.dim(1)?;
        let sp = Span::new("abs_pos", &dev);
        let pos = self.abs_pos(grid)?;
        let mut x = x.broadcast_add(&pos)?;
        drop(sp);
        for blk in &self.blocks {
            x = blk.forward(&x)?;
        }
        Ok(x)
    }

    pub fn config(&self) -> &SamConfig {
        &self.cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_partition_roundtrip_with_padding() {
        let dev = Device::Cpu;
        let x = Tensor::arange(0f32, (2 * 5 * 7 * 3) as f32, &dev)
            .unwrap()
            .reshape((2, 5, 7, 3))
            .unwrap();
        let (win, pad_hw) = window_partition(&x, 3).unwrap();
        assert_eq!(pad_hw, (6, 9));
        assert_eq!(win.dims(), &[2 * 2 * 3, 3, 3, 3]);
        let back = window_unpartition(&win, 3, pad_hw, (5, 7)).unwrap();
        let orig: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let round: Vec<f32> = back.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, round);
    }

    #[test]
    fn rel_pos_exact_when_size_matches() {
        let dev = Device::Cpu;
        let table = Tensor::arange(0f32, (7 * 2) as f32, &dev)
            .unwrap()
            .reshape((7, 2))
            .unwrap();
        let sel = resolve_rel_pos(&table, 4, 4).unwrap();
        assert_eq!(sel.dims(), &[4, 4, 2]);
        let v: Vec<f32> = sel.i((0, 0, ..)).unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![6.0, 7.0]);
        let v: Vec<f32> = sel.i((3, 0, ..)).unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![12.0, 13.0]);
        let v: Vec<f32> = sel.i((0, 3, ..)).unwrap().to_vec1().unwrap();
        assert_eq!(v, vec![0.0, 1.0]);
    }

    #[test]
    fn rel_pos_interpolates_when_rows_differ() {
        let dev = Device::Cpu;
        let table = Tensor::arange(0f32, 127f32, &dev)
            .unwrap()
            .reshape((127, 1))
            .unwrap();
        let sel = resolve_rel_pos(&table, 48, 48).unwrap();
        assert_eq!(sel.dims(), &[48, 48, 1]);
        let center: Vec<f32> = sel.i((0, 0, ..)).unwrap().to_vec1().unwrap();
        assert!((center[0] - 63.0).abs() < 1e-3, "{}", center[0]);
    }

    #[test]
    fn rel_cache_stays_within_its_grid_budget_under_many_distinct_resolutions() {
        let dev = Device::Cpu;
        let table = Tensor::arange(0f32, (7 * 2) as f32, &dev)
            .unwrap()
            .reshape((7, 2))
            .unwrap();
        let attn = SamAttention {
            qkv_w: Tensor::zeros((6, 2), DType::F32, &dev).unwrap(),
            qkv_b: Tensor::zeros(6, DType::F32, &dev).unwrap(),
            proj_w: Tensor::zeros((2, 2), DType::F32, &dev).unwrap(),
            proj_b: Tensor::zeros(2, DType::F32, &dev).unwrap(),
            rel_pos_h: Some(table.clone()),
            rel_pos_w: Some(table),
            num_heads: 1,
            rel_cache: Mutex::new(HashMap::new()),
        };
        for g in 2..=24usize {
            assert!(attn.rel_tables(g, g).unwrap().is_some());
        }
        let len = attn.rel_cache.lock().unwrap().len();
        assert!(
            len <= SAM_REL_CACHE_MAX_DISTINCT_GRIDS_PER_LAYER_8_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_SAM_RUNS_EAGER_AND_INFLIGHT_CLONES_KEEP_TENSORS_ALIVE,
            "a stream of distinct attention grids must not grow the per-layer rel cache past \
             its budget: len={len}"
        );
        assert!(attn.rel_tables(24, 24).unwrap().is_some());
    }

    #[test]
    fn pos_cache_stays_within_its_grid_budget_under_many_distinct_resolutions() {
        let dev = Device::Cpu;
        let cfg = SamConfig {
            embed_dim: 2,
            depth: 0,
            num_heads: 1,
            mlp_ratio: 1,
            patch_size: 16,
            window_size: 14,
            global_attn_indexes: vec![],
            pos_grid: 4,
            ln_eps: 1e-6,
        };
        let enc = SamEncoder {
            cfg,
            patch_w: Tensor::zeros((2, 3, 16, 16), DType::F32, &dev).unwrap(),
            patch_b: Tensor::zeros(2, DType::F32, &dev).unwrap(),
            pos_embed: Tensor::zeros((1, 4, 4, 2), DType::F32, &dev).unwrap(),
            pos_host: vec![0f32; 4 * 4 * 2],
            blocks: vec![],
            pos_cache: Mutex::new(HashMap::new()),
        };
        for grid in [2usize, 3, 5, 6, 7, 8, 9, 10] {
            enc.abs_pos(grid).unwrap();
        }
        let len = enc.pos_cache.lock().unwrap().len();
        assert!(
            len <= SAM_POS_CACHE_MAX_DISTINCT_GRIDS_4_CLEAR_ON_OVERFLOW_IS_SAFE_BECAUSE_SAM_RUNS_EAGER_AND_INFLIGHT_CLONES_KEEP_TENSORS_ALIVE,
            "a stream of distinct pos-embed grids must not grow the cache past its budget: \
             len={len}"
        );
        assert_eq!(enc.abs_pos(10).unwrap().dims(), &[1, 10, 10, 2]);
    }

    #[test]
    fn layer_norm_normalizes_last_dim() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![1f32, 2.0, 3.0, 4.0], (1, 4), &dev).unwrap();
        let w = Tensor::ones(4, DType::F32, &dev).unwrap();
        let b = Tensor::zeros(4, DType::F32, &dev).unwrap();
        let y = layer_norm(&x, &w, &b, 1e-6).unwrap();
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        let mean: f32 = v.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5);
        assert!((v[3] - 1.3416).abs() < 1e-3);
    }
}
