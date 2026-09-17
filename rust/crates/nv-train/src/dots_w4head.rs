use anyhow::{ensure, Context, Result};
use candle_core::backend::BackendStorage;
use candle_core::{DType, Device, Layout, MetalStorage, Shape, Tensor};
use candle_metal_kernels::metal::{Buffer, ComputePipeline};
use objc2_metal::MTLSize;
use std::sync::{Arc, Mutex, OnceLock};

pub const W4HEAD_FUSES_THE_HEAD_CE_AGAINST_THE_CHECKPOINTS_OWN_W4_BYTES_AND_IS_TOLERANCE_CLASS:
    &str =
    "NV_DOTS_QLORA_W4HEAD=1 replaces the fp32 lm_head resident (933 MB) and the candle op chain \
     of the chunked head CE (GEMM, f32 cast, log_softmax fwd+bwd over 151936-wide logits, gather \
     fwd+scatter bwd, dx GEMM) with five MSL kernels riding the quantized checkpoint bytes \
     directly (146 MB resident: u4 words + f32 scales/biases): a dequant-in-kernel simdgroup GEMM \
     for the logits, one row-stats kernel producing lse and the weighted CE rows, an in-place \
     rewrite of the stashed logits into dlogits = w*g*(softmax - onehot) so the backward never \
     allocates a second vocab-wide buffer, a pure dX GEMM over dlogits reading the same quantized \
     words k-major, and a fixed-order n-split reduce. The dequant is the same q*scale+bias f32 \
     fma the resident dequant applies, and the m=37/m=450 parity probes read the logits \
     bit-identical to candle's own GEMM over the dequant resident, but the fused CE reductions \
     and the dX k-chain reassociate, so the arm is tolerance-class BY CONSTRUCTION and gates on \
     eval CE, never on bitwise trajectory. The exp/log in the CE reductions are precise:: \
     intrinsics; every reduction order is fixed, so reruns of the same binary are deterministic. \
     A 2-m-block register-blocked BK=32 retile of both GEMMs measured 2.6x SLOWER than these \
     BM32/BN64/BK64 tiles (2.9 vs 7.6 TF/s) and is a banked negative, not a frontier";

pub const W4HEAD_DX_SPLITS_THE_CONTRACTION_BECAUSE_24_OUTPUT_TILES_STARVE_THE_GRID: usize = 8;

const BM: usize = 32;
const BN: usize = 64;
const GS: usize = 64;

const MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct P {
    uint m;
    uint n;
    uint k;
    uint chunk;
    uint ns;
};

constant constexpr uint BM = 32;
constant constexpr uint BN = 64;
constant constexpr uint TG = 128;

kernel void w4head_logits(
    device const float* x [[buffer(0)]],
    device const uint* wq [[buffer(1)]],
    device const float* sc [[buffer(2)]],
    device const float* bi [[buffer(3)]],
    device float* out [[buffer(4)]],
    constant P& p [[buffer(5)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint wpr = p.k >> 3;
    const uint groups = p.k >> 6;
    const uint m0 = tg.x * BM;
    const uint n0 = tg.y * BN;
    threadgroup float At[BM][64];
    threadgroup float Bt[64][BN + 4];
    threadgroup float St[4][8][9];
    simdgroup_float8x8 acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = simdgroup_float8x8(0.0f);
    const uint kiters = p.k >> 6;
    for (uint kb = 0; kb < kiters; kb++) {
        for (uint i = tid; i < BM * 16; i += TG) {
            const uint r = i >> 4;
            const uint c4 = i & 15;
            float4 v = float4(0.0f);
            if (m0 + r < p.m) {
                v = *(device const float4*)(x + (m0 + r) * p.k + kb * 64 + c4 * 4);
            }
            At[r][c4 * 4 + 0] = v.x;
            At[r][c4 * 4 + 1] = v.y;
            At[r][c4 * 4 + 2] = v.z;
            At[r][c4 * 4 + 3] = v.w;
        }
        for (uint q = tid; q < BN * 8; q += TG) {
            const uint j = q >> 3;
            const uint wi = q & 7;
            const uint nr = n0 + j;
            const uint word = wq[nr * wpr + kb * 8 + wi];
            const float s = sc[nr * groups + kb];
            const float b = bi[nr * groups + kb];
            for (uint e = 0; e < 8; e++) {
                Bt[wi * 8 + e][j] = fma(float((word >> (4 * e)) & 0xFu), s, b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 8; kk++) {
            simdgroup_float8x8 a;
            simdgroup_load(a, &At[sg * 8][kk * 8], 64);
            for (uint nb = 0; nb < 8; nb++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, &Bt[kk * 8][nb * 8], BN + 4);
                simdgroup_multiply_accumulate(acc[nb], a, b, acc[nb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (m0 + BM <= p.m) {
        for (uint nb = 0; nb < 8; nb++) {
            simdgroup_store(acc[nb], out + (ulong)(m0 + sg * 8) * p.n + n0 + nb * 8, p.n);
        }
    } else {
        for (uint nb = 0; nb < 8; nb++) {
            simdgroup_store(acc[nb], &St[sg][0][0], 9);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = e >> 3;
                const uint c = e & 7;
                if (m0 + sg * 8 + r < p.m) {
                    out[(ulong)(m0 + sg * 8 + r) * p.n + n0 + nb * 8 + c] = St[sg][r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void w4head_ce_stats(
    device const float* logits [[buffer(0)]],
    device const uint* tgt [[buffer(1)]],
    device const float* wr [[buffer(2)]],
    device float* lse [[buffer(3)]],
    device float* loss [[buffer(4)]],
    constant P& p [[buffer(5)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float red[8];
    device const float4* r4 = (device const float4*)(logits + (ulong)row * p.n);
    const uint n4 = p.n >> 2;
    float mx = -INFINITY;
    for (uint i = tid; i < n4; i += 256) {
        const float4 v = r4[i];
        mx = max(mx, max(max(v.x, v.y), max(v.z, v.w)));
    }
    mx = simd_max(mx);
    if (lane == 0) red[sg] = mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = max(max(max(red[0], red[1]), max(red[2], red[3])),
             max(max(red[4], red[5]), max(red[6], red[7])));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0.0f;
    for (uint i = tid; i < n4; i += 256) {
        const float4 v = r4[i];
        sum += precise::exp(v.x - mx) + precise::exp(v.y - mx)
             + precise::exp(v.z - mx) + precise::exp(v.w - mx);
    }
    sum = simd_sum(sum);
    if (lane == 0) red[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        sum = ((red[0] + red[1]) + (red[2] + red[3]))
            + ((red[4] + red[5]) + (red[6] + red[7]));
        const float l = mx + precise::log(sum);
        lse[row] = l;
        const float lt = logits[(ulong)row * p.n + tgt[row]];
        loss[row] = wr[row] * (l - lt);
    }
}

kernel void w4head_dlogits_inplace(
    device float* logits [[buffer(0)]],
    device const uint* tgt [[buffer(1)]],
    device const float* wr [[buffer(2)]],
    device const float* g [[buffer(3)]],
    device const float* lse [[buffer(4)]],
    constant P& p [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    const uint n4 = p.n >> 2;
    if (gid >= p.m * n4) return;
    const uint row = gid / n4;
    const uint c0 = (gid - row * n4) << 2;
    const float wg = wr[row] * g[row];
    const float l = lse[row];
    const uint t = tgt[row];
    device float* ptr = logits + (ulong)row * p.n + c0;
    float4 v = *(device float4*)ptr;
    v.x = wg * (precise::exp(v.x - l) - (c0 + 0 == t ? 1.0f : 0.0f));
    v.y = wg * (precise::exp(v.y - l) - (c0 + 1 == t ? 1.0f : 0.0f));
    v.z = wg * (precise::exp(v.z - l) - (c0 + 2 == t ? 1.0f : 0.0f));
    v.w = wg * (precise::exp(v.w - l) - (c0 + 3 == t ? 1.0f : 0.0f));
    *(device float4*)ptr = v;
}

kernel void w4head_dx_partial(
    device const float* dlogits [[buffer(0)]],
    device const uint* wq [[buffer(1)]],
    device const float* sc [[buffer(2)]],
    device const float* bi [[buffer(3)]],
    device float* dxp [[buffer(4)]],
    constant P& p [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const uint wpr = p.k >> 3;
    const uint groups = p.k >> 6;
    const uint m0 = tg.x * BM;
    const uint ko0 = tg.y * 64;
    const uint ntiles = p.n / BN;
    const uint t0 = tg.z * p.chunk;
    const uint t1 = min(t0 + p.chunk, ntiles);
    threadgroup float Dt[BM][BN + 4];
    threadgroup float Wt[BN][64 + 4];
    threadgroup float St[4][8][9];
    simdgroup_float8x8 acc[8];
    for (uint i = 0; i < 8; i++) acc[i] = simdgroup_float8x8(0.0f);
    for (uint nt = t0; nt < t1; nt++) {
        const uint n0 = nt * BN;
        for (uint i = tid; i < BM * 16; i += TG) {
            const uint r = i >> 4;
            const uint c4 = i & 15;
            float4 d = float4(0.0f);
            if (m0 + r < p.m) {
                d = *(device const float4*)(dlogits + (ulong)(m0 + r) * p.n + n0 + c4 * 4);
            }
            Dt[r][c4 * 4 + 0] = d.x;
            Dt[r][c4 * 4 + 1] = d.y;
            Dt[r][c4 * 4 + 2] = d.z;
            Dt[r][c4 * 4 + 3] = d.w;
        }
        for (uint q = tid; q < BN * 8; q += TG) {
            const uint j = q >> 3;
            const uint wi = q & 7;
            const uint nr = n0 + j;
            const uint word = wq[nr * wpr + (ko0 >> 3) + wi];
            const float s = sc[nr * groups + tg.y];
            const float b = bi[nr * groups + tg.y];
            for (uint e = 0; e < 8; e++) {
                Wt[j][wi * 8 + e] = fma(float((word >> (4 * e)) & 0xFu), s, b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < 8; kk++) {
            simdgroup_float8x8 a;
            simdgroup_load(a, &Dt[sg * 8][kk * 8], BN + 4);
            for (uint nb = 0; nb < 8; nb++) {
                simdgroup_float8x8 b;
                simdgroup_load(b, &Wt[kk * 8][nb * 8], 64 + 4);
                simdgroup_multiply_accumulate(acc[nb], a, b, acc[nb]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* base = dxp + (ulong)tg.z * p.m * p.k;
    if (m0 + BM <= p.m) {
        for (uint nb = 0; nb < 8; nb++) {
            simdgroup_store(acc[nb], base + (ulong)(m0 + sg * 8) * p.k + ko0 + nb * 8, p.k);
        }
    } else {
        for (uint nb = 0; nb < 8; nb++) {
            simdgroup_store(acc[nb], &St[sg][0][0], 9);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64; e += 32) {
                const uint r = e >> 3;
                const uint c = e & 7;
                if (m0 + sg * 8 + r < p.m) {
                    base[(ulong)(m0 + sg * 8 + r) * p.k + ko0 + nb * 8 + c] = St[sg][r][c];
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void w4head_dx_reduce(
    device const float* dxp [[buffer(0)]],
    device float* dx [[buffer(1)]],
    constant P& p [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint total = p.m * p.k;
    if (gid >= total) return;
    float s = 0.0f;
    for (uint i = 0; i < p.ns; i++) {
        s += dxp[(ulong)i * total + gid];
    }
    dx[gid] = s;
}
"#;

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    m: u32,
    n: u32,
    k: u32,
    chunk: u32,
    ns: u32,
}

struct Pipes {
    logits: ComputePipeline,
    stats: ComputePipeline,
    dlog: ComputePipeline,
    dxp: ComputePipeline,
    dxred: ComputePipeline,
}

fn cerr(e: impl std::fmt::Display) -> candle_core::Error {
    candle_core::Error::Msg(format!("w4head: {e}"))
}

pub struct DotsW4Head {
    wq: Tensor,
    scales: Tensor,
    biases: Tensor,
    pub n: usize,
    pub k: usize,
    pipes: OnceLock<Pipes>,
}

impl DotsW4Head {
    pub fn from_raw(
        words: Vec<u32>,
        scales_f32: Vec<f32>,
        biases_f32: Vec<f32>,
        n: usize,
        k: usize,
        device: &Device,
    ) -> Result<Arc<Self>> {
        let _ =
            W4HEAD_FUSES_THE_HEAD_CE_AGAINST_THE_CHECKPOINTS_OWN_W4_BYTES_AND_IS_TOLERANCE_CLASS;
        ensure!(device.is_metal(), "the fused W4 head is Metal-only");
        ensure!(
            n % BN == 0 && k % GS == 0 && k % 64 == 0,
            "W4 head needs n % {BN} == 0 and k % {GS} == 0, got n {n} k {k}"
        );
        ensure!(words.len() == n * k / 8, "packed words must be n*k/8");
        ensure!(scales_f32.len() == n * (k / GS), "scales must be n*k/gs");
        ensure!(biases_f32.len() == n * (k / GS), "biases must be n*k/gs");
        let up_u32 = |v: Vec<u32>, shape: (usize, usize)| -> Result<Tensor> {
            Ok(Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(device)?)
        };
        let up_f32 = |v: Vec<f32>, shape: (usize, usize)| -> Result<Tensor> {
            Ok(Tensor::from_vec(v, shape, &Device::Cpu)?.to_device(device)?)
        };
        Ok(Arc::new(Self {
            wq: up_u32(words, (n, k / 8))?,
            scales: up_f32(scales_f32, (n, k / GS))?,
            biases: up_f32(biases_f32, (n, k / GS))?,
            n,
            k,
            pipes: OnceLock::new(),
        }))
    }

    fn pipes(&self, dev: &candle_core::MetalDevice) -> candle_core::Result<&Pipes> {
        if self.pipes.get().is_none() {
            let lib = dev
                .metal_device()
                .new_library_with_source(MSL, None)
                .map_err(cerr)?;
            let mk = |name: &str| -> candle_core::Result<ComputePipeline> {
                let f = lib.get_function(name, None).map_err(cerr)?;
                dev.metal_device()
                    .new_compute_pipeline_state_with_function(&f)
                    .map_err(cerr)
            };
            let pipes = Pipes {
                logits: mk("w4head_logits")?,
                stats: mk("w4head_ce_stats")?,
                dlog: mk("w4head_dlogits_inplace")?,
                dxp: mk("w4head_dx_partial")?,
                dxred: mk("w4head_dx_reduce")?,
            };
            let _ = self.pipes.set(pipes);
        }
        Ok(self.pipes.get().expect("pipes were just set"))
    }

    pub fn resident_bytes(&self) -> usize {
        (self.n * self.k) / 2 + 2 * self.n * (self.k / GS) * 4
    }
}

fn metal_buf_and_off(t: &Tensor, what: &'static str) -> candle_core::Result<(Buffer, usize)> {
    let (storage, layout) = t.storage_and_layout();
    let s = match &*storage {
        candle_core::Storage::Metal(s) => s,
        _ => return Err(cerr(format!("{what} must be Metal-resident"))),
    };
    let (start, _end) = layout
        .contiguous_offsets()
        .ok_or_else(|| cerr(format!("{what} must be contiguous")))?;
    Ok((s.buffer().clone(), start * t.dtype().size_in_bytes()))
}

struct Stash {
    logits: Arc<Buffer>,
    lse: Arc<Buffer>,
    m: usize,
}

fn encode_logits(
    head: &DotsW4Head,
    dev: &candle_core::MetalDevice,
    x_buf: &Buffer,
    x_off: usize,
    m: usize,
) -> candle_core::Result<Arc<Buffer>> {
    let pipes = head.pipes(dev)?;
    let out = dev
        .new_buffer_builder()
        .with_size(m * head.n * DType::F32.size_in_bytes())
        .with_label("w4head-logits")
        .build()?;
    let (wq, wq_off) = metal_buf_and_off(&head.wq, "w4 words")?;
    let (sc, sc_off) = metal_buf_and_off(&head.scales, "w4 scales")?;
    let (bi, bi_off) = metal_buf_and_off(&head.biases, "w4 biases")?;
    let p = Params {
        m: m as u32,
        n: head.n as u32,
        k: head.k as u32,
        chunk: 0,
        ns: 0,
    };
    let enc = dev.command_encoder()?;
    enc.set_compute_pipeline_state(&pipes.logits);
    let e = enc.as_ref();
    e.set_input_buffer(0, Some(x_buf), x_off);
    e.set_input_buffer(1, Some(&wq), wq_off);
    e.set_input_buffer(2, Some(&sc), sc_off);
    e.set_input_buffer(3, Some(&bi), bi_off);
    e.set_output_buffer(4, Some(&*out), 0);
    e.set_bytes(5, &p);
    e.dispatch_thread_groups(
        MTLSize {
            width: m.div_ceil(BM),
            height: head.n / BN,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(out)
}

pub struct W4LogitsOp {
    pub head: Arc<DotsW4Head>,
}

impl candle_core::CustomOp1 for W4LogitsOp {
    fn name(&self) -> &'static str {
        "nv-dots-w4head-logits-forward-no-backward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(cerr("the fused W4 head is Metal-only"))
    }

    fn metal_fwd(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (m, k) = match layout.shape().dims() {
            [m, k] => (*m, *k),
            d => return Err(cerr(format!("W4 logits needs a rank-2 input, got {d:?}"))),
        };
        if k != self.head.k || storage.dtype() != DType::F32 {
            return Err(cerr(format!(
                "W4 logits input [{m}, {k}] {:?} != [{m}, {}] F32",
                storage.dtype(),
                self.head.k
            )));
        }
        let (start, _end) = layout
            .contiguous_offsets()
            .ok_or_else(|| cerr("W4 logits needs a contiguous input"))?;
        let dev = storage.device().clone();
        let out = encode_logits(&self.head, &dev, storage.buffer(), start * 4, m)?;
        Ok((
            MetalStorage::new(out, dev, m * self.head.n, DType::F32),
            (m, self.head.n).into(),
        ))
    }
}

pub struct W4CeRowsOp {
    head: Arc<DotsW4Head>,
    tgt: Tensor,
    wrow: Tensor,
    stash: Mutex<Option<Stash>>,
}

impl W4CeRowsOp {
    pub fn new(head: Arc<DotsW4Head>, tgt: Tensor, wrow: Tensor) -> Result<Self> {
        ensure!(
            tgt.dtype() == DType::U32 && wrow.dtype() == DType::F32,
            "W4 CE targets must be U32 and row weights F32"
        );
        Ok(Self {
            head,
            tgt,
            wrow,
            stash: Mutex::new(None),
        })
    }
}

impl candle_core::CustomOp1 for W4CeRowsOp {
    fn name(&self) -> &'static str {
        "nv-dots-w4head-fused-ce-rows-with-fused-dx-backward"
    }

    fn cpu_fwd(
        &self,
        _storage: &candle_core::CpuStorage,
        _layout: &Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, Shape)> {
        Err(cerr("the fused W4 head CE is Metal-only"))
    }

    fn metal_fwd(
        &self,
        storage: &MetalStorage,
        layout: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        let (m, k) = match layout.shape().dims() {
            [m, k] => (*m, *k),
            d => return Err(cerr(format!("W4 CE needs a rank-2 input, got {d:?}"))),
        };
        if k != self.head.k
            || storage.dtype() != DType::F32
            || self.tgt.dims() != [m]
            || self.wrow.dims() != [m]
        {
            return Err(cerr(format!(
                "W4 CE shapes disagree: x [{m}, {k}], tgt {:?}, w {:?}, hidden {}",
                self.tgt.dims(),
                self.wrow.dims(),
                self.head.k
            )));
        }
        let (start, _end) = layout
            .contiguous_offsets()
            .ok_or_else(|| cerr("W4 CE needs a contiguous input"))?;
        let dev = storage.device().clone();
        let logits = encode_logits(&self.head, &dev, storage.buffer(), start * 4, m)?;
        let pipes = self.head.pipes(&dev)?;
        let lse = dev
            .new_buffer_builder()
            .with_size(m * 4)
            .with_label("w4head-lse")
            .build()?;
        let loss = dev
            .new_buffer_builder()
            .with_size(m * 4)
            .with_label("w4head-loss-rows")
            .build()?;
        let (tgt, tgt_off) = metal_buf_and_off(&self.tgt, "targets")?;
        let (wr, wr_off) = metal_buf_and_off(&self.wrow, "row weights")?;
        let p = Params {
            m: m as u32,
            n: self.head.n as u32,
            k: self.head.k as u32,
            chunk: 0,
            ns: 0,
        };
        {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipes.stats);
            let e = enc.as_ref();
            e.set_input_buffer(0, Some(&*logits), 0);
            e.set_input_buffer(1, Some(&tgt), tgt_off);
            e.set_input_buffer(2, Some(&wr), wr_off);
            e.set_output_buffer(3, Some(&*lse), 0);
            e.set_output_buffer(4, Some(&*loss), 0);
            e.set_bytes(5, &p);
            e.dispatch_thread_groups(
                MTLSize {
                    width: m,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        *self.stash.lock().expect("w4head stash poisoned") = Some(Stash { logits, lse, m });
        Ok((MetalStorage::new(loss, dev, m, DType::F32), (m,).into()))
    }

    fn bwd(
        &self,
        arg: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> candle_core::Result<Option<Tensor>> {
        let stash = self
            .stash
            .lock()
            .expect("w4head stash poisoned")
            .take()
            .ok_or_else(|| cerr("backward ran twice or before forward"))?;
        let dev = match arg.device() {
            Device::Metal(d) => d.clone(),
            _ => return Err(cerr("W4 CE backward needs a Metal arg")),
        };
        let pipes = self.head.pipes(&dev)?;
        let (m, k, n) = (stash.m, self.head.k, self.head.n);
        let g = grad_res.to_dtype(DType::F32)?.contiguous()?;
        if g.dims() != [m] {
            return Err(cerr(format!("upstream grad {:?} != [{m}]", g.dims())));
        }
        let ns = W4HEAD_DX_SPLITS_THE_CONTRACTION_BECAUSE_24_OUTPUT_TILES_STARVE_THE_GRID;
        let dxp = dev
            .new_buffer_builder()
            .with_size(ns * m * k * 4)
            .with_label("w4head-dx-partials")
            .build()?;
        let dx = dev
            .new_buffer_builder()
            .with_size(m * k * 4)
            .with_label("w4head-dx")
            .build()?;
        let (wq, wq_off) = metal_buf_and_off(&self.head.wq, "w4 words")?;
        let (sc, sc_off) = metal_buf_and_off(&self.head.scales, "w4 scales")?;
        let (bi, bi_off) = metal_buf_and_off(&self.head.biases, "w4 biases")?;
        let (tgt, tgt_off) = metal_buf_and_off(&self.tgt, "targets")?;
        let (wr, wr_off) = metal_buf_and_off(&self.wrow, "row weights")?;
        let (gb, g_off) = metal_buf_and_off(&g, "upstream grad")?;
        let ntiles = n / BN;
        let p = Params {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            chunk: ntiles.div_ceil(ns) as u32,
            ns: ns as u32,
        };
        {
            let enc = dev.command_encoder()?;
            enc.set_compute_pipeline_state(&pipes.dlog);
            let e = enc.as_ref();
            e.set_output_buffer(0, Some(&*stash.logits), 0);
            e.set_input_buffer(1, Some(&tgt), tgt_off);
            e.set_input_buffer(2, Some(&wr), wr_off);
            e.set_input_buffer(3, Some(&gb), g_off);
            e.set_input_buffer(4, Some(&*stash.lse), 0);
            e.set_bytes(5, &p);
            e.dispatch_thread_groups(
                MTLSize {
                    width: (m * n / 4).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
            enc.set_compute_pipeline_state(&pipes.dxp);
            e.set_input_buffer(0, Some(&*stash.logits), 0);
            e.set_input_buffer(1, Some(&wq), wq_off);
            e.set_input_buffer(2, Some(&sc), sc_off);
            e.set_input_buffer(3, Some(&bi), bi_off);
            e.set_output_buffer(4, Some(&*dxp), 0);
            e.set_bytes(5, &p);
            e.dispatch_thread_groups(
                MTLSize {
                    width: m.div_ceil(BM),
                    height: k / 64,
                    depth: ns,
                },
                MTLSize {
                    width: 128,
                    height: 1,
                    depth: 1,
                },
            );
            enc.set_compute_pipeline_state(&pipes.dxred);
            e.set_input_buffer(0, Some(&*dxp), 0);
            e.set_output_buffer(1, Some(&*dx), 0);
            e.set_bytes(2, &p);
            e.dispatch_thread_groups(
                MTLSize {
                    width: (m * k).div_ceil(256),
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 256,
                    height: 1,
                    depth: 1,
                },
            );
        }
        let storage = MetalStorage::new(dx, dev, m * k, DType::F32);
        let dx_t = Tensor::from_storage(
            candle_core::Storage::Metal(storage),
            (m, k),
            candle_core::op::BackpropOp::none(),
            false,
        );
        Ok(Some(dx_t))
    }
}

pub fn fused_ce_rows(
    head: &Arc<DotsW4Head>,
    xc: &Tensor,
    tgt: &Tensor,
    wrow: &Tensor,
) -> Result<Tensor> {
    let op = W4CeRowsOp::new(head.clone(), tgt.clone(), wrow.clone())?;
    xc.apply_op1(op).context("fused W4 head CE rows forward")
}

pub fn logits_f32(head: &Arc<DotsW4Head>, xc: &Tensor) -> Result<Tensor> {
    Ok(xc.apply_op1_no_bwd(&W4LogitsOp { head: head.clone() })?)
}
