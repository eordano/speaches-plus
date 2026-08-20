#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::BLOCK_SIZE;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;
mod common;
use common::swizzled_dst as swizzled_scale_dst;

const MIN_TILE: usize = 128;
const E_TOTAL: usize = 256;

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed | 1)
    }

    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn fill_bytes(&mut self, buf: &mut [u8]) {
        let mut it = buf.chunks_exact_mut(8);
        for c in &mut it {
            c.copy_from_slice(&self.next().to_le_bytes());
        }
        let rem = it.into_remainder();
        let last = self.next().to_le_bytes();
        let n = rem.len();
        rem.copy_from_slice(&last[..n]);
    }

    fn scale_byte(&mut self) -> u8 {
        0x30 + (self.next() % 0x18) as u8
    }

    fn unit_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn swizzled_scale_bytes(rows: usize, cols: usize) -> usize {
    let k_blocks = cols / BLOCK_SIZE;
    rows.div_ceil(128) * 128 * k_blocks.div_ceil(4) * 4
}

struct Setup {
    b_packed: CudaSlice<u8>,
    b_scales: CudaSlice<u8>,
    alphas: CudaSlice<f32>,
    a_packed: CudaSlice<u8>,
    a_scales: CudaSlice<u8>,
    id_sets: Vec<CudaSlice<i32>>,
    d: CudaSlice<bf16>,
    n: usize,
    k: usize,
    groups: usize,
}

fn build(stream: &Arc<CudaStream>, rng: &mut Rng, n: usize, k: usize, groups: usize) -> Setup {
    let group_k = k / BLOCK_SIZE;
    let mut b_packed_host = vec![0u8; E_TOTAL * n * k / 2];
    rng.fill_bytes(&mut b_packed_host);
    let sf_per_expert = swizzled_scale_bytes(n, k);
    let mut b_scales_host = vec![0u8; E_TOTAL * sf_per_expert];
    for e in 0..E_TOTAL {
        let base = e * sf_per_expert;
        for r in 0..n {
            for kb in 0..group_k {
                b_scales_host[base + swizzled_scale_dst(r, kb, group_k)] = rng.scale_byte();
            }
        }
    }
    let alphas_host: Vec<f32> = (0..E_TOTAL).map(|_| 0.5 + 1.5 * rng.unit_f32()).collect();

    let rows = groups * MIN_TILE;
    let mut a_packed_host = vec![0u8; rows * k / 2];
    for g in 0..groups {
        let lo = g * MIN_TILE * k / 2;
        rng.fill_bytes(&mut a_packed_host[lo..lo + k / 2]);
    }
    let mut a_scales_host = vec![0u8; swizzled_scale_bytes(rows, k)];
    for g in 0..groups {
        for kb in 0..group_k {
            a_scales_host[swizzled_scale_dst(g * MIN_TILE, kb, group_k)] = rng.scale_byte();
        }
    }

    let n_sets = E_TOTAL / groups.min(E_TOTAL);
    let mut id_sets = Vec::with_capacity(n_sets);
    for s in 0..n_sets {
        let ids_host: Vec<i32> = (0..groups)
            .map(|g| ((s * groups + g) % E_TOTAL) as i32)
            .collect();
        #[allow(deprecated)]
        let ids = stream.memcpy_stod(&ids_host).unwrap();
        id_sets.push(ids);
    }

    #[allow(deprecated)]
    let b_packed = stream.memcpy_stod(&b_packed_host).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_scales_host).unwrap();
    #[allow(deprecated)]
    let alphas = stream.memcpy_stod(&alphas_host).unwrap();
    #[allow(deprecated)]
    let a_packed = stream.memcpy_stod(&a_packed_host).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_scales_host).unwrap();
    let d = stream.alloc_zeros::<bf16>(rows * n).unwrap();
    Setup {
        b_packed,
        b_scales,
        alphas,
        a_packed,
        a_scales,
        id_sets,
        d,
        n,
        k,
        groups,
    }
}

fn launch_gemv(stream: &Arc<CudaStream>, s: &mut Setup, set: usize) {
    let (ap, _g1) = s.a_packed.device_ptr(stream);
    let (asp, _g2) = s.a_scales.device_ptr(stream);
    let (bp, _g3) = s.b_packed.device_ptr(stream);
    let (bsp, _g4) = s.b_scales.device_ptr(stream);
    let (alp, _g5) = s.alphas.device_ptr(stream);
    let (idp, _g7) = s.id_sets[set].device_ptr(stream);
    let (dp, _g6) = s.d.device_ptr_mut(stream);
    let rc = unsafe {
        nv_kernels::cuda::moe_grouped_fp4_gemv_m1_bf16(
            stream.cu_stream() as *mut c_void,
            ap as *const u8,
            asp as *const u8,
            bp as *const u8,
            bsp as *const u8,
            alp as *const f32,
            dp as *mut u16,
            idp as *const i32,
            s.groups as i32,
            E_TOTAL as i32,
            s.n as i32,
            s.k as i32,
            MIN_TILE as i32,
            (MIN_TILE * s.n) as i64,
        )
    };
    assert_eq!(rc, 0);
}

struct CutlassAux {
    eo: CudaSlice<i32>,
    sfo: CudaSlice<i32>,
    ps: CudaSlice<i32>,
    meta_scratch: CudaSlice<u8>,
    gemm_ws: CudaSlice<u8>,
}

fn build_cutlass_aux(stream: &Arc<CudaStream>, n: usize, k: usize, groups: usize) -> CutlassAux {
    let offsets_host: Vec<i32> = (0..groups).map(|g| (g * MIN_TILE) as i32).collect();
    let mut ps_host: Vec<i32> = Vec::with_capacity(groups * 3);
    for _ in 0..groups {
        ps_host.extend_from_slice(&[1, n as i32, k as i32]);
    }
    #[allow(deprecated)]
    let eo = stream.memcpy_stod(&offsets_host).unwrap();
    #[allow(deprecated)]
    let sfo = stream.memcpy_stod(&offsets_host).unwrap();
    #[allow(deprecated)]
    let ps = stream.memcpy_stod(&ps_host).unwrap();
    let meta_scratch = stream.alloc_zeros::<u8>(128 * 1024).unwrap();
    let gemm_ws = stream.alloc_zeros::<u8>(64 * 1024 * 1024).unwrap();
    CutlassAux {
        eo,
        sfo,
        ps,
        meta_scratch,
        gemm_ws,
    }
}

fn launch_cutlass(stream: &Arc<CudaStream>, s: &mut Setup, aux: &mut CutlassAux, set: usize) {
    let (ap, _g1) = s.a_packed.device_ptr(stream);
    let (asp, _g2) = s.a_scales.device_ptr(stream);
    let (bp, _g3) = s.b_packed.device_ptr(stream);
    let (bsp, _g4) = s.b_scales.device_ptr(stream);
    let (alp, _g5) = s.alphas.device_ptr(stream);
    let (aei, _ga) = s.id_sets[set].device_ptr(stream);
    let (eo, _g7) = aux.eo.device_ptr(stream);
    let (sfo, _g8) = aux.sfo.device_ptr(stream);
    let (ps, _g9) = aux.ps.device_ptr(stream);
    let (dp, _g6) = s.d.device_ptr_mut(stream);
    let (ms, _gb) = aux.meta_scratch.device_ptr_mut(stream);
    let (ws, _gc) = aux.gemm_ws.device_ptr_mut(stream);
    unsafe {
        nv_kernels::cuda::cutlass_moe_grouped_fp4_gemm_sm120_bf16(
            stream.cu_stream() as *mut c_void,
            ap as *const c_void,
            asp as *const c_void,
            bp as *const c_void,
            bsp as *const c_void,
            alp as *const f32,
            dp as *mut c_void,
            eo as *const i32,
            sfo as *const i32,
            ps as *const i32,
            aei as *const i32,
            s.n as i32,
            s.k as i32,
            s.groups as i32,
            s.k as i64,
            s.k as i64,
            s.n as i64,
            ms as *mut c_void,
            128 * 1024,
            ws as *mut c_void,
            64 * 1024 * 1024,
        )
        .expect("cutlass grouped FP4 GEMM launch")
    };
}

fn time_arm<F: FnMut(usize)>(
    stream: &Arc<CudaStream>,
    n_sets: usize,
    warmup: usize,
    reps: usize,
    mut f: F,
) -> f64 {
    for i in 0..warmup {
        f(i % n_sets);
    }
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for i in 0..reps {
        f(i % n_sets);
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() / reps as f64
}

#[test]
#[ignore]
fn moe_grouped_fp4_gemv_m1_bench() {
    if std::env::var("NV_MOE_M1_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_MOE_M1_BENCH=1 to run moe_grouped_fp4_gemv_m1_bench");
        return;
    }
    let ctx = CudaContext::new(0).expect("no CUDA device 0");
    let stream = ctx.default_stream();
    let warmup = 40usize;
    let reps = 320usize;

    eprintln!("BENCH basis: E_TOTAL={E_TOTAL} experts resident, ids rotate over all experts across reps (working set per shape ~= {} MiB weights+scales), warmup={warmup} discarded, reps={reps}, wall-clock over reps between stream syncs (launch overhead included), m=1 per group", E_TOTAL * (512 * 2048 / 2 + swizzled_scale_bytes(512, 2048)) * 2 / (1 << 20));

    for &(n, k) in &[(512usize, 2048usize), (2048usize, 512usize)] {
        for &groups in &[8usize, 16, 32] {
            let mut rng = Rng::new(0x1234_5678 ^ ((n as u64) << 24) ^ groups as u64);
            let mut s = build(&stream, &mut rng, n, k, groups);
            let n_sets = s.id_sets.len();
            let mut aux = build_cutlass_aux(&stream, n, k, groups);

            let bytes_w = groups * (n * k / 2 + n * (k / BLOCK_SIZE));
            let bytes_act = groups * (k / 2) + swizzled_scale_bytes(groups * MIN_TILE, k);
            let bytes_out = groups * n * 2;
            let bytes = (bytes_w + bytes_act + bytes_out) as f64;

            let t_cutlass = time_arm(&stream, n_sets, warmup, reps, |set| {
                launch_cutlass(&stream, &mut s, &mut aux, set)
            });
            let t_gemv = time_arm(&stream, n_sets, warmup, reps, |set| {
                launch_gemv(&stream, &mut s, set)
            });

            eprintln!(
                "N={n} K={k} groups={groups}: cutlass={:.2} us/launch ({:.1} GB/s) gemv={:.2} us/launch ({:.1} GB/s) speedup={:.2}x [bytes/launch={:.0}]",
                t_cutlass * 1e6,
                bytes / t_cutlass / 1e9,
                t_gemv * 1e6,
                bytes / t_gemv / 1e9,
                t_cutlass / t_gemv,
                bytes
            );
        }
    }
}
