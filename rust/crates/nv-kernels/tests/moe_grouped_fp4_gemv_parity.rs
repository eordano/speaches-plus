#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUdevice_attribute;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{supports_nvfp4, BLOCK_SIZE};
use std::ffi::c_void;
use std::sync::Arc;
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

    fn range(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit_f32(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn swizzled_scale_bytes(rows: usize, cols: usize) -> usize {
    let k_blocks = cols / BLOCK_SIZE;
    rows.div_ceil(128) * 128 * k_blocks.div_ceil(4) * 4
}

struct Weights {
    b_packed: CudaSlice<u8>,
    b_scales: CudaSlice<u8>,
    alphas: CudaSlice<f32>,
    n: usize,
    k: usize,
}

fn build_weights(stream: &Arc<CudaStream>, rng: &mut Rng, n: usize, k: usize) -> Weights {
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

    #[allow(deprecated)]
    let b_packed = stream.memcpy_stod(&b_packed_host).unwrap();
    #[allow(deprecated)]
    let b_scales = stream.memcpy_stod(&b_scales_host).unwrap();
    #[allow(deprecated)]
    let alphas = stream.memcpy_stod(&alphas_host).unwrap();
    Weights {
        b_packed,
        b_scales,
        alphas,
        n,
        k,
    }
}

struct Acts {
    a_packed: CudaSlice<u8>,
    a_scales: CudaSlice<u8>,
    ids: CudaSlice<i32>,
    groups: usize,
}

fn build_acts(stream: &Arc<CudaStream>, rng: &mut Rng, groups: usize, k: usize) -> Acts {
    let group_k = k / BLOCK_SIZE;
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
    let mut ids_host: Vec<i32> = (0..groups).map(|_| rng.range(E_TOTAL) as i32).collect();
    if groups >= 2 {
        ids_host[1] = ids_host[0];
    }

    #[allow(deprecated)]
    let a_packed = stream.memcpy_stod(&a_packed_host).unwrap();
    #[allow(deprecated)]
    let a_scales = stream.memcpy_stod(&a_scales_host).unwrap();
    #[allow(deprecated)]
    let ids = stream.memcpy_stod(&ids_host).unwrap();
    Acts {
        a_packed,
        a_scales,
        ids,
        groups,
    }
}

fn run_cutlass(stream: &Arc<CudaStream>, w: &Weights, a: &Acts) -> Vec<bf16> {
    let groups = a.groups;
    let (n, k) = (w.n, w.k);
    let mut d = stream.alloc_zeros::<bf16>(groups * MIN_TILE * n).unwrap();
    let mut meta_scratch = stream.alloc_zeros::<u8>(128 * 1024).unwrap();
    let mut gemm_ws = stream.alloc_zeros::<u8>(64 * 1024 * 1024).unwrap();

    let offsets_host: Vec<i32> = (0..groups).map(|g| (g * MIN_TILE) as i32).collect();
    let mut ps_host: Vec<i32> = Vec::with_capacity(groups * 3);
    for _ in 0..groups {
        ps_host.extend_from_slice(&[1, n as i32, k as i32]);
    }
    #[allow(deprecated)]
    let eo_dev = stream.memcpy_stod(&offsets_host).unwrap();
    #[allow(deprecated)]
    let sfo_dev = stream.memcpy_stod(&offsets_host).unwrap();
    #[allow(deprecated)]
    let ps_dev = stream.memcpy_stod(&ps_host).unwrap();

    {
        let (ap, _g1) = a.a_packed.device_ptr(stream);
        let (asp, _g2) = a.a_scales.device_ptr(stream);
        let (bp, _g3) = w.b_packed.device_ptr(stream);
        let (bsp, _g4) = w.b_scales.device_ptr(stream);
        let (alp, _g5) = w.alphas.device_ptr(stream);
        let (dp, _g6) = d.device_ptr_mut(stream);
        let (eo, _g7) = eo_dev.device_ptr(stream);
        let (sfo, _g8) = sfo_dev.device_ptr(stream);
        let (ps, _g9) = ps_dev.device_ptr(stream);
        let (aei, _ga) = a.ids.device_ptr(stream);
        let (ms, _gb) = meta_scratch.device_ptr_mut(stream);
        let (ws, _gc) = gemm_ws.device_ptr_mut(stream);
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
                n as i32,
                k as i32,
                groups as i32,
                k as i64,
                k as i64,
                n as i64,
                ms as *mut c_void,
                128 * 1024,
                ws as *mut c_void,
                64 * 1024 * 1024,
            )
            .expect("cutlass grouped FP4 GEMM launch")
        };
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&d).unwrap()
}

fn run_gemv(stream: &Arc<CudaStream>, w: &Weights, a: &Acts) -> Vec<bf16> {
    let groups = a.groups;
    let (n, k) = (w.n, w.k);
    let mut d = stream.alloc_zeros::<bf16>(groups * MIN_TILE * n).unwrap();
    {
        let (ap, _g1) = a.a_packed.device_ptr(stream);
        let (asp, _g2) = a.a_scales.device_ptr(stream);
        let (bp, _g3) = w.b_packed.device_ptr(stream);
        let (bsp, _g4) = w.b_scales.device_ptr(stream);
        let (alp, _g5) = w.alphas.device_ptr(stream);
        let (dp, _g6) = d.device_ptr_mut(stream);
        let (idp, _g7) = a.ids.device_ptr(stream);
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
                groups as i32,
                E_TOTAL as i32,
                n as i32,
                k as i32,
                MIN_TILE as i32,
                (MIN_TILE * n) as i64,
            )
        };
        assert_eq!(rc, 0, "moe_grouped_fp4_gemv_m1_bf16 rc={rc}");
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&d).unwrap()
}

fn bf16_ulp(x: f32) -> f32 {
    let m = x.abs().max(f32::MIN_POSITIVE);
    let exp = m.log2().floor();
    (exp - 8.0).exp2()
}

fn compare(label: &str, groups: usize, n: usize, cutlass: &[bf16], gemv: &[bf16]) {
    let mut max_abs = 0f64;
    let mut max_ulp = 0f64;
    let mut sum_sq = 0f64;
    let mut ref_sq = 0f64;
    let mut nonzero = 0usize;
    for g in 0..groups {
        let lo = g * MIN_TILE * n;
        for j in 0..n {
            let e = cutlass[lo + j].to_f32();
            let got = gemv[lo + j].to_f32();
            if e != 0.0 {
                nonzero += 1;
            }
            let d = (got - e).abs();
            let ulps = (d / bf16_ulp(e)) as f64;
            if (d as f64) > max_abs {
                max_abs = d as f64;
            }
            if ulps > max_ulp {
                max_ulp = ulps;
            }
            assert!(
                ulps <= 2.0 + 1e-6 || (d as f64) <= 1e-3,
                "{label}: group {g} col {j}: cutlass={e} gemv={got} diff={d} ({ulps:.1} bf16 ulp)"
            );
            sum_sq += (d as f64) * (d as f64);
            ref_sq += (e as f64) * (e as f64);
        }
    }
    let rel_l2 = (sum_sq / ref_sq.max(1e-12)).sqrt();
    eprintln!(
        "  {label}: max_abs={max_abs:.4e} max_ulp={max_ulp:.1} rel_l2={rel_l2:.3e} nonzero={nonzero}/{}",
        groups * n
    );
    assert!(
        nonzero > groups * n / 4,
        "{label}: reference output mostly zero -- CUTLASS arm did not run"
    );
    assert!(rel_l2 <= 1e-3, "{label}: rel_l2={rel_l2} > 1e-3");
}

#[test]
#[ignore]
fn moe_grouped_fp4_gemv_m1_parity_sweep() {
    if std::env::var("NV_MOE_M1_TEST").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_MOE_M1_TEST=1 to run moe_grouped_fp4_gemv_m1_parity_sweep");
        return;
    }
    let ctx = CudaContext::new(0)
        .expect("NV_MOE_M1_TEST=1 but no CUDA device 0 -- this gate panics rather than skipping");
    let major = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .expect("query CC major");
    let minor = ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .expect("query CC minor");
    assert!(
        supports_nvfp4(major) && (major, minor) == (12, 0),
        "NV_MOE_M1_TEST=1 but device is SM {major}.{minor}, not SM 12.0"
    );
    let stream = ctx.default_stream();

    for &(n, k) in &[(512usize, 2048usize), (2048, 512)] {
        for seed in 1u64..=3 {
            let mut rng = Rng::new(seed.wrapping_mul(0x9e3779b97f4a7c15) ^ (n as u64) << 20);
            let w = build_weights(&stream, &mut rng, n, k);
            for &groups in &[1usize, 2, 8, 64, 256] {
                let a = build_acts(&stream, &mut rng, groups, k);
                let cutlass_out = run_cutlass(&stream, &w, &a);
                let gemv_out = run_gemv(&stream, &w, &a);
                let gemv_out2 = run_gemv(&stream, &w, &a);
                let bit_stable = gemv_out.len() == gemv_out2.len()
                    && gemv_out
                        .iter()
                        .zip(gemv_out2.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                assert!(
                    bit_stable,
                    "N={n} K={k} groups={groups} seed={seed}: GEMV is not bit-stable run-to-run"
                );
                let label = format!("N={n} K={k} groups={groups} seed={seed}");
                compare(&label, groups, n, &cutlass_out, &gemv_out);
            }
        }
    }
}
