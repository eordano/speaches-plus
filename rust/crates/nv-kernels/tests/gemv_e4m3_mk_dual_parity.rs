#![cfg(feature = "cuda")]

mod common;
use common::assert_u16_bits;
use common::stream;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const QWEN36_GDN_GEOMETRY_CONV8192_VAL4096_K2048: (usize, usize, usize) = (8192, 4096, 2048);
const QWEN38_GDN_GEOMETRY_CONV10240_VAL6144_K5120: (usize, usize, usize) = (10240, 6144, 5120);
const SMALL_HOST_REF_GEOMETRY_N32_N16_K256: (usize, usize, usize) = (32, 16, 256);
const M_SWEEP_COVERS_M1_ROWS2_2_TO_6_AND_XF32_7_TO_16: [usize; 8] = [1, 2, 3, 4, 5, 6, 7, 16];

struct DualInputs {
    wq_a: Vec<u8>,
    rs_a: Vec<f32>,
    wq_b: Vec<u8>,
    rs_b: Vec<f32>,
    x: Vec<u16>,
}

fn inputs(n_a: usize, n_b: usize, k: usize, m: usize, seed: u64) -> DualInputs {
    let mut lcg = Lcg(seed);
    DualInputs {
        wq_a: lcg.packed_nibbles(n_a * k),
        rs_a: (0..n_a).map(|_| lcg.next_f32().abs() * 0.05 + 1e-3).collect(),
        wq_b: lcg.packed_nibbles(n_b * k),
        rs_b: (0..n_b).map(|_| lcg.next_f32().abs() * 0.05 + 1e-3).collect(),
        x: lcg.bf16_words(m * k, 1.0),
    }
}

fn run_separate_mk(
    stream: &Arc<CudaStream>,
    inp: &DualInputs,
    n_a: usize,
    n_b: usize,
    k: usize,
    m: usize,
) -> Result<(Vec<u16>, Vec<u16>), i32> {
    let mut out = Vec::with_capacity(2);
    for (wq, rs, n) in [(&inp.wq_a, &inp.rs_a, n_a), (&inp.wq_b, &inp.rs_b, n_b)] {
        #[allow(deprecated)]
        let dw: CudaSlice<u8> = stream.clone_htod(wq).unwrap();
        #[allow(deprecated)]
        let ds: CudaSlice<f32> = stream.clone_htod(rs).unwrap();
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.clone_htod(&inp.x).unwrap();
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
        let rc = {
            let (pw, _a) = dw.device_ptr(stream);
            let (ps, _b) = ds.device_ptr(stream);
            let (px, _c) = dx.device_ptr(stream);
            let (py, _d) = dy.device_ptr_mut(stream);
            unsafe {
                cuda::gemv_e4m3_mk(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u8,
                    ps as *const f32,
                    px as *const u16,
                    py as *mut u16,
                    n as i32,
                    k as i32,
                    m as i32,
                )
            }
        };
        if rc != 0 {
            return Err(rc);
        }
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        out.push(stream.memcpy_dtov(&dy).unwrap());
    }
    let b = out.pop().unwrap();
    let a = out.pop().unwrap();
    Ok((a, b))
}

fn run_dual(
    stream: &Arc<CudaStream>,
    inp: &DualInputs,
    n_a: usize,
    n_b: usize,
    k: usize,
    m: usize,
) -> Result<(Vec<u16>, Vec<u16>), i32> {
    #[allow(deprecated)]
    let dwa: CudaSlice<u8> = stream.clone_htod(&inp.wq_a).unwrap();
    #[allow(deprecated)]
    let dsa: CudaSlice<f32> = stream.clone_htod(&inp.rs_a).unwrap();
    #[allow(deprecated)]
    let dwb: CudaSlice<u8> = stream.clone_htod(&inp.wq_b).unwrap();
    #[allow(deprecated)]
    let dsb: CudaSlice<f32> = stream.clone_htod(&inp.rs_b).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&inp.x).unwrap();
    let mut dya: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n_a).unwrap();
    let mut dyb: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n_b).unwrap();
    let rc = {
        let (pwa, _a) = dwa.device_ptr(stream);
        let (psa, _b) = dsa.device_ptr(stream);
        let (pwb, _c) = dwb.device_ptr(stream);
        let (psb, _d) = dsb.device_ptr(stream);
        let (px, _e) = dx.device_ptr(stream);
        let (pya, _f) = dya.device_ptr_mut(stream);
        let (pyb, _g) = dyb.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_e4m3_mk_dual(
                stream.cu_stream() as *mut c_void,
                pwa as *const u8,
                psa as *const f32,
                pwb as *const u8,
                psb as *const f32,
                px as *const u16,
                pya as *mut u16,
                pyb as *mut u16,
                n_a as i32,
                n_b as i32,
                k as i32,
                m as i32,
            )
        }
    };
    if rc != 0 {
        return Err(rc);
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let a = stream.memcpy_dtov(&dya).unwrap();
    #[allow(deprecated)]
    let b = stream.memcpy_dtov(&dyb).unwrap();
    Ok((a, b))
}

#[test]
fn dual_is_bitwise_identical_to_two_separate_mk_launches_across_the_m_dispatch_ladder() {
    let Some(st) =
        stream("dual_is_bitwise_identical_to_two_separate_mk_launches_across_the_m_dispatch_ladder")
    else {
        return;
    };
    for (label, (n_a, n_b, k)) in [
        ("qwen36", QWEN36_GDN_GEOMETRY_CONV8192_VAL4096_K2048),
        ("qwen38", QWEN38_GDN_GEOMETRY_CONV10240_VAL6144_K5120),
    ] {
        for &m in &M_SWEEP_COVERS_M1_ROWS2_2_TO_6_AND_XF32_7_TO_16 {
            let inp = inputs(n_a, n_b, k, m, 0x5eed_0000 + m as u64);
            let sep = match run_separate_mk(&st, &inp, n_a, n_b, k, m) {
                Ok(pair) => pair,
                Err(-1) => {
                    eprintln!(
                        "{label} m={m}: the separate mk baseline itself refuses (-1, bf16 smem \
                         over the optin limit); the dual must refuse too"
                    );
                    match run_dual(&st, &inp, n_a, n_b, k, m) {
                        Err(-1) => continue,
                        other => panic!(
                            "{label} m={m}: dual must refuse when the baseline does, got {:?}",
                            other.map(|(a, b)| (a.len(), b.len()))
                        ),
                    }
                }
                Err(rc) => panic!("{label} m={m}: gemv_e4m3_mk rc={rc}"),
            };
            let (sep_a, sep_b) = sep;
            let dual = match run_dual(&st, &inp, n_a, n_b, k, m) {
                Ok(pair) => pair,
                Err(-1) => {
                    let smem_f32 = m * (k >> 4) * 9 * 8;
                    eprintln!(
                        "{label} m={m}: dual refused (-1, smem_f32={smem_f32}); the call site \
                         falls back to two launches, nothing to compare"
                    );
                    continue;
                }
                Err(rc) => panic!("{label} m={m}: gemv_e4m3_mk_dual rc={rc}"),
            };
            assert_u16_bits(&format!("{label} m={m} qkv-half"), &dual.0, &sep_a);
            assert_u16_bits(&format!("{label} m={m} z-half"), &dual.1, &sep_b);
            eprintln!("{label} m={m}: dual == separate bitwise ({} + {} words)", sep_a.len(), sep_b.len());
        }
    }
}

#[test]
fn dual_matches_host_f64_dequant_reference_so_bitwise_agreement_cannot_hide_a_shared_wrong_answer()
{
    let Some(st) = stream(
        "dual_matches_host_f64_dequant_reference_so_bitwise_agreement_cannot_hide_a_shared_wrong_answer",
    ) else {
        return;
    };
    let (n_a, n_b, k) = SMALL_HOST_REF_GEOMETRY_N32_N16_K256;
    for &m in &[1usize, 3, 8] {
        let inp = inputs(n_a, n_b, k, m, 0xabc0 + m as u64);
        let (dual_a, dual_b) = run_dual(&st, &inp, n_a, n_b, k, m)
            .unwrap_or_else(|rc| panic!("small geometry must be admitted, rc={rc}"));
        for (name, got, wq, rs, n) in [
            ("a", &dual_a, &inp.wq_a, &inp.rs_a, n_a),
            ("b", &dual_b, &inp.wq_b, &inp.rs_b, n_b),
        ] {
            let wf = nv_quant::fp8::dequantize_e4m3_per_row(wq, n, k, rs).unwrap();
            let mut max_rel = 0f64;
            for j in 0..m {
                for r in 0..n {
                    let mut acc = 0f64;
                    for c in 0..k {
                        let xv = bf16::from_bits(inp.x[j * k + c]).to_f32() as f64;
                        acc += wf[r * k + c] as f64 * xv;
                    }
                    let gv = bf16::from_bits(got[j * n + r]).to_f32() as f64;
                    let denom = acc.abs().max(0.25);
                    max_rel = max_rel.max((gv - acc).abs() / denom);
                }
            }
            eprintln!("m={m} half={name}: max_rel_err_vs_f64_ref={max_rel:.3e}");
            assert!(
                max_rel < 2.0e-2,
                "m={m} half={name}: max rel err {max_rel:.3e} exceeds the bf16-output tolerance"
            );
        }
    }
}

#[test]
fn dual_refuses_row_counts_off_the_16_row_block_quantum_instead_of_corrupting_the_pairing() {
    let Some(st) = stream(
        "dual_refuses_row_counts_off_the_16_row_block_quantum_instead_of_corrupting_the_pairing",
    ) else {
        return;
    };
    let (n_a, n_b, k) = (24usize, 16usize, 256usize);
    let inp = inputs(n_a, n_b, k, 2, 0x77);
    match run_dual(&st, &inp, n_a, n_b, k, 2) {
        Err(-1) => {}
        other => panic!("n_a=24 must be refused with -1, got {other:?}"),
    }
}
