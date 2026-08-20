#![cfg(feature = "cuda")]

mod common;
use common::LcgMask23TwoSided as Lcg;
use common::stream;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

fn run_gemv_e4m3_mk(
    stream: &Arc<CudaStream>,
    wq: &[u8],
    row_scale: &[f32],
    x: &[u16],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dw: CudaSlice<u8> = stream.clone_htod(wq).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<f32> = stream.clone_htod(row_scale).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(x).unwrap();
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
    assert_eq!(rc, 0, "gemv_e4m3_mk rc={rc} (n={n} k={k} m={m})");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&dy).unwrap()
}

fn host_reference_f64(
    wq: &[u8],
    row_scale: &[f32],
    x: &[u16],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<f64> {
    let wf = nv_quant::fp8::dequantize_e4m3_per_row(wq, n, k, row_scale).unwrap();
    let mut y = vec![0f64; m * n];
    for j in 0..m {
        for r in 0..n {
            let mut acc = 0f64;
            for c in 0..k {
                let xv = bf16::from_bits(x[j * k + c]).to_f32() as f64;
                acc += wf[r * k + c] as f64 * xv;
            }
            y[j * n + r] = acc;
        }
    }
    y
}

fn assert_close_and_argmax(
    name: &str,
    got: &[u16],
    reference: &[f64],
    n: usize,
    m: usize,
) {
    let mut max_rel = 0f64;
    for (g, r) in got.iter().zip(reference.iter()) {
        let gv = bf16::from_bits(*g).to_f32() as f64;
        let denom = r.abs().max(0.25);
        max_rel = max_rel.max((gv - r).abs() / denom);
    }
    for j in 0..m {
        let row_ref = &reference[j * n..(j + 1) * n];
        let row_got = &got[j * n..(j + 1) * n];
        let am_ref = row_ref
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        let am_got = row_got
            .iter()
            .map(|g| bf16::from_bits(*g).to_f32())
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            am_ref, am_got,
            "{name}: argmax disagrees on output row {j} (m={m}, n={n})"
        );
    }
    eprintln!("{name}: max_rel_err_vs_f64_ref={max_rel:.3e} over {}x{n}", m);
    assert!(
        max_rel < 2.0e-2,
        "{name}: max rel err {max_rel:.3e} exceeds the bf16-output tolerance 2e-2"
    );
}

#[test]
fn gemv_e4m3_mk_matches_host_dequant_reference_at_m1_and_m4() {
    let Some(st) = stream("gemv_e4m3_mk_matches_host_dequant_reference_at_m1_and_m4") else {
        return;
    };
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    for (n, k) in [(1024usize, 512usize), (1000, 5120), (64, 6144)] {
        let w_bf = rng.bf16_words(n * k, 0.05);
        let w_host: Vec<bf16> = w_bf.iter().map(|b| bf16::from_bits(*b)).collect();
        let (wq, scales) = nv_quant::fp8::quantize_e4m3_per_row(&w_host, n, k).unwrap();
        for m in [1usize, 4] {
            let x = rng.bf16_words(m * k, 1.0);
            let got = run_gemv_e4m3_mk(&st, &wq, &scales, &x, n, k, m);
            let reference = host_reference_f64(&wq, &scales, &x, n, k, m);
            assert_close_and_argmax(
                &format!("gemv_e4m3_mk n={n} k={k} m={m}"),
                &got,
                &reference,
                n,
                m,
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_gemv_e4m3_mk_h(
    stream: &Arc<CudaStream>,
    wq: &[u8],
    row_scale: &[f32],
    x: &[u16],
    wn: &[u16],
    rstd: &[f32],
    n: usize,
    k: usize,
    m: usize,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dw: CudaSlice<u8> = stream.clone_htod(wq).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<f32> = stream.clone_htod(row_scale).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(x).unwrap();
    #[allow(deprecated)]
    let dn: CudaSlice<u16> = stream.clone_htod(wn).unwrap();
    #[allow(deprecated)]
    let dr: CudaSlice<f32> = stream.clone_htod(rstd).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let rc = {
        let (pw, _a) = dw.device_ptr(stream);
        let (ps, _b) = ds.device_ptr(stream);
        let (px, _c) = dx.device_ptr(stream);
        let (pn, _e) = dn.device_ptr(stream);
        let (pr, _f) = dr.device_ptr(stream);
        let (py, _d) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_e4m3_mk_h(
                stream.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const f32,
                px as *const u16,
                pn as *const u16,
                pr as *const f32,
                py as *mut u16,
                n as i32,
                k as i32,
                m as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_e4m3_mk_h rc={rc} (n={n} k={k} m={m})");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    stream.memcpy_dtov(&dy).unwrap()
}

#[test]
fn gemv_e4m3_mk_h_prenorm_fold_matches_dead_rmsnorm_then_mk_reference_at_m1_and_m4() {
    let Some(st) =
        stream("gemv_e4m3_mk_h_prenorm_fold_matches_dead_rmsnorm_then_mk_reference_at_m1_and_m4")
    else {
        return;
    };
    let mut rng = Lcg(0x2545f4914f6cdd1d);
    for (n, k) in [(1000usize, 5120usize), (6144, 5120)] {
        let w_bf = rng.bf16_words(n * k, 0.05);
        let w_host: Vec<bf16> = w_bf.iter().map(|b| bf16::from_bits(*b)).collect();
        let (wq, scales) = nv_quant::fp8::quantize_e4m3_per_row(&w_host, n, k).unwrap();
        let wn = rng.bf16_words(k, 1.0);
        for m in [1usize, 4] {
            let x = rng.bf16_words(m * k, 1.0);
            let rstd: Vec<f32> = (0..m)
                .map(|j| {
                    let ssq: f64 = x[j * k..(j + 1) * k]
                        .iter()
                        .map(|b| {
                            let v = bf16::from_bits(*b).to_f32() as f64;
                            v * v
                        })
                        .sum();
                    (1.0 / (ssq / k as f64 + 1e-6).sqrt()) as f32
                })
                .collect();
            let x_normed_rounded_like_the_kernel_smem_stage: Vec<u16> = (0..m * k)
                .map(|i| {
                    let j = i / k;
                    let c = i % k;
                    let v = bf16::from_bits(x[i]).to_f32()
                        * rstd[j]
                        * bf16::from_bits(wn[c]).to_f32();
                    bf16::from_f32(v).to_bits()
                })
                .collect();
            let got = run_gemv_e4m3_mk_h(&st, &wq, &scales, &x, &wn, &rstd, n, k, m);
            let reference = host_reference_f64(
                &wq,
                &scales,
                &x_normed_rounded_like_the_kernel_smem_stage,
                n,
                k,
                m,
            );
            assert_close_and_argmax(
                &format!("gemv_e4m3_mk_h n={n} k={k} m={m}"),
                &got,
                &reference,
                n,
                m,
            );
        }
    }
}

#[test]
fn gemv_e4m3_mk_rows2_arm_on_by_default_matches_host_reference_at_m4_odd_n() {
    let Some(st) = stream("gemv_e4m3_mk_rows2_arm_on_by_default_matches_host_reference_at_m4_odd_n")
    else {
        return;
    };
    assert!(
        std::env::var("NV_Q38_E4M3_MK_R2_OFF").is_err(),
        "unset NV_Q38_E4M3_MK_R2_OFF: this test pins the rows2 dispatch default"
    );
    let mut rng = Lcg(0x853c49e6748fea9b);
    let (n, k, m) = (1000usize, 5120usize, 4usize);
    let w_bf = rng.bf16_words(n * k, 0.05);
    let w_host: Vec<bf16> = w_bf.iter().map(|b| bf16::from_bits(*b)).collect();
    let (wq, scales) = nv_quant::fp8::quantize_e4m3_per_row(&w_host, n, k).unwrap();
    let x = rng.bf16_words(m * k, 1.0);
    let got_rows2 = run_gemv_e4m3_mk(&st, &wq, &scales, &x, n, k, m);
    let reference = host_reference_f64(&wq, &scales, &x, n, k, m);
    assert_close_and_argmax(
        &format!("gemv_e4m3_mk rows2 n={n} k={k} m={m}"),
        &got_rows2,
        &reference,
        n,
        m,
    );
}

#[test]
fn gemv_e4m3_mk_refuses_ragged_k_and_oversized_m_without_launching() {
    let Some(st) = stream("gemv_e4m3_mk_refuses_ragged_k_and_oversized_m_without_launching") else {
        return;
    };
    let n = 8usize;
    let k = 32usize;
    let wq = vec![0u8; n * k];
    let scales = vec![1f32; n];
    let x = vec![0u16; k];
    #[allow(deprecated)]
    let dw: CudaSlice<u8> = st.clone_htod(&wq).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<f32> = st.clone_htod(&scales).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = st.clone_htod(&x).unwrap();
    let mut dy: CudaSlice<u16> = st.alloc_zeros::<u16>(n).unwrap();
    let (pw, _a) = dw.device_ptr(&st);
    let (ps, _b) = ds.device_ptr(&st);
    let (px, _c) = dx.device_ptr(&st);
    let (py, _d) = dy.device_ptr_mut(&st);
    let rc_ragged = unsafe {
        cuda::gemv_e4m3_mk(
            st.cu_stream() as *mut c_void,
            pw as *const u8,
            ps as *const f32,
            px as *const u16,
            py as *mut u16,
            n as i32,
            (k - 8) as i32,
            1,
        )
    };
    assert_eq!(rc_ragged, -1, "K%16!=0 must be refused with -1");
    let rc_big_m = unsafe {
        cuda::gemv_e4m3_mk(
            st.cu_stream() as *mut c_void,
            pw as *const u8,
            ps as *const f32,
            px as *const u16,
            py as *mut u16,
            n as i32,
            k as i32,
            17,
        )
    };
    assert_eq!(rc_big_m, -1, "M>16 must be refused with -1");
}
