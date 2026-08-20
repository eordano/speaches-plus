#![cfg(feature = "cuda")]

mod common;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

fn stream(test: &str) -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => {
            if std::env::var("NV_KERNELS_PARITY_REQUIRE").as_deref() == Ok("1") {
                panic!("{test}: no CUDA device 0: {e}");
            }
            eprintln!("{test}: SKIP no CUDA device 0: {e}");
            None
        }
    }
}

#[test]
fn gemv_bf16_normed_dynsmem_k5120_matches_host_prenorm_fold_reference() {
    let Some(st) = stream("gemv_bf16_normed_dynsmem_k5120_matches_host_prenorm_fold_reference")
    else {
        return;
    };
    let mut rng = Lcg(0xda3e39cb94b95bdb);
    let (n, k) = (96usize, 5120usize);
    assert!(
        k > 4096,
        "this suite exists for the dynamic-smem arm; k must exceed the static cap"
    );
    let w = rng.bf16_words(n * k, 0.2);
    let x = rng.bf16_words(k, 1.0);
    let wn = rng.bf16_words(k, 1.0);
    let ssq: f64 = x
        .iter()
        .map(|b| {
            let v = bf16::from_bits(*b).to_f32() as f64;
            v * v
        })
        .sum();
    let rstd = [(1.0 / (ssq / k as f64 + 1e-6).sqrt()) as f32];

    #[allow(deprecated)]
    let dw: CudaSlice<u16> = st.clone_htod(&w).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = st.clone_htod(&x).unwrap();
    #[allow(deprecated)]
    let dn: CudaSlice<u16> = st.clone_htod(&wn).unwrap();
    #[allow(deprecated)]
    let dr: CudaSlice<f32> = st.clone_htod(&rstd).unwrap();
    let mut dy: CudaSlice<u16> = st.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pw, _a) = dw.device_ptr(&st);
        let (px, _b) = dx.device_ptr(&st);
        let (pn, _c) = dn.device_ptr(&st);
        let (pr, _d) = dr.device_ptr(&st);
        let (py, _e) = dy.device_ptr_mut(&st);
        unsafe {
            cuda::gemv_bf16_normed(
                st.cu_stream() as *mut c_void,
                pw as *const u16,
                px as *const u16,
                pn as *const u16,
                pr as *const f32,
                py as *mut u16,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_bf16_normed dyn-smem arm rc={rc} (k={k})");
    st.synchronize().unwrap();
    #[allow(deprecated)]
    let y = st.memcpy_dtov(&dy).unwrap();

    let mut max_rel = 0f64;
    for r in 0..n {
        let mut acc = 0f64;
        for c in 0..k {
            let normed = bf16::from_f32(
                bf16::from_bits(x[c]).to_f32() * rstd[0] * bf16::from_bits(wn[c]).to_f32(),
            )
            .to_f32() as f64;
            acc += bf16::from_bits(w[r * k + c]).to_f32() as f64 * normed;
        }
        let gv = bf16::from_bits(y[r]).to_f32() as f64;
        let rel = (gv - acc).abs() / acc.abs().max(0.25);
        max_rel = max_rel.max(rel);
    }
    eprintln!("gemv_bf16_normed dyn k={k}: max_rel_err_vs_f64_ref={max_rel:.3e}");
    assert!(
        max_rel < 2.0e-2,
        "gemv_bf16_normed dyn-smem arm max rel err {max_rel:.3e} exceeds 2e-2"
    );
}
