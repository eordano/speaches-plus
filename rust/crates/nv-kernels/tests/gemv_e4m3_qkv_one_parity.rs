#![cfg(feature = "cuda")]

mod common;
use common::stream;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

fn gemv_m1_separate(
    st: &Arc<CudaStream>,
    wq: &[u8],
    scales: &[f32],
    x_dev: &CudaSlice<u16>,
    n: usize,
    k: usize,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dw: CudaSlice<u8> = st.clone_htod(wq).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<f32> = st.clone_htod(scales).unwrap();
    let mut dy: CudaSlice<u16> = st.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pw, _a) = dw.device_ptr(st);
        let (ps, _b) = ds.device_ptr(st);
        let (px, _c) = x_dev.device_ptr(st);
        let (py, _d) = dy.device_ptr_mut(st);
        unsafe {
            cuda::gemv_e4m3_mk(
                st.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const f32,
                px as *const u16,
                py as *mut u16,
                n as i32,
                k as i32,
                1i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_e4m3_mk m=1 rc={rc} (n={n} k={k})");
    st.synchronize().unwrap();
    #[allow(deprecated)]
    st.memcpy_dtov(&dy).unwrap()
}

#[test]
fn qkv_one_launch_is_bitwise_equal_to_three_separate_m1_rows2_launches() {
    let Some(st) = stream("qkv_one_launch_is_bitwise_equal_to_three_separate_m1_rows2_launches")
    else {
        return;
    };
    let mut rng = Lcg(0x2545f4914f6cdd1d);
    for (n_q, n_k, n_v, k) in [
        (12288usize, 1024usize, 1024usize, 5120usize),
        (256, 64, 48, 1024),
    ] {
        let mk = |rng: &mut Lcg, n: usize| {
            let w_bf = rng.bf16_words(n * k, 0.05);
            let w_host: Vec<bf16> = w_bf.iter().map(|b| bf16::from_bits(*b)).collect();
            nv_quant::fp8::quantize_e4m3_per_row(&w_host, n, k).unwrap()
        };
        let (wq_q, sc_q) = mk(&mut rng, n_q);
        let (wq_k, sc_k) = mk(&mut rng, n_k);
        let (wq_v, sc_v) = mk(&mut rng, n_v);
        let x = rng.bf16_words(k, 1.0);
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = st.clone_htod(&x).unwrap();

        let ref_q = gemv_m1_separate(&st, &wq_q, &sc_q, &dx, n_q, k);
        let ref_k = gemv_m1_separate(&st, &wq_k, &sc_k, &dx, n_k, k);
        let ref_v = gemv_m1_separate(&st, &wq_v, &sc_v, &dx, n_v, k);

        #[allow(deprecated)]
        let dwq: CudaSlice<u8> = st.clone_htod(&wq_q).unwrap();
        #[allow(deprecated)]
        let dsq: CudaSlice<f32> = st.clone_htod(&sc_q).unwrap();
        #[allow(deprecated)]
        let dwk: CudaSlice<u8> = st.clone_htod(&wq_k).unwrap();
        #[allow(deprecated)]
        let dsk: CudaSlice<f32> = st.clone_htod(&sc_k).unwrap();
        #[allow(deprecated)]
        let dwv: CudaSlice<u8> = st.clone_htod(&wq_v).unwrap();
        #[allow(deprecated)]
        let dsv: CudaSlice<f32> = st.clone_htod(&sc_v).unwrap();
        let mut dyq: CudaSlice<u16> = st.alloc_zeros::<u16>(n_q).unwrap();
        let mut dyk: CudaSlice<u16> = st.alloc_zeros::<u16>(n_k).unwrap();
        let mut dyv: CudaSlice<u16> = st.alloc_zeros::<u16>(n_v).unwrap();
        let rc = {
            let (pwq, _a) = dwq.device_ptr(&st);
            let (psq, _b) = dsq.device_ptr(&st);
            let (pwk, _c) = dwk.device_ptr(&st);
            let (psk, _d) = dsk.device_ptr(&st);
            let (pwv, _e) = dwv.device_ptr(&st);
            let (psv, _f) = dsv.device_ptr(&st);
            let (px, _g) = dx.device_ptr(&st);
            let (pyq, _h) = dyq.device_ptr_mut(&st);
            let (pyk, _i) = dyk.device_ptr_mut(&st);
            let (pyv, _j) = dyv.device_ptr_mut(&st);
            unsafe {
                cuda::gemv_e4m3_qkv_one_m1(
                    st.cu_stream() as *mut c_void,
                    pwq as *const u8,
                    psq as *const f32,
                    pwk as *const u8,
                    psk as *const f32,
                    pwv as *const u8,
                    psv as *const f32,
                    px as *const u16,
                    pyq as *mut u16,
                    pyk as *mut u16,
                    pyv as *mut u16,
                    n_q as i32,
                    n_k as i32,
                    n_v as i32,
                    k as i32,
                )
            }
        };
        assert_eq!(rc, 0, "gemv_e4m3_qkv_one_m1 rc={rc} ({n_q}+{n_k}+{n_v} x {k})");
        st.synchronize().unwrap();
        #[allow(deprecated)]
        let got_q = st.memcpy_dtov(&dyq).unwrap();
        #[allow(deprecated)]
        let got_k = st.memcpy_dtov(&dyk).unwrap();
        #[allow(deprecated)]
        let got_v = st.memcpy_dtov(&dyv).unwrap();

        assert_eq!(
            got_q, ref_q,
            "q segment not bitwise equal to the standalone rows2 launch ({n_q}x{k})"
        );
        assert_eq!(
            got_k, ref_k,
            "k segment not bitwise equal to the standalone rows2 launch ({n_k}x{k})"
        );
        assert_eq!(
            got_v, ref_v,
            "v segment not bitwise equal to the standalone rows2 launch ({n_v}x{k})"
        );
        eprintln!(
            "qkv_one parity: bitwise equal on ({n_q}+{n_k}+{n_v})x{k} vs three separate launches"
        );
    }
}
