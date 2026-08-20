#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_quant::nvfp4::{decode_e2m1, decode_ue4m3, unswizzle_scales, BLOCK_SIZE};
use std::ffi::c_void;
use std::sync::Arc;

#[path = "nvfp4_true_m_common.rs"]
mod common;
use common::quantize_dev;

pub const MK_MMA_IS_A_FLOOR_PROBE_NO_SERVING_ROUTE_CONSUMES_IT: &str =
    "the padded-LT verify-mlp route out-streams this m<=16 mma arm at q38 dense shapes (bench \
     q38_verify_mlp_narrow_kernel_ab; current numbers: perf/runs.jsonl) and, because verify \
     acceptance is argmax-exactness against the drafter and not precision, the mma arm's \
     bf16-activation verify collapses high-accept brackets, so it stays a parity and floor \
     probe only";

struct Lcg(u64);

impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn next_f32(&mut self) -> f32 {
        ((self.next_u32() & 0x7f_ffff) as f32 / 8388608.0) * 2.0 - 1.0
    }
    fn bf16_words(&mut self, n: usize, gain: f32) -> Vec<u16> {
        (0..n)
            .map(|_| bf16::from_f32(self.next_f32() * gain).to_bits())
            .collect()
    }
}

fn cpu_ref_w4a16(
    x_bf: &[u16],
    packed: &[u8],
    scales_sw: &[u8],
    alpha: f32,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let blocks = k / BLOCK_SIZE;
    let scales_lin = unswizzle_scales(scales_sw, n, blocks);
    let mut w_dec = vec![0f32; n * k];
    for r in 0..n {
        for b in 0..blocks {
            let sf = decode_ue4m3(scales_lin[r * blocks + b]);
            for i in 0..BLOCK_SIZE / 2 {
                let byte = packed[r * k / 2 + b * BLOCK_SIZE / 2 + i];
                let lo = bf16::from_f32(decode_e2m1(byte & 0x0F) * sf).to_f32();
                let hi = bf16::from_f32(decode_e2m1(byte >> 4) * sf).to_f32();
                w_dec[r * k + b * BLOCK_SIZE + i * 2] = lo;
                w_dec[r * k + b * BLOCK_SIZE + i * 2 + 1] = hi;
            }
        }
    }
    let mut y = vec![0f32; m * n];
    for mi in 0..m {
        for r in 0..n {
            let mut acc = 0f64;
            for kk in 0..k {
                let xv = bf16::from_bits(x_bf[mi * k + kk]).to_f64();
                acc += xv * w_dec[r * k + kk] as f64;
            }
            y[mi * n + r] = (acc as f32) * alpha;
        }
    }
    y
}

fn rel_rms(got: &[f32], expect: &[f32]) -> f64 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (g, e) in got.iter().zip(expect.iter()) {
        num += (*g as f64 - *e as f64).powi(2);
        den += (*e as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn run_dual(
    stream: &Arc<CudaStream>,
    wq_a: &CudaSlice<u8>,
    sc_a: &CudaSlice<u8>,
    wq_b: Option<(&CudaSlice<u8>, &CudaSlice<u8>)>,
    x: &CudaSlice<u16>,
    alpha_a: f32,
    alpha_b: f32,
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<u16>, Option<Vec<u16>>) {
    let mut ya: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let mut yb: Option<CudaSlice<u16>> = wq_b.map(|_| stream.alloc_zeros::<u16>(m * n).unwrap());
    let rc = {
        let (pwa, _a) = wq_a.device_ptr(stream);
        let (psa, _b) = sc_a.device_ptr(stream);
        let (pwb, psb) = match wq_b {
            Some((w, s)) => {
                let (pw, _gw) = w.device_ptr(stream);
                let (ps, _gs) = s.device_ptr(stream);
                (pw, ps)
            }
            None => (0u64, 0u64),
        };
        let (px, _c) = x.device_ptr(stream);
        let (pya, _d) = ya.device_ptr_mut(stream);
        let pyb = match yb.as_mut() {
            Some(b) => {
                let (p, _g) = b.device_ptr_mut(stream);
                p
            }
            None => 0u64,
        };
        unsafe {
            nv_kernels::cuda::gemm_nvfp4_w4a16_mk_dual(
                stream.cu_stream() as *mut c_void,
                pwa as *const u8,
                psa as *const u8,
                pwb as *const u8,
                psb as *const u8,
                px as *const u16,
                pya as *mut u16,
                pyb as *mut u16,
                alpha_a,
                alpha_b,
                m as i32,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemm_nvfp4_w4a16_mk_dual rc={rc} m={m} n={n} k={k}");
    let ha: Vec<u16> = stream.memcpy_dtov(&ya).unwrap();
    let hb = yb.map(|b| stream.memcpy_dtov(&b).unwrap());
    (ha, hb)
}

#[test]
fn gemm_w4a16_mk_mma_matches_cpu_reference() {
    let Ok(ctx) = CudaContext::new(0) else {
        eprintln!("skip: no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x2545f4914f6cdd1d);

    for (n, k) in [(64usize, 128usize), (96, 320), (8448, 256)] {
        let w_host = rng.bf16_words(n * k, 0.05);
        #[allow(deprecated)]
        let w_dev: CudaSlice<u16> = stream.clone_htod(&w_host).unwrap();
        let (wq, wsf) = quantize_dev(&stream, &w_dev, n, n, k, 1.0);
        let wq_h: Vec<u8> = stream.memcpy_dtov(&wq).unwrap();
        let wsf_h: Vec<u8> = stream.memcpy_dtov(&wsf).unwrap();

        for m in [1usize, 2, 3, 5, 8, 11, 16] {
            let x_host = rng.bf16_words(m * k, 1.0);
            #[allow(deprecated)]
            let x_dev: CudaSlice<u16> = stream.clone_htod(&x_host).unwrap();
            let alpha = 0.37f32;
            let (ya, _) = run_dual(
                &stream, &wq, &wsf, None, &x_dev, alpha, 0.0, m, n, k,
            );
            let ya_f: Vec<f32> = ya
                .iter()
                .map(|v| bf16::from_bits(*v).to_f32())
                .collect();
            let expect = cpu_ref_w4a16(&x_host, &wq_h, &wsf_h, alpha, m, n, k);
            let rr = rel_rms(&ya_f, &expect);
            assert!(
                rr < 2e-3,
                "mk mma diverged from cpu ref: m={m} n={n} k={k} rel_rms={rr}"
            );
        }
    }

    {
        let (n, k) = (17408usize, 5120usize);
        let m = 8usize;
        let wa_host = rng.bf16_words(n * k, 0.05);
        let wb_host = rng.bf16_words(n * k, 0.05);
        #[allow(deprecated)]
        let wa_dev: CudaSlice<u16> = stream.clone_htod(&wa_host).unwrap();
        #[allow(deprecated)]
        let wb_dev: CudaSlice<u16> = stream.clone_htod(&wb_host).unwrap();
        let (wqa, wsa) = quantize_dev(&stream, &wa_dev, n, n, k, 1.0);
        let (wqb, wsb) = quantize_dev(&stream, &wb_dev, n, n, k, 1.0);
        let wqa_h: Vec<u8> = stream.memcpy_dtov(&wqa).unwrap();
        let wsa_h: Vec<u8> = stream.memcpy_dtov(&wsa).unwrap();
        let wqb_h: Vec<u8> = stream.memcpy_dtov(&wqb).unwrap();
        let wsb_h: Vec<u8> = stream.memcpy_dtov(&wsb).unwrap();
        let x_host = rng.bf16_words(m * k, 1.0);
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_host).unwrap();
        let (ya, yb) = run_dual(
            &stream,
            &wqa,
            &wsa,
            Some((&wqb, &wsb)),
            &x_dev,
            1.1,
            0.9,
            m,
            n,
            k,
        );
        let ya_f: Vec<f32> = ya.iter().map(|v| bf16::from_bits(*v).to_f32()).collect();
        let yb_f: Vec<f32> = yb
            .unwrap()
            .iter()
            .map(|v| bf16::from_bits(*v).to_f32())
            .collect();
        let ea = cpu_ref_w4a16(&x_host, &wqa_h, &wsa_h, 1.1, m, n, k);
        let eb = cpu_ref_w4a16(&x_host, &wqb_h, &wsb_h, 0.9, m, n, k);
        let ra = rel_rms(&ya_f, &ea);
        let rb = rel_rms(&yb_f, &eb);
        assert!(ra < 2e-3, "dual arm a diverged: rel_rms={ra}");
        assert!(rb < 2e-3, "dual arm b diverged: rel_rms={rb}");
    }

    {
        let (n, k) = (5120usize, 17408usize);
        let m = 6usize;
        let w_host = rng.bf16_words(n * k, 0.05);
        #[allow(deprecated)]
        let w_dev: CudaSlice<u16> = stream.clone_htod(&w_host).unwrap();
        let (wq, wsf) = quantize_dev(&stream, &w_dev, n, n, k, 1.0);
        let wq_h: Vec<u8> = stream.memcpy_dtov(&wq).unwrap();
        let wsf_h: Vec<u8> = stream.memcpy_dtov(&wsf).unwrap();
        let x_host = rng.bf16_words(m * k, 1.0);
        #[allow(deprecated)]
        let x_dev: CudaSlice<u16> = stream.clone_htod(&x_host).unwrap();
        let (y, _) = run_dual(&stream, &wq, &wsf, None, &x_dev, 1.0, 0.0, m, n, k);
        let y_f: Vec<f32> = y.iter().map(|v| bf16::from_bits(*v).to_f32()).collect();
        let expect = cpu_ref_w4a16(&x_host, &wq_h, &wsf_h, 1.0, m, n, k);
        let rr = rel_rms(&y_f, &expect);
        assert!(rr < 2e-3, "down-shape mk mma diverged: rel_rms={rr}");
    }
}
