#![cfg(feature = "cuda")]

mod common;
use common::LcgOddSeedShift32F64TwoSided as Lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::sync::Arc;

fn stream() -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(c) => Some(c.default_stream()),
        Err(e) => {
            if std::env::var("NV_KERNELS_W4A16_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_KERNELS_W4A16_ALLOW_SKIP=1): no CUDA device 0: {e}");
                return None;
            }
            panic!(
                "gemv_w4a16_cpu_ref: no CUDA device 0: {e}. This is a correctness gate; \
                 it refuses to report success without running. Set \
                 NV_KERNELS_W4A16_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

fn gen_inputs(n: usize, k: usize, gs: usize, seed: u64) -> (Vec<u32>, Vec<u16>, Vec<u16>) {
    let mut rng = Lcg::new(seed);
    let packed: Vec<u32> = (0..n * k / 8).map(|_| rng.next_u32()).collect();
    let scales: Vec<u16> = (0..n * (k / gs))
        .map(|_| bf16::from_f32(0.005 + 0.01 * rng.next_f32().abs()).to_bits())
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect();
    (packed, scales, x)
}

fn cpu_ref(packed: &[u32], scales: &[u16], x: &[u16], n: usize, k: usize, gs: usize) -> Vec<f32> {
    let xf: Vec<f32> = x.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
    let sf: Vec<f32> = scales
        .iter()
        .map(|&b| bf16::from_bits(b).to_f32())
        .collect();
    let kw = k / 8;
    let kg = k / gs;
    let mut y = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f64;
        for kk in 0..k {
            let word = packed[row * kw + kk / 8];
            let q = ((word >> (4 * (kk % 8))) & 0xF) as i32 - 8;
            acc += (q as f32 * sf[row * kg + kk / gs] * xf[kk]) as f64;
        }
        y[row] = acc as f32;
    }
    y
}

fn gelu_tanh(v: f32) -> f32 {
    let c = 0.797_884_6f32;
    0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
}

fn max_rel(got: &[u16], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "compared buffers differ in length");
    assert!(!got.is_empty(), "nothing was compared");
    assert!(
        want.iter().any(|v| v.abs() > 1e-3),
        "the reference is all zeros; the comparison would be vacuous"
    );
    let mut worst = 0f32;
    for (i, (&g, &e)) in got.iter().zip(want.iter()).enumerate() {
        let gf = bf16::from_bits(g).to_f32();

        assert!(
            gf.is_finite(),
            "row {i}: kernel output is {gf} -- the NaN poison survived, so this row was \
             never written (or the kernel produced a non-finite value)"
        );
        let rel = (gf - e).abs() / e.abs().max(0.5);
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

#[allow(clippy::too_many_arguments)]
fn run_gemv(
    stream: &Arc<CudaStream>,
    packed: &[u32],
    scales: &[u16],
    x: &[u16],
    n: usize,
    k: usize,
    gs: usize,
) -> (i32, Vec<u16>) {
    #[allow(deprecated)]
    let dp: CudaSlice<u32> = stream.memcpy_stod(packed).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<u16> = stream.memcpy_stod(scales).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.memcpy_stod(x).unwrap();

    #[allow(deprecated)]
    let mut dy: CudaSlice<u16> = stream.memcpy_stod(&vec![0x7FC0u16; n]).unwrap();
    let rc = {
        let (pp, _g1) = dp.device_ptr(stream);
        let (sp, _g2) = ds.device_ptr(stream);
        let (xp, _g3) = dx.device_ptr(stream);
        let (yp, _g4) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_w4a16(
                stream.cu_stream() as *mut _,
                pp as *const u32,
                sp as *const u16,
                xp as *const u16,
                yp as *mut u16,
                n as i32,
                k as i32,
                gs as i32,
            )
        }
    };
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&dy).unwrap();
    (rc, out)
}

#[test]
fn gemv_w4a16_matches_cpu_reference() {
    let Some(stream) = stream() else { return };

    let cases: &[(usize, usize, usize)] = &[
        (64, 2560, 32),
        (64, 2560, 16),
        (256, 2048, 128),
        (37, 1024, 64),
        (13, 256, 8),
        (33, 1920, 24),
        (64, 3072, 3072),
        (12, 4096, 32),
        (5, 8192, 16),
        (7, 4096, 128),
        (129, 10240, 32),
        (64, 32, 32),
        (1, 64, 32),
    ];
    let mut worst_overall = 0f32;
    for &(n, k, gs) in cases {
        let (packed, scales, x) = gen_inputs(n, k, gs, 0x51ded ^ ((n * k * gs) as u64));
        let want = cpu_ref(&packed, &scales, &x, n, k, gs);
        let (rc, got) = run_gemv(&stream, &packed, &scales, &x, n, k, gs);
        assert_eq!(
            rc, 0,
            "gemv_w4a16 rejected a supported shape n={n} k={k} gs={gs}"
        );
        let worst = max_rel(&got, &want);
        worst_overall = worst_overall.max(worst);
        eprintln!(
            "gemv_w4a16 n={n} k={k} gs={gs} path={}: max rel err {worst:.3e}",
            if k <= 3072 { "block" } else { "row" }
        );
        assert!(
            worst <= 1e-2,
            "gemv_w4a16 n={n} k={k} gs={gs}: max rel err {worst:.3e} > 1e-2"
        );
    }
    eprintln!(
        "gemv_w4a16 vs CPU reference: worst rel err {worst_overall:.3e} over {} cases",
        cases.len()
    );
}

#[test]
fn cpu_reference_detects_a_single_wrong_scale() {
    let Some(stream) = stream() else { return };
    let (n, k, gs) = (64usize, 2560usize, 32usize);
    let (packed, scales, x) = gen_inputs(n, k, gs, 0xBADC0DE);
    let (rc, got) = run_gemv(&stream, &packed, &scales, &x, n, k, gs);
    assert_eq!(rc, 0);

    let good = cpu_ref(&packed, &scales, &x, n, k, gs);
    assert!(max_rel(&got, &good) <= 1e-2, "control arm must pass");

    let per_row = k / gs;
    assert!(per_row > 1, "rotation is a no-op at one group per row");
    let mut bad_scales = scales.clone();
    for r in 0..n {
        bad_scales[r * per_row..(r + 1) * per_row].rotate_left(1);
    }
    assert!(
        (0..n).any(|r| bad_scales[r * per_row] != scales[r * per_row]),
        "the generator produced constant scale rows; the negative control is vacuous"
    );
    let bad = cpu_ref(&packed, &bad_scales, &x, n, k, gs);
    let rel = max_rel(&got, &bad);
    eprintln!(
        "negative control (scales rotated by one group, {per_row} groups/row): rel err {rel:.3e}"
    );
    assert!(
        rel > 1e-1,
        "rotating every scale by one group moved the reference by only {rel:.3e}; \
         this suite cannot detect a group-indexing bug"
    );
}

#[test]
fn gemv_w4a16_group_sizes_are_rejected_or_correct() {
    let Some(stream) = stream() else { return };
    let (n, k) = (64usize, 1920usize);
    let mut rejected = Vec::new();
    let mut accepted = Vec::new();
    for gs in [
        8usize, 16, 24, 32, 40, 48, 64, 80, 120, 128, 240, 480, 960, 1920,
    ] {
        assert_eq!(
            k % gs,
            0,
            "test shape error: K={k} not divisible by gs={gs}"
        );
        let (packed, scales, x) = gen_inputs(n, k, gs, 0x9005 ^ gs as u64);
        let (rc, got) = run_gemv(&stream, &packed, &scales, &x, n, k, gs);
        if rc != 0 {
            rejected.push(gs);
            continue;
        }
        accepted.push(gs);
        let want = cpu_ref(&packed, &scales, &x, n, k, gs);
        let worst = max_rel(&got, &want);
        assert!(
            worst <= 1e-2,
            "gemv_w4a16 ACCEPTED gs={gs} (rc=0) and answered wrongly: max rel err {worst:.3e}. \
             Silently wrong is the one outcome this contract forbids -- either fix the kernel \
             or reject the group size."
        );
    }
    eprintln!("gemv_w4a16 group sizes: accepted {accepted:?} rejected {rejected:?}");

    for must in [16usize, 32, 64, 128] {
        assert!(
            accepted.contains(&must),
            "gs={must} must stay supported (gemma-4-E4B-it-qat-w4a16-ct is group_size 32; \
             the Metal lanes also serve 16); accepted set was {accepted:?}"
        );
    }
}

#[test]
fn gemv_w4a16_gelu_pli_matches_cpu_reference() {
    let Some(stream) = stream() else { return };
    for &(n, k, gs) in &[
        (64usize, 2048usize, 32usize),
        (19, 1024, 128),
        (7, 3072, 64),
    ] {
        let (packed, scales, x) = gen_inputs(n, k, gs, 0x9e10 ^ ((n * k) as u64));
        let pli: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.03).collect();
        let raw = cpu_ref(&packed, &scales, &x, n, k, gs);
        let want: Vec<f32> = raw
            .iter()
            .zip(pli.iter())
            .map(|(&a, &p)| gelu_tanh(a) * p)
            .collect();

        #[allow(deprecated)]
        let dp: CudaSlice<u32> = stream.memcpy_stod(&packed).unwrap();
        #[allow(deprecated)]
        let ds: CudaSlice<u16> = stream.memcpy_stod(&scales).unwrap();
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.memcpy_stod(&x).unwrap();
        #[allow(deprecated)]
        let dl: CudaSlice<f32> = stream.memcpy_stod(&pli).unwrap();
        #[allow(deprecated)]
        let mut dy: CudaSlice<u16> = stream.memcpy_stod(&vec![0x7FC0u16; n]).unwrap();
        let rc = {
            let (pp, _g1) = dp.device_ptr(&stream);
            let (sp, _g2) = ds.device_ptr(&stream);
            let (xp, _g3) = dx.device_ptr(&stream);
            let (lp, _g4) = dl.device_ptr(&stream);
            let (yp, _g5) = dy.device_ptr_mut(&stream);
            unsafe {
                cuda::gemv_w4a16_gelu_pli(
                    stream.cu_stream() as *mut _,
                    pp as *const u32,
                    sp as *const u16,
                    xp as *const u16,
                    lp as *const f32,
                    yp as *mut u16,
                    n as i32,
                    k as i32,
                    gs as i32,
                )
            }
        };
        assert_eq!(rc, 0, "gelu_pli rejected n={n} k={k} gs={gs}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let got = stream.memcpy_dtov(&dy).unwrap();
        let worst = max_rel(&got, &want);
        eprintln!("gemv_w4a16_gelu_pli n={n} k={k} gs={gs}: max rel err {worst:.3e}");
        assert!(
            worst <= 2e-2,
            "gelu_pli n={n} k={k} gs={gs}: rel {worst:.3e}"
        );
    }
}

#[test]
fn gemv_w4a16_gelu_pli_rejects_grains_it_cannot_express() {
    let Some(stream) = stream() else { return };
    let (n, k) = (64usize, 1920usize);
    for gs in [8usize, 16, 24, 40, 48, 80, 120, 240] {
        let (packed, scales, x) = gen_inputs(n, k, gs, 0x6e10 ^ gs as u64);
        let pli: Vec<f32> = (0..n).map(|i| 0.5 + (i as f32) * 0.03).collect();
        #[allow(deprecated)]
        let dp: CudaSlice<u32> = stream.memcpy_stod(&packed).unwrap();
        #[allow(deprecated)]
        let ds: CudaSlice<u16> = stream.memcpy_stod(&scales).unwrap();
        #[allow(deprecated)]
        let dx: CudaSlice<u16> = stream.memcpy_stod(&x).unwrap();
        #[allow(deprecated)]
        let dl: CudaSlice<f32> = stream.memcpy_stod(&pli).unwrap();
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
        let rc = {
            let (pp, _g1) = dp.device_ptr(&stream);
            let (sp, _g2) = ds.device_ptr(&stream);
            let (xp, _g3) = dx.device_ptr(&stream);
            let (lp, _g4) = dl.device_ptr(&stream);
            let (yp, _g5) = dy.device_ptr_mut(&stream);
            unsafe {
                cuda::gemv_w4a16_gelu_pli(
                    stream.cu_stream() as *mut _,
                    pp as *const u32,
                    sp as *const u16,
                    xp as *const u16,
                    lp as *const f32,
                    yp as *mut u16,
                    n as i32,
                    k as i32,
                    gs as i32,
                )
            }
        };
        stream.synchronize().unwrap();
        assert_ne!(
            rc, 0,
            "gemv_w4a16_gelu_pli accepted gs={gs}, which it has no scale grain for"
        );
    }
    eprintln!("gemv_w4a16_gelu_pli: rejects every non-32-aligned group size");
}
