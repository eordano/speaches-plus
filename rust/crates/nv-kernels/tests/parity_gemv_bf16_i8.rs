#![cfg(all(feature = "cuda", feature = "wgpu"))]

mod common;
use common::backends;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::gemv_bf16 as w_gemv;
use std::ffi::c_void;
use std::sync::Arc;

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
    fn i8s(&mut self, n: usize) -> Vec<i8> {
        (0..n)
            .map(|_| ((self.next_u32() % 255) as i32 - 127) as i8)
            .collect()
    }
    fn f32s(&mut self, n: usize, gain: f32) -> Vec<f32> {
        (0..n)
            .map(|_| (self.next_f32().abs() + 0.05) * gain)
            .collect()
    }
}

fn compare_bf16(name: &str, cu: &[u16], wg: &[u16], reference: &[f64]) -> (usize, i32) {
    let mut mismatch = 0usize;
    let mut max_ulp = 0i32;
    let mut cu_err = 0f64;
    let mut wg_err = 0f64;
    let mut scale = 1e-30f64;
    for ((a, b), r) in cu.iter().zip(wg.iter()).zip(reference.iter()) {
        if a != b {
            mismatch += 1;
            max_ulp = max_ulp.max((*a as i32 - *b as i32).abs());
        }
        let av = bf16::from_bits(*a).to_f32() as f64;
        let bv = bf16::from_bits(*b).to_f32() as f64;
        cu_err = cu_err.max((av - r).abs());
        wg_err = wg_err.max((bv - r).abs());
        scale = scale.max(r.abs());
    }
    eprintln!(
        "{name}: {mismatch}/{} bf16 words differ, max_ulp={max_ulp}; max |err| vs f64 reference: cuda={:.3e} wgpu={:.3e} (max |ref|={scale:.3e})",
        cu.len(),
        cu_err,
        wg_err
    );
    (mismatch, max_ulp)
}

fn cuda_rowquant(stream: &Arc<CudaStream>, w: &[u16], n: usize, k: usize) -> (Vec<i8>, Vec<f32>) {
    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(w).unwrap();
    let mut dq: CudaSlice<i8> = stream.alloc_zeros::<i8>(n * k).unwrap();
    let mut ds: CudaSlice<f32> = stream.alloc_zeros::<f32>(n).unwrap();
    let rc = {
        let (pw, _a) = dw.device_ptr(stream);
        let (pq, _b) = dq.device_ptr_mut(stream);
        let (ps, _c) = ds.device_ptr_mut(stream);
        unsafe {
            cuda::rowquant_i8(
                stream.cu_stream() as *mut c_void,
                pw as *const u16,
                pq as *mut i8,
                ps as *mut f32,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda rowquant_i8 rc={rc} (n={n} k={k})");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let q = stream.memcpy_dtov(&dq).unwrap();
    #[allow(deprecated)]
    let s = stream.memcpy_dtov(&ds).unwrap();
    (q, s)
}

fn rowquant_case(stream: &Arc<CudaStream>, wg: &WgpuContext, n: usize, k: usize, w: &[u16]) {
    let (cu_q, cu_s) = cuda_rowquant(stream, w, n, k);

    let mut wg_q = vec![0i8; n * k];
    let mut wg_s = vec![0f32; n];
    w_gemv::rowquant_i8(wg, w, &mut wg_q, &mut wg_s, n, k).expect("wgpu rowquant_i8");

    let mut qdiff = 0usize;
    let mut max_qdelta = 0i32;
    for (a, b) in cu_q.iter().zip(wg_q.iter()) {
        if a != b {
            qdiff += 1;
            max_qdelta = max_qdelta.max((*a as i32 - *b as i32).abs());
        }
    }
    let mut sdiff = 0usize;
    let mut shown = 0usize;
    for (row, (a, b)) in cu_s.iter().zip(wg_s.iter()).enumerate() {
        if a.to_bits() != b.to_bits() {
            sdiff += 1;
            if shown < 6 {
                let amax = (0..k)
                    .map(|i| bf16::from_bits(w[row * k + i]).to_f32().abs())
                    .fold(0f32, f32::max);
                eprintln!(
                    "  row {row}: amax=0x{:08x} ({amax:e}) cuda_scale=0x{:08x} wgpu_scale=0x{:08x}",
                    amax.to_bits(),
                    a.to_bits(),
                    b.to_bits()
                );
                shown += 1;
            }
        }
    }
    eprintln!(
        "rowquant_i8 n={n} k={k}: {qdiff}/{} int8 bytes differ (max |delta|={max_qdelta}), {sdiff}/{n} row scales differ",
        n * k
    );
    assert_eq!(sdiff, 0, "rowquant_i8 n={n} k={k}: row_scale bits differ");
    assert_eq!(qdiff, 0, "rowquant_i8 n={n} k={k}: int8 bytes differ");
}

#[test]
fn rowquant_i8_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8") else {
        return;
    };
    for (n, k, seed) in [
        (37usize, 1024usize, 0x1234_5678u64),
        (256, 4096, 0x0bad_c0de),
        (5, 48, 0xfeed_face),
        (7, 130, 0xa5a5_5a5a),
        (3, 1, 0x0000_1111),
    ] {
        let mut rng = Lcg(seed);
        let w = rng.bf16_words(n * k, 3.0);
        rowquant_case(&stream, wg, n, k, &w);
    }
}

#[test]
fn rowquant_i8_round_to_nearest_even_ties_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_ties") else {
        return;
    };
    let k = 64usize;
    let patterns: [&[f32]; 4] = [
        &[1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 2.0, -2.0],
        &[0.5, -0.5, 1.0, -1.0, 0.75, -0.75, 0.125, -0.125],
        &[4.0, -4.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5],
        &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let n = patterns.len();
    let mut w = vec![0u16; n * k];
    for (r, pat) in patterns.iter().enumerate() {
        for i in 0..k {
            w[r * k + i] = bf16::from_f32(pat[i % pat.len()]).to_bits();
        }
    }
    rowquant_case(&stream, wg, n, k, &w);
}

#[test]
fn rowquant_i8_bf16_ratio_sweep_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_sweep") else {
        return;
    };
    let k = 2048usize;
    let rows = 128usize;
    let mut w = vec![0u16; rows * k];
    for r in 0..rows {
        w[r * k] = (128u16 << 7) | (r as u16 & 0x7f);
        let mut idx = 1usize;
        'fill: for e in 120u16..128u16 {
            for m in 0u16..128u16 {
                for s in 0u16..2u16 {
                    if idx >= k {
                        break 'fill;
                    }
                    w[r * k + idx] = (s << 15) | (e << 7) | m;
                    idx += 1;
                }
            }
        }
        assert_eq!(idx, k, "sweep row {r} underfilled");
    }
    rowquant_case(&stream, wg, rows, k, &w);
}

fn sweep_row(k: usize, e: u16, m: u16, out: &mut [u16]) {
    out[0] = (e << 7) | m;
    out[1] = 0x8000 | (e << 7) | m;
    for i in 2..k {
        let drop = (i % 9) as u16;
        let ee = e.saturating_sub(drop);
        let mm = ((m as u32 * i as u32 + i as u32 * 37) & 0x7f) as u16;
        let s = ((i & 1) as u16) << 15;
        out[i] = s | (ee << 7) | mm;
    }
}

#[test]
fn rowquant_i8_amax_exponent_sweep_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_exp_sweep") else {
        return;
    };
    let k = 32usize;
    let mut w: Vec<u16> = Vec::new();
    let mut rows = 0usize;
    for e in 0u16..=255 {
        for m in 0u16..128 {
            let mut row = vec![0u16; k];
            sweep_row(k, e, m, &mut row);
            w.extend_from_slice(&row);
            rows += 1;
        }
    }
    rowquant_case(&stream, wg, rows, k, &w);
}

#[test]
fn rowquant_i8_subnormal_and_denormal_scale_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_subnormal") else {
        return;
    };
    let k = 16usize;
    let mut w: Vec<u16> = Vec::new();
    let mut rows = 0usize;
    for e in 0u16..12 {
        for m in [0u16, 1, 5, 64, 127] {
            let mut row = vec![0u16; k];
            sweep_row(k, e, m, &mut row);
            w.extend_from_slice(&row);
            rows += 1;
        }
    }
    for m in 1u16..8 {
        let mut row = vec![0u16; k];
        row[0] = m;
        row[1] = 0x8000 | m;
        w.extend_from_slice(&row);
        rows += 1;
    }
    rowquant_case(&stream, wg, rows, k, &w);
}

#[test]
fn rowquant_i8_wide_grid_and_long_rows_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_wide") else {
        return;
    };
    let mut rng = Lcg(0x0102_0304);
    let (n, k) = (70_001usize, 8usize);
    let w = rng.bf16_words(n * k, 2.0);
    rowquant_case(&stream, wg, n, k, &w);

    let (n2, k2) = (3usize, 8192usize);
    let w2 = rng.bf16_words(n2 * k2, 5.0);
    rowquant_case(&stream, wg, n2, k2, &w2);
}

fn cuda_gemv_bf16_normed(
    stream: &Arc<CudaStream>,
    w: &[u16],
    x: &[u16],
    wn: &[u16],
    rstd: f32,
    n: usize,
    k: usize,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(w).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(x).unwrap();
    #[allow(deprecated)]
    let dn: CudaSlice<u16> = stream.clone_htod(wn).unwrap();
    #[allow(deprecated)]
    let dr: CudaSlice<f32> = stream.clone_htod(&[rstd]).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pw, _a) = dw.device_ptr(stream);
        let (px, _b) = dx.device_ptr(stream);
        let (pn, _c) = dn.device_ptr(stream);
        let (pr, _d) = dr.device_ptr(stream);
        let (py, _e) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_bf16_normed(
                stream.cu_stream() as *mut c_void,
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
    assert_eq!(rc, 0, "cuda gemv_bf16_normed rc={rc} (n={n} k={k})");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&dy).unwrap();
    out
}

fn staged_x(x: &[u16], wn: &[u16], rstd: f32, k: usize, row: usize) -> Vec<f32> {
    (0..k)
        .map(|j| bf16::from_bits(x[row * k + j]).to_f32() * rstd * bf16::from_bits(wn[j]).to_f32())
        .collect()
}

#[test]
fn gemv_bf16_normed_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_bf16_normed") else {
        return;
    };
    for (n, k, seed) in [
        (37usize, 1024usize, 0x1234_5678u64),
        (512, 4096, 0x0bad_c0de),
        (5, 8, 0xfeed_face),
        (129, 512, 0xa5a5_5a5a),
    ] {
        let mut rng = Lcg(seed);
        let w = rng.bf16_words(n * k, 1.0);
        let x = rng.bf16_words(k, 2.0);
        let wn = rng.bf16_words(k, 1.5);
        let rstd = 0.8125f32;

        let cu_out = cuda_gemv_bf16_normed(&stream, &w, &x, &wn, rstd, n, k);
        let mut wg_out = vec![0u16; n];
        w_gemv::gemv_bf16_normed(wg, &w, &x, &wn, rstd, &mut wg_out, n, k)
            .expect("wgpu gemv_bf16_normed");

        let xs = staged_x(&x, &wn, rstd, k, 0);
        let reference: Vec<f64> = (0..n)
            .map(|row| {
                (0..k)
                    .map(|j| bf16::from_bits(w[row * k + j]).to_f32() as f64 * xs[j] as f64)
                    .sum()
            })
            .collect();

        let (mismatch, max_ulp) = compare_bf16(
            &format!("gemv_bf16_normed n={n} k={k}"),
            &cu_out,
            &wg_out,
            &reference,
        );
        assert_eq!(
            max_ulp, 0,
            "gemv_bf16_normed n={n} k={k}: {mismatch} words differ, max_ulp={max_ulp}"
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn cuda_gemv_i8_normed(
    stream: &Arc<CudaStream>,
    wq: &[i8],
    row_scale: &[f32],
    x: &[u16],
    wn: &[u16],
    rstd: f32,
    n: usize,
    k: usize,
) -> Vec<u16> {
    #[allow(deprecated)]
    let dq: CudaSlice<i8> = stream.clone_htod(wq).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<f32> = stream.clone_htod(row_scale).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(x).unwrap();
    #[allow(deprecated)]
    let dn: CudaSlice<u16> = stream.clone_htod(wn).unwrap();
    #[allow(deprecated)]
    let dr: CudaSlice<f32> = stream.clone_htod(&[rstd]).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pq, _a) = dq.device_ptr(stream);
        let (ps, _b) = ds.device_ptr(stream);
        let (px, _c) = dx.device_ptr(stream);
        let (pn, _d) = dn.device_ptr(stream);
        let (pr, _e) = dr.device_ptr(stream);
        let (py, _f) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_i8_normed(
                stream.cu_stream() as *mut c_void,
                pq as *const i8,
                ps as *const f32,
                px as *const u16,
                pn as *const u16,
                pr as *const f32,
                py as *mut u16,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "cuda gemv_i8_normed rc={rc} (n={n} k={k})");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&dy).unwrap();
    out
}

#[allow(clippy::too_many_arguments)]
fn cuda_gemv_i8_normed_mk(
    stream: &Arc<CudaStream>,
    wq: &[i8],
    row_scale: &[f32],
    x: &[u16],
    wn: &[u16],
    rstd: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<u16> {
    let (out, rc) = cuda_gemv_i8_normed_mk_rc(stream, wq, row_scale, x, wn, rstd, m, n, k);
    assert_eq!(rc, 0, "cuda gemv_i8_normed_mk rc={rc} (m={m} n={n} k={k})");
    out
}

#[allow(clippy::too_many_arguments)]
fn cuda_gemv_i8_normed_mk_rc(
    stream: &Arc<CudaStream>,
    wq: &[i8],
    row_scale: &[f32],
    x: &[u16],
    wn: &[u16],
    rstd: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> (Vec<u16>, i32) {
    #[allow(deprecated)]
    let dq: CudaSlice<i8> = stream.clone_htod(wq).unwrap();
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
        let (pq, _a) = dq.device_ptr(stream);
        let (ps, _b) = ds.device_ptr(stream);
        let (px, _c) = dx.device_ptr(stream);
        let (pn, _d) = dn.device_ptr(stream);
        let (pr, _e) = dr.device_ptr(stream);
        let (py, _f) = dy.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_i8_normed_mk(
                stream.cu_stream() as *mut c_void,
                pq as *const i8,
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
    if rc != 0 {
        return (Vec::new(), rc);
    }
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&dy).unwrap();
    (out, rc)
}

fn i8_reference(
    wq: &[i8],
    row_scale: &[f32],
    x: &[u16],
    wn: &[u16],
    rstd: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let mut out = vec![0f64; m * n];
    for j in 0..m {
        let xs = staged_x(x, wn, rstd[j], k, j);
        for row in 0..n {
            let dot: f64 = (0..k).map(|t| wq[row * k + t] as f64 * xs[t] as f64).sum();
            out[j * n + row] = dot * row_scale[row] as f64;
        }
    }
    out
}

#[test]
fn gemv_i8_normed_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed") else {
        return;
    };
    for (n, k, seed) in [
        (37usize, 1024usize, 0x1234_5678u64),
        (512, 4096, 0x0bad_c0de),
        (5, 16, 0xfeed_face),
        (129, 512, 0xa5a5_5a5a),
    ] {
        let mut rng = Lcg(seed);
        let wq = rng.i8s(n * k);
        let row_scale = rng.f32s(n, 0.01);
        let x = rng.bf16_words(k, 2.0);
        let wn = rng.bf16_words(k, 1.5);
        let rstd = 0.8125f32;

        let cu_out = cuda_gemv_i8_normed(&stream, &wq, &row_scale, &x, &wn, rstd, n, k);
        let mut wg_out = vec![0u16; n];
        w_gemv::gemv_i8_normed(wg, &wq, &row_scale, &x, &wn, rstd, &mut wg_out, n, k)
            .expect("wgpu gemv_i8_normed");

        let reference = i8_reference(&wq, &row_scale, &x, &wn, &[rstd], 1, n, k);
        let (mismatch, max_ulp) = compare_bf16(
            &format!("gemv_i8_normed n={n} k={k}"),
            &cu_out,
            &wg_out,
            &reference,
        );
        assert_eq!(
            max_ulp, 0,
            "gemv_i8_normed n={n} k={k}: {mismatch} words differ, max_ulp={max_ulp}"
        );
    }
}

#[test]
fn gemv_i8_normed_mk_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed_mk") else {
        return;
    };
    for (m, n, k, seed) in [
        (1usize, 37usize, 1024usize, 0x1234_5678u64),
        (2, 129, 512, 0xa5a5_5a5a),
        (4, 64, 2048, 0x0bad_c0de),
        (8, 96, 256, 0xfeed_face),
        (3, 8, 16, 0x0000_1111),
    ] {
        let mut rng = Lcg(seed);
        let wq = rng.i8s(n * k);
        let row_scale = rng.f32s(n, 0.01);
        let x = rng.bf16_words(m * k, 2.0);
        let wn = rng.bf16_words(k, 1.5);
        let rstd = rng.f32s(m, 1.25);

        let cu_out = cuda_gemv_i8_normed_mk(&stream, &wq, &row_scale, &x, &wn, &rstd, m, n, k);
        let mut wg_out = vec![0u16; m * n];
        w_gemv::gemv_i8_normed_mk(wg, &wq, &row_scale, &x, &wn, &rstd, &mut wg_out, m, n, k)
            .expect("wgpu gemv_i8_normed_mk");

        let reference = i8_reference(&wq, &row_scale, &x, &wn, &rstd, m, n, k);
        let (mismatch, max_ulp) = compare_bf16(
            &format!("gemv_i8_normed_mk m={m} n={n} k={k}"),
            &cu_out,
            &wg_out,
            &reference,
        );
        assert_eq!(
            max_ulp, 0,
            "gemv_i8_normed_mk m={m} n={n} k={k}: {mismatch} words differ, max_ulp={max_ulp}"
        );
    }
}

#[test]
fn gemv_bf16_normed_boundary_shapes_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_bf16_normed_shapes") else {
        return;
    };
    let mut cases: Vec<(usize, usize, u64)> = vec![
        (1, 8, 0x11),
        (1, 4096, 0x22),
        (8, 4096, 0x33),
        (9, 24, 0x44),
        (2048, 16, 0x55),
        (63, 1000, 0x66),
    ];
    for s in 0..8u64 {
        cases.push((256, 4096, 0xdead_0000 + s));
    }
    let mut worst = 0i32;
    for (n, k, seed) in cases {
        let mut rng = Lcg(seed);
        let w = rng.bf16_words(n * k, 1.0);
        let x = rng.bf16_words(k, 2.0);
        let wn = rng.bf16_words(k, 1.5);
        let rstd = 0.31640625f32 + (seed as f32 % 7.0) * 0.125;

        let cu_out = cuda_gemv_bf16_normed(&stream, &w, &x, &wn, rstd, n, k);
        let mut wg_out = vec![0u16; n];
        w_gemv::gemv_bf16_normed(wg, &w, &x, &wn, rstd, &mut wg_out, n, k)
            .expect("wgpu gemv_bf16_normed");

        let xs = staged_x(&x, &wn, rstd, k, 0);
        let reference: Vec<f64> = (0..n)
            .map(|row| {
                (0..k)
                    .map(|j| bf16::from_bits(w[row * k + j]).to_f32() as f64 * xs[j] as f64)
                    .sum()
            })
            .collect();
        let (mismatch, max_ulp) = compare_bf16(
            &format!("gemv_bf16_normed shape n={n} k={k} seed={seed:#x}"),
            &cu_out,
            &wg_out,
            &reference,
        );
        assert_eq!(mismatch, 0, "gemv_bf16_normed n={n} k={k} seed={seed:#x}");
        worst = worst.max(max_ulp);
    }
    assert_eq!(worst, 0);
}

#[test]
fn gemv_i8_normed_extremes_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed_extremes") else {
        return;
    };
    let (n, k) = (64usize, 128usize);
    let mut wq = vec![0i8; n * k];
    for row in 0..n {
        for t in 0..k {
            wq[row * k + t] = match (row + t) % 5 {
                0 => -128,
                1 => 127,
                2 => -127,
                3 => 0,
                _ => ((t as i32 % 251) - 125) as i8,
            };
        }
    }
    let mut row_scale = vec![0f32; n];
    for (row, s) in row_scale.iter_mut().enumerate() {
        *s = match row % 4 {
            0 => 1.0e38,
            1 => 1.0e-38,
            2 => 0.0,
            _ => 1.0 / 127.0,
        };
    }
    let mut rng = Lcg(0x5150_2020);
    let x = rng.bf16_words(k, 64.0);
    let wn = rng.bf16_words(k, 32.0);
    for rstd in [0.0f32, 1.0, 1.0e-30, 1.0e30] {
        let cu_out = cuda_gemv_i8_normed(&stream, &wq, &row_scale, &x, &wn, rstd, n, k);
        let mut wg_out = vec![0u16; n];
        w_gemv::gemv_i8_normed(wg, &wq, &row_scale, &x, &wn, rstd, &mut wg_out, n, k)
            .expect("wgpu gemv_i8_normed");
        let mut differ = 0usize;
        for (i, (a, b)) in cu_out.iter().zip(wg_out.iter()).enumerate() {
            if a != b {
                if differ < 4 {
                    eprintln!("  row {i}: cuda=0x{a:04x} wgpu=0x{b:04x}");
                }
                differ += 1;
            }
        }
        eprintln!("gemv_i8_normed extremes rstd={rstd:e}: {differ}/{n} bf16 words differ");
        assert_eq!(differ, 0, "gemv_i8_normed extremes rstd={rstd:e}");
    }
}

#[test]
fn gemv_i8_normed_mk_extreme_row_scales_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed_mk_extremes") else {
        return;
    };
    let (m, n, k) = (5usize, 48usize, 64usize);
    let mut rng = Lcg(0x0bee_f00d);
    let mut wq = rng.i8s(n * k);
    for (i, v) in wq.iter_mut().enumerate() {
        if i % 11 == 0 {
            *v = -128;
        }
    }
    let mut row_scale = vec![0f32; n];
    for (row, s) in row_scale.iter_mut().enumerate() {
        *s = match row % 6 {
            0 => 0.0,
            1 => f32::from_bits(1),
            2 => 1.0e-38,
            3 => 5.0e-39,
            4 => 1.0e38,
            _ => 0.0078125,
        };
    }
    let x = rng.bf16_words(m * k, 4.0);
    let wn = rng.bf16_words(k, 2.0);
    let rstd = vec![1.0f32, 0.0, 0.25, 1.0e-20, 4096.0];

    let cu_out = cuda_gemv_i8_normed_mk(&stream, &wq, &row_scale, &x, &wn, &rstd, m, n, k);
    let mut wg_out = vec![0u16; m * n];
    w_gemv::gemv_i8_normed_mk(wg, &wq, &row_scale, &x, &wn, &rstd, &mut wg_out, m, n, k)
        .expect("wgpu gemv_i8_normed_mk");
    let mut differ = 0usize;
    for (i, (a, b)) in cu_out.iter().zip(wg_out.iter()).enumerate() {
        if a != b {
            if differ < 6 {
                eprintln!("  out {i}: cuda=0x{a:04x} wgpu=0x{b:04x}");
            }
            differ += 1;
        }
    }
    eprintln!(
        "gemv_i8_normed_mk extreme row scales: {differ}/{} bf16 words differ",
        m * n
    );
    assert_eq!(differ, 0, "gemv_i8_normed_mk extreme row scales");
}

#[test]
fn gemv_i8_normed_mk_all_row_counts_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed_mk_rows") else {
        return;
    };
    let mut worst = 0i32;
    for m in 1..=8usize {
        for (n, k) in [(1usize, 16usize), (17, 256), (72, 2048)] {
            let mut rng = Lcg(0x3141_5926 + (m as u64) * 7919 + (n as u64));
            let mut wq = rng.i8s(n * k);
            for (i, v) in wq.iter_mut().enumerate() {
                if i % 17 == 0 {
                    *v = -128;
                }
            }
            let row_scale = rng.f32s(n, 0.01);
            let x = rng.bf16_words(m * k, 2.0);
            let wn = rng.bf16_words(k, 1.5);
            let rstd = rng.f32s(m, 1.25);

            let cu_out = cuda_gemv_i8_normed_mk(&stream, &wq, &row_scale, &x, &wn, &rstd, m, n, k);
            let mut wg_out = vec![0u16; m * n];
            w_gemv::gemv_i8_normed_mk(wg, &wq, &row_scale, &x, &wn, &rstd, &mut wg_out, m, n, k)
                .expect("wgpu gemv_i8_normed_mk");
            let reference = i8_reference(&wq, &row_scale, &x, &wn, &rstd, m, n, k);
            let (mismatch, max_ulp) = compare_bf16(
                &format!("gemv_i8_normed_mk m={m} n={n} k={k}"),
                &cu_out,
                &wg_out,
                &reference,
            );
            assert_eq!(mismatch, 0, "gemv_i8_normed_mk m={m} n={n} k={k}");
            worst = worst.max(max_ulp);
        }
    }
    assert_eq!(worst, 0);
}

const MK_SHAPES_SHARING_ONE_PROCESS_WIDE_DYNAMIC_SMEM_OPTIN: [(usize, usize, usize); 8] = [
    (8, 72, 2048),
    (1, 37, 1024),
    (4, 64, 1024),
    (2, 33, 256),
    (8, 16, 512),
    (1, 8, 2048),
    (6, 24, 128),
    (3, 48, 16),
];

#[test]
fn gemv_i8_normed_mk_concurrent_shapes_do_not_lower_each_others_smem_optin() {
    let Some((stream, _wg)) = backends("gemv_i8_normed_mk_smem_optin") else {
        return;
    };
    let ctx = stream.context().clone();
    let gate = Arc::new(std::sync::Barrier::new(
        MK_SHAPES_SHARING_ONE_PROCESS_WIDE_DYNAMIC_SMEM_OPTIN.len(),
    ));
    let mut threads = Vec::new();
    for (m, n, k) in MK_SHAPES_SHARING_ONE_PROCESS_WIDE_DYNAMIC_SMEM_OPTIN {
        let ctx = ctx.clone();
        let gate = gate.clone();
        threads.push(std::thread::spawn(move || {
            let s = ctx.new_stream().expect("new_stream");
            let mut rng = Lcg(0x5eed_0000 + (m * 100_003 + k) as u64);
            let wq = rng.i8s(n * k);
            let row_scale = rng.f32s(n, 0.01);
            let x = rng.bf16_words(m * k, 2.0);
            let wn = rng.bf16_words(k, 1.5);
            let rstd = rng.f32s(m, 1.25);
            gate.wait();
            let (out, rc) = cuda_gemv_i8_normed_mk_rc(&s, &wq, &row_scale, &x, &wn, &rstd, m, n, k);
            (m, n, k, rc, out.iter().filter(|v| **v != 0).count())
        }));
    }
    let mut bad: Vec<String> = Vec::new();
    let mut silent: Vec<String> = Vec::new();
    for t in threads {
        let (m, n, k, rc, nonzero) = t.join().expect("mk launch thread");
        eprintln!("gemv_i8_normed_mk concurrent m={m} n={n} k={k}: rc={rc} nonzero={nonzero}");
        if rc != 0 {
            bad.push(format!("m={m} n={n} k={k} rc={rc}"));
        } else if nonzero == 0 {
            silent.push(format!("m={m} n={n} k={k}"));
        }
    }
    assert!(
        bad.is_empty(),
        "a concurrent smaller shape lowered cudaFuncAttributeMaxDynamicSharedMemorySize below what these launches request, so the launch was rejected with cudaErrorInvalidValue(1) or cudaErrorLaunchOutOfResources(701): {bad:?}"
    );
    assert!(
        silent.is_empty(),
        "launch reported success but wrote an all-zero y, so the kernel did not run: {silent:?}"
    );
}

#[test]
fn rowquant_then_gemv_i8_normed_end_to_end_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_gemv_i8_e2e") else {
        return;
    };
    let (n, k) = (257usize, 1024usize);
    let mut rng = Lcg(0x7777_3333);
    let w = rng.bf16_words(n * k, 1.0);
    let x = rng.bf16_words(k, 2.0);
    let wn = rng.bf16_words(k, 1.5);
    let rstd = 0.640625f32;

    let (cu_q, cu_s) = cuda_rowquant(&stream, &w, n, k);
    let mut wg_q = vec![0i8; n * k];
    let mut wg_s = vec![0f32; n];
    w_gemv::rowquant_i8(wg, &w, &mut wg_q, &mut wg_s, n, k).expect("wgpu rowquant_i8");
    assert_eq!(cu_q, wg_q, "e2e: quantized weights differ");
    assert!(
        cu_s.iter()
            .zip(wg_s.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "e2e: row scales differ"
    );

    let cu_out = cuda_gemv_i8_normed(&stream, &cu_q, &cu_s, &x, &wn, rstd, n, k);
    let mut wg_out = vec![0u16; n];
    w_gemv::gemv_i8_normed(wg, &wg_q, &wg_s, &x, &wn, rstd, &mut wg_out, n, k)
        .expect("wgpu gemv_i8_normed");

    let reference = i8_reference(&cu_q, &cu_s, &x, &wn, &[rstd], 1, n, k);
    let (mismatch, max_ulp) = compare_bf16(
        &format!("rowquant+gemv_i8_normed n={n} k={k}"),
        &cu_out,
        &wg_out,
        &reference,
    );
    assert_eq!(
        max_ulp, 0,
        "rowquant+gemv_i8_normed: {mismatch} words differ, max_ulp={max_ulp}"
    );
}

const BF16_POS_INF: u16 = 0x7f80;
const BF16_NEG_INF: u16 = 0xff80;
const BF16_ONE: u16 = 0x3f80;
const BF16_TWO_POW_M63: u16 = 0x2000;

#[test]
fn rowquant_i8_infinite_and_nan_rows_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_nonfinite") else {
        return;
    };
    let k = 8usize;
    let rows: [[u16; 8]; 12] = [
        [
            BF16_POS_INF,
            0x3f80,
            0x4000,
            0xbf80,
            0x3f00,
            0x0000,
            0x4080,
            0x3f80,
        ],
        [
            BF16_NEG_INF,
            0x3f80,
            0x4000,
            0xbf80,
            0x3f00,
            0x0000,
            0x4080,
            0x3f80,
        ],
        [
            0x7fc0, 0x3f80, 0x4000, 0xbf80, 0x3f00, 0x0000, 0x4080, 0x3f80,
        ],
        [
            0x7fc0, 0x7fc0, 0x7fc0, 0x7fc0, 0x7fc0, 0x7fc0, 0x7fc0, 0x7fc0,
        ],
        [
            BF16_POS_INF,
            BF16_NEG_INF,
            0x7fc0,
            0x3f80,
            0x0000,
            0x0001,
            0x4000,
            0x8000,
        ],
        [
            0x7f81, 0x7fff, 0xffc0, 0xff81, 0x3f80, 0xbf80, 0x0000, 0x0002,
        ],
        [
            0x7fc1, 0x0001, 0x0002, 0x0003, 0x8001, 0x0000, 0x0000, 0x0000,
        ],
        [
            BF16_POS_INF,
            BF16_POS_INF,
            BF16_POS_INF,
            BF16_POS_INF,
            0x7f7f,
            0xff7f,
            0x0001,
            0x8000,
        ],
        [
            0x7fff, 0x7f7f, 0xff7f, 0x0001, 0x0000, 0x3f80, 0xbf80, 0x4000,
        ],
        [
            0x0000, 0x8000, 0x0000, 0x8000, 0x0000, 0x8000, 0x0000, 0x8000,
        ],
        [
            BF16_NEG_INF,
            0x7fc0,
            0x0000,
            0x0000,
            0x0000,
            0x0000,
            0x0000,
            0x0000,
        ],
        [
            0x7fc0, 0x0000, 0x8000, 0x0000, 0x8000, 0x0000, 0x8000, 0x0000,
        ],
    ];
    let n = rows.len();
    let mut w = vec![0u16; n * k];
    for (r, row) in rows.iter().enumerate() {
        w[r * k..(r + 1) * k].copy_from_slice(row);
    }
    rowquant_case(&stream, wg, n, k, &w);
}

#[test]
fn rowquant_i8_non_finite_mixed_into_random_rows_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("rowquant_i8_nonfinite_mixed") else {
        return;
    };
    let (n, k) = (129usize, 256usize);
    let mut rng = Lcg(0x2718_2818);
    let mut w = rng.bf16_words(n * k, 3.0);
    for row in 0..n {
        match row % 5 {
            0 => {}
            1 => w[row * k + (row % k)] = BF16_POS_INF,
            2 => w[row * k + (row % k)] = 0x7fc0,
            3 => {
                w[row * k + (row % k)] = BF16_NEG_INF;
                w[row * k + ((row + 7) % k)] = 0x7fc0;
            }
            _ => {
                for t in 0..k {
                    w[row * k + t] = 0x7fc0 | ((t as u16) & 0x3f);
                }
            }
        }
    }
    rowquant_case(&stream, wg, n, k, &w);
}

fn bf16_from_bits_row(bits: u16, n: usize) -> Vec<u16> {
    vec![bits; n]
}

#[test]
fn gemv_bf16_normed_smallest_normal_products_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_bf16_normed_min_normal") else {
        return;
    };
    let (n, k) = (64usize, 4096usize);
    let one = bf16_from_bits_row(BF16_ONE, k);

    let mut w = vec![0u16; n * k];
    for row in 0..n {
        let sign = if row % 2 == 0 { 0u16 } else { 0x8000 };
        for t in 0..k {
            let m = ((row * 31 + t * 17) % 128) as u16;
            w[row * k + t] = sign | BF16_TWO_POW_M63 | m;
        }
    }
    let x = bf16_from_bits_row(BF16_TWO_POW_M63, k);

    let cu_out = cuda_gemv_bf16_normed(&stream, &w, &x, &one, 1.0, n, k);
    let mut wg_out = vec![0u16; n];
    w_gemv::gemv_bf16_normed(wg, &w, &x, &one, 1.0, &mut wg_out, n, k).expect("wgpu");
    let xs = staged_x(&x, &one, 1.0, k, 0);
    let reference: Vec<f64> = (0..n)
        .map(|row| {
            (0..k)
                .map(|j| bf16::from_bits(w[row * k + j]).to_f32() as f64 * xs[j] as f64)
                .sum()
        })
        .collect();
    let (mismatch, max_ulp) = compare_bf16(
        "gemv_bf16_normed min-normal products",
        &cu_out,
        &wg_out,
        &reference,
    );
    assert!(
        cu_out.iter().all(|v| v & 0x7fff != 0),
        "min-normal case degenerated to zero on the cuda side"
    );
    assert_eq!(mismatch, 0, "gemv_bf16_normed min-normal products");
    assert_eq!(max_ulp, 0);

    let wn = bf16_from_bits_row(BF16_TWO_POW_M63, k);
    let mut w2 = vec![0u16; n * k];
    for row in 0..n {
        for t in 0..k {
            let m = ((row * 13 + t * 7) % 128) as u16;
            w2[row * k + t] = BF16_ONE | m;
        }
    }
    let cu2 = cuda_gemv_bf16_normed(&stream, &w2, &x, &wn, 1.0, n, k);
    let mut wg2 = vec![0u16; n];
    w_gemv::gemv_bf16_normed(wg, &w2, &x, &wn, 1.0, &mut wg2, n, k).expect("wgpu");
    let xs2 = staged_x(&x, &wn, 1.0, k, 0);
    assert!(
        xs2.iter().all(|v| *v == f32::from_bits(0x0080_0000)),
        "staged x is not the smallest normal f32"
    );
    let reference2: Vec<f64> = (0..n)
        .map(|row| {
            (0..k)
                .map(|j| bf16::from_bits(w2[row * k + j]).to_f32() as f64 * xs2[j] as f64)
                .sum()
        })
        .collect();
    let (mismatch2, ulp2) = compare_bf16(
        "gemv_bf16_normed min-normal staged x",
        &cu2,
        &wg2,
        &reference2,
    );
    assert!(cu2.iter().all(|v| v & 0x7fff != 0));
    assert_eq!(mismatch2, 0, "gemv_bf16_normed min-normal staged x");
    assert_eq!(ulp2, 0);
}

#[test]
fn gemv_i8_normed_smallest_normal_staged_x_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed_min_normal") else {
        return;
    };
    let (n, k) = (64usize, 2048usize);
    let x = bf16_from_bits_row(BF16_TWO_POW_M63, k);
    let wn = bf16_from_bits_row(BF16_TWO_POW_M63, k);
    let xs = staged_x(&x, &wn, 1.0, k, 0);
    assert!(xs.iter().all(|v| *v == f32::from_bits(0x0080_0000)));

    let mut wq = vec![0i8; n * k];
    for row in 0..n {
        for t in 0..k {
            wq[row * k + t] = (1 + ((row * 5 + t * 3) % 127)) as i8;
        }
    }
    let row_scale = vec![1.0f32; n];

    let cu_out = cuda_gemv_i8_normed(&stream, &wq, &row_scale, &x, &wn, 1.0, n, k);
    let mut wg_out = vec![0u16; n];
    w_gemv::gemv_i8_normed(wg, &wq, &row_scale, &x, &wn, 1.0, &mut wg_out, n, k).expect("wgpu");
    let reference = i8_reference(&wq, &row_scale, &x, &wn, &[1.0], 1, n, k);
    let (mismatch, max_ulp) = compare_bf16(
        "gemv_i8_normed min-normal staged x",
        &cu_out,
        &wg_out,
        &reference,
    );
    assert!(cu_out.iter().all(|v| v & 0x7fff != 0));
    assert_eq!(mismatch, 0, "gemv_i8_normed min-normal staged x");
    assert_eq!(max_ulp, 0);

    let mut wg_mk = vec![0u16; n];
    w_gemv::gemv_i8_normed_mk(wg, &wq, &row_scale, &x, &wn, &[1.0], &mut wg_mk, 1, n, k)
        .expect("wgpu mk");
    let cu_mk = cuda_gemv_i8_normed_mk(&stream, &wq, &row_scale, &x, &wn, &[1.0], 1, n, k);
    let (mm, mu) = compare_bf16(
        "gemv_i8_normed_mk min-normal staged x",
        &cu_mk,
        &wg_mk,
        &reference,
    );
    assert_eq!(mm, 0, "gemv_i8_normed_mk min-normal staged x");
    assert_eq!(mu, 0);
}

#[test]
fn gemv_bf16_normed_infinite_accumulator_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_bf16_normed_inf") else {
        return;
    };
    let (n, k) = (32usize, 64usize);
    let big = bf16::from_f32(3.0e38).to_bits();
    let one = bf16_from_bits_row(BF16_ONE, k);
    let x = bf16_from_bits_row(big, k);
    let mut w = vec![0u16; n * k];
    for row in 0..n {
        let sign = if row % 2 == 0 { 0u16 } else { 0x8000 };
        for t in 0..k {
            w[row * k + t] = sign | big;
        }
    }
    for rstd in [1.0f32, 4.0, 0.25] {
        let cu_out = cuda_gemv_bf16_normed(&stream, &w, &x, &one, rstd, n, k);
        let mut wg_out = vec![0u16; n];
        w_gemv::gemv_bf16_normed(wg, &w, &x, &one, rstd, &mut wg_out, n, k).expect("wgpu");
        let infs = cu_out.iter().filter(|v| (**v & 0x7fff) == 0x7f80).count();
        let differ = cu_out
            .iter()
            .zip(wg_out.iter())
            .filter(|(a, b)| a != b)
            .count();
        eprintln!(
            "gemv_bf16_normed inf-accum rstd={rstd:e}: {differ}/{n} differ, {infs}/{n} are +-inf"
        );
        assert_eq!(infs, n, "expected every row to overflow to +-inf on cuda");
        assert_eq!(differ, 0, "gemv_bf16_normed inf-accum rstd={rstd:e}");
    }
}

#[test]
fn gemv_i8_normed_infinite_accumulator_cuda_vs_wgpu() {
    let Some((stream, wg)) = backends("gemv_i8_normed_inf") else {
        return;
    };
    let (m, n, k) = (4usize, 24usize, 64usize);
    let big = bf16::from_f32(3.0e38).to_bits();
    let x = bf16_from_bits_row(big, m * k);
    let wn = bf16_from_bits_row(big, k);
    let mut wq = vec![0i8; n * k];
    for row in 0..n {
        let v: i8 = if row % 2 == 0 { 1 } else { -1 };
        for t in 0..k {
            wq[row * k + t] = v;
        }
    }
    let mut row_scale = vec![0f32; n];
    for (row, s) in row_scale.iter_mut().enumerate() {
        *s = match row % 3 {
            0 => 1.0,
            1 => 1.0e-30,
            _ => 7.5,
        };
    }
    let rstd = vec![1.0f32; m];

    let cu1 = cuda_gemv_i8_normed(&stream, &wq, &row_scale, &x[..k], &wn, 1.0, n, k);
    let mut wg1 = vec![0u16; n];
    w_gemv::gemv_i8_normed(wg, &wq, &row_scale, &x[..k], &wn, 1.0, &mut wg1, n, k).expect("wgpu");
    let infs = cu1.iter().filter(|v| (**v & 0x7fff) == 0x7f80).count();
    let d1 = cu1.iter().zip(wg1.iter()).filter(|(a, b)| a != b).count();
    eprintln!("gemv_i8_normed inf-accum: {d1}/{n} differ, {infs}/{n} are +-inf");
    assert_eq!(infs, n, "expected every row to overflow to +-inf on cuda");
    assert_eq!(d1, 0, "gemv_i8_normed inf-accum");

    let cu2 = cuda_gemv_i8_normed_mk(&stream, &wq, &row_scale, &x, &wn, &rstd, m, n, k);
    let mut wg2 = vec![0u16; m * n];
    w_gemv::gemv_i8_normed_mk(wg, &wq, &row_scale, &x, &wn, &rstd, &mut wg2, m, n, k)
        .expect("wgpu mk");
    let infs2 = cu2.iter().filter(|v| (**v & 0x7fff) == 0x7f80).count();
    let d2 = cu2.iter().zip(wg2.iter()).filter(|(a, b)| a != b).count();
    eprintln!(
        "gemv_i8_normed_mk inf-accum: {d2}/{} differ, {infs2}/{} are +-inf",
        m * n,
        m * n
    );
    assert_eq!(
        infs2,
        m * n,
        "expected every mk output to overflow to +-inf on cuda"
    );
    assert_eq!(d2, 0, "gemv_i8_normed_mk inf-accum");
}
