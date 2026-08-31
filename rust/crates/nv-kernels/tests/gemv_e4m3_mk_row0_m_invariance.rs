#![cfg(feature = "cuda")]

mod common;
use common::{lcg, rnd_fp8};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::sync::Arc;

const N_12288_K_5120_THE_QWEN38_VERIFY_SHAPE_THE_STAGED_GEMV_SERVES_TO_M8: (usize, usize) =
    (12288, 5120);
const M_MAX_8_THE_LAST_ROW_COUNT_THE_BF16_STAGED_SMEM_FITS_AT_K5120: usize = 8;

fn rnd_bf16(state: &mut u64) -> u16 {
    half::bf16::from_f32(((lcg(state) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0).to_bits()
}

const EVERY_QWEN38_E4M3_TRUNK_SHAPE_THE_VERIFY_TOUCHES: [(usize, usize); 7] = [
    (12288, 5120),
    (2048, 5120),
    (6144, 5120),
    (10240, 5120),
    (17408, 5120),
    (248320, 5120),
    (5120, 6144),
];

#[test]
fn gemv_e4m3_mk_row0_is_bitwise_m_invariant_2_to_8_because_each_output_row_reads_only_its_own_staged_activation_row(
) {
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the staged-gemv row0 m-invariance gate and must not \
             report success having executed nothing: {e}"
        ),
    };
    let _ = N_12288_K_5120_THE_QWEN38_VERIFY_SHAPE_THE_STAGED_GEMV_SERVES_TO_M8;
    for (n, k) in EVERY_QWEN38_E4M3_TRUNK_SHAPE_THE_VERIFY_TOUCHES {
        check_shape_row0_m_invariant(&stream, n, k);
    }
}

fn check_shape_row0_m_invariant(stream: &Arc<CudaStream>, n: usize, k: usize) {
    let reach = unsafe { cuda::gemv_e4m3_untiled_staging_reaches_this_m(k as i32) } as usize;
    let m_max = M_MAX_8_THE_LAST_ROW_COUNT_THE_BF16_STAGED_SMEM_FITS_AT_K5120.min(reach);
    if m_max < 2 {
        eprintln!("[gemv-minv] n={n} k={k} reach={reach}: below two rows, nothing to sweep");
        return;
    }
    let mut st = 0x9e37_79b9_7f4a_7c15u64 ^ ((n as u64) << 24) ^ k as u64;
    let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8(&mut st)).collect();
    let hs: Vec<f32> = (0..n)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hx: Vec<u16> = (0..m_max * k).map(|_| rnd_bf16(&mut st)).collect();
    let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
    let ds: CudaSlice<f32> = stream.clone_htod(&hs).unwrap();
    let dx: CudaSlice<u16> = stream.clone_htod(&hx).unwrap();
    let mut row0: Vec<(usize, Vec<u16>)> = Vec::new();
    for m in 2..=m_max {
        let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
        let rc = {
            let (pw, _a) = dw.device_ptr(&stream);
            let (ps, _b) = ds.device_ptr(&stream);
            let (px, _c) = dx.device_ptr(&stream);
            let (py, _d) = dy.device_ptr_mut(&stream);
            unsafe {
                cuda::gemv_e4m3_mk(
                    stream.cu_stream() as *mut _,
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
        assert_eq!(
            rc, 0,
            "n={n} k={k} m={m}: gemv_e4m3_mk launcher refused at the shape it serves"
        );
        stream.synchronize().unwrap();
        let out = stream.clone_dtoh(&dy).unwrap();
        row0.push((m, out[..n].to_vec()));
    }
    let (_, base) = &row0[0];
    for (m, r) in &row0 {
        let ndiff = r.iter().zip(base.iter()).filter(|(a, b)| a != b).count();
        assert!(
            r == base,
            "n={n} k={k} m={m}: gemv row 0 differs from the m=2 row 0 in {ndiff}/{n} outputs \
             although row 0's accumulation walks the same K order at every template M; the \
             staging or the per-row indexing broke at this M"
        );
    }
    eprintln!("[gemv-minv] n={n} k={k}: row0 bitwise m-invariant over 2..={}", row0.len() + 1);
}

const N_5120_K_6144_THE_OPROJ_AND_GDN_OUTPROJ_SHAPE_WHOSE_STAGING_REACH_ENDS_INSIDE_THE_VERIFY_RANGE:
    (usize, usize) = (5120, 6144);

#[test]
fn gemv_e4m3_mk_chunked_rows_are_bitwise_the_single_launch_at_the_k6144_shape_the_verify_chunk_arm_serves(
) {
    let stream: Arc<CudaStream> = match CudaContext::new(0) {
        Ok(ctx) => ctx.default_stream(),
        Err(e) => panic!(
            "no CUDA device 0: this is the chunked-gemv bitwise-parity gate and must not \
             report success having executed nothing: {e}"
        ),
    };
    let (n, k) =
        N_5120_K_6144_THE_OPROJ_AND_GDN_OUTPROJ_SHAPE_WHOSE_STAGING_REACH_ENDS_INSIDE_THE_VERIFY_RANGE;
    let reach = unsafe { cuda::gemv_e4m3_untiled_staging_reaches_this_m(k as i32) } as usize;
    for probe_k in [5120i32, 6144, 10240, 17408] {
        let r = unsafe { cuda::gemv_e4m3_untiled_staging_reaches_this_m(probe_k) };
        eprintln!("[gemv-reach] k={probe_k} untiled_staging_reaches_m={r}");
    }
    assert!(
        (2..8).contains(&reach),
        "k={k} staging reach {reach} left the verify interior; the chunk arm in \
         matmul_fp8_small_m_gemv keys on reaches 3..=7 and this gate must fail loudly when the \
         device smem budget moves that window"
    );
    let m = reach;
    let mut st = 0x5157_e2d1_9b1c_0a2fu64;
    let hw: Vec<u8> = (0..n * k).map(|_| rnd_fp8(&mut st)).collect();
    let hs: Vec<f32> = (0..n)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hx: Vec<u16> = (0..m * k).map(|_| rnd_bf16(&mut st)).collect();
    let dw: CudaSlice<u8> = stream.clone_htod(&hw).unwrap();
    let ds: CudaSlice<f32> = stream.clone_htod(&hs).unwrap();
    let dx: CudaSlice<u16> = stream.clone_htod(&hx).unwrap();
    let mut d_single: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let mut d_chunked: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    {
        let (pw, _a) = dw.device_ptr(&stream);
        let (ps, _b) = ds.device_ptr(&stream);
        let (px, _c) = dx.device_ptr(&stream);
        let (py, _d) = d_single.device_ptr_mut(&stream);
        let (pyc, _e) = d_chunked.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::gemv_e4m3_mk(
                stream.cu_stream() as *mut _,
                pw as *const u8,
                ps as *const f32,
                px as *const u16,
                py as *mut u16,
                n as i32,
                k as i32,
                m as i32,
            )
        };
        assert_eq!(rc, 0, "single launch refused at m={m} k={k}");
        let split = m / 2;
        for (start, rows) in [(0usize, split), (split, m - split)] {
            let rc = unsafe {
                cuda::gemv_e4m3_mk(
                    stream.cu_stream() as *mut _,
                    pw as *const u8,
                    ps as *const f32,
                    (px + (start * k * 2) as u64) as *const u16,
                    (pyc + (start * n * 2) as u64) as *mut u16,
                    n as i32,
                    k as i32,
                    rows as i32,
                )
            };
            assert_eq!(rc, 0, "chunk launch refused at start={start} rows={rows} k={k}");
        }
    }
    stream.synchronize().unwrap();
    let single = stream.clone_dtoh(&d_single).unwrap();
    let chunked = stream.clone_dtoh(&d_chunked).unwrap();
    let ndiff = single
        .iter()
        .zip(chunked.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        single == chunked,
        "chunked launches diverge from the single m={m} launch in {ndiff}/{} outputs; the \
         verify chunk arm relies on this identity to keep an m=8 verify on the same arithmetic \
         as m<=7",
        m * n
    );
}
