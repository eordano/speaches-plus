#![cfg(feature = "cuda")]

use cudarc::driver::{CudaContext, CudaSlice, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;

fn cpu_argmax_checked(logits: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        match best {
            Some((_, bv)) if v <= bv => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i as u32)
}

fn gpu_argmax_rows(logits: &[f32], rows: usize, n: usize) -> Vec<u32> {
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let dl: CudaSlice<f32> = stream.clone_htod(logits).unwrap();
    let parts = cuda::argmax_parts();
    let mut dv: CudaSlice<f32> = stream.alloc_zeros::<f32>(rows * parts).unwrap();
    let mut di: CudaSlice<i32> = stream.alloc_zeros::<i32>(rows * parts).unwrap();
    let mut dout: CudaSlice<u32> = stream.alloc_zeros::<u32>(rows).unwrap();
    let rc = {
        let (pl, _g1) = dl.device_ptr(&stream);
        let (pv, _g2) = dv.device_ptr_mut(&stream);
        let (pi, _g3) = di.device_ptr_mut(&stream);
        let (po, _g4) = dout.device_ptr_mut(&stream);
        unsafe {
            cuda::argmax_f32_rows(
                stream.cu_stream() as *mut _,
                pl as *const f32,
                rows as i32,
                n as i32,
                pv as *mut f32,
                pi as *mut i32,
                po as *mut u32,
            )
        }
    };
    assert_eq!(rc, 0, "argmax_f32_rows rc={rc}");
    stream.synchronize().unwrap();
    stream.clone_dtoh(&dout).unwrap()
}

fn check_vs_cpu(logits: &[f32], rows: usize, n: usize, label: &str) -> Vec<u32> {
    let got = gpu_argmax_rows(logits, rows, n);
    for r in 0..rows {
        let want = cpu_argmax_checked(&logits[r * n..(r + 1) * n]).unwrap_or(0);
        assert_eq!(
            got[r], want,
            "{label}: row {r} gpu {} vs cpu {} (n={n})",
            got[r], want
        );
    }
    got
}

#[test]
fn advkern_argmax_part_boundary_ties() {
    let n = 262_144usize;
    let rows = 6usize;
    let mut logits = vec![-5.0f32; rows * n];
    let mut st = 0xdead_beefu64;
    for v in logits.iter_mut() {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((st >> 33) as f32 / (1u64 << 31) as f32) * 10.0 - 8.0;
    }

    logits[127] = 50.0;
    logits[128] = 50.0;

    logits[n + 128 * 256 - 1] = 50.0;
    logits[n + 128 * 256] = 50.0;

    logits[2 * n + 128 * 3 - 1] = f32::NAN;
    logits[2 * n + 128 * 3] = 50.0;
    logits[2 * n + 128 * 3 + 1] = 50.0;
    logits[2 * n + 128 * 4] = f32::NAN;

    for &i in &[64usize, 129, 300, 128 * 256 + 64] {
        logits[3 * n + i] = 50.0;
    }

    logits[4 * n + n - 2] = 50.0;
    logits[4 * n + n - 1] = 50.0;

    for v in logits[5 * n..6 * n].iter_mut() {
        *v = v.min(-0.5);
    }
    logits[5 * n + 999] = -1e-20;
    logits[5 * n + 1000] = -1e-20;

    let got = check_vs_cpu(&logits, rows, n, "part-boundary-ties");
    assert_eq!(got[0], 127);
    assert_eq!(got[1], (128 * 256 - 1) as u32);
    assert_eq!(got[2], (128 * 3) as u32);
    assert_eq!(got[3], 64);
    assert_eq!(got[4], (n - 2) as u32);
    assert_eq!(got[5], 999);
}

#[test]
fn advkern_argmax_row_boundary_isolation_and_nonfinite_rows() {
    let n = 262_145usize;
    let rows = 5usize;
    let mut logits = vec![-2.0f32; rows * n];

    logits[n - 1] = 10.0;
    logits[n] = 99.0;

    for v in logits[2 * n..3 * n].iter_mut() {
        *v = f32::NAN;
    }

    for (i, v) in logits[3 * n..4 * n].iter_mut().enumerate() {
        *v = if i % 2 == 0 {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };
    }

    for v in logits[4 * n..5 * n].iter_mut() {
        *v = f32::NAN;
    }
    logits[4 * n + 128 * 256 * 2] = -1e30;

    let got = check_vs_cpu(&logits, rows, n, "row-boundary");
    assert_eq!(got[0], (n - 1) as u32);
    assert_eq!(got[1], 0);
    assert_eq!(got[2], 0, "all-NaN row must fall back to 0");
    assert_eq!(got[3], 0, "no-finite row must fall back to 0");
    assert_eq!(got[4], (128 * 256 * 2) as u32);
}

#[test]
fn advkern_argmax_small_and_awkward_n() {
    for n in [1usize, 2, 127, 128, 129, 4095, 32768 + 1] {
        let rows = 3usize;
        let mut logits = vec![0f32; rows * n];
        let mut st = n as u64 * 0x9e37;
        for v in logits.iter_mut() {
            st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((st >> 33) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0;
        }
        if n > 1 {
            logits[n - 1] = 7.0;
            logits[n] = 7.5;
        }
        check_vs_cpu(&logits, rows, n, "awkward-n");
    }
}

#[test]
fn advkern_argmax_verify_row_count_16plus1() {
    let n = 262_144usize;
    let rows = 17usize;
    let mut logits = vec![0f32; rows * n];
    let mut st = 0x1357_9bdfu64;
    for v in logits.iter_mut() {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((st >> 33) as f32 / (1u64 << 31) as f32) * 30.0 - 15.0;
    }
    check_vs_cpu(&logits, rows, n, "rows-17");
}
