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

struct Bf16ArgmaxRig {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
    dl: CudaSlice<u16>,
    dv: CudaSlice<f32>,
    di: CudaSlice<i32>,
    dtok: CudaSlice<u32>,
    dpos: CudaSlice<i32>,
    dring: CudaSlice<u32>,
    cap: usize,
    ring_len: usize,
}

impl Bf16ArgmaxRig {
    fn new(cap: usize, ring_len: usize) -> Self {
        let ctx = CudaContext::new(0).expect("cuda device 0");
        let stream = ctx.default_stream();
        let parts = cuda::argmax_parts();
        Self {
            dl: stream.alloc_zeros::<u16>(cap).unwrap(),
            dv: stream.alloc_zeros::<f32>(parts).unwrap(),
            di: stream.alloc_zeros::<i32>(parts).unwrap(),
            dtok: stream.alloc_zeros::<u32>(1).unwrap(),
            dpos: stream.alloc_zeros::<i32>(1).unwrap(),
            dring: stream.alloc_zeros::<u32>(ring_len).unwrap(),
            stream,
            cap,
            ring_len,
        }
    }

    fn run(&mut self, logits: &[half::bf16], pos: i32, with_ring: bool) -> (u32, Vec<u32>) {
        assert!(logits.len() <= self.cap);
        let bits: Vec<u16> = logits.iter().map(|v| v.to_bits()).collect();
        let mut dl_view = self.dl.slice_mut(0..bits.len());
        self.stream.memcpy_htod(&bits, &mut dl_view).unwrap();
        self.stream.memcpy_htod(&[pos], &mut self.dpos).unwrap();
        let poison = vec![0xdead_0000u32; self.ring_len];
        self.stream.memcpy_htod(&poison, &mut self.dring).unwrap();
        let rc = {
            let (lp, _g1) = self.dl.device_ptr(&self.stream);
            let (vp, _g2) = self.dv.device_ptr_mut(&self.stream);
            let (ip, _g3) = self.di.device_ptr_mut(&self.stream);
            let (tp, _g4) = self.dtok.device_ptr_mut(&self.stream);
            let (pp, _g5) = self.dpos.device_ptr(&self.stream);
            let (rp, _g6) = self.dring.device_ptr_mut(&self.stream);
            unsafe {
                cuda::argmax_bf16(
                    self.stream.cu_stream() as *mut _,
                    lp as *const u16,
                    logits.len() as i32,
                    vp as *mut f32,
                    ip as *mut i32,
                    if with_ring { pp as *const i32 } else { std::ptr::null() },
                    tp as *mut u32,
                    if with_ring { rp as *mut u32 } else { std::ptr::null_mut() },
                    (self.ring_len - 1) as i32,
                )
            }
        };
        assert_eq!(rc, 0, "argmax_bf16 rc={rc}");
        self.stream.synchronize().unwrap();
        let tok = self.stream.clone_dtoh(&self.dtok).unwrap()[0];
        let ring = self.stream.clone_dtoh(&self.dring).unwrap();
        (tok, ring)
    }
}

fn bf16_oracle_argmax_checked_semantics(logits: &[half::bf16]) -> u32 {
    let widened: Vec<f32> = logits.iter().map(|v| v.to_f32()).collect();
    cpu_argmax_checked(&widened).unwrap_or(0)
}

#[test]
fn advkern_argmax_bf16_nonfinite_ties_ring_match_argmax_checked() {
    let n = 262_144usize + 37;
    let mut rig = Bf16ArgmaxRig::new(n, 64);
    let base = |seed: u64| -> Vec<half::bf16> {
        let mut st = seed;
        (0..n)
            .map(|_| {
                st = st
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                half::bf16::from_f32(((st >> 33) as f32 / (1u64 << 31) as f32) * 10.0 - 8.0)
            })
            .collect()
    };

    let mut row = base(0x11);
    for v in row.iter_mut() {
        *v = half::bf16::NAN;
    }
    let (tok, ring) = rig.run(&row, 130, true);
    assert_eq!(tok, 0, "all-NaN row must emit token 0, not 0x7fffffff");
    assert_eq!(ring[(130 - 1) & 63], 0, "all-NaN row must feed token 0 into the ring");

    let mut row = base(0x22);
    row[777] = half::bf16::INFINITY;
    row[9000] = half::bf16::from_f32(50.0);
    let (tok, ring) = rig.run(&row, 7, true);
    assert_eq!(tok, 9000, "+inf must be skipped like argmax_checked, finite max wins");
    assert_eq!(ring[(7 - 1) & 63], 9000);

    let mut row = base(0x33);
    for i in (0..n).step_by(3) {
        row[i] = half::bf16::NEG_INFINITY;
    }
    for i in (1..n).step_by(5) {
        row[i] = half::bf16::NAN;
    }
    row[128 * 256 - 1] = half::bf16::from_f32(50.0);
    row[128 * 256] = half::bf16::from_f32(50.0);
    let (tok, _) = rig.run(&row, 1, true);
    assert_eq!(tok, 128 * 256 - 1, "part-boundary tie must break toward the lowest index");
    assert_eq!(tok, bf16_oracle_argmax_checked_semantics(&row));

    let mut row = base(0x44);
    for v in row.iter_mut() {
        *v = half::bf16::NAN;
    }
    row[n - 1] = half::bf16::from_f32(-1e30);
    let (tok, _) = rig.run(&row, 1, false);
    assert_eq!(tok, (n - 1) as u32, "single finite value among NaN must win");

    let mut row = base(0x55);
    row[0] = half::bf16::INFINITY;
    row[n - 1] = half::bf16::INFINITY;
    for i in 1..64 {
        row[i] = half::bf16::NAN;
    }
    let (tok, _) = rig.run(&row, 1, false);
    assert_eq!(tok, bf16_oracle_argmax_checked_semantics(&row));
}

#[test]
fn advkern_argmax_bf16_3000_rows_injected_nonfinite_match_argmax_checked() {
    fn lcg_step(st: &mut u64) -> u64 {
        *st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *st >> 33
    }
    let n = 4099usize;
    let mut rig = Bf16ArgmaxRig::new(n, 64);
    let mut st = 0x0badc0deu64;
    for r in 0..3000usize {
        let mut row: Vec<half::bf16> = (0..n)
            .map(|_| {
                half::bf16::from_f32((lcg_step(&mut st) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0)
            })
            .collect();
        if r % 7 == 0 {
            for _ in 0..17 {
                let i = lcg_step(&mut st) as usize % n;
                row[i] = half::bf16::NAN;
            }
        }
        if r % 11 == 0 {
            let i = lcg_step(&mut st) as usize % n;
            row[i] = half::bf16::INFINITY;
        }
        if r % 13 == 0 {
            let i = lcg_step(&mut st) as usize % n;
            row[i] = half::bf16::NEG_INFINITY;
        }
        if r % 17 == 0 {
            let i = lcg_step(&mut st) as usize % (n - 1);
            let j = i + 1 + lcg_step(&mut st) as usize % (n - i - 1);
            row[i] = half::bf16::from_f32(9.0);
            row[j] = half::bf16::from_f32(9.0);
        }
        if r % 19 == 0 {
            for v in row.iter_mut() {
                *v = half::bf16::NAN;
            }
        }
        let (tok, _) = rig.run(&row, 1, false);
        let want = bf16_oracle_argmax_checked_semantics(&row);
        assert_eq!(tok, want, "row {r}: gpu {tok} vs argmax_checked {want}");
    }
}
