#![cfg(feature = "cuda")]

mod common;
use common::LcgOddSeedShift32F64TwoSided as Lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
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
                "marlin_w4a16_cpu_ref: no CUDA device 0: {e}. This is a correctness gate; \
                 it refuses to report success without running. Set \
                 NV_KERNELS_W4A16_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

struct Weight {
    nib: Vec<u8>,

    scale: Vec<u16>,
    n: usize,
    k: usize,
    gs: usize,
}

impl Weight {
    fn random(n: usize, k: usize, gs: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let nib: Vec<u8> = (0..n * k).map(|_| (rng.next_u32() & 0xF) as u8).collect();
        let scale: Vec<u16> = (0..n * (k / gs))
            .map(|_| bf16::from_f32(0.005 + 0.01 * rng.next_f32().abs()).to_bits())
            .collect();
        Self {
            nib,
            scale,
            n,
            k,
            gs,
        }
    }

    fn packed_ct(&self) -> Vec<u32> {
        let mut p = vec![0u32; self.n * self.k / 8];
        for r in 0..self.n {
            for kk in 0..self.k {
                let idx = r * self.k + kk;
                p[idx / 8] |= (self.nib[idx] as u32) << (4 * (kk % 8));
            }
        }
        p
    }

    fn packed_kmajor(&self) -> Vec<i32> {
        let cols = self.k / 8;
        let ct = self.packed_ct();
        let mut t = vec![0i32; cols * self.n];
        for o in 0..self.n {
            for k8 in 0..cols {
                t[k8 * self.n + o] = ct[o * cols + k8] as i32;
            }
        }
        t
    }

    fn scales_transposed(&self) -> Vec<u16> {
        let ng = self.k / self.gs;
        let mut s = vec![0u16; ng * self.n];
        for o in 0..self.n {
            for g in 0..ng {
                s[g * self.n + o] = self.scale[o * ng + g];
            }
        }
        s
    }

    fn scales_marlin(&self) -> Vec<u16> {
        let s = self.scales_transposed();
        let perm = scale_perm();
        assert_eq!(
            s.len() % 64,
            0,
            "scale array must be a multiple of 64 to permute"
        );
        let mut out = vec![0u16; s.len()];
        for chunk in 0..s.len() / 64 {
            for c in 0..64 {
                out[chunk * 64 + c] = s[chunk * 64 + perm[c]];
            }
        }
        out
    }

    fn abs_dot(&self, a: &[u16], m: usize) -> Vec<f32> {
        let af: Vec<f32> = a.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let sf: Vec<f32> = self
            .scale
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        let ng = self.k / self.gs;
        let mut c = vec![0f32; m * self.n];
        for mi in 0..m {
            for o in 0..self.n {
                let mut acc = 0f64;
                for kk in 0..self.k {
                    let q = self.nib[o * self.k + kk] as i32 - 8;
                    let w = q as f32 * sf[o * ng + kk / self.gs];
                    acc += (w.abs() * af[mi * self.k + kk].abs()) as f64;
                }
                c[mi * self.n + o] = acc as f32;
            }
        }
        c
    }

    fn oracle(&self, a: &[u16], m: usize) -> Vec<f32> {
        let af: Vec<f32> = a.iter().map(|&b| bf16::from_bits(b).to_f32()).collect();
        let sf: Vec<f32> = self
            .scale
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        let ng = self.k / self.gs;
        let mut c = vec![0f32; m * self.n];
        for mi in 0..m {
            for o in 0..self.n {
                let mut acc = 0f64;
                for kk in 0..self.k {
                    let q = self.nib[o * self.k + kk] as i32 - 8;
                    acc += (q as f32 * sf[o * ng + kk / self.gs] * af[mi * self.k + kk]) as f64;
                }
                c[mi * self.n + o] = acc as f32;
            }
        }
        c
    }
}

fn scale_perm() -> Vec<usize> {
    let mut p = Vec::with_capacity(64);
    for i in 0..8usize {
        for j in 0..8usize {
            p.push(i + 8 * j);
        }
    }
    p
}

fn rand_act(m: usize, k: usize, seed: u64) -> Vec<u16> {
    let mut rng = Lcg::new(seed);
    (0..m * k)
        .map(|_| bf16::from_f32(rng.next_f32()).to_bits())
        .collect()
}

struct Marlin {
    b_q: CudaSlice<i32>,
    workspace: CudaSlice<i32>,
}

fn repack(stream: &Arc<CudaStream>, w: &Weight) -> Marlin {
    #[allow(deprecated)]
    let src: CudaSlice<i32> = stream.memcpy_stod(&w.packed_kmajor()).unwrap();
    let mut b_q: CudaSlice<i32> = stream.alloc_zeros::<i32>(w.k * w.n / 8).unwrap();
    let rc = {
        let (sp, _g1) = src.device_ptr(stream);
        let (mp, _g2) = b_q.device_ptr_mut(stream);
        unsafe {
            cuda::marlin_repack_w4a16(
                stream.cu_stream() as *mut _,
                sp as *const c_void,
                mp as *mut c_void,
                w.k as i32,
                w.n as i32,
                4,
            )
        }
    };
    assert_eq!(rc, 0, "marlin_repack_w4a16 rc={rc} k={} n={}", w.k, w.n);
    stream.synchronize().unwrap();

    let mut elems: i32 = 0;
    let rc = unsafe { cuda::marlin_workspace_elems(&mut elems as *mut i32) };
    assert_eq!(rc, 0, "marlin_workspace_elems rc={rc}");
    assert!(elems > 0, "marlin workspace elems = {elems}");

    let workspace: CudaSlice<i32> = stream.alloc_zeros::<i32>(elems as usize).unwrap();
    Marlin { b_q, workspace }
}

#[allow(clippy::too_many_arguments)]
fn gemm(
    stream: &Arc<CudaStream>,
    mk: &mut Marlin,
    scales: &[u16],
    a: &[u16],
    m: usize,
    n: usize,
    k: usize,
    gs: usize,
) -> (i32, Vec<u16>) {
    #[allow(deprecated)]
    let ds: CudaSlice<u16> = stream.memcpy_stod(scales).unwrap();
    #[allow(deprecated)]
    let da: CudaSlice<u16> = stream.memcpy_stod(a).unwrap();
    let mut dc: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let rc = {
        let (ap, _g1) = da.device_ptr(stream);
        let (bp, _g2) = mk.b_q.device_ptr(stream);
        let (sp, _g3) = ds.device_ptr(stream);
        let (cp, _g4) = dc.device_ptr_mut(stream);
        let (wp, _g5) = mk.workspace.device_ptr_mut(stream);
        unsafe {
            cuda::marlin_gemm_w4a16(
                stream.cu_stream() as *mut _,
                ap as *const c_void,
                bp as *const c_void,
                sp as *const c_void,
                cp as *mut c_void,
                wp as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                gs as i32,
                1,
            )
        }
    };
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let out = stream.memcpy_dtov(&dc).unwrap();
    (rc, out)
}

const BF16_U: f32 = 1.0 / 256.0;

fn worst_rounding_units(got: &[u16], want: &[f32], absdot: &[f32]) -> f32 {
    let mut worst = 0f32;
    for ((&g, &e), &ad) in got.iter().zip(want.iter()).zip(absdot.iter()) {
        let denom = BF16_U * ad;
        if denom <= f32::MIN_POSITIVE {
            continue;
        }
        let u = (bf16::from_bits(g).to_f32() - e).abs() / denom;
        if u > worst {
            worst = u;
        }
    }
    worst
}

fn max_rel(got: &[u16], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    assert!(!got.is_empty(), "nothing was compared");
    assert!(
        want.iter().any(|v| v.abs() > 1e-3),
        "the reference is all zeros; the comparison would be vacuous"
    );
    let mut worst = 0f32;
    let mut worst_e = 0f32;
    let mut worst_ulps = 0f32;
    let mut mean_abs = 0f64;
    for (i, (&g, &e)) in got.iter().zip(want.iter()).enumerate() {
        let gf = bf16::from_bits(g).to_f32();
        assert!(gf.is_finite(), "element {i}: marlin produced {gf}");
        mean_abs += e.abs() as f64;
        let rel = (gf - e).abs() / e.abs().max(0.5);
        if rel > worst {
            worst = rel;
            worst_e = e;

            let ulp = (e.abs().max(f32::MIN_POSITIVE) / 256.0).max(f32::MIN_POSITIVE);
            worst_ulps = (gf - e).abs() / ulp;
        }
    }
    eprintln!(
        "    diag: worst rel {worst:.3e} at |expected|={:.3} ({worst_ulps:.2} bf16 ulps); mean |expected|={:.3}",
        worst_e.abs(),
        mean_abs / got.len() as f64
    );
    worst
}

#[test]
fn the_rounding_unit_metric_is_calibrated_and_can_fail() {
    let absdot = [4.0f32];
    let want = [1.0f32];
    let unit = 4.0 * BF16_U;
    for mult in [0.5f32, 1.0, 2.0, 4.0] {
        let got = bf16::from_f32(1.0 + mult * unit);
        assert_eq!(
            got.to_f32(),
            1.0 + mult * unit,
            "test setup is not exact at mult={mult}"
        );
        let u = worst_rounding_units(&[got.to_bits()], &want, &absdot);
        assert!(
            (u - mult).abs() < 1e-3,
            "metric is miscalibrated: {mult} ulps of error read as {u} units"
        );
        assert_eq!(
            u > 1.0,
            mult > 1.0,
            "the 1.0 bound must trip exactly when the error exceeds one ulp \
             (mult={mult}, units={u})"
        );
    }

    assert_eq!(
        worst_rounding_units(&[bf16::from_f32(1.0).to_bits()], &want, &absdot),
        0.0
    );
}

#[test]
fn marlin_w4a16_matches_cpu_reference() {
    let Some(stream) = stream() else { return };
    let mut worst_overall = 0f32;
    let mut worst_units = 0f32;

    for &(n, k) in &[(128usize, 512usize), (256, 256), (128, 1024), (192, 2048)] {
        let gs = 32usize;
        let w = Weight::random(n, k, gs, 0x0A11Cu64 ^ ((n * k) as u64));
        let mut mk = repack(&stream, &w);
        let smarlin = w.scales_marlin();
        for m in [1usize, 2, 4, 8, 16, 33] {
            let a = rand_act(m, k, 0xA1 ^ (m as u64) ^ ((n * k) as u64));
            let want = w.oracle(&a, m);
            let absdot = w.abs_dot(&a, m);
            let (rc, got) = gemm(&stream, &mut mk, &smarlin, &a, m, n, k, gs);
            assert_eq!(rc, 0, "marlin_gemm rejected m={m} n={n} k={k} gs={gs}");
            let worst = max_rel(&got, &want);
            let units = worst_rounding_units(&got, &want, &absdot);
            worst_overall = worst_overall.max(worst);
            worst_units = worst_units.max(units);
            eprintln!(
                "marlin m={m} n={n} k={k} gs={gs}: max rel err {worst:.3e}, {units:.3} rounding units"
            );
            assert!(
                units <= 1.0,
                "marlin m={m} n={n} k={k} gs={gs}: {units:.3} operand-rounding units > 1.0 -- the \
                 kernel exceeded what bf16 operands can explain, so this is a real precision bug \
                 and not the gate. (max rel err was {worst:.3e}, reported for continuity only.)"
            );
        }
    }
    eprintln!(
        "marlin vs CPU reference: worst rel err {worst_overall:.3e}, worst {worst_units:.3} rounding units"
    );
}

#[test]
fn marlin_w4a16_one_hot_localises_layout() {
    let Some(stream) = stream() else { return };
    let (n, k, gs) = (128usize, 512usize, 32usize);
    let (n0, k0) = (77usize, 261usize);
    let mut w = Weight::random(n, k, gs, 0x1B07);
    w.nib.iter_mut().for_each(|v| *v = 8);
    w.nib[n0 * k + k0] = 15;

    let mut mk = repack(&stream, &w);
    let smarlin = w.scales_marlin();
    let m = 4usize;
    let a = rand_act(m, k, 0x0DD);
    let (rc, got) = gemm(&stream, &mut mk, &smarlin, &a, m, n, k, gs);
    assert_eq!(rc, 0, "marlin_gemm rc={rc}");

    let ng = k / gs;
    let s = bf16::from_bits(w.scale[n0 * ng + k0 / gs]).to_f32();
    for mi in 0..m {
        for o in 0..n {
            let g = bf16::from_bits(got[mi * n + o]).to_f32();
            if o == n0 {
                let want = 7.0 * s * bf16::from_bits(a[mi * k + k0]).to_f32();
                let rel = (g - want).abs() / want.abs().max(1e-3);
                assert!(
                    rel <= 2e-2,
                    "one-hot row {mi} col {o}: got {g} want {want} rel {rel:.3e}"
                );
            } else {
                assert!(
                    g.abs() < 1e-6,
                    "one-hot leaked into row {mi} col {o} (expected the only non-zero \
                     column to be {n0}): got {g}. A non-zero at a WRONG column means the \
                     repack transpose or the scale permutation is misaligned."
                );
            }
        }
    }
    eprintln!("marlin one-hot: the only non-zero column is {n0}, value matches 7*s*x");
}

#[test]
fn marlin_w4a16_requires_the_scale_permutation() {
    let Some(stream) = stream() else { return };
    let (n, k, gs) = (128usize, 512usize, 32usize);
    let w = Weight::random(n, k, gs, 0x5CA1E);
    let mut mk = repack(&stream, &w);
    let m = 2usize;
    let a = rand_act(m, k, 0x5EED);
    let want = w.oracle(&a, m);

    let (rc, good) = gemm(&stream, &mut mk, &w.scales_marlin(), &a, m, n, k, gs);
    assert_eq!(rc, 0);
    let rel_good = max_rel(&good, &want);
    assert!(
        rel_good <= 2e-2,
        "permuted scales must be the correct layout; rel {rel_good:.3e}"
    );

    let plain = w.scales_transposed();
    assert_ne!(
        plain,
        w.scales_marlin(),
        "the permutation is a no-op on this shape; the test is vacuous"
    );
    let (rc, bad) = gemm(&stream, &mut mk, &plain, &a, m, n, k, gs);
    assert_eq!(
        rc, 0,
        "unpermuted scales are still a legal *shape*, so rc must be 0"
    );
    let rel_bad = max_rel(&bad, &want);
    eprintln!("marlin scales: permuted rel {rel_good:.3e}, unpermuted rel {rel_bad:.3e}");
    assert!(
        rel_bad > 1e-1,
        "the transposed-but-unpermuted scale array reproduced the oracle to {rel_bad:.3e}. \
         Then INTEGRATION.md's pre-2026-08-09 text was right after all and \
         MarlinLinear::from_raw's index_select is doing unnecessary work -- reconcile them."
    );
}

#[test]
fn marlin_w4a16_group_sizes_are_rejected_or_correct() {
    let Some(stream) = stream() else { return };
    let (n, k) = (128usize, 512usize);
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    for gs in [16usize, 32, 64, 128] {
        let w = Weight::random(n, k, gs, 0x6505 ^ gs as u64);
        let mut mk = repack(&stream, &w);
        let a = rand_act(1, k, 0xAC7 ^ gs as u64);
        let want = w.oracle(&a, 1);
        let (rc, got) = gemm(&stream, &mut mk, &w.scales_marlin(), &a, 1, n, k, gs);
        if rc != 0 {
            rejected.push(gs);
            continue;
        }
        accepted.push(gs);
        let worst = max_rel(&got, &want);
        assert!(
            worst <= 2e-2,
            "marlin ACCEPTED gs={gs} (rc=0) and answered wrongly: rel {worst:.3e}. \
             Silently wrong is the one outcome forbidden here -- instantiate the kernel \
             or reject the group size."
        );
    }
    eprintln!("marlin group sizes: accepted {accepted:?} rejected {rejected:?}");
    assert!(
        accepted.contains(&32),
        "gs=32 must stay supported -- gemma-4-E4B-it-qat-w4a16-ct is group_size 32 and \
         Marlin is its default backend. Accepted set was {accepted:?}"
    );
}
