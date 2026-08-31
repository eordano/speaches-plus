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
            if std::env::var("NV_KERNELS_W4A8_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_KERNELS_W4A8_ALLOW_SKIP=1): no CUDA device 0: {e}");
                return None;
            }
            panic!(
                "marlin_w4a8_fp8_cpu_ref: no CUDA device 0: {e}. This is a correctness gate; \
                 it refuses to report success without running. Set \
                 NV_KERNELS_W4A8_ALLOW_SKIP=1 to skip on purpose."
            );
        }
    }
}

fn e4m3_bits_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 0x7) as f32;
    if exp == 0xF && (b & 0x7) == 0x7 {
        return f32::NAN;
    }
    if exp == 0 {
        sign * (man / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
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
        Self { nib, scale, n, k, gs }
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

    fn scales_marlin_a8_x512(&self) -> Vec<u16> {
        let s = self.scales_transposed();
        let perm = scale_perm_single_because_the_a8_kernel_reads_scale_fragments_in_the_m16n8k32_lane_order();
        assert_eq!(s.len() % 32, 0, "scale array must be a multiple of 32 to permute");
        let mut out = vec![0u16; s.len()];
        for chunk in 0..s.len() / 32 {
            for c in 0..32 {
                let v = bf16::from_bits(s[chunk * 32 + perm[c]]).to_f32();
                out[chunk * 32 + c] = bf16::from_f32(v * 512.0).to_bits();
            }
        }
        out
    }

    fn oracle_and_absdot(&self, a_deq: &[f32], a_row_scale: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
        let sf: Vec<f32> = self
            .scale
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect();
        let ng = self.k / self.gs;
        let mut c = vec![0f32; m * self.n];
        let mut ad = vec![0f32; m * self.n];
        for mi in 0..m {
            for o in 0..self.n {
                let mut acc = 0f64;
                let mut abs = 0f64;
                for kk in 0..self.k {
                    let q = self.nib[o * self.k + kk] as i32 - 8;
                    let w = q as f32 * sf[o * ng + kk / self.gs];
                    let x = a_deq[mi * self.k + kk] * a_row_scale[mi];
                    acc += (w * x) as f64;
                    abs += (w.abs() * x.abs()) as f64;
                }
                c[mi * self.n + o] = acc as f32;
                ad[mi * self.n + o] = abs as f32;
            }
        }
        (c, ad)
    }
}

fn scale_perm_single_because_the_a8_kernel_reads_scale_fragments_in_the_m16n8k32_lane_order(
) -> Vec<usize> {
    let mut p = Vec::with_capacity(32);
    for i in 0..4usize {
        for j in [0usize, 1, 8, 9, 16, 17, 24, 25] {
            p.push(2 * i + j);
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

struct MarlinA8 {
    b_q: CudaSlice<i32>,
    workspace: CudaSlice<i32>,
}

fn preprocess_and_repack_a8(stream: &Arc<CudaStream>, w: &Weight) -> MarlinA8 {
    #[allow(deprecated)]
    let mut src: CudaSlice<i32> = stream.memcpy_stod(&w.packed_kmajor()).unwrap();
    let rc = {
        let (sp, _g) = src.device_ptr_mut(stream);
        unsafe {
            cuda::marlin_int4_fp8_preprocess(
                stream.cu_stream() as *mut _,
                sp as *const c_void,
                sp as *mut c_void,
                (w.k * w.n / 8) as i64,
            )
        }
    };
    assert_eq!(rc, 0, "marlin_int4_fp8_preprocess rc={rc}");
    let mut b_q: CudaSlice<i32> = stream.alloc_zeros::<i32>(w.k * w.n / 8).unwrap();
    let rc = {
        let (sp, _g1) = src.device_ptr(stream);
        let (mp, _g2) = b_q.device_ptr_mut(stream);
        unsafe {
            cuda::marlin_repack_w4a8(
                stream.cu_stream() as *mut _,
                sp as *const c_void,
                mp as *mut c_void,
                w.k as i32,
                w.n as i32,
                4,
            )
        }
    };
    assert_eq!(rc, 0, "marlin_repack_w4a8 rc={rc} k={} n={}", w.k, w.n);
    stream.synchronize().unwrap();

    let mut elems: i32 = 0;
    let rc = unsafe { cuda::marlin_workspace_elems(&mut elems as *mut i32) };
    assert_eq!(rc, 0, "marlin_workspace_elems rc={rc}");
    assert!(elems > 0, "marlin workspace elems = {elems}");

    let workspace: CudaSlice<i32> = stream.alloc_zeros::<i32>(elems as usize).unwrap();
    MarlinA8 { b_q, workspace }
}

fn rowquant_device(
    stream: &Arc<CudaStream>,
    a: &[u16],
    m: usize,
    k: usize,
) -> (CudaSlice<u8>, CudaSlice<f32>, Vec<f32>, Vec<f32>) {
    #[allow(deprecated)]
    let da: CudaSlice<u16> = stream.memcpy_stod(a).unwrap();
    let mut dq: CudaSlice<u8> = stream.alloc_zeros::<u8>(m * k).unwrap();
    let mut dscale: CudaSlice<f32> = stream.alloc_zeros::<f32>(m).unwrap();
    let rc = {
        let (ap, _g1) = da.device_ptr(stream);
        let (qp, _g2) = dq.device_ptr_mut(stream);
        let (sp, _g3) = dscale.device_ptr_mut(stream);
        unsafe {
            cuda::rowquant_e4m3(
                stream.cu_stream() as *mut _,
                ap as *const u16,
                qp as *mut u8,
                sp as *mut f32,
                m as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "rowquant_e4m3 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let qhost = stream.memcpy_dtov(&dq).unwrap();
    #[allow(deprecated)]
    let shost = stream.memcpy_dtov(&dscale).unwrap();
    let deq: Vec<f32> = qhost.iter().map(|&b| e4m3_bits_to_f32(b)).collect();
    for (i, v) in deq.iter().enumerate() {
        assert!(v.is_finite(), "rowquant produced non-finite fp8 at {i}");
    }
    (dq, dscale, deq, shost)
}

#[allow(clippy::too_many_arguments)]
fn gemm_a8(
    stream: &Arc<CudaStream>,
    mk: &mut MarlinA8,
    scales: &[u16],
    a_fp8: &CudaSlice<u8>,
    a_scales: &CudaSlice<f32>,
    m: usize,
    n: usize,
    k: usize,
    gs: usize,
) -> (i32, Vec<u16>) {
    #[allow(deprecated)]
    let ds: CudaSlice<u16> = stream.memcpy_stod(scales).unwrap();
    let mut dc: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let rc = {
        let (ap, _g1) = a_fp8.device_ptr(stream);
        let (asp, _g2) = a_scales.device_ptr(stream);
        let (bp, _g3) = mk.b_q.device_ptr(stream);
        let (sp, _g4) = ds.device_ptr(stream);
        let (cp, _g5) = dc.device_ptr_mut(stream);
        let (wp, _g6) = mk.workspace.device_ptr_mut(stream);
        unsafe {
            cuda::marlin_gemm_w4a8_fp8(
                stream.cu_stream() as *mut _,
                ap as *const c_void,
                asp as *const c_void,
                bp as *const c_void,
                sp as *const c_void,
                cp as *mut c_void,
                wp as *mut c_void,
                m as i32,
                n as i32,
                k as i32,
                gs as i32,
                0,
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
        let gf = bf16::from_bits(g).to_f32();
        assert!(gf.is_finite(), "marlin w4a8 produced {gf}");
        let u = (gf - e).abs() / denom;
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
    for (i, (&g, &e)) in got.iter().zip(want.iter()).enumerate() {
        let gf = bf16::from_bits(g).to_f32();
        assert!(gf.is_finite(), "element {i}: marlin produced {gf}");
        let rel = (gf - e).abs() / e.abs().max(0.5);
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

#[test]
fn marlin_w4a8_fp8_matches_cpu_reference_over_the_device_quantized_operands() {
    let Some(stream) = stream() else { return };
    let mut worst_overall = 0f32;
    let mut worst_units = 0f32;

    for &(n, k) in &[
        (128usize, 512usize),
        (256, 256),
        (128, 1024),
        (192, 2048),
        (20480, 2560),
        (2560, 10240),
    ] {
        let gs = 32usize;
        let w = Weight::random(n, k, gs, 0x0A18u64 ^ ((n * k) as u64));
        let mut mk = preprocess_and_repack_a8(&stream, &w);
        let smarlin = w.scales_marlin_a8_x512();
        for m in [1usize, 2, 4, 8, 16, 33, 512] {
            let a = rand_act(m, k, 0xA8 ^ (m as u64) ^ ((n * k) as u64));
            let (dq, dscale, a_deq, a_row_scale) = rowquant_device(&stream, &a, m, k);
            let (want, absdot) = w.oracle_and_absdot(&a_deq, &a_row_scale, m);
            let (rc, got) = gemm_a8(&stream, &mut mk, &smarlin, &dq, &dscale, m, n, k, gs);
            assert_eq!(rc, 0, "marlin_gemm_w4a8_fp8 rejected m={m} n={n} k={k} gs={gs}");
            let worst = max_rel(&got, &want);
            let units = worst_rounding_units(&got, &want, &absdot);
            worst_overall = worst_overall.max(worst);
            worst_units = worst_units.max(units);
            eprintln!(
                "marlin-w4a8 m={m} n={n} k={k} gs={gs}: max rel err {worst:.3e}, {units:.3} rounding units"
            );
            assert!(
                units <= 1.0,
                "marlin-w4a8 m={m} n={n} k={k} gs={gs}: {units:.3} operand-rounding units > 1.0 -- \
                 the reference already contains the fp8 activation quantization, so error beyond \
                 one bf16 operand ulp means the kernel or the scale plumbing is wrong, not the \
                 format. (max rel err was {worst:.3e}, reported for continuity only.)"
            );
        }
    }
    eprintln!(
        "marlin-w4a8 vs CPU reference: worst rel err {worst_overall:.3e}, worst {worst_units:.3} rounding units"
    );
}

#[test]
#[ignore = "timing probe, not a correctness gate; set NV_W4A8_BENCH=1 -- times the a16 vs a8 \
            marlin GEMM and the rowquant producer at the real E4B MLP shapes on this box, \
            because the vLLM PR's 1.5-3x claims are RTX 4090 numbers and the sm_120 ratio is \
            the fact that decides whether this route widens"]
fn marlin_w4a8_fp8_vs_w4a16_shape_bench() {
    if std::env::var("NV_W4A8_BENCH").ok().as_deref() != Some("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_W4A8_BENCH=1");
    }
    let Some(stream) = stream() else { return };
    for &(n, k) in &[(20480usize, 2560usize), (2560, 10240)] {
        let gs = 32usize;
        let w = Weight::random(n, k, gs, 0xBE4C ^ ((n * k) as u64));
        let mut mk8 = preprocess_and_repack_a8(&stream, &w);
        let s8 = w.scales_marlin_a8_x512();
        #[allow(deprecated)]
        let ds8: CudaSlice<u16> = stream.memcpy_stod(&s8).unwrap();

        #[allow(deprecated)]
        let src16: CudaSlice<i32> = stream.memcpy_stod(&w.packed_kmajor()).unwrap();
        let mut bq16: CudaSlice<i32> = stream.alloc_zeros::<i32>(w.k * w.n / 8).unwrap();
        let rc = {
            let (sp, _g1) = src16.device_ptr(&stream);
            let (mp, _g2) = bq16.device_ptr_mut(&stream);
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
        assert_eq!(rc, 0, "a16 repack rc={rc}");
        let s16full = {
            let st = w.scales_transposed();
            let mut perm = Vec::with_capacity(64);
            for i in 0..8usize {
                for j in 0..8usize {
                    perm.push(i + 8 * j);
                }
            }
            let mut out = vec![0u16; st.len()];
            for chunk in 0..st.len() / 64 {
                for c in 0..64 {
                    out[chunk * 64 + c] = st[chunk * 64 + perm[c]];
                }
            }
            out
        };
        #[allow(deprecated)]
        let ds16: CudaSlice<u16> = stream.memcpy_stod(&s16full).unwrap();
        let mut ws16: CudaSlice<i32> = {
            let mut elems: i32 = 0;
            unsafe { cuda::marlin_workspace_elems(&mut elems as *mut i32) };
            stream.alloc_zeros::<i32>(elems as usize).unwrap()
        };

        for m in [128usize, 256, 512] {
            let a = rand_act(m, k, 0xBE ^ (m as u64));
            #[allow(deprecated)]
            let da: CudaSlice<u16> = stream.memcpy_stod(&a).unwrap();
            let mut dq: CudaSlice<u8> = stream.alloc_zeros::<u8>(m * k).unwrap();
            let mut dsc: CudaSlice<f32> = stream.alloc_zeros::<f32>(m).unwrap();
            let mut dc: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();

            let iters = 50usize;
            let bench = |f: &mut dyn FnMut()| -> f64 {
                for _ in 0..5 {
                    f();
                }
                stream.synchronize().unwrap();
                let t0 = std::time::Instant::now();
                for _ in 0..iters {
                    f();
                }
                stream.synchronize().unwrap();
                t0.elapsed().as_secs_f64() * 1e6 / iters as f64
            };

            let us16 = bench(&mut || {
                let (ap, _g1) = da.device_ptr(&stream);
                let (bp, _g2) = bq16.device_ptr(&stream);
                let (sp, _g3) = ds16.device_ptr(&stream);
                let (cp, _g4) = dc.device_ptr_mut(&stream);
                let (wp, _g5) = ws16.device_ptr_mut(&stream);
                let rc = unsafe {
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
                };
                assert_eq!(rc, 0);
            });

            let usq = bench(&mut || {
                let (ap, _g1) = da.device_ptr(&stream);
                let (qp, _g2) = dq.device_ptr_mut(&stream);
                let (sp, _g3) = dsc.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::rowquant_e4m3(
                        stream.cu_stream() as *mut _,
                        ap as *const u16,
                        qp as *mut u8,
                        sp as *mut f32,
                        m as i32,
                        k as i32,
                    )
                };
                assert_eq!(rc, 0);
            });

            let us8 = bench(&mut || {
                let (qp, _g1) = dq.device_ptr(&stream);
                let (scp, _g2) = dsc.device_ptr(&stream);
                let (bp, _g3) = mk8.b_q.device_ptr(&stream);
                let (sp, _g4) = ds8.device_ptr(&stream);
                let (cp, _g5) = dc.device_ptr_mut(&stream);
                let (wp, _g6) = mk8.workspace.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::marlin_gemm_w4a8_fp8(
                        stream.cu_stream() as *mut _,
                        qp as *const c_void,
                        scp as *const c_void,
                        bp as *const c_void,
                        sp as *const c_void,
                        cp as *mut c_void,
                        wp as *mut c_void,
                        m as i32,
                        n as i32,
                        k as i32,
                        gs as i32,
                        0,
                    )
                };
                assert_eq!(rc, 0);
            });

            eprintln!(
                "W4A8-BENCH n={n} k={k} m={m}: a16={us16:.1}us a8={us8:.1}us rowquant={usq:.1}us \
                 a8_plus_quant={:.1}us ratio_gemm={:.2}x ratio_e2e={:.2}x",
                us8 + usq,
                us16 / us8,
                us16 / (us8 + usq)
            );
        }
    }
}

#[test]
fn marlin_w4a8_fp8_one_hot_localises_layout() {
    let Some(stream) = stream() else { return };
    let (n, k, gs) = (128usize, 512usize, 32usize);
    let (n0, k0) = (77usize, 261usize);
    let mut w = Weight::random(n, k, gs, 0x1B08);
    w.nib.iter_mut().for_each(|v| *v = 8);
    w.nib[n0 * k + k0] = 15;

    let mut mk = preprocess_and_repack_a8(&stream, &w);
    let smarlin = w.scales_marlin_a8_x512();
    let m = 4usize;
    let a = rand_act(m, k, 0x0DD8);
    let (dq, dscale, a_deq, a_row_scale) = rowquant_device(&stream, &a, m, k);
    let (rc, got) = gemm_a8(&stream, &mut mk, &smarlin, &dq, &dscale, m, n, k, gs);
    assert_eq!(rc, 0, "marlin_gemm_w4a8_fp8 rc={rc}");

    let ng = k / gs;
    let s = bf16::from_bits(w.scale[n0 * ng + k0 / gs]).to_f32();
    for mi in 0..m {
        for o in 0..n {
            let g = bf16::from_bits(got[mi * n + o]).to_f32();
            if o == n0 {
                let want = 7.0 * s * a_deq[mi * k + k0] * a_row_scale[mi];
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
                     int4->fp8 preprocess, the a8 repack, or the single-grain scale \
                     permutation is misaligned."
                );
            }
        }
    }
    eprintln!("marlin-w4a8 one-hot: the only non-zero column is {n0}, value matches 7*s*x");
}

#[test]
fn marlin_w4a8_fp8_requires_the_x512_scale_compensation() {
    let Some(stream) = stream() else { return };
    let (n, k, gs) = (128usize, 512usize, 32usize);
    let w = Weight::random(n, k, gs, 0x5CA8);
    let mut mk = preprocess_and_repack_a8(&stream, &w);
    let m = 2usize;
    let a = rand_act(m, k, 0x5EED8);
    let (dq, dscale, a_deq, a_row_scale) = rowquant_device(&stream, &a, m, k);
    let (want, _absdot) = w.oracle_and_absdot(&a_deq, &a_row_scale, m);

    let (rc, good) = gemm_a8(&stream, &mut mk, &w.scales_marlin_a8_x512(), &dq, &dscale, m, n, k, gs);
    assert_eq!(rc, 0);
    let rel_good = max_rel(&good, &want);
    assert!(
        rel_good <= 2e-2,
        "x512-compensated single-grain-permuted scales must reproduce the oracle; rel {rel_good:.3e}"
    );

    let perm = scale_perm_single_because_the_a8_kernel_reads_scale_fragments_in_the_m16n8k32_lane_order();
    let st = w.scales_transposed();
    let mut unmul = vec![0u16; st.len()];
    for chunk in 0..st.len() / 32 {
        for c in 0..32 {
            unmul[chunk * 32 + c] = st[chunk * 32 + perm[c]];
        }
    }
    let (rc, bad) = gemm_a8(&stream, &mut mk, &unmul, &dq, &dscale, m, n, k, gs);
    assert_eq!(rc, 0, "uncompensated scales are still a legal shape, so rc must be 0");
    let rel_bad = max_rel(&bad, &want);
    eprintln!("marlin-w4a8 scales: x512 rel {rel_good:.3e}, uncompensated rel {rel_bad:.3e}");
    assert!(
        rel_bad > 1e-1,
        "scales WITHOUT the x512 compensation reproduced the oracle to {rel_bad:.3e} -- then the \
         int4->fp8 dequant is not the /512 denormal mapping this integration assumes; reconcile \
         entry.cu's preprocess with dequant.h before trusting any a8 number"
    );
}

#[test]
fn marlin_w4a8_fp8_group_sizes_are_rejected_or_correct() {
    let Some(stream) = stream() else { return };
    let (n, k) = (128usize, 512usize);
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();
    for gs in [16usize, 32, 64, 128] {
        let w = Weight::random(n, k, gs, 0x6508 ^ gs as u64);
        let mut mk = preprocess_and_repack_a8(&stream, &w);
        let a = rand_act(1, k, 0xAC8 ^ gs as u64);
        let (dq, dscale, a_deq, a_row_scale) = rowquant_device(&stream, &a, 1, k);
        let (want, _absdot) = w.oracle_and_absdot(&a_deq, &a_row_scale, 1);
        let smarlin = w.scales_marlin_a8_x512();
        let (rc, got) = gemm_a8(&stream, &mut mk, &smarlin, &dq, &dscale, 1, n, k, gs);
        if rc != 0 {
            rejected.push(gs);
            continue;
        }
        accepted.push(gs);
        let worst = max_rel(&got, &want);
        assert!(
            worst <= 2e-2,
            "marlin-w4a8 ACCEPTED gs={gs} (rc=0) and answered wrongly: rel {worst:.3e}. \
             Silently wrong is the one outcome forbidden here -- instantiate the kernel \
             or reject the group size."
        );
    }
    eprintln!("marlin-w4a8 group sizes: accepted {accepted:?} rejected {rejected:?}");
    assert!(
        accepted.contains(&32),
        "gs=32 must stay supported -- gemma-4-E4B-it-qat-w4a16-ct is group_size 32. \
         Accepted set was {accepted:?}"
    );
    assert!(
        rejected.contains(&16),
        "gs=16 must be REJECTED, not silently wrong: the a8 entry validates group_size >= 32 \
         because the PR's kernel set starts at group_blocks=2. Rejected set was {rejected:?}"
    );
}
