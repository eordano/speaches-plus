#![cfg(feature = "cuda")]

mod common;
use common::e4m3_decode;
use common::lcg_f32;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::sync::Arc;

struct Inputs {
    q_f32: Vec<f32>,
    k_bf: Vec<u16>,
    v_bf: Vec<u16>,
    k8: Vec<u8>,
    v8: Vec<u8>,
    ks: Vec<f32>,
    vs: Vec<f32>,
}

fn build_inputs(nh: usize, nkv: usize, hd: usize, total: usize, window: i32, seed: u64) -> Inputs {
    let mut st = seed | 1;
    let q_f32: Vec<f32> = (0..nh * hd).map(|_| lcg_f32(&mut st)).collect();
    let mut k_f: Vec<f32> = (0..total * nkv * hd)
        .map(|_| lcg_f32(&mut st) * 0.25)
        .collect();
    let mut v_f: Vec<f32> = (0..total * nkv * hd).map(|_| lcg_f32(&mut st)).collect();

    if window > 0 && total > window as usize {
        let w = window as usize;

        for (pos, vval) in [(total - w - 1, 0.9f32), (total - w, -0.9f32)] {
            for kvh in 0..nkv {
                for d in 0..hd {
                    let mut qa = 0f32;
                    let group = nh / nkv;
                    for g in 0..group {
                        qa += q_f32[(kvh * group + g) * hd + d];
                    }
                    k_f[(pos * nkv + kvh) * hd + d] = (qa / group as f32).signum() * 3.0;
                    v_f[(pos * nkv + kvh) * hd + d] = vval;
                }
            }
        }
    }

    let k_bf: Vec<u16> = k_f
        .iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect();
    let v_bf: Vec<u16> = v_f
        .iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect();

    let mut k8 = vec![0u8; total * nkv * hd];
    let mut v8 = vec![0u8; total * nkv * hd];
    let mut ks = vec![0f32; total * nkv];
    let mut vs = vec![0f32; total * nkv];

    let table: Vec<(f32, u8)> = {
        let mut t: Vec<(f32, u8)> = (0..=126u8).map(|b| (e4m3_decode(b), b)).collect();
        t.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        t
    };
    let enc = |x: f32, scale: f32| -> u8 {
        let t = x / scale;
        let (mag, sbit) = if t < 0.0 { (-t, 0x80u8) } else { (t, 0u8) };
        let mag = mag.min(448.0);
        let i = table.partition_point(|&(v, _)| v < mag);
        let best = if i == 0 {
            table[0].1
        } else if i >= table.len() {
            table[table.len() - 1].1
        } else if (table[i].0 - mag).abs() < (mag - table[i - 1].0).abs() {
            table[i].1
        } else {
            table[i - 1].1
        };
        best | sbit
    };
    for p in 0..total {
        for h in 0..nkv {
            let row = &k_f[(p * nkv + h) * hd..(p * nkv + h) * hd + hd];
            let amax = row.iter().fold(0f32, |a, &x| a.max(x.abs())).max(1e-8);
            let sc = amax / 448.0;
            ks[p * nkv + h] = sc;
            for d in 0..hd {
                k8[(p * nkv + h) * hd + d] = enc(row[d], sc);
            }
            let rowv = &v_f[(p * nkv + h) * hd..(p * nkv + h) * hd + hd];
            let amax = rowv.iter().fold(0f32, |a, &x| a.max(x.abs())).max(1e-8);
            let sc = amax / 448.0;
            vs[p * nkv + h] = sc;
            for d in 0..hd {
                v8[(p * nkv + h) * hd + d] = enc(rowv[d], sc);
            }
        }
    }
    Inputs {
        q_f32,
        k_bf,
        v_bf,
        k8,
        v8,
        ks,
        vs,
    }
}

fn reference_bf16(
    inp: &Inputs,
    nh: usize,
    nkv: usize,
    hd: usize,
    total: usize,
    window: i32,
) -> Vec<f32> {
    let start = if window > 0 && total > window as usize {
        total - window as usize
    } else {
        0
    };
    let group = nh / nkv;
    let mut out = vec![0f32; nh * hd];
    for h in 0..nh {
        let kvh = h / group;
        let q = &inp.q_f32[h * hd..h * hd + hd];
        let mut scores: Vec<f64> = Vec::new();
        for p in start..total {
            let mut s = 0f64;
            for d in 0..hd {
                let kv = half::bf16::from_bits(inp.k_bf[(p * nkv + kvh) * hd + d]).to_f32();
                s += q[d] as f64 * kv as f64;
            }
            scores.push(s);
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0f64;
        let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
        for &e in &exps {
            denom += e;
        }
        for d in 0..hd {
            let mut acc = 0f64;
            for (i, p) in (start..total).enumerate() {
                let vv = half::bf16::from_bits(inp.v_bf[(p * nkv + kvh) * hd + d]).to_f32();
                acc += exps[i] * vv as f64;
            }
            out[h * hd + d] = if denom > 0.0 {
                (acc / denom) as f32
            } else {
                0.0
            };
        }
    }
    out
}

fn reference_fp8(
    inp: &Inputs,
    nh: usize,
    nkv: usize,
    hd: usize,
    total: usize,
    window: i32,
    scaling: f32,
) -> Vec<f32> {
    let start = if window > 0 && total > window as usize {
        total - window as usize
    } else {
        0
    };
    let group = nh / nkv;
    let mut out = vec![0f32; nh * hd];
    for h in 0..nh {
        let kvh = h / group;
        let q = &inp.q_f32[h * hd..h * hd + hd];
        let mut scores: Vec<f64> = Vec::new();
        for p in start..total {
            let mut s = 0f64;
            for d in 0..hd {
                let kv = e4m3_decode(inp.k8[(p * nkv + kvh) * hd + d]);

                s += q[d] as f64 * kv as f64;
            }
            s *= inp.ks[p * nkv + kvh] as f64 * scaling as f64;
            scores.push(s);
        }
        let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0f64;
        let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
        for &e in &exps {
            denom += e;
        }
        for d in 0..hd {
            let mut acc = 0f64;
            for (i, p) in (start..total).enumerate() {
                let vv = e4m3_decode(inp.v8[(p * nkv + kvh) * hd + d]) as f64
                    * inp.vs[p * nkv + kvh] as f64;
                acc += exps[i] * vv;
            }
            out[h * hd + d] = if denom > 0.0 {
                (acc / denom) as f32
            } else {
                0.0
            };
        }
    }
    out
}

struct Dev {
    stream: Arc<CudaStream>,
    dq_f32: CudaSlice<f32>,
    dq_bf: CudaSlice<u16>,
    dk: CudaSlice<u16>,
    dv: CudaSlice<u16>,
    dk8: CudaSlice<u8>,
    dv8: CudaSlice<u8>,
    dks: CudaSlice<f32>,
    dvs: CudaSlice<f32>,
    dpos: CudaSlice<i32>,
    dscr: CudaSlice<f32>,
    dfan: CudaSlice<u32>,
}

fn upload(stream: &Arc<CudaStream>, inp: &Inputs, nh: usize, hd: usize, pos0: i32) -> Dev {
    let q_bf: Vec<u16> = inp
        .q_f32
        .iter()
        .map(|&x| half::bf16::from_f32(x).to_bits())
        .collect();
    let scr_elems = cuda::flash_splitk_scratch_elems(nh as i32, hd as i32);
    Dev {
        stream: stream.clone(),
        dq_f32: stream.clone_htod(&inp.q_f32).unwrap(),
        dq_bf: stream.clone_htod(&q_bf).unwrap(),
        dk: stream.clone_htod(&inp.k_bf).unwrap(),
        dv: stream.clone_htod(&inp.v_bf).unwrap(),
        dk8: stream.clone_htod(&inp.k8).unwrap(),
        dv8: stream.clone_htod(&inp.v8).unwrap(),
        dks: stream.clone_htod(&inp.ks).unwrap(),
        dvs: stream.clone_htod(&inp.vs).unwrap(),
        dpos: stream.clone_htod(&[pos0]).unwrap(),
        dscr: stream.alloc_zeros::<f32>(scr_elems).unwrap(),
        dfan: stream.alloc_zeros::<u32>(nh).unwrap(),
    }
}

impl Dev {
    fn launch_fused_bf16(
        &mut self,
        dout: &mut CudaSlice<u16>,
        delta: i32,
        nh: i32,
        nkv: i32,
        hd: i32,
        window: i32,
    ) {
        let s = &self.stream;
        let (pq, _a) = self.dq_f32.device_ptr(s);
        let (pk, _b) = self.dk.device_ptr(s);
        let (pv, _c) = self.dv.device_ptr(s);
        let (pp, _d) = self.dpos.device_ptr(s);
        let (po, _e) = dout.device_ptr_mut(s);
        let (ps, _f) = self.dscr.device_ptr_mut(s);
        let (pf, _g) = self.dfan.device_ptr_mut(s);
        let rc = unsafe {
            cuda::flash_decode_fused_bf16kv(
                s.cu_stream() as *mut _,
                pq as *const f32,
                pk as *const u16,
                pv as *const u16,
                po as *mut u16,
                pp as *const i32,
                delta,
                ps as *mut f32,
                pf as *mut u32,
                nh,
                nkv,
                hd,
                window,
            )
        };
        assert_eq!(rc, 0, "flash_decode_fused_bf16kv rc={rc}");
    }

    fn launch_two_stage_bf16(
        &mut self,
        dout: &mut CudaSlice<u16>,
        nh: i32,
        nkv: i32,
        hd: i32,
        window: i32,
    ) {
        let s = &self.stream;
        let (pq, _a) = self.dq_f32.device_ptr(s);
        let (pk, _b) = self.dk.device_ptr(s);
        let (pv, _c) = self.dv.device_ptr(s);
        let (pp, _d) = self.dpos.device_ptr(s);
        let (po, _e) = dout.device_ptr_mut(s);
        let (ps, _f) = self.dscr.device_ptr_mut(s);
        let rc = unsafe {
            cuda::flash_decode_splitk_bf16kv(
                s.cu_stream() as *mut _,
                pq as *const f32,
                pk as *const u16,
                pv as *const u16,
                po as *mut u16,
                pp as *const i32,
                ps as *mut f32,
                nh,
                nkv,
                hd,
                window,
            )
        };
        assert_eq!(rc, 0, "flash_decode_splitk_bf16kv rc={rc}");
    }

    fn launch_fused_fp8(
        &mut self,
        dout: &mut CudaSlice<u16>,
        nh: i32,
        nkv: i32,
        hd: i32,
        window: i32,
        scaling: f32,
    ) {
        let s = &self.stream;
        let (pq, _a) = self.dq_bf.device_ptr(s);
        let (pk, _b) = self.dk8.device_ptr(s);
        let (pv, _c) = self.dv8.device_ptr(s);
        let (pks, _d) = self.dks.device_ptr(s);
        let (pvs, _e) = self.dvs.device_ptr(s);
        let (pp, _f) = self.dpos.device_ptr(s);
        let (po, _g) = dout.device_ptr_mut(s);
        let (ps, _h) = self.dscr.device_ptr_mut(s);
        let (pf, _i) = self.dfan.device_ptr_mut(s);
        let rc = unsafe {
            cuda::flash_decode_fused_fp8kv(
                s.cu_stream() as *mut _,
                pq as *const u16,
                pk as *const u8,
                pv as *const u8,
                pks as *const f32,
                pvs as *const f32,
                po as *mut u16,
                pp as *const i32,
                ps as *mut f32,
                pf as *mut u32,
                nh,
                nkv,
                hd,
                window,
                0,
                scaling,
            )
        };
        assert_eq!(rc, 0, "flash_decode_fused_fp8kv rc={rc}");
    }
}

fn max_abs_err(got_bf: &[u16], want: &[f32]) -> f32 {
    got_bf
        .iter()
        .zip(want.iter())
        .map(|(&g, &w)| (half::bf16::from_bits(g).to_f32() - w).abs())
        .fold(0f32, f32::max)
}

#[test]
fn advkern_fused_matches_reference_window_boundaries() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "advkern_splitk_race: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("advkern_splitk_race: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let (nh, nkv, hd) = (32usize, 16usize, 256usize);

    let cases = [
        (1023usize, 1024i32, 0i32),
        (1024, 1024, 0),
        (1025, 1024, 0),
        (2048, 1024, 0),
        (1025, 1024, 3),
        (1024, 0, 0),
        (13000, 1024, 0),
        (13000, 0, 0),
        (1, 1024, 0),
        (5, 0, 0),
        (1024, 1, 0),
        (1025, 1023, 0),
    ];
    for (total, window, delta) in cases {
        let inp = build_inputs(nh, nkv, hd, total, window, 0x5eed_0000 + total as u64);
        let mut dev = upload(&stream, &inp, nh, hd, total as i32 + delta);
        let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * hd).unwrap();

        dev.launch_fused_bf16(&mut dout, delta, nh as i32, nkv as i32, hd as i32, window);
        stream.synchronize().unwrap();
        let got: Vec<u16> = stream.clone_dtoh(&dout).unwrap();
        let want = reference_bf16(&inp, nh, nkv, hd, total, window);
        let err = max_abs_err(&got, &want);
        eprintln!("bf16 fused total={total} window={window} delta={delta}: max_err={err:.5}");
        assert!(
            err < 2.5e-2,
            "bf16 fused total={total} window={window}: err {err}"
        );

        if delta == 0 {
            dev.launch_two_stage_bf16(&mut dout, nh as i32, nkv as i32, hd as i32, window);
            stream.synchronize().unwrap();
            let got2: Vec<u16> = stream.clone_dtoh(&dout).unwrap();
            let err2 = max_abs_err(&got2, &want);
            assert!(
                err2 < 2.5e-2,
                "two-stage total={total} window={window}: err {err2}"
            );

            let scaling = 0.0625f32;
            dev.launch_fused_fp8(&mut dout, nh as i32, nkv as i32, hd as i32, window, scaling);
            stream.synchronize().unwrap();
            let got3: Vec<u16> = stream.clone_dtoh(&dout).unwrap();
            let want3 = reference_fp8(&inp, nh, nkv, hd, total, window, scaling);
            let err3 = max_abs_err(&got3, &want3);
            eprintln!("fp8  fused total={total} window={window}: max_err={err3:.5}");
            assert!(
                err3 < 4.0e-2,
                "fp8 fused total={total} window={window}: err {err3}"
            );
        }
    }
}

fn bf16_ulp_dist(a: u16, b: u16) -> u32 {
    fn key(x: u16) -> i32 {
        let m = (x & 0x7fff) as i32;
        if x & 0x8000 != 0 {
            -m
        } else {
            m
        }
    }
    (key(a) - key(b)).unsigned_abs()
}

#[doc = "Relaxed contract (2026-07-29, ship-note 8.2; replaces \
advkern_fused_bitwise_matches_two_stage). The fused and two-stage split-K paths are \
NOT bitwise-comparable: nvcc contracts the source-identical accumulator update \
`acc = acc*corr + w*v` differently per kernel (stage1: FMUL w*v then \
FFMA(acc,corr,·); fused, both loop variants: FMUL acc*corr then FFMA(w,v,·) - \
sm_120 SASS), so the f32 split partials legally differ by ~1 ULP per position and \
the merged bf16 outputs by up to ~1-2 bf16 ULPs wherever that crosses a rounding \
boundary. The old bitwise assertion first tripped at nh=32 nkv=16 hd=256 total=1025 \
(1 of 8192 elements, ~1 ULP): accumulation statistics, not a split-boundary defect \
- scratch-partial diffs exist at totals well below 1024 too (see \
splitk_boundary_probe and spp-lanes/shipnotes/note3-splitk-boundary.md). Production \
calls only the fused kernels and relies on (1) per-path run-to-run bitwise \
determinism and (2) both paths tracking the same softmax attention; this test now \
asserts exactly that: each path bitwise-stable across repeated launches, and \
cross-path agreement within 2 bf16 ULPs."]
#[test]
fn advkern_fused_two_stage_agreement() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "advkern_splitk_race: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("advkern_splitk_race: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();

    let mut grand_diffs = 0usize;
    let mut grand_max_ulp = 0u32;
    for (nh, nkv, hd) in [
        (32usize, 16usize, 256usize),
        (8, 4, 128),
        (16, 16, 64),
        (8, 2, 512),
    ] {
        for total in [1usize, 129, 1023, 1024, 1025, 4096, 13000] {
            for window in [0i32, 1024] {
                let inp = build_inputs(nh, nkv, hd, total, window, 0xabcd ^ (total as u64) << 8);
                let mut dev = upload(&stream, &inp, nh, hd, total as i32);
                let mut d1a: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * hd).unwrap();
                let mut d1b: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * hd).unwrap();
                let mut d2a: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * hd).unwrap();
                let mut d2b: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * hd).unwrap();
                dev.launch_fused_bf16(&mut d1a, 0, nh as i32, nkv as i32, hd as i32, window);
                dev.launch_fused_bf16(&mut d1b, 0, nh as i32, nkv as i32, hd as i32, window);
                dev.launch_two_stage_bf16(&mut d2a, nh as i32, nkv as i32, hd as i32, window);
                dev.launch_two_stage_bf16(&mut d2b, nh as i32, nkv as i32, hd as i32, window);
                stream.synchronize().unwrap();
                let f1: Vec<u16> = stream.clone_dtoh(&d1a).unwrap();
                let f2: Vec<u16> = stream.clone_dtoh(&d1b).unwrap();
                let t1: Vec<u16> = stream.clone_dtoh(&d2a).unwrap();
                let t2: Vec<u16> = stream.clone_dtoh(&d2b).unwrap();
                assert_eq!(
                    f1, f2,
                    "fused run-to-run nondeterminism: nh={nh} nkv={nkv} hd={hd} total={total} window={window}"
                );
                assert_eq!(
                    t1, t2,
                    "two-stage run-to-run nondeterminism: nh={nh} nkv={nkv} hd={hd} total={total} window={window}"
                );
                let mut diffs = 0usize;
                let mut max_ulp = 0u32;
                for (&a, &b) in f1.iter().zip(&t1) {
                    let du = bf16_ulp_dist(a, b);
                    if du > 0 {
                        diffs += 1;
                        max_ulp = max_ulp.max(du);
                    }
                }
                assert!(
                    max_ulp <= 2,
                    "fused vs two-stage beyond 2 bf16 ULPs ({max_ulp}): nh={nh} nkv={nkv} hd={hd} total={total} window={window}"
                );
                if diffs > 0 {
                    eprintln!(
                        "cross-path nh={nh} nkv={nkv} hd={hd} total={total} window={window}: {diffs} elems differ, max {max_ulp} ulp"
                    );
                }
                grand_diffs += diffs;
                grand_max_ulp = grand_max_ulp.max(max_ulp);
            }
        }
    }
    eprintln!(
        "fused/two-stage: per-path bitwise-deterministic; cross-path {grand_diffs} differing elems, max {grand_max_ulp} bf16 ulp across sweep"
    );
}

#[test]
fn advkern_fused_determinism_multi_ctx() {
    let Ok(ctx) = CudaContext::new(0) else {
        if std::env::var("NV_KERNELS_CUDA_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "advkern_splitk_race: no CUDA device 0. This gate refuses to report success without \
                 running; set NV_KERNELS_CUDA_ALLOW_SKIP=1 to skip on purpose."
            );
        }
        eprintln!("advkern_splitk_race: SKIP (NV_KERNELS_CUDA_ALLOW_SKIP=1) no CUDA device 0");
        return;
    };
    let stream = ctx.default_stream();
    let (nh, nkv, hd) = (32usize, 16usize, 256usize);
    let iters_per: usize = std::env::var("NV_DET_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10_000);
    let burst = 60usize;

    let cases = [
        (5usize, 0i32),
        (1023, 1024),
        (1024, 1024),
        (1025, 1024),
        (2048, 1024),
        (13000, 0),
    ];
    let mut grand_launches = 0usize;
    for (total, window) in cases {
        let inp = build_inputs(nh, nkv, hd, total, window, 0xfeed ^ total as u64);
        let mut dev = upload(&stream, &inp, nh, hd, total as i32);
        let mut outs_a: Vec<CudaSlice<u16>> = (0..burst)
            .map(|_| stream.alloc_zeros::<u16>(nh * hd).unwrap())
            .collect();

        for kernel in ["bf16", "fp8"] {
            let mut reference: Option<Vec<u16>> = None;
            let mut mism = 0usize;
            let mut done = 0usize;
            while done < iters_per {
                let b = burst.min(iters_per - done);
                for i in 0..b {
                    if kernel == "bf16" {
                        dev.launch_fused_bf16(
                            &mut outs_a[i],
                            0,
                            nh as i32,
                            nkv as i32,
                            hd as i32,
                            window,
                        );
                    } else {
                        dev.launch_fused_fp8(
                            &mut outs_a[i],
                            nh as i32,
                            nkv as i32,
                            hd as i32,
                            window,
                            0.0625,
                        );
                    }
                }
                stream.synchronize().unwrap();
                for out in outs_a.iter().take(b) {
                    let got: Vec<u16> = stream.clone_dtoh(out).unwrap();
                    match &reference {
                        None => reference = Some(got),
                        Some(r) => {
                            if &got != r {
                                mism += 1;
                            }
                        }
                    }
                }
                done += b;
            }
            grand_launches += iters_per;
            eprintln!(
                "det {kernel} total={total} window={window}: {iters_per} launches, {mism} mismatches"
            );
            assert_eq!(
                mism, 0,
                "{kernel} total={total} window={window}: fan-in nondeterminism"
            );
        }
    }
    eprintln!("determinism grand total: {grand_launches} launches, all bitwise-identical");
}
