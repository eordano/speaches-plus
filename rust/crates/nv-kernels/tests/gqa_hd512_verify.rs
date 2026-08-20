#![cfg(feature = "cuda")]

mod common;
use common::lcg;
use common::rnd_f;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::sync::Arc;
use std::time::Instant;
use common::fp8_e4m3_to_f64;
use common::rnd_fp8;

const NH: i32 = 32;
const NKV: i32 = 4;
const HD: i32 = 512;

fn stream_or_skip() -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(ctx.default_stream()),
        Err(e) => {
            if std::env::var("NV_GQA512_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_GQA512_ALLOW_SKIP=1): no CUDA device 0: {e}");
                return None;
            }
            panic!(
                "gqa_hd512_verify: no CUDA device 0: {e}. This is a correctness gate; it \
                 refuses to report success without running. Set NV_GQA512_ALLOW_SKIP=1 to \
                 skip on purpose."
            );
        }
    }
}

fn rnd_bf16(state: &mut u64) -> u16 {
    half::bf16::from_f32(rnd_f(state)).to_bits()
}

fn bf(x: u16) -> f64 {
    half::bf16::from_bits(x).to_f32() as f64
}

fn cpu_ref_fp8(
    hq: &[u16],
    hk8: &[u8],
    hv8: &[u8],
    hks: &[f32],
    hvs: &[f32],
    m: usize,
    nc: usize,
) -> Vec<f32> {
    let nh = NH as usize;
    let nkv = NKV as usize;
    let hd = HD as usize;
    let group = nh / nkv;
    let mut out = vec![0.0f32; m * nh * hd];
    for qi in 0..m {
        let bound = nc + qi + 1;
        for h in 0..nh {
            let kvh = h / group;
            let qrow = &hq[(qi * nh + h) * hd..(qi * nh + h) * hd + hd];
            let mut scores = vec![0.0f64; bound];
            let mut mx = f64::NEG_INFINITY;
            for p in 0..bound {
                let krow = &hk8[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                let ks = hks[p * nkv + kvh] as f64;
                let mut s = 0.0f64;
                for d in 0..hd {
                    s += bf(qrow[d]) * fp8_e4m3_to_f64(krow[d]);
                }
                scores[p] = s * ks;
                if scores[p] > mx {
                    mx = scores[p];
                }
            }
            let mut l = 0.0f64;
            for p in 0..bound {
                scores[p] = (scores[p] - mx).exp();
                l += scores[p];
            }
            let mut acc = vec![0.0f64; hd];
            for p in 0..bound {
                let w = scores[p] / l;
                let vrow = &hv8[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                let vs = hvs[p * nkv + kvh] as f64;
                for d in 0..hd {
                    acc[d] += w * vs * fp8_e4m3_to_f64(vrow[d]);
                }
            }
            for d in 0..hd {
                out[(qi * nh + h) * hd + d] = acc[d] as f32;
            }
        }
    }
    out
}

fn cpu_ref(hq: &[u16], hk: &[u16], hv: &[u16], m: usize, nc: usize) -> Vec<f32> {
    let nh = NH as usize;
    let nkv = NKV as usize;
    let hd = HD as usize;
    let group = nh / nkv;
    let mut out = vec![0.0f32; m * nh * hd];
    for qi in 0..m {
        let bound = nc + qi + 1;
        for h in 0..nh {
            let kvh = h / group;
            let qrow = &hq[(qi * nh + h) * hd..(qi * nh + h) * hd + hd];
            let mut scores = vec![0.0f64; bound];
            let mut mx = f64::NEG_INFINITY;
            for p in 0..bound {
                let krow = &hk[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                let mut s = 0.0f64;
                for d in 0..hd {
                    s += bf(qrow[d]) * bf(krow[d]);
                }
                scores[p] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut l = 0.0f64;
            for p in 0..bound {
                scores[p] = (scores[p] - mx).exp();
                l += scores[p];
            }
            let mut acc = vec![0.0f64; hd];
            for p in 0..bound {
                let w = scores[p] / l;
                let vrow = &hv[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                for d in 0..hd {
                    acc[d] += w * bf(vrow[d]);
                }
            }
            for d in 0..hd {
                out[(qi * nh + h) * hd + d] = acc[d] as f32;
            }
        }
    }
    out
}

fn max_abs_diff_bf16(a: &[u16], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (half::bf16::from_bits(*x).to_f32() - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff_bf16_pair(a: &[u16], b: &[u16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            (half::bf16::from_bits(*x).to_f32() - half::bf16::from_bits(*y).to_f32()).abs()
        })
        .fold(0.0f32, f32::max)
}

fn chain_mask(m: usize) -> Vec<u8> {
    let mut mask = vec![0u8; m * m];
    for qi in 0..m {
        for j in 0..=qi {
            mask[qi * m + j] = 1;
        }
    }
    mask
}

struct Bufs {
    dq: CudaSlice<u16>,
    dk: CudaSlice<u16>,
    dv: CudaSlice<u16>,
    dnc: CudaSlice<i32>,
    dmask: CudaSlice<u8>,
    dscr: CudaSlice<f32>,
    dout_new: CudaSlice<u16>,
    dout_tree: CudaSlice<u16>,
}

fn make_bufs(
    stream: &Arc<CudaStream>,
    m: usize,
    nc: usize,
    seed: u64,
) -> (Bufs, Vec<u16>, Vec<u16>, Vec<u16>) {
    let total = nc + m;
    let mut st = seed;
    let kv_elems = total * NKV as usize * HD as usize;
    let hk: Vec<u16> = (0..kv_elems).map(|_| rnd_bf16(&mut st)).collect();
    let hv: Vec<u16> = (0..kv_elems).map(|_| rnd_bf16(&mut st)).collect();
    let hq: Vec<u16> = (0..m * NH as usize * HD as usize)
        .map(|_| rnd_bf16(&mut st))
        .collect();
    let b = Bufs {
        dq: stream.clone_htod(&hq).unwrap(),
        dk: stream.clone_htod(&hk).unwrap(),
        dv: stream.clone_htod(&hv).unwrap(),
        dnc: stream.clone_htod(&[nc as i32]).unwrap(),
        dmask: stream.clone_htod(&chain_mask(m)).unwrap(),
        dscr: stream
            .alloc_zeros::<f32>(cuda::gqa512_scratch_elems(NH, m as i32, 128))
            .unwrap(),
        dout_new: stream
            .alloc_zeros::<u16>(m * NH as usize * HD as usize)
            .unwrap(),
        dout_tree: stream
            .alloc_zeros::<u16>(m * NH as usize * HD as usize)
            .unwrap(),
    };
    (b, hq, hk, hv)
}

fn launch_new(stream: &Arc<CudaStream>, b: &mut Bufs, m: usize, splits: i32) {
    let (pq, _a) = b.dq.device_ptr(stream);
    let (pk, _b) = b.dk.device_ptr(stream);
    let (pv, _c) = b.dv.device_ptr(stream);
    let (pn, _d) = b.dnc.device_ptr(stream);
    let (ps, _e) = b.dscr.device_ptr_mut(stream);
    let (po, _f) = b.dout_new.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::gqa512_verify_bf16(
            stream.cu_stream() as *mut _,
            pq as *const u16,
            pk as *const u16,
            pv as *const u16,
            po as *mut u16,
            pn as *const i32,
            -(m as i32),
            m as i32,
            ps as *mut f32,
            NH,
            NKV,
            HD,
            splits,
        )
    };
    assert_eq!(rc, 0, "gqa512_verify_bf16 rc={rc}");
}

fn launch_tree(stream: &Arc<CudaStream>, b: &mut Bufs, m: usize) {
    let (pq, _a) = b.dq.device_ptr(stream);
    let (pk, _b) = b.dk.device_ptr(stream);
    let (pv, _c) = b.dv.device_ptr(stream);
    let (pn, _d) = b.dnc.device_ptr(stream);
    let (pm, _e) = b.dmask.device_ptr(stream);
    let (po, _f) = b.dout_tree.device_ptr_mut(stream);
    let rc = unsafe {
        cuda::tree_verify_attn_bf16(
            stream.cu_stream() as *mut _,
            pq as *const u16,
            pk as *const u16,
            pv as *const u16,
            pn as *const i32,
            pm as *const u8,
            std::ptr::null(),
            po as *mut u16,
            NH,
            NKV,
            HD,
            m as i32,
            0,
        )
    };
    assert_eq!(rc, 0, "tree_verify_attn_bf16 rc={rc}");
}

#[test]
fn parity_small() {
    let Some(stream) = stream_or_skip() else {
        return;
    };
    let nc = 200usize;
    for m in 1..=8usize {
        let (mut b, hq, hk, hv) = make_bufs(&stream, m, nc, 0x5eed_0000 + m as u64);
        let reference = cpu_ref(&hq, &hk, &hv, m, nc);
        launch_tree(&stream, &mut b, m);
        for splits in [3i32, 64] {
            launch_new(&stream, &mut b, m, splits);
            stream.synchronize().unwrap();
            let out_new = stream.clone_dtoh(&b.dout_new).unwrap();
            let d_ref = max_abs_diff_bf16(&out_new, &reference);
            assert!(
                d_ref < 0.02,
                "m={m} splits={splits}: new vs cpu ref max diff {d_ref}"
            );
        }
        stream.synchronize().unwrap();
        let out_tree = stream.clone_dtoh(&b.dout_tree).unwrap();
        let out_new = stream.clone_dtoh(&b.dout_new).unwrap();
        let d_tree_ref = max_abs_diff_bf16(&out_tree, &reference);
        let d_pair = max_abs_diff_bf16_pair(&out_new, &out_tree);
        assert!(
            d_tree_ref < 0.02,
            "m={m}: tree vs cpu ref max diff {d_tree_ref}"
        );
        assert!(d_pair < 0.02, "m={m}: new vs tree max diff {d_pair}");
        eprintln!("parity m={m}: new-vs-ref ok, tree-vs-ref ok, pair diff {d_pair:.4}");
    }
}

#[test]
fn parity_fp8() {
    let Some(stream) = stream_or_skip() else {
        return;
    };
    let nc = 200usize;
    for m in 1..=8usize {
        let total = nc + m;
        let mut st = 0xf8f8_0000 + m as u64;
        let kv_elems = total * NKV as usize * HD as usize;
        let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let nsc = total * NKV as usize;
        let hks: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let hvs: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let hq: Vec<u16> = (0..m * NH as usize * HD as usize)
            .map(|_| rnd_bf16(&mut st))
            .collect();
        let reference = cpu_ref_fp8(&hq, &hk8, &hv8, &hks, &hvs, m, nc);

        let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
        let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
        let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
        let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
        let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
        let dnc: CudaSlice<i32> = stream.clone_htod(&[nc as i32]).unwrap();
        let mut dscr: CudaSlice<f32> = stream
            .alloc_zeros::<f32>(cuda::gqa512_scratch_elems(NH, m as i32, 128))
            .unwrap();
        let mut dout: CudaSlice<u16> = stream
            .alloc_zeros::<u16>(m * NH as usize * HD as usize)
            .unwrap();

        for splits in [3i32, 64] {
            {
                let (pq, _a) = dq.device_ptr(&stream);
                let (pk, _b) = dk8.device_ptr(&stream);
                let (pv, _c) = dv8.device_ptr(&stream);
                let (pks, _d) = dks.device_ptr(&stream);
                let (pvs, _e) = dvs.device_ptr(&stream);
                let (pn, _f) = dnc.device_ptr(&stream);
                let (ps, _g) = dscr.device_ptr_mut(&stream);
                let (po, _h) = dout.device_ptr_mut(&stream);
                let rc = unsafe {
                    cuda::gqa512_verify_fp8(
                        stream.cu_stream() as *mut _,
                        pq as *const u16,
                        pk as *const u8,
                        pv as *const u8,
                        pks as *const f32,
                        pvs as *const f32,
                        po as *mut u16,
                        pn as *const i32,
                        -(m as i32),
                        m as i32,
                        ps as *mut f32,
                        NH,
                        NKV,
                        HD,
                        splits,
                        1.0,
                    )
                };
                assert_eq!(rc, 0, "gqa512_verify_fp8 rc={rc}");
            }
            stream.synchronize().unwrap();
            let out_new = stream.clone_dtoh(&dout).unwrap();
            let d_ref = max_abs_diff_bf16(&out_new, &reference);
            assert!(
                d_ref < 0.02,
                "m={m} splits={splits}: fp8 vs cpu ref max diff {d_ref}"
            );
        }
        eprintln!("parity_fp8 m={m}: ok");
    }
}

fn time_ms(stream: &Arc<CudaStream>, launches: usize, mut f: impl FnMut()) -> f64 {
    f();
    stream.synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..launches {
        f();
    }
    stream.synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1e3 / launches as f64
}

#[test]
fn perf_16k() {
    let Some(stream) = stream_or_skip() else {
        return;
    };
    let nh: i32 = std::env::var("NV_PERF_NH").ok().and_then(|v| v.parse().ok()).unwrap_or(NH);
    let nkv: i32 = std::env::var("NV_PERF_NKV").ok().and_then(|v| v.parse().ok()).unwrap_or(NKV);
    assert!(
        nh > 0 && nkv > 0 && nh % nkv == 0,
        "NH={nh} NKV={nkv}: the mk launcher refuses NH % NKV != 0"
    );
    let baselines_available = nh == nkv * 8;
    if !baselines_available {
        println!(
            "ratio {} != 8: gqa512_verify_bf16 requires NKV * kWarps == NH, so the new/tree \
             columns and the in-row cross-check are skipped. fp8mk correctness at this ratio \
             is anchored by fp8_mk_ratio_oracle.rs instead.",
            nh / nkv
        );
    }
    let nc: usize = std::env::var("NV_PERF_NC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(16384);
    let m_max = 8usize;
    let total_max = nc + m_max;
    let mut st = 0xbeef_cafe_u64;

    let kv_elems = total_max * nkv as usize * HD as usize;
    let hk: Vec<u16> = (0..kv_elems).map(|_| rnd_bf16(&mut st)).collect();
    let hv: Vec<u16> = (0..kv_elems).map(|_| rnd_bf16(&mut st)).collect();
    let hq: Vec<u16> = (0..m_max * nh as usize * HD as usize)
        .map(|_| rnd_bf16(&mut st))
        .collect();
    let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let nsc = total_max * nkv as usize;
    let hks: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hvs: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();

    let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
    let dk: CudaSlice<u16> = stream.clone_htod(&hk).unwrap();
    let dv: CudaSlice<u16> = stream.clone_htod(&hv).unwrap();
    let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
    let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
    let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
    let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
    let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh as usize).unwrap();
    let mut dout: CudaSlice<u16> = stream
        .alloc_zeros::<u16>(m_max * nh as usize * HD as usize)
        .unwrap();
    let mut dout_tree: CudaSlice<u16> = stream
        .alloc_zeros::<u16>(m_max * nh as usize * HD as usize)
        .unwrap();
    let mut dscr: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(cuda::gqa512_scratch_elems(nh, m_max as i32, 128))
        .unwrap();
    let mut dscr_mk: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(cuda::flash_splitk_scratch_elems_mk(nh, HD, m_max as i32))
        .unwrap();

    println!("shape: NH={nh} NKV={nkv} HD={HD} nc={nc} chunk={} (chain verify, window=0)",
        std::env::var("NV_MK512_CHUNK").unwrap_or_else(|_| "default".into()));
    println!(
        "unique KV bytes per pass: {:.1} MB",
        (nc + m_max) as f64 * nkv as f64 * HD as f64 * 2.0 * 2.0 / 1e6
    );
    println!("M | splits* | new_ms | newfp8_ms | tree_ms | fp8mk_ms | new GB/s | newfp8 GB/s | vs_tree | vs_fp8mk | fp8_vs_fp8mk");

    for m in 1..=m_max {
        let dnc: CudaSlice<i32> = stream.clone_htod(&[nc as i32]).unwrap();
        let dmask: CudaSlice<u8> = stream.clone_htod(&chain_mask(m)).unwrap();
        let total = nc + m;
        let unique_gb = total as f64 * nkv as f64 * HD as f64 * 2.0 * 2.0 / 1e9;

        let run_new = |splits: i32,
                       stream: &Arc<CudaStream>,
                       dscr: &mut CudaSlice<f32>,
                       dout: &mut CudaSlice<u16>| {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk.device_ptr(stream);
            let (pv, _c) = dv.device_ptr(stream);
            let (pn, _d) = dnc.device_ptr(stream);
            let (ps, _e) = dscr.device_ptr_mut(stream);
            let (po, _f) = dout.device_ptr_mut(stream);
            let rc = unsafe {
                cuda::gqa512_verify_bf16(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u16,
                    pv as *const u16,
                    po as *mut u16,
                    pn as *const i32,
                    -(m as i32),
                    m as i32,
                    ps as *mut f32,
                    nh,
                    nkv,
                    HD,
                    splits,
                )
            };
            assert_eq!(rc, 0);
        };
        let run_tree = |stream: &Arc<CudaStream>, dout_tree: &mut CudaSlice<u16>| {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk.device_ptr(stream);
            let (pv, _c) = dv.device_ptr(stream);
            let (pn, _d) = dnc.device_ptr(stream);
            let (pm, _e) = dmask.device_ptr(stream);
            let (po, _f) = dout_tree.device_ptr_mut(stream);
            let rc = unsafe {
                cuda::tree_verify_attn_bf16(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u16,
                    pv as *const u16,
                    pn as *const i32,
                    pm as *const u8,
                    std::ptr::null(),
                    po as *mut u16,
                    nh,
                    nkv,
                    HD,
                    m as i32,
                    0,
                )
            };
            assert_eq!(rc, 0);
        };
        let run_fp8 = |stream: &Arc<CudaStream>,
                       dscr_mk: &mut CudaSlice<f32>,
                       dfan: &mut CudaSlice<u32>,
                       dout: &mut CudaSlice<u16>| {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk8.device_ptr(stream);
            let (pv, _c) = dv8.device_ptr(stream);
            let (pks, _d) = dks.device_ptr(stream);
            let (pvs, _e) = dvs.device_ptr(stream);
            let (pn, _f) = dnc.device_ptr(stream);
            let (ps, _g) = dscr_mk.device_ptr_mut(stream);
            let (pf, _h) = dfan.device_ptr_mut(stream);
            let (po, _i) = dout.device_ptr_mut(stream);
            let rc = unsafe {
                cuda::flash_decode_fused_fp8kv_mk(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    po as *mut u16,
                    pn as *const i32,
                    -(m as i32),
                    m as i32,
                    ps as *mut f32,
                    pf as *mut u32,
                    nh,
                    nkv,
                    HD,
                    0,
                    0,
                    1.0,
                )
            };
            assert_eq!(rc, 0);
        };

        let run_new_fp8 = |splits: i32,
                           stream: &Arc<CudaStream>,
                           dscr: &mut CudaSlice<f32>,
                           dout: &mut CudaSlice<u16>| {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk8.device_ptr(stream);
            let (pv, _c) = dv8.device_ptr(stream);
            let (pks, _d) = dks.device_ptr(stream);
            let (pvs, _e) = dvs.device_ptr(stream);
            let (pn, _f) = dnc.device_ptr(stream);
            let (ps, _g) = dscr.device_ptr_mut(stream);
            let (po, _h) = dout.device_ptr_mut(stream);
            let rc = unsafe {
                cuda::gqa512_verify_fp8(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    po as *mut u16,
                    pn as *const i32,
                    -(m as i32),
                    m as i32,
                    ps as *mut f32,
                    nh,
                    nkv,
                    HD,
                    splits,
                    1.0,
                )
            };
            assert_eq!(rc, 0);
        };

        let mut best_splits = 64i32;
        let mut best_ms = f64::INFINITY;
        let mut best_splits_f8 = 64i32;
        let mut best_ms_f8 = f64::INFINITY;
        for splits in if baselines_available { &[16i32, 32, 48, 64, 96, 128][..] } else { &[][..] } {
            let splits = *splits;
            let ms = time_ms(&stream, 5, || {
                run_new(splits, &stream, &mut dscr, &mut dout)
            });
            if ms < best_ms {
                best_ms = ms;
                best_splits = splits;
            }
            let ms8 = time_ms(&stream, 5, || {
                run_new_fp8(splits, &stream, &mut dscr, &mut dout)
            });
            if ms8 < best_ms_f8 {
                best_ms_f8 = ms8;
                best_splits_f8 = splits;
            }
        }

        let rounds = 6usize;
        let mut t_new = Vec::new();
        let mut t_new8 = Vec::new();
        let mut t_tree = Vec::new();
        let mut t_fp8 = Vec::new();
        for r in 0..rounds {
            let (a, a8, b) = if baselines_available {
                (
                    time_ms(&stream, 10, || {
                        run_new(best_splits, &stream, &mut dscr, &mut dout)
                    }),
                    time_ms(&stream, 10, || {
                        run_new_fp8(best_splits_f8, &stream, &mut dscr, &mut dout)
                    }),
                    time_ms(&stream, 4, || run_tree(&stream, &mut dout_tree)),
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN)
            };
            let c = time_ms(&stream, 10, || {
                run_fp8(&stream, &mut dscr_mk, &mut dfan, &mut dout)
            });
            if r > 0 {
                if baselines_available {
                    t_new.push(a);
                    t_new8.push(a8);
                    t_tree.push(b);
                }
                t_fp8.push(c);
            }
        }
        let med = |v: &mut Vec<f64>| {
            if v.is_empty() {
                return f64::NAN;
            }
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let new_ms = med(&mut t_new);
        let new8_ms = med(&mut t_new8);
        let tree_ms = med(&mut t_tree);
        let fp8_ms = med(&mut t_fp8);
        let unique_gb_f8 = unique_gb / 2.0;
        println!(
            "{m} | {best_splits}/{best_splits_f8} | {new_ms:.3} | {new8_ms:.3} | {tree_ms:.3} | {fp8_ms:.3} | {:.0} | {:.0} | {:.2}x | {:.2}x | {:.2}x",
            unique_gb / (new_ms / 1e3),
            unique_gb_f8 / (new8_ms / 1e3),
            tree_ms / new_ms,
            fp8_ms / new_ms,
            fp8_ms / new8_ms,
        );

        if !baselines_available {
            continue;
        }
        run_new(best_splits, &stream, &mut dscr, &mut dout);
        run_tree(&stream, &mut dout_tree);
        stream.synchronize().unwrap();
        let out_new = stream.clone_dtoh(&dout).unwrap();
        let out_tree = stream.clone_dtoh(&dout_tree).unwrap();
        let n = m * nh as usize * HD as usize;
        let d = max_abs_diff_bf16_pair(&out_new[..n], &out_tree[..n]);
        assert!(d < 0.02, "m={m}: 16k new vs tree max diff {d}");

        run_new_fp8(best_splits_f8, &stream, &mut dscr, &mut dout);
        run_fp8(&stream, &mut dscr_mk, &mut dfan, &mut dout_tree);
        stream.synchronize().unwrap();
        let out_a = stream.clone_dtoh(&dout).unwrap();
        let out_b = stream.clone_dtoh(&dout_tree).unwrap();
        let d8 = max_abs_diff_bf16_pair(&out_a[..n], &out_b[..n]);
        assert!(d8 < 0.02, "m={m}: 16k newfp8 vs fp8mk max diff {d8}");
    }
}
