#![cfg(feature = "cuda")]

mod common;
use common::lcg;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use nv_kernels::cuda;
use std::sync::Arc;
use common::fp8_e4m3_to_f64;
use common::rnd_fp8;

const HD: usize = 512;
const NC: usize = 200;

fn stream_or_skip() -> Option<Arc<CudaStream>> {
    match CudaContext::new(0) {
        Ok(ctx) => Some(ctx.default_stream()),
        Err(e) => {
            if std::env::var("NV_GQA512_ALLOW_SKIP").as_deref() == Ok("1") {
                eprintln!("SKIP (NV_GQA512_ALLOW_SKIP=1): no CUDA device 0: {e}");
                return None;
            }
            panic!(
                "fp8_mk_ratio_oracle: no CUDA device 0: {e}. This is a correctness gate; \
                 it refuses to report success without running. Set NV_GQA512_ALLOW_SKIP=1 \
                 to skip on purpose."
            );
        }
    }
}

fn rnd_bf16(state: &mut u64) -> u16 {
    half::bf16::from_f32(((lcg(state) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0).to_bits()
}

fn bf(x: u16) -> f64 {
    half::bf16::from_bits(x).to_f32() as f64
}

#[allow(clippy::too_many_arguments)]
fn oracle(
    hq: &[u16],
    hk8: &[u8],
    hv8: &[u8],
    hks: &[f32],
    hvs: &[f32],
    m: usize,
    nh: usize,
    nkv: usize,
) -> Vec<f32> {
    let group = nh / nkv;
    let mut out = vec![0.0f32; m * nh * HD];
    for qi in 0..m {
        let bound = NC + qi + 1;
        for h in 0..nh {
            let kvh = h / group;
            let qrow = &hq[(qi * nh + h) * HD..(qi * nh + h) * HD + HD];
            let mut scores = vec![0.0f64; bound];
            let mut mx = f64::NEG_INFINITY;
            for (p, sc) in scores.iter_mut().enumerate() {
                let krow = &hk8[(p * nkv + kvh) * HD..(p * nkv + kvh) * HD + HD];
                let ks = hks[p * nkv + kvh] as f64;
                let mut s = 0.0f64;
                for d in 0..HD {
                    s += bf(qrow[d]) * fp8_e4m3_to_f64(krow[d]);
                }
                *sc = s * ks;
                if *sc > mx {
                    mx = *sc;
                }
            }
            let mut l = 0.0f64;
            for sc in scores.iter_mut() {
                *sc = (*sc - mx).exp();
                l += *sc;
            }
            let mut acc = vec![0.0f64; HD];
            for (p, sc) in scores.iter().enumerate() {
                let w = sc / l;
                let vrow = &hv8[(p * nkv + kvh) * HD..(p * nkv + kvh) * HD + HD];
                let vs = hvs[p * nkv + kvh] as f64;
                for d in 0..HD {
                    acc[d] += w * vs * fp8_e4m3_to_f64(vrow[d]);
                }
            }
            for d in 0..HD {
                out[(qi * nh + h) * HD + d] = acc[d] as f32;
            }
        }
    }
    out
}

fn force_chunk_3() {

    std::env::set_var("NV_MK512_CHUNK", "3");
}

fn check_shape(stream: &Arc<CudaStream>, nh: usize, nkv: usize) {
    force_chunk_3();
    for m in 1..=8usize {
        let total = NC + m;
        let mut st = 0x8f8f_0000 ^ ((nh as u64) << 32) ^ ((nkv as u64) << 16) ^ m as u64;
        let kv_elems = total * nkv * HD;
        let hq: Vec<u16> = (0..m * nh * HD).map(|_| rnd_bf16(&mut st)).collect();
        let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let nsc = total * nkv;
        let hks: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let hvs: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let reference = oracle(&hq, &hk8, &hv8, &hks, &hvs, m, nh, nkv);

        let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
        let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
        let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
        let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
        let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
        let dnc: CudaSlice<i32> = stream.clone_htod(&[NC as i32]).unwrap();
        let mut dscr: CudaSlice<f32> = stream
            .alloc_zeros::<f32>(cuda::flash_splitk_scratch_elems_mk(
                nh as i32, HD as i32, m as i32,
            ))
            .unwrap();
        let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
        let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * HD).unwrap();

        let rc = {
        let (pq, _a) = dq.device_ptr(stream);
        let (pk, _b) = dk8.device_ptr(stream);
        let (pv, _c) = dv8.device_ptr(stream);
        let (pks, _d) = dks.device_ptr(stream);
        let (pvs, _e) = dvs.device_ptr(stream);
        let (pn, _f) = dnc.device_ptr(stream);
        let (ps, _g) = dscr.device_ptr_mut(stream);
        let (pf, _h) = dfan.device_ptr_mut(stream);
        let (po, _i) = dout.device_ptr_mut(stream);
        unsafe {
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
                nh as i32,
                nkv as i32,
                HD as i32,
                0,
                0,
                1.0,
            )
        }
        };
        assert_eq!(rc, 0, "NH={nh} NKV={nkv} m={m}: launcher refused");
        stream.synchronize().unwrap();
        let out = stream.clone_dtoh(&dout).unwrap();
        let mut worst = 0f32;
        for (x, y) in out.iter().zip(reference.iter()) {
            let g = half::bf16::from_bits(*x).to_f32();
            assert!(g.is_finite(), "NH={nh} NKV={nkv} m={m}: non-finite output");
            worst = worst.max((g - y).abs());
        }
        assert!(
            worst < 0.02,
            "NH={nh} NKV={nkv} m={m} (chunk=3, so m>3 is multi-launch): fp8 mk vs f64 \
             oracle max diff {worst}"
        );
    }
}

#[test]
fn ratio_8_matches_the_oracle_which_ties_this_file_to_the_existing_cross_check() {
    let Some(stream) = stream_or_skip() else { return };
    check_shape(&stream, 32, 4);
}

#[test]
fn ratio_4_matches_the_oracle_which_no_other_test_can_reach() {
    let Some(stream) = stream_or_skip() else { return };
    check_shape(&stream, 32, 8);
}

#[test]
fn ratio_1_pure_mha_matches_the_oracle() {
    let Some(stream) = stream_or_skip() else { return };
    check_shape(&stream, 8, 8);
}

#[test]
fn ratio_32_single_kv_head_matches_the_oracle() {
    let Some(stream) = stream_or_skip() else { return };
    check_shape(&stream, 32, 1);
}
