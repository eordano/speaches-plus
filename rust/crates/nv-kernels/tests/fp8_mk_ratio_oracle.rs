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
    oracle_hd(hq, hk8, hv8, hks, hvs, m, nh, nkv, HD)
}

#[allow(clippy::too_many_arguments)]
fn oracle_hd(
    hq: &[u16],
    hk8: &[u8],
    hv8: &[u8],
    hks: &[f32],
    hvs: &[f32],
    m: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
) -> Vec<f32> {
    let group = nh / nkv;
    let mut out = vec![0.0f32; m * nh * hd];
    for qi in 0..m {
        let bound = NC + qi + 1;
        for h in 0..nh {
            let kvh = h / group;
            let qrow = &hq[(qi * nh + h) * hd..(qi * nh + h) * hd + hd];
            let mut scores = vec![0.0f64; bound];
            let mut mx = f64::NEG_INFINITY;
            for (p, sc) in scores.iter_mut().enumerate() {
                let krow = &hk8[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
                let ks = hks[p * nkv + kvh] as f64;
                let mut s = 0.0f64;
                for d in 0..hd {
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
            let mut acc = vec![0.0f64; hd];
            for (p, sc) in scores.iter().enumerate() {
                let w = sc / l;
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

const HD_256_THE_KVSHARE_TEMPLATE_SHAPE: usize = 256;
const KVSHARE_MAX_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS: usize = 128;
const ORACLE_BAND_0_02_MATCHES_THE_MK_CHECKS_ABOVE: f32 = 0.02;
const CROSS_ROUTE_BAND_IS_BF16_ROUNDING_PLUS_2_PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY: (f32, f32) =
    (1e-3, 0.02);

fn check_kvshare_rowseq_shape(stream: &Arc<CudaStream>, nh: usize, nkv: usize) {
    let hd = HD_256_THE_KVSHARE_TEMPLATE_SHAPE;
    for m in 2..=8usize {
        let total = NC + m;
        let mut st = 0x1d2d_0000 ^ ((nh as u64) << 32) ^ ((nkv as u64) << 16) ^ m as u64;
        let kv_elems = total * nkv * hd;
        let hq: Vec<u16> = (0..m * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
        let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let nsc = total * nkv;
        let hks: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let hvs: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let reference = oracle_hd(&hq, &hk8, &hv8, &hks, &hvs, m, nh, nkv, hd);

        let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
        let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
        let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
        let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
        let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
        let dnc: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
        let scratch_elems = (nh * KVSHARE_MAX_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS * (hd + 2))
            .max(cuda::flash_splitk_scratch_elems_mk(nh as i32, hd as i32, m as i32) as usize);
        let mut dscr: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
        let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
        let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();
        let mut dout_mk: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();

        {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk8.device_ptr(stream);
            let (pv, _c) = dv8.device_ptr(stream);
            let (pks, _d) = dks.device_ptr(stream);
            let (pvs, _e) = dvs.device_ptr(stream);
            let (pn, _f) = dnc.device_ptr(stream);
            let (ps, _g) = dscr.device_ptr_mut(stream);
            let (pf, _h) = dfan.device_ptr_mut(stream);
            let (po, _i) = dout.device_ptr_mut(stream);
            let (pomk, _j) = dout_mk.device_ptr_mut(stream);
            let row_stride_bytes = (nh * hd * std::mem::size_of::<u16>()) as u64;
            for qi in 0..m {
                let rc = unsafe {
                    cuda::flash_decode_kvshare_fp8kv_paged(
                        stream.cu_stream() as *mut _,
                        (pq + qi as u64 * row_stride_bytes) as *const u16,
                        pk as *const u8,
                        pv as *const u8,
                        pks as *const f32,
                        pvs as *const f32,
                        (po + qi as u64 * row_stride_bytes) as *mut u16,
                        pn as *const i32,
                        ps as *mut f32,
                        pf as *mut u32,
                        nh as i32,
                        nkv as i32,
                        hd as i32,
                        0,
                        0,
                        0,
                        1.0,
                        std::ptr::null(),
                        0,
                        (m - 1 - qi) as i32,
                        0,
                        0,
                    )
                };
                assert_eq!(rc, 0, "NH={nh} NKV={nkv} m={m} qi={qi}: kvshare launcher refused");
            }
            let rc = unsafe {
                cuda::flash_decode_fused_fp8kv_mk(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    pomk as *mut u16,
                    pn as *const i32,
                    0,
                    m as i32,
                    ps as *mut f32,
                    pf as *mut u32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    0,
                    0,
                    1.0,
                )
            };
            assert_eq!(rc, 0, "NH={nh} NKV={nkv} m={m}: mk launcher refused");
        }
        stream.synchronize().unwrap();
        let out = stream.clone_dtoh(&dout).unwrap();
        let out_mk = stream.clone_dtoh(&dout_mk).unwrap();

        let mut worst = 0f32;
        for (x, y) in out.iter().zip(reference.iter()) {
            let g = half::bf16::from_bits(*x).to_f32();
            assert!(g.is_finite(), "NH={nh} NKV={nkv} m={m}: non-finite kvshare output");
            worst = worst.max((g - y).abs());
        }
        assert!(
            worst < ORACLE_BAND_0_02_MATCHES_THE_MK_CHECKS_ABOVE,
            "NH={nh} NKV={nkv} m={m}: row-sequential kvshare (delta=m-1-qi) vs f64 oracle max \
             diff {worst}; a delta bug shows up as a whole extra or missing chain row in the \
             softmax, far above this band"
        );

        let max_mag = out_mk
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32().abs())
            .fold(0f32, f32::max);
        let cross = out
            .iter()
            .zip(out_mk.iter())
            .map(|(a, b)| {
                (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs()
            })
            .fold(0f32, f32::max);
        let (abs_band, rel_band) =
            CROSS_ROUTE_BAND_IS_BF16_ROUNDING_PLUS_2_PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY;
        assert!(
            cross <= abs_band + rel_band * max_mag,
            "NH={nh} NKV={nkv} m={m}: kvshare row-sequential vs mk kernel max diff {cross} \
             (max|mk|={max_mag}); both accumulate f32 online-softmax over the same fp8 rows, so \
             beyond reduction-order rounding means an indexing bug"
        );
    }
}

#[test]
fn q38_shape_24q_4kv_hd256_row_sequential_kvshare_with_delta_matches_oracle_and_mk() {
    let Some(stream) = stream_or_skip() else { return };
    check_kvshare_rowseq_shape(&stream, 24, 4);
}

#[test]
fn q36_shape_16q_2kv_hd256_row_sequential_kvshare_with_delta_matches_oracle_and_mk() {
    let Some(stream) = stream_or_skip() else { return };
    check_kvshare_rowseq_shape(&stream, 16, 2);
}

const ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M: usize = 8 * 32;

#[derive(Clone, Copy, PartialEq)]
enum OneLaunchArm {
    RowsharedScalar,
    TctileMma,
}

#[allow(clippy::too_many_arguments)]
fn run_one_launch_mrows(
    arm: OneLaunchArm,
    stream: &Arc<CudaStream>,
    hq: &[u16],
    hk8: &[u8],
    hv8: &[u8],
    hks: &[f32],
    hvs: &[f32],
    m: usize,
    nh: usize,
    nkv: usize,
    hd: usize,
    n_total: usize,
    hole: (i32, i32, i32),
) -> Vec<u16> {
    let dq: CudaSlice<u16> = stream.clone_htod(hq).unwrap();
    let dk8: CudaSlice<u8> = stream.clone_htod(hk8).unwrap();
    let dv8: CudaSlice<u8> = stream.clone_htod(hv8).unwrap();
    let dks: CudaSlice<f32> = stream.clone_htod(hks).unwrap();
    let dvs: CudaSlice<f32> = stream.clone_htod(hvs).unwrap();
    let dnc: CudaSlice<i32> = stream.clone_htod(&[n_total as i32]).unwrap();
    let mut dscr: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(nh * ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M * (hd + 2))
        .unwrap();
    let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
    let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();
    {
        let (pq, _a) = dq.device_ptr(stream);
        let (pk, _b) = dk8.device_ptr(stream);
        let (pv, _c) = dv8.device_ptr(stream);
        let (pks, _d) = dks.device_ptr(stream);
        let (pvs, _e) = dvs.device_ptr(stream);
        let (pn, _f) = dnc.device_ptr(stream);
        let (ps, _g) = dscr.device_ptr_mut(stream);
        let (pf, _h) = dfan.device_ptr_mut(stream);
        let (po, _i) = dout.device_ptr_mut(stream);
        let (hole_from_row, hole_lo_delta, hole_hi_delta) = hole;
        let launcher = match arm {
            OneLaunchArm::RowsharedScalar => cuda::flash_decode_rowshared_fp8kv_mk_paged,
            OneLaunchArm::TctileMma => cuda::flash_decode_rowshared_tctile_fp8kv_mk_paged,
        };
        let rc = unsafe {
            launcher(
                stream.cu_stream() as *mut _,
                pq as *const u16,
                pk as *const u8,
                pv as *const u8,
                pks as *const f32,
                pvs as *const f32,
                po as *mut u16,
                pn as *const i32,
                ps as *mut f32,
                pf as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
                0,
                0,
                (ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M / m) as i32,
                1.0,
                std::ptr::null(),
                0,
                0,
                m as i32,
                hole_from_row,
                hole_lo_delta,
                hole_hi_delta,
            )
        };
        assert_eq!(rc, 0, "NH={nh} NKV={nkv} m={m}: one-launch m-rows launcher refused");
    }
    stream.synchronize().unwrap();
    stream.clone_dtoh(&dout).unwrap()
}

fn check_one_launch_mrows_shape(arm: OneLaunchArm, stream: &Arc<CudaStream>, nh: usize, nkv: usize) {
    let hd = HD_256_THE_KVSHARE_TEMPLATE_SHAPE;
    for m in 2..=8usize {
        let total = NC + m;
        let mut st = 0x2e3e_0000 ^ ((nh as u64) << 32) ^ ((nkv as u64) << 16) ^ m as u64;
        let kv_elems = total * nkv * hd;
        let hq: Vec<u16> = (0..m * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
        let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
        let nsc = total * nkv;
        let hks: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let hvs: Vec<f32> = (0..nsc)
            .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
            .collect();
        let reference = oracle_hd(&hq, &hk8, &hv8, &hks, &hvs, m, nh, nkv, hd);

        let out = run_one_launch_mrows(
            arm, stream, &hq, &hk8, &hv8, &hks, &hvs, m, nh, nkv, hd, total, (m as i32, 0, 0),
        );

        let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
        let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
        let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
        let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
        let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
        let dnc: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
        let mut dscr: CudaSlice<f32> = stream
            .alloc_zeros::<f32>(
                nh * ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M * (hd + 2),
            )
            .unwrap();
        let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
        let mut dout_mk: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();
        {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk8.device_ptr(stream);
            let (pv, _c) = dv8.device_ptr(stream);
            let (pks, _d) = dks.device_ptr(stream);
            let (pvs, _e) = dvs.device_ptr(stream);
            let (pn, _f) = dnc.device_ptr(stream);
            let (ps, _g) = dscr.device_ptr_mut(stream);
            let (pf, _h) = dfan.device_ptr_mut(stream);
            let (pomk, _j) = dout_mk.device_ptr_mut(stream);
            let rc = unsafe {
                cuda::flash_decode_fused_fp8kv_mk(
                    stream.cu_stream() as *mut _,
                    pq as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    pomk as *mut u16,
                    pn as *const i32,
                    0,
                    m as i32,
                    ps as *mut f32,
                    pf as *mut u32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    0,
                    0,
                    1.0,
                )
            };
            assert_eq!(rc, 0, "NH={nh} NKV={nkv} m={m}: mk launcher refused");
        }
        stream.synchronize().unwrap();
        let out_mk = stream.clone_dtoh(&dout_mk).unwrap();

        let mut worst = 0f32;
        for (x, y) in out.iter().zip(reference.iter()) {
            let g = half::bf16::from_bits(*x).to_f32();
            assert!(g.is_finite(), "NH={nh} NKV={nkv} m={m}: non-finite rowshared output");
            worst = worst.max((g - y).abs());
        }
        assert!(
            worst < ORACLE_BAND_0_02_MATCHES_THE_MK_CHECKS_ABOVE,
            "NH={nh} NKV={nkv} m={m}: one-launch rowshared vs f64 oracle max diff {worst}; a \
             per-row causal-bound bug shows up as a whole extra or missing chain row in the \
             softmax, far above this band"
        );

        let max_mag = out_mk
            .iter()
            .map(|b| half::bf16::from_bits(*b).to_f32().abs())
            .fold(0f32, f32::max);
        let cross = out
            .iter()
            .zip(out_mk.iter())
            .map(|(a, b)| {
                (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs()
            })
            .fold(0f32, f32::max);
        let (abs_band, rel_band) =
            CROSS_ROUTE_BAND_IS_BF16_ROUNDING_PLUS_2_PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY;
        assert!(
            cross <= abs_band + rel_band * max_mag,
            "NH={nh} NKV={nkv} m={m}: rowshared one-launch vs mk kernel max diff {cross} \
             (max|mk|={max_mag}); both accumulate f32 online-softmax over the same fp8 rows, so \
             beyond reduction-order rounding means an indexing bug"
        );
    }
}

#[test]
fn q38_shape_24q_4kv_hd256_one_launch_rowshared_matches_oracle_and_mk_for_m_2_to_8() {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_shape(OneLaunchArm::RowsharedScalar, &stream, 24, 4);
}

#[test]
fn q36_shape_16q_2kv_hd256_one_launch_rowshared_matches_oracle_and_mk_for_m_2_to_8() {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_shape(OneLaunchArm::RowsharedScalar, &stream, 16, 2);
}

#[test]
fn q38_shape_24q_4kv_hd256_one_launch_tctile_matches_oracle_and_mk_for_m_2_to_8() {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_shape(OneLaunchArm::TctileMma, &stream, 24, 4);
}

const TCTILE_GROUP8_REFUSAL_RC_MINUS_27_MATCHES_THE_KERNELS_NAMED_CONSTANT: i32 = -27;

#[test]
fn tctile_refuses_the_q36_group8_shape_with_rc_minus_27_so_the_route_can_fall_through() {
    let Some(stream) = stream_or_skip() else { return };
    let (nh, nkv, hd, m) = (16usize, 2usize, HD_256_THE_KVSHARE_TEMPLATE_SHAPE, 4usize);
    let total = NC + m;
    let mut st = 0x5f5f_0004u64;
    let hq: Vec<u16> = (0..m * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
    let hk8: Vec<u8> = (0..total * nkv * hd).map(|_| rnd_fp8(&mut st)).collect();
    let hks: Vec<f32> = (0..total * nkv).map(|_| 0.002f32).collect();
    let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
    let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
    let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
    let dnc: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
    let mut dscr: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(nh * ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M * (hd + 2))
        .unwrap();
    let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
    let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();
    let (pq, _a) = dq.device_ptr(&stream);
    let (pk, _b) = dk8.device_ptr(&stream);
    let (pks, _d) = dks.device_ptr(&stream);
    let (pn, _f) = dnc.device_ptr(&stream);
    let (ps, _g) = dscr.device_ptr_mut(&stream);
    let (pf, _h) = dfan.device_ptr_mut(&stream);
    let (po, _i) = dout.device_ptr_mut(&stream);
    let rc = unsafe {
        cuda::flash_decode_rowshared_tctile_fp8kv_mk_paged(
            stream.cu_stream() as *mut _,
            pq as *const u16,
            pk as *const u8,
            pk as *const u8,
            pks as *const f32,
            pks as *const f32,
            po as *mut u16,
            pn as *const i32,
            ps as *mut f32,
            pf as *mut u32,
            nh as i32,
            nkv as i32,
            hd as i32,
            0,
            0,
            0,
            (ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M / m) as i32,
            1.0,
            std::ptr::null(),
            0,
            0,
            m as i32,
            m as i32,
            0,
            0,
        )
    };
    assert_eq!(
        rc, TCTILE_GROUP8_REFUSAL_RC_MINUS_27_MATCHES_THE_KERNELS_NAMED_CONSTANT,
        "the tctile launcher must refuse group 8 with its own rc so \
         verify_attention_fp8_mk can keep the qwen36 shape on the scalar arms"
    );
}

fn check_one_launch_mrows_row0_m_invariance(arm: OneLaunchArm, stream: &Arc<CudaStream>) {
    let (nh, nkv, hd) = (24usize, 4usize, HD_256_THE_KVSHARE_TEMPLATE_SHAPE);
    let mut st = 0x4c5c_0001u64;
    let kv_elems = (NC + 8) * nkv * hd;
    let hq: Vec<u16> = (0..8 * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
    let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let nsc = (NC + 8) * nkv;
    let hks: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hvs: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let mut row0_by_m: Vec<Vec<u16>> = Vec::new();
    for m in 2..=8usize {
        let out = run_one_launch_mrows(
            arm,
            stream,
            &hq[..m * nh * hd],
            &hk8,
            &hv8,
            &hks,
            &hvs,
            m,
            nh,
            nkv,
            hd,
            NC + m,
            (m as i32, 0, 0),
        );
        row0_by_m.push(out[..nh * hd].to_vec());
    }
    let base = &row0_by_m[0];
    let max_mag = base
        .iter()
        .map(|b| half::bf16::from_bits(*b).to_f32().abs())
        .fold(0f32, f32::max);
    let (abs_band, rel_band) =
        CROSS_ROUTE_BAND_IS_BF16_ROUNDING_PLUS_2_PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY;
    for (i, row0) in row0_by_m.iter().enumerate().skip(1) {
        let m = i + 2;
        let cross = row0
            .iter()
            .zip(base.iter())
            .map(|(a, b)| {
                (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs()
            })
            .fold(0f32, f32::max);
        assert!(
            cross <= abs_band + rel_band * max_mag,
            "m={m}: row0 attends [0, NC+1) at every m, so a diff {cross} beyond the \
             warp-partition reduction-order band (max|row0|={max_mag}) means the per-row causal \
             bound leaks later chain rows into row0"
        );
    }
}

#[test]
fn rowshared_row0_window_is_nc_plus_1_for_every_m_so_row0_output_is_m_invariant_within_the_cross_route_band(
) {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_row0_m_invariance(OneLaunchArm::RowsharedScalar, &stream);
}

#[test]
fn tctile_row0_window_is_nc_plus_1_for_every_m_so_row0_output_is_m_invariant_within_the_cross_route_band(
) {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_row0_m_invariance(OneLaunchArm::TctileMma, &stream);
}

const ROWSHARED_TREE_HOLE_M6_FROM_ROW_4_DELTAS_5_2_IS_THE_SERVING_SIBLING_MASK: (i32, i32, i32) =
    (4, 5, 2);

fn check_one_launch_mrows_hole_matches_mk_hole(arm: OneLaunchArm, stream: &Arc<CudaStream>) {
    let (nh, nkv, hd) = (24usize, 4usize, HD_256_THE_KVSHARE_TEMPLATE_SHAPE);
    let m = 6usize;
    let total = NC + m;
    let mut st = 0x7a8a_0006u64;
    let kv_elems = total * nkv * hd;
    let hq: Vec<u16> = (0..m * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
    let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let hv8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let nsc = total * nkv;
    let hks: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hvs: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hole = ROWSHARED_TREE_HOLE_M6_FROM_ROW_4_DELTAS_5_2_IS_THE_SERVING_SIBLING_MASK;
    let out_hole =
        run_one_launch_mrows(arm, stream, &hq, &hk8, &hv8, &hks, &hvs, m, nh, nkv, hd, total, hole);
    let out_plain = run_one_launch_mrows(
        arm, stream, &hq, &hk8, &hv8, &hks, &hvs, m, nh, nkv, hd, total, (m as i32, 0, 0),
    );
    assert_ne!(
        out_hole[(hole.0 as usize) * nh * hd..],
        out_plain[(hole.0 as usize) * nh * hd..],
        "the hole window must change the masked rows or this test gates nothing"
    );
    assert_eq!(
        out_hole[..(hole.0 as usize) * nh * hd],
        out_plain[..(hole.0 as usize) * nh * hd],
        "rows below hole_from_row must be bitwise unaffected by the hole window"
    );

    let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
    let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
    let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
    let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
    let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
    let dnc: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
    let mut dscr: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(nh * ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M * (hd + 2))
        .unwrap();
    let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
    let mut dout_mk: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();
    {
        let (pq, _a) = dq.device_ptr(&stream);
        let (pk, _b) = dk8.device_ptr(&stream);
        let (pv, _c) = dv8.device_ptr(&stream);
        let (pks, _d) = dks.device_ptr(&stream);
        let (pvs, _e) = dvs.device_ptr(&stream);
        let (pn, _f) = dnc.device_ptr(&stream);
        let (ps, _g) = dscr.device_ptr_mut(&stream);
        let (pf, _h) = dfan.device_ptr_mut(&stream);
        let (pomk, _j) = dout_mk.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::flash_decode_fused_fp8kv_mk_hole(
                stream.cu_stream() as *mut _,
                pq as *const u16,
                pk as *const u8,
                pv as *const u8,
                pks as *const f32,
                pvs as *const f32,
                pomk as *mut u16,
                pn as *const i32,
                0,
                m as i32,
                ps as *mut f32,
                pf as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
                0,
                1.0,
                hole.0,
                hole.1,
                hole.2,
            )
        };
        assert_eq!(rc, 0, "mk hole launcher refused");
    }
    stream.synchronize().unwrap();
    let out_mk = stream.clone_dtoh(&dout_mk).unwrap();
    let max_mag = out_mk
        .iter()
        .map(|b| half::bf16::from_bits(*b).to_f32().abs())
        .fold(0f32, f32::max);
    let cross = out_hole
        .iter()
        .zip(out_mk.iter())
        .map(|(a, b)| {
            (half::bf16::from_bits(*a).to_f32() - half::bf16::from_bits(*b).to_f32()).abs()
        })
        .fold(0f32, f32::max);
    let (abs_band, rel_band) =
        CROSS_ROUTE_BAND_IS_BF16_ROUNDING_PLUS_2_PCT_MATCHING_FLASH_FP8_KVSHARE_PARITY;
    assert!(
        cross <= abs_band + rel_band * max_mag,
        "one-launch hole vs mk hole max diff {cross} (max|mk|={max_mag}): the two arms mask the \
         same [n_total-5, n_total-2) window for rows 4..6, so beyond reduction-order rounding \
         means a hole indexing bug"
    );
}

#[test]
fn rowshared_hole_mask_matches_the_mk_hole_arm_and_actually_excludes_the_hole_rows() {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_hole_matches_mk_hole(OneLaunchArm::RowsharedScalar, &stream);
}

#[test]
fn tctile_hole_mask_matches_the_mk_hole_arm_and_actually_excludes_the_hole_rows() {
    let Some(stream) = stream_or_skip() else { return };
    check_one_launch_mrows_hole_matches_mk_hole(OneLaunchArm::TctileMma, &stream);
}

#[test]
#[ignore = "kernel-only A/B timing at the 172k serving depth, m=4 24q/4kv hd256: median ms for \
            (a) the m-sequential kvshare launches, (b) the one-launch rowshared kernel, (c) the \
            mk fused kernel, (d) the one-launch tctile mma kernel; run solo on an idle GPU with \
            -- --ignored"]
fn rowshared_vs_kvshare_vs_mk_kernel_only_ms_at_the_172k_serving_shape() {
    let Some(stream) = stream_or_skip() else { return };
    let (nh, nkv, hd, m) = (24usize, 4usize, HD_256_THE_KVSHARE_TEMPLATE_SHAPE, 4usize);
    let total = 172_032usize + m;
    let mut st = 0x9b9b_0004u64;
    let kv_elems = total * nkv * hd;
    let hq: Vec<u16> = (0..m * nh * hd).map(|_| rnd_bf16(&mut st)).collect();
    let hk8: Vec<u8> = (0..kv_elems).map(|_| rnd_fp8(&mut st)).collect();
    let hv8: Vec<u8> = hk8.iter().rev().cloned().collect();
    let nsc = total * nkv;
    let hks: Vec<f32> = (0..nsc)
        .map(|_| 0.001f32 + ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 0.003)
        .collect();
    let hvs: Vec<f32> = hks.iter().rev().cloned().collect();

    let dq: CudaSlice<u16> = stream.clone_htod(&hq).unwrap();
    let dk8: CudaSlice<u8> = stream.clone_htod(&hk8).unwrap();
    let dv8: CudaSlice<u8> = stream.clone_htod(&hv8).unwrap();
    let dks: CudaSlice<f32> = stream.clone_htod(&hks).unwrap();
    let dvs: CudaSlice<f32> = stream.clone_htod(&hvs).unwrap();
    let dnc: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
    let scratch_elems = (nh * KVSHARE_MAX_SPLITS_128_THE_KERNELS_KKVSHAREMAXSPLITS * (hd + 2))
        .max(nh * ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M * (hd + 2));
    let mut dscr: CudaSlice<f32> = stream.alloc_zeros::<f32>(scratch_elems).unwrap();
    let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
    let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * nh * hd).unwrap();

    let (pq, _a) = dq.device_ptr(&stream);
    let (pk, _b) = dk8.device_ptr(&stream);
    let (pv, _c) = dv8.device_ptr(&stream);
    let (pks, _d) = dks.device_ptr(&stream);
    let (pvs, _e) = dvs.device_ptr(&stream);
    let (pn, _f) = dnc.device_ptr(&stream);
    let (ps, _g) = dscr.device_ptr_mut(&stream);
    let (pf, _h) = dfan.device_ptr_mut(&stream);
    let (po, _i) = dout.device_ptr_mut(&stream);
    let row_stride_bytes = (nh * hd * std::mem::size_of::<u16>()) as u64;

    let launch_kvshare = || {
        for qi in 0..m {
            let rc = unsafe {
                cuda::flash_decode_kvshare_fp8kv_paged(
                    stream.cu_stream() as *mut _,
                    (pq + qi as u64 * row_stride_bytes) as *const u16,
                    pk as *const u8,
                    pv as *const u8,
                    pks as *const f32,
                    pvs as *const f32,
                    (po + qi as u64 * row_stride_bytes) as *mut u16,
                    pn as *const i32,
                    ps as *mut f32,
                    pf as *mut u32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    0,
                    0,
                    0,
                    1.0,
                    std::ptr::null(),
                    0,
                    (m - 1 - qi) as i32,
                    0,
                    0,
                )
            };
            assert_eq!(rc, 0, "kvshare qi={qi} refused");
        }
    };
    let launch_rowshared = || {
        let rc = unsafe {
            cuda::flash_decode_rowshared_fp8kv_mk_paged(
                stream.cu_stream() as *mut _,
                pq as *const u16,
                pk as *const u8,
                pv as *const u8,
                pks as *const f32,
                pvs as *const f32,
                po as *mut u16,
                pn as *const i32,
                ps as *mut f32,
                pf as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
                0,
                0,
                (ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M / m) as i32,
                1.0,
                std::ptr::null(),
                0,
                0,
                m as i32,
                m as i32,
                0,
                0,
            )
        };
        assert_eq!(rc, 0, "rowshared refused");
    };
    let launch_tctile = || {
        let rc = unsafe {
            cuda::flash_decode_rowshared_tctile_fp8kv_mk_paged(
                stream.cu_stream() as *mut _,
                pq as *const u16,
                pk as *const u8,
                pv as *const u8,
                pks as *const f32,
                pvs as *const f32,
                po as *mut u16,
                pn as *const i32,
                ps as *mut f32,
                pf as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
                0,
                0,
                (ROWSHARED_SPLITS_CAP_IS_MK_SCRATCH_ROWS_8X32_OVER_M / m) as i32,
                1.0,
                std::ptr::null(),
                0,
                0,
                m as i32,
                m as i32,
                0,
                0,
            )
        };
        assert_eq!(rc, 0, "tctile refused");
    };
    let launch_mk = || {
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
                0,
                m as i32,
                ps as *mut f32,
                pf as *mut u32,
                nh as i32,
                nkv as i32,
                hd as i32,
                0,
                0,
                1.0,
            )
        };
        assert_eq!(rc, 0, "mk refused");
    };

    let time_ms = |f: &dyn Fn(), reps: usize| -> f64 {
        for _ in 0..3 {
            f();
        }
        stream.synchronize().unwrap();
        let mut samples: Vec<f64> = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = std::time::Instant::now();
            f();
            stream.synchronize().unwrap();
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        samples[samples.len() / 2]
    };

    let kv_bytes_once = (kv_elems * 2 + nsc * 8) as f64;
    let ms_kvshare = time_ms(&launch_kvshare, 20);
    let ms_rowshared = time_ms(&launch_rowshared, 20);
    let ms_tctile = time_ms(&launch_tctile, 20);
    let ms_mk = time_ms(&launch_mk, 20);
    eprintln!(
        "ROWSHARED-KBENCH total={total} m={m} kvshare_ms={ms_kvshare:.3} \
         rowshared_ms={ms_rowshared:.3} tctile_ms={ms_tctile:.3} mk_ms={ms_mk:.3} \
         kv_gb_once={:.3} kvshare_eff_gbs={:.0} rowshared_eff_gbs={:.0} tctile_eff_gbs={:.0}",
        kv_bytes_once / 1e9,
        kv_bytes_once * (m as f64) / (ms_kvshare / 1e3) / 1e9,
        kv_bytes_once / (ms_rowshared / 1e3) / 1e9,
        kv_bytes_once / (ms_tctile / 1e3) / 1e9,
    );
    assert!(
        ms_rowshared.is_finite() && ms_kvshare.is_finite() && ms_tctile.is_finite()
            && ms_mk.is_finite(),
        "non-finite timing"
    );
}

const DRAFTER_Q1_TOTALS_COVER_SUB_STRIDE_MULTI_ITER_AND_DEEP_VECTOR_LDG: [usize; 3] =
    [37, 300, 4099];

fn bf16_oracle_q1(hq: &[f32], hk: &[u16], hv: &[u16], nh: usize, nkv: usize, hd: usize, total: usize) -> Vec<f32> {
    let group = nh / nkv;
    let mut out = vec![0.0f32; nh * hd];
    for h in 0..nh {
        let kvh = h / group;
        let qrow = &hq[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0f64; total];
        let mut mx = f64::NEG_INFINITY;
        for (p, sc) in scores.iter_mut().enumerate() {
            let krow = &hk[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
            let mut s = 0.0f64;
            for d in 0..hd {
                s += qrow[d] as f64 * bf(krow[d]);
            }
            *sc = s;
            if *sc > mx {
                mx = *sc;
            }
        }
        let mut l = 0.0f64;
        for sc in scores.iter_mut() {
            *sc = (*sc - mx).exp();
            l += *sc;
        }
        let mut acc = vec![0.0f64; hd];
        for (p, sc) in scores.iter().enumerate() {
            let w = sc / l;
            let vrow = &hv[(p * nkv + kvh) * hd..(p * nkv + kvh) * hd + hd];
            for d in 0..hd {
                acc[d] += w * bf(vrow[d]);
            }
        }
        for d in 0..hd {
            out[h * hd + d] = acc[d] as f32;
        }
    }
    out
}

fn check_fused_bf16_q1_shape(stream: &Arc<CudaStream>, nh: usize, nkv: usize, hd: usize) {
    for total in DRAFTER_Q1_TOTALS_COVER_SUB_STRIDE_MULTI_ITER_AND_DEEP_VECTOR_LDG {
        let mut st = 0x51ab_0000 ^ ((nh as u64) << 32) ^ ((hd as u64) << 16) ^ total as u64;
        let hq: Vec<f32> = (0..nh * hd)
            .map(|_| ((lcg(&mut st) >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0)
            .collect();
        let hk: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
        let hv: Vec<u16> = (0..total * nkv * hd).map(|_| rnd_bf16(&mut st)).collect();
        let reference = bf16_oracle_q1(&hq, &hk, &hv, nh, nkv, hd, total);

        let dq: CudaSlice<f32> = stream.clone_htod(&hq).unwrap();
        let dk: CudaSlice<u16> = stream.clone_htod(&hk).unwrap();
        let dv: CudaSlice<u16> = stream.clone_htod(&hv).unwrap();
        let dpos: CudaSlice<i32> = stream.clone_htod(&[total as i32]).unwrap();
        let mut dscr: CudaSlice<f32> = stream
            .alloc_zeros::<f32>(cuda::flash_splitk_scratch_elems(nh as i32, hd as i32))
            .unwrap();
        let mut dfan: CudaSlice<u32> = stream.alloc_zeros::<u32>(nh).unwrap();
        let mut dout: CudaSlice<u16> = stream.alloc_zeros::<u16>(nh * hd).unwrap();
        let rc = {
            let (pq, _a) = dq.device_ptr(stream);
            let (pk, _b) = dk.device_ptr(stream);
            let (pv, _c) = dv.device_ptr(stream);
            let (pp, _d) = dpos.device_ptr(stream);
            let (ps, _e) = dscr.device_ptr_mut(stream);
            let (pf, _f) = dfan.device_ptr_mut(stream);
            let (po, _g) = dout.device_ptr_mut(stream);
            unsafe {
                cuda::flash_decode_fused_bf16kv(
                    stream.cu_stream() as *mut _,
                    pq as *const f32,
                    pk as *const u16,
                    pv as *const u16,
                    po as *mut u16,
                    pp as *const i32,
                    0,
                    ps as *mut f32,
                    pf as *mut u32,
                    nh as i32,
                    nkv as i32,
                    hd as i32,
                    0,
                )
            }
        };
        assert_eq!(rc, 0, "NH={nh} NKV={nkv} HD={hd} total={total}: fused bf16 launcher refused");
        stream.synchronize().unwrap();
        let out = stream.clone_dtoh(&dout).unwrap();
        let mut worst = 0f32;
        for (x, y) in out.iter().zip(reference.iter()) {
            let g = half::bf16::from_bits(*x).to_f32();
            assert!(
                g.is_finite(),
                "NH={nh} NKV={nkv} HD={hd} total={total}: non-finite fused bf16 output"
            );
            worst = worst.max((g - y).abs());
        }
        assert!(
            worst < ORACLE_BAND_0_02_MATCHES_THE_MK_CHECKS_ABOVE,
            "NH={nh} NKV={nkv} HD={hd} total={total}: fused bf16 splitk q1 vs f64 oracle max \
             diff {worst}; this is the NV_Q38_DRAFTER_ATTN_SPLITK drafter route"
        );
    }
}

#[test]
fn drafter_q1_shape_24q_4kv_hd256_fused_bf16_splitk_matches_the_f64_oracle() {
    let Some(stream) = stream_or_skip() else { return };
    check_fused_bf16_q1_shape(&stream, 24, 4, 256);
}
