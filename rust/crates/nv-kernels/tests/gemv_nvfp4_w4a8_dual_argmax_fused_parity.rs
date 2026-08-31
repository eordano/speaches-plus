#![cfg(feature = "cuda")]

mod common;
use common::stream;
use common::LcgMask23TwoSided as Lcg;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;

const FULL_VOCAB_151936_THE_Q38_SERVING_LMHEAD_ROWS: usize = 151936;
const FULL_VOCAB_TILE_ROWS_4096_KEEP_DEBUG_BUILD_WEIGHT_GEN_CHEAP_AND_SEED_MASS_TIES: usize = 4096;
const HIDDEN_5120_THE_Q38_SERVING_LMHEAD_K: usize = 5120;
const E2M1_NIBBLE_PLUS_SIX_ALL_LANES: u8 = 0x77;
const UE4M3_SCALE_BYTE_128X_LIFTS_THE_FORCED_TIE_ROWS_ABOVE_EVERY_RANDOM_ROW: u8 = 0x70;

struct Halves {
    packed_a: Vec<u8>,
    sc_lin_a: Vec<u8>,
    packed_b: Vec<u8>,
    sc_lin_b: Vec<u8>,
    n: usize,
    k: usize,
}

fn random_halves(rng: &mut Lcg, n: usize, k: usize) -> Halves {
    let kb = k / 16;
    Halves {
        packed_a: rng.packed_nibbles(n * k / 2),
        sc_lin_a: rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb),
        packed_b: rng.packed_nibbles(n * k / 2),
        sc_lin_b: rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb),
        n,
        k,
    }
}

fn tiled_halves(rng: &mut Lcg, n: usize, tile_rows: usize, k: usize) -> Halves {
    let kb = k / 16;
    let tile_packed = rng.packed_nibbles(tile_rows * k / 2);
    let tile_sc = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(tile_rows * kb);
    let mut h = Halves {
        packed_a: Vec::with_capacity(n * k / 2),
        sc_lin_a: Vec::with_capacity(n * kb),
        packed_b: Vec::with_capacity(n * k / 2),
        sc_lin_b: Vec::with_capacity(n * kb),
        n,
        k,
    };
    for r in 0..n {
        let t = r % tile_rows;
        h.packed_a
            .extend_from_slice(&tile_packed[t * k / 2..(t + 1) * k / 2]);
        h.sc_lin_a.extend_from_slice(&tile_sc[t * kb..(t + 1) * kb]);
        let t2 = (r + tile_rows / 2) % tile_rows;
        h.packed_b
            .extend_from_slice(&tile_packed[t2 * k / 2..(t2 + 1) * k / 2]);
        h.sc_lin_b
            .extend_from_slice(&tile_sc[t2 * kb..(t2 + 1) * kb]);
    }
    h
}

fn force_identical_max_row_pair(h: &mut Halves, logit_i: usize, logit_j: usize) {
    let (n, k) = (h.n, h.k);
    let kb = k / 16;
    let mut put = |half_b: bool, r: usize| {
        let (packed, sc) = if half_b {
            (&mut h.packed_b, &mut h.sc_lin_b)
        } else {
            (&mut h.packed_a, &mut h.sc_lin_a)
        };
        for b in &mut packed[r * k / 2..(r + 1) * k / 2] {
            *b = E2M1_NIBBLE_PLUS_SIX_ALL_LANES;
        }
        for s in &mut sc[r * kb..(r + 1) * kb] {
            *s = UE4M3_SCALE_BYTE_128X_LIFTS_THE_FORCED_TIE_ROWS_ABOVE_EVERY_RANDOM_ROW;
        }
    };
    put(logit_i >= n, logit_i % n);
    put(logit_j >= n, logit_j % n);
}

struct FusedOut {
    id: u32,
    logit: f32,
}

#[allow(clippy::too_many_arguments)]
fn run_both_and_host_oracle(
    stream: &Arc<CudaStream>,
    h: &Halves,
    x_q8: &[i8],
    x_scale: f32,
    alpha_a: f32,
    alpha_b: f32,
) -> (FusedOut, Vec<u16>, usize) {
    let (n, k) = (h.n, h.k);
    let kb = k / 16;
    let sc_sw_a = nv_quant::nvfp4::swizzle_scales(&h.sc_lin_a, n, kb);
    let sc_sw_b = nv_quant::nvfp4::swizzle_scales(&h.sc_lin_b, n, kb);
    #[allow(deprecated)]
    let dw_a: CudaSlice<u8> = stream.clone_htod(&h.packed_a).unwrap();
    #[allow(deprecated)]
    let dw_b: CudaSlice<u8> = stream.clone_htod(&h.packed_b).unwrap();
    #[allow(deprecated)]
    let ds_a: CudaSlice<u8> = stream.clone_htod(&sc_sw_a).unwrap();
    #[allow(deprecated)]
    let ds_b: CudaSlice<u8> = stream.clone_htod(&sc_sw_b).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<i8> = stream.clone_htod(x_q8).unwrap();
    #[allow(deprecated)]
    let dxs: CudaSlice<f32> = stream.clone_htod(&[x_scale]).unwrap();

    let mut dy_a: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let mut dy_b: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pwa, _a) = dw_a.device_ptr(stream);
        let (pwb, _b) = dw_b.device_ptr(stream);
        let (psa, _c) = ds_a.device_ptr(stream);
        let (psb, _d) = ds_b.device_ptr(stream);
        let (px, _e) = dx.device_ptr(stream);
        let (pxs, _f) = dxs.device_ptr(stream);
        let (pya, _g) = dy_a.device_ptr_mut(stream);
        let (pyb, _h2) = dy_b.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_dual_m1(
                stream.cu_stream() as *mut c_void,
                pwa as *const u8,
                psa as *const u8,
                pwb as *const u8,
                psb as *const u8,
                px as *const i8,
                pxs as *const f32,
                pya as *mut u16,
                pyb as *mut u16,
                alpha_a,
                alpha_b,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_dual_m1 rc={rc}");

    let mut dpacked: CudaSlice<u64> = stream.alloc_zeros::<u64>(1).unwrap();
    let mut did: CudaSlice<u32> = stream.alloc_zeros::<u32>(1).unwrap();
    let mut dlogit: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let rc = {
        let (pwa, _a) = dw_a.device_ptr(stream);
        let (pwb, _b) = dw_b.device_ptr(stream);
        let (psa, _c) = ds_a.device_ptr(stream);
        let (psb, _d) = ds_b.device_ptr(stream);
        let (px, _e) = dx.device_ptr(stream);
        let (pxs, _f) = dxs.device_ptr(stream);
        let (pp, _g) = dpacked.device_ptr_mut(stream);
        let (pi, _h2) = did.device_ptr_mut(stream);
        let (pl, _i) = dlogit.device_ptr_mut(stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_dual_argmax_m1(
                stream.cu_stream() as *mut c_void,
                pwa as *const u8,
                psa as *const u8,
                pwb as *const u8,
                psb as *const u8,
                px as *const i8,
                pxs as *const f32,
                pp as *mut u64,
                pi as *mut u32,
                pl as *mut f32,
                alpha_a,
                alpha_b,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_dual_argmax_m1 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let ya = stream.memcpy_dtov(&dy_a).unwrap();
    #[allow(deprecated)]
    let yb = stream.memcpy_dtov(&dy_b).unwrap();
    #[allow(deprecated)]
    let id = stream.memcpy_dtov(&did).unwrap()[0];
    #[allow(deprecated)]
    let logit = stream.memcpy_dtov(&dlogit).unwrap()[0];

    let mut logits: Vec<u16> = ya;
    logits.extend_from_slice(&yb);
    let host_arg = host_argmax_lowest_index_wins_ties(&logits);
    (FusedOut { id, logit }, logits, host_arg)
}

fn host_argmax_lowest_index_wins_ties(logits_bf16: &[u16]) -> usize {
    let mut best = 0usize;
    let mut best_v = bf16::from_bits(logits_bf16[0]).to_f32();
    for (i, w) in logits_bf16.iter().enumerate().skip(1) {
        let v = bf16::from_bits(*w).to_f32();
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

fn random_x_q8(rng: &mut Lcg, k: usize) -> Vec<i8> {
    (0..k)
        .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
        .collect()
}

fn positive_x_q8(rng: &mut Lcg, k: usize) -> Vec<i8> {
    (0..k).map(|_| (rng.next_u32() % 127 + 1) as i8).collect()
}

#[test]
fn fused_argmax_matches_the_unfused_kernel_plus_host_lowest_index_scan_across_shapes() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_dual_argmax_shapes") else {
        return;
    };
    let mut rng = Lcg(0x1b873593cc9e2d51);
    for (n, k) in [(7usize, 64usize), (8, 64), (25, 512), (256, 512), (129, 2048)] {
        for round in 0..3 {
            let h = random_halves(&mut rng, n, k);
            let x = random_x_q8(&mut rng, k);
            let xs = 0.001 + rng.next_f32().abs() * 0.02;
            let (fused, logits, host_arg) =
                run_both_and_host_oracle(&stream, &h, &x, xs, 0.0125, 0.05);
            assert_eq!(
                fused.id as usize, host_arg,
                "n={n} k={k} round={round}: fused id disagrees with the unfused kernel's \
                 host lowest-index argmax"
            );
            let host_max = bf16::from_bits(logits[host_arg]).to_f32();
            assert_eq!(
                fused.logit, host_max,
                "n={n} k={k} round={round}: fused winning logit must equal the bf16 max the \
                 unfused kernel materialized"
            );
        }
    }
}

#[test]
fn fused_argmax_full_vocab_151936_by_5120_serving_shape_with_tiled_mass_ties() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_dual_argmax_full_vocab") else {
        return;
    };
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let n = FULL_VOCAB_151936_THE_Q38_SERVING_LMHEAD_ROWS / 2;
    let k = HIDDEN_5120_THE_Q38_SERVING_LMHEAD_K;
    let h = tiled_halves(
        &mut rng,
        n,
        FULL_VOCAB_TILE_ROWS_4096_KEEP_DEBUG_BUILD_WEIGHT_GEN_CHEAP_AND_SEED_MASS_TIES,
        k,
    );
    let x = random_x_q8(&mut rng, k);
    let (fused, logits, host_arg) =
        run_both_and_host_oracle(&stream, &h, &x, 0.0043, 0.0125, 0.0125);
    let dup_count = logits
        .iter()
        .filter(|w| **w == logits[host_arg])
        .count();
    assert!(
        dup_count >= 2,
        "the tiled full-vocab case must actually exercise duplicate maxima, got {dup_count}"
    );
    assert_eq!(
        fused.id as usize, host_arg,
        "full-vocab tiled ties: fused id must be the lowest-index maximum"
    );
}

#[test]
fn fused_argmax_adversarial_ties_zero_rows_neg_zero_and_forced_duplicate_maxima() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_dual_argmax_ties") else {
        return;
    };
    let mut rng = Lcg(0x243f6a8885a308d3);
    let (n, k) = (96usize, 512usize);

    let zero_halves = Halves {
        packed_a: vec![0u8; n * k / 2],
        sc_lin_a: vec![0u8; n * k / 16],
        packed_b: vec![0u8; n * k / 2],
        sc_lin_b: vec![0u8; n * k / 16],
        n,
        k,
    };
    let x = positive_x_q8(&mut rng, k);
    let (fused, logits, host_arg) =
        run_both_and_host_oracle(&stream, &zero_halves, &x, 0.01, 1.0, 1.0);
    assert_eq!(host_arg, 0, "all-zero logits: host oracle must pick index 0");
    assert_eq!(
        fused.id, 0,
        "all-zero logits: every index ties at +0.0 and the lowest index must win"
    );

    let (fused, logits_nz, _) =
        run_both_and_host_oracle(&stream, &zero_halves, &x, 0.01, -1.0, 1.0);
    assert_eq!(
        logits_nz[0],
        bf16::from_f32(-0.0).to_bits(),
        "negative alpha over a zero row must materialize -0.0 in the first half"
    );
    assert_eq!(
        logits_nz[n],
        bf16::from_f32(0.0).to_bits(),
        "positive alpha over a zero row must materialize +0.0 in the second half"
    );
    assert_eq!(
        fused.id, 0,
        "-0.0 ties +0.0 exactly as the host float compare does, so index 0 must still win \
         over the +0.0 half"
    );
    assert_eq!(
        fused.logit, 0.0,
        "the winning -0.0 unpacks value-equal to 0.0"
    );
    drop(logits);

    for (lo, hi, label) in [
        (0usize, 2 * n - 1, "first and last logit"),
        (5, 6, "adjacent rows in the top half"),
        (n - 1, n, "the half seam"),
    ] {
        let mut h = random_halves(&mut rng, n, k);
        force_identical_max_row_pair(&mut h, lo, hi);
        let x = positive_x_q8(&mut rng, k);
        let (fused, logits, host_arg) =
            run_both_and_host_oracle(&stream, &h, &x, 0.005, 0.02, 0.02);
        assert_eq!(
            logits[lo], logits[hi],
            "{label}: the forced duplicate rows must produce bit-identical logits"
        );
        assert_eq!(
            host_arg, lo,
            "{label}: the forced pair must be the maximum and the host must take the lower index"
        );
        assert_eq!(
            fused.id as usize, lo,
            "{label}: fused argmax must break the duplicate-maximum tie to the lowest index"
        );
    }
}

#[test]
fn fused_argmax_refuses_ragged_k_without_launching() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_dual_argmax_refusal") else {
        return;
    };
    let mut dpacked: CudaSlice<u64> = stream.alloc_zeros::<u64>(1).unwrap();
    let mut did: CudaSlice<u32> = stream.alloc_zeros::<u32>(1).unwrap();
    let rc = {
        let (pp, _a) = dpacked.device_ptr_mut(&stream);
        let (pi, _b) = did.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_dual_argmax_m1(
                stream.cu_stream() as *mut c_void,
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                std::ptr::null(),
                pp as *mut u64,
                pi as *mut u32,
                std::ptr::null_mut(),
                1.0,
                1.0,
                16,
                24,
            )
        }
    };
    assert_eq!(rc, -1, "ragged K must refuse, got {rc}");
}
