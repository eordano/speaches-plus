#![cfg(feature = "cuda")]

mod common;
use common::stream;
use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use half::bf16;
use nv_kernels::cuda;
use std::ffi::c_void;
use std::sync::Arc;
use std::time::Instant;
use common::LcgMask23TwoSided as Lcg;
use common::assert_rows_close;
use common::host_w_f64;

const MECHANICS_TOL_8E3_IS_TWO_BF16_OUTPUT_ULPS_OVER_EXACT_INT_DOTS: f64 = 8.0e-3;
const HONEST_TOL_1E1_THE_PER_TENSOR_Q8_NOISE_SCALES_WITH_ALPHA_AGAINST_THE_QUARTER_DENOM_FLOOR_SO_ARGMAX_AND_PPL_CARRY_THE_QUALITY_GATE:
    f64 = 1.0e-1;

fn host_rowquant_i8_rn_matching_the_device_float2int_rn(x: &[f32]) -> (Vec<i8>, f64) {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let scale = amax / 127.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let q = x
        .iter()
        .map(|v| ((v * inv).round_ties_even().clamp(-127.0, 127.0)) as i8)
        .collect();
    (q, scale as f64)
}

fn host_dot_rows_f64(wf: &[f64], x: &[f64], n: usize, k: usize) -> Vec<f64> {
    (0..n)
        .map(|r| (0..k).map(|c| wf[r * k + c] * x[c]).sum())
        .collect()
}

fn assert_argmax_agrees(name: &str, got: &[u16], reference: &[f64]) {
    let got_arg = got
        .iter()
        .enumerate()
        .max_by(|a, b| {
            bf16::from_bits(*a.1)
                .to_f32()
                .partial_cmp(&bf16::from_bits(*b.1).to_f32())
                .unwrap()
        })
        .unwrap()
        .0;
    let ref_arg = reference
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    assert_eq!(
        got_arg, ref_arg,
        "{name}: argmax disagrees with the f64 bf16-activation reference"
    );
}

struct DualOut {
    ya: Vec<u16>,
    yb: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
fn run_dual(
    stream: &Arc<CudaStream>,
    packed_a: &[u8],
    sc_sw_a: &[u8],
    packed_b: &[u8],
    sc_sw_b: &[u8],
    x_q8: &[i8],
    x_scale: f32,
    alpha_a: f32,
    alpha_b: f32,
    n: usize,
    k: usize,
) -> DualOut {
    #[allow(deprecated)]
    let dw_a: CudaSlice<u8> = stream.clone_htod(packed_a).unwrap();
    #[allow(deprecated)]
    let dw_b: CudaSlice<u8> = stream.clone_htod(packed_b).unwrap();
    #[allow(deprecated)]
    let ds_a: CudaSlice<u8> = stream.clone_htod(sc_sw_a).unwrap();
    #[allow(deprecated)]
    let ds_b: CudaSlice<u8> = stream.clone_htod(sc_sw_b).unwrap();
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
        let (pyb, _h) = dy_b.device_ptr_mut(stream);
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
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let ya = stream.memcpy_dtov(&dy_a).unwrap();
    #[allow(deprecated)]
    let yb = stream.memcpy_dtov(&dy_b).unwrap();
    DualOut { ya, yb }
}

#[test]
fn gemv_nvfp4_w4a8_dual_matches_the_int8_mechanics_oracle_and_the_bf16_activation_oracle() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_dual") else {
        return;
    };
    let mut rng = Lcg(0x243f6a8885a308d3);
    let (n, k) = (256usize, 512usize);
    let kb = k / 16;
    let packed_a = rng.packed_nibbles(n * k / 2);
    let packed_b = rng.packed_nibbles(n * k / 2);
    let sc_lin_a = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb);
    let sc_lin_b = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb);
    let sc_sw_a = nv_quant::nvfp4::swizzle_scales(&sc_lin_a, n, kb);
    let sc_sw_b = nv_quant::nvfp4::swizzle_scales(&sc_lin_b, n, kb);
    let x_words = rng.bf16_words(k, 1.0);
    let xf32: Vec<f32> = x_words
        .iter()
        .map(|w| bf16::from_bits(*w).to_f32())
        .collect();
    let (x_q8, x_scale) = host_rowquant_i8_rn_matching_the_device_float2int_rn(&xf32);
    let (alpha_a, alpha_b) = (0.0125f32, 0.05f32);

    let out = run_dual(
        &stream, &packed_a, &sc_sw_a, &packed_b, &sc_sw_b, &x_q8, x_scale as f32, alpha_a,
        alpha_b, n, k,
    );

    let wa = host_w_f64(&packed_a, &sc_lin_a, n, k, alpha_a);
    let wb = host_w_f64(&packed_b, &sc_lin_b, n, k, alpha_b);
    let x_deq: Vec<f64> = x_q8.iter().map(|q| *q as f64 * x_scale).collect();
    let mech_a = host_dot_rows_f64(&wa, &x_deq, n, k);
    let mech_b = host_dot_rows_f64(&wb, &x_deq, n, k);
    assert_rows_close(
        "dual gate arm int8-mechanics",
        &out.ya,
        &mech_a,
        MECHANICS_TOL_8E3_IS_TWO_BF16_OUTPUT_ULPS_OVER_EXACT_INT_DOTS,
    );
    assert_rows_close(
        "dual up arm int8-mechanics",
        &out.yb,
        &mech_b,
        MECHANICS_TOL_8E3_IS_TWO_BF16_OUTPUT_ULPS_OVER_EXACT_INT_DOTS,
    );

    let xf: Vec<f64> = xf32.iter().map(|v| *v as f64).collect();
    let honest_a = host_dot_rows_f64(&wa, &xf, n, k);
    let honest_b = host_dot_rows_f64(&wb, &xf, n, k);
    assert_rows_close(
        "dual gate arm bf16-activation",
        &out.ya,
        &honest_a,
        HONEST_TOL_1E1_THE_PER_TENSOR_Q8_NOISE_SCALES_WITH_ALPHA_AGAINST_THE_QUARTER_DENOM_FLOOR_SO_ARGMAX_AND_PPL_CARRY_THE_QUALITY_GATE,
    );
    assert_rows_close(
        "dual up arm bf16-activation",
        &out.yb,
        &honest_b,
        HONEST_TOL_1E1_THE_PER_TENSOR_Q8_NOISE_SCALES_WITH_ALPHA_AGAINST_THE_QUARTER_DENOM_FLOOR_SO_ARGMAX_AND_PPL_CARRY_THE_QUALITY_GATE,
    );
    assert_argmax_agrees("dual gate arm", &out.ya, &honest_a);
    assert_argmax_agrees("dual up arm", &out.yb, &honest_b);
}

#[test]
fn silu_mul_rowquant_then_down_with_residual_match_the_staged_bf16_act_chain_oracle() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_down_chain") else {
        return;
    };
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let (n, k) = (128usize, 1024usize);
    let kb = k / 16;
    let packed = rng.packed_nibbles(n * k / 2);
    let sc_lin = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb);
    let sc_sw = nv_quant::nvfp4::swizzle_scales(&sc_lin, n, kb);
    let gate_words = rng.bf16_words(k, 1.0);
    let up_words = rng.bf16_words(k, 1.0);
    let residual_words = rng.bf16_words(n, 1.0);
    let alpha = 0.02f32;

    #[allow(deprecated)]
    let dg: CudaSlice<u16> = stream.clone_htod(&gate_words).unwrap();
    #[allow(deprecated)]
    let du: CudaSlice<u16> = stream.clone_htod(&up_words).unwrap();
    let mut dact: CudaSlice<i8> = stream.alloc_zeros::<i8>(k).unwrap();
    let mut dact_s: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let rc = {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pq, _c) = dact.device_ptr_mut(&stream);
        let (ps, _d) = dact_s.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_mul_rowquant_i8_m1(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pq as *mut i8,
                ps as *mut f32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "silu_mul_rowquant_i8_m1 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let act_q8 = stream.memcpy_dtov(&dact).unwrap();
    #[allow(deprecated)]
    let act_scale = stream.memcpy_dtov(&dact_s).unwrap()[0];

    let act_bf16: Vec<f32> = gate_words
        .iter()
        .zip(up_words.iter())
        .map(|(g, u)| {
            let gf = bf16::from_bits(*g).to_f32();
            let uf = bf16::from_bits(*u).to_f32();
            bf16::from_f32((gf / (1.0 + (-gf).exp())) * uf).to_f32()
        })
        .collect();
    let (host_q8, host_scale) = host_rowquant_i8_rn_matching_the_device_float2int_rn(&act_bf16);
    let scale_rel = ((act_scale as f64) - host_scale).abs() / host_scale.abs().max(1e-12);
    assert!(
        scale_rel < 1.0e-6,
        "act scale device={act_scale} host={host_scale} rel={scale_rel:.3e}"
    );
    let mismatched = act_q8
        .iter()
        .zip(host_q8.iter())
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1)
        .count();
    assert_eq!(
        mismatched, 0,
        "act_q8 differs from the host bf16-staged rowquant by more than one ulp of q8"
    );

    #[allow(deprecated)]
    let dw: CudaSlice<u8> = stream.clone_htod(&packed).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<u8> = stream.clone_htod(&sc_sw).unwrap();
    #[allow(deprecated)]
    let dres: CudaSlice<u16> = stream.clone_htod(&residual_words).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pw, _a) = dw.device_ptr(&stream);
        let (ps, _b) = ds.device_ptr(&stream);
        let (pq, _c) = dact.device_ptr(&stream);
        let (pqs, _d) = dact_s.device_ptr(&stream);
        let (pr, _e) = dres.device_ptr(&stream);
        let (py, _f) = dy.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_down_residual_m1(
                stream.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const u8,
                pq as *const i8,
                pqs as *const f32,
                pr as *const u16,
                py as *mut u16,
                alpha,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_m1 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let y = stream.memcpy_dtov(&dy).unwrap();

    let w = host_w_f64(&packed, &sc_lin, n, k, alpha);
    let act_deq: Vec<f64> = act_q8.iter().map(|q| *q as f64 * act_scale as f64).collect();
    let mech: Vec<f64> = host_dot_rows_f64(&w, &act_deq, n, k)
        .into_iter()
        .zip(residual_words.iter())
        .map(|(d, r)| d + bf16::from_bits(*r).to_f32() as f64)
        .collect();
    assert_rows_close(
        "down residual int8-mechanics",
        &y,
        &mech,
        MECHANICS_TOL_8E3_IS_TWO_BF16_OUTPUT_ULPS_OVER_EXACT_INT_DOTS,
    );

    let act_f64: Vec<f64> = act_bf16.iter().map(|v| *v as f64).collect();
    let honest: Vec<f64> = host_dot_rows_f64(&w, &act_f64, n, k)
        .into_iter()
        .zip(residual_words.iter())
        .map(|(d, r)| d + bf16::from_bits(*r).to_f32() as f64)
        .collect();
    assert_rows_close(
        "down residual bf16-activation",
        &y,
        &honest,
        HONEST_TOL_1E1_THE_PER_TENSOR_Q8_NOISE_SCALES_WITH_ALPHA_AGAINST_THE_QUARTER_DENOM_FLOOR_SO_ARGMAX_AND_PPL_CARRY_THE_QUALITY_GATE,
    );
    assert_argmax_agrees("down residual", &y, &honest);

    let rstd_eps = 1.0e-6f32;
    let mut dy_emit: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let mut dpack: CudaSlice<f32> = stream.alloc_zeros::<f32>(4).unwrap();
    for round_proving_the_pack_self_resets in 0..2 {
        let rc = {
            let (pw, _a) = dw.device_ptr(&stream);
            let (ps, _b) = ds.device_ptr(&stream);
            let (pq, _c) = dact.device_ptr(&stream);
            let (pqs, _d) = dact_s.device_ptr(&stream);
            let (pr, _e) = dres.device_ptr(&stream);
            let (py, _f) = dy_emit.device_ptr_mut(&stream);
            let (pp, _g) = dpack.device_ptr_mut(&stream);
            unsafe {
                cuda::gemv_nvfp4_w4a8_down_residual_m1_rstd_emit(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u8,
                    ps as *const u8,
                    pq as *const i8,
                    pqs as *const f32,
                    pr as *const u16,
                    py as *mut u16,
                    alpha,
                    pp as *mut f32,
                    rstd_eps,
                    n as i32,
                    k as i32,
                )
            }
        };
        assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_m1_rstd_emit rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let y_emit = stream.memcpy_dtov(&dy_emit).unwrap();
        assert_eq!(
            y, y_emit,
            "rstd-emitting down kernel changed the summed output bits \
             (round {round_proving_the_pack_self_resets})"
        );
        #[allow(deprecated)]
        let pack = stream.memcpy_dtov(&dpack).unwrap();
        let host_ssq: f64 = y
            .iter()
            .map(|b| {
                let v = bf16::from_bits(*b).to_f32() as f64;
                v * v
            })
            .sum();
        let host_rstd = 1.0 / (host_ssq / n as f64 + rstd_eps as f64).sqrt();
        let rel = ((pack[0] as f64) - host_rstd).abs() / host_rstd.abs().max(1e-12);
        assert!(
            rel < 1.0e-5,
            "emitted rstd {} vs host {host_rstd} rel={rel:.3e} \
             (round {round_proving_the_pack_self_resets})",
            pack[0]
        );
        assert_eq!(
            pack[1], 0.0,
            "ssq accumulator must be reset by the last block for the next replay"
        );
        assert_eq!(
            pack[2].to_bits(),
            0,
            "block counter must wrap to zero for the next replay"
        );
    }
}

#[test]
fn silu_mul_rowquant_mk_split_matches_the_single_block_m1_kernel_and_the_host_oracle() {
    let Some(stream) = stream("silu_mul_rowquant_i8_mk") else {
        return;
    };
    let mut rng = Lcg(0x0123456789abcdef);
    let (m, k) = (3usize, 2048usize);
    let gate_words = rng.bf16_words(m * k, 1.0);
    let up_words = rng.bf16_words(m * k, 1.0);
    #[allow(deprecated)]
    let dg: CudaSlice<u16> = stream.clone_htod(&gate_words).unwrap();
    #[allow(deprecated)]
    let du: CudaSlice<u16> = stream.clone_htod(&up_words).unwrap();
    let plen = cuda::silu_mul_rowquant_i8_mk_partials_len(m as i32, k as i32);
    assert!(plen > 0, "partials_len refused m={m} k={k}: {plen}");
    let mut dstage: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * k).unwrap();
    let mut dpart: CudaSlice<f32> = stream.alloc_zeros::<f32>(plen as usize).unwrap();
    let mut dq_mk: CudaSlice<i8> = stream.alloc_zeros::<i8>(m * k).unwrap();
    let mut ds_mk: CudaSlice<f32> = stream.alloc_zeros::<f32>(m).unwrap();
    let rc = {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pst, _c) = dstage.device_ptr_mut(&stream);
        let (pp, _d) = dpart.device_ptr_mut(&stream);
        let (pq, _e) = dq_mk.device_ptr_mut(&stream);
        let (ps, _f) = ds_mk.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_mul_rowquant_i8_mk(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pst as *mut u16,
                pp as *mut f32,
                pq as *mut i8,
                ps as *mut f32,
                m as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "silu_mul_rowquant_i8_mk rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let q_mk = stream.memcpy_dtov(&dq_mk).unwrap();
    #[allow(deprecated)]
    let s_mk = stream.memcpy_dtov(&ds_mk).unwrap();

    for row in 0..m {
        let mut dq_m1: CudaSlice<i8> = stream.alloc_zeros::<i8>(k).unwrap();
        let mut ds_m1: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
        let rc = {
            let (pg, _a) = dg.device_ptr(&stream);
            let (pu, _b) = du.device_ptr(&stream);
            let (pq, _c) = dq_m1.device_ptr_mut(&stream);
            let (ps, _d) = ds_m1.device_ptr_mut(&stream);
            unsafe {
                cuda::silu_mul_rowquant_i8_m1(
                    stream.cu_stream() as *mut c_void,
                    (pg as usize + row * k * 2) as *const u16,
                    (pu as usize + row * k * 2) as *const u16,
                    pq as *mut i8,
                    ps as *mut f32,
                    k as i32,
                )
            }
        };
        assert_eq!(rc, 0, "silu_mul_rowquant_i8_m1 row={row} rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let q_m1 = stream.memcpy_dtov(&dq_m1).unwrap();
        #[allow(deprecated)]
        let s_m1 = stream.memcpy_dtov(&ds_m1).unwrap()[0];
        assert_eq!(
            s_mk[row].to_bits(),
            s_m1.to_bits(),
            "row {row}: mk scale must equal the m1 scale bitwise, fmax reduction order cannot matter"
        );
        assert_eq!(
            &q_mk[row * k..(row + 1) * k],
            &q_m1[..],
            "row {row}: mk act_q8 must equal the m1 act_q8 exactly given identical scales"
        );
        let act_bf16: Vec<f32> = gate_words[row * k..(row + 1) * k]
            .iter()
            .zip(up_words[row * k..(row + 1) * k].iter())
            .map(|(g, u)| {
                let gf = bf16::from_bits(*g).to_f32();
                let uf = bf16::from_bits(*u).to_f32();
                bf16::from_f32((gf / (1.0 + (-gf).exp())) * uf).to_f32()
            })
            .collect();
        let (host_q8, host_scale) =
            host_rowquant_i8_rn_matching_the_device_float2int_rn(&act_bf16);
        let scale_rel = ((s_mk[row] as f64) - host_scale).abs() / host_scale.abs().max(1e-12);
        assert!(
            scale_rel < 1.0e-6,
            "row {row}: scale device={} host={host_scale} rel={scale_rel:.3e}",
            s_mk[row]
        );
        let mismatched = q_mk[row * k..(row + 1) * k]
            .iter()
            .zip(host_q8.iter())
            .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1)
            .count();
        assert_eq!(mismatched, 0, "row {row}: act_q8 differs from host by more than one ulp of q8");
    }
}

#[test]
fn rmsnorm_residual_writeout_rowquant_fused_matches_the_two_kernel_chain() {
    let Some(stream) = stream("rmsnorm_residual_writeout_rowquant_i8_m1") else {
        return;
    };
    let mut rng = Lcg(0xfeedfacecafebeef);
    let hidden = 5120usize;
    let eps = 1.0e-6f32;
    let x_words = rng.bf16_words(hidden, 1.0);
    let r_words = rng.bf16_words(hidden, 1.0);
    let w_words = rng.bf16_words(hidden, 1.0);
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x_words).unwrap();
    #[allow(deprecated)]
    let dr: CudaSlice<u16> = stream.clone_htod(&r_words).unwrap();
    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(&w_words).unwrap();

    let mut dres_ref: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();
    let mut dnormed_ref: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();
    let mut dq_ref: CudaSlice<i8> = stream.alloc_zeros::<i8>(hidden).unwrap();
    let mut ds_ref: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let rc = {
        let (px, _a) = dx.device_ptr(&stream);
        let (pr, _b) = dr.device_ptr(&stream);
        let (pw, _c) = dw.device_ptr(&stream);
        let (pro, _d) = dres_ref.device_ptr_mut(&stream);
        let (pno, _e) = dnormed_ref.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_residual_writeout_bf16(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                pro as *mut u16,
                pno as *mut u16,
                1,
                hidden,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "rmsnorm_residual_writeout_bf16 rc={rc}");
    let rc = {
        let (pno, _a) = dnormed_ref.device_ptr(&stream);
        let (pq, _b) = dq_ref.device_ptr_mut(&stream);
        let (ps, _c) = ds_ref.device_ptr_mut(&stream);
        unsafe {
            cuda::rowquant_i8(
                stream.cu_stream() as *mut c_void,
                pno as *const u16,
                pq as *mut i8,
                ps as *mut f32,
                1,
                hidden as i32,
            )
        }
    };
    assert_eq!(rc, 0, "rowquant_i8 rc={rc}");

    let mut dres_fused: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();
    let mut dq_fused: CudaSlice<i8> = stream.alloc_zeros::<i8>(hidden).unwrap();
    let mut ds_fused: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let rc = {
        let (px, _a) = dx.device_ptr(&stream);
        let (pr, _b) = dr.device_ptr(&stream);
        let (pw, _c) = dw.device_ptr(&stream);
        let (pro, _d) = dres_fused.device_ptr_mut(&stream);
        let (pq, _e) = dq_fused.device_ptr_mut(&stream);
        let (ps, _f) = ds_fused.device_ptr_mut(&stream);
        unsafe {
            cuda::rmsnorm_residual_writeout_rowquant_i8_m1(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                pro as *mut u16,
                pq as *mut i8,
                ps as *mut f32,
                hidden as i32,
                eps,
            )
        }
    };
    assert_eq!(rc, 0, "rmsnorm_residual_writeout_rowquant_i8_m1 rc={rc}");
    stream.synchronize().unwrap();

    #[allow(deprecated)]
    let res_ref = stream.memcpy_dtov(&dres_ref).unwrap();
    #[allow(deprecated)]
    let res_fused = stream.memcpy_dtov(&dres_fused).unwrap();
    assert_eq!(
        res_ref, res_fused,
        "res_out must be bit-identical, x+r rounds per element with no reduction"
    );
    #[allow(deprecated)]
    let s_ref = stream.memcpy_dtov(&ds_ref).unwrap()[0] as f64;
    #[allow(deprecated)]
    let s_fused = stream.memcpy_dtov(&ds_fused).unwrap()[0] as f64;
    let scale_rel = (s_fused - s_ref).abs() / s_ref.abs().max(1e-12);
    assert!(
        scale_rel < 1.0e-5,
        "fused scale {s_fused} vs chain scale {s_ref} rel={scale_rel:.3e}, only sumsq reduction order may differ"
    );
    #[allow(deprecated)]
    let q_ref = stream.memcpy_dtov(&dq_ref).unwrap();
    #[allow(deprecated)]
    let q_fused = stream.memcpy_dtov(&dq_fused).unwrap();
    let mismatched = q_ref
        .iter()
        .zip(q_fused.iter())
        .filter(|(a, b)| (**a as i32 - **b as i32).abs() > 1)
        .count();
    assert_eq!(
        mismatched, 0,
        "fused q8 differs from the two-kernel chain by more than one ulp of q8"
    );
}

#[test]
fn gemv_nvfp4_w4a8_mk_matches_per_row_m1_calls_and_chunks_across_the_smem_token_bound() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_mk") else {
        return;
    };
    let mut rng = Lcg(0x5bd1e9955bd1e995);
    let (n, k, m) = (128usize, 512usize, 4usize);
    let kb = k / 16;
    let packed_a = rng.packed_nibbles(n * k / 2);
    let packed_b = rng.packed_nibbles(n * k / 2);
    let sc_sw_a = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb),
        n,
        kb,
    );
    let sc_sw_b = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb),
        n,
        kb,
    );
    let x_q8: Vec<i8> = (0..m * k)
        .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
        .collect();
    let x_scales: Vec<f32> = (0..m).map(|_| 0.001 + rng.next_f32().abs() * 0.01).collect();
    let (alpha_a, alpha_b) = (0.0125f32, 0.05f32);

    #[allow(deprecated)]
    let dwa: CudaSlice<u8> = stream.clone_htod(&packed_a).unwrap();
    #[allow(deprecated)]
    let dwb: CudaSlice<u8> = stream.clone_htod(&packed_b).unwrap();
    #[allow(deprecated)]
    let dsa: CudaSlice<u8> = stream.clone_htod(&sc_sw_a).unwrap();
    #[allow(deprecated)]
    let dsb: CudaSlice<u8> = stream.clone_htod(&sc_sw_b).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<i8> = stream.clone_htod(&x_q8).unwrap();
    #[allow(deprecated)]
    let dxs: CudaSlice<f32> = stream.clone_htod(&x_scales).unwrap();
    let mut dya: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let mut dyb: CudaSlice<u16> = stream.alloc_zeros::<u16>(m * n).unwrap();
    let rc = {
        let (pwa, _a) = dwa.device_ptr(&stream);
        let (pwb, _b) = dwb.device_ptr(&stream);
        let (psa, _c) = dsa.device_ptr(&stream);
        let (psb, _d) = dsb.device_ptr(&stream);
        let (px, _e) = dx.device_ptr(&stream);
        let (pxs, _f) = dxs.device_ptr(&stream);
        let (pya, _g) = dya.device_ptr_mut(&stream);
        let (pyb, _h) = dyb.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_dual_mk(
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
                m as i32,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_dual_mk rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let ya_mk = stream.memcpy_dtov(&dya).unwrap();
    #[allow(deprecated)]
    let yb_mk = stream.memcpy_dtov(&dyb).unwrap();

    for j in 0..m {
        let out = run_dual(
            &stream,
            &packed_a,
            &sc_sw_a,
            &packed_b,
            &sc_sw_b,
            &x_q8[j * k..(j + 1) * k],
            x_scales[j],
            alpha_a,
            alpha_b,
            n,
            k,
        );
        assert_eq!(
            &ya_mk[j * n..(j + 1) * n],
            &out.ya[..],
            "token {j}: dual mk gate arm must match the m1 kernel bitwise, same dot order"
        );
        assert_eq!(
            &yb_mk[j * n..(j + 1) * n],
            &out.yb[..],
            "token {j}: dual mk up arm must match the m1 kernel bitwise, same dot order"
        );
    }

    let (dn, dk, dm) = (128usize, 16384usize, 8usize);
    let chunk = 96 * 1024 / dk;
    assert!(
        dm > chunk,
        "down mk case must exercise the smem token-bound chunking: m={dm} chunk={chunk}"
    );
    let dkb = dk / 16;
    let packed_d = rng.packed_nibbles(dn * dk / 2);
    let sc_sw_d = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(dn * dkb),
        dn,
        dkb,
    );
    let dx_q8: Vec<i8> = (0..dm * dk)
        .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
        .collect();
    let dx_scales: Vec<f32> = (0..dm).map(|_| 0.001 + rng.next_f32().abs() * 0.01).collect();
    let res_words = rng.bf16_words(dm * dn, 1.0);
    let alpha_d = 0.02f32;
    #[allow(deprecated)]
    let dwd: CudaSlice<u8> = stream.clone_htod(&packed_d).unwrap();
    #[allow(deprecated)]
    let dsd: CudaSlice<u8> = stream.clone_htod(&sc_sw_d).unwrap();
    #[allow(deprecated)]
    let ddx: CudaSlice<i8> = stream.clone_htod(&dx_q8).unwrap();
    #[allow(deprecated)]
    let ddxs: CudaSlice<f32> = stream.clone_htod(&dx_scales).unwrap();
    #[allow(deprecated)]
    let ddres: CudaSlice<u16> = stream.clone_htod(&res_words).unwrap();
    let mut ddy: CudaSlice<u16> = stream.alloc_zeros::<u16>(dm * dn).unwrap();
    let rc = {
        let (pw, _a) = dwd.device_ptr(&stream);
        let (ps, _b) = dsd.device_ptr(&stream);
        let (px, _c) = ddx.device_ptr(&stream);
        let (pxs, _d) = ddxs.device_ptr(&stream);
        let (pr, _e) = ddres.device_ptr(&stream);
        let (py, _f) = ddy.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_down_residual_mk(
                stream.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const u8,
                px as *const i8,
                pxs as *const f32,
                pr as *const u16,
                py as *mut u16,
                alpha_d,
                dm as i32,
                dn as i32,
                dk as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_mk rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let y_mk = stream.memcpy_dtov(&ddy).unwrap();

    for j in 0..dm {
        #[allow(deprecated)]
        let djx: CudaSlice<i8> = stream.clone_htod(&dx_q8[j * dk..(j + 1) * dk]).unwrap();
        #[allow(deprecated)]
        let djxs: CudaSlice<f32> = stream.clone_htod(&dx_scales[j..j + 1]).unwrap();
        #[allow(deprecated)]
        let djres: CudaSlice<u16> = stream.clone_htod(&res_words[j * dn..(j + 1) * dn]).unwrap();
        let mut djy: CudaSlice<u16> = stream.alloc_zeros::<u16>(dn).unwrap();
        let rc = {
            let (pw, _a) = dwd.device_ptr(&stream);
            let (ps, _b) = dsd.device_ptr(&stream);
            let (px, _c) = djx.device_ptr(&stream);
            let (pxs, _d) = djxs.device_ptr(&stream);
            let (pr, _e) = djres.device_ptr(&stream);
            let (py, _f) = djy.device_ptr_mut(&stream);
            unsafe {
                cuda::gemv_nvfp4_w4a8_down_residual_m1(
                    stream.cu_stream() as *mut c_void,
                    pw as *const u8,
                    ps as *const u8,
                    px as *const i8,
                    pxs as *const f32,
                    pr as *const u16,
                    py as *mut u16,
                    alpha_d,
                    dn as i32,
                    dk as i32,
                )
            }
        };
        assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_m1 token={j} rc={rc}");
        stream.synchronize().unwrap();
        #[allow(deprecated)]
        let y_m1 = stream.memcpy_dtov(&djy).unwrap();
        assert_eq!(
            &y_mk[j * dn..(j + 1) * dn],
            &y_m1[..],
            "token {j}: down mk must match the m1 kernel bitwise across the chunk boundary"
        );
    }
}

#[test]
fn down_quant_prologue_matches_the_silu_m1_then_down_m1_chain_bitwise() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_down_quant_prologue") else {
        return;
    };
    let mut rng = Lcg(0xc0ffee1234567890);
    let (n, k) = (128usize, 2048usize);
    let kb = k / 16;
    let packed = rng.packed_nibbles(n * k / 2);
    let sc_sw = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(n * kb),
        n,
        kb,
    );
    let gate_words = rng.bf16_words(k, 1.0);
    let up_words = rng.bf16_words(k, 1.0);
    let residual_words = rng.bf16_words(n, 1.0);
    let alpha = 0.02f32;
    #[allow(deprecated)]
    let dw: CudaSlice<u8> = stream.clone_htod(&packed).unwrap();
    #[allow(deprecated)]
    let ds: CudaSlice<u8> = stream.clone_htod(&sc_sw).unwrap();
    #[allow(deprecated)]
    let dg: CudaSlice<u16> = stream.clone_htod(&gate_words).unwrap();
    #[allow(deprecated)]
    let du: CudaSlice<u16> = stream.clone_htod(&up_words).unwrap();
    #[allow(deprecated)]
    let dres: CudaSlice<u16> = stream.clone_htod(&residual_words).unwrap();

    let mut dact: CudaSlice<i8> = stream.alloc_zeros::<i8>(k).unwrap();
    let mut dact_s: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let mut dy_ref: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pq, _c) = dact.device_ptr_mut(&stream);
        let (ps, _d) = dact_s.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_mul_rowquant_i8_m1(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pq as *mut i8,
                ps as *mut f32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "silu_mul_rowquant_i8_m1 rc={rc}");
    let rc = {
        let (pw, _a) = dw.device_ptr(&stream);
        let (ps, _b) = ds.device_ptr(&stream);
        let (pq, _c) = dact.device_ptr(&stream);
        let (pqs, _d) = dact_s.device_ptr(&stream);
        let (pr, _e) = dres.device_ptr(&stream);
        let (py, _f) = dy_ref.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_down_residual_m1(
                stream.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const u8,
                pq as *const i8,
                pqs as *const f32,
                pr as *const u16,
                py as *mut u16,
                alpha,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_m1 rc={rc}");

    let plen = cuda::silu_mul_rowquant_i8_mk_partials_len(1, k as i32);
    assert!(plen > 0);
    let mut dstage: CudaSlice<u16> = stream.alloc_zeros::<u16>(k).unwrap();
    let mut dpart: CudaSlice<f32> = stream.alloc_zeros::<f32>(plen as usize).unwrap();
    let mut dy_qf: CudaSlice<u16> = stream.alloc_zeros::<u16>(n).unwrap();
    let rc = {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pst, _c) = dstage.device_ptr_mut(&stream);
        let (pp, _d) = dpart.device_ptr_mut(&stream);
        unsafe {
            cuda::silu_mul_stage_partial_absmax_m1(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pst as *mut u16,
                pp as *mut f32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "silu_mul_stage_partial_absmax_m1 rc={rc}");
    let rc = {
        let (pw, _a) = dw.device_ptr(&stream);
        let (ps, _b) = ds.device_ptr(&stream);
        let (pst, _c) = dstage.device_ptr(&stream);
        let (pp, _d) = dpart.device_ptr(&stream);
        let (pr, _e) = dres.device_ptr(&stream);
        let (py, _f) = dy_qf.device_ptr_mut(&stream);
        unsafe {
            cuda::gemv_nvfp4_w4a8_down_residual_quant_prologue_m1(
                stream.cu_stream() as *mut c_void,
                pw as *const u8,
                ps as *const u8,
                pst as *const u16,
                pp as *const f32,
                plen,
                pr as *const u16,
                py as *mut u16,
                alpha,
                n as i32,
                k as i32,
            )
        }
    };
    assert_eq!(rc, 0, "gemv_nvfp4_w4a8_down_residual_quant_prologue_m1 rc={rc}");
    stream.synchronize().unwrap();
    #[allow(deprecated)]
    let y_ref = stream.memcpy_dtov(&dy_ref).unwrap();
    #[allow(deprecated)]
    let y_qf = stream.memcpy_dtov(&dy_qf).unwrap();
    assert_eq!(
        y_ref, y_qf,
        "quant-prologue down must match the unfused chain bitwise: same scale via associative fmax, same round-to-nearest, same dot order"
    );
}

#[test]
fn gemv_nvfp4_w4a8_refuses_ragged_k_and_oversized_smem_without_launching() {
    let Some(stream) = stream("gemv_nvfp4_w4a8_refusal") else {
        return;
    };
    let rc = unsafe {
        cuda::gemv_nvfp4_w4a8_dual_m1(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            1.0,
            1.0,
            16,
            24,
        )
    };
    assert_eq!(rc, -1, "ragged K must refuse, got {rc}");
    let rc = unsafe {
        cuda::gemv_nvfp4_w4a8_down_residual_m1(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            1.0,
            16,
            128 * 1024,
        )
    };
    assert_eq!(rc, -1, "over-smem K must refuse, got {rc}");
    let rc = unsafe {
        cuda::silu_mul_rowquant_i8_m1(
            stream.cu_stream() as *mut c_void,
            std::ptr::null(),
            std::ptr::null(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            64 * 1024,
        )
    };
    assert_eq!(rc, -1, "over-smem K must refuse in the silu quant producer, got {rc}");
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- times the silu-quant producer alternatives (single-block m1 vs multi-block mk split), the post-norm+rowquant fold vs the two-kernel chain, and the small-m dual/down dp4a arms m=1..8 on the q38 dense shapes"]
fn gemv_nvfp4_w4a8_producer_and_fold_alternatives_shape_bench() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x2545f4914f6cdd1d);
    let (hidden, inter) = (5120usize, 17408usize);
    let eps = 1.0e-6f32;
    let gate_words = rng.bf16_words(inter, 1.0);
    let up_words = rng.bf16_words(inter, 1.0);
    let x_words = rng.bf16_words(hidden, 1.0);
    let r_words = rng.bf16_words(hidden, 1.0);
    let w_words = rng.bf16_words(hidden, 1.0);
    #[allow(deprecated)]
    let dg: CudaSlice<u16> = stream.clone_htod(&gate_words).unwrap();
    #[allow(deprecated)]
    let du: CudaSlice<u16> = stream.clone_htod(&up_words).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x_words).unwrap();
    #[allow(deprecated)]
    let dr: CudaSlice<u16> = stream.clone_htod(&r_words).unwrap();
    #[allow(deprecated)]
    let dw: CudaSlice<u16> = stream.clone_htod(&w_words).unwrap();
    let plen = cuda::silu_mul_rowquant_i8_mk_partials_len(1, inter as i32) as usize;
    let mut dstage: CudaSlice<u16> = stream.alloc_zeros::<u16>(inter).unwrap();
    let mut dpart: CudaSlice<f32> = stream.alloc_zeros::<f32>(plen).unwrap();
    let mut dq: CudaSlice<i8> = stream.alloc_zeros::<i8>(inter).unwrap();
    let mut dqs: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let mut dres_out: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();
    let mut dnormed: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();
    let mut dhq: CudaSlice<i8> = stream.alloc_zeros::<i8>(hidden).unwrap();
    let mut dhqs: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();

    let mut time_phase = |name: &str, f: &mut dyn FnMut()| {
        for _ in 0..5 {
            f();
        }
        stream.synchronize().unwrap();
        let iters = 200usize;
        let t0 = Instant::now();
        for _ in 0..iters {
            f();
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        eprintln!("W4A8-ALT-BENCH phase={name} hidden={hidden} inter={inter} us={us:.2}");
    };

    time_phase("siluq_m1_single_block", &mut || {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pq, _c) = dq.device_ptr_mut(&stream);
        let (ps, _d) = dqs.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::silu_mul_rowquant_i8_m1(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pq as *mut i8,
                ps as *mut f32,
                inter as i32,
            )
        };
        assert_eq!(rc, 0);
    });
    time_phase("siluq_mk_split_multiblock", &mut || {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pst, _c) = dstage.device_ptr_mut(&stream);
        let (pp, _d) = dpart.device_ptr_mut(&stream);
        let (pq, _e) = dq.device_ptr_mut(&stream);
        let (ps, _f) = dqs.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::silu_mul_rowquant_i8_mk(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pst as *mut u16,
                pp as *mut f32,
                pq as *mut i8,
                ps as *mut f32,
                1,
                inter as i32,
            )
        };
        assert_eq!(rc, 0);
    });
    time_phase("normquant_two_kernel_chain", &mut || {
        let (px, _a) = dx.device_ptr(&stream);
        let (pr, _b) = dr.device_ptr(&stream);
        let (pw, _c) = dw.device_ptr(&stream);
        let (pro, _d) = dres_out.device_ptr_mut(&stream);
        let (pno, _e) = dnormed.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::rmsnorm_residual_writeout_bf16(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                pro as *mut u16,
                pno as *mut u16,
                1,
                hidden,
                eps,
            )
        };
        assert_eq!(rc, 0);
        let (pq, _f) = dhq.device_ptr_mut(&stream);
        let (ps, _g) = dhqs.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::rowquant_i8(
                stream.cu_stream() as *mut c_void,
                pno as *const u16,
                pq as *mut i8,
                ps as *mut f32,
                1,
                hidden as i32,
            )
        };
        assert_eq!(rc, 0);
    });
    time_phase("normquant_fused", &mut || {
        let (px, _a) = dx.device_ptr(&stream);
        let (pr, _b) = dr.device_ptr(&stream);
        let (pw, _c) = dw.device_ptr(&stream);
        let (pro, _d) = dres_out.device_ptr_mut(&stream);
        let (pq, _e) = dhq.device_ptr_mut(&stream);
        let (ps, _f) = dhqs.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::rmsnorm_residual_writeout_rowquant_i8_m1(
                stream.cu_stream() as *mut c_void,
                px as *const u16,
                pr as *const u16,
                pw as *const u16,
                pro as *mut u16,
                pq as *mut i8,
                ps as *mut f32,
                hidden as i32,
                eps,
            )
        };
        assert_eq!(rc, 0);
    });

    time_phase("siluq_pass1_only", &mut || {
        let (pg, _a) = dg.device_ptr(&stream);
        let (pu, _b) = du.device_ptr(&stream);
        let (pst, _c) = dstage.device_ptr_mut(&stream);
        let (pp, _d) = dpart.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::silu_mul_stage_partial_absmax_m1(
                stream.cu_stream() as *mut c_void,
                pg as *const u16,
                pu as *const u16,
                pst as *mut u16,
                pp as *mut f32,
                inter as i32,
            )
        };
        assert_eq!(rc, 0);
    });

    let packed_g = rng.packed_nibbles(inter * hidden / 2);
    let packed_u = rng.packed_nibbles(inter * hidden / 2);
    let packed_d = rng.packed_nibbles(hidden * inter / 2);
    let sw_g = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(inter * hidden / 16),
        inter,
        hidden / 16,
    );
    let sw_u = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(inter * hidden / 16),
        inter,
        hidden / 16,
    );
    let sw_d = nv_quant::nvfp4::swizzle_scales(
        &rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(hidden * inter / 16),
        hidden,
        inter / 16,
    );
    #[allow(deprecated)]
    let dwg: CudaSlice<u8> = stream.clone_htod(&packed_g).unwrap();
    #[allow(deprecated)]
    let dwu: CudaSlice<u8> = stream.clone_htod(&packed_u).unwrap();
    #[allow(deprecated)]
    let dwd: CudaSlice<u8> = stream.clone_htod(&packed_d).unwrap();
    #[allow(deprecated)]
    let dsg: CudaSlice<u8> = stream.clone_htod(&sw_g).unwrap();
    #[allow(deprecated)]
    let dsu: CudaSlice<u8> = stream.clone_htod(&sw_u).unwrap();
    #[allow(deprecated)]
    let dsd: CudaSlice<u8> = stream.clone_htod(&sw_d).unwrap();
    let max_m = 8usize;
    let xq_all: Vec<i8> = (0..max_m * inter.max(hidden))
        .map(|_| ((rng.next_u32() % 255) as i32 - 127) as i8)
        .collect();
    let xs_all: Vec<f32> = (0..max_m).map(|_| 0.004f32).collect();
    #[allow(deprecated)]
    let dxq_all: CudaSlice<i8> = stream.clone_htod(&xq_all).unwrap();
    #[allow(deprecated)]
    let dxs_all: CudaSlice<f32> = stream.clone_htod(&xs_all).unwrap();
    let res_all = rng.bf16_words(max_m * hidden, 1.0);
    #[allow(deprecated)]
    let dres_all: CudaSlice<u16> = stream.clone_htod(&res_all).unwrap();
    let mut dya: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_m * inter).unwrap();
    let mut dyb: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_m * inter).unwrap();
    let mut dyd: CudaSlice<u16> = stream.alloc_zeros::<u16>(max_m * hidden).unwrap();

    time_phase("down_qfold_m1", &mut || {
        let (pwd, _a) = dwd.device_ptr(&stream);
        let (psd, _b) = dsd.device_ptr(&stream);
        let (pst, _c) = dstage.device_ptr(&stream);
        let (pp, _d) = dpart.device_ptr(&stream);
        let (pr, _e) = dres_all.device_ptr(&stream);
        let (py, _f) = dyd.device_ptr_mut(&stream);
        let rc = unsafe {
            cuda::gemv_nvfp4_w4a8_down_residual_quant_prologue_m1(
                stream.cu_stream() as *mut c_void,
                pwd as *const u8,
                psd as *const u8,
                pst as *const u16,
                pp as *const f32,
                plen as i32,
                pr as *const u16,
                py as *mut u16,
                0.01,
                hidden as i32,
                inter as i32,
            )
        };
        assert_eq!(rc, 0);
    });

    for m in 1..=max_m {
        let name = format!("dual_mk_m{m}");
        time_phase(&name, &mut || {
            let (pwg, _a) = dwg.device_ptr(&stream);
            let (psg, _b) = dsg.device_ptr(&stream);
            let (pwu, _c) = dwu.device_ptr(&stream);
            let (psu, _d) = dsu.device_ptr(&stream);
            let (px, _e) = dxq_all.device_ptr(&stream);
            let (pxs, _f) = dxs_all.device_ptr(&stream);
            let (pya, _g) = dya.device_ptr_mut(&stream);
            let (pyb, _h) = dyb.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_dual_mk(
                    stream.cu_stream() as *mut c_void,
                    pwg as *const u8,
                    psg as *const u8,
                    pwu as *const u8,
                    psu as *const u8,
                    px as *const i8,
                    pxs as *const f32,
                    pya as *mut u16,
                    pyb as *mut u16,
                    0.01,
                    0.01,
                    m as i32,
                    inter as i32,
                    hidden as i32,
                )
            };
            assert_eq!(rc, 0);
        });
        let name = format!("down_mk_m{m}");
        time_phase(&name, &mut || {
            let (pwd, _a) = dwd.device_ptr(&stream);
            let (psd, _b) = dsd.device_ptr(&stream);
            let (px, _c) = dxq_all.device_ptr(&stream);
            let (pxs, _d) = dxs_all.device_ptr(&stream);
            let (pr, _e) = dres_all.device_ptr(&stream);
            let (py, _f) = dyd.device_ptr_mut(&stream);
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_down_residual_mk(
                    stream.cu_stream() as *mut c_void,
                    pwd as *const u8,
                    psd as *const u8,
                    px as *const i8,
                    pxs as *const f32,
                    pr as *const u16,
                    py as *mut u16,
                    0.01,
                    m as i32,
                    hidden as i32,
                    inter as i32,
                )
            };
            assert_eq!(rc, 0);
        });
    }
}

#[test]
#[ignore = "measurement ladder, GPU-time only; set NV_KERNELS_BENCH=1 -- times the four-kernel w4a8 decode MLP (rowquant_i8 x, dual gate+up dp4a, silu-mul rowquant producer, down dp4a with residual epilogue) on the q38 dense shapes (dual N=17408 K=5120, down N=5120 K=17408) so the per-layer cost is comparable against the 154 us/layer padded-A4 tensor-core route and the 157 us/layer w4a16 gemv pair"]
fn gemv_nvfp4_w4a8_decode_mlp_shape_bench() {
    if std::env::var("NV_KERNELS_BENCH").as_deref() != Ok("1") {
        eprintln!("SKIP: set NV_KERNELS_BENCH=1 to run");
        return;
    }
    let ctx = CudaContext::new(0).expect("cuda device 0");
    let stream = ctx.default_stream();
    let mut rng = Lcg(0x9e3779b97f4a7c15);
    let (hidden, inter) = (5120usize, 17408usize);
    let packed_g = rng.packed_nibbles(inter * hidden / 2);
    let packed_u = rng.packed_nibbles(inter * hidden / 2);
    let packed_d = rng.packed_nibbles(hidden * inter / 2);
    let sc_g = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(inter * hidden / 16);
    let sc_u = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(inter * hidden / 16);
    let sc_d = rng.plausible_ue4m3_scale_bytes_biased_to_small_exponents(hidden * inter / 16);
    let sw_g = nv_quant::nvfp4::swizzle_scales(&sc_g, inter, hidden / 16);
    let sw_u = nv_quant::nvfp4::swizzle_scales(&sc_u, inter, hidden / 16);
    let sw_d = nv_quant::nvfp4::swizzle_scales(&sc_d, hidden, inter / 16);
    let x = rng.bf16_words(hidden, 1.0);
    let residual = rng.bf16_words(hidden, 1.0);
    #[allow(deprecated)]
    let dwg: CudaSlice<u8> = stream.clone_htod(&packed_g).unwrap();
    #[allow(deprecated)]
    let dwu: CudaSlice<u8> = stream.clone_htod(&packed_u).unwrap();
    #[allow(deprecated)]
    let dwd: CudaSlice<u8> = stream.clone_htod(&packed_d).unwrap();
    #[allow(deprecated)]
    let dsg: CudaSlice<u8> = stream.clone_htod(&sw_g).unwrap();
    #[allow(deprecated)]
    let dsu: CudaSlice<u8> = stream.clone_htod(&sw_u).unwrap();
    #[allow(deprecated)]
    let dsd: CudaSlice<u8> = stream.clone_htod(&sw_d).unwrap();
    #[allow(deprecated)]
    let dx: CudaSlice<u16> = stream.clone_htod(&x).unwrap();
    #[allow(deprecated)]
    let dres: CudaSlice<u16> = stream.clone_htod(&residual).unwrap();
    let mut dxq: CudaSlice<i8> = stream.alloc_zeros::<i8>(hidden).unwrap();
    let mut dxs: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let mut dg: CudaSlice<u16> = stream.alloc_zeros::<u16>(inter).unwrap();
    let mut du: CudaSlice<u16> = stream.alloc_zeros::<u16>(inter).unwrap();
    let mut dactq: CudaSlice<i8> = stream.alloc_zeros::<i8>(inter).unwrap();
    let mut dacts: CudaSlice<f32> = stream.alloc_zeros::<f32>(1).unwrap();
    let mut dy: CudaSlice<u16> = stream.alloc_zeros::<u16>(hidden).unwrap();

    let phase = std::env::var("NV_KERNELS_BENCH_PHASE").unwrap_or_else(|_| "all".into());
    let want = |name: &str| phase == "all" || phase == name;
    let mut one_layer = || {
        let (px, _a) = dx.device_ptr(&stream);
        let (pxq, _b) = dxq.device_ptr_mut(&stream);
        let (pxs, _c) = dxs.device_ptr_mut(&stream);
        if want("rowquant") {
            let rc = unsafe {
                cuda::rowquant_i8(
                    stream.cu_stream() as *mut c_void,
                    px as *const u16,
                    pxq as *mut i8,
                    pxs as *mut f32,
                    1,
                    hidden as i32,
                )
            };
            assert_eq!(rc, 0, "rowquant rc={rc}");
        }
        let (pwg, _d) = dwg.device_ptr(&stream);
        let (pwu, _e) = dwu.device_ptr(&stream);
        let (psg, _f) = dsg.device_ptr(&stream);
        let (psu, _g) = dsu.device_ptr(&stream);
        let (pg, _h) = dg.device_ptr_mut(&stream);
        let (pu, _i) = du.device_ptr_mut(&stream);
        if want("dual") {
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_dual_m1(
                    stream.cu_stream() as *mut c_void,
                    pwg as *const u8,
                    psg as *const u8,
                    pwu as *const u8,
                    psu as *const u8,
                    pxq as *const i8,
                    pxs as *const f32,
                    pg as *mut u16,
                    pu as *mut u16,
                    0.01,
                    0.01,
                    inter as i32,
                    hidden as i32,
                )
            };
            assert_eq!(rc, 0, "dual rc={rc}");
        }
        let (pactq, _j) = dactq.device_ptr_mut(&stream);
        let (pacts, _k) = dacts.device_ptr_mut(&stream);
        if want("siluq") {
            let rc = unsafe {
                cuda::silu_mul_rowquant_i8_m1(
                    stream.cu_stream() as *mut c_void,
                    pg as *const u16,
                    pu as *const u16,
                    pactq as *mut i8,
                    pacts as *mut f32,
                    inter as i32,
                )
            };
            assert_eq!(rc, 0, "silu quant rc={rc}");
        }
        let (pwd, _l) = dwd.device_ptr(&stream);
        let (psd, _m) = dsd.device_ptr(&stream);
        let (pr, _n) = dres.device_ptr(&stream);
        let (py, _o) = dy.device_ptr_mut(&stream);
        if want("down") {
            let rc = unsafe {
                cuda::gemv_nvfp4_w4a8_down_residual_m1(
                    stream.cu_stream() as *mut c_void,
                    pwd as *const u8,
                    psd as *const u8,
                    pactq as *const i8,
                    pacts as *const f32,
                    pr as *const u16,
                    py as *mut u16,
                    0.01,
                    hidden as i32,
                    inter as i32,
                )
            };
            assert_eq!(rc, 0, "down rc={rc}");
        }
    };

    for _ in 0..3 {
        one_layer();
    }
    stream.synchronize().unwrap();
    let iters = 100usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        one_layer();
    }
    stream.synchronize().unwrap();
    let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
    let weight_bytes = (3 * inter * hidden) / 2 + (3 * inter * hidden) / 16;
    let gbs = weight_bytes as f64 / us / 1e3;
    eprintln!(
        "NVFP4-W4A8-MLP-LAYER phase={phase} hidden={hidden} inter={inter} us_per_layer={us:.1} weight_gbs={gbs:.0}"
    );
}
