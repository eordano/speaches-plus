#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::dequant::bytes_to_words;
use nv_kernels::wgpu_backend::device::{shared_or_reason, WgpuContext};
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4;
use nv_quant::nvfp4::{
    decode_e2m1, decode_ue4m3, quantize_block_with_global, swizzle_scales, unpack_e2m1_pair,
    Nvfp4Tensor, BLOCK_SIZE,
};
use common::LcgShift40Top24TwoSided as Lcg;
use common::ctx_or_skip_reasoned as ctx_or_skip;
use common::swizzled_scale_dst;

fn cpu_gemv_oracle(
    w_packed: &[u8],
    w_scales_swizzled: &[u8],
    x_packed: &[u8],
    x_scales: &[u8],
    alpha: f32,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let k_blocks = k / BLOCK_SIZE;
    let row_bytes = k / 2;
    let mut out = vec![0f32; n];
    for (row, slot) in out.iter_mut().enumerate() {
        let mut acc = 0f32;
        for kb in 0..k_blocks {
            let ws = decode_ue4m3(w_scales_swizzled[swizzled_scale_dst(row, kb, k_blocks)]);
            let xs = decode_ue4m3(x_scales[kb]);
            let w_off = row * row_bytes + kb * 8;
            let x_off = kb * 8;
            let mut dot = 0f32;
            for i in 0..8 {
                let (w_lo, w_hi) = unpack_e2m1_pair(w_packed[w_off + i]);
                let (x_lo, x_hi) = unpack_e2m1_pair(x_packed[x_off + i]);
                dot +=
                    decode_e2m1(w_lo) * decode_e2m1(x_lo) + decode_e2m1(w_hi) * decode_e2m1(x_hi);
            }
            acc += dot * ws * xs;
        }
        *slot = acc * alpha;
    }
    out
}

fn quantize_row_cpu(values: &[f32], stored_global: f32) -> (Vec<u8>, Vec<u8>) {
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    for block in values.chunks(BLOCK_SIZE) {
        let (p, s) = quantize_block_with_global(block, stored_global);
        packed.extend_from_slice(&p);
        scales.push(s);
    }
    (packed, scales)
}

fn random_row(rng: &mut Lcg, k: usize, gain: f32) -> Vec<f32> {
    (0..k)
        .map(|_| bf16::from_f32(rng.next_f32() * gain).to_f32())
        .collect()
}

fn words_to_bytes(words: &[u32], len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((words[i / 4] >> (8 * (i % 4))) & 0xff) as u8)
        .collect()
}

fn dequantize_row(packed: &[u8], scales: &[u8], k: usize, global: f32) -> Vec<f32> {
    let inv_global = if global == 0.0 || !global.is_finite() {
        1.0
    } else {
        1.0 / global
    };
    let mut out = Vec::with_capacity(k);
    for kb in 0..k / BLOCK_SIZE {
        let s = decode_ue4m3(scales[kb]);
        for i in 0..8 {
            let (lo, hi) = unpack_e2m1_pair(packed[kb * 8 + i]);
            out.push(decode_e2m1(lo) * s * inv_global);
            out.push(decode_e2m1(hi) * s * inv_global);
        }
    }
    out
}

fn max_recon_err(row: &[f32], packed: &[u8], scales: &[u8], k: usize, global: f32) -> f32 {
    dequantize_row(packed, scales, k, global)
        .iter()
        .zip(row.iter())
        .fold(0f32, |acc, (got, want)| acc.max((got - want).abs()))
}

#[test]
fn quantize_row_is_bit_exact_on_the_representable_grid() {
    let Some(ctx) = ctx_or_skip("quantize_row_is_bit_exact_on_the_representable_grid") else {
        return;
    };
    let grid = [
        6.0f32, 0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, -6.0, 0.5,
    ];
    let k = 16 * 64;
    let row: Vec<f32> = (0..k).map(|j| grid[j % 16] * 4.0).collect();
    let x_bits: Vec<u16> = row.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();

    for stored_global in [1.0f32, 0.5, 4.0] {
        let mut packed_out = vec![0u32; k / 8];
        let mut scales_out = vec![0u8; k / BLOCK_SIZE];
        gemv_nvfp4::nvfp4_quantize_row_bf16(
            ctx,
            &x_bits,
            stored_global,
            &mut packed_out,
            &mut scales_out,
            k,
        )
        .expect("gpu quantize");

        let (cpu_packed, cpu_scales) = quantize_row_cpu(&row, stored_global);
        assert_eq!(
            packed_out,
            bytes_to_words(&cpu_packed),
            "packed nibbles differ for stored_global {stored_global}"
        );
        assert_eq!(
            scales_out, cpu_scales,
            "e4m3 scale bytes differ for stored_global {stored_global}"
        );
        assert_eq!(
            max_recon_err(&row, &cpu_packed, &cpu_scales, k, stored_global),
            0.0,
            "grid values must round-trip exactly"
        );
    }
}

#[test]
fn quantize_row_tracks_the_cpu_encoder_on_random_data() {
    let Some(ctx) = ctx_or_skip("quantize_row_tracks_the_cpu_encoder_on_random_data") else {
        return;
    };
    let k = 16 * 300;
    let mut rng = Lcg(0x51ed_2701);
    for stored_global in [1.0f32, 0.375, 3.0, 0.0] {
        let row = random_row(&mut rng, k, 7.5);
        let x_bits: Vec<u16> = row.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
        let mut packed_out = vec![0u32; k / 8];
        let mut scales_out = vec![0u8; k / BLOCK_SIZE];
        gemv_nvfp4::nvfp4_quantize_row_bf16(
            ctx,
            &x_bits,
            stored_global,
            &mut packed_out,
            &mut scales_out,
            k,
        )
        .expect("gpu quantize");

        let (cpu_packed, cpu_scales) = quantize_row_cpu(&row, stored_global);
        let gpu_packed = words_to_bytes(&packed_out, k / 2);
        let scale_diffs = scales_out
            .iter()
            .zip(cpu_scales.iter())
            .filter(|(a, b)| a != b)
            .count();
        let gpu_vals = dequantize_row(&gpu_packed, &scales_out, k, stored_global);
        let cpu_vals = dequantize_row(&cpu_packed, &cpu_scales, k, stored_global);
        let nibble_diffs = gpu_vals
            .iter()
            .zip(cpu_vals.iter())
            .filter(|(a, b)| a != b)
            .count();
        let gpu_err = max_recon_err(&row, &gpu_packed, &scales_out, k, stored_global);
        let cpu_err = max_recon_err(&row, &cpu_packed, &cpu_scales, k, stored_global);
        eprintln!(
            "quantize global={stored_global}: scale_byte_diffs={scale_diffs}/{} value_diffs={nibble_diffs}/{k} gpu_recon={gpu_err:.6e} cpu_recon={cpu_err:.6e}",
            cpu_scales.len()
        );
        assert_eq!(scale_diffs, 0, "e4m3 scale bytes must agree exactly");
        assert!(
            gpu_err <= cpu_err * 1.5 + 1e-6,
            "gpu reconstruction {gpu_err} worse than cpu {cpu_err}"
        );
    }
}

fn run_gemv_case(ctx: &WgpuContext, n: usize, k: usize, seed: u64) -> (f32, f32) {
    let mut rng = Lcg(seed);
    let rows: Vec<Vec<f32>> = (0..n).map(|_| random_row(&mut rng, k, 1.0)).collect();
    let x_row = random_row(&mut rng, k, 2.0);

    let stored_weight_global = 0.5f32;
    let stored_input_global = 1.25f32;
    let alpha = (1.0 / stored_weight_global) * (1.0 / stored_input_global);

    let w = Nvfp4Tensor::quantize_rows_with_global(&rows, stored_weight_global);
    let k_blocks = k / BLOCK_SIZE;
    let w_scales_swizzled = swizzle_scales(&w.scales, n, k_blocks);

    let x_bits: Vec<u16> = x_row.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    let mut x_packed_words = vec![0u32; k / 8];
    let mut x_scales = vec![0u8; k_blocks];
    gemv_nvfp4::nvfp4_quantize_row_bf16(
        ctx,
        &x_bits,
        stored_input_global,
        &mut x_packed_words,
        &mut x_scales,
        k,
    )
    .expect("gpu quantize activation");

    let mut x_packed_bytes = vec![0u8; k / 2];
    for (i, b) in x_packed_bytes.iter_mut().enumerate() {
        *b = ((x_packed_words[i / 4] >> (8 * (i % 4))) & 0xff) as u8;
    }

    let w_words = bytes_to_words(&w.data);
    let mut y = vec![0u16; n];
    gemv_nvfp4::nvfp4_gemv_bf16(
        ctx,
        &w_words,
        &w_scales_swizzled,
        &x_packed_words,
        &x_scales,
        alpha,
        &mut y,
        n,
        k,
    )
    .expect("gpu gemv");

    let oracle = cpu_gemv_oracle(
        &w.data,
        &w_scales_swizzled,
        &x_packed_bytes,
        &x_scales,
        alpha,
        n,
        k,
    );

    let mut max_abs = 0f32;
    let mut max_rel = 0f32;
    let mut exact = 0usize;
    let mut max_ulp = 0i32;
    let mut max_abs_bf16 = 0f32;
    for (row, want) in oracle.iter().enumerate() {
        let want_bits = bf16::from_f32(*want).to_bits();
        if y[row] == want_bits {
            exact += 1;
        }
        let d = (bf16::from_bits(y[row]).to_f32() - bf16::from_bits(want_bits).to_f32()).abs();
        if d > max_abs_bf16 {
            max_abs_bf16 = d;
        }
        let ulp = (y[row] as i32 - want_bits as i32).abs();
        if ulp > max_ulp {
            max_ulp = ulp;
        }
        let got = bf16::from_bits(y[row]).to_f32();
        let diff = (got - want).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        let denom = want.abs().max(1e-6);
        let rel = diff / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    eprintln!(
        "gemv n={n} k={k}: max_abs_err_vs_f32_oracle={max_abs:.6e} max_rel_err={max_rel:.6e} max_abs_err_vs_bf16_oracle={max_abs_bf16:.6e} bit_exact={exact}/{n} max_bf16_ulp={max_ulp} (oracle[0]={:.6})",
        oracle[0]
    );
    assert!(
        max_ulp <= 1,
        "row output differs from the cpu oracle by {max_ulp} bf16 ulp"
    );
    (max_abs, max_rel)
}

#[test]
fn gemv_matches_the_cpu_oracle_for_a_tall_k() {
    let Some(ctx) = ctx_or_skip("gemv_matches_the_cpu_oracle_for_a_tall_k") else {
        return;
    };
    let (max_abs, max_rel) = run_gemv_case(ctx, 5, 16 * 300, 0x1234_5678);
    assert!(
        max_rel < 8e-3,
        "relative error {max_rel} too large (abs {max_abs})"
    );
}

#[test]
fn gemv_handles_many_rows_and_short_k() {
    let Some(ctx) = ctx_or_skip("gemv_handles_many_rows_and_short_k") else {
        return;
    };
    let (max_abs, max_rel) = run_gemv_case(ctx, 200, 64, 0x0bad_c0de);
    assert!(
        max_rel < 8e-3,
        "relative error {max_rel} too large (abs {max_abs})"
    );
}

#[test]
fn gemv_dequant_path_is_exact_for_representable_values() {
    let Some(ctx) = ctx_or_skip("gemv_dequant_path_is_exact_for_representable_values") else {
        return;
    };
    let n = 3usize;
    let k = 32usize;
    let grid = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.0, -2.0, -6.0,
    ];
    let rows: Vec<Vec<f32>> = (0..n)
        .map(|r| (0..k).map(|j| grid[(r * 7 + j) % grid.len()]).collect())
        .collect();
    let x_row: Vec<f32> = (0..k).map(|j| grid[(j * 5 + 1) % grid.len()]).collect();

    let w = Nvfp4Tensor::quantize_rows_with_global(&rows, 1.0);
    let k_blocks = k / BLOCK_SIZE;
    let w_scales_swizzled = swizzle_scales(&w.scales, n, k_blocks);
    let (x_packed, x_scales) = quantize_row_cpu(&x_row, 1.0);

    let w_words = bytes_to_words(&w.data);
    let x_words = bytes_to_words(&x_packed);
    let mut y = vec![0u16; n];
    gemv_nvfp4::nvfp4_gemv_bf16(
        ctx,
        &w_words,
        &w_scales_swizzled,
        &x_words,
        &x_scales,
        1.0,
        &mut y,
        n,
        k,
    )
    .expect("gpu gemv");

    let oracle = cpu_gemv_oracle(&w.data, &w_scales_swizzled, &x_packed, &x_scales, 1.0, n, k);
    for (row, want) in oracle.iter().enumerate() {
        assert_eq!(
            y[row],
            bf16::from_f32(*want).to_bits(),
            "row {row}: gpu {:?} cpu {want}",
            bf16::from_bits(y[row]).to_f32()
        );
    }
}

#[test]
fn shape_errors_are_reported_without_a_device() {
    let mut y = vec![0u16; 2];
    let Some(ctx) = ctx_or_skip("shape_errors_are_reported_without_a_device") else {
        return;
    };
    let err = gemv_nvfp4::nvfp4_gemv_bf16(ctx, &[], &[], &[], &[], 1.0, &mut y, 2, 20)
        .expect_err("K must be rejected");
    assert!(format!("{err}").contains("not a multiple of 16"));
}
