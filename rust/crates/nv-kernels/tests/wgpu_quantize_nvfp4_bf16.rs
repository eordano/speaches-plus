#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::lcg;
use half::bf16;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_kernels::wgpu_backend::kernels::quantize_nvfp4_bf16::{
    quantize_nvfp4_bf16, quantize_nvfp4_bf16_per_expert, scale_rows,
    silu_mul_quantize_nvfp4_bf16_per_expert, swizzled_scale_bytes,
};
use nv_quant::nvfp4::{decode_ue4m3, dequantize_block, swizzle_scales, Nvfp4Tensor, BLOCK_SIZE};

fn rand_bf16(state: &mut u64, amp: f32) -> bf16 {
    let u = (lcg(state) >> 40) as f32 / (1u64 << 24) as f32;
    bf16::from_f32((u - 0.5) * 2.0 * amp)
}

fn rows_of(bits: &[u16], rows: usize, k: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| {
            bits[r * k..(r + 1) * k]
                .iter()
                .map(|b| bf16::from_bits(*b).to_f32())
                .collect()
        })
        .collect()
}

fn reference(
    rows: &[Vec<f32>],
    globals: &[f32],
    rows_per_expert: usize,
    m_data_rows: usize,
    dispatch_rows: usize,
    k: usize,
) -> (Vec<u8>, Vec<u8>) {
    let blocks_per_row = k / BLOCK_SIZE;
    let mut packed = vec![0u8; m_data_rows * k / 2];
    let mut linear = vec![0u8; scale_rows(dispatch_rows) * blocks_per_row];
    for (r, row) in rows.iter().enumerate() {
        let e = (r / rows_per_expert).min(globals.len() - 1);
        let q = Nvfp4Tensor::quantize_rows_with_global(std::slice::from_ref(row), globals[e]);
        let row_bytes = k / 2;
        packed[r * row_bytes..(r + 1) * row_bytes].copy_from_slice(&q.data);
        linear[r * blocks_per_row..(r + 1) * blocks_per_row].copy_from_slice(&q.scales);
    }
    let scales = swizzle_scales(&linear, scale_rows(dispatch_rows), blocks_per_row);
    (packed, scales)
}

fn assert_bytes_equal(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mismatches = got.iter().zip(want.iter()).filter(|(a, b)| a != b).count();
    if mismatches != 0 {
        let first = got
            .iter()
            .zip(want.iter())
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(i, (a, b))| (i, *a, *b))
            .unwrap();
        panic!(
            "{what}: {mismatches}/{} bytes differ; first at {} got {:#04x} want {:#04x}",
            got.len(),
            first.0,
            first.1,
            first.2
        );
    }
}

fn max_abs_err(
    orig: &[Vec<f32>],
    packed: &[u8],
    scales_linear: &[u8],
    k: usize,
    globals: &[f32],
    rows_per_expert: usize,
) -> f32 {
    let blocks_per_row = k / BLOCK_SIZE;
    let mut worst = 0.0f32;
    for (r, row) in orig.iter().enumerate() {
        let e = (r / rows_per_expert).min(globals.len() - 1);
        let stored = if globals[e] == 0.0 || !globals[e].is_finite() {
            1.0
        } else {
            globals[e]
        };
        for b in 0..blocks_per_row {
            let off = r * (k / 2) + b * (BLOCK_SIZE / 2);
            let deq = dequantize_block(
                &packed[off..off + BLOCK_SIZE / 2],
                scales_linear[r * blocks_per_row + b],
            );
            for (i, v) in deq.iter().enumerate() {
                let d = (row[b * BLOCK_SIZE + i] - v / stored).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
    }
    worst
}

fn linear_scales(rows: &[Vec<f32>], globals: &[f32], rows_per_expert: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for (r, row) in rows.iter().enumerate() {
        let e = (r / rows_per_expert).min(globals.len() - 1);
        let q = Nvfp4Tensor::quantize_rows_with_global(std::slice::from_ref(row), globals[e]);
        out.extend_from_slice(&q.scales);
    }
    out
}

#[test]
fn wgpu_quantize_nvfp4_bf16_matches_nv_quant_byte_exact() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_nvfp4_bf16_matches_nv_quant_byte_exact") else {
        return;
    };
    let mut st = 0x9e3779b97f4a7c15u64;
    for &(m, k, global, amp) in &[
        (128usize, 128usize, 1.0f32, 2.5f32),
        (128, 512, 0.3, 3.0),
        (17, 64, 2.75, 0.02),
        (1, 16, 1.0, 1.0),
        (200, 96, 0.125, 40.0),
    ] {
        let x: Vec<u16> = (0..m * k)
            .map(|_| rand_bf16(&mut st, amp).to_bits())
            .collect();
        let rows = rows_of(&x, m, k);
        let (packed_want, scales_want) = reference(&rows, &[global], m.max(1), m, m, k);

        let mut packed_got = vec![0xabu8; m * k / 2];
        let mut scales_got = vec![0xcdu8; swizzled_scale_bytes(m, k)];
        quantize_nvfp4_bf16(ctx, &x, global, &mut packed_got, &mut scales_got, m, m, k)
            .expect("quantize_nvfp4_bf16");

        assert_bytes_equal(&format!("packed m={m} k={k}"), &packed_got, &packed_want);
        assert_bytes_equal(&format!("scales m={m} k={k}"), &scales_got, &scales_want);

        let lin = linear_scales(&rows, &[global], m.max(1));
        let err = max_abs_err(&rows, &packed_got, &lin, k, &[global], m.max(1));
        eprintln!(
            "wgpu_quantize_nvfp4_bf16: m={m} k={k} global={global} amp={amp} max_abs_err={err}"
        );
    }
}

#[test]
fn wgpu_quantize_nvfp4_bf16_zero_fills_padding_rows() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_nvfp4_bf16_zero_fills_padding_rows") else {
        return;
    };
    let m_logical = 5usize;
    let m_padded = 12usize;
    let k = 64usize;
    let mut st = 0x243f6a8885a308d3u64;
    let x: Vec<u16> = (0..m_logical * k)
        .map(|_| rand_bf16(&mut st, 1.5).to_bits())
        .collect();
    let rows = rows_of(&x, m_logical, k);
    let (packed_want, scales_want) = reference(
        &rows,
        &[1.0],
        m_logical.max(1),
        m_padded,
        scale_rows(m_padded),
        k,
    );

    let mut packed_got = vec![0x77u8; m_padded * k / 2];
    let mut scales_got = vec![0x55u8; swizzled_scale_bytes(m_padded, k)];
    quantize_nvfp4_bf16(
        ctx,
        &x,
        1.0,
        &mut packed_got,
        &mut scales_got,
        m_logical,
        m_padded,
        k,
    )
    .expect("quantize_nvfp4_bf16");

    assert_bytes_equal("packed padded", &packed_got, &packed_want);
    assert_bytes_equal("scales padded", &scales_got, &scales_want);
    assert!(
        packed_got[m_logical * k / 2..].iter().all(|b| *b == 0),
        "padding rows must be zero-filled"
    );
    eprintln!(
        "wgpu_quantize_nvfp4_bf16_zero_fills_padding_rows: scale bytes {} all-zero-tail ok",
        scales_got.len()
    );
}

#[test]
fn wgpu_quantize_nvfp4_bf16_per_expert_matches_nv_quant_byte_exact() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_nvfp4_bf16_per_expert_matches_nv_quant_byte_exact")
    else {
        return;
    };
    let num_experts = 4usize;
    let m_per_expert = 128usize;
    let k = 256usize;
    let m_total = num_experts * m_per_expert;
    let globals = vec![1.5f32, 0.75, 3.0, 0.25];
    let mut st = 0xdeadbeefcafef00du64;
    let x: Vec<u16> = (0..m_total * k)
        .map(|_| rand_bf16(&mut st, 2.0).to_bits())
        .collect();
    let rows = rows_of(&x, m_total, k);
    let (packed_want, scales_want) = reference(&rows, &globals, m_per_expert, m_total, m_total, k);

    let mut packed_got = vec![0u8; m_total * k / 2];
    let mut scales_got = vec![0u8; swizzled_scale_bytes(m_total, k)];
    let offsets: Vec<i32> = (0..=num_experts)
        .map(|e| (e * m_per_expert) as i32)
        .collect();
    quantize_nvfp4_bf16_per_expert(
        ctx,
        &x,
        &globals,
        &offsets,
        &mut packed_got,
        &mut scales_got,
        num_experts,
        m_per_expert,
        k,
    )
    .expect("quantize_nvfp4_bf16_per_expert");

    assert_bytes_equal("packed per-expert", &packed_got, &packed_want);
    assert_bytes_equal("scales per-expert", &scales_got, &scales_want);

    let lin = linear_scales(&rows, &globals, m_per_expert);
    let err = max_abs_err(&rows, &packed_got, &lin, k, &globals, m_per_expert);
    eprintln!("wgpu_quantize_nvfp4_bf16_per_expert: max_abs_err={err}");
}

#[test]
fn wgpu_silu_mul_quantize_nvfp4_bf16_per_expert_matches_cpu() {
    let Some(ctx) = ctx_or_skip("wgpu_silu_mul_quantize_nvfp4_bf16_per_expert_matches_cpu") else {
        return;
    };
    let num_experts = 2usize;
    let m_per_expert = 128usize;
    let inter = 128usize;
    let m_total = num_experts * m_per_expert;
    let globals = vec![1.5f32, 0.75];
    let mut st = 0x5deece66du64;
    let gate: Vec<u16> = (0..m_total * inter)
        .map(|_| rand_bf16(&mut st, 3.0).to_bits())
        .collect();
    let up: Vec<u16> = (0..m_total * inter)
        .map(|_| rand_bf16(&mut st, 1.2).to_bits())
        .collect();
    let mut gate_up = gate.clone();
    gate_up.extend_from_slice(&up);

    let act: Vec<Vec<f32>> = (0..m_total)
        .map(|r| {
            (0..inter)
                .map(|j| {
                    let g = bf16::from_bits(gate[r * inter + j]).to_f32();
                    let u = bf16::from_bits(up[r * inter + j]).to_f32();
                    (g / (1.0 + (-g).exp())) * u
                })
                .collect()
        })
        .collect();
    let (packed_want, scales_want) =
        reference(&act, &globals, m_per_expert, m_total, m_total, inter);

    let mut packed_got = vec![0u8; m_total * inter / 2];
    let mut scales_got = vec![0u8; swizzled_scale_bytes(m_total, inter)];
    silu_mul_quantize_nvfp4_bf16_per_expert(
        ctx,
        &gate_up,
        &globals,
        &[],
        &mut packed_got,
        &mut scales_got,
        num_experts,
        m_per_expert,
        inter,
    )
    .expect("silu_mul_quantize_nvfp4_bf16_per_expert");

    let total_nibbles = packed_got.len() * 2;
    let mut off_by_one = 0usize;
    let mut worse = 0usize;
    for (a, b) in packed_got.iter().zip(packed_want.iter()) {
        for shift in [0u32, 4] {
            let an = ((*a >> shift) & 0xF) as i32;
            let bn = ((*b >> shift) & 0xF) as i32;
            if an == bn {
                continue;
            }
            if (an >> 3) == (bn >> 3) && ((an & 7) - (bn & 7)).abs() <= 1 {
                off_by_one += 1;
            } else {
                worse += 1;
            }
        }
    }
    let scale_mismatch = scales_got
        .iter()
        .zip(scales_want.iter())
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "wgpu_silu_mul_quantize_nvfp4: {off_by_one}/{total_nibbles} nibbles off-by-one, \
         {worse}/{total_nibbles} larger, {scale_mismatch}/{} scale bytes differ (wgsl exp() \
         is not bit-identical to Rust f32 exp)",
        scales_got.len()
    );
    assert_eq!(worse, 0, "silu path produced non-adjacent FP4 codes");
    assert!(
        off_by_one * 200 < total_nibbles,
        "too many off-by-one nibbles: {off_by_one}/{total_nibbles}"
    );

    let lin = linear_scales(&act, &globals, m_per_expert);
    let err = max_abs_err(&act, &packed_got, &lin, inter, &globals, m_per_expert);
    eprintln!("wgpu_silu_mul_quantize_nvfp4: max_abs_err={err}");
}

#[test]
fn wgpu_quantize_nvfp4_bf16_round_trips_through_shared_decode() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_nvfp4_bf16_round_trips_through_shared_decode")
    else {
        return;
    };
    let m = 8usize;
    let k = 32usize;
    let vals: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.5).collect();
    let x: Vec<u16> = vals.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();
    let rows = rows_of(&x, m, k);
    let mut packed = vec![0u8; m * k / 2];
    let mut scales = vec![0u8; swizzled_scale_bytes(m, k)];
    quantize_nvfp4_bf16(ctx, &x, 1.0, &mut packed, &mut scales, m, m, k).expect("quantize");

    let blocks_per_row = k / BLOCK_SIZE;
    let k_tiles = blocks_per_row.div_ceil(4);
    let mut worst = 0.0f32;
    for r in 0..m {
        for b in 0..blocks_per_row {
            let m_tile = r / 128;
            let d2 = (r / 32) % 4;
            let d3 = r % 32;
            let dst = ((m_tile * k_tiles + b / 4) * 32 + d3) * 16 + d2 * 4 + (b % 4);
            let scale_byte = scales[dst];
            let off = r * (k / 2) + b * (BLOCK_SIZE / 2);
            let deq = dequantize_block(&packed[off..off + BLOCK_SIZE / 2], scale_byte);
            assert!(decode_ue4m3(scale_byte) > 0.0, "scale must be non-zero");
            for (i, v) in deq.iter().enumerate() {
                let d = (rows[r][b * BLOCK_SIZE + i] - v).abs();
                if d > worst {
                    worst = d;
                }
            }
        }
    }
    eprintln!("wgpu_quantize_nvfp4_bf16_round_trips_through_shared_decode: max_abs_err={worst}");
    assert!(
        worst <= 1.0,
        "round-trip error {worst} exceeds one FP4 step"
    );
}

#[test]
fn wgpu_quantize_nvfp4_bf16_bad_k_is_shape_error() {
    let Some(ctx) = ctx_or_skip("wgpu_quantize_nvfp4_bf16_bad_k_is_shape_error") else {
        return;
    };
    let x = vec![0u16; 24];
    let mut packed = vec![0u8; 12];
    let mut scales = vec![0u8; 512];
    let e = quantize_nvfp4_bf16(ctx, &x, 1.0, &mut packed, &mut scales, 1, 1, 24).unwrap_err();
    assert!(
        matches!(e, nv_kernels::wgpu_backend::WgpuError::Shape(_)),
        "{e}"
    );
}
