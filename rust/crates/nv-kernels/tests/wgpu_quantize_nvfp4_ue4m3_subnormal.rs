#![cfg(feature = "wgpu")]

mod common;
use common::wgpu_allow_skip;
use half::bf16;
use nv_kernels::wgpu_backend::dequant::bytes_to_words;
use nv_kernels::wgpu_backend::device::{shared_or_reason, WgpuContext};
use nv_kernels::wgpu_backend::kernels::gemv_nvfp4;
use nv_quant::nvfp4::{
    decode_e2m1, quantize_block_with_global, swizzle_scales, unpack_e2m1_pair, Nvfp4Tensor,
    BLOCK_SIZE,
};

fn ctx_or_skip(what: &str) -> Option<&'static WgpuContext> {
    match shared_or_reason() {
        Ok(ctx) => {
            let q = ctx.qualify();
            if !q.qualified {
                if !wgpu_allow_skip() {
                    panic!(
                        "{what}: wgpu adapter not qualified: {:?}. This gate refuses to \
                         report success without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to \
                         skip on purpose.",
                        q.reason
                    );
                }
                eprintln!("{what}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) not qualified");
                return None;
            }
            eprintln!("{what}: {}", ctx.summary());
            Some(ctx)
        }
        Err(reason) => {
            if !wgpu_allow_skip() {
                panic!(
                    "{what}: no wgpu adapter: {reason}. This gate refuses to report success \
                     without running; set NV_KERNELS_WGPU_ALLOW_SKIP=1 to skip on purpose."
                );
            }
            eprintln!("{what}: SKIP (NV_KERNELS_WGPU_ALLOW_SKIP=1) no adapter: {reason}");
            None
        }
    }
}

fn hw_ue4m3_value(byte: u8) -> Option<f64> {
    let e = ((byte >> 3) & 0x0f) as i32;
    let m = (byte & 0x07) as f64;
    if e == 0x0f && (byte & 0x07) == 0x07 {
        return None;
    }
    if e == 0 {
        Some(m * (-9f64).exp2())
    } else {
        Some((1.0 + m / 8.0) * ((e - 7) as f64).exp2())
    }
}

fn hw_nearest_code(target: f64) -> u8 {
    let mut best = 0u8;
    let mut best_d = f64::INFINITY;
    for b in 0u16..0x80 {
        let Some(v) = hw_ue4m3_value(b as u8) else {
            continue;
        };
        let d = (v - target).abs();
        if d < best_d {
            best_d = d;
            best = b as u8;
        }
    }
    best
}

fn swizzled_scale_dst(m: usize, kb: usize, k_blocks: usize) -> usize {
    let k_tiles = k_blocks.div_ceil(4);
    ((m / 128 * k_tiles + kb / 4) * 32 + m % 32) * 16 + ((m / 32) % 4) * 4 + kb % 4
}

fn f64_gemv_oracle(
    w_packed: &[u8],
    w_scales_swizzled: &[u8],
    x_packed: &[u8],
    x_scales: &[u8],
    alpha: f64,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let k_blocks = k / BLOCK_SIZE;
    let row_bytes = k / 2;
    let mut out = vec![0f64; n];
    for (row, slot) in out.iter_mut().enumerate() {
        let mut acc = 0f64;
        for kb in (0..k_blocks).rev() {
            let ws = hw_ue4m3_value(w_scales_swizzled[swizzled_scale_dst(row, kb, k_blocks)])
                .expect("weight scale byte is the NaN code");
            let xs = hw_ue4m3_value(x_scales[kb]).expect("activation scale byte is the NaN code");
            let w_off = row * row_bytes + kb * 8;
            let x_off = kb * 8;
            let mut dot = 0f64;
            for i in 0..8 {
                let (w_lo, w_hi) = unpack_e2m1_pair(w_packed[w_off + i]);
                let (x_lo, x_hi) = unpack_e2m1_pair(x_packed[x_off + i]);
                dot += decode_e2m1(w_lo) as f64 * decode_e2m1(x_lo) as f64
                    + decode_e2m1(w_hi) as f64 * decode_e2m1(x_hi) as f64;
            }
            acc += dot * ws * xs;
        }
        *slot = acc * alpha;
    }
    out
}

fn amax_for_subnormal_code(m: u8) -> f32 {
    6.0 * m as f32 / 512.0
}

fn subnormal_block(m: u8) -> Vec<f32> {
    let a = amax_for_subnormal_code(m);
    let t = [
        1.0f32, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 1.0, -0.5, 0.25, -0.125, 0.5, -1.0,
        0.125, 0.0,
    ];
    t.iter().map(|f| a * f).collect()
}

#[test]
fn checkpoint_subnormal_weight_scales_decode_by_the_e4m3_definition() {
    let Some(ctx) = ctx_or_skip("checkpoint_subnormal_weight_scales_decode_by_the_e4m3_definition")
    else {
        return;
    };
    let n = 4usize;
    let k = 64usize;
    let k_blocks = k / BLOCK_SIZE;
    let codes = [1u8, 3, 5, 7];

    let rows: Vec<Vec<f32>> = (0..n)
        .map(|r| {
            (0..k_blocks)
                .flat_map(|b| subnormal_block(codes[(r + b) % codes.len()]))
                .collect()
        })
        .collect();

    let w = Nvfp4Tensor::quantize_rows_with_global(&rows, 1.0);

    for (i, b) in w.scales.iter().enumerate() {
        assert!(
            (1..=7).contains(b),
            "fixture no longer reaches the ue4m3 subnormal band: scale[{i}] = {b:#04x}"
        );
    }
    for (r, row) in rows.iter().enumerate() {
        let amax = row.iter().fold(0f32, |a, v| a.max(v.abs()));
        assert!(amax > 0.0, "row {r} is degenerate (all zero)");
    }

    let w_scales_swizzled = swizzle_scales(&w.scales, n, k_blocks);

    let grid = [
        6.0f32, 3.0, 1.5, -6.0, 4.0, -3.0, 2.0, -1.5, 0.5, -0.5, 1.0, -1.0, 6.0, -4.0, 3.0, 2.0,
    ];
    let x_row: Vec<f32> = (0..k).map(|j| grid[j % 16]).collect();
    let mut x_packed = Vec::new();
    let mut x_scales = Vec::new();
    for block in x_row.chunks(BLOCK_SIZE) {
        let (p, s) = quantize_block_with_global(block, 1.0);
        x_packed.extend_from_slice(&p);
        x_scales.push(s);
    }
    for (i, b) in x_scales.iter().enumerate() {
        assert!(
            *b >= 8,
            "activation block {i} was meant to stay in the ue4m3 normal band, got {b:#04x}"
        );
    }

    let mut y = vec![0u16; n];
    gemv_nvfp4::nvfp4_gemv_bf16(
        ctx,
        &bytes_to_words(&w.data),
        &w_scales_swizzled,
        &bytes_to_words(&x_packed),
        &x_scales,
        1.0,
        &mut y,
        n,
        k,
    )
    .expect("gpu gemv");

    let oracle = f64_gemv_oracle(&w.data, &w_scales_swizzled, &x_packed, &x_scales, 1.0, n, k);
    assert!(
        oracle.iter().any(|v| v.abs() > 1e-6),
        "oracle is degenerate: every row is ~0, so agreement proves nothing"
    );

    let mut worst_ulp = 0i32;
    let mut worst_ratio = 1.0f64;
    for (row, want) in oracle.iter().enumerate() {
        let want_bits = bf16::from_f32(*want as f32).to_bits();
        let got = bf16::from_bits(y[row]).to_f32() as f64;
        let ulp = (y[row] as i32 - want_bits as i32).abs();
        worst_ulp = worst_ulp.max(ulp);
        if want.abs() > 1e-12 {
            let r = (got / want).abs();
            if (r - 1.0).abs() > (worst_ratio - 1.0).abs() {
                worst_ratio = r;
            }
        }
        eprintln!("  row {row}: gpu={got:.9} f64_oracle={want:.9} bf16_ulp={ulp}");
    }
    eprintln!(
        "checkpoint_subnormal_weight_scales: worst_bf16_ulp={worst_ulp} worst_gpu/oracle={worst_ratio:.6}"
    );
    assert!(
        worst_ulp <= 1,
        "GEMV disagrees with the e4m3 definition by {worst_ulp} bf16 ulp \
         (worst gpu/oracle ratio {worst_ratio:.6}) on weight scale bytes 0x01-0x07, \
         which a checkpoint emits for any block whose scale falls below 2^-6"
    );
}

#[test]
fn row_quantizer_picks_the_nearest_ue4m3_code_in_the_subnormal_band() {
    let Some(ctx) = ctx_or_skip("row_quantizer_picks_the_nearest_ue4m3_code_in_the_subnormal_band")
    else {
        return;
    };
    let codes = [1u8, 2, 3, 4, 5, 6, 7];
    let k = BLOCK_SIZE * codes.len();
    let row: Vec<f32> = codes.iter().flat_map(|m| subnormal_block(*m)).collect();
    let x_bits: Vec<u16> = row.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();

    let stored_global = 1.0f32;
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

    let packed_bytes: Vec<u8> = (0..k / 2)
        .map(|i| ((packed_out[i / 4] >> (8 * (i % 4))) & 0xff) as u8)
        .collect();

    let mut bad = Vec::new();
    for (b, block) in row.chunks(BLOCK_SIZE).enumerate() {
        let amax = block.iter().fold(0f32, |a, v| a.max(v.abs())) as f64;
        let target = stored_global as f64 * amax / 6.0;
        assert!(
            target > 0.0 && target < (-6f64).exp2(),
            "block {b} target {target:e} is not in the ue4m3 subnormal band"
        );
        let want = hw_nearest_code(target);
        assert!(
            (1..=7).contains(&want),
            "block {b}: f64 nearest code {want:#04x} is not a subnormal code"
        );
        let got = scales_out[b];
        eprintln!(
            "  block {b}: target={target:e} nearest_code={want:#04x}({:e}) gpu={got:#04x}({:?})",
            hw_ue4m3_value(want).unwrap(),
            hw_ue4m3_value(got)
        );
        if got != want {
            bad.push((b, target, want, got));
        }

        let sd = hw_ue4m3_value(got).expect("emitted the NaN code");
        assert!(sd > 0.0, "block {b}: scale byte {got:#04x} decodes to zero");
        let mut worst = 0f64;
        for (i, want_v) in block.iter().enumerate() {
            let byte = packed_bytes[b * (BLOCK_SIZE / 2) + i / 2];
            let (lo, hi) = unpack_e2m1_pair(byte);
            let nib = if i % 2 == 0 { lo } else { hi };
            let got_v = decode_e2m1(nib) as f64 * sd / stored_global as f64;
            worst = worst.max((got_v - *want_v as f64).abs());
        }
        assert!(
            worst <= sd * 1.001,
            "block {b}: round trip through the emitted scale byte {got:#04x} is off by \
             {worst:e}, more than the one e2m1 step ({:e}) it can cost -- q_scale_parts \
             and q_encode_scale disagree about the subnormal codes",
            sd
        );
    }
    assert!(
        bad.is_empty(),
        "{}/{} blocks: the row/activation quantizer did not emit the nearest \
         ue4m3 code below 2^-6. First: block {} target {:e} nearest {:#04x} got {:#04x}",
        bad.len(),
        k / BLOCK_SIZE,
        bad[0].0,
        bad[0].1,
        bad[0].2,
        bad[0].3
    );
}

#[test]
fn normal_band_control_is_unaffected() {
    let Some(ctx) = ctx_or_skip("normal_band_control_is_unaffected") else {
        return;
    };
    let k = BLOCK_SIZE * 8;
    let row: Vec<f32> = (0..k)
        .map(|j| {
            let base = [1.0f32, -0.75, 0.5, 0.25, -1.0, 0.125, 0.625, -0.375][j % 8];
            base * (1u32 << (j / BLOCK_SIZE)) as f32
        })
        .collect();
    let x_bits: Vec<u16> = row.iter().map(|v| bf16::from_f32(*v).to_bits()).collect();

    let mut packed_out = vec![0u32; k / 8];
    let mut scales_out = vec![0u8; k / BLOCK_SIZE];
    gemv_nvfp4::nvfp4_quantize_row_bf16(ctx, &x_bits, 1.0, &mut packed_out, &mut scales_out, k)
        .expect("gpu quantize");

    for (b, block) in row.chunks(BLOCK_SIZE).enumerate() {
        let amax = block.iter().fold(0f32, |a, v| a.max(v.abs())) as f64;
        let target = amax / 6.0;
        assert!(
            target >= (-6f64).exp2(),
            "control block {b} drifted into the subnormal band ({target:e}); it is \
             meant to prove the normal band is untouched"
        );
        let want = hw_nearest_code(target);
        assert_eq!(
            scales_out[b], want,
            "control block {b}: normal-band code changed (target {target:e})"
        );
    }
    eprintln!(
        "normal_band_control_is_unaffected: {} blocks ok",
        k / BLOCK_SIZE
    );
}
