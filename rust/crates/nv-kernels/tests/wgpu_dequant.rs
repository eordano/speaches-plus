#![cfg(feature = "wgpu")]

mod common;
use common::ctx_or_skip;
use common::lcg;
use common::lcg_f32;
use nv_kernels::wgpu_backend::dequant;
use nv_kernels::wgpu_backend::device::WgpuContext;
use nv_quant::nvfp4;

fn e4m3_decode_ref(b: u8) -> f32 {
    let s = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let e = (b >> 3) & 0xf;
    let m = (b & 7) as f32;
    if (b & 0x7f) == 0x7f {
        return f32::NAN;
    }
    if e == 0 {
        s * m * (1.0 / 8.0) * 2f32.powi(-6)
    } else {
        s * (1.0 + m / 8.0) * 2f32.powi(e as i32 - 7)
    }
}

fn e5m2_decode_ref(b: u8) -> f32 {
    let bits = (b as u16) << 8;
    half::f16::from_bits(bits).to_f32()
}

#[test]
fn wgpu_e2m1_decode_is_exact_for_all_sixteen_codes() {
    let Some(ctx) = ctx_or_skip("wgpu_e2m1_decode_is_exact_for_all_sixteen_codes") else {
        return;
    };
    let codes: Vec<u32> = (0u32..16).collect();
    let want: Vec<f32> = (0u8..16).map(nvfp4::decode_e2m1).collect();

    let table = dequant::gpu_decode_e2m1(ctx, &codes).expect("table decode");
    let arith = dequant::gpu_decode_e2m1_arith(ctx, &codes).expect("arith decode");

    for i in 0..16 {
        assert_eq!(table[i], want[i], "table decode of nibble {i}");
        assert_eq!(arith[i], want[i], "branchless decode of nibble {i}");
        if want[i] != 0.0 {
            assert_eq!(
                table[i].is_sign_negative(),
                want[i].is_sign_negative(),
                "sign of nibble {i}: got {} want {}",
                table[i],
                want[i]
            );
        }
    }
    assert_eq!(table[7], 6.0);
    assert_eq!(table[15], -6.0);
    assert_eq!(table[1], 0.5);
    assert_eq!(table[9], -0.5);
}

#[test]
fn wgpu_ue4m3_decode_matches_nv_quant_for_all_256_bytes() {
    let Some(ctx) = ctx_or_skip("wgpu_ue4m3_decode_matches_nv_quant_for_all_256_bytes") else {
        return;
    };
    let codes: Vec<u32> = (0u32..256).collect();
    let got = dequant::gpu_decode_ue4m3(ctx, &codes).expect("ue4m3 decode");
    for b in 0u32..256 {
        let want = nvfp4::decode_ue4m3(b as u8);
        assert_eq!(
            got[b as usize], want,
            "ue4m3 byte {b:#04x}: got {} want {}",
            got[b as usize], want
        );
    }
    assert_eq!(got[0], 0.0);
    assert_eq!(got[7], 7.0 * 0.001953125);
    assert_eq!(got[0x7e], 448.0);
}

#[test]
fn wgpu_signed_e4m3_decode_matches_the_kv_cache_oracle() {
    let Some(ctx) = ctx_or_skip("wgpu_signed_e4m3_decode_matches_the_kv_cache_oracle") else {
        return;
    };
    let codes: Vec<u32> = (0u32..256).collect();
    let got = dequant::gpu_decode_e4m3(ctx, &codes).expect("e4m3 decode");
    for b in 0u32..256 {
        let want = e4m3_decode_ref(b as u8);
        let g = got[b as usize];
        if want.is_nan() {
            assert!(
                g.is_nan(),
                "e4m3 byte {b:#04x} should decode to NaN, got {g}"
            );
            continue;
        }
        assert_eq!(g, want, "e4m3 byte {b:#04x}: got {g} want {want}");
    }
    assert_eq!(got[0x7e], 448.0);
    assert_eq!(got[0xfe], -448.0);
}

#[test]
fn wgpu_e5m2_decode_matches_the_fp16_high_half() {
    let Some(ctx) = ctx_or_skip("wgpu_e5m2_decode_matches_the_fp16_high_half") else {
        return;
    };
    let codes: Vec<u32> = (0u32..256).collect();
    let got = dequant::gpu_decode_e5m2(ctx, &codes).expect("e5m2 decode");
    for b in 0u32..256 {
        let want = e5m2_decode_ref(b as u8);
        let g = got[b as usize];
        if want.is_nan() {
            assert!(
                g.is_nan(),
                "e5m2 byte {b:#04x} should decode to NaN, got {g}"
            );
            continue;
        }
        assert_eq!(g, want, "e5m2 byte {b:#04x}: got {g} want {want}");
    }
}

#[test]
fn wgpu_bf16_round_trips_losslessly() {
    let Some(ctx) = ctx_or_skip("wgpu_bf16_round_trips_losslessly") else {
        return;
    };
    let mut state = 0x1234_5678_9abc_def0u64;
    let mut values: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 6.0, 448.0, 1e-8, 3.4e38];
    for _ in 0..1024 {
        values.push(lcg_f32(&mut state) * 32.0);
    }
    let bf: Vec<half::bf16> = values.iter().map(|v| half::bf16::from_f32(*v)).collect();

    let bits: Vec<u32> = bf.iter().map(|b| b.to_bits() as u32).collect();
    let decoded = dequant::gpu_decode_bf16(ctx, &bits).expect("bf16 decode");
    for (i, b) in bf.iter().enumerate() {
        assert_eq!(decoded[i], b.to_f32(), "bf16 decode of value {i}");
    }

    let encoded = dequant::gpu_encode_bf16(ctx, &values).expect("bf16 encode");
    for (i, b) in bf.iter().enumerate() {
        assert_eq!(
            encoded[i] & 0xffff,
            b.to_bits() as u32,
            "bf16 round-to-nearest-even encode of {}",
            values[i]
        );
    }

    let words: Vec<u32> = bits
        .chunks(2)
        .map(|c| c[0] | (c.get(1).copied().unwrap_or(0) << 16))
        .collect();
    let pairs = dequant::gpu_decode_bf16_pairs(ctx, &words).expect("bf16 pair decode");
    for (i, b) in bf.iter().enumerate() {
        assert_eq!(pairs[i], b.to_f32(), "bf16 packed-pair decode of value {i}");
    }
}

#[test]
fn wgpu_nvfp4_block_path_matches_nv_quant_dequantize() {
    let Some(ctx) = ctx_or_skip("wgpu_nvfp4_block_path_matches_nv_quant_dequantize") else {
        return;
    };
    let mut state = 0xdead_beef_cafe_f00du64;
    let rows_n = 5usize;
    let cols = 64usize;
    let rows: Vec<Vec<f32>> = (0..rows_n)
        .map(|_| (0..cols).map(|_| lcg_f32(&mut state) * 4.0).collect())
        .collect();

    for global in [1.0f32, 0.25, 3.0] {
        let tensor = nvfp4::Nvfp4Tensor::quantize_rows_with_global(&rows, global);
        let want: Vec<f32> = tensor.dequantize().into_iter().flatten().collect();
        let got =
            dequant::gpu_dequantize_nvfp4(ctx, &tensor.data, &tensor.scales, rows_n * cols, 1.0)
                .expect("nvfp4 dequantize");
        assert_eq!(got.len(), want.len());
        for i in 0..want.len() {
            assert_eq!(
                got[i], want[i],
                "global={global} element {i}: got {} want {}",
                got[i], want[i]
            );
        }

        let inv = 1.0f32 / global;
        let scaled =
            dequant::gpu_dequantize_nvfp4(ctx, &tensor.data, &tensor.scales, rows_n * cols, inv)
                .expect("nvfp4 dequantize with global");
        for i in 0..want.len() {
            assert_eq!(
                scaled[i],
                want[i] * inv,
                "global={global} scaled element {i}"
            );
        }
    }
}

#[test]
fn wgpu_nvfp4_scale_swizzle_index_matches_nv_quant() {
    let Some(ctx) = ctx_or_skip("wgpu_nvfp4_scale_swizzle_index_matches_nv_quant") else {
        return;
    };
    for (rows, k_blocks) in [(4usize, 4usize), (32, 6), (160, 9)] {
        let linear: Vec<u8> = (0..rows * k_blocks).map(|i| (i % 251) as u8).collect();
        let swizzled = nvfp4::swizzle_scales(&linear, rows, k_blocks);
        let idx =
            dequant::gpu_nvfp4_scale_swizzle_index(ctx, rows, k_blocks).expect("swizzle index");
        for m in 0..rows {
            for kb in 0..k_blocks {
                let dst = idx[m * k_blocks + kb] as usize;
                assert!(
                    dst < swizzled.len(),
                    "rows={rows} k_blocks={k_blocks} m={m} kb={kb}: dst {dst} out of range"
                );
                assert_eq!(
                    swizzled[dst],
                    linear[m * k_blocks + kb],
                    "rows={rows} k_blocks={k_blocks} m={m} kb={kb}"
                );
            }
        }
    }
}

#[test]
fn wgpu_int4_u4b8_decode_matches_the_cpu_reference() {
    let Some(ctx) = ctx_or_skip("wgpu_int4_u4b8_decode_matches_the_cpu_reference") else {
        return;
    };
    let n = 256usize;
    let mut state = 0x0f0f_0f0f_1234_5678u64;
    let packed: Vec<u32> = (0..n / 8).map(|_| lcg(&mut state) as u32).collect();
    let scale = 0.125f32;
    let got = dequant::gpu_decode_int4_group(ctx, &packed, n, scale, 8.0).expect("int4 decode");
    for i in 0..n {
        let word = packed[i / 8];
        let q = ((word >> (4 * (i % 8))) & 0xf) as i32 - 8;
        let want = q as f32 * scale;
        assert_eq!(got[i], want, "int4 element {i}");
    }

    let unsigned =
        dequant::gpu_decode_int4_group(ctx, &packed, n, scale, 0.0).expect("int4 zp=0 decode");
    for i in 0..n {
        let word = packed[i / 8];
        let q = ((word >> (4 * (i % 8))) & 0xf) as f32;
        assert_eq!(unsigned[i], q * scale, "int4 zp=0 element {i}");
    }
}

#[test]
fn wgpu_capability_report_is_populated() {
    let Some(ctx) = ctx_or_skip("wgpu_capability_report_is_populated") else {
        return;
    };
    let caps = nv_kernels::wgpu_backend::kernels::hello::capability(ctx).expect("capability");
    assert!(!caps.adapter_name.is_empty());
    assert!(!caps.backend.is_empty());
    assert!(caps.max_compute_workgroups_per_dimension >= 65535);
    eprintln!(
        "coop_mat={} shader_f16={} f16_in_f32={} subgroup={}",
        caps.cooperative_matrix, caps.shader_f16, caps.f16_in_f32, caps.subgroup
    );
}
