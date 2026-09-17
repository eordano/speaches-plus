use crate::nvfp4::{decode_ue4m3, unswizzle_scales, BLOCK_SIZE};

pub const MARLIN_SCALE_PERM_64_WIDE_GROUPED_I_PLUS_8J: [usize; 64] = {
    let mut p = [0usize; 64];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            p[i * 8 + j] = i + 8 * j;
            j += 1;
        }
        i += 1;
    }
    p
};

pub const MARLIN_FP4_SCALE_QUAD_PERM_0_2_1_3_FITS_THE_FP8_DEQUANT_LAYOUT: [usize; 4] = [0, 2, 1, 3];

pub const MARLIN_NVFP4_SCALE_HALF_MULT_2_POW_7_SO_THE_E5M3_TOP_EXPONENT_BIT_IS_SET: f32 = 128.0;

pub const MARLIN_NVFP4_MIN_SCALE_1_OVER_64_BELOW_WHICH_THE_E5M3_BYTE_LOSES_ITS_TOP_BIT: f32 =
    1.0 / 64.0;

pub const MARLIN_NVFP4_GLOBAL_EXPONENT_BIAS_BF16_2_POW_119: f32 = 6.646139e35;

pub fn kmajor_int32_from_packed_rows(packed: &[u8], n: usize, k: usize) -> Vec<u32> {
    assert!(k.is_multiple_of(8), "k must be a multiple of 8");
    assert_eq!(
        packed.len(),
        n * k / 2,
        "packed nibbles must be [n][k/2] bytes"
    );
    let words_per_row = k / 8;
    let mut out = vec![0u32; words_per_row * n];
    for row in 0..n {
        let base = row * (k / 2);
        for w in 0..words_per_row {
            let b = &packed[base + w * 4..base + w * 4 + 4];
            out[w * n + row] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
    }
    out
}

pub struct MarlinNvfp4Scales {
    pub bytes_e5m3: Vec<u8>,
    pub folded_pow2: i32,
    pub global_scale_processed: f32,
}

pub fn marlin_scales_from_linear(
    scales_linear: &[u8],
    n: usize,
    k: usize,
    global_scale: f32,
) -> MarlinNvfp4Scales {
    let kb = k / BLOCK_SIZE;
    assert_eq!(
        scales_linear.len(),
        n * kb,
        "linear scales must be [n][k/16]"
    );
    assert!(
        n.is_multiple_of(64) || (kb * n).is_multiple_of(64),
        "scale count must tile by 64"
    );
    let mut vmin = f32::INFINITY;
    let mut vmax = 0f32;
    for &b in scales_linear {
        let v = decode_ue4m3(b);
        if v > 0.0 {
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    let mut fold = 0i32;
    if vmin.is_finite() {
        while vmin * (2f32).powi(fold)
            < MARLIN_NVFP4_MIN_SCALE_1_OVER_64_BELOW_WHICH_THE_E5M3_BYTE_LOSES_ITS_TOP_BIT
        {
            fold += 1;
        }
    }
    assert!(
        vmax * (2f32).powi(fold)
            * MARLIN_NVFP4_SCALE_HALF_MULT_2_POW_7_SO_THE_E5M3_TOP_EXPONENT_BIT_IS_SET
            < 65504.0,
        "folding 2^{fold} into the global scale overflows half for the largest block scale {vmax}"
    );
    let mut transposed = vec![0u8; kb * n];
    for row in 0..n {
        for g in 0..kb {
            transposed[g * n + row] = scales_linear[row * kb + g];
        }
    }
    let mut permuted = vec![0u8; kb * n];
    for chunk in 0..(kb * n / 64) {
        for idx in 0..64 {
            permuted[chunk * 64 + idx] =
                transposed[chunk * 64 + MARLIN_SCALE_PERM_64_WIDE_GROUPED_I_PLUS_8J[idx]];
        }
    }
    let mut quad = vec![0u8; kb * n];
    for q in 0..(kb * n / 4) {
        for j in 0..4 {
            quad[q * 4 + j] =
                permuted[q * 4 + MARLIN_FP4_SCALE_QUAD_PERM_0_2_1_3_FITS_THE_FP8_DEQUANT_LAYOUT[j]];
        }
    }
    let bytes_e5m3: Vec<u8> = quad
        .iter()
        .map(|&b| {
            let v = decode_ue4m3(b) * (2f32).powi(fold);
            if v <= 0.0 {
                return 0u8;
            }
            let h = half::f16::from_f32(
                v * MARLIN_NVFP4_SCALE_HALF_MULT_2_POW_7_SO_THE_E5M3_TOP_EXPONENT_BIT_IS_SET,
            )
            .to_bits();
            let shifted = h << 1;
            (shifted >> 8) as u8
        })
        .collect();
    for (&src, &dst) in quad.iter().zip(&bytes_e5m3) {
        assert!(
            decode_ue4m3(src) <= 0.0 || dst != 0,
            "a nonzero block scale mapped to a zero Marlin scale byte (src {src:#04x})"
        );
    }
    let global_scale_processed =
        global_scale / (2f32).powi(fold) * MARLIN_NVFP4_GLOBAL_EXPONENT_BIAS_BF16_2_POW_119;
    MarlinNvfp4Scales {
        bytes_e5m3,
        folded_pow2: fold,
        global_scale_processed,
    }
}

pub fn marlin_scales_from_swizzled(
    swizzled: &[u8],
    n: usize,
    k: usize,
    global_scale: f32,
) -> MarlinNvfp4Scales {
    let linear = unswizzle_scales(swizzled, n, k / BLOCK_SIZE);
    marlin_scales_from_linear(&linear, n, k, global_scale)
}
