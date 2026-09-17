pub const MARLIN_SCALE_PERM_SINGLE_32_WIDE_PER_CHANNEL: [usize; 32] = {
    let mut p = [0usize; 32];
    let cols = [0usize, 1, 8, 9, 16, 17, 24, 25];
    let mut i = 0;
    while i < 4 {
        let mut j = 0;
        while j < 8 {
            p[i * 8 + j] = 2 * i + cols[j];
            j += 1;
        }
        i += 1;
    }
    p
};

pub const MARLIN_FP8_SCALE_EXPONENT_BIAS_BF16_2_POW_120: f32 = 1.329228e36;

pub fn kmajor_int32_from_e4m3_rows(bytes: &[u8], n: usize, k: usize) -> Vec<u32> {
    assert!(k.is_multiple_of(4), "k must be a multiple of 4");
    assert_eq!(bytes.len(), n * k, "fp8 weight bytes must be [n][k]");
    let words_per_row = k / 4;
    let mut out = vec![0u32; words_per_row * n];
    for row in 0..n {
        let base = row * k;
        for w in 0..words_per_row {
            let b = &bytes[base + w * 4..base + w * 4 + 4];
            out[w * n + row] = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        }
    }
    out
}

pub fn marlin_per_channel_scales_bf16(scale_rows: &[f32], n: usize) -> Vec<u16> {
    assert_eq!(scale_rows.len(), n, "one scale per output channel");
    assert!(
        n.is_multiple_of(32),
        "n must be a multiple of 32 for the per-channel scale permutation"
    );
    let mut out = vec![0u16; n];
    for chunk in 0..(n / 32) {
        for idx in 0..32 {
            let src = chunk * 32 + MARLIN_SCALE_PERM_SINGLE_32_WIDE_PER_CHANNEL[idx];
            let v = scale_rows[src] * MARLIN_FP8_SCALE_EXPONENT_BIAS_BF16_2_POW_120;
            assert!(
                v.is_finite(),
                "fp8 marlin scale overflowed bf16 after the 2^120 exponent fold: {}",
                scale_rows[src]
            );
            out[chunk * 32 + idx] = half::bf16::from_f32(v).to_bits();
        }
    }
    out
}
