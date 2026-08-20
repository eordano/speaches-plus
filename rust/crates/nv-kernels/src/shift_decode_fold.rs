pub const E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE: f32 = f32::from_bits(247u32 << 23);

pub const E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE: f32 = f32::from_bits(253u32 << 23);

pub const E4M3_NAN_CODES_EXCLUDED_BY_QUANTIZER_CONTRACT: [u8; 2] = [0x7f, 0xff];

pub fn e4m3_shift_decode_unscaled(byte: u8) -> f32 {
    let b = byte as u32;
    f32::from_bits(((b & 128) << 24) | ((b & 127) << 20))
}

pub fn e2m1_shift_decode_unscaled(code: u8) -> f32 {
    let n = (code & 15) as u32;
    f32::from_bits(((n & 8) << 28) | ((n & 7) << 22))
}

pub fn fold_scale_for_e4m3_shift_decode(scale: f32) -> f32 {
    let folded = scale * E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE;
    assert!(
        folded.is_finite(),
        "scale {scale:e} times 2^120 leaves f32; the e4m3 shift-decode route needs |scale| under \
         about 2^8, which holds for amax/448 row scales but not for raw magnitudes"
    );
    folded
}

pub fn fold_scale_for_e2m1_shift_decode(scale: f32) -> f32 {
    let folded = scale * E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE;
    assert!(
        folded.is_finite(),
        "scale {scale:e} times 2^126 leaves f32; the e2m1 shift-decode route needs |scale| under \
         about 2^2, so fold the block-scale times global product, never a bare decoded scale byte"
    );
    folded
}

pub fn fold_scales_for_e4m3_shift_decode(scales: &[f32]) -> Vec<f32> {
    scales
        .iter()
        .copied()
        .map(fold_scale_for_e4m3_shift_decode)
        .collect()
}

pub fn fold_scales_for_e2m1_shift_decode(scales: &[f32]) -> Vec<f32> {
    scales
        .iter()
        .copied()
        .map(fold_scale_for_e2m1_shift_decode)
        .collect()
}

pub fn e4m3_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush(byte: u8) -> bool {
    byte & 0x78 == 0 && byte & 7 != 0
}

pub fn e2m1_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush(code: u8) -> bool {
    code & 6 == 0 && code & 1 == 1
}

pub fn e4m3_host_decode_rejecting_nan_codes(byte: u8) -> f32 {
    assert!(
        byte & 0x7f != 0x7f,
        "e4m3 code {byte:#04x} is NaN; quantizers never emit 0x7f or 0xff, and the shift decode \
         maps them to a large finite value past E4M3_MAX=448, so every caller excludes them by \
         contract"
    );
    let e = ((byte >> 3) & 15) as i32;
    let m = (byte & 7) as f32;
    let mag = if e == 0 {
        m * 2f32.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f32.powi(e - 7)
    };
    if byte & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

pub fn e4m3_scale_byte_times_global_prefolded_for_e2m1_shift_decode(
    byte: u8,
    global_scale: f32,
) -> f32 {
    fold_scale_for_e2m1_shift_decode(e4m3_host_decode_rejecting_nan_codes(byte) * global_scale)
}
