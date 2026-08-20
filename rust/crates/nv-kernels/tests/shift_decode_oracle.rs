use nv_kernels::shift_decode_fold::{
    e2m1_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush,
    e2m1_shift_decode_unscaled, e4m3_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush,
    e4m3_host_decode_rejecting_nan_codes,
    e4m3_scale_byte_times_global_prefolded_for_e2m1_shift_decode, e4m3_shift_decode_unscaled,
    fold_scale_for_e2m1_shift_decode, fold_scale_for_e4m3_shift_decode,
    E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE, E4M3_NAN_CODES_EXCLUDED_BY_QUANTIZER_CONTRACT,
    E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE,
};
mod common;
use common::reference_e2m1 as reference_e2m1_from_format_definition;
use common::reference_e4m3 as reference_e4m3_from_format_definition;

const PRODUCT_PARITY_SCALES_STAY_UNDER_THE_2POW2_E2M1_FOLD_CAP: [f32; 6] = [
    1.0,
    0.5,
    0.0078125,
    3.0517578e-5,
    0.371,
    1.0 / 448.0,
];

#[test]
fn the_fold_constants_are_the_exact_powers_of_two_the_names_claim() {
    assert_eq!(
        E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE.to_bits(),
        2f32.powi(120).to_bits()
    );
    assert_eq!(
        E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE.to_bits(),
        2f32.powi(126).to_bits()
    );
}

#[test]
fn every_e4m3_code_shift_decodes_exactly_2pow120_below_the_reference() {
    for b in 0u32..256 {
        let byte = b as u8;
        if byte & 0x7f == 0x7f {
            assert!(
                E4M3_NAN_CODES_EXCLUDED_BY_QUANTIZER_CONTRACT.contains(&byte),
                "the NaN mask 0x7f matched {byte:#04x} outside the documented pair"
            );
            let escaped = e4m3_shift_decode_unscaled(byte) * E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE;
            assert!(
                escaped.is_finite() && escaped.abs() > 448.0,
                "e4m3 codes 0x7f and 0xff are NaN in the format; quantizers never emit them, and \
                 the shift decode turns them into the finite value {escaped} beyond E4M3_MAX=448 \
                 rather than NaN, so callers exclude them by contract and this oracle skips them"
            );
            continue;
        }
        let want = reference_e4m3_from_format_definition(byte);
        let got = e4m3_shift_decode_unscaled(byte) * E4M3_SHIFT_DECODE_LANDS_2POW120_BELOW_TRUE;
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "e4m3 code {byte:#04x}: shift decode times 2^120 gave {got:e}, format reference gave \
             {want:e}; the fold must be bit-exact for normals and subnormals or the route is \
             unusable"
        );
        assert_eq!(
            e4m3_host_decode_rejecting_nan_codes(byte).to_bits(),
            want.to_bits(),
            "e4m3 code {byte:#04x}: the host load-path decoder drifted from the format definition"
        );
    }
}

#[test]
fn every_e2m1_code_shift_decodes_exactly_2pow126_below_the_reference() {
    for c in 0u32..16 {
        let code = c as u8;
        let want = reference_e2m1_from_format_definition(code);
        let got = e2m1_shift_decode_unscaled(code) * E2M1_SHIFT_DECODE_LANDS_2POW126_BELOW_TRUE;
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "e2m1 code {code:#03x}: shift decode times 2^126 gave {got:e}, format reference gave \
             {want:e}"
        );
    }
}

#[test]
fn shift_decode_times_prefolded_scale_is_bitwise_equal_to_reference_times_scale() {
    for scale in PRODUCT_PARITY_SCALES_STAY_UNDER_THE_2POW2_E2M1_FOLD_CAP {
        let folded_e4m3 = fold_scale_for_e4m3_shift_decode(scale);
        for b in 0u32..256 {
            let byte = b as u8;
            if byte & 0x7f == 0x7f {
                continue;
            }
            let want = reference_e4m3_from_format_definition(byte) * scale;
            let got = e4m3_shift_decode_unscaled(byte) * folded_e4m3;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "e4m3 code {byte:#04x} scale {scale:e}: shifted product {got:e} vs reference \
                 product {want:e}"
            );
        }
        let folded_e2m1 = fold_scale_for_e2m1_shift_decode(scale);
        for c in 0u32..16 {
            let code = c as u8;
            let want = reference_e2m1_from_format_definition(code) * scale;
            let got = e2m1_shift_decode_unscaled(code) * folded_e2m1;
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "e2m1 code {code:#03x} scale {scale:e}: shifted product {got:e} vs reference \
                 product {want:e}"
            );
        }
    }
}

#[test]
fn the_scale_byte_load_helper_matches_decode_then_fold_composition() {
    for b in 0u32..256 {
        let byte = b as u8;
        if byte & 0x7f == 0x7f {
            continue;
        }
        let value = reference_e4m3_from_format_definition(byte);
        if !(value.abs() < 2.0) {
            continue;
        }
        let global = 0.03125f32;
        let got = e4m3_scale_byte_times_global_prefolded_for_e2m1_shift_decode(byte, global);
        let want = fold_scale_for_e2m1_shift_decode(value * global);
        assert_eq!(got.to_bits(), want.to_bits(), "scale byte {byte:#04x}");
    }
}

#[test]
fn the_subnormal_band_predicates_name_exactly_the_codes_whose_unscaled_form_is_subnormal() {
    for b in 0u32..256 {
        let byte = b as u8;
        let u = e4m3_shift_decode_unscaled(byte);
        let is_subnormal = u != 0.0 && u.abs() < f32::MIN_POSITIVE;
        assert_eq!(
            e4m3_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush(byte),
            is_subnormal,
            "e4m3 code {byte:#04x}: the hazard predicate must name exactly the codes an \
             FTZ multiplier would flush"
        );
    }
    for c in 0u32..16 {
        let code = c as u8;
        let u = e2m1_shift_decode_unscaled(code);
        let is_subnormal = u != 0.0 && u.abs() < f32::MIN_POSITIVE;
        assert_eq!(
            e2m1_code_shift_decodes_to_f32_subnormal_which_ftz_multipliers_flush(code),
            is_subnormal,
            "e2m1 code {code:#03x}: the hazard predicate must name exactly the codes an \
             FTZ multiplier would flush"
        );
    }
}

#[test]
#[should_panic(expected = "leaves f32")]
fn folding_a_bare_decoded_scale_byte_magnitude_for_e2m1_overflows_and_says_so() {
    fold_scale_for_e2m1_shift_decode(448.0);
}

#[test]
#[should_panic(expected = "leaves f32")]
fn folding_a_raw_weight_magnitude_for_e4m3_overflows_and_says_so() {
    fold_scale_for_e4m3_shift_decode(1024.0);
}

#[test]
#[should_panic(expected = "is NaN")]
fn the_host_decoder_refuses_the_nan_codes_the_shift_route_cannot_represent() {
    e4m3_host_decode_rejecting_nan_codes(0x7f);
}

#[test]
fn the_wgsl_shift_decoders_carry_the_exact_bit_formulas_this_oracle_verified() {
    let src = include_str!("../wgsl/dequant.wgsl");
    for needle in [
        "fn e4m3_shift_decode_scale_must_carry_2pow120",
        "((b & 128u) << 24u) | ((b & 127u) << 20u)",
        "fn e2m1_shift_decode_scale_must_carry_2pow126",
        "((n & 8u) << 28u) | ((n & 7u) << 22u)",
    ] {
        assert!(
            src.contains(needle),
            "dequant.wgsl no longer contains {needle:?}; this oracle pins the host bit math to \
             the shader text, so either restore the formula or re-verify the replacement here"
        );
    }
}
