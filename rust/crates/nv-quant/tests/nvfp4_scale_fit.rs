use nv_quant::nvfp4::{
    block_reconstruction_sq_error, closed_form_scale_byte, decode_ue4m3,
    quantize_block_with_global, quantize_block_with_global_closed_form, BLOCK_SIZE,
    CLOSED_FORM_SCALE_STEPS_ABOVE_NEAREST_BECAUSE_E2M1_STEPS_BY_HALF_BELOW_TWO_AND_BY_TWO_ABOVE_FOUR_SO_PARKING_THE_BLOCK_AMAX_ON_SIX_IS_THE_WORST_PLACE_FOR_IT,
    CLOSED_FORM_SCALE_STEPS_BELOW_NEAREST_BECAUSE_CLIPPING_THE_BLOCK_AMAX_BUYS_INTERIOR_RESOLUTION,
    UE4M3_DECODE_IS_MONOTONIC_IN_THE_BYTE_SO_ONE_GRID_STEP_IS_PLUS_ONE_BYTE,
};

const BLOCKS_PER_ARM: usize = 4096;

fn lcg(state: &mut u64) -> f32 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 40) as f32 / 16777216.0) * 2.0 - 1.0
}

fn gaussian_block(state: &mut u64, sigma: f32) -> Vec<f32> {
    (0..BLOCK_SIZE)
        .map(|_| {
            let u: f32 = (lcg(state) + lcg(state) + lcg(state) + lcg(state)) * 0.5;
            u * sigma
        })
        .collect()
}

fn brute_force_best_byte(values: &[f32], stored_global: f32) -> (u8, f64) {
    let (lo, hi) = UE4M3_DECODE_IS_MONOTONIC_IN_THE_BYTE_SO_ONE_GRID_STEP_IS_PLUS_ONE_BYTE;
    let mut best = lo;
    let mut best_err = f64::INFINITY;
    for byte in lo..=hi {
        let err = block_reconstruction_sq_error(values, byte, stored_global);
        if err < best_err {
            best_err = err;
            best = byte;
        }
    }
    (best, best_err)
}

#[test]
fn ue4m3_decode_is_monotonic_over_the_searchable_byte_range() {
    let (lo, hi) = UE4M3_DECODE_IS_MONOTONIC_IN_THE_BYTE_SO_ONE_GRID_STEP_IS_PLUS_ONE_BYTE;
    for byte in lo..hi {
        assert!(
            decode_ue4m3(byte) < decode_ue4m3(byte + 1),
            "ue4m3 decode is not monotonic at byte {byte:#04x} ({} vs {}); the closed-form \
             scale search walks the alphabet by +/-1 byte and assumes that IS the sorted grid",
            decode_ue4m3(byte),
            decode_ue4m3(byte + 1)
        );
    }
}

#[test]
fn closed_form_scale_byte_is_the_exhaustive_minimum_over_the_whole_ue4m3_alphabet() {
    let mut state = 0x5eed_1234_9abc_def0u64;
    let mut window_hits = 0usize;
    for i in 0..BLOCKS_PER_ARM {
        let sigma = [0.002f32, 0.02, 0.2, 2.0][i % 4];
        let mut block = gaussian_block(&mut state, sigma);
        if i % 7 == 0 {
            block[i % BLOCK_SIZE] *= 12.0;
        }
        let amax = block.iter().fold(0f32, |a, b| a.max(b.abs()));
        let stored_global = 448.0 * 6.0 / (amax * 4.0);
        let fitted = closed_form_scale_byte(&block, stored_global);
        let (brute, brute_err) = brute_force_best_byte(&block, stored_global);
        let fitted_err = block_reconstruction_sq_error(&block, fitted, stored_global);
        assert!(
            fitted_err <= brute_err * (1.0 + 1e-12),
            "block {i}: windowed fit byte {fitted:#04x} err {fitted_err:e} lost to exhaustive \
             byte {brute:#04x} err {brute_err:e}; widen \
             CLOSED_FORM_SCALE_STEPS_{{BELOW,ABOVE}}_NEAREST ({}, {})",
            CLOSED_FORM_SCALE_STEPS_BELOW_NEAREST_BECAUSE_CLIPPING_THE_BLOCK_AMAX_BUYS_INTERIOR_RESOLUTION,
            CLOSED_FORM_SCALE_STEPS_ABOVE_NEAREST_BECAUSE_E2M1_STEPS_BY_HALF_BELOW_TWO_AND_BY_TWO_ABOVE_FOUR_SO_PARKING_THE_BLOCK_AMAX_ON_SIX_IS_THE_WORST_PLACE_FOR_IT
        );
        if fitted != brute {
            window_hits += 1;
        }
    }
    assert_eq!(
        window_hits, 0,
        "the windowed search picked a different byte than exhaustive search on {window_hits} of \
         {BLOCKS_PER_ARM} blocks; ties are allowed only when the errors are equal"
    );
}

#[test]
fn closed_form_beats_round_to_nearest_amax_on_weight_shaped_blocks() {
    let mut state = 0xfeed_face_0bad_c0dEu64;
    let mut rtn_total = 0.0f64;
    let mut fit_total = 0.0f64;
    let mut regressions = 0usize;
    for i in 0..BLOCKS_PER_ARM {
        let block = gaussian_block(&mut state, 0.02);
        let amax = block.iter().fold(0f32, |a, b| a.max(b.abs()));
        let stored_global = 448.0 * 6.0 / (amax * 3.0);
        let (_, rtn_byte) = quantize_block_with_global(&block, stored_global);
        let (_, fit_byte) = quantize_block_with_global_closed_form(&block, stored_global);
        let rtn_err = block_reconstruction_sq_error(&block, rtn_byte, stored_global);
        let fit_err = block_reconstruction_sq_error(&block, fit_byte, stored_global);
        if fit_err > rtn_err {
            regressions += 1;
        }
        rtn_total += rtn_err;
        fit_total += fit_err;
    }
    assert_eq!(
        regressions, 0,
        "closed-form scale fitting regressed on {regressions} blocks; it enumerates a superset of \
         the round-to-nearest choice so it can never lose block-wise"
    );
    let ratio = fit_total / rtn_total;
    assert!(
        ratio < 0.85,
        "closed-form scale fitting only reached mse ratio {ratio:.5} vs round-to-nearest-amax; \
         the measured figure on real qwen3.8 tail MLP weights is 0.731-0.735"
    );
}

#[test]
fn closed_form_quantized_block_round_trips_through_the_stock_dequantizer() {
    let mut state = 0x0123_4567_89ab_cdefu64;
    for _ in 0..256 {
        let block = gaussian_block(&mut state, 0.05);
        let amax = block.iter().fold(0f32, |a, b| a.max(b.abs()));
        let stored_global = 448.0 * 6.0 / amax;
        let (packed, scale_byte) = quantize_block_with_global_closed_form(&block, stored_global);
        assert_eq!(packed.len(), BLOCK_SIZE / 2);
        let inv = stored_global / decode_ue4m3(scale_byte);
        for (i, byte) in packed.iter().enumerate() {
            let (lo, hi) = nv_quant::nvfp4::unpack_e2m1_pair(*byte);
            for (j, nib) in [lo, hi].into_iter().enumerate() {
                let v = block[2 * i + j];
                let expect = nv_quant::nvfp4::encode_e2m1((v * inv).clamp(-6.0, 6.0));
                assert_eq!(
                    nib, expect,
                    "closed-form packing put nibble {nib:#x} where the stock encoder puts \
                     {expect:#x} at element {}; the emitted bytes must be readable by the \
                     unchanged runtime path",
                    2 * i + j
                );
            }
        }
    }
}
