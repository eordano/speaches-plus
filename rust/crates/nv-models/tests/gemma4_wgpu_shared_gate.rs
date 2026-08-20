#![cfg(feature = "wgpu")]

use nv_models::gemma4_wgpu_shared::{bf16_bits, bytes_to_words, pack_pairs, rope_tables};

#[test]
fn pack_pairs_puts_even_indices_in_the_low_halfword_and_odd_in_the_high() {
    let words = pack_pairs(&[0x1111, 0x2222, 0x3333]);
    assert_eq!(words, vec![0x2222_1111, 0x0000_3333]);
    for (i, w) in words.iter().enumerate() {
        let lo = (*w & 0xffff) as u16;
        let hi = (*w >> 16) as u16;
        let src = [0x1111u16, 0x2222, 0x3333, 0];
        assert_eq!(lo, src[2 * i], "low halfword of word {i} must be element {}", 2 * i);
        assert_eq!(hi, src[2 * i + 1], "high halfword of word {i} must be element {}", 2 * i + 1);
    }
}

#[test]
fn empty_inputs_still_produce_one_word_because_wgpu_buffers_cannot_be_empty() {
    assert_eq!(pack_pairs(&[]), vec![0u32]);
    assert_eq!(bytes_to_words(&[]), vec![0u32]);
}

#[test]
fn bytes_pack_little_endian_and_the_tail_is_zero_padded() {
    assert_eq!(bytes_to_words(&[0x01, 0x02, 0x03, 0x04, 0x05]), vec![0x04030201, 0x00000005]);
}

#[test]
fn partial_rotary_leaves_every_angle_past_the_cut_as_the_identity_rotation() {
    let head_dim = 8;
    let (cos, sin) = rope_tables(head_dim, 10_000.0, 0.5, 3);
    let half = head_dim / 2;
    let rope_angles = 2usize;
    for p in 0..3 {
        for i in rope_angles..half {
            assert_eq!(
                (cos[p * half + i], sin[p * half + i]),
                (1.0, 0.0),
                "pos {p} angle {i}: past the partial cut the rotation must be the identity, \
                 or the unrotated half of every head is silently scrambled"
            );
        }
        for i in 0..rope_angles {
            let inv = 1.0 / 10_000f32.powf((i as f32 * 2.0) / head_dim as f32);
            let theta = p as f32 * inv;
            assert!(
                (cos[p * half + i] - theta.cos()).abs() < 1e-6
                    && (sin[p * half + i] - theta.sin()).abs() < 1e-6,
                "pos {p} angle {i} disagrees with the closed form"
            );
        }
    }
    let full = rope_tables(head_dim, 10_000.0, 1.0, 2);
    assert!(
        (0..half).any(|i| full.1[half + i] != 0.0),
        "negative control: with partial = 1.0 some pos-1 angle must actually rotate, or the \
         identity rows above prove nothing"
    );
}

#[test]
fn position_zero_is_always_the_identity_whatever_the_partial() {
    for partial in [0.25f32, 0.5, 1.0] {
        let (cos, sin) = rope_tables(8, 10_000.0, partial, 1);
        assert!(cos.iter().all(|c| *c == 1.0) && sin.iter().all(|s| *s == 0.0));
    }
}

#[test]
fn bf16_bits_matches_half_for_the_values_the_shaders_round_trip() {
    for x in [0.0f32, 1.0, -1.0, 0.5, 3.140625] {
        assert_eq!(bf16_bits(x), half::bf16::from_f32(x).to_bits());
    }
    assert_eq!(bf16_bits(1.0), 0x3F80);
}
