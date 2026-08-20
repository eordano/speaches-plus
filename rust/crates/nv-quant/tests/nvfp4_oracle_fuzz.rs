use nv_quant::nvfp4::{decode_ue4m3, quantize_block_with_global, BLOCK_SIZE};

const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

const UE4M3_MIN_NORMAL: f32 = 0.015625;

const UE4M3_SUBNORMAL_STEP: f32 = 0.001953125;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlantedBug {
    None,
    ScaleMantissaTruncates,
    InvSkipsEncodeDecodeRoundTrip,
    E2m1TieRoundsUp,
    PackOrderSwapped,
}

fn encode_e2m1_by_midpoints(x: f32, bug: PlantedBug) -> u8 {
    let sign = if x.is_sign_negative() { 0b1000 } else { 0 };
    let abs = x.abs();
    let mut idx = 0u8;
    for i in 0..E2M1_VALUES.len() - 1 {
        let mid = (E2M1_VALUES[i] + E2M1_VALUES[i + 1]) * 0.5;
        let go_up = if bug == PlantedBug::E2m1TieRoundsUp {
            abs >= mid
        } else {
            abs > mid
        };
        if go_up {
            idx = (i + 1) as u8;
        }
    }
    sign | idx
}

fn encode_ue4m3_by_bits(scale: f32, bug: PlantedBug) -> u8 {
    if !scale.is_finite() || scale <= 0.0 {
        return 0;
    }
    let clamped = scale.min(448.0);
    if clamped < UE4M3_MIN_NORMAL {
        let m = (clamped / UE4M3_SUBNORMAL_STEP).round() as i32;
        if m <= 0 {
            return 0;
        }
        if m <= 7 {
            return m as u8;
        }
        return 0x08;
    }
    let bits = clamped.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let m23 = bits & 0x7F_FFFF;
    let (mantissa3, exp) = if bug == PlantedBug::ScaleMantissaTruncates {
        ((m23 >> 20) as i32, f32_exp)
    } else {
        let rounded = (m23 + (1 << 19)) >> 20;
        if rounded >= 8 {
            (0, f32_exp + 1)
        } else {
            (rounded as i32, f32_exp)
        }
    };
    let mantissa3 = mantissa3.clamp(0, 7);
    let biased = (exp + 7).clamp(1, 15);
    if biased == 15 && mantissa3 == 7 {
        return 0x7E;
    }
    ((biased as u8) << 3) | (mantissa3 as u8)
}

fn quantize_block_independent(values: &[f32], stored_global: f32, bug: PlantedBug) -> (Vec<u8>, u8) {
    assert_eq!(values.len(), BLOCK_SIZE);
    let amax = values.iter().fold(0f32, |a, b| a.max(b.abs()));
    let stored = if stored_global == 0.0 || !stored_global.is_finite() {
        1.0
    } else {
        stored_global
    };
    let local = if amax == 0.0 { 1.0 } else { amax / 6.0 };
    let scale_byte = encode_ue4m3_by_bits(stored * local, bug);
    let decoded = decode_ue4m3(scale_byte);
    let inv = if bug == PlantedBug::InvSkipsEncodeDecodeRoundTrip {
        if stored * local == 0.0 {
            1.0
        } else {
            stored / (stored * local)
        }
    } else if decoded == 0.0 {
        1.0
    } else {
        stored / decoded
    };
    let mut packed = Vec::with_capacity(BLOCK_SIZE / 2);
    for pair in values.chunks(2) {
        let lo = encode_e2m1_by_midpoints((pair[0] * inv).clamp(-6.0, 6.0), bug);
        let hi = encode_e2m1_by_midpoints((pair[1] * inv).clamp(-6.0, 6.0), bug);
        packed.push(if bug == PlantedBug::PackOrderSwapped {
            (lo << 4) | (hi & 0x0F)
        } else {
            (hi << 4) | (lo & 0x0F)
        });
    }
    (packed, scale_byte)
}

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed | 1)
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 32) as u32
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0
    }
}

fn fuzz_blocks(seed: u64) -> Vec<(Vec<f32>, f32)> {
    let mut rng = Lcg::new(seed);
    let mut cases: Vec<(Vec<f32>, f32)> = Vec::new();
    for scale_mag in [1e-4f32, 3e-3, 0.02, 1.0, 37.0, 500.0] {
        for _ in 0..24 {
            let block: Vec<f32> = (0..BLOCK_SIZE).map(|_| rng.next_f32() * scale_mag).collect();
            cases.push((block, 1.0));
        }
    }
    for global in [0.0f32, 0.0073, 1.0, 3.5, 448.0] {
        for _ in 0..12 {
            let block: Vec<f32> = (0..BLOCK_SIZE).map(|_| rng.next_f32()).collect();
            cases.push((block, global));
        }
    }
    cases.push((vec![0.0; BLOCK_SIZE], 1.0));
    cases.push((vec![-0.0; BLOCK_SIZE], 1.0));
    let mut spikes = vec![0.0f32; BLOCK_SIZE];
    spikes[7] = 6.0;
    spikes[8] = -6.0;
    cases.push((spikes, 1.0));
    for &tie in &[0.25f32, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0] {
        let mut block = vec![0.0f32; BLOCK_SIZE];
        block[0] = tie * 1.0;
        block[1] = -tie;
        block[15] = 6.0;
        cases.push((block, 1.0));
    }
    cases
}

#[test]
fn independent_bit_route_matches_the_shipping_quantizer_byte_for_byte() {
    let mut checked = 0usize;
    for seed in [1u64, 0x9e3779b9, 0xfeedface] {
        for (block, global) in fuzz_blocks(seed) {
            let (packed_a, scale_a) = quantize_block_with_global(&block, global);
            let (packed_b, scale_b) =
                quantize_block_independent(&block, global, PlantedBug::None);
            assert_eq!(
                (packed_a.as_slice(), scale_a),
                (packed_b.as_slice(), scale_b),
                "shipping log2-route quantizer and independent f32-bit-route quantizer \
                 disagree on block {block:?} global {global}; one of them no longer \
                 implements the documented nvfp4 rule"
            );
            checked += 1;
        }
    }
    assert!(checked >= 450, "fuzz shrank to {checked} blocks; coverage lost");
}

#[test]
fn every_planted_bug_changes_bytes_somewhere_in_the_fuzz_set() {
    for &bug in &[
        PlantedBug::ScaleMantissaTruncates,
        PlantedBug::InvSkipsEncodeDecodeRoundTrip,
        PlantedBug::E2m1TieRoundsUp,
        PlantedBug::PackOrderSwapped,
    ] {
        let mut caught = false;
        for (block, global) in fuzz_blocks(0xabcdef) {
            let good = quantize_block_independent(&block, global, PlantedBug::None);
            let bad = quantize_block_independent(&block, global, bug);
            if good != bad {
                caught = true;
                break;
            }
        }
        assert!(
            caught,
            "a planted quantizer bug survived the whole fuzz set; a gate that cannot \
             catch its own seeded mutations vouches for nothing (05.2 planted-bug protocol)"
        );
    }
}

#[test]
fn tie_inputs_are_present_and_decide_the_tie_direction() {
    let mut block = vec![0.0f32; BLOCK_SIZE];
    block[0] = 0.25;
    block[15] = 6.0;
    let good = quantize_block_independent(&block, 1.0, PlantedBug::None);
    let tied_up = quantize_block_independent(&block, 1.0, PlantedBug::E2m1TieRoundsUp);
    assert_ne!(
        good, tied_up,
        "0.25 at scale 1.0 sits exactly on the 0.0/0.5 midpoint; if this stops \
         distinguishing the tie direction the fuzz set has lost its midpoint coverage \
         and E2M1 rounding is ungated"
    );
}
