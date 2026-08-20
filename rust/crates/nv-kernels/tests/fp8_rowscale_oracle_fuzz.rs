use half::bf16;
mod common;
use common::LcgOddSeedShift32F64TwoSided as Lcg;
use common::max_rel_diff;

const E4M3_NAN_CODE_MAG: u8 = 0x7f;

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlantedBug {
    None,
    RowScaleOffByOne,
    SubnormalsDroppedToZero,
    SignBitIgnored,
    NanCodeDecodedAsMax,
}

fn decode_e4m3_float_route(code: u8) -> Option<f32> {
    let mag = code & 0x7f;
    if mag == E4M3_NAN_CODE_MAG {
        return None;
    }
    let e = (mag >> 3) as i32;
    let m = (mag & 7) as f64;
    let v = if e == 0 {
        m * 2f64.powi(-9)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e - 7)
    };
    let v = v as f32;
    Some(if code & 0x80 != 0 { -v } else { v })
}

fn decode_e4m3_integer_route(code: u8, bug: PlantedBug) -> Option<f64> {
    let mag = code & 0x7f;
    if mag == E4M3_NAN_CODE_MAG {
        if bug == PlantedBug::NanCodeDecodedAsMax {
            return Some(448.0);
        }
        return None;
    }
    let e = (mag >> 3) as i64;
    let m = (mag & 7) as i64;
    let v = if e == 0 {
        if bug == PlantedBug::SubnormalsDroppedToZero {
            0.0
        } else {
            m as f64 / 512.0
        }
    } else {
        ((8 + m) as f64) * 2f64.powi((e - 10) as i32)
    };
    if code & 0x80 != 0 && bug != PlantedBug::SignBitIgnored {
        Some(-v)
    } else {
        Some(v)
    }
}

fn ref_kernel_order_f32(
    w: &[u8],
    x_bf16: &[u16],
    row_scale: &[f32],
    n: usize,
    k: usize,
) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for row in 0..n {
        let mut acc = 0f32;
        for kk in 0..k {
            let wv = decode_e4m3_float_route(w[row * k + kk]).unwrap_or(0.0);
            acc += wv * bf16::from_bits(x_bf16[kk]).to_f32();
        }
        y[row] = bf16::from_f32(acc * row_scale[row]).to_f32();
    }
    y
}

fn ref_integer_route_f64(
    w: &[u8],
    x_bf16: &[u16],
    row_scale: &[f32],
    n: usize,
    k: usize,
    bug: PlantedBug,
) -> Vec<f32> {
    let mut y = vec![0f32; n];
    for row in 0..n {
        let scale_row = if bug == PlantedBug::RowScaleOffByOne {
            (row + 1) % n
        } else {
            row
        };
        let mut acc = 0f64;
        for kk in 0..k {
            let wv = decode_e4m3_integer_route(w[row * k + kk], bug).unwrap_or(0.0);
            acc += wv * bf16::from_bits(x_bf16[kk]).to_f64();
        }
        y[row] = bf16::from_f32((acc * row_scale[scale_row] as f64) as f32).to_f32();
    }
    y
}

fn gen_inputs(n: usize, k: usize, seed: u64) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
    gen_inputs_in_regime(n, k, seed, WeightRegime::FullRange)
}

#[derive(Clone, Copy)]
enum WeightRegime {
    FullRange,
    SubnormalsOnlySinceFullRangeBuriesThemUnderNormalsAt2em4Relative,
}

fn gen_inputs_in_regime(
    n: usize,
    k: usize,
    seed: u64,
    regime: WeightRegime,
) -> (Vec<u8>, Vec<u16>, Vec<f32>) {
    let mut rng = Lcg::new(seed);
    let w: Vec<u8> = (0..n * k)
        .map(|_| {
            let b = (rng.next_u32() & 0xff) as u8;
            match regime {
                WeightRegime::FullRange => {
                    if b & 0x7f == E4M3_NAN_CODE_MAG {
                        b & !0x08
                    } else {
                        b
                    }
                }
                WeightRegime::SubnormalsOnlySinceFullRangeBuriesThemUnderNormalsAt2em4Relative => {
                    b & 0x87
                }
            }
        })
        .collect();
    let x: Vec<u16> = (0..k)
        .map(|_| {
            let v = (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16::from_f32(v).to_bits()
        })
        .collect();
    let scale: Vec<f32> = (0..n)
        .map(|_| 0.001 + (rng.next_u32() as f32 / u32::MAX as f32) * 0.05)
        .collect();
    (w, x, scale)
}

const CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION_AND_BF16_OUT: f64 = 1e-2;

const FUZZ_SHAPES: &[(usize, usize)] = &[(1, 64), (3, 96), (5, 512), (16, 4096), (7, 5120), (2, 17408)];

#[test]
fn float_route_and_integer_route_agree_over_the_fuzzed_grid() {
    let mut cases = 0usize;
    for &(n, k) in FUZZ_SHAPES {
        for seed in [1u64, 0x9e3779b9, 0xfeedface] {
            let (w, x, scale) = gen_inputs(n, k, seed);
            let a = ref_kernel_order_f32(&w, &x, &scale, n, k);
            let b = ref_integer_route_f64(&w, &x, &scale, n, k, PlantedBug::None);
            let d = max_rel_diff(&a, &b);
            assert!(
                d < CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION_AND_BF16_OUT,
                "float-formula f32 oracle and integer-fraction f64 oracle diverged \
                 (n={n} k={k} seed={seed:#x}: rel {d:.3e}); one of them no longer implements \
                 the documented rowscale rule (decode e4m3, accumulate, scale once per row, \
                 bf16-encode)"
            );
            cases += 1;
        }
    }
    for seed in [21u64, 22, 23] {
        let (w, x, scale) = gen_inputs_in_regime(
            5,
            512,
            seed,
            WeightRegime::SubnormalsOnlySinceFullRangeBuriesThemUnderNormalsAt2em4Relative,
        );
        let a = ref_kernel_order_f32(&w, &x, &scale, 5, 512);
        let b = ref_integer_route_f64(&w, &x, &scale, 5, 512, PlantedBug::None);
        let d = max_rel_diff(&a, &b);
        assert!(
            d < CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION_AND_BF16_OUT,
            "the two oracles diverge on all-subnormal weights (seed {seed:#x}: rel {d:.3e}); \
             the m/512 fraction and the m*2^-9 float formula must be the same number"
        );
        cases += 1;
    }
    assert!(cases >= 21, "shape grid shrank to {cases} cases");
}

#[test]
fn every_planted_rowscale_bug_is_caught() {
    let (n, k) = (5usize, 512usize);
    for &(bug, name, regime) in &[
        (
            PlantedBug::RowScaleOffByOne,
            "RowScaleOffByOne",
            WeightRegime::FullRange,
        ),
        (
            PlantedBug::SubnormalsDroppedToZero,
            "SubnormalsDroppedToZero",
            WeightRegime::SubnormalsOnlySinceFullRangeBuriesThemUnderNormalsAt2em4Relative,
        ),
        (
            PlantedBug::SignBitIgnored,
            "SignBitIgnored",
            WeightRegime::FullRange,
        ),
        (
            PlantedBug::NanCodeDecodedAsMax,
            "NanCodeDecodedAsMax",
            WeightRegime::FullRange,
        ),
    ] {
        let mut caught = false;
        for seed in [3u64, 4, 5] {
            let (mut w, x, scale) = gen_inputs_in_regime(n, k, seed, regime);
            if bug == PlantedBug::NanCodeDecodedAsMax {
                w[0] = E4M3_NAN_CODE_MAG;
                w[k + 1] = 0x80 | E4M3_NAN_CODE_MAG;
            }
            let good = ref_integer_route_f64(&w, &x, &scale, n, k, PlantedBug::None);
            let bad = ref_integer_route_f64(&w, &x, &scale, n, k, bug);
            if max_rel_diff(&good, &bad)
                > 10.0 * CROSS_DECOMPOSITION_REL_TOL_BOUNDS_F32_VS_F64_ACCUMULATION_AND_BF16_OUT
            {
                caught = true;
                break;
            }
        }
        assert!(
            caught,
            "planted bug {name} survived its own amplification regime; a gate that cannot \
             catch its seeded mutations vouches for nothing (05.2 planted-bug protocol)"
        );
    }
}

#[test]
fn uniform_row_scales_hide_the_off_by_one_so_distinct_scales_are_mandatory() {
    let (n, k) = (4usize, 128usize);
    let (w, x, _) = gen_inputs(n, k, 9);
    let uniform = vec![0.01f32; n];
    let good = ref_integer_route_f64(&w, &x, &uniform, n, k, PlantedBug::None);
    let bad = ref_integer_route_f64(&w, &x, &uniform, n, k, PlantedBug::RowScaleOffByOne);
    assert!(
        max_rel_diff(&good, &bad) < 1e-12,
        "with identical per-row scales the off-by-one is invisible; this pins WHY every \
         rowscale suite must fuzz DISTINCT per-row scales (the w4a16 group-16 incident \
         class, row-scale edition)"
    );
}
