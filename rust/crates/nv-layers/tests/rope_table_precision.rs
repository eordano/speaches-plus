use half::bf16;
use nv_layers::rope::{build_rope_tables_f32, build_rope_tables_f64};

const LAGUNA_SLIDING_DIM: usize = 128;
const LAGUNA_SLIDING_THETA: f64 = 10000.0;
const LAGUNA_FULL_ROT_DIM: usize = 64;
const LAGUNA_FULL_THETA: f64 = 500000.0;
const LAGUNA_FULL_FACTOR: f64 = 32.0;
const LAGUNA_FULL_ORIG: f64 = 8192.0;
const LAGUNA_FULL_BETA_FAST: f64 = 64.0;
const LAGUNA_FULL_BETA_SLOW: f64 = 1.0;
const GEMMA4_DIM: usize = 256;
const GEMMA4_BASE: f64 = 1000000.0;

fn default_inv_freq_f64(dim: usize, base: f64) -> Vec<f64> {
    (0..dim / 2)
        .map(|i| 1.0 / base.powf((i as f64 * 2.0) / dim as f64))
        .collect()
}

fn default_inv_freq_f32_powf(dim: usize, base: f32) -> Vec<f32> {
    (0..dim / 2)
        .map(|i| 1.0 / base.powf((i as f32 * 2.0) / (dim as f32)))
        .collect()
}

fn yarn_inv_freq_f64(dim: usize, base: f64, factor: f64, orig: f64, bf: f64, bs: f64) -> Vec<f64> {
    let find = |num_rot: f64| -> f64 {
        (dim as f64) * (orig / (num_rot * 2.0 * std::f64::consts::PI)).ln() / (2.0 * base.ln())
    };
    let mut low = find(bf).floor();
    let mut high = find(bs).ceil();
    low = low.max(0.0);
    high = high.min(dim as f64 - 1.0);
    if (high - low).abs() < f64::EPSILON {
        high += 0.001;
    }
    (0..dim / 2)
        .map(|i| {
            let pos_freq = base.powf((i as f64 * 2.0) / dim as f64);
            let extrap = 1.0 / pos_freq;
            let interp = 1.0 / (factor * pos_freq);
            let ramp = (((i as f64) - low) / (high - low)).clamp(0.0, 1.0);
            let ef = 1.0 - ramp;
            interp * (1.0 - ef) + extrap * ef
        })
        .collect()
}

fn narrow(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

#[derive(Clone, Copy)]
enum Arm {
    F32,

    F64Prod,

    F64Full,

    HfBf16,

    HfReal,

    Plant1,
}

fn theta_row(arm: Arm, p: usize, inv32: &[f32], inv64: &[f64]) -> Vec<f64> {
    match arm {
        Arm::F32 => inv32.iter().map(|f| ((p as f32) * *f) as f64).collect(),
        Arm::F64Prod => inv32.iter().map(|f| (p as f64) * (*f as f64)).collect(),
        Arm::HfReal => inv32.iter().map(|f| ((p as f32) * *f) as f64).collect(),
        Arm::F64Full | Arm::HfBf16 => inv64.iter().map(|f| (p as f64) * *f).collect(),
        Arm::Plant1 => inv64.iter().map(|f| ((p + 1) as f64) * *f).collect(),
    }
}

fn cos_sin_row(arm: Arm, p: usize, inv32: &[f32], inv64: &[f64]) -> (Vec<f32>, Vec<f32>) {
    let th = theta_row(arm, p, inv32, inv64);
    match arm {
        Arm::F32 | Arm::HfReal => {
            let (c, s) = build_rope_tables_f32(inv32, p + 1);
            let half = inv32.len();
            let (mut cr, mut sr) = (
                c[p * half..(p + 1) * half].to_vec(),
                s[p * half..(p + 1) * half].to_vec(),
            );
            if matches!(arm, Arm::HfReal) {
                for v in cr.iter_mut().chain(sr.iter_mut()) {
                    *v = bf16::from_f32(*v).to_f32();
                }
            }
            (cr, sr)
        }
        Arm::HfBf16 => (
            th.iter()
                .map(|t| bf16::from_f32(t.cos() as f32).to_f32())
                .collect(),
            th.iter()
                .map(|t| bf16::from_f32(t.sin() as f32).to_f32())
                .collect(),
        ),
        _ => (
            th.iter().map(|t| t.cos() as f32).collect(),
            th.iter().map(|t| t.sin() as f32).collect(),
        ),
    }
}

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let x = ((*state >> 33) as u32) as f32 / (u32::MAX >> 1) as f32;
    x * 2.0 - 1.0
}

fn apply_rope_bf16(q: &[bf16], cos: &[f32], sin: &[f32], n_heads: usize) -> Vec<u16> {
    let half = cos.len();
    let head_dim = half * 2;
    let mut out = vec![0u16; n_heads * head_dim];
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let a = q[base + i].to_f32();
            let b = q[base + i + half].to_f32();
            out[base + i] = bf16::from_f32(a * cos[i] - b * sin[i]).to_bits();
            out[base + i + half] = bf16::from_f32(a * sin[i] + b * cos[i]).to_bits();
        }
    }
    out
}

struct Row {
    dtheta_max: f64,
    token_equiv: f64,
    dcos_max: f64,
    flip_frac: f64,
    gt1ulp: f64,
    rel_l1: f64,
}

fn bf16_key(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

fn measure(arm: Arm, p: usize, inv32: &[f32], inv64: &[f64], n_heads: usize) -> Row {
    let ref_th = theta_row(Arm::F64Full, p, inv32, inv64);
    let arm_th = theta_row(arm, p, inv32, inv64);
    let (rc, rs) = cos_sin_row(Arm::F64Full, p, inv32, inv64);
    let (ac, s) = cos_sin_row(arm, p, inv32, inv64);

    let mut dtheta_max = 0f64;
    let mut token_equiv = 0f64;
    for i in 0..ref_th.len() {
        let d = (arm_th[i] - ref_th[i]).abs();
        dtheta_max = dtheta_max.max(d);
        if inv64[i] > 0.0 {
            token_equiv = token_equiv.max(d / inv64[i]);
        }
    }
    let dcos_max = ac
        .iter()
        .zip(rc.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs())
        .fold(0f64, f64::max);

    let head_dim = inv64.len() * 2;
    let mut st = 0x5eed_1234_abcd_0001u64 ^ (p as u64);
    let q: Vec<bf16> = (0..n_heads * head_dim)
        .map(|_| bf16::from_f32(lcg(&mut st)))
        .collect();
    let a_out = apply_rope_bf16(&q, &ac, &s, n_heads);
    let r_out = apply_rope_bf16(&q, &rc, &rs, n_heads);
    let mut diff = 0usize;
    let mut big = 0usize;
    let mut abs_err = 0f64;
    let mut abs_ref = 0f64;
    for (a, b) in a_out.iter().zip(r_out.iter()) {
        if a != b {
            diff += 1;

            if (bf16_key(*a) - bf16_key(*b)).unsigned_abs() > 1 {
                big += 1;
            }
        }
        let av = bf16::from_bits(*a).to_f32() as f64;
        let bv = bf16::from_bits(*b).to_f32() as f64;
        abs_err += (av - bv).abs();
        abs_ref += bv.abs();
    }
    Row {
        dtheta_max,
        token_equiv,
        dcos_max,
        flip_frac: diff as f64 / a_out.len() as f64,
        gt1ulp: big as f64 / a_out.len() as f64,
        rel_l1: if abs_ref > 0.0 {
            abs_err / abs_ref
        } else {
            0.0
        },
    }
}

const POSITIONS: [usize; 8] = [4000, 16000, 64000, 65536, 130000, 250000, 262143, 262144];
const N_HEADS: usize = 48;

fn report(name: &str, inv32: &[f32], inv64: &[f64]) {
    println!(
        "\n=== {name}  (half={}, rows scored={} heads)",
        inv64.len(),
        N_HEADS
    );
    println!(
        "{:>8} {:>5} | {:>9} {:>9} {:>7} | {:>9} {:>7} {:>7} {:>8} | {:>7} {:>7} {:>8} | {:>7}",
        "pos",
        "2^k",
        "dth_f32",
        "dth_prod",
        "tok_eq",
        "dcos_f32",
        "flip",
        "gt1ulp",
        "relL1",
        "flipHF",
        "gt1HF",
        "relL1_HF",
        "flipPrd"
    );
    for p in POSITIONS {
        let a = measure(Arm::F32, p, inv32, inv64, N_HEADS);
        let b = measure(Arm::F64Prod, p, inv32, inv64, N_HEADS);
        let h = measure(Arm::HfBf16, p, inv32, inv64, N_HEADS);
        let pow2 = p.is_power_of_two();
        println!(
            "{:>8} {:>5} | {:>9.3e} {:>9.3e} {:>7.4} | {:>9.3e} {:>7.4} {:>7.4} {:>8.2e} | {:>7.4} {:>7.4} {:>8.2e} | {:>7.4}",
            p,
            if pow2 { "yes" } else { "" },
            a.dtheta_max,
            b.dtheta_max,
            a.token_equiv,
            a.dcos_max,
            a.flip_frac,
            a.gt1ulp,
            a.rel_l1,
            h.flip_frac,
            h.gt1ulp,
            h.rel_l1,
            b.flip_frac,
        );
    }
}

fn compare(a: Arm, b: Arm, p: usize, inv32: &[f32], inv64: &[f64], n_heads: usize) -> (f64, f64) {
    let (ac, asn) = cos_sin_row(a, p, inv32, inv64);
    let (bc, bsn) = cos_sin_row(b, p, inv32, inv64);
    let head_dim = inv64.len() * 2;
    let mut st = 0x5eed_1234_abcd_0001u64 ^ (p as u64);
    let q: Vec<bf16> = (0..n_heads * head_dim)
        .map(|_| bf16::from_f32(lcg(&mut st)))
        .collect();
    let ao = apply_rope_bf16(&q, &ac, &asn, n_heads);
    let bo = apply_rope_bf16(&q, &bc, &bsn, n_heads);
    let mut diff = 0usize;
    let mut err = 0f64;
    let mut mag = 0f64;
    for (x, y) in ao.iter().zip(bo.iter()) {
        if x != y {
            diff += 1;
        }
        let xv = bf16::from_bits(*x).to_f32() as f64;
        let yv = bf16::from_bits(*y).to_f32() as f64;
        err += (xv - yv).abs();
        mag += yv.abs();
    }
    (
        diff as f64 / ao.len() as f64,
        if mag > 0.0 { err / mag } else { 0.0 },
    )
}

fn knob_report(name: &str, inv32: &[f32], inv64: &[f64]) {
    println!("\n--- knob effect: {name} ---");
    println!(
        "{:>8} {:>5} | {:>8} {:>9} | {:>8} {:>9} | {:>8} {:>9}",
        "pos", "2^k", "flipB1", "relL1_B1", "flipB2", "relL1_B2", "flipHFr", "relL1_HFr"
    );
    for p in POSITIONS {
        let (f1, r1) = compare(Arm::F32, Arm::F64Prod, p, inv32, inv64, N_HEADS);
        let (f2, r2) = compare(Arm::F32, Arm::F64Full, p, inv32, inv64, N_HEADS);
        let (fh, rh) = compare(Arm::F32, Arm::HfReal, p, inv32, inv64, N_HEADS);
        println!(
            "{:>8} {:>5} | {:>8.4} {:>9.2e} | {:>8.4} {:>9.2e} | {:>8.4} {:>9.2e}",
            p,
            if p.is_power_of_two() { "yes" } else { "" },
            f1,
            r1,
            f2,
            r2,
            fh,
            rh
        );
    }
}

fn laguna_sliding() -> (Vec<f32>, Vec<f64>) {
    let inv64 = default_inv_freq_f64(LAGUNA_SLIDING_DIM, LAGUNA_SLIDING_THETA);
    (narrow(&inv64), inv64)
}

fn laguna_full() -> (Vec<f32>, Vec<f64>) {
    let inv64 = yarn_inv_freq_f64(
        LAGUNA_FULL_ROT_DIM,
        LAGUNA_FULL_THETA,
        LAGUNA_FULL_FACTOR,
        LAGUNA_FULL_ORIG,
        LAGUNA_FULL_BETA_FAST,
        LAGUNA_FULL_BETA_SLOW,
    );
    (narrow(&inv64), inv64)
}

fn gemma4_shaped() -> (Vec<f32>, Vec<f64>) {
    let inv64 = default_inv_freq_f64(GEMMA4_DIM, GEMMA4_BASE);
    (
        default_inv_freq_f32_powf(GEMMA4_DIM, GEMMA4_BASE as f32),
        inv64,
    )
}

#[test]
fn rope_table_precision_report() {
    let (s32, s64) = laguna_sliding();
    report(
        "laguna sliding (dim128, theta 1e4, f64-then-cast inv_freq)",
        &s32,
        &s64,
    );
    let (f32v, f64v) = laguna_full();
    report(
        "laguna full yarn (rot64, theta 5e5, f64-then-cast inv_freq)",
        &f32v,
        &f64v,
    );
    let (g32, g64) = gemma4_shaped();
    report(
        "gemma4-shaped (dim256, base 1e6, f32 powf inv_freq)",
        &g32,
        &g64,
    );

    println!("\n--- inv_freq construction error (source (a)) ---");
    for (name, v32, v64) in [
        ("laguna sliding f64-then-cast", s32.clone(), s64.clone()),
        ("laguna full yarn f64-then-cast", f32v.clone(), f64v.clone()),
        ("gemma4 f32 powf", g32.clone(), g64.clone()),
    ] {
        let rel = v32
            .iter()
            .zip(v64.iter())
            .map(|(a, b)| ((*a as f64 - *b) / *b).abs())
            .fold(0f64, f64::max);
        let ulps = rel / f64::from(f32::EPSILON);
        println!("{name:<32} max rel err {rel:.4e}  ({ulps:.2} f32 ulp)");
    }

    knob_report("laguna sliding", &s32, &s64);
    knob_report("laguna full yarn", &f32v, &f64v);
    knob_report("gemma4-shaped", &g32, &g64);
}

#[test]
fn f32_table_stays_inside_the_hf_bf16_envelope() {
    for (name, (i32v, i64v)) in [
        ("laguna sliding", laguna_sliding()),
        ("laguna full yarn", laguna_full()),
        ("gemma4-shaped", gemma4_shaped()),
    ] {
        for p in POSITIONS {
            let ours = measure(Arm::F32, p, &i32v, &i64v, N_HEADS);
            let hf = measure(Arm::HfBf16, p, &i32v, &i64v, N_HEADS);
            let hf_real = measure(Arm::HfReal, p, &i32v, &i64v, N_HEADS);
            assert!(
                ours.rel_l1 <= hf.rel_l1,
                "{name} p={p}: our f32 table departs from exact by {:.3e} > HF's own bf16 cast {:.3e}",
                ours.rel_l1,
                hf.rel_l1
            );
            assert!(
                ours.rel_l1 <= hf_real.rel_l1,
                "{name} p={p}: ours {:.3e} > HF-as-shipped {:.3e}",
                ours.rel_l1,
                hf_real.rel_l1
            );

            let planted = measure(Arm::Plant1, p, &i32v, &i64v, N_HEADS);
            assert!(
                planted.rel_l1 > hf.rel_l1,
                "envelope check cannot fail: a 1-token position offset ({:.3e}) \
                 did not exceed the HF bf16 envelope ({:.3e}) at {name} p={p}",
                planted.rel_l1,
                hf.rel_l1
            );
        }
    }
}

#[test]
fn planted_position_offset_is_caught() {
    let (s32, s64) = laguna_sliding();
    for p in [4000usize, 64000, 250000] {
        let plant = measure(Arm::Plant1, p, &s32, &s64, N_HEADS);
        let base = measure(Arm::F32, p, &s32, &s64, N_HEADS);
        println!(
            "PLANT p={p}: token_eq {:.4} flip {:.4} relL1 {:.3e}   vs F32 token_eq {:.6} flip {:.6} relL1 {:.3e}",
            plant.token_equiv,
            plant.flip_frac,
            plant.rel_l1,
            base.token_equiv,
            base.flip_frac,
            base.rel_l1
        );
        assert!(
            plant.token_equiv > 0.99,
            "plant at p={p} must read ~1 token of offset, got {}",
            plant.token_equiv
        );
        assert!(
            plant.flip_frac > 0.5,
            "plant at p={p} must flip most bf16 outputs, got {}",
            plant.flip_frac
        );
        assert!(
            plant.token_equiv > base.token_equiv * 20.0,
            "instrument does not separate plant from baseline at p={p}: {} vs {}",
            plant.token_equiv,
            base.token_equiv
        );
        assert!(
            plant.rel_l1 > base.rel_l1 * 5.0,
            "plant magnitude not separated from baseline at p={p}: {:.3e} vs {:.3e}",
            plant.rel_l1,
            base.rel_l1
        );
    }
}

#[test]
fn power_of_two_positions_hide_the_product_error() {
    let (s32, s64) = laguna_sliding();
    let at_2k = measure(Arm::F32, 262144, &s32, &s64, N_HEADS);
    let prod_2k = measure(Arm::F64Prod, 262144, &s32, &s64, N_HEADS);
    println!(
        "p=262144 (2^18): F32 dtheta {:.4e}, F64Prod dtheta {:.4e}",
        at_2k.dtheta_max, prod_2k.dtheta_max
    );
    assert_eq!(
        at_2k.dtheta_max, prod_2k.dtheta_max,
        "at a power of two the f32 product is exact, so B1 must change nothing"
    );
    let at_odd = measure(Arm::F32, 262143, &s32, &s64, N_HEADS);
    let prod_odd = measure(Arm::F64Prod, 262143, &s32, &s64, N_HEADS);
    println!(
        "p=262143:       F32 dtheta {:.4e}, F64Prod dtheta {:.4e}",
        at_odd.dtheta_max, prod_odd.dtheta_max
    );
    assert!(
        prod_odd.dtheta_max < at_odd.dtheta_max,
        "off a power of two the f64 product must remove some error"
    );
}

#[test]
fn f64_builder_matches_f32_builder_on_exactly_representable_input() {
    let inv32 = vec![0.5f32, 0.25, 0.125, 0.0625];
    let inv64: Vec<f64> = inv32.iter().map(|v| *v as f64).collect();
    let (ca, sa) = build_rope_tables_f32(&inv32, 8);
    let (cb, sb) = build_rope_tables_f64(&inv64, 8);

    for i in 0..ca.len() {
        assert!((ca[i] - cb[i]).abs() < 1e-6, "cos idx {i}");
        assert!((sa[i] - sb[i]).abs() < 1e-6, "sin idx {i}");
    }
}
