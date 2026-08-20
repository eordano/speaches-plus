use nv_quant::nvfp4::{decode_ue4m3, encode_ue4m3};

fn ref_decode(byte: u8) -> Option<f32> {
    let e = (byte >> 3) & 0x0F;
    let m = byte & 0x07;
    if e == 0x0F && m == 0x07 {
        return None;
    }
    if e == 0 {
        Some(m as f32 * (2f32).powi(-9))
    } else {
        Some((8 + m) as f32 * (2f32).powi(e as i32 - 10))
    }
}

fn representable() -> Vec<f32> {
    let mut v: Vec<f32> = (0u16..256).filter_map(|b| ref_decode(b as u8)).collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v.dedup();
    v
}

#[test]
fn decode_matches_hardware_definition_over_full_byte_domain() {
    for b in 0u16..256 {
        let b = b as u8;
        match ref_decode(b) {
            Some(want) => {
                let got = decode_ue4m3(b);
                assert!(
                    got == want,
                    "byte {b:#04x}: decode_ue4m3={got} but e4m3fn hardware value is {want}"
                );
            }
            None => {
                let got = decode_ue4m3(b);
                assert_eq!(
                    got, 480.0,
                    "byte 0x7f: expected the documented software reading 480.0, got {got}"
                );
            }
        }
    }
}

#[test]
fn encode_round_trips_every_representable_byte() {
    for b in 0u8..0x80 {
        let Some(v) = ref_decode(b) else { continue };
        let re = encode_ue4m3(v);

        let want = if v == 0.0 { 0 } else { b };
        assert_eq!(
            re, want,
            "byte {b:#04x} (value {v}): encode(decode(b)) = {re:#04x}, not identity"
        );
    }

    for b in 0x80u16..256 {
        let b = b as u8;
        let alias = b & 0x7F;
        let (a, c) = (decode_ue4m3(b), decode_ue4m3(alias));
        assert!(
            (a == c) || (alias == 0x7F),
            "byte {b:#04x}: software decode {a} != positive alias {c}"
        );
    }
}

#[test]
fn encoder_never_emits_the_nan_byte() {
    let mut probes: Vec<f32> = vec![448.0, 448.0001, 449.0, 460.0, 479.9, 480.0, 1e6, f32::MAX];
    for i in 0..10_000 {
        probes.push(440.0 + i as f32 * 0.01);
    }
    for s in probes {
        assert_ne!(
            encode_ue4m3(s),
            0x7F,
            "encode({s}) produced the e4m3fn NaN byte"
        );
    }
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn probe_set() -> Vec<f32> {
    let mut probes: Vec<f32> = Vec::with_capacity(1_100_000);

    let mut rng = XorShift(0x9e3779b97f4a7c15);
    let lo = (2f64).powi(-14).ln();
    let hi = 600f64.ln();
    for _ in 0..1_000_000 {
        let s = (lo + (hi - lo) * rng.unit_f64()).exp() as f32;
        probes.push(s);
    }

    let step = (2f32).powi(-9);
    for k in 0..=8 {
        let v = k as f32 * step;
        for d in [-3i32, -2, -1, 0, 1, 2, 3] {
            let p = if d < 0 {
                (0..-d).fold(v, |a, _| f32_prev(a))
            } else {
                (0..d).fold(v, |a, _| f32_next(a))
            };
            if p > 0.0 {
                probes.push(p);
            }
        }
        probes.push(v + step / 2.0);
        probes.push(v + step / 2.0 - step / 1024.0);
        probes.push(v + step / 2.0 + step / 1024.0);
    }

    for e in -7i32..=9 {
        let p = (2f32).powi(e);
        for d in [-4i32, -3, -2, -1, 0, 1, 2, 3, 4] {
            let v = if d < 0 {
                (0..-d).fold(p, |a, _| f32_prev(a))
            } else {
                (0..d).fold(p, |a, _| f32_next(a))
            };
            if v > 0.0 {
                probes.push(v);
            }
        }

        let base = (2f32).powi(e - 1);
        probes.push(base * 1.9375);
        probes.push(f32_next(base * 1.9375));
        probes.push(base * 1.94);
        probes.push(base * 1.97);
        probes.push(base * 1.99);
        probes.push(base * 1.999);
    }

    for e in -6i32..=8 {
        let base = (2f32).powi(e);
        for m in 0..8 {
            let mid = base * (1.0 + (m as f32 + 0.5) / 8.0);
            probes.push(mid);
            probes.push(f32_prev(mid));
            probes.push(f32_next(mid));
        }
    }

    probes.extend_from_slice(&[447.9, 448.0, 448.1, 600.0, 1e30, f32::MAX]);
    probes
}

fn f32_next(x: f32) -> f32 {
    f32::from_bits(x.to_bits() + 1)
}
fn f32_prev(x: f32) -> f32 {
    if x == 0.0 {
        0.0
    } else {
        f32::from_bits(x.to_bits() - 1)
    }
}

#[test]
fn encode_rounds_to_nearest_representable_1e6_sweep() {
    let table = representable();
    let probes = probe_set();

    let mut violations = 0usize;
    let mut worst: Option<(f32, f32, f32)> = None;
    let mut worst_excess = 0f64;

    for &s in &probes {
        let byte = encode_ue4m3(s);
        let got = decode_ue4m3(byte);
        let clamped = s.clamp(0.0, 448.0);

        let idx = table.partition_point(|&r| r < clamped);
        let lo = table[idx.saturating_sub(1)];
        let hi = table[idx.min(table.len() - 1)];
        let d_lo = (clamped as f64 - lo as f64).abs();
        let d_hi = (hi as f64 - clamped as f64).abs();
        let (near, d_near) = if d_lo <= d_hi { (lo, d_lo) } else { (hi, d_hi) };
        let d_got = (got as f64 - clamped as f64).abs();

        let slack = (clamped as f64) * 1.2e-7 + f64::MIN_POSITIVE;
        if d_got > d_near + slack {
            violations += 1;
            let excess = d_got - d_near;
            if excess > worst_excess {
                worst_excess = excess;
                worst = Some((s, got, near));
            }
        }
    }

    if let Some((s, got, near)) = worst {
        panic!(
            "{violations}/{} probes did not round to the nearest representable; \
             worst: encode({s}) -> {got} but nearest is {near} \
             (excess abs err {worst_excess:e}) -- mantissa overflow at a \
             power-of-two boundary must carry into the exponent",
            probes.len()
        );
    }
}

#[test]
fn encode_is_monotone_nondecreasing() {
    let mut probes = probe_set();
    probes.retain(|s| s.is_finite() && *s > 0.0);
    probes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut prev_val = -1f32;
    let mut prev_s = 0f32;
    for &s in &probes {
        let v = decode_ue4m3(encode_ue4m3(s));
        assert!(
            v >= prev_val,
            "monotonicity violated: encode({prev_s}) -> {prev_val} but encode({s}) -> {v}"
        );
        prev_val = v;
        prev_s = s;
    }
}
