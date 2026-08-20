const ULAW_BIAS: i32 = 0x84;
const ULAW_CLIP: i32 = 32_635;

pub fn ulaw_decode_byte(u: u8) -> i16 {
    let u = !u;
    let sign = u & 0x80;
    let exponent = (u >> 4) & 0x07;
    let mantissa = u & 0x0F;
    let mut t = ((mantissa as i32) << 3) + ULAW_BIAS;
    t <<= exponent;
    let s = if sign != 0 {
        ULAW_BIAS - t
    } else {
        t - ULAW_BIAS
    };
    s as i16
}

pub fn ulaw_encode_sample(s: i16) -> u8 {
    let mut sample = s as i32;
    let sign = if sample < 0 {
        sample = -sample;
        0x80u8
    } else {
        0
    };
    if sample > ULAW_CLIP {
        sample = ULAW_CLIP;
    }
    sample += ULAW_BIAS;
    let exponent = leading_seg_8bit((sample >> 7) as u32 & 0xFF);
    let mantissa = ((sample >> (exponent + 3)) & 0x0F) as u8;
    !(sign | (exponent << 4) | mantissa)
}

fn leading_seg_8bit(v: u32) -> u8 {
    let v = v & 0xFF;
    if v & 0x80 != 0 {
        7
    } else if v & 0x40 != 0 {
        6
    } else if v & 0x20 != 0 {
        5
    } else if v & 0x10 != 0 {
        4
    } else if v & 0x08 != 0 {
        3
    } else if v & 0x04 != 0 {
        2
    } else if v & 0x02 != 0 {
        1
    } else {
        0
    }
}

pub fn alaw_decode_byte(a: u8) -> i16 {
    let a = a ^ 0x55;
    let mantissa = (a & 0x0F) as i32;
    let seg = ((a & 0x70) >> 4) as i32;
    let mut t = mantissa << 4;
    if seg == 0 {
        t += 8;
    } else if seg == 1 {
        t += 0x108;
    } else {
        t += 0x108;
        t <<= seg - 1;
    }
    let val = if a & 0x80 != 0 { t } else { -t };
    val.clamp(i16::MIN as i32, i16::MAX as i32) as i16
}

pub fn alaw_encode_sample(s: i16) -> u8 {
    let mut pcm = s as i32;
    let mask: u8 = if pcm >= 0 {
        0xD5
    } else {
        pcm = -pcm - 8;
        0x55
    };
    static SEG_END: [i32; 8] = [0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF, 0x3FFF, 0x7FFF];
    let mut seg = 8usize;
    for (i, &end) in SEG_END.iter().enumerate() {
        if pcm <= end {
            seg = i;
            break;
        }
    }
    if seg >= 8 {
        return 0x7F ^ mask;
    }
    let aval = (seg as u8) << 4;
    let mantissa = if seg < 2 {
        ((pcm >> 4) & 0x0F) as u8
    } else {
        ((pcm >> (seg + 3)) & 0x0F) as u8
    };
    (aval | mantissa) ^ mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ulaw_decode_is_total_over_byte_space() {
        for b in 0u16..=0xFF {
            let v = ulaw_decode_byte(b as u8) as i32;
            assert!(v.abs() <= 32_124, "ulaw_decode({b:02x}) = {v} out of range");
        }
    }

    #[test]
    fn alaw_decode_is_total_over_byte_space() {
        for b in 0u16..=0xFF {
            let v = alaw_decode_byte(b as u8) as i32;
            assert!(v.abs() <= 32_256, "alaw_decode({b:02x}) = {v} out of range");
        }
    }

    #[test]
    fn ulaw_encode_is_total_over_i16_space() {
        let mut max_rel_err = 0.0f32;
        for s in i16::MIN..=i16::MAX {
            if s == i16::MIN {
                continue;
            }
            let encoded = ulaw_encode_sample(s);
            let decoded = ulaw_decode_byte(encoded);
            if (s.abs() as i32) > 256 {
                if decoded != 0 {
                    assert_eq!(decoded.signum(), s.signum(), "sign for {s}");
                }
                let err = (decoded as f32 - s as f32).abs() / s.abs() as f32;
                if err > max_rel_err {
                    max_rel_err = err;
                }
            }
        }
        assert!(
            max_rel_err < 0.13,
            "max relative error {max_rel_err} exceeds μ-law spec"
        );
    }

    #[test]
    fn alaw_encode_is_total_over_i16_space() {
        let mut max_rel_err = 0.0f32;
        for s in i16::MIN..=i16::MAX {
            let encoded = alaw_encode_sample(s);
            let decoded = alaw_decode_byte(encoded);

            if s == i16::MIN {
                continue;
            }
            if (s.abs() as i32) > 256 {
                if decoded != 0 {
                    assert_eq!(decoded.signum(), s.signum(), "sign for {s}");
                }
                let err = (decoded as f32 - s as f32).abs() / s.abs() as f32;
                if err > max_rel_err {
                    max_rel_err = err;
                }
            }
        }
        assert!(max_rel_err < 0.13, "max relative error {max_rel_err}");
    }

    #[test]
    fn ulaw_round_trip_preserves_sign_and_magnitude() {
        for s in [-32_000i16, -8_000, -1_000, 0, 1_000, 8_000, 32_000] {
            let encoded = ulaw_encode_sample(s);
            let decoded = ulaw_decode_byte(encoded);
            if s == 0 {
                assert!(decoded.abs() <= ULAW_BIAS as i16 / 2);
            } else {
                assert_eq!(decoded.signum(), s.signum(), "sign for {s}");
                let err = (decoded as f32 - s as f32).abs() / s.abs() as f32;
                assert!(err < 0.13, "relative error {err} for {s}");
            }
        }
    }

    #[test]
    fn alaw_round_trip_preserves_sign_and_magnitude() {
        for s in [-32_000i16, -8_000, -1_000, 0, 1_000, 8_000, 32_000] {
            let encoded = alaw_encode_sample(s);
            let decoded = alaw_decode_byte(encoded);
            if s == 0 {
                assert!(decoded.abs() <= 16);
            } else {
                assert_eq!(decoded.signum(), s.signum(), "sign for {s}");
                let err = (decoded as f32 - s as f32).abs() / s.abs() as f32;
                assert!(err < 0.13, "relative error {err} for {s}");
            }
        }
    }

    #[test]
    fn ulaw_silence_decodes_to_zero() {
        assert!(ulaw_decode_byte(0xFF).abs() < 16);
    }

    #[test]
    fn alaw_silence_decodes_to_zero() {
        assert!(alaw_decode_byte(0xD5).abs() < 16);
    }

    #[test]
    fn alaw_extremes_clip_in_max_segment() {
        let pos = alaw_encode_sample(32_767);
        let neg = alaw_encode_sample(-32_768);

        let dp = alaw_decode_byte(pos) as i32;
        let dn = alaw_decode_byte(neg) as i32;
        assert!(dp > 24_000, "positive extreme decodes to {dp}");
        assert!(dn < -24_000, "negative extreme decodes to {dn}");
    }
}
