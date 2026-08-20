const NVFP4_BLOCK_SIZE: u32 = 16u;
const UE4M3_SUBNORMAL_STEP: f32 = 0.001953125;
const E5M2_SUBNORMAL_STEP: f32 = 0.0000152587890625;
const E2M1_MAX: f32 = 6.0;
const E4M3_MAX: f32 = 448.0;
const F32_QNAN: u32 = 0x7fc00000u;
const F32_INF: u32 = 0x7f800000u;

var<private> E2M1_TABLE: array<f32, 16> = array<f32, 16>(
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
);

fn nvfp4_decode(nibble: u32) -> f32 {
    return E2M1_TABLE[nibble & 15u];
}

fn nvfp4_decode_arith(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let m = f32(n & 1u);
    let e = (n >> 1u) & 3u;
    let mag = select((1.0 + m * 0.5) * exp2(f32(e) - 1.0), m * 0.5, e == 0u);
    return select(mag, -mag, (n >> 3u) == 1u);
}

fn ue4m3_decode(bits: u32) -> f32 {
    let b = bits & 255u;
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    return select(
        bitcast<f32>(((e + 120u) << 23u) | (m << 20u)),
        f32(m) * UE4M3_SUBNORMAL_STEP,
        e == 0u
    );
}

fn e4m3_decode(bits: u32) -> f32 {
    let b = bits & 255u;
    if ((b & 127u) == 127u) {
        return bitcast<f32>(F32_QNAN);
    }
    let e = (b >> 3u) & 15u;
    let m = b & 7u;
    let mag = select(
        bitcast<f32>(((e + 120u) << 23u) | (m << 20u)),
        f32(m) * UE4M3_SUBNORMAL_STEP,
        e == 0u
    );
    return select(mag, -mag, (b & 128u) != 0u);
}

fn e4m3_shift_decode_scale_must_carry_2pow120(bits: u32) -> f32 {
    let b = bits & 255u;
    return bitcast<f32>(((b & 128u) << 24u) | ((b & 127u) << 20u));
}

fn e2m1_shift_decode_scale_must_carry_2pow126(code: u32) -> f32 {
    let n = code & 15u;
    return bitcast<f32>(((n & 8u) << 28u) | ((n & 7u) << 22u));
}

fn e5m2_decode(bits: u32) -> f32 {
    let b = bits & 255u;
    let e = (b >> 2u) & 31u;
    let m = b & 3u;
    var mag: f32;
    if (e == 0u) {
        mag = f32(m) * E5M2_SUBNORMAL_STEP;
    } else if (e == 31u) {
        mag = select(bitcast<f32>(F32_INF), bitcast<f32>(F32_QNAN), m != 0u);
    } else {
        mag = bitcast<f32>(((e + 112u) << 23u) | (m << 21u));
    }
    return select(mag, -mag, (b & 128u) != 0u);
}

fn bf16_decode(bits: u32) -> f32 {
    return bitcast<f32>((bits & 65535u) << 16u);
}

fn bf16_encode(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let r = 0x7fffu + ((b >> 16u) & 1u);
    return select((b + r) >> 16u, 0x7fc0u, x != x);
}

fn bf16_lo(word: u32) -> f32 {
    return bitcast<f32>(word << 16u);
}

fn bf16_hi(word: u32) -> f32 {
    return bitcast<f32>(word & 0xffff0000u);
}

fn bf16_pack(lo: f32, hi: f32) -> u32 {
    return bf16_encode(lo) | (bf16_encode(hi) << 16u);
}

fn byte_at(word: u32, idx: u32) -> u32 {
    return extractBits(word, 8u * (idx & 3u), 8u);
}

fn u16_at(word: u32, idx: u32) -> u32 {
    return extractBits(word, 16u * (idx & 1u), 16u);
}

fn u4_unpack(word: u32, elem: u32) -> u32 {
    return extractBits(word, 4u * (elem & 7u), 4u);
}

fn int4_decode(word: u32, elem: u32, group_scale: f32, zero_point: f32) -> f32 {
    return (f32(u4_unpack(word, elem)) - zero_point) * group_scale;
}

fn int4_decode_u4b8(word: u32, elem: u32, group_scale: f32) -> f32 {
    return int4_decode(word, elem, group_scale, 8.0);
}

fn int8_decode(word: u32, elem: u32) -> f32 {
    return f32(extractBits(bitcast<i32>(word), 8u * (elem & 3u), 8u));
}

fn nvfp4_nibble(word: u32, elem: u32) -> u32 {
    return extractBits(word, 4u * (elem & 7u), 4u);
}

fn nvfp4_data_word_index(row: u32, k_elems: u32, elem: u32) -> u32 {
    return row * (k_elems >> 3u) + (elem >> 3u);
}

fn nvfp4_k_tiles(k_blocks: u32) -> u32 {
    return (k_blocks + 3u) / 4u;
}

fn nvfp4_scale_byte_index(row: u32, block: u32, k_tiles: u32) -> u32 {
    let m_tile = row / 128u;
    let d2 = (row / 32u) % 4u;
    let d3 = row % 32u;
    let k_tile = block / 4u;
    let d5 = block % 4u;
    return ((m_tile * k_tiles + k_tile) * 32u + d3) * 16u + d2 * 4u + d5;
}

fn nvfp4_scale_linear_index(row: u32, block: u32, k_blocks: u32) -> u32 {
    return row * k_blocks + block;
}

fn nvfp4_value(nibble: u32, scale_byte: u32) -> f32 {
    return nvfp4_decode(nibble) * ue4m3_decode(scale_byte);
}

fn nvfp4_value_global(nibble: u32, scale_byte: u32, global_scale: f32) -> f32 {
    return nvfp4_decode(nibble) * ue4m3_decode(scale_byte) * global_scale;
}

fn nvfp4_block_accum(block_dot: f32, w_scale_byte: u32, x_scale_byte: u32) -> f32 {
    return block_dot * ue4m3_decode(w_scale_byte) * ue4m3_decode(x_scale_byte);
}

fn nv_tanhf(x: f32) -> f32 {
    let ax = abs(x);
    let e = exp2(ax * bitcast<f32>(0x4038AA3Bu));
    let d = e + 1.0;
    let r = 1.0 / d;
    let big = fma(r, -2.0, 1.0);
    let sat = select(big, 1.0, ax >= bitcast<f32>(0x41102CB4u));
    let signed_big = bitcast<f32>((bitcast<u32>(x) & 0x80000000u) | bitcast<u32>(sat));
    let x2 = x * x;
    let p0 = fma(bitcast<f32>(0x3C80F082u), x2, bitcast<f32>(0xBD563CAEu));
    let p1 = fma(p0, x2, bitcast<f32>(0x3E085941u));
    let p2 = fma(p1, x2, bitcast<f32>(0xBEAAA9EDu));
    let p3 = fma(p2, x2, 0.0);
    let small = fma(p3, x, x);
    return select(small, signed_big, ax >= bitcast<f32>(0x3F19999Au));
}

fn nvfp4_encode_e2m1(x: f32) -> u32 {
    let sign = (bitcast<u32>(x) >> 31u) << 3u;
    let a = abs(x);
    var best = 0u;
    var best_err = bitcast<f32>(F32_INF);
    for (var i = 0u; i < 8u; i = i + 1u) {
        let err = abs(a - E2M1_TABLE[i]);
        if (err < best_err) {
            best_err = err;
            best = i;
        }
    }
    return sign | best;
}

fn q_pow2(e: i32) -> f32 {
    return bitcast<f32>(u32(e + 127) << 23u);
}

fn q_norm_parts(x: f32) -> vec2<i32> {
    let b = bitcast<u32>(x) & 0x7fffffffu;
    let ef = b >> 23u;
    if (ef == 0u) {
        var m = b;
        var e = -126;
        loop {
            if (m >= 0x800000u) {
                break;
            }
            m = m << 1u;
            e = e - 1;
        }
        return vec2<i32>(i32(m), e);
    }
    return vec2<i32>(i32((b & 0x7fffffu) | 0x800000u), i32(ef) - 127);
}

fn q_finish(sign: u32, q_trunc: u32, r: u32, d: u32, re0: i32) -> f32 {
    var re = re0;
    if (re + 150 >= 255) {
        return bitcast<f32>(sign | F32_INF);
    }
    if (re + 150 >= 1) {
        var q = q_trunc;
        let two_r = 2u * r;
        if (two_r > d || (two_r == d && (q & 1u) == 1u)) {
            q = q + 1u;
        }
        if (q >= 0x1000000u) {
            q = q >> 1u;
            re = re + 1;
            if (re + 150 >= 255) {
                return bitcast<f32>(sign | F32_INF);
            }
        }
        return bitcast<f32>(sign | (u32(re + 150) << 23u) | (q & 0x7fffffu));
    }
    let sh = u32(-149 - re);
    if (sh >= 25u) {
        return bitcast<f32>(sign);
    }
    let k0 = q_trunc >> sh;
    let rem = q_trunc & ((1u << sh) - 1u);
    let num = rem * d + r;
    let den = d << sh;
    var k = k0;
    if (2u * num > den || (2u * num == den && (k0 & 1u) == 1u)) {
        k = k + 1u;
    }
    return bitcast<f32>(sign | k);
}

fn q_div_small(a: f32, d_in: u32, p: i32) -> f32 {
    let d = clamp(d_in, 1u, 15u);
    let bits = bitcast<u32>(a);
    let sign = bits & 0x80000000u;
    let expf = (bits >> 23u) & 255u;
    let frac = bits & 0x7fffffu;
    if (expf == 255u) {
        return a / (f32(d) * q_pow2(p));
    }
    var m: u32;
    var e: i32;
    if (expf == 0u) {
        if (frac == 0u) {
            return bitcast<f32>(sign);
        }
        m = frac;
        e = -149;
        loop {
            if (m >= 0x800000u) {
                break;
            }
            m = m << 1u;
            e = e - 1;
        }
    } else {
        m = frac | 0x800000u;
        e = i32(expf) - 150;
    }
    var s = 0u;
    var n = m;
    var q = m / d;
    loop {
        if (q >= 0x800000u || s >= 4u) {
            break;
        }
        s = s + 1u;
        n = m << s;
        q = n / d;
    }
    let r = n - q * d;
    return q_finish(sign, q, r, d, e - p - i32(s));
}

fn q_scale_parts(scale_byte: u32) -> vec2<i32> {
    let b = scale_byte & 255u;
    let e = (b >> 3u) & 15u;
    var sig: u32;
    var p: i32;
    if (e == 0u) {
        sig = b & 7u;
        p = -9;
    } else {
        sig = 8u + (b & 7u);
        p = i32(e) - 10;
    }
    if (sig == 0u) {
        return vec2<i32>(0, 0);
    }
    loop {
        if ((sig & 1u) == 1u) {
            break;
        }
        sig = sig >> 1u;
        p = p + 1;
    }
    return vec2<i32>(i32(sig), p);
}

fn q_scale_parts_ref(scale_byte: u32) -> vec2<i32> {
    return q_scale_parts(scale_byte);
}

fn q_subnormal_shift(x: f32) -> i32 {
    let b = bitcast<u32>(x) & 0x7fffffffu;
    if (b == 0u || b >= 0x800000u) {
        return 0;
    }
    var m = b;
    var s = 0;
    loop {
        if (m >= 0x800000u) {
            break;
        }
        m = m << 1u;
        s = s + 1;
    }
    return s;
}

fn q_scale_up_pow2(x: f32, s: i32) -> f32 {
    let b = bitcast<u32>(x);
    return bitcast<f32>((b & 0x80000000u) | ((b & 0x7fffffffu) << u32(s)));
}

fn q_scaled_product(v_bf16: u32, inv_up: f32, u_up: i32) -> f32 {
    let v = bf16_decode(v_bf16);
    let t = q_subnormal_shift(v);
    var p = (q_scale_up_pow2(v, t) * inv_up) * q_pow2(-(t + u_up));
    if ((bitcast<u32>(p) & 0x7fffffffu) > F32_INF) {
        p = -E2M1_MAX;
    }
    return clamp(p, -E2M1_MAX, E2M1_MAX);
}

fn q_encode_scale(stored: f32, local_scale: f32) -> u32 {
    let sb = bitcast<u32>(stored);
    let lb = bitcast<u32>(local_scale) & 0x7fffffffu;
    if ((sb >> 31u) != 0u || lb == 0u || lb >= F32_INF) {
        return 0u;
    }
    let sp = q_norm_parts(stored);
    let lp = q_norm_parts(local_scale);
    let af = f32(sp.x);
    let bf = f32(lp.x);
    let hi = af * bf;
    let lo = fma(af, bf, -hi);
    let hp = q_norm_parts(hi);
    var m = u32(hp.x);
    var e = sp.y + lp.y + (hp.y - 46);
    let phi = lo * q_pow2(23 - hp.y);
    if (e < -6) {
        let shs = u32(14 - e);
        if (shs >= 25u) {
            return 0u;
        }
        let qs = m >> shs;
        let rems = m & ((1u << shs) - 1u);
        let halfs = 1u << (shs - 1u);
        var subq = qs;
        if (rems > halfs || (rems == halfs && phi >= 0.0)) {
            subq = subq + 1u;
        }
        if (subq == 0u) {
            return 0u;
        }
        if (subq <= 7u) {
            return subq;
        }
        return 0x08u;
    }
    if (e < -126) {
        let sh = u32(-126 - e);
        if (sh >= 25u) {
            return 0u;
        }
        let q = m >> sh;
        let rem = m & ((1u << sh) - 1u);
        let half = 1u << (sh - 1u);
        var n = q;
        if (rem > half || (rem == half && (phi > 0.0 || (phi == 0.0 && (q & 1u) == 1u)))) {
            n = n + 1u;
        }
        if (n == 0u) {
            return 0u;
        }
        m = n;
        e = -126;
        loop {
            if (m >= 0x800000u) {
                break;
            }
            m = m << 1u;
            e = e - 1;
        }
    }
    if (e > 127) {
        return 0u;
    }
    if (e > 8 || (e == 8 && m > 0xe00000u)) {
        return 0x7eu;
    }
    let frac = m - 0x800000u;
    var mant = (frac + 0x80000u) >> 20u;
    var e_out = e;
    if (mant > 7u) {
        mant = 0u;
        e_out = e_out + 1;
    }
    let biased = u32(clamp(e_out + 7, 0, 15));
    let encoded = (biased << 3u) | mant;
    return select(encoded, 0x7eu, encoded == 0x7fu);
}

fn q_encode_scale_ref(stored: f32, local_scale: f32) -> u32 {
    let scale = stored * local_scale;
    let bits = bitcast<u32>(scale);
    if ((bits & 0x7f800000u) == 0x7f800000u || (bits >> 31u) != 0u || (bits & 0x7fffffffu) == 0u) {
        return 0u;
    }
    let clamped = min(scale, 448.0);
    let cb = bitcast<u32>(clamped);
    let ef = cb >> 23u;
    if (clamped < 0.015625) {
        var mm: u32;
        var ee: i32;
        if (ef == 0u) {
            mm = cb & 0x7fffffu;
            ee = -126;
        } else {
            mm = (cb & 0x7fffffu) | 0x800000u;
            ee = i32(ef) - 127;
        }
        let sh = 14 - ee;
        if (sh >= 25) {
            return 0u;
        }
        let shu = u32(sh);
        let sub = (mm + (1u << (shu - 1u))) >> shu;
        if (sub == 0u) {
            return 0u;
        }
        if (sub <= 7u) {
            return sub;
        }
        return 0x08u;
    }
    let frac = cb & 0x7fffffu;
    var mant = (frac + 0x80000u) >> 20u;
    var e_out = i32(ef) - 127;
    if (mant > 7u) {
        mant = 0u;
        e_out = e_out + 1;
    }
    let biased = u32(clamp(e_out + 7, 1, 15));
    let encoded = (biased << 3u) | mant;
    return select(encoded, 0x7eu, encoded == 0x7fu);
}
