struct DwcParams {
    batch: u32,
    channels: u32,
    seq_len: u32,
    ksize: u32,
    n_elems: u32,
    n_words: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> dwc_x: array<u32>;
@group(0) @binding(1) var<storage, read> dwc_w: array<u32>;
@group(0) @binding(2) var<storage, read_write> dwc_y: array<u32>;
@group(0) @binding(3) var<uniform> dwc_params: DwcParams;

const DWC_WORKGROUP: u32 = 256u;
const DWC_RR_SCALE: f32 = -0.005724980030208826;
const DWC_RR_STEPS: f32 = 252.0;
const DWC_RR_BIAS: f32 = 12582913.0;
const DWC_RR_UNBIAS: f32 = 12583039.0;
const DWC_LOG2E_HI: f32 = -1.4426950216293335;
const DWC_LOG2E_LO: f32 = -1.925963033500011e-8;
const DWC_SUBNORMAL_DEN: f32 = 8.5070591730234616e37;
const DWC_F32_INF: u32 = 0x7f800000u;
const DWC_P2_M149_BITS: u32 = 1249902592u;
const DWC_ULP_BIAS: u32 = 1258291200u;

fn dwc_rcp_rn(d: f32) -> f32 {
    let r0 = 1.0 / d;
    if (!(r0 > 0.0)) {
        return r0;
    }
    let r1 = fma(r0, fma(-d, r0, 1.0), r0);
    let t = fma(r1, fma(-d, r1, 1.0), r1);
    let rem = fma(-d, t, 1.0);
    let tb = bitcast<u32>(t);
    if (rem > 0.0) {
        let up = bitcast<f32>(tb + 1u);
        let dulp = bitcast<f32>(bitcast<u32>(d) + (tb & 0x7f800000u) - DWC_ULP_BIAS);
        let g = dulp - (rem + rem);
        if (g < 0.0 || (g == 0.0 && (tb & 1u) == 1u)) {
            return up;
        }
    } else if (rem < 0.0) {
        let dn = bitcast<f32>(tb - 1u);
        let dulp = bitcast<f32>(bitcast<u32>(d) + ((tb - 1u) & 0x7f800000u) - DWC_ULP_BIAS);
        let g = dulp + (rem + rem);
        if (g < 0.0 || (g == 0.0 && (tb & 1u) == 1u)) {
            return dn;
        }
    }
    return t;
}

fn dwc_mul_rcp_subnormal(acc: f32, d: f32) -> f32 {
    let db = bitcast<u32>(d);
    if (db >= DWC_F32_INF) {
        return acc * (1.0 / d);
    }
    let de = db >> 23u;
    let dm = (db & 0x007fffffu) | 0x00800000u;
    var q = 0u;
    var r = 0u;
    for (var i = 0u; i < 48u; i = i + 1u) {
        r = (r << 1u) | select(0u, 1u, i == 0u);
        q = q << 1u;
        if (r >= dm) {
            r = r - dm;
            q = q + 1u;
        }
    }
    let sh = de - 252u;
    let base = q >> sh;
    let lo = q & ((1u << sh) - 1u);
    let num = (lo * dm + r) << 1u;
    let cmp = dm << sh;
    var m = base;
    if (num > cmp || (num == cmp && (base & 1u) == 1u)) {
        m = m + 1u;
    }
    let p = acc * f32(m);
    return bitcast<f32>(bitcast<u32>(p) - DWC_P2_M149_BITS);
}

fn dwc_floor_steps(t: f32) -> f32 {
    let p = t * DWC_RR_STEPS;
    let pe = fma(t, DWC_RR_STEPS, -p);
    let fp = floor(p);
    return select(fp, fp - 1.0, p == fp && pe < 0.0);
}

fn dwc_silu(acc: f32) -> f32 {
    let t = min(max(fma(acc, DWC_RR_SCALE, 0.5), 0.0), 1.0);
    let j = DWC_RR_BIAS + dwc_floor_steps(t);
    let n = j - DWC_RR_UNBIAS;
    var f = fma(acc, DWC_LOG2E_HI, -n);
    f = fma(acc, DWC_LOG2E_LO, f);
    let scale = bitcast<f32>(bitcast<u32>(j) << 23u);
    let den = fma(exp2(f), scale, 1.0);
    if (den >= DWC_SUBNORMAL_DEN) {
        return dwc_mul_rcp_subnormal(acc, den);
    }
    return acc * dwc_rcp_rn(den);
}

fn dwc_x_word(i: u32) -> u32 {
    return (dwc_x[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
}

fn dwc_w_word(i: u32) -> u32 {
    return (dwc_w[i >> 1u] >> (16u * (i & 1u))) & 0xffffu;
}

fn dwc_u64_shl(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 32u) {
        return vec2<u32>(0u, v.x << (s - 32u));
    }
    return vec2<u32>(v.x << s, (v.y << s) | (v.x >> (32u - s)));
}

fn dwc_u64_shr(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 64u) {
        return vec2<u32>(0u, 0u);
    }
    if (s >= 32u) {
        return vec2<u32>(v.y >> (s - 32u), 0u);
    }
    return vec2<u32>((v.x >> s) | (v.y << (32u - s)), v.y >> s);
}

fn dwc_u64_shr_jam(v: vec2<u32>, s: u32) -> vec2<u32> {
    if (s == 0u) {
        return v;
    }
    if (s >= 64u) {
        return vec2<u32>(select(0u, 1u, (v.x | v.y) != 0u), 0u);
    }
    let kept = dwc_u64_shr(v, s);
    var lost = false;
    if (s >= 32u) {
        let d = s - 32u;
        lost = v.x != 0u;
        if (d > 0u) {
            lost = lost || (v.y & ((1u << d) - 1u)) != 0u;
        }
    } else {
        lost = (v.x & ((1u << s) - 1u)) != 0u;
    }
    return vec2<u32>(kept.x | select(0u, 1u, lost), kept.y);
}

fn dwc_u64_add(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    let lo = a.x + b.x;
    return vec2<u32>(lo, a.y + b.y + select(0u, 1u, lo < a.x));
}

fn dwc_u64_sub(a: vec2<u32>, b: vec2<u32>) -> vec2<u32> {
    return vec2<u32>(a.x - b.x, a.y - b.y - select(0u, 1u, a.x < b.x));
}

fn dwc_u64_ge(a: vec2<u32>, b: vec2<u32>) -> bool {
    if (a.y != b.y) {
        return a.y > b.y;
    }
    return a.x >= b.x;
}

fn dwc_u64_msb(v: vec2<u32>) -> i32 {
    if (v.y != 0u) {
        return 32 + i32(firstLeadingBit(v.y));
    }
    if (v.x != 0u) {
        return i32(firstLeadingBit(v.x));
    }
    return -1;
}

fn dwc_u64_bit(v: vec2<u32>, n: u32) -> u32 {
    if (n >= 64u) {
        return 0u;
    }
    if (n >= 32u) {
        return (v.y >> (n - 32u)) & 1u;
    }
    return (v.x >> n) & 1u;
}

fn dwc_u64_any_below(v: vec2<u32>, n: u32) -> bool {
    if (n == 0u) {
        return false;
    }
    if (n >= 64u) {
        return (v.x | v.y) != 0u;
    }
    if (n >= 32u) {
        let d = n - 32u;
        if (v.x != 0u) {
            return true;
        }
        if (d == 0u) {
            return false;
        }
        return (v.y & ((1u << d) - 1u)) != 0u;
    }
    return (v.x & ((1u << n) - 1u)) != 0u;
}

struct DwcDec {
    s: u32,
    m: u32,
    e: i32,
};

fn dwc_dec_bf16(b: u32) -> DwcDec {
    let e = (b >> 7u) & 0xffu;
    let m = b & 0x7fu;
    if (e == 0u) {
        return DwcDec((b >> 15u) & 1u, m, -133);
    }
    return DwcDec((b >> 15u) & 1u, m | 0x80u, i32(e) - 134);
}

fn dwc_dec_f32(b: u32) -> DwcDec {
    let e = (b >> 23u) & 0xffu;
    let m = b & 0x7fffffu;
    if (e == 0u) {
        return DwcDec(b >> 31u, m, -149);
    }
    return DwcDec(b >> 31u, m | 0x800000u, i32(e) - 150);
}

fn dwc_soft_fma(xb: u32, wb: u32, ab: u32) -> u32 {
    let xe = (xb >> 7u) & 0xffu;
    let we = (wb >> 7u) & 0xffu;
    let ae = (ab >> 23u) & 0xffu;
    let x_zero = (xb & 0x7fffu) == 0u;
    let w_zero = (wb & 0x7fffu) == 0u;
    if (xe == 0xffu || we == 0xffu || ae == 0xffu) {
        let x_nan = xe == 0xffu && (xb & 0x7fu) != 0u;
        let w_nan = we == 0xffu && (wb & 0x7fu) != 0u;
        let a_nan = ae == 0xffu && (ab & 0x7fffffu) != 0u;
        if (x_nan || w_nan || a_nan) {
            return 0x7fc00000u;
        }
        let psign = ((xb >> 15u) & 1u) ^ ((wb >> 15u) & 1u);
        if (xe == 0xffu || we == 0xffu) {
            if (x_zero || w_zero) {
                return 0x7fc00000u;
            }
            if (ae == 0xffu && (ab >> 31u) != psign) {
                return 0x7fc00000u;
            }
            return (psign << 31u) | 0x7f800000u;
        }
        return ab;
    }
    let dx = dwc_dec_bf16(xb);
    let dw = dwc_dec_bf16(wb);
    let da = dwc_dec_f32(ab);
    let ps = dx.s ^ dw.s;
    let pm = dx.m * dw.m;
    let pe = dx.e + dw.e;
    if (pm == 0u) {
        if (da.m == 0u && ps != da.s) {
            return 0u;
        }
        return ab;
    }
    let top = max(pe, da.e);
    let base = top - 38;
    var pv = vec2<u32>(pm, 0u);
    var av = vec2<u32>(da.m, 0u);
    let sp = pe - base;
    let sa = da.e - base;
    if (sp >= 0) {
        pv = dwc_u64_shl(pv, u32(sp));
    } else {
        pv = dwc_u64_shr_jam(pv, u32(-sp));
    }
    if (sa >= 0) {
        av = dwc_u64_shl(av, u32(sa));
    } else {
        av = dwc_u64_shr_jam(av, u32(-sa));
    }
    var mag = vec2<u32>(0u, 0u);
    var rs = ps;
    if (ps == da.s) {
        mag = dwc_u64_add(pv, av);
    } else if (dwc_u64_ge(pv, av)) {
        mag = dwc_u64_sub(pv, av);
    } else {
        mag = dwc_u64_sub(av, pv);
        rs = da.s;
    }
    let h = dwc_u64_msb(mag);
    if (h < 0) {
        return 0u;
    }
    var shift = h - 23;
    let sub_shift = -149 - base;
    if (sub_shift > shift) {
        shift = sub_shift;
    }
    var q = 0u;
    if (shift <= 0) {
        q = dwc_u64_shl(mag, u32(-shift)).x;
    } else {
        let s = u32(shift);
        q = dwc_u64_shr(mag, s).x;
        let rbit = dwc_u64_bit(mag, s - 1u);
        let sticky = dwc_u64_any_below(mag, s - 1u);
        if (rbit == 1u && (sticky || (q & 1u) == 1u)) {
            q = q + 1u;
        }
    }
    var be = base + shift + 150;
    if (q >= 0x1000000u) {
        q = q >> 1u;
        be = be + 1;
    }
    let sign = rs << 31u;
    if (be >= 255) {
        return sign | 0x7f800000u;
    }
    if (q < 0x800000u) {
        return sign | q;
    }
    return sign | (u32(be) << 23u) | (q & 0x7fffffu);
}

fn dwc_soft_acc_bits(idx: u32, t: u32, wbase: u32, kmax: u32) -> u32 {
    var ab = 0u;
    for (var k = 0u; k < dwc_params.ksize; k = k + 1u) {
        let back = kmax - k;
        if (t >= back) {
            ab = dwc_soft_fma(dwc_x_word(idx - back), dwc_w_word(wbase + k), ab);
        }
    }
    return ab;
}

fn dwc_silu_from_bits(ab: u32) -> f32 {
    let e = (ab >> 23u) & 0xffu;
    if (e <= 1u) {
        let m = select(ab & 0x7fffffu, (ab & 0x7fffffu) | 0x800000u, e == 1u);
        let h = m >> 1u;
        return bitcast<f32>((ab & 0x80000000u) | (h + ((m & 1u) & (h & 1u))));
    }
    return dwc_silu(bitcast<f32>(ab));
}

fn dwc_point_bits(idx: u32) -> u32 {
    let t = idx % dwc_params.seq_len;
    let c = (idx / dwc_params.seq_len) % dwc_params.channels;
    let wbase = c * dwc_params.ksize;
    let kmax = dwc_params.ksize - 1u;
    var acc = 0.0;
    var risky = false;
    var nonfinite = false;
    var seen = false;
    for (var k = 0u; k < dwc_params.ksize; k = k + 1u) {
        let back = kmax - k;
        if (t >= back) {
            let xb = dwc_x_word(idx - back);
            let wb = dwc_w_word(wbase + k);
            acc = fma(bitcast<f32>(xb << 16u), bitcast<f32>(wb << 16u), acc);
            let ex = (xb >> 7u) & 0xffu;
            let ew = (wb >> 7u) & 0xffu;
            nonfinite = nonfinite || ex == 0xffu || ew == 0xffu;
            if ((xb & 0x7fffu) != 0u && (wb & 0x7fffu) != 0u) {
                seen = true;
                risky = risky || ex == 0u || ew == 0u || (i32(ex) + i32(ew)) < 174;
            }
            risky = risky || (seen && (bitcast<u32>(acc) & 0x7fffffffu) < 0x17800000u);
        }
    }
    var ab = bitcast<u32>(acc);
    if (risky || nonfinite) {
        ab = dwc_soft_acc_bits(idx, t, wbase, kmax);
    }
    let mag = ab & 0x7fffffffu;
    if (mag > 0x7f800000u) {
        return 0x7fffu;
    }
    if (mag == 0x7f800000u) {
        return select(0x7f80u, 0x7fffu, (ab >> 31u) == 1u);
    }
    return bf16_encode(dwc_silu_from_bits(ab));
}

@compute @workgroup_size(256)
fn depthwise_conv1d_silu_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let w = (wid.y * ng.x + wid.x) * DWC_WORKGROUP + lid.x;
    if (w >= dwc_params.n_words) {
        return;
    }
    let e0 = w * 2u;
    let lo = dwc_point_bits(e0);
    var hi = 0u;
    if (e0 + 1u < dwc_params.n_elems) {
        hi = dwc_point_bits(e0 + 1u);
    }
    dwc_y[w] = lo | (hi << 16u);
}
