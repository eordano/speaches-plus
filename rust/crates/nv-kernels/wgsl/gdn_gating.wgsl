struct GdnGatingParams {
    total: u32,
    num_heads: u32,
};

@group(0) @binding(0) var<storage, read> gdn_a: array<u32>;
@group(0) @binding(1) var<storage, read> gdn_b: array<u32>;
@group(0) @binding(2) var<storage, read> gdn_alog: array<u32>;
@group(0) @binding(3) var<storage, read> gdn_bias: array<u32>;
@group(0) @binding(4) var<storage, read_write> gdn_g: array<f32>;
@group(0) @binding(5) var<storage, read_write> gdn_beta: array<u32>;
@group(0) @binding(6) var<uniform> gdn_params: GdnGatingParams;

const GDN_BLOCK: u32 = 256u;

fn gdn_exp_parts(x: f32) -> vec2<f32> {
    let j = fma(x, bitcast<f32>(0x3BBB989Du), 0.5);
    let sv = clamp(j, 0.0, 1.0);
    let ph = sv * 252.0;
    let pl = fma(sv, 252.0, -ph);
    var fl = floor(ph);
    if (ph == fl && pl < 0.0) {
        fl = fl - 1.0;
    }
    let m = fl + 12582913.0;
    let n = m + bitcast<f32>(0xCB40007Fu);
    let r0 = fma(x, bitcast<f32>(0x3FB8AA3Bu), -n);
    let r = fma(x, bitcast<f32>(0x32A57060u), r0);
    let scale = bitcast<f32>(bitcast<u32>(m) << 23u);
    return vec2<f32>(exp2(r), scale);
}

fn gdn_mul_rn(x: f32, y: f32) -> f32 {
    let xb = bitcast<u32>(x);
    let yb = bitcast<u32>(y);
    let sign = (xb ^ yb) & 0x80000000u;
    let xe = (xb >> 23u) & 0xffu;
    let ye = (yb >> 23u) & 0xffu;
    let xm = xb & 0x7fffffu;
    let ym = yb & 0x7fffffu;
    let x_zero = xe == 0u && xm == 0u;
    let y_zero = ye == 0u && ym == 0u;
    if ((xe == 255u && xm != 0u) || (ye == 255u && ym != 0u)) {
        return bitcast<f32>(0x7fc00000u);
    }
    if (xe == 255u || ye == 255u) {
        if (x_zero || y_zero) {
            return bitcast<f32>(0x7fc00000u);
        }
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if (x_zero || y_zero) {
        return bitcast<f32>(sign);
    }
    var xs = xm | 0x800000u;
    var xexp = i32(xe);
    if (xe == 0u) {
        let sh = countLeadingZeros(xm) - 8u;
        xs = xm << sh;
        xexp = 1 - i32(sh);
    }
    var ys = ym | 0x800000u;
    var yexp = i32(ye);
    if (ye == 0u) {
        let sh = countLeadingZeros(ym) - 8u;
        ys = ym << sh;
        yexp = 1 - i32(sh);
    }
    let a0 = xs & 0xffffu;
    let a1 = xs >> 16u;
    let b0 = ys & 0xffffu;
    let b1 = ys >> 16u;
    let p00 = a0 * b0;
    let mid = a0 * b1 + a1 * b0;
    let lo = p00 + (mid << 16u);
    let carry = select(0u, 1u, lo < p00);
    let hi = a1 * b1 + (mid >> 16u) + carry;
    var msb = 46;
    if ((hi & 0x8000u) != 0u) {
        msb = 47;
    }
    let pexp = xexp + yexp - 300;
    let e = pexp + msb + 127;
    if (e >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    var s = msb - 23;
    if (e < 1) {
        s = -pexp - 149;
    }
    if (s >= 50) {
        return bitcast<f32>(sign);
    }
    var sig = 0u;
    var rbit = 0u;
    var sticky = false;
    if (s < 32) {
        let us = u32(s);
        sig = (hi << (32u - us)) | (lo >> us);
        rbit = (lo >> (us - 1u)) & 1u;
        sticky = (lo & ((1u << (us - 1u)) - 1u)) != 0u;
    } else if (s == 32) {
        sig = hi;
        rbit = lo >> 31u;
        sticky = (lo & 0x7fffffffu) != 0u;
    } else {
        let t = u32(s - 32);
        sig = hi >> t;
        rbit = (hi >> (t - 1u)) & 1u;
        sticky = ((hi & ((1u << (t - 1u)) - 1u)) != 0u) || (lo != 0u);
    }
    if (rbit == 1u && (sticky || (sig & 1u) == 1u)) {
        sig = sig + 1u;
    }
    let val = (u32(max(e - 1, 0)) << 23u) + sig;
    if (val >= 0x7f800000u) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    return bitcast<f32>(sign | val);
}

fn gdn_negate(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ 0x80000000u);
}

fn gdn_expf(x: f32) -> f32 {
    let p = gdn_exp_parts(x);
    return gdn_mul_rn(p.x, p.y);
}

fn gdn_fast_expf(x: f32) -> f32 {
    return exp2(x * bitcast<f32>(0x3FB8AA3Bu));
}

fn gdn_rcp_rn(x: f32) -> f32 {
    let xb = bitcast<u32>(x);
    let sign = xb & 0x80000000u;
    let xe = (xb >> 23u) & 0xffu;
    let xm = xb & 0x7fffffu;
    if (xe == 255u) {
        if (xm != 0u) {
            return bitcast<f32>(0x7fc00000u);
        }
        return bitcast<f32>(sign);
    }
    if (xe == 0u && xm == 0u) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    var xs = xm | 0x800000u;
    var xexp = i32(xe);
    if (xe == 0u) {
        let sh = countLeadingZeros(xm) - 8u;
        xs = xm << sh;
        xexp = 1 - i32(sh);
    }
    var rem = 0x800000u;
    var q = 0u;
    if (rem >= xs) {
        rem = rem - xs;
        q = 1u;
    }
    for (var i = 0u; i < 24u; i = i + 1u) {
        rem = rem << 1u;
        q = q << 1u;
        if (rem >= xs) {
            rem = rem - xs;
            q = q + 1u;
        }
    }
    var sig = q;
    var e = 253 - xexp;
    if (sig >= 0x1000000u) {
        sig = sig >> 1u;
        e = e + 1;
    }
    if (e >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    var s = 0;
    if (e < 1) {
        s = 1 - e;
    }
    if (s >= 26) {
        return bitcast<f32>(sign);
    }
    var up = false;
    if (s == 0) {
        let dbl = rem << 1u;
        up = dbl > xs || (dbl == xs && (sig & 1u) == 1u);
    } else {
        let us = u32(s);
        let half = 1u << (us - 1u);
        let low = sig & ((1u << us) - 1u);
        sig = sig >> us;
        up = low > half || (low == half && (rem != 0u || (sig & 1u) == 1u));
    }
    if (up) {
        sig = sig + 1u;
    }
    let val = (u32(max(e - 1, 0)) << 23u) + sig;
    if (val >= 0x7f800000u) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    return bitcast<f32>(sign | val);
}

fn gdn_sigmoidf(x: f32) -> f32 {
    let p = gdn_exp_parts(-x);
    return gdn_rcp_rn(fma(p.x, p.y, 1.0));
}

fn gdn_log1pf(a: f32) -> f32 {
    let ab0 = bitcast<u32>(a);
    if ((ab0 & 0x7f800000u) == 0x7f800000u && (ab0 & 0x007fffffu) != 0u) {
        return bitcast<f32>(ab0 | 0x00400000u);
    }
    let s = a + 1.0;
    let bv = s - a;
    let av = s - bv;
    let err = (a - av) + (1.0 - bv);
    var u = s;
    if (err < 0.0) {
        u = bitcast<f32>(bitcast<u32>(s) - 1u);
    }
    let ex = (bitcast<u32>(u) + 0xC0C00000u) & 0xFF800000u;
    let ab = bitcast<u32>(a);
    let scaled = bitcast<f32>(ab - ex);
    let inv = bitcast<f32>(1082130432u - ex);
    let base = fma(inv, 0.25, -1.0);
    let m = base + scaled;
    let ef = f32(bitcast<i32>(ex)) * bitcast<f32>(0x34000000u);
    var p = fma(m, bitcast<f32>(0xBD39BF78u), bitcast<f32>(0x3DD80012u));
    p = fma(p, m, bitcast<f32>(0xBE0778E0u));
    p = fma(p, m, bitcast<f32>(0x3E146475u));
    p = fma(p, m, bitcast<f32>(0xBE2A68DDu));
    p = fma(p, m, bitcast<f32>(0x3E4CAF9Eu));
    p = fma(p, m, bitcast<f32>(0xBE800042u));
    p = fma(p, m, bitcast<f32>(0x3EAAAAE6u));
    p = fma(p, m, -0.5);
    let t = m * p;
    let q = fma(t, m, m);
    return fma(ef, bitcast<f32>(0x3F317218u), q);
}

fn gdn_softplus_safe(x: f32) -> f32 {
    if (x > 20.0) {
        return x;
    }
    if (x < -20.0) {
        return gdn_expf(x);
    }
    return gdn_log1pf(gdn_expf(x));
}

fn gdn_index(wg: vec3<u32>, nwg: vec3<u32>, tid: vec3<u32>) -> u32 {
    return (wg.y * nwg.x + wg.x) * GDN_BLOCK + tid.x;
}

@compute @workgroup_size(256)
fn gdn_gating_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = gdn_index(wg, nwg, tid);
    if (idx >= gdn_params.total) {
        return;
    }
    let h = idx % gdn_params.num_heads;

    let sp = gdn_softplus_safe(bitcast<f32>(gdn_a[idx]) + bitcast<f32>(gdn_bias[h]));
    let g = gdn_negate(gdn_mul_rn(sp, gdn_fast_expf(bitcast<f32>(gdn_alog[h]))));
    let beta = gdn_sigmoidf(bitcast<f32>(gdn_b[idx]));

    gdn_g[idx] = g;
    gdn_beta[idx] = bitcast<u32>(beta);
}

@compute @workgroup_size(256)
fn gdn_gating_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = gdn_index(wg, nwg, tid);
    if (idx >= gdn_params.total) {
        return;
    }
    let h = idx % gdn_params.num_heads;

    let sp = gdn_softplus_safe(bf16_decode(gdn_a[idx]) + bf16_decode(gdn_bias[h]));
    let g = gdn_negate(gdn_mul_rn(sp, gdn_fast_expf(bf16_decode(gdn_alog[h]))));
    let beta = gdn_sigmoidf(bf16_decode(gdn_b[idx]));

    gdn_g[idx] = g;
    gdn_beta[idx] = bf16_encode(beta);
}
