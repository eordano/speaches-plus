
const HP_WG: u32 = 256u;
const HP_WARP: u32 = 32u;
const HP_E4M3_MAX: f32 = 448.0;

struct HpParams {
    n_q: u32,
    n_kv: u32,
    head_dim: u32,
    half_dim: u32,
    eps: f32,
    words: u32,
    out_words: u32,
    ring: u32,
};

@group(0) @binding(0) var<storage, read> hp_qa: array<u32>;
@group(0) @binding(1) var<storage, read> hp_ka: array<u32>;
@group(0) @binding(2) var<storage, read> hp_va: array<u32>;
@group(0) @binding(3) var<storage, read> hp_qn: array<u32>;
@group(0) @binding(4) var<storage, read> hp_kn: array<u32>;
@group(0) @binding(5) var<storage, read> hp_vn: array<u32>;
@group(0) @binding(6) var<storage, read_write> hp_qout: array<f32>;
@group(0) @binding(7) var<storage, read_write> hp_kq: array<u32>;
@group(0) @binding(8) var<storage, read_write> hp_ks: array<f32>;
@group(0) @binding(9) var<storage, read_write> hp_vq: array<u32>;
@group(0) @binding(10) var<storage, read_write> hp_vs: array<f32>;
@group(0) @binding(11) var<storage, read> hp_cos: array<f32>;
@group(0) @binding(12) var<storage, read> hp_sin: array<f32>;
@group(0) @binding(13) var<storage, read> hp_pos: array<i32>;
@group(0) @binding(14) var<storage, read> hp_kvstart: array<i32>;
@group(0) @binding(15) var<uniform> hp_params: HpParams;

var<workgroup> hp_rs: array<f32, 256>;
var<workgroup> hp_rs_shared: f32;
var<workgroup> hp_mx: array<f32, 256>;
var<workgroup> hp_mx_shared: f32;
var<workgroup> hp_a: array<u32, HP_MAXW>;
var<workgroup> hp_b: array<u32, HP_MAXW>;

fn hp_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn hp_rms_reduce(lid: u32, local: f32) -> f32 {
    hp_rs[lid] = local;
    workgroupBarrier();
    for (var stride = HP_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (HP_WARP - 1u)) < stride) {
            hp_rs[lid] = hp_rs[lid] + hp_rs[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let a = (hp_rs[0u] + hp_rs[128u]) + (hp_rs[64u] + hp_rs[192u]);
        let b = (hp_rs[32u] + hp_rs[160u]) + (hp_rs[96u] + hp_rs[224u]);
        let sum = a + b;
        let mean = hp_div_rn(sum, f32(hp_params.head_dim));
        hp_rs_shared = inverseSqrt(hp_params.eps + mean);
    }
    workgroupBarrier();
    return hp_rs_shared;
}

fn hp_max_reduce(lid: u32, local: f32) -> f32 {
    hp_mx[lid] = local;
    workgroupBarrier();
    for (var stride = HP_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = hp_mx[lid + stride];
            if (other > hp_mx[lid]) {
                hp_mx[lid] = other;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        hp_mx_shared = hp_mx[0];
    }
    workgroupBarrier();
    return hp_mx_shared;
}

fn hp_e4m3(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let sign = (b >> 31u) << 7u;
    let mag = b & 0x7fffffffu;
    if (mag > 0x7f800000u) {
        return sign | 0x7fu;
    }
    let e = i32(mag >> 23u) - 127;
    if (e >= -6) {
        let lsb = (mag >> 20u) & 1u;
        let r = mag + 0x7ffffu + lsb;
        let e2 = i32(r >> 23u) - 127;
        let m2 = (r >> 20u) & 7u;
        if (e2 > 8 || (e2 == 8 && m2 == 7u)) {
            return sign | 0x7eu;
        }
        return sign | (u32(e2 + 7) << 3u) | m2;
    }
    let s = u32(14 - e);
    if (s >= 32u) {
        return sign;
    }
    let full = 0x800000u | (mag & 0x7fffffu);
    let q = full >> s;
    let round_bit = (full >> (s - 1u)) & 1u;
    let rest = full & ((1u << (s - 1u)) - 1u);
    var n = q;
    if (round_bit == 1u && (rest != 0u || (q & 1u) == 1u)) {
        n = n + 1u;
    }
    return sign | n;
}

fn hp_src(kind: u32, idx: u32) -> u32 {
    if (kind == 0u) {
        return hp_qa[idx];
    }
    if (kind == 1u) {
        return hp_ka[idx];
    }
    return hp_va[idx];
}

fn hp_wt(kind: u32, idx: u32) -> u32 {
    if (kind == 0u) {
        return hp_qn[idx];
    }
    if (kind == 1u) {
        return hp_kn[idx];
    }
    return hp_vn[idx];
}

fn hp_a_at(elem: u32) -> f32 {
    let word = hp_a[elem >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (elem & 1u) == 1u);
}

fn hp_b_at(elem: u32) -> f32 {
    let word = hp_b[elem >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (elem & 1u) == 1u);
}

fn hp_rotate(row_base: u32, elem: u32) -> f32 {
    let half = hp_params.half_dim;
    if (elem < half) {
        let c = hp_cos[row_base + elem];
        let s = hp_sin[row_base + elem];
        let a = hp_a_at(elem);
        let b = hp_a_at(elem + half);
        return fma(a, c, -(b * s));
    }
    let pair = elem - half;
    let c = hp_cos[row_base + pair];
    let s = hp_sin[row_base + pair];
    let a = hp_a_at(pair);
    let b = hp_a_at(elem);
    return fma(a, s, b * c);
}

@compute @workgroup_size(256)
fn g4w_head_prep(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let flat = wg.x + wg.y * nwg.x;
    let n_q = hp_params.n_q;
    let n_kv = hp_params.n_kv;
    if (flat >= n_q + 2u * n_kv) {
        return;
    }
    var kind = 0u;
    var head = flat;
    if (flat >= n_q + n_kv) {
        kind = 2u;
        head = flat - n_q - n_kv;
    } else if (flat >= n_q) {
        kind = 1u;
        head = flat - n_q;
    }

    let lid = tid.x;
    let hd = hp_params.head_dim;
    let words = hp_params.words;
    let base = head * words;

    var local = 0.0;
    for (var i = lid; i < hd; i = i + HP_WG) {
        let word = hp_src(kind, base + (i >> 1u));
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms = hp_rms_reduce(lid, local);
    for (var i = lid; i < words; i = i + HP_WG) {
        let xw = hp_src(kind, base + i);
        let ww = hp_wt(kind, i);
        let lo = bf16_lo(xw) * rms * bf16_lo(ww);
        let hi = bf16_hi(xw) * rms * bf16_hi(ww);
        hp_a[i] = bf16_pack(lo, hi);
    }
    workgroupBarrier();

    if (kind == 2u) {
        for (var i = lid; i < words; i = i + HP_WG) {
            hp_b[i] = hp_a[i];
        }
    } else {
        let row_base = u32(hp_pos[0]) * hp_params.half_dim;
        let q_base = head * hd;
        for (var i = lid; i < words; i = i + HP_WG) {
            let elem = i * 2u;
            let lo = hp_rotate(row_base, elem);
            let hi = hp_rotate(row_base, elem + 1u);
            let word = bf16_pack(lo, hi);
            hp_b[i] = word;
            if (kind == 0u) {
                hp_qout[q_base + elem] = bf16_lo(word);
                hp_qout[q_base + elem + 1u] = bf16_hi(word);
            }
        }
    }
    workgroupBarrier();
    if (kind == 0u) {
        return;
    }

    var lmax = 0.0;
    for (var d = lid; d < hd; d = d + HP_WG) {
        let a = abs(hp_b_at(d));
        if (a > lmax) {
            lmax = a;
        }
    }
    let amax = hp_max_reduce(lid, lmax);
    let positive = amax > 0.0;
    let scale = select(1.0, hp_div_rn(amax, HP_E4M3_MAX), positive);
    let inv_scale = select(1.0, hp_div_rn(HP_E4M3_MAX, amax), positive);

    var slot = u32(max(hp_kvstart[0], 0));
    if (hp_params.ring > 0u) {
        slot = slot % hp_params.ring;
    }
    let sidx = slot * n_kv + head;
    if (lid == 0u) {
        if (kind == 1u) {
            hp_ks[sidx] = scale;
        } else {
            hp_vs[sidx] = scale;
        }
    }
    let dst = (sidx * hd) >> 2u;
    for (var w = lid; w < hp_params.out_words; w = w + HP_WG) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = hp_b_at(d0 + j);
            packed = packed | (hp_e4m3(v * inv_scale) << (8u * j));
        }
        if (kind == 1u) {
            hp_kq[dst + w] = packed;
        } else {
            hp_vq[dst + w] = packed;
        }
    }
}
