struct VfParams {
    k: u32,
    nq: u32,
    nkv: u32,
    hd: u32,
    ring: u32,
    stride: u32,
    q_off: u32,
    k_off: u32,
    v_off: u32,
    eps: f32,
    pad0: u32,
    pad1: u32,
};

struct VfrParams {
    hidden: u32,
    batch: u32,
    words: u32,
    eps: f32,
};

struct VfsParams {
    hidden: u32,
    batch: u32,
    words: u32,
    eps: f32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> vf_qkv: array<u32>;
@group(0) @binding(1) var<storage, read> vf_qw: array<u32>;
@group(0) @binding(2) var<storage, read> vf_kw: array<u32>;
@group(0) @binding(3) var<storage, read> vf_vw: array<u32>;
@group(0) @binding(4) var<storage, read> vf_cos: array<f32>;
@group(0) @binding(5) var<storage, read> vf_sin: array<f32>;
@group(0) @binding(6) var<storage, read> vf_pos: array<i32>;
@group(0) @binding(7) var<storage, read_write> vf_qout: array<u32>;
@group(0) @binding(8) var<storage, read_write> vf_kc: array<u32>;
@group(0) @binding(9) var<storage, read_write> vf_vc: array<u32>;
@group(0) @binding(10) var<storage, read_write> vf_kscale: array<f32>;
@group(0) @binding(11) var<storage, read_write> vf_vscale: array<f32>;
@group(0) @binding(12) var<storage, read> vf_nc: array<i32>;
@group(0) @binding(13) var<uniform> vf_p: VfParams;

@group(0) @binding(14) var<storage, read> vfr_x: array<u32>;
@group(0) @binding(15) var<storage, read> vfr_res: array<u32>;
@group(0) @binding(16) var<storage, read> vfr_w1: array<u32>;
@group(0) @binding(17) var<storage, read> vfr_w2: array<u32>;
@group(0) @binding(18) var<storage, read_write> vfr_sum: array<u32>;
@group(0) @binding(19) var<storage, read_write> vfr_norm: array<u32>;
@group(0) @binding(20) var<uniform> vfr_p: VfrParams;

@group(0) @binding(21) var<storage, read> vfs_x: array<u32>;
@group(0) @binding(22) var<storage, read> vfs_res: array<u32>;
@group(0) @binding(23) var<storage, read> vfs_w: array<u32>;
@group(0) @binding(24) var<storage, read_write> vfs_out: array<u32>;
@group(0) @binding(25) var<uniform> vfs_p: VfsParams;

const VF_BLOCK: u32 = 256u;
const VF_WARP: u32 = 32u;
const VF_FP8_MAX: f32 = 448.0;

var<workgroup> vf_sa: array<f32, 512>;
var<workgroup> vf_sb: array<f32, 512>;
var<workgroup> vf_red: array<f32, 256>;
var<workgroup> vf_bcast: f32;

fn vf_div_rn(a: f32, b: f32) -> f32 {
    let ua = bitcast<u32>(a);
    let ub = bitcast<u32>(b);
    let sign = (ua ^ ub) & 0x80000000u;
    let amag = ua & 0x7fffffffu;
    let bmag = ub & 0x7fffffffu;
    if (amag > 0x7f800000u || bmag > 0x7f800000u
        || (amag == 0u && bmag == 0u)
        || (amag == 0x7f800000u && bmag == 0x7f800000u)) {
        return bitcast<f32>(0x7fc00000u);
    }
    if (amag == 0u || bmag == 0x7f800000u) {
        return bitcast<f32>(sign);
    }
    if (amag == 0x7f800000u || bmag == 0u) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    var ea = i32(amag >> 23u);
    var ma = amag & 0x7fffffu;
    if (ea == 0) {
        let sh = countLeadingZeros(ma) - 8u;
        ma = ma << sh;
        ea = 1 - i32(sh);
    } else {
        ma = ma | 0x800000u;
    }
    var eb = i32(bmag >> 23u);
    var mb = bmag & 0x7fffffu;
    if (eb == 0) {
        let sh = countLeadingZeros(mb) - 8u;
        mb = mb << sh;
        eb = 1 - i32(sh);
    } else {
        mb = mb | 0x800000u;
    }
    var q = 0u;
    var rem = ma;
    for (var i = 0u; i < 26u; i = i + 1u) {
        q = q << 1u;
        if (rem >= mb) {
            rem = rem - mb;
            q = q | 1u;
        }
        rem = rem << 1u;
    }
    var mant: u32;
    var round: u32;
    var sticky = u32(rem != 0u);
    var be: i32;
    if (q >= 0x2000000u) {
        mant = q >> 2u;
        round = (q >> 1u) & 1u;
        sticky = sticky | (q & 1u);
        be = ea - eb + 127;
    } else {
        mant = q >> 1u;
        round = q & 1u;
        be = ea - eb + 126;
    }
    if (be >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if (be <= 0) {
        let s = u32(1 - be);
        if (s > 24u) {
            return bitcast<f32>(sign);
        }
        sticky = sticky | round | u32((mant & ((1u << (s - 1u)) - 1u)) != 0u);
        round = (mant >> (s - 1u)) & 1u;
        mant = mant >> s;
        be = 0;
    }
    if (round == 1u && (sticky != 0u || (mant & 1u) == 1u)) {
        mant = mant + 1u;
    }
    var bits: u32;
    if (be == 0) {
        bits = sign | mant;
    } else {
        bits = sign | ((u32(be) << 23u) + (mant - 0x800000u));
    }
    return bitcast<f32>(bits);
}

fn vf_bf16_round(x: f32) -> f32 {
    return bitcast<f32>(bf16_encode(x) << 16u);
}

fn vf_encode_e4m3(x: f32) -> u32 {
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

fn vf_warp_rms_reduce(lid: u32, local: f32, n: f32, eps: f32) -> f32 {
    vf_red[lid] = local;
    workgroupBarrier();
    for (var stride = VF_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (VF_WARP - 1u)) < stride) {
            vf_red[lid] = vf_red[lid] + vf_red[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let a = (vf_red[0u] + vf_red[128u]) + (vf_red[64u] + vf_red[192u]);
        let b = (vf_red[32u] + vf_red[160u]) + (vf_red[96u] + vf_red[224u]);
        let sum = a + b;
        vf_bcast = inverseSqrt(vf_div_rn(sum, n) + eps);
    }
    workgroupBarrier();
    return vf_bcast;
}

fn vf_tree_rms_reduce(lid: u32, local: f32, n: f32, eps: f32) -> f32 {
    vf_red[lid] = local;
    workgroupBarrier();
    for (var stride = VF_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            vf_red[lid] = vf_red[lid] + vf_red[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        vf_bcast = inverseSqrt(vf_div_rn(vf_red[0u], n) + eps);
    }
    workgroupBarrier();
    return vf_bcast;
}

fn vf_reduce_max(lid: u32, local: f32) -> f32 {
    vf_red[lid] = local;
    workgroupBarrier();
    for (var stride = VF_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = vf_red[lid + stride];
            if (other > vf_red[lid]) {
                vf_red[lid] = other;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        vf_bcast = vf_red[0u];
    }
    workgroupBarrier();
    return vf_bcast;
}

fn vf_qkv_at(idx: u32) -> f32 {
    let word = vf_qkv[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vf_qw_at(idx: u32) -> f32 {
    let word = vf_qw[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vf_kw_at(idx: u32) -> f32 {
    let word = vf_kw[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vf_vw_at(idx: u32) -> f32 {
    let word = vf_vw[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vf_row_rms(lid: u32, row: u32, hd: u32, eps: f32) -> f32 {
    var local = 0.0;
    for (var i = lid; i < hd; i = i + VF_BLOCK) {
        let v = vf_qkv_at(row + i);
        local = fma(v, v, local);
    }
    return vf_warp_rms_reduce(lid, local, f32(hd), eps);
}

@compute @workgroup_size(256)
fn verify_qkv_prep(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let token = wg.y;
    if (token >= vf_p.k) {
        return;
    }
    let hd = vf_p.hd;
    let half = hd >> 1u;
    let lid = tid.x;
    let pos = u32(vf_pos[token]);
    let crow = pos * half;

    if (h < vf_p.nq) {
        let row = token * vf_p.stride + vf_p.q_off + h * hd;
        let rms = vf_row_rms(lid, row, hd, vf_p.eps);
        for (var i = lid; i < hd; i = i + VF_BLOCK) {
            let v = vf_qkv_at(row + i) * rms * vf_qw_at(i);
            vf_sa[i] = vf_bf16_round(v);
        }
        workgroupBarrier();
        if (lid < half) {
            let a = vf_sa[lid];
            let b = vf_sa[lid + half];
            let c = vf_cos[crow + lid];
            let s = vf_sin[crow + lid];
            vf_sb[lid] = fma(a, c, -(b * s));
            vf_sb[lid + half] = fma(a, s, b * c);
        }
        workgroupBarrier();
        let obase = (token * vf_p.nq + h) * hd;
        for (var w = lid; w < half; w = w + VF_BLOCK) {
            vf_qout[(obase >> 1u) + w] = bf16_pack(vf_sb[w * 2u], vf_sb[w * 2u + 1u]);
        }
        return;
    }

    let kvh = h - vf_p.nq;
    if (kvh >= vf_p.nkv) {
        return;
    }
    if (vf_p.ring > 0u && token + vf_p.ring < vf_p.k) {
        return;
    }
    let krow = token * vf_p.stride + vf_p.k_off + kvh * hd;
    let vrow = token * vf_p.stride + vf_p.v_off + kvh * hd;

    let rms_k = vf_row_rms(lid, krow, hd, vf_p.eps);
    for (var i = lid; i < hd; i = i + VF_BLOCK) {
        let v = vf_qkv_at(krow + i) * rms_k * vf_kw_at(i);
        vf_sa[i] = vf_bf16_round(v);
    }
    let rms_v = vf_row_rms(lid, vrow, hd, vf_p.eps);
    for (var i = lid; i < hd; i = i + VF_BLOCK) {
        let v = vf_qkv_at(vrow + i) * rms_v * vf_vw_at(i);
        vf_sb[i] = vf_bf16_round(v);
    }
    workgroupBarrier();

    if (lid < half) {
        let a = vf_sa[lid];
        let b = vf_sa[lid + half];
        let c = vf_cos[crow + lid];
        let s = vf_sin[crow + lid];
        vf_sa[lid] = vf_bf16_round(fma(a, c, -(b * s)));
        vf_sa[lid + half] = vf_bf16_round(fma(a, s, b * c));
    }
    workgroupBarrier();

    var lm_k = 0.0;
    var lm_v = 0.0;
    for (var d = lid; d < hd; d = d + VF_BLOCK) {
        lm_k = max(lm_k, abs(vf_sa[d]));
        lm_v = max(lm_v, abs(vf_sb[d]));
    }
    let amax_k = vf_reduce_max(lid, lm_k);
    let amax_v = vf_reduce_max(lid, lm_v);
    let inv_k = select(1.0, vf_div_rn(VF_FP8_MAX, amax_k), amax_k > 0.0);
    let inv_v = select(1.0, vf_div_rn(VF_FP8_MAX, amax_v), amax_v > 0.0);

    var slot = u32(vf_nc[0] + i32(token));
    if (vf_p.ring > 0u) {
        slot = slot % vf_p.ring;
    }
    if (lid == 0u) {
        vf_kscale[slot * vf_p.nkv + kvh] =
            select(1.0, vf_div_rn(amax_k, VF_FP8_MAX), amax_k > 0.0);
        vf_vscale[slot * vf_p.nkv + kvh] =
            select(1.0, vf_div_rn(amax_v, VF_FP8_MAX), amax_v > 0.0);
    }
    let dbase = (slot * vf_p.nkv + kvh) * hd;
    let quads = hd >> 2u;
    for (var w = lid; w < quads; w = w + VF_BLOCK) {
        var pk = 0u;
        var pv = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            pk = pk | (vf_encode_e4m3(vf_sa[w * 4u + j] * inv_k) << (8u * j));
            pv = pv | (vf_encode_e4m3(vf_sb[w * 4u + j] * inv_v) << (8u * j));
        }
        vf_kc[(dbase >> 2u) + w] = pk;
        vf_vc[(dbase >> 2u) + w] = pv;
    }
}

fn vfr_x_at(idx: u32) -> f32 {
    let word = vfr_x[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vfr_res_at(idx: u32) -> f32 {
    let word = vfr_res[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vfr_w1_at(idx: u32) -> f32 {
    let word = vfr_w1[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vfr_w2_at(idx: u32) -> f32 {
    let word = vfr_w2[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vfr_sum_of(idx: u32, col: u32, rms1: f32) -> f32 {
    let t = vfr_x_at(idx) * rms1 * vfr_w1_at(col);
    return vf_bf16_round(t) + vfr_res_at(idx);
}

@compute @workgroup_size(256)
fn rmsnorm2_residual_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= vfr_p.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = vfr_p.hidden;
    let words = vfr_p.words;
    let base_e = row * hidden;
    let base_w = row * words;

    var local1 = 0.0;
    for (var i = lid; i < hidden; i = i + VF_BLOCK) {
        let v = vfr_x_at(base_e + i);
        local1 = fma(v, v, local1);
    }
    let rms1 = vf_warp_rms_reduce(lid, local1, f32(hidden), vfr_p.eps);

    var local2 = 0.0;
    for (var i = lid; i < hidden; i = i + VF_BLOCK) {
        let s = vfr_sum_of(base_e + i, i, rms1);
        local2 = fma(s, s, local2);
    }
    let rms2 = vf_tree_rms_reduce(lid, local2, f32(hidden), vfr_p.eps);

    for (var w = lid; w < words; w = w + VF_BLOCK) {
        let i0 = base_e + w * 2u;
        let i1 = i0 + 1u;
        let s0 = vfr_sum_of(i0, w * 2u, rms1);
        let s1 = vfr_sum_of(i1, w * 2u + 1u, rms1);
        vfr_sum[base_w + w] = bf16_pack(s0, s1);
        let sb0 = vf_bf16_round(s0);
        let sb1 = vf_bf16_round(s1);
        vfr_norm[base_w + w] = bf16_pack(
            sb0 * rms2 * vfr_w2_at(w * 2u),
            sb1 * rms2 * vfr_w2_at(w * 2u + 1u)
        );
    }
}

fn vfs_x_at(idx: u32) -> f32 {
    let word = vfs_x[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vfs_res_at(idx: u32) -> f32 {
    let word = vfs_res[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn vfs_w_at(idx: u32) -> f32 {
    let word = vfs_w[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

@compute @workgroup_size(256)
fn rmsnorm_residual_scale_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= vfs_p.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = vfs_p.hidden;
    let words = vfs_p.words;
    let base_e = row * hidden;
    let base_w = row * words;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + VF_BLOCK) {
        let v = vfs_x_at(base_e + i);
        local = fma(v, v, local);
    }
    let rms = vf_warp_rms_reduce(lid, local, f32(hidden), vfs_p.eps);

    let scale = vfs_p.scale;
    for (var w = lid; w < words; w = w + VF_BLOCK) {
        let i0 = base_e + w * 2u;
        let i1 = i0 + 1u;
        let n0 = vf_bf16_round(vfs_x_at(i0) * rms * vfs_w_at(w * 2u));
        let n1 = vf_bf16_round(vfs_x_at(i1) * rms * vfs_w_at(w * 2u + 1u));
        let lo = (vfs_res_at(i0) + n0) * scale;
        let hi = (vfs_res_at(i1) + n1) * scale;
        vfs_out[base_w + w] = bf16_pack(lo, hi);
    }
}
