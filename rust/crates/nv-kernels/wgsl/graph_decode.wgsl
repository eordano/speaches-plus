const GD_BLOCK: u32 = 128u;
const GD_ARGMAX_BLOCKS: u32 = 256u;

var<workgroup> gd_scratch: array<f32, 128>;
var<workgroup> gd_warp: array<f32, 4>;
var<workgroup> gd_total: f32;

fn gd_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn gd_pos_inf() -> f32 {
    return bitcast<f32>(0x7f800000u);
}

fn gd_linear(gid: vec3<u32>, nwg: vec3<u32>, wgsize: u32) -> u32 {
    return gid.x + gid.y * nwg.x * wgsize;
}

fn gd_row(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

fn gd_block_sum(lid: u32, v: f32) -> f32 {
    workgroupBarrier();
    gd_scratch[lid] = v;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        if ((lid & 31u) < o) {
            gd_scratch[lid] = gd_scratch[lid] + gd_scratch[lid + o];
        }
        workgroupBarrier();
    }
    if ((lid & 31u) == 0u) {
        gd_warp[lid >> 5u] = gd_scratch[lid];
    }
    workgroupBarrier();
    if (lid == 0u) {
        let a = gd_warp[0] + gd_warp[2];
        let b = gd_warp[1] + gd_warp[3];
        gd_total = a + b;
    }
    workgroupBarrier();
    return gd_total;
}

fn gd_bf16_encode(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let exp_carry = (((b >> 23u) & 0xffu) + 1u) >> 8u;
    let mant_nz = min(b & 0x007fffffu, 1u);
    let mask = 0u - (exp_carry & mant_nz);
    let r = 0x7fffu + ((b >> 16u) & 1u);
    return (((b + r) >> 16u) & ~mask) | (0x7fffu & mask);
}

fn gd_div_rn(a: f32, b: f32) -> f32 {
    let y0 = 1.0 / b;
    let y = fma(fma(-b, y0, 1.0), y0, y0);
    let q = a * y;
    let r = fma(-b, q, a);
    let refined = fma(r, y, q);
    let qb = bitcast<u32>(q);
    let mask = 0u - ((((qb >> 23u) & 0xffu) + 1u) >> 8u);
    return bitcast<f32>((bitcast<u32>(refined) & ~mask) | (qb & mask));
}

struct GdEwParams {
    n: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> gd_ew_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> gd_ew_out: array<u32>;
@group(0) @binding(2) var<uniform> gd_ew_p: GdEwParams;

@compute @workgroup_size(128)
fn cast_bf16_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let i = gd_linear(gid, nwg, GD_BLOCK);
    if (i >= gd_ew_p.n) {
        return;
    }
    gd_ew_out[i] = bitcast<u32>(bf16_decode(gd_ew_in[i]));
}

@compute @workgroup_size(128)
fn cast_f32_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let i = gd_linear(gid, nwg, GD_BLOCK);
    if (i >= gd_ew_p.n) {
        return;
    }
    gd_ew_out[i] = gd_bf16_encode(bitcast<f32>(gd_ew_in[i]));
}

@compute @workgroup_size(128)
fn cast_scale_bf16_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let i = gd_linear(gid, nwg, GD_BLOCK);
    if (i >= gd_ew_p.n) {
        return;
    }
    gd_ew_out[i] = bitcast<u32>(bf16_decode(gd_ew_in[i]) * gd_ew_p.scale);
}

@group(0) @binding(3) var<storage, read> gd_add_a: array<f32>;
@group(0) @binding(4) var<storage, read> gd_add_b: array<f32>;
@group(0) @binding(5) var<storage, read_write> gd_add_y: array<f32>;
@group(0) @binding(6) var<uniform> gd_add_p: GdEwParams;

@compute @workgroup_size(128)
fn add_scale_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let i = gd_linear(gid, nwg, GD_BLOCK);
    if (i >= gd_add_p.n) {
        return;
    }
    gd_add_y[i] = (gd_add_a[i] + gd_add_b[i]) * gd_add_p.scale;
}

@group(0) @binding(7) var<storage, read> gd_gelu_gate: array<u32>;
@group(0) @binding(8) var<storage, read> gd_gelu_pli: array<f32>;
@group(0) @binding(9) var<storage, read_write> gd_gelu_y: array<u32>;
@group(0) @binding(10) var<uniform> gd_gelu_p: GdEwParams;

@compute @workgroup_size(128)
fn gelu_mul_bf16f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let i = gd_linear(gid, nwg, GD_BLOCK);
    if (i >= gd_gelu_p.n) {
        return;
    }
    let g = bf16_decode(gd_gelu_gate[i]);
    let t3 = g * bitcast<f32>(0x3D372713u);
    let t4 = g * t3;
    let t5 = fma(g, t4, g);
    let inner = t5 * bitcast<f32>(0x3F4C422Au);
    let t = nv_tanhf(inner);
    let hg = g * 0.5;
    let s = t + 1.0;
    let gelu = hg * s;
    gd_gelu_y[i] = gd_bf16_encode(gd_gelu_pli[i] * gelu);
}

@group(0) @binding(11) var<storage, read_write> gd_pos: array<i32>;
@group(0) @binding(12) var<storage, read_write> gd_rope_pos: array<i32>;

@compute @workgroup_size(1)
fn incr_pos() {
    gd_pos[0] = gd_pos[0] + 1;
}

@compute @workgroup_size(1)
fn incr_pos_rope() {
    gd_rope_pos[0] = gd_pos[0];
    gd_pos[0] = gd_pos[0] + 1;
}

struct GdKvParams {
    nkv: u32,
    hd: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(13) var<storage, read> gd_wkv_sk: array<f32>;
@group(0) @binding(14) var<storage, read> gd_wkv_sv: array<f32>;
@group(0) @binding(15) var<storage, read_write> gd_wkv_ck: array<f32>;
@group(0) @binding(16) var<storage, read_write> gd_wkv_cv: array<f32>;
@group(0) @binding(17) var<storage, read> gd_wkv_pos: array<i32>;
@group(0) @binding(18) var<uniform> gd_wkv_p: GdKvParams;

@compute @workgroup_size(128)
fn write_kv_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let kvh = gd_row(wg, nwg);
    if (kvh >= gd_wkv_p.nkv) {
        return;
    }
    let slot = gd_wkv_pos[0] - 1;
    if (slot < 0) {
        return;
    }
    let hd = gd_wkv_p.hd;
    let dst = (u32(slot) * gd_wkv_p.nkv + kvh) * hd;
    let src = kvh * hd;
    for (var d = tid.x; d < hd; d = d + GD_BLOCK) {
        gd_wkv_ck[dst + d] = gd_wkv_sk[src + d];
        gd_wkv_cv[dst + d] = gd_wkv_sv[src + d];
    }
}

struct GdRmsParams {
    rows: u32,
    dim: u32,
    eps: f32,
    pad0: u32,
};

@group(0) @binding(19) var<storage, read> gd_rms_x: array<u32>;
@group(0) @binding(20) var<storage, read> gd_rms_w: array<u32>;
@group(0) @binding(21) var<storage, read_write> gd_rms_y: array<f32>;
@group(0) @binding(22) var<uniform> gd_rms_p: GdRmsParams;

@compute @workgroup_size(128)
fn rms_no_weight_bf16_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let r = gd_row(wg, nwg);
    if (r >= gd_rms_p.rows) {
        return;
    }
    let lid = tid.x;
    let dim = gd_rms_p.dim;
    let base = r * dim;
    var partial = 0.0;
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        let v = bf16_decode(gd_rms_x[base + d]);
        partial = fma(v, v, partial);
    }
    let total = gd_block_sum(lid, partial);
    let inv = inverseSqrt(gd_div_rn(total, f32(dim)) + gd_rms_p.eps);
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        gd_rms_y[base + d] = inv * bf16_decode(gd_rms_x[base + d]);
    }
}

@compute @workgroup_size(128)
fn rmsnorm_bf16w_f32out(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let r = gd_row(wg, nwg);
    if (r >= gd_rms_p.rows) {
        return;
    }
    let lid = tid.x;
    let dim = gd_rms_p.dim;
    let base = r * dim;
    var partial = 0.0;
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        let v = bf16_decode(gd_rms_x[base + d]);
        partial = fma(v, v, partial);
    }
    let total = gd_block_sum(lid, partial);
    let inv = inverseSqrt(gd_div_rn(total, f32(dim)) + gd_rms_p.eps);
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        let n = inv * bf16_decode(gd_rms_x[base + d]);
        gd_rms_y[base + d] = n * bf16_decode(gd_rms_w[d]);
    }
}

@compute @workgroup_size(128)
fn rstd_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let r = gd_row(wg, nwg);
    if (r >= gd_rms_p.rows) {
        return;
    }
    let lid = tid.x;
    let dim = gd_rms_p.dim;
    let base = r * dim;
    var partial = 0.0;
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        let v = bf16_decode(gd_rms_x[base + d]);
        partial = fma(v, v, partial);
    }
    let total = gd_block_sum(lid, partial);
    if (lid == 0u) {
        gd_rms_y[r] = inverseSqrt(gd_div_rn(total, f32(dim)) + gd_rms_p.eps);
    }
}

struct GdApplyParams {
    n: u32,
    dim: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(23) var<storage, read> gd_ap_x: array<u32>;
@group(0) @binding(24) var<storage, read> gd_ap_w: array<u32>;
@group(0) @binding(25) var<storage, read> gd_ap_rstd: array<f32>;
@group(0) @binding(26) var<storage, read_write> gd_ap_y: array<u32>;
@group(0) @binding(27) var<uniform> gd_ap_p: GdApplyParams;

@compute @workgroup_size(128)
fn rms_apply_bf16(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>
) {
    let i = gd_linear(gid, nwg, GD_BLOCK);
    if (i >= gd_ap_p.n) {
        return;
    }
    let t = bf16_decode(gd_ap_x[i]) * gd_ap_rstd[i / gd_ap_p.dim];
    gd_ap_y[i] = gd_bf16_encode(t * bf16_decode(gd_ap_w[i % gd_ap_p.dim]));
}

struct GdRasParams {
    rows: u32,
    dim: u32,
    eps: f32,
    scale: f32,
    eps_next: f32,
    flags: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(28) var<storage, read> gd_ras_x: array<u32>;
@group(0) @binding(29) var<storage, read> gd_ras_w: array<u32>;
@group(0) @binding(30) var<storage, read> gd_ras_res: array<u32>;
@group(0) @binding(31) var<storage, read_write> gd_ras_y: array<u32>;
@group(0) @binding(32) var<storage, read_write> gd_ras_rstd: array<f32>;
@group(0) @binding(33) var<storage, read> gd_ras_nextw: array<u32>;
@group(0) @binding(34) var<storage, read_write> gd_ras_normed: array<u32>;
@group(0) @binding(35) var<uniform> gd_ras_p: GdRasParams;

@compute @workgroup_size(128)
fn rmsnorm_add_scale_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let r = gd_row(wg, nwg);
    if (r >= gd_ras_p.rows) {
        return;
    }
    let lid = tid.x;
    let dim = gd_ras_p.dim;
    let base = r * dim;
    var partial = 0.0;
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        let v = bf16_decode(gd_ras_x[base + d]);
        partial = fma(v, v, partial);
    }
    let inv = inverseSqrt(gd_div_rn(gd_block_sum(lid, partial), f32(dim)) + gd_ras_p.eps);
    var out_sq = 0.0;
    for (var d = lid; d < dim; d = d + GD_BLOCK) {
        let t = inv * bf16_decode(gd_ras_x[base + d]);
        let o = fma(t, bf16_decode(gd_ras_w[d]), bf16_decode(gd_ras_res[base + d])) * gd_ras_p.scale;
        let ob = gd_bf16_encode(o);
        gd_ras_y[base + d] = ob;
        let ofv = bf16_decode(ob);
        out_sq = fma(ofv, ofv, out_sq);
    }
    if ((gd_ras_p.flags & 3u) != 0u) {
        let total = gd_block_sum(lid, out_sq);
        let inv2 = inverseSqrt(gd_div_rn(total, f32(dim)) + gd_ras_p.eps_next);
        if ((gd_ras_p.flags & 1u) != 0u && lid == 0u) {
            gd_ras_rstd[r] = inv2;
        }
        if ((gd_ras_p.flags & 2u) != 0u) {
            for (var d = lid; d < dim; d = d + GD_BLOCK) {
                let of2 = bf16_decode(gd_ras_y[base + d]);
                gd_ras_normed[base + d] = gd_bf16_encode(of2 * inv2 * bf16_decode(gd_ras_nextw[d]));
            }
        }
    }
}

struct GdQkvParams {
    nh: u32,
    nkv: u32,
    hd: u32,
    delta: i32,
    eps: f32,
    has_kv: u32,
    pad0: u32,
    pad1: u32,
};

var<workgroup> gd_qkv_ns: array<f32, 512>;

@group(0) @binding(36) var<storage, read> gd_qkv_in: array<u32>;
@group(0) @binding(37) var<storage, read> gd_qkv_qw: array<u32>;
@group(0) @binding(38) var<storage, read> gd_qkv_kw: array<u32>;
@group(0) @binding(39) var<storage, read> gd_qkv_cos: array<f32>;
@group(0) @binding(40) var<storage, read> gd_qkv_sin: array<f32>;
@group(0) @binding(41) var<storage, read> gd_qkv_rope_pos: array<i32>;
@group(0) @binding(42) var<storage, read> gd_qkv_cache_pos: array<i32>;
@group(0) @binding(43) var<storage, read_write> gd_qkv_qout: array<f32>;
@group(0) @binding(44) var<storage, read_write> gd_qkv_kcache: array<u32>;
@group(0) @binding(45) var<storage, read_write> gd_qkv_vcache: array<u32>;
@group(0) @binding(46) var<uniform> gd_qkv_p: GdQkvParams;

fn gd_rope_at(d: u32, half: u32, cbase: u32) -> f32 {
    let i = select(d - half, d, d < half);
    let a = gd_qkv_ns[i];
    let b = gd_qkv_ns[i + half];
    let c = gd_qkv_cos[cbase + i];
    let s = gd_qkv_sin[cbase + i];
    if (d < half) {
        return fma(a, c, -(b * s));
    }
    return fma(a, s, b * c);
}

@compute @workgroup_size(128)
fn qkv_prep(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = gd_row(wg, nwg);
    let nh = gd_qkv_p.nh;
    let nkv = gd_qkv_p.nkv;
    let has_kv = gd_qkv_p.has_kv != 0u;
    let total_heads = select(nh, nh + 2u * nkv, has_kv);
    if (head >= total_heads) {
        return;
    }
    let lid = tid.x;
    let hd = gd_qkv_p.hd;
    let xbase = head * hd;
    var kind = 0u;
    if (head >= nh) {
        kind = select(2u, 1u, head < nh + nkv);
    }

    var partial = 0.0;
    for (var d = lid; d < hd; d = d + GD_BLOCK) {
        let v = bf16_decode(gd_qkv_in[xbase + d]);
        partial = fma(v, v, partial);
    }
    let inv = inverseSqrt(gd_div_rn(gd_block_sum(lid, partial), f32(hd)) + gd_qkv_p.eps);
    for (var d = lid; d < hd; d = d + GD_BLOCK) {
        var n = inv * bf16_decode(gd_qkv_in[xbase + d]);
        if (kind == 0u) {
            n = n * bf16_decode(gd_qkv_qw[d]);
        } else if (kind == 1u) {
            n = n * bf16_decode(gd_qkv_kw[d]);
        }
        gd_qkv_ns[d] = n;
    }
    workgroupBarrier();

    let half = hd >> 1u;
    if (kind == 0u) {
        let p = gd_qkv_rope_pos[0] - gd_qkv_p.delta;
        let cbase = u32(p) * half;
        for (var d = lid; d < hd; d = d + GD_BLOCK) {
            gd_qkv_qout[head * hd + d] = gd_rope_at(d, half, cbase);
        }
        return;
    }
    let slot = gd_qkv_cache_pos[0] - 1 - gd_qkv_p.delta;
    if (slot < 0) {
        return;
    }
    let kvh = head - nh - select(0u, nkv, kind == 2u);
    let dst = (u32(slot) * nkv + kvh) * hd;
    if (kind == 1u) {
        let p = gd_qkv_rope_pos[0] - gd_qkv_p.delta;
        let cbase = u32(p) * half;
        for (var d = lid; d < hd; d = d + GD_BLOCK) {
            gd_qkv_kcache[dst + d] = gd_bf16_encode(gd_rope_at(d, half, cbase));
        }
    } else {
        for (var d = lid; d < hd; d = d + GD_BLOCK) {
            gd_qkv_vcache[dst + d] = gd_bf16_encode(gd_qkv_ns[d]);
        }
    }
}

struct GdArgmaxParams {
    n: u32,
    nparts: u32,
    ring_mask: i32,
    has_ring: u32,
};

var<workgroup> gd_am_val: array<f32, 256>;
var<workgroup> gd_am_idx: array<i32, 256>;

@group(0) @binding(47) var<storage, read> gd_am_logits: array<u32>;
@group(0) @binding(48) var<storage, read_write> gd_am_part_val: array<f32>;
@group(0) @binding(49) var<storage, read_write> gd_am_part_idx: array<i32>;
@group(0) @binding(50) var<uniform> gd_am_p: GdArgmaxParams;
@group(0) @binding(51) var<storage, read> gd_am_pos: array<i32>;
@group(0) @binding(52) var<storage, read_write> gd_am_token: array<u32>;
@group(0) @binding(53) var<storage, read_write> gd_am_ring: array<u32>;

@compute @workgroup_size(128)
fn argmax_bf16_stage1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let lid = tid.x;
    var best = gd_neg_inf();
    var bidx = 2147483647;
    let n = i32(gd_am_p.n);
    let stride = i32(GD_ARGMAX_BLOCKS * GD_BLOCK);
    var i = i32(wg.x * GD_BLOCK + lid);
    while (i < n) {
        let v = bf16_decode(gd_am_logits[u32(i)]);
        if (v > best || (v == best && i < bidx)) {
            best = v;
            bidx = i;
        }
        i = i + stride;
    }
    gd_am_val[lid] = best;
    gd_am_idx[lid] = bidx;
    workgroupBarrier();
    for (var s = GD_BLOCK / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let ov = gd_am_val[lid + s];
            let oi = gd_am_idx[lid + s];
            if (ov > gd_am_val[lid] || (ov == gd_am_val[lid] && oi < gd_am_idx[lid])) {
                gd_am_val[lid] = ov;
                gd_am_idx[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        gd_am_part_val[wg.x] = gd_am_val[0];
        gd_am_part_idx[wg.x] = gd_am_idx[0];
    }
}

@compute @workgroup_size(256)
fn argmax_bf16_stage2(@builtin(local_invocation_id) tid: vec3<u32>) {
    let lid = tid.x;
    if (lid < gd_am_p.nparts) {
        gd_am_val[lid] = gd_am_part_val[lid];
        gd_am_idx[lid] = gd_am_part_idx[lid];
    } else {
        gd_am_val[lid] = gd_neg_inf();
        gd_am_idx[lid] = 2147483647;
    }
    workgroupBarrier();
    for (var s = GD_ARGMAX_BLOCKS / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let ov = gd_am_val[lid + s];
            let oi = gd_am_idx[lid + s];
            if (ov > gd_am_val[lid] || (ov == gd_am_val[lid] && oi < gd_am_idx[lid])) {
                gd_am_val[lid] = ov;
                gd_am_idx[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let t = u32(gd_am_idx[0]);
        gd_am_token[0] = t;
        if (gd_am_p.has_ring != 0u) {
            gd_am_ring[u32((gd_am_pos[0] - 1) & gd_am_p.ring_mask)] = t;
        }
    }
}

struct GdArgmaxRowsParams {
    rows: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(54) var<storage, read> gd_amf_logits: array<f32>;
@group(0) @binding(55) var<storage, read_write> gd_amf_part_val: array<f32>;
@group(0) @binding(56) var<storage, read_write> gd_amf_part_idx: array<i32>;
@group(0) @binding(57) var<storage, read_write> gd_amf_out: array<u32>;
@group(0) @binding(58) var<uniform> gd_amf_p: GdArgmaxRowsParams;

@compute @workgroup_size(128)
fn argmax_f32_rows_stage1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let lid = tid.x;
    let row = wg.y;
    let rowbase = row * gd_amf_p.n;
    var best = gd_neg_inf();
    var bidx = 2147483647;
    let n = i32(gd_amf_p.n);
    let stride = i32(GD_ARGMAX_BLOCKS * GD_BLOCK);
    var i = i32(wg.x * GD_BLOCK + lid);
    while (i < n) {
        let v = gd_amf_logits[rowbase + u32(i)];
        let finite = (v == v) && (abs(v) != gd_pos_inf());
        if (finite && (v > best || (v == best && i < bidx))) {
            best = v;
            bidx = i;
        }
        i = i + stride;
    }
    gd_am_val[lid] = best;
    gd_am_idx[lid] = bidx;
    workgroupBarrier();
    for (var s = GD_BLOCK / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let ov = gd_am_val[lid + s];
            let oi = gd_am_idx[lid + s];
            if (ov > gd_am_val[lid] || (ov == gd_am_val[lid] && oi < gd_am_idx[lid])) {
                gd_am_val[lid] = ov;
                gd_am_idx[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        gd_amf_part_val[row * GD_ARGMAX_BLOCKS + wg.x] = gd_am_val[0];
        gd_amf_part_idx[row * GD_ARGMAX_BLOCKS + wg.x] = gd_am_idx[0];
    }
}

@compute @workgroup_size(256)
fn argmax_f32_rows_stage2(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = gd_row(wg, nwg);
    if (row >= gd_amf_p.rows) {
        return;
    }
    let lid = tid.x;
    gd_am_val[lid] = gd_amf_part_val[row * GD_ARGMAX_BLOCKS + lid];
    gd_am_idx[lid] = gd_amf_part_idx[row * GD_ARGMAX_BLOCKS + lid];
    workgroupBarrier();
    for (var s = GD_ARGMAX_BLOCKS / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let ov = gd_am_val[lid + s];
            let oi = gd_am_idx[lid + s];
            if (ov > gd_am_val[lid] || (ov == gd_am_val[lid] && oi < gd_am_idx[lid])) {
                gd_am_val[lid] = ov;
                gd_am_idx[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        gd_amf_out[row] = select(u32(gd_am_idx[0]), 0u, gd_am_idx[0] == 2147483647);
    }
}

struct GdArgmaxCapParams {
    n: u32,
    cap: f32,
    inv_cap: f32,
    softcap: u32,
};

@group(0) @binding(65) var<storage, read> gd_amc_pk: array<u32>;
@group(0) @binding(66) var<storage, read_write> gd_amc_logits: array<f32>;
@group(0) @binding(67) var<uniform> gd_amc_p: GdArgmaxCapParams;

@compute @workgroup_size(128)
fn argmax_softcap_bf16_stage1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let lid = tid.x;
    var best = gd_neg_inf();
    var bidx = 2147483647;
    let n = i32(gd_amc_p.n);
    let stride = i32(GD_ARGMAX_BLOCKS * GD_BLOCK);
    var i = i32(wg.x * GD_BLOCK + lid);
    while (i < n) {
        let w = gd_amc_pk[u32(i) >> 1u];
        var x = bf16_lo(w);
        if ((u32(i) & 1u) != 0u) {
            x = bf16_hi(w);
        }
        var v = x;
        if (gd_amc_p.softcap != 0u) {
            v = nv_tanhf(x * gd_amc_p.inv_cap) * gd_amc_p.cap;
        }
        gd_amc_logits[u32(i)] = v;
        let finite = (v == v) && (abs(v) != gd_pos_inf());
        if (finite && (v > best || (v == best && i < bidx))) {
            best = v;
            bidx = i;
        }
        i = i + stride;
    }
    gd_am_val[lid] = best;
    gd_am_idx[lid] = bidx;
    workgroupBarrier();
    for (var s = GD_BLOCK / 2u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            let ov = gd_am_val[lid + s];
            let oi = gd_am_idx[lid + s];
            if (ov > gd_am_val[lid] || (ov == gd_am_val[lid] && oi < gd_am_idx[lid])) {
                gd_am_val[lid] = ov;
                gd_am_idx[lid] = oi;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        gd_amf_part_val[wg.x] = gd_am_val[0];
        gd_amf_part_idx[wg.x] = gd_am_idx[0];
    }
}

@group(0) @binding(59) var<storage, read> gd_tm_map: array<u32>;
@group(0) @binding(60) var<storage, read> gd_tm_idx: array<u32>;
@group(0) @binding(61) var<storage, read_write> gd_tm_out: array<u32>;

@compute @workgroup_size(1)
fn token_map_u32() {
    gd_tm_out[0] = gd_tm_map[gd_tm_idx[0]];
}

struct GdZeroParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(62) var<storage, read_write> gd_mz_data: array<u32>;
@group(0) @binding(63) var<storage, read> gd_mz_desc: array<u32>;
@group(0) @binding(64) var<uniform> gd_mz_p: GdZeroParams;

@compute @workgroup_size(256)
fn multi_zero_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let b = gd_row(wg, nwg);
    if (b >= gd_mz_p.n) {
        return;
    }
    let off = gd_mz_desc[b * 2u];
    let cnt = gd_mz_desc[b * 2u + 1u];
    let n4 = cnt / 8u;
    for (var i = tid.x; i < n4; i = i + 256u) {
        let w = off + i * 4u;
        gd_mz_data[w] = 0u;
        gd_mz_data[w + 1u] = 0u;
        gd_mz_data[w + 2u] = 0u;
        gd_mz_data[w + 3u] = 0u;
    }
}
