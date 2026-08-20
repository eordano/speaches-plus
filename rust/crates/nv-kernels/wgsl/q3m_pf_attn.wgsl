
struct Q3parParams {
    tokens: u32,
    q_src_stride_elems: u32,
    k_src_stride_elems: u32,
    q_out_stride_elems: u32,
    k_out_stride_elems: u32,
    qf_out_stride_elems: u32,
    pos_stride_words: u32,
    pad0: u32,
};

@group(0) @binding(9) var<uniform> par_p: Q3parParams;

fn par_rope_at(d: u32, p: u32) -> f32 {
    let rh = ar_p.rot_half;
    let so = ar_p.sin_off;
    if (d < rh) {
        let c = ar_cs[p * rh + d];
        let s = ar_cs[so + p * rh + d];
        return ar_buf[d] * c - ar_buf[d + rh] * s;
    }
    if (d < 2u * rh) {
        let i = d - rh;
        let c = ar_cs[p * rh + i];
        let s = ar_cs[so + p * rh + i];
        return ar_buf[i] * s + ar_buf[d] * c;
    }
    return ar_buf[d];
}

fn par_norm_rope(v0: f32, v1: f32, w_off: u32, lane: u32, t: u32) -> vec2<f32> {
    let hd = ar_p.head_dim;
    let e0 = 2u * lane;
    ar_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            ar_red[lane] = ar_red[lane] + ar_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(ar_red[0] / f32(hd) + ar_p.eps);
    if (e0 < hd) {
        let wi = w_off + e0;
        let w0 = bf16_decode(u16_at(ar_w[wi >> 1u], wi));
        let w1 = bf16_decode(u16_at(ar_w[(wi + 1u) >> 1u], wi + 1u));
        ar_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ar_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    var o = vec2<f32>(0.0, 0.0);
    if (e0 < hd) {
        var p = 0u;
        let pv = ar_pos[t * par_p.pos_stride_words];
        if (pv > 0) {
            p = u32(pv);
        }
        let o0 = par_rope_at(e0, p);
        let o1 = par_rope_at(e0 + 1u, p);
        o = vec2<f32>(o0, o1);
    }
    return o;
}

@compute @workgroup_size(128)
fn q3w_pf_attn_norm_rope_qk_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    let hd = ar_p.head_dim;
    let e0 = 2u * lid.x;
    let is_k = wid.x >= ar_p.n_q;
    var row = wid.x;
    var w_off = ar_p.w_off;
    if (is_k) {
        row = wid.x - ar_p.n_q;
        w_off = ar_p.k_w_off;
    }
    var v = vec2<f32>(0.0, 0.0);
    if (e0 < hd) {
        if (is_k) {
            let base = t * par_p.k_src_stride_elems + ar_p.k_src_off + row * ar_p.k_src_stride + e0;
            v = vec2<f32>(
                bf16_decode(u16_at(ar_ksrc[base >> 1u], base)),
                bf16_decode(u16_at(ar_ksrc[(base + 1u) >> 1u], base + 1u))
            );
        } else {
            let base = t * par_p.q_src_stride_elems + ar_p.src_off + row * ar_p.src_stride + e0;
            v = vec2<f32>(
                bf16_decode(u16_at(ar_src[base >> 1u], base)),
                bf16_decode(u16_at(ar_src[(base + 1u) >> 1u], base + 1u))
            );
        }
    }
    let o = par_norm_rope(v.x, v.y, w_off, lid.x, t);
    if (e0 < hd) {
        let i = row * hd + e0;
        if (is_k) {
            ar_kout[(t * par_p.k_out_stride_elems + i) >> 1u] = bf16_pack(o.x, o.y);
        } else {
            ar_out[(t * par_p.q_out_stride_elems + i) >> 1u] = bf16_pack(o.x, o.y);
            let fb = t * par_p.qf_out_stride_elems + i;
            ar_outf[fb] = bf16_decode(bf16_encode(o.x));
            ar_outf[fb + 1u] = bf16_decode(bf16_encode(o.y));
        }
    }
}

struct Q3pagParams {
    attn_stride_elems: u32,
    qraw_stride_elems: u32,
    out_stride_words: u32,
    pad0: u32,
};

@group(0) @binding(34) var<uniform> pag_p: Q3pagParams;

@compute @workgroup_size(64)
fn q3w_pf_attn_gate_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    let w = wid.x * 64u + lid.x;
    if (w >= ag_p.n_words) {
        return;
    }
    let e0 = w * 2u;
    let ab = t * pag_p.attn_stride_elems;
    let a0 = bf16_decode(bf16_encode(ag_attn[ab + e0]));
    let a1 = bf16_decode(bf16_encode(ag_attn[ab + e0 + 1u]));
    let ow = t * pag_p.out_stride_words + w;
    if (ag_p.has_gate == 0u) {
        ag_out[ow] = bf16_pack(a0, a1);
        return;
    }
    let h = e0 / ag_p.head_dim;
    let d = e0 % ag_p.head_dim;
    let gb = t * pag_p.qraw_stride_elems + h * ag_p.src_stride + ag_p.gate_off + d;
    let g0 = bf16_decode(u16_at(ag_qraw[gb >> 1u], gb));
    let g1 = bf16_decode(u16_at(ag_qraw[(gb + 1u) >> 1u], gb + 1u));
    let s0 = bf16_decode(bf16_encode(1.0 / (1.0 + exp(-g0))));
    let s1 = bf16_decode(bf16_encode(1.0 / (1.0 + exp(-g1))));
    ag_out[ow] = bf16_pack(a0 * s0, a1 * s1);
}
