
struct G4aParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    has_rope: u32,
    n_q: u32,
    n_kv: u32,
    eps: f32,
};

struct G4kvParams {
    words: u32,
    ring: u32,
    k_off_words: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> ga_src: array<u32>;
@group(0) @binding(1) var<storage, read> ga_w: array<u32>;
@group(0) @binding(2) var<storage, read> ga_cos: array<f32>;
@group(0) @binding(3) var<storage, read> ga_sin: array<f32>;
@group(0) @binding(4) var<storage, read> ga_pos: array<i32>;
@group(0) @binding(5) var<storage, read_write> ga_out: array<u32>;
@group(0) @binding(6) var<uniform> ga_p: G4aParams;
@group(0) @binding(7) var<storage, read_write> ga_kc: array<u32>;
@group(0) @binding(8) var<storage, read_write> ga_vc: array<u32>;
@group(0) @binding(9) var<uniform> ga_kv: G4kvParams;

var<workgroup> ga_red: array<f32, 256>;
var<workgroup> ga_buf: array<f32, 512>;
var<workgroup> ga_s: f32;

fn gal_reduce_two_barriers_lane_combines_strides_128_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
    lane: u32,
    local: f32
) -> f32 {
    ga_red[lane] = local;
    workgroupBarrier();
    var acc = 0.0;
    if (lane < 32u) {
        acc = ((ga_red[lane] + ga_red[lane + 128u]) + (ga_red[lane + 64u] + ga_red[lane + 192u]))
            + ((ga_red[lane + 32u] + ga_red[lane + 160u]) + (ga_red[lane + 96u] + ga_red[lane + 224u]));
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u) {
        ga_s = acc;
    }
    workgroupBarrier();
    return ga_s;
}

@compute @workgroup_size(256)
fn g4m_attn_norm_rope_qkv_kvwrite_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let lane = lid.x;
    let hd = ga_p.head_dim;
    let n_q = ga_p.n_q;
    let n_kv = ga_p.n_kv;
    let has_v = ga_p.has_rope;
    var src_row = r;
    var w_off = 0u;
    var rope = true;
    if (r >= n_q + n_kv) {
        let vr = r - n_q - n_kv;
        if (has_v == 1u) {
            src_row = n_q + n_kv + vr;
        } else {
            src_row = n_q + vr;
        }
        w_off = 2u * hd;
        rope = false;
    } else if (r >= n_q) {
        w_off = hd;
    }
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = src_row * ga_p.src_stride + e0;
        v0 = bf16_decode(u16_at(ga_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(ga_src[(base + 1u) >> 1u], base + 1u));
    }
    let sq = gal_reduce_two_barriers_lane_combines_strides_128_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
        lane,
        v0 * v0 + v1 * v1
    );
    let rms = inverseSqrt(sq / f32(hd) + ga_p.eps);
    if (e0 < hd) {
        let we0 = w_off + e0;
        let w0 = bf16_decode(u16_at(ga_w[we0 >> 1u], we0));
        let w1 = bf16_decode(u16_at(ga_w[(we0 + 1u) >> 1u], we0 + 1u));
        ga_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ga_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (ga_pos[0] > 0) {
        p = u32(ga_pos[0]);
    }
    var o0 = ga_buf[e0];
    var o1 = ga_buf[e0 + 1u];
    if (rope) {
        let rh = ga_p.rot_half;
        if (e0 < rh) {
            let c = ga_cos[p * rh + e0];
            let s = ga_sin[p * rh + e0];
            o0 = ga_buf[e0] * c - ga_buf[e0 + rh] * s;
        } else {
            let i0 = e0 - rh;
            let c = ga_cos[p * rh + i0];
            let s = ga_sin[p * rh + i0];
            o0 = ga_buf[i0] * s + ga_buf[e0] * c;
        }
        let e1 = e0 + 1u;
        if (e1 < rh) {
            let c = ga_cos[p * rh + e1];
            let s = ga_sin[p * rh + e1];
            o1 = ga_buf[e1] * c - ga_buf[e1 + rh] * s;
        } else {
            let i1 = e1 - rh;
            let c = ga_cos[p * rh + i1];
            let s = ga_sin[p * rh + i1];
            o1 = ga_buf[i1] * s + ga_buf[e1] * c;
        }
    }
    let outw = bf16_pack(o0, o1);
    ga_out[(r * hd + e0) >> 1u] = outw;
    if (r >= n_q) {
        var slot = p;
        if (ga_kv.ring > 0u) {
            slot = p % ga_kv.ring;
        }
        let base = slot * ga_kv.words;
        if (r < n_q + n_kv) {
            let kr = r - n_q;
            ga_kc[base + ((kr * hd + e0) >> 1u)] = outw;
        } else {
            let vr = r - n_q - n_kv;
            ga_vc[base + ((vr * hd + e0) >> 1u)] = outw;
        }
    }
}
