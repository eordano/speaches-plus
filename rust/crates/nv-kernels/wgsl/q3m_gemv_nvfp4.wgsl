
struct Q34Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    w_e_stride_vec2: u32,
    sf_e_stride_bytes: u32,
    x_slot_stride_vec2: u32,
    xsf_slot_stride_bytes: u32,
    y_slot_stride_words: u32,
    per_expert_alpha: u32,
    pad0: u32,
};

@group(0) @binding(10) var<storage, read> q34_w: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read> q34_ws: array<u32>;
@group(0) @binding(12) var<storage, read> q34_x: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read> q34_xs: array<u32>;
@group(0) @binding(14) var<uniform> q34_p: Q34Params;
@group(0) @binding(15) var<storage, read_write> q34_y: array<u32>;
@group(0) @binding(16) var<storage, read> q34_sel: array<u32>;
@group(0) @binding(17) var<storage, read> q34_alphas: array<f32>;

var<workgroup> q34_red: array<f32, 256>;

@compute @workgroup_size(256)
fn q3w_gemv_nvfp4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = q34_sel[slot];
    let pair = wid.x + wid.y * q34_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q34_p.n_rows;
    let wbase = select(0u, e * q34_p.w_e_stride_vec2 + row * q34_p.k_blocks, live);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let xbase = slot * q34_p.x_slot_stride_vec2;
    let xsfbase = slot * q34_p.xsf_slot_stride_bytes;
    let blocks = select(0u, q34_p.k_blocks, live);

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + 128u) {
        let wsi = sfbase + nvfp4_scale_byte_index(row, kb, q34_p.k_tiles);
        let ws = byte_at(q34_ws[wsi >> 2u], wsi);
        let xsi = xsfbase + kb;
        let xs = byte_at(q34_xs[xsi >> 2u], xsi);
        let bs = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = q34_w[wbase + kb];
        let xv = q34_x[xbase + kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(bs, dot, acc);
    }
    q34_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q34_red[tid] = q34_red[tid] + q34_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        var alpha = q34_p.alpha;
        if (q34_p.per_expert_alpha == 1u) {
            alpha = q34_alphas[e];
        }
        let lo = q34_red[0] * alpha;
        var hi = 0.0;
        if (row + 1u < q34_p.n_rows) {
            hi = q34_red[128] * alpha;
        }
        q34_y[slot * q34_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
