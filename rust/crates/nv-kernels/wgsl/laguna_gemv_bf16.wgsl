
struct LgbParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> lgb_w: array<u32>;
@group(0) @binding(1) var<storage, read> lgb_x: array<u32>;
@group(0) @binding(2) var<uniform> lgb_p: LgbParams;
@group(0) @binding(3) var<storage, read_write> lgb_y: array<u32>;

var<workgroup> lgb_red: array<f32, 256>;

@compute @workgroup_size(256)
fn lgw_gemv_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * lgb_p.groups_x;
    let row = pair * 2u + half;
    let live = row < lgb_p.n_rows;
    let wbase = select(0u, row * lgb_p.w_row_words, live);
    let kw = select(0u, lgb_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = lgb_w[wbase + i];
        let xw = lgb_x[lgb_p.x_off_words + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    lgb_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            lgb_red[tid] = lgb_red[tid] + lgb_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (lgb_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            lgb_y[lgb_p.y_off_words + row] = bitcast<u32>(lgb_red[tid] * lgb_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = lgb_red[0] * lgb_p.alpha;
        var hi = 0.0;
        if (row + 1u < lgb_p.n_rows) {
            hi = lgb_red[128] * lgb_p.alpha;
        }
        lgb_y[lgb_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

struct LgeParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    w_e_stride_words: u32,
    x_slot_stride_words: u32,
    y_slot_stride_words: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> lge_w: array<u32>;
@group(0) @binding(1) var<storage, read> lge_x: array<u32>;
@group(0) @binding(2) var<uniform> lge_p: LgeParams;
@group(0) @binding(3) var<storage, read_write> lge_y: array<u32>;
@group(0) @binding(4) var<storage, read> lge_sel: array<u32>;

var<workgroup> lge_red: array<f32, 256>;

@compute @workgroup_size(256)
fn lgw_gemv_bf16_experts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = lge_sel[slot];
    let pair = wid.x + wid.y * lge_p.groups_x;
    let row = pair * 2u + half;
    let live = row < lge_p.n_rows;
    let wbase = select(0u, e * lge_p.w_e_stride_words + row * lge_p.w_row_words, live);
    let xbase = slot * lge_p.x_slot_stride_words;
    let kw = select(0u, lge_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = lge_w[wbase + i];
        let xw = lge_x[xbase + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    lge_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            lge_red[tid] = lge_red[tid] + lge_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let lo = lge_red[0] * lge_p.alpha;
        var hi = 0.0;
        if (row + 1u < lge_p.n_rows) {
            hi = lge_red[128] * lge_p.alpha;
        }
        lge_y[slot * lge_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
