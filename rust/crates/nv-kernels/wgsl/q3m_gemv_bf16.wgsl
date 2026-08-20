
struct Q3bParams {
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

@group(0) @binding(0) var<storage, read> q3b_w: array<u32>;
@group(0) @binding(1) var<storage, read> q3b_x: array<u32>;
@group(0) @binding(2) var<uniform> q3b_p: Q3bParams;
@group(0) @binding(3) var<storage, read_write> q3b_y: array<u32>;

var<workgroup> q3b_red: array<f32, 256>;

@compute @workgroup_size(256)
fn q3w_gemv_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * q3b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q3b_p.n_rows;
    let wbase = select(0u, row * q3b_p.w_row_words, live);
    let kw = select(0u, q3b_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = q3b_w[wbase + i];
        let xw = q3b_x[q3b_p.x_off_words + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (q3b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            q3b_y[q3b_p.y_off_words + row] = bitcast<u32>(q3b_red[tid] * q3b_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = q3b_red[0] * q3b_p.alpha;
        var hi = 0.0;
        if (row + 1u < q3b_p.n_rows) {
            hi = q3b_red[128] * q3b_p.alpha;
        }
        q3b_y[q3b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

fn q3b_epilogue(tid: u32, lane: u32, row: u32, live: bool, acc: f32) {
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (q3b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            q3b_y[q3b_p.y_off_words + row] = bitcast<u32>(q3b_red[tid] * q3b_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = q3b_red[0] * q3b_p.alpha;
        var hi = 0.0;
        if (row + 1u < q3b_p.n_rows) {
            hi = q3b_red[128] * q3b_p.alpha;
        }
        q3b_y[q3b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn q3w_gemv_bf16_u4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * q3b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q3b_p.n_rows;
    let wbase = select(0u, row * q3b_p.w_row_words, live);
    let kw = select(0u, q3b_p.k_words, live);
    let xb = q3b_p.x_off_words;

    var acc = 0.0;
    var i = lane;
    loop {
        if (i + 384u >= kw) { break; }
        let w0 = q3b_w[wbase + i];
        let w1 = q3b_w[wbase + i + 128u];
        let w2 = q3b_w[wbase + i + 256u];
        let w3 = q3b_w[wbase + i + 384u];
        let x0 = q3b_x[xb + i];
        let x1 = q3b_x[xb + i + 128u];
        let x2 = q3b_x[xb + i + 256u];
        let x3 = q3b_x[xb + i + 384u];
        acc = fma(bf16_lo(w0), bf16_lo(x0), acc);
        acc = fma(bf16_hi(w0), bf16_hi(x0), acc);
        acc = fma(bf16_lo(w1), bf16_lo(x1), acc);
        acc = fma(bf16_hi(w1), bf16_hi(x1), acc);
        acc = fma(bf16_lo(w2), bf16_lo(x2), acc);
        acc = fma(bf16_hi(w2), bf16_hi(x2), acc);
        acc = fma(bf16_lo(w3), bf16_lo(x3), acc);
        acc = fma(bf16_hi(w3), bf16_hi(x3), acc);
        i = i + 512u;
    }
    loop {
        if (i >= kw) { break; }
        let ww = q3b_w[wbase + i];
        let xw = q3b_x[xb + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
        i = i + 128u;
    }
    q3b_epilogue(tid, lane, row, live, acc);
}

@compute @workgroup_size(256)
fn q3w_gemv_bf16_u8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * q3b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q3b_p.n_rows;
    let wbase = select(0u, row * q3b_p.w_row_words, live);
    let kw = select(0u, q3b_p.k_words, live);
    let xb = q3b_p.x_off_words;

    var acc = 0.0;
    var i = lane;
    loop {
        if (i + 896u >= kw) { break; }
        let w0 = q3b_w[wbase + i];
        let w1 = q3b_w[wbase + i + 128u];
        let w2 = q3b_w[wbase + i + 256u];
        let w3 = q3b_w[wbase + i + 384u];
        let w4 = q3b_w[wbase + i + 512u];
        let w5 = q3b_w[wbase + i + 640u];
        let w6 = q3b_w[wbase + i + 768u];
        let w7 = q3b_w[wbase + i + 896u];
        let x0 = q3b_x[xb + i];
        let x1 = q3b_x[xb + i + 128u];
        let x2 = q3b_x[xb + i + 256u];
        let x3 = q3b_x[xb + i + 384u];
        let x4 = q3b_x[xb + i + 512u];
        let x5 = q3b_x[xb + i + 640u];
        let x6 = q3b_x[xb + i + 768u];
        let x7 = q3b_x[xb + i + 896u];
        acc = fma(bf16_lo(w0), bf16_lo(x0), acc);
        acc = fma(bf16_hi(w0), bf16_hi(x0), acc);
        acc = fma(bf16_lo(w1), bf16_lo(x1), acc);
        acc = fma(bf16_hi(w1), bf16_hi(x1), acc);
        acc = fma(bf16_lo(w2), bf16_lo(x2), acc);
        acc = fma(bf16_hi(w2), bf16_hi(x2), acc);
        acc = fma(bf16_lo(w3), bf16_lo(x3), acc);
        acc = fma(bf16_hi(w3), bf16_hi(x3), acc);
        acc = fma(bf16_lo(w4), bf16_lo(x4), acc);
        acc = fma(bf16_hi(w4), bf16_hi(x4), acc);
        acc = fma(bf16_lo(w5), bf16_lo(x5), acc);
        acc = fma(bf16_hi(w5), bf16_hi(x5), acc);
        acc = fma(bf16_lo(w6), bf16_lo(x6), acc);
        acc = fma(bf16_hi(w6), bf16_hi(x6), acc);
        acc = fma(bf16_lo(w7), bf16_lo(x7), acc);
        acc = fma(bf16_hi(w7), bf16_hi(x7), acc);
        i = i + 1024u;
    }
    loop {
        if (i >= kw) { break; }
        let ww = q3b_w[wbase + i];
        let xw = q3b_x[xb + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
        i = i + 128u;
    }
    q3b_epilogue(tid, lane, row, live, acc);
}

