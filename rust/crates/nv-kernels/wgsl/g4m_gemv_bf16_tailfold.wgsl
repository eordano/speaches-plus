
struct GtfGemvParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    wide: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

struct GtfNormParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> gtf_w: array<vec4<u32>>;
@group(0) @binding(1) var<uniform> gtf_p: GtfGemvParams;
@group(0) @binding(2) var<storage, read_write> gtf_y: array<u32>;
@group(0) @binding(3) var<uniform> gtf_n: GtfNormParams;
@group(0) @binding(4) var<storage, read> gtf_x: array<u32>;
@group(0) @binding(5) var<storage, read> gtf_nw: array<u32>;
@group(0) @binding(6) var<storage, read_write> gtf_res_out: array<u32>;
@group(0) @binding(7) var<storage, read> gtf_res_in: array<u32>;
@group(0) @binding(8) var<storage, read> gtf_w2: array<u32>;
@group(0) @binding(9) var<storage, read> gtf_h1: array<u32>;
@group(0) @binding(10) var<storage, read_write> gtf_y2: array<u32>;
@group(0) @binding(11) var<storage, read> gtf_w3: array<u32>;

var<workgroup> gtf_buf: array<u32, 2048>;
var<workgroup> gtf_red: array<f32, 256>;
var<workgroup> gtf_s: f32;

fn gtf_reduce(lid: u32, local: f32) -> f32 {
    gtf_red[lid] = local;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            gtf_red[lid] = gtf_red[lid] + gtf_red[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        gtf_s = inverseSqrt(gtf_red[0] / f32(gtf_n.hidden) + gtf_n.eps);
    }
    workgroupBarrier();
    return gtf_s;
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_tailfold(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>
) {
    let tid = lid3.x;
    let wg0 = wid.x == 0u && wid.y == 0u;

    var local = 0.0;
    for (var w = tid; w < gtf_n.words; w = w + 256u) {
        let xw = gtf_x[w];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s1 = gtf_reduce(tid, local);
    var local2 = 0.0;
    for (var w = tid; w < gtf_n.words; w = w + 256u) {
        let xw = gtf_x[w];
        let ww = gtf_nw[w];
        let h2 = bf16_pack(bf16_lo(xw) * s1 * bf16_lo(ww), bf16_hi(xw) * s1 * bf16_hi(ww));
        let h1w = gtf_h1[w];
        let sum = bf16_pack(
            (bf16_lo(h1w) + bf16_lo(h2)) * 1.0,
            (bf16_hi(h1w) + bf16_hi(h2)) * 1.0
        );
        gtf_buf[w] = sum;
        let r0 = bf16_lo(sum);
        let r1 = bf16_hi(sum);
        local2 = local2 + r0 * r0 + r1 * r1;
    }
    let s2 = gtf_reduce(tid, local2);
    var local3 = 0.0;
    for (var w = tid; w < gtf_n.words; w = w + 256u) {
        let sum = gtf_buf[w];
        let w3 = gtf_w2[w];
        let comb = bf16_pack(bf16_lo(sum) * s2 * bf16_lo(w3), bf16_hi(sum) * s2 * bf16_hi(w3));
        let rw = gtf_res_in[w];
        let outw = bf16_pack(
            (bf16_lo(comb) + bf16_lo(rw)) * gtf_n.scale,
            (bf16_hi(comb) + bf16_hi(rw)) * gtf_n.scale
        );
        if (wg0) {
            gtf_res_out[w] = outw;
        }
        gtf_buf[w] = outw;
        let o0 = bf16_lo(outw);
        let o1 = bf16_hi(outw);
        local3 = local3 + o0 * o0 + o1 * o1;
    }
    let s3 = gtf_reduce(tid, local3);
    for (var w = tid; w < gtf_n.words; w = w + 256u) {
        let outw = gtf_buf[w];
        let wn = gtf_w3[w];
        let yw = bf16_pack(bf16_lo(outw) * s3 * bf16_lo(wn), bf16_hi(outw) * s3 * bf16_hi(wn));
        if (wg0) {
            gtf_y2[w] = yw;
        }
        gtf_buf[w] = yw;
    }
    workgroupBarrier();

    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * gtf_p.groups_x;
    let row = pair * 2u + half;
    let live = row < gtf_p.n_rows;
    let wbase = select(0u, row * gtf_p.w_row_words, live);
    let kw = select(0u, gtf_p.k_words, live);

    var acc = 0.0;
    let wv = wbase >> 2u;
    let kv = kw >> 2u;
    for (var v = lane; v < kv; v = v + 128u) {
        let ww = gtf_w[wv + v];
        let b0 = v * 4u;
        acc = fma(bf16_lo(ww.x), bf16_lo(gtf_buf[b0]), acc);
        acc = fma(bf16_hi(ww.x), bf16_hi(gtf_buf[b0]), acc);
        acc = fma(bf16_lo(ww.y), bf16_lo(gtf_buf[b0 + 1u]), acc);
        acc = fma(bf16_hi(ww.y), bf16_hi(gtf_buf[b0 + 1u]), acc);
        acc = fma(bf16_lo(ww.z), bf16_lo(gtf_buf[b0 + 2u]), acc);
        acc = fma(bf16_hi(ww.z), bf16_hi(gtf_buf[b0 + 2u]), acc);
        acc = fma(bf16_lo(ww.w), bf16_lo(gtf_buf[b0 + 3u]), acc);
        acc = fma(bf16_hi(ww.w), bf16_hi(gtf_buf[b0 + 3u]), acc);
    }
    gtf_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            gtf_red[tid] = gtf_red[tid] + gtf_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (gtf_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            gtf_y[gtf_p.y_off_words + row] = bitcast<u32>(gtf_red[tid] * gtf_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = gtf_red[0] * gtf_p.alpha;
        var hi = 0.0;
        if (row + 1u < gtf_p.n_rows) {
            hi = gtf_red[128] * gtf_p.alpha;
        }
        gtf_y[gtf_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
