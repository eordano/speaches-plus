struct GowVlParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_row_words: u32,
    y_row_words: u32,
    has_bias: u32,
    alpha: f32,
    y_off_words: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> gvl_w: array<u32>;
@group(0) @binding(1) var<storage, read> gvl_x: array<u32>;
@group(0) @binding(2) var<uniform> gvl_p: GowVlParams;
@group(0) @binding(3) var<storage, read_write> gvl_y: array<u32>;
@group(0) @binding(4) var<storage, read> gvl_b: array<u32>;

var<workgroup> gvl_red: array<f32, 256>;

fn gvl_bias(row: u32) -> f32 {
    if (gvl_p.has_bias == 0u) {
        return 0.0;
    }
    return bf16_decode(u16_at(gvl_b[row >> 1u], row));
}

@compute @workgroup_size(256)
fn gow_v_lmhead(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let t = wid.z;
    let pair = wid.x + wid.y * gvl_p.groups_x;
    let row = pair * 2u + half;
    let live = row < gvl_p.n_rows;
    let wbase = select(0u, row * gvl_p.w_row_words, live);
    let kw = select(0u, gvl_p.k_words, live);
    let xbase = t * gvl_p.x_row_words;

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = gvl_w[wbase + i];
        let xw = gvl_x[xbase + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    gvl_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            gvl_red[tid] = gvl_red[tid] + gvl_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && live) {
        let v = gvl_red[tid] * gvl_p.alpha + gvl_bias(row);
        gvl_y[t * gvl_p.y_row_words + gvl_p.y_off_words + row] = bitcast<u32>(v);
    }
}

struct GowVamParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(10) var<storage, read> gvam_x: array<u32>;
@group(0) @binding(11) var<storage, read_write> gvam_pv: array<f32>;
@group(0) @binding(12) var<storage, read_write> gvam_pi: array<u32>;
@group(0) @binding(13) var<storage, read_write> gvam_out: array<u32>;
@group(0) @binding(14) var<uniform> gvam_p: GowVamParams;

var<workgroup> gvam_v: array<f32, 256>;
var<workgroup> gvam_i: array<u32, 256>;

fn gvam_reduce(tid: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let o = tid + s;
            if (gvam_v[o] > gvam_v[tid] || (gvam_v[o] == gvam_v[tid] && gvam_i[o] < gvam_i[tid])) {
                gvam_v[tid] = gvam_v[o];
                gvam_i[tid] = gvam_i[o];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn gow_v_argmax_stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let g = wid.x;
    let t = wid.z;
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    for (var i = g * 256u + tid; i < gvam_p.n; i = i + gvam_p.groups * 256u) {
        let v = bitcast<f32>(gvam_x[t * gvam_p.n + i]);
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    gvam_v[tid] = bv;
    gvam_i[tid] = bi;
    workgroupBarrier();
    gvam_reduce(tid);
    if (tid == 0u) {
        gvam_pv[t * gvam_p.groups + g] = gvam_v[0];
        gvam_pi[t * gvam_p.groups + g] = gvam_i[0];
    }
}

@compute @workgroup_size(256)
fn gow_v_argmax_stage2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.z;
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    if (tid < gvam_p.groups) {
        bv = gvam_pv[t * gvam_p.groups + tid];
        bi = gvam_pi[t * gvam_p.groups + tid];
    }
    gvam_v[tid] = bv;
    gvam_i[tid] = bi;
    workgroupBarrier();
    gvam_reduce(tid);
    if (tid == 0u) {
        gvam_out[t] = gvam_i[0];
    }
}
