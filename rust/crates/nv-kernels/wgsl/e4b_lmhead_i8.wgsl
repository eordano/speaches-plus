
struct I8hParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    dst_word_off: u32,
};

@group(0) @binding(0) var<storage, read> i8h_w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> i8h_scale: array<f32>;
@group(0) @binding(2) var<storage, read> i8h_x: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> i8h_y: array<u32>;
@group(0) @binding(4) var<uniform> i8h_params: I8hParams;

var<workgroup> i8h_partial: array<f32, 256>;

fn i8h_dot4(word: u32, xw0: u32, xw1: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(int8_decode(word, 0u), bf16_lo(xw0), acc);
    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);
    acc = fma(int8_decode(word, 2u), bf16_lo(xw1), acc);
    acc = fma(int8_decode(word, 3u), bf16_hi(xw1), acc);
    return acc;
}

fn i8h_dot16(wv: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    var a = 0.0;
    a = i8h_dot4(wv.x, xa.x, xa.y, a);
    a = i8h_dot4(wv.y, xa.z, xa.w, a);
    a = i8h_dot4(wv.z, xb.x, xb.y, a);
    a = i8h_dot4(wv.w, xb.z, xb.w, a);
    return a;
}

@compute @workgroup_size(256)
fn e4b_lmhead_i8_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let row = (wid.x + wid.y * i8h_params.groups_x) * 8u + warp;
    let live = row < i8h_params.n_rows;
    let kv = select(0u, i8h_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (i8h_params.k_elems >> 4u), live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + 32u) {
        acc = acc + i8h_dot16(i8h_w[wbase + v], i8h_x[v * 2u], i8h_x[v * 2u + 1u]);
    }
    i8h_partial[tid] = acc;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            i8h_partial[tid] = i8h_partial[tid] + i8h_partial[tid + stride];
        }
        workgroupBarrier();
    }
    let total = i8h_partial[tid - lane];
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let hi_live = row + 1u < i8h_params.n_rows;
        let sc_lo = i8h_scale[row];
        let sc_hi = select(0.0, i8h_scale[select(row, row + 1u, hi_live)], hi_live);
        let lo = bf16_encode(total * sc_lo) & 0xffffu;
        let hi = bf16_encode(i8h_partial[tid + 32u] * sc_hi) & 0xffffu;
        i8h_y[i8h_params.dst_word_off + (row >> 1u)] = lo | (select(0u, hi, hi_live) << 16u);
    }
}
