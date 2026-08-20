struct ResidualScaleParams {
    n: u32,
    n_words: u32,
    scale: f32,
    cap: f32,
    inv_cap: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> rs_a: array<u32>;
@group(0) @binding(1) var<storage, read> rs_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> rs_y: array<u32>;
@group(0) @binding(3) var<uniform> rs_params: ResidualScaleParams;
@group(0) @binding(4) var<storage, read_write> rs_yf: array<f32>;

const RS_BLOCK: u32 = 256u;

fn rs_word_index(wg: vec3<u32>, nwg: vec3<u32>, tid: vec3<u32>) -> u32 {
    return (wg.x + wg.y * nwg.x) * RS_BLOCK + tid.x;
}

@compute @workgroup_size(256)
fn residual_add_scale_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = rs_word_index(wg, nwg, tid);
    if (w >= rs_params.n_words) {
        return;
    }
    let aw = rs_a[w];
    let bw = rs_b[w];
    let scale = rs_params.scale;
    let lo = (bf16_lo(aw) + bf16_lo(bw)) * scale;
    let hi = (bf16_hi(aw) + bf16_hi(bw)) * scale;
    rs_y[w] = bf16_pack(lo, hi);
}

@compute @workgroup_size(256)
fn scale_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = rs_word_index(wg, nwg, tid);
    if (w >= rs_params.n_words) {
        return;
    }
    let xw = rs_a[w];
    let scale = rs_params.scale;
    rs_y[w] = bf16_pack(bf16_lo(xw) * scale, bf16_hi(xw) * scale);
}

@compute @workgroup_size(256)
fn tanh_softcap_bf16_to_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = rs_word_index(wg, nwg, tid);
    if (w >= rs_params.n_words) {
        return;
    }
    let xw = rs_a[w];
    let cap = rs_params.cap;
    let inv_cap = rs_params.inv_cap;
    let i = w * 2u;
    rs_yf[i] = nv_tanhf(bf16_lo(xw) * inv_cap) * cap;
    if (i + 1u < rs_params.n) {
        rs_yf[i + 1u] = nv_tanhf(bf16_hi(xw) * inv_cap) * cap;
    }
}

@compute @workgroup_size(256)
fn cast_bf16_to_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = rs_word_index(wg, nwg, tid);
    if (w >= rs_params.n_words) {
        return;
    }
    let xw = rs_a[w];
    let i = w * 2u;
    rs_yf[i] = bf16_lo(xw);
    if (i + 1u < rs_params.n) {
        rs_yf[i + 1u] = bf16_hi(xw);
    }
}
