struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
};

@group(0) @binding(0) var<storage, read> rms_x: array<u32>;
@group(0) @binding(1) var<storage, read> rms_w: array<u32>;
@group(0) @binding(2) var<storage, read_write> rms_y: array<u32>;
@group(0) @binding(3) var<uniform> rms_params: RmsParams;

const RMS_BLOCK: u32 = 256u;
const RMS_WARP: u32 = 32u;

var<workgroup> rms_scratch: array<f32, 256>;
var<workgroup> rms_shared: f32;

fn rms_row_index(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

fn rms_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn rms_reduce(lid: u32, local: f32) -> f32 {
    rms_scratch[lid] = local;
    workgroupBarrier();

    for (var stride = RMS_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (RMS_WARP - 1u)) < stride) {
            rms_scratch[lid] = rms_scratch[lid] + rms_scratch[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let a = (rms_scratch[0u] + rms_scratch[128u]) + (rms_scratch[64u] + rms_scratch[192u]);
        let b = (rms_scratch[32u] + rms_scratch[160u]) + (rms_scratch[96u] + rms_scratch[224u]);
        let sum = a + b;
        let mean = rms_div_rn(sum, f32(rms_params.hidden));
        rms_shared = inverseSqrt(rms_params.eps + mean);
    }
    workgroupBarrier();
    return rms_shared;
}

@compute @workgroup_size(256)
fn rmsnorm_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = rms_row_index(wg, nwg);
    if (row >= rms_params.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = rms_params.hidden;
    let base = row * hidden;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + RMS_BLOCK) {
        let v = bitcast<f32>(rms_x[base + i]);
        local = fma(v, v, local);
    }

    let rms = rms_reduce(lid, local);

    for (var i = lid; i < hidden; i = i + RMS_BLOCK) {
        let v = bitcast<f32>(rms_x[base + i]) * rms * bitcast<f32>(rms_w[i]);
        rms_y[base + i] = bitcast<u32>(v);
    }
}

@compute @workgroup_size(256)
fn rmsnorm_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = rms_row_index(wg, nwg);
    if (row >= rms_params.batch) {
        return;
    }
    let lid = tid.x;
    let words = rms_params.words_per_row;
    let hidden = rms_params.hidden;
    let base = row * words;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + RMS_BLOCK) {
        let word = rms_x[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }

    let rms = rms_reduce(lid, local);

    for (var i = lid; i < words; i = i + RMS_BLOCK) {
        let xw = rms_x[base + i];
        let ww = rms_w[i];
        let lo = bf16_lo(xw) * rms * bf16_lo(ww);
        let hi = bf16_hi(xw) * rms * bf16_hi(ww);
        rms_y[base + i] = bf16_pack(lo, hi);
    }
}
