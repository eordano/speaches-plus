struct RmsResParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
};

@group(0) @binding(0) var<storage, read> rmsres_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> rmsres_res: array<u32>;
@group(0) @binding(2) var<storage, read> rmsres_w: array<u32>;
@group(0) @binding(3) var<storage, read_write> rmsres_out: array<u32>;
@group(0) @binding(4) var<uniform> rmsres_params: RmsResParams;

const RMSRES_BLOCK: u32 = 256u;

var<workgroup> rmsres_scratch: array<f32, 256>;
var<workgroup> rmsres_shared: f32;

fn rmsres_row_index(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

fn rmsres_reduce(lid: u32, local: f32) -> f32 {
    rmsres_scratch[lid] = local;
    workgroupBarrier();
    for (var stride = RMSRES_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            rmsres_scratch[lid] = rmsres_scratch[lid] + rmsres_scratch[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        rmsres_shared = inverseSqrt(rmsres_scratch[0] / f32(rmsres_params.hidden) + rmsres_params.eps);
    }
    workgroupBarrier();
    return rmsres_shared;
}

@compute @workgroup_size(256)
fn rmsnorm_residual_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = rmsres_row_index(wg, nwg);
    if (row >= rmsres_params.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = rmsres_params.hidden;
    let base = row * hidden;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + RMSRES_BLOCK) {
        let s = bitcast<f32>(rmsres_x[base + i]) + bitcast<f32>(rmsres_res[base + i]);
        rmsres_res[base + i] = bitcast<u32>(s);
        local = local + s * s;
    }

    let rms = rmsres_reduce(lid, local);

    for (var i = lid; i < hidden; i = i + RMSRES_BLOCK) {
        let v = bitcast<f32>(rmsres_res[base + i]) * rms * bitcast<f32>(rmsres_w[i]);
        rmsres_out[base + i] = bitcast<u32>(v);
    }
}

@compute @workgroup_size(256)
fn rmsnorm_residual_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = rmsres_row_index(wg, nwg);
    if (row >= rmsres_params.batch) {
        return;
    }
    let lid = tid.x;
    let words = rmsres_params.words_per_row;
    let base = row * words;

    var local = 0.0;
    for (var i = lid; i < words; i = i + RMSRES_BLOCK) {
        let xw = rmsres_x[base + i];
        let rw = rmsres_res[base + i];
        let lo = bf16_lo(xw) + bf16_lo(rw);
        let hi = bf16_hi(xw) + bf16_hi(rw);
        rmsres_res[base + i] = bf16_pack(lo, hi);
        local = local + lo * lo + hi * hi;
    }

    let rms = rmsres_reduce(lid, local);

    for (var i = lid; i < words; i = i + RMSRES_BLOCK) {
        let sw = rmsres_res[base + i];
        let ww = rmsres_w[i];
        let lo = bf16_lo(sw) * rms * bf16_lo(ww);
        let hi = bf16_hi(sw) * rms * bf16_hi(ww);
        rmsres_out[base + i] = bf16_pack(lo, hi);
    }
}
