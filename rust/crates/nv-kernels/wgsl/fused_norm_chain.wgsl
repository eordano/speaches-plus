struct FncParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> fnc_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> fnc_res: array<u32>;
@group(0) @binding(2) var<storage, read> fnc_w1: array<u32>;
@group(0) @binding(3) var<storage, read> fnc_w2: array<u32>;
@group(0) @binding(4) var<storage, read_write> fnc_out: array<u32>;
@group(0) @binding(5) var<uniform> fnc_params: FncParams;
@group(0) @binding(6) var<storage, read_write> fnc_out2: array<u32>;

const FNC_BLOCK: u32 = 256u;
const FNC_WARP: u32 = 32u;

var<workgroup> fnc_scratch: array<f32, 256>;
var<workgroup> fnc_shared: f32;

fn fnc_row_index(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

fn fnc_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn fnc_rms_reduce(lid: u32, local: f32) -> f32 {
    fnc_scratch[lid] = local;
    workgroupBarrier();

    for (var stride = FNC_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (FNC_WARP - 1u)) < stride) {
            fnc_scratch[lid] = fnc_scratch[lid] + fnc_scratch[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let a = (fnc_scratch[0u] + fnc_scratch[128u]) + (fnc_scratch[64u] + fnc_scratch[192u]);
        let b = (fnc_scratch[32u] + fnc_scratch[160u]) + (fnc_scratch[96u] + fnc_scratch[224u]);
        let sum = a + b;
        let mean = fnc_div_rn(sum, f32(fnc_params.hidden));
        fnc_shared = inverseSqrt(fnc_params.eps + mean);
    }
    workgroupBarrier();
    return fnc_shared;
}

fn fnc_res_reduce(lid: u32, local: f32) -> f32 {
    fnc_scratch[lid] = local;
    workgroupBarrier();
    for (var stride = FNC_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            fnc_scratch[lid] = fnc_scratch[lid] + fnc_scratch[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        fnc_shared = inverseSqrt(fnc_scratch[0] / f32(fnc_params.hidden) + fnc_params.eps);
    }
    workgroupBarrier();
    return fnc_shared;
}

@compute @workgroup_size(256)
fn e4b_rms_res_rms_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = fnc_row_index(wg, nwg);
    if (row >= fnc_params.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = fnc_params.hidden;
    let words = fnc_params.words_per_row;
    let base = row * words;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + FNC_BLOCK) {
        let word = fnc_in[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms1 = fnc_rms_reduce(lid, local);

    var local2 = 0.0;
    for (var i = lid; i < words; i = i + FNC_BLOCK) {
        let xw = fnc_in[base + i];
        let w1w = fnc_w1[i];
        let tlo = bf16_lo(xw) * rms1 * bf16_lo(w1w);
        let thi = bf16_hi(xw) * rms1 * bf16_hi(w1w);
        let tw = bf16_pack(tlo, thi);
        let rw = fnc_res[base + i];
        let lo = bf16_lo(tw) + bf16_lo(rw);
        let hi = bf16_hi(tw) + bf16_hi(rw);
        fnc_res[base + i] = bf16_pack(lo, hi);
        local2 = local2 + lo * lo + hi * hi;
    }
    let rms2 = fnc_res_reduce(lid, local2);

    for (var i = lid; i < words; i = i + FNC_BLOCK) {
        let sw = fnc_res[base + i];
        let ww = fnc_w2[i];
        let lo = bf16_lo(sw) * rms2 * bf16_lo(ww);
        let hi = bf16_hi(sw) * rms2 * bf16_hi(ww);
        fnc_out[base + i] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn e4b_res_of_rms_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = fnc_row_index(wg, nwg);
    if (row >= fnc_params.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = fnc_params.hidden;
    let words = fnc_params.words_per_row;
    let base = row * words;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + FNC_BLOCK) {
        let word = fnc_in[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms1 = fnc_rms_reduce(lid, local);

    let scale = fnc_params.scale;
    for (var i = lid; i < words; i = i + FNC_BLOCK) {
        let xw = fnc_in[base + i];
        let w1w = fnc_w1[i];
        let tlo = bf16_lo(xw) * rms1 * bf16_lo(w1w);
        let thi = bf16_hi(xw) * rms1 * bf16_hi(w1w);
        let tw = bf16_pack(tlo, thi);
        let rw = fnc_res[base + i];
        let lo = (bf16_lo(rw) + bf16_lo(tw)) * scale;
        let hi = (bf16_hi(rw) + bf16_hi(tw)) * scale;
        fnc_out[base + i] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn e4b_rms_res_rms_next_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = fnc_row_index(wg, nwg);
    if (row >= fnc_params.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = fnc_params.hidden;
    let words = fnc_params.words_per_row;
    let base = row * words;

    var local = 0.0;
    for (var i = lid; i < hidden; i = i + FNC_BLOCK) {
        let word = fnc_in[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms1 = fnc_rms_reduce(lid, local);

    let scale = fnc_params.scale;
    for (var i = lid; i < words; i = i + FNC_BLOCK) {
        let xw = fnc_in[base + i];
        let w1w = fnc_w1[i];
        let tlo = bf16_lo(xw) * rms1 * bf16_lo(w1w);
        let thi = bf16_hi(xw) * rms1 * bf16_hi(w1w);
        let tw = bf16_pack(tlo, thi);
        let rw = fnc_res[base + i];
        let lo = (bf16_lo(rw) + bf16_lo(tw)) * scale;
        let hi = (bf16_hi(rw) + bf16_hi(tw)) * scale;
        fnc_out[base + i] = bf16_pack(lo, hi);
    }
    storageBarrier();

    var local2 = 0.0;
    for (var i = lid; i < hidden; i = i + FNC_BLOCK) {
        let word = fnc_out[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local2 = fma(v, v, local2);
    }
    let rms2 = fnc_rms_reduce(lid, local2);

    for (var i = lid; i < words; i = i + FNC_BLOCK) {
        let sw = fnc_out[base + i];
        let ww = fnc_w2[i];
        let lo = bf16_lo(sw) * rms2 * bf16_lo(ww);
        let hi = bf16_hi(sw) * rms2 * bf16_hi(ww);
        fnc_out2[base + i] = bf16_pack(lo, hi);
    }
}
