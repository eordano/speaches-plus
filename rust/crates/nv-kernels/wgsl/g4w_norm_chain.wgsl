
const NC_WG: u32 = 256u;
const NC_WARP: u32 = 32u;

struct NcParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
};

@group(0) @binding(20) var<storage, read_write> nc_x: array<u32>;
@group(0) @binding(21) var<storage, read> nc_w1: array<u32>;
@group(0) @binding(22) var<storage, read_write> nc_mid: array<u32>;
@group(0) @binding(23) var<storage, read_write> nc_res: array<u32>;
@group(0) @binding(24) var<storage, read> nc_w2: array<u32>;
@group(0) @binding(25) var<storage, read_write> nc_out: array<u32>;
@group(0) @binding(26) var<uniform> nc_params: NcParams;

var<workgroup> nc_s1: array<f32, 256>;
var<workgroup> nc_v1: f32;
var<workgroup> nc_s2: array<f32, 256>;
var<workgroup> nc_v2: f32;

fn nc_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn nc_rms_reduce(lid: u32, local: f32) -> f32 {
    nc_s1[lid] = local;
    workgroupBarrier();
    for (var stride = NC_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (NC_WARP - 1u)) < stride) {
            nc_s1[lid] = nc_s1[lid] + nc_s1[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        let a = (nc_s1[0u] + nc_s1[128u]) + (nc_s1[64u] + nc_s1[192u]);
        let b = (nc_s1[32u] + nc_s1[160u]) + (nc_s1[96u] + nc_s1[224u]);
        let sum = a + b;
        let mean = nc_div_rn(sum, f32(nc_params.hidden));
        nc_v1 = inverseSqrt(nc_params.eps + mean);
    }
    workgroupBarrier();
    return nc_v1;
}

fn nc_res_reduce(lid: u32, local: f32) -> f32 {
    nc_s2[lid] = local;
    workgroupBarrier();
    for (var stride = NC_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            nc_s2[lid] = nc_s2[lid] + nc_s2[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        nc_v2 = inverseSqrt(nc_s2[0] / f32(nc_params.hidden) + nc_params.eps);
    }
    workgroupBarrier();
    return nc_v2;
}

fn nc_norm_into_mid(lid: u32) {
    let hidden = nc_params.hidden;
    let words = nc_params.words;
    var local = 0.0;
    for (var i = lid; i < hidden; i = i + NC_WG) {
        let word = nc_x[i >> 1u];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms = nc_rms_reduce(lid, local);
    for (var i = lid; i < words; i = i + NC_WG) {
        let xw = nc_x[i];
        let ww = nc_w1[i];
        let lo = bf16_lo(xw) * rms * bf16_lo(ww);
        let hi = bf16_hi(xw) * rms * bf16_hi(ww);
        nc_mid[i] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn g4w_norm_res_norm(@builtin(local_invocation_id) tid: vec3<u32>) {
    let lid = tid.x;
    nc_norm_into_mid(lid);
    storageBarrier();
    workgroupBarrier();

    let words = nc_params.words;
    var local = 0.0;
    for (var i = lid; i < words; i = i + NC_WG) {
        let xw = nc_mid[i];
        let rw = nc_res[i];
        let lo = bf16_lo(xw) + bf16_lo(rw);
        let hi = bf16_hi(xw) + bf16_hi(rw);
        nc_res[i] = bf16_pack(lo, hi);
        local = local + lo * lo + hi * hi;
    }
    let rms = nc_res_reduce(lid, local);
    for (var i = lid; i < words; i = i + NC_WG) {
        let sw = nc_res[i];
        let ww = nc_w2[i];
        let lo = bf16_lo(sw) * rms * bf16_lo(ww);
        let hi = bf16_hi(sw) * rms * bf16_hi(ww);
        nc_x[i] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn g4w_norm_add_norm(@builtin(local_invocation_id) tid: vec3<u32>) {
    let lid = tid.x;
    nc_norm_into_mid(lid);
    storageBarrier();
    workgroupBarrier();

    let words = nc_params.words;
    let scale = nc_params.scale;
    for (var i = lid; i < words; i = i + NC_WG) {
        let aw = nc_res[i];
        let bw = nc_mid[i];
        let lo = (bf16_lo(aw) + bf16_lo(bw)) * scale;
        let hi = (bf16_hi(aw) + bf16_hi(bw)) * scale;
        nc_out[i] = bf16_pack(lo, hi);
    }
    storageBarrier();
    workgroupBarrier();

    let hidden = nc_params.hidden;
    var local = 0.0;
    for (var i = lid; i < hidden; i = i + NC_WG) {
        let word = nc_out[i >> 1u];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms = nc_rms_reduce(lid, local);
    for (var i = lid; i < words; i = i + NC_WG) {
        let xw = nc_out[i];
        let ww = nc_w2[i];
        let lo = bf16_lo(xw) * rms * bf16_lo(ww);
        let hi = bf16_hi(xw) * rms * bf16_hi(ww);
        nc_x[i] = bf16_pack(lo, hi);
    }
}
