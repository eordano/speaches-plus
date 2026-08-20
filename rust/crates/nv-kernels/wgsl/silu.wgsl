struct SiluParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> silu_src_f32: array<f32>;
@group(0) @binding(1) var<storage, read> silu_gate_f32: array<f32>;
@group(0) @binding(2) var<storage, read_write> silu_dst_f32: array<f32>;
@group(0) @binding(3) var<uniform> silu_params: SiluParams;
@group(0) @binding(4) var<storage, read> silu_src_bf16: array<u32>;
@group(0) @binding(5) var<storage, read> silu_gate_bf16: array<u32>;
@group(0) @binding(6) var<storage, read_write> silu_dst_bf16: array<u32>;

const SILU_WORKGROUP: u32 = 256u;
const SILU_EXP_LIMIT: f32 = 88.0;

fn silu_scalar(x: f32) -> f32 {
    if (x < -SILU_EXP_LIMIT) {
        return 0.0;
    }
    return x / (1.0 + exp(clamp(-x, -SILU_EXP_LIMIT, SILU_EXP_LIMIT)));
}

fn silu_flat_index(wid: vec3<u32>, ng: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (wid.y * ng.x + wid.x) * SILU_WORKGROUP + lid.x;
}

@compute @workgroup_size(256)
fn silu_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = silu_flat_index(wid, ng, lid);
    if (i >= silu_params.n) { return; }
    silu_dst_f32[i] = silu_scalar(silu_src_f32[i]);
}

@compute @workgroup_size(256)
fn silu_mul_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = silu_flat_index(wid, ng, lid);
    if (i >= silu_params.n) { return; }
    silu_dst_f32[i] = silu_scalar(silu_src_f32[i]) * silu_gate_f32[i];
}

@compute @workgroup_size(256)
fn silu_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let w = silu_flat_index(wid, ng, lid);
    let words = (silu_params.n + 1u) / 2u;
    if (w >= words) { return; }
    let word = silu_src_bf16[w];
    let lo = silu_scalar(bf16_lo(word));
    var hi = 0.0;
    if (w * 2u + 1u < silu_params.n) {
        hi = silu_scalar(bf16_hi(word));
    }
    silu_dst_bf16[w] = bf16_pack(lo, hi);
}

@compute @workgroup_size(256)
fn silu_mul_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let w = silu_flat_index(wid, ng, lid);
    let words = (silu_params.n + 1u) / 2u;
    if (w >= words) { return; }
    let xw = silu_src_bf16[w];
    let gw = silu_gate_bf16[w];
    let lo = silu_scalar(bf16_lo(xw)) * bf16_lo(gw);
    var hi = 0.0;
    if (w * 2u + 1u < silu_params.n) {
        hi = silu_scalar(bf16_hi(xw)) * bf16_hi(gw);
    }
    silu_dst_bf16[w] = bf16_pack(lo, hi);
}
