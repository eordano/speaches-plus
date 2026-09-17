
struct QagParams {
    total: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> qag_attn: array<f32>;
@group(0) @binding(1) var<storage, read> qag_gate: array<f32>;
@group(0) @binding(2) var<storage, read> qag_dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> qag_dattn: array<f32>;
@group(0) @binding(4) var<storage, read_write> qag_dgate: array<f32>;
@group(0) @binding(5) var<uniform> qag_p: QagParams;

@compute @workgroup_size(256)
fn q38_attn_gate_bwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    if (idx >= qag_p.total) {
        return;
    }
    let sg = 1.0 / (1.0 + exp(-qag_gate[idx]));
    let dyv = qag_dy[idx];
    qag_dattn[idx] = dyv * sg;
    qag_dgate[idx] = dyv * qag_attn[idx] * sg * (1.0 - sg);
}
