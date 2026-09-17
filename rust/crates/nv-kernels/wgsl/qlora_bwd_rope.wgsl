
struct QbrParams {
    tokens: u32,
    heads: u32,
    half_dim: u32,
    max_seq: u32,
};

@group(0) @binding(0) var<storage, read> qbr_x: array<f32>;
@group(0) @binding(1) var<storage, read> qbr_pos: array<u32>;
@group(0) @binding(2) var<storage, read> qbr_cos: array<f32>;
@group(0) @binding(3) var<storage, read> qbr_sin: array<f32>;
@group(0) @binding(4) var<storage, read_write> qbr_out: array<f32>;
@group(0) @binding(5) var<uniform> qbr_p: QbrParams;

const QBR_BLOCK: u32 = 256u;

fn qbr_flat(wg: vec3<u32>, nwg: vec3<u32>, tid: vec3<u32>) -> u32 {
    return (wg.x + wg.y * nwg.x) * QBR_BLOCK + tid.x;
}

@compute @workgroup_size(256)
fn qlora_rope_fwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = qbr_flat(wg, nwg, tid);
    let total = qbr_p.tokens * qbr_p.heads * qbr_p.half_dim;
    if (idx >= total) {
        return;
    }
    let i = idx % qbr_p.half_dim;
    let th = idx / qbr_p.half_dim;
    let t = th / qbr_p.heads;
    let pr = min(qbr_pos[t], qbr_p.max_seq - 1u);
    let base = th * (2u * qbr_p.half_dim);
    let lo = qbr_x[base + i];
    let hi = qbr_x[base + qbr_p.half_dim + i];
    let c = qbr_cos[pr * qbr_p.half_dim + i];
    let s = qbr_sin[pr * qbr_p.half_dim + i];
    let m1 = lo * c;
    let m2 = hi * s;
    let m3 = lo * s;
    let m4 = hi * c;
    qbr_out[base + i] = m1 - m2;
    qbr_out[base + qbr_p.half_dim + i] = m3 + m4;
}

@compute @workgroup_size(256)
fn qlora_rope_bwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = qbr_flat(wg, nwg, tid);
    let total = qbr_p.tokens * qbr_p.heads * qbr_p.half_dim;
    if (idx >= total) {
        return;
    }
    let i = idx % qbr_p.half_dim;
    let th = idx / qbr_p.half_dim;
    let t = th / qbr_p.heads;
    let pr = min(qbr_pos[t], qbr_p.max_seq - 1u);
    let base = th * (2u * qbr_p.half_dim);
    let gl = qbr_x[base + i];
    let gh = qbr_x[base + qbr_p.half_dim + i];
    let c = qbr_cos[pr * qbr_p.half_dim + i];
    let s = qbr_sin[pr * qbr_p.half_dim + i];
    let a1 = gh * s;
    let a2 = gl * c;
    let b1 = gh * c;
    let b2 = gl * s;
    qbr_out[base + i] = a1 + a2;
    qbr_out[base + qbr_p.half_dim + i] = b1 - b2;
}
