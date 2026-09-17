
struct QblGemmParams {
    m_rows: u32,
    i_cols: u32,
    j_cols: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> qblg_p: array<f32>;
@group(0) @binding(1) var<storage, read> qblg_q: array<f32>;
@group(0) @binding(2) var<storage, read_write> qblg_out: array<f32>;
@group(0) @binding(3) var<uniform> qblg_u: QblGemmParams;

const QBLG_BLOCK: u32 = 256u;

fn qblg_flat(wg: vec3<u32>, nwg: vec3<u32>, tid: vec3<u32>) -> u32 {
    return (wg.x + wg.y * nwg.x) * QBLG_BLOCK + tid.x;
}

@compute @workgroup_size(256)
fn qlora_bwd_gemm_nn(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = qblg_flat(wg, nwg, tid);
    if (idx >= qblg_u.total) {
        return;
    }
    let j = idx % qblg_u.j_cols;
    let m = idx / qblg_u.j_cols;
    var acc = 0.0;
    for (var i = 0u; i < qblg_u.i_cols; i = i + 1u) {
        acc = fma(qblg_p[m * qblg_u.i_cols + i], qblg_q[i * qblg_u.j_cols + j], acc);
    }
    qblg_out[idx] = acc;
}

@compute @workgroup_size(256)
fn qlora_bwd_gemm_tn(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = qblg_flat(wg, nwg, tid);
    if (idx >= qblg_u.total) {
        return;
    }
    let j = idx % qblg_u.j_cols;
    let i = idx / qblg_u.j_cols;
    var acc = 0.0;
    for (var m = 0u; m < qblg_u.m_rows; m = m + 1u) {
        acc = fma(qblg_p[m * qblg_u.i_cols + i], qblg_q[m * qblg_u.j_cols + j], acc);
    }
    qblg_out[idx] = acc;
}
