
struct QblDxParams {
    m_rows: u32,
    n_rows: u32,
    k_cols: u32,
    w_row_words: u32,
    scale_stride: u32,
    total: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> qbl_dy: array<f32>;
@group(0) @binding(1) var<storage, read> qbl_w: array<u32>;
@group(0) @binding(2) var<storage, read> qbl_s: array<f32>;
@group(0) @binding(3) var<storage, read> qbl_b: array<f32>;
@group(0) @binding(4) var<storage, read_write> qbl_dx: array<f32>;
@group(0) @binding(5) var<uniform> qbl_p: QblDxParams;

const QBL_BLOCK: u32 = 256u;

fn qbl_flat(wg: vec3<u32>, nwg: vec3<u32>, tid: vec3<u32>) -> u32 {
    return (wg.x + wg.y * nwg.x) * QBL_BLOCK + tid.x;
}

@compute @workgroup_size(256)
fn qlora_bwd_dx_w4(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = qbl_flat(wg, nwg, tid);
    if (idx >= qbl_p.total) {
        return;
    }
    let k = idx % qbl_p.k_cols;
    let m = idx / qbl_p.k_cols;
    let word_off = k >> 3u;
    let g = k / 64u;
    var acc = 0.0;
    for (var n = 0u; n < qbl_p.n_rows; n = n + 1u) {
        let q = f32(u4_unpack(qbl_w[n * qbl_p.w_row_words + word_off], k));
        let sb = n * qbl_p.scale_stride + g;
        let wv = fma(q, qbl_s[sb], qbl_b[sb]);
        acc = fma(qbl_dy[m * qbl_p.n_rows + n], wv, acc);
    }
    qbl_dx[idx] = acc;
}
