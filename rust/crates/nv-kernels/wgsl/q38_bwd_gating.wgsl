
struct QgtParams {
    rows: u32,
    heads: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> qgt_a: array<f32>;
@group(0) @binding(1) var<storage, read> qgt_b: array<f32>;
@group(0) @binding(2) var<storage, read> qgt_alog: array<f32>;
@group(0) @binding(3) var<storage, read> qgt_dtb: array<f32>;
@group(0) @binding(4) var<storage, read> qgt_dg: array<f32>;
@group(0) @binding(5) var<storage, read> qgt_dbeta: array<f32>;
@group(0) @binding(6) var<storage, read_write> qgt_da: array<f32>;
@group(0) @binding(7) var<storage, read_write> qgt_db: array<f32>;
@group(0) @binding(8) var<uniform> qgt_p: QgtParams;

fn qgt_sig(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn q38_gating_bwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    let total = qgt_p.rows * qgt_p.heads;
    if (idx >= total) {
        return;
    }
    let h = idx % qgt_p.heads;
    let ap = qgt_a[idx] + qgt_dtb[h];
    let sp = max(ap, 0.0) + log(1.0 + exp(-abs(ap)));
    let nea = -exp(qgt_alog[h]);
    let ge = exp(sp * nea);
    qgt_da[idx] = qgt_dg[idx] * ge * nea * qgt_sig(ap);
    let beta = qgt_sig(qgt_b[idx]);
    qgt_db[idx] = qgt_dbeta[idx] * beta * (1.0 - beta);
}
