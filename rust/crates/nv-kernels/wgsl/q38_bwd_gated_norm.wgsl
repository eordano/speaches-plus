
struct QgnParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> qgn_x: array<f32>;
@group(0) @binding(1) var<storage, read> qgn_z: array<f32>;
@group(0) @binding(2) var<storage, read> qgn_g: array<f32>;
@group(0) @binding(3) var<storage, read> qgn_dy: array<f32>;
@group(0) @binding(4) var<storage, read_write> qgn_dx: array<f32>;
@group(0) @binding(5) var<storage, read_write> qgn_dz: array<f32>;
@group(0) @binding(6) var<uniform> qgn_p: QgnParams;

var<workgroup> qgn_red_sq: array<f32, 256>;
var<workgroup> qgn_red_dot: array<f32, 256>;

@compute @workgroup_size(256)
fn q38_gated_norm_bwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= qgn_p.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = qgn_p.hidden;
    let base = row * hidden;

    var sq = 0.0;
    var dot = 0.0;
    for (var i = lid; i < hidden; i = i + 256u) {
        let xv = qgn_x[base + i];
        let zv = qgn_z[base + i];
        let gate = zv / (1.0 + exp(-zv));
        sq = fma(xv, xv, sq);
        dot = fma(qgn_dy[base + i] * gate * qgn_g[i], xv, dot);
    }
    qgn_red_sq[lid] = sq;
    qgn_red_dot[lid] = dot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            qgn_red_sq[lid] = qgn_red_sq[lid] + qgn_red_sq[lid + s];
            qgn_red_dot[lid] = qgn_red_dot[lid] + qgn_red_dot[lid + s];
        }
        workgroupBarrier();
    }
    let h = f32(hidden);
    let r = inverseSqrt(qgn_red_sq[0] / h + qgn_p.eps);
    let coef = r * r * r * qgn_red_dot[0] / h;
    for (var i = lid; i < hidden; i = i + 256u) {
        let xv = qgn_x[base + i];
        let zv = qgn_z[base + i];
        let sg = 1.0 / (1.0 + exp(-zv));
        let gate = zv * sg;
        let dyv = qgn_dy[base + i];
        qgn_dx[base + i] = fma(dyv * gate * qgn_g[i], r, -(xv * coef));
        qgn_dz[base + i] = dyv * qgn_g[i] * xv * r * sg * (1.0 + zv * (1.0 - sg));
    }
}
