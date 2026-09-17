
struct QbnParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> qbn_x: array<f32>;
@group(0) @binding(1) var<storage, read> qbn_g: array<f32>;
@group(0) @binding(2) var<storage, read> qbn_dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> qbn_dx: array<f32>;
@group(0) @binding(4) var<uniform> qbn_p: QbnParams;

var<workgroup> qbn_red_sq: array<f32, 256>;
var<workgroup> qbn_red_dot: array<f32, 256>;

@compute @workgroup_size(256)
fn qlora_bwd_rmsnorm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= qbn_p.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = qbn_p.hidden;
    let base = row * hidden;

    var sq = 0.0;
    var dot = 0.0;
    for (var i = lid; i < hidden; i = i + 256u) {
        let xv = qbn_x[base + i];
        sq = fma(xv, xv, sq);
        dot = fma(qbn_dy[base + i] * qbn_g[i], xv, dot);
    }
    qbn_red_sq[lid] = sq;
    qbn_red_dot[lid] = dot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            qbn_red_sq[lid] = qbn_red_sq[lid] + qbn_red_sq[lid + s];
            qbn_red_dot[lid] = qbn_red_dot[lid] + qbn_red_dot[lid + s];
        }
        workgroupBarrier();
    }
    let h = f32(hidden);
    let r = inverseSqrt(qbn_red_sq[0] / h + qbn_p.eps);
    let coef = r * r * r * qbn_red_dot[0] / h;
    for (var i = lid; i < hidden; i = i + 256u) {
        qbn_dx[base + i] = fma(qbn_dy[base + i] * qbn_g[i], r, -(qbn_x[base + i] * coef));
    }
}
