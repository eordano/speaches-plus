
struct Ql2Params {
    hidden: u32,
    batch: u32,
    eps: f32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> ql2_x: array<f32>;
@group(0) @binding(1) var<storage, read> ql2_dy: array<f32>;
@group(0) @binding(2) var<storage, read_write> ql2_dx: array<f32>;
@group(0) @binding(3) var<uniform> ql2_p: Ql2Params;

var<workgroup> ql2_red_sq: array<f32, 256>;
var<workgroup> ql2_red_dot: array<f32, 256>;

@compute @workgroup_size(256)
fn q38_l2norm_bwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= ql2_p.batch) {
        return;
    }
    let lid = tid.x;
    let hidden = ql2_p.hidden;
    let base = row * hidden;

    var sq = 0.0;
    var dot = 0.0;
    for (var i = lid; i < hidden; i = i + 256u) {
        let xv = ql2_x[base + i];
        sq = fma(xv, xv, sq);
        dot = fma(ql2_dy[base + i], xv, dot);
    }
    ql2_red_sq[lid] = sq;
    ql2_red_dot[lid] = dot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            ql2_red_sq[lid] = ql2_red_sq[lid] + ql2_red_sq[lid + s];
            ql2_red_dot[lid] = ql2_red_dot[lid] + ql2_red_dot[lid + s];
        }
        workgroupBarrier();
    }
    let d = sqrt(ql2_red_sq[0] + ql2_p.eps);
    let inv = 1.0 / d;
    let coef = ql2_red_dot[0] * inv * inv * inv;
    for (var i = lid; i < hidden; i = i + 256u) {
        ql2_dx[base + i] = fma(ql2_dy[base + i], inv, -(ql2_x[base + i] * coef));
    }
}
