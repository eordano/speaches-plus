
struct QqrParams {
    tokens: u32,
    heads: u32,
    hidden: u32,
    half: u32,
    max_seq: u32,
    eps: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> qqr_x: array<f32>;
@group(0) @binding(1) var<storage, read> qqr_g: array<f32>;
@group(0) @binding(2) var<storage, read> qqr_pos: array<u32>;
@group(0) @binding(3) var<storage, read> qqr_cos: array<f32>;
@group(0) @binding(4) var<storage, read> qqr_sin: array<f32>;
@group(0) @binding(5) var<storage, read> qqr_dy: array<f32>;
@group(0) @binding(6) var<storage, read_write> qqr_dx: array<f32>;
@group(0) @binding(7) var<uniform> qqr_p: QqrParams;

var<workgroup> qqr_red_sq: array<f32, 256>;
var<workgroup> qqr_red_dot: array<f32, 256>;

fn qqr_dn(base: u32, i: u32, pr: u32) -> f32 {
    let half = qqr_p.half;
    if (i < half) {
        let c = qqr_cos[pr * half + i];
        let s = qqr_sin[pr * half + i];
        return qqr_dy[base + i] * c + qqr_dy[base + half + i] * s;
    }
    if (i < 2u * half) {
        let c = qqr_cos[pr * half + (i - half)];
        let s = qqr_sin[pr * half + (i - half)];
        return qqr_dy[base + i] * c - qqr_dy[base + (i - half)] * s;
    }
    return qqr_dy[base + i];
}

@compute @workgroup_size(256)
fn q38_qknorm_rope_bwd(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= qqr_p.tokens * qqr_p.heads) {
        return;
    }
    let lid = tid.x;
    let hidden = qqr_p.hidden;
    let base = row * hidden;
    let t = row / qqr_p.heads;
    let pr = min(qqr_pos[t], qqr_p.max_seq - 1u);

    var sq = 0.0;
    var dot = 0.0;
    for (var i = lid; i < hidden; i = i + 256u) {
        let xv = qqr_x[base + i];
        sq = fma(xv, xv, sq);
        dot = fma(qqr_dn(base, i, pr) * qqr_g[i], xv, dot);
    }
    qqr_red_sq[lid] = sq;
    qqr_red_dot[lid] = dot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            qqr_red_sq[lid] = qqr_red_sq[lid] + qqr_red_sq[lid + s];
            qqr_red_dot[lid] = qqr_red_dot[lid] + qqr_red_dot[lid + s];
        }
        workgroupBarrier();
    }
    let h = f32(hidden);
    let r = inverseSqrt(qqr_red_sq[0] / h + qqr_p.eps);
    let coef = r * r * r * qqr_red_dot[0] / h;
    for (var i = lid; i < hidden; i = i + 256u) {
        qqr_dx[base + i] = fma(qqr_dn(base, i, pr) * qqr_g[i], r, -(qqr_x[base + i] * coef));
    }
}
