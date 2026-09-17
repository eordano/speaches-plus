
struct QcsParams {
    batch: u32,
    seq: u32,
    channels: u32,
    ktaps: u32,
};

@group(0) @binding(0) var<storage, read> qcs_x: array<f32>;
@group(0) @binding(1) var<storage, read> qcs_w: array<f32>;
@group(0) @binding(2) var<storage, read> qcs_dy: array<f32>;
@group(0) @binding(3) var<storage, read_write> qcs_dx: array<f32>;
@group(0) @binding(4) var<storage, read_write> qcs_dw: array<f32>;
@group(0) @binding(5) var<uniform> qcs_p: QcsParams;

var<workgroup> qcs_red: array<f32, 256>;

fn qcs_pre(bi: u32, r: u32, ci: u32) -> f32 {
    var acc = 0.0;
    for (var j = 0u; j < qcs_p.ktaps; j = j + 1u) {
        let xi = i32(r + j) - i32(qcs_p.ktaps - 1u);
        if (xi >= 0) {
            acc = acc + qcs_w[ci * qcs_p.ktaps + j]
                * qcs_x[((bi * qcs_p.seq + u32(xi)) * qcs_p.channels) + ci];
        }
    }
    return acc;
}

fn qcs_da(bi: u32, r: u32, ci: u32) -> f32 {
    let p = qcs_pre(bi, r, ci);
    let sg = 1.0 / (1.0 + exp(-p));
    return qcs_dy[((bi * qcs_p.seq + r) * qcs_p.channels) + ci] * sg * (1.0 + p * (1.0 - sg));
}

@compute @workgroup_size(256)
fn q38_conv_silu_bwd_dx(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    let total = qcs_p.batch * qcs_p.seq * qcs_p.channels;
    if (idx >= total) {
        return;
    }
    let ci = idx % qcs_p.channels;
    let rest = idx / qcs_p.channels;
    let s = rest % qcs_p.seq;
    let bi = rest / qcs_p.seq;
    var acc = 0.0;
    for (var u = 0u; u < qcs_p.ktaps; u = u + 1u) {
        let r = s + u;
        if (r < qcs_p.seq) {
            acc = acc + qcs_w[ci * qcs_p.ktaps + (qcs_p.ktaps - 1u - u)] * qcs_da(bi, r, ci);
        }
    }
    qcs_dx[idx] = acc;
}

@compute @workgroup_size(256)
fn q38_conv_silu_bwd_dw(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let cj = wg.x + wg.y * nwg.x;
    let lid = tid.x;
    if (cj >= qcs_p.channels * qcs_p.ktaps) {
        return;
    }
    let ci = cj / qcs_p.ktaps;
    let j = cj % qcs_p.ktaps;
    let n = qcs_p.batch * qcs_p.seq;
    var acc = 0.0;
    for (var i = lid; i < n; i = i + 256u) {
        let bi = i / qcs_p.seq;
        let ti = i % qcs_p.seq;
        let xi = i32(ti + j) - i32(qcs_p.ktaps - 1u);
        if (xi >= 0) {
            acc = acc + qcs_da(bi, ti, ci)
                * qcs_x[((bi * qcs_p.seq + u32(xi)) * qcs_p.channels) + ci];
        }
    }
    qcs_red[lid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            qcs_red[lid] = qcs_red[lid] + qcs_red[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        qcs_dw[cj] = qcs_red[0u];
    }
}
