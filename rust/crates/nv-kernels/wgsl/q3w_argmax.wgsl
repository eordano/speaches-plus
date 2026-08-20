struct Q3amParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(40) var<storage, read> am_x: array<u32>;
@group(0) @binding(41) var<storage, read_write> am_pv: array<f32>;
@group(0) @binding(42) var<storage, read_write> am_pi: array<u32>;
@group(0) @binding(43) var<storage, read_write> am_out: array<u32>;
@group(0) @binding(44) var<uniform> am_p: Q3amParams;

var<workgroup> am_v: array<f32, 256>;
var<workgroup> am_i: array<u32, 256>;

fn am_reduce(tid: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let o = tid + s;
            if (am_v[o] > am_v[tid] || (am_v[o] == am_v[tid] && am_i[o] < am_i[tid])) {
                am_v[tid] = am_v[o];
                am_i[tid] = am_i[o];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn q3w_argmax_stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let g = wid.x;
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    for (var i = g * 256u + tid; i < am_p.n; i = i + am_p.groups * 256u) {
        let v = bitcast<f32>(am_x[i]);
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    am_v[tid] = bv;
    am_i[tid] = bi;
    workgroupBarrier();
    am_reduce(tid);
    if (tid == 0u) {
        am_pv[g] = am_v[0];
        am_pi[g] = am_i[0];
    }
}

@compute @workgroup_size(256)
fn q3w_argmax_stage2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    if (tid < am_p.groups) {
        bv = am_pv[tid];
        bi = am_pi[tid];
    }
    am_v[tid] = bv;
    am_i[tid] = bi;
    workgroupBarrier();
    am_reduce(tid);
    if (tid == 0u) {
        am_out[0] = am_i[0];
    }
}
