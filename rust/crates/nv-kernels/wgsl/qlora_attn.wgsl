
struct P {
    g: u32,
    m: u32,
    sk: u32,
    d: u32,
    sq: u32,
    causal: u32,
    window: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qg: array<f32>;
@group(0) @binding(2) var<storage, read> kg: array<f32>;
@group(0) @binding(3) var<storage, read> vg: array<f32>;
@group(0) @binding(4) var<storage, read_write> sp: array<f32>;
@group(0) @binding(5) var<storage, read_write> og: array<f32>;
@group(0) @binding(6) var<storage, read> dog: array<f32>;
@group(0) @binding(7) var<storage, read_write> dsb: array<f32>;
@group(0) @binding(8) var<storage, read_write> drow: array<f32>;
@group(0) @binding(9) var<storage, read_write> dqb: array<f32>;
@group(0) @binding(10) var<storage, read_write> dkb: array<f32>;
@group(0) @binding(11) var<storage, read_write> dvb: array<f32>;

const MASKED_SCORE: f32 = -3.0e38;

var<workgroup> red: array<f32, 256>;

fn is_masked(r: u32, j: u32) -> bool {
    if (p.causal == 0u) {
        return false;
    }
    let i = r % p.sq;
    var off = 0u;
    if (p.sk >= p.sq) {
        off = p.sk - p.sq;
    }
    let last = i + off;
    var first = 0u;
    if (p.window > 0u && last + 1u > p.window) {
        first = last + 1u - p.window;
    }
    return j > last || j < first;
}

@compute @workgroup_size(256)
fn qla_scores(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.m * p.sk) {
        return;
    }
    let j = idx % p.sk;
    let gr = idx / p.sk;
    let r = gr % p.m;
    let g = gr / p.m;
    if (is_masked(r, j)) {
        sp[idx] = MASKED_SCORE;
        return;
    }
    var acc = 0.0;
    let qb = (g * p.m + r) * p.d;
    let kb = (g * p.sk + j) * p.d;
    for (var e = 0u; e < p.d; e = e + 1u) {
        acc = fma(qg[qb + e], kg[kb + e], acc);
    }
    sp[idx] = acc * p.scale;
}

@compute @workgroup_size(256)
fn qla_softmax(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let base = row * p.sk;
    var mx = MASKED_SCORE;
    for (var i = tid; i < p.sk; i = i + 256u) {
        mx = max(mx, sp[base + i]);
    }
    red[tid] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red[tid] = max(red[tid], red[tid + s]);
        }
        workgroupBarrier();
    }
    mx = red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var i = tid; i < p.sk; i = i + 256u) {
        sum = sum + exp(sp[base + i] - mx);
    }
    red[tid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        workgroupBarrier();
    }
    let z = red[0];
    workgroupBarrier();
    for (var i = tid; i < p.sk; i = i + 256u) {
        sp[base + i] = exp(sp[base + i] - mx) / z;
    }
}

@compute @workgroup_size(256)
fn qla_out(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.m * p.d) {
        return;
    }
    let e = idx % p.d;
    let gr = idx / p.d;
    let r = gr % p.m;
    let g = gr / p.m;
    var acc = 0.0;
    let pb = (g * p.m + r) * p.sk;
    let vb = g * p.sk * p.d;
    for (var j = 0u; j < p.sk; j = j + 1u) {
        acc = fma(sp[pb + j], vg[vb + j * p.d + e], acc);
    }
    og[idx] = acc;
}

@compute @workgroup_size(256)
fn qla_dp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.m * p.sk) {
        return;
    }
    let j = idx % p.sk;
    let gr = idx / p.sk;
    let r = gr % p.m;
    let g = gr / p.m;
    var acc = 0.0;
    let ob = (g * p.m + r) * p.d;
    let vb = (g * p.sk + j) * p.d;
    for (var e = 0u; e < p.d; e = e + 1u) {
        acc = fma(dog[ob + e], vg[vb + e], acc);
    }
    dsb[idx] = acc;
}

@compute @workgroup_size(256)
fn qla_dv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.sk * p.d) {
        return;
    }
    let e = idx % p.d;
    let gj = idx / p.d;
    let j = gj % p.sk;
    let g = gj / p.sk;
    var acc = 0.0;
    for (var r = 0u; r < p.m; r = r + 1u) {
        let gr = g * p.m + r;
        acc = fma(sp[gr * p.sk + j], dog[gr * p.d + e], acc);
    }
    dvb[idx] = acc;
}

@compute @workgroup_size(256)
fn qla_drow(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let base = row * p.sk;
    var acc = 0.0;
    for (var i = tid; i < p.sk; i = i + 256u) {
        acc = fma(sp[base + i], dsb[base + i], acc);
    }
    red[tid] = acc;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        drow[row] = red[0];
    }
}

@compute @workgroup_size(256)
fn qla_ds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.m * p.sk) {
        return;
    }
    let row = idx / p.sk;
    dsb[idx] = p.scale * sp[idx] * (dsb[idx] - drow[row]);
}

@compute @workgroup_size(256)
fn qla_dq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.m * p.d) {
        return;
    }
    let e = idx % p.d;
    let gr = idx / p.d;
    let r = gr % p.m;
    let g = gr / p.m;
    var acc = 0.0;
    let db = (g * p.m + r) * p.sk;
    let kb = g * p.sk * p.d;
    for (var j = 0u; j < p.sk; j = j + 1u) {
        acc = fma(dsb[db + j], kg[kb + j * p.d + e], acc);
    }
    dqb[idx] = acc;
}

@compute @workgroup_size(256)
fn qla_dk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= p.g * p.sk * p.d) {
        return;
    }
    let e = idx % p.d;
    let gj = idx / p.d;
    let j = gj % p.sk;
    let g = gj / p.sk;
    var acc = 0.0;
    for (var r = 0u; r < p.m; r = r + 1u) {
        let gr = g * p.m + r;
        acc = fma(dsb[gr * p.sk + j], qg[gr * p.d + e], acc);
    }
    dkb[idx] = acc;
}
