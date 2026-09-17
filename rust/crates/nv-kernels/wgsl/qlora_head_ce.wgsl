
struct HP {
    m: u32,
    n: u32,
    k: u32,
    ns: u32,
    span: u32,
    cap: f32,
    pad1: u32,
    pad2: u32,
}

fn capped(l: f32) -> f32 {
    if (hp.cap > 0.0) {
        return hp.cap * tanh(l / hp.cap);
    }
    return l;
}

fn cap_slope(l: f32) -> f32 {
    if (hp.cap > 0.0) {
        let t = tanh(l / hp.cap);
        return 1.0 - t * t;
    }
    return 1.0;
}

@group(0) @binding(0) var<uniform> hp: HP;
@group(0) @binding(1) var<storage, read> xb: array<f32>;
@group(0) @binding(2) var<storage, read> wq: array<u32>;
@group(0) @binding(3) var<storage, read> sc: array<f32>;
@group(0) @binding(4) var<storage, read> bi: array<f32>;
@group(0) @binding(5) var<storage, read_write> lg: array<f32>;
@group(0) @binding(6) var<storage, read> tg: array<u32>;
@group(0) @binding(7) var<storage, read> wr: array<f32>;
@group(0) @binding(8) var<storage, read_write> lse: array<f32>;
@group(0) @binding(9) var<storage, read_write> lossr: array<f32>;
@group(0) @binding(10) var<storage, read> gup: array<f32>;
@group(0) @binding(11) var<storage, read_write> dxp: array<f32>;
@group(0) @binding(12) var<storage, read_write> dx: array<f32>;

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn w4h_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= hp.m * hp.n) {
        return;
    }
    let mi = idx / hp.n;
    let nr = idx % hp.n;
    let wpr = hp.k >> 3u;
    let groups = hp.k >> 6u;
    var acc = 0.0;
    for (var kw = 0u; kw < wpr; kw = kw + 1u) {
        let word = wq[nr * wpr + kw];
        let grp = kw >> 3u;
        let s = sc[nr * groups + grp];
        let b = bi[nr * groups + grp];
        let x0 = mi * hp.k + kw * 8u;
        for (var e = 0u; e < 8u; e = e + 1u) {
            let w = fma(f32((word >> (4u * e)) & 0xFu), s, b);
            acc = fma(w, xb[x0 + e], acc);
        }
    }
    lg[idx] = acc;
}

@compute @workgroup_size(256)
fn w4h_ce_stats(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let tid = lid.x;
    let base = row * hp.n;
    var mx = -3.0e38;
    for (var i = tid; i < hp.n; i = i + 256u) {
        mx = max(mx, capped(lg[base + i]));
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
    for (var i = tid; i < hp.n; i = i + 256u) {
        sum = sum + exp(capped(lg[base + i]) - mx);
    }
    red[tid] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            red[tid] = red[tid] + red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let l = mx + log(red[0]);
        lse[row] = l;
        lossr[row] = wr[row] * (l - capped(lg[base + tg[row]]));
    }
}

@compute @workgroup_size(256)
fn w4h_dlogits_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= hp.m * hp.n) {
        return;
    }
    let row = idx / hp.n;
    let col = idx % hp.n;
    let wg = wr[row] * gup[row];
    var onehot = 0.0;
    if (col == tg[row]) {
        onehot = 1.0;
    }
    let raw = lg[idx];
    lg[idx] = wg * (exp(capped(raw) - lse[row]) - onehot) * cap_slope(raw);
}

@compute @workgroup_size(256)
fn w4h_dx_partial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let mk = hp.m * hp.k;
    if (idx >= hp.ns * mk) {
        return;
    }
    let z = idx / mk;
    let rem = idx % mk;
    let mi = rem / hp.k;
    let kc = rem % hp.k;
    let wpr = hp.k >> 3u;
    let groups = hp.k >> 6u;
    let kw = kc >> 3u;
    let sh = 4u * (kc & 7u);
    let grp = kc >> 6u;
    let n0 = z * hp.span;
    let n1 = min(n0 + hp.span, hp.n);
    var acc = 0.0;
    for (var nr = n0; nr < n1; nr = nr + 1u) {
        let w = fma(f32((wq[nr * wpr + kw] >> sh) & 0xFu), sc[nr * groups + grp], bi[nr * groups + grp]);
        acc = fma(lg[mi * hp.n + nr], w, acc);
    }
    dxp[idx] = acc;
}

@compute @workgroup_size(256)
fn w4h_dx_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let mk = hp.m * hp.k;
    if (idx >= mk) {
        return;
    }
    var s = 0.0;
    for (var z = 0u; z < hp.ns; z = z + 1u) {
        s = s + dxp[z * mk + idx];
    }
    dx[idx] = s;
}
