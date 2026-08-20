
struct GpnParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> pn_x: array<u32>;
@group(0) @binding(1) var<storage, read> pn_w: array<u32>;
@group(0) @binding(2) var<storage, read_write> pn_y: array<u32>;
@group(0) @binding(3) var<uniform> pn_p: GpnParams;
@group(0) @binding(4) var<storage, read_write> pn_res: array<u32>;
@group(0) @binding(5) var<storage, read> pn_w2: array<u32>;
@group(0) @binding(6) var<storage, read> pn_h1: array<u32>;
@group(0) @binding(7) var<storage, read_write> pn_y2: array<u32>;

const PN_FUSED_MAX_WORDS: u32 = 2048u;

var<workgroup> pn_buf: array<u32, 2048>;

var<workgroup> pn_red: array<f32, 256>;
var<workgroup> pn_s: f32;

fn pn_reduce(lid: u32, local: f32) -> f32 {
    pn_red[lid] = local;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            pn_red[lid] = pn_red[lid] + pn_red[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        pn_s = inverseSqrt(pn_red[0] / f32(pn_p.hidden) + pn_p.eps);
    }
    workgroupBarrier();
    return pn_s;
}

@compute @workgroup_size(256)
fn g4m_norm(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s = pn_reduce(lid, local);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let ww = pn_w[w];
        pn_y[w] = bf16_pack(bf16_lo(xw) * s * bf16_lo(ww), bf16_hi(xw) * s * bf16_hi(ww));
    }
}

@compute @workgroup_size(256)
fn g4m_norm_residual(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let rw = pn_res[w];
        let s0 = bf16_lo(xw) + bf16_lo(rw);
        let s1 = bf16_hi(xw) + bf16_hi(rw);
        pn_res[w] = bf16_pack(s0, s1);
        let sr = pn_res[w];
        let r0 = bf16_lo(sr);
        let r1 = bf16_hi(sr);
        local = local + r0 * r0 + r1 * r1;
    }
    let s = pn_reduce(lid, local);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let sr = pn_res[w];
        let ww = pn_w[w];
        pn_y[w] = bf16_pack(bf16_lo(sr) * s * bf16_lo(ww), bf16_hi(sr) * s * bf16_hi(ww));
    }
}

@compute @workgroup_size(256)
fn g4m_norm_mul(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s = pn_reduce(lid, local);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let ww = pn_w[w];
        let t = bf16_pack(bf16_lo(xw) * s, bf16_hi(xw) * s);
        pn_y[w] = bf16_pack(bf16_lo(t) * bf16_lo(ww), bf16_hi(t) * bf16_hi(ww));
    }
}

@compute @workgroup_size(256)
fn g4m_norm_norm_residual(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s1 = pn_reduce(lid, local);
    var local2 = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let w1 = pn_w[w];
        let t = bf16_pack(bf16_lo(xw) * s1 * bf16_lo(w1), bf16_hi(xw) * s1 * bf16_hi(w1));
        let rw = pn_res[w];
        let sum = bf16_pack(bf16_lo(t) + bf16_lo(rw), bf16_hi(t) + bf16_hi(rw));
        pn_res[w] = sum;
        let r0 = bf16_lo(sum);
        let r1 = bf16_hi(sum);
        local2 = local2 + r0 * r0 + r1 * r1;
    }
    let s2 = pn_reduce(lid, local2);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let sr = pn_res[w];
        let w2 = pn_w2[w];
        pn_y[w] = bf16_pack(bf16_lo(sr) * s2 * bf16_lo(w2), bf16_hi(sr) * s2 * bf16_hi(w2));
    }
}

@compute @workgroup_size(256)
fn g4m_norm_x2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>
) {
    let lid = lid3.x;
    let second = wid.x == 1u;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        var xw = pn_x[w];
        if (second) {
            xw = pn_res[w];
        }
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s = pn_reduce(lid, local);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        var xw = pn_x[w];
        var ww = pn_w[w];
        if (second) {
            xw = pn_res[w];
            ww = pn_w2[w];
        }
        let out = bf16_pack(bf16_lo(xw) * s * bf16_lo(ww), bf16_hi(xw) * s * bf16_hi(ww));
        if (second) {
            pn_y2[w] = out;
        } else {
            pn_y[w] = out;
        }
    }
}

@compute @workgroup_size(256)
fn g4m_norm_add_norm_resout(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let lid = lid3.x;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s1 = pn_reduce(lid, local);
    var local2 = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[w];
        let ww = pn_w[w];
        let h2 = bf16_pack(bf16_lo(xw) * s1 * bf16_lo(ww), bf16_hi(xw) * s1 * bf16_hi(ww));
        let h1w = pn_h1[w];
        let sum = bf16_pack(
            (bf16_lo(h1w) + bf16_lo(h2)) * 1.0,
            (bf16_hi(h1w) + bf16_hi(h2)) * 1.0
        );
        pn_buf[w] = sum;
        let r0 = bf16_lo(sum);
        let r1 = bf16_hi(sum);
        local2 = local2 + r0 * r0 + r1 * r1;
    }
    let s2 = pn_reduce(lid, local2);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let sum = pn_buf[w];
        let w3 = pn_w2[w];
        let comb = bf16_pack(bf16_lo(sum) * s2 * bf16_lo(w3), bf16_hi(sum) * s2 * bf16_hi(w3));
        let rw = pn_res[w];
        pn_y[w] = bf16_pack(
            (bf16_lo(comb) + bf16_lo(rw)) * pn_p.scale,
            (bf16_hi(comb) + bf16_hi(rw)) * pn_p.scale
        );
    }
}
