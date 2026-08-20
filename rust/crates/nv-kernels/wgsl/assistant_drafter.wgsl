struct AdEmbedParams {
    bh: u32,
    rows_per_chunk: u32,
    norm: f32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> ade_t0: array<u32>;
@group(0) @binding(1) var<storage, read> ade_t1: array<u32>;
@group(0) @binding(2) var<storage, read> ade_t2: array<u32>;
@group(0) @binding(3) var<storage, read> ade_t3: array<u32>;
@group(0) @binding(4) var<storage, read> ade_tok: array<u32>;
@group(0) @binding(5) var<storage, read> ade_hidden: array<f32>;
@group(0) @binding(6) var<storage, read_write> ade_out: array<f32>;
@group(0) @binding(7) var<uniform> ade_params: AdEmbedParams;

fn ade_at(c: u32, i: u32) -> u32 {
    if (c == 0u) { return ade_t0[i]; }
    if (c == 1u) { return ade_t1[i]; }
    if (c == 2u) { return ade_t2[i]; }
    return ade_t3[i];
}

@compute @workgroup_size(256)
fn ad_embed_concat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let bh = ade_params.bh;
    let i = gid.x;
    if (i >= bh) {
        return;
    }
    let row = ade_tok[0];
    let c = row / ade_params.rows_per_chunk;
    let e = (row % ade_params.rows_per_chunk) * bh + i;
    let word = ade_at(c, e >> 1u);
    let bits = (word >> (16u * (e & 1u))) & 0xffffu;
    ade_out[i] = bf16_decode(bits) * ade_params.norm;
    ade_out[bh + i] = ade_hidden[i];
}

struct AdGemvParams {
    n: u32,
    k: u32,
    act: u32,
    mode: u32,
};

@group(0) @binding(10) var<storage, read> adg_w: array<u32>;
@group(0) @binding(11) var<storage, read> adg_x: array<f32>;
@group(0) @binding(12) var<storage, read_write> adg_y: array<f32>;
@group(0) @binding(13) var<uniform> adg_params: AdGemvParams;

var<workgroup> adg_red: array<f32, 64>;

fn ad_gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = tanh(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

@compute @workgroup_size(64)
fn ad_gemv(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let r = wg.x + wg.y * nwg.x;
    if (r >= adg_params.n) {
        return;
    }
    let k = adg_params.k;
    let base = r * k;
    var acc = 0.0;
    for (var j = tid.x; j < k; j = j + 64u) {
        let e = base + j;
        let word = adg_w[e >> 1u];
        let bits = (word >> (16u * (e & 1u))) & 0xffffu;
        acc = fma(bf16_decode(bits), adg_x[j], acc);
    }
    adg_red[tid.x] = acc;
    workgroupBarrier();
    for (var off = 32u; off > 0u; off = off >> 1u) {
        if (tid.x < off) {
            adg_red[tid.x] = adg_red[tid.x] + adg_red[tid.x + off];
        }
        workgroupBarrier();
    }
    if (tid.x == 0u) {
        var v = adg_red[0];
        if (adg_params.act == 1u) {
            v = ad_gelu_tanh(v);
        }
        if (adg_params.mode == 1u) {
            adg_y[r] = adg_y[r] * v;
        } else {
            adg_y[r] = v;
        }
    }
}

struct AdRmsParams {
    rows: u32,
    dim: u32,
    eps: f32,
    pad0: u32,
};

@group(0) @binding(20) var<storage, read> adr_x: array<f32>;
@group(0) @binding(21) var<storage, read> adr_w: array<u32>;
@group(0) @binding(22) var<storage, read_write> adr_y: array<f32>;
@group(0) @binding(23) var<uniform> adr_params: AdRmsParams;

var<workgroup> adr_red: array<f32, 256>;

@compute @workgroup_size(256)
fn ad_rmsnorm(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x;
    if (row >= adr_params.rows) {
        return;
    }
    let dim = adr_params.dim;
    let base = row * dim;
    var acc = 0.0;
    for (var j = tid.x; j < dim; j = j + 256u) {
        let v = adr_x[base + j];
        acc = fma(v, v, acc);
    }
    adr_red[tid.x] = acc;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (tid.x < off) {
            adr_red[tid.x] = adr_red[tid.x] + adr_red[tid.x + off];
        }
        workgroupBarrier();
    }
    let inv = inverseSqrt(adr_red[0] / f32(dim) + adr_params.eps);
    for (var j = tid.x; j < dim; j = j + 256u) {
        let word = adr_w[j >> 1u];
        let bits = (word >> (16u * (j & 1u))) & 0xffffu;
        adr_y[base + j] = adr_x[base + j] * inv * bf16_decode(bits);
    }
}

struct AdRopeParams {
    nh: u32,
    hd: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(30) var<storage, read_write> adp_q: array<f32>;
@group(0) @binding(31) var<storage, read> adp_cos: array<f32>;
@group(0) @binding(32) var<storage, read> adp_sin: array<f32>;
@group(0) @binding(33) var<uniform> adp_params: AdRopeParams;

@compute @workgroup_size(256)
fn ad_rope(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = adp_params.hd;
    let total = adp_params.nh * hd;
    let i = gid.x;
    if (i >= total) {
        return;
    }
    let d = i % hd;
    let half = hd / 2u;
    var rot = 0.0;
    if (d < half) {
        rot = -adp_q[i + half];
    } else {
        rot = adp_q[i - half];
    }
    adp_q[i] = fma(adp_q[i], adp_cos[d], rot * adp_sin[d]);
}

struct AdAttnParams {
    n_kv: u32,
    nh: u32,
    hd: u32,
    len: u32,
    start: u32,
    stride: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(40) var<storage, read> ada_q: array<f32>;
@group(0) @binding(41) var<storage, read> ada_k: array<u32>;
@group(0) @binding(42) var<storage, read> ada_v: array<u32>;
@group(0) @binding(43) var<storage, read> ada_kscale: array<f32>;
@group(0) @binding(44) var<storage, read> ada_vscale: array<f32>;
@group(0) @binding(45) var<storage, read_write> ada_scores: array<f32>;
@group(0) @binding(46) var<storage, read_write> ada_ctx: array<f32>;
@group(0) @binding(47) var<uniform> ada_params: AdAttnParams;

@compute @workgroup_size(256)
fn ad_attn_scores(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let len = ada_params.len;
    let i = wg.x * 256u + tid.x;
    let head = wg.y;
    if (i >= len || head >= ada_params.nh) {
        return;
    }
    let hd = ada_params.hd;
    let group = ada_params.nh / ada_params.n_kv;
    let kvh = head / group;
    let slot = (ada_params.start + i) * ada_params.n_kv + kvh;
    let kbase = slot * hd;
    var acc = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        let idx = kbase + d;
        let kd = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ada_k[idx >> 2u], idx));
        acc = fma(ada_q[head * hd + d], kd, acc);
    }
    ada_scores[head * ada_params.stride + i] = acc * (ada_kscale[slot] * bitcast<f32>(0x7B800000u));
}

var<workgroup> ads_red: array<f32, 256>;

@compute @workgroup_size(256)
fn ad_attn_softmax(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x;
    if (head >= ada_params.nh) {
        return;
    }
    let len = ada_params.len;
    let base = head * ada_params.stride;
    var m = bitcast<f32>(0xff800000u);
    for (var i = tid.x; i < len; i = i + 256u) {
        m = max(m, ada_scores[base + i]);
    }
    ads_red[tid.x] = m;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (tid.x < off) {
            ads_red[tid.x] = max(ads_red[tid.x], ads_red[tid.x + off]);
        }
        workgroupBarrier();
    }
    let mx = ads_red[0];
    workgroupBarrier();
    var s = 0.0;
    for (var i = tid.x; i < len; i = i + 256u) {
        let e = exp(ada_scores[base + i] - mx);
        ada_scores[base + i] = e;
        s = s + e;
    }
    ads_red[tid.x] = s;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (tid.x < off) {
            ads_red[tid.x] = ads_red[tid.x] + ads_red[tid.x + off];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / ads_red[0];
    for (var i = tid.x; i < len; i = i + 256u) {
        ada_scores[base + i] = ada_scores[base + i] * inv;
    }
}

@compute @workgroup_size(256)
fn ad_attn_ctx(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hd = ada_params.hd;
    let total = ada_params.nh * hd;
    let i = gid.x;
    if (i >= total) {
        return;
    }
    let head = i / hd;
    let d = i % hd;
    let group = ada_params.nh / ada_params.n_kv;
    let kvh = head / group;
    let len = ada_params.len;
    var acc = 0.0;
    for (var s = 0u; s < len; s = s + 1u) {
        let slot = (ada_params.start + s) * ada_params.n_kv + kvh;
        let idx = slot * hd + d;
        let vd = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ada_v[idx >> 2u], idx));
        acc = fma(ada_scores[head * ada_params.stride + s] * (ada_vscale[slot] * bitcast<f32>(0x7B800000u)), vd, acc);
    }
    ada_ctx[i] = acc;
}

struct AdAddParams {
    n: u32,
    pad0: u32,
    scale: f32,
    pad1: u32,
};

@group(0) @binding(50) var<storage, read> add_a: array<f32>;
@group(0) @binding(51) var<storage, read> add_b: array<f32>;
@group(0) @binding(52) var<storage, read_write> add_y: array<f32>;
@group(0) @binding(53) var<uniform> add_params: AdAddParams;

@compute @workgroup_size(256)
fn ad_add_scale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= add_params.n) {
        return;
    }
    add_y[i] = (add_a[i] + add_b[i]) * add_params.scale;
}

struct AdTopkParams {
    n: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(60) var<storage, read_write> adt_x: array<f32>;
@group(0) @binding(61) var<storage, read_write> adt_idx: array<u32>;
@group(0) @binding(62) var<uniform> adt_params: AdTopkParams;

var<workgroup> adt_val: array<f32, 256>;
var<workgroup> adt_arg: array<u32, 256>;

@compute @workgroup_size(256)
fn ad_topk(@builtin(local_invocation_id) tid: vec3<u32>) {
    let n = adt_params.n;
    for (var round = 0u; round < adt_params.k; round = round + 1u) {
        var best = bitcast<f32>(0xff800000u);
        var barg = 0xffffffffu;
        for (var i = tid.x; i < n; i = i + 256u) {
            let v = adt_x[i];
            if (v > best || (v == best && i < barg)) {
                best = v;
                barg = i;
            }
        }
        adt_val[tid.x] = best;
        adt_arg[tid.x] = barg;
        workgroupBarrier();
        for (var off = 128u; off > 0u; off = off >> 1u) {
            if (tid.x < off) {
                let v = adt_val[tid.x + off];
                let a = adt_arg[tid.x + off];
                if (v > adt_val[tid.x] || (v == adt_val[tid.x] && a < adt_arg[tid.x])) {
                    adt_val[tid.x] = v;
                    adt_arg[tid.x] = a;
                }
            }
            workgroupBarrier();
        }
        if (tid.x == 0u) {
            adt_idx[round] = adt_arg[0];
            adt_x[adt_arg[0]] = bitcast<f32>(0xff800000u);
        }
        workgroupBarrier();
    }
}

struct AdCandParams {
    top: u32,
    vpc: u32,
    h: u32,
    pad0: u32,
};

@group(0) @binding(70) var<storage, read> adc_top: array<u32>;
@group(0) @binding(71) var<storage, read> adc_order: array<u32>;
@group(0) @binding(72) var<storage, read> adc_head: array<u32>;
@group(0) @binding(73) var<storage, read> adc_hn: array<f32>;
@group(0) @binding(74) var<storage, read_write> adc_ids: array<u32>;
@group(0) @binding(75) var<storage, read_write> adc_logits: array<f32>;
@group(0) @binding(76) var<uniform> adc_params: AdCandParams;

@compute @workgroup_size(64)
fn ad_cand_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = adc_params.top * adc_params.vpc;
    let j = gid.x;
    if (j >= total) {
        return;
    }
    let c = adc_top[j / adc_params.vpc];
    let id = adc_order[c * adc_params.vpc + (j % adc_params.vpc)];
    let h = adc_params.h;
    let base = id * h;
    var acc = 0.0;
    for (var m = 0u; m < h; m = m + 1u) {
        let e = base + m;
        let word = adc_head[e >> 1u];
        let bits = (word >> (16u * (e & 1u))) & 0xffffu;
        acc = fma(bf16_decode(bits), adc_hn[m], acc);
    }
    adc_ids[j] = id;
    adc_logits[j] = acc;
}

struct AdPickParams {
    n: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(80) var<storage, read> adk_ids: array<u32>;
@group(0) @binding(81) var<storage, read> adk_logits: array<f32>;
@group(0) @binding(82) var<storage, read_write> adk_tok: array<u32>;
@group(0) @binding(83) var<storage, read_write> adk_steps: array<u32>;
@group(0) @binding(84) var<storage, read_write> adk_count: array<u32>;
@group(0) @binding(85) var<uniform> adk_params: AdPickParams;

var<workgroup> adk_val: array<f32, 256>;
var<workgroup> adk_arg: array<u32, 256>;

@compute @workgroup_size(256)
fn ad_pick(@builtin(local_invocation_id) tid: vec3<u32>) {
    let n = adk_params.n;
    var best = bitcast<f32>(0xff800000u);
    var barg = 0xffffffffu;
    for (var i = tid.x; i < n; i = i + 256u) {
        let v = adk_logits[i];
        if (v > best || (v == best && i < barg)) {
            best = v;
            barg = i;
        }
    }
    adk_val[tid.x] = best;
    adk_arg[tid.x] = barg;
    workgroupBarrier();
    for (var off = 128u; off > 0u; off = off >> 1u) {
        if (tid.x < off) {
            let v = adk_val[tid.x + off];
            let a = adk_arg[tid.x + off];
            if (v > adk_val[tid.x] || (v == adk_val[tid.x] && a < adk_arg[tid.x])) {
                adk_val[tid.x] = v;
                adk_arg[tid.x] = a;
            }
        }
        workgroupBarrier();
    }
    if (tid.x == 0u) {
        let id = adk_ids[adk_arg[0]];
        adk_tok[0] = id;
        adk_steps[adk_count[0]] = id;
        adk_count[0] = adk_count[0] + 1u;
    }
}
