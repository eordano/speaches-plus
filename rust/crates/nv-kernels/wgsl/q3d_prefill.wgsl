
struct Q3ckParams {
    m_live: u32,
    base: u32,
    pad0: u32,
    pad1: u32,
};

struct Q3geParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
};

@group(0) @binding(0) var<storage, read> pge_emb: array<u32>;
@group(0) @binding(1) var<storage, read> pge_tok: array<i32>;
@group(0) @binding(2) var<storage, read_write> pge_out: array<u32>;
@group(0) @binding(3) var<uniform> pge_p: Q3geParams;

@compute @workgroup_size(256)
fn q3w_gather_embed_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    var s = 0u;
    if (pge_tok[t] > 0) {
        s = u32(pge_tok[t]);
    }
    if (s >= pge_p.vocab) {
        s = 0u;
    }
    if (s < pge_p.row_off) {
        return;
    }
    if (s >= pge_p.row_off + pge_p.n_rows) {
        return;
    }
    let base = (s - pge_p.row_off) * pge_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= pge_p.hidden_words) {
        return;
    }
    pge_out[t * pge_p.hidden_words + w] = pge_emb[base + w];
}

struct Q3cvmParams {
    conv_dim: u32,
    kernel: u32,
    x_words: u32,
    mixed_stride: u32,
};

@group(0) @binding(10) var<storage, read> pcv_x: array<u32>;
@group(0) @binding(11) var<storage, read> pcv_w: array<f32>;
@group(0) @binding(12) var<storage, read_write> pcv_state: array<f32>;
@group(0) @binding(13) var<storage, read_write> pcv_out: array<f32>;
@group(0) @binding(14) var<uniform> pcv_p: Q3cvmParams;
@group(0) @binding(15) var<uniform> pcv_ck: Q3ckParams;

fn pcv_xv(t: u32, c: u32) -> f32 {
    let e = t * pcv_p.x_words * 2u + c;
    return bf16_decode(u16_at(pcv_x[e >> 1u], e));
}

@compute @workgroup_size(64)
fn q3w_delta_conv_m(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let t = gid.y;
    if (c >= pcv_p.conv_dim) {
        return;
    }
    let ks = pcv_p.kernel;
    let hist = ks - 1u;
    var acc = 0.0;
    for (var j = 0u; j < hist; j = j + 1u) {
        let idx = i32(t + j) - i32(hist);
        var v = 0.0;
        if (idx < 0) {
            v = pcv_state[c * hist + t + j];
        } else {
            v = pcv_xv(u32(idx), c);
        }
        acc = fma(pcv_w[c * ks + j], v, acc);
    }
    acc = fma(pcv_w[c * ks + hist], pcv_xv(t, c), acc);
    let silu = acc / (1.0 + exp(-acc));
    pcv_out[t * pcv_p.mixed_stride + c] = bf16_decode(bf16_encode(silu));
}

@compute @workgroup_size(64)
fn q3w_delta_conv_shift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= pcv_p.conv_dim) {
        return;
    }
    let hist = pcv_p.kernel - 1u;
    let live = pcv_ck.m_live;
    var tmp: array<f32, 8>;
    for (var j = 0u; j < hist; j = j + 1u) {
        let idx = i32(live + j) - i32(hist);
        if (idx < 0) {
            tmp[j] = pcv_state[c * hist + live + j];
        } else {
            tmp[j] = pcv_xv(u32(idx), c);
        }
    }
    for (var j = 0u; j < hist; j = j + 1u) {
        pcv_state[c * hist + j] = tmp[j];
    }
}

struct Q3dqmParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    mixed_stride: u32,
    pad0: u32,
    scale: f32,
};

@group(0) @binding(20) var<storage, read> pdq_mixed: array<f32>;
@group(0) @binding(21) var<storage, read_write> pdq_q: array<f32>;
@group(0) @binding(22) var<storage, read_write> pdq_k: array<f32>;
@group(0) @binding(23) var<storage, read_write> pdq_v: array<f32>;
@group(0) @binding(24) var<uniform> pdq_p: Q3dqmParams;

var<workgroup> pdq_rq: array<f32, 128>;
var<workgroup> pdq_rk: array<f32, 128>;

@compute @workgroup_size(128)
fn q3w_delta_qkv_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let t = wid.y;
    let d = lid.x;
    let kh = h / pdq_p.v_per_k;
    let mbase = t * pdq_p.mixed_stride;
    var qv = 0.0;
    var kv = 0.0;
    if (d < pdq_p.d_k) {
        qv = pdq_mixed[mbase + kh * pdq_p.d_k + d];
        kv = pdq_mixed[mbase + pdq_p.key_dim + kh * pdq_p.d_k + d];
    }
    pdq_rq[d] = qv * qv;
    pdq_rk[d] = kv * kv;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (d < s) {
            pdq_rq[d] = pdq_rq[d] + pdq_rq[d + s];
            pdq_rk[d] = pdq_rk[d] + pdq_rk[d + s];
        }
        workgroupBarrier();
    }
    let nq = sqrt(pdq_rq[0] + 1e-6);
    let nk = sqrt(pdq_rk[0] + 1e-6);
    if (d < pdq_p.d_k) {
        pdq_q[(t * pdq_p.n_v + h) * pdq_p.d_k + d] = (qv / nq) * pdq_p.scale;
        pdq_k[(t * pdq_p.n_v + h) * pdq_p.d_k + d] = kv / nk;
    }
    if (d < pdq_p.d_v) {
        pdq_v[(t * pdq_p.n_v + h) * pdq_p.d_v + d] =
            pdq_mixed[mbase + 2u * pdq_p.key_dim + h * pdq_p.d_v + d];
    }
}

struct Q3gmpParams {
    n_v: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<storage, read> pdg_ab: array<u32>;
@group(0) @binding(31) var<storage, read> pdg_alog: array<f32>;
@group(0) @binding(32) var<storage, read> pdg_dt: array<f32>;
@group(0) @binding(33) var<storage, read_write> pdg_g: array<f32>;
@group(0) @binding(34) var<storage, read_write> pdg_beta: array<f32>;
@group(0) @binding(35) var<uniform> pdg_p: Q3gmpParams;

@compute @workgroup_size(64)
fn q3w_delta_gating_m(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= pdg_p.n_v) {
        return;
    }
    let ebase = t * 2u * pdg_p.n_v;
    let ea = ebase + i;
    let a = bf16_decode(u16_at(pdg_ab[ea >> 1u], ea));
    let eb = ebase + pdg_p.n_v + i;
    let b = bf16_decode(u16_at(pdg_ab[eb >> 1u], eb));
    pdg_beta[t * pdg_p.n_v + i] = 1.0 / (1.0 + exp(-b));
    let tt = a + pdg_dt[i];
    let sp = max(tt, 0.0) + log(1.0 + exp(-abs(tt)));
    pdg_g[t * pdg_p.n_v + i] = exp(sp * (-exp(pdg_alog[i])));
}

struct Q3rmParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
};

@group(0) @binding(40) var<storage, read> pdr_q: array<f32>;
@group(0) @binding(41) var<storage, read> pdr_k: array<f32>;
@group(0) @binding(42) var<storage, read> pdr_v: array<f32>;
@group(0) @binding(43) var<storage, read> pdr_g: array<f32>;
@group(0) @binding(44) var<storage, read> pdr_beta: array<f32>;
@group(0) @binding(45) var<storage, read_write> pdr_out: array<f32>;
@group(0) @binding(46) var<storage, read_write> pdr_state: array<f32>;
@group(0) @binding(47) var<uniform> pdr_p: Q3rmParams;
@group(0) @binding(48) var<uniform> pdr_ck: Q3ckParams;

var<workgroup> pdr_kb: array<f32, 128>;
var<workgroup> pdr_qb: array<f32, 128>;

@compute @workgroup_size(128)
fn q3w_delta_scan(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let lane = lid.x;
    let dk = pdr_p.d_k;
    let dv = pdr_p.d_v;
    let heads = pdr_p.heads;
    let sbase = h * dk * dv + lane;
    for (var t = 0u; t < pdr_ck.m_live; t = t + 1u) {
        if (lane < dk) {
            pdr_kb[lane] = pdr_k[(t * heads + h) * dk + lane];
            pdr_qb[lane] = pdr_q[(t * heads + h) * dk + lane];
        }
        workgroupBarrier();
        if (lane < dv) {
            let ge = pdr_g[t * heads + h];
            let bt = pdr_beta[t * heads + h];
            var kv_mem = 0.0;
            for (var i = 0u; i < dk; i = i + 1u) {
                let s = pdr_state[sbase + i * dv] * ge;
                pdr_state[sbase + i * dv] = s;
                kv_mem = fma(s, pdr_kb[i], kv_mem);
            }
            let delta = (pdr_v[(t * heads + h) * dv + lane] - kv_mem) * bt;
            var outv = 0.0;
            for (var i = 0u; i < dk; i = i + 1u) {
                let s = fma(pdr_kb[i], delta, pdr_state[sbase + i * dv]);
                pdr_state[sbase + i * dv] = s;
                outv = fma(s, pdr_qb[i], outv);
            }
            pdr_out[(t * heads + h) * dv + lane] = outv;
        }
        workgroupBarrier();
    }
}

const WY_C: u32 = 32u;
const WY_D: u32 = 128u;
const WY_VSPLIT: u32 = 4u;
const WY_G_LOG_CLAMP_KEEPS_LOG_FINITE_WHEN_GATING_UNDERFLOWS: f32 = 1e-30;

var<workgroup> wy_kt: array<vec4<f32>, 1024>;
var<workgroup> wy_qt: array<vec4<f32>, 1024>;
var<workgroup> wy_a: array<f32, 1024>;
var<workgroup> wy_w: array<f32, 1024>;
var<workgroup> wy_cl: array<f32, 32>;
var<workgroup> wy_beta: array<f32, 32>;

fn wy_uget(t: u32, col: u32) -> f32 {
    return wy_qt[t * 32u + (col >> 2u)][col & 3u];
}

fn wy_us(s: u32, n: u32, clend: f32, col: u32) -> f32 {
    return select(
        0.0,
        exp(clend - wy_cl[min(s, n - 1u)]) * wy_uget(s, col),
        s < n
    );
}

@compute @workgroup_size(128)
fn q3w_delta_scan_wy(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let lane = lid.x;
    let heads = pdr_p.heads;
    let sub = lane >> 5u;
    let col = lane & 31u;
    let v = wid.y * 32u + col;
    let jb = sub * 2u;
    let ostr = heads * WY_D;
    for (var c0 = 0u; c0 < pdr_ck.m_live; c0 = c0 + WY_C) {
        let n = min(WY_C, pdr_ck.m_live - c0);
        let ob = (c0 * heads + h) * WY_D + v;
        workgroupBarrier();
        storageBarrier();
        if (lane < n) {
            wy_beta[lane] = pdr_beta[(c0 + lane) * heads + h];
            wy_cl[lane] = log(max(
                pdr_g[(c0 + lane) * heads + h],
                WY_G_LOG_CLAMP_KEEPS_LOG_FINITE_WHEN_GATING_UNDERFLOWS
            ));
        }
        workgroupBarrier();
        if (lane == 0u) {
            for (var t = 1u; t < n; t = t + 1u) {
                wy_cl[t] = wy_cl[t] + wy_cl[t - 1u];
            }
        }
        for (var e = lane; e < n * WY_D; e = e + 128u) {
            let t = e >> 7u;
            let i = e & 127u;
            let idx = ((c0 + t) * heads + h) * WY_D + i;
            wy_kt[i * 8u + (t >> 2u)][t & 3u] = pdr_k[idx];
            wy_qt[i * 8u + (t >> 2u)][t & 3u] = pdr_q[idx];
        }
        workgroupBarrier();
        for (var e = lane; e < n * n; e = e + 128u) {
            let t = e / n;
            let s = e % n;
            if (s <= t) {
                let se = s >> 2u;
                let sc = s & 3u;
                let te = t >> 2u;
                let tc = t & 3u;
                var dkk = 0.0;
                var dqk = 0.0;
                for (var i = 0u; i < WY_D; i = i + 1u) {
                    let ks = wy_kt[i * 8u + se][sc];
                    dkk = fma(ks, wy_kt[i * 8u + te][tc], dkk);
                    dqk = fma(ks, wy_qt[i * 8u + te][tc], dqk);
                }
                let decay = exp(wy_cl[t] - wy_cl[s]);
                if (s < t) {
                    wy_a[t * WY_C + s] = wy_beta[t] * decay * dkk;
                }
                wy_w[t * WY_C + s] = decay * dqk;
            }
        }
        var ak0 = vec4<f32>();
        var ak1 = vec4<f32>();
        var aq0 = vec4<f32>();
        var aq1 = vec4<f32>();
        for (var i = 0u; i < WY_D; i = i + 1u) {
            let s4 = vec4<f32>(pdr_state[h * WY_D * WY_D + i * WY_D + v]);
            let kb = i * 8u + jb;
            ak0 = fma(s4, wy_kt[kb], ak0);
            ak1 = fma(s4, wy_kt[kb + 1u], ak1);
            aq0 = fma(s4, wy_qt[kb], aq0);
            aq1 = fma(s4, wy_qt[kb + 1u], aq1);
        }
        workgroupBarrier();
        let tb = sub * 8u;
        if (tb + 0u < n) {
            let t = tb;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak0.x);
            pdr_out[ob + t * ostr] = ge * aq0.x;
        }
        if (tb + 1u < n) {
            let t = tb + 1u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak0.y);
            pdr_out[ob + t * ostr] = ge * aq0.y;
        }
        if (tb + 2u < n) {
            let t = tb + 2u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak0.z);
            pdr_out[ob + t * ostr] = ge * aq0.z;
        }
        if (tb + 3u < n) {
            let t = tb + 3u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak0.w);
            pdr_out[ob + t * ostr] = ge * aq0.w;
        }
        if (tb + 4u < n) {
            let t = tb + 4u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak1.x);
            pdr_out[ob + t * ostr] = ge * aq1.x;
        }
        if (tb + 5u < n) {
            let t = tb + 5u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak1.y);
            pdr_out[ob + t * ostr] = ge * aq1.y;
        }
        if (tb + 6u < n) {
            let t = tb + 6u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak1.z);
            pdr_out[ob + t * ostr] = ge * aq1.z;
        }
        if (tb + 7u < n) {
            let t = tb + 7u;
            let ge = exp(wy_cl[t]);
            wy_qt[t * 32u + (col >> 2u)][col & 3u] =
                wy_beta[t] * (pdr_v[ob + t * ostr] - ge * ak1.w);
            pdr_out[ob + t * ostr] = ge * aq1.w;
        }
        workgroupBarrier();
        if (sub == 0u) {
            for (var s = 1u; s < n; s = s + 1u) {
                var acc = 0.0;
                for (var r = 0u; r < s; r = r + 1u) {
                    acc = fma(wy_a[s * WY_C + r], wy_uget(r, col), acc);
                }
                wy_qt[s * 32u + (col >> 2u)][col & 3u] = wy_uget(s, col) - acc;
            }
        }
        workgroupBarrier();
        for (var c = 0u; c < 8u; c = c + 1u) {
            let t = tb + c;
            if (t >= n) {
                break;
            }
            var sum = 0.0;
            for (var s = 0u; s <= t; s = s + 1u) {
                sum = fma(wy_w[t * WY_C + s], wy_uget(s, col), sum);
            }
            pdr_out[ob + t * ostr] = pdr_out[ob + t * ostr] + sum;
        }
        let clend = wy_cl[n - 1u];
        let gend = exp(clend);
        let us0 = vec4<f32>(
            wy_us(0u, n, clend, col), wy_us(1u, n, clend, col),
            wy_us(2u, n, clend, col), wy_us(3u, n, clend, col));
        let us1 = vec4<f32>(
            wy_us(4u, n, clend, col), wy_us(5u, n, clend, col),
            wy_us(6u, n, clend, col), wy_us(7u, n, clend, col));
        let us2 = vec4<f32>(
            wy_us(8u, n, clend, col), wy_us(9u, n, clend, col),
            wy_us(10u, n, clend, col), wy_us(11u, n, clend, col));
        let us3 = vec4<f32>(
            wy_us(12u, n, clend, col), wy_us(13u, n, clend, col),
            wy_us(14u, n, clend, col), wy_us(15u, n, clend, col));
        let us4 = vec4<f32>(
            wy_us(16u, n, clend, col), wy_us(17u, n, clend, col),
            wy_us(18u, n, clend, col), wy_us(19u, n, clend, col));
        let us5 = vec4<f32>(
            wy_us(20u, n, clend, col), wy_us(21u, n, clend, col),
            wy_us(22u, n, clend, col), wy_us(23u, n, clend, col));
        let us6 = vec4<f32>(
            wy_us(24u, n, clend, col), wy_us(25u, n, clend, col),
            wy_us(26u, n, clend, col), wy_us(27u, n, clend, col));
        let us7 = vec4<f32>(
            wy_us(28u, n, clend, col), wy_us(29u, n, clend, col),
            wy_us(30u, n, clend, col), wy_us(31u, n, clend, col));
        for (var ii = 0u; ii < 32u; ii = ii + 1u) {
            let i = sub * 32u + ii;
            let sb = h * WY_D * WY_D + i * WY_D + v;
            let kb = i * 8u;
            var sv = fma(gend, pdr_state[sb], dot(wy_kt[kb], us0));
            sv = sv + dot(wy_kt[kb + 1u], us1) + dot(wy_kt[kb + 2u], us2);
            sv = sv + dot(wy_kt[kb + 3u], us3) + dot(wy_kt[kb + 4u], us4);
            sv = sv + dot(wy_kt[kb + 5u], us5) + dot(wy_kt[kb + 6u], us6);
            sv = sv + dot(wy_kt[kb + 7u], us7);
            pdr_state[sb] = sv;
        }
    }
}

struct Q3domParams {
    n_v: u32,
    d_v: u32,
    pad0: u32,
    eps: f32,
};

@group(0) @binding(50) var<storage, read> pdo_core: array<f32>;
@group(0) @binding(51) var<storage, read> pdo_w: array<u32>;
@group(0) @binding(52) var<storage, read> pdo_z: array<u32>;
@group(0) @binding(53) var<storage, read_write> pdo_out: array<u32>;
@group(0) @binding(54) var<uniform> pdo_p: Q3domParams;

var<workgroup> pdo_red: array<f32, 128>;

@compute @workgroup_size(128)
fn q3w_delta_out_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let t = wid.y;
    let lane = lid.x;
    let dv = pdo_p.d_v;
    let vdim = pdo_p.n_v * dv;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < dv) {
        v0 = bf16_decode(bf16_encode(pdo_core[t * vdim + h * dv + e0]));
        v1 = bf16_decode(bf16_encode(pdo_core[t * vdim + h * dv + e0 + 1u]));
    }
    pdo_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            pdo_red[lane] = pdo_red[lane] + pdo_red[lane + s];
        }
        workgroupBarrier();
    }
    if (e0 >= dv) {
        return;
    }
    let rms = inverseSqrt(pdo_red[0] / f32(dv) + pdo_p.eps);
    let w0 = bf16_decode(u16_at(pdo_w[e0 >> 1u], e0));
    let w1 = bf16_decode(u16_at(pdo_w[(e0 + 1u) >> 1u], e0 + 1u));
    let zi = t * vdim + h * dv + e0;
    let z0 = bf16_decode(u16_at(pdo_z[zi >> 1u], zi));
    let z1 = bf16_decode(u16_at(pdo_z[(zi + 1u) >> 1u], zi + 1u));
    let g0 = bf16_decode(bf16_encode(z0 / (1.0 + exp(-z0))));
    let g1 = bf16_decode(bf16_encode(z1 / (1.0 + exp(-z1))));
    let n0 = bf16_decode(bf16_encode(v0 * rms * w0));
    let n1 = bf16_decode(bf16_encode(v1 * rms * w1));
    pdo_out[zi >> 1u] = bf16_pack(n0 * g0, n1 * g1);
}

struct Q3armParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    x_row_elems: u32,
    y_row_elems: u32,
    pad0: u32,
    eps: f32,
};

@group(0) @binding(60) var<storage, read> par_src: array<u32>;
@group(0) @binding(61) var<storage, read> par_w: array<u32>;
@group(0) @binding(62) var<storage, read> par_cos: array<f32>;
@group(0) @binding(63) var<storage, read> par_sin: array<f32>;
@group(0) @binding(64) var<storage, read_write> par_out: array<u32>;
@group(0) @binding(65) var<uniform> par_p: Q3armParams;
@group(0) @binding(66) var<uniform> par_ck: Q3ckParams;

var<workgroup> par_red: array<f32, 128>;
var<workgroup> par_buf: array<f32, 256>;

fn par_rope_at(d: u32, p: u32) -> f32 {
    let rh = par_p.rot_half;
    if (d < rh) {
        let c = par_cos[p * rh + d];
        let s = par_sin[p * rh + d];
        return par_buf[d] * c - par_buf[d + rh] * s;
    }
    if (d < 2u * rh) {
        let i = d - rh;
        let c = par_cos[p * rh + i];
        let s = par_sin[p * rh + i];
        return par_buf[i] * s + par_buf[d] * c;
    }
    return par_buf[d];
}

@compute @workgroup_size(128)
fn q3w_attn_norm_rope_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    if (t >= par_ck.m_live) {
        return;
    }
    let r = wid.x;
    let lane = lid.x;
    let hd = par_p.head_dim;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = t * par_p.x_row_elems + r * par_p.src_stride + e0;
        v0 = bf16_decode(u16_at(par_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(par_src[(base + 1u) >> 1u], base + 1u));
    }
    par_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            par_red[lane] = par_red[lane] + par_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(par_red[0] / f32(hd) + par_p.eps);
    if (e0 < hd) {
        let w0 = bf16_decode(u16_at(par_w[e0 >> 1u], e0));
        let w1 = bf16_decode(u16_at(par_w[(e0 + 1u) >> 1u], e0 + 1u));
        par_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        par_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    let p = par_ck.base + t;
    let o0 = par_rope_at(e0, p);
    let o1 = par_rope_at(e0 + 1u, p);
    par_out[(t * par_p.y_row_elems + r * hd + e0) >> 1u] = bf16_pack(o0, o1);
}

struct Q3kvmParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(70) var<storage, read> pkw_k: array<u32>;
@group(0) @binding(71) var<storage, read> pkw_v: array<u32>;
@group(0) @binding(72) var<storage, read_write> pkw_kc: array<u32>;
@group(0) @binding(73) var<storage, read_write> pkw_vc: array<u32>;
@group(0) @binding(74) var<uniform> pkw_p: Q3kvmParams;
@group(0) @binding(75) var<uniform> pkw_ck: Q3ckParams;

@compute @workgroup_size(64)
fn q3w_kv_write_m(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= pkw_p.words || t >= pkw_ck.m_live) {
        return;
    }
    let base = (pkw_ck.base + t) * pkw_p.words;
    pkw_kc[base + i] = pkw_k[t * pkw_p.words + i];
    pkw_vc[base + i] = pkw_v[t * pkw_p.words + i];
}

struct Q3admParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
};

@group(0) @binding(80) var<storage, read> pad_q: array<u32>;
@group(0) @binding(81) var<storage, read> pad_kc: array<u32>;
@group(0) @binding(82) var<storage, read> pad_vc: array<u32>;
@group(0) @binding(83) var<storage, read_write> pad_scores: array<f32>;
@group(0) @binding(84) var<storage, read_write> pad_out: array<f32>;
@group(0) @binding(85) var<uniform> pad_p: Q3admParams;
@group(0) @binding(86) var<uniform> pad_ck: Q3ckParams;

var<workgroup> pad_qs: array<f32, 256>;
var<workgroup> pad_red: array<f32, 256>;
var<workgroup> pad_m: f32;
var<workgroup> pad_z: f32;

@compute @workgroup_size(256)
fn q3w_attn_decode_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tk = wid.y;
    if (tk >= pad_ck.m_live) {
        return;
    }
    let h = wid.x;
    let tid = lid.x;
    let hd = pad_p.head_dim;
    let p = pad_ck.base + tk;
    let total = p + 1u;
    let kv = h / pad_p.group;
    let srow = (tk * pad_p.n_heads + h) * pad_p.max_seq;
    let qrow = tk * pad_p.n_heads * hd;

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = qrow + h * hd + d;
        pad_qs[d] = bf16_decode(u16_at(pad_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = tid; t < total; t = t + 256u) {
        let kbase = (t * pad_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(pad_kc[idx >> 1u], idx)), pad_qs[d], dot);
        }
        let s = dot * pad_p.scale;
        pad_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    pad_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pad_red[tid] = max(pad_red[tid], pad_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pad_m = pad_red[0];
    }
    workgroupBarrier();
    let m = pad_m;

    var lsum = 0.0;
    for (var t = tid; t < total; t = t + 256u) {
        let e = exp(pad_scores[srow + t] - m);
        pad_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    pad_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pad_red[tid] = pad_red[tid] + pad_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pad_z = pad_red[0];
    }
    workgroupBarrier();
    let z = pad_z;

    if (tid < hd) {
        var acc = 0.0;
        for (var t = 0u; t < total; t = t + 1u) {
            let idx = (t * pad_p.n_kv + kv) * hd + tid;
            acc = fma(pad_scores[srow + t], bf16_decode(u16_at(pad_vc[idx >> 1u], idx)), acc);
        }
        pad_out[qrow + h * hd + tid] = acc / z;
    }
}

struct Q3agmParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    x_row_elems: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(90) var<storage, read> pag_attn: array<f32>;
@group(0) @binding(91) var<storage, read> pag_qraw: array<u32>;
@group(0) @binding(92) var<storage, read_write> pag_out: array<u32>;
@group(0) @binding(93) var<uniform> pag_p: Q3agmParams;

@compute @workgroup_size(64)
fn q3w_attn_gate_m(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let t = gid.y;
    if (w >= pag_p.n_words) {
        return;
    }
    let e0 = w * 2u;
    let fbase = t * pag_p.n_words * 2u;
    let a0 = bf16_decode(bf16_encode(pag_attn[fbase + e0]));
    let a1 = bf16_decode(bf16_encode(pag_attn[fbase + e0 + 1u]));
    if (pag_p.has_gate == 0u) {
        pag_out[t * pag_p.n_words + w] = bf16_pack(a0, a1);
        return;
    }
    let h = e0 / pag_p.head_dim;
    let d = e0 % pag_p.head_dim;
    let gb = t * pag_p.x_row_elems + h * pag_p.src_stride + pag_p.gate_off + d;
    let g0 = bf16_decode(u16_at(pag_qraw[gb >> 1u], gb));
    let g1 = bf16_decode(u16_at(pag_qraw[(gb + 1u) >> 1u], gb + 1u));
    let s0 = bf16_decode(bf16_encode(1.0 / (1.0 + exp(-g0))));
    let s1 = bf16_decode(bf16_encode(1.0 / (1.0 + exp(-g1))));
    pag_out[t * pag_p.n_words + w] = bf16_pack(a0 * s0, a1 * s1);
}
