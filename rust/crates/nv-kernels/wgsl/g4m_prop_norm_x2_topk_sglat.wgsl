
struct GpnParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
};

struct G4tParams {
    n_experts: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> pn_x: array<u32>;
@group(0) @binding(1) var<storage, read> pn_w: array<u32>;
@group(0) @binding(2) var<storage, read_write> pn_y: array<u32>;
@group(0) @binding(3) var<uniform> pn_p: GpnParams;
@group(0) @binding(4) var<storage, read> pn_res: array<u32>;
@group(0) @binding(5) var<storage, read> pn_w2: array<u32>;
@group(0) @binding(7) var<storage, read_write> pn_y2: array<u32>;
@group(0) @binding(10) var<storage, read> tk_logits: array<u32>;
@group(0) @binding(11) var<storage, read_write> tk_ids: array<u32>;
@group(0) @binding(12) var<storage, read_write> tk_wts: array<f32>;
@group(0) @binding(13) var<uniform> tk_p: G4tParams;
@group(0) @binding(14) var<storage, read> tk_pes: array<f32>;

var<workgroup> pn_buf: array<u32, 2048>;
var<workgroup> pn_red: array<f32, 256>;
var<workgroup> pn_s: f32;
var<workgroup> tk_v: array<f32, 256>;
var<workgroup> tk_ch: array<f32, 16>;
var<workgroup> tk_id: array<u32, 16>;

@compute @workgroup_size(256)
fn g4m_norm_x2_topk_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>
) {
    let lid = lid3.x;
    let is_topk = wid.x == 2u;
    let second = wid.x == 1u;
    let ne = tk_p.n_experts;

    if (is_topk) {
        if (lid < ne) {
            tk_v[lid] = bitcast<f32>(tk_logits[lid]);
        }
    } else {
        var local = 0.0;
        for (var w = lid; w < pn_p.words; w = w + 256u) {
            var xw = pn_x[w];
            if (second) {
                xw = pn_res[w];
            }
            pn_buf[w] = xw;
            let x0 = bf16_lo(xw);
            let x1 = bf16_hi(xw);
            local = local + x0 * x0 + x1 * x1;
        }
        pn_red[lid] = local;
    }
    workgroupBarrier();

    if (is_topk) {
        if (lid < ne) {
            let v = tk_v[lid];
            var r = 0u;
            for (var j = 0u; j < ne; j = j + 1u) {
                let w = tk_v[j];
                if (w > v || (w == v && j < lid)) {
                    r = r + 1u;
                }
            }
            if (r < tk_p.k) {
                tk_id[r] = lid;
                tk_ch[r] = v;
                tk_ids[r] = lid;
            }
        }
    } else {
        var acc = 0.0;
        if (lid < 32u) {
            acc = ((pn_red[lid] + pn_red[lid + 128u]) + (pn_red[lid + 64u] + pn_red[lid + 192u]))
                + ((pn_red[lid + 32u] + pn_red[lid + 160u]) + (pn_red[lid + 96u] + pn_red[lid + 224u]));
        }
        acc = acc + subgroupShuffleXor(acc, 16u);
        acc = acc + subgroupShuffleXor(acc, 8u);
        acc = acc + subgroupShuffleXor(acc, 4u);
        acc = acc + subgroupShuffleXor(acc, 2u);
        acc = acc + subgroupShuffleXor(acc, 1u);
        if (lid == 0u) {
            pn_s = inverseSqrt(acc / f32(pn_p.hidden) + pn_p.eps);
        }
    }
    workgroupBarrier();

    if (is_topk) {
        if (lid == 0u) {
            var m = tk_ch[0];
            for (var j = 1u; j < tk_p.k; j = j + 1u) {
                m = max(m, tk_ch[j]);
            }
            var ssum = 0.0;
            for (var j = 0u; j < tk_p.k; j = j + 1u) {
                let e = exp(tk_ch[j] - m);
                tk_ch[j] = e;
                ssum = ssum + e;
            }
            for (var j = 0u; j < tk_p.k; j = j + 1u) {
                tk_wts[j] = tk_ch[j] / ssum * tk_pes[tk_id[j]];
            }
        }
    } else {
        let s = pn_s;
        for (var w = lid; w < pn_p.words; w = w + 256u) {
            let xw = pn_buf[w];
            var ww = pn_w[w];
            if (second) {
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
}
