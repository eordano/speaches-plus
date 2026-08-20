
struct LgmRouterParams {
    n_experts: u32,
    top_k: u32,
    norm_topk: u32,
    softcap: f32,
    routed_scaling: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> lgm_rlogits: array<u32>;
@group(0) @binding(1) var<storage, read> lgm_bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> lgm_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> lgm_wts: array<f32>;
@group(0) @binding(4) var<uniform> lgm_p: LgmRouterParams;

const LGM_NEG_INF: f32 = -3.4028235e38;

const LGM_WG: u32 = 256u;

var<workgroup> lgm_sco: array<f32, 256>;
var<workgroup> lgm_sel: array<f32, 256>;
var<workgroup> lgm_bv: array<f32, 256>;
var<workgroup> lgm_bx: array<u32, 256>;

@compute @workgroup_size(256)
fn lgw_moe_router_topk(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let n = lgm_p.n_experts;
    if (t < n) {
        var l = bitcast<f32>(lgm_rlogits[t]);
        if (lgm_p.softcap > 0.0) {
            l = lgm_p.softcap * tanh(l / lgm_p.softcap);
        }
        let sc = 1.0 / (1.0 + exp(-l));
        lgm_sco[t] = sc;
        lgm_sel[t] = sc + lgm_bias[t];
    } else {
        lgm_sco[t] = 0.0;
        lgm_sel[t] = LGM_NEG_INF;
    }
    workgroupBarrier();

    var sum = 0.0;
    for (var j = 0u; j < lgm_p.top_k; j = j + 1u) {
        lgm_bv[t] = lgm_sel[t];
        lgm_bx[t] = t;
        workgroupBarrier();

        var stride = LGM_WG >> 1u;
        loop {
            if (stride == 0u) {
                break;
            }
            if (t < stride) {
                let ov = lgm_bv[t + stride];
                let ox = lgm_bx[t + stride];
                if (ov > lgm_bv[t] || (ov == lgm_bv[t] && ox < lgm_bx[t])) {
                    lgm_bv[t] = ov;
                    lgm_bx[t] = ox;
                }
            }
            workgroupBarrier();
            stride = stride >> 1u;
        }

        let bi = lgm_bx[0];
        if (t == 0u) {
            lgm_ids[j] = bi;
            lgm_wts[j] = lgm_sco[bi];
        }
        sum = sum + lgm_sco[bi];
        workgroupBarrier();
        if (t == bi) {
            lgm_sel[t] = LGM_NEG_INF;
        }
        workgroupBarrier();
    }
    if (lgm_p.norm_topk == 1u && t == 0u) {
        for (var j = 0u; j < lgm_p.top_k; j = j + 1u) {
            lgm_wts[j] = lgm_wts[j] / sum;
        }
    }
}

struct LgmCombineParams {
    hidden_words: u32,
    top_k: u32,
    slot_stride_words: u32,
    routed_scaling: f32,
};

@group(0) @binding(20) var<storage, read> lgc_y: array<u32>;
@group(0) @binding(21) var<storage, read> lgc_w: array<f32>;
@group(0) @binding(22) var<storage, read> lgc_shared: array<u32>;
@group(0) @binding(23) var<storage, read_write> lgc_out: array<u32>;
@group(0) @binding(24) var<uniform> lgc_p: LgmCombineParams;

@compute @workgroup_size(64)
fn lgw_moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= lgc_p.hidden_words) {
        return;
    }
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < lgc_p.top_k; j = j + 1u) {
        let word = lgc_y[j * lgc_p.slot_stride_words + w];
        let wt = lgc_w[j];
        a0 = fma(bf16_lo(word), wt, a0);
        a1 = fma(bf16_hi(word), wt, a1);
    }
    let sw = lgc_shared[w];
    let o0 = a0 * lgc_p.routed_scaling + bf16_lo(sw);
    let o1 = a1 * lgc_p.routed_scaling + bf16_hi(sw);
    lgc_out[w] = bf16_pack(o0, o1);
}
