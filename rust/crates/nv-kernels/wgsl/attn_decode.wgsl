struct AttnDecodeParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    scaling: f32,
};

@group(0) @binding(0) var<storage, read> ad_q: array<f32>;
@group(0) @binding(1) var<storage, read> ad_k: array<f32>;
@group(0) @binding(2) var<storage, read> ad_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> ad_out: array<f32>;
@group(0) @binding(4) var<uniform> ad_params: AttnDecodeParams;

const AD_BLOCK: u32 = 128u;
const AD_MAX_PER_THREAD: u32 = 4u;
const AD_LOG2_E: f32 = 1.4426950408889634;

var<workgroup> ad_qsh: array<f32, 512>;
var<workgroup> ad_red: array<f32, 128>;

fn ad_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn ad_fast_exp(x: f32) -> f32 {
    return exp2(x * AD_LOG2_E);
}

fn ad_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

@compute @workgroup_size(128)
fn attn_decode_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x + wg.y * nwg.x;
    if (h >= ad_params.n_heads) {
        return;
    }
    let hd = ad_params.head_dim;
    let nkv = ad_params.n_kv_heads;
    let group = ad_params.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;

    for (var d = lid; d < hd; d = d + AD_BLOCK) {
        ad_qsh[d] = ad_q[h * hd + d];
    }
    workgroupBarrier();

    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var m = ad_neg_inf();
    var l = 0.0;

    for (var p = ad_params.start; p < ad_params.total; p = p + 1u) {
        let kbase = (p * nkv + kvh) * hd;
        var partial = 0.0;
        for (var d = lid; d < hd; d = d + AD_BLOCK) {
            partial = fma(ad_qsh[d], ad_k[kbase + d], partial);
        }
        ad_red[lid] = partial;
        workgroupBarrier();
        for (var s = AD_BLOCK / 2u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                ad_red[lid] = ad_red[lid] + ad_red[lid + s];
            }
            workgroupBarrier();
        }
        let score = ad_red[0] * ad_params.scaling;
        workgroupBarrier();

        let m_new = max(m, score);
        let corr = ad_fast_exp(m - m_new);
        let w = ad_fast_exp(score - m_new);
        l = fma(l, corr, w);
        let vbase = (p * nkv + kvh) * hd;
        for (var i = 0u; i < AD_MAX_PER_THREAD; i = i + 1u) {
            let d = lid + i * AD_BLOCK;
            if (d < hd) {
                acc[i] = fma(acc[i], corr, w * ad_v[vbase + d]);
            }
        }
        m = m_new;
    }

    var inv_l = 0.0;
    if (l > 0.0) {
        inv_l = ad_recip(l);
    }
    for (var i = 0u; i < AD_MAX_PER_THREAD; i = i + 1u) {
        let d = lid + i * AD_BLOCK;
        if (d < hd) {
            ad_out[h * hd + d] = acc[i] * inv_l;
        }
    }
}
