
struct Q3pkvParams {
    tokens: u32,
    x_stride_elems: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(9) var<uniform> pkv_p: Q3pkvParams;

@compute @workgroup_size(256)
fn q3w_pf_quantize_kv_fp8_m(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let kv_head = wg.x;
    let token = wg.y;
    if (kv_head >= kvq_params.n_kv || token >= pkv_p.tokens) {
        return;
    }
    let n_kv = kvq_params.n_kv;
    let head_dim = kvq_params.head_dim;

    var slot = u32(max(kvq_start[0], 0)) + token;
    if (kvq_params.ring > 0u) {
        slot = slot % kvq_params.ring;
    }
    let base_src = token * pkv_p.x_stride_elems + kv_head * head_dim;
    let base_dst = (slot * n_kv + kv_head) * head_dim;
    let lid = tid.x;

    var local = 0.0;
    for (var d = lid; d < head_dim; d = d + KV_WG) {
        let a = abs(kv_bf16_at(base_src, d));
        if (a > local) {
            local = a;
        }
    }
    let amax = kv_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, kv_div_rn(amax, KV_FP8_E4M3_MAX), positive);
    let inv_scale = select(1.0, kv_div_rn(KV_FP8_E4M3_MAX, amax), positive);

    if (lid == 0u) {
        kvq_scales[slot * n_kv + kv_head] = scale;
    }

    let out_words = head_dim >> 2u;
    for (var w = lid; w < out_words; w = w + KV_WG) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = kv_bf16_at(base_src, d0 + j);
            packed = packed | (kv_encode_e4m3(v * inv_scale) << (8u * j));
        }
        kvq_out[(base_dst >> 2u) + w] = packed;
    }
}
