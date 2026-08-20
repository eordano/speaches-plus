
@compute @workgroup_size(256)
fn RR_ENTRY_POINT(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    let n = rt_p.n_experts;
    var vt = 0.0;
    if (t < n) {
        vt = bitcast<f32>(rt_logits[t]);
        rtp_v[t] = vt;
    }
    workgroupBarrier();

    if (t < n) {
        var rank = 0u;
        let nw = n - (n % RR_UNROLL_WIDTHu);
        var l = 0u;
        loop {
            if (l >= nw) { break; }
RR_UNROLLED_COMPARES            l = l + RR_UNROLL_WIDTHu;
        }
        loop {
            if (l >= n) { break; }
            let vl = rtp_v[l];
            if (vl > vt || (vl == vt && l < t)) { rank = rank + 1u; }
            l = l + 1u;
        }
        if (rank < rt_p.k) {
            rt_ids[rank] = t;
            rtp_chosen[rank] = vt;
        }
    }
    workgroupBarrier();

    if (t == 0u) {
        var m = rtp_chosen[0];
        for (var j = 1u; j < rt_p.k; j = j + 1u) {
            m = max(m, rtp_chosen[j]);
        }
        var s = 0.0;
        for (var j = 0u; j < rt_p.k; j = j + 1u) {
            let e = exp(rtp_chosen[j] - m);
            rtp_chosen[j] = e;
            s = s + e;
        }
        for (var j = 0u; j < rt_p.k; j = j + 1u) {
            rt_w[j] = rtp_chosen[j] / s;
        }
        if (rt_p.shared_slot == 1u) {
            rt_ids[rt_p.k] = rt_p.n_experts;
        }
    }
}
