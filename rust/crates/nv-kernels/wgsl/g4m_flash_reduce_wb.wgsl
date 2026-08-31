var<workgroup> gf_red: array<f32, 256>;

fn gf_warp_sum(lid: u32, x: f32) -> f32 {
    gf_red[lid] = x;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = gf_red[lid ^ o];
        workgroupBarrier();
        gf_red[lid] = gf_red[lid] + other;
        workgroupBarrier();
    }
    return gf_red[lid];
}
