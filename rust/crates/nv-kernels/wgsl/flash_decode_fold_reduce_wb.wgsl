
var<workgroup> {P}_red: array<f32, 256>;

fn {P}_reduce(lid: u32, x: f32) -> f32 {
    {P}_red[lid] = x;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = {P}_red[lid ^ o];
        workgroupBarrier();
        {P}_red[lid] = {P}_red[lid] + other;
        workgroupBarrier();
    }
    return {P}_red[lid];
}
