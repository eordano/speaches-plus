
fn {P}_reduce(lid: u32, x: f32) -> f32 {
    return subgroupAdd(x);
}
