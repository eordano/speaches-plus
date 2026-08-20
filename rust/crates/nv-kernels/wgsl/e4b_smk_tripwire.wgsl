
@compute @workgroup_size(256)
fn smk_trip_clean(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    if (tid == 0u) {
        smk_partial[0] = 1.0;
    }
    workgroupBarrier();
    let row = wid.x * 256u + tid;
    if (row < smk_params.n_rows) {
        smk_y[row] = bitcast<u32>(smk_partial[tid]);
    }
}

@compute @workgroup_size(256)
fn smk_trip_poisoned(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    smk_partial[lid.x] = bitcast<f32>(0x7fc0deadu | smk_y[0]);
    workgroupBarrier();
    let tid = lid.x;
    if (tid == 0u) {
        smk_partial[0] = 1.0;
    }
    workgroupBarrier();
    let row = wid.x * 256u + tid;
    if (row < smk_params.n_rows) {
        smk_y[row] = bitcast<u32>(smk_partial[tid]);
    }
}
