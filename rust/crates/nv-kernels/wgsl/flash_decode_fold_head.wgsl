
@compute @workgroup_size(256)
fn {E}(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h0 = wg.x * {F}u;
    let split = wg.y;
    if (h0 >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let group = fd_params.n_heads / nkv;
    let kvh = h0 / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd * {F}u; d = d + FD_BLOCK) {
        {P}_qsh[d] = fd_q[h0 * hd + d];
    }
    workgroupBarrier();
