
struct VcParams {
    n_rows: u32,
    row_off: u32,
    vocab: u32,
    m: u32,
    cap: f32,
    inv_cap: f32,
    softcap: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> vc_y: array<u32>;
@group(0) @binding(1) var<storage, read_write> vc_logits: array<f32>;
@group(0) @binding(2) var<uniform> vc_params: VcParams;

@compute @workgroup_size(256)
fn e4b_verify_cap_rows(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    if (i >= vc_params.m * vc_params.n_rows) {
        return;
    }
    let mi = i / vc_params.n_rows;
    let r = i - mi * vc_params.n_rows;
    let v = bf16_lo(vc_y[i]);
    var out = v;
    if (vc_params.softcap != 0u) {
        out = nv_tanhf(v * vc_params.inv_cap) * vc_params.cap;
    }
    vc_logits[mi * vc_params.vocab + vc_params.row_off + r] = out;
}
