
struct AxParams {
    n_words: u32,
    sa: f32,
    sb: f32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> ax_a: array<u32>;
@group(0) @binding(1) var<storage, read> ax_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> ax_y: array<u32>;
@group(0) @binding(3) var<uniform> ax_params: AxParams;

@compute @workgroup_size(256)
fn e4b_axpby_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    if (w >= ax_params.n_words) {
        return;
    }
    let aw = ax_a[w];
    let bw = ax_b[w];
    let lo = bf16_lo(aw) * ax_params.sa + bf16_lo(bw) * ax_params.sb;
    let hi = bf16_hi(aw) * ax_params.sa + bf16_hi(bw) * ax_params.sb;
    ax_y[w] = bf16_pack(lo, hi);
}
