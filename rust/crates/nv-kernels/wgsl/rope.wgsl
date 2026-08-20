struct RopeParams {
    batch: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    half_dim: u32,
    total_heads: u32,
    total_pairs: u32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read_write> rope_q: array<f32>;
@group(0) @binding(1) var<storage, read_write> rope_k: array<f32>;
@group(0) @binding(2) var<storage, read> rope_cos: array<f32>;
@group(0) @binding(3) var<storage, read> rope_sin: array<f32>;
@group(0) @binding(4) var<storage, read> rope_positions: array<i32>;
@group(0) @binding(5) var<uniform> rope_params: RopeParams;

const ROPE_BLOCK: u32 = 256u;

fn rope_flat_index(wg: vec3<u32>, nwg: vec3<u32>, tid: vec3<u32>) -> u32 {
    return (wg.x + wg.y * nwg.x) * ROPE_BLOCK + tid.x;
}

@compute @workgroup_size(256)
fn rope_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = rope_flat_index(wg, nwg, tid);
    if (idx >= rope_params.total_pairs) {
        return;
    }

    let half_dim = rope_params.half_dim;
    let pair_idx = idx % half_dim;
    let rest = idx / half_dim;
    let head_idx = rest % rope_params.total_heads;
    let token_idx = rest / rope_params.total_heads;

    let pos = u32(rope_positions[token_idx]);
    let row = pos * half_dim;
    let c = rope_cos[row + pair_idx];
    let s = rope_sin[row + pair_idx];

    if (head_idx < rope_params.n_heads) {
        let base = (token_idx * rope_params.n_heads + head_idx) * rope_params.head_dim;
        let a = rope_q[base + pair_idx];
        let b = rope_q[base + pair_idx + half_dim];
        rope_q[base + pair_idx] = fma(a, c, -(b * s));
        rope_q[base + pair_idx + half_dim] = fma(a, s, b * c);
    } else {
        let kv_head = head_idx - rope_params.n_heads;
        let base = (token_idx * rope_params.n_kv_heads + kv_head) * rope_params.head_dim;
        let a = rope_k[base + pair_idx];
        let b = rope_k[base + pair_idx + half_dim];
        rope_k[base + pair_idx] = fma(a, c, -(b * s));
        rope_k[base + pair_idx + half_dim] = fma(a, s, b * c);
    }
}
