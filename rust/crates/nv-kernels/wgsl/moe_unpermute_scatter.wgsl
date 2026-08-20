struct MusParams {
    n_tokens: u32,
    k: u32,
    hidden: u32,
    row_stride: u32,
    hidden_tiles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> mus_y_sorted: array<u32>;
@group(0) @binding(1) var<storage, read> mus_weights: array<f32>;
@group(0) @binding(2) var<storage, read> mus_inv_perm: array<i32>;
@group(0) @binding(3) var<storage, read_write> mus_out: array<f32>;
@group(0) @binding(4) var<uniform> mus_params: MusParams;

const MUS_BLOCK: u32 = 256u;

fn mus_load_bf16(elem: u32) -> f32 {
    let word = mus_y_sorted[elem >> 1u];
    if ((elem & 1u) == 0u) {
        return bf16_lo(word);
    }
    return bf16_hi(word);
}

@compute @workgroup_size(256)
fn moe_unpermute_scatter(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let tiles = mus_params.hidden_tiles;
    if (tiles == 0u) {
        return;
    }
    let tile = wg.x + wg.y * nwg.x;
    let n = tile / tiles;
    if (n >= mus_params.n_tokens) {
        return;
    }
    let h = (tile - n * tiles) * MUS_BLOCK + tid.x;
    if (h >= mus_params.hidden) {
        return;
    }

    let base_slot = n * mus_params.k;
    var acc = 0.0;
    for (var s = 0u; s < mus_params.k; s = s + 1u) {
        let slot = base_slot + s;
        let sorted_row = u32(mus_inv_perm[slot]);
        let w = mus_weights[slot];
        let v = mus_load_bf16(sorted_row * mus_params.row_stride + h);
        acc = fma(w, v, acc);
    }
    mus_out[n * mus_params.hidden + h] = acc;
}
