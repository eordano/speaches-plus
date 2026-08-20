
struct PackParams {
    src_off: u32,
    dst_off: u32,
    n_words: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> pk_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> pk_dst: array<u32>;
@group(0) @binding(2) var<uniform> pk_params: PackParams;

@compute @workgroup_size(256)
fn pack_lo16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    if (w >= pk_params.n_words) {
        return;
    }
    let i = pk_params.src_off + w * 2u;
    pk_dst[pk_params.dst_off + w] = (pk_src[i] & 0xffffu) | ((pk_src[i + 1u] & 0xffffu) << 16u);
}

struct Gather2Params {
    split_row: u32,
    hidden_words: u32,
    vocab: u32,
    pad0: u32,
};

@group(0) @binding(3) var<storage, read> g2_lo: array<u32>;
@group(0) @binding(4) var<storage, read> g2_hi: array<u32>;
@group(0) @binding(5) var<storage, read> g2_idx: array<i32>;
@group(0) @binding(6) var<storage, read_write> g2_out: array<u32>;
@group(0) @binding(7) var<uniform> g2_params: Gather2Params;

@compute @workgroup_size(256)
fn gather2_bf16(@builtin(local_invocation_id) tid: vec3<u32>) {
    var s = u32(max(g2_idx[0], 0));
    if (s >= g2_params.vocab) {
        s = 0u;
    }
    let hw = g2_params.hidden_words;
    if (s < g2_params.split_row) {
        let base = s * hw;
        for (var w = tid.x; w < hw; w = w + 256u) {
            g2_out[w] = g2_lo[base + w];
        }
    } else {
        let base = (s - g2_params.split_row) * hw;
        for (var w = tid.x; w < hw; w = w + 256u) {
            g2_out[w] = g2_hi[base + w];
        }
    }
}

@compute @workgroup_size(256)
fn gather2_bf16_mk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let t = wg.x;
    var s = u32(max(g2_idx[t], 0));
    if (s >= g2_params.vocab) {
        s = 0u;
    }
    let hw = g2_params.hidden_words;
    let dst = t * hw;
    if (s < g2_params.split_row) {
        let base = s * hw;
        for (var w = tid.x; w < hw; w = w + 256u) {
            g2_out[dst + w] = g2_lo[base + w];
        }
    } else {
        let base = (s - g2_params.split_row) * hw;
        for (var w = tid.x; w < hw; w = w + 256u) {
            g2_out[dst + w] = g2_hi[base + w];
        }
    }
}
