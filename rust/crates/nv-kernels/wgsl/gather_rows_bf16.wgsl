struct GatherRowsParams {
    m_total_padded: u32,
    hidden_words: u32,
    n_tokens: i32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> gr_x_words: array<u32>;
@group(0) @binding(1) var<storage, read> gr_src_idx: array<i32>;
@group(0) @binding(2) var<storage, read_write> gr_out_words: array<u32>;
@group(0) @binding(3) var<uniform> gr_params: GatherRowsParams;

const GATHER_ROWS_WG: u32 = 256u;

@compute @workgroup_size(256)
fn gather_rows_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_count: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let r = wg_id.x + wg_id.y * wg_count.x;
    if (r >= gr_params.m_total_padded) {
        return;
    }
    let s = gr_src_idx[r];
    let valid = (s >= 0) && (s < gr_params.n_tokens);
    let src_base = select(0u, u32(s), valid) * gr_params.hidden_words;
    let dst_base = r * gr_params.hidden_words;
    for (var w: u32 = lid.x; w < gr_params.hidden_words; w = w + GATHER_ROWS_WG) {
        gr_out_words[dst_base + w] = select(0u, gr_x_words[src_base + w], valid);
    }
}
