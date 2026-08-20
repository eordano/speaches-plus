
@group(0) @binding(6) var<storage, read_write> g4w_rope_f32_out: array<f32>;

@compute @workgroup_size(256)
fn g4w_rope_bf16_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let idx = (wg.x + wg.y * nwg.x) * ROPE_BF16_BLOCK + tid.x;
    if (idx >= rope_bf16_params.total_words) {
        return;
    }
    let half = rope_bf16_params.half_dim;
    let head_row = idx / half;
    let word_in_head = idx - head_row * half;
    let token = head_row / rope_bf16_params.n_heads;
    let base_word = head_row * half;
    let pos = u32(rope_bf16_pos[token]);
    let row_base = pos * half;

    let elem = word_in_head * 2u;
    let lo = rope_bf16_rotate(base_word, row_base, elem);
    let hi = rope_bf16_rotate(base_word, row_base, elem + 1u);
    let word = bf16_pack(lo, hi);
    g4w_rope_f32_out[idx * 2u] = bf16_lo(word);
    g4w_rope_f32_out[idx * 2u + 1u] = bf16_hi(word);
}
