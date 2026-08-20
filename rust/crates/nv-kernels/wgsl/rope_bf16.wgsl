struct RopeBf16Params {
    n_heads: u32,
    half_dim: u32,
    total_words: u32,
    table_rows: u32,
};

@group(0) @binding(0) var<storage, read> rope_bf16_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> rope_bf16_dst: array<u32>;
@group(0) @binding(2) var<storage, read> rope_bf16_cos: array<f32>;
@group(0) @binding(3) var<storage, read> rope_bf16_sin: array<f32>;
@group(0) @binding(4) var<storage, read> rope_bf16_pos: array<i32>;
@group(0) @binding(5) var<uniform> rope_bf16_params: RopeBf16Params;

const ROPE_BF16_BLOCK: u32 = 256u;

fn rope_bf16_load(base_word: u32, elem: u32) -> f32 {
    let word = rope_bf16_src[base_word + (elem >> 1u)];
    if ((elem & 1u) == 0u) {
        return bf16_lo(word);
    }
    return bf16_hi(word);
}

fn rope_bf16_rotate(base_word: u32, row_base: u32, elem: u32) -> f32 {
    let half = rope_bf16_params.half_dim;
    if (elem < half) {
        let c = rope_bf16_cos[row_base + elem];
        let s = rope_bf16_sin[row_base + elem];
        let a = rope_bf16_load(base_word, elem);
        let b = rope_bf16_load(base_word, elem + half);
        return fma(a, c, -(b * s));
    }
    let pair = elem - half;
    let c = rope_bf16_cos[row_base + pair];
    let s = rope_bf16_sin[row_base + pair];
    let a = rope_bf16_load(base_word, pair);
    let b = rope_bf16_load(base_word, elem);
    return fma(a, s, b * c);
}

@compute @workgroup_size(256)
fn rope_bf16(
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
    rope_bf16_dst[idx] = bf16_pack(lo, hi);
}
