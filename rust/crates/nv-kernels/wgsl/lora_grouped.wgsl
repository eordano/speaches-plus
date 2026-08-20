struct LoraGroupedParams {
    m: u32,
    rank: u32,
    k: u32,
    cta_m_num: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    y_row_stride: u32,
    scale: f32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> lg_x: array<u32>;
@group(0) @binding(1) var<storage, read> lg_a: array<u32>;
@group(0) @binding(2) var<storage, read_write> lg_buf: array<f32>;
@group(0) @binding(3) var<storage, read> lg_meta: array<i32>;
@group(0) @binding(4) var<uniform> lg_p: LoraGroupedParams;
@group(0) @binding(5) var<storage, read> lg_b: array<u32>;
@group(0) @binding(6) var<storage, read_write> lg_y: array<u32>;

const LORA_BLOCK_M: u32 = 16u;
const LORA_BLOCK_N: u32 = 16u;

fn lg_x_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lg_x[e >> 1u], e));
}

fn lg_a_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lg_a[e >> 1u], e));
}

fn lg_b_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lg_b[e >> 1u], e));
}

@compute @workgroup_size(16, 16)
fn lora_shrink(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lora_id = lg_meta[lg_p.off_active + wg.z];
    if (lora_id == -1) {
        return;
    }
    let group_size = u32(lg_meta[lg_p.off_counts + wg.z]);
    let pid_m = wg.x % lg_p.cta_m_num;
    let pid_n = wg.x / lg_p.cta_m_num;
    let m_offset = pid_m * LORA_BLOCK_M;
    if (m_offset >= group_size) {
        return;
    }
    let mi = m_offset + lid.y;
    let n = pid_n * LORA_BLOCK_N + lid.x;
    if (mi >= group_size || n >= lg_p.rank) {
        return;
    }
    let slice_id = wg.y;
    let start = u32(lg_meta[lg_p.off_start + wg.z]);
    let row = u32(lg_meta[start + mi]);
    let a_base = slice_id * lg_p.a_slice_stride + u32(lora_id) * lg_p.a_d0_stride + n * lg_p.k;
    let x_base = row * lg_p.k;
    var acc = 0.0;
    for (var kk = 0u; kk < lg_p.k; kk = kk + 1u) {
        acc = fma(lg_x_at(x_base + kk), lg_a_at(a_base + kk), acc);
    }
    lg_buf[(slice_id * lg_p.m + row) * lg_p.rank + n] = acc * lg_p.scale;
}

@compute @workgroup_size(16, 16)
fn lora_expand(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lora_id = lg_meta[lg_p.off_active + wg.z];
    if (lora_id == -1) {
        return;
    }
    let group_size = u32(lg_meta[lg_p.off_counts + wg.z]);
    let pid_m = wg.x % lg_p.cta_m_num;
    let pid_n = wg.x / lg_p.cta_m_num;
    let m_offset = pid_m * LORA_BLOCK_M;
    if (m_offset >= group_size) {
        return;
    }
    let slice_id = wg.y;
    let curr_n = u32(lg_meta[lg_p.off_slice_n + slice_id]);
    if (pid_n * LORA_BLOCK_N >= curr_n) {
        return;
    }
    let mi = m_offset + lid.y;
    let n = pid_n * LORA_BLOCK_N + lid.x;
    if (mi >= group_size || n >= curr_n) {
        return;
    }
    let start = u32(lg_meta[lg_p.off_start + wg.z]);
    let row = u32(lg_meta[start + mi]);
    let b_base = u32(lg_meta[lg_p.off_b_off + slice_id])
        + u32(lora_id) * curr_n * lg_p.rank
        + n * lg_p.rank;
    let buf_base = (slice_id * lg_p.m + row) * lg_p.rank;
    var acc = 0.0;
    for (var r = 0u; r < lg_p.rank; r = r + 1u) {
        acc = fma(lg_buf[buf_base + r], lg_b_at(b_base + r), acc);
    }
    let yi = row * lg_p.y_row_stride + u32(lg_meta[lg_p.off_slice_start + slice_id]) + n;
    lg_y[yi] = bf16_encode(bf16_decode(lg_y[yi]) + acc);
}
