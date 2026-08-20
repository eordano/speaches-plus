struct LoraFusedParams {
    m: u32,
    rank: u32,
    k: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    y_row_stride: u32,
    win_off: u32,
    win_len: u32,
    scale: f32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
};

@group(0) @binding(0) var<storage, read> lf_x: array<u32>;
@group(0) @binding(1) var<storage, read> lf_a: array<u32>;
@group(0) @binding(2) var<storage, read> lf_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> lf_y: array<u32>;
@group(0) @binding(4) var<storage, read> lf_meta: array<i32>;
@group(0) @binding(5) var<uniform> lf_p: LoraFusedParams;

const LORA_FUSED_N_CHUNK: u32 = 512u;
const LORA_FUSED_WARPS: u32 = 16u;
const LORA_FUSED_LANES: u32 = 32u;
const LORA_FUSED_MAX_RANK: u32 = 64u;

var<workgroup> lf_h: array<f32, 64>;
var<workgroup> lf_partial: array<f32, 512>;

fn lf_x_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lf_x[e >> 1u], e));
}

fn lf_a_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lf_a[e >> 1u], e));
}

fn lf_b_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lf_b[e >> 1u], e));
}

@compute @workgroup_size(32, 16)
fn lora_fused(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let wid = lid.y;
    let tid = wid * LORA_FUSED_LANES + lane;

    let lora_id = lf_meta[lf_p.off_active + wg.z];
    let group_size = u32(lf_meta[lf_p.off_counts + wg.z]);
    let pid_m = wg.x % lf_p.m;
    let pid_n = wg.x / lf_p.m;
    let slice_id = wg.y;
    let curr_n = u32(lf_meta[lf_p.off_slice_n + slice_id]);
    let n_base = pid_n * LORA_FUSED_N_CHUNK;
    let s_start = u32(lf_meta[lf_p.off_slice_start + slice_id]);
    let chunk_end = min(n_base + LORA_FUSED_N_CHUNK, curr_n);
    let alive = (lora_id != -1)
        && (pid_m < group_size)
        && (n_base < curr_n)
        && (s_start + chunk_end > lf_p.win_off)
        && (s_start + n_base < lf_p.win_off + lf_p.win_len);

    let slot = select(0u, u32(lora_id), lora_id != -1);
    var row = 0u;
    if (alive) {
        let start = u32(lf_meta[lf_p.off_start + wg.z]);
        row = u32(lf_meta[start + pid_m]);
    }
    let x_base = row * lf_p.k;
    let a_row0 = slice_id * lf_p.a_slice_stride + slot * lf_p.a_d0_stride;

    let rank_iters = (lf_p.rank + LORA_FUSED_WARPS - 1u) / LORA_FUSED_WARPS;
    for (var rr = 0u; rr < rank_iters; rr = rr + 1u) {
        let r = rr * LORA_FUSED_WARPS + wid;
        var acc = 0.0;
        if (alive && r < lf_p.rank) {
            let a_base = a_row0 + r * lf_p.k;
            if ((lf_p.k & 1u) == 0u) {
                let k2 = lf_p.k >> 1u;
                for (var kk = lane; kk < k2; kk = kk + LORA_FUSED_LANES) {
                    acc = fma(lf_x_at(x_base + 2u * kk), lf_a_at(a_base + 2u * kk), acc);
                    acc = fma(lf_x_at(x_base + 2u * kk + 1u), lf_a_at(a_base + 2u * kk + 1u), acc);
                }
            } else {
                for (var kk = lane; kk < lf_p.k; kk = kk + LORA_FUSED_LANES) {
                    acc = fma(lf_x_at(x_base + kk), lf_a_at(a_base + kk), acc);
                }
            }
        }
        lf_partial[tid] = acc;
        workgroupBarrier();
        for (var off = LORA_FUSED_LANES >> 1u; off > 0u; off = off >> 1u) {
            if (lane < off) {
                lf_partial[tid] = lf_partial[tid] + lf_partial[tid + off];
            }
            workgroupBarrier();
        }
        if (lane == 0u && alive && r < lf_p.rank) {
            lf_h[r] = lf_partial[tid] * lf_p.scale;
        }
        workgroupBarrier();
    }

    let nl = n_base + tid;
    let col = s_start + nl;
    let live_out = alive
        && nl < curr_n
        && col >= lf_p.win_off
        && col < lf_p.win_off + lf_p.win_len;
    if (live_out) {
        let b_base = u32(lf_meta[lf_p.off_b_off + slice_id])
            + slot * u32(lf_meta[lf_p.off_b_d0 + slice_id])
            + nl * lf_p.rank;
        var acc = 0.0;
        if ((lf_p.rank & 1u) == 0u) {
            let r2 = lf_p.rank >> 1u;
            for (var r = 0u; r < r2; r = r + 1u) {
                acc = fma(lf_h[2u * r], lf_b_at(b_base + 2u * r), acc);
                acc = fma(lf_h[2u * r + 1u], lf_b_at(b_base + 2u * r + 1u), acc);
            }
        } else {
            for (var r = 0u; r < lf_p.rank; r = r + 1u) {
                acc = fma(lf_h[r], lf_b_at(b_base + r), acc);
            }
        }
        let yi = row * lf_p.y_row_stride + (col - lf_p.win_off);
        lf_y[yi] = bf16_encode(bf16_decode(lf_y[yi]) + acc);
    }
}
