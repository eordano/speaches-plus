struct LoraPkParams {
    m: u32,
    rank: u32,
    k: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    scale: f32,
    k_tile: u32,
    n_tiles: u32,
    n_slices: u32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
    off_dest_buf: u32,
    off_dest_col: u32,
    off_dest_words: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> lpt_x: array<u32>;
@group(0) @binding(1) var<storage, read> lpt_a: array<u32>;
@group(0) @binding(2) var<storage, read> lpt_b: array<u32>;
@group(0) @binding(3) var<storage, read_write> lpt_y0: array<u32>;
@group(0) @binding(4) var<storage, read_write> lpt_y1: array<u32>;
@group(0) @binding(5) var<storage, read_write> lpt_y2: array<u32>;
@group(0) @binding(6) var<storage, read> lpt_meta: array<i32>;
@group(0) @binding(7) var<uniform> lpt_p: LoraPkParams;
@group(0) @binding(8) var<storage, read_write> lpt_partials: array<f32>;

const LORA_PK_N_CHUNK: u32 = 1024u;
const LORA_PK_SHRINK_WARPS: u32 = 8u;
const LORA_PK_LANES: u32 = 32u;
const LORA_PK_MAX_RANK: u32 = 64u;

var<workgroup> lpt_lanes: array<f32, 256>;
var<workgroup> lpt_h: array<f32, 64>;

fn lpt_x_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lpt_x[e >> 1u], e));
}

fn lpt_a_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lpt_a[e >> 1u], e));
}

fn lpt_b_at(e: u32) -> f32 {
    return bf16_decode(u16_at(lpt_b[e >> 1u], e));
}

fn lpt_y_load(buf: u32, i: u32) -> u32 {
    if (buf == 0u) {
        return lpt_y0[i];
    }
    if (buf == 1u) {
        return lpt_y1[i];
    }
    return lpt_y2[i];
}

fn lpt_y_store(buf: u32, i: u32, w: u32) {
    if (buf == 0u) {
        lpt_y0[i] = w;
    } else if (buf == 1u) {
        lpt_y1[i] = w;
    } else {
        lpt_y2[i] = w;
    }
}

fn lpt_partial_index(z: u32, slice_id: u32, tile: u32, r: u32) -> u32 {
    return ((z * lpt_p.n_slices + slice_id) * lpt_p.n_tiles + tile) * lpt_p.rank + r;
}

@compute @workgroup_size(32, 8)
fn lora_pk_shrink_tiles(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let wid = lid.y;
    let tile = wg.x;
    let slice_id = wg.y;
    let g = wg.z / lpt_p.m;
    let pid_m = wg.z - g * lpt_p.m;

    let lora_id = lpt_meta[lpt_p.off_active + g];
    let group_size = u32(lpt_meta[lpt_p.off_counts + g]);
    let alive = (lora_id != -1) && (pid_m < group_size);
    let slot = select(0u, u32(lora_id), lora_id != -1);
    var row = 0u;
    if (alive) {
        let start = u32(lpt_meta[lpt_p.off_start + g]);
        row = u32(lpt_meta[start + pid_m]);
    }
    let x_base = row * lpt_p.k;
    let a_row0 = slice_id * lpt_p.a_slice_stride + slot * lpt_p.a_d0_stride;
    let k0 = tile * lpt_p.k_tile;
    let k1 = min(k0 + lpt_p.k_tile, lpt_p.k);

    let r_iters = (lpt_p.rank + LORA_PK_SHRINK_WARPS - 1u) / LORA_PK_SHRINK_WARPS;
    for (var rr = 0u; rr < r_iters; rr = rr + 1u) {
        let r = rr * LORA_PK_SHRINK_WARPS + wid;
        var acc = 0.0;
        if (alive && r < lpt_p.rank && k0 < k1) {
            let a_base = a_row0 + r * lpt_p.k;
            if (((k0 | k1) & 1u) == 0u) {
                let span2 = (k1 - k0) >> 1u;
                for (var kk = lane; kk < span2; kk = kk + LORA_PK_LANES) {
                    let e = k0 + 2u * kk;
                    acc = fma(lpt_x_at(x_base + e), lpt_a_at(a_base + e), acc);
                    acc = fma(lpt_x_at(x_base + e + 1u), lpt_a_at(a_base + e + 1u), acc);
                }
            } else {
                for (var kk = k0 + lane; kk < k1; kk = kk + LORA_PK_LANES) {
                    acc = fma(lpt_x_at(x_base + kk), lpt_a_at(a_base + kk), acc);
                }
            }
        }
        lpt_lanes[wid * LORA_PK_LANES + lane] = acc;
        workgroupBarrier();
        if (lane == 0u && alive && r < lpt_p.rank) {
            var s = 0.0;
            for (var l = 0u; l < LORA_PK_LANES; l = l + 1u) {
                s = s + lpt_lanes[wid * LORA_PK_LANES + l];
            }
            lpt_partials[lpt_partial_index(wg.z, slice_id, tile, r)] = s;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(32, 16)
fn lora_pk_expand_tiles(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let wid = lid.y;
    let tid = wid * LORA_PK_LANES + lane;

    let g = wg.z;
    let lora_id = lpt_meta[lpt_p.off_active + g];
    let group_size = u32(lpt_meta[lpt_p.off_counts + g]);
    let pid_m = wg.x % lpt_p.m;
    let pid_n = wg.x / lpt_p.m;
    let slice_id = wg.y;
    let curr_n = u32(lpt_meta[lpt_p.off_slice_n + slice_id]);
    let n_base = pid_n * LORA_PK_N_CHUNK;
    let alive = (lora_id != -1) && (pid_m < group_size) && (n_base < curr_n);

    let slot = select(0u, u32(lora_id), lora_id != -1);
    var row = 0u;
    if (alive) {
        let start = u32(lpt_meta[lpt_p.off_start + g]);
        row = u32(lpt_meta[start + pid_m]);
    }

    if (tid < lpt_p.rank) {
        var s = 0.0;
        let z = g * lpt_p.m + pid_m;
        for (var tile = 0u; tile < lpt_p.n_tiles; tile = tile + 1u) {
            s = s + lpt_partials[lpt_partial_index(z, slice_id, tile, tid)];
        }
        lpt_h[tid] = s * lpt_p.scale;
    }
    workgroupBarrier();

    let nl = n_base + 2u * tid;
    if (alive && nl < curr_n) {
        let b_base = u32(lpt_meta[lpt_p.off_b_off + slice_id])
            + slot * u32(lpt_meta[lpt_p.off_b_d0 + slice_id])
            + nl * lpt_p.rank;
        var acc0 = 0.0;
        var acc1 = 0.0;
        if ((lpt_p.rank & 1u) == 0u) {
            let r2 = lpt_p.rank >> 1u;
            for (var r = 0u; r < r2; r = r + 1u) {
                acc0 = fma(lpt_h[2u * r], lpt_b_at(b_base + 2u * r), acc0);
                acc0 = fma(lpt_h[2u * r + 1u], lpt_b_at(b_base + 2u * r + 1u), acc0);
            }
            for (var r = 0u; r < r2; r = r + 1u) {
                acc1 = fma(lpt_h[2u * r], lpt_b_at(b_base + lpt_p.rank + 2u * r), acc1);
                acc1 = fma(lpt_h[2u * r + 1u], lpt_b_at(b_base + lpt_p.rank + 2u * r + 1u), acc1);
            }
        } else {
            for (var r = 0u; r < lpt_p.rank; r = r + 1u) {
                acc0 = fma(lpt_h[r], lpt_b_at(b_base + r), acc0);
            }
            for (var r = 0u; r < lpt_p.rank; r = r + 1u) {
                acc1 = fma(lpt_h[r], lpt_b_at(b_base + lpt_p.rank + r), acc1);
            }
        }
        let dcol = u32(lpt_meta[lpt_p.off_dest_col + slice_id]) + nl;
        let dbuf = u32(lpt_meta[lpt_p.off_dest_buf + slice_id]);
        let dwords = u32(lpt_meta[lpt_p.off_dest_words + slice_id]);
        let wi = row * dwords + (dcol >> 1u);
        let old = lpt_y_load(dbuf, wi);
        let lo = bf16_encode(bf16_decode(u16_at(old, 0u)) + acc0);
        let hi = bf16_encode(bf16_decode(u16_at(old, 1u)) + acc1);
        lpt_y_store(dbuf, wi, lo | (hi << 16u));
    }
}
