
struct Q3bParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> q3b_w: array<u32>;
@group(0) @binding(1) var<storage, read> q3b_x: array<u32>;
@group(0) @binding(2) var<uniform> q3b_p: Q3bParams;
@group(0) @binding(3) var<storage, read_write> q3b_y: array<u32>;
@group(0) @binding(4) var<storage, read> q3b_row_scale_folded_2pow120: array<f32>;

var<workgroup> q3b_red: array<f32, 256>;

@compute @workgroup_size(256)
fn q3w_gemv_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * q3b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q3b_p.n_rows;
    let wbase = select(0u, row * q3b_p.w_row_words, live);
    let kw = select(0u, q3b_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = q3b_w[wbase + i];
        let xw = q3b_x[q3b_p.x_off_words + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (q3b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            q3b_y[q3b_p.y_off_words + row] = bitcast<u32>(q3b_red[tid] * q3b_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = q3b_red[0] * q3b_p.alpha;
        var hi = 0.0;
        if (row + 1u < q3b_p.n_rows) {
            hi = q3b_red[128] * q3b_p.alpha;
        }
        q3b_y[q3b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn q3w_gemv_fp8_rowscale(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * q3b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q3b_p.n_rows;
    let wbase = select(0u, row * q3b_p.w_row_words, live);
    let kw = select(0u, q3b_p.k_words, live);
    let xb = q3b_p.x_off_words;

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = q3b_w[wbase + i];
        let x0 = q3b_x[xb + 2u * i];
        let x1 = q3b_x[xb + 2u * i + 1u];
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 0u)), bf16_lo(x0), acc);
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 1u)), bf16_hi(x0), acc);
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 2u)), bf16_lo(x1), acc);
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 3u)), bf16_hi(x1), acc);
    }
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (q3b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            let s = q3b_row_scale_folded_2pow120[row];
            q3b_y[q3b_p.y_off_words + row] = bitcast<u32>(q3b_red[tid] * s * q3b_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = q3b_red[0] * q3b_row_scale_folded_2pow120[pair * 2u] * q3b_p.alpha;
        var hi = 0.0;
        if (row + 1u < q3b_p.n_rows) {
            hi = q3b_red[128] * q3b_row_scale_folded_2pow120[pair * 2u + 1u] * q3b_p.alpha;
        }
        q3b_y[q3b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

struct Q3bLoraFoldParams {
    rank: u32,
    n_tiles: u32,
    part_base: u32,
    b_base: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(11) var<storage, read> q3b_lora_b: array<u32>;
@group(0) @binding(12) var<storage, read> q3b_lora_partials: array<f32>;
@group(0) @binding(13) var<uniform> q3b_lp: Q3bLoraFoldParams;

const Q3B_LORA_FOLD_MAX_RANK_64_MATCHES_FUSED_MAX_RANK: u32 = 64u;

var<workgroup> q3b_lora_h: array<f32, Q3B_LORA_FOLD_MAX_RANK_64_MATCHES_FUSED_MAX_RANK>;

fn q3b_lora_stage_h(tid: u32, part_base: u32, n_tiles: u32, rank: u32, scale: f32) {
    if (tid < rank) {
        var s = 0.0;
        for (var t = 0u; t < n_tiles; t = t + 1u) {
            s = s + q3b_lora_partials[part_base + t * rank + tid];
        }
        q3b_lora_h[tid] = s * scale;
    }
}

fn q3b_lora_row_delta(b_base: u32, row: u32, rank: u32) -> f32 {
    var d = 0.0;
    let e0 = b_base + row * rank;
    for (var r = 0u; r < rank; r = r + 1u) {
        let e = e0 + r;
        let w = q3b_lora_b[e >> 1u];
        d = fma(q3b_lora_h[r], select(bf16_lo(w), bf16_hi(w), (e & 1u) != 0u), d);
    }
    return d;
}

@compute @workgroup_size(256)
fn q3w_gemv_fp8_rowscale_lora_fold(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    q3b_lora_stage_h(tid, q3b_lp.part_base, q3b_lp.n_tiles, q3b_lp.rank, q3b_lp.scale);
    workgroupBarrier();
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * q3b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < q3b_p.n_rows;
    let wbase = select(0u, row * q3b_p.w_row_words, live);
    let kw = select(0u, q3b_p.k_words, live);
    let xb = q3b_p.x_off_words;

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = q3b_w[wbase + i];
        let x0 = q3b_x[xb + 2u * i];
        let x1 = q3b_x[xb + 2u * i + 1u];
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 0u)), bf16_lo(x0), acc);
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 1u)), bf16_hi(x0), acc);
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 2u)), bf16_lo(x1), acc);
        acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 3u)), bf16_hi(x1), acc);
    }
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (q3b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            let s = q3b_row_scale_folded_2pow120[row];
            let y = q3b_red[tid] * s * q3b_p.alpha;
            let d = q3b_lora_row_delta(q3b_lp.b_base, row, q3b_lp.rank);
            q3b_y[q3b_p.y_off_words + row] = bitcast<u32>(select(y, y + d, d != 0.0));
        }
    } else if (tid == 0u) {
        let lo = q3b_red[0] * q3b_row_scale_folded_2pow120[pair * 2u] * q3b_p.alpha;
        let dlo = q3b_lora_row_delta(q3b_lp.b_base, pair * 2u, q3b_lp.rank);
        let lo2 = select(lo, lo + dlo, dlo != 0.0);
        var hi = 0.0;
        if (row + 1u < q3b_p.n_rows) {
            let h0 = q3b_red[128] * q3b_row_scale_folded_2pow120[pair * 2u + 1u] * q3b_p.alpha;
            let dhi = q3b_lora_row_delta(q3b_lp.b_base, pair * 2u + 1u, q3b_lp.rank);
            hi = select(h0, h0 + dhi, dhi != 0.0);
        }
        q3b_y[q3b_p.y_off_words + (row >> 1u)] = bf16_pack(lo2, hi);
    }
}

struct Q3mgParams {
    qkv_pairs: u32,
    z_pairs: u32,
    ab_pairs: u32,
    qkv_rows: u32,
    z_rows: u32,
    ab_rows: u32,
    fp8_row_words: u32,
    bf16_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(5) var<storage, read> mg_w_z: array<u32>;
@group(0) @binding(6) var<storage, read> mg_s_z_folded_2pow120: array<f32>;
@group(0) @binding(7) var<storage, read_write> mg_y_z: array<u32>;
@group(0) @binding(8) var<storage, read> mg_w_ab: array<u32>;
@group(0) @binding(9) var<storage, read_write> mg_y_ab: array<u32>;
@group(0) @binding(10) var<uniform> mg_p: Q3mgParams;

@compute @workgroup_size(256)
fn q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let gpair = wid.x + wid.y * mg_p.groups_x;

    if (gpair < mg_p.qkv_pairs) {
        let pair = gpair;
        let row = pair * 2u + half;
        let live = row < mg_p.qkv_rows;
        let wbase = select(0u, row * mg_p.fp8_row_words, live);
        let kw = select(0u, mg_p.fp8_row_words, live);
        var acc = 0.0;
        for (var i = lane; i < kw; i = i + 128u) {
            let ww = q3b_w[wbase + i];
            let x0 = q3b_x[2u * i];
            let x1 = q3b_x[2u * i + 1u];
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 0u)), bf16_lo(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 1u)), bf16_hi(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 2u)), bf16_lo(x1), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 3u)), bf16_hi(x1), acc);
        }
        q3b_red[tid] = acc;
        workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            let lo = q3b_red[0] * q3b_row_scale_folded_2pow120[pair * 2u];
            var hi = 0.0;
            if (row + 1u < mg_p.qkv_rows) {
                hi = q3b_red[128] * q3b_row_scale_folded_2pow120[pair * 2u + 1u];
            }
            q3b_y[row >> 1u] = bf16_pack(lo, hi);
        }
        return;
    }
    if (gpair < mg_p.qkv_pairs + mg_p.z_pairs) {
        let pair = gpair - mg_p.qkv_pairs;
        let row = pair * 2u + half;
        let live = row < mg_p.z_rows;
        let wbase = select(0u, row * mg_p.fp8_row_words, live);
        let kw = select(0u, mg_p.fp8_row_words, live);
        var acc = 0.0;
        for (var i = lane; i < kw; i = i + 128u) {
            let ww = mg_w_z[wbase + i];
            let x0 = q3b_x[2u * i];
            let x1 = q3b_x[2u * i + 1u];
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 0u)), bf16_lo(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 1u)), bf16_hi(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 2u)), bf16_lo(x1), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 3u)), bf16_hi(x1), acc);
        }
        q3b_red[tid] = acc;
        workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            let lo = q3b_red[0] * mg_s_z_folded_2pow120[pair * 2u];
            var hi = 0.0;
            if (row + 1u < mg_p.z_rows) {
                hi = q3b_red[128] * mg_s_z_folded_2pow120[pair * 2u + 1u];
            }
            mg_y_z[row >> 1u] = bf16_pack(lo, hi);
        }
        return;
    }
    let pair = gpair - mg_p.qkv_pairs - mg_p.z_pairs;
    let row = pair * 2u + half;
    let live = row < mg_p.ab_rows;
    let wbase = select(0u, row * mg_p.bf16_row_words, live);
    let kw = select(0u, mg_p.bf16_row_words, live);
    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = mg_w_ab[wbase + i];
        let xw = q3b_x[i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let lo = q3b_red[0];
        var hi = 0.0;
        if (row + 1u < mg_p.ab_rows) {
            hi = q3b_red[128];
        }
        mg_y_ab[row >> 1u] = bf16_pack(lo, hi);
    }
}

struct Q3mgLoraFoldParams {
    rank: u32,
    n_tiles: u32,
    ab_a_rows: u32,
    pad0: u32,
    b_base: vec4<u32>,
    scale: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(14) var<uniform> mg_lp: Q3mgLoraFoldParams;

@compute @workgroup_size(256)
fn q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab_lora_fold(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let gpair = wid.x + wid.y * mg_p.groups_x;

    if (gpair < mg_p.qkv_pairs) {
        q3b_lora_stage_h(tid, 0u, mg_lp.n_tiles, mg_lp.rank, mg_lp.scale);
        workgroupBarrier();
        let pair = gpair;
        let row = pair * 2u + half;
        let live = row < mg_p.qkv_rows;
        let wbase = select(0u, row * mg_p.fp8_row_words, live);
        let kw = select(0u, mg_p.fp8_row_words, live);
        var acc = 0.0;
        for (var i = lane; i < kw; i = i + 128u) {
            let ww = q3b_w[wbase + i];
            let x0 = q3b_x[2u * i];
            let x1 = q3b_x[2u * i + 1u];
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 0u)), bf16_lo(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 1u)), bf16_hi(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 2u)), bf16_lo(x1), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 3u)), bf16_hi(x1), acc);
        }
        q3b_red[tid] = acc;
        workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            let lo = q3b_red[0] * q3b_row_scale_folded_2pow120[pair * 2u];
            let dlo = q3b_lora_row_delta(mg_lp.b_base.x, pair * 2u, mg_lp.rank);
            let lo2 = select(lo, lo + dlo, dlo != 0.0);
            var hi = 0.0;
            if (row + 1u < mg_p.qkv_rows) {
                let h0 = q3b_red[128] * q3b_row_scale_folded_2pow120[pair * 2u + 1u];
                let dhi = q3b_lora_row_delta(mg_lp.b_base.x, pair * 2u + 1u, mg_lp.rank);
                hi = select(h0, h0 + dhi, dhi != 0.0);
            }
            q3b_y[row >> 1u] = bf16_pack(lo2, hi);
        }
        return;
    }
    if (gpair < mg_p.qkv_pairs + mg_p.z_pairs) {
        q3b_lora_stage_h(tid, mg_lp.n_tiles * mg_lp.rank, mg_lp.n_tiles, mg_lp.rank, mg_lp.scale);
        workgroupBarrier();
        let pair = gpair - mg_p.qkv_pairs;
        let row = pair * 2u + half;
        let live = row < mg_p.z_rows;
        let wbase = select(0u, row * mg_p.fp8_row_words, live);
        let kw = select(0u, mg_p.fp8_row_words, live);
        var acc = 0.0;
        for (var i = lane; i < kw; i = i + 128u) {
            let ww = mg_w_z[wbase + i];
            let x0 = q3b_x[2u * i];
            let x1 = q3b_x[2u * i + 1u];
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 0u)), bf16_lo(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 1u)), bf16_hi(x0), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 2u)), bf16_lo(x1), acc);
            acc = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(ww, 3u)), bf16_hi(x1), acc);
        }
        q3b_red[tid] = acc;
        workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            let lo = q3b_red[0] * mg_s_z_folded_2pow120[pair * 2u];
            let dlo = q3b_lora_row_delta(mg_lp.b_base.y, pair * 2u, mg_lp.rank);
            let lo2 = select(lo, lo + dlo, dlo != 0.0);
            var hi = 0.0;
            if (row + 1u < mg_p.z_rows) {
                let h0 = q3b_red[128] * mg_s_z_folded_2pow120[pair * 2u + 1u];
                let dhi = q3b_lora_row_delta(mg_lp.b_base.y, pair * 2u + 1u, mg_lp.rank);
                hi = select(h0, h0 + dhi, dhi != 0.0);
            }
            mg_y_z[row >> 1u] = bf16_pack(lo2, hi);
        }
        return;
    }
    let pair = gpair - mg_p.qkv_pairs - mg_p.z_pairs;
    let ab_slice = select(2u, 3u, pair * 2u >= mg_lp.ab_a_rows);
    q3b_lora_stage_h(
        tid,
        ab_slice * mg_lp.n_tiles * mg_lp.rank,
        mg_lp.n_tiles,
        mg_lp.rank,
        mg_lp.scale,
    );
    workgroupBarrier();
    let row = pair * 2u + half;
    let live = row < mg_p.ab_rows;
    let wbase = select(0u, row * mg_p.bf16_row_words, live);
    let kw = select(0u, mg_p.bf16_row_words, live);
    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = mg_w_ab[wbase + i];
        let xw = q3b_x[i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    q3b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            q3b_red[tid] = q3b_red[tid] + q3b_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let ab_row0 = pair * 2u - select(0u, mg_lp.ab_a_rows, ab_slice == 3u);
        let ab_b_base = mg_lp.b_base[ab_slice];
        let lo = q3b_red[0];
        let dlo = q3b_lora_row_delta(ab_b_base, ab_row0, mg_lp.rank);
        let lo2 = select(lo, lo + dlo, dlo != 0.0);
        var hi = 0.0;
        if (row + 1u < mg_p.ab_rows) {
            let h0 = q3b_red[128];
            let dhi = q3b_lora_row_delta(ab_b_base, ab_row0 + 1u, mg_lp.rank);
            hi = select(h0, h0 + dhi, dhi != 0.0);
        }
        mg_y_ab[row >> 1u] = bf16_pack(lo2, hi);
    }
}
