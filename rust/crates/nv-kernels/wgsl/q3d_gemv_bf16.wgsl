
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
