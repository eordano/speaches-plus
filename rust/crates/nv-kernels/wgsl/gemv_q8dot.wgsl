struct Q8dQuantParams {
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Q8dGemvParams {
    n_rows: u32,
    k_blocks: u32,
    group_blocks: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> q8dq_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> q8dq_q: array<u32>;
@group(0) @binding(2) var<storage, read_write> q8dq_ds: array<u32>;
@group(0) @binding(3) var<uniform> q8dq_p: Q8dQuantParams;

fn q8d_bf16_lo(w: u32) -> f32 {
    return bitcast<f32>((w & 0xffffu) << 16u);
}

fn q8d_bf16_hi(w: u32) -> f32 {
    return bitcast<f32>(w & 0xffff0000u);
}

@compute @workgroup_size(64)
fn q8d_quantize_x(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= q8dq_p.k_blocks) {
        return;
    }
    var vals: array<f32, 32>;
    var amax = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let w = q8dq_x[b * 16u + i];
        let lo = q8d_bf16_lo(w);
        let hi = q8d_bf16_hi(w);
        vals[2u * i] = lo;
        vals[2u * i + 1u] = hi;
        amax = max(amax, max(abs(lo), abs(hi)));
    }
    let d = amax / 127.0;
    let dinv = select(0.0, 1.0 / d, d != 0.0);
    var isum = 0i;
    for (var i = 0u; i < 8u; i = i + 1u) {
        var word = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let q = i32(round(vals[4u * i + j] * dinv));
            isum = isum + q;
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        q8dq_q[b * 8u + i] = word;
    }
    q8dq_ds[b] = pack2x16float(vec2<f32>(d, d * f32(isum)));
}

@group(0) @binding(4) var<storage, read> q8dg_wq: array<u32>;
@group(0) @binding(5) var<storage, read> q8dg_ws: array<f32>;
@group(0) @binding(6) var<storage, read> q8dg_xq: array<u32>;
@group(0) @binding(7) var<storage, read> q8dg_xds: array<u32>;
@group(0) @binding(8) var<uniform> q8dg_p: Q8dGemvParams;
@group(0) @binding(9) var<storage, read_write> q8dg_y: array<f32>;

var<workgroup> q8dg_sh: array<f32, 64>;

@group(0) @binding(10) var<storage, read_write> q8dq_df: array<f32>;

@compute @workgroup_size(64)
fn q8d_quantize_x_df(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= q8dq_p.k_blocks) {
        return;
    }
    var vals: array<f32, 32>;
    var amax = 0.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
        let w = q8dq_x[b * 16u + i];
        let lo = q8d_bf16_lo(w);
        let hi = q8d_bf16_hi(w);
        vals[2u * i] = lo;
        vals[2u * i + 1u] = hi;
        amax = max(amax, max(abs(lo), abs(hi)));
    }
    let d = amax / 127.0;
    let dinv = select(0.0, 1.0 / d, d != 0.0);
    for (var i = 0u; i < 8u; i = i + 1u) {
        var word = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let q = i32(round(vals[4u * i + j] * dinv));
            word = word | ((u32(q) & 0xffu) << (8u * j));
        }
        q8dq_q[b * 8u + i] = word;
    }
    q8dq_df[b] = d;
}

@group(0) @binding(11) var<storage, read> q8dg_xdf: array<f32>;

@compute @workgroup_size(256)
fn q8d_gemv_sg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let row = (wg.x + wg.y * q8dg_p.groups_x) * 8u + sgid;
    let live = row < q8dg_p.n_rows;
    let k_words = select(0u, q8dg_p.k_blocks * 8u, live);
    let wrow = row * q8dg_p.k_blocks * 8u;
    let sbase = row * (q8dg_p.k_blocks / q8dg_p.group_blocks);
    var acc = 0.0;
    for (var i = lane; i < k_words; i = i + 32u) {
        let gb = i >> 3u;
        let idot = dot4I8Packed(q8dg_wq[wrow + i], q8dg_xq[i]);
        acc = fma(f32(idot) * q8dg_xdf[gb], q8dg_ws[sbase + gb / q8dg_p.group_blocks], acc);
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u && live) {
        q8dg_y[row] = acc;
    }
}

var<workgroup> q8ds_x: array<u32, 256>;
var<workgroup> q8ds_ds: array<u32, 32>;

@compute @workgroup_size(256)
fn q8d_gemv_smem(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let tid = lid.x;
    let row = (wg.x + wg.y * q8dg_p.groups_x) * 8u + sgid;
    let live = row < q8dg_p.n_rows;
    let kb = q8dg_p.k_blocks;
    let k_words = kb * 8u;
    let wrow = select(0u, row * k_words, live);
    let sbase = select(0u, row * (kb / q8dg_p.group_blocks), live);
    var acc = 0.0;
    let tiles = (kb + 31u) / 32u;
    for (var t = 0u; t < tiles; t = t + 1u) {
        let tile_word0 = t * 256u;
        let widx = tile_word0 + tid;
        if (widx < k_words) {
            q8ds_x[tid] = q8dg_xq[widx];
        }
        let b = t * 32u + tid;
        if (tid < 32u && b < kb) {
            q8ds_ds[tid] = q8dg_xds[b];
        }
        workgroupBarrier();
        for (var i = lane; i < 256u; i = i + 32u) {
            let gw = tile_word0 + i;
            if (gw < k_words) {
                let blk = i >> 3u;
                let gb = t * 32u + blk;
                let d = unpack2x16float(q8ds_ds[blk]).x;
                let s = q8dg_ws[sbase + gb / q8dg_p.group_blocks];
                let idot = dot4I8Packed(q8dg_wq[wrow + gw], q8ds_x[i]);
                acc = acc + f32(idot) * d * s;
            }
        }
        workgroupBarrier();
    }
    let total = subgroupAdd(acc);
    if (lane == 0u && live) {
        q8dg_y[row] = total;
    }
}

@compute @workgroup_size(64)
fn q8d_gemv(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wg.x + wg.y * q8dg_p.groups_x;
    let tid = lid.x;
    let row_live = row < q8dg_p.n_rows;
    let blocks = select(0u, q8dg_p.k_blocks, row_live);
    let spr = q8dg_p.k_blocks / q8dg_p.group_blocks;
    let wbase = row * q8dg_p.k_blocks * 8u;
    let sbase = row * spr;
    var acc = 0.0;
    var b = tid;
    while (b < blocks) {
        var idot = 0i;
        let wb = wbase + b * 8u;
        let xb = b * 8u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            idot = idot + dot4I8Packed(q8dg_wq[wb + i], q8dg_xq[xb + i]);
        }
        let ds = unpack2x16float(q8dg_xds[b]);
        let gi = b / q8dg_p.group_blocks;
        acc = acc + f32(idot) * ds.x * q8dg_ws[sbase + gi];
        b = b + 64u;
    }
    q8dg_sh[tid] = acc;
    workgroupBarrier();
    var s = 32u;
    while (s > 0u) {
        if (tid < s) {
            q8dg_sh[tid] = q8dg_sh[tid] + q8dg_sh[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (tid == 0u && row_live) {
        q8dg_y[row] = q8dg_sh[0];
    }
}
