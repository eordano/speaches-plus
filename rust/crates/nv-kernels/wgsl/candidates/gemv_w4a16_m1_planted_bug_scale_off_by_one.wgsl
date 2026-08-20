struct ForgeParams {
    n_rows: u32,
    kv: u32,
    w_row_words: u32,
    split: u32,
    rows_per_group: u32,
    max_v: u32,
    groups_x: u32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read> planted_packed: array<u32>;
@group(0) @binding(1) var<storage, read> planted_scale: array<u32>;
@group(0) @binding(2) var<storage, read> planted_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> planted_y: array<u32>;
@group(0) @binding(4) var<uniform> planted_params: ForgeParams;

const PLANTED_LANES: u32 = 32u;

var<workgroup> planted_red: array<f32, 256>;

fn planted_nibble(word: u32, elem: u32) -> f32 {
    return f32(u4_unpack(word, elem)) - 8.0;
}

fn planted_dot8(pv: u32, kbase: u32, acc_in: f32) -> f32 {
    var a = acc_in;
    let xb = kbase >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let word = planted_x[xb + i];
        a = fma(planted_nibble(pv, 2u * i), bf16_lo(word), a);
        a = fma(planted_nibble(pv, 2u * i + 1u), bf16_hi(word), a);
    }
    return a;
}

fn planted_dot32(wbase: u32, kbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        a = planted_dot8(planted_packed[wbase + j], kbase + j * 8u, a);
    }
    return a;
}

@compute @workgroup_size(256)
fn gemv_w4a16_m1_planted_bug_scale_off_by_one_w8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let t = tid.x;
    let lane = t & (PLANTED_LANES - 1u);
    let warp = t >> 5u;
    let gid = wg.x + wg.y * planted_params.groups_x;
    let row = gid * planted_params.rows_per_group + warp;
    let live = row < planted_params.n_rows;

    var acc = 0.0;
    if (live) {
        let kv = planted_params.kv;
        let wbase = row * planted_params.w_row_words;
        let sbase = row * kv;
        for (var j = 0u; j < planted_params.max_v; j = j + 1u) {
            let v = lane + j * PLANTED_LANES;
            if (v < kv) {
                let sc = bf16_decode(planted_scale[sbase + (v + 1u) % kv]);
                acc = fma(sc, planted_dot32(wbase + v * 4u, v * 32u), acc);
            }
        }
    }

    planted_red[t] = acc;
    workgroupBarrier();
    for (var off = PLANTED_LANES >> 1u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            planted_red[t] = planted_red[t] + planted_red[t + off];
        }
        workgroupBarrier();
    }

    if (lane == 0u && live) {
        planted_y[row] = bf16_encode(planted_red[t]);
    }
}
