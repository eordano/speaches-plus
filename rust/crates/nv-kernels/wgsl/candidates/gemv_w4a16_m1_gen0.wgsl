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

@group(0) @binding(0) var<storage, read> gen0_packed: array<u32>;
@group(0) @binding(1) var<storage, read> gen0_scale: array<u32>;
@group(0) @binding(2) var<storage, read> gen0_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> gen0_y: array<u32>;
@group(0) @binding(4) var<uniform> gen0_params: ForgeParams;

const GEN0_LANES: u32 = 32u;

var<workgroup> gen0_red: array<f32, 256>;

fn gen0_nibble(word: u32, elem: u32) -> f32 {
    return f32(u4_unpack(word, elem)) - 8.0;
}

fn gen0_dot8(pv: u32, kbase: u32, acc_in: f32) -> f32 {
    var a = acc_in;
    let xb = kbase >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let word = gen0_x[xb + i];
        a = fma(gen0_nibble(pv, 2u * i), bf16_lo(word), a);
        a = fma(gen0_nibble(pv, 2u * i + 1u), bf16_hi(word), a);
    }
    return a;
}

fn gen0_dot32(wbase: u32, kbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        a = gen0_dot8(gen0_packed[wbase + j], kbase + j * 8u, a);
    }
    return a;
}

@compute @workgroup_size(256)
fn gemv_w4a16_m1_gen0_w8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let t = tid.x;
    let lane = t & (GEN0_LANES - 1u);
    let warp = t >> 5u;
    let gid = wg.x + wg.y * gen0_params.groups_x;
    let row = gid * gen0_params.rows_per_group + warp;
    let live = row < gen0_params.n_rows;

    var acc = 0.0;
    if (live) {
        let kv = gen0_params.kv;
        let wbase = row * gen0_params.w_row_words;
        let sbase = row * kv;
        for (var j = 0u; j < gen0_params.max_v; j = j + 1u) {
            let v = lane + j * GEN0_LANES;
            if (v < kv) {
                let sc = bf16_decode(gen0_scale[sbase + v]);
                acc = fma(sc, gen0_dot32(wbase + v * 4u, v * 32u), acc);
            }
        }
    }

    gen0_red[t] = acc;
    workgroupBarrier();

    if (lane == 0u && live) {
        var sum = 0.0;
        for (var i = 0u; i < GEN0_LANES; i = i + 1u) {
            sum = sum + gen0_red[t + i];
        }
        gen0_y[row] = bf16_encode(sum);
    }
}
