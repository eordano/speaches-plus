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

@group(0) @binding(0) var<storage, read> gen2_packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> gen2_scale: array<u32>;
@group(0) @binding(2) var<storage, read> gen2_x: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> gen2_y: array<u32>;
@group(0) @binding(4) var<uniform> gen2_params: ForgeParams;

const GEN2_LANES: u32 = 32u;

var<workgroup> gen2_red: array<f32, 256>;

fn gen2_nibble(word: u32, elem: u32) -> f32 {
    return f32(u4_unpack(word, elem)) - 8.0;
}

fn gen2_dot8(pv: u32, xq: vec4<u32>, acc_in: f32) -> f32 {
    var a = acc_in;
    a = fma(gen2_nibble(pv, 0u), bf16_lo(xq.x), a);
    a = fma(gen2_nibble(pv, 1u), bf16_hi(xq.x), a);
    a = fma(gen2_nibble(pv, 2u), bf16_lo(xq.y), a);
    a = fma(gen2_nibble(pv, 3u), bf16_hi(xq.y), a);
    a = fma(gen2_nibble(pv, 4u), bf16_lo(xq.z), a);
    a = fma(gen2_nibble(pv, 5u), bf16_hi(xq.z), a);
    a = fma(gen2_nibble(pv, 6u), bf16_lo(xq.w), a);
    a = fma(gen2_nibble(pv, 7u), bf16_hi(xq.w), a);
    return a;
}

fn gen2_dot32(wquad: u32, xquad: u32) -> f32 {
    let pv = gen2_packed[wquad];
    var a = 0.0;
    a = gen2_dot8(pv.x, gen2_x[xquad], a);
    a = gen2_dot8(pv.y, gen2_x[xquad + 1u], a);
    a = gen2_dot8(pv.z, gen2_x[xquad + 2u], a);
    a = gen2_dot8(pv.w, gen2_x[xquad + 3u], a);
    return a;
}

@compute @workgroup_size(256)
fn gemv_w4a16_m1_gen2_w8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let t = tid.x;
    let lane = t & (GEN2_LANES - 1u);
    let warp = t >> 5u;
    let gid = wg.x + wg.y * gen2_params.groups_x;
    let row = gid * gen2_params.rows_per_group + warp;
    let live = row < gen2_params.n_rows;

    var acc = 0.0;
    if (live) {
        let kv = gen2_params.kv;
        let wquad_base = row * (gen2_params.w_row_words >> 2u);
        let sbase = row * kv;
        for (var j = 0u; j < gen2_params.max_v; j = j + 1u) {
            let v = lane + j * GEN2_LANES;
            if (v < kv) {
                let sc = bf16_decode(gen2_scale[sbase + v]);
                acc = fma(sc, gen2_dot32(wquad_base + v, v * 4u), acc);
            }
        }
    }

    gen2_red[t] = acc;
    workgroupBarrier();
    for (var off = GEN2_LANES >> 1u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            gen2_red[t] = gen2_red[t] + gen2_red[t + off];
        }
        workgroupBarrier();
    }

    if (lane == 0u && live) {
        gen2_y[row] = bf16_encode(gen2_red[t]);
    }
}
