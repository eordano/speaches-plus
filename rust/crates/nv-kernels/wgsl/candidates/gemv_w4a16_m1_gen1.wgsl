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

@group(0) @binding(0) var<storage, read> gen1_packed: array<u32>;
@group(0) @binding(1) var<storage, read> gen1_scale: array<u32>;
@group(0) @binding(2) var<storage, read> gen1_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> gen1_y: array<u32>;
@group(0) @binding(4) var<uniform> gen1_params: ForgeParams;

const GEN1_LANES: u32 = 32u;

var<workgroup> gen1_red: array<f32, 256>;

fn gen1_nibble(word: u32, elem: u32) -> f32 {
    return f32(u4_unpack(word, elem)) - 8.0;
}

fn gen1_dot8(pv: u32, kbase: u32, acc_in: f32) -> f32 {
    var a = acc_in;
    let xb = kbase >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let word = gen1_x[xb + i];
        a = fma(gen1_nibble(pv, 2u * i), bf16_lo(word), a);
        a = fma(gen1_nibble(pv, 2u * i + 1u), bf16_hi(word), a);
    }
    return a;
}

fn gen1_dot32(wbase: u32, kbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        a = gen1_dot8(gen1_packed[wbase + j], kbase + j * 8u, a);
    }
    return a;
}

@compute @workgroup_size(256)
fn gemv_w4a16_m1_gen1_w8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let t = tid.x;
    let lane = t & (GEN1_LANES - 1u);
    let warp = t >> 5u;
    let gid = wg.x + wg.y * gen1_params.groups_x;
    let row = gid * gen1_params.rows_per_group + warp;
    let live = row < gen1_params.n_rows;

    var even_acc = 0.0;
    var odd_acc = 0.0;
    if (live) {
        let kv = gen1_params.kv;
        let wbase = row * gen1_params.w_row_words;
        let sbase = row * kv;
        for (var j = 0u; j < gen1_params.max_v; j = j + 2u) {
            let va = lane + j * GEN1_LANES;
            let vb = va + GEN1_LANES;
            if (va < kv) {
                let sa = bf16_decode(gen1_scale[sbase + va]);
                even_acc = fma(sa, gen1_dot32(wbase + va * 4u, va * 32u), even_acc);
            }
            if (vb < kv) {
                let sb = bf16_decode(gen1_scale[sbase + vb]);
                odd_acc = fma(sb, gen1_dot32(wbase + vb * 4u, vb * 32u), odd_acc);
            }
        }
    }

    gen1_red[t] = even_acc + odd_acc;
    workgroupBarrier();
    for (var off = GEN1_LANES >> 1u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            gen1_red[t] = gen1_red[t] + gen1_red[t + off];
        }
        workgroupBarrier();
    }

    if (lane == 0u && live) {
        gen1_y[row] = bf16_encode(gen1_red[t]);
    }
}
