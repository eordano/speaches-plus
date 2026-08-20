struct QuantGemvParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    group_shift: u32,
    scales_per_row: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> qg_w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> qg_row_scale: array<f32>;
@group(0) @binding(2) var<storage, read> qg_x4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> qg_y: array<u32>;
@group(0) @binding(4) var<uniform> qg_params: QuantGemvParams;
@group(0) @binding(5) var<storage, read> qg_scale_words: array<u32>;

const QG_LANES: u32 = 32u;
const QG_TREE_ROWS: u32 = 8u;
const QG_SG_ROWS: u32 = 4u;

var<workgroup> qg_partial: array<f32, 256>;

fn qg_e2m1(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let s = (n >> 3u) << 31u;
    let e = (n >> 1u) & 3u;
    let m = n & 1u;
    let bits = select(m * 0x3f000000u, ((126u + e) << 23u) | (m << 22u), e != 0u);
    return bitcast<f32>(s | bits);
}

fn qg_e8m0(byte: u32) -> f32 {
    return bitcast<f32>((byte & 255u) << 23u);
}

fn qg_reduce(tid: u32, lane: u32, acc: f32) -> f32 {
    qg_partial[tid] = acc;
    workgroupBarrier();
    for (var stride = QG_LANES >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            qg_partial[tid] = qg_partial[tid] + qg_partial[tid + stride];
        }
        workgroupBarrier();
    }
    return qg_partial[tid - lane];
}

fn qg_butterfly(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn qg_dot4_i8(word: u32, xw0: u32, xw1: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(int8_decode(word, 0u), bf16_lo(xw0), acc);
    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);
    acc = fma(int8_decode(word, 2u), bf16_lo(xw1), acc);
    acc = fma(int8_decode(word, 3u), bf16_hi(xw1), acc);
    return acc;
}

fn qg_dot4_e4m3(word: u32, xw0: u32, xw1: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(e4m3_decode(byte_at(word, 0u)), bf16_lo(xw0), acc);
    acc = fma(e4m3_decode(byte_at(word, 1u)), bf16_hi(xw0), acc);
    acc = fma(e4m3_decode(byte_at(word, 2u)), bf16_lo(xw1), acc);
    acc = fma(e4m3_decode(byte_at(word, 3u)), bf16_hi(xw1), acc);
    return acc;
}

fn qg_dot8_fp4(word: u32, xw: vec4<u32>) -> f32 {
    var a = 0.0;
    a = fma(qg_e2m1(nvfp4_nibble(word, 0u)), bf16_lo(xw.x), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 1u)), bf16_hi(xw.x), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 2u)), bf16_lo(xw.y), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 3u)), bf16_hi(xw.y), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 4u)), bf16_lo(xw.z), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 5u)), bf16_hi(xw.z), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 6u)), bf16_lo(xw.w), a);
    a = fma(qg_e2m1(nvfp4_nibble(word, 7u)), bf16_hi(xw.w), a);
    return a;
}

fn qg_row_acc_i8(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, qg_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + QG_LANES) {
        let wv = qg_w4[wbase + v];
        let xa = qg_x4[2u * v];
        let xb = qg_x4[2u * v + 1u];
        acc = qg_dot4_i8(wv.x, xa.x, xa.y, acc);
        acc = qg_dot4_i8(wv.y, xa.z, xa.w, acc);
        acc = qg_dot4_i8(wv.z, xb.x, xb.y, acc);
        acc = qg_dot4_i8(wv.w, xb.z, xb.w, acc);
    }
    return acc;
}

fn qg_row_acc_e4m3(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, qg_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + QG_LANES) {
        let wv = qg_w4[wbase + v];
        let xa = qg_x4[2u * v];
        let xb = qg_x4[2u * v + 1u];
        acc = qg_dot4_e4m3(wv.x, xa.x, xa.y, acc);
        acc = qg_dot4_e4m3(wv.y, xa.z, xa.w, acc);
        acc = qg_dot4_e4m3(wv.z, xb.x, xb.y, acc);
        acc = qg_dot4_e4m3(wv.w, xb.z, xb.w, acc);
    }
    return acc;
}

fn qg_dot16_e4m3(wv: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    var d = 0.0;
    d = qg_dot4_e4m3(wv.x, xa.x, xa.y, d);
    d = qg_dot4_e4m3(wv.y, xa.z, xa.w, d);
    d = qg_dot4_e4m3(wv.z, xb.x, xb.y, d);
    d = qg_dot4_e4m3(wv.w, xb.z, xb.w, d);
    return d;
}

fn qg_dot16_i8(wv: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    var d = 0.0;
    d = qg_dot4_i8(wv.x, xa.x, xa.y, d);
    d = qg_dot4_i8(wv.y, xa.z, xa.w, d);
    d = qg_dot4_i8(wv.z, xb.x, xb.y, d);
    d = qg_dot4_i8(wv.w, xb.z, xb.w, d);
    return d;
}

fn qg_group_acc_e4m3(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, qg_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);
    let sbase = select(0u, row * qg_params.scales_per_row, live);
    let sh = qg_params.group_shift;
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + QG_LANES) {
        let d = qg_dot16_e4m3(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v + 1u]);
        acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);
    }
    return acc;
}

fn qg_group_acc_i8(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, qg_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);
    let sbase = select(0u, row * qg_params.scales_per_row, live);
    let sh = qg_params.group_shift;
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + QG_LANES) {
        let d = qg_dot16_i8(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v + 1u]);
        acc = fma(qg_row_scale[sbase + (v >> sh)], d, acc);
    }
    return acc;
}

fn qg_row_acc_mx(row: u32, live: bool, lane: u32) -> f32 {
    let kb = select(0u, qg_params.k_elems >> 5u, live);
    let base = select(0u, row * (qg_params.k_elems >> 5u), live);
    var acc = 0.0;
    for (var v = lane; v < kb; v = v + QG_LANES) {
        let wv = qg_w4[base + v];
        let xi = 4u * v;
        var dot = qg_dot8_fp4(wv.x, qg_x4[xi]);
        dot = dot + qg_dot8_fp4(wv.y, qg_x4[xi + 1u]);
        dot = dot + qg_dot8_fp4(wv.z, qg_x4[xi + 2u]);
        dot = dot + qg_dot8_fp4(wv.w, qg_x4[xi + 3u]);
        let si = base + v;
        acc = fma(qg_e8m0(byte_at(qg_scale_words[si >> 2u], si)), dot, acc);
    }
    return acc;
}

@compute @workgroup_size(256)
fn gemv_int8_rowscale(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_row_acc_i8(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total * qg_row_scale[row]);
    }
}

@compute @workgroup_size(256)
fn gemv_fp8_rowscale(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_row_acc_e4m3(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total * qg_row_scale[row]);
    }
}

@compute @workgroup_size(256)
fn gemv_fp8_group(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_group_acc_e4m3(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_int8_group(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_group_acc_i8(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(128)
fn gemv_fp8_group_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let total = qg_butterfly(qg_group_acc_e4m3(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(128)
fn gemv_int8_group_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let total = qg_butterfly(qg_group_acc_i8(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_mxfp4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_row_acc_mx(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(128)
fn gemv_int8_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let total = qg_butterfly(qg_row_acc_i8(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total * qg_row_scale[row]);
    }
}

@compute @workgroup_size(128)
fn gemv_fp8_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let total = qg_butterfly(qg_row_acc_e4m3(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total * qg_row_scale[row]);
    }
}

const QG_GELU_SQRT2_OVER_PI: f32 = 0.7978845608028654;
const QG_GELU_CUBIC_COEFF: f32 = 0.044715;
const QG_GELU_TANH_CLAMP: f32 = 10.0;

fn qg_gelu_mul_bits(gate_bits: u32, up_bits: u32) -> u32 {
    let gate = bf16_decode(gate_bits);
    let up = bf16_decode(up_bits);
    let g3 = gate * gate * gate;
    let inner = QG_GELU_SQRT2_OVER_PI * (gate + QG_GELU_CUBIC_COEFF * g3);
    let clamped = clamp(inner, -QG_GELU_TANH_CLAMP, QG_GELU_TANH_CLAMP);
    let t = select(nv_tanhf(clamped), inner, inner != inner);
    let mag = 0.5 * abs(gate) * (1.0 + t) * abs(up);
    let sgn = ((gate_bits ^ up_bits) & 0x8000u) << 16u;
    return bf16_encode(bitcast<f32>((bitcast<u32>(mag) & 0x7fffffffu) | sgn)) & 0xffffu;
}

var<workgroup> qg_gelu_bits: array<u32, QG_TREE_ROWS>;
var<workgroup> qg_gelu_sg_bits: array<u32, QG_SG_ROWS>;

@compute @workgroup_size(256)
fn gemv_int8_group_gelu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let half = QG_TREE_ROWS >> 1u;
    let inter = qg_params.n_rows >> 1u;
    let base = (wid.x + wid.y * qg_params.groups_x) * half;
    let sub = warp & (half - 1u);
    let out = base + sub;
    let live = out < inter;
    let row = select(out, inter + out, warp >= half);
    let total = qg_reduce(tid, lane, qg_group_acc_i8(row, live, lane));
    if (lane == 0u) {
        qg_gelu_bits[warp] = bf16_encode(total) & 0xffffu;
    }
    workgroupBarrier();
    if (lane == 0u && warp < (half >> 1u)) {
        let o = base + warp * 2u;
        if (o < inter) {
            let j = warp * 2u;
            var word = qg_gelu_mul_bits(qg_gelu_bits[j], qg_gelu_bits[half + j]);
            if (o + 1u < inter) {
                word = word | (qg_gelu_mul_bits(qg_gelu_bits[j + 1u], qg_gelu_bits[half + j + 1u]) << 16u);
            }
            qg_y[o >> 1u] = word;
        }
    }
}

@compute @workgroup_size(128)
fn gemv_int8_group_gelu_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let half = QG_SG_ROWS >> 1u;
    let inter = qg_params.n_rows >> 1u;
    let base = (wid.x + wid.y * qg_params.groups_x) * half;
    let sub = sgid & (half - 1u);
    let out = base + sub;
    let live = out < inter;
    let row = select(out, inter + out, sgid >= half);
    let bits = bf16_encode(qg_butterfly(qg_group_acc_i8(row, live, lane))) & 0xffffu;
    if (lane == 0u) {
        qg_gelu_sg_bits[sgid] = bits;
    }
    workgroupBarrier();
    if (lane == 0u && sgid == 0u && base < inter) {
        var word = qg_gelu_mul_bits(qg_gelu_sg_bits[0u], qg_gelu_sg_bits[half]);
        if (base + 1u < inter) {
            word = word | (qg_gelu_mul_bits(qg_gelu_sg_bits[1u], qg_gelu_sg_bits[half + 1u]) << 16u);
        }
        qg_y[base >> 1u] = word;
    }
}

@compute @workgroup_size(256)
fn gemv_fp8_group_gelu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let half = QG_TREE_ROWS >> 1u;
    let inter = qg_params.n_rows >> 1u;
    let base = (wid.x + wid.y * qg_params.groups_x) * half;
    let sub = warp & (half - 1u);
    let out = base + sub;
    let live = out < inter;
    let row = select(out, inter + out, warp >= half);
    let total = qg_reduce(tid, lane, qg_group_acc_e4m3(row, live, lane));
    if (lane == 0u) {
        qg_gelu_bits[warp] = bf16_encode(total) & 0xffffu;
    }
    workgroupBarrier();
    if (lane == 0u && warp < (half >> 1u)) {
        let o = base + warp * 2u;
        if (o < inter) {
            let j = warp * 2u;
            var word = qg_gelu_mul_bits(qg_gelu_bits[j], qg_gelu_bits[half + j]);
            if (o + 1u < inter) {
                word = word | (qg_gelu_mul_bits(qg_gelu_bits[j + 1u], qg_gelu_bits[half + j + 1u]) << 16u);
            }
            qg_y[o >> 1u] = word;
        }
    }
}

@compute @workgroup_size(128)
fn gemv_fp8_group_gelu_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let half = QG_SG_ROWS >> 1u;
    let inter = qg_params.n_rows >> 1u;
    let base = (wid.x + wid.y * qg_params.groups_x) * half;
    let sub = sgid & (half - 1u);
    let out = base + sub;
    let live = out < inter;
    let row = select(out, inter + out, sgid >= half);
    let bits = bf16_encode(qg_butterfly(qg_group_acc_e4m3(row, live, lane))) & 0xffffu;
    if (lane == 0u) {
        qg_gelu_sg_bits[sgid] = bits;
    }
    workgroupBarrier();
    if (lane == 0u && sgid == 0u && base < inter) {
        var word = qg_gelu_mul_bits(qg_gelu_sg_bits[0u], qg_gelu_sg_bits[half]);
        if (base + 1u < inter) {
            word = word | (qg_gelu_mul_bits(qg_gelu_sg_bits[1u], qg_gelu_sg_bits[half + 1u]) << 16u);
        }
        qg_y[base >> 1u] = word;
    }
}

@compute @workgroup_size(128)
fn gemv_mxfp4_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let total = qg_butterfly(qg_row_acc_mx(row, live, lane));
    if (lane == 0u && live) {
        qg_y[row] = bf16_encode(total);
    }
}
