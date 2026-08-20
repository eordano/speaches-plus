struct ProtoParams {
    n_rows: u32,
    kv: u32,
    w_row_words: u32,
    split: u32,
    rows_per_group: u32,
    max_v: u32,
    groups_x: u32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read> proto_packed: array<u32>;
@group(0) @binding(1) var<storage, read> proto_scale: array<u32>;
@group(0) @binding(2) var<storage, read> proto_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> proto_y: array<u32>;
@group(0) @binding(4) var<uniform> proto_params: ProtoParams;

const PROTO_LANES: u32 = 32u;

var<workgroup> proto_red: array<f32, 512>;
var<workgroup> proto_partials: array<f32, 16>;

fn proto_nibble(word: u32, elem: u32) -> f32 {
    return f32(u4_unpack(word, elem)) - 8.0;
}

fn proto_dot8(pv: u32, kbase: u32, acc_in: f32) -> f32 {
    var a = acc_in;
    let xb = kbase >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let word = proto_x[xb + i];
        a = fma(proto_nibble(pv, 2u * i), bf16_lo(word), a);
        a = fma(proto_nibble(pv, 2u * i + 1u), bf16_hi(word), a);
    }
    return a;
}

fn proto_dot32(wbase: u32, kbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        a = proto_dot8(proto_packed[wbase + j], kbase + j * 8u, a);
    }
    return a;
}

fn proto_body(tid: u32, gid: u32) {
    let lane = tid & (PROTO_LANES - 1u);
    let warp = tid >> 5u;
    let split = proto_params.split;
    let rw = warp / split;
    let part = warp % split;
    let n = gid * proto_params.rows_per_group + rw;
    let live = n < proto_params.n_rows;

    var acc = 0.0;
    if (live) {
        let kv = proto_params.kv;
        let wbase = n * proto_params.w_row_words;
        let sbase = n * kv;
        let stride = PROTO_LANES * split;
        let v0 = part * PROTO_LANES + lane;
        for (var j = 0u; j < proto_params.max_v; j = j + 1u) {
            let v = v0 + j * stride;
            if (v < kv) {
                let sc = bf16_decode(proto_scale[sbase + v]);
                acc = fma(sc, proto_dot32(wbase + v * 4u, v * 32u), acc);
            }
        }
    }

    proto_red[tid] = acc;
    workgroupBarrier();
    for (var off = PROTO_LANES >> 1u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            proto_red[tid] = proto_red[tid] + proto_red[tid + off];
        }
        workgroupBarrier();
    }
    let warp_total = proto_red[tid - lane];
    if (lane == 0u) {
        proto_partials[warp] = warp_total;
    }
    workgroupBarrier();

    if (lane == 0u && live) {
        if (split == 1u) {
            proto_y[n] = bf16_encode(warp_total);
        } else if (part == 0u) {
            var sum = 0.0;
            for (var s = 0u; s < split; s = s + 1u) {
                sum = sum + proto_partials[rw * split + s];
            }
            proto_y[n] = bf16_encode(sum);
        }
    }
}

@compute @workgroup_size(128)
fn gemv_w4a16_m1_proto_w4(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    proto_body(tid.x, wg.x + wg.y * proto_params.groups_x);
}

@compute @workgroup_size(256)
fn gemv_w4a16_m1_proto_w8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    proto_body(tid.x, wg.x + wg.y * proto_params.groups_x);
}

@compute @workgroup_size(512)
fn gemv_w4a16_m1_proto_w16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    proto_body(tid.x, wg.x + wg.y * proto_params.groups_x);
}
