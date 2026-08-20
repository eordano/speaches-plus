
struct FncuParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> fncu_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> fncu_res: array<u32>;
@group(0) @binding(2) var<storage, read> fncu_w1: array<u32>;
@group(0) @binding(3) var<storage, read> fncu_w2: array<u32>;
@group(0) @binding(4) var<storage, read_write> fncu_out: array<u32>;
@group(0) @binding(5) var<uniform> fncu_params: FncuParams;
@group(0) @binding(6) var<storage, read_write> fncu_out2: array<u32>;

const FNCU_WARP: u32 = 32u;

var<workgroup> fncu_scratch: array<f32, 256>;
var<workgroup> fncu_shared: f32;

fn fncu_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn fncu_rms_reduce(lid: u32, local: f32) -> f32 {
    fncu_scratch[lid] = local;
    workgroupBarrier();

    for (var stride = FNCU_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (FNCU_WARP - 1u)) < stride) {
            fncu_scratch[lid] = fncu_scratch[lid] + fncu_scratch[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let a = (fncu_scratch[0u] + fncu_scratch[128u]) + (fncu_scratch[64u] + fncu_scratch[192u]);
        let b = (fncu_scratch[32u] + fncu_scratch[160u]) + (fncu_scratch[96u] + fncu_scratch[224u]);
        let sum = a + b;
        let mean = fncu_div_rn(sum, f32(fncu_params.hidden));
        fncu_shared = inverseSqrt(fncu_params.eps + mean);
    }
    workgroupBarrier();
    return fncu_shared;
}

fn fncu_res_reduce(lid: u32, local: f32) -> f32 {
    fncu_scratch[lid] = local;
    workgroupBarrier();
    for (var stride = 256u / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            fncu_scratch[lid] = fncu_scratch[lid] + fncu_scratch[lid + stride];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        fncu_shared = inverseSqrt(fncu_scratch[0] / f32(fncu_params.hidden) + fncu_params.eps);
    }
    workgroupBarrier();
    return fncu_shared;
}
