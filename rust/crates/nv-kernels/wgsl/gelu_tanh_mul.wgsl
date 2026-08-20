const GELU_SQRT2_OVER_PI: f32 = 0.7978845608028654;
const GELU_CUBIC_COEFF: f32 = 0.044715;
const GELU_TANH_CLAMP: f32 = 10.0;
const GELU_WG: u32 = 256u;

struct GeluParams {
    inter: u32,
    inter_words: u32,
    rows: u32,
    tot_pairs: u32,
};

fn gelu_tanh_mul_scalar(gate: f32, up: f32) -> f32 {
    let g3 = gate * gate * gate;
    let inner = GELU_SQRT2_OVER_PI * (gate + GELU_CUBIC_COEFF * g3);
    let clamped = clamp(inner, -GELU_TANH_CLAMP, GELU_TANH_CLAMP);
    let t = select(nv_tanhf(clamped), inner, inner != inner);
    let mag = 0.5 * abs(gate) * (1.0 + t) * abs(up);
    let sgn = (bitcast<u32>(gate) ^ bitcast<u32>(up)) & 0x80000000u;
    return bitcast<f32>((bitcast<u32>(mag) & 0x7fffffffu) | sgn);
}

fn gelu_tanh_mul_word(gate_word: u32, up_word: u32) -> u32 {
    return bf16_pack(
        gelu_tanh_mul_scalar(bf16_lo(gate_word), bf16_lo(up_word)),
        gelu_tanh_mul_scalar(bf16_hi(gate_word), bf16_hi(up_word))
    );
}

@group(0) @binding(0) var<storage, read> split_gate: array<u32>;
@group(0) @binding(1) var<storage, read> split_up: array<u32>;
@group(0) @binding(2) var<storage, read_write> split_y: array<u32>;

@compute @workgroup_size(256)
fn gelu_tanh_mul_split(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>
) {
    let n_words = arrayLength(&split_y);
    let stride = ng.x * GELU_WG;
    for (var w = gid.x; w < n_words; w = w + stride) {
        split_y[w] = gelu_tanh_mul_word(split_gate[w], split_up[w]);
    }
}

@group(0) @binding(3) var<storage, read> fused_src: array<u32>;
@group(0) @binding(4) var<storage, read_write> fused_y: array<u32>;
@group(0) @binding(5) var<uniform> fused_params: GeluParams;

@compute @workgroup_size(256)
fn gelu_tanh_mul_fused_even(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>
) {
    let iw = fused_params.inter_words;
    let stride_x = ng.x * GELU_WG;
    for (var bs = gid.y; bs < fused_params.rows; bs = bs + ng.y) {
        let gate_base = bs * fused_params.inter;
        let up_base = gate_base + iw;
        let out_base = bs * iw;
        for (var w = gid.x; w < iw; w = w + stride_x) {
            fused_y[out_base + w] = gelu_tanh_mul_word(
                fused_src[gate_base + w],
                fused_src[up_base + w]
            );
        }
    }
}

fn fused_gate_up_at(idx: u32) -> vec2<f32> {
    let i = idx % fused_params.inter;
    let bs = idx / fused_params.inter;
    let off = bs * 2u * fused_params.inter;
    let gi = off + i;
    let ui = off + fused_params.inter + i;
    let g = bf16_decode(u16_at(fused_src[gi >> 1u], gi));
    let u = bf16_decode(u16_at(fused_src[ui >> 1u], ui));
    return vec2<f32>(g, u);
}

@compute @workgroup_size(256)
fn gelu_tanh_mul_fused_general(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>
) {
    let n_words = arrayLength(&fused_y);
    let stride = ng.x * GELU_WG;
    for (var k = gid.x; k < n_words; k = k + stride) {
        let i0 = k * 2u;
        let p0 = fused_gate_up_at(i0);
        let lo = gelu_tanh_mul_scalar(p0.x, p0.y);
        var hi = 0.0;
        let i1 = i0 + 1u;
        if (i1 < fused_params.tot_pairs) {
            let p1 = fused_gate_up_at(i1);
            hi = gelu_tanh_mul_scalar(p1.x, p1.y);
        }
        fused_y[k] = bf16_pack(lo, hi);
    }
}
