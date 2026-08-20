
struct GmParams {
    n_words: u32,
    pli_word_off: u32,
    tok_words: u32,
    pli_stride: u32,
};

@group(0) @binding(0) var<storage, read> gm_gate: array<u32>;
@group(0) @binding(1) var<storage, read> gm_pli: array<u32>;
@group(0) @binding(2) var<storage, read_write> gm_y: array<u32>;
@group(0) @binding(3) var<uniform> gm_params: GmParams;

const GM_SQRT2_OVER_PI: f32 = 0.7978845608028654;
const GM_CUBIC: f32 = 0.044715;
const GM_CLAMP: f32 = 10.0;

fn gm_gelu(x: f32) -> f32 {
    let x3 = x * x * x;
    let inner = GM_SQRT2_OVER_PI * (x + GM_CUBIC * x3);
    let clamped = clamp(inner, -GM_CLAMP, GM_CLAMP);
    let t = select(tanh(clamped), inner, inner != inner);
    return 0.5 * x * (1.0 + t);
}

@compute @workgroup_size(256)
fn e4b_gate_mul_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    if (w >= gm_params.n_words) {
        return;
    }
    let gw = gm_gate[w];
    let pw = gm_pli[gm_params.pli_word_off + w];
    gm_y[w] = bf16_pack(
        gm_gelu(bf16_lo(gw)) * bf16_lo(pw),
        gm_gelu(bf16_hi(gw)) * bf16_hi(pw)
    );
}

@compute @workgroup_size(256)
fn e4b_gate_mul_bf16_mk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let w = (wg.x + wg.y * nwg.x) * 256u + tid.x;
    if (w >= gm_params.n_words) {
        return;
    }
    let tw = gm_params.tok_words;
    let t = w / tw;
    let wo = w - t * tw;
    let gw = gm_gate[w];
    let pw = gm_pli[t * gm_params.pli_stride + gm_params.pli_word_off + wo];
    gm_y[w] = bf16_pack(
        gm_gelu(bf16_lo(gw)) * bf16_lo(pw),
        gm_gelu(bf16_hi(gw)) * bf16_hi(pw)
    );
}
