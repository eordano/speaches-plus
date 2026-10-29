use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_kernels::wgpu_backend::dispatch::{self, PassRef};
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};
use nv_weights::{TensorSource, WeightLoader};
use wgpu::{BindGroup, Buffer, ComputePipeline};

use crate::gemma4_wgpu_shared::pack_pairs;
use crate::pod;
use crate::voxtral::{
    merge_adapter_rows, pad_audio_for_streaming, time_embedding, HfLayouts, TekkenVocab,
    VoxtralConfig, VoxtralMel, VoxtralQuant, DECODER_ROPE_ROWS,
    ENCODER_ROPE_ROWS_COVER_35_MINUTES_OF_50HZ_FRAMES, HF_CONFIG_FILE, HF_WEIGHTS_FILE, HOP_LENGTH,
    N_FFT, SAMPLES_PER_TOKEN, TEKKEN_FILE, VOXTRAL_QUANT_ENV,
};

pub const TOKEN_RING_SLOTS: usize = 64;
pub const REPLAYS_PER_BATCH_TRADE_EOS_LATENCY_AGAINST_HOST_ROUND_TRIPS: usize = 16;
pub const DECODE_ATTENTION_SPLITS: u32 = 16;
pub const GEMM_TILE: usize = 64;
pub const GEMM_K_STEP: usize = 32;
pub const ARGMAX_PARTS: u32 = 256;
pub const PAD_DRAFT_ROWS_MAX: usize = 8;
pub const PAD_DRAFT_ROWS_DEFAULT_FOR_NVFP4: usize = 4;
pub const WGPU_SPEC_ROWS_ENV: &str = "NV_VOXTRAL_SPEC_ROWS";
pub const RING_POS_SLOT: usize = TOKEN_RING_SLOTS;

pub const THE_STEP_READS_ITS_TOKEN_AND_POSITION_FROM_DEVICE_BUFFERS_SO_A_BATCH_OF_STEPS_NEEDS_NO_HOST_SYNC:
    &str = "the argmax writes the token into a device buffer the next step's embed gather reads, \
     and every position-dependent pass (audio row, rope row, KV slot, attention length, ring \
     slot) reads one device position scalar that the last pass of the step increments, so the \
     prebuilt step pass list is submitted k times back to back and the host reads the token \
     ring once per batch, mirroring the CUDA graph's replay batches";

pub const ENCODER_CONV_STEM_IS_TWO_IM2COL_GEMMS: &str =
    "conv1 (k=3, pad-left 2) is an im2col built on the host beside the log-mel, conv2 (k=3, \
     stride 2, pad-left 1) an im2col gather pass over conv1's output; both then run through the \
     same tiled bf16 GEMM as every encoder linear, with bias and GELU(erf) fused in its epilogue";

pub const ENCODER_PASSES_ARE_REBUILT_PER_CLIP_BECAUSE_ITS_ACTIVATIONS_ARE_SIZED_TO_THE_CLIP: &str =
    "a 480 s chunk is 24000 encoder frames and 2.3 GB of transient activations, so the encoder \
     allocates per clip and rebuilds its bind groups against those buffers; the decoder's \
     prefill and step lists bind capacity-sized buffers once at construction";

pub const WGPU_NVFP4_IS_THE_DEFAULT_BY_THE_LIBRISPEECH_73_CLIP_CORPUS_TEST: &str =
    "NV_VOXTRAL_QUANT=nvfp4 quantizes the 26 decoder layers' qkv, o, gate/up, down and the lm_head \
     exactly as voxtral.rs does on CUDA (per-tensor global 448*6/amax, block-16 ue4m3 scales) and, \
     like the CUDA path at every M <= 64, keeps the activations bf16: W4A16, not the W4A4 route the \
     chat decoders take through q3w_gemv_nvfp4; it is the default because the 73-clip word-error \
     instrument that blessed the CUDA flip, run here as wgpu_corpus_word_errors_reproduce_the_cuda_\
     flip_instrument in tests/voxtral_wgpu.rs, scores 77 of 1150 against bf16's 77 at 3.4 ms per \
     step (best of 8 on the fixture, prefix prefill included) against bf16's 7.3; \
     NV_VOXTRAL_QUANT=bf16 restores the dense weights";

pub const THE_NVFP4_STEP_FUSES_RMSNORM_AND_RESIDUAL_ADDS_INTO_ITS_GEMVS: &str =
    "vx_norm_rows writes each row's rmsnorm (attn_norm before qkv, the delay-folded ffn_norm \
     before gate/up, the final norm before the lm_head) in the GEMV's permuted layout, and o and \
     down accumulate straight into the residual stream, so a layer is 10 passes and a replay \
     26 x 10 + 5";

pub const THE_PREFIX_ROWS_THAT_SEE_ONLY_THE_LEFT_PAD_ARE_SEEDED_ONCE_PER_MODEL: &str =
    "prefix row r adds audio row r (BOS included), and an audio row whose mel frames all end \
     before the first STFT window that reaches past the zero left pad is the same across clips \
     through the causal encoder, so those rows' K/V is computed by the seed list on the first \
     clip and kept; each clip prefills only the remaining prefix rows, and the suite covering the \
     prefix seed asserts the audio rows bitwise across two clips";

pub const E2M1_DECODES_THROUGH_THE_F16_SUBNORMAL_LADDER: &str =
    "a nibble's three magnitude bits placed at f16 bits 9..11 read as e2m1 times 2^-14 for \
     every code, the zero-exponent codes landing on f16 subnormals of the same spacing, so \
     unpack2x16float of two such patterns and an exact power-of-two scale decode a byte in a \
     dozen ALU ops with no lookup table in workgroup memory; the values are the LUT's bitwise";

pub const THE_NVFP4_STEP_DRAFTS_PADS_LIKE_THE_CUDA_GRAPH_AND_STAYS_BITWISE_PER_ROW: &str =
    "row 0 carries the last committed token and rows 1..M the streaming pad; vx_gemv_nv4 keeps \
     one lane sequence per weight pair and sweeps the M staged rows inside it, the decode \
     attention runs each row's own prefix in the one-row order, and vx_argmax2 commits rows while \
     the previous row's argmax is the pad and advances the device position by the count, so \
     every committed token is the one-row route's; the per-clip prefix prefill is the same M-row \
     block over the rows the seed leaves, with the lm_head on its last row only";

pub const UPLOAD_STAGING_IS_FREED_ONLY_AT_A_DEVICE_POLL_SO_THE_LOADER_FLUSHES_PER_LAYER: &str =
    "create_buffer_init parks each upload's staging copy until the device is polled, and on a \
     resizable-BAR adapter that staging lives in device memory, so without a per-layer flush the \
     loader's peak is twice the checkpoint and a half-free GPU refuses a model that fits";

fn pad_draft_tokens(first: u32) -> Vec<u32> {
    let mut t = vec![crate::voxtral::STREAMING_PAD_TOKEN; PAD_DRAFT_ROWS_MAX];
    t[0] = first;
    t
}

pub fn spec_rows_from_env(quant: VoxtralQuant) -> Result<usize> {
    let rows = match std::env::var(WGPU_SPEC_ROWS_ENV).ok().as_deref() {
        None | Some("") => match quant {
            VoxtralQuant::Nvfp4 => PAD_DRAFT_ROWS_DEFAULT_FOR_NVFP4,
            _ => 1,
        },
        Some(v) => v.parse().map_err(|_| anyhow!("{WGPU_SPEC_ROWS_ENV}={v:?}: expected 1..={PAD_DRAFT_ROWS_MAX}"))?,
    };
    ensure!((1..=PAD_DRAFT_ROWS_MAX).contains(&rows), "{WGPU_SPEC_ROWS_ENV}={rows}: expected 1..={PAD_DRAFT_ROWS_MAX}");
    ensure!(rows == 1 || quant == VoxtralQuant::Nvfp4, "{WGPU_SPEC_ROWS_ENV}={rows}: only the nvfp4 step drafts pads");
    Ok(rows)
}

pub fn wgpu_quant_from_env() -> Result<VoxtralQuant> {
    let _ = WGPU_NVFP4_IS_THE_DEFAULT_BY_THE_LIBRISPEECH_73_CLIP_CORPUS_TEST;
    match std::env::var(VOXTRAL_QUANT_ENV).ok().as_deref() {
        None | Some("") | Some("nvfp4") => Ok(VoxtralQuant::Nvfp4),
        Some("bf16") | Some("f32") => Ok(VoxtralQuant::Bf16),
        Some(other) => bail!("{VOXTRAL_QUANT_ENV}={other:?}: the wgpu twin serves bf16 or nvfp4"),
    }
}

pub(crate) const LABEL: &str = "nv_models_voxtral_wgpu";

const GEMV_BF16_WGSL: &str = nv_kernels::wgsl!("../../nv-kernels/wgsl/q3d_gemv_bf16.wgsl");

pub(crate) const VX_SHARED_WGSL: &str = r#"
fn nv_e2m1(nib: u32) -> f32 {
    let n = nib & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn nv_perm(i: u32) -> u32 {
    return ((i >> 7u) << 7u) | ((i & 3u) << 5u) | ((i >> 2u) & 31u);
}

fn nv_perm_word(o: u32) -> u32 {
    return (nv_perm(o >> 2u) << 2u) | (o & 3u);
}

fn nv_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn nv_sq4(v: vec4<u32>) -> f32 {
    var d = 0.0;
    d = fma(bf16_lo(v.x), bf16_lo(v.x), d);
    d = fma(bf16_hi(v.x), bf16_hi(v.x), d);
    d = fma(bf16_lo(v.y), bf16_lo(v.y), d);
    d = fma(bf16_hi(v.y), bf16_hi(v.y), d);
    d = fma(bf16_lo(v.z), bf16_lo(v.z), d);
    d = fma(bf16_hi(v.z), bf16_hi(v.z), d);
    d = fma(bf16_lo(v.w), bf16_lo(v.w), d);
    d = fma(bf16_hi(v.w), bf16_hi(v.w), d);
    return d;
}

fn nv_norm4(x: vec4<u32>, w: vec4<u32>, rstd: f32) -> vec4<u32> {
    return vec4<u32>(
        bf16_pack(bf16_lo(x.x) * rstd * bf16_lo(w.x), bf16_hi(x.x) * rstd * bf16_hi(w.x)),
        bf16_pack(bf16_lo(x.y) * rstd * bf16_lo(w.y), bf16_hi(x.y) * rstd * bf16_hi(w.y)),
        bf16_pack(bf16_lo(x.z) * rstd * bf16_lo(w.z), bf16_hi(x.z) * rstd * bf16_hi(w.z)),
        bf16_pack(bf16_lo(x.w) * rstd * bf16_lo(w.w), bf16_hi(x.w) * rstd * bf16_hi(w.w)));
}

"#;

const VX_WGSL: &str = r#"
struct VxGemmParams { m: u32, n: u32, k_words: u32, flags: u32, alpha: f32, k_tiles: u32, k_split: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> gm_x: array<u32>;
@group(0) @binding(1) var<storage, read> gm_w: array<u32>;
@group(0) @binding(2) var<storage, read> gm_bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> gm_y: array<u32>;
@group(0) @binding(4) var<uniform> gm_p: VxGemmParams;
@group(0) @binding(42) var<storage, read_write> gm_part: array<f32>;
@group(0) @binding(43) var<storage, read> gm_sf: array<u32>;

const GM_LD: u32 = 33u;
var<workgroup> gm_xs: array<f32, 2112>;
var<workgroup> gm_ws: array<f32, 2112>;

fn vx_erf(x: f32) -> f32 {
    let a = abs(x);
    let t = 1.0 / fma(0.3275911, a, 1.0);
    let poly = t * fma(t, fma(t, fma(t, fma(t, 1.061405429, -1.453152027), 1.421413741), -0.284496736), 0.254829592);
    let e = 1.0 - poly * exp(-a * a);
    return select(e, -e, x < 0.0);
}

fn vx_gelu(x: f32) -> f32 {
    return 0.5 * x * (1.0 + vx_erf(x * 0.70710678118654752));
}

fn gm_store(r: u32, c: u32, v0_in: f32, v1_in: f32) {
    var v0 = v0_in;
    var v1 = v1_in;
    if ((gm_p.flags & 1u) != 0u) { v0 = v0 + gm_bias[c]; v1 = v1 + gm_bias[c + 1u]; }
    if ((gm_p.flags & 2u) != 0u) { v0 = vx_gelu(v0); v1 = vx_gelu(v1); }
    if ((gm_p.flags & 4u) != 0u) {
        let idx = r * gm_p.n + c;
        if ((gm_p.flags & 8u) != 0u) { v0 = v0 + bitcast<f32>(gm_y[idx]); v1 = v1 + bitcast<f32>(gm_y[idx + 1u]); }
        gm_y[idx] = bitcast<u32>(v0);
        gm_y[idx + 1u] = bitcast<u32>(v1);
    } else {
        let widx = (r * gm_p.n + c) >> 1u;
        if ((gm_p.flags & 8u) != 0u) { let old = gm_y[widx]; v0 = v0 + bf16_lo(old); v1 = v1 + bf16_hi(old); }
        gm_y[widx] = bf16_pack(v0, v1);
    }
}

fn gm_emit(r: u32, c: u32, v0: f32, v1: f32, z: u32) {
    if (r >= gm_p.m || c >= gm_p.n) { return; }
    if (gm_p.k_split == 1u) {
        gm_store(r, c, v0, v1);
    } else {
        let b = (z * gm_p.m + r) * gm_p.n + c;
        gm_part[b] = v0;
        gm_part[b + 1u] = v1;
    }
}

fn gm_stage8(base: u32, v: vec4<u32>) {
    gm_xs[base] = bf16_lo(v.x);
    gm_xs[base + 1u] = bf16_hi(v.x);
    gm_xs[base + 2u] = bf16_lo(v.y);
    gm_xs[base + 3u] = bf16_hi(v.y);
    gm_xs[base + 4u] = bf16_lo(v.z);
    gm_xs[base + 5u] = bf16_hi(v.z);
    gm_xs[base + 6u] = bf16_lo(v.w);
    gm_xs[base + 7u] = bf16_hi(v.w);
}

fn gm_stage8w(base: u32, v: vec4<u32>) {
    gm_ws[base] = bf16_lo(v.x);
    gm_ws[base + 1u] = bf16_hi(v.x);
    gm_ws[base + 2u] = bf16_lo(v.y);
    gm_ws[base + 3u] = bf16_hi(v.y);
    gm_ws[base + 4u] = bf16_lo(v.z);
    gm_ws[base + 5u] = bf16_hi(v.z);
    gm_ws[base + 6u] = bf16_lo(v.w);
    gm_ws[base + 7u] = bf16_hi(v.w);
}

fn gm_load_x(row: u32, kt: u32, lc: u32) -> vec4<u32> {
    if (row >= gm_p.m) { return vec4<u32>(0u); }
    let b = row * gm_p.k_words + kt * 16u + lc * 4u;
    return vec4<u32>(gm_x[b], gm_x[b + 1u], gm_x[b + 2u], gm_x[b + 3u]);
}

fn gm_load_w(col: u32, kt: u32, lc: u32) -> vec4<u32> {
    if (col >= gm_p.n) { return vec4<u32>(0u); }
    let b = col * gm_p.k_words + kt * 16u + lc * 4u;
    return vec4<u32>(gm_w[b], gm_w[b + 1u], gm_w[b + 2u], gm_w[b + 3u]);
}

fn gm_load_w4(col: u32, kt: u32, lc: u32) -> vec4<u32> {
    if (col >= gm_p.n) { return vec4<u32>(0u); }
    return vec4<u32>(gm_w[col * (gm_p.k_words >> 2u) + kt * 4u + lc], 0u, 0u, 0u);
}

fn gm_load_sf(col: u32, kt: u32, lc: u32) -> u32 {
    if (col >= gm_p.n) { return 0u; }
    return gm_sf[nvfp4_scale_byte_index(col, kt * 2u + (lc >> 1u), gm_p.k_tiles) >> 2u];
}

fn gm_stage8w4(base: u32, wn: u32, sw: u32, col: u32, kt: u32, lc: u32) {
    let si = nvfp4_scale_byte_index(min(col, gm_p.n - 1u), kt * 2u + (lc >> 1u), gm_p.k_tiles);
    let sc = ue4m3_decode(byte_at(sw, si)) * gm_p.alpha;
    for (var jj = 0u; jj < 8u; jj = jj + 1u) { gm_ws[base + jj] = nv_e2m1(wn >> (4u * jj)) * sc; }
}

@compute @workgroup_size(64)
fn vx_gemm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let tx = tid & 7u;
    let ty = tid >> 3u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let kw = gm_p.k_words;
    let n_kt = kw / 16u;
    let per = (n_kt + gm_p.k_split - 1u) / gm_p.k_split;
    let kt0 = wid.z * per;
    let kt1 = min(n_kt, kt0 + per);
    let lr = tid >> 2u;
    let lc = tid & 3u;
    let w4 = (gm_p.flags & 16u) != 0u;
    GEMM_ACC_DECL
    var xr0 = vec4<u32>(0u);
    var xr1 = vec4<u32>(0u);
    var xr2 = vec4<u32>(0u);
    var xr3 = vec4<u32>(0u);
    var wr0 = vec4<u32>(0u);
    var wr1 = vec4<u32>(0u);
    var wr2 = vec4<u32>(0u);
    var wr3 = vec4<u32>(0u);
    var sw0 = 0u;
    var sw1 = 0u;
    var sw2 = 0u;
    var sw3 = 0u;
    var kt = kt0;
    if (kt < kt1) {
        xr0 = gm_load_x(row0 + lr, kt, lc);
        xr1 = gm_load_x(row0 + lr + 16u, kt, lc);
        xr2 = gm_load_x(row0 + lr + 32u, kt, lc);
        xr3 = gm_load_x(row0 + lr + 48u, kt, lc);
        if (w4) {
            wr0 = gm_load_w4(col0 + lr, kt, lc);
            wr1 = gm_load_w4(col0 + lr + 16u, kt, lc);
            wr2 = gm_load_w4(col0 + lr + 32u, kt, lc);
            wr3 = gm_load_w4(col0 + lr + 48u, kt, lc);
            sw0 = gm_load_sf(col0 + lr, kt, lc);
            sw1 = gm_load_sf(col0 + lr + 16u, kt, lc);
            sw2 = gm_load_sf(col0 + lr + 32u, kt, lc);
            sw3 = gm_load_sf(col0 + lr + 48u, kt, lc);
        } else {
            wr0 = gm_load_w(col0 + lr, kt, lc);
            wr1 = gm_load_w(col0 + lr + 16u, kt, lc);
            wr2 = gm_load_w(col0 + lr + 32u, kt, lc);
            wr3 = gm_load_w(col0 + lr + 48u, kt, lc);
        }
    }
    for (; kt < kt1; kt = kt + 1u) {
        let xs = lr * GM_LD + lc * 8u;
        gm_stage8(xs, xr0);
        gm_stage8(xs + 16u * GM_LD, xr1);
        gm_stage8(xs + 32u * GM_LD, xr2);
        gm_stage8(xs + 48u * GM_LD, xr3);
        if (w4) {
            gm_stage8w4(xs, wr0.x, sw0, col0 + lr, kt, lc);
            gm_stage8w4(xs + 16u * GM_LD, wr1.x, sw1, col0 + lr + 16u, kt, lc);
            gm_stage8w4(xs + 32u * GM_LD, wr2.x, sw2, col0 + lr + 32u, kt, lc);
            gm_stage8w4(xs + 48u * GM_LD, wr3.x, sw3, col0 + lr + 48u, kt, lc);
        } else {
            gm_stage8w(xs, wr0);
            gm_stage8w(xs + 16u * GM_LD, wr1);
            gm_stage8w(xs + 32u * GM_LD, wr2);
            gm_stage8w(xs + 48u * GM_LD, wr3);
        }
        let kn = kt + 1u;
        if (kn < kt1) {
            xr0 = gm_load_x(row0 + lr, kn, lc);
            xr1 = gm_load_x(row0 + lr + 16u, kn, lc);
            xr2 = gm_load_x(row0 + lr + 32u, kn, lc);
            xr3 = gm_load_x(row0 + lr + 48u, kn, lc);
            if (w4) {
                wr0 = gm_load_w4(col0 + lr, kn, lc);
                wr1 = gm_load_w4(col0 + lr + 16u, kn, lc);
                wr2 = gm_load_w4(col0 + lr + 32u, kn, lc);
                wr3 = gm_load_w4(col0 + lr + 48u, kn, lc);
                sw0 = gm_load_sf(col0 + lr, kn, lc);
                sw1 = gm_load_sf(col0 + lr + 16u, kn, lc);
                sw2 = gm_load_sf(col0 + lr + 32u, kn, lc);
                sw3 = gm_load_sf(col0 + lr + 48u, kn, lc);
            } else {
                wr0 = gm_load_w(col0 + lr, kn, lc);
                wr1 = gm_load_w(col0 + lr + 16u, kn, lc);
                wr2 = gm_load_w(col0 + lr + 32u, kn, lc);
                wr3 = gm_load_w(col0 + lr + 48u, kn, lc);
            }
        }
        workgroupBarrier();
        let xb = ty * 8u * GM_LD;
        let wb = tx * 8u * GM_LD;
        for (var kk = 0u; kk < 32u; kk = kk + 1u) {
            GEMM_FMA_BODY
        }
        workgroupBarrier();
    }
    let r0 = row0 + ty * 8u;
    let c0 = col0 + tx * 8u;
    GEMM_EPILOGUE
}

@compute @workgroup_size(256)
fn vx_gemm_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    let pairs = gm_p.n >> 1u;
    if (idx >= gm_p.m * pairs) { return; }
    let r = idx / pairs;
    let c = (idx % pairs) * 2u;
    var v0 = 0.0;
    var v1 = 0.0;
    for (var z = 0u; z < gm_p.k_split; z = z + 1u) {
        let b = (z * gm_p.m + r) * gm_p.n + c;
        v0 = v0 + gm_part[b];
        v1 = v1 + gm_part[b + 1u];
    }
    gm_store(r, c, v0, v1);
}

struct VxRopeParams { rows: u32, nh: u32, nkv: u32, hd: u32, qkv_stride: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(5) var<storage, read> rk_qkv: array<f32>;
@group(0) @binding(6) var<storage, read> rk_cos: array<f32>;
@group(0) @binding(7) var<storage, read> rk_sin: array<f32>;
@group(0) @binding(8) var<storage, read> rk_pos: array<i32>;
@group(0) @binding(9) var<storage, read_write> rk_q: array<f32>;
@group(0) @binding(10) var<storage, read_write> rk_k: array<u32>;
@group(0) @binding(11) var<storage, read_write> rk_v: array<u32>;
@group(0) @binding(12) var<uniform> rk_p: VxRopeParams;

fn rk_rot(base: u32, d: u32, half: u32, cbase: u32) -> f32 {
    if (d < half) {
        return fma(rk_qkv[base + d], rk_cos[cbase + d], -(rk_qkv[base + d + half] * rk_sin[cbase + d]));
    }
    let i = d - half;
    return fma(rk_qkv[base + d], rk_cos[cbase + i], rk_qkv[base + i] * rk_sin[cbase + i]);
}

@compute @workgroup_size(256)
fn vx_rope_kv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = (wid.y * nwg.x + wid.x) * 256u + lid.x;
    let words = rk_p.hd >> 1u;
    let half = words;
    let heads = rk_p.nh + 2u * rk_p.nkv;
    if (idx >= rk_p.rows * heads * words) { return; }
    let j = idx % words;
    let h = (idx / words) % heads;
    let r = idx / (words * heads);
    let pos = u32(rk_pos[0] + i32(r));
    let base = r * rk_p.qkv_stride + h * rk_p.hd;
    let d = 2u * j;
    if (h < rk_p.nh) {
        let o = (r * rk_p.nh + h) * rk_p.hd + d;
        rk_q[o] = rk_rot(base, d, half, pos * half);
        rk_q[o + 1u] = rk_rot(base, d + 1u, half, pos * half);
    } else if (h < rk_p.nh + rk_p.nkv) {
        let slot = (pos * rk_p.nkv + (h - rk_p.nh)) * words + j;
        rk_k[slot] = bf16_pack(rk_rot(base, d, half, pos * half), rk_rot(base, d + 1u, half, pos * half));
    } else {
        let slot = (pos * rk_p.nkv + (h - rk_p.nh - rk_p.nkv)) * words + j;
        rk_v[slot] = bf16_pack(rk_qkv[base + d], rk_qkv[base + d + 1u]);
    }
}

struct VxAttnParams { rows: u32, nh: u32, nkv: u32, hd: u32, window: u32, scale: f32, splits: u32, perm: u32 };
@group(0) @binding(13) var<storage, read> at_q: array<f32>;
@group(0) @binding(14) var<storage, read> at_k: array<u32>;
@group(0) @binding(15) var<storage, read> at_v: array<u32>;
@group(0) @binding(16) var<storage, read> at_pos: array<i32>;
@group(0) @binding(17) var<storage, read_write> at_out: array<u32>;
@group(0) @binding(18) var<uniform> at_p: VxAttnParams;
@group(0) @binding(19) var<storage, read_write> at_scratch: array<f32>;

var<workgroup> ad_q: array<f32, 128>;
var<workgroup> ad_m: array<f32, 16>;
var<workgroup> ad_l: array<f32, 16>;
var<workgroup> ad_acc: array<f32, 2048>;

fn ad_neg_inf() -> f32 { return bitcast<f32>(0xff800000u); }

@compute @workgroup_size(256)
fn vx_attn_decode1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_id) warp: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let h = wid.x;
    let split = wid.y;
    let row = wid.z;
    let tid = lid.x;
    let nwarps = 256u / sg_size;
    let hd = at_p.hd;
    let words = hd >> 1u;
    let nkv = at_p.nkv;
    let kvh = h / (at_p.nh / nkv);
    let total = u32(at_pos[0]) + 1u + row;
    for (var d = tid; d < hd; d = d + 256u) { ad_q[d] = at_q[(row * at_p.nh + h) * hd + d]; }
    workgroupBarrier();
    let stride = at_p.splits * nwarps;
    let base = split * nwarps;
    var rounds = 0u;
    if (total > base) { rounds = (total - base + stride - 1u) / stride; }
    var m = ad_neg_inf();
    var l = 0.0;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    let lane2 = lane + sg_size;
    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var partial = 0.0;
        if (live) {
            let kb = (p * nkv + kvh) * words;
            for (var w = lane; w < words; w = w + sg_size) {
                let kw = at_k[kb + w];
                partial = fma(ad_q[2u * w], bf16_lo(kw), partial);
                partial = fma(ad_q[2u * w + 1u], bf16_hi(kw), partial);
            }
        }
        let s = subgroupAdd(partial) * at_p.scale;
        if (live) {
            let m_new = max(m, s);
            let corr = exp(m - m_new);
            let pw = exp(s - m_new);
            l = fma(l, corr, pw);
            let vb = (p * nkv + kvh) * words;
            if (lane < words) {
                let w0 = at_v[vb + lane];
                a0 = fma(a0, corr, pw * bf16_lo(w0));
                a1 = fma(a1, corr, pw * bf16_hi(w0));
            }
            if (lane2 < words) {
                let w1 = at_v[vb + lane2];
                a2 = fma(a2, corr, pw * bf16_lo(w1));
                a3 = fma(a3, corr, pw * bf16_hi(w1));
            }
            m = m_new;
        }
    }
    if (lane == 0u) { ad_m[warp] = m; ad_l[warp] = l; }
    if (lane < words) {
        ad_acc[warp * 128u + 2u * lane] = a0;
        ad_acc[warp * 128u + 2u * lane + 1u] = a1;
    }
    if (lane2 < words) {
        ad_acc[warp * 128u + 2u * lane2] = a2;
        ad_acc[warp * 128u + 2u * lane2 + 1u] = a3;
    }
    workgroupBarrier();
    var m_blk = ad_neg_inf();
    for (var w = 0u; w < nwarps; w = w + 1u) { m_blk = max(m_blk, ad_m[w]); }
    let slot = ((row * at_p.nh + h) * at_p.splits + split) * (hd + 2u);
    if (tid == 0u) {
        var l_blk = 0.0;
        if (m_blk > ad_neg_inf()) {
            for (var w = 0u; w < nwarps; w = w + 1u) {
                if (ad_m[w] > ad_neg_inf()) { l_blk = l_blk + ad_l[w] * exp(ad_m[w] - m_blk); }
            }
        }
        at_scratch[slot] = m_blk;
        at_scratch[slot + 1u] = l_blk;
    }
    for (var d = tid; d < hd; d = d + 256u) {
        var a = 0.0;
        if (m_blk > ad_neg_inf()) {
            for (var w = 0u; w < nwarps; w = w + 1u) {
                if (ad_m[w] > ad_neg_inf()) { a = a + ad_acc[w * 128u + d] * exp(ad_m[w] - m_blk); }
            }
        }
        at_scratch[slot + 2u + d] = a;
    }
}

@compute @workgroup_size(256)
fn vx_attn_decode2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.x;
    let row = wid.y;
    let tid = lid.x;
    let hd = at_p.hd;
    let splits = at_p.splits;
    let stride = hd + 2u;
    let base = (row * at_p.nh + h) * splits * stride;
    let words = hd >> 1u;
    let groups = 256u / words;
    let g = tid / words;
    let w = tid % words;
    var m_glob = ad_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) { m_glob = max(m_glob, at_scratch[base + s * stride]); }
    var l_glob = 0.0;
    var x0 = 0.0;
    var x1 = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let ms = at_scratch[base + s * stride];
        if (ms > ad_neg_inf()) {
            let sc = exp(ms - m_glob);
            l_glob = fma(at_scratch[base + s * stride + 1u], sc, l_glob);
            if (s % groups == g) {
                x0 = fma(at_scratch[base + s * stride + 2u + 2u * w], sc, x0);
                x1 = fma(at_scratch[base + s * stride + 3u + 2u * w], sc, x1);
            }
        }
    }
    ad_acc[tid * 2u] = x0;
    ad_acc[tid * 2u + 1u] = x1;
    workgroupBarrier();
    if (g != 0u) { return; }
    for (var q = 1u; q < groups; q = q + 1u) {
        x0 = x0 + ad_acc[(q * words + w) * 2u];
        x1 = x1 + ad_acc[(q * words + w) * 2u + 1u];
    }
    var inv = 0.0;
    if (l_glob > 0.0) { inv = 1.0 / l_glob; }
    let o = (row * at_p.nh + h) * words + w;
    at_out[select(o, nv_perm_word(o), at_p.perm != 0u)] = bf16_pack(x0 * inv, x1 * inv);
}

struct VxEwParams { rows: u32, words: u32, a: u32, b: u32, c: i32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(20) var<storage, read> ea_embed: array<u32>;
@group(0) @binding(21) var<storage, read> ea_tok: array<i32>;
@group(0) @binding(22) var<storage, read> ea_audio: array<u32>;
@group(0) @binding(23) var<storage, read> ea_pos: array<i32>;
@group(0) @binding(24) var<storage, read_write> ea_out: array<u32>;
@group(0) @binding(25) var<uniform> ea_p: VxEwParams;

fn vx_flat(wid: vec3<u32>, nwg: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (wid.y * nwg.x + wid.x) * 256u + lid.x;
}

@compute @workgroup_size(256)
fn vx_embed_add(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = vx_flat(wid, nwg, lid);
    if (idx >= ea_p.rows * ea_p.words) { return; }
    let r = idx / ea_p.words;
    let w = idx % ea_p.words;
    let e = ea_embed[u32(ea_tok[r]) * ea_p.words + w];
    let a = ea_audio[u32(ea_pos[0] + i32(r)) * ea_p.words + w];
    ea_out[idx] = bf16_pack(bf16_lo(e) + bf16_lo(a), bf16_hi(e) + bf16_hi(a));
}

@group(0) @binding(26) var<storage, read> sm_gu: array<u32>;
@group(0) @binding(27) var<storage, read_write> sm_out: array<u32>;
@group(0) @binding(28) var<uniform> sm_p: VxEwParams;

fn vx_silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn vx_silu_mul_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = vx_flat(wid, nwg, lid);
    if (idx >= sm_p.rows * sm_p.words) { return; }
    let r = idx / sm_p.words;
    let w = idx % sm_p.words;
    let g = sm_gu[r * 2u * sm_p.words + w];
    let u = sm_gu[r * 2u * sm_p.words + sm_p.words + w];
    sm_out[select(idx, nv_perm_word(idx), sm_p.a != 0u)] = bf16_pack(vx_silu(bf16_lo(g)) * bf16_lo(u), vx_silu(bf16_hi(g)) * bf16_hi(u));
}

@group(0) @binding(29) var<storage, read> ic_src: array<u32>;
@group(0) @binding(30) var<storage, read_write> ic_dst: array<u32>;
@group(0) @binding(31) var<uniform> ic_p: VxEwParams;

@compute @workgroup_size(256)
fn vx_im2col3(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = vx_flat(wid, nwg, lid);
    let cw = ic_p.words;
    if (idx >= ic_p.rows * 3u * cw) { return; }
    let t2 = idx / (3u * cw);
    let rem = idx % (3u * cw);
    let tap = rem / cw;
    let c = rem % cw;
    let src_t = i32(t2 * ic_p.a + tap) - i32(ic_p.b);
    var v = 0u;
    if (src_t >= 0 && src_t < ic_p.c) { v = ic_src[u32(src_t) * cw + c]; }
    ic_dst[idx] = v;
}

@group(0) @binding(32) var<storage, read_write> ai_y: array<u32>;
@group(0) @binding(33) var<storage, read> ai_b: array<u32>;
@group(0) @binding(34) var<uniform> ai_p: VxEwParams;

@compute @workgroup_size(256)
fn vx_add_inplace(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = vx_flat(wid, nwg, lid);
    if (idx >= ai_p.words) { return; }
    let y = ai_y[idx];
    let b = ai_b[idx];
    ai_y[idx] = bf16_pack(bf16_lo(y) + bf16_lo(b), bf16_hi(y) + bf16_hi(b));
}

struct VxArgmaxParams { n: u32, nparts: u32, ring_mask: i32, incr_pos: u32, rows: u32, pad: i32, pad0: u32, pad1: u32 };
@group(0) @binding(35) var<storage, read> am_logits: array<f32>;
@group(0) @binding(36) var<storage, read_write> am_pval: array<f32>;
@group(0) @binding(37) var<storage, read_write> am_pidx: array<i32>;
@group(0) @binding(38) var<uniform> am_p: VxArgmaxParams;
@group(0) @binding(39) var<storage, read_write> am_pos: array<i32>;
@group(0) @binding(40) var<storage, read_write> am_tok: array<u32>;
@group(0) @binding(41) var<storage, read_write> am_ring: array<u32>;

var<workgroup> am_val: array<f32, 256>;
var<workgroup> am_idx: array<i32, 256>;
var<workgroup> am_out: array<i32, 8>;

fn am_fold(lid: u32, s: u32) {
    if (lid < s) {
        let ov = am_val[lid + s];
        let oi = am_idx[lid + s];
        if (ov > am_val[lid] || (ov == am_val[lid] && oi < am_idx[lid])) {
            am_val[lid] = ov;
            am_idx[lid] = oi;
        }
    }
}

@compute @workgroup_size(128)
fn vx_argmax1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var best = ad_neg_inf();
    var bidx = 2147483647;
    let n = i32(am_p.n);
    let stride = i32(am_p.nparts * 128u);
    let lb = wid.y * am_p.n;
    var i = i32(wid.x * 128u + t);
    while (i < n) {
        let v = am_logits[lb + u32(i)];
        if (v > best || (v == best && i < bidx)) { best = v; bidx = i; }
        i = i + stride;
    }
    am_val[t] = best;
    am_idx[t] = bidx;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        am_fold(t, s);
        workgroupBarrier();
    }
    if (t == 0u) {
        am_pval[wid.y * am_p.nparts + wid.x] = am_val[0];
        am_pidx[wid.y * am_p.nparts + wid.x] = am_idx[0];
    }
}

@compute @workgroup_size(256)
fn vx_argmax2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    for (var r = 0u; r < am_p.rows; r = r + 1u) {
        if (t < am_p.nparts) {
            am_val[t] = am_pval[r * am_p.nparts + t];
            am_idx[t] = am_pidx[r * am_p.nparts + t];
        } else {
            am_val[t] = ad_neg_inf();
            am_idx[t] = 2147483647;
        }
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            am_fold(t, s);
            workgroupBarrier();
        }
        if (t == 0u) { am_out[r] = select(am_idx[0], 0, am_idx[0] == 2147483647); }
        workgroupBarrier();
    }
    if (t == 0u) {
        let p = am_pos[0];
        var n = 1u;
        while (n < am_p.rows && am_out[n - 1u] == am_p.pad) { n = n + 1u; }
        for (var i = 0u; i < n; i = i + 1u) { am_ring[u32((p + i32(i)) & am_p.ring_mask)] = u32(am_out[i]); }
        am_tok[0] = u32(am_out[n - 1u]);
        if (am_p.incr_pos == 1u) {
            am_pos[0] = p + i32(n);
            am_ring[RINGPOSu] = u32(p + i32(n));
        }
    }
}

struct VxNormParams { kvec: u32, x_off_vec4: u32, eps: f32, pad0: u32 };
@group(0) @binding(50) var<storage, read> nr_x: array<vec4<u32>>;
@group(0) @binding(51) var<storage, read> nr_w: array<vec4<u32>>;
@group(0) @binding(52) var<storage, read_write> nr_y: array<vec4<u32>>;
@group(0) @binding(53) var<uniform> nr_p: VxNormParams;

var<workgroup> nr_red: array<f32, 256>;

@compute @workgroup_size(256)
fn vx_norm_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let tid = lid.x;
    let kvec = nr_p.kvec;
    let base = nr_p.x_off_vec4 + r * kvec;
    var local = 0.0;
    for (var i = tid; i < kvec; i = i + 256u) { local = local + nv_sq4(nr_x[base + i]); }
    nr_red[tid] = local;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) { nr_red[tid] = nr_red[tid] + nr_red[tid + stride]; }
        workgroupBarrier();
    }
    let rstd = inverseSqrt(nr_p.eps + nv_div_rn(nr_red[0], f32(kvec * 8u)));
    for (var i = tid; i < kvec; i = i + 256u) { nr_y[r * kvec + nv_perm(i)] = nv_norm4(nr_x[base + i], nr_w[i], rstd); }
}
"#;

const VX_GEMV_NV4_WGSL: &str = r#"
struct VxNv4Params { n_rows: u32, k_blocks: u32, k_tiles: u32, groups_x: u32, alpha: f32, flags: u32, x_off_vec4: u32, sg_per_pair: u32, pad0: u32, pad1: u32, pad2: u32, pad3: u32 };
@group(0) @binding(44) var<storage, read> nv_w: array<vec4<u32>>;
@group(0) @binding(45) var<storage, read> nv_sf: array<u32>;
@group(0) @binding(46) var<storage, read> nv_x: array<vec4<u32>>;
@group(0) @binding(47) var<storage, read_write> nv_y: array<u32>;
@group(0) @binding(48) var<uniform> nv_p: VxNv4Params;

const NV_M: u32 = MVALu;
var<workgroup> nv_red: array<f32, NVREDu>;

fn nv_dot8(ea: vec4<f32>, eb: vec4<f32>, x: vec4<u32>) -> f32 {
    var d = 0.0;
    d = fma(ea.x, bf16_lo(x.x), d);
    d = fma(ea.y, bf16_hi(x.x), d);
    d = fma(ea.z, bf16_lo(x.y), d);
    d = fma(ea.w, bf16_hi(x.y), d);
    d = fma(eb.x, bf16_lo(x.z), d);
    d = fma(eb.y, bf16_hi(x.z), d);
    d = fma(eb.z, bf16_lo(x.w), d);
    d = fma(eb.w, bf16_hi(x.w), d);
    return d;
}

fn nv_dec2(by: u32) -> vec2<f32> {
    let lo = ((by & 7u) << 9u) | ((by & 8u) << 12u);
    let hi = ((by & 0x70u) << 21u) | ((by & 0x80u) << 24u);
    return unpack2x16float(lo | hi) * 16384.0;
}

fn nv_dec_lo(wn: u32) -> vec4<f32> {
    let b = unpack4xU8(wn);
    return vec4<f32>(nv_dec2(b.x), nv_dec2(b.y));
}

fn nv_dec_hi(wn: u32) -> vec4<f32> {
    let b = unpack4xU8(wn);
    return vec4<f32>(nv_dec2(b.z), nv_dec2(b.w));
}

fn nv_row2(e0a: vec4<f32>, e0b: vec4<f32>, e1a: vec4<f32>, e1b: vec4<f32>, e2a: vec4<f32>, e2b: vec4<f32>, e3a: vec4<f32>, e3b: vec4<f32>, s0: f32, s1: f32, xr: u32, acc: f32) -> f32 {
    let a = fma(s0, nv_dot8(e0a, e0b, nv_x[xr]) + nv_dot8(e1a, e1b, nv_x[xr + 32u]), acc);
    return fma(s1, nv_dot8(e2a, e2b, nv_x[xr + 64u]) + nv_dot8(e3a, e3b, nv_x[xr + 96u]), a);
}

@compute @workgroup_size(256)
fn vx_gemv_nv4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_size) sg_size: u32,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let xo = nv_p.x_off_vec4;
    let kvec = nv_p.k_blocks * 2u;
    let spp = nv_p.sg_per_pair;
    let pairs_per_group = (256u / sg_size) / spp;
    let pair = (wid.x + wid.y * nv_p.groups_x) * pairs_per_group + sgid / spp;
    let part = sgid % spp;
    let row0 = pair * 2u;
    let live = row0 < nv_p.n_rows;
    let bp = nv_p.k_blocks >> 1u;
    let bpairs = select(0u, bp, live);
    let wb0 = select(0u, row0 * bp, live);
    let wb1 = select(0u, (row0 + 1u) * bp, live);
    let sr0 = select(0u, row0, live);
    let sr1 = select(0u, row0 + 1u, live);
    let stride = spp * sg_size;
    ACC_DECL
    for (var p = part * sg_size + lane; p < bpairs; p = p + stride) {
        let kb = p * 2u;
        let si0 = nvfp4_scale_byte_index(sr0, kb, nv_p.k_tiles);
        let si1 = nvfp4_scale_byte_index(sr1, kb, nv_p.k_tiles);
        let sw0 = nv_sf[si0 >> 2u];
        let sw1 = nv_sf[si1 >> 2u];
        let s00 = ue4m3_decode(byte_at(sw0, si0));
        let s01 = ue4m3_decode(byte_at(sw0, si0 + 1u));
        let s10 = ue4m3_decode(byte_at(sw1, si1));
        let s11 = ue4m3_decode(byte_at(sw1, si1 + 1u));
        let w0 = nv_w[wb0 + p];
        let w1 = nv_w[wb1 + p];
        let a0 = nv_dec_lo(w0.x);
        let a1 = nv_dec_hi(w0.x);
        let a2 = nv_dec_lo(w0.y);
        let a3 = nv_dec_hi(w0.y);
        let a4 = nv_dec_lo(w0.z);
        let a5 = nv_dec_hi(w0.z);
        let a6 = nv_dec_lo(w0.w);
        let a7 = nv_dec_hi(w0.w);
        let b0 = nv_dec_lo(w1.x);
        let b1 = nv_dec_hi(w1.x);
        let b2 = nv_dec_lo(w1.y);
        let b3 = nv_dec_hi(w1.y);
        let b4 = nv_dec_lo(w1.z);
        let b5 = nv_dec_hi(w1.z);
        let b6 = nv_dec_lo(w1.w);
        let b7 = nv_dec_hi(w1.w);
        let xb = xo + ((p >> 5u) << 7u) + (p & 31u);
        ACC_ROWS
    }
    ACC_REDUCE
    workgroupBarrier();
    if (lane != 0u || part != 0u || !live) { return; }
    for (var r = 0u; r < NV_M; r = r + 1u) {
        var lo = 0.0;
        var hi = 0.0;
        for (var q = 0u; q < spp; q = q + 1u) {
            lo = lo + nv_red[(sgid + q) * 2u * NV_M + 2u * r];
            hi = hi + nv_red[(sgid + q) * 2u * NV_M + 2u * r + 1u];
        }
        lo = lo * nv_p.alpha;
        hi = hi * nv_p.alpha;
        let o = r * nv_p.n_rows + row0;
        if ((nv_p.flags & 1u) != 0u) {
            nv_y[o] = bitcast<u32>(lo);
            nv_y[o + 1u] = bitcast<u32>(hi);
        } else {
            if ((nv_p.flags & 2u) != 0u) {
                let old = nv_y[o >> 1u];
                lo = lo + bf16_lo(old);
                hi = hi + bf16_hi(old);
            }
            nv_y[o >> 1u] = bf16_pack(lo, hi);
        }
    }
}
"#;

const VX_ATTN_PREFILL_WGSL: &str = r#"
struct VxAttnParams { rows: u32, nh: u32, nkv: u32, hd: u32, window: u32, scale: f32, splits: u32, perm: u32 };
@group(0) @binding(13) var<storage, read> at_q: array<f32>;
@group(0) @binding(14) var<storage, read> at_k: array<u32>;
@group(0) @binding(15) var<storage, read> at_v: array<u32>;
@group(0) @binding(16) var<storage, read> at_pos: array<i32>;
@group(0) @binding(17) var<storage, read_write> at_out: array<u32>;
@group(0) @binding(18) var<uniform> at_p: VxAttnParams;

const AP_HD: u32 = HDVALu;
const AP_W: u32 = AP_HD / 2u;
const AP_D: u32 = AP_HD / 4u;
const AP_DW: u32 = AP_D / 2u;
const AP_TK: u32 = 4096u / AP_HD;
var<workgroup> ap_ks: array<u32, 2048>;
var<workgroup> ap_vs: array<u32, 2048>;

fn ap_neg_inf() -> f32 { return bitcast<f32>(0xff800000u); }

@compute @workgroup_size(256)
fn vx_attn_prefill_hdHDVAL(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let h = wid.y;
    let tile0 = wid.x * 64u;
    let t = lid.x >> 2u;
    let c = lid.x & 3u;
    let r = tile0 + t;
    let rows = at_p.rows;
    let live = r < rows;
    let start = u32(at_pos[0]);
    let kvh = h / (at_p.nh / at_p.nkv);
    let qpos = start + min(r, rows - 1u);
    let tile_last = start + min(tile0 + 63u, rows - 1u);
    var kmin = 0u;
    var tile_kmin = 0u;
    if (at_p.window > 0u) {
        kmin = qpos - min(qpos, at_p.window - 1u);
        tile_kmin = (start + tile0) - min(start + tile0, at_p.window - 1u);
    }
    var q: array<f32, AP_D>;
    var acc: array<f32, AP_D>;
    let qb = (min(r, rows - 1u) * at_p.nh + h) * AP_HD + c * AP_D;
    for (var d = 0u; d < AP_D; d = d + 1u) {
        q[d] = at_q[qb + d] * at_p.scale;
        acc[d] = 0.0;
    }
    var m = ap_neg_inf();
    var l = 0.0;
    for (var kb = tile_kmin; kb <= tile_last; kb = kb + AP_TK) {
        for (var i = lid.x; i < AP_TK * AP_W; i = i + 256u) {
            let key = kb + i / AP_W;
            let w = i % AP_W;
            var kw = 0u;
            var vw = 0u;
            if (key <= tile_last) {
                let b = (key * at_p.nkv + kvh) * AP_W + w;
                kw = at_k[b];
                vw = at_v[b];
            }
            ap_ks[i] = kw;
            ap_vs[i] = vw;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < AP_TK; kk = kk + 1u) {
            let key = kb + kk;
            let sb = kk * AP_W + c * AP_DW;
            var s = 0.0;
            for (var w = 0u; w < AP_DW; w = w + 1u) {
                let kw = ap_ks[sb + w];
                s = fma(q[2u * w], bf16_lo(kw), s);
                s = fma(q[2u * w + 1u], bf16_hi(kw), s);
            }
            s = s + subgroupShuffleXor(s, 1u);
            s = s + subgroupShuffleXor(s, 2u);
            if (live && key >= kmin && key <= qpos) {
                let m_new = max(m, s);
                let corr = exp(m - m_new);
                let pw = exp(s - m_new);
                l = fma(l, corr, pw);
                for (var w = 0u; w < AP_DW; w = w + 1u) {
                    let vw = ap_vs[sb + w];
                    acc[2u * w] = fma(acc[2u * w], corr, pw * bf16_lo(vw));
                    acc[2u * w + 1u] = fma(acc[2u * w + 1u], corr, pw * bf16_hi(vw));
                }
                m = m_new;
            }
        }
        workgroupBarrier();
    }
    if (live) {
        var inv = 0.0;
        if (l > 0.0) { inv = 1.0 / l; }
        let ob = (r * at_p.nh + h) * AP_W + c * AP_DW;
        for (var w = 0u; w < AP_DW; w = w + 1u) {
            at_out[ob + w] = bf16_pack(acc[2u * w] * inv, acc[2u * w + 1u] * inv);
        }
    }
}
"#;

pod! {
    #[derive(Debug)]
    struct GemmParams { m: u32, n: u32, k_words: u32, flags: u32, alpha: f32, k_tiles: u32, k_split: u32, pad1: u32 }
    #[derive(Debug)]
    struct RopeParams { rows: u32, nh: u32, nkv: u32, hd: u32, qkv_stride: u32, pad0: u32, pad1: u32, pad2: u32 }
    #[derive(Debug)]
    struct AttnParams { rows: u32, nh: u32, nkv: u32, hd: u32, window: u32, scale: f32, splits: u32, perm: u32 }
    #[derive(Debug)]
    struct EwParams { rows: u32, words: u32, a: u32, b: u32, c: i32, pad0: u32, pad1: u32, pad2: u32 }
    #[derive(Debug)]
    struct ArgmaxParams { n: u32, nparts: u32, ring_mask: i32, incr_pos: u32, rows: u32, pad: i32, pad0: u32, pad1: u32 }
    #[derive(Debug)]
    struct Nv4Params { n_rows: u32, k_blocks: u32, k_tiles: u32, groups_x: u32, alpha: f32, flags: u32, x_off_vec4: u32, sg_per_pair: u32, pad0: u32, pad1: u32, pad2: u32, pad3: u32 }
    #[derive(Debug)]
    struct NormParams { kvec: u32, x_off_vec4: u32, eps: f32, pad0: u32 }
    #[derive(Debug)]
    struct GemvBf16Params { n_rows: u32, k_words: u32, groups_x: u32, out_f32: u32, w_row_words: u32, x_off_words: u32, y_off_words: u32, pad0: u32, alpha: f32, pad1: u32, pad2: u32, pad3: u32 }
    #[derive(Debug)]
    struct RmsParams { hidden: u32, batch: u32, eps: f32, words_per_row: u32 }
}

pub(crate) const GEMM_BIAS: u32 = 1;
pub(crate) const GEMM_GELU: u32 = 2;
pub(crate) const GEMM_OUT_F32: u32 = 4;
pub(crate) const GEMM_ACCUM: u32 = 8;
pub(crate) const GEMM_W4: u32 = 16;
const GEMM_SPLIT_K_TARGETS_THIS_MANY_WORKGROUPS: usize = 768;
const GEMM_SPLIT_K_MAX: usize = 16;
pub(crate) const NV4_OUT_F32: u32 = 1;
pub(crate) const NV4_ACCUM: u32 = 2;
const NV4_ROWS_ARE_PERMUTED_SO_A_SUBGROUP_READS_ITS_32_BLOCK_PAIRS_CONTIGUOUSLY: usize = 1024;

fn gemm_source(vx: &str) -> String {
    let ij = || (0..8).flat_map(|i| (0..8).map(move |j| (i, j)));
    let decl: Vec<String> = ij().map(|(i, j)| format!("var acc{i}_{j} = 0.0;")).collect();
    let loads: Vec<String> = (0..8)
        .map(|i| format!("let x{i} = gm_xs[xb + {i}u * GM_LD + kk];\n            let w{i} = gm_ws[wb + {i}u * GM_LD + kk];"))
        .collect();
    let fmas: Vec<String> = ij().map(|(i, j)| format!("acc{i}_{j} = fma(x{i}, w{j}, acc{i}_{j});")).collect();
    let epi: Vec<String> = (0..8)
        .flat_map(|i| (0..8).step_by(2).map(move |j| format!("gm_emit(r0 + {i}u, c0 + {j}u, acc{i}_{j}, acc{i}_{}, wid.z);", j + 1)))
        .collect();
    vx.replace("GEMM_ACC_DECL", &decl.join("\n    "))
        .replace("GEMM_FMA_BODY", &format!("{}\n            {}", loads.join("\n            "), fmas.join("\n            ")))
        .replace("GEMM_EPILOGUE", &epi.join("\n    "))
}

fn gemv_nv4_source(m: usize) -> String {
    let rows = |f: &dyn Fn(usize) -> String| (0..m).map(f).collect::<Vec<_>>().join("\n");
    VX_GEMV_NV4_WGSL
        .replace("ACC_DECL", &rows(&|r| format!("var acc{r}a = 0.0;\n    var acc{r}b = 0.0;")))
        .replace(
            "ACC_ROWS",
            &rows(&|r| {
                format!(
                    "let xr{r} = xb + {r}u * kvec;\n        acc{r}a = nv_row2(a0, a1, a2, a3, a4, a5, a6, a7, s00, s01, xr{r}, acc{r}a);\n        acc{r}b = nv_row2(b0, b1, b2, b3, b4, b5, b6, b7, s10, s11, xr{r}, acc{r}b);"
                )
            }),
        )
        .replace(
            "ACC_REDUCE",
            &rows(&|r| {
                format!(
                    "let t{r}a = subgroupAdd(acc{r}a);\n    let t{r}b = subgroupAdd(acc{r}b);\n    if (lane == 0u) {{ nv_red[sgid * 2u * NV_M + {}u] = t{r}a; nv_red[sgid * 2u * NV_M + {}u] = t{r}b; }}",
                    2 * r,
                    2 * r + 1
                )
            }),
        )
        .replace("NVRED", &(16 * m).to_string())
        .replace("MVAL", &m.to_string())
}

pub const A_PASS_HOLDS_ITS_BOUND_BUFFERS_BECAUSE_THE_WEBGPU_BACKEND_DESTROYS_A_BUFFER_WHEN_ITS_LAST_HANDLE_DROPS: &str =
    "native wgpu keeps a buffer alive through the bind group that names it, so a uniform or \
     scratch buffer created inside a pass builder may go out of scope; the vendored WebGPU \
     backend destroys the GPUBuffer when the last Rust clone drops and Chromium then rejects the \
     submit with 'used in submit while destroyed', so every pass clones the handles it binds";

pub(crate) struct Pass {
    pub(crate) pipeline: Arc<ComputePipeline>,
    pub(crate) bind: BindGroup,
    pub(crate) grid: (u32, u32, u32),
    pub(crate) label: String,
    pub(crate) _held: Vec<Buffer>,
}

impl Pass {
    pub(crate) fn named(mut self, lin: &Lin) -> Self {
        self.label = format!("{}:{}", self.label, lin.name);
        self
    }
}

pub(crate) fn refs(passes: &[Pass]) -> Vec<PassRef<'_>> {
    passes
        .iter()
        .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
        .collect()
}

pub(crate) enum Weights {
    Bf16(Buffer),
    Nv4 { w: Buffer, sf: Buffer, alpha: f32 },
}

pub(crate) fn quantize_nv4_rows(bits: &[u16], n: usize, k: usize) -> (Vec<u8>, Vec<u8>, f32) {
    let bv = |b: &u16| f32::from_bits((*b as u32) << 16);
    let amax = bits.iter().map(|b| bv(b).abs()).fold(0f32, f32::max);
    let stored = if amax.is_finite() && amax > 0.0 { 448.0 * 6.0 / amax } else { 1.0 };
    let alpha = if stored.is_finite() && stored != 0.0 { 1.0 / stored } else { 1.0 };
    let threads = std::thread::available_parallelism().map_or(8, |t| t.get()).clamp(1, 64);
    let rows_per = n.div_ceil(threads).max(1);
    let parts: Vec<(Vec<u8>, Vec<u8>)> = std::thread::scope(|s| {
        let handles: Vec<_> = bits
            .chunks(rows_per * k)
            .map(|chunk| {
                s.spawn(move || {
                    let rows: Vec<Vec<f32>> = chunk.chunks(k).map(|r| r.iter().map(bv).collect()).collect();
                    let q = nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&rows, stored);
                    (q.data, q.scales)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("nvfp4 row quantizer thread")).collect()
    });
    let mut packed = Vec::with_capacity(n * k / 2);
    let mut scales = Vec::with_capacity(n * k / 16);
    for (d, sc) in parts {
        packed.extend_from_slice(&d);
        scales.extend_from_slice(&sc);
    }
    (packed, nv_quant::nvfp4::swizzle_scales(&scales, n, k / 16), alpha)
}

pub(crate) struct Sources {
    vx: String,
    gemv_nv4: Vec<String>,
    attn_pf_64: String,
    attn_pf_128: String,
    gemv: String,
    rms: String,
    rmsres: String,
}

pub const UNGRANTED_SUBGROUPS_SELECT_WORKGROUP_MEMORY_TWINS_OF_THE_TWO_ATTENTION_REDUCTIONS: &str =
    "Tint rejects a module carrying a subgroup builtin on a device without the feature, so an \
     ungranted device (a browser without 'subgroups', or NV_WGPU_SUBGROUPS_UNGRANTED=1) builds vx \
     with its subgroup fns stripped plus vx_attn_decode1_nosg, whose eight 32-lane virtual warps \
     meet in workgroup memory where the granted kernel calls subgroupAdd, and prefill twins whose \
     four quarter-dots meet in workgroup memory in the butterfly's own association; a granting \
     device keeps its sources byte-identical, and nvfp4 stays refused without subgroups";

const NOSG_WARP_SUM_WGSL: &str = r#"
var<workgroup> ad_red: array<f32, 256>;

fn ad_warp_sum(v: f32, tid: u32, lane: u32) -> f32 {
    ad_red[tid] = v;
    workgroupBarrier();
    for (var s = 16u; s > 0u; s = s >> 1u) {
        if (lane < s) { ad_red[tid] = ad_red[tid] + ad_red[tid + s]; }
        workgroupBarrier();
    }
    let r = ad_red[tid - lane];
    workgroupBarrier();
    return r;
}
"#;

fn replaced(src: &str, from: &str, to: &str) -> String {
    assert!(src.contains(from), "nosg twin: marker {from:?} is gone from the kernel source");
    src.replace(from, to)
}

pub(crate) fn nosg_decode_source() -> String {
    let start = VX_WGSL.find("@compute @workgroup_size(256)\nfn vx_attn_decode1(").expect("vx_attn_decode1 in VX_WGSL");
    let end = VX_WGSL.find("@compute @workgroup_size(256)\nfn vx_attn_decode2(").expect("vx_attn_decode2 in VX_WGSL");
    let body = replaced(&VX_WGSL[start..end], "fn vx_attn_decode1(", "fn vx_attn_decode1_nosg(");
    let body = replaced(
        &body,
        ",\n    @builtin(subgroup_size) sg_size: u32,\n    @builtin(subgroup_id) warp: u32,\n    @builtin(subgroup_invocation_id) lane: u32\n) {",
        "\n) {\n    let sg_size = 32u;\n    let warp = lid.x >> 5u;\n    let lane = lid.x & 31u;",
    );
    let body = replaced(&body, "let s = subgroupAdd(partial) * at_p.scale;", "let s = ad_warp_sum(partial, lid.x, lane) * at_p.scale;");
    format!("{NOSG_WARP_SUM_WGSL}\n{body}")
}

pub(crate) fn nosg_prefill_source(hd: &str) -> String {
    let src = replaced(VX_ATTN_PREFILL_WGSL, "fn vx_attn_prefill_hdHDVAL(", "fn vx_attn_prefill_hdHDVAL_nosg(");
    let src = replaced(
        &src,
        "            s = s + subgroupShuffleXor(s, 1u);\n            s = s + subgroupShuffleXor(s, 2u);\n",
        "            ap_red[lid.x] = s;\n            workgroupBarrier();\n            let q4 = lid.x & ~3u;\n            s = (ap_red[q4] + ap_red[q4 + 1u]) + (ap_red[q4 + 2u] + ap_red[q4 + 3u]);\n            workgroupBarrier();\n",
    );
    compose(&format!("var<workgroup> ap_red: array<f32, 256>;\n{}", src.replace("HDVAL", hd)))
}

impl Sources {
    pub(crate) fn with_subgroups(sg: bool) -> Self {
        let _ = UNGRANTED_SUBGROUPS_SELECT_WORKGROUP_MEMORY_TWINS_OF_THE_TWO_ATTENTION_REDUCTIONS;
        let vx = compose(&format!("{VX_SHARED_WGSL}{}", gemm_source(&VX_WGSL.replace("RINGPOS", &RING_POS_SLOT.to_string()))));
        let (vx, attn_pf_64, attn_pf_128) = if sg {
            (vx, compose(&VX_ATTN_PREFILL_WGSL.replace("HDVAL", "64")), compose(&VX_ATTN_PREFILL_WGSL.replace("HDVAL", "128")))
        } else {
            (
                format!("{}\n{}", nv_kernels::wgpu_backend::strip_subgroup_fns(&vx), nosg_decode_source()),
                nosg_prefill_source("64"),
                nosg_prefill_source("128"),
            )
        };
        Self {
            vx,
            gemv_nv4: (1..=PAD_DRAFT_ROWS_MAX)
                .map(|m| compose(&format!("{VX_SHARED_WGSL}{}", gemv_nv4_source(m))))
                .collect(),
            attn_pf_64,
            attn_pf_128,
            gemv: compose(GEMV_BF16_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
        }
    }
}

pub(crate) struct Lin {
    pub(crate) w: Weights,
    pub(crate) bias: Option<Buffer>,
    pub(crate) n: usize,
    pub(crate) k: usize,
    pub(crate) name: &'static str,
}

pub(crate) struct Kit {
    pub(crate) ctx: &'static WgpuContext,
    pub(crate) src: Sources,
    pub(crate) dummy: Buffer,
    pub(crate) dummy_rw: Buffer,
}

pub(crate) fn err(e: nv_kernels::wgpu_backend::WgpuError) -> anyhow::Error {
    anyhow!("{e}")
}

impl Kit {
    pub(crate) fn new(ctx: &'static WgpuContext) -> Self {
        Self {
            ctx,
            src: Sources::with_subgroups(ctx.caps.subgroup),
            dummy: dispatch::storage_zeroed(ctx, "vx-dummy", 16),
            dummy_rw: dispatch::storage_zeroed(ctx, "vx-dummy-rw", 16),
        }
    }

    pub(crate) fn upload_u32(&self, label: &str, data: &[u32]) -> Buffer {
        dispatch::storage_from_slice(self.ctx, label, data)
    }

    pub(crate) fn upload_bf16(&self, label: &str, bits: &[u16]) -> Buffer {
        self.upload_u32(label, &pack_pairs(bits))
    }

    pub(crate) fn upload_f32(&self, label: &str, data: &[f32]) -> Buffer {
        dispatch::storage_from_slice(self.ctx, label, data)
    }

    pub(crate) fn upload_i32(&self, label: &str, data: &[i32]) -> Buffer {
        dispatch::storage_from_slice(self.ctx, label, data)
    }

    pub(crate) fn zeroed(&self, label: &str, bytes: usize) -> Buffer {
        dispatch::storage_zeroed(self.ctx, label, bytes as u64)
    }

    pub(crate) fn uni<T: bytemuck::Pod>(&self, label: &str, v: &T) -> Buffer {
        dispatch::uniform_from(self.ctx, label, v)
    }

    pub(crate) fn lin(&self, label: &'static str, bits: &[u16], n: usize, k: usize, bias: Option<&[f32]>) -> Result<Lin> {
        ensure!(bits.len() == n * k, "{label}: {} elements, want {n}x{k}", bits.len());
        ensure!(k % GEMM_K_STEP == 0 && n % 2 == 0, "{label}: {n}x{k} is not GEMM-tileable");
        let bytes = (n * k * 2) as u64;
        ensure!(
            bytes <= self.ctx.caps.max_storage_buffer_binding_size,
            "{label}: {bytes} bytes exceed the adapter's storage binding limit {}",
            self.ctx.caps.max_storage_buffer_binding_size
        );
        if let Some(b) = bias {
            ensure!(b.len() == n, "{label}: bias {} != {n}", b.len());
        }
        Ok(Lin {
            w: Weights::Bf16(self.upload_bf16(label, bits)),
            bias: bias.map(|b| self.upload_f32(label, b)),
            n,
            k,
            name: label,
        })
    }

    pub(crate) fn lin_quant(&self, label: &'static str, bits: &[u16], n: usize, k: usize, quant: VoxtralQuant) -> Result<Lin> {
        match quant {
            VoxtralQuant::Nvfp4 => self.lin_nv4(label, bits, n, k),
            _ => self.lin(label, bits, n, k, None),
        }
    }

    pub(crate) fn lin_nv4(&self, label: &'static str, bits: &[u16], n: usize, k: usize) -> Result<Lin> {
        ensure!(bits.len() == n * k, "{label}: {} elements, want {n}x{k}", bits.len());
        ensure!(k % GEMM_K_STEP == 0 && n % 2 == 0 && k % 8 == 0, "{label}: {n}x{k} is not nvfp4-tileable");
        ensure!(
            self.ctx.caps.subgroup,
            "{label}: the nvfp4 decode GEMV reduces across a subgroup; {VOXTRAL_QUANT_ENV}=bf16 serves this adapter"
        );
        let (packed, sf, alpha) = quantize_nv4_rows(bits, n, k);
        Ok(Lin {
            w: Weights::Nv4 {
                w: self.upload_u32(label, &nv_kernels::wgpu_backend::dequant::bytes_to_words(&packed)),
                sf: self.upload_u32(label, &nv_kernels::wgpu_backend::dequant::bytes_to_words(&sf)),
                alpha,
            },
            bias: None,
            n,
            k,
            name: label,
        })
    }

    pub(crate) fn pass(
        &self,
        source: &str,
        entry: &str,
        binds: &[(u32, &Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<Pass> {
        let _ = A_PASS_HOLDS_ITS_BOUND_BUFFERS_BECAUSE_THE_WEBGPU_BACKEND_DESTROYS_A_BUFFER_WHEN_ITS_LAST_HANDLE_DROPS;
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, LABEL, source, entry)
            .map_err(|e| anyhow!("pipeline {entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        Ok(Pass {
            pipeline,
            bind,
            grid,
            label: entry.to_string(),
            _held: binds.iter().map(|(_, b)| (*b).clone()).collect(),
        })
    }

    pub(crate) fn grid_flat(&self, invocations: usize, wg: u32) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations as u64, wg)
    }

    pub(crate) fn grid_rows(&self, rows: usize) -> (u32, u32, u32) {
        let limit = self.ctx.caps.max_compute_workgroups_per_dimension.max(1) as usize;
        if rows <= limit {
            (rows.max(1) as u32, 1, 1)
        } else {
            (limit as u32, rows.div_ceil(limit) as u32, 1)
        }
    }

    pub(crate) fn gemm(&self, x: &Buffer, lin: &Lin, y: &Buffer, m: usize, flags: u32) -> Result<Pass> {
        Ok(self.gemm_split(x, lin, y, m, flags, &self.dummy_rw, 1)?.pop().unwrap())
    }

    pub(crate) fn split_k_for(&self, lin: &Lin, m: usize) -> usize {
        let tiles = lin.n.div_ceil(GEMM_TILE) * m.div_ceil(GEMM_TILE);
        let n_kt = lin.k / GEMM_K_STEP;
        GEMM_SPLIT_K_TARGETS_THIS_MANY_WORKGROUPS
            .div_ceil(tiles)
            .min(GEMM_SPLIT_K_MAX)
            .min((n_kt / 4).max(1))
    }

    pub(crate) fn split_part(&self, shapes: &[(&Lin, usize)]) -> Buffer {
        let widest = shapes.iter().map(|(l, m)| self.split_k_for(l, *m) * l.n * m).max().unwrap_or(0);
        self.zeroed("vx-gemm-part", widest.max(4) * 4)
    }

    pub(crate) fn gemm_split(
        &self,
        x: &Buffer,
        lin: &Lin,
        y: &Buffer,
        m: usize,
        flags: u32,
        part: &Buffer,
        splits: usize,
    ) -> Result<Vec<Pass>> {
        let flags = flags | if lin.bias.is_some() { GEMM_BIAS } else { 0 };
        let (w, sf, alpha, flags) = match &lin.w {
            Weights::Bf16(w) => (w, &self.dummy, 1.0, flags),
            Weights::Nv4 { w, sf, alpha } => (w, sf, *alpha, flags | GEMM_W4),
        };
        ensure!(splits >= 1 && splits <= lin.k / GEMM_K_STEP, "{}: {splits} k-splits over {} k-tiles", lin.name, lin.k / GEMM_K_STEP);
        ensure!(
            splits == 1 || part.size() >= (splits * m * lin.n * 4) as u64,
            "{}: split-k scratch of {} bytes holds no {splits}x{m}x{} partials",
            lin.name,
            part.size(),
            lin.n
        );
        let p = self.uni(
            "vx-gemm-p",
            &GemmParams {
                m: m as u32,
                n: lin.n as u32,
                k_words: (lin.k / 2) as u32,
                flags,
                alpha,
                k_tiles: (lin.k / 16).div_ceil(4) as u32,
                k_split: splits as u32,
                pad1: 0,
            },
        );
        let bias = lin.bias.as_ref().unwrap_or(&self.dummy);
        let mut passes = vec![self
            .pass(
                &self.src.vx,
                "vx_gemm",
                &[(0, x), (1, w), (2, bias), (3, y), (4, &p), (42, part), (43, sf)],
                (lin.n.div_ceil(GEMM_TILE) as u32, m.div_ceil(GEMM_TILE) as u32, splits as u32),
            )?
            .named(lin)];
        if splits > 1 {
            passes.push(
                self.pass(
                    &self.src.vx,
                    "vx_gemm_reduce",
                    &[(2, bias), (3, y), (4, &p), (42, part)],
                    self.grid_flat(m * lin.n / 2, 256),
                )?
                .named(lin),
            );
        }
        Ok(passes)
    }

    pub(crate) fn gemv_nv4(&self, x: &Buffer, lin: &Lin, y: &Buffer, x_off_words: usize, rows: usize, flags: u32) -> Result<Pass> {
        let Weights::Nv4 { w, sf, alpha } = &lin.w else {
            bail!("{}: the nvfp4 GEMV needs nvfp4 weights", lin.name)
        };
        let perm = NV4_ROWS_ARE_PERMUTED_SO_A_SUBGROUP_READS_ITS_32_BLOCK_PAIRS_CONTIGUOUSLY;
        ensure!(x_off_words % 4 == 0, "{}: x offset {x_off_words} words is not vec4-aligned", lin.name);
        ensure!(lin.k % perm == 0, "{}: k={} is not a multiple of the {perm}-element input permutation", lin.name, lin.k);
        ensure!((1..=PAD_DRAFT_ROWS_MAX).contains(&rows), "{}: {rows} rows exceed the {PAD_DRAFT_ROWS_MAX}-row GEMV", lin.name);
        let sg = self.ctx.subgroup_width().unwrap_or(32) as usize;
        ensure!(sg == 32, "{}: the permuted input layout is laid out for 32-wide subgroups, not {sg}", lin.name);
        let subgroups = 256 / sg;
        let (pairs, bpairs) = (lin.n / 2, lin.k / 32);
        let mut spp = 1;
        while spp * 2 <= subgroups && pairs * spp / subgroups < 512 && bpairs / (spp * 2) >= sg {
            spp *= 2;
        }
        let grid = self.grid_rows(pairs.div_ceil(subgroups / spp));
        let p = self.uni(
            "vx-nv4-p",
            &Nv4Params {
                n_rows: lin.n as u32,
                k_blocks: (lin.k / 16) as u32,
                k_tiles: (lin.k / 16).div_ceil(4) as u32,
                groups_x: grid.0,
                alpha: *alpha,
                flags,
                x_off_vec4: (x_off_words / 4) as u32,
                sg_per_pair: spp as u32,
                pad0: 0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
        );
        Ok(self
            .pass(
                &self.src.gemv_nv4[rows - 1],
                "vx_gemv_nv4",
                &[(44, w), (45, sf), (46, x), (47, y), (48, &p)],
                grid,
            )?
            .named(lin))
    }

    pub(crate) fn norm_rows(&self, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, k: usize, x_off_words: usize, eps: f32) -> Result<Pass> {
        let perm = NV4_ROWS_ARE_PERMUTED_SO_A_SUBGROUP_READS_ITS_32_BLOCK_PAIRS_CONTIGUOUSLY;
        ensure!(k % perm == 0 && x_off_words % 4 == 0, "norm_rows: k={k} offset {x_off_words} do not tile the {perm}-element permutation");
        let p = self.uni(
            "vx-norm-p",
            &NormParams {
                kvec: (k / 8) as u32,
                x_off_vec4: (x_off_words / 4) as u32,
                eps,
                pad0: 0,
            },
        );
        self.pass(&self.src.vx, "vx_norm_rows", &[(50, x), (51, w), (52, y), (53, &p)], (rows as u32, 1, 1))
    }

    pub(crate) fn gemv(&self, x: &Buffer, lin: &Lin, y: &Buffer, x_off_words: usize, out_f32: bool) -> Result<Pass> {
        ensure!(lin.bias.is_none(), "the decode GEMV carries no bias");
        let w = match &lin.w {
            Weights::Bf16(w) => w,
            Weights::Nv4 { .. } => {
                return self.gemv_nv4(x, lin, y, x_off_words, 1, if out_f32 { NV4_OUT_F32 } else { 0 })
            }
        };
        let grid = self.grid_flat(lin.n.div_ceil(2), 1);
        let p = self.uni(
            "vx-gemv-p",
            &GemvBf16Params {
                n_rows: lin.n as u32,
                k_words: (lin.k / 2) as u32,
                groups_x: grid.0,
                out_f32: u32::from(out_f32),
                w_row_words: (lin.k / 2) as u32,
                x_off_words: x_off_words as u32,
                y_off_words: 0,
                pad0: 0,
                alpha: 1.0,
                pad1: 0,
                pad2: 0,
                pad3: 0,
            },
        );
        Ok(self
            .pass(
                &self.src.gemv,
                "q3w_gemv_bf16",
                &[(0, w), (1, x), (2, &p), (3, y)],
                grid,
            )?
            .named(lin))
    }

    pub(crate) fn rmsnorm(&self, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, hidden: usize, eps: f32) -> Result<Pass> {
        let p = self.uni(
            "vx-rms-p",
            &RmsParams {
                hidden: hidden as u32,
                batch: rows as u32,
                eps,
                words_per_row: (hidden / 2) as u32,
            },
        );
        self.pass(
            &self.src.rms,
            "rmsnorm_bf16",
            &[(0, x), (1, w), (2, y), (3, &p)],
            self.grid_rows(rows),
        )
    }

    pub(crate) fn rmsnorm_residual(
        &self,
        x: &Buffer,
        res: &Buffer,
        w: &Buffer,
        y: &Buffer,
        rows: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<Pass> {
        let p = self.uni(
            "vx-rmsres-p",
            &RmsParams {
                hidden: hidden as u32,
                batch: rows as u32,
                eps,
                words_per_row: (hidden / 2) as u32,
            },
        );
        self.pass(
            &self.src.rmsres,
            "rmsnorm_residual_bf16",
            &[(0, x), (1, res), (2, w), (3, y), (4, &p)],
            self.grid_rows(rows),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_kv(
        &self,
        qkv: &Buffer,
        cos: &Buffer,
        sin: &Buffer,
        pos: &Buffer,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
    ) -> Result<Pass> {
        let p = self.uni(
            "vx-rope-p",
            &RopeParams {
                rows: rows as u32,
                nh: nh as u32,
                nkv: nkv as u32,
                hd: hd as u32,
                qkv_stride: ((nh + 2 * nkv) * hd) as u32,
                ..Default::default()
            },
        );
        self.pass(
            &self.src.vx,
            "vx_rope_kv",
            &[(5, qkv), (6, cos), (7, sin), (8, pos), (9, q), (10, k), (11, v), (12, &p)],
            self.grid_flat(rows * (nh + 2 * nkv) * hd / 2, 256),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_prefill(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        pos: &Buffer,
        out: &Buffer,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
        scale: f32,
    ) -> Result<Pass> {
        let (src, entry) = match (hd, self.ctx.caps.subgroup) {
            (64, true) => (&self.src.attn_pf_64, "vx_attn_prefill_hd64"),
            (128, true) => (&self.src.attn_pf_128, "vx_attn_prefill_hd128"),
            (64, false) => (&self.src.attn_pf_64, "vx_attn_prefill_hd64_nosg"),
            (128, false) => (&self.src.attn_pf_128, "vx_attn_prefill_hd128_nosg"),
            _ => bail!("prefill attention is instantiated for head_dim 64 and 128, not {hd}"),
        };
        let p = self.uni(
            "vx-attnpf-p",
            &AttnParams {
                rows: rows as u32,
                nh: nh as u32,
                nkv: nkv as u32,
                hd: hd as u32,
                window: window as u32,
                scale,
                splits: 0,
                perm: 0,
            },
        );
        self.pass(
            src,
            entry,
            &[(13, q), (14, k), (15, v), (16, pos), (17, out), (18, &p)],
            (rows.div_ceil(64) as u32, nh as u32, 1),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attn_decode(
        &self,
        q: &Buffer,
        k: &Buffer,
        v: &Buffer,
        pos: &Buffer,
        out: &Buffer,
        scratch: &Buffer,
        rows: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        perm: bool,
    ) -> Result<[Pass; 2]> {
        ensure!(hd == 128 || hd == 64, "decode attention covers head_dim 64 and 128, not {hd}");
        ensure!(scratch.size() >= (rows * nh * DECODE_ATTENTION_SPLITS as usize * (hd + 2) * 4) as u64, "decode attention scratch holds no {rows} rows");
        let entry1 = if self.ctx.caps.subgroup {
            let sg = self.ctx.subgroup_width().unwrap_or(32) as usize;
            ensure!(sg >= 16 && 4 * sg >= hd, "decode attention holds two value words per lane: subgroup {sg} cannot cover head_dim {hd}");
            "vx_attn_decode1"
        } else {
            "vx_attn_decode1_nosg"
        };
        let p = self.uni(
            "vx-attnd-p",
            &AttnParams {
                rows: 1,
                nh: nh as u32,
                nkv: nkv as u32,
                hd: hd as u32,
                window: 0,
                scale: 1.0,
                splits: DECODE_ATTENTION_SPLITS,
                perm: u32::from(perm),
            },
        );
        Ok([
            self.pass(
                &self.src.vx,
                entry1,
                &[(13, q), (14, k), (15, v), (16, pos), (18, &p), (19, scratch)],
                (nh as u32, DECODE_ATTENTION_SPLITS, rows as u32),
            )?,
            self.pass(
                &self.src.vx,
                "vx_attn_decode2",
                &[(17, out), (18, &p), (19, scratch)],
                (nh as u32, rows as u32, 1),
            )?,
        ])
    }

    pub(crate) fn embed_add(
        &self,
        embed: &Buffer,
        tok: &Buffer,
        audio: &Buffer,
        pos: &Buffer,
        out: &Buffer,
        rows: usize,
        hidden: usize,
    ) -> Result<Pass> {
        let words = hidden / 2;
        let p = self.uni(
            "vx-embed-p",
            &EwParams {
                rows: rows as u32,
                words: words as u32,
                ..Default::default()
            },
        );
        self.pass(
            &self.src.vx,
            "vx_embed_add",
            &[(20, embed), (21, tok), (22, audio), (23, pos), (24, out), (25, &p)],
            self.grid_flat(rows * words, 256),
        )
    }

    pub(crate) fn silu_mul_rows(&self, gu: &Buffer, out: &Buffer, rows: usize, inter: usize, perm: bool) -> Result<Pass> {
        let words = inter / 2;
        let p = self.uni(
            "vx-silu-p",
            &EwParams {
                rows: rows as u32,
                words: words as u32,
                a: u32::from(perm),
                ..Default::default()
            },
        );
        self.pass(
            &self.src.vx,
            "vx_silu_mul_rows",
            &[(26, gu), (27, out), (28, &p)],
            self.grid_flat(rows * words, 256),
        )
    }

    pub(crate) fn im2col3(
        &self,
        src: &Buffer,
        dst: &Buffer,
        t_out: usize,
        channels: usize,
        stride: usize,
        pad_left: usize,
        t_in: usize,
    ) -> Result<Pass> {
        let words = channels / 2;
        let p = self.uni(
            "vx-im2col-p",
            &EwParams {
                rows: t_out as u32,
                words: words as u32,
                a: stride as u32,
                b: pad_left as u32,
                c: t_in as i32,
                ..Default::default()
            },
        );
        self.pass(
            &self.src.vx,
            "vx_im2col3",
            &[(29, src), (30, dst), (31, &p)],
            self.grid_flat(t_out * 3 * words, 256),
        )
    }

    pub(crate) fn add_inplace(&self, y: &Buffer, b: &Buffer, elems: usize) -> Result<Pass> {
        let words = elems / 2;
        let p = self.uni(
            "vx-add-p",
            &EwParams {
                rows: 1,
                words: words as u32,
                ..Default::default()
            },
        );
        self.pass(
            &self.src.vx,
            "vx_add_inplace",
            &[(32, y), (33, b), (34, &p)],
            self.grid_flat(words, 256),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn argmax(
        &self,
        logits: &Buffer,
        pval: &Buffer,
        pidx: &Buffer,
        pos: &Buffer,
        tok: &Buffer,
        ring: &Buffer,
        n: usize,
        rows: usize,
        incr_pos: bool,
    ) -> Result<[Pass; 2]> {
        ensure!((1..=PAD_DRAFT_ROWS_MAX).contains(&rows), "the accept pass holds at most {PAD_DRAFT_ROWS_MAX} rows, not {rows}");
        let p = self.uni(
            "vx-argmax-p",
            &ArgmaxParams {
                n: n as u32,
                nparts: ARGMAX_PARTS,
                ring_mask: (TOKEN_RING_SLOTS - 1) as i32,
                incr_pos: u32::from(incr_pos),
                rows: rows as u32,
                pad: crate::voxtral::STREAMING_PAD_TOKEN as i32,
                pad0: 0,
                pad1: 0,
            },
        );
        Ok([
            self.pass(
                &self.src.vx,
                "vx_argmax1",
                &[(35, logits), (36, pval), (37, pidx), (38, &p)],
                (ARGMAX_PARTS, rows as u32, 1),
            )?,
            self.pass(
                &self.src.vx,
                "vx_argmax2",
                &[(36, pval), (37, pidx), (38, &p), (39, pos), (40, tok), (41, ring)],
                (1, 1, 1),
            )?,
        ])
    }

    pub(crate) fn submit(&self, passes: &[Pass]) {
        dispatch::submit_pass_list(self.ctx, refs(passes).into_iter());
    }

    pub(crate) fn release_upload_staging(&self) -> Result<()> {
        let _ = UPLOAD_STAGING_IS_FREED_ONLY_AT_A_DEVICE_POLL_SO_THE_LOADER_FLUSHES_PER_LAYER;
        self.ctx.queue.submit([]);
        self.ctx.poll_blocking().map_err(err)
    }

    pub(crate) fn submit_profiled(&self, passes: &[Pass]) -> Result<()> {
        let labels: Vec<String> = passes.iter().map(|p| p.label.clone()).collect();
        dispatch::submit_profiled_slices(self.ctx, &refs(passes), &labels).map_err(err)
    }

    pub(crate) fn submit_repeated(&self, passes: &[Pass], times: usize) {
        let one = refs(passes);
        let all: Vec<PassRef<'_>> = (0..times).flat_map(|_| one.iter().copied()).collect();
        dispatch::submit_pass_list(self.ctx, all.into_iter());
    }

    pub(crate) fn read_u32(&self, b: &Buffer, len: usize) -> Result<Vec<u32>> {
        dispatch::read_back::<u32>(self.ctx, b, len).map_err(err)
    }

    #[cfg(test)]
    pub(crate) fn read_f32(&self, b: &Buffer, len: usize) -> Result<Vec<f32>> {
        dispatch::read_back::<f32>(self.ctx, b, len).map_err(err)
    }
}

struct EncLayer {
    attn_norm: Buffer,
    qkv: Lin,
    o: Lin,
    ffn_norm: Buffer,
    gate_up: Lin,
    down: Lin,
}

struct Encoder {
    conv1: Lin,
    conv2: Lin,
    layers: Vec<EncLayer>,
    norm: Buffer,
    proj1: Lin,
    proj2: Lin,
    cos: Buffer,
    sin: Buffer,
}

struct DecLayer {
    attn_norm: Buffer,
    qkv: Lin,
    o: Lin,
    ffn_norm: Buffer,
    gate_up: Lin,
    down: Lin,
    kcache: Buffer,
    vcache: Buffer,
}

struct Decoder {
    embed: Buffer,
    layers: Vec<DecLayer>,
    norm: Buffer,
    lm_head: Lin,
    cos: Buffer,
    sin: Buffer,
}

pub struct VoxtralWgpu {
    kit: Kit,
    cfg: VoxtralConfig,
    quant: VoxtralQuant,
    vocab: TekkenVocab,
    mel: VoxtralMel,
    enc: Encoder,
    dec: Decoder,
    capacity: usize,
    audio: Buffer,
    pos: Buffer,
    zero_pos: Buffer,
    tok: Buffer,
    ring: Buffer,
    seed_passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    step_passes: Vec<Pass>,
    step_rows: usize,
    replays: usize,
    seeded: bool,
    validated: bool,
}

pub(crate) fn tensor_bits(t: &Tensor) -> Result<Vec<u16>> {
    let v: Vec<bf16> = t.to_dtype(DType::BF16)?.flatten_all()?.to_vec1()?;
    Ok(v.into_iter().map(|x| x.to_bits()).collect())
}

fn tensor_f32(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
}

fn get(w: &HfLayouts, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = w.get(name, DType::BF16).with_context(|| format!("load {name}"))?;
    ensure!(t.dims() == shape, "{name}: shape {:?} != {shape:?}", t.dims());
    Ok(t)
}

fn bits(w: &HfLayouts, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    tensor_bits(&get(w, name, shape)?)
}

fn f32s(w: &HfLayouts, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
    tensor_f32(&get(w, name, shape)?)
}

fn conv_taps_to_gemm_rows(w: &[u16], out_ch: usize, in_ch: usize) -> Vec<u16> {
    let mut out = vec![0u16; out_ch * in_ch * 3];
    for co in 0..out_ch {
        for ci in 0..in_ch {
            for tap in 0..3 {
                out[co * in_ch * 3 + tap * in_ch + ci] = w[(co * in_ch + ci) * 3 + tap];
            }
        }
    }
    out
}

fn concat_bits(parts: &[&[u16]]) -> Vec<u16> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

pub(crate) fn rope_buffers(kit: &Kit, label: &str, hd: usize, theta: f32, rows: usize) -> (Buffer, Buffer) {
    let half = hd / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / theta.powf((i as f32 * 2.0) / (hd as f32)))
        .collect();
    let (cos, sin) = nv_layers::rope::build_rope_tables_f32(&inv_freq, rows);
    (
        kit.upload_f32(&format!("{label}-cos"), &cos),
        kit.upload_f32(&format!("{label}-sin"), &sin),
    )
}

impl VoxtralWgpu {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        Self::load_dir_with_adapter(dir, None)
    }

    pub fn load_dir_with_adapter(dir: &Path, adapter_dir: Option<&Path>) -> Result<Self> {
        let cpu = Device::Cpu;
        let adapter = match adapter_dir {
            Some(d) => Some(
                nv_weights::lora_adapter::LoraAdapter::load(d, &cpu)
                    .with_context(|| format!("load adapter {}", d.display()))?,
            ),
            None => None,
        };
        let cfg = VoxtralConfig::from_json_str(
            &std::fs::read_to_string(dir.join(HF_CONFIG_FILE))
                .with_context(|| format!("read {}", dir.join(HF_CONFIG_FILE).display()))?,
        )?;
        let vocab = TekkenVocab::from_json_str(
            &std::fs::read_to_string(dir.join(TEKKEN_FILE))
                .with_context(|| format!("read {}", dir.join(TEKKEN_FILE).display()))?,
        )?;
        let weights_path = dir.join(HF_WEIGHTS_FILE);
        let weights = WeightLoader::open_file(&weights_path, &cpu)
            .with_context(|| format!("open {}", weights_path.display()))?;
        Self::from_loader(cfg, vocab, &weights, adapter.as_ref())
    }

    pub fn from_loader(
        cfg: VoxtralConfig,
        vocab: TekkenVocab,
        loader: &WeightLoader,
        adapter: Option<&nv_weights::lora_adapter::LoraAdapter>,
    ) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow!("no wgpu adapter: {e}"))?;
        let quant = wgpu_quant_from_env()?;
        let kit = Kit::new(ctx);
        let w = &HfLayouts(loader);
        let a = &cfg.audio_config;
        let t = &cfg.text_config;
        ensure!(a.head_dim == 64 && t.head_dim == 128, "attention kernels are instantiated for the shipped head dims");
        let _ = ENCODER_CONV_STEM_IS_TWO_IM2COL_GEMMS;
        let conv1 = kit.lin(
            "vx-enc-conv1",
            &conv_taps_to_gemm_rows(
                &bits(w, "model.audio_tower.embedder.conv1.weight", &[a.hidden_size, a.num_mel_bins, 3])?,
                a.hidden_size,
                a.num_mel_bins,
            ),
            a.hidden_size,
            3 * a.num_mel_bins,
            Some(&f32s(w, "model.audio_tower.embedder.conv1.bias", &[a.hidden_size])?),
        )?;
        let conv2 = kit.lin(
            "vx-enc-conv2",
            &conv_taps_to_gemm_rows(
                &bits(w, "model.audio_tower.embedder.conv2.weight", &[a.hidden_size, a.hidden_size, 3])?,
                a.hidden_size,
                a.hidden_size,
            ),
            a.hidden_size,
            3 * a.hidden_size,
            Some(&f32s(w, "model.audio_tower.embedder.conv2.bias", &[a.hidden_size])?),
        )?;
        let qkv_rows = a.num_attention_heads * a.head_dim;
        let mut enc_layers = Vec::with_capacity(a.num_hidden_layers);
        for i in 0..a.num_hidden_layers {
            let p = format!("model.audio_tower.layers.{i}");
            let q = bits(w, &format!("{p}.self_attn.q_proj.weight"), &[qkv_rows, a.hidden_size])?;
            let k = bits(w, &format!("{p}.self_attn.k_proj.weight"), &[qkv_rows, a.hidden_size])?;
            let v = bits(w, &format!("{p}.self_attn.v_proj.weight"), &[qkv_rows, a.hidden_size])?;
            let mut qkv_bias = f32s(w, &format!("{p}.self_attn.q_proj.bias"), &[qkv_rows])?;
            qkv_bias.extend(std::iter::repeat_n(0.0, qkv_rows));
            qkv_bias.extend(f32s(w, &format!("{p}.self_attn.v_proj.bias"), &[qkv_rows])?);
            let gate = bits(w, &format!("{p}.mlp.gate_proj.weight"), &[a.intermediate_size, a.hidden_size])?;
            let up = bits(w, &format!("{p}.mlp.up_proj.weight"), &[a.intermediate_size, a.hidden_size])?;
            enc_layers.push(EncLayer {
                attn_norm: kit.upload_bf16(
                    "vx-enc-attn-norm",
                    &bits(w, &format!("{p}.self_attn_layer_norm.weight"), &[a.hidden_size])?,
                ),
                qkv: kit.lin("vx-enc-qkv", &concat_bits(&[&q, &k, &v]), 3 * qkv_rows, a.hidden_size, Some(&qkv_bias))?,
                o: kit.lin(
                    "vx-enc-o",
                    &bits(w, &format!("{p}.self_attn.o_proj.weight"), &[a.hidden_size, qkv_rows])?,
                    a.hidden_size,
                    qkv_rows,
                    Some(&f32s(w, &format!("{p}.self_attn.o_proj.bias"), &[a.hidden_size])?),
                )?,
                ffn_norm: kit.upload_bf16(
                    "vx-enc-ffn-norm",
                    &bits(w, &format!("{p}.final_layer_norm.weight"), &[a.hidden_size])?,
                ),
                gate_up: kit.lin("vx-enc-gu", &concat_bits(&[&gate, &up]), 2 * a.intermediate_size, a.hidden_size, None)?,
                down: kit.lin(
                    "vx-enc-down",
                    &bits(w, &format!("{p}.mlp.down_proj.weight"), &[a.hidden_size, a.intermediate_size])?,
                    a.hidden_size,
                    a.intermediate_size,
                    Some(&f32s(w, &format!("{p}.mlp.down_proj.bias"), &[a.hidden_size])?),
                )?,
            });
            kit.release_upload_staging()?;
        }
        let proj_in = a.hidden_size * cfg.downsample_factor;
        let (enc_cos, enc_sin) = rope_buffers(
            &kit,
            "vx-enc-rope",
            a.head_dim,
            a.rope_parameters.rope_theta,
            ENCODER_ROPE_ROWS_COVER_35_MINUTES_OF_50HZ_FRAMES,
        );
        let enc = Encoder {
            conv1,
            conv2,
            layers: enc_layers,
            norm: kit.upload_bf16("vx-enc-norm", &bits(w, "model.audio_tower.norm.weight", &[a.hidden_size])?),
            proj1: kit.lin(
                "vx-proj1",
                &bits(w, "model.multi_modal_projector.linear_1.weight", &[t.hidden_size, proj_in])?,
                t.hidden_size,
                proj_in,
                None,
            )?,
            proj2: kit.lin(
                "vx-proj2",
                &bits(w, "model.multi_modal_projector.linear_2.weight", &[t.hidden_size, t.hidden_size])?,
                t.hidden_size,
                t.hidden_size,
                None,
            )?,
            cos: enc_cos,
            sin: enc_sin,
        };

        let prefix_len = 1 + vocab.streaming_left_pad_tokens + vocab.num_delay_tokens();
        let max_audio_tokens = t.sliding_window.unwrap_or(DECODER_ROPE_ROWS).min(DECODER_ROPE_ROWS) - prefix_len;
        let capacity = max_audio_tokens + prefix_len + 1 + 2 * PAD_DRAFT_ROWS_MAX;
        let step_rows = spec_rows_from_env(quant)?;
        let cpu = Device::Cpu;
        let t_cond = Tensor::from_vec(time_embedding(vocab.num_delay_tokens(), t.hidden_size), (1, t.hidden_size), &cpu)?;
        let q_rows = t.num_attention_heads * t.head_dim;
        let kv_rows = t.num_key_value_heads * t.head_dim;
        let kv_words = capacity * kv_rows * 2;
        let mut dec_layers = Vec::with_capacity(t.num_hidden_layers);
        for i in 0..t.num_hidden_layers {
            let p = format!("model.language_model.layers.{i}");
            let ada1 = get(w, &format!("{p}.ada_rms_norm.linear1.weight"), &[32, t.hidden_size])?.to_dtype(DType::F32)?;
            let ada2 = get(w, &format!("{p}.ada_rms_norm.linear2.weight"), &[t.hidden_size, 32])?.to_dtype(DType::F32)?;
            let _ = crate::voxtral::THE_DELAY_SCALE_FOLDS_INTO_THE_POST_ATTENTION_NORM_WEIGHT;
            let hidden = t_cond.matmul(&ada1.t()?)?.gelu_erf()?;
            let ada_scale = (hidden.matmul(&ada2.t()?)? + 1.0)?.reshape(t.hidden_size)?;
            let ffn_norm_w = get(w, &format!("{p}.post_attention_layernorm.weight"), &[t.hidden_size])?;
            let ffn_norm = (ffn_norm_w.to_dtype(DType::F32)? * ada_scale)?;
            let _ = crate::voxtral::Q_ROWS_CARRY_THE_SOFTMAX_SCALE_SO_EVERY_ATTENTION_PATH_SEES_SCALE_ONE;
            let q_scaled = (get(w, &format!("{p}.self_attn.q_proj.weight"), &[q_rows, t.hidden_size])?.to_dtype(DType::F32)?
                * (1.0 / (t.head_dim as f64).sqrt()))?
            .to_dtype(DType::BF16)?;
            let qkv = Tensor::cat(
                &[
                    q_scaled,
                    get(w, &format!("{p}.self_attn.k_proj.weight"), &[kv_rows, t.hidden_size])?,
                    get(w, &format!("{p}.self_attn.v_proj.weight"), &[kv_rows, t.hidden_size])?,
                ],
                0,
            )?;
            let gate_up = Tensor::cat(
                &[
                    get(w, &format!("{p}.mlp.gate_proj.weight"), &[t.intermediate_size, t.hidden_size])?,
                    get(w, &format!("{p}.mlp.up_proj.weight"), &[t.intermediate_size, t.hidden_size])?,
                ],
                0,
            )?;
            let qkv = merge_adapter_rows(
                qkv,
                &format!("{p}.self_attn"),
                &[("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)],
                adapter,
            )?;
            let o = merge_adapter_rows(
                get(w, &format!("{p}.self_attn.o_proj.weight"), &[t.hidden_size, q_rows])?,
                &format!("{p}.self_attn"),
                &[("o_proj", t.hidden_size)],
                adapter,
            )?;
            let gate_up = merge_adapter_rows(
                gate_up,
                &format!("{p}.mlp"),
                &[("gate_proj", t.intermediate_size), ("up_proj", t.intermediate_size)],
                adapter,
            )?;
            let down = merge_adapter_rows(
                get(w, &format!("{p}.mlp.down_proj.weight"), &[t.hidden_size, t.intermediate_size])?,
                &format!("{p}.mlp"),
                &[("down_proj", t.hidden_size)],
                adapter,
            )?;
            dec_layers.push(DecLayer {
                attn_norm: kit.upload_bf16(
                    "vx-dec-attn-norm",
                    &bits(w, &format!("{p}.input_layernorm.weight"), &[t.hidden_size])?,
                ),
                qkv: kit.lin_quant("vx-dec-qkv", &tensor_bits(&qkv)?, q_rows + 2 * kv_rows, t.hidden_size, quant)?,
                o: kit.lin_quant("vx-dec-o", &tensor_bits(&o)?, t.hidden_size, q_rows, quant)?,
                ffn_norm: kit.upload_bf16("vx-dec-ffn-norm", &tensor_bits(&ffn_norm)?),
                gate_up: kit.lin_quant("vx-dec-gu", &tensor_bits(&gate_up)?, 2 * t.intermediate_size, t.hidden_size, quant)?,
                down: kit.lin_quant("vx-dec-down", &tensor_bits(&down)?, t.hidden_size, t.intermediate_size, quant)?,
                kcache: kit.zeroed("vx-kcache", kv_words),
                vcache: kit.zeroed("vx-vcache", kv_words),
            });
            kit.release_upload_staging()?;
        }
        let embed_bits = bits(w, "model.language_model.embed_tokens.weight", &[t.vocab_size, t.hidden_size])?;
        let embed_bytes = (embed_bits.len() * 2) as u64;
        ensure!(
            embed_bytes <= kit.ctx.caps.max_storage_buffer_binding_size,
            "the {embed_bytes}-byte embedding table exceeds the adapter's storage binding limit {}",
            kit.ctx.caps.max_storage_buffer_binding_size
        );
        let embed = kit.upload_bf16("vx-embed", &embed_bits);
        kit.release_upload_staging()?;
        let lm_head = if w.has("lm_head.weight") {
            kit.lin_quant(
                "vx-lm-head",
                &bits(w, "lm_head.weight", &[t.vocab_size, t.hidden_size])?,
                t.vocab_size,
                t.hidden_size,
                quant,
            )?
        } else if quant == VoxtralQuant::Nvfp4 {
            let _ = crate::voxtral::THE_LM_HEAD_READS_A_QUARTER_OF_EACH_STEP_SO_IT_QUANTIZES_WITH_THE_DECODER;
            kit.lin_nv4("vx-lm-head", &embed_bits, t.vocab_size, t.hidden_size)?
        } else {
            Lin {
                w: Weights::Bf16(embed.clone()),
                bias: None,
                n: t.vocab_size,
                k: t.hidden_size,
                name: "vx-embed-lm-head",
            }
        };
        drop(embed_bits);
        let (dec_cos, dec_sin) = rope_buffers(&kit, "vx-dec-rope", t.head_dim, t.rope_parameters.rope_theta, DECODER_ROPE_ROWS);
        let dec = Decoder {
            embed,
            layers: dec_layers,
            norm: kit.upload_bf16("vx-dec-norm", &bits(w, "model.language_model.norm.weight", &[t.hidden_size])?),
            lm_head,
            cos: dec_cos,
            sin: dec_sin,
        };
        let mel = VoxtralMel::new(a.num_mel_bins);
        let mut m = Self {
            audio: kit.zeroed("vx-audio", capacity * t.hidden_size * 2),
            pos: kit.upload_i32("vx-pos", &[0]),
            zero_pos: kit.upload_i32("vx-zero-pos", &[0]),
            tok: kit.upload_u32("vx-tok", &pad_draft_tokens(0)),
            ring: kit.zeroed("vx-ring", (TOKEN_RING_SLOTS + 1) * 4),
            kit,
            cfg,
            quant,
            vocab,
            mel,
            enc,
            dec,
            capacity,
            seed_passes: Vec::new(),
            pf_passes: Vec::new(),
            step_passes: Vec::new(),
            step_rows,
            replays: 0,
            seeded: false,
            validated: false,
        };
        m.build_decoder_passes()?;
        Ok(m)
    }

    fn build_decoder_passes(&mut self) -> Result<()> {
        let _ = THE_STEP_READS_ITS_TOKEN_AND_POSITION_FROM_DEVICE_BUFFERS_SO_A_BATCH_OF_STEPS_NEEDS_NO_HOST_SYNC;
        let kit = &self.kit;
        let t = &self.cfg.text_config;
        let (nh, nkv, hd, hidden, inter) = (
            t.num_attention_heads,
            t.num_key_value_heads,
            t.head_dim,
            t.hidden_size,
            t.intermediate_size,
        );
        let eps = t.rms_norm_eps as f32;
        let _ = THE_PREFIX_ROWS_THAT_SEE_ONLY_THE_LEFT_PAD_ARE_SEEDED_ONCE_PER_MODEL;
        let (s, m_pf) = (self.seed_rows(), self.prefix_len() - self.seed_rows());
        let nv4 = self.quant == VoxtralQuant::Nvfp4;
        let m_step = self.step_rows;
        ensure!(!nv4 || m_pf <= PAD_DRAFT_ROWS_MAX, "the {m_pf} unseeded prefix rows exceed the {PAD_DRAFT_ROWS_MAX}-row block");
        let m = s.max(m_pf).max(m_step);
        let m_blk = m_pf.max(m_step);
        let prefix: Vec<i32> = self.prefix_tokens().iter().map(|&x| x as i32).collect();
        let seed_tok = kit.upload_i32("vx-seed-tok", &prefix[..s]);
        let pf_tok = kit.upload_i32("vx-pf-tok", &prefix[s..]);
        let pf_pos = kit.upload_i32("vx-pf-pos", &[s as i32]);
        let pf_last_pos = kit.upload_i32("vx-pf-last-pos", &[(s + m_pf - 1) as i32]);
        let x = kit.zeroed("vx-x", m * hidden * 2);
        let h = kit.zeroed("vx-h", m * hidden * 2);
        let qkv = kit.zeroed("vx-qkv", m * (nh + 2 * nkv) * hd * 4);
        let q = kit.zeroed("vx-q", m * nh * hd * 4);
        let attn = kit.zeroed("vx-attn", m * nh * hd * 2);
        let ao = kit.zeroed("vx-ao", m * hidden * 2);
        let gu = kit.zeroed("vx-gu", m * 2 * inter * 2);
        let act = kit.zeroed("vx-act", m * inter * 2);
        let hn = kit.zeroed("vx-hn", m * hidden * 2);
        let xn = kit.zeroed("vx-xn", m * hidden * 2);
        let logits = kit.zeroed("vx-logits", m_blk * t.vocab_size * 4);
        let pval = kit.zeroed("vx-am-val", m_blk * ARGMAX_PARTS as usize * 4);
        let pidx = kit.zeroed("vx-am-idx", m_blk * ARGMAX_PARTS as usize * 4);
        let scratch = kit.zeroed("vx-fd-scratch", m_blk * nh * DECODE_ATTENTION_SPLITS as usize * (hd + 2) * 4);
        let part = kit.split_part(&self.dec.layers.iter().flat_map(|l| [(&l.qkv, m), (&l.o, m), (&l.gate_up, m), (&l.down, m)]).collect::<Vec<_>>());

        let gemm_rows = |m: usize, tok: &Buffer, pos: &Buffer| -> Result<Vec<Pass>> {
            let mut pf = vec![kit.embed_add(&self.dec.embed, tok, &self.audio, pos, &x, m, hidden)?];
            for l in &self.dec.layers {
                pf.push(kit.rmsnorm(&x, &l.attn_norm, &h, m, hidden, eps)?);
                pf.extend(kit.gemm_split(&h, &l.qkv, &qkv, m, GEMM_OUT_F32, &part, kit.split_k_for(&l.qkv, m))?);
                pf.push(kit.rope_kv(&qkv, &self.dec.cos, &self.dec.sin, pos, &q, &l.kcache, &l.vcache, m, nh, nkv, hd)?);
                pf.push(kit.attn_prefill(&q, &l.kcache, &l.vcache, pos, &attn, m, nh, nkv, hd, 0, 1.0)?);
                pf.extend(kit.gemm_split(&attn, &l.o, &ao, m, 0, &part, kit.split_k_for(&l.o, m))?);
                pf.push(kit.rmsnorm_residual(&ao, &x, &l.ffn_norm, &h, m, hidden, eps)?);
                pf.extend(kit.gemm_split(&h, &l.gate_up, &gu, m, 0, &part, kit.split_k_for(&l.gate_up, m))?);
                pf.push(kit.silu_mul_rows(&gu, &act, m, inter, false)?);
                pf.extend(kit.gemm_split(&act, &l.down, &x, m, GEMM_ACCUM, &part, kit.split_k_for(&l.down, m))?);
            }
            Ok(pf)
        };
        let nv4_rows = |m: usize, tok: &Buffer, pos: &Buffer| -> Result<Vec<Pass>> {
            let _ = THE_NVFP4_STEP_FUSES_RMSNORM_AND_RESIDUAL_ADDS_INTO_ITS_GEMVS;
            let _ = THE_NVFP4_STEP_DRAFTS_PADS_LIKE_THE_CUDA_GRAPH_AND_STAYS_BITWISE_PER_ROW;
            let _ = E2M1_DECODES_THROUGH_THE_F16_SUBNORMAL_LADDER;
            let mut st = vec![kit.embed_add(&self.dec.embed, tok, &self.audio, pos, &x, m, hidden)?];
            for l in &self.dec.layers {
                st.push(kit.norm_rows(&x, &l.attn_norm, &xn, m, hidden, 0, eps)?);
                st.push(kit.gemv_nv4(&xn, &l.qkv, &qkv, 0, m, NV4_OUT_F32)?);
                st.push(kit.rope_kv(&qkv, &self.dec.cos, &self.dec.sin, pos, &q, &l.kcache, &l.vcache, m, nh, nkv, hd)?);
                st.extend(kit.attn_decode(&q, &l.kcache, &l.vcache, pos, &attn, &scratch, m, nh, nkv, hd, true)?);
                st.push(kit.gemv_nv4(&attn, &l.o, &x, 0, m, NV4_ACCUM)?);
                st.push(kit.norm_rows(&x, &l.ffn_norm, &xn, m, hidden, 0, eps)?);
                st.push(kit.gemv_nv4(&xn, &l.gate_up, &gu, 0, m, 0)?);
                st.push(kit.silu_mul_rows(&gu, &act, m, inter, true)?);
                st.push(kit.gemv_nv4(&act, &l.down, &x, 0, m, NV4_ACCUM)?);
            }
            Ok(st)
        };
        let seed = gemm_rows(s, &seed_tok, &self.zero_pos)?;
        let mut pf = if nv4 {
            let mut pf = nv4_rows(m_pf, &pf_tok, &pf_pos)?;
            pf.push(kit.norm_rows(&x, &self.dec.norm, &xn, 1, hidden, (m_pf - 1) * hidden / 2, eps)?);
            pf.push(kit.gemv_nv4(&xn, &self.dec.lm_head, &logits, 0, 1, NV4_OUT_F32)?);
            pf
        } else {
            let mut pf = gemm_rows(m_pf, &pf_tok, &pf_pos)?;
            pf.push(kit.rmsnorm(&x, &self.dec.norm, &hn, m_pf, hidden, eps)?);
            pf.push(kit.gemv(&hn, &self.dec.lm_head, &logits, (m_pf - 1) * hidden / 2, true)?);
            pf
        };
        pf.extend(kit.argmax(&logits, &pval, &pidx, &pf_last_pos, &self.tok, &self.ring, t.vocab_size, 1, false)?);

        let mut st = if nv4 {
            let mut st = nv4_rows(m_step, &self.tok, &self.pos)?;
            st.push(kit.norm_rows(&x, &self.dec.norm, &xn, m_step, hidden, 0, eps)?);
            st.push(kit.gemv_nv4(&xn, &self.dec.lm_head, &logits, 0, m_step, NV4_OUT_F32)?);
            st
        } else {
            let mut st = vec![kit.embed_add(&self.dec.embed, &self.tok, &self.audio, &self.pos, &x, 1, hidden)?];
            for l in &self.dec.layers {
                st.push(kit.rmsnorm(&x, &l.attn_norm, &h, 1, hidden, eps)?);
                st.push(kit.gemv(&h, &l.qkv, &qkv, 0, true)?);
                st.push(kit.rope_kv(&qkv, &self.dec.cos, &self.dec.sin, &self.pos, &q, &l.kcache, &l.vcache, 1, nh, nkv, hd)?);
                st.extend(kit.attn_decode(&q, &l.kcache, &l.vcache, &self.pos, &attn, &scratch, 1, nh, nkv, hd, false)?);
                st.push(kit.gemv(&attn, &l.o, &ao, 0, false)?);
                st.push(kit.rmsnorm_residual(&ao, &x, &l.ffn_norm, &h, 1, hidden, eps)?);
                st.push(kit.gemv(&h, &l.gate_up, &gu, 0, false)?);
                st.push(kit.silu_mul_rows(&gu, &act, 1, inter, false)?);
                st.push(kit.gemv(&act, &l.down, &ao, 0, false)?);
                st.push(kit.add_inplace(&x, &ao, hidden)?);
            }
            st.push(kit.rmsnorm(&x, &self.dec.norm, &hn, 1, hidden, eps)?);
            st.push(kit.gemv(&hn, &self.dec.lm_head, &logits, 0, true)?);
            st
        };
        st.extend(kit.argmax(&logits, &pval, &pidx, &self.pos, &self.tok, &self.ring, t.vocab_size, m_step, true)?);
        self.seed_passes = seed;
        self.pf_passes = pf;
        self.step_passes = st;
        Ok(())
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.cfg
    }

    pub fn quant(&self) -> VoxtralQuant {
        self.quant
    }

    pub fn vocab(&self) -> &TekkenVocab {
        &self.vocab
    }

    pub fn prefix_tokens(&self) -> Vec<u32> {
        let mut t = vec![self.cfg.text_config.bos_token_id];
        t.extend(std::iter::repeat_n(
            crate::voxtral::STREAMING_PAD_TOKEN,
            self.vocab.streaming_left_pad_tokens + self.vocab.num_delay_tokens(),
        ));
        t
    }

    pub fn prefix_len(&self) -> usize {
        1 + self.vocab.streaming_left_pad_tokens + self.vocab.num_delay_tokens()
    }

    pub fn max_audio_tokens(&self) -> usize {
        self.capacity - self.prefix_len() - 1 - 2 * PAD_DRAFT_ROWS_MAX
    }

    pub fn step_rows(&self) -> usize {
        self.step_rows
    }

    pub fn replays(&self) -> usize {
        self.replays
    }

    pub fn seed_rows(&self) -> usize {
        let clean_frames = (self.vocab.streaming_left_pad_tokens * SAMPLES_PER_TOKEN - N_FFT / 2) / HOP_LENGTH + 1;
        clean_frames / (2 * self.cfg.downsample_factor)
    }

    pub fn step_pass_count(&self) -> usize {
        self.step_passes.len()
    }

    pub fn log_mel(&self, audio_16k_mono: &[f32]) -> (Vec<f32>, usize) {
        let padded = pad_audio_for_streaming(
            audio_16k_mono,
            self.vocab.streaming_left_pad_tokens,
            self.vocab.right_pad_tokens(),
        );
        self.mel.log_mel(&padded)
    }

    pub fn encode(&mut self, audio_16k_mono: &[f32]) -> Result<usize> {
        let _ = ENCODER_PASSES_ARE_REBUILT_PER_CLIP_BECAUSE_ITS_ACTIVATIONS_ARE_SIZED_TO_THE_CLIP;
        let (mel, frames) = self.log_mel(audio_16k_mono);
        let a = &self.cfg.audio_config;
        let ds = self.cfg.downsample_factor;
        let n_mels = a.num_mel_bins;
        ensure!(frames >= 2 && frames % (2 * ds) == 0, "{frames} mel frames do not tile into 80 ms tokens");
        let t2 = frames / 2;
        let n_tokens = t2 / ds;
        ensure!(n_tokens <= self.capacity, "clip of {n_tokens} tokens exceeds the decoder capacity {}", self.capacity);
        ensure!(
            t2 <= ENCODER_ROPE_ROWS_COVER_35_MINUTES_OF_50HZ_FRAMES,
            "{t2} encoder frames exceed the rope table"
        );
        let mut c1in = vec![0u16; frames * 3 * n_mels];
        for t in 0..frames {
            for tap in 0..3 {
                if t + tap < 2 {
                    continue;
                }
                let src = t + tap - 2;
                for ci in 0..n_mels {
                    c1in[t * 3 * n_mels + tap * n_mels + ci] = bf16::from_f32(mel[ci * frames + src]).to_bits();
                }
            }
        }
        let kit = &self.kit;
        let (nh, hd, hidden, inter) = (a.num_attention_heads, a.head_dim, a.hidden_size, a.intermediate_size);
        let eps = a.rms_norm_eps as f32;
        let qkv_rows = nh * hd;
        let c1in = kit.upload_bf16("vx-enc-c1in", &c1in);
        let c1out = kit.zeroed("vx-enc-c1out", frames * hidden * 2);
        let c2in = kit.zeroed("vx-enc-c2in", t2 * 3 * hidden * 2);
        let x = kit.zeroed("vx-enc-x", t2 * hidden * 2);
        let h = kit.zeroed("vx-enc-h", t2 * hidden * 2);
        let qkv = kit.zeroed("vx-enc-qkv", t2 * 3 * qkv_rows * 4);
        let q = kit.zeroed("vx-enc-q", t2 * qkv_rows * 4);
        let k = kit.zeroed("vx-enc-k", t2 * qkv_rows * 2);
        let v = kit.zeroed("vx-enc-v", t2 * qkv_rows * 2);
        let attn = kit.zeroed("vx-enc-attn", t2 * qkv_rows * 2);
        let gu = kit.zeroed("vx-enc-gu", t2 * 2 * inter * 2);
        let act = kit.zeroed("vx-enc-act", t2 * inter * 2);
        let p1 = kit.zeroed("vx-enc-p1", n_tokens * self.cfg.text_config.hidden_size * 2);
        let scope = kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut shapes = vec![(&self.enc.conv1, frames), (&self.enc.conv2, t2), (&self.enc.proj1, n_tokens), (&self.enc.proj2, n_tokens)];
        shapes.extend(self.enc.layers.iter().flat_map(|l| [(&l.qkv, t2), (&l.o, t2), (&l.gate_up, t2), (&l.down, t2)]));
        let part = kit.split_part(&shapes);
        let gemm = |x: &Buffer, lin: &Lin, y: &Buffer, m: usize, flags: u32| kit.gemm_split(x, lin, y, m, flags, &part, kit.split_k_for(lin, m));
        let mut passes = Vec::with_capacity(4 + 9 * self.enc.layers.len());
        passes.extend(gemm(&c1in, &self.enc.conv1, &c1out, frames, GEMM_GELU)?);
        passes.push(kit.im2col3(&c1out, &c2in, t2, hidden, 2, 1, frames)?);
        passes.extend(gemm(&c2in, &self.enc.conv2, &x, t2, GEMM_GELU)?);
        let scale = 1.0 / (hd as f32).sqrt();
        let window = a.sliding_window.unwrap_or(0);
        for l in &self.enc.layers {
            passes.push(kit.rmsnorm(&x, &l.attn_norm, &h, t2, hidden, eps)?);
            passes.extend(gemm(&h, &l.qkv, &qkv, t2, GEMM_OUT_F32)?);
            passes.push(kit.rope_kv(&qkv, &self.enc.cos, &self.enc.sin, &self.zero_pos, &q, &k, &v, t2, nh, nh, hd)?);
            passes.push(kit.attn_prefill(&q, &k, &v, &self.zero_pos, &attn, t2, nh, nh, hd, window, scale)?);
            passes.extend(gemm(&attn, &l.o, &x, t2, GEMM_ACCUM)?);
            passes.push(kit.rmsnorm(&x, &l.ffn_norm, &h, t2, hidden, eps)?);
            passes.extend(gemm(&h, &l.gate_up, &gu, t2, 0)?);
            passes.push(kit.silu_mul_rows(&gu, &act, t2, inter, false)?);
            passes.extend(gemm(&act, &l.down, &x, t2, GEMM_ACCUM)?);
        }
        passes.push(kit.rmsnorm(&x, &self.enc.norm, &h, t2, hidden, eps)?);
        passes.extend(gemm(&h, &self.enc.proj1, &p1, n_tokens, GEMM_GELU)?);
        passes.extend(gemm(&p1, &self.enc.proj2, &self.audio, n_tokens, 0)?);
        if dispatch::profile::enabled() && kit.ctx.caps.timestamp_query {
            dispatch::profile::reset();
            kit.submit_profiled(&passes)?;
            eprintln!("{}", dispatch::profile::table(&format!("voxtral wgpu encoder ({t2} frames)"), passes.len()));
        } else {
            kit.submit(&passes);
        }
        if let Some(e) = dispatch::pop_scope(scope) {
            bail!("voxtral wgpu encoder validation: {e}");
        }
        Ok(n_tokens)
    }

    pub fn audio_embeds(&self, n_tokens: usize) -> Result<Vec<f32>> {
        let hidden = self.cfg.text_config.hidden_size;
        let words = self.kit.read_u32(&self.audio, n_tokens * hidden / 2)?;
        Ok(words
            .iter()
            .flat_map(|w| [f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000)])
            .collect())
    }

    pub fn decode_stream(&mut self, n_audio: usize) -> Result<Vec<u32>> {
        let profiling = dispatch::profile::enabled() && self.kit.ctx.caps.timestamp_query;
        let out = self.decode_stream_inner(n_audio, profiling)?;
        if profiling {
            eprintln!(
                "{}",
                dispatch::profile::table(
                    &format!("voxtral wgpu {} decode step", self.quant.label()),
                    self.step_passes.len()
                )
            );
        }
        Ok(out)
    }

    fn decode_stream_inner(&mut self, n_audio: usize, profiling: bool) -> Result<Vec<u32>> {
        let prefix_len = self.prefix_len();
        if n_audio <= prefix_len {
            bail!("audio yields {n_audio} 80 ms tokens, not more than the {prefix_len}-token streaming prefix");
        }
        if n_audio > self.max_audio_tokens() + prefix_len {
            bail!(
                "audio yields {n_audio} tokens; the decoder serves at most {} per call (split the clip)",
                self.max_audio_tokens()
            );
        }
        let eos = self.cfg.text_config.eos_token_id;
        let first = self.prefill()?;
        let kit = &self.kit;
        let mut out = Vec::with_capacity(n_audio - prefix_len);
        if profiling {
            dispatch::profile::reset();
        }
        if first == eos {
            return Ok(out);
        }
        out.push(first);
        kit.ctx
            .queue
            .write_buffer(&self.pos, 0, bytemuck::bytes_of(&(prefix_len as i32)));
        let mut next = prefix_len;
        let last_input = n_audio - 1;
        let m = self.step_rows;
        self.replays = 0;
        while next < last_input {
            let remaining = last_input - next;
            let k = (remaining / m)
                .clamp(1, REPLAYS_PER_BATCH_TRADE_EOS_LATENCY_AGAINST_HOST_ROUND_TRIPS)
                .min(TOKEN_RING_SLOTS / m);
            let scope = (!self.validated).then(|| kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation));
            if profiling {
                for _ in 0..k {
                    kit.submit_profiled(&self.step_passes)?;
                }
            } else {
                kit.submit_repeated(&self.step_passes, k);
            }
            self.replays += k;
            let ring = kit.read_u32(&self.ring, TOKEN_RING_SLOTS + 1)?;
            if let Some(scope) = scope {
                if let Some(e) = dispatch::pop_scope(scope) {
                    bail!("voxtral wgpu decode step validation: {e}");
                }
                self.validated = true;
            }
            let advanced = (ring[RING_POS_SLOT] as usize).saturating_sub(next).min(remaining);
            ensure!(advanced >= 1, "the accept pass left the position at {} after {k} replays from {next}", ring[RING_POS_SLOT]);
            for i in 0..advanced {
                let t = ring[(next + i) & (TOKEN_RING_SLOTS - 1)];
                if t == eos {
                    return Ok(out);
                }
                out.push(t);
            }
            next += advanced;
        }
        Ok(out)
    }

    pub fn prefill(&mut self) -> Result<u32> {
        let kit = &self.kit;
        let scope = (!self.validated).then(|| kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation));
        if !self.seeded {
            kit.submit(&self.seed_passes);
            self.seeded = true;
        }
        kit.submit(&self.pf_passes);
        let first = kit.read_u32(&self.tok, 1)?[0];
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                bail!("voxtral wgpu prefill validation: {e}");
            }
        }
        Ok(first)
    }

    pub fn transcribe(&mut self, audio_16k_mono: &[f32]) -> Result<(String, Vec<u32>)> {
        let n_audio = self.encode(audio_16k_mono)?;
        let tokens = self.decode_stream(n_audio)?;
        Ok((self.vocab.decode(&tokens), tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kit() -> Option<Kit> {
        match WgpuContext::shared() {
            Ok(ctx) => Some(Kit::new(ctx)),
            Err(e) => {
                eprintln!("skip: no wgpu adapter: {e}");
                None
            }
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn bf(&mut self) -> u16 {
            bf16::from_f32(self.f()).to_bits()
        }
    }

    fn bv(b: u16) -> f32 {
        f32::from_bits((b as u32) << 16)
    }

    fn round_bf(x: f32) -> f32 {
        bv(bf16::from_f32(x).to_bits())
    }

    fn perm_vec4(i: usize) -> usize {
        ((i >> 7) << 7) | ((i & 3) << 5) | ((i >> 2) & 31)
    }

    fn perm_word(o: usize) -> usize {
        (perm_vec4(o >> 2) << 2) | (o & 3)
    }

    fn permute_rows(bits: &[u16], k: usize) -> Vec<u16> {
        let mut out = vec![0u16; bits.len()];
        for (e, b) in bits.iter().enumerate() {
            let (r, i) = (e / k, e % k);
            out[r * k + perm_vec4(i / 8) * 8 + i % 8] = *b;
        }
        out
    }

    fn unpack(words: &[u32]) -> Vec<f32> {
        words
            .iter()
            .flat_map(|w| [f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000)])
            .collect()
    }

    fn run(kit: &Kit, passes: &[Pass]) {
        let scope = kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        kit.submit(passes);
        if let Some(e) = dispatch::pop_scope(scope) {
            panic!("validation: {e}");
        }
    }

    fn max_rel(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        got.iter()
            .zip(want)
            .map(|(g, w)| (g - w).abs() / scale)
            .fold(0f32, f32::max)
    }

    fn gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + libm_erf(x / 2f32.sqrt()))
    }

    fn libm_erf(x: f32) -> f32 {
        let x = x as f64;
        let t = 1.0 / (1.0 + 0.5 * x.abs());
        let y = 1.0
            - t * (-x * x - 1.26551223
                + t * (1.00002368
                    + t * (0.37409196
                        + t * (0.09678418
                            + t * (-0.18628806
                                + t * (0.27886807
                                    + t * (-1.13520398
                                        + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
                .exp();
        (if x >= 0.0 { y } else { -y }) as f32
    }

    #[test]
    fn gemm_matches_the_host_product_with_bias_gelu_f32_and_accumulate() {
        let Some(kit) = kit() else { return };
        let (m, n, k) = (37usize, 128usize, 96usize);
        let mut rng = Lcg(7);
        let x: Vec<u16> = (0..m * k).map(|_| rng.bf()).collect();
        let w: Vec<u16> = (0..n * k).map(|_| rng.bf()).collect();
        let bias: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        let lin = kit.lin("t-w", &w, n, k, Some(&bias)).unwrap();
        let xb = kit.upload_bf16("t-x", &x);
        let y = kit.zeroed("t-y", m * n * 2);
        let yf = kit.upload_f32("t-yf", &vec![0.5f32; m * n]);
        run(
            &kit,
            &[
                kit.gemm(&xb, &lin, &y, m, GEMM_GELU).unwrap(),
                kit.gemm(&xb, &lin, &yf, m, GEMM_OUT_F32 | GEMM_ACCUM).unwrap(),
            ],
        );
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut acc = bias[c];
                for i in 0..k {
                    acc += bv(x[r * k + i]) * bv(w[c * k + i]);
                }
                want[r * n + c] = acc;
            }
        }
        let got = unpack(&kit.read_u32(&y, m * n / 2).unwrap());
        let want_gelu: Vec<f32> = want.iter().map(|v| gelu(*v)).collect();
        assert!(max_rel(&got, &want_gelu) < 1e-2, "bf16 gelu out: {}", max_rel(&got, &want_gelu));
        let got_f = kit.read_f32(&yf, m * n).unwrap();
        let want_f: Vec<f32> = want.iter().map(|v| v + 0.5).collect();
        assert!(max_rel(&got_f, &want_f) < 1e-4, "f32 accumulate: {}", max_rel(&got_f, &want_f));

        let part = kit.zeroed("t-part", 3 * m * n * 4);
        let ys = kit.zeroed("t-ys", m * n * 2);
        run(&kit, &kit.gemm_split(&xb, &lin, &ys, m, GEMM_GELU, &part, 3).unwrap());
        let got_s = unpack(&kit.read_u32(&ys, m * n / 2).unwrap());
        assert!(max_rel(&got_s, &want_gelu) < 1e-2, "split-k gelu out: {}", max_rel(&got_s, &want_gelu));
    }

    #[test]
    fn gemv_q3w_matches_the_host_product() {
        let Some(kit) = kit() else { return };
        let (n, k) = (70usize, 64usize);
        let mut rng = Lcg(3);
        let x: Vec<u16> = (0..2 * k).map(|_| rng.bf()).collect();
        let w: Vec<u16> = (0..n * k).map(|_| rng.bf()).collect();
        let lin = kit.lin("t-w", &w, n, k, None).unwrap();
        let xb = kit.upload_bf16("t-x", &x);
        let yf = kit.zeroed("t-yf", n * 4);
        let yb = kit.zeroed("t-yb", n * 2);
        run(
            &kit,
            &[
                kit.gemv(&xb, &lin, &yf, k / 2, true).unwrap(),
                kit.gemv(&xb, &lin, &yb, k / 2, false).unwrap(),
            ],
        );
        let want: Vec<f32> = (0..n)
            .map(|c| (0..k).map(|i| bv(x[k + i]) * bv(w[c * k + i])).sum())
            .collect();
        assert!(max_rel(&kit.read_f32(&yf, n).unwrap(), &want) < 1e-4);
        assert!(max_rel(&unpack(&kit.read_u32(&yb, n / 2).unwrap()), &want) < 1e-2);
    }

    fn host_rope(x: &[f32], hd: usize, pos: usize, theta: f32) -> Vec<f32> {
        let half = hd / 2;
        let mut out = vec![0f32; hd];
        for i in 0..half {
            let inv = 1.0 / theta.powf((i as f32 * 2.0) / hd as f32);
            let ang = pos as f32 * inv;
            let (s, c) = (ang.sin(), ang.cos());
            out[i] = x[i] * c - x[i + half] * s;
            out[i + half] = x[i + half] * c + x[i] * s;
        }
        out
    }

    #[test]
    fn rope_kv_rotates_q_and_k_and_writes_the_cache_at_the_device_position() {
        let Some(kit) = kit() else { return };
        let (rows, nh, nkv, hd, base) = (3usize, 4usize, 2usize, 8usize, 5usize);
        let stride = (nh + 2 * nkv) * hd;
        let mut rng = Lcg(11);
        let qkv: Vec<f32> = (0..rows * stride).map(|_| rng.f()).collect();
        let (cos, sin) = rope_buffers(&kit, "t", hd, 1000.0, 32);
        let qkv_b = kit.upload_f32("t-qkv", &qkv);
        let pos = kit.upload_i32("t-pos", &[base as i32]);
        let q = kit.zeroed("t-q", rows * nh * hd * 4);
        let kc = kit.zeroed("t-k", 16 * nkv * hd * 2);
        let vc = kit.zeroed("t-v", 16 * nkv * hd * 2);
        run(&kit, &[kit.rope_kv(&qkv_b, &cos, &sin, &pos, &q, &kc, &vc, rows, nh, nkv, hd).unwrap()]);
        let got_q = kit.read_f32(&q, rows * nh * hd).unwrap();
        let got_k = unpack(&kit.read_u32(&kc, 16 * nkv * hd / 2).unwrap());
        let got_v = unpack(&kit.read_u32(&vc, 16 * nkv * hd / 2).unwrap());
        for r in 0..rows {
            let p = base + r;
            for h in 0..nh {
                let src = &qkv[r * stride + h * hd..r * stride + (h + 1) * hd];
                let want = host_rope(src, hd, p, 1000.0);
                let got = &got_q[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                assert!(max_rel(got, &want) < 1e-5, "q row {r} head {h}");
            }
            for kv in 0..nkv {
                let src = &qkv[r * stride + (nh + kv) * hd..r * stride + (nh + kv + 1) * hd];
                let want: Vec<f32> = host_rope(src, hd, p, 1000.0).iter().map(|v| round_bf(*v)).collect();
                let got = &got_k[(p * nkv + kv) * hd..(p * nkv + kv + 1) * hd];
                assert!(max_rel(got, &want) < 1e-5, "k row {r} head {kv}");
                let vsrc: Vec<f32> = qkv[r * stride + (nh + nkv + kv) * hd..r * stride + (nh + nkv + kv + 1) * hd]
                    .iter()
                    .map(|v| round_bf(*v))
                    .collect();
                let gotv = &got_v[(p * nkv + kv) * hd..(p * nkv + kv + 1) * hd];
                assert!(max_rel(gotv, &vsrc) < 1e-5, "v row {r} head {kv}");
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn host_attention(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        rows: usize,
        start: usize,
        nh: usize,
        nkv: usize,
        hd: usize,
        window: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0f32; rows * nh * hd];
        for r in 0..rows {
            let qpos = start + r;
            let kmin = if window > 0 { qpos.saturating_sub(window - 1) } else { 0 };
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let qv = &q[(r * nh + h) * hd..(r * nh + h + 1) * hd];
                let scores: Vec<f32> = (kmin..=qpos)
                    .map(|key| {
                        let kv = &k[(key * nkv + kvh) * hd..(key * nkv + kvh + 1) * hd];
                        qv.iter().zip(kv).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let ws: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f32 = ws.iter().sum();
                for (j, key) in (kmin..=qpos).enumerate() {
                    let vv = &v[(key * nkv + kvh) * hd..(key * nkv + kvh + 1) * hd];
                    for d in 0..hd {
                        out[(r * nh + h) * hd + d] += ws[j] / l * vv[d];
                    }
                }
            }
        }
        out
    }

    fn attention_case(hd: usize, window: usize, rows: usize, start: usize) {
        let Some(kit) = kit() else { return };
        let (nh, nkv) = (2usize, 1usize);
        let total = start + rows;
        let mut rng = Lcg(hd as u64 + window as u64);
        let q: Vec<f32> = (0..rows * nh * hd).map(|_| rng.f()).collect();
        let kb: Vec<u16> = (0..total * nkv * hd).map(|_| rng.bf()).collect();
        let vb: Vec<u16> = (0..total * nkv * hd).map(|_| rng.bf()).collect();
        let k: Vec<f32> = kb.iter().map(|b| bv(*b)).collect();
        let v: Vec<f32> = vb.iter().map(|b| bv(*b)).collect();
        let scale = 0.3;
        let qg = kit.upload_f32("t-q", &q);
        let kg = kit.upload_bf16("t-k", &kb);
        let vg = kit.upload_bf16("t-v", &vb);
        let pos = kit.upload_i32("t-pos", &[start as i32]);
        let out = kit.zeroed("t-out", rows * nh * hd * 2);
        run(&kit, &[kit.attn_prefill(&qg, &kg, &vg, &pos, &out, rows, nh, nkv, hd, window, scale).unwrap()]);
        let got = unpack(&kit.read_u32(&out, rows * nh * hd / 2).unwrap());
        let want = host_attention(&q, &k, &v, rows, start, nh, nkv, hd, window, scale);
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-2, "prefill attention hd{hd} window {window}: rel {rel}");
    }

    #[test]
    fn prefill_attention_hd64_honours_the_sliding_window_across_key_tiles() {
        attention_case(64, 5, 150, 0);
        attention_case(64, 100, 70, 3);
    }

    #[test]
    fn prefill_attention_hd128_is_causal_over_the_cache_prefix() {
        attention_case(128, 0, 39, 0);
        attention_case(128, 0, 50, 20);
    }

    #[test]
    fn decode_attention_split_k_matches_the_host_softmax() {
        let Some(kit) = kit() else { return };
        let (nh, nkv, hd, total) = (8usize, 2usize, 128usize, 301usize);
        let mut rng = Lcg(99);
        let q: Vec<f32> = (0..nh * hd).map(|_| rng.f()).collect();
        let kb: Vec<u16> = (0..total * nkv * hd).map(|_| rng.bf()).collect();
        let vb: Vec<u16> = (0..total * nkv * hd).map(|_| rng.bf()).collect();
        let k: Vec<f32> = kb.iter().map(|b| bv(*b)).collect();
        let v: Vec<f32> = vb.iter().map(|b| bv(*b)).collect();
        let qg = kit.upload_f32("t-q", &q);
        let kg = kit.upload_bf16("t-k", &kb);
        let vg = kit.upload_bf16("t-v", &vb);
        let pos = kit.upload_i32("t-pos", &[(total - 1) as i32]);
        let out = kit.zeroed("t-out", nh * hd * 2);
        let scratch = kit.zeroed("t-scratch", nh * DECODE_ATTENTION_SPLITS as usize * (hd + 2) * 4);
        let passes = kit.attn_decode(&qg, &kg, &vg, &pos, &out, &scratch, 1, nh, nkv, hd, false).unwrap();
        run(&kit, &passes);
        let got = unpack(&kit.read_u32(&out, nh * hd / 2).unwrap());
        let want = host_attention(&q, &k, &v, 1, total - 1, nh, nkv, hd, 0, 1.0);
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-2, "decode attention: rel {rel}");
        let pos1 = kit.upload_i32("t-pos1", &[0]);
        let passes = kit.attn_decode(&qg, &kg, &vg, &pos1, &out, &scratch, 1, nh, nkv, hd, false).unwrap();
        run(&kit, &passes);
        let got = unpack(&kit.read_u32(&out, nh * hd / 2).unwrap());
        let want = host_attention(&q, &k, &v, 1, 0, nh, nkv, hd, 0, 1.0);
        assert!(max_rel(&got, &want) < 1e-2, "single-key decode attention");

        let q3: Vec<f32> = (0..2 * nh * hd).map(|_| rng.f()).chain(q.iter().copied()).collect();
        let qg3 = kit.upload_f32("t-q3", &q3);
        let pos3 = kit.upload_i32("t-pos3", &[(total - 3) as i32]);
        let out3 = kit.zeroed("t-out3", 3 * nh * hd * 2);
        let scratch3 = kit.zeroed("t-scratch3", 3 * nh * DECODE_ATTENTION_SPLITS as usize * (hd + 2) * 4);
        let passes = kit.attn_decode(&qg3, &kg, &vg, &pos3, &out3, &scratch3, 3, nh, nkv, hd, false).unwrap();
        run(&kit, &passes);
        let passes = kit.attn_decode(&qg, &kg, &vg, &pos, &out, &scratch, 1, nh, nkv, hd, false).unwrap();
        run(&kit, &passes);
        let got3 = kit.read_u32(&out3, 3 * nh * hd / 2).unwrap();
        assert_eq!(got3[nh * hd..], kit.read_u32(&out, nh * hd / 2).unwrap()[..], "row 2 of the three-row decode attention is the one-row result bitwise");
        let want0 = host_attention(&q3[..nh * hd], &k, &v, 1, total - 3, nh, nkv, hd, 0, 1.0);
        assert!(max_rel(&unpack(&got3[..nh * hd / 2]), &want0) < 1e-2, "row 0 of the three-row decode attention attends its own prefix");
        let outp = kit.zeroed("t-outp", 3 * nh * hd * 2);
        let passes = kit.attn_decode(&qg3, &kg, &vg, &pos3, &outp, &scratch3, 3, nh, nkv, hd, true).unwrap();
        run(&kit, &passes);
        let gotp = kit.read_u32(&outp, 3 * nh * hd / 2).unwrap();
        assert!((0..3 * nh * hd / 2).all(|o| gotp[perm_word(o)] == got3[o]), "the permuted attention output is the natural one relabelled");
    }

    const METAL_COMPILES_EACH_KERNELS_REDUCTION_IN_ITS_OWN_ORDER_SO_THE_TWINS_ARE_A_BAND_THERE: f32 = 1e-2;

    #[test]
    fn attention_nosg_twins_match_the_subgroup_kernels() {
        let Some(kit) = kit() else { return };
        if !kit.ctx.caps.subgroup {
            eprintln!("skip: the twins are compared against the subgroup kernels");
            return;
        }
        let nosg = Sources::with_subgroups(false);
        assert!(!nosg.vx.contains("subgroup") && !nosg.attn_pf_128.contains("subgroup"), "the ungranted sources carry no subgroup token");
        let (nh, nkv) = (2usize, 1usize);
        for (hd, window, rows, start) in [(128usize, 0usize, 50usize, 20usize), (64, 5, 150, 0)] {
            let total = start + rows;
            let mut rng = Lcg(hd as u64 + 7);
            let q = kit.upload_f32("t-q", &(0..rows * nh * hd).map(|_| rng.f()).collect::<Vec<_>>());
            let k = kit.upload_bf16("t-k", &(0..total * nkv * hd).map(|_| rng.bf()).collect::<Vec<_>>());
            let v = kit.upload_bf16("t-v", &(0..total * nkv * hd).map(|_| rng.bf()).collect::<Vec<_>>());
            let pos = kit.upload_i32("t-pos", &[start as i32]);
            let out = kit.zeroed("t-out", rows * nh * hd * 2);
            run(&kit, &[kit.attn_prefill(&q, &k, &v, &pos, &out, rows, nh, nkv, hd, window, 0.3).unwrap()]);
            let want = kit.read_u32(&out, rows * nh * hd / 2).unwrap();
            let p = kit.uni("t-p", &AttnParams { rows: rows as u32, nh: nh as u32, nkv: nkv as u32, hd: hd as u32, window: window as u32, scale: 0.3, splits: 0, perm: 0 });
            let (src, entry) = if hd == 64 { (&nosg.attn_pf_64, "vx_attn_prefill_hd64_nosg") } else { (&nosg.attn_pf_128, "vx_attn_prefill_hd128_nosg") };
            let out2 = kit.zeroed("t-out2", rows * nh * hd * 2);
            run(&kit, &[kit.pass(src, entry, &[(13, &q), (14, &k), (15, &v), (16, &pos), (17, &out2), (18, &p)], (rows.div_ceil(64) as u32, nh as u32, 1)).unwrap()]);
            let got = kit.read_u32(&out2, rows * nh * hd / 2).unwrap();
            if kit.ctx.info.backend == wgpu::Backend::Metal {
                let rel = max_rel(&unpack(&got), &unpack(&want));
                assert!(rel < METAL_COMPILES_EACH_KERNELS_REDUCTION_IN_ITS_OWN_ORDER_SO_THE_TWINS_ARE_A_BAND_THERE, "prefill hd{hd} on Metal: rel {rel}");
            } else {
                assert_eq!(got, want, "prefill hd{hd}: the workgroup-memory butterfly is the subgroup butterfly bitwise");
            }
        }
        let (nh, nkv, hd, total) = (8usize, 2usize, 128usize, 301usize);
        let mut rng = Lcg(99);
        let q: Vec<f32> = (0..nh * hd).map(|_| rng.f()).collect();
        let kb: Vec<u16> = (0..total * nkv * hd).map(|_| rng.bf()).collect();
        let vb: Vec<u16> = (0..total * nkv * hd).map(|_| rng.bf()).collect();
        let qg = kit.upload_f32("t-q", &q);
        let kg = kit.upload_bf16("t-k", &kb);
        let vg = kit.upload_bf16("t-v", &vb);
        let pos = kit.upload_i32("t-pos", &[(total - 1) as i32]);
        let out = kit.zeroed("t-out", nh * hd * 2);
        let scratch = kit.zeroed("t-scratch", nh * DECODE_ATTENTION_SPLITS as usize * (hd + 2) * 4);
        let p = kit.uni("t-p", &AttnParams { rows: 1, nh: nh as u32, nkv: nkv as u32, hd: hd as u32, window: 0, scale: 1.0, splits: DECODE_ATTENTION_SPLITS, perm: 0 });
        run(
            &kit,
            &[
                kit.pass(&nosg.vx, "vx_attn_decode1_nosg", &[(13, &qg), (14, &kg), (15, &vg), (16, &pos), (18, &p), (19, &scratch)], (nh as u32, DECODE_ATTENTION_SPLITS, 1)).unwrap(),
                kit.pass(&nosg.vx, "vx_attn_decode2", &[(17, &out), (18, &p), (19, &scratch)], (nh as u32, 1, 1)).unwrap(),
            ],
        );
        let got = unpack(&kit.read_u32(&out, nh * hd / 2).unwrap());
        let k: Vec<f32> = kb.iter().map(|b| bv(*b)).collect();
        let v: Vec<f32> = vb.iter().map(|b| bv(*b)).collect();
        let want = host_attention(&q, &k, &v, 1, total - 1, nh, nkv, hd, 0, 1.0);
        let rel = max_rel(&got, &want);
        assert!(rel < 1e-2, "decode attention without subgroups: rel {rel}");
    }

    #[test]
    fn elementwise_passes_embed_add_silu_mul_im2col_add_and_argmax() {
        let Some(kit) = kit() else { return };
        let mut rng = Lcg(5);
        let (vocab, hidden) = (10usize, 16usize);
        let embed: Vec<u16> = (0..vocab * hidden).map(|_| rng.bf()).collect();
        let audio: Vec<u16> = (0..8 * hidden).map(|_| rng.bf()).collect();
        let eb = kit.upload_bf16("t-e", &embed);
        let ab = kit.upload_bf16("t-a", &audio);
        let tok = kit.upload_i32("t-tok", &[3, 7]);
        let pos = kit.upload_i32("t-pos", &[4]);
        let out = kit.zeroed("t-out", 2 * hidden * 2);
        run(&kit, &[kit.embed_add(&eb, &tok, &ab, &pos, &out, 2, hidden).unwrap()]);
        let got = unpack(&kit.read_u32(&out, hidden).unwrap());
        for (r, t) in [3usize, 7].iter().enumerate() {
            for i in 0..hidden {
                let want = round_bf(bv(embed[t * hidden + i]) + bv(audio[(4 + r) * hidden + i]));
                assert_eq!(got[r * hidden + i], want, "embed_add row {r} col {i}");
            }
        }

        let (rows, inter) = (3usize, 12usize);
        let gu: Vec<u16> = (0..rows * 2 * inter).map(|_| rng.bf()).collect();
        let gub = kit.upload_bf16("t-gu", &gu);
        let act = kit.zeroed("t-act", rows * inter * 2);
        run(&kit, &[kit.silu_mul_rows(&gub, &act, rows, inter, false).unwrap()]);
        let got = unpack(&kit.read_u32(&act, rows * inter / 2).unwrap());
        for r in 0..rows {
            for i in 0..inter {
                let g = bv(gu[r * 2 * inter + i]);
                let u = bv(gu[r * 2 * inter + inter + i]);
                let want = g / (1.0 + (-g).exp()) * u;
                assert!((got[r * inter + i] - want).abs() <= 1e-2 * want.abs().max(1e-2), "silu row {r} col {i}");
            }
        }
        let (rows, inter) = (2usize, 2048usize);
        let gu: Vec<u16> = (0..rows * 2 * inter).map(|_| rng.bf()).collect();
        let gub = kit.upload_bf16("t-gu2", &gu);
        let act = kit.zeroed("t-act2", rows * inter * 2);
        let actp = kit.zeroed("t-actp", rows * inter * 2);
        run(&kit, &[kit.silu_mul_rows(&gub, &act, rows, inter, false).unwrap(), kit.silu_mul_rows(&gub, &actp, rows, inter, true).unwrap()]);
        let (nat, per) = (kit.read_u32(&act, rows * inter / 2).unwrap(), kit.read_u32(&actp, rows * inter / 2).unwrap());
        assert!((0..rows * inter / 2).all(|o| per[perm_word(o)] == nat[o]), "the permuted silu output is the natural one relabelled");

        let (t_in, ch) = (10usize, 4usize);
        let src: Vec<u16> = (0..t_in * ch).map(|i| bf16::from_f32(i as f32).to_bits()).collect();
        let sb = kit.upload_bf16("t-src", &src);
        let t_out = t_in / 2;
        let dst = kit.zeroed("t-dst", t_out * 3 * ch * 2);
        run(&kit, &[kit.im2col3(&sb, &dst, t_out, ch, 2, 1, t_in).unwrap()]);
        let got = unpack(&kit.read_u32(&dst, t_out * 3 * ch / 2).unwrap());
        for t2 in 0..t_out {
            for tap in 0..3 {
                for c in 0..ch {
                    let st = (2 * t2 + tap) as i64 - 1;
                    let want = if st >= 0 && (st as usize) < t_in { (st as usize * ch + c) as f32 } else { 0.0 };
                    assert_eq!(got[t2 * 3 * ch + tap * ch + c], want, "im2col t2 {t2} tap {tap} c {c}");
                }
            }
        }

        let y: Vec<u16> = (0..hidden).map(|_| rng.bf()).collect();
        let b: Vec<u16> = (0..hidden).map(|_| rng.bf()).collect();
        let yb = kit.upload_bf16("t-y", &y);
        let bb = kit.upload_bf16("t-b", &b);
        run(&kit, &[kit.add_inplace(&yb, &bb, hidden).unwrap()]);
        let got = unpack(&kit.read_u32(&yb, hidden / 2).unwrap());
        for i in 0..hidden {
            assert_eq!(got[i], round_bf(bv(y[i]) + bv(b[i])), "add_inplace {i}");
        }

        let n = 70_001usize;
        let mut logits: Vec<f32> = (0..n).map(|_| rng.f()).collect();
        logits[43_210] = 7.5;
        logits[43_211] = 7.5;
        let lb = kit.upload_f32("t-logits", &logits);
        let pval = kit.zeroed("t-pv", ARGMAX_PARTS as usize * 4);
        let pidx = kit.zeroed("t-pi", ARGMAX_PARTS as usize * 4);
        let apos = kit.upload_i32("t-apos", &[70]);
        let tok = kit.zeroed("t-tok", 4);
        let ring = kit.zeroed("t-ring", (TOKEN_RING_SLOTS + 1) * 4);
        let passes = kit.argmax(&lb, &pval, &pidx, &apos, &tok, &ring, n, 1, false).unwrap();
        run(&kit, &passes);
        assert_eq!(kit.read_u32(&tok, 1).unwrap()[0], 43_210, "argmax picks the lowest index on ties");
        assert_eq!(kit.read_u32(&ring, TOKEN_RING_SLOTS).unwrap()[70 & (TOKEN_RING_SLOTS - 1)], 43_210);
        assert_eq!(kit.read_u32(&apos, 1).unwrap()[0], 70, "the prefill argmax leaves its position alone");
        let passes = kit.argmax(&lb, &pval, &pidx, &apos, &tok, &ring, n, 1, true).unwrap();
        run(&kit, &passes);
        assert_eq!(kit.read_u32(&apos, 1).unwrap()[0], 71, "the step argmax advances the device position");
        assert_eq!(kit.read_u32(&ring, TOKEN_RING_SLOTS + 1).unwrap()[RING_POS_SLOT], 71, "the ring's last slot mirrors the position");

        let pad = crate::voxtral::STREAMING_PAD_TOKEN as usize;
        let mut rows = vec![0f32; 4 * n];
        for (r, best) in [pad, pad, 500, pad].iter().enumerate() {
            rows[r * n..(r + 1) * n].copy_from_slice(&logits);
            rows[r * n + best] = 9.0;
        }
        let lb4 = kit.upload_f32("t-logits4", &rows);
        let pval4 = kit.zeroed("t-pv4", 4 * ARGMAX_PARTS as usize * 4);
        let pidx4 = kit.zeroed("t-pi4", 4 * ARGMAX_PARTS as usize * 4);
        let passes = kit.argmax(&lb4, &pval4, &pidx4, &apos, &tok, &ring, n, 4, true).unwrap();
        run(&kit, &passes);
        let ring_now = kit.read_u32(&ring, TOKEN_RING_SLOTS + 1).unwrap();
        assert_eq!(kit.read_u32(&apos, 1).unwrap()[0], 74, "two pads then a word commit three rows");
        assert_eq!(&ring_now[71 & (TOKEN_RING_SLOTS - 1)..=73 & (TOKEN_RING_SLOTS - 1)], &[pad as u32, pad as u32, 500]);
        assert_eq!(kit.read_u32(&tok, 1).unwrap()[0], 500, "the committed token is the last accepted row's");
        assert_eq!(ring_now[RING_POS_SLOT], 74);
        rows[2 * n + 500] = 0.0;
        rows[2 * n + pad] = 9.0;
        let lb4 = kit.upload_f32("t-logits4b", &rows);
        let passes = kit.argmax(&lb4, &pval4, &pidx4, &apos, &tok, &ring, n, 4, true).unwrap();
        run(&kit, &passes);
        assert_eq!(kit.read_u32(&apos, 1).unwrap()[0], 78, "an all-pad draft commits every row");
        rows[777] = 9.5;
        let lb4 = kit.upload_f32("t-logits4c", &rows);
        let passes = kit.argmax(&lb4, &pval4, &pidx4, &apos, &tok, &ring, n, 4, true).unwrap();
        run(&kit, &passes);
        assert_eq!(kit.read_u32(&apos, 1).unwrap()[0], 79, "a word in row 0 commits one row");
        assert_eq!(kit.read_u32(&tok, 1).unwrap()[0], 777);
    }

    fn host_nv4_dequant(lin: &Lin, packed: &[u8], sf: &[u8], alpha: f32) -> Vec<f32> {
        crate::nvfp4_host::dequantize_nvfp4_host(&crate::nvfp4_host::HostNvfp4Lin {
            packed: packed.to_vec(),
            scales_swizzled: sf.to_vec(),
            alpha,
            input_global: 1.0,
            n: lin.n,
            k: lin.k,
        })
    }

    #[test]
    fn nvfp4_gemm_and_gemv_match_the_host_dequantized_product_with_prenorm_and_accumulate() {
        let Some(kit) = kit() else { return };
        if !kit.ctx.caps.subgroup {
            eprintln!("skip: no subgroups");
            return;
        }
        let (m, n, k) = (37usize, 70usize, 1024usize);
        let mut rng = Lcg(21);
        let w: Vec<u16> = (0..n * k).map(|_| rng.bf()).collect();
        let (packed, sf, alpha) = quantize_nv4_rows(&w, n, k);
        let lin = kit.lin_nv4("t-nv4", &w, n, k).unwrap();
        let wf = host_nv4_dequant(&lin, &packed, &sf, alpha);
        let max_err = w
            .iter()
            .zip(&wf)
            .map(|(a, b)| (bv(*a) - b).abs())
            .fold(0f32, f32::max);
        assert!(max_err < 0.2, "nvfp4 round trip of unit-range weights: {max_err}");

        let x: Vec<u16> = (0..m * k).map(|_| rng.bf()).collect();
        let xb = kit.upload_bf16("t-x", &x);
        let yf = kit.zeroed("t-yf", m * n * 4);
        run(&kit, &[kit.gemm(&xb, &lin, &yf, m, GEMM_OUT_F32).unwrap()]);
        let got = kit.read_f32(&yf, m * n).unwrap();
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] = (0..k).map(|i| bv(x[r * k + i]) * wf[c * k + i]).sum();
            }
        }
        assert!(max_rel(&got, &want) < 1e-4, "nvfp4 gemm: {}", max_rel(&got, &want));

        let xp = kit.upload_bf16("t-xp", &permute_rows(&x, k));
        let xrow: Vec<f32> = (0..k).map(|i| bv(x[k + i])).collect();
        let row_want: Vec<f32> = (0..n).map(|c| (0..k).map(|i| xrow[i] * wf[c * k + i]).sum()).collect();
        let yg = kit.zeroed("t-yg", n * 4);
        run(&kit, &[kit.gemv_nv4(&xp, &lin, &yg, k / 2, 1, NV4_OUT_F32).unwrap()]);
        assert!(max_rel(&kit.read_f32(&yg, n).unwrap(), &row_want) < 1e-4, "nvfp4 gemv f32");

        let y0: Vec<u16> = (0..n).map(|_| rng.bf()).collect();
        let yb = kit.upload_bf16("t-yb", &y0);
        run(&kit, &[kit.gemv_nv4(&xp, &lin, &yb, k / 2, 1, NV4_ACCUM).unwrap()]);
        let acc_want: Vec<f32> = (0..n).map(|c| row_want[c] + bv(y0[c])).collect();
        assert!(max_rel(&unpack(&kit.read_u32(&yb, n / 2).unwrap()), &acc_want) < 1e-2, "nvfp4 gemv accumulate");

        let nw: Vec<u16> = (0..k).map(|_| rng.bf()).collect();
        let nwb = kit.upload_bf16("t-nw", &nw);
        let eps = 1e-5f32;
        let rstd = 1.0 / (xrow.iter().map(|v| v * v).sum::<f32>() / k as f32 + eps).sqrt();
        let xn: Vec<f32> = (0..k).map(|i| round_bf(xrow[i] * rstd * bv(nw[i]))).collect();
        let norm_want: Vec<f32> = (0..n).map(|c| (0..k).map(|i| xn[i] * wf[c * k + i]).sum()).collect();
        let yn = kit.zeroed("t-yn", n * 4);
        let xnb = kit.zeroed("t-xn", 3 * k * 2);
        run(
            &kit,
            &[
                kit.norm_rows(&xb, &nwb, &xnb, 1, k, k / 2, eps).unwrap(),
                kit.gemv_nv4(&xnb, &lin, &yn, 0, 1, NV4_OUT_F32).unwrap(),
            ],
        );
        let xn_got = unpack(&kit.read_u32(&xnb, k / 2).unwrap());
        assert!(
            (0..k).all(|i| (xn_got[perm_vec4(i / 8) * 8 + i % 8] - xn[i]).abs() <= 1e-2 * xn[i].abs().max(1e-2)),
            "norm_rows writes the bf16 rmsnorm in the permuted layout"
        );
        let rel = max_rel(&kit.read_f32(&yn, n).unwrap(), &norm_want);
        assert!(rel < 1e-2, "nvfp4 gemv prenorm: {rel}");

        let y3 = kit.zeroed("t-y3", 3 * n * 4);
        let yn3 = kit.zeroed("t-yn3", 3 * n * 4);
        run(
            &kit,
            &[
                kit.gemv_nv4(&xp, &lin, &y3, 0, 3, NV4_OUT_F32).unwrap(),
                kit.norm_rows(&xb, &nwb, &xnb, 3, k, 0, eps).unwrap(),
                kit.gemv_nv4(&xnb, &lin, &yn3, 0, 3, NV4_OUT_F32).unwrap(),
            ],
        );
        let got3 = kit.read_f32(&y3, 3 * n).unwrap();
        assert_eq!(got3[n..2 * n], kit.read_f32(&yg, n).unwrap()[..], "the three-row GEMV's row 1 is the one-row GEMV bitwise");
        assert!(max_rel(&got3[..n], &(0..n).map(|c| (0..k).map(|i| bv(x[i]) * wf[c * k + i]).sum()).collect::<Vec<f32>>()) < 1e-4, "three-row GEMV row 0");
        assert_eq!(kit.read_f32(&yn3, 3 * n).unwrap()[n..2 * n], kit.read_f32(&yn, n).unwrap()[..], "the three-row normed GEMV's row 1 is the one-row normed GEMV bitwise");
        let yb3 = kit.upload_bf16("t-yb3", &[y0.clone(), y0.clone(), y0.clone()].concat());
        run(&kit, &[kit.gemv_nv4(&xp, &lin, &yb3, 0, 3, NV4_ACCUM).unwrap()]);
        assert_eq!(kit.read_u32(&yb3, 3 * n / 2).unwrap()[n / 2..n], kit.read_u32(&yb, n / 2).unwrap()[..], "three-row accumulate row 1 bitwise");
    }

    #[test]
    #[ignore]
    fn gemv_nv4_row_ladder_probe() {
        let Some(kit) = kit() else { return };
        let mut rng = Lcg(5);
        for (name, n, k) in [("gu", 18432usize, 3072usize), ("down", 3072, 9216), ("qkv", 6144, 3072), ("o", 3072, 4096)] {
            let w: Vec<u16> = (0..n * k).map(|_| rng.bf()).collect();
            let lin = kit.lin_nv4("p-w", &w, n, k).unwrap();
            let x: Vec<u16> = (0..PAD_DRAFT_ROWS_MAX * k).map(|_| rng.bf()).collect();
            let xb = kit.upload_bf16("p-x", &x);
            let nw: Vec<u16> = (0..k).map(|_| rng.bf()).collect();
            let nwb = kit.upload_bf16("p-nw", &nw);
            let y = kit.zeroed("p-y", PAD_DRAFT_ROWS_MAX * n * 4);
            let mut line = format!("gemv_nv4 {name} {n}x{k}:");
            for m in [1usize, 2, 4, 8] {
                let pass = kit.gemv_nv4(&xb, &lin, &y, 0, m, NV4_OUT_F32).unwrap();
                kit.submit_repeated(std::slice::from_ref(&pass), 10);
                kit.ctx.poll_blocking().unwrap();
                let reps = 200;
                let t = std::time::Instant::now();
                kit.submit_repeated(std::slice::from_ref(&pass), reps);
                kit.ctx.poll_blocking().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
                let gbs = (n * k) as f64 * 0.53125 / us / 1e3;
                let norm = kit.norm_rows(&xb, &nwb, &y, m, k, 0, 1e-5).unwrap();
                kit.submit_repeated(std::slice::from_ref(&norm), reps);
                kit.ctx.poll_blocking().unwrap();
                let t = std::time::Instant::now();
                kit.submit_repeated(std::slice::from_ref(&norm), reps);
                kit.ctx.poll_blocking().unwrap();
                let nus = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
                line += &format!("  m={m} {us:.1} us ({gbs:.0} GB/s) +norm {nus:.1} us");
            }
            eprintln!("{line}");
        }
    }

    #[test]
    #[ignore]
    fn gemm_encoder_shapes_probe() {
        let Some(kit) = kit() else { return };
        let mut rng = Lcg(9);
        let m = 376usize;
        for (name, n, k) in [("gu", 10240usize, 1280usize), ("qkv", 3840, 1280), ("down", 1280, 5120), ("o", 1280, 1280)] {
            let w: Vec<u16> = (0..n * k).map(|_| rng.bf()).collect();
            let lin = kit.lin("p-w", &w, n, k, None).unwrap();
            let x: Vec<u16> = (0..m * k).map(|_| rng.bf()).collect();
            let xb = kit.upload_bf16("p-x", &x);
            let y = kit.zeroed("p-y", m * n * 4);
            let part = kit.split_part(&[(&lin, m)]);
            let splits = kit.split_k_for(&lin, m);
            let passes = kit.gemm_split(&xb, &lin, &y, m, GEMM_OUT_F32, &part, splits).unwrap();
            kit.submit_repeated(&passes, 10);
            kit.ctx.poll_blocking().unwrap();
            let reps = 100;
            let t = std::time::Instant::now();
            kit.submit_repeated(&passes, reps);
            kit.ctx.poll_blocking().unwrap();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            let tflops = 2.0 * (m * n * k) as f64 / us / 1e6;
            eprintln!("gemm {name} {m}x{n}x{k} split {splits}: {us:.1} us ({tflops:.1} TFLOPS)");
        }
    }

    #[test]
    fn conv_taps_regroup_into_tap_major_gemm_rows() {
        let w: Vec<u16> = (0..2 * 3 * 3).map(|i| i as u16).collect();
        let g = conv_taps_to_gemm_rows(&w, 2, 3);
        assert_eq!(g[0 * 9 + 1 * 3 + 2], w[(0 * 3 + 2) * 3 + 1]);
        assert_eq!(g[1 * 9 + 2 * 3 + 0], w[(1 * 3 + 0) * 3 + 2]);
    }
}
