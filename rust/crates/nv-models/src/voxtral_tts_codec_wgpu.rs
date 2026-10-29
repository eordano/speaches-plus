use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, ensure, Context, Result};
use candle_core::{DType, Device};
use half::bf16;
use nv_weights::WeightLoader;
use nv_kernels::wgpu_backend::dispatch::{self, PassRef, ReadbackArm};
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};
use wgpu::{BindGroup, Buffer, ComputePipeline};

use crate::gemma4_wgpu_shared::pack_pairs;
use crate::pod;
use crate::voxtral_tts_codec::{
    frames_before_end_of_audio, CODEC_FRAME_TOKENS, CODEC_N_AUDIO_SPECIAL, CODEC_WEIGHTS_FILE,
};

const LABEL: &str = "nv_models_voxtral_tts_codec_wgpu";
const DIM: usize = 1024;
const HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const FFN: usize = 4096;
const LATENT: usize = 292;
const SEMANTIC_ROWS: usize = 8192;
const SEMANTIC_DIM: usize = 256;
const PATCH: usize = 240;
const ACOUSTIC_LEVELS: i32 = 21;
const STAGE_BLOCKS: [usize; 4] = [1, 3, 5, 7];
const STAGE_WINDOWS: [usize; 4] = [2, 4, 8, 16];
const UPSAMPLER_BLOCKS: [usize; 3] = [2, 4, 6];
const INPUT_CONV_TAPS: usize = 3;
const OUTPUT_CONV_TAPS: usize = 7;
const INPUT_CONV_K_876_PADS_TO_896_FOR_THE_32_WIDE_GEMM_K_STEP: usize = 896;
const QK_NORM_EPS: f32 = 1e-6;
const BLOCK_NORM_EPS: f32 = 1e-2;
const CLUSTER_USAGE_FLOOR: f32 = 1e-5;
const GEMM_TILE: usize = 64;
const GEMM_K_STEP: usize = 32;
const GEMM_SPLIT_K_TARGETS_THIS_MANY_WORKGROUPS: usize = 768;
const GEMM_SPLIT_K_MAX: usize = 16;
const GEMV_TAKES_UP_TO_THIS_MANY_INPUT_ROWS_BECAUSE_A_64_WIDE_TILE_WASTES_THE_REST: usize = 8;
const GEMV_LANES_PER_ROW_COVER_K_IN_ONE_LOAD_OF_8_BUT_LEAVE_A_WORKGROUP_TWO_ROWS_FOR_THE_PACKED_PAIR_STORE: usize = 128;
const GEMM_SCALE_ACCUM: u32 = 1;
const PAD_ZERO: u32 = 0;
const PAD_REPLICATE: u32 = 1;
const PAD_REFLECT: u32 = 2;
const SCRATCH_GROWS_TO_THE_LARGEST_CALL_AND_STARTS_AT_THIS_MANY_FRAMES: usize = 64;

pub const THE_STATE_IS_DEVICE_RESIDENT_SO_A_FRAME_REPLAYS_A_PREBUILT_PASS_LIST: &str =
    "a frame counter, the k-1 trailing columns of every conv and a window-slot K/V ring per \
     layer live in device buffers; the im2col, attention and commit passes read the counter to \
     pick pad-vs-history and the valid ring span, and an advance pass bumps it, so the pass list \
     for a given frame count is built once per state and replayed with no host work but the \
     code upload and the sample readback";

const CX_WGSL: &str = r#"
struct CxGemmParams { m: u32, n: u32, k_words: u32, flags: u32, k_split: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> gm_x: array<u32>;
@group(0) @binding(1) var<storage, read> gm_w: array<u32>;
@group(0) @binding(2) var<storage, read> gm_scale: array<f32>;
@group(0) @binding(3) var<storage, read_write> gm_y: array<u32>;
@group(0) @binding(4) var<uniform> gm_p: CxGemmParams;
@group(0) @binding(5) var<storage, read_write> gm_part: array<f32>;

const GM_LD: u32 = 33u;
var<workgroup> gm_xs: array<f32, 2112>;
var<workgroup> gm_ws: array<f32, 2112>;

fn cx_rbf(x: f32) -> f32 { return bf16_lo(bf16_pack(x, 0.0)); }

fn gm_store(r: u32, c: u32, v0_in: f32, v1_in: f32) {
    var v0 = v0_in;
    var v1 = v1_in;
    let widx = (r * gm_p.n + c) >> 1u;
    if ((gm_p.flags & 1u) != 0u) {
        let old = gm_y[widx];
        v0 = bf16_lo(old) + cx_rbf(cx_rbf(v0) * gm_scale[c]);
        v1 = bf16_hi(old) + cx_rbf(cx_rbf(v1) * gm_scale[c + 1u]);
    }
    gm_y[widx] = bf16_pack(v0, v1);
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

@compute @workgroup_size(64)
fn cx_gemm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let tx = tid & 7u;
    let ty = tid >> 3u;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;
    let n_kt = gm_p.k_words / 16u;
    let per = (n_kt + gm_p.k_split - 1u) / gm_p.k_split;
    let kt0 = wid.z * per;
    let kt1 = min(n_kt, kt0 + per);
    let lr = tid >> 2u;
    let lc = tid & 3u;
    GEMM_ACC_DECL
    var xr0 = vec4<u32>(0u);
    var xr1 = vec4<u32>(0u);
    var xr2 = vec4<u32>(0u);
    var xr3 = vec4<u32>(0u);
    var wr0 = vec4<u32>(0u);
    var wr1 = vec4<u32>(0u);
    var wr2 = vec4<u32>(0u);
    var wr3 = vec4<u32>(0u);
    var kt = kt0;
    if (kt < kt1) {
        xr0 = gm_load_x(row0 + lr, kt, lc);
        xr1 = gm_load_x(row0 + lr + 16u, kt, lc);
        xr2 = gm_load_x(row0 + lr + 32u, kt, lc);
        xr3 = gm_load_x(row0 + lr + 48u, kt, lc);
        wr0 = gm_load_w(col0 + lr, kt, lc);
        wr1 = gm_load_w(col0 + lr + 16u, kt, lc);
        wr2 = gm_load_w(col0 + lr + 32u, kt, lc);
        wr3 = gm_load_w(col0 + lr + 48u, kt, lc);
    }
    for (; kt < kt1; kt = kt + 1u) {
        let xs = lr * GM_LD + lc * 8u;
        gm_stage8(xs, xr0);
        gm_stage8(xs + 16u * GM_LD, xr1);
        gm_stage8(xs + 32u * GM_LD, xr2);
        gm_stage8(xs + 48u * GM_LD, xr3);
        gm_stage8w(xs, wr0);
        gm_stage8w(xs + 16u * GM_LD, wr1);
        gm_stage8w(xs + 32u * GM_LD, wr2);
        gm_stage8w(xs + 48u * GM_LD, wr3);
        let kn = kt + 1u;
        if (kn < kt1) {
            xr0 = gm_load_x(row0 + lr, kn, lc);
            xr1 = gm_load_x(row0 + lr + 16u, kn, lc);
            xr2 = gm_load_x(row0 + lr + 32u, kn, lc);
            xr3 = gm_load_x(row0 + lr + 48u, kn, lc);
            wr0 = gm_load_w(col0 + lr, kn, lc);
            wr1 = gm_load_w(col0 + lr + 16u, kn, lc);
            wr2 = gm_load_w(col0 + lr + 32u, kn, lc);
            wr3 = gm_load_w(col0 + lr + 48u, kn, lc);
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
fn cx_gemm_reduce(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
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

const GV_M: u32 = 8u;
var<workgroup> gv_red: array<f32, 2048>;

fn gv_fma8(w: vec4<u32>, x: vec4<u32>, acc: f32) -> f32 {
    var a = fma(bf16_lo(w.x), bf16_lo(x.x), acc);
    a = fma(bf16_hi(w.x), bf16_hi(x.x), a);
    a = fma(bf16_lo(w.y), bf16_lo(x.y), a);
    a = fma(bf16_hi(w.y), bf16_hi(x.y), a);
    a = fma(bf16_lo(w.z), bf16_lo(x.z), a);
    a = fma(bf16_hi(w.z), bf16_hi(x.z), a);
    a = fma(bf16_lo(w.w), bf16_lo(x.w), a);
    return fma(bf16_hi(w.w), bf16_hi(x.w), a);
}

@compute @workgroup_size(256)
fn cx_gemv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let lanes = gm_p.k_split;
    let rows = 256u / lanes;
    let r = tid / lanes;
    let lane = tid % lanes;
    let row = wid.x * rows + r;
    var acc: array<f32, 8>;
    for (var j = 0u; j < GV_M; j = j + 1u) { acc[j] = 0.0; }
    if (row < gm_p.n) {
        let wb = row * gm_p.k_words;
        for (var kw = lane * 4u; kw < gm_p.k_words; kw = kw + lanes * 4u) {
            let w = vec4<u32>(gm_w[wb + kw], gm_w[wb + kw + 1u], gm_w[wb + kw + 2u], gm_w[wb + kw + 3u]);
            for (var j = 0u; j < GV_M; j = j + 1u) {
                if (j < gm_p.m) {
                    let xb = j * gm_p.k_words + kw;
                    acc[j] = gv_fma8(w, vec4<u32>(gm_x[xb], gm_x[xb + 1u], gm_x[xb + 2u], gm_x[xb + 3u]), acc[j]);
                }
            }
        }
    }
    for (var j = 0u; j < GV_M; j = j + 1u) { gv_red[(r * GV_M + j) * lanes + lane] = acc[j]; }
    workgroupBarrier();
    if (tid < rows * GV_M) {
        var s = 0.0;
        for (var l = 0u; l < lanes; l = l + 1u) { s = s + gv_red[tid * lanes + l]; }
        gv_red[tid * lanes] = s;
    }
    workgroupBarrier();
    if (tid < (rows / 2u) * GV_M) {
        let p = tid / GV_M;
        let j = tid % GV_M;
        let c = wid.x * rows + p * 2u;
        if (j < gm_p.m && c < gm_p.n) {
            gm_store(j, c, gv_red[(p * 2u * GV_M + j) * lanes], gv_red[((p * 2u + 1u) * GV_M + j) * lanes]);
        }
    }
}

fn cx_flat(wid: vec3<u32>, nwg: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (wid.y * nwg.x + wid.x) * 256u + lid.x;
}

@group(0) @binding(40) var<storage, read> cx_ctl: array<u32>;

struct CxColParams { rows: u32, c_words: u32, taps: u32, kpad_words: u32, mode: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(6) var<storage, read> ic_x: array<u32>;
@group(0) @binding(7) var<storage, read> ic_hist: array<u32>;
@group(0) @binding(8) var<storage, read_write> ic_dst: array<u32>;
@group(0) @binding(9) var<uniform> ic_p: CxColParams;

@compute @workgroup_size(256)
fn cx_im2col(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = cx_flat(wid, nwg, lid);
    if (idx >= ic_p.rows * ic_p.kpad_words) { return; }
    let t = idx / ic_p.kpad_words;
    let w = idx % ic_p.kpad_words;
    let tap = w / ic_p.c_words;
    let cw = w % ic_p.c_words;
    var v = 0u;
    if (tap < ic_p.taps) {
        let src_t = i32(t + tap + 1u) - i32(ic_p.taps);
        if (src_t >= 0) {
            v = ic_x[u32(src_t) * ic_p.c_words + cw];
        } else if (cx_ctl[0] != 0u) {
            v = ic_hist[u32(i32(ic_p.taps) - 1 + src_t) * ic_p.c_words + cw];
        } else if (ic_p.mode == 1u) {
            v = ic_x[cw];
        } else if (ic_p.mode == 2u) {
            v = ic_x[u32(-src_t) * ic_p.c_words + cw];
        }
    }
    ic_dst[idx] = v;
}

struct CxEwParams { rows: u32, words: u32, off: u32, pad0: u32 };
@group(0) @binding(10) var<storage, read> cp_src: array<u32>;
@group(0) @binding(11) var<storage, read_write> cp_dst: array<u32>;
@group(0) @binding(12) var<uniform> cp_p: CxEwParams;

@compute @workgroup_size(256)
fn cx_carry(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = cx_flat(wid, nwg, lid);
    if (i >= cp_p.words) { return; }
    cp_dst[i] = cp_src[cp_p.off + i];
}

@group(0) @binding(13) var<storage, read> lt_codes: array<i32>;
@group(0) @binding(14) var<storage, read> lt_table: array<u32>;
@group(0) @binding(15) var<storage, read_write> lt_out: array<u32>;
@group(0) @binding(16) var<uniform> lt_p: CxEwParams;
const LT_WORDS: u32 = 146u;
const LT_SEM_WORDS: u32 = 128u;

fn lt_acoustic(code: i32) -> f32 { return f32((code - 2) * 2) / 20.0 - 1.0; }

@compute @workgroup_size(256)
fn cx_latent(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = cx_flat(wid, nwg, lid);
    if (idx >= lt_p.rows * LT_WORDS) { return; }
    let t = idx / LT_WORDS;
    let w = idx % LT_WORDS;
    let frame = t * 37u;
    if (w < LT_SEM_WORDS) {
        lt_out[idx] = lt_table[u32(lt_codes[frame] - 2) * LT_SEM_WORDS + w];
    } else {
        let a = frame + 1u + (w - LT_SEM_WORDS) * 2u;
        lt_out[idx] = bf16_pack(lt_acoustic(lt_codes[a]), lt_acoustic(lt_codes[a + 1u]));
    }
}

struct CxNormParams { hidden: u32, x_stride_words: u32, x_off_words: u32, y_stride_words: u32, eps: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(17) var<storage, read> nr_x: array<u32>;
@group(0) @binding(18) var<storage, read> nr_w: array<u32>;
@group(0) @binding(19) var<storage, read_write> nr_y: array<u32>;
@group(0) @binding(20) var<uniform> nr_p: CxNormParams;
@group(0) @binding(41) var<storage, read> nr_w2: array<u32>;
@group(0) @binding(42) var<storage, read_write> nr_y2: array<u32>;
var<workgroup> nr_red: array<f32, 256>;

fn cx_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

@compute @workgroup_size(256)
fn cx_norm_qk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let which = wid.y;
    let tid = lid.x;
    let words = nr_p.hidden >> 1u;
    let base = which * nr_p.x_off_words + r * nr_p.x_stride_words;
    var local = 0.0;
    for (var i = tid; i < words; i = i + 256u) {
        let xw = nr_x[base + i];
        local = fma(bf16_lo(xw), bf16_lo(xw), local);
        local = fma(bf16_hi(xw), bf16_hi(xw), local);
    }
    nr_red[tid] = local;
    workgroupBarrier();
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) { nr_red[tid] = nr_red[tid] + nr_red[tid + stride]; }
        workgroupBarrier();
    }
    let rstd = inverseSqrt(nr_p.eps + cx_div_rn(nr_red[0], f32(nr_p.hidden)));
    for (var i = tid; i < words; i = i + 256u) {
        let xw = nr_x[base + i];
        if (which == 0u) {
            let ww = nr_w[i];
            nr_y[r * nr_p.y_stride_words + i] = bf16_pack(bf16_lo(xw) * rstd * bf16_lo(ww), bf16_hi(xw) * rstd * bf16_hi(ww));
        } else {
            let ww = nr_w2[i];
            nr_y2[r * nr_p.y_stride_words + i] = bf16_pack(bf16_lo(xw) * rstd * bf16_lo(ww), bf16_hi(xw) * rstd * bf16_hi(ww));
        }
    }
}

struct CxAttnParams { rows: u32, window: u32, rows_per_frame: u32, v_stride_words: u32, v_off_words: u32, scale: f32, pad0: u32, pad1: u32 };
@group(0) @binding(21) var<storage, read> at_q: array<u32>;
@group(0) @binding(22) var<storage, read> at_k: array<u32>;
@group(0) @binding(23) var<storage, read> at_v: array<u32>;
@group(0) @binding(24) var<storage, read> at_kring: array<u32>;
@group(0) @binding(25) var<storage, read> at_vring: array<u32>;
@group(0) @binding(26) var<storage, read_write> at_out: array<u32>;
@group(0) @binding(27) var<uniform> at_p: CxAttnParams;
const AT_HW: u32 = 64u;
const AT_RW: u32 = 512u;
const AT_KEYS: u32 = 17u;
const AT_RING: u32 = 0x80000000u;
var<workgroup> at_part: array<f32, 1088>;
var<workgroup> at_sc: array<f32, AT_KEYS>;
var<workgroup> at_kb: array<u32, AT_KEYS>;
var<workgroup> at_dist: array<f32, AT_KEYS>;

fn at_vword(b: u32, h: u32, d: u32) -> u32 {
    if ((b & AT_RING) != 0u) { return at_vring[(b & ~AT_RING) * AT_RW + h * AT_HW + d]; }
    return at_v[b * at_p.v_stride_words + at_p.v_off_words + h * AT_HW + d];
}

@compute @workgroup_size(64)
fn cx_attn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let g = wid.y * nwg.x + wid.x;
    let i = g / 8u;
    let h = g % 8u;
    let d = lid.x;
    if (i >= at_p.rows) { return; }
    let w = at_p.window;
    let f = cx_ctl[0] * at_p.rows_per_frame;
    let p = f + i;
    let hb = h * AT_HW + d;
    let qw = at_q[i * AT_RW + hb];
    let qlo = bf16_lo(qw);
    let qhi = bf16_hi(qw);
    var nk = 0u;
    for (var pos = select(0u, p - w, p > w); pos < f; pos = pos + 1u) {
        let slot = pos & (w - 1u);
        let kw = at_kring[slot * AT_RW + hb];
        at_part[nk * 64u + d] = fma(qlo, bf16_lo(kw), qhi * bf16_hi(kw));
        if (d == 0u) { at_kb[nk] = slot | AT_RING; at_dist[nk] = f32(p - pos); }
        nk = nk + 1u;
    }
    for (var j = select(0u, i - w, i > w); j <= i; j = j + 1u) {
        let kw = at_k[j * AT_RW + hb];
        at_part[nk * 64u + d] = fma(qlo, bf16_lo(kw), qhi * bf16_hi(kw));
        if (d == 0u) { at_kb[nk] = j; at_dist[nk] = f32(i - j); }
        nk = nk + 1u;
    }
    workgroupBarrier();
    if (d < nk) {
        var s = 0.0;
        for (var t = 0u; t < 64u; t = t + 1u) { s = s + at_part[d * 64u + t]; }
        at_sc[d] = s * at_p.scale - exp2(-f32(h)) * at_dist[d];
    }
    workgroupBarrier();
    var mx = -3.0e38;
    for (var k = 0u; k < nk; k = k + 1u) { mx = max(mx, at_sc[k]); }
    var l = 0.0;
    var lo = 0.0;
    var hi = 0.0;
    for (var k = 0u; k < nk; k = k + 1u) {
        let e = exp(at_sc[k] - mx);
        let vw = at_vword(at_kb[k], h, d);
        l = l + e;
        lo = fma(e, bf16_lo(vw), lo);
        hi = fma(e, bf16_hi(vw), hi);
    }
    at_out[i * AT_RW + hb] = bf16_pack(lo / l, hi / l);
}

@group(0) @binding(29) var<storage, read> kc_k: array<u32>;
@group(0) @binding(30) var<storage, read> kc_v: array<u32>;
@group(0) @binding(31) var<storage, read_write> kc_kring: array<u32>;
@group(0) @binding(32) var<storage, read_write> kc_vring: array<u32>;
@group(0) @binding(33) var<uniform> kc_p: CxAttnParams;

@compute @workgroup_size(256)
fn cx_kv_commit(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = cx_flat(wid, nwg, lid);
    let cnt = min(kc_p.rows, kc_p.window);
    if (idx >= cnt * AT_RW) { return; }
    let r = idx / AT_RW;
    let c = idx % AT_RW;
    let j = kc_p.rows - cnt + r;
    let slot = (cx_ctl[0] * kc_p.rows_per_frame + j) & (kc_p.window - 1u);
    kc_kring[slot * AT_RW + c] = kc_k[j * AT_RW + c];
    kc_vring[slot * AT_RW + c] = kc_v[j * kc_p.v_stride_words + kc_p.v_off_words + c];
}

@group(0) @binding(35) var<storage, read> sm_gu: array<u32>;
@group(0) @binding(36) var<storage, read_write> sm_out: array<u32>;
@group(0) @binding(37) var<uniform> sm_p: CxEwParams;

fn cx_silu(x: f32) -> f32 { return x / (1.0 + exp(-x)); }

@compute @workgroup_size(256)
fn cx_silu_mul_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = cx_flat(wid, nwg, lid);
    if (idx >= sm_p.rows * sm_p.words) { return; }
    let r = idx / sm_p.words;
    let w = idx % sm_p.words;
    let g = sm_gu[r * 2u * sm_p.words + w];
    let u = sm_gu[r * 2u * sm_p.words + sm_p.words + w];
    sm_out[idx] = bf16_pack(cx_silu(bf16_lo(g)) * bf16_lo(u), cx_silu(bf16_hi(g)) * bf16_hi(u));
}

@group(0) @binding(38) var<storage, read_write> ad_ctl: array<u32>;
@group(0) @binding(39) var<uniform> ad_p: CxEwParams;

@compute @workgroup_size(1)
fn cx_advance() {
    ad_ctl[0] = ad_ctl[0] + ad_p.rows;
}
"#;

pod! {
    #[derive(Debug)]
    struct GemmParams { m: u32, n: u32, k_words: u32, flags: u32, k_split: u32, pad0: u32, pad1: u32, pad2: u32 }
    #[derive(Debug)]
    struct ColParams { rows: u32, c_words: u32, taps: u32, kpad_words: u32, mode: u32, pad0: u32, pad1: u32, pad2: u32 }
    #[derive(Debug)]
    struct EwParams { rows: u32, words: u32, off: u32, pad0: u32 }
    #[derive(Debug)]
    struct NormParams { hidden: u32, x_stride_words: u32, x_off_words: u32, y_stride_words: u32, eps: f32, pad0: u32, pad1: u32, pad2: u32 }
    #[derive(Debug)]
    struct AttnParams { rows: u32, window: u32, rows_per_frame: u32, v_stride_words: u32, v_off_words: u32, scale: f32, pad0: u32, pad1: u32 }
    #[derive(Debug)]
    struct RmsParams { hidden: u32, batch: u32, eps: f32, words_per_row: u32 }
}

fn gemm_source() -> String {
    let ij = || (0..8).flat_map(|i| (0..8).map(move |j| (i, j)));
    let decl: Vec<String> = ij().map(|(i, j)| format!("var acc{i}_{j} = 0.0;")).collect();
    let loads: Vec<String> = (0..8)
        .map(|i| format!("let x{i} = gm_xs[xb + {i}u * GM_LD + kk];\n            let w{i} = gm_ws[wb + {i}u * GM_LD + kk];"))
        .collect();
    let fmas: Vec<String> = ij().map(|(i, j)| format!("acc{i}_{j} = fma(x{i}, w{j}, acc{i}_{j});")).collect();
    let epi: Vec<String> = (0..8)
        .flat_map(|i| (0..8).step_by(2).map(move |j| format!("gm_emit(r0 + {i}u, c0 + {j}u, acc{i}_{j}, acc{i}_{}, wid.z);", j + 1)))
        .collect();
    compose(
        &CX_WGSL
            .replace("GEMM_ACC_DECL", &decl.join("\n    "))
            .replace("GEMM_FMA_BODY", &format!("{}\n            {}", loads.join("\n            "), fmas.join("\n            ")))
            .replace("GEMM_EPILOGUE", &epi.join("\n    ")),
    )
}

struct Pass {
    pipeline: Arc<ComputePipeline>,
    bind: BindGroup,
    grid: (u32, u32, u32),
    label: String,
    _held: Vec<Buffer>,
}

fn refs(passes: &[Pass]) -> Vec<PassRef<'_>> {
    passes.iter().map(|p| (p.pipeline.as_ref(), &p.bind, p.grid)).collect()
}

struct Lin {
    w: Buffer,
    n: usize,
    k: usize,
    name: &'static str,
}

struct Kit {
    ctx: &'static WgpuContext,
    cx: String,
    rms: String,
    dummy: Buffer,
}

fn err(e: nv_kernels::wgpu_backend::WgpuError) -> anyhow::Error {
    anyhow!("{e}")
}

impl Kit {
    fn upload_bf16(&self, label: &str, bits: &[u16]) -> Buffer {
        dispatch::storage_from_slice(self.ctx, label, &pack_pairs(bits))
    }

    fn upload_f32(&self, label: &str, data: &[f32]) -> Buffer {
        dispatch::storage_from_slice(self.ctx, label, data)
    }

    fn zeroed(&self, label: &str, words: usize) -> Buffer {
        dispatch::storage_zeroed(self.ctx, label, (words * 4) as u64)
    }

    fn uni<T: bytemuck::Pod>(&self, v: &T) -> Buffer {
        dispatch::uniform_from(self.ctx, "cx-p", v)
    }

    fn lin(&self, name: &'static str, bits: &[u16], n: usize, k: usize) -> Result<Lin> {
        ensure!(bits.len() == n * k, "{name}: {} elements, want {n}x{k}", bits.len());
        ensure!(k % GEMM_K_STEP == 0 && n % 2 == 0, "{name}: {n}x{k} is not GEMM-tileable");
        Ok(Lin { w: self.upload_bf16(name, bits), n, k, name })
    }

    fn pass(&self, source: &str, entry: &str, binds: &[(u32, &Buffer)], grid: (u32, u32, u32)) -> Result<Pass> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, LABEL, source, entry)
            .map_err(|e| anyhow!("pipeline {entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        let _ = crate::voxtral_wgpu::A_PASS_HOLDS_ITS_BOUND_BUFFERS_BECAUSE_THE_WEBGPU_BACKEND_DESTROYS_A_BUFFER_WHEN_ITS_LAST_HANDLE_DROPS;
        Ok(Pass { pipeline, bind, grid, label: entry.to_string(), _held: binds.iter().map(|(_, b)| (*b).clone()).collect() })
    }

    fn grid_flat(&self, invocations: usize, wg: u32) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations as u64, wg)
    }

    fn split_k_for(&self, lin: &Lin, m: usize) -> usize {
        let tiles = lin.n.div_ceil(GEMM_TILE) * m.div_ceil(GEMM_TILE);
        GEMM_SPLIT_K_TARGETS_THIS_MANY_WORKGROUPS
            .div_ceil(tiles)
            .min(GEMM_SPLIT_K_MAX)
            .min((lin.k / GEMM_K_STEP / 4).max(1))
    }

    fn part_words(&self, lin: &Lin, m: usize) -> usize {
        self.split_k_for(lin, m) * m * lin.n
    }

    fn gemv_lanes_per_row(&self, lin: &Lin) -> usize {
        let per_row_loads = lin.k / 8;
        (per_row_loads.min(GEMV_LANES_PER_ROW_COVER_K_IN_ONE_LOAD_OF_8_BUT_LEAVE_A_WORKGROUP_TWO_ROWS_FOR_THE_PACKED_PAIR_STORE) + 1).next_power_of_two() / 2
    }

    fn gemm(&self, x: &Buffer, lin: &Lin, scale: Option<&Buffer>, y: &Buffer, m: usize, part: &Buffer) -> Result<Vec<Pass>> {
        let gemv = m <= GEMV_TAKES_UP_TO_THIS_MANY_INPUT_ROWS_BECAUSE_A_64_WIDE_TILE_WASTES_THE_REST;
        let splits = if gemv { self.gemv_lanes_per_row(lin) } else { self.split_k_for(lin, m) };
        ensure!(part.size() >= (self.part_words(lin, m) * 4) as u64, "{}: split-k scratch too small for m={m}", lin.name);
        let p = self.uni(&GemmParams {
            m: m as u32,
            n: lin.n as u32,
            k_words: (lin.k / 2) as u32,
            flags: if scale.is_some() { GEMM_SCALE_ACCUM } else { 0 },
            k_split: splits as u32,
            ..Default::default()
        });
        let scale = scale.unwrap_or(&self.dummy);
        let named = |mut pass: Pass| {
            pass.label = format!("{}:{}", pass.label, lin.name);
            pass
        };
        let mut passes = vec![named(if gemv {
            self.pass(&self.cx, "cx_gemv", &[(0, x), (1, &lin.w), (2, scale), (3, y), (4, &p)], self.grid_flat(lin.n.div_ceil(256 / splits) * 256, 256))?
        } else {
            self.pass(
                &self.cx,
                "cx_gemm",
                &[(0, x), (1, &lin.w), (2, scale), (3, y), (4, &p), (5, part)],
                (lin.n.div_ceil(GEMM_TILE) as u32, m.div_ceil(GEMM_TILE) as u32, splits as u32),
            )?
        })];
        if splits > 1 && !gemv {
            passes.push(named(self.pass(
                &self.cx,
                "cx_gemm_reduce",
                &[(2, scale), (3, y), (4, &p), (5, part)],
                self.grid_flat(m * lin.n / 2, 256),
            )?));
        }
        Ok(passes)
    }

    #[allow(clippy::too_many_arguments)]
    fn im2col(&self, x: &Buffer, hist: &Buffer, ctl: &Buffer, dst: &Buffer, rows: usize, channels: usize, taps: usize, kpad: usize, mode: u32) -> Result<Pass> {
        let p = self.uni(&ColParams {
            rows: rows as u32,
            c_words: (channels / 2) as u32,
            taps: taps as u32,
            kpad_words: (kpad / 2) as u32,
            mode,
            ..Default::default()
        });
        self.pass(&self.cx, "cx_im2col", &[(6, x), (7, hist), (8, dst), (9, &p), (40, ctl)], self.grid_flat(rows * kpad / 2, 256))
    }

    fn carry(&self, col: &Buffer, hist: &Buffer, rows: usize, channels: usize, taps: usize, kpad: usize) -> Result<Pass> {
        let words = (taps - 1) * channels / 2;
        let p = self.uni(&EwParams { rows: 0, words: words as u32, off: ((rows - 1) * kpad / 2 + channels / 2) as u32, pad0: 0 });
        self.pass(&self.cx, "cx_carry", &[(10, col), (11, hist), (12, &p)], self.grid_flat(words, 256))
    }

    fn latent(&self, codes: &Buffer, table: &Buffer, lat: &Buffer, rows: usize) -> Result<Pass> {
        let p = self.uni(&EwParams { rows: rows as u32, ..Default::default() });
        self.pass(&self.cx, "cx_latent", &[(13, codes), (14, table), (15, lat), (16, &p)], self.grid_flat(rows * LATENT / 2, 256))
    }

    fn norm_qk(&self, qkv: &Buffer, layer: &Layer, qn: &Buffer, kn: &Buffer, rows: usize) -> Result<Pass> {
        let p = self.uni(&NormParams {
            hidden: DIM as u32,
            x_stride_words: (3 * DIM / 2) as u32,
            x_off_words: (DIM / 2) as u32,
            y_stride_words: (DIM / 2) as u32,
            eps: QK_NORM_EPS,
            ..Default::default()
        });
        self.pass(&self.cx, "cx_norm_qk", &[(17, qkv), (18, &layer.q_norm), (19, qn), (20, &p), (41, &layer.k_norm), (42, kn)], (rows as u32, 2, 1))
    }

    fn rmsnorm(&self, x: &Buffer, w: &Buffer, y: &Buffer, rows: usize, eps: f32) -> Result<Pass> {
        let p = self.uni(&RmsParams { hidden: DIM as u32, batch: rows as u32, eps, words_per_row: (DIM / 2) as u32 });
        self.pass(&self.rms, "rmsnorm_bf16", &[(0, x), (1, w), (2, y), (3, &p)], (rows as u32, 1, 1))
    }

    fn attn_params(&self, rows: usize, window: usize, rows_per_frame: usize) -> Buffer {
        self.uni(&AttnParams {
            rows: rows as u32,
            window: window as u32,
            rows_per_frame: rows_per_frame as u32,
            v_stride_words: (3 * DIM / 2) as u32,
            v_off_words: DIM as u32,
            scale: 1.0 / (HEAD_DIM as f32).sqrt(),
            pad0: 0,
            pad1: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn attn(&self, q: &Buffer, k: &Buffer, qkv: &Buffer, ring: &(Buffer, Buffer), ctl: &Buffer, out: &Buffer, rows: usize, window: usize, rows_per_frame: usize) -> Result<Pass> {
        let p = self.attn_params(rows, window, rows_per_frame);
        self.pass(
            &self.cx,
            "cx_attn",
            &[(21, q), (22, k), (23, qkv), (24, &ring.0), (25, &ring.1), (26, out), (27, &p), (40, ctl)],
            self.grid_flat(rows * HEADS * 64, 64),
        )
    }

    fn kv_commit(&self, k: &Buffer, qkv: &Buffer, ring: &(Buffer, Buffer), ctl: &Buffer, rows: usize, window: usize, rows_per_frame: usize) -> Result<Pass> {
        let p = self.attn_params(rows, window, rows_per_frame);
        self.pass(
            &self.cx,
            "cx_kv_commit",
            &[(29, k), (30, qkv), (31, &ring.0), (32, &ring.1), (33, &p), (40, ctl)],
            self.grid_flat(rows.min(window) * DIM / 2, 256),
        )
    }

    fn silu_mul(&self, gu: &Buffer, out: &Buffer, rows: usize) -> Result<Pass> {
        let p = self.uni(&EwParams { rows: rows as u32, words: (FFN / 2) as u32, off: 0, pad0: 0 });
        self.pass(&self.cx, "cx_silu_mul_rows", &[(35, gu), (36, out), (37, &p)], self.grid_flat(rows * FFN / 2, 256))
    }

    fn advance(&self, ctl: &Buffer, frames: usize) -> Result<Pass> {
        let p = self.uni(&EwParams { rows: frames as u32, ..Default::default() });
        self.pass(&self.cx, "cx_advance", &[(38, ctl), (39, &p)], (1, 1, 1))
    }

    fn submit(&self, passes: &[Pass]) -> Result<()> {
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            let labels: Vec<String> = passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &refs(passes), &labels).map_err(err)
        } else {
            dispatch::submit_pass_list(self.ctx, refs(passes).into_iter());
            Ok(())
        }
    }
}

struct Layer {
    attn_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    ffn_norm: Buffer,
    attn_scale: Buffer,
    ffn_scale: Buffer,
    qkv: Lin,
    wo: Lin,
    gate_up: Lin,
    w2: Lin,
}

struct Persist {
    ctl: Buffer,
    lat_hist: Buffer,
    rings: Vec<(Buffer, Buffer)>,
    hist_up: [Buffer; 3],
    hist_out: Buffer,
}

struct Scratch {
    cap: usize,
    codes: Buffer,
    lat: Buffer,
    col_in: Buffer,
    x: [Buffer; 2],
    h: Buffer,
    qkv: Buffer,
    qn: Buffer,
    kn: Buffer,
    o: Buffer,
    gu: Buffer,
    act: Buffer,
    col_up: Buffer,
    col_out: Buffer,
    out: Buffer,
    part: Buffer,
}

#[derive(Default)]
pub struct CodecState {
    persist: Option<Persist>,
    scratch: Option<Scratch>,
    passes: HashMap<usize, Vec<Pass>>,
}

impl CodecState {
    pub fn reset(&mut self, codec: &VoxtralTtsCodecWgpu) {
        if let Some(p) = &self.persist {
            codec.kit.ctx.queue.write_buffer(&p.ctl, 0, bytemuck::bytes_of(&[0u32; 4]));
        }
    }

    pub fn pass_count(&self, frames: usize) -> Option<usize> {
        self.passes.get(&frames).map(Vec::len)
    }
}

pub struct VoxtralTtsCodecWgpu {
    kit: Kit,
    semantic: Buffer,
    input_conv: Lin,
    stages: [[Layer; 2]; 4],
    upsamplers: [Lin; 3],
    output_conv: Lin,
    full: Mutex<CodecState>,
}

fn to_bits(v: &[f32]) -> Vec<u16> {
    v.iter().map(|x| bf16::from_f32(*x).to_bits()).collect()
}

fn fold_weight_norm(g: &[f32], v: &[f32], rows: usize) -> Vec<f32> {
    let per = v.len() / rows;
    let mut out = Vec::with_capacity(v.len());
    for r in 0..rows {
        let row = &v[r * per..(r + 1) * per];
        let s = g[r] / row.iter().map(|x| x * x).sum::<f32>().sqrt();
        out.extend(row.iter().map(|x| x * s));
    }
    out
}

fn conv_gemm_rows(w: &[f32], out_ch: usize, in_ch: usize, taps: usize, kpad: usize) -> Vec<u16> {
    let mut rows = vec![0u16; out_ch * kpad];
    for o in 0..out_ch {
        for ci in 0..in_ch {
            for t in 0..taps {
                rows[o * kpad + t * in_ch + ci] = bf16::from_f32(w[(o * in_ch + ci) * taps + t]).to_bits();
            }
        }
    }
    rows
}

fn transpose_conv_as_causal_conv_rows(w_in_out_4: &[f32], ch: usize) -> Vec<u16> {
    let mut rows = vec![0u16; 2 * ch * 2 * ch];
    for j in 0..2 {
        for o in 0..ch {
            let row = (j * ch + o) * 2 * ch;
            for i in 0..ch {
                rows[row + i] = bf16::from_f32(w_in_out_4[(i * ch + o) * 4 + j + 2]).to_bits();
                rows[row + ch + i] = bf16::from_f32(w_in_out_4[(i * ch + o) * 4 + j]).to_bits();
            }
        }
    }
    rows
}

impl VoxtralTtsCodecWgpu {
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join(CODEC_WEIGHTS_FILE);
        Self::from_loader(&WeightLoader::open_file(&path, &Device::Cpu).with_context(|| format!("open {}", path.display()))?)
    }

    pub fn from_loader(w: &WeightLoader) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(err)?;
        let kit = Kit {
            ctx,
            cx: gemm_source(),
            rms: compose(wk::rmsnorm::WGSL),
            dummy: dispatch::storage_zeroed(ctx, "cx-dummy", 16),
        };
        let f32s = |name: &str| -> Result<Vec<f32>> {
            Ok(w.get(&format!("audio_tokenizer.{name}"), DType::F32)
                .with_context(|| format!("load audio_tokenizer.{name}"))?
                .flatten_all()?
                .to_vec1()?)
        };
        let bf = |name: &str| -> Result<Buffer> { Ok(kit.upload_bf16(name, &to_bits(&f32s(name)?))) };
        let lin = |label: &'static str, names: &[&str], n: usize, k: usize| -> Result<Lin> {
            let mut bits = Vec::with_capacity(n * k);
            for name in names {
                bits.extend(to_bits(&f32s(name)?));
            }
            kit.lin(label, &bits, n, k)
        };
        let folded = |prefix: &str, rows: usize| -> Result<Vec<f32>> {
            Ok(fold_weight_norm(
                &f32s(&format!("{prefix}.parametrizations.weight.original0"))?,
                &f32s(&format!("{prefix}.parametrizations.weight.original1"))?,
                rows,
            ))
        };
        let usage = f32s("quantizer.semantic_codebook.cluster_usage")?;
        let mut table = f32s("quantizer.semantic_codebook.embedding_sum")?;
        ensure!(table.len() == SEMANTIC_ROWS * SEMANTIC_DIM && usage.len() == SEMANTIC_ROWS, "semantic codebook shape");
        for (r, u) in usage.iter().enumerate() {
            let inv = 1.0 / u.max(CLUSTER_USAGE_FLOOR);
            for x in &mut table[r * SEMANTIC_DIM..(r + 1) * SEMANTIC_DIM] {
                *x *= inv;
            }
        }
        let semantic = kit.upload_bf16("cx-semantic", &to_bits(&table));
        let input_conv = kit.lin(
            "input_conv",
            &conv_gemm_rows(&folded("decoder_blocks.0.conv", DIM)?, DIM, LATENT, INPUT_CONV_TAPS, INPUT_CONV_K_876_PADS_TO_896_FOR_THE_32_WIDE_GEMM_K_STEP),
            DIM,
            INPUT_CONV_K_876_PADS_TO_896_FOR_THE_32_WIDE_GEMM_K_STEP,
        )?;
        let layer = |block: usize, l: usize| -> Result<Layer> {
            let p = format!("decoder_blocks.{block}.layers.{l}");
            let layer = Layer {
                attn_norm: bf(&format!("{p}.attention_norm.weight"))?,
                q_norm: bf(&format!("{p}.attention.q_norm.weight"))?,
                k_norm: bf(&format!("{p}.attention.k_norm.weight"))?,
                ffn_norm: bf(&format!("{p}.ffn_norm.weight"))?,
                attn_scale: kit.upload_f32("cx-attn-scale", &f32s(&format!("{p}.attention_scale"))?),
                ffn_scale: kit.upload_f32("cx-ffn-scale", &f32s(&format!("{p}.ffn_scale"))?),
                qkv: lin(
                    "qkv",
                    &[&format!("{p}.attention.wq.weight"), &format!("{p}.attention.wk.weight"), &format!("{p}.attention.wv.weight")],
                    3 * DIM,
                    DIM,
                )?,
                wo: lin("wo", &[&format!("{p}.attention.wo.weight")], DIM, DIM)?,
                gate_up: lin("gate_up", &[&format!("{p}.feed_forward.w1.weight"), &format!("{p}.feed_forward.w3.weight")], 2 * FFN, DIM)?,
                w2: lin("w2", &[&format!("{p}.feed_forward.w2.weight")], DIM, FFN)?,
            };
            ctx.queue.submit([]);
            ctx.poll_blocking().map_err(err)?;
            Ok(layer)
        };
        let stage = |b: usize| -> Result<[Layer; 2]> { Ok([layer(b, 0)?, layer(b, 1)?]) };
        let upsampler = |b: usize| -> Result<Lin> {
            kit.lin("upsampler", &transpose_conv_as_causal_conv_rows(&folded(&format!("decoder_blocks.{b}.conv"), DIM)?, DIM), 2 * DIM, 2 * DIM)
        };
        let codec = Self {
            semantic,
            input_conv,
            stages: [stage(STAGE_BLOCKS[0])?, stage(STAGE_BLOCKS[1])?, stage(STAGE_BLOCKS[2])?, stage(STAGE_BLOCKS[3])?],
            upsamplers: [upsampler(UPSAMPLER_BLOCKS[0])?, upsampler(UPSAMPLER_BLOCKS[1])?, upsampler(UPSAMPLER_BLOCKS[2])?],
            output_conv: kit.lin(
                "output_conv",
                &conv_gemm_rows(&folded("output_proj.conv", PATCH)?, PATCH, DIM, OUTPUT_CONV_TAPS, OUTPUT_CONV_TAPS * DIM),
                PATCH,
                OUTPUT_CONV_TAPS * DIM,
            )?,
            kit,
            full: Mutex::new(CodecState::default()),
        };
        ctx.queue.submit([]);
        ctx.poll_blocking().map_err(err)?;
        Ok(codec)
    }

    pub fn decode(&self, codes: &[i32]) -> Result<Vec<f32>> {
        dispatch::drive_blocking(self.decode_arm(codes, ReadbackArm::Blocking))
    }

    pub async fn decode_arm(&self, codes: &[i32], arm: ReadbackArm) -> Result<Vec<f32>> {
        let mut state = self.full.lock().map_err(|_| anyhow!("codec scratch state poisoned"))?;
        state.reset(self);
        self.decode_frames_arm(&mut state, frames_before_end_of_audio(codes), arm).await
    }

    pub fn decode_frames(&self, state: &mut CodecState, frames: &[i32]) -> Result<Vec<f32>> {
        dispatch::drive_blocking(self.decode_frames_arm(state, frames, ReadbackArm::Blocking))
    }

    pub async fn decode_frames_arm(&self, state: &mut CodecState, frames: &[i32], arm: ReadbackArm) -> Result<Vec<f32>> {
        ensure!(
            !frames.is_empty() && frames.len() % CODEC_FRAME_TOKENS == 0,
            "codec frames are [T, {CODEC_FRAME_TOKENS}] with T >= 1, got {} values",
            frames.len()
        );
        for frame in frames.chunks_exact(CODEC_FRAME_TOKENS) {
            let s = frame[0] - CODEC_N_AUDIO_SPECIAL;
            ensure!((0..SEMANTIC_ROWS as i32).contains(&s), "semantic code {} out of range", frame[0]);
            for &c in &frame[1..] {
                ensure!((0..ACOUSTIC_LEVELS).contains(&(c - CODEC_N_AUDIO_SPECIAL)), "acoustic code {c} out of range");
            }
        }
        let n = frames.len() / CODEC_FRAME_TOKENS;
        self.ensure_state(state, n)?;
        let (persist, scratch) = (state.persist.as_ref().unwrap(), state.scratch.as_ref().unwrap());
        self.kit.ctx.queue.write_buffer(&scratch.codes, 0, bytemuck::cast_slice(frames));
        if !state.passes.contains_key(&n) {
            let passes = self.build_passes(persist, scratch, n)?;
            state.passes.insert(n, passes);
        }
        self.kit.submit(&state.passes[&n])?;
        let words: Vec<u32> = dispatch::read_back_arm(self.kit.ctx, &scratch.out, n * 8 * PATCH / 2, arm).await.map_err(err)?;
        Ok(words
            .iter()
            .flat_map(|w| [f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000)])
            .map(|s| s.clamp(-1.0, 1.0))
            .collect())
    }

    fn gemm_shapes(&self, n: usize) -> Vec<(&Lin, usize)> {
        let mut v = vec![(&self.input_conv, n)];
        for (s, layers) in self.stages.iter().enumerate() {
            let m = n << s;
            for l in layers {
                v.extend([(&l.qkv, m), (&l.wo, m), (&l.gate_up, m), (&l.w2, m)]);
            }
            if s < 3 {
                v.push((&self.upsamplers[s], m));
            }
        }
        v.push((&self.output_conv, 8 * n));
        v
    }

    fn ensure_state(&self, state: &mut CodecState, n: usize) -> Result<()> {
        let kit = &self.kit;
        if state.persist.is_none() {
            state.persist = Some(Persist {
                ctl: kit.zeroed("cx-ctl", 4),
                lat_hist: kit.zeroed("cx-lat-hist", (INPUT_CONV_TAPS - 1) * LATENT / 2),
                rings: (0..8)
                    .map(|i| (kit.zeroed("cx-kring", STAGE_WINDOWS[i / 2] * DIM / 2), kit.zeroed("cx-vring", STAGE_WINDOWS[i / 2] * DIM / 2)))
                    .collect(),
                hist_up: [kit.zeroed("cx-hist-up", DIM / 2), kit.zeroed("cx-hist-up", DIM / 2), kit.zeroed("cx-hist-up", DIM / 2)],
                hist_out: kit.zeroed("cx-hist-out", (OUTPUT_CONV_TAPS - 1) * DIM / 2),
            });
        }
        if state.scratch.as_ref().is_some_and(|s| s.cap >= n) {
            return Ok(());
        }
        let cap = n.max(SCRATCH_GROWS_TO_THE_LARGEST_CALL_AND_STARTS_AT_THIS_MANY_FRAMES);
        let rows = 8 * cap;
        let part = (1..=cap)
            .flat_map(|f| self.gemm_shapes(f).into_iter().map(|(l, m)| kit.part_words(l, m)).collect::<Vec<_>>())
            .max()
            .unwrap_or(4);
        state.passes.clear();
        state.scratch = Some(Scratch {
            cap,
            codes: kit.zeroed("cx-codes", cap * CODEC_FRAME_TOKENS),
            lat: kit.zeroed("cx-lat", cap * LATENT / 2),
            col_in: kit.zeroed("cx-col-in", cap * INPUT_CONV_K_876_PADS_TO_896_FOR_THE_32_WIDE_GEMM_K_STEP / 2),
            x: [kit.zeroed("cx-x0", rows * DIM / 2), kit.zeroed("cx-x1", rows * DIM / 2)],
            h: kit.zeroed("cx-h", rows * DIM / 2),
            qkv: kit.zeroed("cx-qkv", rows * 3 * DIM / 2),
            qn: kit.zeroed("cx-qn", rows * DIM / 2),
            kn: kit.zeroed("cx-kn", rows * DIM / 2),
            o: kit.zeroed("cx-o", rows * DIM / 2),
            gu: kit.zeroed("cx-gu", rows * FFN),
            act: kit.zeroed("cx-act", rows * FFN / 2),
            col_up: kit.zeroed("cx-col-up", rows / 2 * DIM),
            col_out: kit.zeroed("cx-col-out", rows * OUTPUT_CONV_TAPS * DIM / 2),
            out: kit.zeroed("cx-out", rows * PATCH / 2),
            part: kit.zeroed("cx-part", part),
        });
        Ok(())
    }

    fn build_passes(&self, st: &Persist, sc: &Scratch, n: usize) -> Result<Vec<Pass>> {
        let kit = &self.kit;
        let kin = INPUT_CONV_K_876_PADS_TO_896_FOR_THE_32_WIDE_GEMM_K_STEP;
        let mut p = vec![
            kit.latent(&sc.codes, &self.semantic, &sc.lat, n)?,
            kit.im2col(&sc.lat, &st.lat_hist, &st.ctl, &sc.col_in, n, LATENT, INPUT_CONV_TAPS, kin, PAD_REPLICATE)?,
            kit.carry(&sc.col_in, &st.lat_hist, n, LATENT, INPUT_CONV_TAPS, kin)?,
        ];
        p.extend(kit.gemm(&sc.col_in, &self.input_conv, None, &sc.x[0], n, &sc.part)?);
        let mut cur = 0;
        for (s, layers) in self.stages.iter().enumerate() {
            let m = n << s;
            let (w, rpf) = (STAGE_WINDOWS[s], 1 << s);
            let x = &sc.x[cur];
            for (l, layer) in layers.iter().enumerate() {
                let ring = &st.rings[s * 2 + l];
                p.push(kit.rmsnorm(x, &layer.attn_norm, &sc.h, m, BLOCK_NORM_EPS)?);
                p.extend(kit.gemm(&sc.h, &layer.qkv, None, &sc.qkv, m, &sc.part)?);
                p.push(kit.norm_qk(&sc.qkv, layer, &sc.qn, &sc.kn, m)?);
                p.push(kit.attn(&sc.qn, &sc.kn, &sc.qkv, ring, &st.ctl, &sc.o, m, w, rpf)?);
                p.push(kit.kv_commit(&sc.kn, &sc.qkv, ring, &st.ctl, m, w, rpf)?);
                p.extend(kit.gemm(&sc.o, &layer.wo, Some(&layer.attn_scale), x, m, &sc.part)?);
                p.push(kit.rmsnorm(x, &layer.ffn_norm, &sc.h, m, BLOCK_NORM_EPS)?);
                p.extend(kit.gemm(&sc.h, &layer.gate_up, None, &sc.gu, m, &sc.part)?);
                p.push(kit.silu_mul(&sc.gu, &sc.act, m)?);
                p.extend(kit.gemm(&sc.act, &layer.w2, Some(&layer.ffn_scale), x, m, &sc.part)?);
            }
            if s < 3 {
                p.push(kit.im2col(x, &st.hist_up[s], &st.ctl, &sc.col_up, m, DIM, 2, 2 * DIM, PAD_ZERO)?);
                p.push(kit.carry(&sc.col_up, &st.hist_up[s], m, DIM, 2, 2 * DIM)?);
                p.extend(kit.gemm(&sc.col_up, &self.upsamplers[s], None, &sc.x[1 - cur], m, &sc.part)?);
                cur = 1 - cur;
            }
        }
        let m = 8 * n;
        let kout = OUTPUT_CONV_TAPS * DIM;
        p.push(kit.im2col(&sc.x[cur], &st.hist_out, &st.ctl, &sc.col_out, m, DIM, OUTPUT_CONV_TAPS, kout, PAD_REFLECT)?);
        p.push(kit.carry(&sc.col_out, &st.hist_out, m, DIM, OUTPUT_CONV_TAPS, kout)?);
        p.extend(kit.gemm(&sc.col_out, &self.output_conv, None, &sc.out, m, &sc.part)?);
        p.push(kit.advance(&st.ctl, n)?);
        Ok(p)
    }
}
