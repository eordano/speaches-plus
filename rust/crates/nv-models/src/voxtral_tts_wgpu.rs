use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, bail, ensure, Context, Result};
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_kernels::wgpu_backend::dispatch::{self, ReadbackArm};
use nv_kernels::wgpu_backend::{compose, WgpuContext};
use nv_weights::WeightLoader;
use serde::Deserialize;
use wgpu::Buffer;

use crate::pod;
use crate::voxtral::{VoxtralQuant, DECODER_ROPE_ROWS, VOXTRAL_QUANT_ENV};
use crate::voxtral_tekken::TekkenEncoder;
use crate::voxtral_tts::{
    bf16_timesteps, read_torch_tensor, sinusoidal_time_embedding, CodebookLayout, FrameTrace,
    Synthesis, AUDIO_TOKEN, BEGIN_AUDIO_TOKEN, BOS_TOKEN, DEFAULT_CFG_ALPHA, DEFAULT_NFE, EMPTY_AUDIO,
    END_AUDIO, NEXT_AUDIO_TEXT_TOKEN, N_AUDIO_SPECIAL, PARAMS_FILE, REPEAT_AUDIO_TEXT_TOKEN,
    TEKKEN_FILE, VOICE_DIR, WEIGHTS_FILE,
};
use crate::voxtral_wgpu::{
    err, rope_buffers, tensor_bits, Kit, Lin, Pass, DECODE_ATTENTION_SPLITS,
    GEMM_ACCUM, GEMM_OUT_F32, NV4_ACCUM, NV4_OUT_F32, VX_SHARED_WGSL,
};

pub const KV_CAPACITY_ROWS: usize = 4096;
pub const FLOW_ROWS: usize = 6;
pub const FLOW_TOKENS: usize = 3;
pub const INPUT_PROJ_K_PADDED_TO_THE_GEMM_K_STEP: usize = 64;
const VT_EULER_HARDCODES_THE_SPECIAL_IDS: () = assert!(EMPTY_AUDIO == 0 && END_AUDIO == 1 && N_AUDIO_SPECIAL == 2);

pub const EULER_AND_FSQ_RUN_ON_THE_DEVICE_WITH_ONE_BF16_ROUNDING_PER_OPERATION: &str =
    "vt_euler rounds alpha * v_cond, (1 - alpha) * v_uncond, their sum, the product with dt and \
     the sum with x_t to bf16 in that order, and the FSQ rounds (c + 1), * 0.5 and * levels the \
     same way before round-half-to-even, which is the arithmetic voxtral_tts.rs does on the host \
     in half::bf16, so the codes depend on the backend only through the velocity GEMMs";

pub const THE_TTS_TWIN_SERVES_BF16_BECAUSE_NVFP4_FAILS_THE_FRAME_LOOP_GATE: &str =
    "nvfp4 on the backbone and flow layers, quantized as the CUDA DecoderLinearBuilder does, \
     leaves the teacher-forced hidden states outside the relative band, flips semantic ids at \
     margins far above two bf16 ulps and moves acoustic codes by more than one level in the suite \
     covering the frame loop (tests/voxtral_tts_wgpu.rs), so NV_VOXTRAL_QUANT unset serves bf16 \
     here while the STT twin's default stays nvfp4; NV_VOXTRAL_QUANT=nvfp4 is the opt-in the \
     suite rejects";

pub const THE_SEMANTIC_ARGMAX_READS_BF16_LOGITS_SO_TIES_BREAK_AS_THE_REFERENCE_BREAKS_THEM: &str =
    "the reference's logits are bf16 GEMM outputs and its fixtures hold frames whose top two tie \
     in bf16; the head GEMV writes bf16 and vt_semantic_argmax takes the lowest index among equal \
     values over ids 1..semantic_rows, torch's first-index rule; it writes only ids[0], and the \
     FSQ pass copies it into the frame, so a free run submits the flow and the next step together \
     and the frame buffer still holds this frame when the host reads it";

pub const HEADS_ONLY_LOADS_NO_BACKBONE_LAYER_AND_NO_TEXT_EMBEDDING_SO_A_BROWSER_CAN_RUN_THE_FLOW_FROM_SERVED_HIDDEN_STATES: &str =
    "from_parts with with_backbone false uploads the final norm, both heads, the codebook embedding \
     and the flow head alone (under a gigabyte of bf16 against the backbone's six), builds the step \
     list over zero layers and refuses synthesize; frame_from_hidden then runs the heads and the nfe \
     Euler steps on a normed hidden state the caller supplies, the per-frame contract under which a \
     server-side backbone feeds a browser, and the pumped arm of every readback is what a browser \
     awaits where the native entry points drive the blocking arm";

pub const THE_FLOW_HEAD_SWEEPS_THE_SIX_CFG_ROWS_PER_WEIGHT_WORD_LIKE_THE_NVFP4_GEMV: &str =
    "an Euler step is three tokens for the conditional and three for the unconditional row, one \
     M=6 product per linear; vx_gemm at six rows runs compute-bound on its 64-row tile at several \
     times the one-row GEMV's time for the same bytes, so vt_gemv_rows reads each weight word once \
     and accumulates the six rows in registers, the bf16 analogue of vx_gemv_nv4's M sweep, and \
     its residual accumulate rounds the product to bf16 before the add, the reference's two \
     roundings; the prebuilt pass list of all nfe steps is submitted at once, so a frame is one submit for the \
     flow and one for the backbone step";

#[derive(Deserialize)]
struct Params {
    dim: usize,
    n_layers: usize,
    head_dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    rope_theta: f32,
    norm_eps: f64,
    vocab_size: usize,
    model_type: String,
    multimodal: Multimodal,
}

#[derive(Deserialize)]
struct Multimodal {
    audio_model_args: AudioModelArgs,
}

#[derive(Deserialize)]
struct AudioModelArgs {
    semantic_codebook_size: usize,
    acoustic_codebook_size: usize,
    n_acoustic_codebook: usize,
    acoustic_transformer_args: AcousticArgs,
}

#[derive(Deserialize)]
struct AcousticArgs {
    dim: usize,
    n_layers: usize,
    head_dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    #[serde(default)]
    n_decoding_steps: Option<usize>,
}

#[derive(Deserialize)]
struct TekkenJson {
    config: TekkenConfig,
    vocab: Vec<TekkenEntry>,
    audio: TekkenAudio,
}

#[derive(Deserialize)]
struct TekkenConfig {
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Deserialize)]
struct TekkenEntry {
    rank: usize,
    token_bytes: String,
}

#[derive(Deserialize)]
struct TekkenAudio {
    voice_num_audio_tokens: BTreeMap<String, usize>,
}

#[derive(Clone, Copy)]
struct Stack {
    d: usize,
    layers: usize,
    hd: usize,
    inter: usize,
    nh: usize,
    nkv: usize,
    eps: f32,
    theta: f32,
}

const VT_WGSL: &str = r#"
struct VtParams { words: u32, a: u32, b: u32, c: u32, f0: f32, f1: f32, f2: f32, f3: f32 };
struct VtEulerParams { n: u32, fsq: u32, semantic_rows: u32, acoustic_rows: u32, alpha: f32, one_minus_alpha: f32, dt: f32, levels: f32 };

fn vt_flat(wid: vec3<u32>, nwg: vec3<u32>, lid: vec3<u32>) -> u32 {
    return (wid.y * nwg.x + wid.x) * 256u + lid.x;
}

fn vt_rb(x: f32) -> f32 {
    return bitcast<f32>(bf16_encode(x) << 16u);
}

@group(0) @binding(0) var<storage, read> pr_embed: array<u32>;
@group(0) @binding(1) var<storage, read> pr_ids: array<u32>;
@group(0) @binding(2) var<storage, read> pr_voice: array<u32>;
@group(0) @binding(3) var<storage, read_write> pr_out: array<u32>;
@group(0) @binding(4) var<uniform> pr_p: VtParams;

@compute @workgroup_size(256)
fn vt_prompt_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = vt_flat(wid, nwg, lid);
    if (idx >= pr_p.a * pr_p.words) { return; }
    let r = idx / pr_p.words;
    let w = idx % pr_p.words;
    if (r >= pr_p.b && r < pr_p.b + pr_p.c) {
        pr_out[idx] = pr_voice[(r - pr_p.b) * pr_p.words + w];
    } else {
        pr_out[idx] = pr_embed[pr_ids[r] * pr_p.words + w];
    }
}

@group(0) @binding(5) var<storage, read> fe_table: array<u32>;
@group(0) @binding(6) var<storage, read> fe_ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> fe_out: array<u32>;
@group(0) @binding(8) var<uniform> fe_p: VtParams;

@compute @workgroup_size(256)
fn vt_frame_embed(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let w = vt_flat(wid, nwg, lid);
    if (w >= fe_p.words) { return; }
    var lo = 0.0;
    var hi = 0.0;
    for (var k = 0u; k < fe_p.a; k = k + 1u) {
        let v = fe_table[fe_ids[k] * fe_p.words + w];
        lo = lo + bf16_lo(v);
        hi = hi + bf16_hi(v);
    }
    fe_out[w] = bf16_pack(lo, hi);
}

@group(0) @binding(9) var<storage, read> sa_logits: array<u32>;
@group(0) @binding(10) var<storage, read_write> sa_ids: array<u32>;
@group(0) @binding(12) var<storage, read_write> sa_pos: array<i32>;
@group(0) @binding(13) var<uniform> sa_p: VtParams;

var<workgroup> sa_val: array<f32, 256>;
var<workgroup> sa_idx: array<u32, 256>;

@compute @workgroup_size(256)
fn vt_semantic_argmax(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    var best = bitcast<f32>(0xff800000u);
    var bidx = 0xffffffffu;
    for (var i = 1u + t; i < sa_p.a; i = i + 256u) {
        let w = sa_logits[i >> 1u];
        let v = select(bf16_lo(w), bf16_hi(w), (i & 1u) != 0u);
        if (v > best) { best = v; bidx = i; }
    }
    sa_val[t] = best;
    sa_idx[t] = bidx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            let ov = sa_val[t + s];
            let oi = sa_idx[t + s];
            if (ov > sa_val[t] || (ov == sa_val[t] && oi < sa_idx[t])) { sa_val[t] = ov; sa_idx[t] = oi; }
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        let id = select(sa_idx[0], 1u, sa_idx[0] == 0xffffffffu);
        sa_ids[0] = id;
        sa_pos[0] = sa_pos[0] + i32(sa_p.b);
    }
}

@group(0) @binding(14) var<storage, read> ft_xs: array<u32>;
@group(0) @binding(31) var<storage, read> ft_ip: array<u32>;
@group(0) @binding(15) var<storage, read> ft_time: array<u32>;
@group(0) @binding(16) var<storage, read> ft_cond: array<u32>;
@group(0) @binding(17) var<storage, read_write> ft_out: array<u32>;
@group(0) @binding(18) var<uniform> ft_p: VtParams;

@compute @workgroup_size(256)
fn vt_flow_tokens(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let idx = vt_flat(wid, nwg, lid);
    if (idx >= 6u * ft_p.words) { return; }
    let r = idx / ft_p.words;
    let w = idx % ft_p.words;
    let s = r % 3u;
    var v = 0u;
    if (s == 0u) {
        let kw = ft_p.b;
        var a0 = 0.0;
        var a1 = 0.0;
        for (var i = 0u; i < kw; i = i + 1u) {
            let xw = ft_xs[i];
            let w0 = ft_ip[2u * w * kw + i];
            let w1 = ft_ip[(2u * w + 1u) * kw + i];
            a0 = fma(bf16_hi(w0), bf16_hi(xw), fma(bf16_lo(w0), bf16_lo(xw), a0));
            a1 = fma(bf16_hi(w1), bf16_hi(xw), fma(bf16_lo(w1), bf16_lo(xw), a1));
        }
        v = bf16_pack(a0, a1);
    } else if (s == 1u) { v = ft_time[ft_p.a * ft_p.words + w]; } else if (r == 2u) { v = ft_cond[w]; }
    ft_out[idx] = v;
}

@group(0) @binding(19) var<storage, read> fa_qkv: array<u32>;
@group(0) @binding(20) var<storage, read_write> fa_out: array<u32>;
@group(0) @binding(21) var<uniform> fa_p: VtParams;

var<workgroup> fa_red: array<f32, 192>;

@compute @workgroup_size(64)
fn vt_flow_attn(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let r = wid.x;
    let h = wid.y;
    let t = lid.x;
    let words = fa_p.words;
    let b = r / 3u;
    let kvh = h / (fa_p.a / fa_p.b);
    let stride = (fa_p.a + 2u * fa_p.b) * words;
    let qw = fa_qkv[r * stride + h * words + t];
    let q0 = bf16_lo(qw);
    let q1 = bf16_hi(qw);
    for (var j = 0u; j < 3u; j = j + 1u) {
        let kw = fa_qkv[(b * 3u + j) * stride + (fa_p.a + kvh) * words + t];
        fa_red[j * 64u + t] = fma(q0, bf16_lo(kw), q1 * bf16_hi(kw));
    }
    workgroupBarrier();
    var s = vec3<f32>(0.0);
    for (var i = 0u; i < 64u; i = i + 1u) {
        s = s + vec3<f32>(fa_red[i], fa_red[64u + i], fa_red[128u + i]);
    }
    s = s * fa_p.f0;
    let m = max(s.x, max(s.y, s.z));
    let e = exp(s - vec3<f32>(m));
    let p = e / (e.x + e.y + e.z);
    var o0 = 0.0;
    var o1 = 0.0;
    for (var j = 0u; j < 3u; j = j + 1u) {
        let vw = fa_qkv[(b * 3u + j) * stride + (fa_p.a + fa_p.b + kvh) * words + t];
        o0 = fma(p[j], bf16_lo(vw), o0);
        o1 = fma(p[j], bf16_hi(vw), o1);
    }
    let o = (r * fa_p.a + h) * words + t;
    fa_out[select(o, nv_perm_word(o), fa_p.c != 0u)] = bf16_pack(o0, o1);
}

@group(0) @binding(22) var<storage, read> eu_v: array<u32>;
@group(0) @binding(23) var<storage, read_write> eu_x: array<u32>;
@group(0) @binding(24) var<storage, read_write> eu_frame: array<u32>;
@group(0) @binding(25) var<storage, read_write> eu_ids: array<u32>;
@group(0) @binding(26) var<uniform> eu_p: VtEulerParams;

fn vt_step(x: f32, vc: f32, vu: f32) -> f32 {
    let vt = vt_rb(vt_rb(eu_p.alpha * vc) + vt_rb(eu_p.one_minus_alpha * vu));
    return vt_rb(x + vt_rb(vt * eu_p.dt));
}

fn vt_code(x: f32) -> u32 {
    let c = clamp(x, -1.0, 1.0);
    let y = vt_rb(vt_rb(vt_rb(c + 1.0) * 0.5) * eu_p.levels);
    return u32(clamp(round(y), 0.0, eu_p.levels));
}

@compute @workgroup_size(64)
fn vt_euler(@builtin(local_invocation_id) lid: vec3<u32>) {
    let w = lid.x;
    let words = eu_p.n >> 1u;
    if (w >= words) { return; }
    let vc = eu_v[w];
    let vu = eu_v[3u * words + w];
    let xw = eu_x[w];
    let x0 = vt_step(bf16_lo(xw), bf16_lo(vc), bf16_lo(vu));
    let x1 = vt_step(bf16_hi(xw), bf16_hi(vc), bf16_hi(vu));
    eu_x[w] = bf16_pack(x0, x1);
    if (eu_p.fsq != 0u) {
        let sem = eu_ids[0];
        let end = sem == 1u;
        if (w == 0u) { eu_frame[0] = sem; }
        let c0 = select(vt_code(x0), 0u, end) + 2u;
        let c1 = select(vt_code(x1), 0u, end) + 2u;
        let k0 = 1u + 2u * w;
        eu_frame[k0] = c0;
        eu_frame[k0 + 1u] = c1;
        eu_ids[k0] = eu_p.semantic_rows + (k0 - 1u) * eu_p.acoustic_rows + c0;
        eu_ids[k0 + 1u] = eu_p.semantic_rows + k0 * eu_p.acoustic_rows + c1;
    }
}
"#;

const VT_GEMV_WGSL: &str = r#"
struct VtGemvParams { n_rows: u32, k_vec: u32, groups_x: u32, flags: u32, x_off_vec4: u32, sg_per_unit: u32, m: u32, eps: f32 };
@group(0) @binding(27) var<storage, read> vg_w: array<vec4<u32>>;
@group(0) @binding(28) var<storage, read> vg_x: array<vec4<u32>>;
@group(0) @binding(29) var<storage, read_write> vg_y: array<u32>;
@group(0) @binding(30) var<uniform> vg_p: VtGemvParams;
@group(0) @binding(32) var<storage, read> vg_nw: array<vec4<u32>>;
Q8_DECL

const VG_M: u32 = MVALu;
const VG_R: u32 = RVALu;
const VG_XV: u32 = XVALu;
var<workgroup> vg_red: array<f32, 16u * VG_M * VG_R>;
var<workgroup> vg_ss: array<f32, 256>;

fn vt_rb(x: f32) -> f32 {
    return bitcast<f32>(bf16_encode(x) << 16u);
}

fn vt_silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn vt_dot8(w: vec4<u32>, x: vec4<u32>, a: f32) -> f32 {
    var acc = fma(bf16_lo(w.x), bf16_lo(x.x), a);
    acc = fma(bf16_hi(w.x), bf16_hi(x.x), acc);
    acc = fma(bf16_lo(w.y), bf16_lo(x.y), acc);
    acc = fma(bf16_hi(w.y), bf16_hi(x.y), acc);
    acc = fma(bf16_lo(w.z), bf16_lo(x.z), acc);
    acc = fma(bf16_hi(w.z), bf16_hi(x.z), acc);
    acc = fma(bf16_lo(w.w), bf16_lo(x.w), acc);
    acc = fma(bf16_hi(w.w), bf16_hi(x.w), acc);
    return acc;
}

SUM_FN

@compute @workgroup_size(256)
fn vt_gemv(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>SG_BUILTINS) {
    NOSG_IDS
    let spp = vg_p.sg_per_unit;
    let unit = (wid.x + wid.y * vg_p.groups_x) * ((256u / sg_size) / spp) + sgid / spp;
    let part = sgid % spp;
    let kv = vg_p.k_vec;
    let gu = (vg_p.flags & 4u) != 0u;
    let nrm = (vg_p.flags & 16u) != 0u;
    let cols = select(vg_p.n_rows, vg_p.n_rows >> 1u, gu);
    let r0 = unit * 2u;
    let live = r0 < cols;
    let kend = select(0u, kv, live);
    let xb = vg_p.x_off_vec4;
    let wb0 = r0 * kv;
    let wb1 = wb0 + kv;
    let wb2 = (r0 + cols) * kv;
    let wb3 = wb2 + kv;
    let stride = VG_M * VG_R;
    var rstd = 0.0;
    if (nrm) {
        var ss = 0.0;
        for (var i = lid.x; i < kv * VG_XV; i = i + 256u) { ss = ss + nv_sq4(vg_x[xb + i]); }
        vg_ss[lid.x] = ss;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (lid.x < s) { vg_ss[lid.x] = vg_ss[lid.x] + vg_ss[lid.x + s]; }
            workgroupBarrier();
        }
        rstd = inverseSqrt(vg_p.eps + nv_div_rn(vg_ss[0], f32(kv * 8u * VG_XV)));
    }
    ACC_DECL
    for (var v = part * sg_size + lane; v < kend; v = v + spp * sg_size) {
        W_LOADS
        ACC_ROWS
    }
    ACC_REDUCE
    workgroupBarrier();
    if (lane != 0u || part != 0u || !live) { return; }
    for (var m = 0u; m < vg_p.m; m = m + 1u) {
        var s0 = 0.0;
        var s1 = 0.0;
        var s2 = 0.0;
        var s3 = 0.0;
        for (var q = 0u; q < spp; q = q + 1u) {
            let b = (sgid + q) * stride + m * VG_R;
            s0 = s0 + vg_red[b];
            s1 = s1 + vg_red[b + 1u];
            if (VG_R == 4u) {
                s2 = s2 + vg_red[b + 2u];
                s3 = s3 + vg_red[b + 3u];
            }
        }
        if (gu) {
            vg_y[m * (cols >> 1u) + unit] = bf16_pack(vt_silu(vt_rb(s0)) * vt_rb(s2), vt_silu(vt_rb(s1)) * vt_rb(s3));
        } else if ((vg_p.flags & 1u) != 0u) {
            vg_y[m * cols + r0] = bitcast<u32>(s0);
            vg_y[m * cols + r0 + 1u] = bitcast<u32>(s1);
        } else {
            let o = m * (cols >> 1u) + unit;
            if ((vg_p.flags & 2u) != 0u) {
                let old = vg_y[o];
                s0 = vt_rb(s0) + bf16_lo(old);
                s1 = vt_rb(s1) + bf16_hi(old);
            }
            vg_y[o] = bf16_pack(s0, s1);
        }
    }
}
"#;

const VT_Q8_WGSL: &str = r#"
@group(0) @binding(33) var<storage, read> vg_s: array<u32>;

fn vt_q8s(sb: u32, v: u32) -> f32 {
    return unpack2x16float(vg_s[sb + (v >> 2u)])[(v >> 1u) & 1u];
}

fn vt_i8x4(w: u32) -> vec4<f32> {
    return vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}

fn vt_dot4(q: vec4<f32>, xa: u32, xb: u32, a: f32) -> f32 {
    var acc = fma(q.x, bf16_lo(xa), a);
    acc = fma(q.y, bf16_hi(xa), acc);
    acc = fma(q.z, bf16_lo(xb), acc);
    acc = fma(q.w, bf16_hi(xb), acc);
    return acc;
}

fn vt_dot16(w: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    var d = vt_dot4(vt_i8x4(w.x), xa.x, xa.y, 0.0);
    d = vt_dot4(vt_i8x4(w.y), xa.z, xa.w, d);
    d = vt_dot4(vt_i8x4(w.z), xb.x, xb.y, d);
    d = vt_dot4(vt_i8x4(w.w), xb.z, xb.w, d);
    return d;
}
"#;

const VG_OUT_F32: u32 = 1;
const VG_ACCUM: u32 = 2;
const VG_GATE_UP_SILU: u32 = 4;
const VG_NORM_PROLOGUE: u32 = 16;

pub const THE_M_ROW_GEMV_READS_VEC4_WEIGHTS_PER_SUBGROUP_LANE_AND_FUSES_SILU_AND_THE_RESIDUAL: &str =
    "vt_gemv is generated per (M, R): a unit of sg_per_unit subgroups owns R weight rows (two \
     adjacent rows, or the gate and up rows of two columns when R is four) and every lane reads \
     one 16-byte vec4 of each row per step, so a 32-lane subgroup reads 512 contiguous bytes per \
     load and the M input rows come from L1; the lanes meet in subgroupAdd and the parts in \
     workgroup memory; the epilogue writes f32 for the rope, rounds the product to bf16 before a \
     residual add, or applies silu(gate) * up, which removes the silu and add passes of a layer; \
     an ungranted device gets the same kernel with eight 32-lane virtual warps summing in \
     workgroup memory";

fn gemv_source(m: usize, r: usize, q8: bool, sg: bool) -> String {
    let _ = THE_M_ROW_GEMV_READS_VEC4_WEIGHTS_PER_SUBGROUP_LANE_AND_FUSES_SILU_AND_THE_RESIDUAL;
    let rows = |n: usize, f: &dyn Fn(usize) -> String| (0..n).map(f).collect::<Vec<_>>().join("\n        ");
    let acc = |j: usize, i: usize| format!("acc{j}_{i}");
    let src = VT_GEMV_WGSL
        .replace("ACC_DECL", &rows(m * r, &|k| format!("var {} = 0.0;", acc(k / m, k % m))))
        .replace("Q8_DECL", if q8 { VT_Q8_WGSL } else { "" })
        .replace("XVAL", if q8 { "2" } else { "1" })
        .replace(
            "W_LOADS",
            &rows(r, &|j| {
                if q8 {
                    format!("let w{j} = vg_w[wb{j} + v]; let s{j} = vt_q8s(wb{j} >> 2u, v);")
                } else {
                    format!("let w{j} = vg_w[wb{j} + v];")
                }
            }),
        )
        .replace(
            "ACC_ROWS",
            &rows(m, &|i| {
                if q8 {
                    let fm = (0..r).map(|j| format!("{a} = fma(s{j}, vt_dot16(w{j}, x{i}a, x{i}b), {a});", a = acc(j, i))).collect::<Vec<_>>().join(" ");
                    format!("let x{i}a = vg_x[xb + ({i}u * kv + v) * 2u]; let x{i}b = vg_x[xb + ({i}u * kv + v) * 2u + 1u]; {fm}")
                } else {
                    let fm = (0..r).map(|j| format!("{a} = vt_dot8(w{j}, x{i}, {a});", a = acc(j, i))).collect::<Vec<_>>().join(" ");
                    format!("var x{i} = vg_x[xb + {i}u * kv + v]; if (nrm) {{ x{i} = nv_norm4(x{i}, vg_nw[v], rstd); }} {fm}")
                }
            }),
        )
        .replace(
            "ACC_REDUCE",
            &rows(m * r, &|k| {
                let (j, i) = (k / m, k % m);
                format!(
                    "let t{j}_{i} = vt_sum({a}, lid.x, lane); if (lane == 0u) {{ vg_red[sgid * stride + {}u] = t{j}_{i}; }}",
                    i * r + j,
                    a = acc(j, i)
                )
            }),
        )
        .replace("MVAL", &m.to_string())
        .replace("RVAL", &r.to_string());
    if sg {
        src.replace("SUM_FN", "fn vt_sum(v: f32, tid: u32, lane: u32) -> f32 { return subgroupAdd(v); }")
            .replace("SG_BUILTINS", ", @builtin(subgroup_size) sg_size: u32, @builtin(subgroup_id) sgid: u32, @builtin(subgroup_invocation_id) lane: u32")
            .replace("NOSG_IDS", "")
    } else {
        src.replace(
            "SUM_FN",
            "var<workgroup> vg_warp: array<f32, 256>;\nfn vt_sum(v: f32, tid: u32, lane: u32) -> f32 {\n    vg_warp[tid] = v;\n    workgroupBarrier();\n    for (var s = 16u; s > 0u; s = s >> 1u) {\n        if (lane < s) { vg_warp[tid] = vg_warp[tid] + vg_warp[tid + s]; }\n        workgroupBarrier();\n    }\n    let r = vg_warp[tid - lane];\n    workgroupBarrier();\n    return r;\n}",
        )
        .replace("SG_BUILTINS", "")
        .replace("NOSG_IDS", "let sg_size = 32u;\n    let sgid = lid.x >> 5u;\n    let lane = lid.x & 31u;")
    }
}

pod! {
    #[derive(Debug)]
    struct VtParams { words: u32, a: u32, b: u32, c: u32, f0: f32, f1: f32, f2: f32, f3: f32 }
    #[derive(Debug)]
    struct VtEulerParams { n: u32, fsq: u32, semantic_rows: u32, acoustic_rows: u32, alpha: f32, one_minus_alpha: f32, dt: f32, levels: f32 }
    #[derive(Debug)]
    struct VtGemvParams { n_rows: u32, k_vec: u32, groups_x: u32, flags: u32, x_off_vec4: u32, sg_per_unit: u32, m: u32, eps: f32 }
}

struct Layer<L = Lin> {
    attn_norm: Buffer,
    qkv: L,
    o: L,
    ffn_norm: Buffer,
    gate_up: L,
    down: L,
    kcache: Buffer,
    vcache: Buffer,
}

struct Q8 {
    w: Buffer,
    s: Buffer,
    n: usize,
    k: usize,
    name: &'static str,
}

enum FlowLin {
    Bf16(Lin),
    Q8(Q8),
}

enum GemvW<'a> {
    Bf16(&'a Lin),
    Q8(&'a Q8),
}

impl FlowLin {
    fn lin(&self) -> Result<&Lin> {
        match self {
            Self::Bf16(l) => Ok(l),
            Self::Q8(q) => bail!("{}: the q8_0 flow head has no nvfp4 arm", q.name),
        }
    }

    fn w(&self) -> GemvW<'_> {
        match self {
            Self::Bf16(l) => GemvW::Bf16(l),
            Self::Q8(q) => GemvW::Q8(q),
        }
    }
}

pub const THE_Q8_0_FLOW_HEAD_STAYS_AN_OPT_IN_BECAUSE_THE_ES_FIXTURE_LEAVES_ITS_LAST_EULER_STEP_OUTSIDE_THE_FLOW_BAND: &str =
    "q8_0 flow rows (ggml's block-32 absmax scales) read half the bytes per Euler evaluation and \
     score inside the paired per-row noise of bf16 in the suite covering corpus word errors \
     (wgpu_corpus_word_errors_reproduce_the_cuda_eager_instrument), but the frame-loop suite's \
     es fixture puts the unconditional velocity of the last Euler step just outside the relative \
     band while en stays inside, so the default flow rows stay bf16 and NV_VOXTRAL_TTS_FLOW_QUANT=q8_0 \
     is the opt-in";

pub const FLOW_QUANT_ENV: &str = "NV_VOXTRAL_TTS_FLOW_QUANT";
pub const Q8_0_BLOCK: usize = 32;

fn q8_rows(bits: &[u16], k: usize) -> (Vec<u32>, Vec<u32>) {
    let blocks = bits.len() / Q8_0_BLOCK;
    let mut w = vec![0u32; bits.len() / 4];
    let mut s = vec![0u32; blocks.div_ceil(2)];
    for (b, block) in bits.chunks(Q8_0_BLOCK).enumerate() {
        let x: Vec<f32> = block.iter().map(|&v| bf16::from_bits(v).to_f32()).collect();
        let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
        let d = amax / 127.0;
        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
        s[b / 2] |= (half::f16::from_f32(d).to_bits() as u32) << (16 * (b % 2));
        for (j, v) in x.iter().enumerate() {
            let q = (v * id).round().clamp(-127.0, 127.0) as i8;
            let e = b * Q8_0_BLOCK + j;
            w[e / 4] |= (q as u8 as u32) << (8 * (e % 4));
        }
    }
    assert!(k % Q8_0_BLOCK == 0 && (k / Q8_0_BLOCK) % 2 == 0, "q8_0 rows need an even block count, k={k}");
    (w, s)
}

struct Scratch {
    x: Buffer,
    h: Buffer,
    xn: Buffer,
    qkv: Buffer,
    q: Buffer,
    attn: Buffer,
    ao: Buffer,
    gu: Buffer,
    act: Buffer,
}

pub struct VoxtralTtsWgpu {
    kit: Kit,
    src: String,
    gemv_src: BTreeMap<(usize, usize, bool), String>,
    quant: VoxtralQuant,
    bb: Stack,
    fl: Stack,
    codebooks: CodebookLayout,
    encoder: TekkenEncoder,
    voices: BTreeMap<String, usize>,
    dir: Option<PathBuf>,
    embed: Buffer,
    codebook_embed: Buffer,
    layers: Vec<Layer>,
    norm: Buffer,
    cos: Buffer,
    sin: Buffer,
    semantic_head: Lin,
    llm_proj: Lin,
    input_proj: Lin,
    flow_quant: VoxtralQuant,
    flow_layers: Vec<Layer<FlowLin>>,
    flow_norm: Buffer,
    out: Lin,
    time_tokens: Buffer,
    nfe: usize,
    cfg_alpha: bf16,
    dts: Vec<bf16>,
    voice_cache: Mutex<BTreeMap<String, Buffer>>,
    pos: Buffer,
    zero_pos: Buffer,
    ids: Buffer,
    frame: Buffer,
    hn: Buffer,
    logits: Buffer,
    cond: Buffer,
    x_state: Buffer,
    vf: Buffer,
    step_x: Option<Buffer>,
    step_passes: Vec<Pass>,
    flow_passes: Vec<Pass>,
    hidden_passes: Vec<Pass>,
    flow_eval_len: usize,
    validated: bool,
}

fn neox_row_permutation(rows: usize, head_dim: usize) -> Vec<u32> {
    let half = head_dim / 2;
    (0..rows)
        .map(|r| {
            let (head, i) = (r / head_dim, r % head_dim);
            let src = if i < half { 2 * i } else { 2 * (i - half) + 1 };
            (head * head_dim + src) as u32
        })
        .collect()
}

fn unpack_bf16(words: &[u32]) -> Vec<f32> {
    words
        .iter()
        .flat_map(|w| [f32::from_bits(w << 16), f32::from_bits(w & 0xffff_0000)])
        .collect()
}

fn pack_bf16(v: &[f32], words: usize) -> Vec<u32> {
    let mut out = vec![0u32; words];
    for (i, x) in v.iter().enumerate() {
        out[i / 2] |= (bf16::from_f32(*x).to_bits() as u32) << (16 * (i % 2));
    }
    out
}

fn get(w: &WeightLoader, name: &str, shape: &[usize]) -> Result<Tensor> {
    let t = w.get(name, DType::BF16).with_context(|| format!("load {name}"))?;
    ensure!(t.dims() == shape, "{name}: shape {:?} != {shape:?}", t.dims());
    Ok(t)
}

fn bits(w: &WeightLoader, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    tensor_bits(&get(w, name, shape)?)
}

impl VoxtralTtsWgpu {
    pub fn quant_from_env() -> Result<VoxtralQuant> {
        let _ = THE_TTS_TWIN_SERVES_BF16_BECAUSE_NVFP4_FAILS_THE_FRAME_LOOP_GATE;
        match std::env::var(VOXTRAL_QUANT_ENV).ok().as_deref() {
            None | Some("") | Some("bf16") => Ok(VoxtralQuant::Bf16),
            Some("nvfp4") => Ok(VoxtralQuant::Nvfp4),
            Some(other) => bail!("{VOXTRAL_QUANT_ENV}={other:?}: the TTS twin serves bf16 or nvfp4"),
        }
    }

    pub fn flow_quant_from_env() -> Result<VoxtralQuant> {
        match std::env::var(FLOW_QUANT_ENV).ok().as_deref() {
            None | Some("") | Some("bf16") => Ok(VoxtralQuant::Bf16),
            Some("q8_0") => {
                let _ = THE_Q8_0_FLOW_HEAD_STAYS_AN_OPT_IN_BECAUSE_THE_ES_FIXTURE_LEAVES_ITS_LAST_EULER_STEP_OUTSIDE_THE_FLOW_BAND;
                Ok(VoxtralQuant::GgmlQ8_0)
            }
            Some("nvfp4") => Ok(VoxtralQuant::Nvfp4),
            Some(other) => bail!("{FLOW_QUANT_ENV}={other:?}: the flow head serves bf16, q8_0 or nvfp4"),
        }
    }

    pub fn flow_quant(&self) -> VoxtralQuant {
        self.flow_quant
    }

    pub fn load_dir(dir: &Path) -> Result<Self> {
        Self::load_dir_with(dir, Self::quant_from_env()?, Self::flow_quant_from_env()?, DEFAULT_NFE, DEFAULT_CFG_ALPHA)
    }

    pub fn load_dir_with(dir: &Path, quant: VoxtralQuant, flow_quant: VoxtralQuant, nfe: usize, cfg_alpha: f32) -> Result<Self> {
        let read = |f: &str| {
            std::fs::read_to_string(dir.join(f)).with_context(|| format!("read {}", dir.join(f).display()))
        };
        let w = WeightLoader::open_file(dir.join(WEIGHTS_FILE), &Device::Cpu)
            .with_context(|| format!("open {}", dir.join(WEIGHTS_FILE).display()))?;
        Self::from_parts(&read(PARAMS_FILE)?, &read(TEKKEN_FILE)?, &w, Some(dir), quant, flow_quant, nfe, cfg_alpha, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        params_json: &str,
        tekken_json: &str,
        w: &WeightLoader,
        voice_dir: Option<&Path>,
        quant: VoxtralQuant,
        flow_quant: VoxtralQuant,
        nfe: usize,
        cfg_alpha: f32,
        with_backbone: bool,
    ) -> Result<Self> {
        let _ = HEADS_ONLY_LOADS_NO_BACKBONE_LAYER_AND_NO_TEXT_EMBEDDING_SO_A_BROWSER_CAN_RUN_THE_FLOW_FROM_SERVED_HIDDEN_STATES;
        let params: Params = serde_json::from_str(params_json).context("parse params.json")?;
        ensure!(params.model_type == "voxtral_tts", "params.json model_type {:?} is not voxtral_tts", params.model_type);
        let audio = &params.multimodal.audio_model_args;
        let ac = &audio.acoustic_transformer_args;
        let bb = Stack {
            d: params.dim,
            layers: params.n_layers,
            hd: params.head_dim,
            inter: params.hidden_dim,
            nh: params.n_heads,
            nkv: params.n_kv_heads,
            eps: params.norm_eps as f32,
            theta: params.rope_theta,
        };
        let fl = Stack { d: ac.dim, layers: ac.n_layers, hd: ac.head_dim, inter: ac.hidden_dim, nh: ac.n_heads, nkv: ac.n_kv_heads, ..bb };
        ensure!(bb.hd == 128 && fl.hd == 128 && fl.d == bb.d, "the attention kernels are instantiated for head_dim 128 and one width");
        let codebooks = CodebookLayout {
            semantic_size: audio.semantic_codebook_size,
            acoustic_levels: audio.acoustic_codebook_size,
            n_acoustic: audio.n_acoustic_codebook,
        };
        let nfe = ac.n_decoding_steps.unwrap_or(nfe);
        let tekken: TekkenJson = serde_json::from_str(tekken_json).context("parse tekken.json")?;
        let n_special = tekken.config.default_num_special_tokens;
        let n_regular = tekken.config.default_vocab_size - n_special;
        let mut token_bytes = vec![Vec::new(); n_regular];
        for e in tekken.vocab.iter().filter(|e| e.rank < n_regular) {
            token_bytes[e.rank] = base64::engine::general_purpose::STANDARD
                .decode(&e.token_bytes)
                .with_context(|| format!("tekken rank {} token_bytes is not base64", e.rank))?;
        }
        let encoder = TekkenEncoder::new(&token_bytes, n_special)?;

        let ctx = WgpuContext::shared().map_err(|e| anyhow!("no wgpu adapter: {e}"))?;
        let kit = Kit::new(ctx);
        let src = compose(&format!("{VX_SHARED_WGSL}{VT_WGSL}"));
        let gemv_src = [1, FLOW_ROWS]
            .into_iter()
            .flat_map(|m| [(2, false), (4, false), (2, true), (4, true)].map(|(r, q8)| ((m, r, q8), compose(&format!("{VX_SHARED_WGSL}{}", gemv_source(m, r, q8, ctx.caps.subgroup))))))
            .collect();
        let d = bb.d;
        let embed = if with_backbone {
            kit.upload_bf16("vt-embed", &bits(w, "mm_audio_embeddings.tok_embeddings.weight", &[params.vocab_size, d])?)
        } else {
            kit.zeroed("vt-embed", 16)
        };
        kit.release_upload_staging()?;
        let cb = w.get("mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight", DType::BF16)?;
        ensure!(
            cb.dims().len() == 2 && cb.dims()[1] == d && cb.dims()[0] >= codebooks.rows_used(),
            "audio_codebook_embeddings {:?} holds fewer than the {} rows the 37 codebooks address",
            cb.dims(),
            codebooks.rows_used()
        );
        let codebook_embed = kit.upload_bf16("vt-codebook-embed", &tensor_bits(&cb)?);
        let bf16_lin = |label: &'static str, bits: &[u16], n: usize, k: usize| kit.lin_quant(label, bits, n, k, quant);
        let flow_lin = |label: &'static str, bits: &[u16], n: usize, k: usize| -> Result<FlowLin> {
            if flow_quant != VoxtralQuant::GgmlQ8_0 {
                return Ok(FlowLin::Bf16(kit.lin_quant(label, bits, n, k, flow_quant)?));
            }
            let (w, s) = q8_rows(bits, k);
            Ok(FlowLin::Q8(Q8 { w: kit.upload_u32(label, &w), s: kit.upload_u32(label, &s), n, k, name: label }))
        };
        fn load_stack<L>(
            kit: &Kit,
            w: &WeightLoader,
            prefix: &str,
            t: &Stack,
            neox: bool,
            kv_rows_cap: usize,
            n_layers: usize,
            lin: &dyn Fn(&'static str, &[u16], usize, usize) -> Result<L>,
        ) -> Result<(Vec<Layer<L>>, Buffer)> {
            let cpu = Device::Cpu;
            let (q_rows, kv_rows) = (t.nh * t.hd, t.nkv * t.hd);
            let mut layers = Vec::with_capacity(n_layers);
            for i in 0..n_layers {
                let p = format!("{prefix}layers.{i}.");
                let mut q = get(w, &format!("{p}attention.wq.weight"), &[q_rows, t.d])?;
                let mut k = get(w, &format!("{p}attention.wk.weight"), &[kv_rows, t.d])?;
                if neox {
                    let _ = crate::voxtral_tts::Q_AND_K_ROWS_ARE_PERMUTED_TO_NEOX_ORDER_SO_THE_STT_ROPE_KERNEL_ROTATES_ADJACENT_PAIRS;
                    let perm = |rows: usize| Tensor::new(neox_row_permutation(rows, t.hd).as_slice(), &cpu);
                    q = (q.index_select(&perm(q_rows)?, 0)?.to_dtype(DType::F32)? * (1.0 / (t.hd as f64).sqrt()))?
                        .to_dtype(DType::BF16)?;
                    k = k.index_select(&perm(kv_rows)?, 0)?;
                }
                let v = get(w, &format!("{p}attention.wv.weight"), &[kv_rows, t.d])?;
                let gate_up = Tensor::cat(
                    &[
                        get(w, &format!("{p}feed_forward.w1.weight"), &[t.inter, t.d])?,
                        get(w, &format!("{p}feed_forward.w3.weight"), &[t.inter, t.d])?,
                    ],
                    0,
                )?;
                layers.push(Layer {
                    attn_norm: kit.upload_bf16("vt-attn-norm", &bits(w, &format!("{p}attention_norm.weight"), &[t.d])?),
                    qkv: lin("vt-qkv", &tensor_bits(&Tensor::cat(&[q, k, v], 0)?)?, q_rows + 2 * kv_rows, t.d)?,
                    o: lin("vt-o", &bits(w, &format!("{p}attention.wo.weight"), &[t.d, q_rows])?, t.d, q_rows)?,
                    ffn_norm: kit.upload_bf16("vt-ffn-norm", &bits(w, &format!("{p}ffn_norm.weight"), &[t.d])?),
                    gate_up: lin("vt-gu", &tensor_bits(&gate_up)?, 2 * t.inter, t.d)?,
                    down: lin("vt-down", &bits(w, &format!("{p}feed_forward.w2.weight"), &[t.d, t.inter])?, t.d, t.inter)?,
                    kcache: kit.zeroed("vt-kcache", kv_rows_cap * kv_rows * 2),
                    vcache: kit.zeroed("vt-vcache", kv_rows_cap * kv_rows * 2),
                });
                kit.release_upload_staging()?;
            }
            Ok((layers, kit.upload_bf16("vt-norm", &bits(w, &format!("{prefix}norm.weight"), &[t.d])?)))
        }
        let (layers, norm) = load_stack(&kit, w, "", &bb, true, KV_CAPACITY_ROWS, if with_backbone { bb.layers } else { 0 }, &bf16_lin)?;
        let fp = "acoustic_transformer.";
        let (flow_layers, flow_norm) = load_stack(&kit, w, fp, &fl, false, 1, fl.layers, &flow_lin)?;
        let (cos, sin) = rope_buffers(&kit, "vt-rope", bb.hd, bb.theta, DECODER_ROPE_ROWS);
        let semantic_rows = codebooks.semantic_rows().div_ceil(128) * 128;
        let semantic_head = kit.lin("vt-semantic-head", &bits(&w, &format!("{fp}semantic_codebook_output.weight"), &[semantic_rows, d])?, semantic_rows, d, None)?;
        let llm_proj = kit.lin("vt-llm-proj", &bits(&w, &format!("{fp}llm_projection.weight"), &[d, d])?, d, d, None)?;
        let n = codebooks.n_acoustic;
        let kp = INPUT_PROJ_K_PADDED_TO_THE_GEMM_K_STEP;
        ensure!(n <= kp && n % 2 == 0, "{n} acoustic codebooks do not fit the padded input projection");
        let ip = bits(&w, &format!("{fp}input_projection.weight"), &[d, n])?;
        let mut ip_pad = vec![0u16; d * kp];
        for r in 0..d {
            ip_pad[r * kp..r * kp + n].copy_from_slice(&ip[r * n..(r + 1) * n]);
        }
        let input_proj = kit.lin("vt-input-proj", &ip_pad, d, kp, None)?;
        let out = kit.lin("vt-out", &bits(&w, &format!("{fp}acoustic_codebook_output.weight"), &[n, d])?, n, d, None)?;
        let time_proj = kit.lin("vt-time-proj", &bits(&w, &format!("{fp}time_projection.weight"), &[d, d])?, d, d, None)?;
        let timesteps = bf16_timesteps(nfe);
        let time_rows: Vec<u16> = timesteps[..nfe]
            .iter()
            .flat_map(|t| sinusoidal_time_embedding(t.to_f32(), d))
            .map(|v| bf16::from_f32(v).to_bits())
            .collect();
        let time_in = kit.upload_bf16("vt-time-in", &time_rows);
        let time_tokens = kit.zeroed("vt-time-tokens", nfe * d * 2);
        kit.submit(&[kit.gemm(&time_in, &time_proj, &time_tokens, nfe, 0)?]);
        kit.release_upload_staging()?;
        drop(time_proj);
        let dts = timesteps.windows(2).map(|p| p[1] - p[0]).collect();
        let mut m = Self {
            pos: kit.upload_i32("vt-pos", &[0]),
            zero_pos: kit.upload_i32("vt-zero-pos", &[0]),
            ids: kit.zeroed("vt-ids", codebooks.n_codebooks() * 4),
            frame: kit.zeroed("vt-frame", codebooks.n_codebooks() * 4),
            hn: kit.zeroed("vt-hn", d * 2),
            logits: kit.zeroed("vt-logits", semantic_rows * 2),
            cond: kit.zeroed("vt-cond", d * 2),
            x_state: kit.zeroed("vt-x-state", kp * 2),
            vf: kit.zeroed("vt-vf", FLOW_ROWS * n * 2),
            kit,
            src,
            gemv_src,
            quant,
            flow_quant,
            bb,
            fl,
            codebooks,
            encoder,
            voices: tekken.audio.voice_num_audio_tokens,
            dir: voice_dir.map(Path::to_path_buf),
            embed,
            codebook_embed,
            layers,
            norm,
            cos,
            sin,
            semantic_head,
            llm_proj,
            input_proj,
            flow_layers,
            flow_norm,
            out,
            time_tokens,
            nfe,
            cfg_alpha: bf16::from_f32(cfg_alpha),
            dts,
            voice_cache: Mutex::new(BTreeMap::new()),
            step_x: None,
            step_passes: Vec::new(),
            flow_passes: Vec::new(),
            hidden_passes: Vec::new(),
            flow_eval_len: 0,
            validated: false,
        };
        m.build_passes()?;
        Ok(m)
    }

    fn scratch(&self, t: &Stack, m: usize) -> Scratch {
        let kit = &self.kit;
        Scratch {
            x: kit.zeroed("vt-x", m * t.d * 2),
            h: kit.zeroed("vt-h", m * t.d * 2),
            xn: kit.zeroed("vt-xn", m * t.d * 2),
            qkv: kit.zeroed("vt-qkv", m * (t.nh + 2 * t.nkv) * t.hd * 4),
            q: kit.zeroed("vt-q", m * t.nh * t.hd * 4),
            attn: kit.zeroed("vt-attn", m * t.nh * t.hd * 2),
            ao: kit.zeroed("vt-ao", m * t.d * 2),
            gu: kit.zeroed("vt-gu", m * 2 * t.inter * 2),
            act: kit.zeroed("vt-act", m * t.inter * 2),
        }
    }

    fn vt(&self, entry: &str, binds: &[(u32, &Buffer)], grid: (u32, u32, u32)) -> Result<Pass> {
        self.kit.pass(&self.src, entry, binds, grid)
    }

    fn params(&self, p: &VtParams) -> Buffer {
        self.kit.uni("vt-p", p)
    }

    fn gemv(&self, x: &Buffer, lin: GemvW<'_>, y: &Buffer, x_off_words: usize, m: usize, flags: u32, norm: Option<&Buffer>) -> Result<Pass> {
        let (w, scales, n, k, name) = match lin {
            GemvW::Bf16(l) => match &l.w {
                crate::voxtral_wgpu::Weights::Bf16(w) => (w, None, l.n, l.k, l.name),
                crate::voxtral_wgpu::Weights::Nv4 { .. } => bail!("{}: the M-row GEMV reads bf16 or q8_0 weights", l.name),
            },
            GemvW::Q8(q) => (&q.w, Some(&q.s), q.n, q.k, q.name),
        };
        let gate_up = flags & VG_GATE_UP_SILU != 0;
        let r = if gate_up { 4 } else { 2 };
        let cols = if gate_up { n / 2 } else { n };
        let xv = if scales.is_some() { 2 } else { 1 };
        ensure!(cols % 2 == 0 && k % (8 * xv) == 0 && x_off_words % 4 == 0, "{name}: n={n} k={k} x offset {x_off_words} do not tile the vec4 GEMV");
        ensure!(norm.is_none() || (m == 1 && scales.is_none()), "{name}: the rmsnorm prologue normalizes one bf16 row");
        let flags = flags | if norm.is_some() { VG_NORM_PROLOGUE } else { 0 };
        let src = self.gemv_src.get(&(m, r, scales.is_some())).ok_or_else(|| anyhow!("{name}: no {m}-row GEMV is generated"))?;
        let ctx = self.kit.ctx;
        let sg = if ctx.caps.subgroup { ctx.subgroup_width().unwrap_or(32) as usize } else { 32 };
        let subgroups = 256 / sg;
        let (units, kvec) = (cols / 2, k / (8 * xv));
        let mut spp = 1;
        while spp * 2 <= subgroups && units * spp / subgroups < 512 && kvec / (spp * 2) >= sg {
            spp *= 2;
        }
        let grid = self.kit.grid_rows(units.div_ceil(subgroups / spp));
        let p = self.kit.uni(
            "vt-gemv-p",
            &VtGemvParams {
                n_rows: n as u32,
                k_vec: kvec as u32,
                groups_x: grid.0,
                flags,
                x_off_vec4: (x_off_words / 4) as u32,
                sg_per_unit: spp as u32,
                m: m as u32,
                eps: self.bb.eps,
            },
        );
        let mut binds = vec![(27, w), (28, x), (29, y), (30, &p)];
        binds.push(scales.map_or((32, norm.unwrap_or(&self.kit.dummy)), |s| (33, s)));
        let mut pass = self.kit.pass(src, "vt_gemv", &binds, grid)?;
        pass.label = format!("{}:{name}:m{m}", pass.label);
        Ok(pass)
    }

    fn semantic_argmax_pass(&self, incr: usize) -> Result<Pass> {
        let _ = THE_SEMANTIC_ARGMAX_READS_BF16_LOGITS_SO_TIES_BREAK_AS_THE_REFERENCE_BREAKS_THEM;
        let p = self.params(&VtParams { a: self.codebooks.semantic_rows() as u32, b: incr as u32, ..Default::default() });
        self.vt("vt_semantic_argmax", &[(9, &self.logits), (10, &self.ids), (12, &self.pos), (13, &p)], (1, 1, 1))
    }

    fn head_passes(&self, x: &Buffer, hn: &Buffer, rows: usize, incr: usize) -> Result<Vec<Pass>> {
        let kit = &self.kit;
        let off = (rows - 1) * self.bb.d / 2;
        Ok(vec![
            kit.rmsnorm(x, &self.norm, hn, rows, self.bb.d, self.bb.eps)?,
            self.gemv(hn, GemvW::Bf16(&self.semantic_head), &self.logits, off, 1, 0, None)?,
            self.gemv(hn, GemvW::Bf16(&self.llm_proj), &self.cond, off, 1, 0, None)?,
            self.semantic_argmax_pass(incr)?,
        ])
    }

    fn prefill_layers(&self, s: &Scratch, m: usize) -> Result<Vec<Pass>> {
        let kit = &self.kit;
        let t = &self.bb;
        let part = kit.split_part(&self.layers.iter().flat_map(|l| [(&l.qkv, m), (&l.o, m), (&l.gate_up, m), (&l.down, m)]).collect::<Vec<_>>());
        let mut pf = Vec::with_capacity(13 * t.layers);
        for l in &self.layers {
            pf.push(kit.rmsnorm(&s.x, &l.attn_norm, &s.h, m, t.d, t.eps)?);
            pf.extend(kit.gemm_split(&s.h, &l.qkv, &s.qkv, m, GEMM_OUT_F32, &part, kit.split_k_for(&l.qkv, m))?);
            pf.push(kit.rope_kv(&s.qkv, &self.cos, &self.sin, &self.zero_pos, &s.q, &l.kcache, &l.vcache, m, t.nh, t.nkv, t.hd)?);
            pf.push(kit.attn_prefill(&s.q, &l.kcache, &l.vcache, &self.zero_pos, &s.attn, m, t.nh, t.nkv, t.hd, 0, 1.0)?);
            pf.extend(kit.gemm_split(&s.attn, &l.o, &s.ao, m, 0, &part, kit.split_k_for(&l.o, m))?);
            pf.push(kit.rmsnorm_residual(&s.ao, &s.x, &l.ffn_norm, &s.h, m, t.d, t.eps)?);
            pf.extend(kit.gemm_split(&s.h, &l.gate_up, &s.gu, m, 0, &part, kit.split_k_for(&l.gate_up, m))?);
            pf.push(kit.silu_mul_rows(&s.gu, &s.act, m, t.inter, false)?);
            pf.extend(kit.gemm_split(&s.act, &l.down, &s.x, m, GEMM_ACCUM, &part, kit.split_k_for(&l.down, m))?);
        }
        Ok(pf)
    }

    fn step_layers(&self, s: &Scratch, scratch: &Buffer) -> Result<Vec<Pass>> {
        let kit = &self.kit;
        let t = &self.bb;
        let nv4 = self.quant == VoxtralQuant::Nvfp4;
        let mut st = Vec::with_capacity(11 * t.layers);
        for l in &self.layers {
            if nv4 {
                st.push(kit.norm_rows(&s.x, &l.attn_norm, &s.xn, 1, t.d, 0, t.eps)?);
                st.push(kit.gemv_nv4(&s.xn, &l.qkv, &s.qkv, 0, 1, NV4_OUT_F32)?);
                st.push(kit.rope_kv(&s.qkv, &self.cos, &self.sin, &self.pos, &s.q, &l.kcache, &l.vcache, 1, t.nh, t.nkv, t.hd)?);
                st.extend(kit.attn_decode(&s.q, &l.kcache, &l.vcache, &self.pos, &s.attn, scratch, 1, t.nh, t.nkv, t.hd, true)?);
                st.push(kit.gemv_nv4(&s.attn, &l.o, &s.x, 0, 1, NV4_ACCUM)?);
                st.push(kit.norm_rows(&s.x, &l.ffn_norm, &s.xn, 1, t.d, 0, t.eps)?);
                st.push(kit.gemv_nv4(&s.xn, &l.gate_up, &s.gu, 0, 1, 0)?);
                st.push(kit.silu_mul_rows(&s.gu, &s.act, 1, t.inter, true)?);
                st.push(kit.gemv_nv4(&s.act, &l.down, &s.x, 0, 1, NV4_ACCUM)?);
            } else {
                st.push(self.gemv(&s.x, GemvW::Bf16(&l.qkv), &s.qkv, 0, 1, VG_OUT_F32, Some(&l.attn_norm))?);
                st.push(kit.rope_kv(&s.qkv, &self.cos, &self.sin, &self.pos, &s.q, &l.kcache, &l.vcache, 1, t.nh, t.nkv, t.hd)?);
                st.extend(kit.attn_decode(&s.q, &l.kcache, &l.vcache, &self.pos, &s.attn, scratch, 1, t.nh, t.nkv, t.hd, false)?);
                st.push(self.gemv(&s.attn, GemvW::Bf16(&l.o), &s.x, 0, 1, VG_ACCUM, None)?);
                st.push(self.gemv(&s.x, GemvW::Bf16(&l.gate_up), &s.act, 0, 1, VG_GATE_UP_SILU, Some(&l.ffn_norm))?);
                st.push(self.gemv(&s.act, GemvW::Bf16(&l.down), &s.x, 0, 1, VG_ACCUM, None)?);
            }
        }
        Ok(st)
    }

    fn flow_eval(&self, s: &Scratch, i: usize) -> Result<Vec<Pass>> {
        let _ = THE_FLOW_HEAD_SWEEPS_THE_SIX_CFG_ROWS_PER_WEIGHT_WORD_LIKE_THE_NVFP4_GEMV;
        let _ = VT_EULER_HARDCODES_THE_SPECIAL_IDS;
        let _ = EULER_AND_FSQ_RUN_ON_THE_DEVICE_WITH_ONE_BF16_ROUNDING_PER_OPERATION;
        let kit = &self.kit;
        let t = &self.fl;
        let m = FLOW_ROWS;
        let words = t.d / 2;
        let nv4 = self.flow_quant == VoxtralQuant::Nvfp4;
        let crate::voxtral_wgpu::Weights::Bf16(ip) = &self.input_proj.w else { bail!("input_projection is bf16") };
        let p = self.params(&VtParams { words: words as u32, a: i as u32, b: (self.input_proj.k / 2) as u32, ..Default::default() });
        let mut ev = vec![self.vt(
            "vt_flow_tokens",
            &[(14, &self.x_state), (31, ip), (15, &self.time_tokens), (16, &self.cond), (17, &s.x), (18, &p)],
            kit.grid_flat(m * words, 256),
        )?];
        let ap = self.params(&VtParams {
            words: (t.hd / 2) as u32,
            a: t.nh as u32,
            b: t.nkv as u32,
            c: u32::from(nv4),
            f0: 1.0 / (t.hd as f32).sqrt(),
            ..Default::default()
        });
        for l in &self.flow_layers {
            if nv4 {
                ev.push(kit.norm_rows(&s.x, &l.attn_norm, &s.xn, m, t.d, 0, t.eps)?);
                ev.push(kit.gemv_nv4(&s.xn, l.qkv.lin()?, &s.qkv, 0, m, 0)?);
                ev.push(self.vt("vt_flow_attn", &[(19, &s.qkv), (20, &s.attn), (21, &ap)], (m as u32, t.nh as u32, 1))?);
                ev.push(kit.gemv_nv4(&s.attn, l.o.lin()?, &s.x, 0, m, NV4_ACCUM)?);
                ev.push(kit.norm_rows(&s.x, &l.ffn_norm, &s.xn, m, t.d, 0, t.eps)?);
                ev.push(kit.gemv_nv4(&s.xn, l.gate_up.lin()?, &s.gu, 0, m, 0)?);
                ev.push(kit.silu_mul_rows(&s.gu, &s.act, m, t.inter, true)?);
                ev.push(kit.gemv_nv4(&s.act, l.down.lin()?, &s.x, 0, m, NV4_ACCUM)?);
            } else {
                ev.push(kit.rmsnorm(&s.x, &l.attn_norm, &s.h, m, t.d, t.eps)?);
                ev.push(self.gemv(&s.h, l.qkv.w(), &s.qkv, 0, m, 0, None)?);
                ev.push(self.vt("vt_flow_attn", &[(19, &s.qkv), (20, &s.attn), (21, &ap)], (m as u32, t.nh as u32, 1))?);
                ev.push(self.gemv(&s.attn, l.o.w(), &s.ao, 0, m, 0, None)?);
                ev.push(kit.rmsnorm_residual(&s.ao, &s.x, &l.ffn_norm, &s.h, m, t.d, t.eps)?);
                ev.push(self.gemv(&s.h, l.gate_up.w(), &s.act, 0, m, VG_GATE_UP_SILU, None)?);
                ev.push(self.gemv(&s.act, l.down.w(), &s.x, 0, m, VG_ACCUM, None)?);
            }
        }
        ev.push(kit.rmsnorm(&s.x, &self.flow_norm, &s.h, m, t.d, t.eps)?);
        ev.push(self.gemv(&s.h, GemvW::Bf16(&self.out), &self.vf, 0, m, 0, None)?);
        let cb = self.codebooks;
        let ep = self.kit.uni(
            "vt-euler-p",
            &VtEulerParams {
                n: cb.n_acoustic as u32,
                fsq: u32::from(i + 1 == self.nfe),
                semantic_rows: cb.semantic_rows() as u32,
                acoustic_rows: cb.acoustic_rows() as u32,
                alpha: self.cfg_alpha.to_f32(),
                one_minus_alpha: (bf16::ONE - self.cfg_alpha).to_f32(),
                dt: self.dts[i].to_f32(),
                levels: bf16::from_f32((cb.acoustic_levels - 1) as f32).to_f32(),
            },
        );
        ev.push(self.vt("vt_euler", &[(22, &self.vf), (23, &self.x_state), (24, &self.frame), (25, &self.ids), (26, &ep)], (1, 1, 1))?);
        Ok(ev)
    }

    fn build_passes(&mut self) -> Result<()> {
        let kit = &self.kit;
        let t = self.bb;
        let s = self.scratch(&t, 1);
        let scratch = kit.zeroed("vt-fd-scratch", t.nh * DECODE_ATTENTION_SPLITS as usize * (t.hd + 2) * 4);
        let fp = self.params(&VtParams { words: (t.d / 2) as u32, a: self.codebooks.n_codebooks() as u32, ..Default::default() });
        let mut st = vec![self.vt("vt_frame_embed", &[(5, &self.codebook_embed), (6, &self.ids), (7, &s.x), (8, &fp)], kit.grid_flat(t.d / 2, 256))?];
        st.extend(self.step_layers(&s, &scratch)?);
        st.extend(self.head_passes(&s.x, &self.hn, 1, 1)?);

        let fs = self.scratch(&self.fl, FLOW_ROWS);
        let mut fl = Vec::new();
        for i in 0..self.nfe {
            let ev = self.flow_eval(&fs, i)?;
            if i == 0 {
                self.flow_eval_len = ev.len();
            }
            ensure!(ev.len() == self.flow_eval_len, "every Euler step builds the same pass list");
            fl.extend(ev);
        }
        self.step_x = Some(s.x.clone());
        self.step_passes = st;
        self.flow_passes = fl;
        self.hidden_passes = vec![
            self.gemv(&self.hn, GemvW::Bf16(&self.semantic_head), &self.logits, 0, 1, 0, None)?,
            self.gemv(&self.hn, GemvW::Bf16(&self.llm_proj), &self.cond, 0, 1, 0, None)?,
            self.semantic_argmax_pass(0)?,
        ];
        Ok(())
    }

    pub fn quant(&self) -> VoxtralQuant {
        self.quant
    }

    pub fn nfe(&self) -> usize {
        self.nfe
    }

    pub fn timesteps(&self) -> Vec<f32> {
        bf16_timesteps(self.nfe).iter().map(|t| t.to_f32()).collect()
    }

    pub fn codebooks(&self) -> CodebookLayout {
        self.codebooks
    }

    pub fn voices(&self) -> impl Iterator<Item = (&str, usize)> {
        self.voices.iter().map(|(k, v)| (k.as_str(), *v))
    }

    pub fn encoder(&self) -> &TekkenEncoder {
        &self.encoder
    }

    pub fn step_pass_count(&self) -> usize {
        self.step_passes.len()
    }

    pub fn flow_pass_count(&self) -> usize {
        self.flow_passes.len()
    }

    pub fn voice_rows(&self, voice: &str) -> Result<usize> {
        self.voices.get(voice).copied().ok_or_else(|| {
            anyhow!("unknown voice {voice:?}; presets: {}", self.voices.keys().cloned().collect::<Vec<_>>().join(", "))
        })
    }

    pub fn prompt_ids(&self, text: &str, voice: &str) -> Result<Vec<u32>> {
        let n_voice = self.voice_rows(voice)?;
        let mut ids = vec![BOS_TOKEN, BEGIN_AUDIO_TOKEN];
        ids.extend(std::iter::repeat_n(AUDIO_TOKEN, n_voice));
        ids.push(NEXT_AUDIO_TEXT_TOKEN);
        ids.extend(self.encoder.encode(text));
        ids.extend([REPEAT_AUDIO_TEXT_TOKEN, BEGIN_AUDIO_TOKEN]);
        Ok(ids)
    }

    fn voice_buffer(&self, voice: &str) -> Result<Buffer> {
        let n_voice = self.voice_rows(voice)?;
        let mut cache = self.voice_cache.lock().map_err(|_| anyhow!("voice cache poisoned"))?;
        if let Some(b) = cache.get(voice) {
            return Ok(b.clone());
        }
        let dir = self.dir.as_ref().ok_or_else(|| anyhow!("voice {voice:?} is not staged and this model has no checkpoint dir to read {VOICE_DIR}/{voice}.pt from"))?;
        let t = read_torch_tensor(&dir.join(VOICE_DIR).join(format!("{voice}.pt")))?;
        ensure!(t.dims() == [n_voice, self.bb.d], "voice {voice}: tensor {:?} does not match tekken's {n_voice} audio tokens", t.dims());
        let b = self.kit.upload_bf16("vt-voice", &tensor_bits(&t)?);
        cache.insert(voice.to_string(), b.clone());
        Ok(b)
    }

    pub fn stage_voice(&self, voice: &str, bits: &[u16]) -> Result<()> {
        let n_voice = self.voice_rows(voice)?;
        ensure!(bits.len() == n_voice * self.bb.d, "voice {voice}: {} values do not match tekken's {n_voice} audio tokens by {}", bits.len(), self.bb.d);
        let b = self.kit.upload_bf16("vt-voice", bits);
        self.voice_cache.lock().map_err(|_| anyhow!("voice cache poisoned"))?.insert(voice.to_string(), b);
        Ok(())
    }

    pub fn semantic_argmax(&self, logits: &[f32]) -> u32 {
        self.semantic_top2(logits)[0].0
    }

    pub fn semantic_top2(&self, logits: &[f32]) -> [(u32, f32); 2] {
        let rows = self.codebooks.semantic_rows();
        let mut top = [(END_AUDIO, f32::NEG_INFINITY), (END_AUDIO, f32::NEG_INFINITY)];
        for (i, &l) in logits.iter().enumerate().take(rows).skip(1) {
            if l > top[0].1 {
                top[1] = top[0];
                top[0] = (i as u32, l);
            } else if l > top[1].1 {
                top[1] = (i as u32, l);
            }
        }
        top
    }

    fn frame_ids(&self, frame: &[u32]) -> Vec<u32> {
        let cb = self.codebooks;
        frame.iter().enumerate().map(|(k, &c)| (cb.offset(k) + c as usize) as u32).collect()
    }

    fn run(&self, passes: &[Pass], profiling: bool) -> Result<()> {
        if profiling {
            self.kit.submit_profiled(passes)
        } else {
            self.kit.submit(passes);
            Ok(())
        }
    }

    async fn read_bf16(&self, b: &Buffer, offset_words: usize, words: usize, arm: ReadbackArm) -> Result<Vec<f32>> {
        Ok(unpack_bf16(&dispatch::read_back_at_arm::<u32>(self.kit.ctx, b, offset_words, words, arm).await.map_err(err)?))
    }

    pub fn frame_from_hidden(&self, hidden: &[f32], x0: &[f32], trace: bool) -> Result<(Vec<u32>, FrameTrace)> {
        dispatch::drive_blocking(self.frame_from_hidden_arm(hidden, x0, trace, ReadbackArm::Blocking))
    }

    pub async fn frame_from_hidden_arm(&self, hidden: &[f32], x0: &[f32], trace: bool, arm: ReadbackArm) -> Result<(Vec<u32>, FrameTrace)> {
        let cb = self.codebooks;
        let ctx = self.kit.ctx;
        ensure!(hidden.len() == self.bb.d, "hidden state has {} values, the heads read {}", hidden.len(), self.bb.d);
        ensure!(x0.len() == cb.n_acoustic, "noise row has {} values, the flow head needs {}", x0.len(), cb.n_acoustic);
        ctx.queue.write_buffer(&self.hn, 0, bytemuck::cast_slice(&pack_bf16(hidden, self.bb.d / 2)));
        ctx.queue.write_buffer(&self.x_state, 0, bytemuck::cast_slice(&pack_bf16(x0, INPUT_PROJ_K_PADDED_TO_THE_GEMM_K_STEP / 2)));
        self.kit.submit(&self.hidden_passes);
        let mut tr = FrameTrace::default();
        if trace {
            tr.semantic_logits = self.read_bf16(&self.logits, 0, self.semantic_head.n / 2, arm).await?;
            tr.x.push(x0.to_vec());
            for i in 0..self.nfe {
                self.kit.submit(&self.flow_passes[i * self.flow_eval_len..(i + 1) * self.flow_eval_len]);
                let v = self.read_bf16(&self.vf, 0, FLOW_ROWS * cb.n_acoustic / 2, arm).await?;
                tr.v_cond.push(v[..cb.n_acoustic].to_vec());
                tr.v_uncond.push(v[3 * cb.n_acoustic..4 * cb.n_acoustic].to_vec());
                tr.x.push(self.read_bf16(&self.x_state, 0, cb.n_acoustic / 2, arm).await?);
            }
        } else {
            self.kit.submit(&self.flow_passes);
        }
        let frame = dispatch::read_back_arm::<u32>(ctx, &self.frame, cb.n_codebooks(), arm).await.map_err(err)?;
        Ok((frame, tr))
    }

    pub fn synthesize(
        &mut self,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        trace: bool,
        teacher: Option<&[Vec<u32>]>,
    ) -> Result<Synthesis> {
        dispatch::drive_blocking(self.synthesize_arm(text, voice, noise, max_frames, trace, teacher, ReadbackArm::Blocking))
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn synthesize_arm(
        &mut self,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        trace: bool,
        teacher: Option<&[Vec<u32>]>,
        arm: ReadbackArm,
    ) -> Result<Synthesis> {
        ensure!(!self.layers.is_empty(), "this model loaded no backbone layers: frame_from_hidden is its only frame path");
        let ids = self.prompt_ids(text, voice)?;
        let max_frames = teacher.map_or(max_frames, <[Vec<u32>]>::len);
        let m = ids.len();
        ensure!(m + max_frames + 1 <= KV_CAPACITY_ROWS, "prompt of {m} tokens plus {max_frames} frames exceeds the {KV_CAPACITY_ROWS}-row cache");
        let voice_b = self.voice_buffer(voice)?;
        let n_voice = self.voice_rows(voice)?;
        let first_audio = ids.iter().position(|&t| t == AUDIO_TOKEN).ok_or_else(|| anyhow!("prompt has no [AUDIO] run"))?;
        let kit = &self.kit;
        let ctx = kit.ctx;
        let profiling = dispatch::profile::enabled() && ctx.caps.timestamp_query;
        let (d, words) = (self.bb.d, self.bb.d / 2);
        let cb = self.codebooks;
        let n_cb = cb.n_codebooks();
        let scope = (!self.validated).then(|| ctx.device.push_error_scope(wgpu::ErrorFilter::Validation));
        ctx.queue.write_buffer(&self.pos, 0, bytemuck::bytes_of(&0i32));
        let s = self.scratch(&self.bb, m);
        let hn_pf = kit.zeroed("vt-hn-pf", m * d * 2);
        let ids_b = kit.upload_u32("vt-prompt-ids", &ids);
        let pp = self.params(&VtParams { words: words as u32, a: m as u32, b: first_audio as u32, c: n_voice as u32, ..Default::default() });
        let mut pf = vec![self.vt("vt_prompt_rows", &[(0, &self.embed), (1, &ids_b), (2, &voice_b), (3, &s.x), (4, &pp)], kit.grid_flat(m * words, 256))?];
        pf.extend(self.prefill_layers(&s, m)?);
        pf.extend(self.head_passes(&s.x, &hn_pf, m, m)?);
        if profiling {
            dispatch::profile::reset();
        }
        let mut out = Synthesis { prompt_ids: ids.clone(), ..Default::default() };
        #[cfg(not(target_arch = "wasm32"))]
        let mut t0 = std::time::Instant::now();
        let mut lap = |slot: &mut f64| -> Result<()> {
            #[cfg(not(target_arch = "wasm32"))]
            if trace {
                ctx.poll_blocking().map_err(err)?;
                let now = std::time::Instant::now();
                *slot += (now - t0).as_secs_f64() * 1000.0;
                t0 = now;
            }
            #[cfg(target_arch = "wasm32")]
            let _ = slot;
            Ok(())
        };
        self.run(&pf, profiling)?;
        lap(&mut out.stage_ms.prefill)?;
        let x_words = INPUT_PROJ_K_PADDED_TO_THE_GEMM_K_STEP / 2;
        for frame_idx in 0..max_frames {
            let hidden = if frame_idx == 0 { (&hn_pf, (m - 1) * words) } else { (&self.hn, 0) };
            if trace {
                let logits = self.read_bf16(&self.logits, 0, self.semantic_head.n / 2, arm).await?;
                out.semantic_top2.push(self.semantic_top2(&logits));
                if frame_idx == 0 {
                    out.frame0 = Some(FrameTrace { semantic_logits: logits, ..Default::default() });
                }
            }
            lap(&mut out.stage_ms.semantic)?;
            let x0 = noise(frame_idx);
            ensure!(x0.len() == cb.n_acoustic, "noise row has {} values, the flow head needs {}", x0.len(), cb.n_acoustic);
            ctx.queue.write_buffer(&self.x_state, 0, bytemuck::cast_slice(&pack_bf16(&x0, x_words)));
            let tracing_frame = trace && frame_idx == 0;
            if tracing_frame {
                let tr = out.frame0.as_mut().expect("frame 0 trace");
                tr.x.push(x0.clone());
                for i in 0..self.nfe {
                    self.run(&self.flow_passes[i * self.flow_eval_len..(i + 1) * self.flow_eval_len], profiling)?;
                    let v = self.read_bf16(&self.vf, 0, FLOW_ROWS * cb.n_acoustic / 2, arm).await?;
                    tr.v_cond.push(v[..cb.n_acoustic].to_vec());
                    tr.v_uncond.push(v[3 * cb.n_acoustic..4 * cb.n_acoustic].to_vec());
                    tr.x.push(self.read_bf16(&self.x_state, 0, cb.n_acoustic / 2, arm).await?);
                }
            } else {
                self.run(&self.flow_passes, profiling)?;
            }
            let free_step = teacher.is_none() && !trace;
            if free_step {
                self.run(&self.step_passes, profiling)?;
            }
            let frame = dispatch::read_back_arm::<u32>(ctx, &self.frame, n_cb, arm).await.map_err(err)?;
            lap(&mut out.stage_ms.flow)?;
            let semantic = frame[0];
            let ends = teacher.map_or(semantic == END_AUDIO, |f| frame_idx + 1 == f.len());
            if trace && (frame_idx < 3 || ends) {
                out.hidden.insert(frame_idx, self.read_bf16(hidden.0, hidden.1, words, arm).await?);
            }
            out.frames.push(frame);
            if ends {
                out.ended = semantic == END_AUDIO;
                break;
            }
            if let Some(f) = teacher {
                ctx.queue.write_buffer(&self.ids, 0, bytemuck::cast_slice(&self.frame_ids(&f[frame_idx])));
            }
            if !free_step {
                if tracing_frame {
                    self.run(&self.step_passes[..1], profiling)?;
                    out.frame0.as_mut().expect("frame 0 trace").frame_embed = self.read_bf16(self.step_x.as_ref().expect("step scratch"), 0, words, arm).await?;
                    self.run(&self.step_passes[1..], profiling)?;
                } else {
                    self.run(&self.step_passes, profiling)?;
                }
            }
            lap(&mut out.stage_ms.backbone)?;
            out.stage_ms.steps += 1;
        }
        if let Some(scope) = scope {
            ctx.poll_blocking().map_err(err)?;
            if let Some(e) = dispatch::pop_scope(scope) {
                bail!("voxtral tts wgpu validation: {e}");
            }
            self.validated = true;
        }
        if profiling {
            eprintln!(
                "{}",
                dispatch::profile::table(
                    &format!("voxtral tts wgpu {} frame", self.quant.label()),
                    self.step_passes.len() + self.flow_passes.len()
                )
            );
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rig {
        kit: Kit,
        src: String,
    }

    fn rig() -> Option<Rig> {
        match WgpuContext::shared() {
            Ok(ctx) => Some(Rig { kit: Kit::new(ctx), src: compose(&format!("{VX_SHARED_WGSL}{VT_WGSL}")) }),
            Err(e) => {
                eprintln!("skip: no wgpu adapter: {e}");
                None
            }
        }
    }

    impl Rig {
        fn run(&self, entry: &str, binds: &[(u32, &Buffer)], grid: (u32, u32, u32)) {
            let scope = self.kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pass = self.kit.pass(&self.src, entry, binds, grid).unwrap();
            self.kit.submit(std::slice::from_ref(&pass));
            if let Some(e) = dispatch::pop_scope(scope) {
                panic!("validation: {e}");
            }
        }
    }

    struct Lcg(u64);

    impl Lcg {
        fn f(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
        fn bf(&mut self) -> f32 {
            bf16::from_f32(self.f()).to_f32()
        }
    }

    #[test]
    fn euler_and_fsq_reproduce_the_host_bf16_arithmetic_bitwise() {
        let Some(r) = rig() else { return };
        let (n, levels) = (36usize, 21usize);
        let alpha = bf16::from_f32(DEFAULT_CFG_ALPHA);
        let dts = {
            let t = bf16_timesteps(7);
            t.windows(2).map(|p| p[1] - p[0]).collect::<Vec<_>>()
        };
        let mut rng = Lcg(17);
        for (step, &dt) in dts.iter().enumerate() {
            let x0: Vec<f32> = (0..n).map(|_| rng.bf() * 1.5).collect();
            let v: Vec<f32> = (0..FLOW_ROWS * n).map(|_| rng.bf() * 3.0).collect();
            let mut want: Vec<bf16> = x0.iter().map(|x| bf16::from_f32(*x)).collect();
            let oma = bf16::ONE - alpha;
            for j in 0..n {
                let vt = alpha * bf16::from_f32(v[j]) + oma * bf16::from_f32(v[3 * n + j]);
                want[j] = want[j] + vt * dt;
            }
            let lv = bf16::from_f32((levels - 1) as f32);
            let codes: Vec<u32> = want
                .iter()
                .map(|&x| {
                    let c = x.clamp(bf16::NEG_ONE, bf16::ONE);
                    let y = (c + bf16::ONE) * bf16::from_f32(0.5) * lv;
                    y.to_f32().round_ties_even().clamp(0.0, (levels - 1) as f32) as u32
                })
                .collect();
            let xb = r.kit.upload_u32("t-x", &pack_bf16(&x0, 32));
            let vb = r.kit.upload_u32("t-v", &pack_bf16(&v, FLOW_ROWS * n / 2));
            let frame = r.kit.zeroed("t-frame", (n + 1) * 4);
            let ids = r.kit.upload_u32("t-ids", &vec![if step % 2 == 0 { 7 } else { END_AUDIO }; n + 1]);
            let p = r.kit.uni(
                "t-p",
                &VtEulerParams {
                    n: n as u32,
                    fsq: 1,
                    semantic_rows: 8194,
                    acoustic_rows: 23,
                    alpha: alpha.to_f32(),
                    one_minus_alpha: oma.to_f32(),
                    dt: dt.to_f32(),
                    levels: lv.to_f32(),
                },
            );
            r.run("vt_euler", &[(22, &vb), (23, &xb), (24, &frame), (25, &ids), (26, &p)], (1, 1, 1));
            let got = unpack_bf16(&r.kit.read_u32(&xb, n / 2).unwrap());
            assert_eq!(got, want.iter().map(|v| v.to_f32()).collect::<Vec<_>>(), "euler step {step}");
            let got_frame = r.kit.read_u32(&frame, n + 1).unwrap();
            let got_ids = r.kit.read_u32(&ids, n + 1).unwrap();
            let ended = step % 2 == 1;
            assert_eq!(got_frame[0], got_ids[0], "the FSQ pass copies the semantic id into the frame");
            for j in 0..n {
                let code = if ended { 0 } else { codes[j] } + N_AUDIO_SPECIAL;
                assert_eq!(got_frame[1 + j], code, "code {j} step {step}");
                assert_eq!(got_ids[1 + j], 8194 + j as u32 * 23 + code, "id {j} step {step}");
            }
        }
    }

    #[test]
    fn semantic_argmax_masks_id_zero_and_the_padded_rows_and_takes_the_lowest_tied_index() {
        let Some(r) = rig() else { return };
        let rows = 8194usize;
        let mut rng = Lcg(3);
        let mut logits: Vec<f32> = (0..8320).map(|_| rng.bf()).collect();
        logits[0] = 50.0;
        logits[8200] = 50.0;
        logits[3040] = 10.8125;
        logits[10] = 10.8125;
        let lb = r.kit.upload_u32("t-logits", &pack_bf16(&logits, 8320 / 2));
        let ids = r.kit.zeroed("t-ids", 37 * 4);
        let pos = r.kit.upload_i32("t-pos", &[5]);
        let p = r.kit.uni("t-p", &VtParams { a: rows as u32, b: 233, ..Default::default() });
        r.run("vt_semantic_argmax", &[(9, &lb), (10, &ids), (12, &pos), (13, &p)], (1, 1, 1));
        assert_eq!(r.kit.read_u32(&ids, 1).unwrap()[0], 10);
        assert_eq!(r.kit.read_u32(&pos, 1).unwrap()[0], 238, "the argmax advances the device position by incr");
    }

    #[test]
    fn flow_attention_matches_the_host_softmax_over_three_tokens_per_cfg_row() {
        let Some(r) = rig() else { return };
        let (nh, nkv, hd) = (4usize, 2usize, 128usize);
        let stride = (nh + 2 * nkv) * hd;
        let mut rng = Lcg(9);
        let qkv: Vec<f32> = (0..FLOW_ROWS * stride).map(|_| rng.bf()).collect();
        let qb = r.kit.upload_u32("t-qkv", &pack_bf16(&qkv, FLOW_ROWS * stride / 2));
        let out = r.kit.zeroed("t-out", FLOW_ROWS * nh * hd * 2);
        let scale = 1.0 / (hd as f32).sqrt();
        let p = r.kit.uni("t-p", &VtParams { words: (hd / 2) as u32, a: nh as u32, b: nkv as u32, f0: scale, ..Default::default() });
        r.run("vt_flow_attn", &[(19, &qb), (20, &out), (21, &p)], (FLOW_ROWS as u32, nh as u32, 1));
        let got = unpack_bf16(&r.kit.read_u32(&out, FLOW_ROWS * nh * hd / 2).unwrap());
        for row in 0..FLOW_ROWS {
            let b = row / 3;
            for h in 0..nh {
                let kvh = h / (nh / nkv);
                let q = &qkv[row * stride + h * hd..row * stride + (h + 1) * hd];
                let scores: Vec<f32> = (0..3)
                    .map(|j| {
                        let k = &qkv[(b * 3 + j) * stride + (nh + kvh) * hd..][..hd];
                        q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - m).exp()).collect();
                let l: f32 = e.iter().sum();
                for d in 0..hd {
                    let want: f32 = (0..3).map(|j| e[j] / l * qkv[(b * 3 + j) * stride + (nh + nkv + kvh) * hd + d]).sum();
                    let g = got[(row * nh + h) * hd + d];
                    assert!((g - want).abs() <= 1e-2 * want.abs().max(1e-2), "row {row} head {h} dim {d}: {g} vs {want}");
                }
            }
        }
    }

    #[test]
    fn the_m_row_gemv_matches_the_host_product_in_every_epilogue_on_both_subgroup_arms() {
        let Some(r) = rig() else { return };
        let (n, k) = (76usize, 1024usize);
        let mut rng = Lcg(23);
        let w: Vec<u16> = (0..n * k).map(|_| bf16::from_f32(rng.f()).to_bits()).collect();
        let x: Vec<f32> = (0..FLOW_ROWS * k).map(|_| rng.bf()).collect();
        let y0: Vec<f32> = (0..FLOW_ROWS * n).map(|_| rng.bf()).collect();
        let nw: Vec<f32> = (0..k).map(|_| rng.bf()).collect();
        let eps = 1e-5f32;
        let wb = r.kit.upload_bf16("t-w", &w);
        let xb = r.kit.upload_u32("t-x", &pack_bf16(&x, FLOW_ROWS * k / 2));
        let nb = r.kit.upload_u32("t-nw", &pack_bf16(&nw, k / 2));
        let rstd = 1.0 / (x[..k].iter().map(|v| v * v).sum::<f32>() / k as f32 + eps).sqrt();
        let xn: Vec<f32> = x[..k].iter().zip(&nw).map(|(v, g)| bf16::from_f32(v * rstd * g).to_f32()).collect();
        let dot = |m: usize, c: usize, normed: bool| {
            (0..k).map(|i| if normed { xn[i] } else { x[m * k + i] } * bf16::from_bits(w[c * k + i]).to_f32()).sum::<f32>()
        };
        let close = |got: f32, want: f32, what: &str| assert!((got - want).abs() <= 1e-2 * want.abs().max(1e-2), "{what}: {got} vs {want}");
        for sg in [r.kit.ctx.caps.subgroup, false] {
            for (m, flags, spp) in [
                (1, VG_OUT_F32, 2),
                (1, VG_OUT_F32 | VG_NORM_PROLOGUE, 2),
                (FLOW_ROWS, 0, 1),
                (FLOW_ROWS, VG_ACCUM, 4),
                (1, VG_GATE_UP_SILU, 1),
                (1, VG_GATE_UP_SILU | VG_NORM_PROLOGUE, 1),
                (FLOW_ROWS, VG_GATE_UP_SILU, 2),
            ] {
                let gu = flags & VG_GATE_UP_SILU != 0;
                let rr = if gu { 4 } else { 2 };
                let normed = flags & VG_NORM_PROLOGUE != 0;
                let src = compose(&format!("{VX_SHARED_WGSL}{}", gemv_source(m, rr, false, sg)));
                let cols = if gu { n / 2 } else { n };
                let units = cols / 2;
                let grid = r.kit.grid_rows(units.div_ceil(8 / spp));
                let y = r.kit.upload_u32("t-y", &pack_bf16(&y0, FLOW_ROWS * n / 2));
                let p = r.kit.uni(
                    "t-p",
                    &VtGemvParams { n_rows: n as u32, k_vec: (k / 8) as u32, groups_x: grid.0, flags, x_off_vec4: 0, sg_per_unit: spp as u32, m: m as u32, eps },
                );
                let scope = r.kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
                let pass = r.kit.pass(&src, "vt_gemv", &[(27, &wb), (28, &xb), (29, &y), (30, &p), (32, &nb)], grid).unwrap();
                r.kit.submit(std::slice::from_ref(&pass));
                if let Some(e) = dispatch::pop_scope(scope) {
                    panic!("validation: {e}");
                }
                let what = format!("sg {sg} m {m} flags {flags}");
                if flags & VG_OUT_F32 != 0 {
                    let got = r.kit.read_f32(&y, m * n).unwrap();
                    for c in 0..n {
                        close(got[c], dot(0, c, normed), &format!("{what} col {c}"));
                    }
                    continue;
                }
                let got = unpack_bf16(&r.kit.read_u32(&y, FLOW_ROWS * n / 2).unwrap());
                for mi in 0..m {
                    for c in 0..cols {
                        let want = if gu {
                            let (g, u) = (bf16::from_f32(dot(mi, c, normed)).to_f32(), bf16::from_f32(dot(mi, c + cols, normed)).to_f32());
                            g / (1.0 + (-g).exp()) * u
                        } else if flags & VG_ACCUM != 0 {
                            bf16::from_f32(dot(mi, c, false)).to_f32() + y0[mi * n + c]
                        } else {
                            dot(mi, c, false)
                        };
                        close(got[mi * cols + c], want, &format!("{what} row {mi} col {c}"));
                    }
                }
            }
        }
    }

    #[test]
    fn frame_embed_sums_the_codebook_rows_in_f32_and_rounds_once() {
        let Some(r) = rig() else { return };
        let (rows, d, n) = (64usize, 32usize, 37usize);
        let mut rng = Lcg(5);
        let table: Vec<f32> = (0..rows * d).map(|_| rng.bf()).collect();
        let ids: Vec<u32> = (0..n).map(|k| ((k * 7) % rows) as u32).collect();
        let tb = r.kit.upload_u32("t-table", &pack_bf16(&table, rows * d / 2));
        let ib = r.kit.upload_u32("t-ids", &ids);
        let out = r.kit.zeroed("t-out", d * 2);
        let p = r.kit.uni("t-p", &VtParams { words: (d / 2) as u32, a: n as u32, ..Default::default() });
        r.run("vt_frame_embed", &[(5, &tb), (6, &ib), (7, &out), (8, &p)], r.kit.grid_flat(d / 2, 256));
        let got = unpack_bf16(&r.kit.read_u32(&out, d / 2).unwrap());
        for c in 0..d {
            let want = bf16::from_f32(ids.iter().map(|&i| table[i as usize * d + c]).sum::<f32>()).to_f32();
            assert_eq!(got[c], want, "column {c}");
        }
    }

    #[test]
    fn the_q8_0_gemv_matches_the_host_product_over_the_dequantized_rows() {
        let Some(r) = rig() else { return };
        let (n, k) = (76usize, 1024usize);
        let mut rng = Lcg(29);
        let w: Vec<u16> = (0..n * k).map(|_| bf16::from_f32(rng.f()).to_bits()).collect();
        let x: Vec<f32> = (0..FLOW_ROWS * k).map(|_| rng.bf()).collect();
        let (wq, sq) = q8_rows(&w, k);
        let deq: Vec<f32> = (0..n * k)
            .map(|e| {
                let b = e / Q8_0_BLOCK;
                let d = half::f16::from_bits(((sq[b / 2] >> (16 * (b % 2))) & 0xffff) as u16).to_f32();
                d * (((wq[e / 4] >> (8 * (e % 4))) & 0xff) as u8 as i8) as f32
            })
            .collect();
        for (e, &bits) in w.iter().enumerate().take(64) {
            let v = bf16::from_bits(bits).to_f32();
            assert!((deq[e] - v).abs() <= 1.0 / 127.0 + 1e-6, "q8_0 element {e}: {} vs {v}", deq[e]);
        }
        let wb = r.kit.upload_u32("t-wq", &wq);
        let sb = r.kit.upload_u32("t-sq", &sq);
        let xb = r.kit.upload_u32("t-x", &pack_bf16(&x, FLOW_ROWS * k / 2));
        let dot = |m: usize, c: usize| (0..k).map(|i| x[m * k + i] * deq[c * k + i]).sum::<f32>();
        for (m, flags, spp) in [(FLOW_ROWS, 0, 1), (FLOW_ROWS, VG_GATE_UP_SILU, 2), (1, 0, 4)] {
            let gu = flags & VG_GATE_UP_SILU != 0;
            let rr = if gu { 4 } else { 2 };
            let src = compose(&format!("{VX_SHARED_WGSL}{}", gemv_source(m, rr, true, r.kit.ctx.caps.subgroup)));
            let cols = if gu { n / 2 } else { n };
            let units = cols / 2;
            let grid = r.kit.grid_rows(units.div_ceil(8 / spp));
            let y = r.kit.zeroed("t-y", FLOW_ROWS * n * 2);
            let p = r.kit.uni(
                "t-p",
                &VtGemvParams { n_rows: n as u32, k_vec: (k / 16) as u32, groups_x: grid.0, flags, x_off_vec4: 0, sg_per_unit: spp as u32, m: m as u32, eps: 0.0 },
            );
            let scope = r.kit.ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let pass = r.kit.pass(&src, "vt_gemv", &[(27, &wb), (28, &xb), (29, &y), (30, &p), (33, &sb)], grid).unwrap();
            r.kit.submit(std::slice::from_ref(&pass));
            if let Some(e) = dispatch::pop_scope(scope) {
                panic!("validation: {e}");
            }
            let got = unpack_bf16(&r.kit.read_u32(&y, FLOW_ROWS * n / 2).unwrap());
            for mi in 0..m {
                for c in 0..cols {
                    let want = if gu {
                        let (g, u) = (bf16::from_f32(dot(mi, c)).to_f32(), bf16::from_f32(dot(mi, c + cols)).to_f32());
                        g / (1.0 + (-g).exp()) * u
                    } else {
                        dot(mi, c)
                    };
                    let g = got[mi * cols + c];
                    assert!((g - want).abs() <= 1e-2 * want.abs().max(1e-2), "q8 m {m} flags {flags} row {mi} col {c}: {g} vs {want}");
                }
            }
        }
    }
}
