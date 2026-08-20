use anyhow::{anyhow, ensure, Result};
use nv_kernels::wgpu_backend::dispatch::{GpuTensor, GpuUniform, Recorded};
use nv_kernels::wgpu_backend::kernels::attn_decode_small_m as smk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::chain::{accept_prefix_argmax, build_chain_batch};
use crate::suffix_automaton::SuffixAutomaton;

pub const MAX_HEAD_DIM: usize = 64;

pub const SMALLM_ENV: &str = "NV_WGPU_SPEC_SMALLM";

const WGSL: &str = r#"
struct SpParams {
  h: u32,
  nh: u32,
  nkv: u32,
  hd: u32,
  inter: u32,
  vocab: u32,
  max_seq: u32,
  eps: f32,
  scale: f32,
  theta: f32,
  pad0: u32,
  pad1: u32,
};

struct SpStep {
  committed: u32,
  kb: u32,
  pad0: u32,
  pad1: u32,
};

@group(0) @binding(0) var<uniform> P: SpParams;
@group(0) @binding(1) var<uniform> S: SpStep;
@group(0) @binding(2) var<storage, read> tokens: array<u32>;
@group(0) @binding(4) var<storage, read> embed: array<f32>;
@group(0) @binding(5) var<storage, read> ln1: array<f32>;
@group(0) @binding(6) var<storage, read> wq: array<f32>;
@group(0) @binding(7) var<storage, read> wk: array<f32>;
@group(0) @binding(8) var<storage, read> wv: array<f32>;
@group(0) @binding(9) var<storage, read> wo: array<f32>;
@group(0) @binding(10) var<storage, read> ln2: array<f32>;
@group(0) @binding(11) var<storage, read> wg: array<f32>;
@group(0) @binding(12) var<storage, read> wu: array<f32>;
@group(0) @binding(13) var<storage, read> wd: array<f32>;
@group(0) @binding(14) var<storage, read> lnf: array<f32>;
@group(0) @binding(15) var<storage, read> wlm: array<f32>;
@group(0) @binding(16) var<storage, read_write> x: array<f32>;
@group(0) @binding(17) var<storage, read_write> xn: array<f32>;
@group(0) @binding(18) var<storage, read_write> qb: array<f32>;
@group(0) @binding(19) var<storage, read_write> kbuf: array<f32>;
@group(0) @binding(20) var<storage, read_write> vbuf: array<f32>;
@group(0) @binding(21) var<storage, read_write> kc: array<f32>;
@group(0) @binding(22) var<storage, read_write> vc: array<f32>;
@group(0) @binding(23) var<storage, read_write> ao: array<f32>;
@group(0) @binding(24) var<storage, read_write> h1: array<f32>;
@group(0) @binding(25) var<storage, read_write> tn: array<f32>;
@group(0) @binding(26) var<storage, read_write> act: array<f32>;
@group(0) @binding(27) var<storage, read_write> h2: array<f32>;
@group(0) @binding(28) var<storage, read_write> xf: array<f32>;
@group(0) @binding(29) var<storage, read_write> logits: array<f32>;
@group(0) @binding(30) var<storage, read_write> amax: array<u32>;
@group(0) @binding(31) var<storage, read> rc: array<f32>;
@group(0) @binding(32) var<storage, read> rs: array<f32>;

@compute @workgroup_size(64)
fn sp_embed(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.h) { return; }
  let row = idx / P.h;
  let c = idx % P.h;
  x[idx] = embed[tokens[row] * P.h + c];
}

@compute @workgroup_size(64)
fn sp_rms1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.h) { return; }
  let row = idx / P.h;
  let c = idx % P.h;
  var ss = 0.0;
  for (var i = 0u; i < P.h; i++) {
    let v = x[row * P.h + i];
    ss = ss + v * v;
  }
  let rr = 1.0 / sqrt(ss / f32(P.h) + P.eps);
  xn[idx] = x[idx] * rr * ln1[c];
}

@compute @workgroup_size(64)
fn sp_qkv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let qdim = P.nh * P.hd;
  let kvdim = P.nkv * P.hd;
  let per = qdim + 2u * kvdim;
  if (idx >= S.kb * per) { return; }
  let row = idx / per;
  var o = idx % per;
  var accv = 0.0;
  if (o < qdim) {
    for (var i = 0u; i < P.h; i++) {
      accv = accv + xn[row * P.h + i] * wq[o * P.h + i];
    }
    qb[row * qdim + o] = accv;
  } else if (o < qdim + kvdim) {
    o = o - qdim;
    for (var i = 0u; i < P.h; i++) {
      accv = accv + xn[row * P.h + i] * wk[o * P.h + i];
    }
    kbuf[row * kvdim + o] = accv;
  } else {
    o = o - qdim - kvdim;
    for (var i = 0u; i < P.h; i++) {
      accv = accv + xn[row * P.h + i] * wv[o * P.h + i];
    }
    vbuf[row * kvdim + o] = accv;
  }
}

@compute @workgroup_size(64)
fn sp_rope(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let half = P.hd / 2u;
  let nq = P.nh * half;
  let nk = P.nkv * half;
  if (idx >= S.kb * (nq + nk)) { return; }
  let row = idx / (nq + nk);
  var r = idx % (nq + nk);
  let pos = S.committed + row;
  if (r < nq) {
    let head = r / half;
    let j = r % half;
    let base = row * P.nh * P.hd + head * P.hd;
    let c = rc[pos * half + j];
    let s = rs[pos * half + j];
    let a = qb[base + j];
    let b = qb[base + j + half];
    qb[base + j] = a * c - b * s;
    qb[base + j + half] = a * s + b * c;
  } else {
    r = r - nq;
    let head = r / half;
    let j = r % half;
    let base = row * P.nkv * P.hd + head * P.hd;
    let c = rc[pos * half + j];
    let s = rs[pos * half + j];
    let a = kbuf[base + j];
    let b = kbuf[base + j + half];
    kbuf[base + j] = a * c - b * s;
    kbuf[base + j + half] = a * s + b * c;
  }
}

@compute @workgroup_size(64)
fn sp_kvwrite(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  let kvdim = P.nkv * P.hd;
  if (idx >= S.kb * kvdim) { return; }
  let row = idx / kvdim;
  let r = idx % kvdim;
  let slot = S.committed + row;
  kc[slot * kvdim + r] = kbuf[idx];
  vc[slot * kvdim + r] = vbuf[idx];
}

@compute @workgroup_size(64)
fn sp_attn(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.nh) { return; }
  let row = idx / P.nh;
  let head = idx % P.nh;
  let grp = P.nh / P.nkv;
  let kvh = head / grp;
  let qdim = P.nh * P.hd;
  let kvdim = P.nkv * P.hd;
  let qoff = row * qdim + head * P.hd;
  var m = -3.0e38;
  var l = 0.0;
  var acc: array<f32, 64>;
  for (var d = 0u; d < P.hd; d++) { acc[d] = 0.0; }
  let total = S.committed + row + 1u;
  for (var s = 0u; s < total; s++) {
    let koff = s * kvdim + kvh * P.hd;
    var dt = 0.0;
    for (var d = 0u; d < P.hd; d++) {
      dt = dt + qb[qoff + d] * kc[koff + d];
    }
    let sc = dt * P.scale;
    if (sc > m) {
      let coef = exp(m - sc);
      l = l * coef;
      for (var d = 0u; d < P.hd; d++) { acc[d] = acc[d] * coef; }
      m = sc;
    }
    let w = exp(sc - m);
    l = l + w;
    for (var d = 0u; d < P.hd; d++) {
      acc[d] = acc[d] + w * vc[koff + d];
    }
  }
  for (var d = 0u; d < P.hd; d++) { ao[qoff + d] = acc[d] / l; }
}

@compute @workgroup_size(64)
fn sp_oproj(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.h) { return; }
  let row = idx / P.h;
  let c = idx % P.h;
  let qdim = P.nh * P.hd;
  var accv = 0.0;
  for (var o = 0u; o < qdim; o++) {
    accv = accv + ao[row * qdim + o] * wo[c * qdim + o];
  }
  h1[idx] = x[idx] + accv;
}

@compute @workgroup_size(64)
fn sp_rms2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.h) { return; }
  let row = idx / P.h;
  let c = idx % P.h;
  var ss = 0.0;
  for (var i = 0u; i < P.h; i++) {
    let v = h1[row * P.h + i];
    ss = ss + v * v;
  }
  let rr = 1.0 / sqrt(ss / f32(P.h) + P.eps);
  tn[idx] = h1[idx] * rr * ln2[c];
}

@compute @workgroup_size(64)
fn sp_gateup(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.inter) { return; }
  let row = idx / P.inter;
  let mcol = idx % P.inter;
  var g = 0.0;
  var u = 0.0;
  for (var i = 0u; i < P.h; i++) {
    let t = tn[row * P.h + i];
    g = g + t * wg[mcol * P.h + i];
    u = u + t * wu[mcol * P.h + i];
  }
  act[idx] = (g / (1.0 + exp(-g))) * u;
}

@compute @workgroup_size(64)
fn sp_down(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.h) { return; }
  let row = idx / P.h;
  let c = idx % P.h;
  var accv = 0.0;
  for (var mcol = 0u; mcol < P.inter; mcol++) {
    accv = accv + act[row * P.inter + mcol] * wd[c * P.inter + mcol];
  }
  h2[idx] = h1[idx] + accv;
}

@compute @workgroup_size(64)
fn sp_rmsf(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.h) { return; }
  let row = idx / P.h;
  let c = idx % P.h;
  var ss = 0.0;
  for (var i = 0u; i < P.h; i++) {
    let v = h2[row * P.h + i];
    ss = ss + v * v;
  }
  let rr = 1.0 / sqrt(ss / f32(P.h) + P.eps);
  xf[idx] = h2[idx] * rr * lnf[c];
}

@compute @workgroup_size(64)
fn sp_logits(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb * P.vocab) { return; }
  let row = idx / P.vocab;
  let v = idx % P.vocab;
  var accv = 0.0;
  for (var i = 0u; i < P.h; i++) {
    accv = accv + xf[row * P.h + i] * wlm[v * P.h + i];
  }
  logits[idx] = accv;
}

@compute @workgroup_size(64)
fn sp_argmax(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx = gid.x;
  if (idx >= S.kb) { return; }
  var best = logits[idx * P.vocab];
  var bi = 0u;
  for (var v = 1u; v < P.vocab; v++) {
    let val = logits[idx * P.vocab + v];
    if (val > best) {
      best = val;
      bi = v;
    }
  }
  amax[idx] = bi;
}
"#;

pub const SP_ENTRIES: &[&str] = &[
    "sp_embed",
    "sp_rms1",
    "sp_qkv",
    "sp_rope",
    "sp_kvwrite",
    "sp_attn",
    "sp_oproj",
    "sp_rms2",
    "sp_gateup",
    "sp_down",
    "sp_rmsf",
    "sp_logits",
    "sp_argmax",
];

pub fn sp_wgsl() -> &'static str {
    WGSL
}

type SpParamsRaw = [u32; 12];
type SpStepRaw = [u32; 4];
type SmParamsRaw = [u32; 8];

fn sm_params_raw(d: &SpecDims, m_rows: usize, total: usize) -> SmParamsRaw {
    [
        d.nh as u32,
        d.nkv as u32,
        d.hd as u32,
        total as u32,
        m_rows as u32,
        0,
        d.scale().to_bits(),
        0,
    ]
}

fn smallm_env() -> bool {
    std::env::var(SMALLM_ENV).ok().as_deref() == Some("1")
}

fn smallm_supported(ctx: &WgpuContext, dims: &SpecDims, k_max: usize) -> bool {
    let scratch = (smk::MAX_M * smk::MAX_HEAD_DIM * 4) as u32 + smk::WORKGROUP_SIZE * 4;
    k_max <= smk::MAX_M
        && dims.hd <= smk::MAX_HEAD_DIM
        && ctx.caps.max_compute_invocations_per_workgroup >= smk::WORKGROUP_SIZE
        && ctx.caps.max_compute_workgroup_size_x >= smk::WORKGROUP_SIZE
        && ctx.caps.workgroup_storage_fits(scratch)
}

fn params_raw(d: &SpecDims) -> SpParamsRaw {
    [
        d.h as u32,
        d.nh as u32,
        d.nkv as u32,
        d.hd as u32,
        d.inter as u32,
        d.vocab as u32,
        d.max_seq as u32,
        d.eps.to_bits(),
        d.scale().to_bits(),
        d.rope_theta.to_bits(),
        0,
        0,
    ]
}

fn step_raw(committed: usize, kb: usize) -> SpStepRaw {
    [committed as u32, kb as u32, 0, 0]
}

#[derive(Clone, Copy, Debug)]
pub struct SpecDims {
    pub h: usize,
    pub nh: usize,
    pub nkv: usize,
    pub hd: usize,
    pub inter: usize,
    pub vocab: usize,
    pub max_seq: usize,
    pub eps: f32,
    pub rope_theta: f32,
}

impl SpecDims {
    pub fn qdim(&self) -> usize {
        self.nh * self.hd
    }

    pub fn kvdim(&self) -> usize {
        self.nkv * self.hd
    }

    pub fn scale(&self) -> f32 {
        1.0 / (self.hd as f32).sqrt()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(self.h > 0 && self.nh > 0 && self.nkv > 0 && self.hd > 0);
        ensure!(self.inter > 0 && self.vocab > 1 && self.max_seq > 0);
        ensure!(
            self.hd <= MAX_HEAD_DIM,
            "hd {} exceeds {MAX_HEAD_DIM}",
            self.hd
        );
        ensure!(self.hd % 2 == 0, "hd {} must be even for rope", self.hd);
        ensure!(
            self.nh % self.nkv == 0,
            "nh {} not a multiple of nkv {}",
            self.nh,
            self.nkv
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SpecWeights {
    pub embed: Vec<f32>,
    pub ln1: Vec<f32>,
    pub wq: Vec<f32>,
    pub wk: Vec<f32>,
    pub wv: Vec<f32>,
    pub wo: Vec<f32>,
    pub ln2: Vec<f32>,
    pub wg: Vec<f32>,
    pub wu: Vec<f32>,
    pub wd: Vec<f32>,
    pub lnf: Vec<f32>,
    pub wlm: Vec<f32>,
}

fn det_mat(n: usize, seed: f32, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i as f32) * 0.7181 + seed * 13.37).sin() * scale)
        .collect()
}

fn det_norm(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| 1.0 + ((i as f32) * 0.31 + seed).sin() * 0.1)
        .collect()
}

impl SpecWeights {
    pub fn synthetic(d: &SpecDims, seed: u32) -> Self {
        let s = seed as f32;
        let sc = 0.2;
        Self {
            embed: det_mat(d.vocab * d.h, s + 0.1, sc),
            ln1: det_norm(d.h, s + 0.2),
            wq: det_mat(d.qdim() * d.h, s + 0.3, sc),
            wk: det_mat(d.kvdim() * d.h, s + 0.4, sc),
            wv: det_mat(d.kvdim() * d.h, s + 0.5, sc),
            wo: det_mat(d.h * d.qdim(), s + 0.6, sc),
            ln2: det_norm(d.h, s + 0.7),
            wg: det_mat(d.inter * d.h, s + 0.8, sc),
            wu: det_mat(d.inter * d.h, s + 0.9, sc),
            wd: det_mat(d.h * d.inter, s + 1.1, sc),
            lnf: det_norm(d.h, s + 1.2),
            wlm: det_mat(d.vocab * d.h, s + 1.3, sc),
        }
    }

    pub fn check(&self, d: &SpecDims) -> Result<()> {
        ensure!(self.embed.len() == d.vocab * d.h);
        ensure!(self.ln1.len() == d.h && self.ln2.len() == d.h && self.lnf.len() == d.h);
        ensure!(self.wq.len() == d.qdim() * d.h);
        ensure!(self.wk.len() == d.kvdim() * d.h);
        ensure!(self.wv.len() == d.kvdim() * d.h);
        ensure!(self.wo.len() == d.h * d.qdim());
        ensure!(self.wg.len() == d.inter * d.h && self.wu.len() == d.inter * d.h);
        ensure!(self.wd.len() == d.h * d.inter);
        ensure!(self.wlm.len() == d.vocab * d.h);
        Ok(())
    }
}

pub fn rope_tables(d: &SpecDims) -> (Vec<f32>, Vec<f32>) {
    let half = d.hd / 2;
    let mut cos = Vec::with_capacity(d.max_seq * half);
    let mut sin = Vec::with_capacity(d.max_seq * half);
    for pos in 0..d.max_seq {
        for j in 0..half {
            let inv = (d.rope_theta as f64).powf(-((2 * j) as f64) / (d.hd as f64));
            let angle = (pos as f64) * inv;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    (cos, sin)
}

pub struct WgpuSpecModel {
    ctx: &'static WgpuContext,
    dims: SpecDims,
    k_max: usize,
    committed: usize,
    _params: GpuUniform<SpParamsRaw>,
    step: GpuUniform<SpStepRaw>,
    sm: GpuUniform<SmParamsRaw>,
    smallm: bool,
    tokens: GpuTensor<u32>,
    amax: GpuTensor<u32>,
    rec1: Recorded,
    reck: Option<Recorded>,
    replays: usize,
    _resident: Vec<GpuTensor<f32>>,
}

struct BufIdx {
    embed: usize,
    ln1: usize,
    wq: usize,
    wk: usize,
    wv: usize,
    wo: usize,
    ln2: usize,
    wg: usize,
    wu: usize,
    wd: usize,
    lnf: usize,
    wlm: usize,
    x: usize,
    xn: usize,
    qb: usize,
    kbuf: usize,
    vbuf: usize,
    kc: usize,
    vc: usize,
    ao: usize,
    h1: usize,
    tn: usize,
    act: usize,
    h2: usize,
    xf: usize,
    logits: usize,
    rc: usize,
    rs: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpStage {
    X,
    Xn,
    Qb,
    Kbuf,
    Vbuf,
    Kc,
    Vc,
    Ao,
    H1,
    Tn,
    Act,
    H2,
    Xf,
    Logits,
}

impl BufIdx {
    fn slot(&self, s: SpStage) -> usize {
        match s {
            SpStage::X => self.x,
            SpStage::Xn => self.xn,
            SpStage::Qb => self.qb,
            SpStage::Kbuf => self.kbuf,
            SpStage::Vbuf => self.vbuf,
            SpStage::Kc => self.kc,
            SpStage::Vc => self.vc,
            SpStage::Ao => self.ao,
            SpStage::H1 => self.h1,
            SpStage::Tn => self.tn,
            SpStage::Act => self.act,
            SpStage::H2 => self.h2,
            SpStage::Xf => self.xf,
            SpStage::Logits => self.logits,
        }
    }
}

fn alloc_bufs(
    ctx: &'static WgpuContext,
    d: &SpecDims,
    weights: &SpecWeights,
    k_max: usize,
) -> (Vec<GpuTensor<f32>>, BufIdx) {
    let (cos, sin) = rope_tables(d);
    let up = |label: &str, data: &[f32]| GpuTensor::upload(ctx, label, data);
    let zt = |label: &str, len: usize| GpuTensor::<f32>::zeroed(ctx, label, len.max(1));
    let mut resident: Vec<GpuTensor<f32>> = Vec::new();
    let push = |t: GpuTensor<f32>, r: &mut Vec<GpuTensor<f32>>| -> usize {
        r.push(t);
        r.len() - 1
    };
    let idx = BufIdx {
        embed: push(up("sp-embed", &weights.embed), &mut resident),
        ln1: push(up("sp-ln1", &weights.ln1), &mut resident),
        wq: push(up("sp-wq", &weights.wq), &mut resident),
        wk: push(up("sp-wk", &weights.wk), &mut resident),
        wv: push(up("sp-wv", &weights.wv), &mut resident),
        wo: push(up("sp-wo", &weights.wo), &mut resident),
        ln2: push(up("sp-ln2", &weights.ln2), &mut resident),
        wg: push(up("sp-wg", &weights.wg), &mut resident),
        wu: push(up("sp-wu", &weights.wu), &mut resident),
        wd: push(up("sp-wd", &weights.wd), &mut resident),
        lnf: push(up("sp-lnf", &weights.lnf), &mut resident),
        wlm: push(up("sp-wlm", &weights.wlm), &mut resident),
        x: push(zt("sp-x", k_max * d.h), &mut resident),
        xn: push(zt("sp-xn", k_max * d.h), &mut resident),
        qb: push(zt("sp-qb", k_max * d.qdim()), &mut resident),
        kbuf: push(zt("sp-kbuf", k_max * d.kvdim()), &mut resident),
        vbuf: push(zt("sp-vbuf", k_max * d.kvdim()), &mut resident),
        kc: push(zt("sp-kc", d.max_seq * d.kvdim()), &mut resident),
        vc: push(zt("sp-vc", d.max_seq * d.kvdim()), &mut resident),
        ao: push(zt("sp-ao", k_max * d.qdim()), &mut resident),
        h1: push(zt("sp-h1", k_max * d.h), &mut resident),
        tn: push(zt("sp-tn", k_max * d.h), &mut resident),
        act: push(zt("sp-act", k_max * d.inter), &mut resident),
        h2: push(zt("sp-h2", k_max * d.h), &mut resident),
        xf: push(zt("sp-xf", k_max * d.h), &mut resident),
        logits: push(zt("sp-logits", k_max * d.vocab), &mut resident),
        rc: push(up("sp-ropecos", &cos), &mut resident),
        rs: push(up("sp-ropesin", &sin), &mut resident),
    };
    (resident, idx)
}

struct SpBind<'a> {
    params: &'a GpuUniform<SpParamsRaw>,
    step: &'a GpuUniform<SpStepRaw>,
    tokens: &'a GpuTensor<u32>,
    amax: &'a GpuTensor<u32>,
    b: &'a [GpuTensor<f32>],
    idx: &'a BufIdx,
}

type SpPass<'a> = (
    &'static str,
    Vec<(u32, &'a dyn nv_kernels::wgpu_backend::dispatch::GpuBind)>,
    usize,
);

fn sp_pass_table<'a>(d: &SpecDims, kb: usize, r: &SpBind<'a>) -> Vec<SpPass<'a>> {
    let b = r.b;
    let idx = r.idx;
    let half = d.hd / 2;
    let p = r.params;
    let s = r.step;
    vec![
        (
            "sp_embed",
            vec![
                (0u32, p as _),
                (1, s as _),
                (2, r.tokens as _),
                (4, &b[idx.embed] as _),
                (16, &b[idx.x] as _),
            ],
            kb * d.h,
        ),
        (
            "sp_rms1",
            vec![
                (0, p as _),
                (1, s as _),
                (16, &b[idx.x] as _),
                (5, &b[idx.ln1] as _),
                (17, &b[idx.xn] as _),
            ],
            kb * d.h,
        ),
        (
            "sp_qkv",
            vec![
                (0, p as _),
                (1, s as _),
                (17, &b[idx.xn] as _),
                (6, &b[idx.wq] as _),
                (7, &b[idx.wk] as _),
                (8, &b[idx.wv] as _),
                (18, &b[idx.qb] as _),
                (19, &b[idx.kbuf] as _),
                (20, &b[idx.vbuf] as _),
            ],
            kb * (d.qdim() + 2 * d.kvdim()),
        ),
        (
            "sp_rope",
            vec![
                (0, p as _),
                (1, s as _),
                (18, &b[idx.qb] as _),
                (19, &b[idx.kbuf] as _),
                (31, &b[idx.rc] as _),
                (32, &b[idx.rs] as _),
            ],
            kb * (d.nh * half + d.nkv * half),
        ),
        (
            "sp_kvwrite",
            vec![
                (0, p as _),
                (1, s as _),
                (19, &b[idx.kbuf] as _),
                (20, &b[idx.vbuf] as _),
                (21, &b[idx.kc] as _),
                (22, &b[idx.vc] as _),
            ],
            kb * d.kvdim(),
        ),
        (
            "sp_attn",
            vec![
                (0, p as _),
                (1, s as _),
                (18, &b[idx.qb] as _),
                (21, &b[idx.kc] as _),
                (22, &b[idx.vc] as _),
                (23, &b[idx.ao] as _),
            ],
            kb * d.nh,
        ),
        (
            "sp_oproj",
            vec![
                (0, p as _),
                (1, s as _),
                (23, &b[idx.ao] as _),
                (9, &b[idx.wo] as _),
                (16, &b[idx.x] as _),
                (24, &b[idx.h1] as _),
            ],
            kb * d.h,
        ),
        (
            "sp_rms2",
            vec![
                (0, p as _),
                (1, s as _),
                (24, &b[idx.h1] as _),
                (10, &b[idx.ln2] as _),
                (25, &b[idx.tn] as _),
            ],
            kb * d.h,
        ),
        (
            "sp_gateup",
            vec![
                (0, p as _),
                (1, s as _),
                (25, &b[idx.tn] as _),
                (11, &b[idx.wg] as _),
                (12, &b[idx.wu] as _),
                (26, &b[idx.act] as _),
            ],
            kb * d.inter,
        ),
        (
            "sp_down",
            vec![
                (0, p as _),
                (1, s as _),
                (26, &b[idx.act] as _),
                (13, &b[idx.wd] as _),
                (24, &b[idx.h1] as _),
                (27, &b[idx.h2] as _),
            ],
            kb * d.h,
        ),
        (
            "sp_rmsf",
            vec![
                (0, p as _),
                (1, s as _),
                (27, &b[idx.h2] as _),
                (14, &b[idx.lnf] as _),
                (28, &b[idx.xf] as _),
            ],
            kb * d.h,
        ),
        (
            "sp_logits",
            vec![
                (0, p as _),
                (1, s as _),
                (28, &b[idx.xf] as _),
                (15, &b[idx.wlm] as _),
                (29, &b[idx.logits] as _),
            ],
            kb * d.vocab,
        ),
        (
            "sp_argmax",
            vec![
                (0, p as _),
                (1, s as _),
                (29, &b[idx.logits] as _),
                (30, r.amax as _),
            ],
            kb,
        ),
    ]
}

fn groups_1d(total: usize) -> (u32, u32, u32) {
    (((total + 63) / 64) as u32, 1, 1)
}

pub struct SpProbe {
    ctx: &'static WgpuContext,
    dims: SpecDims,
    k_max: usize,
    kb: usize,
    params: GpuUniform<SpParamsRaw>,
    step: GpuUniform<SpStepRaw>,
    tokens: GpuTensor<u32>,
    amax: GpuTensor<u32>,
    b: Vec<GpuTensor<f32>>,
    idx: BufIdx,
}

impl SpProbe {
    pub fn new(
        ctx: &'static WgpuContext,
        dims: SpecDims,
        weights: &SpecWeights,
        k_max: usize,
    ) -> Result<Self> {
        dims.validate()?;
        weights.check(&dims)?;
        ensure!(k_max >= 1, "k_max must be >= 1");
        ensure!(k_max <= dims.max_seq, "k_max exceeds max_seq");
        let (b, idx) = alloc_bufs(ctx, &dims, weights, k_max);
        let params = GpuUniform::new(ctx, "sp-probe-params", &params_raw(&dims));
        let step = GpuUniform::new(ctx, "sp-probe-step", &step_raw(0, 1));
        let tokens = GpuTensor::<u32>::zeroed(ctx, "sp-probe-tokens", k_max);
        let amax = GpuTensor::<u32>::zeroed(ctx, "sp-probe-amax", k_max);
        Ok(Self {
            ctx,
            dims,
            k_max,
            kb: 1,
            params,
            step,
            tokens,
            amax,
            b,
            idx,
        })
    }

    pub fn dims(&self) -> &SpecDims {
        &self.dims
    }

    pub fn set_step(&mut self, committed: usize, kb: usize) {
        assert!(
            kb >= 1 && kb <= self.k_max,
            "sp probe: kb {kb} outside 1..={}",
            self.k_max
        );
        assert!(
            committed + kb <= self.dims.max_seq,
            "sp probe: committed {committed} + kb {kb} exceeds max_seq {}",
            self.dims.max_seq
        );
        self.kb = kb;
        self.step.write(self.ctx, &step_raw(committed, kb));
    }

    pub fn stage_len(&self, s: SpStage) -> usize {
        self.b[self.idx.slot(s)].len()
    }

    pub fn write_stage(&self, s: SpStage, data: &[f32]) {
        let t = &self.b[self.idx.slot(s)];
        assert_eq!(
            data.len(),
            t.len(),
            "sp probe: stage {s:?} takes {} elements, got {}",
            t.len(),
            data.len()
        );
        t.write(self.ctx, data)
            .unwrap_or_else(|e| panic!("sp probe: write {s:?}: {e}"));
    }

    pub fn read_stage(&self, s: SpStage) -> Vec<f32> {
        self.b[self.idx.slot(s)]
            .download(self.ctx)
            .unwrap_or_else(|e| panic!("sp probe: read {s:?}: {e}"))
    }

    pub fn write_tokens(&self, toks: &[u32]) {
        assert!(
            toks.len() <= self.k_max,
            "sp probe: {} tokens exceeds k_max {}",
            toks.len(),
            self.k_max
        );
        let mut padded = vec![0u32; self.k_max];
        padded[..toks.len()].copy_from_slice(toks);
        self.tokens
            .write(self.ctx, &padded)
            .unwrap_or_else(|e| panic!("sp probe: write tokens: {e}"));
    }

    pub fn read_amax(&self) -> Vec<u32> {
        let all = self
            .amax
            .download(self.ctx)
            .unwrap_or_else(|e| panic!("sp probe: read amax: {e}"));
        all[..self.kb].to_vec()
    }

    pub fn run(&self, entry: &str) {
        let bind = SpBind {
            params: &self.params,
            step: &self.step,
            tokens: &self.tokens,
            amax: &self.amax,
            b: &self.b,
            idx: &self.idx,
        };
        let table = sp_pass_table(&self.dims, self.kb, &bind);
        let (name, bindings, total) = table
            .into_iter()
            .find(|(n, _, _)| *n == entry)
            .unwrap_or_else(|| {
                panic!("sp probe: entry {entry} is not in the shipped pass table {SP_ENTRIES:?}")
            });
        let mut rec = Recorded::new();
        rec.push(
            self.ctx,
            &format!("nv_specdecode_{name}"),
            WGSL,
            name,
            &bindings,
            groups_1d(total),
        )
        .unwrap_or_else(|e| panic!("sp probe: record {name}: {e}"));
        rec.replay(self.ctx)
            .unwrap_or_else(|e| panic!("sp probe: replay {name}: {e}"));
    }
}

impl WgpuSpecModel {
    pub fn new(
        ctx: &'static WgpuContext,
        dims: SpecDims,
        weights: &SpecWeights,
        k_max: usize,
    ) -> Result<Self> {
        Self::new_with_smallm(ctx, dims, weights, k_max, smallm_env())
    }

    pub fn new_with_smallm(
        ctx: &'static WgpuContext,
        dims: SpecDims,
        weights: &SpecWeights,
        k_max: usize,
        smallm: bool,
    ) -> Result<Self> {
        dims.validate()?;
        weights.check(&dims)?;
        ensure!(k_max >= 1, "k_max must be >= 1");
        ensure!(k_max <= dims.max_seq, "k_max exceeds max_seq");
        let smallm = if smallm && !smallm_supported(ctx, &dims, k_max) {
            eprintln!(
                "[wgpu_spec] small-m attention requested but unsupported here (k_max {k_max} > {} or hd {} > {} or caps); falling back to sp_attn",
                smk::MAX_M,
                dims.hd,
                smk::MAX_HEAD_DIM
            );
            false
        } else {
            smallm
        };

        let (resident, idx) = alloc_bufs(ctx, &dims, weights, k_max);

        let params = GpuUniform::new(ctx, "sp-params", &params_raw(&dims));
        let step = GpuUniform::new(ctx, "sp-step", &step_raw(0, 1));
        let sm = GpuUniform::new(ctx, "sp-sm-params", &sm_params_raw(&dims, 1, 1));
        let sm_src = compose(smk::WGSL);
        let tokens = GpuTensor::<u32>::zeroed(ctx, "sp-tokens", k_max);
        let amax = GpuTensor::<u32>::zeroed(ctx, "sp-amax", k_max);

        let record = |kb: usize| -> Result<Recorded> {
            let mut rec = Recorded::new();
            let d = &dims;
            let b = &resident;
            let bind = SpBind {
                params: &params,
                step: &step,
                tokens: &tokens,
                amax: &amax,
                b,
                idx: &idx,
            };
            let passes = sp_pass_table(d, kb, &bind);
            for (entry, bindings, total) in passes {
                if smallm && entry == "sp_attn" {
                    rec.push(
                        ctx,
                        "nv_specdecode_sp_attn_small_m",
                        &sm_src,
                        "attn_decode_small_m_f32",
                        &[
                            (0, &b[idx.qb] as _),
                            (1, &b[idx.kc] as _),
                            (2, &b[idx.vc] as _),
                            (3, &b[idx.ao] as _),
                            (4, &sm as _),
                        ],
                        (d.nh as u32, 1, 1),
                    )
                    .map_err(|e| anyhow!("record sp_attn_small_m: {e}"))?;
                    continue;
                }
                rec.push(
                    ctx,
                    &format!("nv_specdecode_{entry}"),
                    WGSL,
                    entry,
                    &bindings,
                    groups_1d(total),
                )
                .map_err(|e| anyhow!("record {entry}: {e}"))?;
            }
            Ok(rec)
        };

        let rec1 = record(1)?;
        let reck = if k_max > 1 {
            Some(record(k_max)?)
        } else {
            None
        };

        Ok(Self {
            ctx,
            dims,
            k_max,
            committed: 0,
            _params: params,
            step,
            sm,
            smallm,
            tokens,
            amax,
            rec1,
            reck,
            replays: 0,
            _resident: resident,
        })
    }

    pub fn smallm(&self) -> bool {
        self.smallm
    }

    pub fn dims(&self) -> &SpecDims {
        &self.dims
    }

    pub fn committed(&self) -> usize {
        self.committed
    }

    pub fn k_max(&self) -> usize {
        self.k_max
    }

    pub fn pass_count(&self) -> usize {
        self.rec1.len() + self.reck.as_ref().map_or(0, |r| r.len())
    }

    pub fn replays(&self) -> usize {
        self.replays
    }

    pub fn reset(&mut self) {
        self.committed = 0;
    }

    fn write_inputs(&mut self, batch: &[u32], kb: usize) -> Result<()> {
        let mut padded = vec![0u32; self.k_max];
        padded[..batch.len()].copy_from_slice(batch);
        self.tokens
            .write(self.ctx, &padded)
            .map_err(|e| anyhow!("tokens write: {e}"))?;
        self.step.write(self.ctx, &step_raw(self.committed, kb));
        if self.smallm {
            self.sm.write(
                self.ctx,
                &sm_params_raw(&self.dims, kb, self.committed + kb),
            );
        }
        Ok(())
    }

    fn read_amax(&self, kb: usize) -> Result<Vec<u32>> {
        let all = self
            .amax
            .download(self.ctx)
            .map_err(|e| anyhow!("amax readback: {e}"))?;
        Ok(all[..kb].to_vec())
    }

    pub fn decode1(&mut self, token: u32) -> Result<u32> {
        ensure!(
            self.committed + 1 <= self.dims.max_seq,
            "decode1 overflow: committed={} max_seq={}",
            self.committed,
            self.dims.max_seq
        );
        self.write_inputs(&[token], 1)?;
        self.rec1
            .replay(self.ctx)
            .map_err(|e| anyhow!("decode1 replay: {e}"))?;
        self.replays += 1;
        self.committed += 1;
        Ok(self.read_amax(1)?[0])
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let mut last = 0;
        for &t in tokens {
            last = self.decode1(t)?;
        }
        Ok(last)
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        ensure!(
            self.k_max > 1,
            "verify_chain needs a model built with k_max > 1"
        );
        ensure!(
            batch.len() == self.k_max,
            "verify_chain batch len {} != k_max {}",
            batch.len(),
            self.k_max
        );
        ensure!(
            self.committed + self.k_max <= self.dims.max_seq,
            "verify_chain overflow: committed={} k={} max_seq={}",
            self.committed,
            self.k_max,
            self.dims.max_seq
        );
        self.write_inputs(batch, self.k_max)?;
        let rec = self.reck.as_mut().expect("reck exists when k_max > 1");
        rec.replay(self.ctx)
            .map_err(|e| anyhow!("verify replay: {e}"))?;
        self.replays += 1;
        self.read_amax(self.k_max)
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        ensure!(
            self.committed + n <= self.dims.max_seq,
            "advance past max_seq"
        );
        self.committed += n;
        Ok(())
    }

    pub fn rollback_to(&mut self, n: usize) -> Result<()> {
        ensure!(
            n <= self.committed,
            "rollback_to {} beyond committed {}",
            n,
            self.committed
        );
        self.committed = n;
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    pub emitted: Vec<u32>,
    pub rounds: usize,
    pub drafted: usize,
    pub accepted_drafts: usize,
}

impl SpecStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            return 0.0;
        }
        self.accepted_drafts as f64 / self.drafted as f64
    }
}

pub struct WgpuChainSpec {
    pub drafter: WgpuSpecModel,
    pub verifier: WgpuSpecModel,
    k: usize,
}

impl WgpuChainSpec {
    pub fn new(
        ctx: &'static WgpuContext,
        dims: SpecDims,
        verifier_weights: &SpecWeights,
        drafter_weights: &SpecWeights,
        k: usize,
    ) -> Result<Self> {
        Self::new_with_smallm(
            ctx,
            dims,
            verifier_weights,
            drafter_weights,
            k,
            smallm_env(),
        )
    }

    pub fn new_with_smallm(
        ctx: &'static WgpuContext,
        dims: SpecDims,
        verifier_weights: &SpecWeights,
        drafter_weights: &SpecWeights,
        k: usize,
        smallm: bool,
    ) -> Result<Self> {
        ensure!(k >= 2, "chain spec needs k >= 2, got {k}");
        let verifier = WgpuSpecModel::new_with_smallm(ctx, dims, verifier_weights, k, smallm)?;
        let drafter = WgpuSpecModel::new_with_smallm(ctx, dims, drafter_weights, 1, smallm)?;
        Ok(Self {
            drafter,
            verifier,
            k,
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn generate(&mut self, prompt: &[u32], n_tokens: usize) -> Result<SpecStats> {
        ensure!(!prompt.is_empty(), "generate needs a non-empty prompt");
        self.verifier.reset();
        self.drafter.reset();
        let mut context = prompt.to_vec();
        let mut bonus = self.verifier.prefill(prompt)?;
        self.drafter.prefill(prompt)?;

        let mut stats = SpecStats::default();
        while stats.emitted.len() < n_tokens {
            if self.verifier.committed() + self.k > self.verifier.dims().max_seq {
                break;
            }
            while self.drafter.committed() < context.len() {
                let t = context[self.drafter.committed()];
                self.drafter.decode1(t)?;
            }
            let base = context.len();
            let mut drafts = Vec::with_capacity(self.k - 1);
            let mut cur = bonus;
            for _ in 0..self.k - 1 {
                let t = self.drafter.decode1(cur)?;
                drafts.push(t);
                cur = t;
            }
            let batch = build_chain_batch(bonus, &drafts, self.k, true)?;
            let amax = self.verifier.verify_chain(&batch)?;
            let acc = accept_prefix_argmax(&batch, &amax)?;
            self.verifier.advance(acc.commit_len)?;
            context.extend_from_slice(&batch[..acc.commit_len]);
            stats.emitted.extend_from_slice(&batch[..acc.commit_len]);
            self.drafter
                .rollback_to(base + acc.commit_len.min(self.k - 1))?;
            stats.rounds += 1;
            stats.drafted += self.k - 1;
            stats.accepted_drafts += acc.draft_accepted;
            bonus = acc.next_bonus;
        }
        Ok(stats)
    }
}

pub trait StepDecoder {
    fn label(&self) -> String;
    fn vocab(&self) -> usize;
    fn pos(&self) -> usize;
    fn reset_state(&mut self) -> Result<()>;
    fn step(&mut self, token: u32) -> Result<u32>;
}

impl StepDecoder for WgpuSpecModel {
    fn label(&self) -> String {
        format!(
            "wgpu-spec-synthetic[h={},vocab={}]",
            self.dims.h, self.dims.vocab
        )
    }

    fn vocab(&self) -> usize {
        self.dims.vocab
    }

    fn pos(&self) -> usize {
        self.committed
    }

    fn reset_state(&mut self) -> Result<()> {
        self.reset();
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<u32> {
        self.decode1(token)
    }
}

impl StepDecoder for nv_models::gemma4_wgpu::Gemma4Wgpu {
    fn label(&self) -> String {
        format!("gemma4-wgpu[{}L]", self.config().num_hidden_layers)
    }

    fn vocab(&self) -> usize {
        self.config().vocab_size
    }

    fn pos(&self) -> usize {
        self.current_pos()
    }

    fn reset_state(&mut self) -> Result<()> {
        self.reset();
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<u32> {
        self.decode_step(token)
    }
}

impl StepDecoder for nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu {
    fn label(&self) -> String {
        format!("gemma4-e4b-wgpu[{}L]", self.config().num_hidden_layers)
    }

    fn vocab(&self) -> usize {
        self.config().vocab_size
    }

    fn pos(&self) -> usize {
        self.current_pos()
    }

    fn reset_state(&mut self) -> Result<()> {
        self.reset();
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<u32> {
        self.decode_step(token)
    }
}

impl StepDecoder for nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu {
    fn label(&self) -> String {
        format!("qwen3.5-moe-wgpu[{}L]", self.config().num_hidden_layers)
    }

    fn vocab(&self) -> usize {
        self.config().vocab_size
    }

    fn pos(&self) -> usize {
        self.current_pos()
    }

    fn reset_state(&mut self) -> Result<()> {
        self.reset()
    }

    fn step(&mut self, token: u32) -> Result<u32> {
        self.decode_step(token)
    }
}

pub trait ChainDrafter {
    fn label(&self) -> String;
    fn reset_state(&mut self) -> Result<()>;
    fn observe(&mut self, token: u32) -> Result<Option<u32>>;
}

pub struct ModelDrafter<D: StepDecoder> {
    pub inner: D,
    last: Option<u32>,
}

impl<D: StepDecoder> ModelDrafter<D> {
    pub fn new(inner: D) -> Self {
        Self { inner, last: None }
    }

    pub fn last_proposal(&self) -> Option<u32> {
        self.last
    }
}

impl<D: StepDecoder> ChainDrafter for ModelDrafter<D> {
    fn label(&self) -> String {
        format!("model:{}", self.inner.label())
    }

    fn reset_state(&mut self) -> Result<()> {
        self.last = None;
        self.inner.reset_state()
    }

    fn observe(&mut self, token: u32) -> Result<Option<u32>> {
        let t = self.inner.step(token)?;
        self.last = Some(t);
        Ok(Some(t))
    }
}

pub struct PromptLookupDrafter {
    sam: SuffixAutomaton,
    min_match: usize,
    hits: usize,
    calls: usize,
}

impl PromptLookupDrafter {
    pub fn new(min_match: usize) -> Self {
        Self {
            sam: SuffixAutomaton::new(),
            min_match: min_match.max(1),
            hits: 0,
            calls: 0,
        }
    }

    pub fn hits(&self) -> usize {
        self.hits
    }

    pub fn calls(&self) -> usize {
        self.calls
    }
}

impl ChainDrafter for PromptLookupDrafter {
    fn label(&self) -> String {
        format!("prompt-lookup[min_match={}]", self.min_match)
    }

    fn reset_state(&mut self) -> Result<()> {
        self.sam = SuffixAutomaton::new();
        self.hits = 0;
        self.calls = 0;
        Ok(())
    }

    fn observe(&mut self, token: u32) -> Result<Option<u32>> {
        self.sam.extend(token);
        self.calls += 1;
        match self.sam.propose(1, self.min_match) {
            Some(p) => {
                let t = p.tokens.first().copied();
                if t.is_some() {
                    self.hits += 1;
                }
                Ok(t)
            }
            None => Ok(None),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct RealSpecStats {
    pub k: usize,
    pub emitted: Vec<u32>,
    pub prefill_steps: usize,
    pub verifier_steps: usize,
    pub rounds: usize,
    pub draft_slots: usize,
    pub accepted_drafts: usize,
    pub round_lengths: Vec<usize>,
    pub hit_eos: bool,
    pub truncated_round: bool,
}

impl RealSpecStats {
    pub fn acceptance_rate(&self) -> f64 {
        if self.draft_slots == 0 {
            return 0.0;
        }
        self.accepted_drafts as f64 / self.draft_slots as f64
    }

    pub fn tokens_per_round(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        let committed: usize = self.round_lengths.iter().sum();
        committed as f64 / self.rounds as f64
    }

    pub fn summary(&self) -> String {
        format!(
            "k={} rounds={} emitted={} draft_slots={} accepted={} acceptance_rate={:.3} tokens_per_verifier_batch={:.3} eos={}",
            self.k,
            self.rounds,
            self.emitted.len(),
            self.draft_slots,
            self.accepted_drafts,
            self.acceptance_rate(),
            self.tokens_per_round(),
            self.hit_eos,
        )
    }
}

pub struct LockstepChainSpec<V: StepDecoder, D: ChainDrafter> {
    pub verifier: V,
    pub drafter: D,
    k: usize,
    eos: Vec<u32>,
}

impl<V: StepDecoder, D: ChainDrafter> LockstepChainSpec<V, D> {
    pub fn new(verifier: V, drafter: D, k: usize, eos: Vec<u32>) -> Result<Self> {
        ensure!(k >= 2, "chain spec needs k >= 2, got {k}");
        Ok(Self {
            verifier,
            drafter,
            k,
            eos,
        })
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn eos(&self) -> &[u32] {
        &self.eos
    }

    pub fn greedy(&mut self, prompt: &[u32], max_new: usize) -> Result<Vec<u32>> {
        ensure!(!prompt.is_empty(), "greedy needs a non-empty prompt");
        self.verifier.reset_state()?;
        let mut next = 0u32;
        for &t in prompt {
            next = self.verifier.step(t)?;
        }
        let mut out = Vec::with_capacity(max_new);
        while out.len() < max_new {
            out.push(next);
            if self.eos.contains(&next) {
                break;
            }
            next = self.verifier.step(next)?;
        }
        Ok(out)
    }

    pub fn generate(&mut self, prompt: &[u32], max_new: usize) -> Result<RealSpecStats> {
        ensure!(!prompt.is_empty(), "generate needs a non-empty prompt");
        ensure!(max_new > 0, "generate needs max_new > 0");
        self.verifier.reset_state()?;
        self.drafter.reset_state()?;

        let mut stats = RealSpecStats {
            k: self.k,
            ..Default::default()
        };
        let mut cur = 0u32;
        for &t in prompt {
            cur = self.verifier.step(t)?;
            self.drafter.observe(t)?;
            stats.prefill_steps += 1;
        }

        'outer: loop {
            let mut round_tokens = 0usize;
            let mut round_accepted = 0usize;
            loop {
                if stats.emitted.len() >= max_new {
                    stats.truncated_round = round_tokens > 0;
                    break 'outer;
                }
                stats.emitted.push(cur);
                if self.eos.contains(&cur) {
                    stats.hit_eos = true;
                    stats.truncated_round = round_tokens > 0;
                    break 'outer;
                }
                let v_next = self.verifier.step(cur)?;
                let d_next = self.drafter.observe(cur)?;
                stats.verifier_steps += 1;
                round_tokens += 1;

                let accepted = matches!(d_next, Some(d) if d == v_next);
                let full = round_tokens == self.k;
                if accepted && !full {
                    round_accepted += 1;
                    cur = v_next;
                    continue;
                }
                stats.rounds += 1;
                stats.draft_slots += self.k - 1;
                stats.accepted_drafts += round_accepted;
                stats.round_lengths.push(round_tokens);
                cur = v_next;
                break;
            }
        }
        Ok(stats)
    }
}
