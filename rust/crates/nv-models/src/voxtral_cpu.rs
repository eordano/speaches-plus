use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use candle_core::{CpuStorage, DType, Device, Storage, Tensor, TensorId};
use nv_layers::cpu_qmatvec::{arm, matmul_rows_bf16, par_chunks_mut};
use nv_layers::linear::Linear;

use nv_layers::norm::RmsNorm;
use nv_layers::rope::Rope;

use crate::voxtral::{
    clip_independent_prefix_positions, cpu_profile_report, DecoderLayer, Lap, VoxtralRealtime,
    VoxtralTextConfig, STREAMING_PAD_TOKEN,
};

pub const THE_CPU_STEP_IS_PLAIN_F32_LOOPS_AROUND_THE_QUANTIZED_LINEARS: &str =
    "on CPU every op between the cpu_qmatvec linears (rmsnorm, rope, the KV write, the \
     attention over the cache, silu*up, the residual adds and the argmax) is a loop over host \
     f32 slices in candle's own evaluation order: sequential sums, x/denom*w, a plain product at \
     depth zero then fused multiply-adds for the two attention products (gemm's gemv), so the \
     hidden state is bitwise the candle eager path's and the only tensors a step builds are the \
     linears' inputs and outputs; the streaming prefix is one block so each weight is read once";

pub const KEYS_ARE_STORED_TRANSPOSED_SO_THE_SCORE_LOOP_STREAMS_EIGHT_POSITIONS_PER_FMA: &str =
    "K lives as [kv_head][head_dim][position] padded to eight positions, so one fused \
     multiply-add over consecutive positions keeps every position's accumulation sequential in \
     head_dim (gemm's order); V stays [kv_head][position][head_dim] so the weighted sum streams \
     one position's row per fma";

pub(crate) struct F32Host<'a>(std::sync::RwLockReadGuard<'a, Storage>, usize, usize);

impl std::ops::Deref for F32Host<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match &*self.0 {
            Storage::Cpu(CpuStorage::F32(v)) => &v[self.1..self.2],
            _ => unreachable!("F32Host is only built over contiguous f32 host storage"),
        }
    }
}

pub(crate) fn host_f32(t: &Tensor) -> Result<F32Host<'_>> {
    let (s, l) = t.storage_and_layout();
    if !matches!(&*s, Storage::Cpu(CpuStorage::F32(_))) {
        bail!("voxtral cpu step: {:?} on {:?} is not f32 host storage", t.dtype(), t.device());
    }
    let (o1, o2) = l
        .contiguous_offsets()
        .ok_or_else(|| anyhow!("voxtral cpu step: tensor {:?} is not contiguous", t.dims()))?;
    Ok(F32Host(s, o1, o2))
}

pub const A_DENSE_BF16_LINEAR_ON_CPU_RUNS_THE_LANE_MATVEC_OVER_ITS_HOST_BITS: &str =
    "candle's CPU backend has no bf16 matmul, so a Linear holding a bf16 weight on the host goes \
     through matmul_rows_bf16: the weights are the CUDA path's bits and the accumulation is f32";

pub(crate) fn linear(l: &Linear, x: &[f32], rows: usize) -> Result<Vec<f32>> {
    if let Some(w) = l.weight().filter(|w| w.dtype() == DType::BF16 && w.device().is_cpu()) {
        let _ = A_DENSE_BF16_LINEAR_ON_CPU_RUNS_THE_LANE_MATVEC_OVER_ITS_HOST_BITS;
        let (n, k) = w.dims2()?;
        let (s, layout) = w.storage_and_layout();
        let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| anyhow!("voxtral cpu step: bf16 weight is not contiguous"))?;
        let Storage::Cpu(CpuStorage::BF16(v)) = &*s else { bail!("voxtral cpu step: bf16 weight is not host storage") };
        let bits: &[u16] = unsafe { std::slice::from_raw_parts(v[o1..o2].as_ptr() as *const u16, o2 - o1) };
        let mut out_t = vec![0f32; n * rows];
        matmul_rows_bf16(arm(), bits, k, x, &mut out_t)?;
        return Ok(if rows == 1 { out_t } else { (0..rows * n).map(|i| out_t[(i % n) * rows + i / n]).collect() });
    }
    let y = l.forward(&Tensor::from_slice(x, (rows, x.len() / rows), &Device::Cpu)?)?;
    let out = host_f32(&y)?.to_vec();
    Ok(out)
}

pub(crate) fn rmsnorm_rows(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = w.len();
    let scale = (1.0 / n as f64) as f32;
    for (x, out) in x.chunks_exact(n).zip(out.chunks_exact_mut(n)) {
        let mut s = 0f32;
        for &v in x {
            s += v * v;
        }
        let denom = (s * scale + 0.0 + eps).sqrt();
        for i in 0..n {
            out[i] = x[i] / denom * w[i];
        }
    }
}

fn rope_in_place(x: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = cos.len();
    for head in x.chunks_exact_mut(2 * half) {
        let (lo, hi) = head.split_at_mut(half);
        for i in 0..half {
            let (l, h) = (lo[i], hi[i]);
            lo[i] = l * cos[i] - h * sin[i];
            hi[i] = l * sin[i] + h * cos[i];
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn fma_lanes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
}

fn scores(kt: &[f32], cap: usize, q: &[f32], n_pad: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if fma_lanes() {
        return unsafe { scores_avx2(kt, cap, q, n_pad, out) };
    }
    for j in 0..n_pad {
        let mut acc = kt[j] * q[0];
        for (d, &qd) in q.iter().enumerate().skip(1) {
            acc = kt[d * cap + j].mul_add(qd, acc);
        }
        out[j] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn scores_avx2(kt: &[f32], cap: usize, q: &[f32], n_pad: usize, out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        for (d, &qd) in q.iter().enumerate() {
            let qv = _mm256_set1_ps(qd);
            let row = kt.as_ptr().add(d * cap);
            for j in (0..n_pad).step_by(8) {
                let o = out.as_mut_ptr().add(j);
                let k = _mm256_loadu_ps(row.add(j));
                let acc = if d == 0 {
                    _mm256_mul_ps(k, qv)
                } else {
                    _mm256_fmadd_ps(k, qv, _mm256_loadu_ps(o))
                };
                _mm256_storeu_ps(o, acc);
            }
        }
    }
}

fn weighted_sum(v: &[f32], d: usize, p: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if fma_lanes() {
        return unsafe { weighted_sum_avx2(v, d, p, out) };
    }
    for i in 0..d {
        let mut acc = v[i] * p[0];
        for (j, &pj) in p.iter().enumerate().skip(1) {
            acc = v[j * d + i].mul_add(pj, acc);
        }
        out[i] = acc;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn weighted_sum_avx2(v: &[f32], d: usize, p: &[f32], out: &mut [f32]) {
    unsafe {
        use std::arch::x86_64::*;
        for (j, &pj) in p.iter().enumerate() {
            let pv = _mm256_set1_ps(pj);
            let row = v.as_ptr().add(j * d);
            for i in (0..d).step_by(8) {
                let o = out.as_mut_ptr().add(i);
                let acc = if j == 0 {
                    _mm256_mul_ps(_mm256_loadu_ps(row.add(i)), pv)
                } else {
                    _mm256_fmadd_ps(_mm256_loadu_ps(row.add(i)), pv, _mm256_loadu_ps(o))
                };
                _mm256_storeu_ps(o, acc);
            }
        }
    }
}

pub(crate) fn silu_mul_rows(gate_up: &[f32], inter: usize, act: &mut [f32]) {
    let lane = (1..=inter.min(1024)).rev().find(|l| inter.is_multiple_of(*l)).unwrap_or(1);
    let per_row = inter / lane;
    par_chunks_mut(act, lane, |c, a| {
        let gu = &gate_up[(c / per_row) * 2 * inter + (c % per_row) * lane..];
        for (i, a) in a.iter_mut().enumerate() {
            let g = gu[i];
            *a = g / (1.0 + (-g).exp()) * gu[inter + i];
        }
    });
}

pub(crate) fn softmax_in_place(s: &mut [f32]) {
    let mut max = s[0];
    for &v in &s[1..] {
        max = max.max(v);
    }
    let mut sum = 0f32;
    for v in s.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in s.iter_mut() {
        *v /= sum;
    }
}

pub const PAD_DRAFT_ROWS_DEFAULT_FOR_CPU: usize = 4;

pub const THE_FIRST_DECODE_OF_A_MODEL_KEEPS_THE_CLIP_INDEPENDENT_PREFIX_KV_AND_LATER_CLIPS_PREFILL_FROM_IT: &str =
    "the K/V of the clip_independent_prefix_positions are the same for every clip, so the first \
     decode of a model copies them out of its full prefill and every later clip seeds its cache \
     from the copy and prefills the remaining prefix rows from there, bitwise the full prefill \
     because every row's attention reads the same K/V";

pub const PAD_DRAFT_ROWS_COMMIT_EVERY_LEADING_PAD_OF_AN_M_ROW_STEP_AND_READ_THE_WEIGHTS_ONCE: &str =
    "row 0 carries the last committed token and rows 1..M carry [STREAMING_PAD] drafted at the \
     next positions; cpu_qmatvec reads each weight row once for any M and every loop keeps one \
     lane per row, so the M rows cost one weight read and are bitwise the single-row rows; the \
     rows commit while the previous row's argmax is the pad, the first non-pad token ends the \
     chain and its position is where the next step starts, over the same K/V slots";

pub(crate) struct PrefixSeed {
    positions: usize,
    kt: Vec<f32>,
    v: Vec<f32>,
}

pub(crate) struct CpuStack<'a> {
    pub(crate) layers: &'a [DecoderLayer],
    pub(crate) rope: &'a Rope,
    pub(crate) norm: &'a RmsNorm,
    pub(crate) cfg: &'a VoxtralTextConfig,
    pub(crate) device: &'a Device,
}

pub(crate) struct CpuDecoder<'a> {
    m: CpuStack<'a>,
    head: Option<&'a Linear>,
    cos: F32Host<'a>,
    sin: F32Host<'a>,
    norms: Vec<(F32Host<'a>, F32Host<'a>)>,
    final_norm: F32Host<'a>,
    cap: usize,
    d: usize,
    kt: Vec<f32>,
    v: Vec<f32>,
    scores: Vec<f32>,
}

impl<'a> CpuDecoder<'a> {
    pub(crate) fn new(m: &'a VoxtralRealtime, positions: usize) -> Result<Self> {
        let stack = CpuStack {
            layers: &m.dec_layers,
            rope: &m.dec_rope,
            norm: &m.dec_norm,
            cfg: &m.cfg.text_config,
            device: &m.device,
        };
        let mut dec = Self::over(stack, positions)?;
        dec.head = Some(&m.lm_head);
        Ok(dec)
    }

    pub(crate) fn over(m: CpuStack<'a>, positions: usize) -> Result<Self> {
        let _ = THE_CPU_STEP_IS_PLAIN_F32_LOOPS_AROUND_THE_QUANTIZED_LINEARS;
        let t = m.cfg;
        if t.head_dim % 8 != 0 || t.num_attention_heads % t.num_key_value_heads != 0 {
            bail!("voxtral cpu step: head_dim {} must be a multiple of 8 and heads {} a multiple of kv heads {}", t.head_dim, t.num_attention_heads, t.num_key_value_heads);
        }
        let cap = positions.div_ceil(8) * 8;
        let per_layer = t.num_key_value_heads * t.head_dim * cap;
        let norms = m
            .layers
            .iter()
            .map(|l| Ok((host_f32(l.attn_norm.weight())?, host_f32(l.ffn_norm.weight())?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            cos: host_f32(m.rope.cos())?,
            sin: host_f32(m.rope.sin())?,
            norms,
            final_norm: host_f32(m.norm.weight())?,
            m,
            head: None,
            cap,
            d: t.head_dim,
            kt: vec![0f32; t.num_hidden_layers * per_layer],
            v: vec![0f32; t.num_hidden_layers * per_layer],
            scores: vec![0f32; cap],
        })
    }

    pub(crate) fn prefix_seed(&self, positions: usize) -> PrefixSeed {
        let (cap, d) = (self.cap, self.d);
        PrefixSeed {
            positions,
            kt: self.kt.chunks_exact(cap).flat_map(|row| &row[..positions]).copied().collect(),
            v: self.v.chunks_exact(d * cap).flat_map(|h| &h[..positions * d]).copied().collect(),
        }
    }

    pub(crate) fn seed(&mut self, s: &PrefixSeed) {
        let (cap, d, p) = (self.cap, self.d, s.positions);
        for (row, src) in self.kt.chunks_exact_mut(cap).zip(s.kt.chunks_exact(p)) {
            row[..p].copy_from_slice(src);
        }
        for (h, src) in self.v.chunks_exact_mut(d * cap).zip(s.v.chunks_exact(p * d)) {
            h[..p * d].copy_from_slice(src);
        }
    }

    pub(crate) fn step(&mut self, x: &mut [f32], start: usize) -> Result<Vec<f32>> {
        let hidden = self.m.cfg.hidden_size;
        let mut normed = self.forward(x, start)?;
        normed.drain(..x.len() - hidden);
        Ok(normed)
    }

    pub(crate) fn forward(&mut self, x: &mut [f32], start: usize) -> Result<Vec<f32>> {
        let _ = KEYS_ARE_STORED_TRANSPOSED_SO_THE_SCORE_LOOP_STREAMS_EIGHT_POSITIONS_PER_FMA;
        let m = &self.m;
        let t = m.cfg;
        let (hidden, d, heads, kv_heads, inter) = (
            t.hidden_size,
            t.head_dim,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.intermediate_size,
        );
        let (q_rows, kv_rows, half) = (heads * d, kv_heads * d, d / 2);
        let (cap, group) = (self.cap, heads / kv_heads);
        let rows = x.len() / hidden;
        let eps = t.rms_norm_eps as f32;
        let per_layer = kv_rows * cap;
        let mut h = vec![0f32; rows * hidden];
        let mut attn = vec![0f32; rows * q_rows];
        let mut act = vec![0f32; rows * inter];
        let mut lap = Lap::start(m.device);
        for (li, layer) in m.layers.iter().enumerate() {
            rmsnorm_rows(x, &self.norms[li].0, eps, &mut h);
            lap.lap(1);
            let mut qkv = linear(&layer.qkv_proj, &h, rows)?;
            lap.lap(2);
            let kt = &mut self.kt[li * per_layer..(li + 1) * per_layer];
            let v = &mut self.v[li * per_layer..(li + 1) * per_layer];
            for (r, row) in qkv.chunks_exact_mut(q_rows + 2 * kv_rows).enumerate() {
                let pos = start + r;
                let (cos, sin) = (&self.cos[pos * half..(pos + 1) * half], &self.sin[pos * half..(pos + 1) * half]);
                rope_in_place(&mut row[..q_rows + kv_rows], cos, sin);
                lap.lap(4);
                for g in 0..kv_heads {
                    let k_new = &row[q_rows + g * d..q_rows + (g + 1) * d];
                    for (i, &k) in k_new.iter().enumerate() {
                        kt[g * d * cap + i * cap + pos] = k;
                    }
                    let v_new = &row[q_rows + kv_rows + g * d..q_rows + kv_rows + (g + 1) * d];
                    v[g * d * cap + pos * d..g * d * cap + (pos + 1) * d].copy_from_slice(v_new);
                }
                lap.lap(5);
            }
            let (kt, v) = (&*kt, &*v);
            let attend = |r: usize, hd: usize, s: &mut [f32], o: &mut [f32]| {
                let row = &qkv[r * (q_rows + 2 * kv_rows)..];
                let n = start + r + 1;
                let g = hd / group;
                scores(&kt[g * d * cap..(g + 1) * d * cap], cap, &row[hd * d..(hd + 1) * d], n.div_ceil(8) * 8, s);
                softmax_in_place(&mut s[..n]);
                weighted_sum(&v[g * d * cap..g * d * cap + n * d], d, &s[..n], o);
            };
            if rows == 1 {
                for (hd, o) in attn.chunks_exact_mut(d).enumerate() {
                    attend(0, hd, &mut self.scores, o);
                }
            } else {
                par_chunks_mut(&mut attn, d, |i, o| attend(i / heads, i % heads, &mut vec![0f32; cap], o));
            }
            lap.lap(6);
            let o = linear(&layer.o_proj, &attn, rows)?;
            lap.lap(7);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi = *oi + *xi;
            }
            rmsnorm_rows(x, &self.norms[li].1, eps, &mut h);
            lap.lap(8);
            let gate_up = linear(&layer.gate_up_proj, &h, rows)?;
            lap.lap(9);
            silu_mul_rows(&gate_up, inter, &mut act);
            lap.lap(10);
            let down = linear(&layer.down_proj, &act, rows)?;
            for (xi, di) in x.iter_mut().zip(&down) {
                *xi += *di;
            }
            lap.lap(11);
        }
        rmsnorm_rows(x, &self.final_norm, eps, &mut h);
        lap.lap(12);
        Ok(h)
    }

    pub(crate) fn argmax(&self, normed: &[f32]) -> Result<u32> {
        Ok(self.argmax_rows(normed)?[0])
    }

    pub(crate) fn argmax_rows(&self, normed: &[f32]) -> Result<Vec<u32>> {
        let mut lap = Lap::start(self.m.device);
        let rows = normed.len() / self.m.cfg.hidden_size;
        let head = self.head.ok_or_else(|| anyhow!("voxtral cpu step: this stack has no lm_head"))?;
        let logits = linear(head, normed, rows)?;
        let toks = logits
            .chunks_exact(logits.len() / rows)
            .map(|l| {
                let mut best = (0usize, l[0]);
                for (i, &v) in l.iter().enumerate() {
                    if v > best.1 {
                        best = (i, v);
                    }
                }
                best.0 as u32
            })
            .collect();
        lap.lap(13);
        Ok(toks)
    }
}

fn prefix_seeds() -> &'static Mutex<Option<(TensorId, Arc<PrefixSeed>)>> {
    static SEEDS: Mutex<Option<(TensorId, Arc<PrefixSeed>)>> = Mutex::new(None);
    &SEEDS
}

pub(crate) fn decode_stream(m: &VoxtralRealtime, audio_embeds: &Tensor) -> Result<Vec<u32>> {
    Ok(decode_stream_rows(m, audio_embeds, PAD_DRAFT_ROWS_DEFAULT_FOR_CPU)?.0)
}

pub(crate) fn decode_stream_rows(
    m: &VoxtralRealtime,
    audio_embeds: &Tensor,
    rows: usize,
) -> Result<(Vec<u32>, usize)> {
    let _ = THE_FIRST_DECODE_OF_A_MODEL_KEEPS_THE_CLIP_INDEPENDENT_PREFIX_KV_AND_LATER_CLIPS_PREFILL_FROM_IT;
    let seed = prefix_seeds()
        .lock()
        .unwrap()
        .as_ref()
        .filter(|(id, _)| *id == m.embed.id())
        .map(|(_, s)| s.clone());
    let (out, steps, harvested) = decode_from(m, audio_embeds, rows, seed.as_deref())?;
    if let Some(s) = harvested {
        *prefix_seeds().lock().unwrap() = Some((m.embed.id(), Arc::new(s)));
    }
    Ok((out, steps))
}

fn decode_from(
    m: &VoxtralRealtime,
    audio_embeds: &Tensor,
    rows: usize,
    seed: Option<&PrefixSeed>,
) -> Result<(Vec<u32>, usize, Option<PrefixSeed>)> {
    let _ = PAD_DRAFT_ROWS_COMMIT_EVERY_LEADING_PAD_OF_AN_M_ROW_STEP_AND_READ_THE_WEIGHTS_ONCE;
    let n_audio = audio_embeds.dims()[1];
    m.audio_tokens_in_range(n_audio)?;
    let hidden = m.cfg.text_config.hidden_size;
    let eos = m.cfg.text_config.eos_token_id;
    let prefix = m.prefix_tokens();
    let audio = audio_embeds.contiguous()?;
    let audio = host_f32(&audio)?;
    let embed = host_f32(&m.embed)?;
    let input = |tok: u32, pos: usize, x: &mut [f32]| {
        let e = &embed[tok as usize * hidden..(tok as usize + 1) * hidden];
        let a = &audio[pos * hidden..(pos + 1) * hidden];
        for i in 0..hidden {
            x[i] = e[i] + a[i];
        }
    };
    let mut dec = CpuDecoder::new(m, n_audio + 1)?;
    let start = seed.map_or(0, |s| {
        dec.seed(s);
        s.positions
    });
    let mut x = vec![0f32; (prefix.len() - start) * hidden];
    for (r, &tok) in prefix.iter().enumerate().skip(start) {
        input(tok, r, &mut x[(r - start) * hidden..(r - start + 1) * hidden]);
    }
    let normed = dec.step(&mut x, start)?;
    let mut tok = dec.argmax(&normed)?;
    let harvested = seed
        .is_none()
        .then(|| dec.prefix_seed(clip_independent_prefix_positions(m.vocab().streaming_left_pad_tokens)));
    cpu_profile_report("prefill", 1);
    let mut out = Vec::with_capacity(n_audio - prefix.len());
    let (mut pos, end, mut steps) = (prefix.len(), n_audio - 1, 0);
    if tok != eos {
        out.push(tok);
    }
    'decode: while tok != eos && pos < end {
        let rows = rows.min(end - pos);
        let mut lap = Lap::start(&m.device);
        x.resize(rows * hidden, 0.0);
        for r in 0..rows {
            let t = if r == 0 { tok } else { STREAMING_PAD_TOKEN };
            input(t, pos + r, &mut x[r * hidden..(r + 1) * hidden]);
        }
        lap.lap(0);
        let normed = dec.forward(&mut x, pos)?;
        let toks = dec.argmax_rows(&normed)?;
        steps += 1;
        let mut n = 1;
        while n < rows && toks[n - 1] == STREAMING_PAD_TOKEN {
            n += 1;
        }
        for &t in &toks[..n] {
            if t == eos {
                break 'decode;
            }
            out.push(t);
        }
        tok = toks[n - 1];
        pos += n;
    }
    cpu_profile_report("decode", steps);
    Ok((out, steps, harvested))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> Option<VoxtralRealtime> {
        let dir = std::env::var_os("VOXTRAL_MODEL_DIR").map(std::path::PathBuf::from)?;
        Some(VoxtralRealtime::load_dir(&dir, &Device::Cpu).unwrap())
    }

    fn pcm16_wav(path: &str) -> Vec<f32> {
        let b = std::fs::read(path).unwrap();
        let at = b.windows(4).position(|w| w == b"data").unwrap() + 8;
        b[at..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0).collect()
    }

    fn fixture() -> (std::path::PathBuf, Vec<u32>) {
        let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/data/voxtral_librispeech_8297-275156-0000");
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(base.with_extension("json")).unwrap()).unwrap();
        let tokens = json["tokens"].as_array().unwrap().iter().map(|t| t.as_u64().unwrap() as u32).collect();
        (base.with_extension("wav"), tokens)
    }

    fn corpus_clips() -> Vec<String> {
        let rows = std::env::var("NV_VOXTRAL_CORPUS").unwrap_or_else(|_| "/tmp/voxtral-corpus/rows.jsonl".into());
        let Ok(raw) = std::fs::read_to_string(&rows) else {
            eprintln!("no corpus at {rows}: fixture only");
            return Vec::new();
        };
        raw.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["audio"].as_str().unwrap().to_string())
            .collect()
    }

    fn load() -> String {
        let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        let up = std::fs::read_to_string("/proc/uptime").unwrap_or_default();
        format!(
            "load {} up {:.0} min",
            load.split_whitespace().take(3).collect::<Vec<_>>().join(" "),
            up.split_whitespace().next().and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0) / 60.0
        )
    }

    fn embeds(m: &VoxtralRealtime, clips: &[String]) -> Vec<Tensor> {
        clips.iter().map(|p| m.encode(&m.features(&pcm16_wav(p)).unwrap()).unwrap()).collect()
    }

    #[test]
    #[ignore]
    fn pad_draft_rows_reproduce_the_single_row_decode_on_the_fixture_and_the_corpus() {
        let Some(m) = model() else { return };
        let (wav, reference) = fixture();
        let mut clips = vec![wav.display().to_string()];
        clips.extend(corpus_clips());
        let embeds = embeds(&m, &clips);
        let ladder: Vec<usize> = std::env::var("NV_VOXTRAL_SPEC_ROWS_LADDER")
            .unwrap_or_else(|_| PAD_DRAFT_ROWS_DEFAULT_FOR_CPU.to_string())
            .split(',')
            .map(|r| r.parse().unwrap())
            .collect();
        let run = |rows: usize, seed: Option<&PrefixSeed>| {
            let t0 = std::time::Instant::now();
            let out: Vec<(Vec<u32>, usize, Option<PrefixSeed>)> =
                embeds.iter().map(|e| decode_from(&m, e, rows, seed).unwrap()).collect();
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            let steps: usize = out.iter().map(|o| o.1).sum();
            let tokens: usize = out.iter().map(|o| o.0.len()).sum();
            eprintln!(
                "rows={rows} {}: {} clips, {tokens} tokens, {steps} steps, {:.2} committed/step, {ms:.0} ms total, {:.2} ms/step, {:.2} ms/token, {}",
                if seed.is_some() { "seeded" } else { "unseeded" },
                clips.len(),
                tokens as f64 / steps.max(1) as f64,
                ms / steps.max(1) as f64,
                ms / tokens.max(1) as f64,
                load()
            );
            out.into_iter().map(|o| (o.0, o.2)).unzip::<_, _, Vec<_>, Vec<_>>()
        };
        let differing = |a: &[Vec<u32>], b: &[Vec<u32>]| -> Vec<&String> {
            clips.iter().zip(a).zip(b).filter(|((_, a), b)| a != b).map(|((c, _), _)| c).collect()
        };
        let (unseeded, seeds) = run(1, None);
        assert_eq!(unseeded[0], reference, "the single-row CPU decode diverges from the oracle");
        let seed = seeds[0].as_ref().unwrap();
        let seed_diffs: Vec<&String> = clips
            .iter()
            .zip(&seeds)
            .filter(|(_, s)| {
                let s = s.as_ref().unwrap();
                bits(&s.kt) != bits(&seed.kt) || bits(&s.v) != bits(&seed.v)
            })
            .map(|(c, _)| c)
            .collect();
        eprintln!("{} of {} clips harvest a prefix seed that differs bitwise from the fixture's", seed_diffs.len(), clips.len());
        let (single, _) = run(1, Some(seed));
        let diffs = differing(&unseeded, &single);
        eprintln!("rows=1 seeded: {} of {} clips differ from the unseeded full-prefill decode", diffs.len(), clips.len());
        let mut failures = Vec::new();
        if !diffs.is_empty() {
            failures.push(format!("seeded rows=1: {diffs:?}"));
        }
        for rows in ladder {
            let (spec, _) = run(rows, Some(seed));
            let diffs = differing(&single, &spec);
            eprintln!("rows={rows}: {} of {} clips differ from the single-row decode", diffs.len(), clips.len());
            if !diffs.is_empty() {
                failures.push(format!("rows={rows}: {diffs:?}"));
            }
        }
        assert!(seed_diffs.is_empty(), "the prefix seed depends on the clip: {seed_diffs:?}");
        assert!(failures.is_empty(), "pad-draft changes the token stream: {failures:?}");
    }

    #[test]
    #[ignore]
    fn the_prefix_kv_is_clip_independent_below_the_last_left_pad() {
        let Some(m) = model() else { return };
        let mut clips = vec![fixture().0.display().to_string()];
        clips.extend(corpus_clips().into_iter().take(2));
        let embeds = embeds(&m, &clips);
        let hidden = m.cfg.text_config.hidden_size;
        let prefix = m.prefix_tokens();
        let p = clip_independent_prefix_positions(m.vocab().streaming_left_pad_tokens);
        let seeds: Vec<(PrefixSeed, Vec<f32>, f64, f64)> = embeds
            .iter()
            .map(|e| {
                let e = host_f32(e).unwrap().to_vec();
                let embed = host_f32(&m.embed).unwrap();
                let mut x = vec![0f32; prefix.len() * hidden];
                for (r, &t) in prefix.iter().enumerate() {
                    for i in 0..hidden {
                        x[r * hidden + i] = embed[t as usize * hidden + i] + e[r * hidden + i];
                    }
                }
                let mut dec = CpuDecoder::new(&m, e.len() / hidden + 1).unwrap();
                let mut x_full = x.clone();
                let t0 = std::time::Instant::now();
                let full = dec.step(&mut x_full, 0).unwrap();
                let full_ms = t0.elapsed().as_secs_f64() * 1e3;
                let seed = dec.prefix_seed(p);
                let mut dec2 = CpuDecoder::new(&m, e.len() / hidden + 1).unwrap();
                dec2.seed(&seed);
                let t0 = std::time::Instant::now();
                let seeded = dec2.step(&mut x[p * hidden..], p).unwrap();
                let seeded_ms = t0.elapsed().as_secs_f64() * 1e3;
                assert_eq!(bits(&full), bits(&seeded), "the seeded prefill differs from the full one");
                (seed, e, full_ms, seeded_ms)
            })
            .collect();
        let first_embed_diff = |a: &[f32], b: &[f32]| a.chunks_exact(hidden).zip(b.chunks_exact(hidden)).position(|(x, y)| bits(x) != bits(y));
        for (i, (s, e, full_ms, seeded_ms)) in seeds.iter().enumerate() {
            eprintln!(
                "clip {i}: full {}-row prefill {full_ms:.1} ms, seeded {}-row prefill {seeded_ms:.1} ms, audio embeds first differ from clip 0 at position {:?}, {}",
                prefix.len(),
                prefix.len() - p,
                first_embed_diff(&seeds[0].1, e),
                load()
            );
            assert_eq!(bits(&seeds[0].0.kt), bits(&s.kt), "clip {i}: K differs from clip 0 within the first {p} positions");
            assert_eq!(bits(&seeds[0].0.v), bits(&s.v), "clip {i}: V differs from clip 0 within the first {p} positions");
            assert!(first_embed_diff(&seeds[0].1, e).is_none_or(|d| d >= p), "clip {i}: audio embeds differ within the first {p} positions");
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }
}
