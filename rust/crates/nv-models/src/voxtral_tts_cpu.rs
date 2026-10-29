use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result, anyhow, bail, ensure};
use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::cpu_qmatvec::{Placement, hot_l3_bytes, with_placement};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::{load_linear, load_tensor};
use nv_weights::WeightLoader;

use crate::voxtral::{
    DECODER_ROPE_ROWS, DecoderLayer, DecoderLinearBuilder, Lap, VoxtralQuant, VoxtralTextConfig,
};
use crate::voxtral_cpu::{
    CpuDecoder, CpuStack, host_f32, linear, rmsnorm_rows, silu_mul_rows, softmax_in_place,
};
use crate::voxtral_tekken::TekkenEncoder;
use crate::voxtral_tts::{
    AUDIO_TOKEN, BEGIN_AUDIO_TOKEN, BOS_TOKEN, CodebookLayout, DEFAULT_CFG_ALPHA, DEFAULT_NFE, EMPTY_AUDIO,
    END_AUDIO, FrameTrace, Manifest, N_AUDIO_SPECIAL, NEXT_AUDIO_TEXT_TOKEN, REPEAT_AUDIO_TEXT_TOKEN,
    Synthesis, VOICE_DIR, VoxtralTts, WEIGHTS_FILE, bf16_timesteps, load_stack, read_manifest,
    read_torch_tensor, sinusoidal_time_embedding,
};

pub const VOXTRAL_TTS_QUANT_ENV: &str = "NV_VOXTRAL_TTS_QUANT";

pub const THE_CPU_DEFAULT_IS_Q4_K_BY_THE_73_CLIP_CORPUS_WORD_ERROR_INSTRUMENT: CpuGrade =
    CpuGrade::Q4K;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuGrade {
    F32,
    Bf16,
    Q8_0,
    Q4K,
}

impl CpuGrade {
    pub fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "f32" => Self::F32,
            "bf16" => Self::Bf16,
            "q8_0" => Self::Q8_0,
            "q4_k" => Self::Q4K,
            other => bail!("{VOXTRAL_TTS_QUANT_ENV}={other:?}: expected f32, bf16, q8_0 or q4_k"),
        })
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::Bf16 => "bf16",
            Self::Q8_0 => "q8_0",
            Self::Q4K => "q4_k",
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q8_0 | Self::Q4K)
    }

    fn builder_quant(self) -> VoxtralQuant {
        match self {
            Self::F32 | Self::Bf16 => VoxtralQuant::Bf16,
            Self::Q8_0 => VoxtralQuant::GgmlQ8_0,
            Self::Q4K => VoxtralQuant::GgmlQ4K,
        }
    }

    fn bytes_per_weight(self) -> f64 {
        match self {
            Self::F32 => 4.0,
            Self::Bf16 => 2.0,
            Self::Q8_0 => 34.0 / 32.0,
            Self::Q4K => 144.0 / 256.0,
        }
    }
}

pub const THE_FLOW_HEAD_KEEPS_FOUR_FIFTHS_OF_THE_LARGEST_L3_IN_HOT_ROWS_AND_STREAMS_THE_REST: &str =
    "the seven flow evaluations of a frame read the same three layers, so the leading rows of      every flow linear, sized to four fifths of the largest L3, are hot rows the large-L3 CCD      serves from cache after the first evaluation, while the remaining rows and the whole      backbone stream with non-temporal prefetch so they do not evict them; the share is 0 on      a single-L3 machine and the arithmetic is unchanged by the placement";

const HOT_ROWS_FILL_OF_THE_LARGEST_L3: f64 = 0.8;

const BACKBONE_STREAMS: Placement = Placement { hot_share: 0.0, streaming: true };

struct Grader {
    grade: CpuGrade,
    builder: DecoderLinearBuilder,
}

impl Grader {
    fn build(&self, weight: Tensor) -> Result<Linear> {
        if self.grade == CpuGrade::Bf16 {
            return Linear::new_no_pretranspose(weight.to_dtype(DType::BF16)?, None);
        }
        self.builder.build(weight, &Device::Cpu)
    }
}

pub const THE_CPU_FRAME_IS_THE_STT_CPU_STEP_OVER_THE_BACKBONE_AND_SIX_ROWS_THROUGH_THE_FLOW_LAYERS: &str =
    "the backbone step is CpuDecoder over the 26 TTS layers (the STT decoder's shape, so the \
     same f32 loops and the same M-row cpu_qmatvec linears serve it, the 230-row prompt as one \
     block); each flow evaluation runs its two CFG rows times three tokens as six rows through \
     the three flow layers, so every flow weight is read once per evaluation; the semantic head \
     and llm_projection take the backbone's grade, input_projection, time_projection and the \
     36-row acoustic output stay dense f32; the Euler, CFG and FSQ arithmetic is the host bf16 \
     of VoxtralTts::flow_frame; the f32 grade is candle's gemm over f32 weights, bitwise the \
     eager CPU path, the bf16 grade is the CUDA path's bf16 weights through the lane matvec";

fn norm_weights(layers: &[DecoderLayer]) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
    layers
        .iter()
        .map(|l| {
            Ok((
                host_f32(l.attn_norm.weight())?.to_vec(),
                host_f32(l.ffn_norm.weight())?.to_vec(),
            ))
        })
        .collect()
}

pub fn cpu_grade_from_env() -> Result<CpuGrade> {
    match std::env::var(VOXTRAL_TTS_QUANT_ENV).ok().as_deref() {
        None | Some("") => Ok(THE_CPU_DEFAULT_IS_Q4_K_BY_THE_73_CLIP_CORPUS_WORD_ERROR_INSTRUMENT),
        Some(s) => CpuGrade::parse(s),
    }
}

pub fn semantic_top2(rows: usize, logits: &[f32]) -> [(u32, f32); 2] {
    let mut top = [
        (END_AUDIO, f32::NEG_INFINITY),
        (END_AUDIO, f32::NEG_INFINITY),
    ];
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

struct FlowHead {
    input_proj: Linear,
    llm_proj: Linear,
    layers: Vec<DecoderLayer>,
    norms: Vec<(Vec<f32>, Vec<f32>)>,
    norm: Vec<f32>,
    out: Linear,
    nfe: usize,
    cfg_alpha: bf16,
    time_tokens: Vec<f32>,
    dts: Vec<bf16>,
    placement: Placement,
}

pub struct VoxtralTtsCpu {
    backbone_cfg: VoxtralTextConfig,
    flow_cfg: VoxtralTextConfig,
    codebooks: CodebookLayout,
    grade: CpuGrade,
    encoder: TekkenEncoder,
    voices: BTreeMap<String, usize>,
    dir: PathBuf,
    embed: Tensor,
    codebook_embed: Vec<f32>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rope: Rope,
    semantic_head: Linear,
    flow: FlowHead,
    voice_cache: Mutex<BTreeMap<String, Vec<f32>>>,
}

impl VoxtralTtsCpu {
    pub fn load_dir(dir: &Path) -> Result<Self> {
        Self::load_dir_with(dir, cpu_grade_from_env()?, DEFAULT_NFE, DEFAULT_CFG_ALPHA)
    }

    pub fn load_dir_with(dir: &Path, grade: CpuGrade, nfe: usize, cfg_alpha: f32) -> Result<Self> {
        let _ = THE_CPU_FRAME_IS_THE_STT_CPU_STEP_OVER_THE_BACKBONE_AND_SIX_ROWS_THROUGH_THE_FLOW_LAYERS;
        let dev = Device::Cpu;
        let Manifest { backbone_cfg, flow_cfg, codebooks, nfe, encoder, voices } =
            read_manifest(dir, nfe)?;
        let builder = Grader {
            grade,
            builder: DecoderLinearBuilder::new(grade.builder_quant(), &dev)?,
        };
        let w = WeightLoader::open_file(dir.join(WEIGHTS_FILE), &dev)
            .with_context(|| format!("open {}", dir.join(WEIGHTS_FILE).display()))?;
        let d = backbone_cfg.hidden_size;
        let embed = load_tensor(
            &w,
            "mm_audio_embeddings.tok_embeddings.weight",
            &[backbone_cfg.vocab_size, d],
            DType::BF16,
        )?;
        let codebook_embed = w.get(
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
            DType::F32,
        )?;
        ensure!(
            codebook_embed.dims().len() == 2
                && codebook_embed.dims()[1] == d
                && codebook_embed.dims()[0] >= codebooks.rows_used(),
            "audio_codebook_embeddings {:?} holds fewer than the {} rows the 37 codebooks address",
            codebook_embed.dims(),
            codebooks.rows_used()
        );
        let codebook_embed = host_f32(&codebook_embed)?.to_vec();
        let build = |t: Tensor| builder.build(t);
        let (layers, norm) = load_stack(&w, &w, "", &backbone_cfg, &build, &dev, DType::F32, true)?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: backbone_cfg.head_dim,
                max_seq_len: DECODER_ROPE_ROWS,
                base: backbone_cfg.rope_parameters.rope_theta,
                kind: RopeKind::Standard,
            },
            &dev,
        )?;
        let fp = "acoustic_transformer.";
        let semantic_head = builder.build(load_tensor(
            &w,
            &format!("{fp}semantic_codebook_output.weight"),
            &[codebooks.semantic_rows().div_ceil(128) * 128, d],
            DType::F32,
        )?)?;
        let (flow_layers, flow_norm) = load_stack(&w, &w, fp, &flow_cfg, &build, &dev, DType::F32, false)?;
        let _ = THE_FLOW_HEAD_KEEPS_FOUR_FIFTHS_OF_THE_LARGEST_L3_IN_HOT_ROWS_AND_STREAMS_THE_REST;
        let (fd, fh, fkv, fi) = (
            flow_cfg.hidden_size as f64,
            (flow_cfg.num_attention_heads * flow_cfg.head_dim) as f64,
            (flow_cfg.num_key_value_heads * flow_cfg.head_dim) as f64,
            flow_cfg.intermediate_size as f64,
        );
        let flow_bytes = flow_cfg.num_hidden_layers as f64
            * grade.bytes_per_weight()
            * (fd * (fh + 2.0 * fkv) + fh * fd + 3.0 * fi * fd);
        let placement = Placement {
            hot_share: (HOT_ROWS_FILL_OF_THE_LARGEST_L3 * hot_l3_bytes() as f64 / flow_bytes).min(1.0) as f32,
            streaming: true,
        };
        let time_proj = load_linear(&w, &format!("{fp}time_projection.weight"), d, d, DType::F32)?;
        let timesteps = bf16_timesteps(nfe);
        let time_rows: Vec<f32> = timesteps[..nfe]
            .iter()
            .flat_map(|t| sinusoidal_time_embedding(t.to_f32(), d))
            .collect();
        let flow = FlowHead {
            input_proj: load_linear(
                &w,
                &format!("{fp}input_projection.weight"),
                d,
                codebooks.n_acoustic,
                DType::F32,
            )?,
            llm_proj: builder.build(load_tensor(
                &w,
                &format!("{fp}llm_projection.weight"),
                &[d, d],
                DType::F32,
            )?)?,
            norms: norm_weights(&flow_layers)?,
            layers: flow_layers,
            norm: host_f32(flow_norm.weight())?.to_vec(),
            out: load_linear(
                &w,
                &format!("{fp}acoustic_codebook_output.weight"),
                codebooks.n_acoustic,
                d,
                DType::F32,
            )?,
            nfe,
            cfg_alpha: bf16::from_f32(cfg_alpha),
            time_tokens: linear(&time_proj, &time_rows, nfe)?,
            dts: timesteps.windows(2).map(|p| p[1] - p[0]).collect(),
            placement,
        };
        Ok(Self {
            backbone_cfg,
            flow_cfg,
            codebooks,
            grade,
            encoder,
            voices,
            dir: dir.to_path_buf(),
            embed,
            codebook_embed,
            layers,
            norm,
            rope,
            semantic_head,
            flow,
            voice_cache: Mutex::new(BTreeMap::new()),
        })
    }

    pub fn grade(&self) -> CpuGrade {
        self.grade
    }

    pub fn nfe(&self) -> usize {
        self.flow.nfe
    }

    pub fn timesteps(&self) -> Vec<f32> {
        bf16_timesteps(self.flow.nfe)
            .iter()
            .map(|t| t.to_f32())
            .collect()
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

    pub fn voice_rows(&self, voice: &str) -> Result<usize> {
        self.voices.get(voice).copied().ok_or_else(|| {
            anyhow!(
                "unknown voice {voice:?}; presets: {}",
                self.voices.keys().cloned().collect::<Vec<_>>().join(", ")
            )
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

    fn voice_embedding(&self, voice: &str) -> Result<Vec<f32>> {
        let n_voice = self.voice_rows(voice)?;
        let mut cache = self
            .voice_cache
            .lock()
            .map_err(|_| anyhow!("voice cache poisoned"))?;
        if let Some(t) = cache.get(voice) {
            return Ok(t.clone());
        }
        let t = read_torch_tensor(&self.dir.join(VOICE_DIR).join(format!("{voice}.pt")))?
            .to_dtype(DType::F32)?;
        ensure!(
            t.dims() == [n_voice, self.backbone_cfg.hidden_size],
            "voice {voice}: tensor {:?} does not match tekken's {n_voice} audio tokens",
            t.dims()
        );
        let v = host_f32(&t)?.to_vec();
        cache.insert(voice.to_string(), v.clone());
        Ok(v)
    }

    fn prompt_embeds(&self, ids: &[u32], voice: &str) -> Result<Vec<f32>> {
        let d = self.backbone_cfg.hidden_size;
        let voice_rows = self.voice_embedding(voice)?;
        let n_voice = voice_rows.len() / d;
        let e = self
            .embed
            .index_select(&Tensor::new(ids, &Device::Cpu)?, 0)?
            .to_dtype(DType::F32)?;
        let e = host_f32(&e)?;
        let first_audio = ids
            .iter()
            .position(|&t| t == AUDIO_TOKEN)
            .ok_or_else(|| anyhow!("prompt has no [AUDIO] run"))?;
        ensure!(
            ids[first_audio..first_audio + n_voice]
                .iter()
                .all(|&t| t == AUDIO_TOKEN),
            "the [AUDIO] run is shorter than the voice tensor"
        );
        let mut x = Vec::with_capacity(ids.len() * d);
        x.extend_from_slice(&e[..first_audio * d]);
        x.extend_from_slice(&voice_rows);
        x.extend_from_slice(&e[(first_audio + n_voice) * d..]);
        Ok(x)
    }

    pub fn semantic_argmax(&self, logits: &[f32]) -> u32 {
        self.semantic_top2(logits)[0].0
    }

    pub fn semantic_top2(&self, logits: &[f32]) -> [(u32, f32); 2] {
        semantic_top2(self.codebooks.semantic_rows(), logits)
    }

    fn flow_eval(&self, x: &mut [f32]) -> Result<Vec<f32>> {
        with_placement(self.flow.placement, || self.flow_eval_rows(x))
    }

    fn flow_eval_rows(&self, x: &mut [f32]) -> Result<Vec<f32>> {
        let t = &self.flow_cfg;
        let (dim, d, heads, kv_heads, inter) = (
            t.hidden_size,
            t.head_dim,
            t.num_attention_heads,
            t.num_key_value_heads,
            t.intermediate_size,
        );
        let (q_rows, kv_rows, group, seq) = (heads * d, kv_heads * d, heads / kv_heads, 3);
        let stride = q_rows + 2 * kv_rows;
        let rows = x.len() / dim;
        let eps = t.rms_norm_eps as f32;
        let scale = 1.0 / (d as f32).sqrt();
        let mut h = vec![0f32; rows * dim];
        let mut attn = vec![0f32; rows * q_rows];
        let mut act = vec![0f32; rows * inter];
        let mut scores = vec![0f32; seq];
        let mut lap = Lap::start(&Device::Cpu);
        for (li, layer) in self.flow.layers.iter().enumerate() {
            rmsnorm_rows(x, &self.flow.norms[li].0, eps, &mut h);
            lap.lap(1);
            let qkv = linear(&layer.qkv_proj, &h, rows)?;
            lap.lap(2);
            for r in 0..rows {
                let b = r / seq;
                for hd in 0..heads {
                    let g = hd / group;
                    let q = &qkv[r * stride + hd * d..][..d];
                    for j in 0..seq {
                        let k = &qkv[(b * seq + j) * stride + q_rows + g * d..][..d];
                        scores[j] = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    softmax_in_place(&mut scores);
                    let o = &mut attn[r * q_rows + hd * d..][..d];
                    o.fill(0.0);
                    for (j, &p) in scores.iter().enumerate() {
                        let v = &qkv[(b * seq + j) * stride + q_rows + kv_rows + g * d..][..d];
                        for (o, &v) in o.iter_mut().zip(v) {
                            *o += p * v;
                        }
                    }
                }
            }
            lap.lap(6);
            let o = linear(&layer.o_proj, &attn, rows)?;
            lap.lap(7);
            for (xi, oi) in x.iter_mut().zip(&o) {
                *xi += *oi;
            }
            rmsnorm_rows(x, &self.flow.norms[li].1, eps, &mut h);
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
        let first: Vec<f32> = x
            .chunks_exact(seq * dim)
            .flat_map(|b| &b[..dim])
            .copied()
            .collect();
        let mut normed = vec![0f32; first.len()];
        rmsnorm_rows(&first, &self.flow.norm, eps, &mut normed);
        lap.lap(12);
        let v = linear(&self.flow.out, &normed, rows / seq);
        lap.lap(13);
        v
    }

    pub fn flow_frame(
        &self,
        h: &[f32],
        x0: &[f32],
        trace: Option<&mut FrameTrace>,
    ) -> Result<Vec<u32>> {
        let f = &self.flow;
        let n = self.codebooks.n_acoustic;
        ensure!(
            x0.len() == n,
            "noise row has {} values, the flow head needs {n}",
            x0.len()
        );
        let d = self.flow_cfg.hidden_size;
        let cond = with_placement(BACKBONE_STREAMS, || linear(&f.llm_proj, h, 1))?;
        let mut x: Vec<bf16> = x0.iter().map(|v| bf16::from_f32(*v)).collect();
        let mut trace = trace;
        if let Some(tr) = trace.as_deref_mut() {
            tr.x.push(x.iter().map(|v| v.to_f32()).collect());
        }
        let one_minus_alpha = bf16::ONE - f.cfg_alpha;
        let mut tokens = vec![0f32; 6 * d];
        for i in 0..f.nfe {
            let xf: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
            let xin = linear(&f.input_proj, &xf, 1)?;
            let tp = &f.time_tokens[i * d..(i + 1) * d];
            for b in 0..2 {
                tokens[b * 3 * d..][..d].copy_from_slice(&xin);
                tokens[b * 3 * d + d..][..d].copy_from_slice(tp);
            }
            tokens[2 * d..3 * d].copy_from_slice(&cond);
            tokens[5 * d..6 * d].fill(0.0);
            let v = self.flow_eval(&mut tokens)?;
            let (vc, vu) = v.split_at(n);
            for j in 0..n {
                let vt =
                    f.cfg_alpha * bf16::from_f32(vc[j]) + one_minus_alpha * bf16::from_f32(vu[j]);
                x[j] = x[j] + vt * f.dts[i];
            }
            if let Some(tr) = trace.as_deref_mut() {
                tr.v_cond
                    .push(vc.iter().map(|v| bf16::from_f32(*v).to_f32()).collect());
                tr.v_uncond
                    .push(vu.iter().map(|v| bf16::from_f32(*v).to_f32()).collect());
                tr.x.push(x.iter().map(|v| v.to_f32()).collect());
            }
        }
        let levels = bf16::from_f32((self.codebooks.acoustic_levels - 1) as f32);
        Ok(x.iter()
            .map(|&v| {
                let c = v.clamp(bf16::NEG_ONE, bf16::ONE);
                let y = (c + bf16::ONE) * bf16::from_f32(0.5) * levels;
                y.to_f32()
                    .round_ties_even()
                    .clamp(0.0, (self.codebooks.acoustic_levels - 1) as f32) as u32
            })
            .collect())
    }

    pub fn frame_embedding(&self, frame: &[u32]) -> Result<Vec<f32>> {
        let cb = self.codebooks;
        let d = self.backbone_cfg.hidden_size;
        ensure!(
            frame.len() == cb.n_codebooks(),
            "frame has {} tokens, not {}",
            frame.len(),
            cb.n_codebooks()
        );
        let mut e = vec![0f32; d];
        for (k, &c) in frame.iter().enumerate() {
            let row = &self.codebook_embed[(cb.offset(k) + c as usize) * d..][..d];
            for (e, r) in e.iter_mut().zip(row) {
                *e += r;
            }
        }
        Ok(e)
    }

    pub fn synthesize(
        &self,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        trace: bool,
        teacher: Option<&[Vec<u32>]>,
    ) -> Result<Synthesis> {
        let ids = self.prompt_ids(text, voice)?;
        let max_frames = teacher.map_or(max_frames, <[Vec<u32>]>::len);
        let mut dec = CpuDecoder::over(
            CpuStack {
                layers: &self.layers,
                rope: &self.rope,
                norm: &self.norm,
                cfg: &self.backbone_cfg,
                device: &Device::Cpu,
            },
            ids.len() + max_frames + 1,
        )?;
        let mut out = Synthesis {
            prompt_ids: ids.clone(),
            ..Default::default()
        };
        let mut t0 = std::time::Instant::now();
        let mut lap = |slot: &mut f64| {
            let now = std::time::Instant::now();
            *slot += (now - t0).as_secs_f64() * 1000.0;
            t0 = now;
        };
        let mut x = self.prompt_embeds(&ids, voice)?;
        let mut h = with_placement(BACKBONE_STREAMS, || dec.step(&mut x, 0))?;
        lap(&mut out.stage_ms.prefill);
        crate::voxtral::cpu_profile_report("prefill", 1);
        let cb = self.codebooks;
        let mut pos = ids.len();
        for frame_idx in 0..max_frames {
            let logits = with_placement(BACKBONE_STREAMS, || linear(&self.semantic_head, &h, 1))?;
            let top2 = self.semantic_top2(&logits);
            let semantic = top2[0].0;
            if trace {
                out.semantic_top2.push(top2);
            }
            lap(&mut out.stage_ms.semantic);
            let mut frame_trace = (trace && frame_idx == 0).then(FrameTrace::default);
            if let Some(tr) = frame_trace.as_mut() {
                tr.semantic_logits = logits;
            }
            let codes = self.flow_frame(&h, &noise(frame_idx), frame_trace.as_mut())?;
            lap(&mut out.stage_ms.flow);
            let mut frame = Vec::with_capacity(cb.n_codebooks());
            frame.push(semantic);
            frame.extend(codes.iter().map(|&c| if semantic == END_AUDIO { EMPTY_AUDIO } else { c } + N_AUDIO_SPECIAL));
            let fed = teacher.map_or_else(|| frame.clone(), |f| f[frame_idx].clone());
            let ends = teacher.map_or(semantic == END_AUDIO, |f| frame_idx + 1 == f.len());
            if trace && (frame_idx < 3 || ends) {
                out.hidden.insert(frame_idx, h.clone());
            }
            out.frames.push(frame);
            if ends {
                out.ended = semantic == END_AUDIO;
                break;
            }
            x = self.frame_embedding(&fed)?;
            if let Some(tr) = frame_trace.as_mut() {
                tr.frame_embed = x.clone();
            }
            out.frame0 = out.frame0.take().or(frame_trace);
            lap(&mut out.stage_ms.embed);
            h = with_placement(BACKBONE_STREAMS, || dec.step(&mut x, pos))?;
            pos += 1;
            lap(&mut out.stage_ms.backbone);
            out.stage_ms.steps += 1;
        }
        Ok(out)
    }
}

pub enum VoxtralTtsModel {
    Eager(VoxtralTts),
    Cpu(VoxtralTtsCpu),
}

impl VoxtralTtsModel {
    pub fn load_dir(dir: &Path, device: &Device) -> Result<Self> {
        Ok(if device.is_cpu() {
            Self::Cpu(VoxtralTtsCpu::load_dir(dir)?)
        } else {
            Self::Eager(VoxtralTts::load_dir(dir, device)?)
        })
    }

    pub fn nfe(&self) -> usize {
        match self {
            Self::Eager(m) => m.nfe(),
            Self::Cpu(m) => m.nfe(),
        }
    }

    pub fn timesteps(&self) -> Vec<f32> {
        match self {
            Self::Eager(m) => m.timesteps(),
            Self::Cpu(m) => m.timesteps(),
        }
    }

    pub fn prompt_ids(&self, text: &str, voice: &str) -> Result<Vec<u32>> {
        match self {
            Self::Eager(m) => m.prompt_ids(text, voice),
            Self::Cpu(m) => m.prompt_ids(text, voice),
        }
    }

    pub fn semantic_argmax(&self, logits: &[f32]) -> u32 {
        match self {
            Self::Eager(m) => m.semantic_argmax(logits),
            Self::Cpu(m) => m.semantic_argmax(logits),
        }
    }

    pub fn semantic_top2(&self, logits: &[f32]) -> [(u32, f32); 2] {
        match self {
            Self::Eager(m) => m.semantic_top2(logits),
            Self::Cpu(m) => m.semantic_top2(logits),
        }
    }

    pub fn synthesize(
        &self,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        trace: bool,
        teacher: Option<&[Vec<u32>]>,
    ) -> Result<Synthesis> {
        match self {
            Self::Eager(m) => m.synthesize(text, voice, noise, max_frames, trace, teacher),
            Self::Cpu(m) => m.synthesize(text, voice, noise, max_frames, trace, teacher),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cuda")]
    use crate::voxtral_tts::SeededNoise;

    #[cfg(feature = "cuda")]
    const CORPUS_MAX_FRAMES: usize = 512;

    fn opted_in() -> Option<PathBuf> {
        if std::env::var("NV_VOXTRAL_TTS_TEST").ok().as_deref() != Some("1") {
            eprintln!(
                "skip: NV_VOXTRAL_TTS_TEST=1 opts in (needs the Voxtral-4B-TTS-2603 checkpoint)"
            );
            return None;
        }
        Some(VoxtralTts::default_dir().unwrap())
    }

    fn load() -> String {
        let load = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        let up = std::fs::read_to_string("/proc/uptime").unwrap_or_default();
        format!(
            "load {} up {:.0} min",
            load.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" "),
            up.split_whitespace()
                .next()
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(0.0)
                / 60.0
        )
    }

    fn fixture(case: &str) -> (String, String, Vec<Vec<f32>>, Vec<Vec<u32>>) {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../tests/data/voxtral_tts_{case}"));
        let json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(base.with_extension("json")).unwrap()).unwrap();
        let st = candle_core::safetensors::load(base.with_extension("safetensors"), &Device::Cpu)
            .unwrap();
        let noise = st["noise"]
            .to_dtype(DType::F32)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap();
        let frames = json["frames"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| {
                f.as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_u64().unwrap() as u32)
                    .collect()
            })
            .collect();
        (
            json["text"].as_str().unwrap().into(),
            json["voice"].as_str().unwrap().into(),
            noise,
            frames,
        )
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    fn stage_report(s: &crate::voxtral_tts::StageMs, codec_ms: f64) -> String {
        let n = s.steps.max(1) as f64;
        format!(
            "prefill {:.0} ms; per frame: backbone {:.1} ms, semantic {:.2} ms, flow {:.1} ms, embed {:.2} ms, codec {:.1} ms; {}",
            s.prefill,
            s.backbone / n,
            s.semantic / n,
            s.flow / n,
            s.embed / n,
            codec_ms,
            load()
        )
    }

    #[test]
    #[ignore]
    fn the_f32_grade_reproduces_the_eager_cpu_path_on_the_fixture() {
        let Some(dir) = opted_in() else { return };
        let (text, voice, noise, frames) = fixture("en");
        let eager = {
            let m = VoxtralTts::load_dir(&dir, &Device::Cpu).unwrap();
            m.synthesize(
                &text,
                &voice,
                &mut |i| noise[i].clone(),
                0,
                true,
                Some(&frames),
            )
            .unwrap()
        };
        eprintln!("eager f32: {}", stage_report(&eager.stage_ms, 0.0));
        let m = VoxtralTtsCpu::load_dir_with(&dir, CpuGrade::F32, DEFAULT_NFE, DEFAULT_CFG_ALPHA)
            .unwrap();
        let ours = m
            .synthesize(
                &text,
                &voice,
                &mut |i| noise[i].clone(),
                0,
                true,
                Some(&frames),
            )
            .unwrap();
        eprintln!("cpu step f32: {}", stage_report(&ours.stage_ms, 0.0));
        let (a, b) = (
            eager.frame0.as_ref().unwrap(),
            ours.frame0.as_ref().unwrap(),
        );
        let max_abs = |x: &[f32], y: &[f32]| {
            x.iter()
                .zip(y)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max)
        };
        for (k, h) in &eager.hidden {
            eprintln!(
                "hidden frame {k}: max abs {:.3e}, bitwise {}",
                max_abs(h, &ours.hidden[k]),
                bits(h) == bits(&ours.hidden[k])
            );
        }
        eprintln!(
            "semantic logits frame 0: max abs {:.3e}, bitwise {}",
            max_abs(&a.semantic_logits, &b.semantic_logits),
            bits(&a.semantic_logits) == bits(&b.semantic_logits)
        );
        for i in 0..a.v_cond.len() {
            eprintln!(
                "flow step {i}: v_cond max abs {:.3e} v_uncond {:.3e} x {:.3e}",
                max_abs(&a.v_cond[i], &b.v_cond[i]),
                max_abs(&a.v_uncond[i], &b.v_uncond[i]),
                max_abs(&a.x[i + 1], &b.x[i + 1])
            );
        }
        eprintln!(
            "frame_embed_0 bitwise {}",
            bits(&a.frame_embed) == bits(&b.frame_embed)
        );
        let frame_diffs = eager
            .frames
            .iter()
            .zip(&ours.frames)
            .filter(|(a, b)| a != b)
            .count();
        eprintln!("frames differing: {frame_diffs} of {}", eager.frames.len());
        for (k, h) in &eager.hidden {
            assert_eq!(
                bits(h),
                bits(&ours.hidden[k]),
                "hidden state of frame {k} differs from the eager CPU path"
            );
        }
        assert_eq!(
            bits(&a.semantic_logits),
            bits(&b.semantic_logits),
            "semantic logits differ from the eager CPU path"
        );
        assert_eq!(
            frame_diffs, 0,
            "teacher-forced frames differ from the eager CPU path"
        );
    }

    #[cfg(feature = "cuda")]
    fn resample_24k_to_16k(x: &[f32]) -> Vec<f32> {
        const HALF: isize = 24;
        (0..x.len() * 2 / 3)
            .map(|i| {
                let c = i as f64 * 1.5;
                let k0 = c.floor() as isize;
                let mut acc = 0f64;
                for k in (k0 - HALF).max(0)..=(k0 + HALF).min(x.len() as isize - 1) {
                    let t = k as f64 - c;
                    let w = 0.5 + 0.5 * (std::f64::consts::PI * t / (HALF as f64 + 1.0)).cos();
                    let a = std::f64::consts::PI * 2.0 * t / 3.0;
                    let s = if t == 0.0 { 1.0 } else { a.sin() / a };
                    acc += x[k as usize] as f64 * (2.0 / 3.0) * s * w;
                }
                acc as f32
            })
            .collect()
    }

    #[cfg(feature = "cuda")]
    fn stt_dir() -> PathBuf {
        if let Some(d) = std::env::var_os("VOXTRAL_MODEL_DIR") {
            return PathBuf::from(d);
        }
        let snaps = PathBuf::from(std::env::var("HOME").unwrap()).join(
            ".cache/huggingface/hub/models--mistralai--Voxtral-Mini-4B-Realtime-2602/snapshots",
        );
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .expect("stt snapshots")
            .flatten()
            .map(|e| e.path())
            .collect();
        entries.sort();
        entries
            .into_iter()
            .find(|p| p.join("model.safetensors").is_file())
            .expect("a Voxtral realtime snapshot")
    }

    #[test]
    #[ignore]
    fn fixture_free_run_stage_ms_per_grade() {
        use crate::voxtral_tts_codec::{
            CODEC_FRAME_TOKENS, CodecState, VoxtralTtsCodec, frames_before_end_of_audio,
        };
        let Some(dir) = opted_in() else { return };
        let case = std::env::var("NV_VOXTRAL_TTS_CASE").unwrap_or_else(|_| "en".into());
        let (text, voice, noise, _) = fixture(&case);
        let codec = VoxtralTtsCodec::load(&dir, &Device::Cpu).unwrap();
        for grade in std::env::var("NV_VOXTRAL_TTS_GRADES")
            .unwrap_or_else(|_| "q4_k,q8_0,bf16,f32".into())
            .split(',')
        {
            let grade = CpuGrade::parse(grade).unwrap();
            let t0 = std::time::Instant::now();
            let m =
                VoxtralTtsCpu::load_dir_with(&dir, grade, DEFAULT_NFE, DEFAULT_CFG_ALPHA).unwrap();
            let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let run = || {
                m.synthesize(
                    &text,
                    &voice,
                    &mut |i| noise[i.min(noise.len() - 1)].clone(),
                    4 * noise.len(),
                    true,
                    None,
                )
                .unwrap()
            };
            let _warm = run();
            crate::voxtral::cpu_profile_report("warm-up run", 1);
            let s = run();
            crate::voxtral::cpu_profile_report("prefill + frames", s.stage_ms.steps);
            let (h0, x0) = (&s.hidden[&0], &noise[0]);
            let flow_reps = 10;
            let t0 = std::time::Instant::now();
            for _ in 0..flow_reps {
                m.flow_frame(h0, x0, None).unwrap();
            }
            let flow_only_ms = t0.elapsed().as_secs_f64() * 1000.0 / flow_reps as f64;
            crate::voxtral::cpu_profile_report("flow evaluation", flow_reps * m.nfe());
            let codes: Vec<i32> = s.frames.iter().flatten().map(|&c| c as i32).collect();
            let codes = frames_before_end_of_audio(&codes);
            let mut state = CodecState::default();
            let _warm = codec.decode(&codes[..CODEC_FRAME_TOKENS]).unwrap();
            let t0 = std::time::Instant::now();
            for frame in codes.chunks_exact(CODEC_FRAME_TOKENS) {
                codec.decode_frames(&mut state, frame).unwrap();
            }
            let codec_ms = t0.elapsed().as_secs_f64() * 1000.0
                / (codes.len() / CODEC_FRAME_TOKENS).max(1) as f64;
            let n = s.stage_ms.steps.max(1) as f64;
            eprintln!(
                "{case} {}: loaded in {load_ms:.0} ms; {} frames; {}; flow alone {flow_only_ms:.1} ms (hot share {:.3}); frame total {:.1} ms",
                grade.label(),
                s.frames.len(),
                stage_report(&s.stage_ms, codec_ms),
                m.flow.placement.hot_share,
                (s.stage_ms.backbone + s.stage_ms.semantic + s.stage_ms.flow + s.stage_ms.embed)
                    / n
                    + codec_ms
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn fixture_free_run_frames_close_the_loop_through_the_cpu_codec_and_the_stt_port() {
        use crate::voxtral::VoxtralRealtime;
        use crate::voxtral_tts_codec::VoxtralTtsCodec;
        let Some(dir) = opted_in() else { return };
        let case = std::env::var("NV_VOXTRAL_TTS_CASE").unwrap_or_else(|_| "es".into());
        let grade = cpu_grade_from_env().unwrap();
        let (text, voice, noise, oracle) = fixture(&case);
        let path = format!("/tmp/voxtral-tts-free-{case}-{}.json", grade.label());
        let frames: Vec<Vec<u32>> = match std::fs::read(&path) {
            Ok(b) => serde_json::from_slice(&b).unwrap(),
            Err(_) => {
                let m = VoxtralTtsCpu::load_dir_with(&dir, grade, DEFAULT_NFE, DEFAULT_CFG_ALPHA)
                    .unwrap();
                let s = m
                    .synthesize(
                        &text,
                        &voice,
                        &mut |i| noise[i.min(noise.len() - 1)].clone(),
                        4 * noise.len(),
                        true,
                        None,
                    )
                    .unwrap();
                eprintln!(
                    "{case} {} free run: {} frames, ended {}; {}",
                    grade.label(),
                    s.frames.len(),
                    s.ended,
                    stage_report(&s.stage_ms, 0.0)
                );
                std::fs::write(&path, serde_json::to_vec(&s.frames).unwrap()).unwrap();
                s.frames
            }
        };
        let codec = VoxtralTtsCodec::load(&dir, &Device::Cpu).unwrap();
        let stt = VoxtralRealtime::load_dir(&stt_dir(), &Device::new_cuda_with_stream(0).unwrap())
            .unwrap();
        for (label, fr) in [("oracle frames", &oracle), ("free run", &frames)] {
            let codes: Vec<i32> = fr.iter().flatten().map(|&c| c as i32).collect();
            let wav = codec.decode(&codes).unwrap();
            let (hyp, _) = stt.transcribe(&resample_24k_to_16k(&wav)).unwrap();
            eprintln!(
                "{case} {}: {label} -> {hyp:?} (text {text:?}, {})",
                grade.label(),
                if hyp.trim() == text.trim() {
                    "verbatim"
                } else {
                    "DIFFERS"
                }
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn corpus_word_errors_per_grade_through_the_cpu_codec_and_the_stt_port() {
        use crate::voxtral::VoxtralRealtime;
        use crate::voxtral_train::word_error_rate;
        use crate::voxtral_tts_codec::VoxtralTtsCodec;
        let Some(dir) = opted_in() else { return };
        let rows_path = std::env::var("NV_VOXTRAL_CORPUS")
            .unwrap_or_else(|_| "/tmp/voxtral-corpus/rows.jsonl".into());
        let texts: Vec<String> = std::fs::read_to_string(&rows_path)
            .expect("corpus rows")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l).unwrap()["text"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        let grades: Vec<CpuGrade> = std::env::var("NV_VOXTRAL_TTS_GRADES")
            .unwrap_or_else(|_| "f32,q8_0,q4_k".into())
            .split(',')
            .map(|g| CpuGrade::parse(g).unwrap())
            .collect();
        let voice =
            std::env::var("NV_VOXTRAL_TTS_VOICE").unwrap_or_else(|_| "neutral_female".into());
        let codec = VoxtralTtsCodec::load(&dir, &Device::Cpu).unwrap();
        let stt = VoxtralRealtime::load_dir(&stt_dir(), &Device::new_cuda_with_stream(0).unwrap())
            .unwrap();
        for grade in grades {
            let m =
                VoxtralTtsCpu::load_dir_with(&dir, grade, DEFAULT_NFE, DEFAULT_CFG_ALPHA).unwrap();
            let mut stage = crate::voxtral_tts::StageMs::default();
            let (mut errors, mut words, mut bad_rows, mut unfinished, mut frames, mut codec_ms) =
                (0, 0, 0, 0, 0, 0f64);
            let mut log = String::new();
            for (i, text) in texts.iter().enumerate() {
                let mut noise = SeededNoise::new(1 + i as u64);
                let s = m
                    .synthesize(
                        text,
                        &voice,
                        &mut |_| noise.frame(36),
                        CORPUS_MAX_FRAMES,
                        true,
                        None,
                    )
                    .unwrap();
                let codes: Vec<i32> = s.frames.iter().flatten().map(|&c| c as i32).collect();
                let t0 = std::time::Instant::now();
                let wav = codec.decode(&codes).unwrap();
                codec_ms += t0.elapsed().as_secs_f64() * 1000.0;
                let (hyp, _) = stt.transcribe(&resample_24k_to_16k(&wav)).unwrap();
                let (e, w) = word_error_rate(&hyp, text);
                errors += e;
                words += w;
                bad_rows += (e > 0) as usize;
                unfinished += (!s.ended) as usize;
                frames += s.frames.len();
                stage.prefill += s.stage_ms.prefill;
                stage.backbone += s.stage_ms.backbone;
                stage.semantic += s.stage_ms.semantic;
                stage.flow += s.stage_ms.flow;
                stage.embed += s.stage_ms.embed;
                stage.steps += s.stage_ms.steps;
                log += &format!(
                    "{}\n",
                    serde_json::json!({"row": i, "text": text, "hypothesis": hyp, "errors": e, "frames": s.frames.len(), "ended": s.ended})
                );
            }
            stage.prefill /= texts.len() as f64;
            let path = format!("/tmp/voxtral-tts-corpus-{}.jsonl", grade.label());
            std::fs::write(&path, log).unwrap();
            eprintln!(
                "grade {}: {errors} word errors over {words} words in {} rows ({bad_rows} rows with errors, {unfinished} unfinished at {CORPUS_MAX_FRAMES} frames), {frames} frames; {}; rows in {path}",
                grade.label(),
                texts.len(),
                stage_report(&stage, codec_ms / frames.max(1) as f64)
            );
        }
    }
}
