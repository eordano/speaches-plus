use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use base64::Engine;
use candle_core::{DType, Device, Tensor, D};
use nv_layers::attn::{flash_attn, flash_attn_windowed, sdpa, sdpa_windowed, AttnConfig};
use nv_layers::conv::Conv1d;
use nv_layers::linear::Linear;
use nv_layers::mlp::Mlp;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::{load_linear, load_linear_with_bias_names, load_rmsnorm, load_tensor};
use nv_weights::{TensorSource, WeightLoader};
use serde::Deserialize;

use crate::qwen3::KvCache;

pub const SAMPLE_RATE: usize = 16_000;
pub const HOP_LENGTH: usize = 160;
pub const N_FFT: usize = 400;
pub const GLOBAL_LOG_MEL_MAX: f32 = 1.5;
pub const FRAME_MS: usize = 80;
pub const SAMPLES_PER_TOKEN: usize = SAMPLE_RATE * FRAME_MS / 1000;
pub const MEL_FRAMES_PER_TOKEN: usize = SAMPLES_PER_TOKEN / HOP_LENGTH;

pub const BOS_TOKEN: u32 = 1;
pub const EOS_TOKEN: u32 = 2;
pub const STREAMING_PAD_TOKEN: u32 = 32;
pub const STREAMING_WORD_TOKEN: u32 = 33;

pub const OFFLINE_STREAMING_BUFFER_TOKENS_COVER_ONE_LONG_WORD_AFTER_THE_AUDIO_ENDS: usize = 10;
pub const TIME_EMBEDDING_THETA: f32 = 10_000.0;
pub const ENCODER_ROPE_ROWS_COVER_35_MINUTES_OF_50HZ_FRAMES: usize = 1 << 17;
pub const DECODER_ROPE_ROWS: usize = 1 << 14;

pub const HF_CONFIG_FILE: &str = "config.json";
pub const TEKKEN_FILE: &str = "tekken.json";
pub const HF_WEIGHTS_FILE: &str = "model.safetensors";

#[derive(Clone, Debug, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VoxtralEncoderConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub num_mel_bins: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub rope_parameters: RopeParameters,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VoxtralTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub rope_parameters: RopeParameters,
    pub vocab_size: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

#[derive(Clone, Debug, Deserialize)]
pub struct VoxtralConfig {
    pub audio_config: VoxtralEncoderConfig,
    pub text_config: VoxtralTextConfig,
    pub downsample_factor: usize,
    pub audio_length_per_tok: usize,
    pub default_num_delay_tokens: usize,
    #[serde(default)]
    pub model_type: String,
}

impl VoxtralConfig {
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(raw).context("parse voxtral config.json")?;
        if cfg.model_type != "voxtral_realtime" {
            bail!(
                "config.json model_type {:?} is not voxtral_realtime (the offline `voxtral` graph \
                 is a different model and is not served)",
                cfg.model_type
            );
        }
        if cfg.audio_length_per_tok != MEL_FRAMES_PER_TOKEN {
            bail!(
                "audio_length_per_tok {} != {MEL_FRAMES_PER_TOKEN} mel frames per 80 ms token",
                cfg.audio_length_per_tok
            );
        }
        if cfg.audio_config.num_key_value_heads != cfg.audio_config.num_attention_heads {
            bail!("the encoder attention is MHA; GQA encoders are not wired");
        }
        Ok(cfg)
    }
}

pub struct TekkenVocab {
    token_bytes: Vec<Vec<u8>>,
    num_special_tokens: usize,
    pub transcription_delay_ms: u32,
    pub streaming_left_pad_tokens: usize,
}

#[derive(Deserialize)]
struct TekkenJson {
    config: TekkenConfig,
    vocab: Vec<TekkenEntry>,
    audio: Option<TekkenAudio>,
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
    transcription_delay_ms: Option<u32>,
    streaming_n_left_pad_tokens: Option<usize>,
}

impl TekkenVocab {
    pub fn from_json_str(raw: &str) -> Result<Self> {
        let parsed: TekkenJson = serde_json::from_str(raw).context("parse tekken.json")?;
        let num_special_tokens = parsed.config.default_num_special_tokens;
        let n_regular = parsed
            .config
            .default_vocab_size
            .checked_sub(num_special_tokens)
            .ok_or_else(|| anyhow!("tekken vocab smaller than its special-token block"))?;
        let mut token_bytes = vec![Vec::new(); n_regular];
        let engine = base64::engine::general_purpose::STANDARD;
        for e in parsed.vocab.iter().filter(|e| e.rank < n_regular) {
            token_bytes[e.rank] = engine
                .decode(&e.token_bytes)
                .with_context(|| format!("tekken rank {} token_bytes is not base64", e.rank))?;
        }
        let audio = parsed.audio.ok_or_else(|| {
            anyhow!("tekken.json carries no `audio` block: not a Voxtral tokenizer")
        })?;
        Ok(Self {
            token_bytes,
            num_special_tokens,
            transcription_delay_ms: audio
                .transcription_delay_ms
                .ok_or_else(|| anyhow!("tekken.json audio.transcription_delay_ms missing"))?,
            streaming_left_pad_tokens: audio
                .streaming_n_left_pad_tokens
                .ok_or_else(|| anyhow!("tekken.json audio.streaming_n_left_pad_tokens missing"))?,
        })
    }

    pub fn num_delay_tokens(&self) -> usize {
        (self.transcription_delay_ms as usize).div_ceil(FRAME_MS)
    }

    pub fn right_pad_tokens(&self) -> usize {
        self.num_delay_tokens()
            + 1
            + OFFLINE_STREAMING_BUFFER_TOKENS_COVER_ONE_LONG_WORD_AFTER_THE_AUDIO_ENDS
    }

    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.token_bytes
            .get((id as usize).checked_sub(self.num_special_tokens)?)
            .map(Vec::as_slice)
    }

    pub fn starts_with_space(&self, id: u32) -> bool {
        self.token_bytes(id)
            .is_some_and(|b| b.first() == Some(&b' '))
    }

    pub fn encoder(&self) -> Result<crate::voxtral_tekken::TekkenEncoder> {
        crate::voxtral_tekken::TekkenEncoder::new(&self.token_bytes, self.num_special_tokens)
    }

    pub fn is_special(&self, id: u32) -> bool {
        (id as usize) < self.num_special_tokens
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if self.is_special(id) {
                continue;
            }
            if let Some(b) = self.token_bytes.get(id as usize - self.num_special_tokens) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

pub struct HfLayouts<'a>(pub &'a WeightLoader);

pub fn hf_name_candidates(name: &str) -> Vec<String> {
    let mut v = vec![name.to_string()];
    if let Some(s) = name.strip_prefix("model.") {
        v.push(s.to_string());
        if let Some(rest) = s.strip_prefix("language_model.") {
            v.push(format!("language_model.model.{rest}"));
        }
    }
    v
}

impl TensorSource for HfLayouts<'_> {
    fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let candidates = hf_name_candidates(name);
        match candidates.iter().find(|c| self.0.has(c)) {
            Some(c) => self.0.get(c, dtype),
            None => bail!("tensor not found under any HF layout name: {candidates:?}"),
        }
    }

    fn has(&self, name: &str) -> bool {
        hf_name_candidates(name).iter().any(|c| self.0.has(c))
    }
}

pub fn merge_adapter_rows(
    weight: Tensor,
    prefix: &str,
    parts: &[(&str, usize)],
    adapter: Option<&nv_weights::lora_adapter::LoraAdapter>,
) -> Result<Tensor> {
    let Some(adapter) = adapter else {
        return Ok(weight);
    };
    let dtype = weight.dtype();
    let mut merged: Option<Tensor> = None;
    let mut off = 0;
    for (leaf, rows) in parts {
        let name = nv_weights::lora_adapter::normalize_module_name(&format!("{prefix}.{leaf}"));
        if let Some(lw) = adapter.loras.get(&name) {
            if lw.lora_b.dims()[0] != *rows || lw.lora_a.dims()[1] != weight.dims()[1] {
                bail!(
                    "adapter module {name}: B {:?} / A {:?} do not fit rows {rows} x in {}",
                    lw.lora_b.dims(),
                    lw.lora_a.dims(),
                    weight.dims()[1]
                );
            }
            let base = merged.take().unwrap_or(weight.to_dtype(DType::F32)?);
            let delta = (lw
                .lora_b
                .to_dtype(DType::F32)?
                .matmul(&lw.lora_a.to_dtype(DType::F32)?)?
                * lw.scaling)?;
            let block = (base.narrow(0, off, *rows)? + delta)?;
            let mut pieces = Vec::new();
            if off > 0 {
                pieces.push(base.narrow(0, 0, off)?);
            }
            pieces.push(block);
            let rest = base.dims()[0] - off - rows;
            if rest > 0 {
                pieces.push(base.narrow(0, off + rows, rest)?);
            }
            merged = Some(Tensor::cat(&pieces, 0)?);
        }
        off += rows;
    }
    match merged {
        Some(m) => Ok(m.to_dtype(dtype)?),
        None => Ok(weight),
    }
}

pub fn pad_audio_for_streaming(
    audio: &[f32],
    left_pad_tokens: usize,
    right_pad_tokens: usize,
) -> Vec<f32> {
    let left = left_pad_tokens * SAMPLES_PER_TOKEN;
    let align = (SAMPLES_PER_TOKEN - audio.len() % SAMPLES_PER_TOKEN) % SAMPLES_PER_TOKEN;
    let right = align + right_pad_tokens * SAMPLES_PER_TOKEN;
    let mut out = vec![0.0f32; left + audio.len() + right];
    out[left..left + audio.len()].copy_from_slice(audio);
    out
}

pub struct VoxtralMel {
    n_mels: usize,
    filters: Vec<f32>,
    bands: Vec<(usize, usize)>,
    hann: Vec<f32>,
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
}

pub const EACH_MEL_FILTER_SUMS_ITS_NONZERO_BAND_ONLY_BECAUSE_ADDING_POSITIVE_ZERO_IS_THE_IDENTITY: &str =
    "a slaney filter is one triangle over a contiguous run of FFT bins and zero elsewhere, and \
     every skipped term is 0 * power = +0.0, which leaves an f32 sum unchanged, so the banded \
     dot product is bitwise the full 201-bin one at a tenth of the multiplies";

pub fn filter_bands(filters: &[f32], n_mels: usize, n_bins: usize) -> Vec<(usize, usize)> {
    let _ = EACH_MEL_FILTER_SUMS_ITS_NONZERO_BAND_ONLY_BECAUSE_ADDING_POSITIVE_ZERO_IS_THE_IDENTITY;
    (0..n_mels)
        .map(|m| {
            let row = &filters[m * n_bins..(m + 1) * n_bins];
            let lo = row.iter().position(|w| *w != 0.0).unwrap_or(0);
            let hi = row.iter().rposition(|w| *w != 0.0).map_or(0, |k| k + 1);
            (lo, hi.max(lo))
        })
        .collect()
}

fn hz_to_slaney_mel(f: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if f >= MIN_LOG_HZ {
        min_log_mel + (f / MIN_LOG_HZ).ln() / logstep
    } else {
        f / F_SP
    }
}

fn slaney_mel_to_hz(m: f32) -> f32 {
    const F_SP: f32 = 200.0 / 3.0;
    const MIN_LOG_HZ: f32 = 1000.0;
    let min_log_mel = MIN_LOG_HZ / F_SP;
    let logstep = (6.4f32).ln() / 27.0;
    if m >= min_log_mel {
        MIN_LOG_HZ * ((m - min_log_mel) * logstep).exp()
    } else {
        F_SP * m
    }
}

pub fn slaney_mel_filters(n_mels: usize, n_fft: usize, sample_rate: usize) -> Vec<f32> {
    let n_bins = n_fft / 2 + 1;
    let m_min = hz_to_slaney_mel(0.0);
    let m_max = hz_to_slaney_mel(sample_rate as f32 / 2.0);
    let hz_points: Vec<f32> = (0..n_mels + 2)
        .map(|i| slaney_mel_to_hz(m_min + (m_max - m_min) * i as f32 / (n_mels as f32 + 1.0)))
        .collect();
    let mut filters = vec![0.0f32; n_mels * n_bins];
    for m in 0..n_mels {
        let (lower, center, upper) = (hz_points[m], hz_points[m + 1], hz_points[m + 2]);
        let enorm = 2.0 / (upper - lower).max(f32::EPSILON);
        for k in 0..n_bins {
            let freq = k as f32 * sample_rate as f32 / n_fft as f32;
            let w = if freq >= lower && freq <= center {
                (freq - lower) / (center - lower).max(f32::EPSILON)
            } else if freq > center && freq <= upper {
                (upper - freq) / (upper - center).max(f32::EPSILON)
            } else {
                0.0
            };
            filters[m * n_bins + k] = w * enorm;
        }
    }
    filters
}

impl VoxtralMel {
    pub fn new(n_mels: usize) -> Self {
        let mut planner = realfft::RealFftPlanner::<f32>::new();
        let hann = (0..N_FFT)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / N_FFT as f32).cos())
            .collect();
        let filters = slaney_mel_filters(n_mels, N_FFT, SAMPLE_RATE);
        let bands = filter_bands(&filters, n_mels, N_FFT / 2 + 1);
        Self {
            n_mels,
            filters,
            bands,
            hann,
            fft: planner.plan_fft_forward(N_FFT),
        }
    }

    pub fn n_mels(&self) -> usize {
        self.n_mels
    }

    pub fn log_mel(&self, audio: &[f32]) -> (Vec<f32>, usize) {
        let n_frames = audio.len() / HOP_LENGTH;
        let n_bins = N_FFT / 2 + 1;
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; audio.len() + N_FFT];
        for i in 0..pad.min(audio.len().saturating_sub(1)) {
            padded[i] = audio[pad - i];
        }
        padded[pad..pad + audio.len()].copy_from_slice(audio);
        for i in 0..pad {
            let src = audio.len().saturating_sub(2 + i);
            padded[pad + audio.len() + i] = audio.get(src).copied().unwrap_or(0.0);
        }
        let mut input = self.fft.make_input_vec();
        let mut output = self.fft.make_output_vec();
        let mut power = vec![0.0f32; n_bins];
        let mut mel = vec![0.0f32; self.n_mels * n_frames];
        let floor = GLOBAL_LOG_MEL_MAX - 8.0;
        for frame in 0..n_frames {
            let start = frame * HOP_LENGTH;
            for i in 0..N_FFT {
                input[i] = padded[start + i] * self.hann[i];
            }
            self.fft.process(&mut input, &mut output).expect("rfft");
            for (k, c) in output.iter().enumerate() {
                power[k] = c.re * c.re + c.im * c.im;
            }
            for (m, &(lo, hi)) in self.bands.iter().enumerate() {
                let row = &self.filters[m * n_bins + lo..m * n_bins + hi];
                let sum: f32 = row.iter().zip(&power[lo..hi]).map(|(a, b)| a * b).sum();
                let v = sum.max(1e-10).log10().max(floor);
                mel[m * n_frames + frame] = (v + 4.0) / 4.0;
            }
        }
        (mel, n_frames)
    }
}

struct EncoderLayer {
    attn_norm: RmsNorm,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    ffn_norm: RmsNorm,
    mlp: Mlp,
}

pub(crate) struct DecoderLayer {
    pub(crate) attn_norm: RmsNorm,
    pub(crate) qkv_proj: Linear,
    pub(crate) o_proj: Linear,
    pub(crate) ffn_norm: RmsNorm,
    pub(crate) gate_up_proj: Linear,
    pub(crate) down_proj: Linear,
}

pub const THE_DELAY_SCALE_FOLDS_INTO_THE_POST_ATTENTION_NORM_WEIGHT: &str =
    "the ada-RMSNorm multiplies the normed hidden state by 1 + ada(t_cond), a per-layer vector \
     fixed by the transcription delay, and RMSNorm's weight multiplies the same normed state, so \
     the product is one weight vector computed at load and no kernel runs for it per step";

pub const Q_ROWS_CARRY_THE_SOFTMAX_SCALE_SO_EVERY_ATTENTION_PATH_SEES_SCALE_ONE: &str =
    "1/sqrt(head_dim) is folded into the q rows of the fused qkv weight at load; RoPE is \
     linear so the rotated query keeps the scale, and both the flash_attn path and the \
     graph-captured split-K decode kernel then run with softmax_scale = 1";

pub const FUSED_QKV_AND_GATE_UP_HALVE_THE_DECODE_GEMV_LAUNCHES_PER_LAYER: &str =
    "q, k and v share one [4096+1024+1024, 3072] GEMV and gate/up one [18432, 3072] GEMV; a \
     decode step is memory-bound on weight reads, so the fused rows cost nothing extra and the \
     launch count per layer drops from seven linears to four";

pub const VOXTRAL_QUANT_ENV: &str = "NV_VOXTRAL_QUANT";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoxtralQuant {
    Bf16,
    Nvfp4,
    GgmlQ8_0,
    GgmlQ4K,
}

pub const NVFP4_IS_THE_CUDA_DEFAULT_BY_THE_LIBRISPEECH_73_CLIP_WER_INSTRUMENT: &str =
    "voxtral-tune eval over the 73 clips of hf-internal-testing/librispeech_asr_dummy (1150 \
     reference words) scored 77 word errors for bf16 graphed, bf16 eager and nvfp4 alike, and \
     the fixture stream matches the oracle with zero token diffs; the flip reads a quarter of \
     the bytes per step and NV_VOXTRAL_QUANT=bf16 restores the dense weights";

pub const CPU_SERVES_GGML_QUANTS_BECAUSE_F32_WEIGHTS_READ_TWELVE_GIGABYTES_A_STEP: &str =
    "candle's CPU k-quant kernels (AVX2/AVX-512 dot products under rayon) stream Q8_0 or Q4_K \
     rows instead of f32 ones, so a CPU decode step reads a quarter to an eighth of the bytes; \
     the CPU default is Q4_K, the GPU default stays bf16";

pub const FP8_IS_NOT_OFFERED_BECAUSE_CUBLASLT_SERVES_NO_PER_ROW_FP8_GEMM_ON_SM_120: &str =
    "the repo's fp8 Linear needs cuBLASLt's OUTER_VEC_32F per-row scaling, which the Blackwell \
     workstation part refuses at the load-time probe; NVFP4 reads half the bytes of fp8 anyway";

pub const DECODER_WEIGHTS_SET_THE_STEP_FLOOR_SO_ONLY_THEY_ARE_QUANTIZED: &str =
    "one 80 ms step reads every decoder linear once and the encoder runs a single prefill per \
     clip, so the quantized storage covers the 26 decoder layers' qkv, o, gate/up and down and \
     the lm_head; the embedding table and the encoder stay bf16";

pub const NVFP4_GEMV_ROWS_COVER_THE_39_TOKEN_STREAMING_PREFIX: usize = 64;
pub const PREFILL_ROWS_PER_BLOCK: usize = 8;

pub const THE_LM_HEAD_READS_A_QUARTER_OF_EACH_STEP_SO_IT_QUANTIZES_WITH_THE_DECODER: &str =
    "the tied 131072x3072 head is 805 MB in bf16 and the CUPTI trace put its GEMV at a quarter \
     of a decode step, so under NV_VOXTRAL_QUANT=nvfp4 the head gets its own NVFP4 copy while \
     the embedding table it is tied to stays bf16";

impl VoxtralQuant {
    pub fn from_env_for(device: &Device) -> Result<Self> {
        let _ = CPU_SERVES_GGML_QUANTS_BECAUSE_F32_WEIGHTS_READ_TWELVE_GIGABYTES_A_STEP;
        let _ = NVFP4_IS_THE_CUDA_DEFAULT_BY_THE_LIBRISPEECH_73_CLIP_WER_INSTRUMENT;
        let on_cpu = matches!(device, Device::Cpu);
        match std::env::var(VOXTRAL_QUANT_ENV).ok().as_deref() {
            None | Some("") => Ok(if on_cpu { Self::GgmlQ4K } else { Self::Nvfp4 }),
            Some("bf16") | Some("f32") => Ok(Self::Bf16),
            Some("nvfp4") => Ok(Self::Nvfp4),
            Some("q8_0") => Ok(Self::GgmlQ8_0),
            Some("q4_k") => Ok(Self::GgmlQ4K),
            Some(other) => {
                let _ = FP8_IS_NOT_OFFERED_BECAUSE_CUBLASLT_SERVES_NO_PER_ROW_FP8_GEMM_ON_SM_120;
                bail!("{VOXTRAL_QUANT_ENV}={other:?}: expected bf16, nvfp4, q8_0 or q4_k")
            }
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Nvfp4 => "nvfp4",
            Self::GgmlQ8_0 => "q8_0",
            Self::GgmlQ4K => "q4_k",
        }
    }

    fn ggml_dtype(self) -> Option<candle_core::quantized::GgmlDType> {
        match self {
            Self::GgmlQ8_0 => Some(candle_core::quantized::GgmlDType::Q8_0),
            Self::GgmlQ4K => Some(candle_core::quantized::GgmlDType::Q4K),
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct DecoderLinearBuilder {
    quant: VoxtralQuant,
    nvfp4: Option<std::sync::Arc<std::sync::Mutex<nv_quant::nvfp4::Nvfp4GemmRunner>>>,
}

#[cfg(feature = "cuda")]
impl DecoderLinearBuilder {
    pub(crate) fn new(quant: VoxtralQuant, device: &Device) -> Result<Self> {
        let _ = DECODER_WEIGHTS_SET_THE_STEP_FLOOR_SO_ONLY_THEY_ARE_QUANTIZED;
        let stream = match device {
            Device::Cuda(d) => Some(d.cuda_stream()),
            _ if quant == VoxtralQuant::Nvfp4 => {
                bail!("{VOXTRAL_QUANT_ENV}={} needs a CUDA device", quant.label())
            }
            _ => None,
        };
        Ok(Self {
            quant,
            nvfp4: match (quant, stream) {
                (VoxtralQuant::Nvfp4, Some(stream)) => Some(std::sync::Arc::new(
                    std::sync::Mutex::new(nv_quant::nvfp4::Nvfp4GemmRunner::new(stream)?),
                )),
                _ => None,
            },
        })
    }

    pub(crate) fn build(&self, weight: Tensor, device: &Device) -> Result<Linear> {
        if let Some(gdt) = self.quant.ggml_dtype() {
            return Linear::from_bf16_quantized_ggml(&weight.contiguous()?, None, gdt);
        }
        match self.quant {
            VoxtralQuant::Bf16 | VoxtralQuant::GgmlQ8_0 | VoxtralQuant::GgmlQ4K => {
                Linear::new(weight.contiguous()?, None)
            }
            VoxtralQuant::Nvfp4 => {
                let mut l = Linear::from_bf16_quantized_nvfp4_dev(
                    &weight,
                    None,
                    device,
                    self.nvfp4.clone().expect("nvfp4 runner"),
                )?;
                l.set_nvfp4_bf16act_rows_before_sharing(
                    NVFP4_GEMV_ROWS_COVER_THE_39_TOKEN_STREAMING_PREFIX,
                )?;
                Ok(l)
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub(crate) struct DecoderLinearBuilder(VoxtralQuant);

#[cfg(not(feature = "cuda"))]
impl DecoderLinearBuilder {
    pub(crate) fn new(quant: VoxtralQuant, _device: &Device) -> Result<Self> {
        if quant == VoxtralQuant::Nvfp4 {
            bail!(
                "{VOXTRAL_QUANT_ENV}={} needs the cuda feature",
                quant.label()
            );
        }
        Ok(Self(quant))
    }

    pub(crate) fn build(&self, weight: Tensor, _device: &Device) -> Result<Linear> {
        if let Some(gdt) = self.0.ggml_dtype() {
            return Linear::from_bf16_quantized_ggml(&weight.contiguous()?, None, gdt);
        }
        Linear::new(weight.contiguous()?, None)
    }
}

pub struct VoxtralRealtime {
    pub(crate) cfg: VoxtralConfig,
    vocab: TekkenVocab,
    mel: VoxtralMel,
    enc_conv1: Conv1d,
    enc_conv2: Conv1d,
    enc_layers: Vec<EncoderLayer>,
    enc_norm: RmsNorm,
    enc_rope: Rope,
    proj1: Linear,
    proj2: Linear,
    pub(crate) embed: Tensor,
    pub(crate) dec_layers: Vec<DecoderLayer>,
    pub(crate) dec_norm: RmsNorm,
    pub(crate) lm_head: Linear,
    pub(crate) dec_rope: Rope,
    pub(crate) device: Device,
    dtype: DType,
    quant: VoxtralQuant,
    prefix_kv: std::sync::Mutex<Option<Vec<(Tensor, Tensor)>>>,
    draft_head: std::sync::Mutex<Option<DraftHead>>,
    #[cfg(feature = "cuda")]
    graph: std::sync::Mutex<Option<crate::voxtral_graph::GraphState>>,
}

pub const DRAFT_HEAD_FILE: &str = "voxtral-draft-head.safetensors";
pub const VOXTRAL_DRAFT_HEAD_ENV: &str = "NV_VOXTRAL_DRAFT_HEAD";

pub const THE_DRAFT_HEAD_IS_A_LINEAR_PROBE_OVER_THE_AUDIO_EMBEDDING_THE_DECODER_SAW: &str =
    "the token at stream position q is the decoder's argmax over the input at q - 1, whose \
     audio embedding already carries the causal encoder's view of the clip, so one linear over \
     that embedding predicts a class per position ([STREAMING_PAD], [STREAMING_WORD], each of \
     the most frequent word pieces, or other) for the whole clip in one matmul before the \
     decode, and the per-position table drafts the class's token; greedy verification keeps \
     the committed stream the single-row graph's whatever the probe says";

pub struct DraftHead {
    pub weight: Tensor,
    pub bias: Tensor,
    pub classes: Vec<u32>,
}

impl DraftHead {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let _ = THE_DRAFT_HEAD_IS_A_LINEAR_PROBE_OVER_THE_AUDIO_EMBEDDING_THE_DECODER_SAW;
        let t = candle_core::safetensors::load(path, device)
            .with_context(|| format!("load draft head {}", path.display()))?;
        let get = |n: &str| t.get(n).cloned().ok_or_else(|| anyhow!("draft head lacks {n}"));
        let classes = get("classes")?.to_vec1::<u32>()?;
        let weight = get("weight")?.to_dtype(DType::F32)?;
        ensure!(weight.dims()[0] == classes.len(), "draft head weight rows != classes");
        Ok(Self {
            weight,
            bias: get("bias")?.to_dtype(DType::F32)?,
            classes,
        })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        let classes = Tensor::new(self.classes.as_slice(), self.weight.device())?;
        candle_core::safetensors::save(
            &std::collections::HashMap::from([
                ("weight", self.weight.clone()),
                ("bias", self.bias.clone()),
                ("classes", classes),
            ]),
            path,
        )?;
        Ok(())
    }

    pub fn predict_classes(&self, audio_embeds: &Tensor) -> Result<Vec<u32>> {
        let (_, n, h) = audio_embeds.dims3()?;
        let x = audio_embeds.reshape((n, h))?.to_dtype(DType::F32)?;
        let logits = x.matmul(&self.weight.t()?)?.broadcast_add(&self.bias)?;
        Ok(logits.argmax(D::Minus1)?.to_vec1::<u32>()?)
    }

    pub fn draft_table(&self, audio_embeds: &Tensor) -> Result<Vec<u32>> {
        let classes = self.predict_classes(audio_embeds)?;
        let mut table = vec![STREAMING_PAD_TOKEN];
        table.extend(classes[..classes.len() - 1].iter().map(|&c| self.classes[c as usize]));
        Ok(table)
    }
}

pub const THE_LEFT_PAD_POSITIONS_ARE_CLIP_INDEPENDENT_BECAUSE_THE_ENCODER_IS_CAUSAL: &str =
    "pad_audio_for_streaming puts 32 tokens of zeros before the clip, the mel frame of a \
     position reads 200 samples past it, and conv1, conv2, the sliding-window causal \
     attention and the 4x stacking all look backwards, so every 80 ms position whose last mel \
     frame ends inside the zero pad has one audio embedding for every clip; the suite covering \
     it holds those encoder rows bitwise across clips, and the decoder K/V of those positions \
     is computed once per model from one token of silence and seeds every clip's cache";

pub fn clip_independent_prefix_positions(left_pad_tokens: usize) -> usize {
    let _ = THE_LEFT_PAD_POSITIONS_ARE_CLIP_INDEPENDENT_BECAUSE_THE_ENCODER_IS_CAUSAL;
    let zero_samples = left_pad_tokens * SAMPLES_PER_TOKEN;
    let zero_frames = (zero_samples - N_FFT / 2) / HOP_LENGTH + 1;
    zero_frames / MEL_FRAMES_PER_TOKEN
}

pub fn time_embedding(num_delay_tokens: usize, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let t = num_delay_tokens as f32;
    let mut out = vec![0.0f32; dim];
    for i in 0..half {
        let inv_freq = (-(TIME_EMBEDDING_THETA.ln()) * i as f32 / half as f32).exp();
        out[i] = (t * inv_freq).cos();
        out[half + i] = (t * inv_freq).sin();
    }
    out
}

pub(crate) fn serving_dtype(device: &Device) -> DType {
    if device.is_cuda() || device.is_metal() {
        DType::BF16
    } else {
        DType::F32
    }
}

pub(crate) fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: &AttnConfig,
    window: Option<usize>,
) -> Result<Tensor> {
    let (q, k, v) = (q.contiguous()?, k.contiguous()?, v.contiguous()?);
    match (q.device().is_cuda(), window) {
        (true, Some(wd)) => flash_attn_windowed(&q, &k, &v, cfg, Some(wd - 1), Some(0)),
        (true, None) => flash_attn(&q, &k, &v, cfg),
        (false, Some(wd)) => sdpa_windowed(&q, &k, &v, cfg, wd),
        (false, None) => sdpa(&q, &k, &v, cfg),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decoder_hidden_over(
    layers: &[DecoderLayer],
    rope: &Rope,
    norm: &RmsNorm,
    t: &VoxtralTextConfig,
    device: &Device,
    inputs_embeds: &Tensor,
    start: usize,
    cache: &mut KvCache,
) -> Result<Tensor> {
    let seq = inputs_embeds.dims()[1];
    let positions = Tensor::from_vec(
        (start as i32..(start + seq) as i32).collect::<Vec<i32>>(),
        seq,
        device,
    )?;
    let new_total = start + seq;
    let q_rows = t.num_attention_heads * t.head_dim;
    let kv_rows = t.num_key_value_heads * t.head_dim;
    let inter = t.intermediate_size;
    let attn_cfg = AttnConfig {
        num_heads: t.num_attention_heads,
        num_kv_heads: t.num_key_value_heads,
        head_dim: t.head_dim,
        softmax_scale: 1.0,
        causal: true,
    };
    let mut x = inputs_embeds.clone();
    for (i, layer) in layers.iter().enumerate() {
        let h = layer.attn_norm.forward(&x)?;
        let qkv = layer.qkv_proj.forward(&h)?;
        let q = qkv.narrow(2, 0, q_rows)?.contiguous()?.reshape((
            1,
            seq,
            t.num_attention_heads,
            t.head_dim,
        ))?;
        let k = qkv.narrow(2, q_rows, kv_rows)?.contiguous()?.reshape((
            1,
            seq,
            t.num_key_value_heads,
            t.head_dim,
        ))?;
        let v = qkv
            .narrow(2, q_rows + kv_rows, kv_rows)?
            .contiguous()?
            .reshape((1, seq, t.num_key_value_heads, t.head_dim))?;
        let (q, k) = rope.apply(&q, &k, &positions)?;
        cache.write_at(i, start, &k.contiguous()?, &v)?;
        let (k_full, v_full) = cache.view(i, new_total)?;
        let attn = attention(&q, &k_full, &v_full, &attn_cfg, None)?;
        let attn = layer.o_proj.forward(&attn.reshape((1, seq, q_rows))?)?;
        let (h, x_mid) = layer.ffn_norm.forward_residual(&attn, &x)?;
        let gate_up = layer.gate_up_proj.forward(&h)?;
        let gate = gate_up.narrow(2, 0, inter)?;
        let up = gate_up.narrow(2, inter, inter)?;
        let act = candle_nn::ops::silu(&gate.contiguous()?)?.mul(&up.contiguous()?)?;
        x = (&x_mid + layer.down_proj.forward(&act)?)?;
    }
    cache.advance(seq);
    let last = x.narrow(1, seq - 1, 1)?;
    norm.forward(&last)
}

impl VoxtralRealtime {
    pub fn load_dir(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_dir_with_adapter(dir, device, None)
    }

    pub const ADAPTERS_SERVE_MERGED_SO_THE_CAPTURED_STEP_NEVER_SEES_A_HOOK: &'static str =
        "a PEFT adapter is folded into the bf16 fused weights before they are quantized and \
         built, so serving it costs no kernel and the graph capture is unchanged; LoRA hooks \
         exist for training and eager evaluation only, and a model carrying hooks decodes \
         eagerly";

    pub fn load_dir_with_adapter(
        dir: &Path,
        device: &Device,
        adapter_dir: Option<&Path>,
    ) -> Result<Self> {
        let _ = Self::ADAPTERS_SERVE_MERGED_SO_THE_CAPTURED_STEP_NEVER_SEES_A_HOOK;
        let adapter = match adapter_dir {
            Some(d) => Some(
                nv_weights::lora_adapter::LoraAdapter::load(d, device)
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
        let weights = WeightLoader::open_file(&weights_path, device)
            .with_context(|| format!("open {}", weights_path.display()))?;
        let model = Self::from_loader(cfg, vocab, &weights, device, adapter.as_ref())?;
        let head = std::env::var_os(VOXTRAL_DRAFT_HEAD_ENV)
            .map(std::path::PathBuf::from)
            .or_else(|| adapter_dir.map(|d| d.join(DRAFT_HEAD_FILE)).filter(|p| p.is_file()));
        if let Some(path) = head {
            model.set_draft_head(Some(DraftHead::load(&path, device)?));
        }
        Ok(model)
    }

    pub fn set_draft_head(&self, head: Option<DraftHead>) {
        if let Ok(mut guard) = self.draft_head.lock() {
            *guard = head;
        }
    }

    pub fn draft_table(&self, audio_embeds: &Tensor) -> Result<Option<Vec<u32>>> {
        let guard = self
            .draft_head
            .lock()
            .map_err(|_| anyhow!("voxtral draft head mutex poisoned"))?;
        guard.as_ref().map(|h| h.draft_table(audio_embeds)).transpose()
    }

    pub fn from_loader(
        cfg: VoxtralConfig,
        vocab: TekkenVocab,
        w: &WeightLoader,
        device: &Device,
        adapter: Option<&nv_weights::lora_adapter::LoraAdapter>,
    ) -> Result<Self> {
        let w = &HfLayouts(w);
        let dtype = serving_dtype(device);
        let quant = VoxtralQuant::from_env_for(device)?;
        let dec_linear = DecoderLinearBuilder::new(quant, device)?;
        let a = &cfg.audio_config;
        let t = &cfg.text_config;
        let conv_dtype = dtype;
        let enc_conv1 = Conv1d::new(
            load_tensor(
                w,
                "model.audio_tower.embedder.conv1.weight",
                &[a.hidden_size, a.num_mel_bins, 3],
                conv_dtype,
            )?,
            Some(load_tensor(
                w,
                "model.audio_tower.embedder.conv1.bias",
                &[a.hidden_size],
                conv_dtype,
            )?),
            1,
            0,
        )?;
        let enc_conv2 = Conv1d::new(
            load_tensor(
                w,
                "model.audio_tower.embedder.conv2.weight",
                &[a.hidden_size, a.hidden_size, 3],
                conv_dtype,
            )?,
            Some(load_tensor(
                w,
                "model.audio_tower.embedder.conv2.bias",
                &[a.hidden_size],
                conv_dtype,
            )?),
            2,
            0,
        )?;
        let mut enc_layers = Vec::with_capacity(a.num_hidden_layers);
        for i in 0..a.num_hidden_layers {
            let p = format!("model.audio_tower.layers.{i}");
            let qkv = a.num_attention_heads * a.head_dim;
            enc_layers.push(EncoderLayer {
                attn_norm: load_rmsnorm(
                    w,
                    &format!("{p}.self_attn_layer_norm.weight"),
                    a.hidden_size,
                    a.rms_norm_eps,
                    dtype,
                )?,
                q_proj: load_linear_with_bias_names(
                    w,
                    &format!("{p}.self_attn.q_proj.weight"),
                    qkv,
                    a.hidden_size,
                    Some(&format!("{p}.self_attn.q_proj.bias")),
                    dtype,
                )?,
                k_proj: load_linear(
                    w,
                    &format!("{p}.self_attn.k_proj.weight"),
                    qkv,
                    a.hidden_size,
                    dtype,
                )?,
                v_proj: load_linear_with_bias_names(
                    w,
                    &format!("{p}.self_attn.v_proj.weight"),
                    qkv,
                    a.hidden_size,
                    Some(&format!("{p}.self_attn.v_proj.bias")),
                    dtype,
                )?,
                o_proj: load_linear_with_bias_names(
                    w,
                    &format!("{p}.self_attn.o_proj.weight"),
                    a.hidden_size,
                    qkv,
                    Some(&format!("{p}.self_attn.o_proj.bias")),
                    dtype,
                )?,
                ffn_norm: load_rmsnorm(
                    w,
                    &format!("{p}.final_layer_norm.weight"),
                    a.hidden_size,
                    a.rms_norm_eps,
                    dtype,
                )?,
                mlp: Mlp::new(
                    load_linear(
                        w,
                        &format!("{p}.mlp.gate_proj.weight"),
                        a.intermediate_size,
                        a.hidden_size,
                        dtype,
                    )?,
                    load_linear(
                        w,
                        &format!("{p}.mlp.up_proj.weight"),
                        a.intermediate_size,
                        a.hidden_size,
                        dtype,
                    )?,
                    load_linear_with_bias_names(
                        w,
                        &format!("{p}.mlp.down_proj.weight"),
                        a.hidden_size,
                        a.intermediate_size,
                        Some(&format!("{p}.mlp.down_proj.bias")),
                        dtype,
                    )?,
                )?,
            });
        }
        let enc_norm = load_rmsnorm(
            w,
            "model.audio_tower.norm.weight",
            a.hidden_size,
            a.rms_norm_eps,
            dtype,
        )?;
        let enc_rope = Rope::new(
            RopeConfig {
                head_dim: a.head_dim,
                max_seq_len: ENCODER_ROPE_ROWS_COVER_35_MINUTES_OF_50HZ_FRAMES,
                base: a.rope_parameters.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        let proj_in = a.hidden_size * cfg.downsample_factor;
        let proj1 = load_linear(
            w,
            "model.multi_modal_projector.linear_1.weight",
            t.hidden_size,
            proj_in,
            dtype,
        )?;
        let proj2 = load_linear(
            w,
            "model.multi_modal_projector.linear_2.weight",
            t.hidden_size,
            t.hidden_size,
            dtype,
        )?;
        let embed = load_tensor(
            w,
            "model.language_model.embed_tokens.weight",
            &[t.vocab_size, t.hidden_size],
            dtype,
        )?;
        let t_cond = Tensor::from_vec(
            time_embedding(vocab.num_delay_tokens(), t.hidden_size),
            (1, t.hidden_size),
            device,
        )?
        .to_dtype(dtype)?;
        let mut dec_layers = Vec::with_capacity(t.num_hidden_layers);
        let _ = FUSED_QKV_AND_GATE_UP_HALVE_THE_DECODE_GEMV_LAUNCHES_PER_LAYER;
        let q_rows = t.num_attention_heads * t.head_dim;
        let kv_rows = t.num_key_value_heads * t.head_dim;
        for i in 0..t.num_hidden_layers {
            let p = format!("model.language_model.layers.{i}");
            let ada1 = load_tensor(
                w,
                &format!("{p}.ada_rms_norm.linear1.weight"),
                &[32, t.hidden_size],
                dtype,
            )?;
            let ada2 = load_tensor(
                w,
                &format!("{p}.ada_rms_norm.linear2.weight"),
                &[t.hidden_size, 32],
                dtype,
            )?;
            let _ = THE_DELAY_SCALE_FOLDS_INTO_THE_POST_ATTENTION_NORM_WEIGHT;
            let hidden = t_cond.matmul(&ada1.t()?)?.gelu_erf()?;
            let ada_scale = (hidden.matmul(&ada2.t()?)? + 1.0)?.reshape(t.hidden_size)?;
            let ffn_norm_w = load_tensor(
                w,
                &format!("{p}.post_attention_layernorm.weight"),
                &[t.hidden_size],
                dtype,
            )?;
            let ffn_norm = RmsNorm::new(
                (ffn_norm_w.to_dtype(DType::F32)? * ada_scale.to_dtype(DType::F32)?)?
                    .to_dtype(dtype)?,
                t.rms_norm_eps,
            );
            let _ = Q_ROWS_CARRY_THE_SOFTMAX_SCALE_SO_EVERY_ATTENTION_PATH_SEES_SCALE_ONE;
            let q_scaled = (load_tensor(
                w,
                &format!("{p}.self_attn.q_proj.weight"),
                &[q_rows, t.hidden_size],
                dtype,
            )?
            .to_dtype(DType::F32)?
                * (1.0 / (t.head_dim as f64).sqrt()))?
            .to_dtype(dtype)?;
            let qkv = Tensor::cat(
                &[
                    q_scaled,
                    load_tensor(
                        w,
                        &format!("{p}.self_attn.k_proj.weight"),
                        &[kv_rows, t.hidden_size],
                        dtype,
                    )?,
                    load_tensor(
                        w,
                        &format!("{p}.self_attn.v_proj.weight"),
                        &[kv_rows, t.hidden_size],
                        dtype,
                    )?,
                ],
                0,
            )?;
            let gate_up = Tensor::cat(
                &[
                    load_tensor(
                        w,
                        &format!("{p}.mlp.gate_proj.weight"),
                        &[t.intermediate_size, t.hidden_size],
                        dtype,
                    )?,
                    load_tensor(
                        w,
                        &format!("{p}.mlp.up_proj.weight"),
                        &[t.intermediate_size, t.hidden_size],
                        dtype,
                    )?,
                ],
                0,
            )?;
            dec_layers.push(DecoderLayer {
                attn_norm: load_rmsnorm(
                    w,
                    &format!("{p}.input_layernorm.weight"),
                    t.hidden_size,
                    t.rms_norm_eps,
                    dtype,
                )?,
                qkv_proj: dec_linear.build(
                    merge_adapter_rows(
                        qkv,
                        &format!("{p}.self_attn"),
                        &[("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)],
                        adapter,
                    )?,
                    device,
                )?,
                o_proj: dec_linear.build(
                    merge_adapter_rows(
                        load_tensor(
                            w,
                            &format!("{p}.self_attn.o_proj.weight"),
                            &[t.hidden_size, q_rows],
                            dtype,
                        )?,
                        &format!("{p}.self_attn"),
                        &[("o_proj", t.hidden_size)],
                        adapter,
                    )?,
                    device,
                )?,
                ffn_norm,
                gate_up_proj: dec_linear.build(
                    merge_adapter_rows(
                        gate_up,
                        &format!("{p}.mlp"),
                        &[
                            ("gate_proj", t.intermediate_size),
                            ("up_proj", t.intermediate_size),
                        ],
                        adapter,
                    )?,
                    device,
                )?,
                down_proj: dec_linear.build(
                    merge_adapter_rows(
                        load_tensor(
                            w,
                            &format!("{p}.mlp.down_proj.weight"),
                            &[t.hidden_size, t.intermediate_size],
                            dtype,
                        )?,
                        &format!("{p}.mlp"),
                        &[("down_proj", t.hidden_size)],
                        adapter,
                    )?,
                    device,
                )?,
            });
        }
        let dec_norm = load_rmsnorm(
            w,
            "model.language_model.norm.weight",
            t.hidden_size,
            t.rms_norm_eps,
            dtype,
        )?;
        let lm_head_weight = if w.has("lm_head.weight") {
            load_tensor(w, "lm_head.weight", &[t.vocab_size, t.hidden_size], dtype)?
        } else {
            embed.clone()
        };
        let _ = THE_LM_HEAD_READS_A_QUARTER_OF_EACH_STEP_SO_IT_QUANTIZES_WITH_THE_DECODER;
        let lm_head = dec_linear.build(lm_head_weight, device)?;
        let dec_rope = Rope::new(
            RopeConfig {
                head_dim: t.head_dim,
                max_seq_len: DECODER_ROPE_ROWS,
                base: t.rope_parameters.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        let mel = VoxtralMel::new(a.num_mel_bins);
        Ok(Self {
            cfg,
            vocab,
            mel,
            enc_conv1,
            enc_conv2,
            enc_layers,
            enc_norm,
            enc_rope,
            proj1,
            proj2,
            embed,
            dec_layers,
            dec_norm,
            lm_head,
            dec_rope,
            device: device.clone(),
            dtype,
            quant,
            prefix_kv: std::sync::Mutex::new(None),
            draft_head: std::sync::Mutex::new(None),
            #[cfg(feature = "cuda")]
            graph: std::sync::Mutex::new(None),
        })
    }

    pub fn quant(&self) -> VoxtralQuant {
        self.quant
    }

    pub fn has_lora_hooks(&self) -> bool {
        self.dec_layers.iter().any(|l| {
            l.qkv_proj.has_lora()
                || l.o_proj.has_lora()
                || l.gate_up_proj.has_lora()
                || l.down_proj.has_lora()
        }) || self.lm_head.has_lora()
    }

    pub const ADAPTER_CHANGES_INVALIDATE_THE_CAPTURED_STEP: &'static str =
        "a captured graph bakes the hook kernels and the adapter pointers present at capture \
         time, so attaching or detaching a LoRA drops the GraphState and the next clip \
         captures afresh";

    pub fn drop_graph_state(&self) {
        let _ = Self::ADAPTER_CHANGES_INVALIDATE_THE_CAPTURED_STEP;
        if let Ok(mut guard) = self.prefix_kv.lock() {
            *guard = None;
        }
        #[cfg(feature = "cuda")]
        if let Ok(mut guard) = self.graph.lock() {
            *guard = None;
        }
    }

    pub fn config(&self) -> &VoxtralConfig {
        &self.cfg
    }

    pub fn graph_rows_steps_committed(&self) -> Option<(usize, usize, usize)> {
        #[cfg(feature = "cuda")]
        if let Ok(guard) = self.graph.lock() {
            return guard.as_ref().map(|g| {
                let (steps, committed) = g.steps_and_committed_positions();
                (g.rows(), steps, committed)
            });
        }
        None
    }

    pub fn vocab(&self) -> &TekkenVocab {
        &self.vocab
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn prefix_tokens(&self) -> Vec<u32> {
        let mut t = vec![self.cfg.text_config.bos_token_id];
        t.extend(std::iter::repeat_n(
            STREAMING_PAD_TOKEN,
            self.vocab.streaming_left_pad_tokens + self.vocab.num_delay_tokens(),
        ));
        t
    }

    pub fn max_audio_tokens(&self) -> usize {
        self.cfg
            .text_config
            .sliding_window
            .unwrap_or(DECODER_ROPE_ROWS)
            .min(DECODER_ROPE_ROWS)
            - self.prefix_tokens().len()
    }

    pub fn features(&self, audio_16k_mono: &[f32]) -> Result<Tensor> {
        let padded = pad_audio_for_streaming(
            audio_16k_mono,
            self.vocab.streaming_left_pad_tokens,
            self.vocab.right_pad_tokens(),
        );
        let (mel, frames) = self.mel.log_mel(&padded);
        Ok(
            Tensor::from_vec(mel, (1, self.mel.n_mels(), frames), &self.device)?
                .to_dtype(self.dtype)?,
        )
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        cfg: &AttnConfig,
        window: Option<usize>,
    ) -> Result<Tensor> {
        attention(q, k, v, cfg, window)
    }

    pub fn encode(&self, features: &Tensor) -> Result<Tensor> {
        let a = &self.cfg.audio_config;
        let x = self
            .enc_conv1
            .forward(&features.pad_with_zeros(2, 2, 0)?)?
            .gelu_erf()?;
        let x = self
            .enc_conv2
            .forward(&x.pad_with_zeros(2, 1, 0)?)?
            .gelu_erf()?;
        let mut x = x.transpose(1, 2)?.contiguous()?;
        let seq = x.dims()[1];
        let positions = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &self.device)?;
        let attn_cfg = AttnConfig {
            num_heads: a.num_attention_heads,
            num_kv_heads: a.num_key_value_heads,
            head_dim: a.head_dim,
            softmax_scale: 1.0 / (a.head_dim as f32).sqrt(),
            causal: true,
        };
        for layer in &self.enc_layers {
            let h = layer.attn_norm.forward(&x)?;
            let q =
                layer
                    .q_proj
                    .forward(&h)?
                    .reshape((1, seq, a.num_attention_heads, a.head_dim))?;
            let k =
                layer
                    .k_proj
                    .forward(&h)?
                    .reshape((1, seq, a.num_key_value_heads, a.head_dim))?;
            let v =
                layer
                    .v_proj
                    .forward(&h)?
                    .reshape((1, seq, a.num_key_value_heads, a.head_dim))?;
            let (q, k) = self
                .enc_rope
                .apply(&q.contiguous()?, &k.contiguous()?, &positions)?;
            let attn = self.attention(&q, &k, &v, &attn_cfg, a.sliding_window)?;
            let attn = layer.o_proj.forward(&attn.reshape((
                1,
                seq,
                a.num_attention_heads * a.head_dim,
            ))?)?;
            x = (x + attn)?;
            let h = layer.ffn_norm.forward(&x)?;
            x = (&x + layer.mlp.forward(&h)?)?;
        }
        let x = self.enc_norm.forward(&x)?;
        let ds = self.cfg.downsample_factor;
        let n_tokens = seq / ds;
        let x = x
            .narrow(1, 0, n_tokens * ds)?
            .reshape((1, n_tokens, a.hidden_size * ds))?;
        let x = self.proj1.forward(&x)?.gelu_erf()?;
        self.proj2.forward(&x)
    }

    fn decoder_hidden(
        &self,
        inputs_embeds: &Tensor,
        start: usize,
        cache: &mut KvCache,
    ) -> Result<Tensor> {
        decoder_hidden_over(
            &self.dec_layers,
            &self.dec_rope,
            &self.dec_norm,
            &self.cfg.text_config,
            &self.device,
            inputs_embeds,
            start,
            cache,
        )
    }

    pub fn prefix_len(&self) -> usize {
        1 + self.vocab.streaming_left_pad_tokens + self.vocab.num_delay_tokens()
    }

    pub(crate) fn embed_tokens(&self, ids: &[u32]) -> Result<Tensor> {
        let idx = Tensor::new(ids, &self.device)?;
        Ok(self.embed.index_select(&idx, 0)?.reshape((
            1,
            ids.len(),
            self.cfg.text_config.hidden_size,
        ))?)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn embed_table(&self) -> &Tensor {
        &self.embed
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn decoder_layers(&self) -> &[DecoderLayer] {
        &self.dec_layers
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn decoder_rope(&self) -> &Rope {
        &self.dec_rope
    }

    pub(crate) fn decoder_linears(&self) -> Vec<(String, &Linear, Vec<(&'static str, usize)>)> {
        let t = &self.cfg.text_config;
        let q_rows = t.num_attention_heads * t.head_dim;
        let kv_rows = t.num_key_value_heads * t.head_dim;
        let mut out = Vec::with_capacity(self.dec_layers.len() * 4);
        for (i, l) in self.dec_layers.iter().enumerate() {
            let p = format!("model.language_model.layers.{i}");
            out.push((
                format!("{p}.self_attn"),
                &l.qkv_proj,
                vec![("q_proj", q_rows), ("k_proj", kv_rows), ("v_proj", kv_rows)],
            ));
            out.push((
                format!("{p}.self_attn"),
                &l.o_proj,
                vec![("o_proj", t.hidden_size)],
            ));
            out.push((
                format!("{p}.mlp"),
                &l.gate_up_proj,
                vec![
                    ("gate_proj", t.intermediate_size),
                    ("up_proj", t.intermediate_size),
                ],
            ));
            out.push((
                format!("{p}.mlp"),
                &l.down_proj,
                vec![("down_proj", t.hidden_size)],
            ));
        }
        out
    }

    pub const TRAINING_FORWARD_USES_THE_DIFFERENTIABLE_PATHS_OF_EVERY_LAYER: &'static str =
        "RmsNorm::forward_candle, Rope::apply_candle on f32 q/k, a plain-ops causal SDPA with \
         scale 1 (the q rows already carry 1/sqrt(head_dim)) and Linear::forward's dx-only \
         route keep the whole decoder on candle's autograd, so LoRA hooks attached to the fused \
         linears receive gradients while the base weights stay frozen";

    pub fn decoder_logits_train(&self, inputs_embeds: &Tensor) -> Result<Tensor> {
        let _ = Self::TRAINING_FORWARD_USES_THE_DIFFERENTIABLE_PATHS_OF_EVERY_LAYER;
        let t = &self.cfg.text_config;
        let seq = inputs_embeds.dims()[1];
        let q_rows = t.num_attention_heads * t.head_dim;
        let kv_rows = t.num_key_value_heads * t.head_dim;
        let inter = t.intermediate_size;
        let positions = Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &self.device)?;
        let mut x = inputs_embeds.clone();
        for layer in &self.dec_layers {
            let h = layer.attn_norm.forward_candle(&x)?;
            let qkv = layer.qkv_proj.forward(&h)?.to_dtype(DType::F32)?;
            let q =
                qkv.narrow(2, 0, q_rows)?
                    .reshape((1, seq, t.num_attention_heads, t.head_dim))?;
            let k = qkv.narrow(2, q_rows, kv_rows)?.reshape((
                1,
                seq,
                t.num_key_value_heads,
                t.head_dim,
            ))?;
            let v = qkv.narrow(2, q_rows + kv_rows, kv_rows)?.reshape((
                1,
                seq,
                t.num_key_value_heads,
                t.head_dim,
            ))?;
            let (q, k) = self.dec_rope.apply_candle(&q, &k, &positions)?;
            let attn = crate::voxtral_train::sdpa_train_scale_one(
                &q,
                &k,
                &v,
                t.num_attention_heads,
                t.num_key_value_heads,
                t.head_dim,
            )?
            .to_dtype(x.dtype())?;
            let attn = layer.o_proj.forward(&attn.reshape((1, seq, q_rows))?)?;
            let x_mid = (&x + attn)?;
            let h = layer.ffn_norm.forward_candle(&x_mid)?;
            let gate_up = layer.gate_up_proj.forward(&h)?;
            let gate = gate_up.narrow(2, 0, inter)?;
            let up = gate_up.narrow(2, inter, inter)?;
            let act = candle_nn::ops::silu(&gate)?.mul(&up)?;
            x = (&x_mid + layer.down_proj.forward(&act)?)?;
        }
        let normed = self.dec_norm.forward_candle(&x)?;
        let logits = self.lm_head.forward(&normed)?.to_dtype(DType::F32)?;
        logits.reshape((seq, t.vocab_size)).map_err(Into::into)
    }

    pub fn inputs_embeds_for_stream(
        &self,
        stream: &[u32],
        audio_embeds: &Tensor,
    ) -> Result<Tensor> {
        let n = stream.len();
        let embeds =
            (self.embed_tokens(stream)? + audio_embeds.narrow(1, 0, n)?.to_dtype(self.dtype)?)?;
        Ok(embeds)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn logits_from_hidden(&self, x: &Tensor) -> Result<Tensor> {
        let normed = self.dec_norm.forward(x)?;
        self.lm_head.forward(&normed)
    }

    fn argmax(&self, hidden: &Tensor) -> Result<u32> {
        let logits = self.lm_head.forward(hidden)?.flatten_all()?;
        let id = logits.argmax(D::Minus1)?.to_scalar::<u32>()?;
        Ok(id)
    }

    pub fn decode_stream(&self, audio_embeds: &Tensor) -> Result<Vec<u32>> {
        if matches!(self.device, Device::Cpu) {
            return crate::voxtral_cpu::decode_stream(self, audio_embeds);
        }
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() && graph_decode_enabled() && !self.has_lora_hooks() {
            let mut guard = self
                .graph
                .lock()
                .map_err(|_| anyhow!("voxtral graph state mutex poisoned"))?;
            if guard.is_none() {
                let capacity = self.max_audio_tokens() + self.prefix_tokens().len() + 1;
                let rows = crate::voxtral_graph::pad_draft_rows_for(self.quant)?;
                *guard = Some(crate::voxtral_graph::GraphState::new(
                    self, capacity, rows,
                )?);
            }
            let state = guard.as_mut().expect("graph state");
            let draft = self.draft_table(audio_embeds)?;
            return crate::voxtral_graph::decode_stream_graphed(
                self,
                state,
                audio_embeds,
                draft.as_deref(),
            );
        }
        self.decode_stream_eager(audio_embeds)
    }

    pub(crate) fn audio_tokens_in_range(&self, n_audio: usize) -> Result<()> {
        let prefix_len = self.prefix_len();
        if n_audio <= prefix_len {
            bail!("audio yields {n_audio} 80 ms tokens, not more than the {prefix_len}-token streaming prefix");
        }
        if n_audio > self.max_audio_tokens() + prefix_len {
            bail!(
                "audio yields {n_audio} tokens; the decoder serves at most {} per call (split the \
                 clip)",
                self.max_audio_tokens()
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prefill_into(&self, audio_embeds: &Tensor, cache: &mut KvCache) -> Result<u32> {
        self.audio_tokens_in_range(audio_embeds.dims()[1])?;
        let hidden = self.prefill_hidden(audio_embeds, cache)?;
        self.argmax(&hidden)
    }

    pub const THE_PREFIX_PREFILLS_IN_EIGHT_ROW_BLOCKS_SO_THE_NVFP4_LINEAR_READS_ITS_WEIGHTS_FIVE_TIMES_NOT_THIRTY_NINE: &'static str =
        "the NVFP4 bf16-activation GEMV serves row blocks up to \
         NVFP4_GEMV_MROWS_LADDER_MATCHES_THE_CUDA_SWITCH_AND_ROWS_ABOVE_IT_STILL_LOOP in one \
         launch and loops single rows above it, so a 39-row prefill read every decoder weight \
         39 times; eight-row blocks through the same decoder_hidden keep each row on the kernel \
         decode uses and the stream token-identical on the fixture and the 73-clip corpus";

    pub fn prefill_rows(
        &self,
        audio_embeds: &Tensor,
        cache: &mut KvCache,
        from: usize,
        to: usize,
    ) -> Result<Tensor> {
        let _ = Self::THE_PREFIX_PREFILLS_IN_EIGHT_ROW_BLOCKS_SO_THE_NVFP4_LINEAR_READS_ITS_WEIGHTS_FIVE_TIMES_NOT_THIRTY_NINE;
        let prefix = self.prefix_tokens();
        ensure!(from < to && to <= prefix.len(), "prefill rows {from}..{to} outside the prefix");
        let mut hidden = None;
        for start in (from..to).step_by(PREFILL_ROWS_PER_BLOCK) {
            let len = (to - start).min(PREFILL_ROWS_PER_BLOCK);
            let embeds = (self.embed_tokens(&prefix[start..start + len])?
                + audio_embeds.narrow(1, start, len)?)?;
            hidden = Some(self.decoder_hidden(&embeds, start, cache)?);
        }
        hidden.ok_or_else(|| anyhow!("empty prefill block"))
    }

    pub fn cached_prefix_positions(&self) -> usize {
        if self.device.is_cuda() {
            clip_independent_prefix_positions(self.vocab.streaming_left_pad_tokens)
        } else {
            0
        }
    }

    pub fn silence_audio_embeds(&self) -> Result<Tensor> {
        self.encode(&self.features(&[0.0; SAMPLES_PER_TOKEN])?)
    }

    pub fn prefix_kv_rows(&self, audio_embeds: &Tensor, n: usize) -> Result<Vec<(Tensor, Tensor)>> {
        let t = &self.cfg.text_config;
        let mut scratch = KvCache::with_dims(
            t.num_hidden_layers,
            n,
            t.num_key_value_heads,
            t.head_dim,
            &self.device,
            self.dtype,
        )?;
        self.prefill_rows(audio_embeds, &mut scratch, 0, n)?;
        (0..t.num_hidden_layers)
            .map(|i| {
                let (k, v) = scratch.view(i, n)?;
                Ok((k.contiguous()?, v.contiguous()?))
            })
            .collect()
    }

    fn seed_prefix(&self, cache: &mut KvCache) -> Result<usize> {
        let n = self.cached_prefix_positions();
        if n == 0 {
            return Ok(0);
        }
        let mut guard = self
            .prefix_kv
            .lock()
            .map_err(|_| anyhow!("voxtral prefix mutex poisoned"))?;
        if guard.is_none() {
            *guard = Some(self.prefix_kv_rows(&self.silence_audio_embeds()?, n)?);
        }
        for (i, (k, v)) in guard.as_ref().expect("prefix kv").iter().enumerate() {
            cache.write_at(i, 0, k, v)?;
        }
        cache.advance(n);
        Ok(n)
    }

    fn prefill_hidden(&self, audio_embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        let from = self.seed_prefix(cache)?;
        self.prefill_rows(audio_embeds, cache, from, self.prefix_len())
    }

    fn prefill(&self, audio_embeds: &Tensor) -> Result<(KvCache, u32)> {
        let n_audio = audio_embeds.dims()[1];
        self.audio_tokens_in_range(n_audio)?;
        let t = &self.cfg.text_config;
        let mut cache = KvCache::with_dims(
            t.num_hidden_layers,
            n_audio + 1,
            t.num_key_value_heads,
            t.head_dim,
            &self.device,
            self.dtype,
        )?;
        let hidden = self.prefill_hidden(audio_embeds, &mut cache)?;
        let tok = self.argmax(&hidden)?;
        Ok((cache, tok))
    }

    pub fn decode_stream_eager(&self, audio_embeds: &Tensor) -> Result<Vec<u32>> {
        let n_audio = audio_embeds.dims()[1];
        let prefix = self.prefix_tokens();
        self.audio_tokens_in_range(n_audio)?;
        let eos = self.cfg.text_config.eos_token_id;
        let (mut cache, mut tok) = self.prefill(audio_embeds)?;
        let mut out = Vec::with_capacity(n_audio - prefix.len());
        if tok == eos {
            return Ok(out);
        }
        out.push(tok);
        for pos in prefix.len()..n_audio - 1 {
            let embeds = (self.embed_tokens(&[tok])? + audio_embeds.narrow(1, pos, 1)?)?;
            let hidden = self.decoder_hidden(&embeds, pos, &mut cache)?;
            tok = self.argmax(&hidden)?;
            if tok == eos {
                break;
            }
            out.push(tok);
        }
        Ok(out)
    }

    pub fn transcribe(&self, audio_16k_mono: &[f32]) -> Result<(String, Vec<u32>)> {
        let features = self.features(audio_16k_mono)?;
        let audio_embeds = self.encode(&features)?;
        let tokens = self.decode_stream(&audio_embeds)?;
        Ok((self.vocab.decode(&tokens), tokens))
    }
}

pub const VOXTRAL_CPU_PROFILE_ENV: &str = "NV_VOXTRAL_CPU_PROFILE";
const CPU_PROFILE_OPS: [&str; 14] = [
    "embed", "attn_norm", "qkv", "split", "rope", "kv_write", "attn", "o_proj", "ffn_norm",
    "gate_up", "silu_mul", "down", "final_norm", "lm_head",
];
thread_local! {
    static CPU_PROFILE: std::cell::RefCell<[f64; 14]> = const { std::cell::RefCell::new([0.0; 14]) };
}

pub(crate) fn cpu_profile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(VOXTRAL_CPU_PROFILE_ENV).ok().as_deref() == Some("1"))
}

pub(crate) struct Lap(Option<std::time::Instant>);

impl Lap {
    pub(crate) fn start(device: &Device) -> Self {
        Self((matches!(device, Device::Cpu) && cpu_profile_enabled()).then(std::time::Instant::now))
    }

    pub(crate) fn lap(&mut self, op: usize) {
        if let Some(t) = &mut self.0 {
            let now = std::time::Instant::now();
            CPU_PROFILE.with(|p| p.borrow_mut()[op] += (now - *t).as_secs_f64());
            *t = now;
        }
    }
}

pub(crate) fn cpu_profile_report(label: &str, steps: usize) {
    if !cpu_profile_enabled() {
        return;
    }
    let ms = CPU_PROFILE.with(|p| std::mem::take(&mut *p.borrow_mut()));
    let per: Vec<String> = CPU_PROFILE_OPS
        .iter()
        .zip(&ms)
        .map(|(n, s)| format!("{n} {:.2}", s * 1000.0 / steps.max(1) as f64))
        .collect();
    eprintln!(
        "cpu step profile {label} ({steps} steps, ms/step): {} | total {:.2}",
        per.join(" "),
        ms.iter().sum::<f64>() * 1000.0 / steps.max(1) as f64
    );
}

pub const VOXTRAL_GRAPH_ENV: &str = "NV_VOXTRAL_GRAPH";

pub fn graph_decode_enabled() -> bool {
    std::env::var(VOXTRAL_GRAPH_ENV).ok().as_deref() != Some("0")
}

pub fn text_token_count(tokens: &[u32]) -> usize {
    tokens
        .iter()
        .filter(|&&t| t != STREAMING_PAD_TOKEN && t != STREAMING_WORD_TOKEN && t > EOS_TOKEN)
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn cpu_encoder_stage_profile() {
        let Some(dir) = std::env::var_os("VOXTRAL_MODEL_DIR").map(std::path::PathBuf::from) else {
            return;
        };
        if std::env::var("NV_VOXTRAL_CPU_PROFILE").ok().as_deref() != Some("1") {
            return;
        }
        let m = VoxtralRealtime::load_dir(&dir, &Device::Cpu).unwrap();
        let audio = vec![0.01f32; SAMPLE_RATE * 3];
        let feats = m.features(&audio).unwrap();
        let a = &m.cfg.audio_config;
        let t0 = std::time::Instant::now();
        let x = m
            .enc_conv1
            .forward(&feats.pad_with_zeros(2, 2, 0).unwrap())
            .unwrap()
            .gelu_erf()
            .unwrap();
        let x = m
            .enc_conv2
            .forward(&x.pad_with_zeros(2, 1, 0).unwrap())
            .unwrap()
            .gelu_erf()
            .unwrap();
        let x = x.transpose(1, 2).unwrap().contiguous().unwrap();
        let t1 = std::time::Instant::now();
        let seq = x.dims()[1];
        let layer = &m.enc_layers[0];
        let h = layer.attn_norm.forward(&x).unwrap();
        let t2 = std::time::Instant::now();
        let q = layer.q_proj.forward(&h).unwrap();
        let _k = layer.k_proj.forward(&h).unwrap();
        let _v = layer.v_proj.forward(&h).unwrap();
        let t3 = std::time::Instant::now();
        let positions =
            Tensor::from_vec((0..seq as i32).collect::<Vec<i32>>(), seq, &Device::Cpu).unwrap();
        let q4 = q
            .reshape((1, seq, a.num_attention_heads, a.head_dim))
            .unwrap();
        let k4 = _k
            .reshape((1, seq, a.num_key_value_heads, a.head_dim))
            .unwrap();
        let (qr, kr) = m
            .enc_rope
            .apply(
                &q4.contiguous().unwrap(),
                &k4.contiguous().unwrap(),
                &positions,
            )
            .unwrap();
        let t4 = std::time::Instant::now();
        let cfg = AttnConfig {
            num_heads: a.num_attention_heads,
            num_kv_heads: a.num_key_value_heads,
            head_dim: a.head_dim,
            softmax_scale: 1.0 / (a.head_dim as f32).sqrt(),
            causal: true,
        };
        let v4 = _v
            .reshape((1, seq, a.num_key_value_heads, a.head_dim))
            .unwrap()
            .contiguous()
            .unwrap();
        let att = m.attention(&qr, &kr, &v4, &cfg, a.sliding_window).unwrap();
        let t5 = std::time::Instant::now();
        let o = layer
            .o_proj
            .forward(
                &att.reshape((1, seq, a.num_attention_heads * a.head_dim))
                    .unwrap(),
            )
            .unwrap();
        let t6 = std::time::Instant::now();
        let h2 = layer.ffn_norm.forward(&o).unwrap();
        let _mlp = layer.mlp.forward(&h2).unwrap();
        let t7 = std::time::Instant::now();
        let t8 = std::time::Instant::now();
        let _e = m.encode(&feats).unwrap();
        let t9 = std::time::Instant::now();
        eprintln!(
            "cpu encoder seq {seq}: conv {} ms | norm {} ms | qkv {} ms | rope {} ms | attn {} ms | o {} ms | mlp {} ms | full encode {} ms",
            (t1 - t0).as_millis(),
            (t2 - t1).as_millis(),
            (t3 - t2).as_millis(),
            (t4 - t3).as_millis(),
            (t5 - t4).as_millis(),
            (t6 - t5).as_millis(),
            (t7 - t6).as_millis(),
            (t9 - t8).as_millis()
        );
    }

    #[test]
    #[ignore]
    fn cpu_step_is_bitwise_with_the_candle_eager_path() {
        let Some(dir) = std::env::var_os("VOXTRAL_MODEL_DIR").map(std::path::PathBuf::from) else {
            return;
        };
        let m = VoxtralRealtime::load_dir(&dir, &Device::Cpu).unwrap();
        let audio: Vec<f32> = (0..SAMPLE_RATE * 2)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                0.3 * (2.0 * std::f32::consts::PI * (180.0 + 600.0 * t) * t).sin()
                    + 0.05 * ((i as f32 * 12.9898).sin() * 43758.547).fract()
            })
            .collect();
        let embeds = m.encode(&m.features(&audio).unwrap()).unwrap();
        let n_audio = embeds.dims()[1];
        let hidden = m.cfg.text_config.hidden_size;
        let prefix = m.prefix_tokens();
        let (mut cache, mut tok) = m.prefill(&embeds).unwrap();
        let mut dec = crate::voxtral_cpu::CpuDecoder::new(&m, n_audio + 1).unwrap();
        let mut x = m
            .inputs_embeds_for_stream(&prefix, &embeds)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let normed = dec.step(&mut x, 0).unwrap();
        let mut cpu_tok = dec.argmax(&normed).unwrap();
        let mut mismatched_steps = 0;
        let mut max_abs = 0f32;
        for pos in prefix.len()..n_audio - 1 {
            assert_eq!(cpu_tok, tok, "token at pos {pos}");
            let embeds_step = (m.embed_tokens(&[tok]).unwrap() + embeds.narrow(1, pos, 1).unwrap()).unwrap();
            let reference = m.decoder_hidden(&embeds_step, pos, &mut cache).unwrap();
            tok = m.argmax(&reference).unwrap();
            x = embeds_step.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let normed = dec.step(&mut x, pos).unwrap();
            cpu_tok = dec.argmax(&normed).unwrap();
            let reference = reference.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let diff: Vec<f32> = reference.iter().zip(&normed).map(|(a, b)| (a - b).abs()).collect();
            let worst = diff.iter().cloned().fold(0f32, f32::max);
            if worst > 0.0 {
                mismatched_steps += 1;
                max_abs = max_abs.max(worst);
                eprintln!(
                    "pos {pos}: {} of {hidden} hidden values differ, max abs {worst:e}",
                    diff.iter().filter(|d| **d > 0.0).count()
                );
            }
        }
        eprintln!("cpu step vs candle eager: {mismatched_steps} steps with any hidden-state difference, max abs {max_abs:e}");
        assert_eq!(mismatched_steps, 0, "the CPU step must reproduce the candle eager hidden state bitwise");
    }

    #[test]
    fn hf_name_candidates_cover_the_5_2_checkpoint_and_the_5_17_remap() {
        assert_eq!(
            hf_name_candidates("model.language_model.layers.0.mlp.up_proj.weight"),
            vec![
                "model.language_model.layers.0.mlp.up_proj.weight",
                "language_model.layers.0.mlp.up_proj.weight",
                "language_model.model.layers.0.mlp.up_proj.weight",
            ]
        );
        assert_eq!(
            hf_name_candidates("model.audio_tower.norm.weight"),
            vec!["model.audio_tower.norm.weight", "audio_tower.norm.weight"]
        );
        assert_eq!(hf_name_candidates("lm_head.weight"), vec!["lm_head.weight"]);
    }

    #[test]
    fn streaming_padding_matches_mistral_common_for_the_librispeech_fixture() {
        let padded = pad_audio_for_streaming(&vec![0.5; 57_280], 32, 17);
        assert_eq!(padded.len() % SAMPLES_PER_TOKEN, 0);
        assert_eq!(padded.len() / SAMPLES_PER_TOKEN, 94);
        assert_eq!(padded[32 * SAMPLES_PER_TOKEN], 0.5);
        assert_eq!(padded[32 * SAMPLES_PER_TOKEN - 1], 0.0);
    }

    #[test]
    fn mel_frames_are_one_per_hop_and_bounded_by_the_global_max() {
        let mel = VoxtralMel::new(128);
        let audio: Vec<f32> = (0..SAMPLES_PER_TOKEN * 3)
            .map(|i| (i as f32 * 0.05).sin() * 0.3)
            .collect();
        let (m, frames) = mel.log_mel(&audio);
        assert_eq!(frames, 24);
        assert_eq!(m.len(), 128 * 24);
        let hi = (GLOBAL_LOG_MEL_MAX + 4.0) / 4.0 + 0.5;
        let lo = (GLOBAL_LOG_MEL_MAX - 8.0 + 4.0) / 4.0;
        assert!(m
            .iter()
            .all(|v| v.is_finite() && *v >= lo - 1e-6 && *v < hi));
        assert!(m.iter().any(|v| *v > lo + 1e-3));
    }

    #[test]
    fn banded_mel_filters_are_bitwise_the_full_bin_dot_products() {
        let mel = VoxtralMel::new(128);
        let n_bins = N_FFT / 2 + 1;
        assert!(mel.bands.iter().all(|&(lo, hi)| lo < hi && hi <= n_bins));
        assert!(mel.bands.iter().map(|&(lo, hi)| hi - lo).sum::<usize>() * 8 < 128 * n_bins);
        let audio: Vec<f32> = (0..SAMPLES_PER_TOKEN * 5)
            .map(|i| ((i as f32 * 12.9898).sin() * 43758.547).fract() * 0.4 + (i as f32 * 0.03).sin())
            .collect();
        let (got, frames) = mel.log_mel(&audio);
        let n_frames = audio.len() / HOP_LENGTH;
        assert_eq!(frames, n_frames);
        let pad = N_FFT / 2;
        let mut padded = vec![0.0f32; audio.len() + N_FFT];
        for i in 0..pad {
            padded[i] = audio[pad - i];
        }
        padded[pad..pad + audio.len()].copy_from_slice(&audio);
        for i in 0..pad {
            padded[pad + audio.len() + i] = audio[audio.len() - 2 - i];
        }
        let mut input = mel.fft.make_input_vec();
        let mut output = mel.fft.make_output_vec();
        let floor = GLOBAL_LOG_MEL_MAX - 8.0;
        for frame in 0..n_frames {
            for i in 0..N_FFT {
                input[i] = padded[frame * HOP_LENGTH + i] * mel.hann[i];
            }
            mel.fft.process(&mut input, &mut output).unwrap();
            let power: Vec<f32> = output.iter().map(|c| c.re * c.re + c.im * c.im).collect();
            for m in 0..128 {
                let row = &mel.filters[m * n_bins..(m + 1) * n_bins];
                let sum: f32 = row.iter().zip(&power).map(|(a, b)| a * b).sum();
                let v = sum.max(1e-10).log10().max(floor);
                assert_eq!(got[m * n_frames + frame].to_bits(), ((v + 4.0) / 4.0).to_bits(), "mel {m} frame {frame}");
            }
        }
    }

    #[test]
    fn time_embedding_is_cos_then_sin_of_delay_times_inv_freq() {
        let e = time_embedding(6, 8);
        assert!((e[0] - (6.0f32).cos()).abs() < 1e-6);
        assert!((e[4] - (6.0f32).sin()).abs() < 1e-6);
        let inv1 = (-(10_000f32).ln() / 4.0).exp();
        assert!((e[1] - (6.0 * inv1).cos()).abs() < 1e-6);
        assert!((e[5] - (6.0 * inv1).sin()).abs() < 1e-6);
    }

    #[test]
    fn tekken_decodes_regular_ids_after_the_special_block_and_skips_specials() {
        let raw = r#"{"config":{"default_vocab_size":1003,"default_num_special_tokens":1000},
            "vocab":[{"rank":0,"token_bytes":"SGk="},{"rank":1,"token_bytes":"IHRoZXJl"},{"rank":2,"token_bytes":"IQ=="},{"rank":3,"token_bytes":"eng="}],
            "special_tokens":[],
            "audio":{"transcription_delay_ms":480,"streaming_n_left_pad_tokens":32}}"#;
        let v = TekkenVocab::from_json_str(raw).unwrap();
        assert_eq!(v.num_delay_tokens(), 6);
        assert_eq!(v.right_pad_tokens(), 17);
        assert_eq!(
            v.decode(&[32, 33, 1000, 32, 33, 1001, 1002, 2]),
            "Hi there!"
        );
    }

    #[test]
    fn text_token_count_ignores_pads_boundaries_and_specials() {
        assert_eq!(
            text_token_count(&[32, 32, 33, 5675, 1584, 32, 33, 1636, 2]),
            3
        );
    }
}
