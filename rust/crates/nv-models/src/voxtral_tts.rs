use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, ensure};
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_layers::attn::{AttnConfig, attention};
use nv_layers::linear::Linear;
use nv_layers::norm::RmsNorm;
use nv_layers::rope::{Rope, RopeConfig, RopeKind};
use nv_layers::weight_load::{load_rmsnorm, load_tensor};
use nv_weights::WeightLoader;
use serde::Deserialize;

use crate::qwen3::KvCache;
use crate::voxtral::{
    DECODER_ROPE_ROWS, DecoderLayer, DecoderLinearBuilder, RopeParameters, TIME_EMBEDDING_THETA,
    VoxtralQuant, VoxtralTextConfig, decoder_hidden_over, serving_dtype,
};
use crate::voxtral_tekken::TekkenEncoder;

pub const PARAMS_FILE: &str = "params.json";
pub const TEKKEN_FILE: &str = "tekken.json";
pub const WEIGHTS_FILE: &str = "consolidated.safetensors";
pub const VOICE_DIR: &str = "voice_embedding";
pub const HUB_REPO: &str = "models--mistralai--Voxtral-4B-TTS-2603";
pub const MODEL_DIR_ENV: &str = "VOXTRAL_TTS_MODEL_DIR";

pub const BOS_TOKEN: u32 = 1;
pub const AUDIO_TOKEN: u32 = 24;
pub const BEGIN_AUDIO_TOKEN: u32 = 25;
pub const REPEAT_AUDIO_TEXT_TOKEN: u32 = 35;
pub const NEXT_AUDIO_TEXT_TOKEN: u32 = 36;
pub const EMPTY_AUDIO: u32 = 0;
pub const END_AUDIO: u32 = 1;
pub const N_AUDIO_SPECIAL: u32 = 2;

pub const SAMPLE_RATE: usize = 24_000;
pub const SAMPLES_PER_FRAME: usize = 1920;
pub const DEFAULT_NFE: usize = 7;
pub const DEFAULT_CFG_ALPHA: f32 = 1.2;
pub const MAX_FRAMES_DEFAULT: usize = 2048;

pub const Q_AND_K_ROWS_ARE_PERMUTED_TO_NEOX_ORDER_SO_THE_STT_ROPE_KERNEL_ROTATES_ADJACENT_PAIRS:
    &str = "the checkpoint rotates adjacent pairs (2i, 2i+1) of each head with frequency i; the \
     shared Rope rotates (i, i + 64) with frequency i, so row 2i of a head moves to i and row \
     2i + 1 to i + 64 in wq and wk at load, and the same permutation of q and k leaves every \
     q.k dot product unchanged";

pub const THE_FLOW_HEAD_KEEPS_THE_SOFTMAX_SCALE_IN_ATTENTION_NOT_IN_Q: &str = "1/sqrt(128) is not a power of two, so folding it into bf16 q rows rounds every weight; \
     the three-token flow attention passes the scale to the kernel instead and only the \
     backbone, which reuses the STT step, carries the folded scale";

pub const EULER_AND_FSQ_ARITHMETIC_RUN_ON_THE_HOST_IN_BF16_TO_MATCH_THE_REFERENCE_ROUNDING: &str = "the reference blends alpha * v_cond + (1 - alpha) * v_uncond, scales by dt, adds to x_t, \
     and quantizes round((x + 1) / 2 * 20) with one bf16 rounding per operation; thirty-six \
     values per step cost nothing on the host and the rounding order is then exact on every \
     backend";

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

#[derive(Clone, Copy, Debug)]
pub struct CodebookLayout {
    pub semantic_size: usize,
    pub acoustic_levels: usize,
    pub n_acoustic: usize,
}

impl CodebookLayout {
    pub fn semantic_rows(&self) -> usize {
        self.semantic_size + N_AUDIO_SPECIAL as usize
    }

    pub fn acoustic_rows(&self) -> usize {
        self.acoustic_levels + N_AUDIO_SPECIAL as usize
    }

    pub fn n_codebooks(&self) -> usize {
        1 + self.n_acoustic
    }

    pub fn offset(&self, codebook: usize) -> usize {
        if codebook == 0 {
            0
        } else {
            self.semantic_rows() + (codebook - 1) * self.acoustic_rows()
        }
    }

    pub fn rows_used(&self) -> usize {
        self.offset(self.n_codebooks())
    }
}

fn stack_config(
    dim: usize,
    n_layers: usize,
    head_dim: usize,
    hidden_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    norm_eps: f64,
    rope_theta: f32,
    vocab_size: usize,
) -> VoxtralTextConfig {
    VoxtralTextConfig {
        hidden_size: dim,
        intermediate_size: hidden_dim,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv_heads,
        head_dim,
        num_hidden_layers: n_layers,
        rms_norm_eps: norm_eps,
        sliding_window: None,
        rope_parameters: RopeParameters { rope_theta },
        vocab_size,
        bos_token_id: BOS_TOKEN,
        eos_token_id: 2,
        tie_word_embeddings: true,
    }
}

pub(crate) fn neox_row_permutation(rows: usize, head_dim: usize) -> Vec<u32> {
    let half = head_dim / 2;
    (0..rows)
        .map(|r| {
            let (head, i) = (r / head_dim, r % head_dim);
            let src = if i < half { 2 * i } else { 2 * (i - half) + 1 };
            (head * head_dim + src) as u32
        })
        .collect()
}

pub(crate) struct Manifest {
    pub(crate) backbone_cfg: VoxtralTextConfig,
    pub(crate) flow_cfg: VoxtralTextConfig,
    pub(crate) codebooks: CodebookLayout,
    pub(crate) nfe: usize,
    pub(crate) encoder: TekkenEncoder,
    pub(crate) voices: BTreeMap<String, usize>,
}

pub(crate) fn read_manifest(dir: &Path, nfe: usize) -> Result<Manifest> {
    let read = |f: &str| {
        std::fs::read_to_string(dir.join(f))
            .with_context(|| format!("read {}", dir.join(f).display()))
    };
    let params: Params = serde_json::from_str(&read(PARAMS_FILE)?).context("parse params.json")?;
    ensure!(
        params.model_type == "voxtral_tts",
        "params.json model_type {:?} is not voxtral_tts",
        params.model_type
    );
    let audio = &params.multimodal.audio_model_args;
    let ac = &audio.acoustic_transformer_args;
    let tekken: TekkenJson =
        serde_json::from_str(&read(TEKKEN_FILE)?).context("parse tekken.json")?;
    let n_special = tekken.config.default_num_special_tokens;
    let n_regular = tekken.config.default_vocab_size - n_special;
    let mut token_bytes = vec![Vec::new(); n_regular];
    for e in tekken.vocab.iter().filter(|e| e.rank < n_regular) {
        token_bytes[e.rank] = base64::engine::general_purpose::STANDARD
            .decode(&e.token_bytes)
            .with_context(|| format!("tekken rank {} token_bytes is not base64", e.rank))?;
    }
    Ok(Manifest {
        backbone_cfg: stack_config(
            params.dim,
            params.n_layers,
            params.head_dim,
            params.hidden_dim,
            params.n_heads,
            params.n_kv_heads,
            params.norm_eps,
            params.rope_theta,
            params.vocab_size,
        ),
        flow_cfg: stack_config(
            ac.dim,
            ac.n_layers,
            ac.head_dim,
            ac.hidden_dim,
            ac.n_heads,
            ac.n_kv_heads,
            params.norm_eps,
            params.rope_theta,
            params.vocab_size,
        ),
        codebooks: CodebookLayout {
            semantic_size: audio.semantic_codebook_size,
            acoustic_levels: audio.acoustic_codebook_size,
            n_acoustic: audio.n_acoustic_codebook,
        },
        nfe: ac.n_decoding_steps.unwrap_or(nfe),
        encoder: TekkenEncoder::new(&token_bytes, n_special)?,
        voices: tekken.audio.voice_num_audio_tokens,
    })
}

pub const BF16_LINEARS_KEEP_NO_PRETRANSPOSED_TWIN_BECAUSE_IT_DOUBLES_THE_RESIDENT_WEIGHTS: &str = "Linear::new stores a transposed copy of every bf16 weight for the row-major cuBLAS route; \
     the one-row GEMV, the M-row GEMV and the transposed-B cuBLAS route read the weight as \
     stored, so the TTS builds its bf16 linears with new_no_pretranspose and the resident bytes \
     are the checkpoint's";

struct TtsLinearBuilder {
    quant: VoxtralQuant,
    quantized: DecoderLinearBuilder,
}

impl TtsLinearBuilder {
    fn new(quant: VoxtralQuant, device: &Device) -> Result<Self> {
        Ok(Self {
            quant,
            quantized: DecoderLinearBuilder::new(quant, device)?,
        })
    }

    fn build(&self, weight: Tensor, device: &Device) -> Result<Linear> {
        let _ = BF16_LINEARS_KEEP_NO_PRETRANSPOSED_TWIN_BECAUSE_IT_DOUBLES_THE_RESIDENT_WEIGHTS;
        match self.quant {
            VoxtralQuant::Bf16 => Linear::new_no_pretranspose(weight.contiguous()?, None),
            _ => self.quantized.build(weight, device),
        }
    }
}

fn load_plain_linear(
    w: &WeightLoader,
    name: &str,
    out: usize,
    inp: usize,
    dtype: DType,
) -> Result<Linear> {
    Linear::new_no_pretranspose(load_tensor(w, name, &[out, inp], dtype)?, None)
}

pub const FUSED_WEIGHTS_ARE_PREPARED_ON_THE_HOST_SO_THE_DEVICE_POOL_HOLDS_ONLY_FINAL_WEIGHTS: &str = "the q and k row permutation, the folded q scale and the q|k|v and w1|w3 concatenations run \
     on host tensors over the mmapped file and each fused weight is uploaded once; device-side \
     temporaries interleaved with the layers' weights left every pool chunk partly live, which no \
     trim could return and which doubled the resident bytes of a load";

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_stack(
    w: &WeightLoader,
    host: &WeightLoader,
    prefix: &str,
    t: &VoxtralTextConfig,
    build: &dyn Fn(Tensor) -> Result<Linear>,
    device: &Device,
    dtype: DType,
    neox_q_rows_with_folded_scale: bool,
) -> Result<(Vec<DecoderLayer>, RmsNorm)> {
    let q_rows = t.num_attention_heads * t.head_dim;
    let kv_rows = t.num_key_value_heads * t.head_dim;
    let (d, inter) = (t.hidden_size, t.intermediate_size);
    let mut layers = Vec::with_capacity(t.num_hidden_layers);
    for i in 0..t.num_hidden_layers {
        let p = format!("{prefix}layers.{i}.");
        let _ = FUSED_WEIGHTS_ARE_PREPARED_ON_THE_HOST_SO_THE_DEVICE_POOL_HOLDS_ONLY_FINAL_WEIGHTS;
        let mut q = load_tensor(
            host,
            &format!("{p}attention.wq.weight"),
            &[q_rows, d],
            dtype,
        )?;
        let mut k = load_tensor(
            host,
            &format!("{p}attention.wk.weight"),
            &[kv_rows, d],
            dtype,
        )?;
        if neox_q_rows_with_folded_scale {
            let _ = Q_AND_K_ROWS_ARE_PERMUTED_TO_NEOX_ORDER_SO_THE_STT_ROPE_KERNEL_ROTATES_ADJACENT_PAIRS;
            let perm = |rows: usize| {
                Tensor::new(
                    neox_row_permutation(rows, t.head_dim).as_slice(),
                    &Device::Cpu,
                )
            };
            q = (q.index_select(&perm(q_rows)?, 0)?.to_dtype(DType::F32)?
                * (1.0 / (t.head_dim as f64).sqrt()))?
            .to_dtype(dtype)?;
            k = k.index_select(&perm(kv_rows)?, 0)?;
        }
        let v = load_tensor(
            host,
            &format!("{p}attention.wv.weight"),
            &[kv_rows, d],
            dtype,
        )?;
        let gate_up = Tensor::cat(
            &[
                load_tensor(
                    host,
                    &format!("{p}feed_forward.w1.weight"),
                    &[inter, d],
                    dtype,
                )?,
                load_tensor(
                    host,
                    &format!("{p}feed_forward.w3.weight"),
                    &[inter, d],
                    dtype,
                )?,
            ],
            0,
        )?
        .to_device(device)?;
        let qkv = build(Tensor::cat(&[q, k, v], 0)?.to_device(device)?)?;
        layers.push(DecoderLayer {
            attn_norm: load_rmsnorm(
                w,
                &format!("{p}attention_norm.weight"),
                d,
                t.rms_norm_eps,
                dtype,
            )?,
            qkv_proj: qkv,
            o_proj: build(load_tensor(
                w,
                &format!("{p}attention.wo.weight"),
                &[d, q_rows],
                dtype,
            )?)?,
            ffn_norm: load_rmsnorm(w, &format!("{p}ffn_norm.weight"), d, t.rms_norm_eps, dtype)?,
            gate_up_proj: build(gate_up)?,
            down_proj: build(load_tensor(
                w,
                &format!("{p}feed_forward.w2.weight"),
                &[d, inter],
                dtype,
            )?)?,
        });
    }
    let norm = load_rmsnorm(w, &format!("{prefix}norm.weight"), d, t.rms_norm_eps, dtype)?;
    Ok((layers, norm))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlowQuant {
    Bf16,
    Nvfp4,
    E4m3,
}

pub const THE_E4M3_FLOW_HEAD_KEEPS_ITS_OWN_ROW_SCALED_WEIGHTS_AND_THE_SIX_ROW_E4M3_GEMV: &str = "the fp8 Linear needs cuBLASLt's per-row scaling, which this part refuses at the probe; the \
     flow head's six-row evaluations go through gemv_e4m3_mk over per-row absmax e4m3 weights \
     quantized on the device at load, so an e4m3 flow layer holds no bf16 twin and the three \
     layers read half the bytes of bf16 per evaluation";

#[cfg(feature = "cuda")]
pub(crate) struct E4m3Rows {
    wq: cudarc::driver::CudaSlice<u8>,
    scale: cudarc::driver::CudaSlice<f32>,
    n: usize,
    k: usize,
}

#[cfg(feature = "cuda")]
impl E4m3Rows {
    fn from_linear(lin: &Linear) -> Result<Self> {
        let _ = THE_E4M3_FLOW_HEAD_KEEPS_ITS_OWN_ROW_SCALED_WEIGHTS_AND_THE_SIX_ROW_E4M3_GEMV;
        let w = lin
            .weight()
            .context("an e4m3 flow layer quantizes a bf16 weight")?;
        let (wq, scale) = nv_layers::lm_head_resident_quant::rowquant_e4m3_dev(w)?;
        Ok(Self {
            wq,
            scale,
            n: lin.out_features(),
            k: lin.in_features(),
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct FlowE4m3Layer {
    attn_norm: RmsNorm,
    qkv: E4m3Rows,
    o: E4m3Rows,
    ffn_norm: RmsNorm,
    gate_up: E4m3Rows,
    down: E4m3Rows,
}

#[cfg(feature = "cuda")]
impl FlowE4m3Layer {
    fn from_decoder_layer(l: DecoderLayer) -> Result<Self> {
        Ok(Self {
            qkv: E4m3Rows::from_linear(&l.qkv_proj)?,
            o: E4m3Rows::from_linear(&l.o_proj)?,
            gate_up: E4m3Rows::from_linear(&l.gate_up_proj)?,
            down: E4m3Rows::from_linear(&l.down_proj)?,
            attn_norm: l.attn_norm,
            ffn_norm: l.ffn_norm,
        })
    }
}

pub(crate) struct FlowHead {
    pub(crate) input_proj: Linear,
    pub(crate) llm_proj: Linear,
    pub(crate) layers: Vec<DecoderLayer>,
    #[cfg(feature = "cuda")]
    pub(crate) e4m3: Vec<FlowE4m3Layer>,
    pub(crate) norm: RmsNorm,
    pub(crate) out: Linear,
    pub(crate) nfe: usize,
    pub(crate) cfg_alpha: bf16,
    pub(crate) time_tokens: Tensor,
    pub(crate) dts: Vec<bf16>,
}

impl FlowHead {
    fn is_e4m3(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            !self.e4m3.is_empty()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

pub fn bf16_timesteps(nfe: usize) -> Vec<bf16> {
    (0..=nfe)
        .map(|i| bf16::from_f32(i as f32 / nfe as f32))
        .collect()
}

pub fn sinusoidal_time_embedding(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    for j in 0..half {
        let inv_freq = (-(TIME_EMBEDDING_THETA.ln()) * j as f32 / half as f32).exp();
        out[j] = (t * inv_freq).cos();
        out[half + j] = (t * inv_freq).sin();
    }
    out
}

pub struct SeededNoise(u64);

impl SeededNoise {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn frame(&mut self, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                let u1 = ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
                let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                let g = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                bf16::from_f32(g as f32).to_f32()
            })
            .collect()
    }
}

#[derive(Default, Debug, Clone)]
pub struct FrameTrace {
    pub semantic_logits: Vec<f32>,
    pub v_cond: Vec<Vec<f32>>,
    pub v_uncond: Vec<Vec<f32>>,
    pub x: Vec<Vec<f32>>,
    pub frame_embed: Vec<f32>,
}

#[derive(Default, Debug, Clone)]
pub struct StageMs {
    pub prefill: f64,
    pub backbone: f64,
    pub semantic: f64,
    pub flow: f64,
    pub embed: f64,
    pub graph: f64,
    pub gpu: f64,
    pub steps: usize,
}

#[derive(Default, Debug)]
pub struct Synthesis {
    pub prompt_ids: Vec<u32>,
    pub frames: Vec<Vec<u32>>,
    pub ended: bool,
    pub hidden: BTreeMap<usize, Vec<f32>>,
    pub semantic_top2: Vec<[(u32, f32); 2]>,
    pub frame0: Option<FrameTrace>,
    pub stage_ms: StageMs,
}

impl Synthesis {
    pub fn codec_codes(&self) -> Vec<Vec<u32>> {
        self.frames
            .iter()
            .take_while(|f| f[0] != END_AUDIO)
            .map(|f| f.iter().map(|c| c - N_AUDIO_SPECIAL).collect())
            .collect()
    }
}

pub struct VoxtralTts {
    pub(crate) backbone_cfg: VoxtralTextConfig,
    pub(crate) flow_cfg: VoxtralTextConfig,
    codebooks: CodebookLayout,
    encoder: TekkenEncoder,
    voices: BTreeMap<String, usize>,
    dir: PathBuf,
    embed: Tensor,
    pub(crate) codebook_embed: Tensor,
    pub(crate) layers: Vec<DecoderLayer>,
    pub(crate) norm: RmsNorm,
    pub(crate) rope: Rope,
    pub(crate) semantic_head: Linear,
    pub(crate) flow: FlowHead,
    voice_cache: std::sync::Mutex<BTreeMap<String, Tensor>>,
    #[cfg(feature = "cuda")]
    graph: std::sync::Mutex<Option<crate::voxtral_tts_graph::TtsGraphState>>,
    device: Device,
    dtype: DType,
}

pub const GRAPH_CAPACITY_MIN: usize = 3072;

pub const CUDA_SYNTHESIS_REPLAYS_THE_FRAME_GRAPH_BY_THE_LIBRISPEECH_73_ROW_WORD_ERROR_INSTRUMENT:
    &str = "the corpus instrument scored the graphed frame against the eager loop at 54 and 56 word \
     errors to 48 and 51 over 1150 words at seeds 42 and 43, inside the instrument's own \
     seed-to-seed spread, and both fixtures' graphed free runs transcribe verbatim, so an \
     untraced synthesis on CUDA replays the graph; a traced run and the CPU keep the eager loop";

impl VoxtralTts {
    pub fn default_dir() -> Result<PathBuf> {
        if let Some(d) = std::env::var_os(MODEL_DIR_ENV).map(PathBuf::from) {
            ensure!(
                d.join(PARAMS_FILE).is_file(),
                "{MODEL_DIR_ENV}={} holds no {PARAMS_FILE}",
                d.display()
            );
            return Ok(d);
        }
        let home = std::env::var("HOME").context("HOME unset")?;
        let snaps = PathBuf::from(home)
            .join(".cache/huggingface/hub")
            .join(HUB_REPO)
            .join("snapshots");
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
            .map(|rd| rd.flatten().map(|e| e.path()).collect())
            .unwrap_or_default();
        entries.sort();
        entries
            .into_iter()
            .find(|p| p.join(WEIGHTS_FILE).is_file())
            .ok_or_else(|| {
                anyhow!(
                    "no Voxtral TTS checkpoint under {} and {MODEL_DIR_ENV} unset",
                    snaps.display()
                )
            })
    }

    pub fn load_dir(dir: &Path, device: &Device) -> Result<Self> {
        Self::load_dir_with(
            dir,
            device,
            VoxtralQuant::Bf16,
            FlowQuant::Bf16,
            DEFAULT_NFE,
            DEFAULT_CFG_ALPHA,
        )
    }

    pub fn load_dir_with(
        dir: &Path,
        device: &Device,
        quant: VoxtralQuant,
        flow_quant: FlowQuant,
        nfe: usize,
        cfg_alpha: f32,
    ) -> Result<Self> {
        let Manifest {
            backbone_cfg,
            flow_cfg,
            codebooks,
            nfe,
            encoder,
            voices,
        } = read_manifest(dir, nfe)?;
        let dtype = serving_dtype(device);
        let builder = TtsLinearBuilder::new(quant, device)?;
        let flow_builder = TtsLinearBuilder::new(
            match flow_quant {
                FlowQuant::Nvfp4 => VoxtralQuant::Nvfp4,
                FlowQuant::Bf16 | FlowQuant::E4m3 => VoxtralQuant::Bf16,
            },
            device,
        )?;
        ensure!(
            flow_quant != FlowQuant::E4m3 || device.is_cuda(),
            "an e4m3 flow head runs on CUDA only"
        );
        let w = WeightLoader::open_file(dir.join(WEIGHTS_FILE), device)
            .with_context(|| format!("open {}", dir.join(WEIGHTS_FILE).display()))?;
        let host = WeightLoader::open_file(dir.join(WEIGHTS_FILE), &Device::Cpu)?;
        let d = backbone_cfg.hidden_size;
        let embed = load_tensor(
            &w,
            "mm_audio_embeddings.tok_embeddings.weight",
            &[backbone_cfg.vocab_size, d],
            dtype,
        )?;
        let codebook_embed = w.get(
            "mm_audio_embeddings.audio_codebook_embeddings.embeddings.weight",
            dtype,
        )?;
        ensure!(
            codebook_embed.dims().len() == 2
                && codebook_embed.dims()[1] == d
                && codebook_embed.dims()[0] >= codebooks.rows_used(),
            "audio_codebook_embeddings {:?} holds fewer than the {} rows the 37 codebooks address",
            codebook_embed.dims(),
            codebooks.rows_used()
        );
        let (layers, norm) = load_stack(
            &w,
            &host,
            "",
            &backbone_cfg,
            &|t| builder.build(t, device),
            device,
            dtype,
            true,
        )?;
        let rope = Rope::new(
            RopeConfig {
                head_dim: backbone_cfg.head_dim,
                max_seq_len: DECODER_ROPE_ROWS,
                base: backbone_cfg.rope_parameters.rope_theta,
                kind: RopeKind::Standard,
            },
            device,
        )?;
        let fp = "acoustic_transformer.";
        let semantic_head = load_plain_linear(
            &w,
            &format!("{fp}semantic_codebook_output.weight"),
            codebooks.semantic_rows().div_ceil(128) * 128,
            d,
            dtype,
        )?;
        let (flow_layers, flow_norm) = load_stack(
            &w,
            &host,
            fp,
            &flow_cfg,
            &|t| flow_builder.build(t, device),
            device,
            dtype,
            false,
        )?;
        #[cfg(feature = "cuda")]
        let (flow_layers, e4m3) = match flow_quant {
            FlowQuant::E4m3 => (
                Vec::new(),
                flow_layers
                    .into_iter()
                    .map(FlowE4m3Layer::from_decoder_layer)
                    .collect::<Result<Vec<_>>>()?,
            ),
            FlowQuant::Bf16 | FlowQuant::Nvfp4 => (flow_layers, Vec::new()),
        };
        let time_proj = load_plain_linear(&w, &format!("{fp}time_projection.weight"), d, d, dtype)?;
        let timesteps = bf16_timesteps(nfe);
        let time_rows: Vec<f32> = timesteps[..nfe]
            .iter()
            .flat_map(|t| sinusoidal_time_embedding(t.to_f32(), d))
            .collect();
        let time_tokens =
            time_proj.forward(&Tensor::from_vec(time_rows, (nfe, d), device)?.to_dtype(dtype)?)?;
        let flow = FlowHead {
            input_proj: load_plain_linear(
                &w,
                &format!("{fp}input_projection.weight"),
                d,
                codebooks.n_acoustic,
                dtype,
            )?,
            llm_proj: load_plain_linear(&w, &format!("{fp}llm_projection.weight"), d, d, dtype)?,
            layers: flow_layers,
            #[cfg(feature = "cuda")]
            e4m3,
            norm: flow_norm,
            out: load_plain_linear(
                &w,
                &format!("{fp}acoustic_codebook_output.weight"),
                codebooks.n_acoustic,
                d,
                dtype,
            )?,
            nfe,
            cfg_alpha: bf16::from_f32(cfg_alpha),
            time_tokens,
            dts: timesteps.windows(2).map(|p| p[1] - p[0]).collect(),
        };
        release_the_load_time_pool_slack(device)?;
        Ok(Self {
            backbone_cfg,
            flow_cfg,
            codebooks,
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
            voice_cache: std::sync::Mutex::new(BTreeMap::new()),
            #[cfg(feature = "cuda")]
            graph: std::sync::Mutex::new(None),
            device: device.clone(),
            dtype,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
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

    pub fn voice_embedding(&self, voice: &str) -> Result<Tensor> {
        let n_voice = self.voice_rows(voice)?;
        let mut cache = self
            .voice_cache
            .lock()
            .map_err(|_| anyhow!("voice cache poisoned"))?;
        if let Some(t) = cache.get(voice) {
            return Ok(t.clone());
        }
        let path = self.dir.join(VOICE_DIR).join(format!("{voice}.pt"));
        let t = read_torch_tensor(&path)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        ensure!(
            t.dims() == [n_voice, self.backbone_cfg.hidden_size],
            "voice {voice}: tensor {:?} does not match tekken's {n_voice} audio tokens",
            t.dims()
        );
        cache.insert(voice.to_string(), t.clone());
        Ok(t)
    }

    pub fn prompt_embeds(&self, ids: &[u32], voice: &str) -> Result<Tensor> {
        let voice_t = self.voice_embedding(voice)?;
        let n_voice = voice_t.dims()[0];
        let e = self
            .embed
            .index_select(&Tensor::new(ids, &self.device)?, 0)?;
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
        let tail = ids.len() - first_audio - n_voice;
        Tensor::cat(
            &[
                e.narrow(0, 0, first_audio)?,
                voice_t,
                e.narrow(0, first_audio + n_voice, tail)?,
            ],
            0,
        )?
        .unsqueeze(0)
        .map_err(Into::into)
    }

    pub(crate) fn step(&self, embeds: &Tensor, cache: &mut KvCache) -> Result<Tensor> {
        decoder_hidden_over(
            &self.layers,
            &self.rope,
            &self.norm,
            &self.backbone_cfg,
            &self.device,
            embeds,
            cache.current_len(),
            cache,
        )
    }

    pub fn semantic_logits(&self, h: &Tensor) -> Result<Vec<f32>> {
        Ok(self
            .semantic_head
            .forward(h)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?)
    }

    pub fn semantic_argmax(&self, logits: &[f32]) -> u32 {
        self.semantic_top2(logits)[0].0
    }

    pub fn semantic_top2(&self, logits: &[f32]) -> [(u32, f32); 2] {
        let rows = self.codebooks.semantic_rows();
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

    fn flow_transformer(&self, x: &Tensor) -> Result<Tensor> {
        let t = &self.flow_cfg;
        ensure!(
            !self.flow.is_e4m3(),
            "the e4m3 flow head has no candle path"
        );
        let (b, s, _) = x.dims3()?;
        let q_rows = t.num_attention_heads * t.head_dim;
        let kv_rows = t.num_key_value_heads * t.head_dim;
        let inter = t.intermediate_size;
        let _ = THE_FLOW_HEAD_KEEPS_THE_SOFTMAX_SCALE_IN_ATTENTION_NOT_IN_Q;
        let cfg = AttnConfig {
            num_heads: t.num_attention_heads,
            num_kv_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            softmax_scale: 1.0 / (t.head_dim as f32).sqrt(),
            causal: false,
        };
        let mut x = x.clone();
        for layer in &self.flow.layers {
            let h = layer.attn_norm.forward(&x)?;
            let qkv = layer.qkv_proj.forward(&h)?;
            let q = qkv.narrow(2, 0, q_rows)?.contiguous()?.reshape((
                b,
                s,
                t.num_attention_heads,
                t.head_dim,
            ))?;
            let k = qkv.narrow(2, q_rows, kv_rows)?.contiguous()?.reshape((
                b,
                s,
                t.num_key_value_heads,
                t.head_dim,
            ))?;
            let v = qkv
                .narrow(2, q_rows + kv_rows, kv_rows)?
                .contiguous()?
                .reshape((b, s, t.num_key_value_heads, t.head_dim))?;
            let attn = attention(&q, &k, &v, &cfg)?;
            let attn = layer.o_proj.forward(&attn.reshape((b, s, q_rows))?)?;
            let (h, x_mid) = layer.ffn_norm.forward_residual(&attn, &x)?;
            let gate_up = layer.gate_up_proj.forward(&h)?;
            let gate = gate_up.narrow(2, 0, inter)?;
            let up = gate_up.narrow(2, inter, inter)?;
            let act = candle_nn::ops::silu(&gate.contiguous()?)?.mul(&up.contiguous()?)?;
            x = (&x_mid + layer.down_proj.forward(&act)?)?;
        }
        let first = x.narrow(1, 0, 1)?;
        Ok(self
            .flow
            .norm
            .forward(&first)?
            .reshape((b, t.hidden_size))?)
    }

    fn flow_velocity(&self, tokens: &Tensor) -> Result<(Vec<bf16>, Vec<bf16>)> {
        #[cfg(feature = "cuda")]
        if tokens.device().is_cuda() {
            let v = self
                .flow_velocity_device(tokens.clone())?
                .to_vec2::<bf16>()?;
            return Ok((v[0].clone(), v[FLOW_TOKENS].clone()));
        }
        let v = self
            .flow
            .out
            .forward(&self.flow_transformer(tokens)?)?
            .to_dtype(DType::BF16)?
            .to_vec2::<bf16>()?;
        Ok((v[0].clone(), v[1].clone()))
    }

    pub fn flow_frame(
        &self,
        h: &Tensor,
        x0: &[f32],
        trace: Option<&mut FrameTrace>,
    ) -> Result<Vec<u32>> {
        let _ = EULER_AND_FSQ_ARITHMETIC_RUN_ON_THE_HOST_IN_BF16_TO_MATCH_THE_REFERENCE_ROUNDING;
        let f = &self.flow;
        let n = self.codebooks.n_acoustic;
        ensure!(
            x0.len() == n,
            "noise row has {} values, the flow head needs {n}",
            x0.len()
        );
        let d = self.flow_cfg.hidden_size;
        let mut x: Vec<bf16> = x0.iter().map(|v| bf16::from_f32(*v)).collect();
        #[cfg(feature = "cuda")]
        if trace.is_none() && self.device.is_cuda() {
            let xt = Tensor::from_vec(x, (1, n), &self.device)?;
            self.flow_ode_device(h, &xt)?;
            return Ok(self.fsq_codes(&xt.to_vec2::<bf16>()?[0]));
        }
        let cond = f.llm_proj.forward(&h.reshape((1, d))?)?;
        let cond = Tensor::cat(&[&cond, &cond.zeros_like()?], 0)?;
        let mut trace = trace;
        if let Some(tr) = trace.as_deref_mut() {
            tr.x.push(x.iter().map(|v| v.to_f32()).collect());
        }
        let one_minus_alpha = bf16::ONE - f.cfg_alpha;
        for i in 0..f.nfe {
            let xt = Tensor::from_vec(x.clone(), (1, n), &self.device)?.to_dtype(self.dtype)?;
            let xin = f.input_proj.forward(&xt)?;
            let tp = f.time_tokens.narrow(0, i, 1)?;
            let tokens = Tensor::stack(
                &[
                    Tensor::cat(&[&xin, &xin], 0)?,
                    Tensor::cat(&[&tp, &tp], 0)?,
                    cond.clone(),
                ],
                1,
            )?;
            let (vc, vu) = self.flow_velocity(&tokens)?;
            let (vc, vu) = (&vc, &vu);
            for j in 0..n {
                let vt = f.cfg_alpha * vc[j] + one_minus_alpha * vu[j];
                x[j] = x[j] + vt * f.dts[i];
            }
            if let Some(tr) = trace.as_deref_mut() {
                tr.v_cond.push(vc.iter().map(|v| v.to_f32()).collect());
                tr.v_uncond.push(vu.iter().map(|v| v.to_f32()).collect());
                tr.x.push(x.iter().map(|v| v.to_f32()).collect());
            }
        }
        Ok(self.fsq_codes(&x))
    }

    fn fsq_codes(&self, x: &[bf16]) -> Vec<u32> {
        let levels = bf16::from_f32((self.codebooks.acoustic_levels - 1) as f32);
        x.iter()
            .map(|&v| {
                let c = v.clamp(bf16::NEG_ONE, bf16::ONE);
                let y = (c + bf16::ONE) * bf16::from_f32(0.5) * levels;
                y.to_f32()
                    .round_ties_even()
                    .clamp(0.0, (self.codebooks.acoustic_levels - 1) as f32) as u32
            })
            .collect()
    }

    pub fn frame_embedding(&self, frame: &[u32]) -> Result<Tensor> {
        let cb = self.codebooks;
        ensure!(
            frame.len() == cb.n_codebooks(),
            "frame has {} tokens, not {}",
            frame.len(),
            cb.n_codebooks()
        );
        let ids: Vec<u32> = frame
            .iter()
            .enumerate()
            .map(|(k, &c)| (cb.offset(k) + c as usize) as u32)
            .collect();
        let rows = self
            .codebook_embed
            .index_select(&Tensor::new(ids.as_slice(), &self.device)?, 0)?;
        Ok(rows
            .to_dtype(DType::F32)?
            .sum_keepdim(0)?
            .to_dtype(self.dtype)?
            .unsqueeze(0)?)
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
        self.synthesize_with(text, voice, noise, max_frames, trace, teacher, &mut |_| {
            Ok(true)
        })
    }

    pub fn synthesize_streaming(
        &self,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Synthesis> {
        self.synthesize_with(text, voice, noise, max_frames, false, None, on_frame)
    }

    #[allow(clippy::too_many_arguments)]
    fn synthesize_with(
        &self,
        text: &str,
        voice: &str,
        noise: &mut dyn FnMut(usize) -> Vec<f32>,
        max_frames: usize,
        trace: bool,
        teacher: Option<&[Vec<u32>]>,
        on_frame: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<Synthesis> {
        let t = &self.backbone_cfg;
        let ids = self.prompt_ids(text, voice)?;
        let max_frames = teacher.map_or(max_frames, <[Vec<u32>]>::len);
        #[cfg(feature = "cuda")]
        if !trace && self.device.is_cuda() {
            let _ = CUDA_SYNTHESIS_REPLAYS_THE_FRAME_GRAPH_BY_THE_LIBRISPEECH_73_ROW_WORD_ERROR_INSTRUMENT;
            let need = ids.len() + max_frames + 1;
            let mut slot = self
                .graph
                .lock()
                .map_err(|_| anyhow!("graph state poisoned"))?;
            if slot.as_ref().is_none_or(|g| g.capacity() < need) {
                *slot = None;
                *slot = Some(crate::voxtral_tts_graph::TtsGraphState::new(
                    self,
                    need.max(GRAPH_CAPACITY_MIN),
                )?);
            }
            let state = slot.as_mut().expect("graph state just set");
            return self
                .synthesize_graphed_with(state, text, voice, noise, max_frames, teacher, on_frame);
        }
        let mut cache = KvCache::with_dims(
            t.num_hidden_layers,
            ids.len() + max_frames + 1,
            t.num_key_value_heads,
            t.head_dim,
            &self.device,
            self.dtype,
        )?;
        let mut out = Synthesis {
            prompt_ids: ids.clone(),
            ..Default::default()
        };
        let mut lap = Lap::new(&self.device, trace);
        let mut h = self.step(&self.prompt_embeds(&ids, voice)?, &mut cache)?;
        lap.record(&mut out.stage_ms.prefill)?;
        let cb = self.codebooks;
        for frame_idx in 0..max_frames {
            let logits = self.semantic_logits(&h)?;
            let top2 = self.semantic_top2(&logits);
            let semantic = top2[0].0;
            if trace {
                out.semantic_top2.push(top2);
            }
            lap.record(&mut out.stage_ms.semantic)?;
            let mut frame_trace = (trace && frame_idx == 0).then(FrameTrace::default);
            if let Some(tr) = frame_trace.as_mut() {
                tr.semantic_logits = logits;
            }
            let codes = self.flow_frame(&h, &noise(frame_idx), frame_trace.as_mut())?;
            lap.record(&mut out.stage_ms.flow)?;
            let mut frame = Vec::with_capacity(cb.n_codebooks());
            frame.push(semantic);
            frame.extend(codes.iter().map(|&c| if semantic == END_AUDIO { EMPTY_AUDIO } else { c } + N_AUDIO_SPECIAL));
            let fed = teacher.map_or_else(|| frame.clone(), |f| f[frame_idx].clone());
            let ends = teacher.map_or(semantic == END_AUDIO, |f| frame_idx + 1 == f.len());
            if trace && (frame_idx < 3 || ends) {
                out.hidden.insert(
                    frame_idx,
                    h.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?,
                );
            }
            let go_on = on_frame(&frame)?;
            out.frames.push(frame);
            if ends || !go_on {
                out.ended = semantic == END_AUDIO;
                break;
            }
            let e = self.frame_embedding(&fed)?;
            if let Some(tr) = frame_trace.as_mut() {
                tr.frame_embed = e.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            }
            out.frame0 = out.frame0.take().or(frame_trace);
            lap.record(&mut out.stage_ms.embed)?;
            h = self.step(&e, &mut cache)?;
            lap.record(&mut out.stage_ms.backbone)?;
            out.stage_ms.steps += 1;
        }
        Ok(out)
    }
}

impl VoxtralTts {
    pub fn hidden_after_prompt(&self, text: &str, voice: &str) -> Result<Tensor> {
        let ids = self.prompt_ids(text, voice)?;
        let t = &self.backbone_cfg;
        let mut cache = KvCache::with_dims(
            t.num_hidden_layers,
            ids.len() + 1,
            t.num_key_value_heads,
            t.head_dim,
            &self.device,
            self.dtype,
        )?;
        self.step(&self.prompt_embeds(&ids, voice)?, &mut cache)
    }
}

#[cfg(feature = "cuda")]
mod cuda_flow {
    use std::ffi::c_void;
    use std::sync::Arc;

    use anyhow::{Result, bail, ensure};
    use candle_core::{DType, Device, Tensor};
    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{CudaStream, DevicePtr};
    use half::bf16;
    use nv_kernels::voxtral_tts_flow as flow;
    use nv_kernels::voxtral_tts_frame as frame;
    use nv_layers::linear::Linear;
    use nv_layers::norm::RmsNorm;
    use nv_layers::step_arena;

    use super::{
        E4m3Rows, FLOW_BATCH, FLOW_TOKENS,
        THE_EAGER_AND_GRAPHED_FLOW_HEADS_SHARE_ONE_KERNEL_SET_SO_A_FRAME_DIFFERS_ONLY_BY_THE_BACKBONE_ATTENTION,
        THE_FLOW_ATTENTION_IS_ITS_OWN_KERNEL_BECAUSE_CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_NULL_STREAM,
        VoxtralTts,
    };

    pub const THE_FLOW_GEMM_KEEPS_GEMM_BF16_MK_LANE_ORDER_WITH_FOUR_WEIGHT_LOADS_IN_FLIGHT_AND_FUSES_THE_SILU_AND_THE_RESIDUAL_INTO_ITS_EPILOGUE: &str =
        "voxtral_tts_flow_gemm is gemm_bf16_mk's six-row dot in gemm_bf16_mk's lane order with four \
         weight loads in flight, whose epilogue either stores, or pairs gate row n with up row n to \
         write silu(gate) * up as spec_silu_mul_halves does, or adds the residual as candle's bf16 \
         add does; the norms stay the nv-layers kernels because a per-block norm prologue repeats a \
         latency-bound reduction in every one of the hundreds of blocks and costs more than the \
         launch it saves; an e4m3 weight dequantizes in the same loop with gemv_e4m3_mk's order \
         and its per-row scale in the epilogue";

    pub const THE_EULER_STEP_THE_INPUT_PROJECTION_AND_THE_TOKEN_ASSEMBLY_ARE_ONE_KERNEL_PER_EVALUATION: &str =
        "voxtral_tts_flow_step advances x_t with tts_euler's rounded bf16 arithmetic, projects it \
         with gemv_bf16's lane-strided dot and writes the six token rows, so an evaluation pays \
         one launch where it paid three, and the trajectory is bitwise; the state ping-pongs \
         through arena slots because every block recomputes the step and the last one writes \
         the caller's buffer";

    pub(super) struct Cu {
        dev: Device,
        stream: Arc<CudaStream>,
        cu: *mut c_void,
    }

    pub(super) enum FlowW<'a> {
        Dense(&'a Linear),
        E4m3(&'a E4m3Rows),
    }

    struct FlowLayerRef<'a> {
        attn_norm: &'a RmsNorm,
        ffn_norm: &'a RmsNorm,
        qkv: FlowW<'a>,
        o: FlowW<'a>,
        gate_up: FlowW<'a>,
        down: FlowW<'a>,
    }

    fn tp(t: &Tensor) -> Result<u64> {
        step_arena::device_ptr_for_a_kernel_to_write(t)
    }

    fn arena(dev: &Device, dtype: DType, shape: &[usize]) -> Result<Tensor> {
        step_arena::output_tensor_uninitialised_for_a_kernel_to_fill(dev, dtype, shape)
    }

    fn rc(code: i32, what: &str) -> Result<()> {
        ensure!(code == 0, "{what} returned {code}");
        Ok(())
    }

    impl VoxtralTts {
        pub(super) fn cu(&self) -> Result<Cu> {
            let Device::Cuda(cuda) = &self.device else {
                bail!("the flow head kernels run on CUDA")
            };
            let stream = nv_layers::cuda_stream::current_stream(cuda);
            Ok(Cu {
                dev: self.device.clone(),
                cu: stream.cu_stream() as *mut c_void,
                stream,
            })
        }

        fn flow_layer_refs(&self) -> Vec<FlowLayerRef<'_>> {
            if self.flow.e4m3.is_empty() {
                self.flow
                    .layers
                    .iter()
                    .map(|l| FlowLayerRef {
                        attn_norm: &l.attn_norm,
                        ffn_norm: &l.ffn_norm,
                        qkv: FlowW::Dense(&l.qkv_proj),
                        o: FlowW::Dense(&l.o_proj),
                        gate_up: FlowW::Dense(&l.gate_up_proj),
                        down: FlowW::Dense(&l.down_proj),
                    })
                    .collect()
            } else {
                self.flow
                    .e4m3
                    .iter()
                    .map(|l| FlowLayerRef {
                        attn_norm: &l.attn_norm,
                        ffn_norm: &l.ffn_norm,
                        qkv: FlowW::E4m3(&l.qkv),
                        o: FlowW::E4m3(&l.o),
                        gate_up: FlowW::E4m3(&l.gate_up),
                        down: FlowW::E4m3(&l.down),
                    })
                    .collect()
            }
        }

        fn flow_gemm(
            &self,
            c: &Cu,
            epilogue: i32,
            w: FlowW,
            x: &Tensor,
            epi_aux: Option<&Tensor>,
        ) -> Result<Tensor> {
            let _ = THE_FLOW_GEMM_KEEPS_GEMM_BF16_MK_LANE_ORDER_WITH_FOUR_WEIGHT_LOADS_IN_FLIGHT_AND_FUSES_THE_SILU_AND_THE_RESIDUAL_INTO_ITS_EPILOGUE;
            let (m, cols) = x.dims2()?;
            let x = x.contiguous()?;
            let (wp, sp, n, k, fp8) = match w {
                FlowW::Dense(lin) => {
                    let wt = lin.weight().expect("dense flow weight");
                    (tp(wt)?, 0u64, lin.out_features(), lin.in_features(), 0)
                }
                FlowW::E4m3(r) => (
                    r.wq.device_ptr(&c.stream).0,
                    r.scale.device_ptr(&c.stream).0,
                    r.n,
                    r.k,
                    1,
                ),
            };
            ensure!(
                m == flow::FLOW_ROWS && cols == k,
                "flow gemm: activations {:?}, expected [{}, {k}]",
                x.dims(),
                flow::FLOW_ROWS
            );
            let n_out = if epilogue == flow::EPILOGUE_SILU_OF_PAIRED_GATE_AND_UP_ROWS {
                n / 2
            } else {
                n
            };
            let out = arena(&c.dev, DType::BF16, &[m, n_out])?;
            let code = unsafe {
                flow::nv_kernels_voxtral_tts_flow_gemm(
                    c.cu,
                    epilogue,
                    fp8,
                    wp as *const c_void,
                    sp as *const f32,
                    tp(&x)? as *const u16,
                    epi_aux.map_or(Ok(0u64), tp)? as *const u16,
                    tp(&out)? as *mut u16,
                    n as i32,
                    k as i32,
                )
            };
            rc(code, "voxtral_tts_flow_gemm")?;
            Ok(out)
        }

        pub(crate) fn flow_velocity_device(&self, tokens: Tensor) -> Result<Tensor> {
            let _ = THE_FLOW_ATTENTION_IS_ITS_OWN_KERNEL_BECAUSE_CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_NULL_STREAM;
            let _ = THE_EAGER_AND_GRAPHED_FLOW_HEADS_SHARE_ONE_KERNEL_SET_SO_A_FRAME_DIFFERS_ONLY_BY_THE_BACKBONE_ATTENTION;
            let c = self.cu()?;
            let t = &self.flow_cfg;
            let (nh, nkv, hd, d) = (
                t.num_attention_heads,
                t.num_key_value_heads,
                t.head_dim,
                t.hidden_size,
            );
            let rows = flow::FLOW_ROWS;
            let store = flow::EPILOGUE_STORE;
            let mut x = tokens.reshape((rows, d))?;
            for layer in self.flow_layer_refs() {
                let h = layer.attn_norm.forward(&x)?;
                let qkv = self.flow_gemm(&c, store, layer.qkv, &h, None)?;
                let attn = arena(&c.dev, DType::BF16, &[rows, nh * hd])?;
                let code = unsafe {
                    frame::nv_kernels_voxtral_tts_flow_attn(
                        c.cu,
                        tp(&qkv)? as *const u16,
                        tp(&attn)? as *mut u16,
                        FLOW_BATCH as i32,
                        FLOW_TOKENS as i32,
                        nh as i32,
                        nkv as i32,
                        hd as i32,
                        1.0 / (hd as f32).sqrt(),
                    )
                };
                rc(code, "tts_flow_attn")?;
                let o = self.flow_gemm(&c, store, layer.o, &attn, None)?;
                let (h, x_mid) = layer.ffn_norm.forward_residual(&o, &x)?;
                let act = self.flow_gemm(
                    &c,
                    flow::EPILOGUE_SILU_OF_PAIRED_GATE_AND_UP_ROWS,
                    layer.gate_up,
                    &h,
                    None,
                )?;
                x = self.flow_gemm(
                    &c,
                    flow::EPILOGUE_ADD_RESIDUAL,
                    layer.down,
                    &act,
                    Some(&x_mid),
                )?;
            }
            self.flow_gemm(
                &c,
                store,
                FlowW::Dense(&self.flow.out),
                &self.flow.norm.forward(&x)?,
                None,
            )
        }

        pub(crate) fn flow_ode_device(&self, h: &Tensor, x: &Tensor) -> Result<()> {
            let _ = THE_EULER_STEP_THE_INPUT_PROJECTION_AND_THE_TOKEN_ASSEMBLY_ARE_ONE_KERNEL_PER_EVALUATION;
            let c = self.cu()?;
            let f = &self.flow;
            let d = self.flow_cfg.hidden_size;
            let n = self.codebooks.n_acoustic;
            ensure!(
                x.dims() == [1, n] && x.dtype() == DType::BF16,
                "flow state {:?}: expected bf16 [1, {n}]",
                x.dims()
            );
            let cond = f.llm_proj.forward(&h.reshape((1, d))?)?.contiguous()?;
            let w_in = f.input_proj.weight().expect("input_projection stays bf16");
            let alpha = f.cfg_alpha.to_f32();
            let one_minus_alpha = (bf16::ONE - f.cfg_alpha).to_f32();
            let step = |x_in: &Tensor,
                        v: Option<&Tensor>,
                        dt: f32,
                        x_out: Option<&Tensor>,
                        tokens: Option<&Tensor>,
                        tok: Option<&Tensor>|
             -> Result<()> {
                let opt = |t: Option<&Tensor>| t.map_or(Ok(0u64), tp);
                let code = unsafe {
                    flow::nv_kernels_voxtral_tts_flow_step(
                        c.cu,
                        tp(x_in)? as *const u16,
                        opt(v)? as *const u16,
                        (FLOW_TOKENS * n) as i32,
                        alpha,
                        one_minus_alpha,
                        dt,
                        opt(x_out)? as *mut u16,
                        tp(w_in)? as *const u16,
                        opt(tok)? as *const u16,
                        tp(&cond)? as *const u16,
                        opt(tokens)? as *mut u16,
                        n as i32,
                        d as i32,
                    )
                };
                rc(code, "voxtral_tts_flow_step")
            };
            let mut cur = x.clone();
            let mut last_v: Option<Tensor> = None;
            for i in 0..f.nfe {
                let tokens = arena(&c.dev, DType::BF16, &[FLOW_BATCH, FLOW_TOKENS, d])?;
                let tok = f.time_tokens.narrow(0, i, 1)?;
                let next = match &last_v {
                    None => None,
                    Some(_) => Some(arena(&c.dev, DType::BF16, &[1, n])?),
                };
                let dt = if i == 0 { 0.0 } else { f.dts[i - 1].to_f32() };
                step(
                    &cur,
                    last_v.as_ref(),
                    dt,
                    next.as_ref(),
                    Some(&tokens),
                    Some(&tok),
                )?;
                cur = next.unwrap_or(cur);
                last_v = Some(self.flow_velocity_device(tokens)?.contiguous()?);
            }
            step(
                &cur,
                last_v.as_ref(),
                f.dts[f.nfe - 1].to_f32(),
                Some(x),
                None,
                None,
            )
        }

        pub fn flow_ode_graphed_ms(
            &self,
            h: &Tensor,
            x0: &[f32],
            replays: usize,
        ) -> Result<(Vec<f64>, Vec<u16>)> {
            let c = self.cu()?;
            let Device::Cuda(cuda) = &self.device else {
                bail!("the flow head kernels run on CUDA")
            };
            let ctx = cuda.cuda_stream().context().clone();
            if ctx.is_event_tracking() {
                unsafe { ctx.disable_event_tracking() };
            }
            let n = self.codebooks.n_acoustic;
            let x = Tensor::zeros((1, n), DType::BF16, &self.device)?;
            let h = h.reshape((1, self.flow_cfg.hidden_size))?.contiguous()?;
            let x0: Vec<bf16> = x0.iter().map(|v| bf16::from_f32(*v)).collect();
            let reset = || -> Result<()> {
                c.stream.synchronize()?;
                unsafe { cudarc::driver::result::memcpy_htod_sync(tp(&x)?, &x0)? };
                Ok(())
            };
            let body = |m: &Self| -> Result<()> { m.flow_ode_device(&h, &x) };
            let mut arena = Some(step_arena::StepArena::new());
            let mut runner = nv_kernels::graph::CudaGraphRunner::new(c.stream.clone());
            let mut run = |eager: bool| -> Result<()> {
                step_arena::install(arena.take().expect("arena is parked between runs"));
                let _htod_cache = cuda.enable_cuda_graph_htod_cache();
                let r = if eager {
                    step_arena::begin_step();
                    body(self)
                } else {
                    runner.run(1, |_s| {
                        step_arena::begin_step();
                        body(self)
                    })
                };
                arena = step_arena::uninstall();
                r
            };
            reset()?;
            run(true)?;
            reset()?;
            run(false)?;
            let mut out = Vec::with_capacity(replays);
            for _ in 0..replays {
                reset()?;
                let e0 = c
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                run(false)?;
                let e1 = c
                    .stream
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                c.stream.synchronize()?;
                out.push(e0.elapsed_ms(&e1)? as f64);
            }
            let bits = x.to_vec2::<bf16>()?[0]
                .iter()
                .map(|v| v.to_bits())
                .collect();
            Ok((out, bits))
        }
    }
}

pub const FLOW_TOKENS: usize = 3;
pub const FLOW_BATCH: usize = 2;

pub const THE_FLOW_ATTENTION_IS_ITS_OWN_KERNEL_BECAUSE_CANDLE_FLASH_ATTN_LAUNCHES_ON_THE_NULL_STREAM: &str =
    "a capture on the device stream cannot see a kernel launched on the legacy null stream, so \
     the three-token bidirectional attention of the flow head runs tts_flow_attn (one block per \
     batch row and head, f32 scores and softmax) inside the graph";

pub const THE_EAGER_AND_GRAPHED_FLOW_HEADS_SHARE_ONE_KERNEL_SET_SO_A_FRAME_DIFFERS_ONLY_BY_THE_BACKBONE_ATTENTION: &str =
    "flow_ode_device is the flow head for both paths: the six-row GEMMs, the three-token attention \
     and the fused Euler-projection step are the same kernels in the same order whether launched \
     eagerly or captured, so the acoustic trajectory of a frame is bitwise between the paths given \
     the same h, and the paths diverge only where the backbone's decode attention rounds differently \
     from flash-attn at a semantic knife edge";

pub const THE_STREAM_ORDERED_POOL_KEEPS_THE_LOAD_TIME_TEMPORARIES_UNTIL_TRIMMED: &str = "the q and k permutation, the f32 scale fold and the qkv and gate-up concatenations free \
     their temporaries back to the device's memory pool, which keeps the reservation; trimming \
     the pool after load hands those bytes back and the pool regrows only to what a frame needs";

#[cfg(feature = "cuda")]
fn release_the_load_time_pool_slack(device: &Device) -> Result<()> {
    let _ = THE_STREAM_ORDERED_POOL_KEEPS_THE_LOAD_TIME_TEMPORARIES_UNTIL_TRIMMED;
    let Device::Cuda(dev) = device else {
        return Ok(());
    };
    device.synchronize()?;
    use candle_core::cuda::cudarc::driver::result;
    unsafe {
        let pool = result::device::get_mem_pool(dev.cuda_stream().context().cu_device())?;
        result::mem_pool::trim_to(pool, 0)?;
    }
    Ok(())
}

#[cfg(not(feature = "cuda"))]
fn release_the_load_time_pool_slack(_device: &Device) -> Result<()> {
    Ok(())
}

struct Lap(Option<(Device, std::time::Instant)>);

impl Lap {
    fn new(device: &Device, on: bool) -> Self {
        Self(on.then(|| (device.clone(), std::time::Instant::now())))
    }

    fn record(&mut self, slot: &mut f64) -> Result<()> {
        if let Some((device, t)) = &mut self.0 {
            device.synchronize()?;
            let now = std::time::Instant::now();
            *slot += (now - *t).as_secs_f64() * 1000.0;
            *t = now;
        }
        Ok(())
    }
}

fn le_u16(b: &[u8], at: usize) -> usize {
    u16::from_le_bytes([b[at], b[at + 1]]) as usize
}

fn le_u32(b: &[u8], at: usize) -> usize {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
}

pub fn stored_zip_entries(bytes: &[u8]) -> Result<BTreeMap<String, (usize, usize)>> {
    let eocd = (0..bytes.len().saturating_sub(21))
        .rev()
        .find(|&i| bytes[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
        .ok_or_else(|| anyhow!("not a zip archive: no end-of-central-directory record"))?;
    let n = le_u16(bytes, eocd + 10);
    let mut at = le_u32(bytes, eocd + 16);
    let mut out = BTreeMap::new();
    for _ in 0..n {
        ensure!(
            bytes[at..at + 4] == [0x50, 0x4b, 0x01, 0x02],
            "zip central directory corrupt"
        );
        let (method, size) = (le_u16(bytes, at + 10), le_u32(bytes, at + 24));
        let (nlen, elen, clen) = (
            le_u16(bytes, at + 28),
            le_u16(bytes, at + 30),
            le_u16(bytes, at + 32),
        );
        let local = le_u32(bytes, at + 42);
        let name = String::from_utf8_lossy(&bytes[at + 46..at + 46 + nlen]).into_owned();
        ensure!(
            method == 0 && size != 0xFFFF_FFFF,
            "zip entry {name}: only stored, non-zip64 entries are read"
        );
        let data = local + 30 + le_u16(bytes, local + 26) + le_u16(bytes, local + 28);
        out.insert(name, (data, size));
        at += 46 + nlen + elen + clen;
    }
    Ok(out)
}

pub fn read_torch_tensor(path: &Path) -> Result<Tensor> {
    use candle_core::pickle::{Object, Stack};
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let entries = stored_zip_entries(&bytes)?;
    let (pkl_name, &(pkl_at, pkl_len)) = entries
        .iter()
        .find(|(k, _)| k.ends_with("data.pkl"))
        .ok_or_else(|| anyhow!("{}: no data.pkl", path.display()))?;
    let mut stack = Stack::empty();
    let mut reader: &[u8] = &bytes[pkl_at..pkl_at + pkl_len];
    stack.read_loop(&mut reader)?;
    let dir = PathBuf::from(pkl_name.strip_suffix(".pkl").expect("ends with data.pkl"));
    let info = stack
        .finalize()?
        .into_tensor_info(Object::Unicode("tensor".into()), &dir)?
        .ok_or_else(|| anyhow!("{}: pickle does not rebuild a tensor", path.display()))?;
    ensure!(
        info.layout.is_contiguous() && info.layout.start_offset() == 0,
        "{}: only contiguous tensors at storage offset 0 are read",
        path.display()
    );
    let &(at, len) = entries
        .get(&info.path)
        .ok_or_else(|| anyhow!("{}: storage {} missing", path.display(), info.path))?;
    let dims = info.layout.shape().dims().to_vec();
    let want = dims.iter().product::<usize>() * info.dtype.size_in_bytes();
    ensure!(
        len == want,
        "{}: storage holds {len} bytes, shape {dims:?} needs {want}",
        path.display()
    );
    Ok(Tensor::from_raw_buffer(
        &bytes[at..at + len],
        info.dtype,
        &dims,
        &Device::Cpu,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neox_permutation_moves_even_rows_to_the_first_half_and_odd_rows_to_the_second() {
        let p = neox_row_permutation(8, 4);
        assert_eq!(p, vec![0, 2, 1, 3, 4, 6, 5, 7]);
    }

    #[test]
    fn codebook_offsets_are_cumulative_with_two_specials_per_codebook() {
        let cb = CodebookLayout {
            semantic_size: 8192,
            acoustic_levels: 21,
            n_acoustic: 36,
        };
        assert_eq!(cb.offset(0), 0);
        assert_eq!(cb.offset(1), 8194);
        assert_eq!(cb.offset(2), 8194 + 23);
        assert_eq!(cb.rows_used(), 9022);
    }

    #[test]
    fn bf16_timesteps_for_seven_steps_match_torch_linspace_cast() {
        let t: Vec<f32> = bf16_timesteps(7).iter().map(|v| v.to_f32()).collect();
        assert_eq!(
            t,
            vec![
                0.0,
                0.142578125,
                0.28515625,
                0.427734375,
                0.5703125,
                0.71484375,
                0.85546875,
                1.0
            ]
        );
    }

    #[test]
    fn seeded_noise_is_deterministic_and_bf16_valued() {
        let a = SeededNoise::new(42).frame(36);
        let b = SeededNoise::new(42).frame(36);
        assert_eq!(a, b);
        assert!(a.iter().all(|v| bf16::from_f32(*v).to_f32() == *v));
        assert!(a.iter().any(|v| v.abs() > 0.5));
    }
}
