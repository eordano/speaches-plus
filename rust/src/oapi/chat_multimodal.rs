use std::io::Cursor;
use std::path::Path;

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use candle_core::{DType, Device, IndexOp, Tensor};
use serde::Deserialize;

use nv_models::gemma4_audio::{
    Gemma4AudioTower, GEMMA4_AUDIO_FRAME_LENGTH, GEMMA4_AUDIO_HOP_LENGTH, GEMMA4_AUDIO_MEL_BINS,
    GEMMA4_AUDIO_SAMPLE_RATE, GEMMA4_AUDIO_SEQ_LENGTH,
};
use nv_models::gemma4_mm_splice::{
    audio_num_soft_tokens, expand_audio_placeholder, expand_image_placeholder,
    splice_mm_embeddings, MmItem, Modality,
};
use nv_models::gemma4_vision::{Gemma4VisionConfig, Gemma4VisionTower};
use nv_weights::WeightLoader;

const FFT_LENGTH: usize = 512;
const MEL_FLOOR: f32 = 1.0e-3;
const MEL_MAX_FREQUENCY: f32 = 8000.0;
const MAX_AUDIO_SAMPLES: usize = GEMMA4_AUDIO_SAMPLE_RATE * 30;

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum ImageUrlSpec {
    Url(String),
    Obj {
        url: String,
        #[serde(default)]
        detail: Option<String>,
    },
}

impl ImageUrlSpec {
    pub fn url(&self) -> &str {
        match self {
            ImageUrlSpec::Url(u) => u,
            ImageUrlSpec::Obj { url, .. } => url,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct InputAudioSpec {
    pub data: String,
    pub format: String,
}

#[derive(Clone, Debug)]
pub enum MmContentPart {
    Text(String),
    ImageUrl(ImageUrlSpec),
    InputAudio(InputAudioSpec),
}

#[derive(Clone, Debug)]
pub struct MmMessage {
    pub role: String,
    pub parts: Vec<MmContentPart>,
}

fn parse_part(p: &serde_json::Value) -> Result<MmContentPart> {
    let ty = p
        .get("type")
        .and_then(|t| t.as_str())
        .context("content part missing \"type\"")?;
    match ty {
        "text" => {
            let text = p
                .get("text")
                .and_then(|t| t.as_str())
                .context("text part missing \"text\"")?;
            Ok(MmContentPart::Text(text.to_string()))
        }
        "image_url" => {
            let raw = p
                .get("image_url")
                .context("image_url part missing \"image_url\"")?;
            let spec: ImageUrlSpec = serde_json::from_value(raw.clone())
                .context("image_url must be a URL string or {\"url\": ...}")?;
            Ok(MmContentPart::ImageUrl(spec))
        }
        "input_audio" => {
            let raw = p
                .get("input_audio")
                .context("input_audio part missing \"input_audio\"")?;
            let spec: InputAudioSpec = serde_json::from_value(raw.clone())
                .context("input_audio must be {\"data\": <base64>, \"format\": ...}")?;
            Ok(MmContentPart::InputAudio(spec))
        }
        other => bail!(
            "unsupported content part type {other:?} (supported: text, image_url, input_audio)"
        ),
    }
}

pub fn parse_mm_messages(messages: &serde_json::Value) -> Result<Vec<MmMessage>> {
    let arr = messages.as_array().context("messages must be an array")?;
    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let role = m
            .get("role")
            .and_then(|r| r.as_str())
            .with_context(|| format!("messages[{i}] missing role"))?
            .to_string();
        let parts = match m.get("content") {
            None | Some(serde_json::Value::Null) => Vec::new(),
            Some(serde_json::Value::String(s)) => vec![MmContentPart::Text(s.clone())],
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .map(parse_part)
                .collect::<Result<Vec<_>>>()
                .with_context(|| format!("messages[{i}].content"))?,
            Some(_) => bail!("messages[{i}].content must be a string or an array of parts"),
        };
        out.push(MmMessage { role, parts });
    }
    Ok(out)
}

pub fn messages_have_mm_parts(messages: &serde_json::Value) -> bool {
    let Some(arr) = messages.as_array() else {
        return false;
    };
    arr.iter().any(|m| {
        m.get("content")
            .and_then(|c| c.as_array())
            .map(|parts| {
                parts.iter().any(|p| {
                    matches!(
                        p.get("type").and_then(|t| t.as_str()),
                        Some("image_url") | Some("input_audio")
                    )
                })
            })
            .unwrap_or(false)
    })
}

pub fn decode_data_url_bytes(url: &str) -> Result<Vec<u8>> {
    let Some(rest) = url.strip_prefix("data:") else {
        bail!(
            "image_url must be an inline data: URL (data:<mime>;base64,<payload>); remote URLs \
             and local file references are not accepted"
        );
    };
    let (meta, payload) = rest
        .split_once(',')
        .context("malformed data: URL (no comma)")?;
    if !meta.ends_with(";base64") {
        bail!("data: URL must be base64-encoded (got {meta:?})");
    }
    B64.decode(payload.trim()).context("decode base64 image data")
}

pub fn decode_b64(data: &str) -> Result<Vec<u8>> {
    B64.decode(data.trim()).context("decode base64 data")
}

pub fn decode_image_ref(url: &str) -> Result<image::RgbImage> {
    let bytes = decode_data_url_bytes(url)?;
    nv_imgdec::decode_rgb8(&bytes).context("decode image bytes")
}

fn resample_linear(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate || samples.is_empty() {
        return samples.to_vec();
    }
    let out_len = ((samples.len() as u64 * to_rate as u64) / from_rate as u64).max(1) as usize;
    let ratio = from_rate as f64 / to_rate as f64;
    (0..out_len)
        .map(|i| {
            let src = i as f64 * ratio;
            let i0 = src.floor() as usize;
            let frac = (src - i0 as f64) as f32;
            let a = samples[i0.min(samples.len() - 1)];
            let b = samples[(i0 + 1).min(samples.len() - 1)];
            a + (b - a) * frac
        })
        .collect()
}

pub fn decode_audio_input(spec: &InputAudioSpec) -> Result<Vec<f32>> {
    if spec.format != "wav" {
        bail!(
            "unsupported input_audio format {:?} (supported: wav)",
            spec.format
        );
    }
    let bytes = B64
        .decode(spec.data.trim())
        .context("decode base64 audio data")?;
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).context("parse WAV audio")?;
    let hspec = reader.spec();
    let channels = hspec.channels.max(1) as usize;
    let mono: Vec<f32> = match hspec.sample_format {
        hound::SampleFormat::Float => {
            let all: Vec<f32> = reader
                .samples::<f32>()
                .collect::<std::result::Result<_, _>>()
                .context("read WAV float samples")?;
            all.chunks(channels)
                .map(|c| c.iter().sum::<f32>() / channels as f32)
                .collect()
        }
        hound::SampleFormat::Int => {
            let scale = 1.0f32 / (1i64 << (hspec.bits_per_sample - 1)) as f32;
            let all: Vec<i32> = reader
                .samples::<i32>()
                .collect::<std::result::Result<_, _>>()
                .context("read WAV int samples")?;
            all.chunks(channels)
                .map(|c| c.iter().map(|&s| s as f32 * scale).sum::<f32>() / channels as f32)
                .collect()
        }
    };
    if mono.is_empty() {
        bail!("WAV audio contains no samples");
    }
    Ok(resample_linear(
        &mono,
        hspec.sample_rate,
        GEMMA4_AUDIO_SAMPLE_RATE as u32,
    ))
}

#[derive(Debug)]
pub struct ImagePatches {
    pub pixel_values: Tensor,
    pub position_ids: Tensor,
    pub host_pixels: Vec<f32>,
    pub grid: (usize, usize),
    pub num_soft_tokens: usize,
    pub target_width: usize,
    pub target_height: usize,
}

pub fn preprocess_image(
    img: &image::RgbImage,
    cfg: &Gemma4VisionConfig,
    device: &Device,
) -> Result<ImagePatches> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    if w == 0 || h == 0 {
        bail!("image has zero width or height");
    }
    let (tw, th) = cfg.target_resolution(w, h, None);
    let num_soft = cfg.compute_num_soft_tokens(w, h, None);
    let resized = image::imageops::resize(
        img,
        tw as u32,
        th as u32,
        image::imageops::FilterType::CatmullRom,
    );
    let patch = cfg.patch_size;
    let grid_w = tw / patch;
    let grid_h = th / patch;
    let n = grid_w * grid_h;
    let pp = cfg.patch_pixels();
    let mut pixels = Vec::with_capacity(n * pp);
    let mut positions = Vec::with_capacity(n * 2);
    for py in 0..grid_h {
        for px in 0..grid_w {
            positions.push(px as i64);
            positions.push(py as i64);
            for dy in 0..patch {
                for dx in 0..patch {
                    let p = resized.get_pixel((px * patch + dx) as u32, (py * patch + dy) as u32);
                    for c in 0..3 {
                        pixels.push(p.0[c] as f32 / 255.0);
                    }
                }
            }
        }
    }
    Ok(ImagePatches {
        pixel_values: Tensor::from_slice(&pixels, (n, pp), device)?,
        position_ids: Tensor::from_vec(positions, (n, 2), device)?,
        host_pixels: pixels,
        grid: (grid_w, grid_h),
        num_soft_tokens: num_soft,
        target_width: tw,
        target_height: th,
    })
}

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10f32.powf(mel / 2595.0) - 1.0)
}

fn mel_filterbank() -> Vec<Vec<f32>> {
    let bins = FFT_LENGTH / 2 + 1;
    let mel_lo = hz_to_mel(0.0);
    let mel_hi = hz_to_mel(MEL_MAX_FREQUENCY);
    let edges: Vec<f32> = (0..GEMMA4_AUDIO_MEL_BINS + 2)
        .map(|i| {
            mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f32 / (GEMMA4_AUDIO_MEL_BINS + 1) as f32)
        })
        .collect();
    let bin_hz = GEMMA4_AUDIO_SAMPLE_RATE as f32 / FFT_LENGTH as f32;
    (0..GEMMA4_AUDIO_MEL_BINS)
        .map(|m| {
            let (left, center, right) = (edges[m], edges[m + 1], edges[m + 2]);
            (0..bins)
                .map(|k| {
                    let f = k as f32 * bin_hz;
                    let up = if center > left {
                        (f - left) / (center - left)
                    } else {
                        0.0
                    };
                    let down = if right > center {
                        (right - f) / (right - center)
                    } else {
                        0.0
                    };
                    up.min(down).max(0.0)
                })
                .collect()
        })
        .collect()
}

pub fn gemma4_log_mel(samples: &[f32]) -> (Vec<f32>, usize) {
    let frame = GEMMA4_AUDIO_FRAME_LENGTH;
    let hop = GEMMA4_AUDIO_HOP_LENGTH;
    let unfold = frame + 1;
    let pad_left = frame / 2;
    let mut padded = vec![0f32; pad_left + samples.len()];
    padded[pad_left..].copy_from_slice(samples);
    if padded.len() < unfold {
        return (Vec::new(), 0);
    }
    let num_frames = (padded.len() - unfold) / hop + 1;
    let window: Vec<f32> = (0..frame)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (frame - 1) as f32).cos()))
        .collect();
    let fbank = mel_filterbank();
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(FFT_LENGTH);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let mut mel = Vec::with_capacity(num_frames * GEMMA4_AUDIO_MEL_BINS);
    for t in 0..num_frames {
        let start = t * hop;
        input.iter_mut().for_each(|x| *x = 0.0);
        for i in 0..frame {
            input[i] = padded[start + i] * window[i];
        }
        fft.process(&mut input, &mut output).expect("fft process");
        let mag: Vec<f32> = output
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect();
        for taps in &fbank {
            let e: f32 = taps.iter().zip(mag.iter()).map(|(w, m)| w * m).sum();
            mel.push(e.max(MEL_FLOOR).ln());
        }
    }
    (mel, num_frames)
}

pub fn mel_tensor(samples: &[f32], device: &Device) -> Result<(Tensor, usize)> {
    let (mel, frames) = gemma4_log_mel(samples);
    if frames == 0 {
        bail!("audio too short: fewer samples than one mel frame");
    }
    let t = Tensor::from_vec(mel, (1, frames, GEMMA4_AUDIO_MEL_BINS), device)?;
    Ok((t, frames))
}

pub struct Gemma4MmTowers {
    pub model_id: String,
    pub vision: Option<Gemma4VisionTower>,
    pub audio: Option<Gemma4AudioTower>,
    #[cfg(feature = "cuda")]
    pub vision_graph: Option<nv_models::gemma4_vision_graph::Gemma4VisionGraph>,
}

impl Gemma4MmTowers {
    pub fn new(
        model_id: impl Into<String>,
        vision: Option<Gemma4VisionTower>,
        audio: Option<Gemma4AudioTower>,
    ) -> Self {
        Self {
            model_id: model_id.into(),
            vision,
            audio,
            #[cfg(feature = "cuda")]
            vision_graph: None,
        }
    }

    #[cfg(feature = "cuda")]
    fn maybe_vision_graph(
        vision: &Option<Gemma4VisionTower>,
        device: &Device,
    ) -> Option<nv_models::gemma4_vision_graph::Gemma4VisionGraph> {
        use nv_models::gemma4_vision_graph::{vision_graph_enabled, Gemma4VisionGraph};
        if vision.is_none() || !vision_graph_enabled() {
            return None;
        }
        match Gemma4VisionGraph::new(device) {
            Ok(g) => Some(g),
            Err(e) => {
                tracing::warn!("NV_VISION_GRAPH=1 but graph capture is unavailable: {e:#}");
                None
            }
        }
    }

    pub fn from_model_dir(dir: &Path, device: &Device) -> Result<Self> {
        fn tower_enabled(which: &str) -> bool {
            match std::env::var("NV_MM_TOWERS") {
                Ok(v) => v.split(',').any(|t| t.trim() == which),
                Err(_) => true,
            }
        }
        let vision_dtype = match std::env::var("NV_MM_VISION_DTYPE").as_deref() {
            Ok("int8") => DType::U8,
            Ok("bf16") => DType::BF16,
            _ => DType::F32,
        };
        let model_id = dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| dir.display().to_string());
        let cfg_path = dir.join("config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let v: serde_json::Value =
            serde_json::from_str(&raw).with_context(|| format!("parse {}", cfg_path.display()))?;
        let vision = match v.get("vision_config") {
            _ if !tower_enabled("vision") => None,
            None | Some(serde_json::Value::Null) => None,
            Some(_) => {
                let vcfg = Gemma4VisionConfig::from_hf_json_str(&raw)
                    .context("parse gemma4 vision_config")?;
                let loader = WeightLoader::open_dir(dir, device)
                    .with_context(|| format!("open weights in {}", dir.display()))?;
                Some(
                    Gemma4VisionTower::load(vcfg, &loader, device, vision_dtype)
                        .context("load gemma4 vision tower")?,
                )
            }
        };
        let audio = if tower_enabled("audio") {
            Gemma4AudioTower::maybe_from_model_dir(dir, device)
                .context("load gemma4 audio tower")?
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let vision_graph = Self::maybe_vision_graph(&vision, device);
        Ok(Self {
            model_id,
            vision,
            audio,
            #[cfg(feature = "cuda")]
            vision_graph,
        })
    }

    fn require_vision(&self) -> Result<&Gemma4VisionTower> {
        self.vision.as_ref().with_context(|| {
            format!(
                "model {} has no vision tower (vision_config missing); image input is not supported by this model",
                self.model_id
            )
        })
    }

    fn require_audio(&self) -> Result<&Gemma4AudioTower> {
        self.audio.as_ref().with_context(|| {
            format!(
                "model {} has no audio tower (audio_config is null); audio input is not supported by this model",
                self.model_id
            )
        })
    }
}

pub enum PromptSegment {
    Text(String),
    Image(image::RgbImage),
    Audio(Vec<f32>),
}

pub fn segments_from_message(msg: &MmMessage) -> Result<Vec<PromptSegment>> {
    msg.parts
        .iter()
        .map(|p| match p {
            MmContentPart::Text(t) => Ok(PromptSegment::Text(t.clone())),
            MmContentPart::ImageUrl(spec) => {
                Ok(PromptSegment::Image(decode_image_ref(spec.url())?))
            }
            MmContentPart::InputAudio(spec) => Ok(PromptSegment::Audio(decode_audio_input(spec)?)),
        })
        .collect()
}

#[derive(Debug)]
pub struct PlannedImage {
    pub position: usize,
    pub patches: ImagePatches,
}

#[derive(Debug)]
pub struct PlannedAudio {
    pub position: usize,
    pub mel: Tensor,
    pub mel_frames: usize,
    pub num_soft_tokens: usize,
}

#[derive(Debug)]
pub struct MmPlan {
    pub tokens: Vec<u32>,
    pub images: Vec<PlannedImage>,
    pub audios: Vec<PlannedAudio>,
}

impl MmPlan {
    pub fn is_multimodal(&self) -> bool {
        !self.images.is_empty() || !self.audios.is_empty()
    }
}

fn plan_image_expansion(
    towers: &Gemma4MmTowers,
    img: &image::RgbImage,
    tokens: &mut Vec<u32>,
    images: &mut Vec<PlannedImage>,
    device: &Device,
) -> Result<()> {
    let tower = towers.require_vision()?;
    let patches = preprocess_image(img, tower.config(), device)?;
    let position = tokens.len() + 1;
    tokens.extend(expand_image_placeholder(patches.num_soft_tokens));
    images.push(PlannedImage { position, patches });
    Ok(())
}

fn plan_audio_expansion(
    towers: &Gemma4MmTowers,
    samples: &[f32],
    tokens: &mut Vec<u32>,
    audios: &mut Vec<PlannedAudio>,
    device: &Device,
) -> Result<()> {
    towers.require_audio()?;
    if samples.len() > MAX_AUDIO_SAMPLES {
        tracing::warn!(
            samples = samples.len(),
            kept = MAX_AUDIO_SAMPLES,
            "input_audio longer than the 30s gemma4 audio window; the tail past 30s is dropped"
        );
    }
    let clipped = &samples[..samples.len().min(MAX_AUDIO_SAMPLES)];
    let num_soft = audio_num_soft_tokens(
        clipped.len(),
        GEMMA4_AUDIO_SAMPLE_RATE,
        GEMMA4_AUDIO_SEQ_LENGTH,
    );
    if num_soft == 0 {
        bail!(
            "audio too short: {} samples yields no audio tokens (need at least ~{} ms)",
            clipped.len(),
            2 * GEMMA4_AUDIO_HOP_LENGTH * 1000 / GEMMA4_AUDIO_SAMPLE_RATE
        );
    }
    let (mel, mel_frames) = mel_tensor(clipped, device)?;
    let position = tokens.len() + 1;
    tokens.extend(expand_audio_placeholder(num_soft));
    audios.push(PlannedAudio {
        position,
        mel,
        mel_frames,
        num_soft_tokens: num_soft,
    });
    Ok(())
}

pub fn plan_prompt<F>(
    towers: &Gemma4MmTowers,
    segments: &[PromptSegment],
    device: &Device,
    mut tokenize: F,
) -> Result<MmPlan>
where
    F: FnMut(&str) -> Result<Vec<u32>>,
{
    let mut tokens: Vec<u32> = Vec::new();
    let mut images = Vec::new();
    let mut audios = Vec::new();
    for seg in segments {
        match seg {
            PromptSegment::Text(t) => tokens.extend(tokenize(t)?),
            PromptSegment::Image(img) => {
                plan_image_expansion(towers, img, &mut tokens, &mut images, device)?
            }
            PromptSegment::Audio(samples) => {
                plan_audio_expansion(towers, samples, &mut tokens, &mut audios, device)?
            }
        }
    }
    Ok(MmPlan {
        tokens,
        images,
        audios,
    })
}

#[derive(Clone, Debug, Default)]
pub struct MmMedia {
    pub images: Vec<image::RgbImage>,
    pub audios: Vec<Vec<f32>>,
}

pub fn plan_from_marked_tokens(
    towers: &Gemma4MmTowers,
    rendered_ids: &[u32],
    media: &MmMedia,
    device: &Device,
) -> Result<MmPlan> {
    use nv_models::gemma4_mm_splice::{GEMMA4_BOA_TOKEN_ID, GEMMA4_BOI_TOKEN_ID};
    let mut tokens = Vec::with_capacity(rendered_ids.len());
    let mut images = Vec::new();
    let mut audios = Vec::new();
    let mut img_it = media.images.iter();
    let mut aud_it = media.audios.iter();
    for &id in rendered_ids {
        match id {
            GEMMA4_BOI_TOKEN_ID => {
                let img = img_it.next().context(
                    "prompt contains more image markers than image_url parts; the boi marker \
                     token is reserved for image_url parts and cannot be written as text",
                )?;
                plan_image_expansion(towers, img, &mut tokens, &mut images, device)?;
            }
            GEMMA4_BOA_TOKEN_ID => {
                let samples = aud_it.next().context(
                    "prompt contains more audio markers than input_audio parts; the boa marker \
                     token is reserved for input_audio parts and cannot be written as text",
                )?;
                plan_audio_expansion(towers, samples, &mut tokens, &mut audios, device)?;
            }
            other => tokens.push(other),
        }
    }
    if img_it.next().is_some() {
        bail!("request carries more image_url parts than the rendered prompt has image markers");
    }
    if aud_it.next().is_some() {
        bail!("request carries more input_audio parts than the rendered prompt has audio markers");
    }
    Ok(MmPlan {
        tokens,
        images,
        audios,
    })
}

pub fn run_towers(towers: &Gemma4MmTowers, plan: &MmPlan, dtype: DType) -> Result<Vec<MmItem>> {
    let vision_scale: f64 = std::env::var("NV_MM_VISION_SCALE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let mut items = Vec::with_capacity(plan.images.len() + plan.audios.len());
    for img in &plan.images {
        let tower = towers.require_vision()?;
        #[cfg(feature = "cuda")]
        let emb = match &towers.vision_graph {
            Some(g) => {
                let (gw, gh) = img.patches.grid;
                g.forward(tower, &img.patches.host_pixels, gw, gh)
                    .context("gemma4 vision tower graphed forward")?
            }
            None => tower
                .forward(&img.patches.pixel_values, &img.patches.position_ids)
                .context("gemma4 vision tower forward")?,
        };
        #[cfg(not(feature = "cuda"))]
        let emb = tower
            .forward(&img.patches.pixel_values, &img.patches.position_ids)
            .context("gemma4 vision tower forward")?;
        let emb = if vision_scale != 1.0 {
            emb.affine(vision_scale, 0.0)?
        } else {
            emb
        };
        let rows = emb.dims2()?.0;
        if rows != img.patches.num_soft_tokens {
            bail!(
                "vision tower produced {} soft tokens but the prompt reserves {}",
                rows,
                img.patches.num_soft_tokens
            );
        }
        items.push(MmItem {
            modality: Modality::Image,
            position: img.position,
            embedding: emb.to_dtype(dtype)?,
        });
    }
    for aud in &plan.audios {
        let tower = towers.require_audio()?;
        let (enc, sub_lens) = tower
            .encoder
            .forward(&aud.mel, &[aud.mel_frames])
            .context("gemma4 audio encoder forward")?;
        if sub_lens[0] != aud.num_soft_tokens {
            bail!(
                "audio encoder produced {} soft tokens but the prompt reserves {}",
                sub_lens[0],
                aud.num_soft_tokens
            );
        }
        let emb = tower
            .embedder
            .forward(&enc)
            .context("gemma4 audio embedder forward")?;
        let emb = emb.i(0)?.narrow(0, 0, aud.num_soft_tokens)?;
        items.push(MmItem {
            modality: Modality::Audio,
            position: aud.position,
            embedding: emb.to_dtype(dtype)?,
        });
    }
    Ok(items)
}

pub fn embed_prompt(embed_weight: &Tensor, embed_scale: f64, tokens: &[u32]) -> Result<Tensor> {
    if tokens.is_empty() {
        bail!("cannot embed an empty token sequence");
    }
    let vocab = embed_weight.dims2()?.0;
    if let Some(&bad) = tokens.iter().find(|&&t| t as usize >= vocab) {
        bail!("token id {bad} out of range for vocab {vocab}");
    }
    let ids = Tensor::from_vec(tokens.to_vec(), tokens.len(), embed_weight.device())?;
    let emb = embed_weight.index_select(&ids, 0)?.to_dtype(DType::F32)?;
    Ok(emb.affine(embed_scale, 0.0)?)
}

pub fn mm_embeddings(
    towers: &Gemma4MmTowers,
    plan: &MmPlan,
    embed_weight: &Tensor,
    embed_scale: f64,
) -> Result<Tensor> {
    let text = embed_prompt(embed_weight, embed_scale, &plan.tokens)?;
    if !plan.is_multimodal() {
        return splice_mm_embeddings(&text, &plan.tokens, &[]);
    }
    let items = run_towers(towers, plan, text.dtype())?;
    splice_mm_embeddings(&text, &plan.tokens, &items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_image_ref_rejects_file_urls_without_leaking_the_path() {
        for url in [
            "file:///etc/passwd",
            "/etc/passwd",
            "../../etc/passwd",
            "http://example.com/a.png",
            "https://example.com/a.png",
        ] {
            let err = decode_image_ref(url).unwrap_err().to_string();
            assert!(err.contains("data: URL"), "unexpected error: {err}");
            assert!(!err.contains("passwd"), "error leaks the path: {err}");
            assert!(!err.contains("example.com"), "error leaks the path: {err}");
        }
    }

    #[test]
    fn decode_image_ref_still_requires_base64_data_urls() {
        let err = decode_image_ref("data:image/png,abc")
            .unwrap_err()
            .to_string();
        assert!(err.contains("base64"), "unexpected error: {err}");
    }
}
