use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{anyhow, ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use half::bf16;
use nv_kernels::wgpu_backend::dispatch::ReadbackArm;
use nv_models::voxtral::VoxtralQuant;
use nv_models::voxtral_tts::{
    stored_zip_entries, SeededNoise, DEFAULT_CFG_ALPHA, DEFAULT_NFE, PARAMS_FILE, TEKKEN_FILE,
};
use nv_models::voxtral_tts_codec_wgpu::{CodecState, VoxtralTtsCodecWgpu};
use nv_models::voxtral_tts_wgpu::VoxtralTtsWgpu;
use nv_weights::{ExternBytes, ExternReadFn, WeightLoader};

pub const THE_CHECKPOINT_IS_A_FEW_VIRTUAL_EXTERN_SHARDS_READ_BY_RANGE_SO_NEITHER_HEAP_HOLDS_THE_8_GB_FILE: &str =
    "nv-weights addresses a shard in usize, so the 8 GB consolidated file cannot be one shard on \
     wasm32; the worker slices its header into sub-4-GiB virtual shards (codec, heads, embedding, \
     two halves of the backbone), each a rebuilt header plus a synchronous range reader (a sync \
     XHR with a Range header, legal in a dedicated worker) that maps the virtual offsets back to \
     the file, so every tensor transits wasm memory once, at its own size, on its way to the \
     device, and a tier touches only the shards its tensors live in";

thread_local! {
    static TEXTS: RefCell<BTreeMap<String, String>> = const { RefCell::new(BTreeMap::new()) };
    static CHECKPOINT: RefCell<Option<WeightLoader>> = const { RefCell::new(None) };
    static CODEC: RefCell<Option<VoxtralTtsCodecWgpu>> = const { RefCell::new(None) };
    static CODEC_STATE: RefCell<Option<CodecState>> = const { RefCell::new(None) };
    static TTS: RefCell<Option<VoxtralTtsWgpu>> = const { RefCell::new(None) };
}

pub fn tts_clear() {
    TEXTS.with(|t| t.borrow_mut().clear());
    CHECKPOINT.with(|c| *c.borrow_mut() = None);
    CODEC.with(|c| *c.borrow_mut() = None);
    CODEC_STATE.with(|c| *c.borrow_mut() = None);
    TTS.with(|t| *t.borrow_mut() = None);
}

pub fn tts_stage_text(name: &str, text: String) {
    TEXTS.with(|t| t.borrow_mut().insert(name.to_string(), text));
}

pub fn tts_open_checkpoint(shards: Vec<(String, u64, ExternReadFn)>) -> Result<()> {
    let _ = THE_CHECKPOINT_IS_A_FEW_VIRTUAL_EXTERN_SHARDS_READ_BY_RANGE_SO_NEITHER_HEAP_HOLDS_THE_8_GB_FILE;
    let loader = WeightLoader::open_extern_shards(
        shards.into_iter().map(|(name, len, read)| (name, ExternBytes { len, read })),
        &Device::Cpu,
    )?;
    CHECKPOINT.with(|c| *c.borrow_mut() = Some(loader));
    Ok(())
}

fn with_checkpoint<T>(f: impl FnOnce(&WeightLoader) -> Result<T>) -> Result<T> {
    CHECKPOINT.with(|c| {
        f(c.borrow()
            .as_ref()
            .ok_or_else(|| anyhow!("no checkpoint open: call tts_open_checkpoint first"))?)
    })
}

fn text(name: &str) -> Result<String> {
    TEXTS.with(|t| {
        t.borrow()
            .get(name)
            .cloned()
            .ok_or_else(|| anyhow!("{name} is not staged: tts_stage_text first"))
    })
}

pub fn codec_boot() -> Result<String> {
    let codec = with_checkpoint(VoxtralTtsCodecWgpu::from_loader)?;
    CODEC.with(|c| *c.borrow_mut() = Some(codec));
    CODEC_STATE.with(|c| *c.borrow_mut() = Some(CodecState::default()));
    Ok(serde_json::json!({"codec": "voxtral_tts_codec_wgpu"}).to_string())
}

async fn with_codec<T, F, Fut>(f: F) -> Result<T>
where
    F: FnOnce(VoxtralTtsCodecWgpu, CodecState) -> Fut,
    Fut: core::future::Future<Output = (VoxtralTtsCodecWgpu, CodecState, Result<T>)>,
{
    let codec = CODEC
        .with(|c| c.borrow_mut().take())
        .ok_or_else(|| anyhow!("codec not booted: call codec_boot first"))?;
    let state = CODEC_STATE.with(|c| c.borrow_mut().take()).unwrap_or_default();
    let (codec, state, res) = f(codec, state).await;
    CODEC.with(|c| *c.borrow_mut() = Some(codec));
    CODEC_STATE.with(|c| *c.borrow_mut() = Some(state));
    res
}

pub async fn codec_decode(codes: Vec<i32>, arm: ReadbackArm) -> Result<Vec<f32>> {
    with_codec(|codec, state| async move {
        let r = codec.decode_arm(&codes, arm).await;
        (codec, state, r)
    })
    .await
}

pub fn codec_stream_reset() -> Result<()> {
    CODEC.with(|c| {
        let codec = c.borrow();
        let codec = codec.as_ref().ok_or_else(|| anyhow!("codec not booted"))?;
        CODEC_STATE.with(|s| {
            if let Some(st) = s.borrow_mut().as_mut() {
                st.reset(codec);
            }
        });
        Ok(())
    })
}

pub async fn codec_decode_frames(frames: Vec<i32>, arm: ReadbackArm) -> Result<Vec<f32>> {
    with_codec(|codec, mut state| async move {
        let r = codec.decode_frames_arm(&mut state, &frames, arm).await;
        (codec, state, r)
    })
    .await
}

pub fn tts_boot(with_backbone: bool) -> Result<String> {
    let (params, tekken) = (text(PARAMS_FILE)?, text(TEKKEN_FILE)?);
    let model = with_checkpoint(|w| {
        VoxtralTtsWgpu::from_parts(&params, &tekken, w, None, VoxtralQuant::Bf16, VoxtralQuant::Bf16, DEFAULT_NFE, DEFAULT_CFG_ALPHA, with_backbone)
    })?;
    let info = serde_json::json!({
        "nfe": model.nfe(),
        "with_backbone": with_backbone,
        "voices": model.voices().map(|(v, n)| serde_json::json!({"voice": v, "rows": n})).collect::<Vec<_>>(),
        "step_passes": model.step_pass_count(),
        "flow_passes": model.flow_pass_count(),
    })
    .to_string();
    TTS.with(|t| *t.borrow_mut() = Some(model));
    Ok(info)
}

pub fn voice_bits_from_torch(bytes: &[u8]) -> Result<Vec<u16>> {
    use candle_core::pickle::{Object, Stack};
    let entries = stored_zip_entries(bytes)?;
    let (pkl_name, &(at, len)) = entries
        .iter()
        .find(|(k, _)| k.ends_with("data.pkl"))
        .context("voice .pt has no data.pkl")?;
    let mut stack = Stack::empty();
    let mut reader: &[u8] = &bytes[at..at + len];
    stack.read_loop(&mut reader)?;
    let dir = PathBuf::from(pkl_name.strip_suffix(".pkl").expect("ends with data.pkl"));
    let info = stack
        .finalize()?
        .into_tensor_info(Object::Unicode("tensor".into()), &dir)?
        .context("voice .pt pickle does not rebuild a tensor")?;
    ensure!(info.layout.is_contiguous() && info.layout.start_offset() == 0, "voice tensor is not contiguous at offset 0");
    let &(at, len) = entries.get(&info.path).with_context(|| format!("voice storage {} missing", info.path))?;
    let t = Tensor::from_raw_buffer(&bytes[at..at + len], info.dtype, info.layout.shape().dims(), &Device::Cpu)?;
    Ok(t.to_dtype(DType::BF16)?.flatten_all()?.to_vec1::<bf16>()?.iter().map(|v| v.to_bits()).collect())
}

pub fn tts_stage_voice_torch(voice: &str, pt_bytes: &[u8]) -> Result<()> {
    let bits = voice_bits_from_torch(pt_bytes)?;
    TTS.with(|t| {
        t.borrow()
            .as_ref()
            .ok_or_else(|| anyhow!("tts not booted: call tts_boot first"))?
            .stage_voice(voice, &bits)
    })
}

async fn with_tts<T, F, Fut>(f: F) -> Result<T>
where
    F: FnOnce(VoxtralTtsWgpu) -> Fut,
    Fut: core::future::Future<Output = (VoxtralTtsWgpu, Result<T>)>,
{
    let model = TTS
        .with(|t| t.borrow_mut().take())
        .ok_or_else(|| anyhow!("tts not booted: call tts_boot first"))?;
    let (model, res) = f(model).await;
    TTS.with(|t| *t.borrow_mut() = Some(model));
    res
}

pub async fn tts_frame_from_hidden(hidden: Vec<f32>, x0: Vec<f32>, trace: bool, arm: ReadbackArm) -> Result<String> {
    with_tts(|model| async move {
        let r = model.frame_from_hidden_arm(&hidden, &x0, trace, arm).await.map(|(frame, tr)| {
            serde_json::json!({
                "frame": frame,
                "semantic_logits": tr.semantic_logits,
                "v_cond": tr.v_cond,
                "v_uncond": tr.v_uncond,
                "x": tr.x,
            })
            .to_string()
        });
        (model, r)
    })
    .await
}

pub struct SynthRequest {
    pub text: String,
    pub voice: String,
    pub noise_rows: Vec<f32>,
    pub seed: u64,
    pub max_frames: usize,
    pub teacher_frames: Vec<u32>,
}

pub async fn tts_synthesize(req: SynthRequest, arm: ReadbackArm) -> Result<serde_json::Value> {
    with_tts(|mut model| async move {
        let n = model.codebooks().n_acoustic;
        let n_cb = model.codebooks().n_codebooks();
        let teacher: Option<Vec<Vec<u32>>> =
            (!req.teacher_frames.is_empty()).then(|| req.teacher_frames.chunks(n_cb).map(<[u32]>::to_vec).collect());
        let rows = req.noise_rows.len() / n.max(1);
        let mut seeded = SeededNoise::new(req.seed);
        let mut noise = |i: usize| -> Vec<f32> {
            if rows == 0 {
                seeded.frame(n)
            } else {
                let r = i.min(rows - 1);
                req.noise_rows[r * n..(r + 1) * n].to_vec()
            }
        };
        let r = model
            .synthesize_arm(&req.text, &req.voice, &mut noise, req.max_frames, false, teacher.as_deref(), arm)
            .await
            .map(|s| {
                serde_json::json!({
                    "prompt_ids": s.prompt_ids,
                    "frames": s.frames,
                    "ended": s.ended,
                    "codec_codes": s.codec_codes(),
                })
            });
        (model, r)
    })
    .await
}

pub async fn tts_speak(req: SynthRequest, arm: ReadbackArm) -> Result<Vec<f32>> {
    let synth = tts_synthesize(req, arm).await?;
    let codes: Vec<i32> = synth["frames"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|f| f.as_array().into_iter().flatten())
        .map(|c| c.as_u64().unwrap_or(0) as i32)
        .collect();
    ensure!(!codes.is_empty(), "the free run produced no frames");
    codec_decode(codes, arm).await
}
