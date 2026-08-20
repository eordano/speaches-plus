#![allow(dead_code)]

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use super::{AudioPadAlignment, EouModel};
use crate::defaults;

pub const SAMPLE_RATE: u32 = defaults::eou::audio::SAMPLE_RATE;
pub const N_MELS: usize = defaults::eou::audio::N_MELS;
pub const N_FFT: usize = defaults::eou::audio::N_FFT;
pub const HOP_LENGTH: usize = defaults::eou::audio::HOP_LENGTH;
pub const CHUNK_LENGTH_S: usize = defaults::eou::audio::CHUNK_LENGTH_S;
pub const TARGET_SAMPLES: usize = defaults::eou::audio::TARGET_SAMPLES;
pub const N_FRAMES: usize = defaults::eou::audio::N_FRAMES;

pub struct AudioEouModel {
    session: Arc<Mutex<Session>>,
    audio_window_ms: u32,
    pad_alignment: AudioPadAlignment,
    mel_filters: Arc<Vec<f32>>,
    hann: Arc<Vec<f32>>,
}

impl AudioEouModel {
    pub fn load(
        model_path: impl AsRef<Path>,
        audio_window_ms: u32,
        pad_alignment: AudioPadAlignment,
    ) -> Result<Self> {
        let path = model_path.as_ref();
        let session = Session::builder()
            .map_err(crate::vad::ort_err)?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(crate::vad::ort_err)?
            .with_intra_threads(1)
            .map_err(crate::vad::ort_err)?
            .commit_from_file(path)
            .map_err(crate::vad::ort_err)
            .with_context(|| format!("load smart-turn from {}", path.display()))?;
        Ok(Self::from_session(
            Arc::new(Mutex::new(session)),
            audio_window_ms,
            pad_alignment,
        ))
    }

    pub fn from_session(
        session: Arc<Mutex<Session>>,
        audio_window_ms: u32,
        pad_alignment: AudioPadAlignment,
    ) -> Self {
        let mel_filters = Arc::new(crate::mel_scale::build_mel_filters(N_MELS, N_FFT, SAMPLE_RATE as usize));
        let hann = Arc::new(crate::mel_scale::hann_window(N_FFT));
        Self {
            session,
            audio_window_ms,
            pad_alignment,
            mel_filters,
            hann,
        }
    }

    pub fn audio_window_ms(&self) -> u32 {
        self.audio_window_ms
    }

    pub fn pad_alignment(&self) -> AudioPadAlignment {
        self.pad_alignment
    }

    pub fn run(&self, audio: &[f32], sample_rate: u32) -> Result<f32> {
        if sample_rate != SAMPLE_RATE {
            return Err(anyhow!(
                "smart-turn expects {} Hz, got {}",
                SAMPLE_RATE,
                sample_rate
            ));
        }
        let prepared = prepare_audio(audio, self.audio_window_ms, self.pad_alignment);
        let mel = log_mel_spectrogram(&prepared, &self.hann, &self.mel_filters);
        debug_assert_eq!(mel.len(), N_MELS * N_FRAMES);

        let tensor =
            Tensor::<f32>::from_array(([1usize, N_MELS, N_FRAMES], mel.into_boxed_slice()))
                .map_err(crate::vad::ort_err)?;

        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("smart-turn session poisoned"))?;
        let outputs = session
            .run(ort::inputs!["input_features" => tensor])
            .map_err(crate::vad::ort_err)?;
        let first_name = outputs
            .iter()
            .next()
            .map(|(name, _)| name.to_string())
            .ok_or_else(|| anyhow!("smart-turn produced no outputs"))?;
        let (_, data) = outputs[first_name.as_str()]
            .try_extract_tensor::<f32>()
            .map_err(crate::vad::ort_err)?;
        let raw = *data
            .first()
            .ok_or_else(|| anyhow!("smart-turn empty output"))?;
        Ok(normalize_output(raw))
    }
}

impl EouModel for AudioEouModel {
    fn score(&self, _context: &str) -> f32 {
        f32::NAN
    }

    fn score_with_audio(&self, _context: &str, audio: &[f32], sample_rate: u32) -> f32 {
        match self.run(audio, sample_rate) {
            Ok(p) if p.is_finite() && (0.0..=1.0).contains(&p) => p,
            Ok(p) => {
                tracing::warn!(value = p, "smart-turn returned out-of-range value");
                f32::NAN
            }
            Err(err) => {
                tracing::warn!(error = %err, "smart-turn inference failed");
                f32::NAN
            }
        }
    }
}

fn normalize_output(raw: f32) -> f32 {
    if !raw.is_finite() {
        return raw;
    }
    if (0.0..=1.0).contains(&raw) {
        raw
    } else {
        sigmoid(raw)
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn prepare_audio(
    audio: &[f32],
    audio_window_ms: u32,
    pad_alignment: AudioPadAlignment,
) -> Vec<f32> {
    let target = TARGET_SAMPLES;
    let max_window = (audio_window_ms as usize).saturating_mul(SAMPLE_RATE as usize) / 1000;
    let max_window = max_window.min(target);

    let mut working: Vec<f32> = if audio.len() > max_window {
        audio[audio.len() - max_window..].to_vec()
    } else {
        audio.to_vec()
    };
    if working.len() < target {
        let pad = target - working.len();
        let mut padded = Vec::with_capacity(target);
        match pad_alignment {
            AudioPadAlignment::Leading => {
                padded.resize(pad, 0.0);
                padded.extend_from_slice(&working);
            }
            AudioPadAlignment::Trailing => {
                padded.extend_from_slice(&working);
                padded.resize(target, 0.0);
            }
        }
        working = padded;
    } else if working.len() > target {
        working = working[working.len() - target..].to_vec();
    }
    debug_assert_eq!(working.len(), target);
    for sample in working.iter_mut() {
        *sample = if sample.is_finite() {
            sample.clamp(-1.0, 1.0)
        } else {
            0.0
        };
    }
    working
}

pub fn log_mel_spectrogram(audio: &[f32], hann: &[f32], mel_filters: &[f32]) -> Vec<f32> {
    assert_eq!(hann.len(), N_FFT);
    let n_bins = N_FFT / 2 + 1;
    assert_eq!(mel_filters.len(), N_MELS * n_bins);

    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut input_buf = fft.make_input_vec();
    let mut output_buf = fft.make_output_vec();

    let mut padded = vec![0.0f32; audio.len() + N_FFT];
    let pad = N_FFT / 2;
    for i in 0..pad {
        padded[i] = audio[pad - i];
    }
    padded[pad..pad + audio.len()].copy_from_slice(audio);
    for i in 0..pad {
        let src = audio.len().saturating_sub(2 + i);
        padded[pad + audio.len() + i] = audio[src];
    }

    let mut power = vec![0.0f32; n_bins * N_FRAMES];
    for frame in 0..N_FRAMES {
        let start = frame * HOP_LENGTH;
        for i in 0..N_FFT {
            input_buf[i] = padded[start + i] * hann[i];
        }
        fft.process(&mut input_buf, &mut output_buf)
            .expect("FFT process");
        for (k, c) in output_buf.iter().enumerate() {
            let mag = c.re * c.re + c.im * c.im;
            power[frame * n_bins + k] = mag;
        }
    }

    let mut mel = vec![0.0f32; N_MELS * N_FRAMES];
    for m in 0..N_MELS {
        for frame in 0..N_FRAMES {
            let mut sum = 0.0f32;
            for k in 0..n_bins {
                sum += mel_filters[m * n_bins + k] * power[frame * n_bins + k];
            }
            mel[m * N_FRAMES + frame] = sum;
        }
    }

    let eps = 1e-10f32;
    let mut log_mel = mel.iter().map(|v| v.max(eps).log10()).collect::<Vec<_>>();
    let max_val = log_mel.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let floor = max_val - 8.0;
    for v in log_mel.iter_mut() {
        if *v < floor {
            *v = floor;
        }
        *v = (*v + 4.0) / 4.0;
    }
    log_mel
}

pub fn try_load_from_env(
    cfg_window_ms: u32,
    cfg_alignment: AudioPadAlignment,
) -> Option<Arc<AudioEouModel>> {
    let path = std::env::var(defaults::env::EOU_AUDIO_MODEL_PATH).ok()?;
    let path = path.trim();
    if path.is_empty() {
        return None;
    }
    let p = Path::new(path);
    if !p.exists() {
        tracing::warn!(
            path = %path,
            "EOU_AUDIO_MODEL_PATH set but file not found; falling back to stub"
        );
        return None;
    }
    match AudioEouModel::load(p, cfg_window_ms, cfg_alignment) {
        Ok(m) => {
            tracing::info!(
                path = %path,
                window_ms = cfg_window_ms,
                "smart-turn audio EOU loaded"
            );
            Some(Arc::new(m))
        }
        Err(err) => {
            tracing::warn!(
                path = %path,
                error = %err,
                "smart-turn audio EOU load failed; falling back to stub"
            );
            None
        }
    }
}

pub fn shared_audio_eou_model(
    window_ms: u32,
    alignment: AudioPadAlignment,
) -> Option<Arc<dyn EouModel>> {
    try_load_from_env(window_ms, alignment).map(|m| m as Arc<dyn EouModel>)
}

pub fn resolve_audio_eou_paths() -> Option<String> {
    std::env::var(defaults::env::EOU_AUDIO_MODEL_PATH)
        .ok()
        .filter(|s| !s.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prepare_audio_pads_leading_when_short() {
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 + 1.0) / 2000.0).collect();
        let prepared = prepare_audio(&audio, 8000, AudioPadAlignment::Leading);
        assert_eq!(prepared.len(), TARGET_SAMPLES);
        let pad_len = TARGET_SAMPLES - 1600;
        for i in 0..pad_len {
            assert_eq!(prepared[i], 0.0, "expected zero pad at {}", i);
        }
        for i in 0..1600 {
            assert!(
                (prepared[pad_len + i] - audio[i]).abs() < 1e-6,
                "mismatch at {} got {} want {}",
                i,
                prepared[pad_len + i],
                audio[i]
            );
        }
    }

    #[test]
    fn prepare_audio_pads_trailing_when_configured() {
        let audio: Vec<f32> = (0..1600).map(|i| (i as f32 + 1.0) / 2000.0).collect();
        let prepared = prepare_audio(&audio, 8000, AudioPadAlignment::Trailing);
        assert_eq!(prepared.len(), TARGET_SAMPLES);
        for i in 0..1600 {
            assert!(
                (prepared[i] - audio[i]).abs() < 1e-6,
                "mismatch at {} got {} want {}",
                i,
                prepared[i],
                audio[i]
            );
        }
        for i in 1600..TARGET_SAMPLES {
            assert_eq!(prepared[i], 0.0);
        }
    }

    #[test]
    fn prepare_audio_truncates_to_last_window() {
        let mut audio: Vec<f32> = vec![0.0f32; TARGET_SAMPLES + 10000];
        for i in 0..audio.len() {
            audio[i] = (((i as f32) % 1000.0) - 500.0) / 1000.0;
        }
        let prepared = prepare_audio(&audio, 8000, AudioPadAlignment::Leading);
        assert_eq!(prepared.len(), TARGET_SAMPLES);
        let want_first = audio[audio.len() - TARGET_SAMPLES];
        assert!(
            (prepared[0] - want_first).abs() < 1e-9,
            "got {} want {}",
            prepared[0],
            want_first
        );
        let want_last = audio[audio.len() - 1];
        assert!((prepared[TARGET_SAMPLES - 1] - want_last).abs() < 1e-9);
    }

    #[test]
    fn prepare_audio_clamps_out_of_range_samples() {
        let mut audio = vec![0.0_f32; 5];
        audio[0] = 5.0;
        audio[1] = -3.0;
        audio[2] = f32::NAN;
        audio[3] = 0.5;
        audio[4] = f32::INFINITY;
        let prepared = prepare_audio(&audio, 8000, AudioPadAlignment::Leading);
        assert_eq!(prepared.len(), TARGET_SAMPLES);
        let tail = &prepared[TARGET_SAMPLES - audio.len()..];
        assert_eq!(tail[0], 1.0);
        assert_eq!(tail[1], -1.0);
        assert_eq!(tail[2], 0.0);
        assert!((tail[3] - 0.5).abs() < 1e-6);
        assert_eq!(tail[4], 0.0);
    }

    #[test]
    fn mel_filters_are_finite_and_non_negative() {
        let filters = crate::mel_scale::build_mel_filters(N_MELS, N_FFT, SAMPLE_RATE as usize);
        for (i, v) in filters.iter().enumerate() {
            assert!(v.is_finite(), "filter[{}] not finite", i);
            assert!(*v >= 0.0, "filter[{}] negative: {}", i, v);
        }
    }

    #[test]
    fn log_mel_shape_is_correct() {
        let hann = crate::mel_scale::hann_window(N_FFT);
        let filters = crate::mel_scale::build_mel_filters(N_MELS, N_FFT, SAMPLE_RATE as usize);
        let audio = vec![0.1f32; TARGET_SAMPLES];
        let mel = log_mel_spectrogram(&audio, &hann, &filters);
        assert_eq!(mel.len(), N_MELS * N_FRAMES);
        for v in &mel {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn try_load_skips_when_model_path_unset() {
        std::env::remove_var(defaults::env::EOU_AUDIO_MODEL_PATH);
        assert!(try_load_from_env(8000, AudioPadAlignment::Leading).is_none());
    }

    #[test]
    fn try_load_skips_when_model_absent() {
        std::env::set_var(
            defaults::env::EOU_AUDIO_MODEL_PATH,
            "/this/does/not/exist/smart-turn.onnx",
        );
        let m = try_load_from_env(8000, AudioPadAlignment::Leading);
        assert!(m.is_none());
        std::env::remove_var(defaults::env::EOU_AUDIO_MODEL_PATH);
    }

    #[test]
    fn smart_turn_loads_when_model_present_and_runs() {
        let path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("models/smart-turn-v3.onnx");
        if !path.exists() {
            eprintln!("skip: {} missing", path.display());
            return;
        }
        let model =
            AudioEouModel::load(&path, 8000, AudioPadAlignment::Leading).expect("load smart-turn");
        let silence = vec![0.0f32; TARGET_SAMPLES];
        let p = model.run(&silence, SAMPLE_RATE).expect("score");
        assert!((0.0..=1.0).contains(&p), "out of range: {p}");
    }
}
