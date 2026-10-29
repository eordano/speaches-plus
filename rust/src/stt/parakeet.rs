use super::TokenLogprob;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::Tensor;

use crate::diarization::ep;
use crate::vad::ort_err;

pub const PARAKEET_DIR_ENV: &str = "STT_PARAKEET_DIR";
pub const PARAKEET_HUB_REPO: &str = "models--istupakov--parakeet-tdt-0.6b-v2-onnx";

const SR: usize = 16_000;
const BLANK_ID: i32 = 1024;
const TOKEN_LOGITS: usize = 1025;
const TDT_DURATION_BINS: [usize; 5] = [0, 1, 2, 3, 4];
const STATE_LAYERS: usize = 2;
const STATE_DIM: usize = 640;
const ENC_DIM: usize = 1024;
const WINDOW_SAMPLES_BOUNDS_FULL_ATTENTION_COST: usize = 60 * SR;
const MAX_SYMBOLS_PER_FRAME_BREAKS_EMISSION_LOOPS: usize = 10;

pub struct ParakeetTdt {
    pre: Mutex<Session>,
    encoder: Mutex<Session>,
    joint: Mutex<Session>,
    vocab: Vec<String>,
    encoder_ep: &'static str,
}

pub fn parakeet_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var(PARAKEET_DIR_ENV) {
        return Ok(PathBuf::from(d));
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let snaps = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(PARAKEET_HUB_REPO)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snaps)
        .with_context(|| {
            format!(
                "no parakeet snapshot under {}; set {PARAKEET_DIR_ENV} to a checkpoint dir \
                 holding nemo128.onnx + encoder-model.onnx(.data) + decoder_joint-model.onnx + \
                 vocab.txt",
                snaps.display()
            )
        })?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    entries.sort();
    entries
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("empty snapshots dir {}", snaps.display()))
}

fn load_session(dir: &Path, file: &str) -> Result<ep::Loaded> {
    ep::load_session(&dir.join(file), file)
        .with_context(|| format!("load parakeet {file} from {}", dir.display()))
}

impl ParakeetTdt {
    pub fn load(dir: &Path) -> Result<Self> {
        let vocab_raw = std::fs::read_to_string(dir.join("vocab.txt"))
            .with_context(|| format!("read {}/vocab.txt", dir.display()))?;
        let vocab: Vec<String> = vocab_raw
            .lines()
            .map(|l| l.split(' ').next().unwrap_or_default().to_string())
            .collect();
        if vocab.len() != TOKEN_LOGITS {
            return Err(anyhow!(
                "vocab.txt has {} entries, the TDT head emits {TOKEN_LOGITS} token logits",
                vocab.len()
            ));
        }
        let pre = load_session(dir, "nemo128.onnx")?;
        let encoder = load_session(dir, "encoder-model.onnx")?;
        let joint = load_session(dir, "decoder_joint-model.onnx")?;
        let encoder_ep = encoder.provider;
        Ok(Self {
            pre: Mutex::new(pre.session),
            encoder: Mutex::new(encoder.session),
            joint: Mutex::new(joint.session),
            vocab,
            encoder_ep,
        })
    }

    pub fn ep(&self) -> &'static str {
        self.encoder_ep
    }

    pub fn transcribe(&self, audio_16k_mono: &[f32]) -> Result<String> {
        Ok(self.transcribe_scored(audio_16k_mono)?.0)
    }

    pub fn transcribe_scored(&self, audio_16k_mono: &[f32]) -> Result<(String, Vec<TokenLogprob>)> {
        let mut text = String::new();
        let mut tokens = Vec::new();
        for window in audio_16k_mono.chunks(WINDOW_SAMPLES_BOUNDS_FULL_ATTENTION_COST) {
            let scored = self.decode_window(window)?;
            let ids: Vec<i32> = scored.iter().map(|(id, _)| *id).collect();
            let part = self.detokenize(&ids);
            if !part.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&part);
            }
            tokens.extend(scored.into_iter().map(|(id, logprob)| TokenLogprob {
                token: self.vocab[id as usize].replace('\u{2581}', " "),
                logprob,
            }));
        }
        Ok((text, tokens))
    }

    fn decode_window(&self, audio: &[f32]) -> Result<Vec<(i32, f32)>> {
        if audio.is_empty() {
            return Ok(Vec::new());
        }
        let n = audio.len();
        let wave = Tensor::<f32>::from_array(([1usize, n], audio.to_vec().into_boxed_slice()))
            .map_err(ort_err)?;
        let wave_len = Tensor::<i64>::from_array(([1usize], vec![n as i64].into_boxed_slice()))
            .map_err(ort_err)?;
        let (feat_shape, feats, flen) = {
            let mut pre = self
                .pre
                .lock()
                .map_err(|_| anyhow!("parakeet pre poisoned"))?;
            let out = pre
                .run(ort::inputs!["waveforms" => wave, "waveforms_lens" => wave_len])
                .map_err(ort_err)?;
            let (shape, data) = out["features"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let (_, lens) = out["features_lens"]
                .try_extract_tensor::<i64>()
                .map_err(ort_err)?;
            (shape.to_vec(), data.to_vec(), lens[0])
        };
        let feat_dims: Vec<usize> = feat_shape.iter().map(|&d| d as usize).collect();
        let feat_tensor =
            Tensor::<f32>::from_array((feat_dims, feats.into_boxed_slice())).map_err(ort_err)?;
        let feat_len = Tensor::<i64>::from_array(([1usize], vec![flen].into_boxed_slice()))
            .map_err(ort_err)?;
        let (enc, t_len) = {
            let mut encoder = self
                .encoder
                .lock()
                .map_err(|_| anyhow!("parakeet encoder poisoned"))?;
            let out = encoder
                .run(ort::inputs!["audio_signal" => feat_tensor, "length" => feat_len])
                .map_err(ort_err)?;
            let (_, data) = out["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let (_, lens) = out["encoded_lengths"]
                .try_extract_tensor::<i64>()
                .map_err(ort_err)?;
            (data.to_vec(), lens[0] as usize)
        };
        self.tdt_greedy(&enc, t_len)
    }

    fn tdt_greedy(&self, enc: &[f32], t_len: usize) -> Result<Vec<(i32, f32)>> {
        let mut joint = self
            .joint
            .lock()
            .map_err(|_| anyhow!("parakeet joint poisoned"))?;
        let mut s1 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut s2 = vec![0f32; STATE_LAYERS * STATE_DIM];
        let mut last = BLANK_ID;
        let mut ids: Vec<(i32, f32)> = Vec::new();
        let mut t = 0usize;
        let mut symbols_this_frame = 0usize;
        while t < t_len {
            let mut frame = vec![0f32; ENC_DIM];
            for (c, slot) in frame.iter_mut().enumerate() {
                *slot = enc[c * t_len + t];
            }
            let enc_t = Tensor::<f32>::from_array(([1usize, ENC_DIM, 1], frame.into_boxed_slice()))
                .map_err(ort_err)?;
            let targets = Tensor::<i32>::from_array(([1usize, 1], vec![last].into_boxed_slice()))
                .map_err(ort_err)?;
            let target_len = Tensor::<i32>::from_array(([1usize], vec![1i32].into_boxed_slice()))
                .map_err(ort_err)?;
            let st1 = Tensor::<f32>::from_array((
                [STATE_LAYERS, 1usize, STATE_DIM],
                s1.clone().into_boxed_slice(),
            ))
            .map_err(ort_err)?;
            let st2 = Tensor::<f32>::from_array((
                [STATE_LAYERS, 1usize, STATE_DIM],
                s2.clone().into_boxed_slice(),
            ))
            .map_err(ort_err)?;
            let out = joint
                .run(ort::inputs![
                    "encoder_outputs" => enc_t,
                    "targets" => targets,
                    "target_length" => target_len,
                    "input_states_1" => st1,
                    "input_states_2" => st2,
                ])
                .map_err(ort_err)?;
            let (_, logits) = out["outputs"]
                .try_extract_tensor::<f32>()
                .map_err(ort_err)?;
            let tok = argmax(&logits[..TOKEN_LOGITS]);
            let dur = argmax(&logits[TOKEN_LOGITS..TOKEN_LOGITS + TDT_DURATION_BINS.len()]);
            let mut skip = TDT_DURATION_BINS[dur];
            if tok as i32 != BLANK_ID {
                ids.push((tok as i32, log_softmax_at(&logits[..TOKEN_LOGITS], tok)));
                last = tok as i32;
                let (_, o1) = out["output_states_1"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?;
                let (_, o2) = out["output_states_2"]
                    .try_extract_tensor::<f32>()
                    .map_err(ort_err)?;
                s1.copy_from_slice(o1);
                s2.copy_from_slice(o2);
                symbols_this_frame += 1;
                if symbols_this_frame >= MAX_SYMBOLS_PER_FRAME_BREAKS_EMISSION_LOOPS && skip == 0 {
                    skip = 1;
                }
            } else if skip == 0 {
                skip = 1;
            }
            if skip > 0 {
                t += skip;
                symbols_this_frame = 0;
            }
        }
        Ok(ids)
    }

    fn detokenize(&self, ids: &[i32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let piece = &self.vocab[id as usize];
            if piece == "<unk>" {
                continue;
            }
            if let Some(rest) = piece.strip_prefix('\u{2581}') {
                if !out.is_empty() {
                    out.push(' ');
                }
                out.push_str(rest);
            } else {
                out.push_str(piece);
            }
        }
        out.trim().to_string()
    }
}

fn log_softmax_at(logits: &[f32], idx: usize) -> f32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = logits.iter().map(|&l| (l - max).exp()).sum::<f32>().ln();
    logits[idx] - max - log_sum
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_size_mismatch_names_the_head_width_and_log_softmax_is_a_probability() {
        let dir = std::env::temp_dir().join("parakeet-vocab-mismatch-test");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("vocab.txt"), "a 0\nb 1\n").unwrap();
        let Err(err) = ParakeetTdt::load(&dir) else {
            panic!("a 2-entry vocab must fail to load")
        };
        let err = err.to_string();
        assert!(err.contains("1025"), "unexpected error: {err}");
        {
            let logits = [1.0f32, 2.0, 3.0];
            let total: f32 = (0..3)
                .map(|i| super::log_softmax_at(&logits, i).exp())
                .sum();
            assert!((total - 1.0).abs() < 1e-5, "probabilities sum to {total}");
            assert!(super::log_softmax_at(&logits, 2) > super::log_softmax_at(&logits, 0));
            assert!(super::log_softmax_at(&[0.0f32, 0.0], 0) < 0.0);
        }
    }
}
