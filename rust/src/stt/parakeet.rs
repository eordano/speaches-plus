use super::TokenLogprob;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use ort::session::Session;
use ort::value::Tensor;

use crate::diarization::ep;
use crate::vad::ort_err;

pub const PARAKEET_DIR_ENV: &str = "STT_PARAKEET_DIR";
pub const PARAKEET_HUB_REPOS: [&str; 2] = [
    "models--istupakov--parakeet-tdt-0.6b-v3-onnx",
    "models--istupakov--parakeet-tdt-0.6b-v2-onnx",
];

const SR: usize = 16_000;
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
    vocab: Vocab,
    encoder_ep: &'static str,
}

#[derive(Debug)]
struct Vocab {
    pieces: Vec<String>,
    special: Vec<bool>,
    blank: usize,
    multilingual: bool,
}

pub const V3_LANGUAGES: [&str; 25] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

fn parse_vocab(raw: &str) -> Result<Vocab> {
    let mut pieces = Vec::new();
    for (i, line) in raw.lines().enumerate() {
        let (piece, id) = line
            .rsplit_once(' ')
            .ok_or_else(|| anyhow!("vocab.txt line {}: expected `<piece> <id>`", i + 1))?;
        if id.trim().parse::<usize>().ok() != Some(i) {
            return Err(anyhow!("vocab.txt line {}: id {id:?} is not {i}", i + 1));
        }
        pieces.push(piece.to_string());
    }
    let blank = pieces
        .iter()
        .position(|p| p == "<blk>")
        .ok_or_else(|| anyhow!("vocab.txt has no <blk> token"))?;
    let special = pieces
        .iter()
        .map(|p| {
            matches!(p.as_str(), "<unk>" | "<pad>" | "<blk>") || (p.starts_with("<|") && p.ends_with("|>"))
        })
        .collect();
    let multilingual = pieces.iter().any(|p| p == "<|es|>");
    Ok(Vocab {
        pieces,
        special,
        blank,
        multilingual,
    })
}

fn detokenize(vocab: &Vocab, ids: &[i32]) -> String {
    let mut out = String::new();
    for &id in ids {
        let id = id as usize;
        if vocab.special[id] {
            continue;
        }
        let piece = &vocab.pieces[id];
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

pub fn parakeet_dir() -> Result<PathBuf> {
    if let Ok(d) = std::env::var(PARAKEET_DIR_ENV) {
        return Ok(PathBuf::from(d));
    }
    let home = std::env::var("HOME").context("HOME unset")?;
    let hub = PathBuf::from(home).join(".cache/huggingface/hub");
    let snaps = PARAKEET_HUB_REPOS
        .iter()
        .map(|repo| hub.join(repo).join("snapshots"))
        .find(|p| p.is_dir())
        .unwrap_or_else(|| hub.join(PARAKEET_HUB_REPOS[0]).join("snapshots"));
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

pub fn hub_id(snapshot: &Path) -> Option<String> {
    let repo = snapshot.parent()?.parent()?.file_name()?.to_str()?;
    let rest = repo.strip_prefix("models--")?;
    Some(rest.replacen("--", "/", 1))
}

fn load_session(dir: &Path, file: &str) -> Result<ep::Loaded> {
    ep::load_session(&dir.join(file), file)
        .with_context(|| format!("load parakeet {file} from {}", dir.display()))
}

impl ParakeetTdt {
    pub fn load(dir: &Path) -> Result<Self> {
        let vocab_raw = std::fs::read_to_string(dir.join("vocab.txt"))
            .with_context(|| format!("read {}/vocab.txt", dir.display()))?;
        let vocab = parse_vocab(&vocab_raw).with_context(|| format!("{}/vocab.txt", dir.display()))?;
        tracing::info!(
            tokens = vocab.pieces.len(),
            blank = vocab.blank,
            multilingual = vocab.multilingual,
            dir = %dir.display(),
            "parakeet vocabulary loaded"
        );
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

    pub fn multilingual(&self) -> bool {
        self.vocab.multilingual
    }

    pub fn default_id(&self) -> &'static str {
        if self.vocab.multilingual {
            "nvidia/parakeet-tdt-0.6b-v3"
        } else {
            "nvidia/parakeet-tdt-0.6b-v2"
        }
    }

    pub fn supports_language(&self, lang: &str) -> bool {
        if self.vocab.multilingual {
            V3_LANGUAGES.contains(&lang)
        } else {
            lang == "en"
        }
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
            let part = detokenize(&self.vocab, &ids);
            if !part.is_empty() {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&part);
            }
            tokens.extend(
                scored
                    .into_iter()
                    .filter(|(id, _)| !self.vocab.special[*id as usize])
                    .map(|(id, logprob)| TokenLogprob {
                        token: self.vocab.pieces[id as usize].replace('\u{2581}', " "),
                        logprob,
                    }),
            );
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
        let token_logits = self.vocab.pieces.len();
        let blank = self.vocab.blank as i32;
        let mut last = blank;
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
            if logits.len() != token_logits + TDT_DURATION_BINS.len() {
                return Err(anyhow!(
                    "parakeet joint emits {} logits; vocab.txt gives {token_logits} tokens + {} \
                     durations: vocab.txt and decoder_joint-model.onnx are from different checkpoints",
                    logits.len(),
                    TDT_DURATION_BINS.len()
                ));
            }
            let tok = argmax(&logits[..token_logits]);
            let dur = argmax(&logits[token_logits..]);
            let mut skip = TDT_DURATION_BINS[dur];
            if tok as i32 != blank {
                ids.push((tok as i32, log_softmax_at(&logits[..token_logits], tok)));
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
    fn both_vocab_layouts_parse_and_special_tokens_never_reach_the_text() {
        let v2 = "<unk> 0\n\u{2581}t 1\nh 2\n\u{2581}a 3\n<blk> 4\n";
        let v = parse_vocab(v2).unwrap();
        assert_eq!((v.pieces.len(), v.blank, v.multilingual), (5, 4, false));
        assert_eq!(detokenize(&v, &[1, 2, 3]), "th a");

        let v3 = "<unk> 0\n<|nospeech|> 1\n<pad> 2\n<|es|> 3\n\u{2581}hola 4\n\u{2581}mundo 5\n<blk> 6\n";
        let v = parse_vocab(v3.replace("<|es|>", "<|en|>").as_str()).unwrap();
        assert!(!v.multilingual, "only a <|es|> tag marks the multilingual head");
        let v = parse_vocab(v3).unwrap();
        assert_eq!((v.pieces.len(), v.blank, v.multilingual), (7, 6, true));
        assert_eq!(detokenize(&v, &[3, 4, 1, 5, 0, 2]), "hola mundo");

        assert!(parse_vocab("a 0\nb 1\n").unwrap_err().to_string().contains("<blk>"));
        assert!(parse_vocab("a 0\n<blk> 2\n").unwrap_err().to_string().contains("line 2"));
    }

    #[test]
    fn the_model_id_comes_from_a_hub_snapshot_path_only() {
        let hub = Path::new("/c/hub/models--istupakov--parakeet-tdt-0.6b-v3-onnx/snapshots/8f23f0c");
        assert_eq!(hub_id(hub).as_deref(), Some("istupakov/parakeet-tdt-0.6b-v3-onnx"));
        assert_eq!(hub_id(Path::new("/var/lib/speaches/models/parakeet-tdt")), None);
        assert_eq!(
            hub_id(Path::new("/nix/store/r1k44-hf-model-istupakov-parakeet-tdt-0.6b-v3-onnx-8f23f0c")),
            None
        );
    }

    #[test]
    fn log_softmax_is_a_probability() {
        let logits = [1.0f32, 2.0, 3.0];
        let total: f32 = (0..3).map(|i| log_softmax_at(&logits, i).exp()).sum();
        assert!((total - 1.0).abs() < 1e-5, "probabilities sum to {total}");
        assert!(log_softmax_at(&logits, 2) > log_softmax_at(&logits, 0));
        assert!(log_softmax_at(&[0.0f32, 0.0], 0) < 0.0);
    }

    #[test]
    #[ignore = "needs PARAKEET_TEST_DIR (a checkpoint) and PARAKEET_TEST_F32 (16 kHz mono f32le files, comma separated)"]
    fn a_real_checkpoint_transcribes_each_clip() {
        let dir = std::env::var("PARAKEET_TEST_DIR").expect("PARAKEET_TEST_DIR");
        let model = ParakeetTdt::load(Path::new(&dir)).unwrap();
        for f in std::env::var("PARAKEET_TEST_F32").expect("PARAKEET_TEST_F32").split(',') {
            let bytes = std::fs::read(f).unwrap();
            let audio: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let started = std::time::Instant::now();
            let text = model.transcribe(&audio).unwrap();
            eprintln!(
                "parakeet {f}: {:.1} s audio in {} ms ({}): {text}",
                audio.len() as f32 / SR as f32,
                started.elapsed().as_millis(),
                model.ep()
            );
            assert!(!text.is_empty(), "{f}: empty transcript");
            assert!(!text.contains("<|"), "{f}: special token leaked: {text}");
        }
    }
}
